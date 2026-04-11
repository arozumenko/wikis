"""Import service — restores a wiki from a .wikiexport bundle.

The bundle format is produced by ``ExportService.build_wikis_bundle`` and
contains a manifest, wiki artifact pages, and optional cache files.
"""

from __future__ import annotations

import asyncio
import hashlib
import io
import json
import logging
import os
import zipfile
from typing import TYPE_CHECKING

from app.models.db_models import WikiRecord
from app.storage.base import ArtifactStorage

if TYPE_CHECKING:
    from app.config import Settings
    from app.services.wiki_management import WikiManagementService

logger = logging.getLogger(__name__)

# Required manifest keys (bundle_version checked separately)
_REQUIRED_MANIFEST_KEYS = ("repo_url", "branch", "title")


class BundleValidationError(Exception):
    """Raised when the bundle archive is malformed or fails validation."""


class BundleVersionError(Exception):
    """Raised when the bundle version is unsupported."""


class ImportService:
    """Restores a wiki from a ``.wikiexport`` bundle.

    Args:
        storage: Artifact storage backend (must be local).
        wiki_management: WikiManagementService for record persistence.
        settings: Application settings (used for ``storage_backend`` and ``cache_dir``).
    """

    SUPPORTED_BUNDLE_VERSION = 1

    def __init__(
        self,
        storage: ArtifactStorage,
        wiki_management: WikiManagementService,
        settings: Settings,
    ) -> None:
        self.storage = storage
        self.wiki_management = wiki_management
        self.settings = settings

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def restore_wiki(self, bundle_bytes: bytes, owner_id: str) -> WikiRecord:
        """Restore a wiki from a ``.wikiexport`` bundle.

        Args:
            bundle_bytes: Raw bytes of the ``.wikiexport`` ZIP archive.
            owner_id: The importing user's ID — always overrides any ID in the bundle.

        Returns:
            The upserted ``WikiRecord`` for the restored wiki.

        Raises:
            NotImplementedError: When ``settings.storage_backend`` is not ``"local"``.
            BundleValidationError: When the bundle is invalid (bad zip, missing manifest,
                forbidden files, missing required fields).
            BundleVersionError: When the ``bundle_version`` in the manifest is unsupported.
        """
        # Step 1 — S3 guard
        if self.settings.storage_backend != "local":
            raise NotImplementedError(
                "Wiki import not supported for S3 storage backend"
            )

        # Step 2 — Parse ZIP
        try:
            zf = zipfile.ZipFile(io.BytesIO(bundle_bytes))
        except zipfile.BadZipFile:
            raise BundleValidationError("Invalid bundle: not a zip file")

        with zf:
            # Step 3 — Validate manifest
            if "manifest.json" not in zf.namelist():
                raise BundleValidationError("Invalid bundle: missing manifest.json")

            try:
                manifest = json.loads(zf.read("manifest.json").decode("utf-8"))
            except Exception as exc:
                raise BundleValidationError(
                    f"Invalid bundle: manifest.json is not valid JSON"
                ) from exc

            bundle_version = manifest.get("bundle_version")
            if bundle_version != self.SUPPORTED_BUNDLE_VERSION:
                raise BundleVersionError(
                    f"Unsupported bundle version: {bundle_version}"
                )

            for key in _REQUIRED_MANIFEST_KEYS:
                if not manifest.get(key):
                    raise BundleValidationError(
                        f"Invalid bundle: manifest missing required field '{key}'"
                    )

            repo_url: str = manifest["repo_url"]
            branch: str = manifest["branch"]
            title: str = manifest["title"]
            commit_hash: str | None = manifest.get("commit_hash") or None
            page_count: int = int(manifest.get("page_count") or 0)

            # Step 4 — Reject pickle files
            for entry_name in zf.namelist():
                if entry_name.endswith(".docs.pkl"):
                    raise BundleValidationError(
                        "Invalid bundle: .docs.pkl files are not permitted"
                    )

            # Step 5 — Derive wiki_id from repo_url + branch (never trust bundle)
            wiki_id = hashlib.sha256(
                f"{repo_url}:{branch}".encode()
            ).hexdigest()[:16]

            # Step 6 — Collision handling
            existing = await self.wiki_management.get_wiki_record(wiki_id)
            if existing is not None:
                await self.wiki_management.delete_wiki(
                    wiki_id,
                    user_id=None,  # system-level delete; bypass ownership check
                    cache_dir=self.settings.cache_dir,
                )
                logger.info("Removed existing wiki %s before import", wiki_id)

            # Step 7 — Write pages to artifact storage
            page_entries = [
                name for name in zf.namelist()
                if name.startswith("pages/") and not name.endswith("/")
            ]
            for entry_name in page_entries:
                # Strip the leading "pages/" prefix to get the storage key
                artifact_key = entry_name[len("pages/"):]
                artifact_data = zf.read(entry_name)
                await self.storage.upload("wiki_artifacts", artifact_key, artifact_data)

            # Step 8 — Write cache files
            cache_key: str | None = None
            if "cache/cache_key.txt" in zf.namelist():
                cache_key = zf.read("cache/cache_key.txt").decode("utf-8").strip() or None

            if cache_key:
                cache_entries = [
                    name for name in zf.namelist()
                    if name.startswith("cache/")
                    and not name.endswith("/")
                    and name not in ("cache/cache_key.txt", "cache/cache_index_fragment.json")
                ]

                def _write_cache_files(
                    entries: list[tuple[str, bytes]],
                    cache_dir: str,
                ) -> None:
                    from pathlib import Path
                    Path(cache_dir).mkdir(parents=True, exist_ok=True)
                    for filename, file_bytes in entries:
                        dest = Path(cache_dir) / filename
                        dest.write_bytes(file_bytes)

                entries_data = [
                    (os.path.basename(name), zf.read(name))
                    for name in cache_entries
                ]
                await asyncio.to_thread(
                    _write_cache_files, entries_data, self.settings.cache_dir
                )

            # Step 9 — Atomic cache_index.json patch
            fragment: dict = {}
            if "cache/cache_index_fragment.json" in zf.namelist():
                try:
                    fragment = json.loads(
                        zf.read("cache/cache_index_fragment.json").decode("utf-8")
                    )
                except Exception:
                    fragment = {}

            if fragment:
                await asyncio.to_thread(
                    _patch_cache_index,
                    self.settings.cache_dir,
                    fragment,
                )

        # Step 10 — Upsert WikiRecord (outside the zipfile context — no more zip reads needed)
        await self.wiki_management.register_wiki(
            wiki_id=wiki_id,
            repo_url=repo_url,
            branch=branch,
            title=title,
            page_count=page_count,
            owner_id=owner_id,
            status="complete",
            commit_hash=commit_hash,
        )

        # Step 11 — Return the upserted WikiRecord
        record = await self.wiki_management.get_wiki_record(wiki_id)
        assert record is not None, "WikiRecord must exist after register_wiki"
        return record


# ---------------------------------------------------------------------------
# Module-level helpers (blocking I/O, called via asyncio.to_thread)
# ---------------------------------------------------------------------------


def _patch_cache_index(cache_dir: str, fragment: dict) -> None:
    """Atomically merge *fragment* into ``cache_index.json``.

    Only this wiki's entries are added/replaced; all other keys are preserved.
    The write uses ``os.replace`` (atomic on POSIX) via a ``.tmp`` file.
    """
    from pathlib import Path

    cache_path = Path(cache_dir)
    cache_path.mkdir(parents=True, exist_ok=True)
    index_file = cache_path / "cache_index.json"
    tmp_file = cache_path / "cache_index.json.tmp"

    # Read existing index (empty dict if missing)
    existing: dict = {}
    if index_file.exists():
        try:
            with open(index_file) as f:
                existing = json.load(f)
        except Exception:
            existing = {}

    # Merge: add/replace entries from fragment, preserve all others.
    # Fragment structure mirrors the full index (top-level keys).
    for k, v in fragment.items():
        if k == "refs":
            # Merge the refs sub-dict carefully
            existing_refs = existing.setdefault("refs", {})
            if isinstance(v, dict):
                existing_refs.update(v)
        else:
            existing[k] = v

    try:
        with open(tmp_file, "w") as f:
            json.dump(existing, f, indent=2)
        os.replace(tmp_file, index_file)
        logger.info("Patched cache_index.json with imported wiki fragment")
    except Exception as exc:
        logger.warning("Failed to patch cache_index.json: %s", exc)
        # Non-fatal — wiki still imported; cache just won't have the index entry
