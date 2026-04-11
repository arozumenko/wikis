"""Export service — builds downloadable vault archives from wiki artifacts.

Currently supports the Obsidian vault zip format and Wikis-to-Wikis bundle format.
"""

from __future__ import annotations

import io
import json
import logging
import re
import zipfile
from datetime import datetime, timezone
from typing import TYPE_CHECKING, AsyncIterator

from app.storage.base import ArtifactStorage

if TYPE_CHECKING:
    from app.config import Settings
    from app.services.wiki_management import WikiManagementService

logger = logging.getLogger(__name__)


class WikiNotFoundError(Exception):
    """Raised when a wiki_id does not match any known wiki record."""


class ExportService:
    """Builds exportable archives from stored wiki artifacts.

    Args:
        storage: Artifact storage backend (local or S3).
        wiki_management: WikiManagementService for record look-ups.
        settings: Application settings (needed for cache_dir and storage_backend).
    """

    def __init__(
        self,
        storage: ArtifactStorage,
        wiki_management: WikiManagementService,
        settings: Settings | None = None,
    ) -> None:
        self.storage = storage
        self.wiki_management = wiki_management
        self.settings = settings

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def build_obsidian_zip(self, wiki_id: str) -> AsyncIterator[bytes]:
        """Stream a ZIP archive formatted as an Obsidian vault.

        The archive layout is::

            {vault_title}/
                README.md
                .obsidian/app.json
                {page_title}.md     ← one per wiki page, with YAML frontmatter
            ...

        Args:
            wiki_id: Identifier of the wiki to export.

        Yields:
            Raw bytes of the ZIP archive, in 64 KiB chunks.

        Raises:
            WikiNotFoundError: When no WikiRecord exists for *wiki_id*.

        TODO: Replace the buffer-then-yield approach with true streaming using a
              pipe pattern (e.g. threading.Pipe + ZipFile) once the artifact sizes
              warrant it.  The current BytesIO buffer is acceptable for typical
              wiki sizes (< 50 MB).
        """
        record = await self.wiki_management.get_wiki_record(wiki_id)
        if record is None:
            raise WikiNotFoundError(f"No wiki found for id={wiki_id!r}")

        # Sanitize vault_title to prevent path traversal entries in the ZIP.
        vault_title = re.sub(r"[^\w\s\-]", "", record.title or wiki_id).strip() or "wiki"
        repo_url = record.repo_url or ""
        generated_at = datetime.now(tz=timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

        # Collect only .md artifacts stored under the wiki_pages/ sub-path.
        # Skip any README.md — the generated one at vault root is the authoritative copy.
        all_artifacts = await self.storage.list_artifacts("wiki_artifacts", prefix=wiki_id)
        md_artifacts = [
            a for a in all_artifacts
            if a.endswith(".md") and "wiki_pages/" in a and not a.endswith("/README.md")
        ]

        buf = io.BytesIO()
        with zipfile.ZipFile(buf, mode="w", compression=zipfile.ZIP_DEFLATED) as zf:
            # .obsidian/app.json
            obsidian_config = json.dumps({"legacyEditor": False})
            zf.writestr(f"{vault_title}/.obsidian/app.json", obsidian_config)

            # README.md at vault root
            readme = self._build_readme(vault_title, repo_url, generated_at)
            zf.writestr(f"{vault_title}/README.md", readme)

            # Wiki pages
            for artifact_key in md_artifacts:
                raw_bytes = await self.storage.download("wiki_artifacts", artifact_key)
                raw_content = raw_bytes.decode("utf-8") if isinstance(raw_bytes, bytes) else raw_bytes

                page_title = self._page_title_from_key(artifact_key)
                content_with_fm = self._inject_frontmatter(raw_content, page_title, repo_url, generated_at)

                zf.writestr(f"{vault_title}/{page_title}.md", content_with_fm)

        buf.seek(0)
        chunk_size = 65536  # 64 KiB
        while chunk := buf.read(chunk_size):
            yield chunk

    async def build_wikis_bundle(self, wiki_id: str) -> AsyncIterator[bytes]:
        """Stream a Wikis-to-Wikis bundle (.wikiexport) for re-importing into another instance.

        The bundle is a ZIP archive containing:

        - ``manifest.json`` — metadata (no wiki_id; derived on import)
        - ``pages/{wiki_id}/wiki_pages/*.md`` — all markdown artifacts, sub-paths preserved
        - ``cache/cache_key.txt`` — the raw cache key string
        - ``cache/cache_index_fragment.json`` — only this wiki's slice of cache_index.json
        - ``cache/{key}.faiss``, ``{key}.docstore.bin``, etc. — all cache files EXCEPT .docs.pkl

        Args:
            wiki_id: Identifier of the wiki to export.

        Yields:
            Raw bytes of the ZIP archive, in 64 KiB chunks.

        Raises:
            NotImplementedError: When ``settings.storage_backend`` is not ``"local"``.
            WikiNotFoundError: When no WikiRecord exists for *wiki_id*.
        """
        import asyncio
        import glob as glob_mod
        import json as json_mod
        from pathlib import Path

        from app.services.wiki_management import _derive_cache_key

        if self.settings is not None and self.settings.storage_backend != "local":
            raise NotImplementedError(
                "Wikis-bundle export not supported for S3 storage backend"
            )

        record = await self.wiki_management.get_wiki_record(wiki_id)
        if record is None:
            raise WikiNotFoundError(f"No wiki found for id={wiki_id!r}")

        repo_url = record.repo_url or ""
        branch = record.branch or "main"
        title = record.title or wiki_id
        commit_hash = record.commit_hash or ""
        page_count = record.page_count or 0
        exported_at = datetime.now(tz=timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

        # Collect all artifacts for this wiki
        all_artifacts = await self.storage.list_artifacts("wiki_artifacts", prefix=wiki_id)

        buf = io.BytesIO()
        with zipfile.ZipFile(buf, mode="w", compression=zipfile.ZIP_DEFLATED) as zf:
            # --- pages/ ---
            for artifact_key in all_artifacts:
                raw_bytes = await self.storage.download("wiki_artifacts", artifact_key)
                zf.writestr(f"pages/{artifact_key}", raw_bytes)

            # --- cache/ ---
            cache_key: str | None = None
            cache_index_fragment: dict = {}

            if self.settings is not None:
                cache_dir = self.settings.cache_dir

                def _load_cache_data() -> tuple[str | None, dict]:
                    """Blocking I/O: resolve key and read index fragment."""
                    key = _derive_cache_key(cache_dir, repo_url, branch)
                    fragment: dict = {}
                    index_file = Path(cache_dir) / "cache_index.json"
                    if key and index_file.exists():
                        try:
                            with open(index_file) as fp:
                                full_index = json_mod.load(fp)
                        except Exception:
                            full_index = {}
                        # Build a minimal fragment: refs entry + key mapping
                        url = repo_url.rstrip("/")
                        if url.endswith(".git"):
                            url = url[:-4]
                        parts = url.split("/")
                        owner_repo = f"{parts[-2]}/{parts[-1]}" if len(parts) >= 2 else url
                        repo_identifier = f"{owner_repo}:{branch}"
                        refs = full_index.get("refs", {})
                        resolved = refs.get(repo_identifier, repo_identifier)
                        fragment_refs: dict = {}
                        if repo_identifier in refs:
                            fragment_refs[repo_identifier] = refs[repo_identifier]
                        entry: dict = {"refs": fragment_refs}
                        if resolved in full_index:
                            entry[resolved] = full_index[resolved]
                        elif repo_identifier in full_index:
                            entry[repo_identifier] = full_index[repo_identifier]
                        fragment = entry
                    return key, fragment

                cache_key, cache_index_fragment = await asyncio.to_thread(_load_cache_data)

                if cache_key:
                    # Include all cache files for this key, EXCEPT .docs.pkl
                    def _collect_cache_files() -> list[tuple[str, bytes]]:
                        result: list[tuple[str, bytes]] = []
                        for path_str in glob_mod.glob(
                            str(Path(cache_dir) / f"{cache_key}.*")
                        ):
                            p = Path(path_str)
                            if p.suffix == ".pkl" or p.name.endswith(".docs.pkl"):
                                continue
                            try:
                                result.append((p.name, p.read_bytes()))
                            except Exception as exc:
                                logger.warning("Skipping cache file %s: %s", p, exc)
                        return result

                    cache_files = await asyncio.to_thread(_collect_cache_files)
                    for filename, file_bytes in cache_files:
                        zf.writestr(f"cache/{filename}", file_bytes)

            # cache_key.txt
            zf.writestr("cache/cache_key.txt", cache_key or "")

            # cache_index_fragment.json
            zf.writestr(
                "cache/cache_index_fragment.json",
                json.dumps(cache_index_fragment, indent=2),
            )

            # --- manifest.json ---
            manifest = {
                "bundle_version": 1,
                "title": title,
                "repo_url": repo_url,
                "branch": branch,
                "commit_hash": commit_hash,
                "exported_at": exported_at,
                "page_count": page_count,
            }
            zf.writestr("manifest.json", json.dumps(manifest, indent=2))

        buf.seek(0)
        chunk_size = 65536  # 64 KiB
        while chunk := buf.read(chunk_size):
            yield chunk

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _page_title_from_key(artifact_key: str) -> str:
        """Derive a human-readable page title from the storage key.

        Example: ``"abc123/architecture-overview.md"`` → ``"architecture-overview"``
        """
        filename = artifact_key.rsplit("/", 1)[-1]
        stem = filename[: -len(".md")] if filename.endswith(".md") else filename
        return stem

    @staticmethod
    def _inject_frontmatter(content: str, page_title: str, repo_url: str, generated_at: str) -> str:
        """Prepend YAML frontmatter to *content*, stripping any existing block first.

        The injected block contains:
        - ``title``: derived from the filename
        - ``repo``: repository URL
        - ``generated_at``: ISO-8601 UTC timestamp
        - ``tags``: ``[wiki, generated]``
        """
        # Strip existing frontmatter if present.
        stripped = re.sub(r"^---\n.*?\n---\n?", "", content, count=1, flags=re.DOTALL)

        # Escape double-quotes in the title for safe YAML embedding.
        safe_title = page_title.replace('"', '\\"')
        safe_repo = repo_url.replace('"', '\\"')

        frontmatter = (
            "---\n"
            f'title: "{safe_title}"\n'
            f'repo: "{safe_repo}"\n'
            f'generated_at: "{generated_at}"\n'
            "tags: [wiki, generated]\n"
            "---\n"
        )
        return frontmatter + stripped

    @staticmethod
    def _build_readme(vault_title: str, repo_url: str, generated_at: str) -> str:
        """Return a minimal README.md for the vault root."""
        return (
            f"# {vault_title}\n\n"
            f"This vault was automatically generated from the repository:\n\n"
            f"**Repo:** {repo_url}\n\n"
            f"**Generated at:** {generated_at}\n\n"
            "---\n\n"
            "*Generated by [Wikis](https://github.com/arozumenko/wikis)*\n"
        )
