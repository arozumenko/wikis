"""Export service — builds downloadable vault archives from wiki artifacts.

Currently supports the Obsidian vault zip format.
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
    from app.services.wiki_management import WikiManagementService

logger = logging.getLogger(__name__)


class WikiNotFoundError(Exception):
    """Raised when a wiki_id does not match any known wiki record."""


class ExportService:
    """Builds exportable archives from stored wiki artifacts.

    Args:
        storage: Artifact storage backend (local or S3).
        wiki_management: WikiManagementService for record look-ups.
    """

    def __init__(self, storage: ArtifactStorage, wiki_management: WikiManagementService) -> None:
        self.storage = storage
        self.wiki_management = wiki_management

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

        vault_title = record.title or wiki_id
        repo_url = record.repo_url or ""
        generated_at = datetime.now(tz=timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

        # Collect only .md artifacts for this wiki.
        all_artifacts = await self.storage.list_artifacts("wiki_artifacts", prefix=wiki_id)
        md_artifacts = [a for a in all_artifacts if a.endswith(".md")]

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
