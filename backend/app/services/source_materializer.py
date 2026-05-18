"""Materialise any SourceToolkit into a local tmpdir for the filesystem indexer.

The filesystem indexer (FilesystemRepositoryIndexer) expects files on disk.
SourceMaterializer bridges the SourceToolkit interface — which yields FileInfo
objects and FileContent on demand — to a concrete directory tree the indexer
can traverse directly.

Token lifecycle note
--------------------
Credentials (access_token, refresh_token, client_id) live exclusively in the
SourceToolkit instance for the duration of the run.  They are NEVER written to
the DB, NEVER logged (TokenRedactionFilter is the safety net; primary defence
is "don't log dicts containing these keys").  When the toolkit goes out of
scope after __aexit__, tokens go with it.

Atlassian mid-run token rotation
---------------------------------
If AtlassianClient refreshes an access_token during Confluence/Jira
materialisation (401 → refresh → retry), the rotated token stays in the
in-memory client only.  It is NOT propagated back to the SPA in this PR —
the SPA will reuse its (potentially stale) refresh token on the next call.
This is an accepted limitation; a future PR can add a token-rotation callback.
"""

from __future__ import annotations

import logging
import os
import shutil
import tempfile
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from app.core.sources.base import SourceToolkit

logger = logging.getLogger(__name__)


class SourceMaterializer:
    """Walk a SourceToolkit and write its contents to a local tmpdir as flat files.

    Usage (async context manager)::

        async with SourceMaterializer(toolkit) as workdir:
            # workdir is a Path — pass it to FilesystemRepositoryIndexer
            indexer.index_repository(f"file://{workdir}", ...)

    Or call explicitly::

        mat = SourceMaterializer(toolkit)
        workdir = await mat.materialize()
        ...
        await mat.cleanup()

    File naming
    -----------
    - Git-sourced files keep their original repo-relative path and extension.
    - Confluence / Jira files use the FileInfo.path (already sanitized by the
      toolkit) with a ``.md`` extension appended when language=="markdown".
    - For any other language, the path is used as-is.

    Path safety
    -----------
    Even though toolkit implementations already sanitize paths, we apply a
    secondary escape-check: any FileInfo.path that resolves outside *workdir*
    raises ValueError.  This is cheap insurance against hypothetical toolkit
    bugs.
    """

    def __init__(self, toolkit: "SourceToolkit", workdir: Path | None = None) -> None:
        self._toolkit = toolkit
        self._workdir = workdir
        self._owned: bool = workdir is None  # True when we created the tmpdir

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def materialize(self) -> Path:
        """Write all files from the toolkit to the workdir, return workdir."""
        if self._workdir is None:
            self._workdir = Path(tempfile.mkdtemp(prefix="wikis_source_"))

        workdir = self._workdir
        written = 0
        errors = 0

        # Collect file infos first (list_files is an async generator — no await).
        file_infos = []
        async for info in self._toolkit.list_files():
            file_infos.append(info)

        logger.debug(
            "Materializing %d files from %s toolkit into %s",
            len(file_infos),
            self._toolkit.source_type,
            workdir,
        )

        for info in file_infos:
            dest = self._resolve_dest(workdir, info)
            try:
                content_obj = await self._toolkit.fetch_content(info.origin)
                dest.parent.mkdir(parents=True, exist_ok=True)
                dest.write_text(content_obj.content, encoding="utf-8", errors="replace")
                written += 1
            except Exception as exc:  # noqa: BLE001
                logger.warning("Failed to materialize %s: %s", info.path, exc)
                errors += 1

        logger.info(
            "Materialisation complete: %d files written, %d errors (source=%s, dest=%s)",
            written,
            errors,
            self._toolkit.source_type,
            workdir,
        )
        return workdir

    async def cleanup(self) -> None:
        """Remove the workdir (only if we created it)."""
        if self._owned and self._workdir is not None:
            try:
                shutil.rmtree(self._workdir, ignore_errors=True)
            except Exception as exc:  # noqa: BLE001
                logger.warning("Failed to clean up materializer workdir %s: %s", self._workdir, exc)
            finally:
                self._workdir = None

    # ------------------------------------------------------------------
    # Async context manager
    # ------------------------------------------------------------------

    async def __aenter__(self) -> Path:
        return await self.materialize()

    async def __aexit__(self, *_exc: object) -> None:
        await self.cleanup()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _resolve_dest(self, workdir: Path, info: "SourceToolkit") -> Path:  # type: ignore[type-arg]
        """Return the absolute destination Path for a FileInfo, inside *workdir*.

        Appends ``.md`` when language is ``"markdown"`` and the path doesn't
        already end in ``.md`` — Confluence and Jira use this convention.

        Raises:
            ValueError: If the resolved path would escape *workdir*.
        """
        from app.core.sources.base import FileInfo  # avoid circular at module level

        # info is actually a FileInfo but TYPE_CHECKING reference is SourceToolkit
        # (pyright workaround) — cast at runtime.
        if not isinstance(info, FileInfo):
            raise TypeError(f"Expected FileInfo, got {type(info)!r}")

        raw_path = info.path

        # Append .md for markdown content that doesn't already have the extension.
        if info.language == "markdown" and not raw_path.endswith(".md"):
            raw_path = raw_path + ".md"

        # Guard against path traversal embedded in FileInfo.path.
        candidate = (workdir / raw_path).resolve()
        workdir_resolved = workdir.resolve()
        if not candidate.is_relative_to(workdir_resolved):
            raise ValueError(
                f"FileInfo.path {info.path!r} resolves to {candidate} which "
                f"escapes workdir {workdir_resolved}"
            )
        return candidate
