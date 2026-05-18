"""Validate a source configuration and return a preview without persistence.

Drives Step 3 of the source-ingestion wizard (#208). The endpoint takes the
same ``source_type`` / ``scope`` / ``auth`` shape the wizard would post to
``/api/v1/wikis``, runs a lightweight reachability check, and returns
preview stats. Nothing is written — no ``WikiRecord``, no invocation, no
token persisted. The toolkit's async context cleans up any tmpdir it
created during the shallow clone, even on raises.
"""

from __future__ import annotations

import logging
from pathlib import Path

from app.core.sources.exceptions import (
    SourceAuthError,
    SourceNotFoundError,
    SourceUnavailableError,
)
from app.core.sources.git_toolkit import GitToolkit
from app.models.api import GitScanPreview, ScanRequest, ScanResponse

logger = logging.getLogger(__name__)


class ScanError(Exception):
    """Raised when a scan cannot complete. Maps to HTTP 400 at the route."""

    def __init__(self, message: str, reachable: bool = False) -> None:
        super().__init__(message)
        self.reachable = reachable


class SourceScanService:
    """Per-source-type scan dispatch. Reuses existing toolkit logic."""

    # Cap how many file paths we enumerate during scan. Real repos can be
    # massive; the preview only needs an honest order-of-magnitude count
    # and a few top-level paths, not a full walk.
    _MAX_FILES_ENUMERATED = 50_000
    _MAX_TOP_PATHS = 12

    async def scan(self, request: ScanRequest) -> ScanResponse:
        """Dispatch to the per-source-type scanner."""
        if request.source_type == "git":
            return await self._scan_git(request)
        # Atlassian connectors land in a follow-up (#211). Surface as 501 at
        # the route so the wizard's Step 3 can render a "preview not
        # supported for this source yet" affordance.
        raise NotImplementedError(
            f"Scan not implemented for source_type={request.source_type!r}"
        )

    async def _scan_git(self, request: ScanRequest) -> ScanResponse:
        scope = request.scope or {}
        repo_url = scope.get("repo_url")
        if not repo_url:
            raise ScanError("scope.repo_url is required for git scans")
        branch = scope.get("branch", "main")
        token = (request.auth or {}).get("pat")

        warnings: list[str] = []

        toolkit = GitToolkit.from_config(
            {"repo_url": repo_url, "branch": branch, "token": token}
        )
        async with toolkit:
            # ls-remote first — cheap reachability + auth probe before we
            # commit to a clone. Translates source-layer exceptions into
            # ScanError so the route can return a uniform 400 with a
            # redacted, user-safe message.
            try:
                await toolkit.test_connection()
            except SourceAuthError as exc:
                raise ScanError(str(exc), reachable=False) from exc
            except SourceNotFoundError as exc:
                raise ScanError(str(exc), reachable=False) from exc
            except SourceUnavailableError as exc:
                raise ScanError(str(exc), reachable=False) from exc

            # Trigger the shallow clone (depth=1, single-branch). The async
            # context manager guarantees tmpdir cleanup via ``close()`` on
            # exit, even if file enumeration below raises.
            try:
                toolkit._ensure_workdir()  # noqa: SLF001 — internal API by design
            except SourceAuthError as exc:
                raise ScanError(str(exc), reachable=False) from exc
            except SourceNotFoundError as exc:
                raise ScanError(str(exc), reachable=False) from exc
            except SourceUnavailableError as exc:
                raise ScanError(str(exc), reachable=False) from exc

            workdir = toolkit._workdir  # noqa: SLF001
            commit_hash = toolkit._commit_sha  # noqa: SLF001
            if workdir is None:
                raise ScanError("Clone completed but workdir is unset")

            file_count, total_size, truncated = _enumerate(
                workdir, self._MAX_FILES_ENUMERATED
            )
            if truncated:
                warnings.append(
                    f"File count truncated at {self._MAX_FILES_ENUMERATED} — repository is larger"
                )

            top_paths = _top_level_paths(workdir, self._MAX_TOP_PATHS)

        preview = GitScanPreview(
            default_branch=None,  # populated once we add ls-remote --symref support
            resolved_branch=branch,
            commit_hash=commit_hash,
            file_count=file_count,
            top_paths=top_paths,
            size_bytes=total_size,
        )
        return ScanResponse(
            source_type="git",
            reachable=True,
            preview=preview,
            warnings=warnings,
        )


def _enumerate(workdir: Path, cap: int) -> tuple[int, int, bool]:
    """Walk *workdir* and return ``(file_count, total_bytes, truncated)``.

    Skips the ``.git`` directory. Stops counting at *cap* and signals truncation.
    """
    count = 0
    total = 0
    truncated = False
    for path in workdir.rglob("*"):
        # Skip git internals — the shallow clone leaves a small .git but it
        # is not user content and would inflate the preview.
        if ".git" in path.parts:
            continue
        if not path.is_file():
            continue
        count += 1
        try:
            total += path.stat().st_size
        except OSError:
            # Broken symlinks or perms — count the file, skip its size.
            continue
        if count >= cap:
            truncated = True
            break
    return count, total, truncated


def _top_level_paths(workdir: Path, limit: int) -> list[str]:
    """Return up to *limit* top-level entries sorted alphabetically.

    Directories get a trailing slash so the UI can render them distinctly.
    """
    entries: list[str] = []
    for entry in sorted(workdir.iterdir(), key=lambda p: p.name):
        if entry.name == ".git":
            continue
        entries.append(f"{entry.name}/" if entry.is_dir() else entry.name)
        if len(entries) >= limit:
            break
    return entries
