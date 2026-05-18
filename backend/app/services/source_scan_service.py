"""Validate a source configuration and return a preview without persistence.

Drives Step 3 of the source-ingestion wizard (#208). The endpoint takes the
same ``source_type`` / ``scope`` / ``auth`` shape the wizard would post to
``/api/v1/wikis``, runs a lightweight reachability check, and returns
preview stats. Nothing is written — no ``WikiRecord``, no invocation, no
token persisted. The toolkit's async context cleans up any tmpdir it
created during the shallow clone, even on raises.
"""

from __future__ import annotations

import asyncio
import logging
from pathlib import Path

from app.config import get_settings
from app.core.local_repo_provider import (
    extract_git_metadata,
    is_local_path,
    validate_local_path,
)
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
    # Total-time budget for a single scan (Rio: prevent worker-thread
    # exhaustion under concurrency even when the clone itself runs in a
    # thread pool). git's own `ls-remote` timeout is 30s; clone is
    # unbounded — 90s is the combined cap.
    _SCAN_BUDGET_SEC = 90.0

    async def scan(self, request: ScanRequest) -> ScanResponse:
        """Dispatch to the per-source-type scanner under a total-time budget."""
        if request.source_type == "git":
            try:
                return await asyncio.wait_for(
                    self._scan_git(request), timeout=self._SCAN_BUDGET_SEC
                )
            except asyncio.TimeoutError as exc:
                raise ScanError(
                    f"Scan exceeded the {self._SCAN_BUDGET_SEC:.0f}s budget — "
                    "the source may be slow or unreachable",
                    reachable=False,
                ) from exc
        # Atlassian connectors land in a follow-up (#211). Surface as 501 at
        # the route so the wizard's Step 3 can render a "preview not
        # supported for this source yet" affordance.
        raise NotImplementedError(
            f"Scan not implemented for source_type={request.source_type!r}"
        )

    async def _scan_git(self, request: ScanRequest) -> ScanResponse:
        scope = request.scope or {}
        repo_url = scope.get("repo_url")
        if not isinstance(repo_url, str) or not repo_url:
            raise ScanError("scope.repo_url must be a non-empty string for git scans")
        branch = scope.get("branch", "main")
        if not isinstance(branch, str) or not branch:
            raise ScanError("scope.branch must be a non-empty string")
        token = (request.auth or {}).get("pat")

        # Security gate: local paths must clear ALLOWED_LOCAL_PATHS before
        # the toolkit touches them. ``GitToolkit.from_config`` documents
        # that callers are responsible for this — without the gate, a
        # ``file:///etc`` or bare ``/etc`` payload would scan host
        # directories. ``is_local_path`` covers both ``/`` and
        # ``file://`` prefixes.
        if is_local_path(repo_url):
            settings = get_settings()
            allowed: list[str] = []
            if settings.allowed_local_paths:
                allowed = [
                    p.strip()
                    for p in settings.allowed_local_paths.split(",")
                    if p.strip()
                ]
            try:
                validate_local_path(repo_url.removeprefix("file://"), allowed or None)
            except ValueError as exc:
                raise ScanError(str(exc), reachable=False) from exc

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
            except (SourceAuthError, SourceNotFoundError, SourceUnavailableError) as exc:
                raise ScanError(str(exc), reachable=False) from exc

            # Shallow clone (depth=1, single-branch). Offloaded to a worker
            # thread so a slow remote can't block the FastAPI event loop.
            # The async context manager guarantees tmpdir cleanup via
            # ``close()`` on exit, even if enumeration below raises.
            try:
                await asyncio.to_thread(toolkit._ensure_workdir)  # noqa: SLF001
            except (SourceAuthError, SourceNotFoundError, SourceUnavailableError) as exc:
                raise ScanError(str(exc), reachable=False) from exc

            workdir = toolkit._workdir  # noqa: SLF001
            commit_hash = toolkit._commit_sha  # noqa: SLF001
            is_local_clone = toolkit._is_local  # noqa: SLF001
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

            # For remote clones the single-branch clone makes the requested
            # branch authoritative. For local repos the toolkit doesn't
            # check out anything, so the actual HEAD branch may differ from
            # ``branch``. Read it from ``extract_git_metadata`` so the
            # response reflects reality.
            if is_local_clone:
                meta = extract_git_metadata(workdir)
                resolved_branch = meta.branch or branch
            else:
                resolved_branch = branch

        preview = GitScanPreview(
            default_branch=None,  # populated once we add ls-remote --symref support
            resolved_branch=resolved_branch,
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

    Skips the ``.git`` directory and any symlink (whether to a file or a
    directory) — symlinks could escape the repo and inflate the preview
    with host content (Copilot C4). Truncation is signalled only when at
    least one file beyond ``cap`` exists (off-by-one fix: Copilot C3).
    """
    count = 0
    total = 0
    truncated = False
    for path in workdir.rglob("*"):
        if ".git" in path.parts:
            continue
        # Skip symlinks before any stat call — ``is_file()`` would follow
        # them and let a crafted symlink count host files toward the
        # preview total. ``GitToolkit.list_files`` already enforces this;
        # the scan preview must match.
        if path.is_symlink():
            continue
        if not path.is_file():
            continue
        if count >= cap:
            # We found a file past the cap → truncation is real.
            truncated = True
            break
        count += 1
        try:
            total += path.stat().st_size
        except OSError:
            # Broken file or perms — count it, skip the size.
            continue
    return count, total, truncated


def _top_level_paths(workdir: Path, limit: int) -> list[str]:
    """Return up to *limit* top-level entries sorted alphabetically.

    Symlinks at the top level are skipped (same rationale as
    ``_enumerate``). Directories get a trailing slash so the UI can render
    them distinctly.
    """
    entries: list[str] = []
    for entry in sorted(workdir.iterdir(), key=lambda p: p.name):
        if entry.name == ".git":
            continue
        if entry.is_symlink():
            continue
        entries.append(f"{entry.name}/" if entry.is_dir() else entry.name)
        if len(entries) >= limit:
            break
    return entries
