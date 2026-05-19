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
from app.core.sources._atlassian import AtlassianClient
from app.core.sources.exceptions import (
    SourceAuthError,
    SourceConnectionError,
    SourceNotFoundError,
    SourceUnavailableError,
)
from app.core.sources.git_toolkit import GitToolkit
from app.models.api import (
    AtlassianAuth,
    ConfluenceScanPreview,
    ConfluenceSpaceInfo,
    ConfluenceScanRequest,
    GitScanPreview,
    GitScanRequest,
    JiraScanPreview,
    JiraScanRequest,
    ScanRequest,
    ScanResponse,
)

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
            scanner = self._scan_git(request)
        elif request.source_type == "confluence":
            scanner = self._scan_confluence(request)
        elif request.source_type == "jira":
            scanner = self._scan_jira(request)
        else:
            raise NotImplementedError(
                f"Scan not implemented for source_type={request.source_type!r}"
            )

        try:
            return await asyncio.wait_for(scanner, timeout=self._SCAN_BUDGET_SEC)
        except asyncio.TimeoutError as exc:
            raise ScanError(
                f"Scan exceeded the {self._SCAN_BUDGET_SEC:.0f}s budget — "
                "the source may be slow or unreachable",
                reachable=False,
            ) from exc

    async def _scan_git(self, request: GitScanRequest) -> ScanResponse:
        # Pydantic enforces `repo_url: str` is present and the right type, but
        # an *empty* string is still a valid `str` — the guard below is the
        # actual enforcement layer for non-emptiness (Rio #224 review).
        repo_url = request.scope.repo_url
        if not repo_url:
            raise ScanError("scope.repo_url must be a non-empty string for git scans")
        branch = request.scope.branch or "main"
        token = request.auth.pat

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

            # Shallow clone (depth=1, single-branch). The toolkit's async
            # ``_ensure_workdir`` (#214) owns the asyncio.to_thread offloading
            # internally, so callers just await it directly. The async
            # context manager still guarantees tmpdir cleanup via ``close()``
            # on exit, even if enumeration below raises.
            try:
                await toolkit._ensure_workdir()  # noqa: SLF001
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


    async def _scan_confluence(self, request: ConfluenceScanRequest) -> ScanResponse:
        """Scan a Confluence instance and return a space-level preview.

        Page-count strategy: ``GET /wiki/api/v2/spaces/{key}/pages?limit=1``
        does not return a reliable total in the v2 API — the v2 cursor-based
        pagination omits ``total_size``.  We fall back to the v1 CQL content
        search (``GET /wiki/rest/api/content/search`` with ``cqlcontext``), which
        does return ``totalSize`` in its envelope.  If that field is missing (e.g.
        filtered by permissions), ``page_count`` is left as ``None`` so the UI can
        render ``"?"``.

        scope.space_keys is validated as list[str] by ConfluenceScope; empty
        list means "all spaces". Per-element non-emptiness is not enforced
        by Pydantic — the C1 guard below is the actual element-level check.
        """
        # Pydantic enforces the field shape (str / list[str]) but accepts
        # empty strings — the guards below are the actual non-empty checks
        # (Rio #224 review).
        base_url = request.scope.base_url
        if not base_url:
            raise ScanError(
                "scope.base_url must be a non-empty string for confluence scans"
            )
        requested_keys: list[str] = list(request.scope.space_keys)
        # C1 guard still applies: Pydantic validates list[str] but empty
        # strings are allowed by the type system — reject them explicitly.
        if not all(isinstance(k, str) and k for k in requested_keys):
            raise ScanError(
                "scope.space_keys must be a list of non-empty strings"
            )

        client_kwargs = _atlassian_client_kwargs(request.auth)

        try:
            async with AtlassianClient(base_url=base_url, **client_kwargs) as client:
                # Fetch spaces — filter to requested keys when provided.
                spaces_data = await client.get(
                    "/wiki/rest/api/space",
                    params={"limit": 250, "type": "global"},
                )
                all_spaces = spaces_data.get("results", [])

                if requested_keys:
                    key_set = set(requested_keys)
                    all_spaces = [s for s in all_spaces if s.get("key") in key_set]

                space_infos: list[ConfluenceSpaceInfo] = []
                for space in all_spaces:
                    key = space.get("key", "")
                    name = space.get("name", key)
                    page_count = await _confluence_page_count(client, key)
                    space_infos.append(
                        ConfluenceSpaceInfo(key=key, name=name, page_count=page_count)
                    )

        except SourceAuthError as exc:
            raise ScanError(str(exc), reachable=False) from exc
        except (SourceNotFoundError, SourceUnavailableError, SourceConnectionError) as exc:
            raise ScanError(str(exc), reachable=False) from exc

        # C2+C5 — if ALL spaces have a known page count, sum them. If ANY are
        # None (permission-filtered or API didn't return totalSize), report None
        # so the wizard renders "?" rather than a silently undercounted total.
        if all(s.page_count is not None for s in space_infos):
            total_pages: int | None = sum(s.page_count for s in space_infos)  # type: ignore[misc]
        else:
            total_pages = None
        preview = ConfluenceScanPreview(spaces=space_infos, total_pages=total_pages)
        return ScanResponse(
            source_type="confluence",
            reachable=True,
            preview=preview,
        )

    async def _scan_jira(self, request: JiraScanRequest) -> ScanResponse:
        """Scan a Jira instance, validate JQL, and return an issue-count preview.

        Uses ``GET /rest/api/3/search`` with ``maxResults=3`` — the search
        endpoint validates JQL and returns ``total`` even when only a small
        slice of issues is requested.  A JQL syntax error surfaces as a 400
        from Jira, which the AtlassianClient maps to ``SourceConnectionError``
        (HTTP 4xx catchall). We re-raise that as ``ScanError(reachable=False)``
        because the scan itself failed — the site may well be reachable, but
        reporting ``reachable=True`` when the preview cannot be computed would
        mislead the wizard into advancing to the next step.

        scope.base_url and scope.jql are typed as `str` by JiraScope —
        Pydantic enforces the type but accepts empty strings. The guards
        below are the actual non-empty checks (Rio #224 review).
        """
        # See class docstring on the empty-string semantics.
        base_url = request.scope.base_url
        if not base_url:
            raise ScanError(
                "scope.base_url must be a non-empty string for jira scans"
            )
        jql = request.scope.jql
        if not jql:
            raise ScanError("scope.jql must be a non-empty string")

        client_kwargs = _atlassian_client_kwargs(request.auth)

        try:
            async with AtlassianClient(base_url=base_url, **client_kwargs) as client:
                data = await client.get(
                    "/rest/api/3/search",
                    params={
                        "jql": jql,
                        "maxResults": 3,
                        "fields": "summary",
                    },
                )
        except SourceAuthError as exc:
            raise ScanError(str(exc), reachable=False) from exc
        except (
            SourceNotFoundError,
            SourceUnavailableError,
            SourceConnectionError,
        ) as exc:
            # SourceConnectionError covers Jira's 400 for invalid JQL — the
            # AtlassianClient maps all 4xx (other than 401/403/404) to this
            # exception.  We treat a failed scan as reachable=False: the wizard
            # should not advance to the next step when the preview is absent.
            raise ScanError(str(exc), reachable=False) from exc

        total = data.get("total", 0)
        sample_keys = [issue["key"] for issue in data.get("issues", [])]
        preview = JiraScanPreview(
            matching_issues=total,
            sample_issue_keys=sample_keys,
            jql_validated=True,
        )
        return ScanResponse(
            source_type="jira",
            reachable=True,
            preview=preview,
        )


def _atlassian_client_kwargs(auth: AtlassianAuth) -> dict[str, str | None]:
    """Convert an AtlassianAuth into the kwargs ``AtlassianClient`` accepts.

    Returns either the OAuth bundle (``access_token`` + optional
    ``refresh_token`` / ``client_id``) or the basic-auth bundle
    (``email`` + ``api_token``).  The two shapes are mutually exclusive —
    ``AtlassianAuth``'s model validator already enforced that, so we
    just pick the correct one here.
    """
    if auth.uses_basic_auth:
        return {"email": auth.email, "api_token": auth.api_token}
    return {
        "access_token": auth.access_token,
        "refresh_token": auth.refresh_token or None,
        "client_id": auth.client_id or None,
    }


async def _confluence_page_count(client: AtlassianClient, space_key: str) -> int | None:
    """Return the total page count for *space_key* or ``None`` when unavailable.

    Uses the v1 CQL content search which reliably returns ``totalSize`` in its
    envelope.  ``limit=1`` minimises data transfer — we only need the count.
    """
    try:
        data = await client.get(
            "/wiki/rest/api/content/search",
            params={
                "cql": f'space.key="{space_key}" AND type=page',
                "limit": 1,
            },
        )
    except (SourceNotFoundError, SourceUnavailableError, SourceConnectionError):
        return None

    total = data.get("totalSize")
    if total is None:
        return None
    try:
        return int(total)
    except (TypeError, ValueError):
        return None


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
