"""#207 — POST /api/v1/sources/scan validates + previews a source.

Covers the scan service directly (clone path, file enumeration, top-paths,
tmpdir cleanup-on-raise) and the route (status codes, redacted error shape,
501 for unimplemented connectors).
"""

from __future__ import annotations

import os
import shutil
import subprocess
import tempfile
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.api.routes import router
from app.auth import get_current_user
from app.core.sources.exceptions import (
    SourceAuthError,
    SourceNotFoundError,
    SourceUnavailableError,
)
from app.models.api import ScanRequest
from app.services.source_scan_service import ScanError, SourceScanService


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _build_app():
    app = FastAPI()
    app.include_router(router)
    app.dependency_overrides[get_current_user] = lambda: MagicMock(id="user-1")
    return app


def _make_fake_git_repo() -> Path:
    """Create a real on-disk git repo with a few files for end-to-end scans.

    Faster and more honest than mocking subprocess.run — the GitToolkit
    shells out to git for clone, ls-remote, rev-parse; mocking that chain
    would test the mock more than the code. A tiny local repo exercises the
    actual code path.
    """
    src = Path(tempfile.mkdtemp(prefix="scan_src_"))
    (src / "README.md").write_text("# fake repo")
    (src / "src").mkdir()
    (src / "src" / "main.py").write_text("print('hi')\n")
    (src / "docs").mkdir()
    (src / "docs" / "intro.md").write_text("hello\n")
    subprocess.run(["git", "init", "-q", "--initial-branch=main"], cwd=src, check=True)
    subprocess.run(["git", "config", "user.email", "t@t.t"], cwd=src, check=True)
    subprocess.run(["git", "config", "user.name", "t"], cwd=src, check=True)
    subprocess.run(["git", "add", "."], cwd=src, check=True)
    subprocess.run(
        ["git", "commit", "-q", "-m", "init", "--no-gpg-sign"], cwd=src, check=True
    )
    return src


# ---------------------------------------------------------------------------
# Service-level: happy path against a real local repo
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_git_scan_happy_path() -> None:
    repo = _make_fake_git_repo()
    try:
        service = SourceScanService()
        request = ScanRequest(
            source_type="git",
            scope={"repo_url": f"file://{repo}", "branch": "main"},
            auth={"pat": None},
        )
        resp = await service.scan(request)

        assert resp.source_type == "git"
        assert resp.reachable is True
        assert resp.preview is not None
        assert resp.preview.resolved_branch == "main"
        # README.md + src/main.py + docs/intro.md
        assert resp.preview.file_count == 3
        # File-byte total non-zero
        assert resp.preview.size_bytes > 0
        # Top-level paths reported as a/, b/, file (dirs get trailing slash)
        assert "README.md" in resp.preview.top_paths
        assert "src/" in resp.preview.top_paths
        assert "docs/" in resp.preview.top_paths
        # `.git` excluded from top paths
        assert ".git" not in resp.preview.top_paths
        assert ".git/" not in resp.preview.top_paths
        assert resp.warnings == []
    finally:
        shutil.rmtree(repo, ignore_errors=True)


# ---------------------------------------------------------------------------
# Service-level: error translations
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_unreachable_url_raises_scan_error() -> None:
    """ls-remote against a nonexistent URL → SourceUnavailableError → ScanError."""
    service = SourceScanService()
    request = ScanRequest(
        source_type="git",
        scope={"repo_url": "https://example.invalid/does/not/exist", "branch": "main"},
    )
    with pytest.raises(ScanError) as exc_info:
        await service.scan(request)
    assert exc_info.value.reachable is False


@pytest.mark.asyncio
async def test_auth_error_translates_to_scan_error() -> None:
    """SourceAuthError from test_connection → ScanError, never leaks the token."""
    service = SourceScanService()
    request = ScanRequest(
        source_type="git",
        scope={"repo_url": "https://github.com/owner/private", "branch": "main"},
        auth={"pat": "ghp_SECRET"},
    )
    with patch(
        "app.services.source_scan_service.GitToolkit.test_connection",
        new_callable=AsyncMock,
        side_effect=SourceAuthError("Auth failure for https://github.com/owner/private"),
    ):
        with pytest.raises(ScanError) as exc_info:
            await service.scan(request)
    assert exc_info.value.reachable is False
    # Critical: the redacted token string must not leak through the message.
    assert "ghp_SECRET" not in str(exc_info.value)


@pytest.mark.asyncio
async def test_missing_repo_url_raises() -> None:
    service = SourceScanService()
    with pytest.raises(ScanError):
        await service.scan(ScanRequest(source_type="git", scope={}))


@pytest.mark.asyncio
async def test_tmpdir_cleaned_up_on_clone_failure() -> None:
    """If ``_ensure_workdir`` raises mid-clone, the tmpdir must still be removed.

    ``GitToolkit`` owns this via ``async with`` + ``close()``; the scan
    service must use the context manager so the guarantee carries over.
    """
    service = SourceScanService()
    request = ScanRequest(
        source_type="git",
        scope={"repo_url": "https://example.invalid/x/y", "branch": "main"},
    )

    # Snapshot existing /tmp/git_toolkit_* dirs so we can assert nothing
    # is left over after the failure.
    before = {p for p in os.listdir(tempfile.gettempdir()) if p.startswith("git_toolkit_")}
    with pytest.raises(ScanError):
        await service.scan(request)
    after = {p for p in os.listdir(tempfile.gettempdir()) if p.startswith("git_toolkit_")}
    assert after == before, f"leaked tmpdirs: {after - before}"


# ---------------------------------------------------------------------------
# Service-level: 501 for unimplemented connectors
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_confluence_scan_raises_not_implemented() -> None:
    service = SourceScanService()
    with pytest.raises(NotImplementedError):
        await service.scan(ScanRequest(source_type="confluence", scope={"base_url": "x"}))


@pytest.mark.asyncio
async def test_jira_scan_raises_not_implemented() -> None:
    service = SourceScanService()
    with pytest.raises(NotImplementedError):
        await service.scan(ScanRequest(source_type="jira", scope={"base_url": "x", "jql": "y"}))


# ---------------------------------------------------------------------------
# Route-level: status codes + error shape
# ---------------------------------------------------------------------------


def test_route_returns_200_on_happy_path() -> None:
    repo = _make_fake_git_repo()
    try:
        app = _build_app()
        with TestClient(app) as client:
            resp = client.post(
                "/api/v1/sources/scan",
                json={
                    "source_type": "git",
                    "scope": {"repo_url": f"file://{repo}", "branch": "main"},
                    "auth": {"pat": None},
                },
            )
        assert resp.status_code == 200
        body = resp.json()
        assert body["source_type"] == "git"
        assert body["reachable"] is True
        assert body["preview"]["file_count"] == 3
    finally:
        shutil.rmtree(repo, ignore_errors=True)


def test_route_returns_400_on_scan_error() -> None:
    app = _build_app()
    with TestClient(app) as client:
        resp = client.post(
            "/api/v1/sources/scan",
            json={
                "source_type": "git",
                "scope": {"repo_url": "https://example.invalid/x/y", "branch": "main"},
            },
        )
    assert resp.status_code == 400
    body = resp.json()
    assert body["detail"]["reachable"] is False
    assert "error" in body["detail"]


def test_route_returns_501_for_confluence() -> None:
    app = _build_app()
    with TestClient(app) as client:
        resp = client.post(
            "/api/v1/sources/scan",
            json={"source_type": "confluence", "scope": {"base_url": "https://x.atlassian.net"}},
        )
    assert resp.status_code == 501


def test_route_returns_501_for_jira() -> None:
    app = _build_app()
    with TestClient(app) as client:
        resp = client.post(
            "/api/v1/sources/scan",
            json={"source_type": "jira", "scope": {"base_url": "https://x.atlassian.net", "jql": "x"}},
        )
    assert resp.status_code == 501
