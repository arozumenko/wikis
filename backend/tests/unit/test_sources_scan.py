"""#207 — POST /api/v1/sources/scan validates + previews a source.

Covers the scan service directly (clone path, file enumeration, top-paths,
tmpdir cleanup-on-raise, total-time budget, local-path allowlist) and the
route (status codes, redacted error shape, 501 for unimplemented
connectors). All network-touching paths are mocked so the suite is fast
and deterministic in CI.
"""

from __future__ import annotations

import asyncio
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
from app.core.sources.git_toolkit import GitToolkit
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


@pytest.fixture
def fake_repo():
    """Yields a fresh on-disk repo path with ALLOWED_LOCAL_PATHS configured."""
    repo = _make_fake_git_repo()
    with patch("app.services.source_scan_service.get_settings") as get_settings:
        get_settings.return_value = MagicMock(allowed_local_paths=str(repo.parent))
        yield repo
    shutil.rmtree(repo, ignore_errors=True)


# ---------------------------------------------------------------------------
# Service-level: happy path against a real local repo
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_git_scan_happy_path(fake_repo) -> None:
    service = SourceScanService()
    request = ScanRequest(
        source_type="git",
        scope={"repo_url": f"file://{fake_repo}", "branch": "main"},
        auth={"pat": None},
    )
    resp = await service.scan(request)

    assert resp.source_type == "git"
    assert resp.reachable is True
    assert resp.preview is not None
    # C7 — for local repos, resolved_branch is read from extract_git_metadata.
    assert resp.preview.resolved_branch == "main"
    assert resp.preview.commit_hash and len(resp.preview.commit_hash) >= 7
    # README.md + src/main.py + docs/intro.md
    assert resp.preview.file_count == 3
    assert resp.preview.size_bytes > 0
    # Top-level paths reported as README.md, src/, docs/ (dirs get trailing slash).
    assert "README.md" in resp.preview.top_paths
    assert "src/" in resp.preview.top_paths
    assert "docs/" in resp.preview.top_paths
    # .git excluded everywhere.
    assert ".git" not in resp.preview.top_paths
    assert ".git/" not in resp.preview.top_paths
    assert resp.warnings == []


# ---------------------------------------------------------------------------
# C1 — validate_local_path enforcement on local paths
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_local_path_rejected_when_allowlist_unset() -> None:
    """File-system access without ``ALLOWED_LOCAL_PATHS`` must be refused.

    This is the security regression Copilot caught: without the gate, a
    crafted ``file:///etc`` payload would let the toolkit walk host dirs.
    """
    with patch("app.services.source_scan_service.get_settings") as get_settings:
        get_settings.return_value = MagicMock(allowed_local_paths=None)
        service = SourceScanService()
        with pytest.raises(ScanError) as exc_info:
            await service.scan(
                ScanRequest(
                    source_type="git",
                    scope={"repo_url": "file:///etc", "branch": "main"},
                )
            )
    assert "Local path access is disabled" in str(exc_info.value)
    assert exc_info.value.reachable is False


@pytest.mark.asyncio
async def test_bare_slash_local_path_also_gated() -> None:
    """``/etc`` (no ``file://`` prefix) must hit the same allowlist gate.

    ``is_local_path`` accepts both forms; the gate must cover both.
    """
    with patch("app.services.source_scan_service.get_settings") as get_settings:
        get_settings.return_value = MagicMock(allowed_local_paths=None)
        service = SourceScanService()
        with pytest.raises(ScanError):
            await service.scan(
                ScanRequest(
                    source_type="git",
                    scope={"repo_url": "/etc", "branch": "main"},
                )
            )


# ---------------------------------------------------------------------------
# C5 — mocked failure paths (no real network)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_unreachable_url_raises_scan_error() -> None:
    """SourceUnavailableError from the toolkit → ScanError (no real network)."""
    service = SourceScanService()
    with patch.object(
        GitToolkit,
        "test_connection",
        new_callable=AsyncMock,
        side_effect=SourceUnavailableError("Timed out connecting to https://x/y"),
    ):
        with pytest.raises(ScanError) as exc_info:
            await service.scan(
                ScanRequest(
                    source_type="git",
                    scope={"repo_url": "https://x/y", "branch": "main"},
                )
            )
    assert exc_info.value.reachable is False


@pytest.mark.asyncio
async def test_not_found_translates_to_scan_error() -> None:
    service = SourceScanService()
    with patch.object(
        GitToolkit,
        "test_connection",
        new_callable=AsyncMock,
        side_effect=SourceNotFoundError("Repository not found: https://x/y"),
    ):
        with pytest.raises(ScanError):
            await service.scan(
                ScanRequest(
                    source_type="git",
                    scope={"repo_url": "https://x/y", "branch": "main"},
                )
            )


@pytest.mark.asyncio
async def test_clone_failure_cleans_up_tmpdir() -> None:
    """``_ensure_workdir`` raise must still cause ``close()`` to remove the tmpdir.

    Uses the toolkit's real context manager but stubs the subprocess so
    the cleanup contract is exercised without any network call.
    """
    service = SourceScanService()
    before = {
        p for p in os.listdir(tempfile.gettempdir()) if p.startswith("git_toolkit_")
    }
    with patch.object(
        GitToolkit,
        "test_connection",
        new_callable=AsyncMock,
        return_value="ok",
    ):
        with patch.object(
            GitToolkit,
            "_ensure_workdir",
            side_effect=SourceUnavailableError("clone failed"),
        ):
            with pytest.raises(ScanError):
                await service.scan(
                    ScanRequest(
                        source_type="git",
                        scope={"repo_url": "https://x/y", "branch": "main"},
                    )
                )
    after = {
        p for p in os.listdir(tempfile.gettempdir()) if p.startswith("git_toolkit_")
    }
    assert after == before, f"leaked tmpdirs: {after - before}"


# ---------------------------------------------------------------------------
# C9 — actual redaction is exercised (token must not appear in the error)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_token_never_leaks_through_scan_error() -> None:
    """The toolkit's ``_sanitise`` strips the raw PAT from any error it raises.

    The earlier version of this test passed even when redaction was broken
    because the mocked exception message never contained the token. This
    version uses the toolkit's real sanitisation path: we instantiate a
    real GitToolkit with a known token, then force its underlying clone to
    fail with a message that includes the token; the toolkit must scrub
    the token before raising.
    """
    secret = "ghp_VERY_SECRET_TOKEN_THAT_MUST_NOT_LEAK"
    tk = GitToolkit(
        repo_url="https://github.com/owner/private", branch="main", token=secret
    )
    # Force ls-remote to "fail" with stderr containing the raw token.
    fake_result = MagicMock()
    fake_result.returncode = 128
    fake_result.stderr = f"fatal: Authentication failed for {secret}"
    with patch(
        "app.core.sources.git_toolkit.subprocess.run", return_value=fake_result
    ):
        with pytest.raises(SourceAuthError) as exc_info:
            await tk.test_connection()
    msg = str(exc_info.value)
    assert secret not in msg, f"raw token leaked: {msg!r}"
    assert "***" in msg, "redaction marker should appear where token was"


# ---------------------------------------------------------------------------
# C6 — type validation on scope fields
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_missing_repo_url_raises_scan_error() -> None:
    service = SourceScanService()
    with pytest.raises(ScanError):
        await service.scan(ScanRequest(source_type="git", scope={}))


@pytest.mark.asyncio
async def test_non_string_repo_url_raises_scan_error() -> None:
    """Non-string repo_url must surface as 400 / ScanError, not 500."""
    service = SourceScanService()
    with pytest.raises(ScanError) as exc_info:
        await service.scan(
            ScanRequest(source_type="git", scope={"repo_url": 123, "branch": "main"})
        )
    assert "non-empty string" in str(exc_info.value)


# ---------------------------------------------------------------------------
# C3 — truncation off-by-one
# ---------------------------------------------------------------------------


def test_enumerate_does_not_warn_when_exactly_at_cap(tmp_path) -> None:
    """At exactly ``cap`` files, the warning must NOT fire."""
    from app.services.source_scan_service import _enumerate

    for i in range(5):
        (tmp_path / f"f{i}.txt").write_text("x")
    count, _, truncated = _enumerate(tmp_path, cap=5)
    assert count == 5
    assert truncated is False


def test_enumerate_warns_when_strictly_over_cap(tmp_path) -> None:
    from app.services.source_scan_service import _enumerate

    for i in range(7):
        (tmp_path / f"f{i}.txt").write_text("x")
    count, _, truncated = _enumerate(tmp_path, cap=5)
    assert count == 5  # we stop at the cap
    assert truncated is True


# ---------------------------------------------------------------------------
# C4 — symlinks are not followed by the preview
# ---------------------------------------------------------------------------


def test_enumerate_skips_symlinks(tmp_path) -> None:
    """A symlink to a host file must not be counted into the preview."""
    from app.services.source_scan_service import _enumerate

    (tmp_path / "real.txt").write_text("hello")
    # Symlink pointing outside the repo — would be /etc/passwd in the wild.
    outside = tmp_path.parent / "outside.txt"
    outside.write_text("secret host content")
    (tmp_path / "leak.txt").symlink_to(outside)

    count, total, _ = _enumerate(tmp_path, cap=100)
    assert count == 1  # only real.txt
    assert total == len("hello")  # symlink target's bytes excluded


def test_top_level_paths_skips_symlinks(tmp_path) -> None:
    from app.services.source_scan_service import _top_level_paths

    (tmp_path / "real.txt").write_text("x")
    outside = tmp_path.parent / "elsewhere"
    outside.mkdir(exist_ok=True)
    (tmp_path / "link").symlink_to(outside)
    paths = _top_level_paths(tmp_path, limit=10)
    assert "real.txt" in paths
    assert "link" not in paths
    assert "link/" not in paths


# ---------------------------------------------------------------------------
# C8 — total-time budget terminates a hung scan
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_scan_budget_cancels_hung_scan(monkeypatch) -> None:
    """A scan that exceeds ``_SCAN_BUDGET_SEC`` must raise ScanError, not hang."""

    async def _slow_scan(self, request):
        await asyncio.sleep(10)
        raise AssertionError("should have been cancelled")

    service = SourceScanService()
    monkeypatch.setattr(service, "_SCAN_BUDGET_SEC", 0.05)
    monkeypatch.setattr(SourceScanService, "_scan_git", _slow_scan)

    with pytest.raises(ScanError) as exc_info:
        await service.scan(
            ScanRequest(source_type="git", scope={"repo_url": "x", "branch": "main"})
        )
    assert "budget" in str(exc_info.value)
    assert exc_info.value.reachable is False


# ---------------------------------------------------------------------------
# Service-level: 501 for unimplemented connectors
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_confluence_scan_raises_not_implemented() -> None:
    service = SourceScanService()
    with pytest.raises(NotImplementedError):
        await service.scan(
            ScanRequest(source_type="confluence", scope={"base_url": "x"})
        )


@pytest.mark.asyncio
async def test_jira_scan_raises_not_implemented() -> None:
    service = SourceScanService()
    with pytest.raises(NotImplementedError):
        await service.scan(
            ScanRequest(source_type="jira", scope={"base_url": "x", "jql": "y"})
        )


# ---------------------------------------------------------------------------
# Route-level: status codes + error shape
# ---------------------------------------------------------------------------


def test_route_returns_200_on_happy_path(fake_repo) -> None:
    app = _build_app()
    with TestClient(app) as client:
        resp = client.post(
            "/api/v1/sources/scan",
            json={
                "source_type": "git",
                "scope": {"repo_url": f"file://{fake_repo}", "branch": "main"},
                "auth": {"pat": None},
            },
        )
    assert resp.status_code == 200
    body = resp.json()
    assert body["source_type"] == "git"
    assert body["reachable"] is True
    assert body["preview"]["file_count"] == 3


def test_route_returns_400_on_scan_error() -> None:
    """Route translates ScanError → 400 with the documented body shape."""
    app = _build_app()
    with patch(
        "app.services.source_scan_service.SourceScanService.scan",
        new_callable=AsyncMock,
        side_effect=ScanError("Repository not found: x/y", reachable=False),
    ):
        with TestClient(app) as client:
            resp = client.post(
                "/api/v1/sources/scan",
                json={
                    "source_type": "git",
                    "scope": {"repo_url": "https://x/y", "branch": "main"},
                },
            )
    assert resp.status_code == 400
    body = resp.json()
    assert body["detail"]["reachable"] is False
    assert "Repository not found" in body["detail"]["error"]


def test_route_returns_501_for_confluence() -> None:
    app = _build_app()
    with TestClient(app) as client:
        resp = client.post(
            "/api/v1/sources/scan",
            json={
                "source_type": "confluence",
                "scope": {"base_url": "https://x.atlassian.net"},
            },
        )
    assert resp.status_code == 501


def test_route_returns_501_for_jira() -> None:
    app = _build_app()
    with TestClient(app) as client:
        resp = client.post(
            "/api/v1/sources/scan",
            json={
                "source_type": "jira",
                "scope": {"base_url": "https://x.atlassian.net", "jql": "x"},
            },
        )
    assert resp.status_code == 501
