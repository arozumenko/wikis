"""#215 — discriminated-union scope/auth models for ScanRequest and GenerateWikiRequest.

Two acceptance tests that pin the new discrimination behavior:
  (i)  invalid combo (git source_type + AtlassianAuth fields) → 422 at the route.
  (ii) new typed-input (GitScope/GitAuth instances) still works end-to-end at the route.

Additional unit tests:
  - ScanRequest with typed model instances round-trips correctly.
  - AtlassianAuth / GitAuth field validation.
  - GenerateWikiRequest discriminated union works for all three source types.
  - Legacy flat-fields backward compat in GenerateWikiRequest still works.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from pydantic import ValidationError

from app.api.routes import router
from app.auth import get_current_user
from app.models.api import (
    AtlassianAuth,
    ConfluenceScanRequest,
    ConfluenceScope,
    GenerateWikiRequest,
    GitAuth,
    GitScanRequest,
    GitScope,
    JiraScanRequest,
    JiraScope,
    ScanRequest,
)


def _build_app():
    app = FastAPI()
    app.include_router(router)
    app.dependency_overrides[get_current_user] = lambda: MagicMock(id="user-1")
    return app


# ---------------------------------------------------------------------------
# (i) AC: invalid combo → 422 at the route
# ---------------------------------------------------------------------------


def test_route_rejects_git_source_with_atlassian_auth_fields() -> None:
    """git source_type + access_token (AtlassianAuth) must yield 422 at the route.

    After the discriminated union, GitScanRequest only accepts GitAuth
    (``pat`` field), not AtlassianAuth (``access_token`` field). Passing
    ``access_token`` to a ``source_type=git`` request must be rejected by
    Pydantic before the service layer is reached.
    """
    app = _build_app()
    with TestClient(app) as client:
        resp = client.post(
            "/api/v1/sources/scan",
            json={
                "source_type": "git",
                "scope": {"repo_url": "https://github.com/org/repo", "branch": "main"},
                # AtlassianAuth shape sent for a git source — must be rejected
                "auth": {"access_token": "at_xxx", "refresh_token": "rt_yyy"},
            },
        )
    assert resp.status_code == 422, (
        f"Expected 422 for git+AtlassianAuth combo, got {resp.status_code}: {resp.text}"
    )


def test_route_rejects_confluence_source_with_git_scope() -> None:
    """confluence source_type + repo_url scope must yield 422 at the route.

    ConfluenceScanRequest requires ConfluenceScope (base_url) not GitScope
    (repo_url). Passing the wrong scope shape must be caught by Pydantic.
    """
    app = _build_app()
    with TestClient(app) as client:
        resp = client.post(
            "/api/v1/sources/scan",
            json={
                "source_type": "confluence",
                # GitScope shape sent for a confluence source — must be rejected
                "scope": {"repo_url": "https://github.com/org/repo", "branch": "main"},
                "auth": {"access_token": "at_xxx"},
            },
        )
    assert resp.status_code == 422, (
        f"Expected 422 for confluence+GitScope combo, got {resp.status_code}: {resp.text}"
    )


# ---------------------------------------------------------------------------
# (ii) AC: new typed-input still works at the route
# ---------------------------------------------------------------------------


def test_route_accepts_typed_git_scan_request(fake_repo_for_discrimination) -> None:
    """New typed GitScanRequest (with GitScope + GitAuth) → 200 at the route."""
    app = _build_app()
    with TestClient(app) as client:
        resp = client.post(
            "/api/v1/sources/scan",
            json={
                "source_type": "git",
                "scope": {
                    "repo_url": f"file://{fake_repo_for_discrimination}",
                    "branch": "main",
                },
                "auth": {"pat": None},
            },
        )
    assert resp.status_code == 200, (
        f"Expected 200 for valid GitScanRequest, got {resp.status_code}: {resp.text}"
    )
    body = resp.json()
    assert body["source_type"] == "git"
    assert body["reachable"] is True


# ---------------------------------------------------------------------------
# Unit: typed model instances round-trip correctly (no route involved)
# ---------------------------------------------------------------------------


def test_git_scan_request_accepts_typed_scope_and_auth() -> None:
    """GitScanRequest constructed with GitScope + GitAuth instances is valid."""
    req = GitScanRequest(
        scope=GitScope(repo_url="https://github.com/org/repo", branch="develop"),
        auth=GitAuth(pat="ghp_token"),
    )
    assert req.source_type == "git"
    assert req.scope.repo_url == "https://github.com/org/repo"
    assert req.scope.branch == "develop"
    assert req.auth.pat == "ghp_token"


def test_git_scan_request_accepts_dict_scope() -> None:
    """GitScanRequest constructed with dict scope is coerced into GitScope."""
    req = GitScanRequest(
        scope={"repo_url": "https://github.com/org/repo", "branch": "main"},
    )
    assert req.scope.repo_url == "https://github.com/org/repo"
    assert req.scope.branch == "main"
    assert req.auth.pat is None  # default


def test_confluence_scan_request_accepts_typed_scope_and_auth() -> None:
    """ConfluenceScanRequest constructed with ConfluenceScope + AtlassianAuth is valid."""
    req = ConfluenceScanRequest(
        scope=ConfluenceScope(base_url="https://acme.atlassian.net", space_keys=["ENG"]),
        auth=AtlassianAuth(access_token="at_xxx", refresh_token="rt_yyy", client_id="cid"),
    )
    assert req.source_type == "confluence"
    assert req.scope.base_url == "https://acme.atlassian.net"
    assert req.scope.space_keys == ["ENG"]
    assert req.auth.access_token == "at_xxx"


def test_jira_scan_request_accepts_typed_scope_and_auth() -> None:
    """JiraScanRequest constructed with JiraScope + AtlassianAuth is valid."""
    req = JiraScanRequest(
        scope=JiraScope(base_url="https://acme.atlassian.net", jql="project=ENG"),
        auth=AtlassianAuth(access_token="at_aaa"),
    )
    assert req.source_type == "jira"
    assert req.scope.jql == "project=ENG"
    assert req.auth.refresh_token is None  # default


def test_scan_request_discriminates_by_source_type() -> None:
    """ScanRequest (discriminated union) dispatches to the correct variant."""
    from pydantic import TypeAdapter

    ta = TypeAdapter(ScanRequest)

    git_req = ta.validate_python(
        {
            "source_type": "git",
            "scope": {"repo_url": "https://github.com/org/repo", "branch": "main"},
        }
    )
    assert isinstance(git_req, GitScanRequest)

    conf_req = ta.validate_python(
        {
            "source_type": "confluence",
            "scope": {"base_url": "https://acme.atlassian.net"},
            "auth": {"access_token": "at_xxx"},
        }
    )
    assert isinstance(conf_req, ConfluenceScanRequest)

    jira_req = ta.validate_python(
        {
            "source_type": "jira",
            "scope": {"base_url": "https://acme.atlassian.net", "jql": "project=ENG"},
            "auth": {"access_token": "at_yyy"},
        }
    )
    assert isinstance(jira_req, JiraScanRequest)


def test_git_scope_requires_repo_url() -> None:
    """GitScope rejects construction without repo_url."""
    with pytest.raises(ValidationError):
        GitScope(branch="main")  # type: ignore[call-arg]  # missing required repo_url


def test_atlassian_auth_requires_access_token() -> None:
    """AtlassianAuth rejects construction without access_token."""
    with pytest.raises(ValidationError):
        AtlassianAuth()  # type: ignore[call-arg]  # missing required access_token


def test_jira_scope_requires_jql() -> None:
    """JiraScope rejects construction without jql."""
    with pytest.raises(ValidationError):
        JiraScope(base_url="https://acme.atlassian.net")  # type: ignore[call-arg]


# ---------------------------------------------------------------------------
# GenerateWikiRequest: discriminated union + legacy backward compat
# ---------------------------------------------------------------------------


def test_generate_wiki_request_git_typed() -> None:
    """GenerateWikiRequest with typed GitScope/GitAuth works correctly."""
    req = GenerateWikiRequest(
        source_type="git",
        scope={"repo_url": "https://github.com/org/repo", "branch": "feat/215"},
        auth={"pat": "ghp_abc"},
    )
    assert req.source_type == "git"
    # scope is stored as a dict (GenerateWikiRequest keeps dict shape for legacy compat)
    assert req.scope["repo_url"] == "https://github.com/org/repo"
    # Back-filled legacy fields must still work
    assert req.repo_url == "https://github.com/org/repo"
    assert req.branch == "feat/215"
    assert req.access_token == "ghp_abc"


def test_generate_wiki_request_confluence_typed() -> None:
    """GenerateWikiRequest with ConfluenceScope shape works correctly."""
    req = GenerateWikiRequest(
        source_type="confluence",
        scope={"base_url": "https://acme.atlassian.net", "space_keys": ["ENG"]},
        auth={"access_token": "at_xxx", "refresh_token": "rt_yyy"},
    )
    assert req.source_type == "confluence"
    assert req.scope["space_keys"] == ["ENG"]
    # Stub back-fill: repo_url set to base_url for serialisation compat
    assert req.repo_url == "https://acme.atlassian.net"


def test_generate_wiki_request_jira_typed() -> None:
    """GenerateWikiRequest with JiraScope shape works correctly."""
    req = GenerateWikiRequest(
        source_type="jira",
        scope={"base_url": "https://acme.atlassian.net", "jql": "project=ENG"},
        auth={"access_token": "at_aaa"},
    )
    assert req.source_type == "jira"
    assert req.scope["jql"] == "project=ENG"


def test_generate_wiki_request_legacy_flat_fields_still_work() -> None:
    """Legacy repo_url+branch+access_token flat fields must still normalize correctly.

    This is the backward-compat contract: the _normalize_legacy validator
    must survive the discriminated-union migration unchanged.
    """
    req = GenerateWikiRequest(
        repo_url="https://github.com/org/repo",
        branch="develop",
        access_token="ghp_legacy_token",
    )
    assert req.source_type == "git"
    assert req.scope["repo_url"] == "https://github.com/org/repo"
    assert req.scope["branch"] == "develop"
    assert req.auth.get("pat") == "ghp_legacy_token"
    assert req.access_token == "ghp_legacy_token"


# ---------------------------------------------------------------------------
# Fixture
# ---------------------------------------------------------------------------

import shutil
import subprocess
import tempfile
from pathlib import Path


@pytest.fixture
def fake_repo_for_discrimination():
    """Create a minimal on-disk git repo for route-level discrimination tests."""
    src = Path(tempfile.mkdtemp(prefix="scan_disc_"))
    (src / "README.md").write_text("# test")
    subprocess.run(["git", "init", "-q", "--initial-branch=main"], cwd=src, check=True)
    subprocess.run(["git", "config", "user.email", "t@t.t"], cwd=src, check=True)
    subprocess.run(["git", "config", "user.name", "t"], cwd=src, check=True)
    subprocess.run(["git", "add", "."], cwd=src, check=True)
    subprocess.run(
        ["git", "commit", "-q", "-m", "init", "--no-gpg-sign"], cwd=src, check=True
    )
    with patch("app.services.source_scan_service.get_settings") as gs:
        gs.return_value = MagicMock(allowed_local_paths=str(src.parent))
        yield src
    shutil.rmtree(src, ignore_errors=True)
