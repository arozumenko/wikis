"""#203 — POST /api/v1/wikis must reach the same handler as POST /api/v1/generate.

The frontend's ``generateWikiMultiSource`` posts to ``/api/v1/wikis``.
The module docstring of ``app/models/api.py`` advertises the same URL.
PR #195 introduced both — but the backend route alias was missed, leaving
the dialog with a 405 on every click. These tests pin the registration
and behavior parity so the alias can't silently drop again.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.api.routes import router
from app.auth import get_current_user
from app.dependencies import get_wiki_service


def _build_app(invocation):
    app = FastAPI()
    # The router already carries prefix="/api/v1".
    app.include_router(router)

    mock_service = AsyncMock()
    mock_service.generate = AsyncMock(return_value=invocation)

    app.dependency_overrides[get_wiki_service] = lambda: mock_service
    app.dependency_overrides[get_current_user] = lambda: MagicMock(id="user-1")
    return app, mock_service


def _fake_invocation():
    inv = MagicMock()
    inv.wiki_id = "owner--repo--main"
    inv.id = "inv-1"
    inv.status = "generating"
    return inv


@pytest.fixture
def request_body():
    return {
        "source_type": "git",
        "scope": {"repo_url": "https://github.com/owner/repo", "branch": "main"},
        "auth": {"pat": None},
    }


def _registered(method: str, suffix: str) -> bool:
    """Return True if the router exposes ``method <prefix><suffix>``.

    APIRouter stores routes with the prefix baked in, so the path attribute
    is the full ``/api/v1/wikis`` form rather than the bare decorator arg.
    """
    method = method.upper()
    return any(
        getattr(route, "path", "").endswith(suffix) and method in getattr(route, "methods", set())
        for route in router.routes
    )


def test_post_wikis_registered() -> None:
    """The ``POST /wikis`` route exists (regression guard for #203).

    The module docstring on ``app/models/api.py`` advertises this URL as the
    documented contract; the absence of the route is exactly what made #195
    ship with the Generate dialog broken.
    """
    assert _registered("POST", "/wikis"), "POST /api/v1/wikis missing — #203 regression"


def test_post_generate_still_registered() -> None:
    """``POST /generate`` stays as a legacy alias for any pre-#195 caller."""
    assert _registered("POST", "/generate"), "POST /api/v1/generate missing (legacy alias)"


def test_post_wikis_returns_202(request_body) -> None:
    app, mock_service = _build_app(_fake_invocation())
    with TestClient(app) as client:
        resp = client.post("/api/v1/wikis", json=request_body)
    assert resp.status_code == 202
    body = resp.json()
    assert body["wiki_id"] == "owner--repo--main"
    assert body["invocation_id"] == "inv-1"
    mock_service.generate.assert_awaited_once()


def test_post_generate_returns_202(request_body) -> None:
    """The legacy alias keeps working for backward compatibility."""
    app, mock_service = _build_app(_fake_invocation())
    with TestClient(app) as client:
        resp = client.post("/api/v1/generate", json=request_body)
    assert resp.status_code == 202
    mock_service.generate.assert_awaited_once()


def test_both_routes_reach_the_same_handler(request_body) -> None:
    """Two routes, one handler — both must produce the same response shape."""
    app, _ = _build_app(_fake_invocation())
    with TestClient(app) as client:
        a = client.post("/api/v1/wikis", json=request_body)
        b = client.post("/api/v1/generate", json=request_body)
    assert a.status_code == b.status_code == 202
    assert a.json() == b.json()
