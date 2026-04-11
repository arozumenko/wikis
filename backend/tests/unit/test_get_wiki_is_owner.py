"""Tests for is_owner field in GET /api/v1/wikis/{wiki_id} and _get_repo_context
in DeepResearchEngine.

Covers:
- Main response path: is_owner present and correct
- Early-return for non-complete DB state: is_owner present (False)
- Early-return for in-flight invocation: is_owner present (False)
- DeepResearchEngine._get_repo_context description key handling
- DeepResearchEngine._get_repo_context with None repo_analysis
"""

from __future__ import annotations

from datetime import datetime
from unittest.mock import AsyncMock, MagicMock

import pytest
from app.core.deep_research.research_engine import DeepResearchEngine
from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient

# ---------------------------------------------------------------------------
# DeepResearchEngine._get_repo_context tests
# ---------------------------------------------------------------------------


def _make_research_engine(**kwargs) -> DeepResearchEngine:
    return DeepResearchEngine(
        retriever_stack=MagicMock(),
        graph_manager=MagicMock(),
        code_graph=MagicMock(),
        **kwargs,
    )


class TestResearchEngineGetRepoContext:
    def test_description_prepended_before_other_fields(self):
        """description key appears first in the context output."""
        engine = _make_research_engine(
            repo_analysis={
                "description": "A payments API",
                "summary": "handles billing",
                "languages": "Python",
            }
        )
        ctx = engine._get_repo_context()
        assert ctx.startswith("Project Description: A payments API")
        assert "Repository Summary: handles billing" in ctx
        assert "Languages: Python" in ctx

    def test_description_only(self):
        """Only description key → output starts with the description."""
        engine = _make_research_engine(repo_analysis={"description": "A payments API"})
        ctx = engine._get_repo_context()
        assert ctx.startswith("Project Description: A payments API")

    def test_no_description_existing_keys_unchanged(self):
        """Without description, summary/key_components/languages still appear."""
        engine = _make_research_engine(
            repo_analysis={"summary": "Test repo", "key_components": "auth", "languages": "Go"}
        )
        ctx = engine._get_repo_context()
        assert "Project Description" not in ctx
        assert "Repository Summary: Test repo" in ctx
        assert "Key Components: auth" in ctx
        assert "Languages: Go" in ctx

    def test_none_repo_analysis_returns_fallback(self):
        """repo_analysis=None returns fallback without crashing."""
        engine = _make_research_engine()
        engine.repo_analysis = None  # bypass __init__ default
        ctx = engine._get_repo_context()
        assert ctx == "No repository overview available."

    def test_empty_repo_analysis_returns_fallback(self):
        engine = _make_research_engine(repo_analysis={})
        ctx = engine._get_repo_context()
        assert ctx == "No repository overview available."


# ---------------------------------------------------------------------------
# GET /api/v1/wikis/{wiki_id} — is_owner field tests
# ---------------------------------------------------------------------------


def _make_wiki_summary(status: str = "complete", is_owner: bool = True) -> MagicMock:
    """Build a mock WikiSummary with required attributes."""
    ws = MagicMock()
    ws.wiki_id = "wiki-1"
    ws.repo_url = "https://github.com/example/repo"
    ws.branch = "main"
    ws.title = "My Wiki"
    ws.page_count = 5
    ws.created_at = datetime(2024, 1, 1)
    ws.indexed_at = None
    ws.commit_hash = None
    ws.status = status
    ws.error = None
    ws.requires_token = False
    ws.description = "test wiki"
    ws.is_owner = is_owner
    ws.invocation_id = None
    return ws


def _make_wiki_record() -> MagicMock:
    record = MagicMock()
    record.id = "wiki-1"
    record.owner_id = "user-1"
    return record


@pytest.fixture
def app_with_mocks():
    """Build a FastAPI test app with the routes router and mocked dependencies."""
    from app.api.routes import router
    from app.auth import get_current_user
    from app.dependencies import get_wiki_management, get_wiki_service

    app = FastAPI()
    app.include_router(router)

    mock_management = AsyncMock()
    mock_service = MagicMock()
    mock_service.invocations = {}

    app.dependency_overrides[get_current_user] = lambda: MagicMock(id="user-1")
    app.dependency_overrides[get_wiki_management] = lambda: mock_management
    app.dependency_overrides[get_wiki_service] = lambda: mock_service

    return app, mock_management, mock_service


@pytest.mark.asyncio
async def test_get_wiki_main_response_includes_is_owner(app_with_mocks):
    """Complete wiki returns is_owner from wiki_meta."""
    app, mock_management, mock_service = app_with_mocks

    wiki_record = _make_wiki_record()
    wiki_summary = _make_wiki_summary(status="complete", is_owner=True)

    mock_management.get_wiki = AsyncMock(return_value=wiki_record)
    mock_management.storage.list_artifacts = AsyncMock(return_value=[])

    from app.services.wiki_management import WikiManagementService

    with (
        pytest.MonkeyPatch().context() as mp,
    ):
        mp.setattr(
            WikiManagementService,
            "_record_to_summary",
            staticmethod(lambda record, user_id: wiki_summary),
        )

        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            resp = await client.get("/api/v1/wikis/wiki-1")

    assert resp.status_code == 200
    data = resp.json()
    assert "is_owner" in data
    assert data["is_owner"] is True


@pytest.mark.asyncio
async def test_get_wiki_main_response_is_owner_false_for_non_owner(app_with_mocks):
    """Complete wiki returns is_owner=False for non-owner."""
    app, mock_management, mock_service = app_with_mocks

    wiki_record = _make_wiki_record()
    wiki_summary = _make_wiki_summary(status="complete", is_owner=False)

    mock_management.get_wiki = AsyncMock(return_value=wiki_record)
    mock_management.storage.list_artifacts = AsyncMock(return_value=[])

    from app.services.wiki_management import WikiManagementService

    with pytest.MonkeyPatch().context() as mp:
        mp.setattr(
            WikiManagementService,
            "_record_to_summary",
            staticmethod(lambda record, user_id: wiki_summary),
        )

        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            resp = await client.get("/api/v1/wikis/wiki-1")

    assert resp.status_code == 200
    assert resp.json()["is_owner"] is False


@pytest.mark.asyncio
async def test_get_wiki_non_complete_db_state_has_is_owner(app_with_mocks):
    """Non-complete DB state early-return includes is_owner from wiki_meta."""
    app, mock_management, mock_service = app_with_mocks

    wiki_record = _make_wiki_record()
    wiki_summary = _make_wiki_summary(status="generating", is_owner=True)

    mock_management.get_wiki = AsyncMock(return_value=wiki_record)

    from app.services.wiki_management import WikiManagementService

    with pytest.MonkeyPatch().context() as mp:
        mp.setattr(
            WikiManagementService,
            "_record_to_summary",
            staticmethod(lambda record, user_id: wiki_summary),
        )

        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            resp = await client.get("/api/v1/wikis/wiki-1")

    assert resp.status_code == 200
    data = resp.json()
    assert "is_owner" in data
    assert data["is_owner"] is True


@pytest.mark.asyncio
async def test_get_wiki_inflight_invocation_has_is_owner_false(app_with_mocks):
    """In-flight invocation early-return includes is_owner=False."""
    app, mock_management, mock_service = app_with_mocks

    # No DB record — only in-memory invocation
    mock_management.get_wiki = AsyncMock(return_value=None)

    inv = MagicMock()
    inv.wiki_id = "wiki-1"
    inv.repo_url = "https://github.com/example/repo"
    inv.branch = "main"
    inv.pages_completed = 3
    inv.created_at = datetime(2024, 1, 1)
    inv.status = "generating"
    inv.id = "inv-1"
    inv.error = None
    inv.owner_id = "user-1"

    mock_service.invocations = {"inv-1": inv}

    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        resp = await client.get("/api/v1/wikis/wiki-1")

    assert resp.status_code == 200
    data = resp.json()
    assert "is_owner" in data
    assert data["is_owner"] is False
