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


# ---------------------------------------------------------------------------
# #175 — list/detail endpoints must agree on status when a live
# in-progress invocation supersedes a stale DB record.
#
# Repro shape: a wiki has ``status='failed'`` + ``error='old…'`` from
# a prior failed run. The user retries; a new generating invocation
# now lives in ``service.invocations``. Both the list endpoint
# (dashboard cards) and the detail endpoint (wiki viewer) MUST
# surface ``status='generating'`` with no leftover ``error`` so the
# SPA stops showing "Generation failed" while a refresh is live.
# ---------------------------------------------------------------------------


def _make_inflight_invocation(
    status: str = "generating",
    progress: float = 0.1,
    invocation_id: str = "inv-new",
) -> MagicMock:
    """Build a mock in-flight invocation matching the live refresh."""
    inv = MagicMock()
    inv.id = invocation_id
    inv.wiki_id = "wiki-1"
    inv.repo_url = "https://github.com/example/repo"
    inv.branch = "main"
    inv.status = status
    inv.progress = progress
    inv.error = None
    inv.pages_completed = 0
    inv.created_at = datetime(2024, 1, 2)
    inv.owner_id = "user-1"
    return inv


@pytest.mark.asyncio
async def test_get_wiki_override_clears_stale_error_when_refresh_is_live(
    app_with_mocks,
):
    """Detail endpoint: when a generating invocation supersedes a
    prior-failed DB record, response must surface generating status
    AND drop the stale error string."""
    app, mock_management, mock_service = app_with_mocks

    wiki_record = _make_wiki_record()
    wiki_summary = _make_wiki_summary(status="failed", is_owner=True)
    wiki_summary.error = "Git clone failed (from prior attempt)"

    mock_management.get_wiki = AsyncMock(return_value=wiki_record)
    mock_management.storage.list_artifacts = AsyncMock(return_value=[])

    mock_service.invocations = {
        "inv-new": _make_inflight_invocation(status="generating", progress=0.42),
    }

    from app.services.wiki_management import WikiManagementService

    with pytest.MonkeyPatch().context() as mp:
        mp.setattr(
            WikiManagementService,
            "_record_to_summary",
            staticmethod(lambda record, user_id: wiki_summary),
        )
        async with AsyncClient(
            transport=ASGITransport(app=app), base_url="http://test",
        ) as client:
            resp = await client.get("/api/v1/wikis/wiki-1")

    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "generating"
    assert data["error"] is None
    assert data["progress"] == 0.42
    assert data["invocation_id"] == "inv-new"


@pytest.mark.asyncio
async def test_get_wiki_override_fires_for_running_incremental_refresh(
    app_with_mocks,
):
    """Detail endpoint must honour ``running`` (incremental refresh)
    the same way as ``generating`` (full re-generation)."""
    app, mock_management, mock_service = app_with_mocks

    wiki_record = _make_wiki_record()
    wiki_summary = _make_wiki_summary(status="complete", is_owner=True)

    mock_management.get_wiki = AsyncMock(return_value=wiki_record)
    mock_management.storage.list_artifacts = AsyncMock(return_value=[])

    mock_service.invocations = {
        "inv-r": _make_inflight_invocation(status="running", progress=0.3),
    }

    from app.services.wiki_management import WikiManagementService

    with pytest.MonkeyPatch().context() as mp:
        mp.setattr(
            WikiManagementService,
            "_record_to_summary",
            staticmethod(lambda record, user_id: wiki_summary),
        )
        async with AsyncClient(
            transport=ASGITransport(app=app), base_url="http://test",
        ) as client:
            resp = await client.get("/api/v1/wikis/wiki-1")

    assert resp.status_code == 200
    assert resp.json()["status"] == "running"
    assert resp.json()["progress"] == 0.3


@pytest.mark.asyncio
async def test_get_wiki_terminal_invocation_does_not_override(app_with_mocks):
    """Counter-test: a leftover *failed* in-memory invocation must NOT
    shadow a *successfully completed* DB record. The DB is the source
    of truth for terminal states; only in-progress invocations override.
    """
    app, mock_management, mock_service = app_with_mocks

    wiki_record = _make_wiki_record()
    wiki_summary = _make_wiki_summary(status="complete", is_owner=True)

    mock_management.get_wiki = AsyncMock(return_value=wiki_record)
    mock_management.storage.list_artifacts = AsyncMock(return_value=[])

    mock_service.invocations = {
        "inv-old": _make_inflight_invocation(status="failed"),
    }

    from app.services.wiki_management import WikiManagementService

    with pytest.MonkeyPatch().context() as mp:
        mp.setattr(
            WikiManagementService,
            "_record_to_summary",
            staticmethod(lambda record, user_id: wiki_summary),
        )
        async with AsyncClient(
            transport=ASGITransport(app=app), base_url="http://test",
        ) as client:
            resp = await client.get("/api/v1/wikis/wiki-1")

    # Status stays "complete" — terminal in-memory invocation does
    # not regress the DB's authoritative state.
    assert resp.json()["status"] == "complete"


def _make_serializable_wiki_summary(
    status: str = "complete",
    is_owner: bool = True,
    error: str | None = None,
) -> MagicMock:
    """Like ``_make_wiki_summary`` but populates every field the
    ``WikiListResponse`` Pydantic schema requires as concrete typed
    values, not MagicMock auto-attrs (which fail validation)."""
    ws = _make_wiki_summary(status=status, is_owner=is_owner)
    ws.error = error
    ws.progress = None
    ws.owner_id = "user-1"
    ws.visibility = "personal"
    return ws


@pytest.mark.asyncio
async def test_list_wikis_override_clears_stale_error_when_refresh_is_live(
    app_with_mocks,
):
    """List endpoint: same contract as the detail endpoint above. The
    dashboard card must show generating + no error when a refresh
    supersedes a prior failed run."""
    app, mock_management, mock_service = app_with_mocks

    wiki_summary = _make_serializable_wiki_summary(
        status="failed", is_owner=True,
        error="Git clone failed (from prior attempt)",
    )

    wiki_list = MagicMock()
    wiki_list.wikis = [wiki_summary]
    mock_management.list_wikis = AsyncMock(return_value=wiki_list)
    mock_management.storage.list_artifacts = AsyncMock(return_value=[])

    mock_service.invocations = {
        "inv-new": _make_inflight_invocation(status="generating", progress=0.42),
    }

    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://test",
    ) as client:
        resp = await client.get("/api/v1/wikis")

    assert resp.status_code == 200
    wikis = resp.json()["wikis"]
    assert len(wikis) == 1
    assert wikis[0]["status"] == "generating"
    assert wikis[0]["error"] is None
    assert wikis[0]["progress"] == 0.42


@pytest.mark.asyncio
async def test_list_wikis_override_fires_for_running_incremental_refresh(
    app_with_mocks,
):
    """List endpoint: running (incremental refresh) is honoured."""
    app, mock_management, mock_service = app_with_mocks

    wiki_summary = _make_serializable_wiki_summary(
        status="complete", is_owner=True,
    )

    wiki_list = MagicMock()
    wiki_list.wikis = [wiki_summary]
    mock_management.list_wikis = AsyncMock(return_value=wiki_list)
    mock_management.storage.list_artifacts = AsyncMock(return_value=[])

    mock_service.invocations = {
        "inv-r": _make_inflight_invocation(status="running", progress=0.3),
    }

    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://test",
    ) as client:
        resp = await client.get("/api/v1/wikis")

    assert resp.json()["wikis"][0]["status"] == "running"
    assert resp.json()["wikis"][0]["progress"] == 0.3
