"""Unit tests for GET /api/v1/projects/{project_id}/search."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.core.project_search_engine import ProjectSearchResult
from app.core.wiki_search_engine import (
    NeighborItem,
    ResultItem,
    WikiSummaryItem as EngineWikiSummaryItem,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_wiki_record(wiki_id: str, wiki_name: str = "Test Wiki") -> MagicMock:
    record = MagicMock()
    record.id = wiki_id
    record.title = wiki_name
    record.owner_id = "user-1"
    return record


def _make_result_item(
    wiki_id: str = "wiki-1",
    wiki_name: str = "Test Wiki",
    page_title: str = "Auth",
    score: float = 0.9,
) -> ResultItem:
    return ResultItem(
        wiki_id=wiki_id,
        wiki_name=wiki_name,
        page_title=page_title,
        snippet=f"Desc of {page_title}",
        score=score,
        neighbors=[NeighborItem(title="Session", rel="links_to")],
    )


def _make_wiki_summary(
    wiki_id: str = "wiki-1",
    wiki_name: str = "Test Wiki",
    match_count: int = 1,
    relevance: float = 0.5,
) -> EngineWikiSummaryItem:
    return EngineWikiSummaryItem(
        wiki_id=wiki_id,
        wiki_name=wiki_name,
        match_count=match_count,
        relevance=relevance,
    )


def _build_app(
    project_wikis: list[MagicMock] | None = None,  # None = project not found
    search_result: ProjectSearchResult | None = None,
) -> FastAPI:
    """Build a minimal test FastAPI app for the project search endpoint."""
    from app.api.routes import router
    from app.auth import get_current_user
    from app.dependencies import (
        get_project_service,
        get_wiki_index_cache,
    )

    app = FastAPI()
    app.include_router(router)

    # Mock project service
    mock_project_svc = AsyncMock()
    if project_wikis is None:
        # Project not found
        mock_project_svc.list_project_wikis = AsyncMock(return_value=None)
    else:
        mock_project_svc.list_project_wikis = AsyncMock(return_value=project_wikis)

    # Mock wiki page index cache
    mock_cache = AsyncMock()
    mock_cache.get = AsyncMock(return_value=MagicMock(pages={}))

    # Mock settings
    app.state.settings = MagicMock(cache_dir="/tmp/test_cache")
    app.state.wiki_index_cache = mock_cache

    # Store result for use in patch
    app.state._project_search_result = search_result

    # Override DI
    app.dependency_overrides[get_project_service] = lambda: mock_project_svc
    app.dependency_overrides[get_wiki_index_cache] = lambda: mock_cache
    app.dependency_overrides[get_current_user] = lambda: MagicMock(id="user-1")

    return app


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestProjectSearchRoute:
    """Tests for GET /api/v1/projects/{project_id}/search."""

    def test_returns_200_with_project_search_response_shape(self):
        """Returns 200 with ProjectSearchResponse shape on valid request."""
        wikis = [_make_wiki_record("wiki-1", "Test Wiki")]
        result = ProjectSearchResult(
            query="auth",
            results=[_make_result_item()],
            wiki_summary=[_make_wiki_summary()],
        )
        app = _build_app(project_wikis=wikis, search_result=result)

        with patch("app.core.project_search_engine.ProjectSearchEngine") as MockEngine:
            mock_engine = MagicMock()
            mock_engine.search = AsyncMock(return_value=result)
            MockEngine.return_value = mock_engine

            with patch("app.core.unified_db.UnifiedWikiDB"):
                client = TestClient(app)
                resp = client.get("/api/v1/projects/proj-1/search?q=auth")

        assert resp.status_code == 200
        data = resp.json()
        assert data["query"] == "auth"
        assert isinstance(data["results"], list)
        assert isinstance(data["wiki_summary"], list)
        assert len(data["results"]) == 1
        assert data["results"][0]["page_title"] == "Auth"
        assert data["results"][0]["score"] == pytest.approx(0.9)
        assert data["wiki_summary"][0]["wiki_id"] == "wiki-1"

    def test_hop_depth_6_returns_422(self):
        """hop_depth=6 exceeds maximum of 5 and returns 422."""
        wikis = [_make_wiki_record("wiki-1")]
        app = _build_app(project_wikis=wikis)

        client = TestClient(app)
        resp = client.get("/api/v1/projects/proj-1/search?q=test&hop_depth=6")
        assert resp.status_code == 422

    def test_unknown_project_id_returns_404(self):
        """Unknown project_id returns 404."""
        app = _build_app(project_wikis=None)

        client = TestClient(app)
        resp = client.get("/api/v1/projects/no-such-project/search?q=test")
        assert resp.status_code == 404

    def test_empty_wiki_list_returns_200_with_empty_results(self):
        """Project with no wikis returns 200 with empty results and empty wiki_summary."""
        empty_result = ProjectSearchResult(query="test", results=[], wiki_summary=[])
        app = _build_app(project_wikis=[], search_result=empty_result)

        with patch("app.core.project_search_engine.ProjectSearchEngine") as MockEngine:
            mock_engine = MagicMock()
            mock_engine.search = AsyncMock(return_value=empty_result)
            MockEngine.return_value = mock_engine

            client = TestClient(app)
            resp = client.get("/api/v1/projects/proj-1/search?q=test")

        assert resp.status_code == 200
        data = resp.json()
        assert data["query"] == "test"
        assert data["results"] == []
        assert data["wiki_summary"] == []

    def test_multi_wiki_fanout_merges_results(self):
        """Multi-wiki project search returns merged results from all wikis."""
        wikis = [
            _make_wiki_record("wiki-1", "Wiki A"),
            _make_wiki_record("wiki-2", "Wiki B"),
        ]
        merged_result = ProjectSearchResult(
            query="auth",
            results=[
                _make_result_item("wiki-1", "Wiki A", "AuthPage", score=0.9),
                _make_result_item("wiki-2", "Wiki B", "LoginPage", score=0.7),
            ],
            wiki_summary=[
                _make_wiki_summary("wiki-1", "Wiki A", match_count=1),
                _make_wiki_summary("wiki-2", "Wiki B", match_count=1),
            ],
        )
        app = _build_app(project_wikis=wikis, search_result=merged_result)

        with patch("app.core.project_search_engine.ProjectSearchEngine") as MockEngine:
            mock_engine = MagicMock()
            mock_engine.search = AsyncMock(return_value=merged_result)
            MockEngine.return_value = mock_engine

            with patch("app.core.unified_db.UnifiedWikiDB"):
                client = TestClient(app)
                resp = client.get("/api/v1/projects/proj-1/search?q=auth")

        assert resp.status_code == 200
        data = resp.json()
        page_titles = {r["page_title"] for r in data["results"]}
        assert "AuthPage" in page_titles
        assert "LoginPage" in page_titles
        wiki_ids = {s["wiki_id"] for s in data["wiki_summary"]}
        assert "wiki-1" in wiki_ids
        assert "wiki-2" in wiki_ids

    def test_top_k_parameter_is_passed_through(self):
        """top_k is forwarded to the engine's search call."""
        wikis = [_make_wiki_record("wiki-1")]
        empty_result = ProjectSearchResult(query="q", results=[], wiki_summary=[])
        app = _build_app(project_wikis=wikis, search_result=empty_result)

        with patch("app.core.project_search_engine.ProjectSearchEngine") as MockEngine:
            mock_engine = MagicMock()
            mock_engine.search = AsyncMock(return_value=empty_result)
            MockEngine.return_value = mock_engine

            with patch("app.core.unified_db.UnifiedWikiDB"):
                client = TestClient(app)
                resp = client.get("/api/v1/projects/proj-1/search?q=q&top_k=5")

        assert resp.status_code == 200
        # Verify top_k was passed to engine.search
        call_kwargs = mock_engine.search.call_args
        assert call_kwargs is not None
        # top_k can be positional or keyword
        args, kwargs = call_kwargs
        passed_top_k = kwargs.get("top_k") or (args[2] if len(args) > 2 else None)
        assert passed_top_k == 5

    def test_hop_depth_parameter_is_passed_through(self):
        """hop_depth is forwarded to the engine's search call."""
        wikis = [_make_wiki_record("wiki-1")]
        empty_result = ProjectSearchResult(query="q", results=[], wiki_summary=[])
        app = _build_app(project_wikis=wikis, search_result=empty_result)

        with patch("app.core.project_search_engine.ProjectSearchEngine") as MockEngine:
            mock_engine = MagicMock()
            mock_engine.search = AsyncMock(return_value=empty_result)
            MockEngine.return_value = mock_engine

            with patch("app.core.unified_db.UnifiedWikiDB"):
                client = TestClient(app)
                resp = client.get("/api/v1/projects/proj-1/search?q=q&hop_depth=3")

        assert resp.status_code == 200
        call_kwargs = mock_engine.search.call_args
        assert call_kwargs is not None
        args, kwargs = call_kwargs
        passed_hop_depth = kwargs.get("hop_depth") or (args[1] if len(args) > 1 else None)
        assert passed_hop_depth == 3

    def test_missing_q_parameter_returns_422(self):
        """Missing required query param q returns 422."""
        wikis = [_make_wiki_record("wiki-1")]
        app = _build_app(project_wikis=wikis)

        client = TestClient(app)
        resp = client.get("/api/v1/projects/proj-1/search")
        assert resp.status_code == 422
