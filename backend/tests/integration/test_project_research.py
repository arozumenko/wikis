"""Integration tests for the project-based Research / CodeMap feature (Phase 3).

Tests:
- ResearchRequest with project_id validates correctly
- ResearchRequest with neither wiki_id nor project_id returns 422
- Single-wiki research path unaffected (regression)
- POST /api/v1/projects/{project_id}/map route exists and returns SSE / JSON
- Partial wiki availability (one wiki failed) doesn't block research
- MultiGraphQueryService unit tests

Run with:
    AUTH_ENABLED=false pytest tests/integration/test_project_research.py -v
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from httpx import ASGITransport, AsyncClient
from pydantic import ValidationError
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

from app.main import create_app
from app.models.api import ResearchRequest
from app.models.db_models import Base, ProjectRecord, ProjectWikiRecord, WikiRecord


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


async def _make_session_factory():
    engine = create_async_engine("sqlite+aiosqlite://", connect_args={"check_same_thread": False})
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    return engine, async_sessionmaker(engine, expire_on_commit=False)


@pytest.fixture
async def client(tmp_path, monkeypatch):
    """FastAPI test client with in-memory SQLite and AUTH_ENABLED=false."""
    monkeypatch.setenv("AUTH_ENABLED", "false")
    monkeypatch.setenv("LLM_API_KEY", "test-key")

    engine, session_factory = await _make_session_factory()
    app = create_app()

    async with app.router.lifespan_context(app):
        from app.services.wiki_management import WikiManagementService
        from app.storage.local import LocalArtifactStorage

        storage = LocalArtifactStorage(str(tmp_path))
        wiki_mgmt = WikiManagementService(storage, session_factory)
        app.state.storage = storage
        app.state.wiki_management = wiki_mgmt
        app.state.session_factory = session_factory

        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as c:
            yield c, app, session_factory

    await engine.dispose()


async def _seed_wiki(
    session_factory,
    wiki_id: str = "test-wiki-1",
    owner_id: str = "dev-user",
    title: str = "Test Wiki",
) -> None:
    async with session_factory() as session:
        async with session.begin():
            record = WikiRecord(
                id=wiki_id,
                owner_id=owner_id,
                repo_url="https://github.com/test/repo",
                branch="main",
                title=title,
                page_count=1,
                status="complete",
            )
            session.add(record)


async def _seed_project_with_wikis(
    session_factory,
    project_id: str,
    wiki_ids: list[str],
    owner_id: str = "dev-user",
) -> None:
    """Insert a project and attach the given wikis to it."""
    async with session_factory() as session:
        async with session.begin():
            project = ProjectRecord(
                id=project_id,
                owner_id=owner_id,
                name="Research Project",
                visibility="personal",
            )
            session.add(project)
            for wid in wiki_ids:
                membership = ProjectWikiRecord(
                    project_id=project_id,
                    wiki_id=wid,
                    added_by=owner_id,
                )
                session.add(membership)


# ---------------------------------------------------------------------------
# Model validation tests (pure Pydantic)
# ---------------------------------------------------------------------------


class TestResearchRequestValidation:
    def test_wiki_id_only_is_valid(self):
        req = ResearchRequest(wiki_id="w1", question="What is auth?")
        assert req.wiki_id == "w1"
        assert req.project_id is None

    def test_project_id_only_is_valid(self):
        req = ResearchRequest(project_id="p1", question="How does auth work?")
        assert req.project_id == "p1"
        assert req.wiki_id is None

    def test_both_wiki_and_project_is_valid(self):
        """Both can be set — service branches on project_id first."""
        req = ResearchRequest(wiki_id="w1", project_id="p1", question="q?")
        assert req.wiki_id == "w1"
        assert req.project_id == "p1"

    def test_neither_wiki_nor_project_raises_validation_error(self):
        with pytest.raises(ValidationError) as exc_info:
            ResearchRequest(question="What is auth?")
        errors = exc_info.value.errors()
        assert any("wiki_id" in str(e) or "project_id" in str(e) for e in errors)

    def test_research_type_defaults_to_general(self):
        req = ResearchRequest(wiki_id="w1", question="q?")
        assert req.research_type == "general"

    def test_codemap_research_type_allowed(self):
        req = ResearchRequest(project_id="p1", question="q?", research_type="codemap")
        assert req.research_type == "codemap"


# ---------------------------------------------------------------------------
# HTTP route validation tests
# ---------------------------------------------------------------------------


class TestResearchRouteValidation:
    @pytest.mark.asyncio
    async def test_research_without_wiki_or_project_returns_422(self, client):
        c, _app, _sf = client
        resp = await c.post(
            "/api/v1/research",
            json={"question": "What is auth?"},
        )
        assert resp.status_code == 422

    @pytest.mark.asyncio
    async def test_research_with_wiki_id_nonexistent_returns_error(self, client):
        c, _app, _sf = client
        resp = await c.post(
            "/api/v1/research",
            json={"wiki_id": "nonexistent-wiki", "question": "What is auth?"},
        )
        # 404 if wiki not found, 501 if engine not provisioned
        assert resp.status_code in (400, 404, 501)

    @pytest.mark.asyncio
    async def test_project_map_route_nonexistent_project_returns_404(self, client):
        c, _app, _sf = client
        resp = await c.post(
            "/api/v1/projects/nonexistent-project/map",
            json={"question": "How does auth work?"},
        )
        assert resp.status_code == 404

    @pytest.mark.asyncio
    async def test_project_map_route_accessible_project_triggers_service(self, client):
        """A valid project returns a response (mocked service)."""
        c, app, session_factory = client

        await _seed_wiki(session_factory, wiki_id="wiki-1")
        await _seed_wiki(session_factory, wiki_id="wiki-2")
        await _seed_project_with_wikis(
            session_factory,
            project_id="proj-1",
            wiki_ids=["wiki-1", "wiki-2"],
        )

        mock_response = {
            "answer": "Mocked answer",
            "sources": [],
            "research_steps": [],
            "code_map": None,
        }

        from app.services.research_service import ResearchService

        with patch.object(
            ResearchService,
            "codemap_sync",
            new_callable=AsyncMock,
        ) as mock_codemap:
            from app.models.api import ResearchResponse

            mock_codemap.return_value = ResearchResponse(**mock_response)

            resp = await c.post(
                "/api/v1/projects/proj-1/map",
                json={"question": "How does auth work?"},
            )

        assert resp.status_code == 200
        data = resp.json()
        assert data["answer"] == "Mocked answer"

    @pytest.mark.asyncio
    async def test_project_map_route_returns_sse_when_requested(self, client):
        """With Accept: text/event-stream the route streams SSE."""
        c, app, session_factory = client

        await _seed_wiki(session_factory, wiki_id="wiki-sse-1")
        await _seed_project_with_wikis(
            session_factory,
            project_id="proj-sse",
            wiki_ids=["wiki-sse-1"],
        )

        from app.services.research_service import ResearchService

        async def _mock_codemap_stream(self, request, user_id=None):
            yield {"event_type": "research_complete", "data": {"report": "done", "sources": [], "code_map": None}}

        with patch.object(ResearchService, "codemap_stream", _mock_codemap_stream):
            resp = await c.post(
                "/api/v1/projects/proj-sse/map",
                json={"question": "How does auth work?"},
                params={"accept": "text/event-stream"},
            )

        assert resp.status_code == 200
        assert "text/event-stream" in resp.headers["content-type"]


# ---------------------------------------------------------------------------
# ResearchService._get_multi_wiki_components unit tests
# ---------------------------------------------------------------------------


class TestResearchServiceMultiWikiComponents:
    """Unit tests for multi-wiki component loading in ResearchService."""

    def _make_mock_session_factory(self):
        mock_session = AsyncMock()
        mock_session.__aenter__ = AsyncMock(return_value=mock_session)
        mock_session.__aexit__ = AsyncMock(return_value=False)
        return MagicMock(return_value=mock_session)

    @pytest.mark.asyncio
    async def test_get_multi_wiki_components_empty_project_raises(self, tmp_path, monkeypatch):
        """A project with no wikis raises ValueError."""
        monkeypatch.setenv("LLM_API_KEY", "test-key")
        from app.config import Settings
        from app.services.research_service import ResearchService
        from app.storage.local import LocalArtifactStorage

        settings = Settings()
        storage = LocalArtifactStorage(str(tmp_path))
        service = ResearchService(settings=settings, storage=storage)

        mock_factory = self._make_mock_session_factory()

        with patch("app.db.get_session_factory", return_value=mock_factory):
            with patch(
                "app.services.project_service.ProjectService.list_project_wikis",
                new_callable=AsyncMock,
                return_value=[],
            ):
                with pytest.raises(ValueError, match="no accessible wikis"):
                    await service._get_multi_wiki_components("project-1", "user-1")

    @pytest.mark.asyncio
    async def test_get_multi_wiki_components_inaccessible_project_raises(self, tmp_path, monkeypatch):
        """A project not accessible to the user raises ValueError."""
        monkeypatch.setenv("LLM_API_KEY", "test-key")
        from app.config import Settings
        from app.services.research_service import ResearchService
        from app.storage.local import LocalArtifactStorage

        settings = Settings()
        storage = LocalArtifactStorage(str(tmp_path))
        service = ResearchService(settings=settings, storage=storage)

        mock_factory = self._make_mock_session_factory()

        with patch("app.db.get_session_factory", return_value=mock_factory):
            with patch(
                "app.services.project_service.ProjectService.list_project_wikis",
                new_callable=AsyncMock,
                return_value=None,
            ):
                with pytest.raises(ValueError, match="not found or not accessible"):
                    await service._get_multi_wiki_components("project-1", "user-1")


# ---------------------------------------------------------------------------
# build_multi_wiki_components helper unit tests
# ---------------------------------------------------------------------------


class TestBuildMultiWikiComponents:
    """Tests for the shared helper in multi_wiki_components.py."""

    def _make_wiki_record(self, wiki_id: str) -> MagicMock:
        m = MagicMock()
        m.id = wiki_id
        return m

    def _make_mock_session_factory(self):
        mock_session = AsyncMock()
        mock_session.__aenter__ = AsyncMock(return_value=mock_session)
        mock_session.__aexit__ = AsyncMock(return_value=False)
        return MagicMock(return_value=mock_session)

    @pytest.mark.asyncio
    async def test_all_wikis_loaded_successfully(self):
        """All wikis load → MultiWikiRetrieverStack is populated."""
        from app.services.multi_wiki_components import build_multi_wiki_components
        from app.services.toolkit_bridge import EngineComponents

        mock_retriever = MagicMock()
        comp_a = EngineComponents(retriever_stack=mock_retriever, llm=MagicMock())
        comp_b = EngineComponents(retriever_stack=mock_retriever, llm=MagicMock())

        wiki_records = [self._make_wiki_record("w1"), self._make_wiki_record("w2")]

        async def mock_get_components(wiki_id: str) -> EngineComponents:
            return comp_a if wiki_id == "w1" else comp_b

        mock_factory = self._make_mock_session_factory()

        with patch(
            "app.services.project_service.ProjectService.list_project_wikis",
            new_callable=AsyncMock,
            return_value=wiki_records,
        ):
            result = await build_multi_wiki_components(
                project_id="proj-1",
                user_id="user-1",
                get_components_fn=mock_get_components,
                session_factory=mock_factory,
            )

        from app.core.multi_retriever import MultiWikiRetrieverStack

        assert isinstance(result.retriever_stack, MultiWikiRetrieverStack)

    @pytest.mark.asyncio
    async def test_partial_failure_does_not_block(self):
        """If one wiki fails to load, others still succeed."""
        from app.services.multi_wiki_components import build_multi_wiki_components
        from app.services.toolkit_bridge import EngineComponents

        mock_retriever = MagicMock()
        comp_ok = EngineComponents(retriever_stack=mock_retriever, llm=MagicMock())

        wiki_records = [
            self._make_wiki_record("wiki-ok"),
            self._make_wiki_record("wiki-bad"),
        ]

        async def mock_get_components(wiki_id: str) -> EngineComponents:
            if wiki_id == "wiki-bad":
                raise FileNotFoundError("missing artifacts")
            return comp_ok

        mock_factory = self._make_mock_session_factory()

        with patch(
            "app.services.project_service.ProjectService.list_project_wikis",
            new_callable=AsyncMock,
            return_value=wiki_records,
        ):
            result = await build_multi_wiki_components(
                project_id="proj-1",
                user_id="user-1",
                get_components_fn=mock_get_components,
                session_factory=mock_factory,
            )

        # Should succeed with one wiki
        assert result.retriever_stack is not None

    @pytest.mark.asyncio
    async def test_all_wikis_fail_raises_value_error(self):
        """All wikis failing raises ValueError."""
        from app.services.multi_wiki_components import build_multi_wiki_components

        wiki_records = [self._make_wiki_record("w1")]

        async def mock_get_components(wiki_id: str):
            raise FileNotFoundError("missing")

        mock_factory = self._make_mock_session_factory()

        with patch(
            "app.services.project_service.ProjectService.list_project_wikis",
            new_callable=AsyncMock,
            return_value=wiki_records,
        ):
            with pytest.raises(ValueError, match="No usable components"):
                await build_multi_wiki_components(
                    project_id="proj-1",
                    user_id="user-1",
                    get_components_fn=mock_get_components,
                    session_factory=mock_factory,
                )


# ---------------------------------------------------------------------------
# MultiGraphQueryService unit tests
# ---------------------------------------------------------------------------


class TestMultiGraphQueryService:
    """Unit tests for MultiGraphQueryService (no DB, no HTTP)."""

    def _make_gqs(self, node_ids: list[str] | None = None) -> MagicMock:
        """Create a minimal GraphQueryService-like mock."""
        import networkx as nx

        gqs = MagicMock()
        graph = nx.DiGraph()
        if node_ids:
            for nid in node_ids:
                graph.add_node(nid, symbol_name=nid, symbol_type="function")
        gqs.graph = graph
        return gqs

    def test_resolve_symbol_fan_out_finds_match(self):
        from app.core.code_graph.multi_graph_query_service import MultiGraphQueryService

        gqs_a = self._make_gqs()
        gqs_a.resolve_symbol.return_value = None

        gqs_b = self._make_gqs()
        gqs_b.resolve_symbol.return_value = "node-b-1"

        svc = MultiGraphQueryService({"wiki-a": gqs_a, "wiki-b": gqs_b})
        result = svc.resolve_symbol("MyClass")
        assert result == "node-b-1"

    def test_resolve_symbol_with_wiki_id_dispatches_to_correct_graph(self):
        from app.core.code_graph.multi_graph_query_service import MultiGraphQueryService

        gqs_a = self._make_gqs()
        gqs_a.resolve_symbol.return_value = "node-a-1"
        gqs_b = self._make_gqs()
        gqs_b.resolve_symbol.return_value = "node-b-1"

        svc = MultiGraphQueryService({"wiki-a": gqs_a, "wiki-b": gqs_b})
        result = svc.resolve_symbol("MyClass", wiki_id="wiki-b")
        assert result == "node-b-1"
        gqs_a.resolve_symbol.assert_not_called()

    def test_get_relationships_finds_node_in_correct_graph(self):
        from app.core.code_graph.multi_graph_query_service import MultiGraphQueryService

        gqs_a = self._make_gqs(["node-a"])
        gqs_a.get_relationships.return_value = []

        gqs_b = self._make_gqs(["node-b"])
        gqs_b.get_relationships.return_value = [MagicMock()]

        svc = MultiGraphQueryService({"wiki-a": gqs_a, "wiki-b": gqs_b})
        result = svc.get_relationships("node-b")
        assert len(result) == 1
        gqs_a.get_relationships.assert_not_called()

    def test_get_relationships_unknown_node_returns_empty(self):
        from app.core.code_graph.multi_graph_query_service import MultiGraphQueryService

        gqs_a = self._make_gqs()
        svc = MultiGraphQueryService({"wiki-a": gqs_a})
        result = svc.get_relationships("unknown-node")
        assert result == []

    def test_search_merges_results_from_all_graphs(self):
        from app.core.code_graph.graph_query_service import SymbolResult
        from app.core.code_graph.multi_graph_query_service import MultiGraphQueryService

        sym_a = SymbolResult(
            node_id="n-a", symbol_name="AuthService", symbol_type="class",
            score=0.8,
        )
        sym_b = SymbolResult(
            node_id="n-b", symbol_name="TokenManager", symbol_type="class",
            score=0.9,
        )

        gqs_a = self._make_gqs()
        gqs_a.search.return_value = [sym_a]
        gqs_b = self._make_gqs()
        gqs_b.search.return_value = [sym_b]

        svc = MultiGraphQueryService({"wiki-a": gqs_a, "wiki-b": gqs_b})
        results = svc.search("auth", k=10)

        assert len(results) == 2
        # Higher score first
        assert results[0].node_id == "n-b"

    def test_graph_property_returns_primary_graph(self):
        import networkx as nx

        from app.core.code_graph.multi_graph_query_service import MultiGraphQueryService

        gqs_a = self._make_gqs()
        graph = nx.DiGraph()
        gqs_a.graph = graph
        svc = MultiGraphQueryService({"wiki-a": gqs_a})
        assert svc.graph is graph

    def test_empty_service_graph_property_returns_empty_digraph(self):
        import networkx as nx

        from app.core.code_graph.multi_graph_query_service import MultiGraphQueryService

        svc = MultiGraphQueryService({})
        assert isinstance(svc.graph, nx.DiGraph)

    def test_wiki_ids_returns_keys(self):
        from app.core.code_graph.multi_graph_query_service import MultiGraphQueryService

        gqs_a = self._make_gqs()
        gqs_b = self._make_gqs()
        svc = MultiGraphQueryService({"w1": gqs_a, "w2": gqs_b})
        assert svc.wiki_ids() == ["w1", "w2"]
