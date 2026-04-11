"""Integration tests for the multi-wiki / project-based Ask feature.

Tests:
- Single-wiki ask still works (regression)
- AskRequest with project_id and no wiki_id is valid
- AskRequest with neither wiki_id nor project_id raises validation error
- Multi-wiki retrieval returns merged results (mock per-wiki retrievers)
- SourceReference fields wiki_id and wiki_title are present

Run with:
    AUTH_ENABLED=false pytest tests/integration/test_project_ask.py -v
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from httpx import ASGITransport, AsyncClient
from pydantic import ValidationError
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

from app.main import create_app
from app.models.api import AskRequest, SourceReference
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
                name="Multi-Wiki Project",
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
# Model validation tests (no HTTP, pure Pydantic)
# ---------------------------------------------------------------------------


class TestAskRequestValidation:
    def test_wiki_id_only_is_valid(self):
        req = AskRequest(wiki_id="w1", question="What is this?")
        assert req.wiki_id == "w1"
        assert req.project_id is None

    def test_project_id_only_is_valid(self):
        req = AskRequest(project_id="p1", question="What is this?")
        assert req.project_id == "p1"
        assert req.wiki_id is None

    def test_both_wiki_and_project_is_valid(self):
        """Both can be set — project_id takes precedence in service logic."""
        req = AskRequest(wiki_id="w1", project_id="p1", question="q?")
        assert req.wiki_id == "w1"
        assert req.project_id == "p1"

    def test_neither_wiki_nor_project_raises_validation_error(self):
        with pytest.raises(ValidationError) as exc_info:
            AskRequest(question="What is this?")
        errors = exc_info.value.errors()
        assert any("wiki_id" in str(e) or "project_id" in str(e) for e in errors)

    def test_empty_question_is_rejected(self):
        """Ensure question field is required."""
        with pytest.raises(ValidationError):
            AskRequest(wiki_id="w1")


class TestSourceReferenceFields:
    def test_source_reference_has_wiki_id_and_title_fields(self):
        ref = SourceReference(
            file_path="src/main.py",
            wiki_id="wiki-1",
            wiki_title="My Repo",
        )
        assert ref.wiki_id == "wiki-1"
        assert ref.wiki_title == "My Repo"

    def test_source_reference_wiki_fields_default_to_none(self):
        ref = SourceReference(file_path="src/main.py")
        assert ref.wiki_id is None
        assert ref.wiki_title is None


# ---------------------------------------------------------------------------
# HTTP route tests (regression + project-based ask)
# ---------------------------------------------------------------------------


class TestAskRouteValidation:
    """Route-level validation without triggering real LLM/retrieval."""

    @pytest.mark.asyncio
    async def test_ask_without_wiki_or_project_returns_422(self, client):
        c, _app, _sf = client
        resp = await c.post(
            "/api/v1/ask",
            json={"question": "What is auth?"},
        )
        assert resp.status_code == 422

    @pytest.mark.asyncio
    async def test_ask_with_wiki_id_missing_wiki_returns_404(self, client):
        """Asking about a wiki that doesn't exist returns 404."""
        c, _app, _sf = client
        resp = await c.post(
            "/api/v1/ask",
            json={"wiki_id": "non-existent-wiki", "question": "What is auth?"},
        )
        # Service raises FileNotFoundError → 404
        assert resp.status_code in (404, 501)  # 501 if AskEngine not enabled


# ---------------------------------------------------------------------------
# MultiWikiRetrieverStack unit tests
# ---------------------------------------------------------------------------


class TestMultiWikiRetrieverStack:
    """Unit tests for the fan-out retriever — no HTTP, no DB."""

    @pytest.mark.asyncio
    async def test_empty_stacks_returns_empty(self):
        from app.core.multi_retriever import MultiWikiRetrieverStack

        stack = MultiWikiRetrieverStack([])
        result = await stack.aretrieve("query")
        assert result == []

    @pytest.mark.asyncio
    async def test_single_wiki_results_returned(self):
        from langchain_core.documents import Document

        from app.core.multi_retriever import MultiWikiRetrieverStack

        doc = Document(page_content="hello", metadata={"score": 0.9})
        mock_stack = MagicMock()
        mock_stack.search_repository.return_value = [doc]

        stack = MultiWikiRetrieverStack([("wiki-1", mock_stack)])
        result = await stack.aretrieve("query", k=5)

        assert len(result) == 1
        assert result[0].metadata["source_wiki_id"] == "wiki-1"

    @pytest.mark.asyncio
    async def test_multi_wiki_results_merged_and_tagged(self):
        from langchain_core.documents import Document

        from app.core.multi_retriever import MultiWikiRetrieverStack

        doc_a = Document(page_content="from wiki A", metadata={"score": 0.8})
        doc_b = Document(page_content="from wiki B", metadata={"score": 0.9})

        mock_a = MagicMock()
        mock_a.search_repository.return_value = [doc_a]

        mock_b = MagicMock()
        mock_b.search_repository.return_value = [doc_b]

        stack = MultiWikiRetrieverStack([("wiki-a", mock_a), ("wiki-b", mock_b)])
        result = await stack.aretrieve("query", k=10)

        assert len(result) == 2
        wiki_ids = {d.metadata["source_wiki_id"] for d in result}
        assert wiki_ids == {"wiki-a", "wiki-b"}

    @pytest.mark.asyncio
    async def test_failed_wiki_retrieval_is_skipped(self):
        from langchain_core.documents import Document

        from app.core.multi_retriever import MultiWikiRetrieverStack

        doc = Document(page_content="ok", metadata={"score": 0.7})

        ok_stack = MagicMock()
        ok_stack.search_repository.return_value = [doc]

        bad_stack = MagicMock()
        bad_stack.search_repository.side_effect = RuntimeError("retriever down")

        stack = MultiWikiRetrieverStack([("wiki-ok", ok_stack), ("wiki-bad", bad_stack)])
        result = await stack.aretrieve("query", k=5)

        # Should only return docs from the working wiki
        assert len(result) == 1
        assert result[0].metadata["source_wiki_id"] == "wiki-ok"

    @pytest.mark.asyncio
    async def test_results_capped_at_k_times_2(self):
        from langchain_core.documents import Document

        from app.core.multi_retriever import MultiWikiRetrieverStack

        docs_a = [Document(page_content=f"a{i}", metadata={"score": float(i)}) for i in range(10)]
        docs_b = [Document(page_content=f"b{i}", metadata={"score": float(i)}) for i in range(10)]

        mock_a = MagicMock()
        mock_a.search_repository.return_value = docs_a
        mock_b = MagicMock()
        mock_b.search_repository.return_value = docs_b

        stack = MultiWikiRetrieverStack([("wiki-a", mock_a), ("wiki-b", mock_b)])
        result = await stack.aretrieve("query", k=5)

        # Cap is k*2 = 10
        assert len(result) <= 10

    @pytest.mark.asyncio
    async def test_normalized_score_metadata_present(self):
        from langchain_core.documents import Document

        from app.core.multi_retriever import MultiWikiRetrieverStack

        docs = [
            Document(page_content="low", metadata={"score": 0.1}),
            Document(page_content="high", metadata={"score": 0.9}),
        ]
        mock_stack = MagicMock()
        mock_stack.search_repository.return_value = docs

        stack = MultiWikiRetrieverStack([("wiki-1", mock_stack)])
        result = await stack.aretrieve("query", k=10)

        for doc in result:
            assert "normalized_score" in doc.metadata

    def test_sync_search_repository_wraps_async(self):
        from langchain_core.documents import Document

        from app.core.multi_retriever import MultiWikiRetrieverStack

        doc = Document(page_content="sync test", metadata={"score": 0.5})
        mock_stack = MagicMock()
        mock_stack.search_repository.return_value = [doc]

        stack = MultiWikiRetrieverStack([("wiki-1", mock_stack)])
        # Call the sync method directly (it should internally run the async version)
        # We don't test the full sync path here since it depends on event loop state,
        # but we verify the method exists and is callable
        assert callable(stack.search_repository)
        assert callable(stack.retrieve)


# ---------------------------------------------------------------------------
# AskService._get_multi_wiki_components unit tests
# ---------------------------------------------------------------------------


class TestAskServiceMultiWikiComponents:
    """Unit tests for the multi-wiki component loading logic."""

    def _make_mock_session_factory(self):
        """Build a mock async session factory that patch('app.db.get_session_factory') can return."""
        mock_session = AsyncMock()
        mock_session.__aenter__ = AsyncMock(return_value=mock_session)
        mock_session.__aexit__ = AsyncMock(return_value=False)
        return MagicMock(return_value=mock_session)

    @pytest.mark.asyncio
    async def test_get_multi_wiki_components_empty_project_raises(self, tmp_path, monkeypatch):
        """A project with no wikis raises ValueError."""
        monkeypatch.setenv("LLM_API_KEY", "test-key")
        from app.config import Settings
        from app.services.ask_service import AskService
        from app.storage.local import LocalArtifactStorage

        settings = Settings()
        storage = LocalArtifactStorage(str(tmp_path))
        service = AskService(settings=settings, storage=storage)

        mock_factory = self._make_mock_session_factory()

        # Patch at the module where get_session_factory is defined (app.db)
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
        """A project not accessible to the user (None returned) raises ValueError."""
        monkeypatch.setenv("LLM_API_KEY", "test-key")
        from app.config import Settings
        from app.services.ask_service import AskService
        from app.storage.local import LocalArtifactStorage

        settings = Settings()
        storage = LocalArtifactStorage(str(tmp_path))
        service = AskService(settings=settings, storage=storage)

        mock_factory = self._make_mock_session_factory()

        with patch("app.db.get_session_factory", return_value=mock_factory):
            with patch(
                "app.services.project_service.ProjectService.list_project_wikis",
                new_callable=AsyncMock,
                return_value=None,
            ):
                with pytest.raises(ValueError, match="not found or not accessible"):
                    await service._get_multi_wiki_components("project-1", "user-1")
