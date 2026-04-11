"""Integration tests for the REST search endpoints.

Tests the HTTP surface of:
  GET /api/v1/wikis/{wiki_id}/search
  GET /api/v1/wikis/{wiki_id}/pages
  GET /api/v1/wikis/{wiki_id}/pages/{page_title}/neighbors
  GET /api/v1/projects/{project_id}/search

Run with:
    AUTH_ENABLED=false pytest tests/integration/test_search_routes.py -v
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from httpx import ASGITransport, AsyncClient
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

from app.auth import CurrentUser, get_current_user
from app.core.wiki_page_index import PageMeta, WikiPageIndex, WikiPageIndexCache
from app.main import create_app
from app.models.db_models import Base, WikiRecord
from app.services.wiki_management import WikiManagementService
from app.storage.local import LocalArtifactStorage


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


async def _make_session_factory():
    engine = create_async_engine("sqlite+aiosqlite://", connect_args={"check_same_thread": False})
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    return engine, async_sessionmaker(engine, expire_on_commit=False)


async def _seed_wiki(
    session_factory,
    wiki_id: str = "test-wiki",
    owner_id: str = "dev-user",
    status: str = "complete",
) -> WikiRecord:
    """Insert a minimal WikiRecord into the in-memory DB."""
    async with session_factory() as session:
        async with session.begin():
            record = WikiRecord(
                id=wiki_id,
                owner_id=owner_id,
                repo_url="https://github.com/test/repo",
                branch="main",
                title="Test Wiki",
                page_count=3,
                status=status,
            )
            session.add(record)
    return record


def _make_fake_page_index(wiki_id: str = "test-wiki") -> WikiPageIndex:
    """Return a WikiPageIndex with two pre-populated pages and a wikilink edge."""
    index = WikiPageIndex.__new__(WikiPageIndex)
    index._wiki_id = wiki_id
    index._storage = None

    page_a = PageMeta(
        title="Architecture",
        description="High-level architecture overview",
        file_path=f"{wiki_id}/repo/wiki_pages/intro/Architecture.md",
    )
    page_b = PageMeta(
        title="Database",
        description="Database schema and access patterns",
        file_path=f"{wiki_id}/repo/wiki_pages/intro/Database.md",
    )
    index.pages = {"Architecture": page_a, "Database": page_b}
    # Architecture → Database (forward edge)
    index._forward = {"Architecture": {"Database"}, "Database": set()}
    # Database ← Architecture (backward edge)
    index._backward = {"Database": {"Architecture"}, "Architecture": set()}
    return index


# ---------------------------------------------------------------------------
# Shared fixture
# ---------------------------------------------------------------------------


@pytest.fixture
async def client_ctx(tmp_path, monkeypatch):
    """Yield (client, app, session_factory) with in-memory SQLite and AUTH_ENABLED=false."""
    monkeypatch.setenv("AUTH_ENABLED", "false")
    monkeypatch.setenv("LLM_API_KEY", "test-key")

    engine, session_factory = await _make_session_factory()

    app = create_app()
    async with app.router.lifespan_context(app):
        storage = LocalArtifactStorage(str(tmp_path))
        wiki_mgmt = WikiManagementService(storage, session_factory)
        app.state.storage = storage
        app.state.wiki_management = wiki_mgmt
        app.state.session_factory = session_factory

        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as c:
            yield c, app, session_factory

    await engine.dispose()


# ---------------------------------------------------------------------------
# Wiki Search — GET /api/v1/wikis/{wiki_id}/search
# ---------------------------------------------------------------------------


class TestWikiSearch:
    """Tests for GET /api/v1/wikis/{wiki_id}/search."""

    @pytest.mark.asyncio
    async def test_authenticated_wiki_search_returns_200(self, client_ctx):
        """Scenario 1: authenticated search returns 200 with WikiSearchResponse shape."""
        c, app, session_factory = client_ctx
        wiki_id = "wiki-search-1"
        await _seed_wiki(session_factory, wiki_id=wiki_id)

        fake_index = _make_fake_page_index(wiki_id)
        mock_cache = MagicMock(spec=WikiPageIndexCache)
        mock_cache.get = AsyncMock(return_value=fake_index)
        app.state.wiki_index_cache = mock_cache

        # Mock both UnifiedWikiDB (file access) and WikiSearchEngine.search
        from app.core.unified_db import UnifiedWikiDB
        from app.core.wiki_search_engine import WikiSearchEngine

        fake_result = MagicMock()
        fake_result.query = "test"
        fake_result.results = []
        fake_result.wiki_summary = []

        mock_db = MagicMock(spec=UnifiedWikiDB)
        mock_db.close = MagicMock()

        with (
            patch("app.core.unified_db.UnifiedWikiDB", return_value=mock_db),
            patch.object(WikiSearchEngine, "search", new_callable=AsyncMock, return_value=fake_result),
        ):
            resp = await c.get(f"/api/v1/wikis/{wiki_id}/search?q=test")

        assert resp.status_code == 200, resp.text
        data = resp.json()
        assert data["query"] == "test"
        assert "results" in data
        assert "wiki_summary" in data

    @pytest.mark.asyncio
    async def test_unauthenticated_wiki_search_returns_401(self, client_ctx):
        """Scenario 2: no auth token → 401."""
        c, app, session_factory = client_ctx
        wiki_id = "wiki-unauth"
        await _seed_wiki(session_factory, wiki_id=wiki_id)

        # Force auth on for this request by overriding the dependency to raise 401
        from fastapi import HTTPException

        async def require_auth():
            raise HTTPException(status_code=401, detail="Not authenticated")

        app.dependency_overrides[get_current_user] = require_auth
        try:
            resp = await c.get(f"/api/v1/wikis/{wiki_id}/search?q=test")
            assert resp.status_code == 401, resp.text
        finally:
            app.dependency_overrides.pop(get_current_user, None)

    @pytest.mark.asyncio
    async def test_wiki_belonging_to_other_user_returns_404(self, client_ctx):
        """Scenario 3: wiki owned by another user → 404 (not 403)."""
        c, app, session_factory = client_ctx
        wiki_id = "wiki-other-owner"
        # Seed the wiki under a different owner
        await _seed_wiki(session_factory, wiki_id=wiki_id, owner_id="other-user")

        # dev-user (the default AUTH_ENABLED=false identity) should not see it
        resp = await c.get(f"/api/v1/wikis/{wiki_id}/search?q=test")
        assert resp.status_code == 404, resp.text

    @pytest.mark.asyncio
    async def test_hop_depth_6_returns_422(self, client_ctx):
        """Scenario 5: hop_depth exceeds max=5 → 422 Unprocessable Entity."""
        c, _app, session_factory = client_ctx
        wiki_id = "wiki-hop-422"
        await _seed_wiki(session_factory, wiki_id=wiki_id)

        resp = await c.get(f"/api/v1/wikis/{wiki_id}/search?q=test&hop_depth=6")
        assert resp.status_code == 422, resp.text

    @pytest.mark.asyncio
    async def test_top_k_default_of_10_applied(self, client_ctx):
        """Scenario 6: omitting top_k still returns up to 10 results."""
        c, app, session_factory = client_ctx
        wiki_id = "wiki-topk-default"
        await _seed_wiki(session_factory, wiki_id=wiki_id)

        fake_index = _make_fake_page_index(wiki_id)
        mock_cache = MagicMock(spec=WikiPageIndexCache)
        mock_cache.get = AsyncMock(return_value=fake_index)
        app.state.wiki_index_cache = mock_cache

        from app.core.unified_db import UnifiedWikiDB
        from app.core.wiki_search_engine import WikiSearchEngine

        captured: dict = {}

        async def _mock_search(self_engine, q, *, hop_depth=1, top_k=10):
            captured["top_k"] = top_k
            result = MagicMock()
            result.query = q
            result.results = []
            result.wiki_summary = []
            return result

        mock_db = MagicMock(spec=UnifiedWikiDB)
        mock_db.close = MagicMock()

        with (
            patch("app.core.unified_db.UnifiedWikiDB", return_value=mock_db),
            patch.object(WikiSearchEngine, "search", new=_mock_search),
        ):
            resp = await c.get(f"/api/v1/wikis/{wiki_id}/search?q=hello")

        assert resp.status_code == 200, resp.text
        assert captured["top_k"] == 10, f"Expected default top_k=10, got {captured['top_k']}"


# ---------------------------------------------------------------------------
# Wiki Pages — GET /api/v1/wikis/{wiki_id}/pages
# ---------------------------------------------------------------------------


class TestWikiPages:
    """Tests for GET /api/v1/wikis/{wiki_id}/pages."""

    @pytest.mark.asyncio
    async def test_list_wiki_pages_returns_page_list_with_descriptions(self, client_ctx):
        """Scenario 7: returns page list with titles and descriptions."""
        c, app, session_factory = client_ctx
        wiki_id = "wiki-pages-1"
        await _seed_wiki(session_factory, wiki_id=wiki_id)

        fake_index = _make_fake_page_index(wiki_id)
        mock_cache = MagicMock(spec=WikiPageIndexCache)
        mock_cache.get = AsyncMock(return_value=fake_index)
        app.state.wiki_index_cache = mock_cache

        resp = await c.get(f"/api/v1/wikis/{wiki_id}/pages")
        assert resp.status_code == 200, resp.text
        data = resp.json()
        assert data["wiki_id"] == wiki_id
        pages = data["pages"]
        assert len(pages) == 2

        by_title = {p["page_title"]: p for p in pages}
        assert "Architecture" in by_title
        assert "Database" in by_title
        assert by_title["Architecture"]["description"] == "High-level architecture overview"
        assert by_title["Database"]["description"] == "Database schema and access patterns"
        # Section is extracted from the path segment after wiki_pages/
        assert by_title["Architecture"]["section"] == "intro"


# ---------------------------------------------------------------------------
# Page Neighbors — GET /api/v1/wikis/{wiki_id}/pages/{page_title}/neighbors
# ---------------------------------------------------------------------------


class TestPageNeighbors:
    """Tests for GET /api/v1/wikis/{wiki_id}/pages/{page_title}/neighbors."""

    @pytest.mark.asyncio
    async def test_get_page_neighbors_returns_links_to_and_linked_from(self, client_ctx):
        """Scenario 8: returns links_to and linked_from for the requested page."""
        c, app, session_factory = client_ctx
        wiki_id = "wiki-neighbors-1"
        await _seed_wiki(session_factory, wiki_id=wiki_id)

        fake_index = _make_fake_page_index(wiki_id)
        mock_cache = MagicMock(spec=WikiPageIndexCache)
        mock_cache.get = AsyncMock(return_value=fake_index)
        app.state.wiki_index_cache = mock_cache

        # "Architecture" links_to "Database"; "Database" is linked_from "Architecture"
        resp = await c.get(f"/api/v1/wikis/{wiki_id}/pages/Architecture/neighbors")
        assert resp.status_code == 200, resp.text
        data = resp.json()
        assert data["wiki_id"] == wiki_id
        assert data["page_title"] == "Architecture"
        assert "Database" in data["links_to"]
        assert isinstance(data["linked_from"], list)

    @pytest.mark.asyncio
    async def test_get_neighbors_unknown_page_returns_404(self, client_ctx):
        """Page that doesn't exist in the index returns 404."""
        c, app, session_factory = client_ctx
        wiki_id = "wiki-neighbors-2"
        await _seed_wiki(session_factory, wiki_id=wiki_id)

        fake_index = _make_fake_page_index(wiki_id)
        mock_cache = MagicMock(spec=WikiPageIndexCache)
        mock_cache.get = AsyncMock(return_value=fake_index)
        app.state.wiki_index_cache = mock_cache

        resp = await c.get(f"/api/v1/wikis/{wiki_id}/pages/NonExistentPage/neighbors")
        assert resp.status_code == 404, resp.text


# ---------------------------------------------------------------------------
# Project Search — GET /api/v1/projects/{project_id}/search
# ---------------------------------------------------------------------------


class TestProjectSearch:
    """Tests for GET /api/v1/projects/{project_id}/search."""

    @pytest.mark.asyncio
    async def test_project_search_wiki_summary_always_present(self, client_ctx):
        """Scenario 9: wiki_summary is always present in the response."""
        c, app, session_factory = client_ctx

        # Create a project
        create_resp = await c.post("/api/v1/projects", json={"name": "Search Project"})
        assert create_resp.status_code == 201, create_resp.text
        project_id = create_resp.json()["id"]

        from app.core.project_search_engine import ProjectSearchEngine

        fake_result = MagicMock()
        fake_result.query = "architecture"
        fake_result.results = []
        fake_result.wiki_summary = []

        with patch.object(ProjectSearchEngine, "search", new_callable=AsyncMock, return_value=fake_result):
            resp = await c.get(f"/api/v1/projects/{project_id}/search?q=architecture")

        assert resp.status_code == 200, resp.text
        data = resp.json()
        assert "wiki_summary" in data
        assert isinstance(data["wiki_summary"], list)

    @pytest.mark.asyncio
    async def test_project_search_zero_hit_wikis_returns_match_count_zero(self, client_ctx):
        """Scenario 4: wikis with no results have match_count=0 in wiki_summary."""
        c, app, session_factory = client_ctx
        wiki_id = "wiki-zero-hits"
        await _seed_wiki(session_factory, wiki_id=wiki_id)

        # Create project and add wiki
        create_resp = await c.post("/api/v1/projects", json={"name": "Zero Hit Project"})
        project_id = create_resp.json()["id"]
        await c.post(f"/api/v1/projects/{project_id}/wikis", json={"wiki_id": wiki_id})

        from app.models.search import WikiSummaryItem

        zero_hit_summary = WikiSummaryItem(
            wiki_id=wiki_id,
            wiki_name="Test Wiki",
            match_count=0,
            relevance=0.0,
        )

        from app.core.project_search_engine import ProjectSearchEngine

        fake_result = MagicMock()
        fake_result.query = "obscure-query"
        fake_result.results = []
        fake_result.wiki_summary = [zero_hit_summary]

        fake_index = _make_fake_page_index(wiki_id)
        mock_cache = MagicMock(spec=WikiPageIndexCache)
        mock_cache.get = AsyncMock(return_value=fake_index)
        app.state.wiki_index_cache = mock_cache

        with patch.object(ProjectSearchEngine, "search", new_callable=AsyncMock, return_value=fake_result):
            resp = await c.get(f"/api/v1/projects/{project_id}/search?q=obscure-query")

        assert resp.status_code == 200, resp.text
        data = resp.json()
        summaries = {s["wiki_id"]: s for s in data["wiki_summary"]}
        assert wiki_id in summaries
        assert summaries[wiki_id]["match_count"] == 0

    @pytest.mark.asyncio
    async def test_project_search_hop_depth_6_returns_422(self, client_ctx):
        """hop_depth=6 exceeds max=5 → 422 Unprocessable Entity."""
        c, _app, session_factory = client_ctx

        create_resp = await c.post("/api/v1/projects", json={"name": "HopDepth Project"})
        project_id = create_resp.json()["id"]

        resp = await c.get(f"/api/v1/projects/{project_id}/search?q=test&hop_depth=6")
        assert resp.status_code == 422, resp.text

    @pytest.mark.asyncio
    async def test_project_search_nonexistent_project_returns_404(self, client_ctx):
        """Searching a project that does not exist returns 404."""
        c, _app, _sf = client_ctx
        resp = await c.get("/api/v1/projects/does-not-exist/search?q=test")
        assert resp.status_code == 404, resp.text


# ---------------------------------------------------------------------------
# AUTH_ENABLED=false mode
# ---------------------------------------------------------------------------


class TestAuthEnabledFalseMode:
    """Scenario 10: full suite must run without JWT setup (AUTH_ENABLED=false).

    The conftest sets AUTH_ENABLED=false globally, so these tests confirm
    that the default dev-user identity is injected automatically and the
    search endpoints are reachable without providing any token.
    """

    @pytest.mark.asyncio
    async def test_wiki_search_works_without_jwt(self, client_ctx):
        """No Authorization header required when AUTH_ENABLED=false."""
        c, app, session_factory = client_ctx
        wiki_id = "wiki-noauth-search"
        await _seed_wiki(session_factory, wiki_id=wiki_id)

        fake_index = _make_fake_page_index(wiki_id)
        mock_cache = MagicMock(spec=WikiPageIndexCache)
        mock_cache.get = AsyncMock(return_value=fake_index)
        app.state.wiki_index_cache = mock_cache

        from app.core.unified_db import UnifiedWikiDB
        from app.core.wiki_search_engine import WikiSearchEngine

        fake_result = MagicMock()
        fake_result.query = "auth"
        fake_result.results = []
        fake_result.wiki_summary = []

        mock_db = MagicMock(spec=UnifiedWikiDB)
        mock_db.close = MagicMock()

        with (
            patch("app.core.unified_db.UnifiedWikiDB", return_value=mock_db),
            patch.object(WikiSearchEngine, "search", new_callable=AsyncMock, return_value=fake_result),
        ):
            # Deliberately send no Authorization header
            resp = await c.get(f"/api/v1/wikis/{wiki_id}/search?q=auth")

        assert resp.status_code == 200, f"Expected 200 without token in AUTH_ENABLED=false mode, got {resp.status_code}"

    @pytest.mark.asyncio
    async def test_wiki_pages_works_without_jwt(self, client_ctx):
        """list_wiki_pages is reachable without a token when auth is disabled."""
        c, app, session_factory = client_ctx
        wiki_id = "wiki-noauth-pages"
        await _seed_wiki(session_factory, wiki_id=wiki_id)

        fake_index = _make_fake_page_index(wiki_id)
        mock_cache = MagicMock(spec=WikiPageIndexCache)
        mock_cache.get = AsyncMock(return_value=fake_index)
        app.state.wiki_index_cache = mock_cache

        resp = await c.get(f"/api/v1/wikis/{wiki_id}/pages")
        assert resp.status_code == 200, f"Expected 200 without token in AUTH_ENABLED=false mode, got {resp.status_code}"
