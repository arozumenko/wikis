"""Unit tests for wiki search, pages, and neighbors REST endpoints."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.core.wiki_page_index import PageMeta, WikiPageIndex, WikiPageIndexCache
from app.core.wiki_search_engine import (
    NeighborItem,
    ResultItem,
    WikiSearchResult,
    WikiSummaryItem as EngineWikiSummaryItem,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_page_meta(title: str, description: str = "", section_path: bool = False) -> PageMeta:
    file_path = (
        f"wiki-1/repo/wiki_pages/section-a/{title}.md"
        if section_path
        else f"wiki-1/repo/wiki_pages/{title}.md"
    )
    return PageMeta(title=title, description=description or f"Desc of {title}", file_path=file_path)


def _make_page_index(
    titles: list[str],
    *,
    neighbors_map: dict[str, list[str]] | None = None,
    backlinks_map: dict[str, list[str]] | None = None,
    section_path: bool = False,
) -> WikiPageIndex:
    index = MagicMock(spec=WikiPageIndex)
    index.pages = {t: _make_page_meta(t, section_path=section_path) for t in titles}

    def _neighbors(title, hop_depth=1):
        titles_list = (neighbors_map or {}).get(title, [])
        return [index.pages[t] for t in titles_list if t in index.pages]

    def _backlinks(title):
        return list((backlinks_map or {}).get(title, []))

    index.neighbors.side_effect = _neighbors
    index.backlinks.side_effect = _backlinks
    return index


def _make_wiki_record(wiki_id: str = "wiki-1", title: str = "Test Wiki") -> MagicMock:
    record = MagicMock()
    record.id = wiki_id
    record.title = title
    record.owner_id = "user-1"
    return record


# ---------------------------------------------------------------------------
# App fixture factory
# ---------------------------------------------------------------------------


def _build_app(
    wiki_record=None,
    page_index=None,
    storage_content: bytes | None = None,
    search_result: WikiSearchResult | None = None,
) -> FastAPI:
    from app.api.routes import router
    from app.auth import get_current_user
    from app.dependencies import (
        get_wiki_index_cache,
        get_wiki_management,
    )

    app = FastAPI()
    app.include_router(router)

    # Mock management
    mock_management = AsyncMock()
    mock_management.get_wiki = AsyncMock(return_value=wiki_record)
    mock_storage = AsyncMock()
    mock_storage.download = AsyncMock(return_value=storage_content or b"")
    mock_management.storage = mock_storage

    # Mock page index cache
    mock_cache = AsyncMock(spec=WikiPageIndexCache)
    mock_cache.get = AsyncMock(return_value=page_index or MagicMock(spec=WikiPageIndex, pages={}))

    # Mock settings, session factory, and other state
    app.state.settings = MagicMock(cache_dir="/tmp/test_cache")
    app.state.session_factory = MagicMock()
    app.state.wiki_management = mock_management
    app.state.wiki_index_cache = mock_cache

    # Override DI
    app.dependency_overrides[get_wiki_management] = lambda: mock_management
    app.dependency_overrides[get_wiki_index_cache] = lambda: mock_cache
    app.dependency_overrides[get_current_user] = lambda: MagicMock(id="user-1")

    # Attach the search result for patching
    app.state._search_result = search_result

    return app


# ---------------------------------------------------------------------------
# Tests: GET /api/v1/wikis/{wiki_id}/search
# ---------------------------------------------------------------------------


class TestSearchEndpoint:
    def _make_search_result(self) -> WikiSearchResult:
        return WikiSearchResult(
            query="auth",
            results=[
                ResultItem(
                    wiki_id="wiki-1",
                    wiki_name="Test Wiki",
                    page_title="Authentication",
                    snippet="Desc of Authentication",
                    score=0.9,
                    neighbors=[NeighborItem(title="Authorization", rel="links_to")],
                )
            ],
            wiki_summary=[
                EngineWikiSummaryItem(
                    wiki_id="wiki-1",
                    wiki_name="Test Wiki",
                    match_count=1,
                    relevance=0.5,
                )
            ],
        )

    def test_search_returns_200_with_correct_shape(self):
        """Search returns 200 with WikiSearchResponse shape."""
        wiki = _make_wiki_record()
        page_index = _make_page_index(["Authentication", "Authorization"])
        search_result = self._make_search_result()
        app = _build_app(wiki_record=wiki, page_index=page_index, search_result=search_result)

        with patch("app.core.wiki_search_engine.WikiSearchEngine") as MockEngine:
            mock_engine = MagicMock()
            mock_engine.search = AsyncMock(return_value=search_result)
            MockEngine.return_value = mock_engine

            client = TestClient(app)
            resp = client.get("/api/v1/wikis/wiki-1/search?q=auth")

        assert resp.status_code == 200
        data = resp.json()
        assert data["query"] == "auth"
        assert len(data["results"]) == 1
        assert data["results"][0]["page_title"] == "Authentication"
        assert data["results"][0]["score"] == pytest.approx(0.9)
        assert data["results"][0]["neighbors"][0]["title"] == "Authorization"
        assert data["results"][0]["neighbors"][0]["rel"] == "links_to"
        assert len(data["wiki_summary"]) == 1
        assert data["wiki_summary"][0]["match_count"] == 1

    def test_search_hop_depth_too_large_returns_422(self):
        """hop_depth=6 exceeds maximum and returns 422."""
        wiki = _make_wiki_record()
        app = _build_app(wiki_record=wiki, page_index=_make_page_index([]))
        client = TestClient(app)
        resp = client.get("/api/v1/wikis/wiki-1/search?q=test&hop_depth=6")
        assert resp.status_code == 422

    def test_search_unknown_wiki_returns_404(self):
        """Search on unknown wiki_id returns 404."""
        app = _build_app(wiki_record=None, page_index=_make_page_index([]))
        client = TestClient(app)
        resp = client.get("/api/v1/wikis/no-such-wiki/search?q=test")
        assert resp.status_code == 404

    def test_search_empty_query_returns_200(self):
        """Search with empty query string still returns 200."""
        wiki = _make_wiki_record()
        page_index = _make_page_index([])
        empty_result = WikiSearchResult(query="", results=[], wiki_summary=[])
        app = _build_app(wiki_record=wiki, page_index=page_index)

        with patch("app.core.wiki_search_engine.WikiSearchEngine") as MockEngine:
            mock_engine = MagicMock()
            mock_engine.search = AsyncMock(return_value=empty_result)
            MockEngine.return_value = mock_engine

            client = TestClient(app)
            resp = client.get("/api/v1/wikis/wiki-1/search?q=")

        assert resp.status_code == 200
        data = resp.json()
        assert data["query"] == ""
        assert data["results"] == []


# ---------------------------------------------------------------------------
# Tests: GET /api/v1/wikis/{wiki_id}/pages
# ---------------------------------------------------------------------------


class TestPagesListEndpoint:
    def test_pages_list_returns_200_with_all_pages(self):
        """Pages list returns 200 with all pages from the index."""
        wiki = _make_wiki_record()
        page_index = _make_page_index(["Page A", "Page B", "Page C"])
        app = _build_app(wiki_record=wiki, page_index=page_index)

        client = TestClient(app)
        resp = client.get("/api/v1/wikis/wiki-1/pages")

        assert resp.status_code == 200
        data = resp.json()
        assert data["wiki_id"] == "wiki-1"
        titles = {p["page_title"] for p in data["pages"]}
        assert titles == {"Page A", "Page B", "Page C"}

    def test_pages_list_unknown_wiki_returns_404(self):
        """Pages list on unknown wiki returns 404."""
        app = _build_app(wiki_record=None, page_index=_make_page_index([]))
        client = TestClient(app)
        resp = client.get("/api/v1/wikis/no-such-wiki/pages")
        assert resp.status_code == 404

    def test_pages_list_section_extracted_from_nested_path(self):
        """Section is extracted from nested wiki_pages/ path."""
        wiki = _make_wiki_record()
        page_index = _make_page_index(["Deep Page"], section_path=True)
        app = _build_app(wiki_record=wiki, page_index=page_index)

        client = TestClient(app)
        resp = client.get("/api/v1/wikis/wiki-1/pages")

        assert resp.status_code == 200
        data = resp.json()
        assert len(data["pages"]) == 1
        assert data["pages"][0]["section"] == "section-a"

    def test_pages_list_top_level_page_has_no_section(self):
        """Top-level pages (no subdirectory) have section=null."""
        wiki = _make_wiki_record()
        page_index = _make_page_index(["Top Page"], section_path=False)
        app = _build_app(wiki_record=wiki, page_index=page_index)

        client = TestClient(app)
        resp = client.get("/api/v1/wikis/wiki-1/pages")

        assert resp.status_code == 200
        data = resp.json()
        page = next(p for p in data["pages"] if p["page_title"] == "Top Page")
        assert page["section"] is None


# ---------------------------------------------------------------------------
# Tests: GET /api/v1/wikis/{wiki_id}/pages/{page_title}
# ---------------------------------------------------------------------------

_SAMPLE_MD = b"""---
title: My Page
description: A sample page
---
# Introduction
Some text here.
## Details
More details.
"""


class TestGetPageEndpoint:
    def test_get_page_returns_200_with_content_and_sections(self):
        """Get page returns 200 with content and extracted sections."""
        wiki = _make_wiki_record()
        page_index = _make_page_index(["My Page"])
        app = _build_app(wiki_record=wiki, page_index=page_index, storage_content=_SAMPLE_MD)

        client = TestClient(app)
        resp = client.get("/api/v1/wikis/wiki-1/pages/My Page")

        assert resp.status_code == 200
        data = resp.json()
        assert data["wiki_id"] == "wiki-1"
        assert data["page_title"] == "My Page"
        assert "Some text here" in data["content"]
        assert "Introduction" in data["sections"]
        assert "Details" in data["sections"]

    def test_get_unknown_page_title_returns_404(self):
        """Get unknown page title returns 404."""
        wiki = _make_wiki_record()
        page_index = _make_page_index(["Real Page"])
        app = _build_app(wiki_record=wiki, page_index=page_index)

        client = TestClient(app)
        resp = client.get("/api/v1/wikis/wiki-1/pages/Nonexistent Page")
        assert resp.status_code == 404

    def test_get_page_unknown_wiki_returns_404(self):
        """Get page on unknown wiki returns 404."""
        app = _build_app(wiki_record=None, page_index=_make_page_index([]))
        client = TestClient(app)
        resp = client.get("/api/v1/wikis/no-such-wiki/pages/Any Page")
        assert resp.status_code == 404


# ---------------------------------------------------------------------------
# Tests: GET /api/v1/wikis/{wiki_id}/pages/{page_title}/neighbors
# ---------------------------------------------------------------------------


class TestNeighborsEndpoint:
    def test_neighbors_returns_200_with_links_to_and_linked_from(self):
        """Neighbors endpoint returns 200 with forward and backward links."""
        wiki = _make_wiki_record()
        page_index = _make_page_index(
            ["Auth", "Session", "Token"],
            neighbors_map={"Auth": ["Session", "Token"]},
            backlinks_map={"Auth": ["Session"]},
        )
        app = _build_app(wiki_record=wiki, page_index=page_index)

        client = TestClient(app)
        resp = client.get("/api/v1/wikis/wiki-1/pages/Auth/neighbors")

        assert resp.status_code == 200
        data = resp.json()
        assert data["wiki_id"] == "wiki-1"
        assert data["page_title"] == "Auth"
        assert set(data["links_to"]) == {"Session", "Token"}
        assert data["linked_from"] == ["Session"]

    def test_neighbors_hop_depth_too_large_returns_422(self):
        """hop_depth=6 exceeds maximum and returns 422."""
        wiki = _make_wiki_record()
        page_index = _make_page_index(["Auth"])
        app = _build_app(wiki_record=wiki, page_index=page_index)

        client = TestClient(app)
        resp = client.get("/api/v1/wikis/wiki-1/pages/Auth/neighbors?hop_depth=6")
        assert resp.status_code == 422

    def test_neighbors_unknown_page_returns_404(self):
        """Unknown page title for neighbors returns 404."""
        wiki = _make_wiki_record()
        page_index = _make_page_index(["Auth"])
        app = _build_app(wiki_record=wiki, page_index=page_index)

        client = TestClient(app)
        resp = client.get("/api/v1/wikis/wiki-1/pages/Nonexistent/neighbors")
        assert resp.status_code == 404

    def test_neighbors_unknown_wiki_returns_404(self):
        """Unknown wiki for neighbors returns 404."""
        app = _build_app(wiki_record=None, page_index=_make_page_index([]))
        client = TestClient(app)
        resp = client.get("/api/v1/wikis/no-such-wiki/pages/Auth/neighbors")
        assert resp.status_code == 404
