"""Tests for MCP server tools (HTTP fallback path and direct-mode access control)."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from mcp_server.server import ask_codebase, get_page_neighbors, get_wiki_page, search_wikis


def _mock_response(status_code=200, json_data=None):
    resp = MagicMock()
    resp.status_code = status_code
    resp.json.return_value = json_data or {}
    resp.raise_for_status = MagicMock()
    if status_code >= 400:
        resp.raise_for_status.side_effect = Exception(f"HTTP {status_code}")
    return resp


def _patch_httpx(mock_client):
    """Patch httpx.AsyncClient as an async context manager returning mock_client."""
    cm = AsyncMock()
    cm.__aenter__ = AsyncMock(return_value=mock_client)
    cm.__aexit__ = AsyncMock(return_value=False)
    return patch("httpx.AsyncClient", return_value=cm)


@pytest.fixture(autouse=True)
def _reset_mcp_globals():
    """Ensure MCP server globals are reset so HTTP fallback path is used."""
    import mcp_server.server as srv

    orig_storage = srv._storage
    orig_ask = srv._ask_service
    orig_mgmt = srv._wiki_management
    orig_qa = srv._qa_service
    srv._storage = None
    srv._ask_service = None
    srv._wiki_management = None
    srv._qa_service = None
    yield
    srv._storage = orig_storage
    srv._ask_service = orig_ask
    srv._wiki_management = orig_mgmt
    srv._qa_service = orig_qa


class TestSearchWikis:
    @pytest.mark.asyncio
    async def test_returns_matching_wikis(self):
        wikis_resp = _mock_response(
            json_data={
                "wikis": [
                    {
                        "wiki_id": "w1",
                        "title": "Test",
                        "repo_url": "https://github.com/a/b",
                        "branch": "main",
                        "page_count": 5,
                    }
                ]
            }
        )

        mock_client = AsyncMock()
        mock_client.get = AsyncMock(return_value=wikis_resp)

        with _patch_httpx(mock_client):
            result = await search_wikis("")
            assert result["count"] == 1
            assert result["wikis"][0]["wiki_id"] == "w1"

    @pytest.mark.asyncio
    async def test_empty_result_for_no_match(self):
        wikis_resp = _mock_response(json_data={"wikis": []})
        mock_client = AsyncMock()
        mock_client.get = AsyncMock(return_value=wikis_resp)

        with _patch_httpx(mock_client):
            result = await search_wikis("nope")
            assert result["count"] == 0
            assert result["wikis"] == []


class TestGetWikiPage:
    @pytest.mark.asyncio
    async def test_returns_page_content(self):
        resp = _mock_response(json_data={"page_id": "p1", "content": "# Hello"})
        mock_client = AsyncMock()
        mock_client.get = AsyncMock(return_value=resp)

        with _patch_httpx(mock_client):
            result = await get_wiki_page("w1", "p1")
            assert result["content"] == "# Hello"

    @pytest.mark.asyncio
    async def test_page_not_found(self):
        resp = _mock_response(status_code=404)
        mock_client = AsyncMock()
        mock_client.get = AsyncMock(return_value=resp)

        with _patch_httpx(mock_client):
            result = await get_wiki_page("w1", "nope")
            assert "error" in result


class TestGetWikiPageAccessControl:
    """Direct-mode (non-HTTP) access control for get_wiki_page."""

    @pytest.mark.asyncio
    async def test_blocks_inaccessible_wiki(self):
        """get_wiki_page must return an error dict (not content) when the wiki is not
        in the current user's list_wikis result set."""
        import mcp_server.server as srv

        # Build a wiki management mock that returns an empty list (no wikis for this user).
        wiki_list_result = MagicMock()
        wiki_list_result.wikis = []

        mgmt = AsyncMock()
        mgmt.list_wikis = AsyncMock(return_value=wiki_list_result)

        storage = AsyncMock()

        orig_mgmt = srv._wiki_management
        orig_storage = srv._storage
        try:
            srv._wiki_management = mgmt
            srv._storage = storage
            result = await get_wiki_page("w-other", "some-page")
            assert "error" in result
            assert "Wiki not found" in result["error"]
            # Storage must not be touched for an inaccessible wiki.
            storage.list_artifacts.assert_not_called()
        finally:
            srv._wiki_management = orig_mgmt
            srv._storage = orig_storage

    @pytest.mark.asyncio
    async def test_allows_accessible_wiki(self):
        """get_wiki_page proceeds to storage when the wiki is accessible to the user."""
        import mcp_server.server as srv

        wiki = MagicMock()
        wiki.wiki_id = "w1"

        wiki_list_result = MagicMock()
        wiki_list_result.wikis = [wiki]

        mgmt = AsyncMock()
        mgmt.list_wikis = AsyncMock(return_value=wiki_list_result)

        storage = AsyncMock()
        storage.list_artifacts = AsyncMock(return_value=["w1/repo/wiki_pages/some-page.md"])
        storage.download = AsyncMock(return_value=b"# Hello")

        orig_mgmt = srv._wiki_management
        orig_storage = srv._storage
        try:
            srv._wiki_management = mgmt
            srv._storage = storage
            result = await get_wiki_page("w1", "some-page")
            assert "error" not in result
            assert result["content"] == "# Hello"
        finally:
            srv._wiki_management = orig_mgmt
            srv._storage = orig_storage


class TestGetPageNeighborsAccessControl:
    """Direct-mode (non-HTTP) access control for get_page_neighbors."""

    @pytest.mark.asyncio
    async def test_blocks_inaccessible_wiki(self):
        """get_page_neighbors must return error dict when wiki is not accessible."""
        import mcp_server.server as srv

        wiki_list_result = MagicMock()
        wiki_list_result.wikis = []

        mgmt = AsyncMock()
        mgmt.list_wikis = AsyncMock(return_value=wiki_list_result)

        page_index_cache = AsyncMock()

        orig_mgmt = srv._wiki_management
        orig_cache = srv._page_index_cache
        try:
            srv._wiki_management = mgmt
            srv._page_index_cache = page_index_cache
            result = await get_page_neighbors("w-other", "SomePage")
            assert "error" in result
            assert "Wiki not found" in result["error"]
            # page_index_cache.get must not be called for an inaccessible wiki.
            page_index_cache.get.assert_not_called()
        finally:
            srv._wiki_management = orig_mgmt
            srv._page_index_cache = orig_cache

    @pytest.mark.asyncio
    async def test_allows_accessible_wiki(self):
        """get_page_neighbors proceeds when the wiki is accessible to the user."""
        import mcp_server.server as srv

        wiki = MagicMock()
        wiki.wiki_id = "w1"

        wiki_list_result = MagicMock()
        wiki_list_result.wikis = [wiki]

        mgmt = AsyncMock()
        mgmt.list_wikis = AsyncMock(return_value=wiki_list_result)

        page_meta = MagicMock()
        page_meta.title = "SomePage"

        page_index = MagicMock()
        page_index.pages = {"SomePage": page_meta}
        page_index.neighbors = MagicMock(return_value=[])
        page_index.backlinks = MagicMock(return_value=[])

        page_index_cache = AsyncMock()
        page_index_cache.get = AsyncMock(return_value=page_index)

        orig_mgmt = srv._wiki_management
        orig_cache = srv._page_index_cache
        try:
            srv._wiki_management = mgmt
            srv._page_index_cache = page_index_cache
            result = await get_page_neighbors("w1", "SomePage")
            assert "error" not in result
            assert result["wiki_id"] == "w1"
            assert result["page_title"] == "SomePage"
        finally:
            srv._wiki_management = orig_mgmt
            srv._page_index_cache = orig_cache


class TestAskCodebase:
    @pytest.mark.asyncio
    async def test_returns_answer(self):
        resp = _mock_response(json_data={"answer": "Auth uses JWT.", "sources": []})
        mock_client = AsyncMock()
        mock_client.post = AsyncMock(return_value=resp)

        with _patch_httpx(mock_client):
            result = await ask_codebase("w1", "How does auth work?")
            assert result["answer"] == "Auth uses JWT."

    @pytest.mark.asyncio
    async def test_wiki_not_found(self):
        resp = _mock_response(status_code=404)
        mock_client = AsyncMock()
        mock_client.post = AsyncMock(return_value=resp)

        with _patch_httpx(mock_client):
            result = await ask_codebase("nope", "question")
            assert "error" in result
