"""Tests for MCP server tools (HTTP fallback path)."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from mcp_server.server import ask_codebase, get_wiki_page, search_wikis


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
