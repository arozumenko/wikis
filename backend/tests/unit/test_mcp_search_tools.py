"""Tests for MCP search tools: search_wiki, get_page_neighbors, search_project.

Also covers list_projects and search_project HTTP fallback paths.
"""
from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_result_item(wiki_id="wiki-1", wiki_name="My Wiki", page_title="Overview"):
    item = MagicMock()
    item.wiki_id = wiki_id
    item.wiki_name = wiki_name
    item.page_title = page_title
    item.snippet = "A brief description"
    item.score = 0.95
    n = MagicMock()
    n.title = "Related Page"
    n.rel = "links_to"
    item.neighbors = [n]
    return item


def _make_wiki_summary(wiki_id="wiki-1", wiki_name="My Wiki", match_count=1, relevance=0.5):
    s = MagicMock()
    s.wiki_id = wiki_id
    s.wiki_name = wiki_name
    s.match_count = match_count
    s.relevance = relevance
    return s


def _make_search_result(query="test", items=None, summaries=None):
    result = MagicMock()
    result.query = query
    result.results = items or [_make_result_item()]
    result.wiki_summary = summaries or [_make_wiki_summary()]
    return result


def _make_page_meta(title: str, description: str = ""):
    meta = MagicMock()
    meta.title = title
    meta.description = description
    return meta


def _make_page_index(pages=None):
    index = MagicMock()
    index.pages = pages or {"Overview": _make_page_meta("Overview", "Main overview")}
    index.neighbors = MagicMock(return_value=[_make_page_meta("Linked Page")])
    index.backlinks = MagicMock(return_value=["Source Page"])
    return index


def _make_wiki_record(wiki_id="wiki-1", title="My Wiki"):
    w = MagicMock()
    w.wiki_id = wiki_id
    w.title = title
    return w


def _make_project_wiki(id="wiki-1", title="My Wiki"):
    w = MagicMock()
    w.id = id
    w.title = title
    return w


# ---------------------------------------------------------------------------
# TASK-009: search_wiki — direct mode
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_search_wiki_returns_correct_shape_direct_mode():
    """search_wiki in direct mode returns dict with query, results, wiki_summary."""
    import mcp_server.server as srv

    mock_cache = AsyncMock()
    mock_cache.get = AsyncMock(return_value=_make_page_index())

    mock_session_factory = MagicMock()

    mock_mgmt = AsyncMock()
    wiki_list = MagicMock()
    wiki_list.wikis = [_make_wiki_record()]
    mock_mgmt.list_wikis = AsyncMock(return_value=wiki_list)

    mock_engine = AsyncMock()
    mock_engine.search = AsyncMock(return_value=_make_search_result())

    token = srv._current_user_id.set("user-1")
    old_cache = srv._page_index_cache
    old_session_factory = srv._session_factory
    old_mgmt = srv._wiki_management
    try:
        srv._page_index_cache = mock_cache
        srv._session_factory = mock_session_factory
        srv._wiki_management = mock_mgmt

        with patch("app.core.wiki_search_engine.WikiSearchEngine") as MockEngine:
            MockEngine.return_value = mock_engine
            result = await srv.search_wiki(wiki_id="wiki-1", query="authentication")

        assert "query" in result
        assert "results" in result
        assert "wiki_summary" in result
    finally:
        srv._current_user_id.reset(token)
        srv._page_index_cache = old_cache
        srv._session_factory = old_session_factory
        srv._wiki_management = old_mgmt


@pytest.mark.asyncio
async def test_search_wiki_raises_on_hop_depth_out_of_range():
    """search_wiki raises ValueError when hop_depth > 5."""
    import mcp_server.server as srv

    with pytest.raises(ValueError, match="hop_depth"):
        await srv.search_wiki(wiki_id="wiki-1", query="test", hop_depth=6)


@pytest.mark.asyncio
async def test_search_wiki_raises_on_hop_depth_below_range():
    """search_wiki raises ValueError when hop_depth < 1."""
    import mcp_server.server as srv

    with pytest.raises(ValueError, match="hop_depth"):
        await srv.search_wiki(wiki_id="wiki-1", query="test", hop_depth=0)


@pytest.mark.asyncio
async def test_search_wiki_calls_http_fallback_in_standalone_mode():
    """search_wiki calls HTTP fallback when no page_index_cache is configured."""
    import mcp_server.server as srv

    old_cache = srv._page_index_cache
    old_session_factory = srv._session_factory
    try:
        srv._page_index_cache = None
        srv._session_factory = None

        with patch("mcp_server.server._http_search_wiki", new_callable=AsyncMock) as mock_http:
            mock_http.return_value = {"query": "test", "results": [], "wiki_summary": []}
            result = await srv.search_wiki(wiki_id="wiki-1", query="test")

        mock_http.assert_called_once_with("wiki-1", "test", 1, 10)
        assert "results" in result
    finally:
        srv._page_index_cache = old_cache
        srv._session_factory = old_session_factory


# ---------------------------------------------------------------------------
# TASK-010: get_page_neighbors — direct mode
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_get_page_neighbors_returns_links_to_and_linked_from():
    """get_page_neighbors returns dict with links_to and linked_from lists."""
    import mcp_server.server as srv

    page_index = _make_page_index(pages={"Overview": _make_page_meta("Overview")})

    mock_cache = AsyncMock()
    mock_cache.get = AsyncMock(return_value=page_index)

    # Access-control mock: user can see "wiki-1".
    wiki_list = MagicMock()
    wiki_list.wikis = [_make_wiki_record("wiki-1")]
    mock_mgmt = AsyncMock()
    mock_mgmt.list_wikis = AsyncMock(return_value=wiki_list)

    token = srv._current_user_id.set("user-1")
    old_cache = srv._page_index_cache
    old_mgmt = srv._wiki_management
    try:
        srv._page_index_cache = mock_cache
        srv._wiki_management = mock_mgmt
        result = await srv.get_page_neighbors(wiki_id="wiki-1", page_title="Overview")

        assert result["wiki_id"] == "wiki-1"
        assert result["page_title"] == "Overview"
        assert "links_to" in result
        assert "linked_from" in result
        assert isinstance(result["links_to"], list)
        assert isinstance(result["linked_from"], list)
    finally:
        srv._current_user_id.reset(token)
        srv._page_index_cache = old_cache
        srv._wiki_management = old_mgmt


@pytest.mark.asyncio
async def test_get_page_neighbors_raises_on_unknown_page():
    """get_page_neighbors raises ValueError when page_title not in index."""
    import mcp_server.server as srv

    page_index = _make_page_index(pages={})  # empty index
    mock_cache = AsyncMock()
    mock_cache.get = AsyncMock(return_value=page_index)

    # Access-control mock: user can see "wiki-1" so we reach the page check.
    wiki_list = MagicMock()
    wiki_list.wikis = [_make_wiki_record("wiki-1")]
    mock_mgmt = AsyncMock()
    mock_mgmt.list_wikis = AsyncMock(return_value=wiki_list)

    old_cache = srv._page_index_cache
    old_mgmt = srv._wiki_management
    try:
        srv._page_index_cache = mock_cache
        srv._wiki_management = mock_mgmt
        with pytest.raises(ValueError, match="Page not found"):
            await srv.get_page_neighbors(wiki_id="wiki-1", page_title="NonExistent")
    finally:
        srv._page_index_cache = old_cache
        srv._wiki_management = old_mgmt


@pytest.mark.asyncio
async def test_get_page_neighbors_raises_on_hop_depth_out_of_range():
    """get_page_neighbors raises ValueError when hop_depth > 5."""
    import mcp_server.server as srv

    with pytest.raises(ValueError, match="hop_depth"):
        await srv.get_page_neighbors(wiki_id="wiki-1", page_title="Overview", hop_depth=6)


@pytest.mark.asyncio
async def test_get_page_neighbors_calls_http_fallback_in_standalone_mode():
    """get_page_neighbors calls HTTP fallback when no page_index_cache configured."""
    import mcp_server.server as srv

    old_cache = srv._page_index_cache
    try:
        srv._page_index_cache = None

        with patch("mcp_server.server._http_get_page_neighbors", new_callable=AsyncMock) as mock_http:
            mock_http.return_value = {"wiki_id": "wiki-1", "page_title": "Overview", "links_to": [], "linked_from": []}
            result = await srv.get_page_neighbors(wiki_id="wiki-1", page_title="Overview", hop_depth=2)

        mock_http.assert_called_once_with("wiki-1", "Overview", 2)
        assert result["page_title"] == "Overview"
    finally:
        srv._page_index_cache = old_cache


# ---------------------------------------------------------------------------
# TASK-011: search_project — direct mode
# ---------------------------------------------------------------------------


def _make_fake_session_factory_for_search(mock_project_svc):
    """Build a fake async session factory returning a mock ProjectService."""
    mock_session = AsyncMock()
    fake_ctx = AsyncMock()
    fake_ctx.__aenter__ = AsyncMock(return_value=mock_session)
    fake_ctx.__aexit__ = AsyncMock(return_value=False)
    fake_factory = MagicMock(return_value=fake_ctx)
    return fake_factory, mock_session


@pytest.mark.asyncio
async def test_search_project_fans_out_across_wikis_direct_mode():
    """search_project fans out to ProjectSearchEngine with wikis from the project."""
    import mcp_server.server as srv

    wikis = [_make_project_wiki("wiki-1", "Repo One"), _make_project_wiki("wiki-2", "Repo Two")]

    mock_project_svc = AsyncMock()
    mock_project_svc.list_project_wikis = AsyncMock(return_value=wikis)

    fake_factory, _ = _make_fake_session_factory_for_search(mock_project_svc)

    page_index = _make_page_index()
    mock_cache = AsyncMock()
    mock_cache.get = AsyncMock(return_value=page_index)

    merged_result = _make_search_result(
        "auth",
        items=[_make_result_item("wiki-1"), _make_result_item("wiki-2")],
        summaries=[_make_wiki_summary("wiki-1"), _make_wiki_summary("wiki-2")],
    )

    token = srv._current_user_id.set("user-1")
    old_cache = srv._page_index_cache
    old_factory = srv._session_factory
    try:
        srv._page_index_cache = mock_cache
        srv._session_factory = fake_factory

        with patch("app.services.project_service.ProjectService") as MockPS, \
             patch("app.core.wiki_search_engine.WikiSearchEngine"), \
             patch("app.core.project_search_engine.ProjectSearchEngine") as MockPSE:
            MockPS.return_value = mock_project_svc
            mock_pse_instance = AsyncMock()
            mock_pse_instance.search = AsyncMock(return_value=merged_result)
            MockPSE.return_value = mock_pse_instance

            result = await srv.search_project(project_id="proj-1", query="auth")

        assert "results" in result
        assert "wiki_summary" in result
    finally:
        srv._current_user_id.reset(token)
        srv._page_index_cache = old_cache
        srv._session_factory = old_factory


@pytest.mark.asyncio
async def test_search_project_raises_on_hop_depth_out_of_range():
    """search_project raises ValueError when hop_depth > 5."""
    import mcp_server.server as srv

    with pytest.raises(ValueError, match="hop_depth"):
        await srv.search_project(project_id="proj-1", query="test", hop_depth=6)


@pytest.mark.asyncio
async def test_search_project_calls_http_fallback_in_standalone_mode():
    """search_project calls HTTP fallback when services not configured."""
    import mcp_server.server as srv

    old_cache = srv._page_index_cache
    old_factory = srv._session_factory
    try:
        srv._page_index_cache = None
        srv._session_factory = None

        with patch("mcp_server.server._http_search_project", new_callable=AsyncMock) as mock_http:
            mock_http.return_value = {"query": "test", "results": [], "wiki_summary": []}
            result = await srv.search_project(project_id="proj-1", query="test")

        mock_http.assert_called_once_with("proj-1", "test", 1, 10)
        assert "results" in result
    finally:
        srv._page_index_cache = old_cache
        srv._session_factory = old_factory


# ---------------------------------------------------------------------------
# Auth passthrough: _current_user_id used in direct-mode tools
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_search_wiki_passes_user_id_to_wiki_management():
    """search_wiki passes _current_user_id to wiki management list_wikis."""
    import mcp_server.server as srv

    mock_cache = AsyncMock()
    mock_cache.get = AsyncMock(return_value=_make_page_index())

    mock_session_factory = MagicMock()

    mock_mgmt = AsyncMock()
    wiki_list = MagicMock()
    wiki_list.wikis = [_make_wiki_record()]
    mock_mgmt.list_wikis = AsyncMock(return_value=wiki_list)

    mock_engine = AsyncMock()
    mock_engine.search = AsyncMock(return_value=_make_search_result())

    token = srv._current_user_id.set("expected-user-42")
    old_cache = srv._page_index_cache
    old_session_factory = srv._session_factory
    old_mgmt = srv._wiki_management
    try:
        srv._page_index_cache = mock_cache
        srv._session_factory = mock_session_factory
        srv._wiki_management = mock_mgmt

        with patch("app.core.wiki_search_engine.WikiSearchEngine") as MockEngine:
            MockEngine.return_value = mock_engine
            await srv.search_wiki(wiki_id="wiki-1", query="test")

        mock_mgmt.list_wikis.assert_called_once_with(user_id="expected-user-42")
    finally:
        srv._current_user_id.reset(token)
        srv._page_index_cache = old_cache
        srv._session_factory = old_session_factory
        srv._wiki_management = old_mgmt


@pytest.mark.asyncio
async def test_get_page_neighbors_auth_direct_mode():
    """get_page_neighbors works correctly with user context set."""
    import mcp_server.server as srv

    page_index = _make_page_index(pages={"Overview": _make_page_meta("Overview")})
    mock_cache = AsyncMock()
    mock_cache.get = AsyncMock(return_value=page_index)

    # Access-control mock: "auth-user-99" can see "wiki-1".
    wiki_list = MagicMock()
    wiki_list.wikis = [_make_wiki_record("wiki-1")]
    mock_mgmt = AsyncMock()
    mock_mgmt.list_wikis = AsyncMock(return_value=wiki_list)

    token = srv._current_user_id.set("auth-user-99")
    old_cache = srv._page_index_cache
    old_mgmt = srv._wiki_management
    try:
        srv._page_index_cache = mock_cache
        srv._wiki_management = mock_mgmt
        result = await srv.get_page_neighbors(wiki_id="wiki-1", page_title="Overview")
        # Should succeed and return the right structure
        assert result["wiki_id"] == "wiki-1"
        assert result["page_title"] == "Overview"
    finally:
        srv._current_user_id.reset(token)
        srv._page_index_cache = old_cache
        srv._wiki_management = old_mgmt


@pytest.mark.asyncio
async def test_search_project_auth_direct_mode():
    """search_project passes _current_user_id to project service."""
    import mcp_server.server as srv

    wikis = [_make_project_wiki("wiki-1", "Repo One")]
    mock_project_svc = AsyncMock()
    mock_project_svc.list_project_wikis = AsyncMock(return_value=wikis)

    fake_factory, _ = _make_fake_session_factory_for_search(mock_project_svc)

    page_index = _make_page_index()
    mock_cache = AsyncMock()
    mock_cache.get = AsyncMock(return_value=page_index)

    merged_result = _make_search_result("q")

    token = srv._current_user_id.set("auth-user-77")
    old_cache = srv._page_index_cache
    old_factory = srv._session_factory
    try:
        srv._page_index_cache = mock_cache
        srv._session_factory = fake_factory

        with patch("app.services.project_service.ProjectService") as MockPS, \
             patch("app.core.wiki_search_engine.WikiSearchEngine"), \
             patch("app.core.project_search_engine.ProjectSearchEngine") as MockPSE:
            MockPS.return_value = mock_project_svc
            mock_pse_instance = AsyncMock()
            mock_pse_instance.search = AsyncMock(return_value=merged_result)
            MockPSE.return_value = mock_pse_instance

            await srv.search_project(project_id="proj-1", query="q")

        # list_project_wikis should be called with the user_id from context
        mock_project_svc.list_project_wikis.assert_called_once_with("proj-1", user_id="auth-user-77")
    finally:
        srv._current_user_id.reset(token)
        srv._page_index_cache = old_cache
        srv._session_factory = old_factory


@pytest.mark.asyncio
async def test_search_wiki_blocks_inaccessible_wiki():
    """search_wiki must return an error dict when the wiki is not in the user's list."""
    import mcp_server.server as srv

    wiki_list_result = MagicMock()
    wiki_list_result.wikis = []  # user has no wikis

    mock_mgmt = AsyncMock()
    mock_mgmt.list_wikis = AsyncMock(return_value=wiki_list_result)

    mock_cache = AsyncMock()
    mock_session_factory = MagicMock()

    token = srv._current_user_id.set("test-user")
    old_mgmt = srv._wiki_management
    old_cache = srv._page_index_cache
    old_session_factory = srv._session_factory
    try:
        srv._wiki_management = mock_mgmt
        srv._page_index_cache = mock_cache
        srv._session_factory = mock_session_factory

        result = await srv.search_wiki(wiki_id="secret-wiki", query="anything")

        assert "error" in result
        assert "Wiki not found" in result["error"]
        # Cache must NOT be touched for an inaccessible wiki.
        mock_cache.get.assert_not_called()
        # user_id must be passed through.
        mock_mgmt.list_wikis.assert_called_once_with(user_id="test-user")
    finally:
        srv._current_user_id.reset(token)
        srv._wiki_management = old_mgmt
        srv._page_index_cache = old_cache
        srv._session_factory = old_session_factory
