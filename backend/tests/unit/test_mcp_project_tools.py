"""Tests for MCP project tools: list_projects, ask_project, research_project, map_project."""
from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_project_record(id: str, name: str, description: str | None = None,
                          visibility: str = "personal", owner_id: str = "user-1") -> MagicMock:
    record = MagicMock()
    record.id = id
    record.name = name
    record.description = description
    record.visibility = visibility
    record.owner_id = owner_id
    return record


# ---------------------------------------------------------------------------
# list_projects
# ---------------------------------------------------------------------------


def _make_fake_session_factory(mock_ps):
    """Build a fake async session factory that returns a mock ProjectService via context manager."""
    mock_session = AsyncMock()

    fake_ctx = AsyncMock()
    fake_ctx.__aenter__ = AsyncMock(return_value=mock_session)
    fake_ctx.__aexit__ = AsyncMock(return_value=False)

    fake_factory = MagicMock(return_value=fake_ctx)
    return fake_factory, mock_session


@pytest.mark.asyncio
async def test_list_projects_returns_all_projects():
    """list_projects returns all accessible projects with wiki counts."""
    import mcp_server.server as srv

    projects = [
        _make_project_record("p-1", "Alpha"),
        _make_project_record("p-2", "Beta"),
    ]

    mock_ps = AsyncMock()
    mock_ps.list_projects = AsyncMock(return_value=projects)
    mock_ps.batch_get_wiki_counts = AsyncMock(return_value={"p-1": 3, "p-2": 1})

    token = srv._current_user_id.set("user-1")
    old_factory = srv._session_factory
    fake_factory, _ = _make_fake_session_factory(mock_ps)
    try:
        srv._session_factory = fake_factory

        # Patch at the import location inside the tool function
        with patch("app.services.project_service.ProjectService") as MockPS:
            MockPS.return_value = mock_ps
            result = await srv.list_projects()

        assert result["total"] == 2
        assert len(result["projects"]) == 2
        ids = {p["project_id"] for p in result["projects"]}
        assert "p-1" in ids
        assert "p-2" in ids
        counts = {p["project_id"]: p["wiki_count"] for p in result["projects"]}
        assert counts["p-1"] == 3
        assert counts["p-2"] == 1
    finally:
        srv._current_user_id.reset(token)
        srv._session_factory = old_factory


@pytest.mark.asyncio
async def test_list_projects_filters_by_query():
    """list_projects filters by name prefix when query is provided."""
    import mcp_server.server as srv

    all_projects = [
        _make_project_record("p-1", "Alpha"),
        _make_project_record("p-2", "Beta"),
        _make_project_record("p-3", "Alphabet"),
    ]

    mock_ps = AsyncMock()
    mock_ps.list_projects = AsyncMock(return_value=all_projects)
    mock_ps.batch_get_wiki_counts = AsyncMock(return_value={"p-1": 0, "p-3": 0})

    token = srv._current_user_id.set("user-1")
    old_factory = srv._session_factory
    fake_factory, _ = _make_fake_session_factory(mock_ps)
    try:
        srv._session_factory = fake_factory

        with patch("app.services.project_service.ProjectService") as MockPS:
            MockPS.return_value = mock_ps
            result = await srv.list_projects(query="alp")

        assert result["total"] == 2
        names = [p["name"] for p in result["projects"]]
        assert "Alpha" in names
        assert "Alphabet" in names
        assert "Beta" not in names
    finally:
        srv._current_user_id.reset(token)
        srv._session_factory = old_factory


@pytest.mark.asyncio
async def test_list_projects_no_session_factory_returns_error():
    """list_projects returns error dict when no session factory is configured."""
    import mcp_server.server as srv

    token = srv._current_user_id.set("user-1")
    old_factory = srv._session_factory
    try:
        srv._session_factory = None
        result = await srv.list_projects()
        assert "error" in result
    finally:
        srv._current_user_id.reset(token)
        srv._session_factory = old_factory


# ---------------------------------------------------------------------------
# ask_project
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_ask_project_delegates_to_ask_service():
    """ask_project calls ask_service with project_id and returns the response."""
    import mcp_server.server as srv
    from app.models.api import AskResponse

    response = AskResponse(answer="cross-repo answer", sources=[], tool_steps=2, qa_id="qa-x")
    mock_result = MagicMock()
    mock_result.response = response
    mock_result.recording = None

    mock_ask = AsyncMock()
    mock_ask.ask_sync = AsyncMock(return_value=mock_result)

    token = srv._current_user_id.set("user-1")
    old_ask = srv._ask_service
    try:
        srv._ask_service = mock_ask
        result = await srv.ask_project(project_id="proj-1", question="How does auth work?")
        assert result["answer"] == "cross-repo answer"
        mock_ask.ask_sync.assert_called_once()
        call_args = mock_ask.ask_sync.call_args
        req = call_args[0][0]
        assert req.project_id == "proj-1"
        assert req.question == "How does auth work?"
    finally:
        srv._current_user_id.reset(token)
        srv._ask_service = old_ask


@pytest.mark.asyncio
async def test_ask_project_no_service_returns_error():
    """ask_project returns error dict when ask_service is not available."""
    import mcp_server.server as srv

    token = srv._current_user_id.set("user-1")
    old_ask = srv._ask_service
    try:
        srv._ask_service = None
        result = await srv.ask_project(project_id="p", question="q")
        assert "error" in result
    finally:
        srv._current_user_id.reset(token)
        srv._ask_service = old_ask


# ---------------------------------------------------------------------------
# research_project
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_research_project_delegates_to_research_service():
    """research_project calls research_service with project_id and returns response."""
    import mcp_server.server as srv
    from app.models.api import ResearchResponse

    response = ResearchResponse(answer="deep answer", sources=[], research_steps=["step1"])
    mock_research = AsyncMock()
    mock_research.research_sync = AsyncMock(return_value=response)

    token = srv._current_user_id.set("user-1")
    old_research = srv._research_service
    try:
        srv._research_service = mock_research
        result = await srv.research_project(project_id="proj-1", question="deep question")
        assert result["answer"] == "deep answer"
        mock_research.research_sync.assert_called_once()
        call_args = mock_research.research_sync.call_args
        req = call_args[0][0]
        assert req.project_id == "proj-1"
        assert req.question == "deep question"
    finally:
        srv._current_user_id.reset(token)
        srv._research_service = old_research


@pytest.mark.asyncio
async def test_research_project_no_service_returns_error():
    """research_project returns error dict when research_service is not available."""
    import mcp_server.server as srv

    token = srv._current_user_id.set("user-1")
    old_research = srv._research_service
    try:
        srv._research_service = None
        result = await srv.research_project(project_id="p", question="q")
        assert "error" in result
    finally:
        srv._current_user_id.reset(token)
        srv._research_service = old_research


# ---------------------------------------------------------------------------
# map_project
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_map_project_delegates_to_codemap_sync():
    """map_project calls research_service.codemap_sync with project_id and entry_points."""
    import mcp_server.server as srv
    from app.models.api import ResearchResponse

    response = ResearchResponse(answer="codemap answer", sources=[], research_steps=[])
    mock_research = AsyncMock()
    mock_research.codemap_sync = AsyncMock(return_value=response)

    token = srv._current_user_id.set("user-1")
    old_research = srv._research_service
    try:
        srv._research_service = mock_research
        result = await srv.map_project(project_id="proj-1", entry_points=["main.py", "auth.py"])
        assert result["answer"] == "codemap answer"
        mock_research.codemap_sync.assert_called_once()
        call_args = mock_research.codemap_sync.call_args
        req = call_args[0][0]
        assert req.project_id == "proj-1"
    finally:
        srv._current_user_id.reset(token)
        srv._research_service = old_research


@pytest.mark.asyncio
async def test_map_project_no_service_returns_error():
    """map_project returns error dict when research_service is not available."""
    import mcp_server.server as srv

    token = srv._current_user_id.set("user-1")
    old_research = srv._research_service
    try:
        srv._research_service = None
        result = await srv.map_project(project_id="p", entry_points=[])
        assert "error" in result
    finally:
        srv._current_user_id.reset(token)
        srv._research_service = old_research
