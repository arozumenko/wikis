"""Tests for QA Knowledge Flywheel route integration (Steps 9, 10, 12)."""

from __future__ import annotations

import json
import uuid
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from httpx import ASGITransport, AsyncClient

from app.models.api import AskRequest, AskResponse
from app.models.qa_api import AskResult, QAListResponse, QARecordingPayload, QARecordResponse, QAStatsResponse


# ---- Fixtures ----


@pytest.fixture
def mock_ask_service():
    service = AsyncMock()
    service.evict_cache = MagicMock()
    return service


@pytest.fixture
def mock_qa_service():
    service = AsyncMock()
    service.record_interaction = AsyncMock()
    service.list_qa = AsyncMock(return_value=([], 0))
    service.get_stats = AsyncMock(
        return_value={
            "total_count": 0,
            "cache_hit_count": 0,
            "validated_count": 0,
            "rejected_count": 0,
            "hit_rate": 0.0,
        }
    )
    service.invalidate_wiki = AsyncMock()
    return service


@pytest.fixture
def mock_wiki_management():
    mgmt = AsyncMock()
    wiki = MagicMock()
    wiki.id = "wiki-1"
    wiki.owner_id = "user-1"
    mgmt.get_wiki = AsyncMock(return_value=wiki)
    mgmt.delete_wiki = AsyncMock(return_value=MagicMock(deleted=True, message=None))
    return mgmt


@pytest.fixture
def mock_wiki_service():
    service = MagicMock()
    service.invocations = {}
    service.persist_invocations = AsyncMock()
    return service


@pytest.fixture
def mock_research_service():
    service = MagicMock()
    service.evict_cache = MagicMock()
    return service


@pytest.fixture
def app(mock_ask_service, mock_qa_service, mock_wiki_management, mock_wiki_service, mock_research_service):
    """Create a FastAPI app with mocked services."""
    from app.api.routes import router

    app = FastAPI()
    app.include_router(router)

    # Wire mock services to app.state
    app.state.ask_service = mock_ask_service
    app.state.qa_service = mock_qa_service
    app.state.wiki_management = mock_wiki_management
    app.state.wiki_service = mock_wiki_service
    app.state.research_service = mock_research_service
    app.state.settings = MagicMock(cache_dir="/tmp/cache")

    # Override DI
    from app.dependencies import (
        get_ask_service,
        get_qa_service,
        get_wiki_management,
        get_wiki_service,
        get_research_service,
    )
    from app.auth import get_current_user

    app.dependency_overrides[get_ask_service] = lambda: mock_ask_service
    app.dependency_overrides[get_qa_service] = lambda: mock_qa_service
    app.dependency_overrides[get_wiki_management] = lambda: mock_wiki_management
    app.dependency_overrides[get_wiki_service] = lambda: mock_wiki_service
    app.dependency_overrides[get_research_service] = lambda: mock_research_service
    app.dependency_overrides[get_current_user] = lambda: MagicMock(id="user-1")

    return app


@pytest.fixture
async def client(app):
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as c:
        yield c


# ---- Step 9: /ask recording ----


@pytest.mark.asyncio
async def test_ask_sync_returns_response_with_qa_id(client, mock_ask_service):
    """Sync /ask returns response with qa_id."""
    response = AskResponse(answer="test", sources=[], tool_steps=1, qa_id="qa-123")
    recording = QARecordingPayload(
        qa_id="qa-123",
        wiki_id="wiki-1",
        question="q",
        answer="test",
        sources_json="[]",
        tool_steps=1,
        mode="fast",
        user_id=None,
        is_cache_hit=False,
        source_qa_id=None,
        embedding=None,
    )
    mock_ask_service.ask_sync = AsyncMock(return_value=AskResult(response=response, recording=recording))

    resp = await client.post(
        "/api/v1/ask",
        json={"wiki_id": "wiki-1", "question": "What is auth?"},
    )
    assert resp.status_code == 200
    data = resp.json()
    assert data["answer"] == "test"
    assert data["qa_id"] == "qa-123"


@pytest.mark.asyncio
async def test_ask_sync_cache_hit(client, mock_ask_service, mock_qa_service):
    """Cache hit returns cached answer with BackgroundTask recording scheduled."""
    response = AskResponse(answer="cached", sources=[], tool_steps=0, qa_id="qa-hit")
    recording = QARecordingPayload(
        qa_id="qa-hit",
        wiki_id="wiki-1",
        question="q",
        answer="cached",
        sources_json="[]",
        tool_steps=0,
        mode="fast",
        user_id=None,
        is_cache_hit=True,
        source_qa_id="qa-orig",
        embedding=None,
    )
    mock_ask_service.ask_sync = AsyncMock(return_value=AskResult(response=response, recording=recording))

    resp = await client.post(
        "/api/v1/ask",
        json={"wiki_id": "wiki-1", "question": "cached q"},
    )
    assert resp.status_code == 200
    assert resp.json()["answer"] == "cached"
    # BackgroundTask should have scheduled recording
    mock_qa_service.record_interaction.assert_called_once()


# ---- Step 10: QA endpoints ----


@pytest.mark.asyncio
async def test_list_qa_returns_paginated(client, mock_qa_service):
    """GET /qa returns paginated QARecords."""
    record = MagicMock()
    record.id = "qa-1"
    record.wiki_id = "wiki-1"
    record.question = "q"
    record.answer = "a"
    record.sources_json = "[]"
    record.tool_steps = 0
    record.mode = "fast"
    record.status = "pending"
    record.is_cache_hit = 0
    record.has_context = 0
    record.created_at = None

    mock_qa_service.list_qa = AsyncMock(return_value=([record], 1))

    resp = await client.get("/api/v1/wikis/wiki-1/qa")
    assert resp.status_code == 200
    data = resp.json()
    assert data["total"] == 1
    assert len(data["items"]) == 1
    assert data["items"][0]["id"] == "qa-1"


@pytest.mark.asyncio
async def test_list_qa_with_status_filter(client, mock_qa_service):
    """GET /qa with status filter passes filter to service."""
    mock_qa_service.list_qa = AsyncMock(return_value=([], 0))

    resp = await client.get("/api/v1/wikis/wiki-1/qa?status=validated")
    assert resp.status_code == 200
    mock_qa_service.list_qa.assert_called_once_with(
        "wiki-1",
        status="validated",
        limit=20,
        offset=0,
    )


@pytest.mark.asyncio
async def test_qa_stats_returns_counts(client, mock_qa_service):
    """GET /qa/stats returns correct stats."""
    mock_qa_service.get_stats = AsyncMock(
        return_value={
            "total_count": 100,
            "cache_hit_count": 25,
            "validated_count": 10,
            "rejected_count": 5,
            "hit_rate": 0.25,
        }
    )

    resp = await client.get("/api/v1/wikis/wiki-1/qa/stats")
    assert resp.status_code == 200
    data = resp.json()
    assert data["total_count"] == 100
    assert data["hit_rate"] == 0.25


@pytest.mark.asyncio
async def test_list_qa_unauthorized(client, mock_wiki_management):
    """GET /qa returns 404 for unauthorized wiki."""
    mock_wiki_management.get_wiki = AsyncMock(return_value=None)

    resp = await client.get("/api/v1/wikis/secret-wiki/qa")
    assert resp.status_code == 404


@pytest.mark.asyncio
async def test_qa_stats_unauthorized(client, mock_wiki_management):
    """GET /qa/stats returns 404 for unauthorized wiki."""
    mock_wiki_management.get_wiki = AsyncMock(return_value=None)

    resp = await client.get("/api/v1/wikis/secret-wiki/qa/stats")
    assert resp.status_code == 404


# ---- Step 12: Wiki lifecycle ----


@pytest.mark.asyncio
async def test_delete_wiki_invalidates_qa(client, mock_qa_service, mock_wiki_management, mock_wiki_service):
    """Deleting a wiki removes all QARecords and FAISS file."""
    resp = await client.delete("/api/v1/wikis/wiki-1")
    assert resp.status_code == 200
    mock_qa_service.invalidate_wiki.assert_called_once_with("wiki-1", delete=True)


@pytest.mark.asyncio
async def test_refresh_wiki_invalidates_qa(client, mock_qa_service, mock_wiki_management, mock_wiki_service):
    """Refreshing a wiki marks pending→stale and deletes FAISS file."""
    invocation = MagicMock()
    invocation.wiki_id = "wiki-1"
    invocation.id = "inv-1"
    invocation.status = "running"
    mock_wiki_service.refresh = AsyncMock(return_value=invocation)
    from app.dependencies import get_wiki_service

    client._transport.app.dependency_overrides[get_wiki_service] = lambda: mock_wiki_service

    resp = await client.post("/api/v1/wikis/wiki-1/refresh")
    assert resp.status_code == 202
    mock_qa_service.invalidate_wiki.assert_called_once_with("wiki-1", delete=False)


@pytest.mark.asyncio
async def test_refresh_wiki_evicts_component_caches(
    client, mock_ask_service, mock_research_service, mock_qa_service, mock_wiki_management, mock_wiki_service
):
    """Refresh must evict Ask/Research component caches so stale indexes aren't reused."""
    invocation = MagicMock()
    invocation.wiki_id = "wiki-1"
    invocation.id = "inv-1"
    invocation.status = "running"
    mock_wiki_service.refresh = AsyncMock(return_value=invocation)
    from app.dependencies import get_wiki_service

    client._transport.app.dependency_overrides[get_wiki_service] = lambda: mock_wiki_service

    resp = await client.post("/api/v1/wikis/wiki-1/refresh")
    assert resp.status_code == 202
    mock_ask_service.evict_cache.assert_called_once_with("wiki-1")
    mock_research_service.evict_cache.assert_called_once_with("wiki-1")


@pytest.mark.asyncio
async def test_refresh_wiki_not_found_does_not_invalidate_qa(
    client, mock_qa_service, mock_wiki_management, mock_wiki_service
):
    """A refresh that returns 404 (wiki not found) must NOT invalidate QA cache."""
    mock_wiki_service.refresh = AsyncMock(return_value=None)
    mock_wiki_management.get_wiki = AsyncMock(return_value=None)
    from app.dependencies import get_wiki_service

    client._transport.app.dependency_overrides[get_wiki_service] = lambda: mock_wiki_service

    resp = await client.post("/api/v1/wikis/nonexistent/refresh")
    assert resp.status_code == 404
    mock_qa_service.invalidate_wiki.assert_not_called()


@pytest.mark.asyncio
async def test_refresh_wiki_unauthorized_does_not_invalidate_qa(
    client, mock_qa_service, mock_wiki_management, mock_wiki_service
):
    """A refresh rejected with 403 (wrong owner) must NOT invalidate QA cache."""
    # Return a wiki owned by a different user
    wrong_owner_wiki = MagicMock()
    wrong_owner_wiki.id = "wiki-1"
    wrong_owner_wiki.owner_id = "other-user"
    mock_wiki_management.get_wiki = AsyncMock(return_value=wrong_owner_wiki)

    resp = await client.post("/api/v1/wikis/wiki-1/refresh")
    assert resp.status_code == 403
    mock_qa_service.invalidate_wiki.assert_not_called()
