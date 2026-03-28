"""Tests for dependency injection and lifespan setup."""

from __future__ import annotations

from unittest.mock import AsyncMock

import pytest
from httpx import ASGITransport, AsyncClient

from app.main import create_app
from app.models.api import AskResponse, ResearchResponse
from app.models.qa_api import AskResult
from app.models.invocation import Invocation
from app.services.ask_service import AskService
from app.services.research_service import ResearchService
from app.services.wiki_management import WikiManagementService
from app.services.wiki_service import WikiService


@pytest.fixture
async def test_app():
    """Create app and run lifespan."""
    app = create_app()
    async with app.router.lifespan_context(app):
        yield app


@pytest.fixture
async def client(test_app):
    async with AsyncClient(transport=ASGITransport(app=test_app), base_url="http://test") as c:
        yield c, test_app


class TestLifespanInitialization:
    @pytest.mark.asyncio
    async def test_services_initialized(self, test_app):
        assert isinstance(test_app.state.wiki_service, WikiService)
        assert isinstance(test_app.state.ask_service, AskService)
        assert isinstance(test_app.state.research_service, ResearchService)
        assert isinstance(test_app.state.wiki_management, WikiManagementService)

    @pytest.mark.asyncio
    async def test_settings_loaded(self, test_app):
        assert test_app.state.settings is not None
        assert test_app.state.settings.llm_provider in ("openai", "anthropic", "custom", "ollama", "gemini", "bedrock", "copilot", "github")

    @pytest.mark.asyncio
    async def test_storage_initialized(self, test_app):
        assert test_app.state.storage is not None


class TestDependencyInjection:
    @pytest.mark.asyncio
    async def test_generate_route_returns_202(self, client):
        c, app = client
        mock_inv = Invocation(id="inv-1", wiki_id="w1", status="generating")
        app.state.wiki_service.generate = AsyncMock(return_value=mock_inv)
        resp = await c.post("/api/v1/generate", json={"repo_url": "https://github.com/t/r"})
        assert resp.status_code == 202

    @pytest.mark.asyncio
    async def test_ask_route_resolves_service(self, client):
        c, app = client
        app.state.ask_service.ask_sync = AsyncMock(
            return_value=AskResult(response=AskResponse(answer="test", sources=[]), recording=None)
        )
        resp = await c.post("/api/v1/ask", json={"wiki_id": "w1", "question": "test"})
        assert resp.status_code == 200

    @pytest.mark.asyncio
    async def test_research_route_resolves_service(self, client):
        c, app = client
        app.state.research_service.research_sync = AsyncMock(
            return_value=ResearchResponse(answer="test", sources=[], research_steps=[])
        )
        resp = await c.post("/api/v1/research", json={"wiki_id": "w1", "question": "test"})
        assert resp.status_code == 200

    @pytest.mark.asyncio
    async def test_list_wikis_returns_200(self, client):
        c, _ = client
        resp = await c.get("/api/v1/wikis")
        assert resp.status_code == 200

    @pytest.mark.asyncio
    async def test_delete_wiki_returns_404_when_missing(self, client):
        c, _ = client
        resp = await c.delete("/api/v1/wikis/nonexistent")
        assert resp.status_code == 404

    @pytest.mark.asyncio
    async def test_health_returns_200(self, client):
        c, _ = client
        resp = await c.get("/api/v1/health")
        assert resp.status_code == 200
