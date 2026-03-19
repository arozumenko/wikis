"""Tests for OpenAPI spec and route behavior."""

from __future__ import annotations

from unittest.mock import AsyncMock

import pytest
from httpx import ASGITransport, AsyncClient

from app.main import create_app
from app.models.api import AskResponse, ResearchResponse
from app.models.invocation import Invocation


@pytest.fixture
async def client():
    app = create_app()
    async with app.router.lifespan_context(app):
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as c:
            yield c, app


class TestOpenAPISpec:
    @pytest.mark.asyncio
    async def test_openapi_json_accessible(self, client):
        c, _ = client
        resp = await c.get("/openapi.json")
        assert resp.status_code == 200
        spec = resp.json()
        assert spec["info"]["title"] == "Wikis API"
        assert spec["info"]["version"] == "0.1.0"

    @pytest.mark.asyncio
    async def test_all_endpoints_in_spec(self, client):
        c, _ = client
        resp = await c.get("/openapi.json")
        paths = resp.json()["paths"]
        assert "/api/v1/generate" in paths
        assert "/api/v1/ask" in paths
        assert "/api/v1/research" in paths
        assert "/api/v1/wikis" in paths
        assert "/api/v1/wikis/{wiki_id}" in paths
        assert "/api/v1/health" in paths

    @pytest.mark.asyncio
    async def test_models_in_schema(self, client):
        c, _ = client
        resp = await c.get("/openapi.json")
        schemas = resp.json()["components"]["schemas"]
        for model in ["GenerateWikiRequest", "AskRequest", "ResearchRequest", "HealthResponse", "ErrorResponse"]:
            assert model in schemas, f"{model} missing from OpenAPI schemas"


class TestRoutes:
    @pytest.mark.asyncio
    async def test_health_returns_ok(self, client):
        c, _ = client
        resp = await c.get("/api/v1/health")
        assert resp.status_code == 200
        assert resp.json()["status"] in ("ok", "degraded")

    @pytest.mark.asyncio
    async def test_generate_returns_202(self, client):
        c, app = client
        mock_inv = Invocation(id="inv-1", wiki_id="w1", status="generating")
        app.state.wiki_service.generate = AsyncMock(return_value=mock_inv)
        resp = await c.post("/api/v1/generate", json={"repo_url": "https://github.com/t/r"})
        assert resp.status_code == 202

    @pytest.mark.asyncio
    async def test_ask_returns_200(self, client):
        c, app = client
        app.state.ask_service.ask_sync = AsyncMock(return_value=AskResponse(answer="ok", sources=[]))
        resp = await c.post("/api/v1/ask", json={"wiki_id": "w1", "question": "test"})
        assert resp.status_code == 200

    @pytest.mark.asyncio
    async def test_research_returns_200(self, client):
        c, app = client
        app.state.research_service.research_sync = AsyncMock(
            return_value=ResearchResponse(answer="ok", sources=[], research_steps=[])
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
        resp = await c.delete("/api/v1/wikis/w1")
        assert resp.status_code == 404

    @pytest.mark.asyncio
    async def test_generate_validation_error(self, client):
        c, _ = client
        resp = await c.post("/api/v1/generate", json={})
        assert resp.status_code == 422
