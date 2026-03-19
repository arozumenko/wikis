"""Tests for wiki refresh endpoint."""

from __future__ import annotations

from unittest.mock import AsyncMock

import pytest
from httpx import ASGITransport, AsyncClient
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

from app.main import create_app
from app.models.db_models import Base
from app.models.invocation import Invocation
from app.services.wiki_management import WikiManagementService
from app.storage.local import LocalArtifactStorage


async def _make_session_factory():
    engine = create_async_engine("sqlite+aiosqlite://", connect_args={"check_same_thread": False})
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    return async_sessionmaker(engine, expire_on_commit=False)


class TestRefreshEndpoint:
    @pytest.fixture
    async def client(self, tmp_path, monkeypatch):
        monkeypatch.setenv("AUTH_ENABLED", "false")
        session_factory = await _make_session_factory()
        app = create_app()
        async with app.router.lifespan_context(app):
            storage = LocalArtifactStorage(str(tmp_path))
            app.state.storage = storage
            app.state.wiki_management = WikiManagementService(storage, session_factory)
            async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as c:
                yield c, app

    @pytest.mark.asyncio
    async def test_refresh_existing_wiki(self, client):
        c, app = client
        mgmt = app.state.wiki_management
        await mgmt.register_wiki("w1", "https://github.com/a/b", "main", "Wiki", 5)
        mock_inv = Invocation(id="inv-r", wiki_id="w1", status="generating")
        app.state.wiki_service.generate = AsyncMock(return_value=mock_inv)
        resp = await c.post("/api/v1/wikis/w1/refresh")
        assert resp.status_code == 202
        assert resp.json()["wiki_id"] == "w1"

    @pytest.mark.asyncio
    async def test_refresh_nonexistent_wiki(self, client):
        c, _ = client
        resp = await c.post("/api/v1/wikis/nope/refresh")
        assert resp.status_code == 404

    @pytest.mark.asyncio
    async def test_refresh_preserves_old_wiki_during_generation(self, client):
        c, app = client
        mgmt = app.state.wiki_management
        await mgmt.register_wiki("w1", "https://github.com/a/b", "main", "Wiki", 5)
        mock_inv = Invocation(id="inv-r", wiki_id="w1", status="generating")
        app.state.wiki_service.generate = AsyncMock(return_value=mock_inv)
        await c.post("/api/v1/wikis/w1/refresh")
        # Old wiki still listed
        list_resp = await c.get("/api/v1/wikis")
        assert len(list_resp.json()["wikis"]) == 1
