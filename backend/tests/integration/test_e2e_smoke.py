"""E2E smoke test — validates full generate → list → ask → research → delete flow.

This test uses mocked core modules (no real LLM/git calls) but exercises
the full HTTP API surface end-to-end through the FastAPI app.

Run with: pytest tests/integration/test_e2e_smoke.py -v
"""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, patch

import pytest
from httpx import ASGITransport, AsyncClient
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

from app.main import create_app
from app.models.db_models import Base
from app.services.wiki_management import WikiManagementService
from app.storage.local import LocalArtifactStorage


async def _make_session_factory():
    engine = create_async_engine("sqlite+aiosqlite://", connect_args={"check_same_thread": False})
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    return async_sessionmaker(engine, expire_on_commit=False)


@pytest.fixture
async def e2e_client(tmp_path, monkeypatch):
    """Full app client with mocked generation backend."""
    monkeypatch.setenv("AUTH_ENABLED", "false")
    monkeypatch.setenv("LLM_API_KEY", "test-key")
    monkeypatch.setenv("HEALTH_CHECK_ON_STARTUP", "false")

    session_factory = await _make_session_factory()
    app = create_app()
    async with app.router.lifespan_context(app):
        # Use tmp_path storage so tests are isolated
        storage = LocalArtifactStorage(str(tmp_path))
        wiki_mgmt = WikiManagementService(storage, session_factory)
        app.state.storage = storage
        app.state.wiki_management = wiki_mgmt

        # Rewire wiki_service to use our storage + management
        from app.services.wiki_service import WikiService

        app.state.wiki_service = WikiService(app.state.settings, storage, wiki_management=wiki_mgmt)

        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as c:
            yield c, app


def _mock_generation_result():
    """Create a realistic wiki-generation result payload.

    The production generate path now runs through
    ``WikiService._run_wiki_subprocess()``. The smoke test should mock that
    seam directly so it still exercises the real HTTP route, background task,
    invocation tracking, page storage, wiki registration, and delete flow.
    """
    return {
        "success": True,
        "generated_pages": {
            "getting-started": "# Getting Started\nWelcome to the project.",
            "architecture": "# Architecture\nThe system uses microservices.",
            "api-reference": "# API Reference\nGET /health → 200",
        },
        "artifacts": [],
        "execution_time": 45.2,
        "errors": [],
    }


class TestE2ESmokeFlow:
    """Full flow: generate → poll → list → ask → research → delete."""

    @pytest.mark.asyncio
    async def test_full_e2e_flow(self, e2e_client):
        c, app = e2e_client

        with patch.object(
            app.state.wiki_service,
            "_run_wiki_subprocess",
            new_callable=AsyncMock,
            return_value=_mock_generation_result(),
        ) as mock_run:
            # === Step 1: Generate wiki ===
            resp = await c.post(
                "/api/v1/generate",
                json={
                    "repo_url": "https://github.com/test/smoke-repo",
                    "branch": "main",
                    "wiki_title": "Smoke Test Wiki",
                },
            )
            assert resp.status_code == 202, f"Generate failed: {resp.text}"
            gen_data = resp.json()
            wiki_id = gen_data["wiki_id"]
            assert wiki_id
            assert "invocations/" in gen_data["message"]

            # Extract invocation ID from message
            inv_id = gen_data["message"].split("invocations/")[-1]

            # === Step 2: Poll invocation until complete ===
            for _ in range(50):  # max 5 seconds
                resp = await c.get(f"/api/v1/invocations/{inv_id}")
                assert resp.status_code == 200
                inv_data = resp.json()
                if inv_data["status"] in ("complete", "failed"):
                    break
                await asyncio.sleep(0.1)

            assert inv_data["status"] == "complete", f"Generation failed: {inv_data}"
            assert inv_data["progress"] == 1.0

            # Verify the real execution seam was used.
            assert mock_run.await_count == 1
            run_kwargs = mock_run.await_args.kwargs
            assert run_kwargs["request"].repo_url == "https://github.com/test/smoke-repo"
            assert run_kwargs["invocation"].id == inv_id
            assert run_kwargs["invocation"].wiki_id == wiki_id

        # === Step 3: List wikis — should contain our wiki ===
        resp = await c.get("/api/v1/wikis")
        assert resp.status_code == 200
        wikis = resp.json()["wikis"]
        assert len(wikis) >= 1
        our_wiki = next((w for w in wikis if w["wiki_id"] == wiki_id), None)
        assert our_wiki is not None, f"Wiki {wiki_id} not in list: {wikis}"
        assert our_wiki["title"] == "Smoke Test Wiki"
        assert our_wiki["page_count"] == 3

        # === Step 4: Ask a question ===
        from app.models.api import AskResponse
        from app.models.qa_api import AskResult

        with patch.object(
            app.state.ask_service,
            "ask_sync",
            new_callable=AsyncMock,
            return_value=AskResult(
                response=AskResponse(answer="The project uses microservices.", sources=[]),
                recording=None,
            ),
        ):
            resp = await c.post(
                "/api/v1/ask",
                json={
                    "wiki_id": wiki_id,
                    "question": "What architecture does this project use?",
                },
            )
            assert resp.status_code == 200
            assert "microservices" in resp.json()["answer"]

        # === Step 5: Deep research ===
        from app.models.api import ResearchResponse

        app.state.research_service.research_sync = AsyncMock(
            return_value=ResearchResponse(
                answer="Detailed analysis shows microservice pattern.",
                sources=[],
                research_steps=["search_symbols", "analyze_graph"],
            )
        )
        resp = await c.post(
            "/api/v1/research",
            json={
                "wiki_id": wiki_id,
                "question": "Explain the architecture in detail",
            },
        )
        assert resp.status_code == 200
        assert len(resp.json()["research_steps"]) == 2

        # === Step 6: Delete wiki ===
        resp = await c.delete(f"/api/v1/wikis/{wiki_id}")
        assert resp.status_code == 200
        assert resp.json()["deleted"] is True

        # === Step 7: Verify deleted ===
        resp = await c.get("/api/v1/wikis")
        assert resp.status_code == 200
        remaining = [w for w in resp.json()["wikis"] if w["wiki_id"] == wiki_id]
        assert len(remaining) == 0, "Wiki should be deleted"

    @pytest.mark.asyncio
    async def test_health_endpoint(self, e2e_client):
        c, _ = e2e_client
        resp = await c.get("/api/v1/health")
        assert resp.status_code == 200
        assert resp.json()["status"] in ("ok", "degraded")  # degraded if LLM unreachable in test

    @pytest.mark.asyncio
    async def test_generate_validation_error(self, e2e_client):
        c, _ = e2e_client
        resp = await c.post("/api/v1/generate", json={})
        assert resp.status_code == 422

    @pytest.mark.asyncio
    async def test_delete_nonexistent_wiki(self, e2e_client):
        c, _ = e2e_client
        resp = await c.delete("/api/v1/wikis/nonexistent-id")
        assert resp.status_code == 404
