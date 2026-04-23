"""Tests for WikiService — background generation, invocations, cancellation."""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from httpx import ASGITransport, AsyncClient
from pydantic import SecretStr

from app.config import Settings
from app.main import create_app
from app.models.api import GenerateWikiRequest
from app.models.invocation import Invocation
from app.services.wiki_service import WikiService
from app.storage.local import LocalArtifactStorage


def _settings() -> Settings:
    return Settings(llm_api_key=SecretStr("test-key"))


@pytest.fixture
def storage(tmp_path):
    return LocalArtifactStorage(str(tmp_path))


@pytest.fixture
def service(storage):
    return WikiService(_settings(), storage)


@pytest.fixture
def request_obj():
    return GenerateWikiRequest(repo_url="https://github.com/test/repo")


def _mock_generation_result(generated_pages=None, artifacts=None, success=True):
    return {
        "success": success,
        "generated_pages": generated_pages or {},
        "artifacts": artifacts or [],
    }


class TestWikiServiceUnit:
    def test_make_wiki_id_deterministic(self):
        id1 = WikiService._make_wiki_id("https://github.com/a/b", "main")
        id2 = WikiService._make_wiki_id("https://github.com/a/b", "main")
        assert id1 == id2
        assert len(id1) == 16

    def test_make_wiki_id_varies_by_branch(self):
        id1 = WikiService._make_wiki_id("https://github.com/a/b", "main")
        id2 = WikiService._make_wiki_id("https://github.com/a/b", "dev")
        assert id1 != id2

    @pytest.mark.asyncio
    async def test_generate_returns_invocation(self, service, request_obj):
        with patch.object(service, "_run_generation", new_callable=AsyncMock):
            invocation = await service.generate(request_obj)
            assert isinstance(invocation, Invocation)
            assert invocation.status == "generating"
            assert invocation.wiki_id

    @pytest.mark.asyncio
    async def test_get_invocation(self, service, request_obj):
        with patch.object(service, "_run_generation", new_callable=AsyncMock):
            invocation = await service.generate(request_obj)
            found = await service.get_invocation(invocation.id)
            assert found is not None
            assert found.id == invocation.id

    @pytest.mark.asyncio
    async def test_get_invocation_not_found(self, service):
        result = await service.get_invocation("nonexistent")
        assert result is None

    @pytest.mark.asyncio
    async def test_cancel_nonexistent_returns_false(self, service):
        result = await service.cancel_invocation("nonexistent")
        assert result is False

    @pytest.mark.asyncio
    async def test_generation_failure_sets_status(self, service, request_obj):
        """If subprocess generation raises, invocation should be marked failed."""
        invocation = Invocation(id="test-id", wiki_id="test-wiki", status="generating")
        service._invocations["test-id"] = invocation

        with patch.object(service, "_run_wiki_subprocess", new_callable=AsyncMock, side_effect=ValueError("bad key")):
            await service._run_generation(invocation, request_obj)

        assert invocation.status == "failed"
        assert "bad key" in invocation.error


class TestRunGeneration:
    """Tests for _run_generation covering success, overrides, artifacts, cancellation."""

    @pytest.fixture
    def invocation(self):
        return Invocation(id="inv-1", wiki_id="wiki-1", status="generating")

    @pytest.mark.asyncio
    async def test_successful_generation_stores_artifacts(self, service, request_obj, invocation):
        service._invocations[invocation.id] = invocation

        with patch.object(
            service,
            "_run_wiki_subprocess",
            new_callable=AsyncMock,
            return_value=_mock_generation_result(
                generated_pages={
                    "getting-started": "# Getting Started\nWelcome.",
                    "architecture": "# Architecture\nMicroservices.",
                },
                artifacts=[{"name": "index.json", "data": b"{}"}],
            ),
        ):
            await service._run_generation(invocation, request_obj)

        assert invocation.status == "complete"
        assert invocation.progress == 1.0
        assert invocation.completed_at is not None
        # Verify pages were stored
        data1 = await service.storage.download("wiki_artifacts", "wiki-1/getting-started.md")
        assert data1 == b"# Getting Started\nWelcome."
        data2 = await service.storage.download("wiki_artifacts", "wiki-1/architecture.md")
        assert data2 == b"# Architecture\nMicroservices."

    @pytest.mark.asyncio
    async def test_llm_model_override_passed(self, service, invocation):
        req = GenerateWikiRequest(repo_url="https://github.com/t/r", llm_model="gpt-4o")
        service._invocations[invocation.id] = invocation

        with patch.object(
            service,
            "_run_wiki_subprocess",
            new_callable=AsyncMock,
            return_value=_mock_generation_result(),
        ) as mock_run:
            await service._run_generation(invocation, req)

        run_kwargs = mock_run.await_args.kwargs
        assert run_kwargs["request"].llm_model == "gpt-4o"
        assert run_kwargs["planner_type"] == "agent"
        assert run_kwargs["exclude_tests"] is False

    @pytest.mark.asyncio
    async def test_embedding_model_override_passed(self, service, invocation):
        req = GenerateWikiRequest(repo_url="https://github.com/t/r", embedding_model="text-embedding-ada-002")
        service._invocations[invocation.id] = invocation

        with patch.object(
            service,
            "_run_wiki_subprocess",
            new_callable=AsyncMock,
            return_value=_mock_generation_result(),
        ) as mock_run:
            await service._run_generation(invocation, req)

        run_kwargs = mock_run.await_args.kwargs
        assert run_kwargs["request"].embedding_model == "text-embedding-ada-002"
        assert run_kwargs["planner_type"] == "agent"

    @pytest.mark.asyncio
    async def test_non_dict_result_sets_failed(self, service, request_obj, invocation):
        service._invocations[invocation.id] = invocation

        with patch.object(service, "_run_wiki_subprocess", new_callable=AsyncMock, return_value="not a dict"):
            await service._run_generation(invocation, request_obj)

        assert invocation.status == "failed"
        assert "non-dict" in invocation.error

    @pytest.mark.asyncio
    async def test_cancellation_sets_cancelled_status(self, service, request_obj, invocation):
        service._invocations[invocation.id] = invocation

        with patch.object(service, "_run_wiki_subprocess", new_callable=AsyncMock, side_effect=asyncio.CancelledError()):
            await service._run_generation(invocation, request_obj)

        assert invocation.status == "cancelled"
        assert invocation.completed_at is not None

    @pytest.mark.asyncio
    async def test_task_removed_from_tasks_dict_on_completion(self, service, request_obj, invocation):
        service._invocations[invocation.id] = invocation
        service._tasks[invocation.id] = MagicMock()

        with patch.object(
            service,
            "_run_wiki_subprocess",
            new_callable=AsyncMock,
            return_value=_mock_generation_result(),
        ):
            await service._run_generation(invocation, request_obj)

        assert invocation.id not in service._tasks

    @pytest.mark.asyncio
    async def test_task_removed_from_tasks_dict_on_failure(self, service, request_obj, invocation):
        service._invocations[invocation.id] = invocation
        service._tasks[invocation.id] = MagicMock()

        with patch.object(service, "_run_wiki_subprocess", new_callable=AsyncMock, side_effect=ValueError("boom")):
            await service._run_generation(invocation, request_obj)

        assert invocation.id not in service._tasks


class TestEmitHelpers:
    @pytest.mark.asyncio
    async def test_emit_progress_updates_state(self):
        inv = Invocation(id="t", wiki_id="w")
        await WikiService._emit_progress(inv, "indexing", 0.5, "Indexing repo")
        assert inv.current_phase == "indexing"
        assert inv.progress == 0.5
        assert inv.message == "Indexing repo"

    @pytest.mark.asyncio
    async def test_emit_error_sends_event(self):
        inv = Invocation(id="t", wiki_id="w")
        await WikiService._emit_error(inv, "LLM timeout", recoverable=True)
        # Event is in the broadcast event log
        assert len(inv._event_log) == 1
        event = inv._event_log[0]
        assert event.event == "task_failed"
        assert event.params.get("error") == "LLM timeout"
        assert event.params.get("recoverable") is True


class TestCancelInvocation:
    @pytest.mark.asyncio
    async def test_cancel_running_task(self, service):
        mock_task = MagicMock()
        mock_task.done.return_value = False
        inv = Invocation(id="inv-1", wiki_id="w1", status="generating")
        service._invocations["inv-1"] = inv
        service._tasks["inv-1"] = mock_task

        result = await service.cancel_invocation("inv-1")
        assert result is True
        mock_task.cancel.assert_called_once()

    @pytest.mark.asyncio
    async def test_cancel_already_completed_returns_false(self, service):
        inv = Invocation(id="inv-1", wiki_id="w1", status="complete")
        service._invocations["inv-1"] = inv

        result = await service.cancel_invocation("inv-1")
        assert result is False

    @pytest.mark.asyncio
    async def test_cancel_done_task_returns_false(self, service):
        mock_task = MagicMock()
        mock_task.done.return_value = True
        inv = Invocation(id="inv-1", wiki_id="w1", status="generating")
        service._invocations["inv-1"] = inv
        service._tasks["inv-1"] = mock_task

        result = await service.cancel_invocation("inv-1")
        assert result is False


class TestWikiServiceRoutes:
    @pytest.fixture
    async def client(self, monkeypatch):
        monkeypatch.setenv("AUTH_ENABLED", "false")
        app = create_app()
        async with app.router.lifespan_context(app):
            async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as c:
                yield c, app

    @pytest.mark.asyncio
    async def test_generate_returns_202(self, client):
        c, app = client
        with patch.object(
            app.state.wiki_service,
            "generate",
            new_callable=AsyncMock,
            return_value=Invocation(id="inv-1", wiki_id="wiki-1", status="generating"),
        ):
            resp = await c.post("/api/v1/generate", json={"repo_url": "https://github.com/t/r"})
            assert resp.status_code == 202
            data = resp.json()
            assert data["wiki_id"] == "wiki-1"
            assert "inv-1" in data["message"]

    @pytest.mark.asyncio
    async def test_get_invocation_200(self, client):
        c, app = client
        inv = Invocation(id="inv-1", wiki_id="wiki-1", status="generating", progress=0.5)
        with patch.object(
            app.state.wiki_service,
            "get_invocation",
            new_callable=AsyncMock,
            return_value=inv,
        ):
            resp = await c.get("/api/v1/invocations/inv-1")
            assert resp.status_code == 200
            assert resp.json()["progress"] == 0.5

    @pytest.mark.asyncio
    async def test_get_invocation_404(self, client):
        c, app = client
        with patch.object(
            app.state.wiki_service,
            "get_invocation",
            new_callable=AsyncMock,
            return_value=None,
        ):
            resp = await c.get("/api/v1/invocations/nope")
            assert resp.status_code == 404

    @pytest.mark.asyncio
    async def test_cancel_invocation_success(self, client):
        c, app = client
        with patch.object(
            app.state.wiki_service,
            "cancel_invocation",
            new_callable=AsyncMock,
            return_value=True,
        ):
            resp = await c.delete("/api/v1/invocations/inv-1")
            assert resp.status_code == 200
            assert resp.json()["cancelled"] is True

    @pytest.mark.asyncio
    async def test_cancel_invocation_404(self, client):
        c, app = client
        with patch.object(
            app.state.wiki_service,
            "cancel_invocation",
            new_callable=AsyncMock,
            return_value=False,
        ):
            resp = await c.delete("/api/v1/invocations/inv-1")
            assert resp.status_code == 404
