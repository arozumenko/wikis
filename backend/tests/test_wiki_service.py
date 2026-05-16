"""Tests for WikiService — background generation, invocations, cancellation."""

from __future__ import annotations

import asyncio
import json
from datetime import datetime, timedelta
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


class TestLoadPersistedInvocationsOrphanSweep:
    """#146 Gap 1: ``load_persisted_invocations`` must flip ALL orphaned
    non-terminal statuses (``"generating"`` AND ``"running"``) to
    ``"failed"`` on startup. Before the fix, only ``"generating"`` was
    swept — leaving stranded incremental refreshes to 409-lock the
    wiki on the next request after a server restart.
    """

    @pytest.mark.asyncio
    async def test_running_status_swept_on_load(self, storage) -> None:
        # Pre-seed the persistence layer with a stranded "running"
        # invocation (no live task — simulates a server restart
        # mid-incremental-refresh).
        #
        # Use a created_at slightly in the past so the test doesn't
        # depend on clock skew.
        seeded = {
            "stranded-inv-1": {
                "id": "stranded-inv-1",
                "wiki_id": "wiki-A",
                "repo_url": "https://github.com/x/y",
                "branch": "main",
                "owner_id": "u1",
                "status": "running",
                "created_at": (datetime.now() - timedelta(minutes=10)).isoformat(),
            },
            "stranded-inv-2": {
                "id": "stranded-inv-2",
                "wiki_id": "wiki-B",
                "repo_url": "https://github.com/x/y",
                "branch": "main",
                "owner_id": "u1",
                "status": "generating",
                "created_at": (datetime.now() - timedelta(minutes=10)).isoformat(),
            },
        }
        raw = json.dumps(seeded).encode("utf-8")
        await storage.upload(
            WikiService.INVOCATIONS_BUCKET,
            WikiService.INVOCATIONS_KEY,
            raw,
        )

        service = WikiService(_settings(), storage)
        await service.load_persisted_invocations()

        # Both stranded entries flipped to "failed". Pre-fix, only the
        # "generating" one would have been swept; "running" would have
        # silently stayed in-flight and 409-locked the wiki.
        for inv_id in ("stranded-inv-1", "stranded-inv-2"):
            assert inv_id in service._invocations
            inv = service._invocations[inv_id]
            assert inv.status == "failed", (
                f"{inv_id} still {inv.status!r} after sweep"
            )
            assert inv.completed_at is not None
            assert "Server restarted" in (inv.error or "")

    @pytest.mark.asyncio
    async def test_terminal_statuses_not_touched_on_load(self, storage) -> None:
        """Already-terminal statuses must NOT be flipped — they're
        legitimate prior-run records that the cleanup-purge can later
        reclaim. Without this guard, the sweep would clobber the
        history of completed runs."""
        seeded = {
            "done-inv": {
                "id": "done-inv",
                "wiki_id": "wiki-C",
                "repo_url": "https://github.com/x/y",
                "branch": "main",
                "owner_id": "u1",
                "status": "complete",
                "created_at": datetime.now().isoformat(),
                "completed_at": datetime.now().isoformat(),
            },
            "failed-inv": {
                "id": "failed-inv",
                "wiki_id": "wiki-D",
                "repo_url": "https://github.com/x/y",
                "branch": "main",
                "owner_id": "u1",
                "status": "failed",
                "created_at": datetime.now().isoformat(),
                "completed_at": datetime.now().isoformat(),
            },
        }
        raw = json.dumps(seeded).encode("utf-8")
        await storage.upload(
            WikiService.INVOCATIONS_BUCKET,
            WikiService.INVOCATIONS_KEY,
            raw,
        )

        service = WikiService(_settings(), storage)
        await service.load_persisted_invocations()

        assert service._invocations["done-inv"].status == "complete"
        assert service._invocations["failed-inv"].status == "failed"


class TestWikiServiceShutdown:
    """Regression guard for CI segfault on fixture teardown.

    The crash happened when ``pytest_asyncio`` closed the event loop
    while a background task was still awaiting ``asyncio.to_thread()``
    for SQLite work. Cancellation interrupts the await but leaves the
    OS thread running against soon-to-be-closed storage.
    ``WikiService.shutdown()`` must wait for the task — and through it,
    the worker thread — to finish naturally before returning.
    """

    @pytest.mark.asyncio
    async def test_shutdown_waits_for_to_thread_workers_to_finish(self, service):
        """A task that internally calls ``asyncio.to_thread`` must be
        awaited to completion — the thread keeps running even if the
        asyncio.Task is cancelled, so we explicitly do NOT cancel.

        Without ``shutdown()``, this exact pattern is what segfaults
        in CI: shutdown returns, the loop closes, the OS thread keeps
        writing to a connection that's already closed.
        """
        import time

        thread_finished = False

        def slow_thread_work():
            # Tiny sleep simulates the SQLite FTS rebuild that
            # actually segfaults in CI when interrupted mid-flight.
            time.sleep(0.05)
            nonlocal thread_finished
            thread_finished = True

        async def run_with_to_thread():
            await asyncio.to_thread(slow_thread_work)

        task = asyncio.create_task(run_with_to_thread())
        service._tasks["test-invocation"] = task

        # Shutdown should block until the to_thread worker drains.
        shutdown_started = time.monotonic()
        await service.shutdown(timeout=5.0)
        elapsed = time.monotonic() - shutdown_started

        assert task.done()
        assert thread_finished, (
            "shutdown() returned before the worker thread finished — "
            "this is the exact race that causes CI segfaults"
        )
        # Shutdown must have actually waited, not returned instantly.
        assert elapsed >= 0.04, (
            f"shutdown returned in {elapsed:.3f}s — it should have "
            f"waited for the ~0.05s worker thread to finish"
        )

    @pytest.mark.asyncio
    async def test_shutdown_is_a_noop_when_no_tasks_pending(self, service):
        """Should not raise or hang when there's nothing to drain."""
        await service.shutdown(timeout=1.0)

    @pytest.mark.asyncio
    async def test_shutdown_does_not_cancel_pending_tasks(self, service):
        """Cancelling would mark the task done but leave the inner
        thread running — the bug we're guarding against. Verify
        ``shutdown`` lets tasks complete instead of cancelling."""
        completed = False

        async def finish_naturally():
            nonlocal completed
            await asyncio.sleep(0.05)
            completed = True

        task = asyncio.create_task(finish_naturally())
        service._tasks["nat"] = task

        await service.shutdown(timeout=2.0)

        assert task.done()
        assert not task.cancelled(), (
            "task was cancelled instead of awaited — the underlying "
            "to_thread worker would still be running"
        )
        assert completed

    @pytest.mark.asyncio
    async def test_shutdown_does_not_cancel_to_thread_tasks(self, service):
        """The QA + tech-lead review on PR #165 pointed out that the
        first ``does_not_cancel`` test uses ``asyncio.sleep``, not
        ``asyncio.to_thread`` — so it doesn't actually exercise the
        case that segfaults in CI. This variant uses the real
        to_thread pattern: verifies that the asyncio.Task completes
        normally (not cancelled) AND the worker thread runs to
        completion."""
        import time

        thread_finished = False

        def slow_thread_work():
            time.sleep(0.05)
            nonlocal thread_finished
            thread_finished = True

        async def run_with_to_thread():
            await asyncio.to_thread(slow_thread_work)

        task = asyncio.create_task(run_with_to_thread())
        service._tasks["tt"] = task

        await service.shutdown(timeout=5.0)

        assert task.done()
        assert not task.cancelled(), (
            "to_thread-wrapped task was cancelled — the worker thread "
            "would still be running against soon-closed storage"
        )
        assert thread_finished

    @pytest.mark.asyncio
    async def test_shutdown_swallows_task_exceptions(self, service):
        """``return_exceptions=True`` on the gather: shutdown should
        not re-raise an error from one task and abandon the others."""

        async def boom():
            raise RuntimeError("simulated failure inside task")

        async def slow_ok():
            await asyncio.sleep(0.05)

        task_bad = asyncio.create_task(boom())
        task_good = asyncio.create_task(slow_ok())
        service._tasks["bad"] = task_bad
        service._tasks["good"] = task_good

        # Must not raise.
        await service.shutdown(timeout=2.0)

        assert task_bad.done()
        assert task_good.done()
        # Consume the exception so pytest doesn't warn about an
        # un-retrieved task exception.
        assert task_bad.exception() is not None

    @pytest.mark.asyncio
    async def test_shutdown_logs_and_returns_on_timeout(
        self, service, caplog,
    ):
        """A task that won't finish within the timeout should not
        block shutdown forever — it logs and returns."""

        async def never_finish():
            await asyncio.sleep(60)

        task = asyncio.create_task(never_finish())
        service._tasks["wedged"] = task

        with caplog.at_level("ERROR"):
            await service.shutdown(timeout=0.1)

        assert any(
            "WikiService.shutdown timeout" in rec.message
            for rec in caplog.records
        )
        # Clean up the still-pending task so pytest doesn't warn.
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass

    @pytest.mark.asyncio
    async def test_shutdown_does_not_cancel_on_timeout(self, service):
        """Reviewer-flagged Critical regression test: the timeout
        path must NOT cancel still-pending tasks. An earlier draft
        used ``asyncio.wait_for(gather(...))`` which cancelled the
        children on timeout — reintroducing the original segfault.
        """
        async def slow():
            await asyncio.sleep(0.5)

        task = asyncio.create_task(slow())
        service._tasks["slow"] = task

        await service.shutdown(timeout=0.05)

        # The task should still be pending — shutdown bailed without
        # cancelling. Its worker thread (if any) is free to finish.
        assert not task.done()
        assert not task.cancelled()

        # Clean up so pytest doesn't warn about a dangling task.
        await task
