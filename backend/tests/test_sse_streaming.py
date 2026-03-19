"""Tests for SSE streaming of wiki generation progress."""

from __future__ import annotations

import asyncio

import pytest

import app.events as events
from app.models.invocation import Invocation


class TestInvocationEventBroadcast:
    """Tests for the EventStore-inspired broadcast pattern."""

    @pytest.mark.asyncio
    async def test_emit_and_receive_events(self):
        inv = Invocation(id="test", wiki_id="w1")
        await inv.emit(events.progress("test", 10, 100, "Indexing"))
        await inv.emit(
            events.task_status("test", "completed", "Done", wiki_id="w1", page_count=5, execution_time=10.0)
        )

        received = []
        async for _idx, event in inv.events():
            received.append(event)

        assert len(received) == 2
        assert received[0].event == "progress"
        assert received[1].event == "task_complete"

    @pytest.mark.asyncio
    async def test_error_event_terminates_stream(self):
        inv = Invocation(id="test", wiki_id="w1")
        await inv.emit(events.progress("test", 10, 100, "Start"))
        await inv.emit(events.task_status("test", "failed", "LLM timeout", error="LLM timeout"))

        received = []
        async for _idx, event in inv.events():
            received.append(event)

        assert len(received) == 2
        assert received[-1].event == "task_failed"

    @pytest.mark.asyncio
    async def test_multiple_progress_events(self):
        inv = Invocation(id="test", wiki_id="w1")
        phases = ["configuring", "indexing", "generating"]
        for i, phase in enumerate(phases):
            await inv.emit(events.progress("test", (i + 1) * 30, 100, phase, phase=phase))
        await inv.emit(
            events.task_status("test", "completed", "Done", wiki_id="w1", page_count=10, execution_time=60.0)
        )

        received = []
        async for _idx, event in inv.events():
            received.append(event)

        assert len(received) == 4
        assert [e.event for e in received] == ["progress", "progress", "progress", "task_complete"]

    @pytest.mark.asyncio
    async def test_two_concurrent_readers_both_receive_all_events(self):
        """Two subscribers should each get all events (broadcast, not split)."""
        inv = Invocation(id="test", wiki_id="w1")

        reader1_events: list = []
        reader2_events: list = []

        async def reader(target: list):
            async for _idx, event in inv.events():
                target.append(event)

        # Start both readers
        t1 = asyncio.create_task(reader(reader1_events))
        t2 = asyncio.create_task(reader(reader2_events))

        # Give readers time to subscribe
        await asyncio.sleep(0.05)

        # Emit events
        await inv.emit(events.progress("test", 50, 100, "Half done"))
        await inv.emit(
            events.task_status("test", "completed", "Done", wiki_id="w1", page_count=3, execution_time=5.0)
        )

        await asyncio.gather(t1, t2)

        assert len(reader1_events) == 2
        assert len(reader2_events) == 2
        assert reader1_events[0].event == "progress"
        assert reader2_events[0].event == "progress"

    @pytest.mark.asyncio
    async def test_late_connecting_reader_gets_replay(self):
        """A reader connecting after events were emitted gets full replay."""
        inv = Invocation(id="test", wiki_id="w1")

        # Emit events with no subscribers
        await inv.emit(events.progress("test", 10, 100, "Step 1"))
        await inv.emit(events.progress("test", 50, 100, "Step 2"))
        await inv.emit(
            events.task_status("test", "completed", "Done", wiki_id="w1", page_count=1, execution_time=1.0)
        )

        # Late-connecting reader
        received = []
        async for _idx, event in inv.events():
            received.append(event)

        assert len(received) == 3
        assert received[0].event == "progress"
        assert received[2].event == "task_complete"

    @pytest.mark.asyncio
    async def test_after_event_id_reconnection(self):
        """Reconnection with after_event_id replays only newer events."""
        inv = Invocation(id="test", wiki_id="w1")

        await inv.emit(events.progress("test", 10, 100, "Step 1"))  # idx 0
        await inv.emit(events.progress("test", 50, 100, "Step 2"))  # idx 1
        await inv.emit(
            events.task_status("test", "completed", "Done", wiki_id="w1", page_count=1, execution_time=1.0)
        )  # idx 2

        # Reconnect after event 0 — should get events 1 and 2
        received = []
        async for idx, event in inv.events(after_event_id=0):
            received.append((idx, event))

        assert len(received) == 2
        assert received[0][0] == 1  # event index 1
        assert received[0][1].event == "progress"
        assert received[1][0] == 2  # event index 2
        assert received[1][1].event == "task_complete"

    @pytest.mark.asyncio
    async def test_terminal_invocation_returns_status(self):
        """A completed invocation yields a terminal event immediately."""
        inv = Invocation(id="test", wiki_id="w1", status="complete", pages_completed=5)
        inv.completed_at = inv.created_at

        received = []
        async for _idx, event in inv.events():
            received.append(event)

        assert len(received) == 1
        assert received[0].event == "task_complete"

    @pytest.mark.asyncio
    async def test_emit_sync_from_thread(self):
        """emit_sync called from a worker thread delivers events to an async reader."""
        inv = Invocation(id="test", wiki_id="w1")

        received: list = []

        async def reader():
            async for _idx, event in inv.events():
                received.append(event)

        task = asyncio.create_task(reader())
        await asyncio.sleep(0.05)  # let reader subscribe

        # Emit from a worker thread
        def thread_fn():
            inv.emit_sync(events.progress("test", 50, 100, "From thread"))
            inv.emit_sync(
                events.task_status("test", "completed", "Done", wiki_id="w1", page_count=1, execution_time=1.0)
            )

        loop = asyncio.get_running_loop()
        await loop.run_in_executor(None, thread_fn)
        await task

        assert len(received) == 2
        assert received[0].event == "progress"

    @pytest.mark.asyncio
    async def test_event_indices_are_sequential(self):
        """Event indices should match their position in the event log."""
        inv = Invocation(id="test", wiki_id="w1")

        await inv.emit(events.progress("test", 10, 100, "A"))
        await inv.emit(events.progress("test", 50, 100, "B"))
        await inv.emit(
            events.task_status("test", "completed", "Done", wiki_id="w1", page_count=1, execution_time=1.0)
        )

        indices = []
        async for idx, _event in inv.events():
            indices.append(idx)

        assert indices == [0, 1, 2]


class TestSSEEndpoint:
    @pytest.mark.asyncio
    async def test_stream_returns_text_event_stream(self):
        from httpx import ASGITransport, AsyncClient

        from app.main import create_app

        app = create_app()
        async with app.router.lifespan_context(app):
            # Create an invocation with pre-loaded events
            inv = Invocation(id="inv-1", wiki_id="w1")
            app.state.wiki_service._invocations["inv-1"] = inv

            await inv.emit(events.progress("inv-1", 10, 100, "Indexing", phase="indexing"))
            await inv.emit(
                events.task_status("inv-1", "completed", "Done", wiki_id="w1", page_count=5, execution_time=10.0)
            )

            async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as c:
                resp = await c.get("/api/v1/invocations/inv-1/stream")
                assert resp.status_code == 200
                assert "text/event-stream" in resp.headers["content-type"]
                body = resp.text
                assert "event: progress" in body or "event:progress" in body
                assert "event: task_complete" in body or "event:task_complete" in body

    @pytest.mark.asyncio
    async def test_stream_404_for_missing_invocation(self):
        from httpx import ASGITransport, AsyncClient

        from app.main import create_app

        app = create_app()
        async with app.router.lifespan_context(app):
            async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as c:
                resp = await c.get("/api/v1/invocations/nope/stream")
                assert resp.status_code == 404
