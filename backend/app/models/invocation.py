"""Invocation tracking model for background wiki generation.

Uses an EventStore-inspired broadcast pattern (modeled after
mcp.server.streamable_http.EventStore) so that multiple SSE clients
can subscribe concurrently and late-connecting clients get full replay.
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncGenerator
from datetime import datetime
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

TERMINAL_EVENTS = frozenset({"task_complete", "task_failed", "task_cancelled"})

# Cap event log to prevent unbounded memory growth during long generations.
# Typical generation emits ~50–200 events; 2000 is generous headroom.
# Note: Last-Event-ID reconnection only works within the current process
# lifetime — event log is not persisted across server restarts.
MAX_EVENT_LOG = 2000


class Invocation(BaseModel):
    """Tracks the state of a background wiki generation job."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    id: str
    wiki_id: str
    repo_url: str = ""
    branch: str = ""
    owner_id: str = ""
    status: str = "generating"  # generating | complete | failed | cancelled | partial
    progress: float = 0.0
    current_phase: str = ""
    message: str = ""
    created_at: datetime = Field(default_factory=datetime.now)
    completed_at: datetime | None = None
    error: str | None = None
    pages_completed: int = 0
    pages_total: int | None = None
    estimated_tokens: int | None = None
    model_context_limit: int | None = None

    # --- Broadcast event store (excluded from serialization) ---
    _event_log: list = []
    _subscribers: list = []
    _loop: asyncio.AbstractEventLoop | None = None

    def model_post_init(self, __context: Any) -> None:
        # Each instance gets its own mutable containers
        object.__setattr__(self, "_event_log", [])
        object.__setattr__(self, "_subscribers", [])
        try:
            object.__setattr__(self, "_loop", asyncio.get_running_loop())
        except RuntimeError:
            object.__setattr__(self, "_loop", None)

    # --- Emit (event loop thread) ---

    async def emit(self, event: Any) -> None:
        """Push an SSE event to all subscribers and append to the event log."""
        self.emit_sync(event)

    def emit_sync(self, event: Any) -> None:
        """Thread-safe emit for use from worker threads.

        Appends to the event log (list.append is GIL-atomic in CPython)
        and notifies subscribers via the event loop.
        """
        self._event_log.append(event)
        # Prevent unbounded growth — drop oldest non-terminal events.
        if len(self._event_log) > MAX_EVENT_LOG:
            self._event_log = self._event_log[-MAX_EVENT_LOG:]

        if not self._subscribers:
            return

        loop = self._loop
        if loop is None:
            return

        # From the event-loop thread we can put_nowait directly;
        # from worker threads we schedule on the loop.
        try:
            running = asyncio.get_running_loop() is loop
        except RuntimeError:
            running = False

        if running:
            # Same event loop — safe to call put_nowait directly
            dead: list[asyncio.Queue] = []
            for q in list(self._subscribers):
                try:
                    q.put_nowait(event)
                except asyncio.QueueFull:
                    dead.append(q)
            for q in dead:
                try:
                    self._subscribers.remove(q)
                except ValueError:
                    pass
        else:
            # Worker thread — schedule on the event loop.
            # Wrap put_nowait so QueueFull removes the dead subscriber.
            subs = self._subscribers  # capture reference for closure

            for q in list(subs):
                def _put(q=q, event=event):  # noqa: E731
                    try:
                        q.put_nowait(event)
                    except asyncio.QueueFull:
                        try:
                            subs.remove(q)
                        except ValueError:
                            pass

                loop.call_soon_threadsafe(_put)

    # --- Subscribe & consume ---

    async def events(self, after_event_id: int | None = None) -> AsyncGenerator[tuple[int, object], None]:
        """Yield ``(event_index, event)`` tuples for SSE streaming.

        * Late-connecting clients get full replay from ``_event_log``.
        * ``after_event_id`` enables ``Last-Event-ID`` reconnection — replay
          starts from the event *after* this index.
        * Multiple concurrent callers each get their own subscriber queue
          so every client sees every event (broadcast, not single-consumer).
        """
        import app.events as _events

        # ----- Terminal: replay from log if available, else synthesize -----
        if self.status in ("complete", "failed", "cancelled", "partial"):
            if self._event_log:
                # Replay the event log so reconnecting clients see full history
                start = (after_event_id + 1) if after_event_id is not None else 0
                for idx in range(start, len(self._event_log)):
                    yield (idx, self._event_log[idx])
                return
            # No log (e.g., loaded from persistence) — synthesize terminal event
            if self.status == "complete":
                elapsed = (self.completed_at - self.created_at).total_seconds() if self.completed_at else 0
                yield (0, _events.task_status(
                    self.id,
                    "completed",
                    "Wiki generation complete",
                    wiki_id=self.wiki_id,
                    page_count=self.pages_completed,
                    execution_time=elapsed,
                ))
            elif self.status == "cancelled":
                yield (0, _events.task_status(
                    self.id,
                    "cancelled",
                    self.message or "Generation cancelled",
                ))
            else:
                yield (0, _events.task_status(
                    self.id,
                    "failed",
                    self.error or self.message or self.status,
                    error=self.error or self.message or self.status,
                ))
            return

        # ----- Subscribe BEFORE replay to avoid missing events -----
        q: asyncio.Queue = asyncio.Queue(maxsize=4096)
        self._subscribers.append(q)

        try:
            # Replay from event log
            start = (after_event_id + 1) if after_event_id is not None else 0
            cursor = start
            while cursor < len(self._event_log):
                event = self._event_log[cursor]
                yield (cursor, event)
                if getattr(event, "event", "") in TERMINAL_EVENTS:
                    return
                cursor += 1

            # Drain subscriber queue of events already covered by replay.
            # Because we subscribed before replay, events emitted during
            # replay landed in both _event_log (read by cursor) and q.
            # Only drain exactly (cursor - start) items — that's how many
            # the replay loop yielded. Any extra items in q are NEW events
            # emitted after replay finished and must NOT be discarded.
            replayed_count = cursor - start
            for _ in range(replayed_count):
                try:
                    q.get_nowait()
                except asyncio.QueueEmpty:
                    break

            # ----- Live events -----
            while True:
                try:
                    event = await asyncio.wait_for(q.get(), timeout=300)
                except TimeoutError:
                    yield (cursor, _events.task_status(
                        self.id, "failed", "Stream timeout", error="Stream timeout",
                    ))
                    return

                yield (cursor, event)
                cursor += 1
                if getattr(event, "event", "") in TERMINAL_EVENTS:
                    return
        finally:
            try:
                self._subscribers.remove(q)
            except ValueError:
                pass
