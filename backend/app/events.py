"""MCP-standard event builders for SSE streaming.

All events follow JSON-RPC 2.0 notification format, compatible with MCP's
notifications/progress, notifications/message, and tasks/status patterns.

Usage:
    from app.events import progress, task_status, message

    await invocation.emit(progress("tok-123", 4, 12, "Generating page: Auth"))
    await invocation.emit(task_status("inv-abc", "working", "Indexing repository..."))
    await invocation.emit(message("info", "Retrying with fallback model"))
    await invocation.emit(task_result("inv-abc", wiki_id="x", page_count=12, execution_time=42.5))
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import Any


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


# ---------------------------------------------------------------------------
# JSON-RPC 2.0 notification envelope
# ---------------------------------------------------------------------------


class MCPEvent:
    """A JSON-RPC 2.0 notification that can be serialized for SSE or MCP."""

    __slots__ = ("method", "params", "_event_alias")

    def __init__(self, method: str, params: dict[str, Any], *, event_alias: str | None = None) -> None:
        self.method = method
        self.params = params
        # SSE event: line uses a short alias (e.g. "progress") instead of the
        # full method name for brevity.  Falls back to last segment of method.
        self._event_alias = event_alias or method.rsplit("/", 1)[-1]

    # -- Pydantic-like interface so Invocation.emit() works unchanged ---------

    @property
    def event(self) -> str:
        """SSE event type identifier (used by Invocation.events() for terminal detection)."""
        return self._event_alias

    def model_dump_json(self) -> str:
        """Serialize to JSON-RPC 2.0 notification string."""
        return json.dumps(
            {"jsonrpc": "2.0", "method": self.method, "params": self.params},
            default=str,
        )


# ---------------------------------------------------------------------------
# notifications/progress
# ---------------------------------------------------------------------------


def progress(
    progress_token: str,
    progress: int,
    total: int,
    message: str,
    *,
    phase: str | None = None,
    page_id: str | None = None,
) -> MCPEvent:
    """MCP progress notification — monotonically increasing progress counter.

    Maps to: notifications/progress
    """
    params: dict[str, Any] = {
        "progressToken": progress_token,
        "progress": progress,
        "total": total,
        "message": message,
    }
    if phase:
        params["_phase"] = phase
    if page_id:
        params["_pageId"] = page_id
    return MCPEvent("notifications/progress", params, event_alias="progress")


# ---------------------------------------------------------------------------
# notifications/tasks/status
# ---------------------------------------------------------------------------

# Valid status values per MCP spec
TASK_WORKING = "working"
TASK_COMPLETED = "completed"
TASK_FAILED = "failed"
TASK_CANCELLED = "cancelled"

_TERMINAL_STATUSES = frozenset({TASK_COMPLETED, TASK_FAILED, TASK_CANCELLED})


def task_status(
    task_id: str,
    status: str,
    status_message: str = "",
    *,
    # Extra fields for result payloads (flattened into params for SSE convenience)
    wiki_id: str | None = None,
    page_count: int | None = None,
    execution_time: float | None = None,
    error: str | None = None,
    recoverable: bool = False,
    error_type: str | None = None,
    model_limit: int | None = None,
    suggested_actions: list[str] | None = None,
    answer: str | None = None,
    sources: list[dict[str, Any]] | None = None,
    report: str | None = None,
    steps: int | None = None,
    research_steps: list[str] | None = None,
) -> MCPEvent:
    """MCP task status notification.

    For terminal statuses (completed/failed/cancelled), extra payload fields
    carry the result or error details.

    Maps to: notifications/tasks/status
    """
    params: dict[str, Any] = {
        "taskId": task_id,
        "status": status,
        "statusMessage": status_message,
        "timestamp": _now_iso(),
    }
    # Completion payload
    if wiki_id is not None:
        params["wikiId"] = wiki_id
    if page_count is not None:
        params["pageCount"] = page_count
    if execution_time is not None:
        params["executionTime"] = execution_time
    # Error payload
    if error is not None:
        params["error"] = error
    if recoverable:
        params["recoverable"] = True
    if error_type:
        params["errorType"] = error_type
    if model_limit is not None:
        params["modelLimit"] = model_limit
    if suggested_actions:
        params["suggestedActions"] = suggested_actions
    # Ask/Research completion payload
    if answer is not None:
        params["answer"] = answer
    if sources is not None:
        params["sources"] = sources
    if report is not None:
        params["report"] = report
    if steps is not None:
        params["steps"] = steps
    if research_steps is not None:
        params["researchSteps"] = research_steps

    # Determine SSE event alias based on status for terminal detection
    alias = "task_status"
    if status == TASK_COMPLETED:
        alias = "task_complete"
    elif status == TASK_FAILED:
        alias = "task_failed"
    elif status == TASK_CANCELLED:
        alias = "task_cancelled"

    return MCPEvent("notifications/tasks/status", params, event_alias=alias)


# ---------------------------------------------------------------------------
# notifications/message  (logging / informational)
# ---------------------------------------------------------------------------


def message(
    level: str,
    data: str,
    *,
    logger: str | None = None,
    tool: str | None = None,
    attempt: int | None = None,
    max_attempts: int | None = None,
    wait_seconds: float | None = None,
    from_model: str | None = None,
    to_model: str | None = None,
) -> MCPEvent:
    """MCP log/message notification for informational events.

    Maps to: notifications/message
    """
    params: dict[str, Any] = {
        "level": level,
        "data": data,
        "timestamp": _now_iso(),
    }
    if logger:
        params["logger"] = logger
    if tool:
        params["tool"] = tool
    if attempt is not None:
        params["attempt"] = attempt
    if max_attempts is not None:
        params["maxAttempts"] = max_attempts
    if wait_seconds is not None:
        params["waitSeconds"] = wait_seconds
    if from_model:
        params["fromModel"] = from_model
    if to_model:
        params["toModel"] = to_model
    return MCPEvent("notifications/message", params, event_alias="message")


# ---------------------------------------------------------------------------
# Thinking step (tool call / tool result) — custom extension
# ---------------------------------------------------------------------------


def thinking_step(
    progress_token: str,
    step: int,
    step_type: str,  # "tool_call" or "tool_result"
    tool: str,
    *,
    tool_call_id: str | None = None,
    call_id: str | None = None,
    input: str | None = None,
    output: str | None = None,
    output_preview: str | None = None,
    output_length: int | None = None,
) -> MCPEvent:
    """Agent thinking step — tool invocation or result.

    This is a custom extension on top of notifications/progress that carries
    structured tool call metadata. Frontend uses this for the ToolCallPanel.
    """
    params: dict[str, Any] = {
        "progressToken": progress_token,
        "step": step,
        "stepType": step_type,
        "tool": tool,
        "timestamp": _now_iso(),
    }
    if tool_call_id:
        params["toolCallId"] = tool_call_id
    if call_id:
        params["callId"] = call_id
    if input is not None:
        params["input"] = input
    if output is not None:
        params["output"] = output
    if output_preview is not None:
        params["outputPreview"] = output_preview
    if output_length is not None:
        params["outputLength"] = output_length
    return MCPEvent("notifications/progress/thinking", params, event_alias="thinking_step")


# ---------------------------------------------------------------------------
# Answer chunk — streaming text content
# ---------------------------------------------------------------------------


def answer_chunk(
    progress_token: str,
    chunk: str,
) -> MCPEvent:
    """Incremental answer text for streaming display."""
    return MCPEvent(
        "notifications/progress/content",
        {
            "progressToken": progress_token,
            "chunk": chunk,
            "timestamp": _now_iso(),
        },
        event_alias="answer_chunk",
    )


# ---------------------------------------------------------------------------
# Legacy compatibility helpers
# ---------------------------------------------------------------------------


def page_complete(progress_token: str, page_id: str, page_title: str) -> MCPEvent:
    """Convenience wrapper — page completion as a progress notification."""
    return MCPEvent(
        "notifications/progress",
        {
            "progressToken": progress_token,
            "message": f"Page complete: {page_title}",
            "_pageId": page_id,
            "_pageTitle": page_title,
            "timestamp": _now_iso(),
        },
        event_alias="page_complete",
    )
