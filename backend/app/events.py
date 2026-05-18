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
from collections.abc import Mapping
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
    output_full: str | None = None,
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
    if output_full is not None:
        params["outputFull"] = output_full
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


# ---------------------------------------------------------------------------
# #116 PR 5 — incremental regen events (per-page regime + run summary)
# ---------------------------------------------------------------------------
#
# These are emitted by :class:`IncrementalRegenService` via its
# ``progress_callback``. Each per-page event carries the page_id + title
# so the SPA can render a live list; the summary event arrives once at the
# end with the aggregate :class:`IncrementalRegenStats` payload.
#
# Event aliases match the named regimes from #116 issue:
# ``page_unchanged`` / ``page_patched`` / ``page_edited`` /
# ``page_regenerated`` / ``page_deleted`` / ``incremental_summary``.


def _incremental_page_event(
    alias: str,
    progress_token: str,
    page_id: str,
    page_title: str,
    **extra: Any,
) -> MCPEvent:
    payload = {
        "progressToken": progress_token,
        "_pageId": page_id,
        "_pageTitle": page_title,
        "timestamp": _now_iso(),
        **extra,
    }
    return MCPEvent(
        "notifications/progress",
        payload,
        event_alias=alias,
    )


def page_unchanged(progress_token: str, page_id: str, page_title: str) -> MCPEvent:
    """Page was untouched by the change set — no LLM call."""
    return _incremental_page_event(
        "page_unchanged", progress_token, page_id, page_title,
    )


def page_patched(
    progress_token: str,
    page_id: str,
    page_title: str,
    citation_count: int = 0,
) -> MCPEvent:
    """Trivial-regime patch — file paths rewritten via regex, no LLM."""
    return _incremental_page_event(
        "page_patched", progress_token, page_id, page_title,
        citationCount=citation_count,
    )


def page_edited(
    progress_token: str,
    page_id: str,
    page_title: str,
    diff_ratio: float = 0.0,
) -> MCPEvent:
    """Edit-regime surgical LLM patch passed the quality gate."""
    return _incremental_page_event(
        "page_edited", progress_token, page_id, page_title,
        diffRatio=diff_ratio,
    )


def page_regenerated(
    progress_token: str,
    page_id: str,
    page_title: str,
    demoted_from_edit: bool = False,
) -> MCPEvent:
    """Structural regen — full single-page LLM regenerate.

    ``demoted_from_edit`` is True when the page started in the edit
    regime but the quality gate rejected the surgical output, forcing
    a structural fallback. PR 5 telemetry distinguishes the two cases.
    """
    return _incremental_page_event(
        "page_regenerated", progress_token, page_id, page_title,
        demotedFromEdit=demoted_from_edit,
    )


def page_deleted(progress_token: str, page_id: str, page_title: str) -> MCPEvent:
    """A page's primary symbol vanished — its row was removed.

    Reserved for the cluster-vanished case (entire module deleted).
    PR 3's dispatcher routes the page through structural regen; a
    dedicated event fires only when the page is actually dropped, not
    just regenerated.

    NOTE: builder defined for completeness but the orchestrator does
    not currently emit it — PR 3's classifier never produces a
    "delete this page" regime. Tracked in the #116 follow-up issues
    for when cluster-vanished detection lands.
    """
    return _incremental_page_event(
        "page_deleted", progress_token, page_id, page_title,
    )


def incremental_summary(
    progress_token: str,
    stats: Mapping[str, Any],
) -> MCPEvent:
    """Run summary emitted once at the end of an incremental refresh.

    ``stats`` is :meth:`IncrementalRegenStats.as_dict()` — all regime
    counts + ``avg_diff_ratio`` + ``total_pages``. The SPA renders this
    as a banner like "12 unchanged · 3 patched · 1 regenerated".
    """
    return MCPEvent(
        "notifications/progress",
        {
            "progressToken": progress_token,
            "stats": dict(stats),
            "timestamp": _now_iso(),
        },
        event_alias="incremental_summary",
    )


# ---------------------------------------------------------------------------
# Phase 9 — Project-level lifecycle events
# ---------------------------------------------------------------------------

PHASE_PROJECT_RELATEDNESS = "project_relatedness"
PHASE_CROSS_REPO_LINKER = "cross_repo_linker"
PHASE_PROJECT_CLUSTERING = "project_clustering"


def project_relatedness_progress(
    progress_token: str,
    progress: int,
    total: int,
    message: str,
    *,
    project_id: str | None = None,
) -> MCPEvent:
    """Project-pair relatedness scoring progress.

    Emitted by the Phase 8 cross-repo pipeline once it lands. Wraps a
    standard ``notifications/progress`` so existing SSE clients keep
    working unchanged; consumers that care about the lifecycle stage
    can branch on the ``_phase`` parameter.
    """
    params: dict[str, Any] = {
        "progressToken": progress_token,
        "progress": progress,
        "total": total,
        "message": message,
        "_phase": PHASE_PROJECT_RELATEDNESS,
    }
    if project_id:
        params["_projectId"] = project_id
    return MCPEvent(
        "notifications/progress",
        params,
        event_alias="project_relatedness_progress",
    )


def cross_repo_linker_progress(
    progress_token: str,
    progress: int,
    total: int,
    message: str,
    *,
    project_id: str | None = None,
    pair: tuple[str, str] | None = None,
) -> MCPEvent:
    """Cross-repo linker progress (per wiki pair)."""
    params: dict[str, Any] = {
        "progressToken": progress_token,
        "progress": progress,
        "total": total,
        "message": message,
        "_phase": PHASE_CROSS_REPO_LINKER,
    }
    if project_id:
        params["_projectId"] = project_id
    if pair and len(pair) == 2:
        params["_pair"] = list(pair)
    return MCPEvent(
        "notifications/progress",
        params,
        event_alias="cross_repo_linker_progress",
    )


def project_clustering_progress(
    progress_token: str,
    progress: int,
    total: int,
    message: str,
    *,
    project_id: str | None = None,
    community_count: int | None = None,
) -> MCPEvent:
    """Project-level Leiden clustering progress."""
    params: dict[str, Any] = {
        "progressToken": progress_token,
        "progress": progress,
        "total": total,
        "message": message,
        "_phase": PHASE_PROJECT_CLUSTERING,
    }
    if project_id:
        params["_projectId"] = project_id
    if community_count is not None:
        params["_communityCount"] = int(community_count)
    return MCPEvent(
        "notifications/progress",
        params,
        event_alias="project_clustering_progress",
    )
