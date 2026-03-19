"""SSE event types for real-time wiki generation progress."""

from __future__ import annotations

from pydantic import BaseModel


class SSEEvent(BaseModel):
    """Generic SSE event wrapper."""

    event: str
    data: dict


class ProgressEvent(BaseModel):
    """Progress update during wiki generation."""

    event: str = "progress"
    phase: str  # "indexing" | "planning" | "generating" | "enhancing"
    progress: float  # 0.0 → 1.0
    message: str
    page_id: str | None = None


class PageCompleteEvent(BaseModel):
    """Fired when a single wiki page finishes generating."""

    event: str = "page_complete"
    page_id: str
    page_title: str


class WikiCompleteEvent(BaseModel):
    """Fired when the entire wiki generation is done."""

    event: str = "wiki_complete"
    wiki_id: str
    page_count: int
    execution_time: float


class RetryEvent(BaseModel):
    """Fired when an LLM call is being retried."""

    event: str = "retry"
    message: str
    attempt: int
    max_attempts: int
    wait_seconds: float


class FallbackEvent(BaseModel):
    """Fired when switching to a fallback model."""

    event: str = "fallback"
    message: str
    from_model: str
    to_model: str


class ErrorEvent(BaseModel):
    """Fired on error during generation."""

    event: str = "error"
    error: str
    recoverable: bool = False
    error_type: str | None = None  # "context_overflow" | None
    model_limit: int | None = None
    suggested_actions: list[str] | None = None


class TokenUsageEvent(BaseModel):
    """Fired after ranked truncation to report per-page token budget usage."""

    event: str = "token_usage"
    page_id: str
    page_title: str
    docs_retrieved: int
    docs_included: int
    docs_dropped: int
    tokens_before: int
    tokens_after: int
    token_budget: int
    utilization_pct: float
