"""Pydantic request/response models for all API endpoints."""

from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, Field, model_validator

# ---------------------------------------------------------------------------
# Generate Wiki
# ---------------------------------------------------------------------------


class GenerateWikiRequest(BaseModel):
    """Request to generate a wiki from a repository."""

    repo_url: str
    branch: str = "main"
    provider: str = "github"  # github | gitlab | bitbucket | ado | local
    access_token: str | None = None
    wiki_title: str | None = None
    include_research: bool = True
    include_diagrams: bool = True
    force_rebuild_index: bool = True
    visibility: str = "personal"  # "personal" or "shared"
    # LLM overrides (optional, falls back to server config)
    llm_model: str | None = None
    embedding_model: str | None = None

    @model_validator(mode="after")
    def _validate_local_path(self) -> GenerateWikiRequest:
        from app.config import get_settings
        from app.core.local_repo_provider import is_local_path, validate_local_path

        if not is_local_path(self.repo_url):
            return self

        # Auto-tag provider
        if self.provider == "github":
            self.provider = "local"

        settings = get_settings()
        allowed: list[str] = []
        if settings.allowed_local_paths:
            allowed = [p.strip() for p in settings.allowed_local_paths.split(",") if p.strip()]

        # Raises ValueError → Pydantic turns into 422
        path = self.repo_url.removeprefix("file://")
        validate_local_path(path, allowed or None)
        return self


class GenerateWikiResponse(BaseModel):
    """Response after initiating wiki generation."""

    wiki_id: str
    invocation_id: str
    status: str  # "generating" | "complete" | "failed"
    message: str


# ---------------------------------------------------------------------------
# Ask (Q&A)
# ---------------------------------------------------------------------------


class ChatMessage(BaseModel):
    """A single chat message."""

    role: str  # "user" | "assistant"
    content: str


class SourceReference(BaseModel):
    """A reference to a source code location."""

    file_path: str
    line_start: int | None = None
    line_end: int | None = None
    snippet: str | None = None
    symbol: str | None = None
    symbol_type: str | None = None
    relevance_score: float | None = None


class AskRequest(BaseModel):
    """Request to ask a question about a wiki."""

    wiki_id: str
    question: str
    chat_history: list[ChatMessage] = Field(default_factory=list)
    k: int = 15  # Retrieval doc count


class AskResponse(BaseModel):
    """Response to an ask question."""

    answer: str
    sources: list[SourceReference] = Field(default_factory=list)
    tool_steps: int = 0  # Number of tool calls the agent made (0 = no tools used)
    qa_id: str = ""  # Unique Q&A interaction ID (empty for backward compat)


# ---------------------------------------------------------------------------
# Deep Research
# ---------------------------------------------------------------------------


class ResearchRequest(BaseModel):
    """Request to perform deep research on a wiki."""

    wiki_id: str
    question: str
    research_type: str = "general"
    chat_history: list[ChatMessage] = Field(default_factory=list)


class ResearchResponse(BaseModel):
    """Response from deep research."""

    answer: str
    sources: list[SourceReference] = Field(default_factory=list)
    research_steps: list[str] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# List / Delete Wikis
# ---------------------------------------------------------------------------


class WikiSummary(BaseModel):
    """Summary of a generated wiki."""

    wiki_id: str
    repo_url: str
    branch: str
    title: str
    created_at: datetime
    page_count: int
    status: str = "complete"  # complete | generating | failed | partial | cancelled
    progress: float = 1.0
    invocation_id: str | None = None
    error: str | None = None
    indexed_at: datetime | None = None
    commit_hash: str | None = None
    owner_id: str | None = None
    visibility: str = "personal"  # "personal" or "shared"
    is_owner: bool = False
    requires_token: bool = False


class WikiListResponse(BaseModel):
    """Response listing all wikis."""

    wikis: list[WikiSummary] = Field(default_factory=list)


class DeleteWikiResponse(BaseModel):
    """Response after deleting a wiki."""

    deleted: bool
    wiki_id: str
    message: str | None = None


class UpdateWikiVisibilityRequest(BaseModel):
    """Request to update the visibility of a wiki."""

    visibility: str  # "personal" or "shared"


class RefreshWikiRequest(BaseModel):
    """Request body for wiki refresh — allows passing an access token for private repos."""

    access_token: str | None = None


# ---------------------------------------------------------------------------
# Health
# ---------------------------------------------------------------------------


class HealthResponse(BaseModel):
    """Health check response."""

    status: str  # "ok" | "degraded"
    version: str
    services: dict[str, str] = Field(default_factory=dict)


# ---------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------


class ErrorResponse(BaseModel):
    """Standard error response."""

    error: str
    detail: str | None = None
    code: str  # machine-readable error code
