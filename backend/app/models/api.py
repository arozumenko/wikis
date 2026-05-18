"""Pydantic request/response models for all API endpoints.

Generate-wiki API contract (frontend-facing)
--------------------------------------------
POST /api/v1/wikis accepts:

    {
      "source_type": "git" | "confluence" | "jira",  // default "git"
      "scope": {
        // git:        {"repo_url": str, "branch": str}
        // confluence: {"base_url": str, "space_keys": list[str]}
        // jira:       {"base_url": str, "jql": str}
      },
      "auth": {
        // git:        {"pat": str | null}                (optional)
        // confluence: {"access_token": str,
        //              "refresh_token": str | null,
        //              "client_id": str | null}
        // jira:       {"access_token": str,
        //              "refresh_token": str | null,
        //              "client_id": str | null}
      },
      "wiki_title": "optional string",
      "structure_planner": "agentic" | "graph_clustering"  // optional
    }

Backwards compatibility: the legacy flat fields (``repo_url``, ``branch``,
``access_token``) are still accepted and auto-normalised into the
``source_type`` / ``scope`` / ``auth`` shape by the model validator, so
existing API consumers and the entire test suite continue to work.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, Field, model_validator

# ---------------------------------------------------------------------------
# Generate Wiki
# ---------------------------------------------------------------------------


class GenerateWikiRequest(BaseModel):
    """Request to generate a wiki from a repository or Atlassian source.

    Preferred (new) shape
    ---------------------
    source_type + scope + auth — see module docstring for field shapes.

    Backwards-compatible (legacy) shape
    ------------------------------------
    repo_url + branch + access_token — auto-normalized into the new shape
    by the ``_normalize_legacy`` validator so old callers keep working.
    """

    # --- new multi-source fields -------------------------------------------
    source_type: Literal["git", "confluence", "jira"] = "git"
    scope: dict[str, Any] = Field(default_factory=dict)
    auth: dict[str, Any] = Field(default_factory=dict)

    # --- legacy git-only shim (kept for backwards compat) ------------------
    repo_url: str | None = None
    branch: str | None = None
    provider: str = "github"  # github | gitlab | bitbucket | ado | local
    access_token: str | None = None

    # --- common options -------------------------------------------------------
    wiki_title: str | None = None
    include_research: bool = True
    include_diagrams: bool = True
    force_rebuild_index: bool = True
    visibility: str = "personal"  # "personal" or "shared"
    # LLM overrides (optional, falls back to server config)
    llm_model: str | None = None
    embedding_model: str | None = None
    # Structure planner override — "agentic" maps to "agent", "graph_clustering" to "cluster".
    # The legacy "planner_type" field still accepted to avoid breaking internal callers.
    structure_planner: Literal["agentic", "graph_clustering"] | None = None
    planner_type: Literal["agent", "cluster"] | None = None
    # Test exclusion (optional; only honoured when planner_type == "cluster").
    exclude_tests: bool | None = None

    # ------------------------------------------------------------------
    # Validators
    # ------------------------------------------------------------------

    @model_validator(mode="before")
    @classmethod
    def _normalize_legacy(cls, values: Any) -> Any:
        """Normalize legacy flat-field shape into the new scope/auth shape.

        If ``repo_url`` is present and ``scope`` is absent or empty, build:
            source_type = "git"
            scope = {"repo_url": <repo_url>, "branch": <branch or "main">}
            auth  = {"pat": <access_token>}  (only when access_token set)

        Also maps ``structure_planner`` → ``planner_type`` for the subprocess.
        """
        if not isinstance(values, dict):
            return values

        repo_url = values.get("repo_url")
        scope = values.get("scope")
        if repo_url and not scope:
            branch = values.get("branch") or "main"
            values["source_type"] = "git"
            values["scope"] = {"repo_url": repo_url, "branch": branch}
            if values.get("access_token"):
                values["auth"] = {"pat": values["access_token"]}

        # Map structure_planner → planner_type
        sp = values.get("structure_planner")
        if sp and not values.get("planner_type"):
            if sp == "agentic":
                values["planner_type"] = "agent"
            elif sp == "graph_clustering":
                values["planner_type"] = "cluster"

        return values

    @model_validator(mode="after")
    def _validate_and_backfill(self) -> "GenerateWikiRequest":
        """Back-fill legacy fields from scope so downstream code still works."""
        if self.source_type == "git":
            # Ensure repo_url and branch are set (legacy pipeline reads them directly).
            if not self.repo_url:
                self.repo_url = self.scope.get("repo_url", "")
            if not self.branch:
                self.branch = self.scope.get("branch") or "main"
            if not self.access_token:
                self.access_token = self.auth.get("pat") or None

            # Local path validation (unchanged from original).
            from app.core.local_repo_provider import is_local_path, validate_local_path

            if self.repo_url and is_local_path(self.repo_url):
                if self.provider == "github":
                    self.provider = "local"
                from app.config import get_settings

                settings = get_settings()
                allowed: list[str] = []
                if settings.allowed_local_paths:
                    allowed = [p.strip() for p in settings.allowed_local_paths.split(",") if p.strip()]
                path = self.repo_url.removeprefix("file://")
                validate_local_path(path, allowed or None)
        else:
            # For non-git sources, set stubs so invocation serialisation
            # (which references repo_url / branch) doesn't blow up.
            if not self.repo_url:
                self.repo_url = self.scope.get("base_url", "")
            if not self.branch:
                self.branch = "main"

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
    wiki_id: str | None = None    # None for single-wiki answers
    wiki_title: str | None = None  # Human-readable wiki title for cross-repo answers
    # #120: edge-confidence label propagated from the underlying graph
    # node's strongest incoming relationship. ``EXTRACTED`` (explicit
    # parser observation) / ``INFERRED`` (name-only resolution) /
    # ``AMBIGUOUS`` (multiple candidates). ``None`` when the citation
    # comes from a path that doesn't surface edge confidence — e.g. a
    # pure FTS hit with no graph linkage. Stable enum string;
    # downstream SPA + MCP consumers filter on this.
    confidence: str | None = None


class AskRequest(BaseModel):
    """Request to ask a question about a wiki."""

    wiki_id: str | None = None      # Optional when project_id is provided
    project_id: str | None = None   # Query across all wikis in a project
    question: str
    chat_history: list[ChatMessage] = Field(default_factory=list)
    k: int = 15  # Retrieval doc count
    # #120/#157: minimum edge-confidence floor for graph expansion.
    # None (default) keeps the prior behavior of including all edges
    # regardless of label. ``"EXTRACTED"`` keeps only direct parser
    # observations; ``"INFERRED"`` includes name-only resolution
    # edges too; ``"AMBIGUOUS"`` would also include reserved
    # multi-target candidates (no callsites today).
    #
    # Validator rejects typos so MCP clients passing ``"extrcted"``
    # or similar see a 422 instead of silently getting unfiltered
    # results (the rank lookup would return 0 for unknown keys and
    # admit every edge — code-review I1).
    min_confidence: str | None = None

    @model_validator(mode="after")
    def _normalise_min_confidence(self) -> "AskRequest":
        if self.min_confidence is None:
            return self
        # Lazy import to keep the model module dependency-light.
        from app.core.confidence_filter import ACCEPTED_CONFIDENCE_LEVELS

        normalized = self.min_confidence.upper()
        if normalized not in ACCEPTED_CONFIDENCE_LEVELS:
            raise ValueError(
                f"min_confidence must be one of "
                f"{sorted(ACCEPTED_CONFIDENCE_LEVELS)!r} (case-insensitive); "
                f"got {self.min_confidence!r}"
            )
        # Always store uppercase so downstream comparisons are stable.
        self.min_confidence = normalized
        return self

    @model_validator(mode="after")
    def _require_target(self) -> "AskRequest":
        if not self.wiki_id and not self.project_id:
            raise ValueError("Either wiki_id or project_id is required")
        return self


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

    wiki_id: str | None = None      # Optional when project_id is provided
    project_id: str | None = None   # Research across all wikis in a project
    question: str
    research_type: str = "general"
    chat_history: list[ChatMessage] = Field(default_factory=list)

    @model_validator(mode="after")
    def _require_target(self) -> ResearchRequest:
        if not self.wiki_id and not self.project_id:
            raise ValueError("Either wiki_id or project_id is required")
        return self


class ProjectCodeMapRequest(BaseModel):
    """Request to build a code map for a project (all wikis)."""

    project_id: str
    question: str
    research_type: str = "codemap"
    chat_history: list[ChatMessage] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# Code Map — hierarchical call-tree view (like DeepWiki's code-map panel)
# ---------------------------------------------------------------------------


class CodeMapSymbol(BaseModel):
    """A single symbol (function / class / variable) shown as a leaf in the tree."""

    id: str
    name: str
    symbol_type: str = ""   # function, class, method, variable, …
    file_path: str = ""     # relative path
    line_start: int | None = None
    description: str = ""   # LLM-generated 1-sentence explanation of this symbol's role
    relationships: list[str] = Field(default_factory=list)  # e.g. ["calls: validate_token"]


class CodeMapSection(BaseModel):
    """A top-level section in the call-tree (usually one source file)."""

    id: str
    title: str          # short label, e.g. file basename
    file_path: str = "" # relative path
    description: str = ""
    symbols: list[CodeMapSymbol] = Field(default_factory=list)


class CallStackStep(BaseModel):
    """One step in a call stack chain."""

    symbol: str        # e.g. "MCPAuthMiddleware.__call__"
    file_path: str = ""
    description: str = ""  # what happens at this step

class CallStack(BaseModel):
    """A named call-flow chain, e.g. 'Frontend Auth' or 'MCP Auth'."""

    title: str         # e.g. "Frontend Session Auth"
    steps: list[CallStackStep] = Field(default_factory=list)

class CodeMapData(BaseModel):
    """Hierarchical call-tree for the code-map right panel."""

    summary: str = ""   # LLM-generated 1-2 sentence overview
    call_stacks: list[CallStack] = Field(default_factory=list)  # rendered call chains
    sections: list[CodeMapSection] = Field(default_factory=list)


class ResearchResponse(BaseModel):
    """Response from deep research."""

    answer: str
    sources: list[SourceReference] = Field(default_factory=list)
    research_steps: list[str] = Field(default_factory=list)
    code_map: CodeMapData | None = None


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
    description: str | None = None


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


class UpdateWikiDescriptionRequest(BaseModel):
    """Request to update the description of a wiki."""

    description: str | None = None


class RefreshWikiRequest(BaseModel):
    """Request body for wiki refresh — allows passing an access token for private repos."""

    access_token: str | None = None


# ---------------------------------------------------------------------------
# Projects
# ---------------------------------------------------------------------------


class ProjectCreateRequest(BaseModel):
    """Request to create a new project."""

    name: str = Field(..., min_length=1, max_length=100)
    description: str | None = None
    visibility: Literal["personal", "shared"] = "personal"


class ProjectUpdateRequest(BaseModel):
    """Request to update an existing project."""

    name: str | None = Field(None, min_length=1, max_length=100)
    description: str | None = None
    visibility: Literal["personal", "shared"] | None = None


class ProjectResponse(BaseModel):
    """Response for a single project."""

    id: str
    name: str
    description: str | None
    visibility: str
    owner_id: str
    is_owner: bool = False
    created_at: datetime
    wiki_count: int = 0


class ProjectListResponse(BaseModel):
    """Response listing all projects."""

    projects: list[ProjectResponse]


class ProjectAddWikiRequest(BaseModel):
    """Request to add a wiki to a project."""

    wiki_id: str


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


# ---------------------------------------------------------------------------
# Change detection (#116 PR 2 — incremental regen diagnostics)
# ---------------------------------------------------------------------------


class ParsedNodeInput(BaseModel):
    """One node from a fresh parse, supplied to the diff endpoint.

    Only the three fields needed for change classification are required;
    callers can pass the same dicts they would otherwise hand to
    ``upsert_nodes_batch`` (extra keys are ignored).
    """

    node_id: str
    content_hash: str | None = None
    rel_path: str = ""


class DiffWikiRequest(BaseModel):
    """Body for ``POST /wikis/{wiki_id}/diff`` — supply the freshly parsed
    nodes; the server diffs them against the indexed snapshot.

    Capped at 50 000 nodes per request to bound the cost of the per-node
    reverse-index lookup. The largest repos in our corpus parse to ~30k
    architectural nodes; 50k gives ~1.5x headroom before PR 3 swaps in the
    batched ``get_pages_citing_nodes`` query.
    """

    parsed_nodes: list[ParsedNodeInput] = Field(default_factory=list, max_length=50_000)


class NodeChangeResponse(BaseModel):
    """One change in the response. Field population depends on ``kind``."""

    kind: Literal["added", "modified", "deleted", "moved"]
    node_id: str
    old_hash: str | None = None
    new_hash: str | None = None
    old_path: str | None = None
    new_path: str | None = None


class ChangeSetResponse(BaseModel):
    """Aggregate change set — one entry per kind."""

    added: list[NodeChangeResponse] = Field(default_factory=list)
    modified: list[NodeChangeResponse] = Field(default_factory=list)
    moved: list[NodeChangeResponse] = Field(default_factory=list)
    deleted: list[NodeChangeResponse] = Field(default_factory=list)
    total: int


class AffectedPageResponse(BaseModel):
    """One wiki page touched by the change set."""

    page_id: str
    title: str | None = None
    anchor_slug: str | None = None
    changes: list[NodeChangeResponse] = Field(default_factory=list)


class DiffWikiResponse(BaseModel):
    """Result of the diff endpoint — no writes occur."""

    wiki_id: str
    change_set: ChangeSetResponse
    affected_pages: list[AffectedPageResponse] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# Incremental refresh (#116 PR 5 — three-regime dispatch)
# ---------------------------------------------------------------------------


class IncrementalRefreshRequest(BaseModel):
    """POST ``/wikis/{id}/incremental-refresh`` body.

    Caller supplies pre-parsed nodes (same shape as ``/diff``). Capped at
    50_000 nodes per request to bound the orchestrator's snapshot work.
    """

    parsed_nodes: list[ParsedNodeInput] = Field(
        default_factory=list, max_length=50_000,
    )


class IncrementalRefreshResponse(BaseModel):
    """Initial response — actual progress flows over SSE.

    The orchestrator runs asynchronously after this returns; subscribe
    to ``/invocations/{invocation_id}/stream`` for ``page_unchanged``,
    ``page_patched``, ``page_edited``, ``page_regenerated``, and the
    final ``incremental_summary`` event with the run stats.
    """

    wiki_id: str
    invocation_id: str
    status: str  # "running" — terminal state arrives via SSE
    page_count: int  # how many pages the run will potentially touch
    message: str | None = None
