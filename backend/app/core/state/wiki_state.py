"""
Optimized State Definition for Wiki Generation

Following LangGraph best practices with clean state management.
"""

from __future__ import annotations

import operator
from datetime import datetime
from enum import StrEnum
from typing import Annotated, Any

from pydantic import AliasChoices, BaseModel, ConfigDict, Field, field_validator


class DictAccessMixin:
    """Mixin to add dict-style access (.get, [], in) to Pydantic models.

    Required for compatibility with core agents that use state.get('key')
    and state['key'] patterns (39 sites in wiki_graph_optimized.py).
    """

    def get(self, key: str, default: Any = None) -> Any:
        return getattr(self, key, default)

    def __getitem__(self, key: str) -> Any:
        try:
            return getattr(self, key)
        except AttributeError as e:
            raise KeyError(key) from e

    def __setitem__(self, key: str, value: Any) -> None:
        setattr(self, key, value)

    def __contains__(self, key: str) -> bool:
        return hasattr(self, key)


class WikiStyle(StrEnum):
    """Wiki generation styles"""

    COMPREHENSIVE = "comprehensive"
    TECHNICAL = "technical"
    MINIMAL = "minimal"
    TUTORIAL = "tutorial"


class TargetAudience(StrEnum):
    """Target audiences for wiki content"""

    DEVELOPERS = "developers"
    BEGINNERS = "beginners"
    MIXED = "mixed"
    TECHNICAL = "technical"
    BUSINESS = "business"


# Remove WikiConfiguration class entirely as it's now handled by direct parameters


class PageSpec(BaseModel):
    """Specification for a wiki page with graph-based retrieval support"""

    page_name: str
    page_order: int
    description: str
    content_focus: str
    rationale: str  # Why this page is needed and what problems it solves
    target_symbols: list[str] = Field(default_factory=list)  # Exact class/function names for graph-based retrieval
    target_docs: list[str] = Field(
        default_factory=list
    )  # Documentation files for page context (e.g., README.md, docs/auth.md)
    target_folders: list[str] = Field(default_factory=list)  # Folders in repo this page references
    key_files: list[str] = Field(default_factory=list)  # Specific files this page will reference
    retrieval_query: str = Field(default="")  # Fallback query for vector store retrieval (documentation files)
    metadata: dict[str, Any] = Field(default_factory=dict)  # Planner-specific metadata (cluster ids, node lists)


class SectionSpec(BaseModel):
    """Specification for a wiki section"""

    section_name: str
    section_order: int
    description: str
    rationale: str
    pages: list[PageSpec]


class WikiStructureSpec(BaseModel):
    """Complete wiki structure specification"""

    wiki_title: str
    overview: str
    sections: list[SectionSpec]
    total_pages: int


class QualityAssessment(BaseModel):
    """Content quality assessment results"""

    scores: dict[str, float]
    overall_score: float
    strengths: list[str]
    weaknesses: list[str]
    improvement_suggestions: list[str]
    passes_quality_gate: bool
    confidence_level: float


class ValidationResult(BaseModel):
    """Content validation results"""

    validation_results: dict[str, bool]
    diagram_analysis: dict[str, Any]
    issues_found: list[str]
    severity_level: str
    passes_validation: bool
    recommendations: list[str]


class WikiPage(BaseModel):
    """Generated wiki page data"""

    page_id: str
    title: str
    content: str
    status: str  # generating, completed, failed, enhanced
    quality_assessment: QualityAssessment | None = None
    validation_result: ValidationResult | None = None
    retry_count: int = 0
    generated_at: datetime = Field(default_factory=datetime.now)


class WikiPageBatch(BaseModel):
    """Batch of wiki pages for parallel processing"""

    batch_id: str
    page_ids: list[str]
    batch_status: str  # pending, processing, completed, failed
    completed_at: datetime | None = None


# Main State Types


class WikiState(DictAccessMixin, BaseModel):
    """Main state for wiki generation workflow - Following LangGraph best practices"""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    # Repository analysis (from indexer)
    repository_analysis: Any | None = None  # LLM analysis text or RepositoryAnalysis object
    repository_context: str | None = None  # Formatted context for LLM
    repository_tree: str | None = None  # Full repository tree structure
    readme_content: str | None = None  # Complete README content
    repository_tree_stats: dict[str, int] | None = None  # Parsed stats footer
    conceptual_summary_raw: str | None = None  # Raw LLM JSON/text prior to parsing
    conceptual_summary: ConceptualRepositorySummary | None = None  # Parsed structured conceptual model
    page_budget: int | None = None  # Adaptive page budget (min 5, max 55)
    page_budget_justification: str | None = None  # Reasoning from LLM for budget
    budget_compression_recommendation: str | None = None  # If compression / consolidation recommended

    # Generation options (from API request / config defaults)
    planner_type: str = "agent"  # agent | cluster
    exclude_tests: bool = False  # soft-skip test nodes during clustering
    unified_db: Any | None = None  # UnifiedWikiDB instance (set by agent)

    # Wiki structure planning (LLM-driven)
    wiki_structure_spec: WikiStructureSpec | None = None
    structure_planning_complete: bool | None = False

    # Phase 1: Content Generation - Natural accumulation with operator.add
    wiki_pages: Annotated[list[WikiPage], operator.add] = Field(default_factory=list)

    # Phase 2: Quality Assessment - Natural accumulation with operator.add
    quality_assessments: Annotated[list[dict[str, Any]], operator.add] = Field(default_factory=list)

    # Phase 3: Enhancement (only for failed pages) - Natural accumulation with operator.add
    enhanced_pages: Annotated[list[WikiPage], operator.add] = Field(default_factory=list)

    # Retry tracking
    retry_counts: dict[str, int] = Field(default_factory=dict)

    # Progress tracking
    start_time: float | None = 0.0
    current_phase: str | None = ""

    # Final results
    export_complete: bool | None = False
    artifacts: list[dict] = Field(default_factory=list)
    generation_summary: dict[str, Any] | None = None

    # Monitoring and debugging - Lists that accumulate
    messages: Annotated[list[str], operator.add] = Field(default_factory=list)
    errors: Annotated[list[str], operator.add] = Field(default_factory=list)


# Parallel Processing State Types - Following LangGraph Send patterns


class PageGenerationState(DictAccessMixin, BaseModel):
    """State for individual page generation tasks"""

    model_config = ConfigDict(arbitrary_types_allowed=True)
    page_id: str = ""
    page_spec: PageSpec | None = None
    batch_id: str = ""
    repository_context: str = ""


class QualityAssessmentState(DictAccessMixin, BaseModel):
    """State for quality assessment tasks"""

    page_id: str = ""
    page_content: str = ""
    page_title: str = ""


class EnhancementState(DictAccessMixin, BaseModel):
    """State for content enhancement tasks"""

    model_config = ConfigDict(arbitrary_types_allowed=True)
    page_id: str = ""
    page_content: str = ""
    page_title: str = ""
    quality_assessment: QualityAssessment | None = None
    repository_context: str = ""


# Helper Types for Complex Operations


class RepositoryAnalysis(BaseModel):
    """Structured repository analysis results"""

    total_files: int
    programming_languages: list[str]
    primary_language: str
    has_readme: bool
    readme_content: str
    total_symbols: int
    file_types: dict[str, int]
    project_complexity: str  # simple, moderate, complex
    key_directories: list[str]
    main_modules: list[str]
    documentation_files: list[str]
    configuration_files: list[str]
    test_files: list[str]


class ContentContext(BaseModel):
    """Context for content generation"""

    repository_summary: str
    relevant_code_snippets: list[dict[str, str]]
    related_documentation: list[dict[str, str]]
    file_references: list[str]
    technical_concepts: list[str]


class GenerationMetrics(BaseModel):
    """Metrics for generation performance"""

    total_pages_generated: int
    total_diagrams_created: int
    average_quality_score: float
    validation_pass_rate: float
    total_generation_time: float
    pages_per_minute: float
    retry_rate: float
    enhancement_applied: int


class ExportSummary(BaseModel):
    """Summary of exported wiki"""

    wiki_title: str
    total_sections: int
    total_pages: int
    total_diagrams: int
    average_quality_score: float
    export_formats: list[str]
    artifact_files: list[str]
    generation_timestamp: str
    quality_metrics: dict[str, float]


# ============================================================================
# SECTION-BASED PLANNING MODELS (for agentic generator)
# ============================================================================


class SymbolSummary(BaseModel):
    """Lightweight symbol representation for planning phase.

    Contains just enough information for the LLM to decide which symbols
    to include in each section, without loading full code.
    Built from document metadata - no regex parsing needed.
    """

    name: str = Field(description="Symbol name, e.g., 'UserService'")
    qualified_name: str = Field(description="Full path, e.g., 'src.services.user::UserService'")
    symbol_type: str = Field(description="Type: 'class', 'function', 'interface', 'method'")
    file_path: str = Field(description="Source file path, e.g., 'src/services/user.py'")
    brief: str = Field(default="", description="First line of docstring (from metadata)")
    relationships: list[str] = Field(
        default_factory=list,
        description="Relationship strings from expansion: 'extends BaseService', 'uses UserRepository'",
    )
    estimated_tokens: int = Field(default=0, description="Estimated token count of full code")


class OverviewSection(BaseModel):
    """A single overview section for high-level content.

    LLM decides title and content during planning phase when full context is available.
    Examples: "Architecture Overview", "Key Concepts", "Design Patterns", "Data Flow"
    """

    title: str = Field(description="Section title - LLM decides based on topic needs")
    content: str = Field(description="Full markdown content with optional <code_context path> citations")


class DetailSectionSpec(BaseModel):
    """Specification for a detail section to be generated with focused context.

    These sections are generated AFTER planning, with only the relevant symbols loaded.
    """

    section_id: str = Field(description="Unique identifier for this section, e.g., 'auth_implementation'")
    title: str = Field(description="Section title, e.g., 'Authentication Implementation'")
    description: str = Field(description="What this section should cover in detail")

    # Optional intent for section-specific retrieval/expansion policy
    intent: str | None = Field(
        default=None, description="Optional intent tag (e.g., 'graph_construction', 'api', 'config', 'runtime')"
    )

    # Symbols needed for this section
    primary_symbols: list[str] = Field(
        default_factory=list, description="Symbol names whose FULL code is needed (5-10 per section)"
    )
    supporting_symbols: list[str] = Field(
        default_factory=list, description="Symbol names where signatures/docstrings are sufficient"
    )
    retrieval_queries: list[str] = Field(
        default_factory=list, description="Search queries used to retrieve symbols for this section"
    )
    symbol_paths: list[str] = Field(default_factory=list, description="File paths for the symbols (for code citations)")

    model_config = ConfigDict(extra="ignore")

    @field_validator("primary_symbols", "supporting_symbols", mode="before")
    @classmethod
    def _coerce_symbol_lists(cls, v):
        if v is None:
            return []
        if isinstance(v, list):
            return v
        if isinstance(v, str):
            return [v]
        if isinstance(v, dict):
            # Some models return {symbol: path} or {idx: symbol}; pick keys as names.
            return [str(k) for k in v.keys()]
        return v

    @field_validator("symbol_paths", mode="before")
    @classmethod
    def _coerce_symbol_paths(cls, v):
        if v is None:
            return []
        if isinstance(v, list):
            return [str(x) for x in v if x is not None]
        if isinstance(v, str):
            return [v]
        if isinstance(v, dict):
            # Common drift: model returns {SymbolName: "path/to/file.py"}
            return [str(x) for x in v.values() if x is not None]
        return v

    @field_validator("retrieval_queries", mode="before")
    @classmethod
    def _coerce_retrieval_queries(cls, v):
        if v is None:
            return []
        if isinstance(v, list):
            return [str(x) for x in v if x is not None]
        if isinstance(v, str):
            return [v]
        if isinstance(v, dict):
            return [str(x) for x in v.values() if x is not None]
        return v

    # Content guidance
    suggested_elements: list[str] = Field(
        default_factory=list,
        description="Suggested content elements: 'workflow diagram', 'code example', 'sequence diagram'",
    )


class PageStructure(BaseModel):
    """Complete page structure from planning - HYBRID approach.

    Generated during planning phase with full context:
    - Introduction (required): Multi-audience, value proposition
    - Overview sections (free-form): LLM decides count and titles
    - Detail section specs (structured): Specs for focused generation
    - Conclusion (required): Takeaways and next steps
    """

    title: str = Field(description="Page title", validation_alias=AliasChoices("title", "page_title"))

    # REQUIRED: Introduction - written during planning with complete context
    introduction: str = Field(
        description="Comprehensive introduction with WHY/WHAT/WHO, multi-audience, can include <code_context> citations"
    )

    # FREE-FORM: Overview sections - LLM decides how many and what titles
    overview_sections: list[OverviewSection] = Field(
        default_factory=list,
        description="Overview sections (0 or more) - LLM decides titles like 'Architecture', 'Key Concepts', etc.",
    )

    # STRUCTURED: Detail section specifications for focused generation
    detail_sections: list[DetailSectionSpec] = Field(
        default_factory=list, description="Specifications for sections generated with focused context"
    )

    # REQUIRED: Conclusion - written during planning with complete context
    conclusion: str = Field(
        description="Comprehensive conclusion with takeaways and next steps, can include <code_context> citations"
    )

    model_config = ConfigDict(extra="ignore")


# -------------------- Phase 2 Conceptual Summary Models --------------------


class Component(BaseModel):
    name: str
    purpose: str
    key_files: list[str] = Field(default_factory=list)
    key_directories: list[str] = Field(default_factory=list)
    responsibilities: list[str] = Field(default_factory=list)
    dependencies: list[str] = Field(default_factory=list)
    risks: list[str] = Field(default_factory=list)
    maturity: str = Field(default="unknown")  # experimental|stable|legacy|unknown


class DataFlow(BaseModel):
    name: str
    description: str
    sources: list[str] = Field(default_factory=list)
    sinks: list[str] = Field(default_factory=list)
    transformation_summary: str
    critical_path: bool = False
    failure_modes: list[str] = Field(default_factory=list)


class ExtensionPoint(BaseModel):
    name: str
    mechanism: str  # plugin/hooks/config/override
    locations: list[str] = Field(default_factory=list)
    stability: str = Field(default="internal")  # stable|volatile|internal
    typical_use_cases: list[str] = Field(default_factory=list)


class GlossaryItem(BaseModel):
    term: str
    definition: str
    related_terms: list[str] = Field(default_factory=list)


class RecommendedSection(BaseModel):
    title: str
    rationale: str
    priority: int
    suggested_pages: list[str] = Field(default_factory=list)


class ConceptualRepositorySummary(BaseModel):
    high_level_architecture: str
    major_components: list[Component]
    data_flows: list[DataFlow]
    extension_points: list[ExtensionPoint]
    glossary: list[GlossaryItem]
    recommended_documentation_sections: list[RecommendedSection]
    risks: list[str]
    notes: str
