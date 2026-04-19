"""
Optimized LangGraph Wiki Generation Agent

Following LangGraph best practices with clean state management and configuration separation.
Configuration comes from API wrapper, state only contains workflow execution data.
"""

import heapq
import json
import logging
import math
import os
import random
import re
import time
import traceback
import uuid
from collections import Counter
from datetime import datetime
from typing import Any

from langchain_core.documents import Document
from langchain_core.language_models import BaseLanguageModel
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph
from langgraph.types import Send

from ..code_graph.expansion_engine import SMART_EXPANSION_ENABLED, expand_smart, has_relationship
from ..constants import (
    ARCHITECTURAL_SYMBOLS,
    COVERAGE_IGNORE_DIRS,
    DOC_SYMBOL_TYPES,
    DOCUMENTATION_EXTENSIONS_SET,
    DOT_DIR_WHITELIST,
    EXPANSION_SYMBOL_TYPES,
    KNOWN_FILENAMES,
)
from ..document_compressor import DocumentCompressor
from ..document_ranker import DocumentRanker
from ..prompts.wiki_prompts_enhanced import (
    ENHANCED_CONTENT_GENERATION_PROMPT_V3_TONE_ADJUSTED,
    ENHANCED_REPO_ANALYSIS_PROMPT,
    ENHANCED_WIKI_STRUCTURE_PROMPT,
    STRUCTURED_REPO_ANALYSIS_PROMPT,
    TARGET_AUDIENCES,
)
from ..state.wiki_state import (
    PageGenerationState,
    PageSpec,
    RepositoryAnalysis,
    SectionSpec,
    TargetAudience,
    WikiPage,
    WikiState,
    WikiStructureSpec,
    WikiStyle,
)
from ..token_counter import CONTEXT_TOKEN_BUDGET, get_token_counter

logger = logging.getLogger(__name__)

# Toggle to skip repository_context in page generation (test mode for token savings)
SKIP_REPO_CONTEXT_FOR_PAGES = os.getenv("WIKIS_SKIP_REPO_CONTEXT_FOR_PAGES", "0") == "1"

# Toggle to use structured JSON analysis instead of verbose markdown (~10x token reduction)
# Set to "1" to use new STRUCTURED_REPO_ANALYSIS_PROMPT (outputs ~5-8K JSON)
# Set to "0" to use legacy ENHANCED_REPO_ANALYSIS_PROMPT (outputs ~67K markdown)
USE_STRUCTURED_REPO_ANALYSIS = os.getenv("WIKIS_USE_STRUCTURED_REPO_ANALYSIS", "0") == "1"

# =============================================================================
# Feature Flags for Doc/Code Separation
# =============================================================================
# When enabled, documentation is retrieved from vector store (semantic) instead of graph
# Requires WIKIS_DOC_SEPARATE_INDEX=1 during indexing to be effective
SEPARATE_DOC_INDEX = os.getenv("WIKIS_DOC_SEPARATE_INDEX", "0") == "1"

# When enabled, use Ensemble + EmbeddingsFilter for semantic doc retrieval
# This provides content-based matching instead of filename/path matching
USE_SEMANTIC_DOC_RETRIEVAL = os.getenv("WIKIS_DOC_SEMANTIC_RETRIEVAL", "0") == "1"

# When enabled, docs are auto-retrieved using retrieval_query during generation
# LLM doesn't need to manually specify target_docs during structure planning
AUTO_TARGET_DOCS = os.getenv("WIKIS_AUTO_TARGET_DOCS", "0") == "1"

# Log feature flags at import time
if SEPARATE_DOC_INDEX or USE_SEMANTIC_DOC_RETRIEVAL or AUTO_TARGET_DOCS or SMART_EXPANSION_ENABLED:
    logger.info(
        f"[FEATURE FLAGS] Doc retrieval: separate_index={SEPARATE_DOC_INDEX}, "
        f"semantic_retrieval={USE_SEMANTIC_DOC_RETRIEVAL}, auto_target={AUTO_TARGET_DOCS}, "
        f"smart_expansion={SMART_EXPANSION_ENABLED}"
    )

# Optional thinking emitter (safe no-op outside task context)
try:  # pragma: no cover
    from tools import this  # type: ignore
except Exception:  # pragma: no cover
    this = None  # type: ignore


def add_jittered_delay(min_delay: float = 0.5, max_delay: float = 1.0):
    """Add a random delay with jitter before parallel page content generation to avoid backend throttling.

    Rate limiting is managed by litellm and backend services, but we add a small
    delay on the plugin side to spread out parallel page generation requests.
    Only used for parallel operations; sequential calls don't need delays.

    Args:
        min_delay: Minimum delay in seconds (default: 0.1s)
        max_delay: Maximum delay in seconds (default: 0.5s)
    """
    delay = random.uniform(min_delay, max_delay)  # noqa: S311 — used for jitter/backoff delay, not security
    time.sleep(delay)


class OptimizedWikiGenerationAgent:
    """Optimized wiki generation agent with proper configuration management"""

    def __init__(
        self,
        indexer,  # Accept any indexer (filesystem-based expected)
        retriever_stack: Any,  # UnifiedRetriever
        llm: BaseLanguageModel,
        repository_url: str,
        branch: str = "main",
        bucket_name: str = "wiki_artifacts",
        # Basic parameters (formerly WikiConfiguration)
        wiki_style: WikiStyle = WikiStyle.COMPREHENSIVE,
        target_audience: TargetAudience = TargetAudience.MIXED,
        # Quality parameters
        quality_threshold: float = 0.7,
        require_diagrams: bool = True,
        require_code_examples: bool = True,
        min_content_length: int = 300,
        # Processing parameters
        max_concurrent_pages: int = 4,
        max_retries: int = 3,
        enable_progress_tracking: bool = True,
        # Advanced features
        enable_quality_enhancement: bool = True,
        enable_diagram_optimization: bool = True,
        enable_structure_analysis: bool = True,
        # Graph text index (FTS5) — optional, for O(log N) symbol lookup
        graph_text_index: Any | None = None,
        # Tiered LLM — optional low-cost model for quality/enhancement
        llm_low: BaseLanguageModel | None = None,
        # Progress callback — (phase, progress, message) for SSE streaming
        progress_callback: Any | None = None,
        # Structure planner options
        planner_type: str = "agent",
        exclude_tests: bool = False,
    ):

        # Validate all required components
        if not indexer:
            raise ValueError("Indexer is required")
        if not retriever_stack:
            raise ValueError("Retriever stack is required")
        if not llm:
            raise ValueError("BaseLanguageModel is required")
        if not repository_url:
            raise ValueError("Repository URL is required")

        # Store configuration as instance variables (not in state)
        self.indexer = indexer
        self.retriever_stack = retriever_stack
        self.llm = llm
        self.llm_low = llm_low or llm  # fallback to primary if not set
        self.repository_url = repository_url
        self.branch = branch

        self.bucket_name = bucket_name

        # Store wiki configuration parameters directly
        self.wiki_style = wiki_style
        self.target_audience = target_audience
        self.quality_threshold = quality_threshold
        self.require_diagrams = require_diagrams
        self.require_code_examples = require_code_examples
        self.min_content_length = min_content_length
        self.max_concurrent_pages = max_concurrent_pages
        self.max_retries = max_retries
        self.enable_progress_tracking = enable_progress_tracking
        self.enable_quality_enhancement = enable_quality_enhancement
        self.enable_diagram_optimization = enable_diagram_optimization
        self.enable_structure_analysis = enable_structure_analysis
        self.graph_text_index = graph_text_index  # FTS5 index for O(log N) symbol lookup
        self.progress_callback = progress_callback
        self.planner_type = planner_type
        self.exclude_tests = exclude_tests
        self._pages_generated = 0
        self._total_pages = 0
        self._progress_lock = __import__("threading").Lock()

        # Log component initialization for debugging
        logger.info("OptimizedWikiGenerationAgent initialized with:")
        logger.info(f"  - Repository: {repository_url} (branch: {branch})")
        logger.info(f"  - Indexer: {type(indexer).__name__}")
        logger.info(f"  - Retriever: {type(retriever_stack).__name__}")
        logger.info(f"  - LLM: {type(llm).__name__}")

        # Cluster expansion state (lazy-opened)
        self._cluster_db = None       # UnifiedWikiDB instance (opened on first use)
        self._cluster_db_path = None  # Cached path for the unified DB file

        # Build code_graph
        self.graph = self._build_graph()

        # Progress tracking callback (if available)
        try:
            from langchain_core.callbacks import dispatch_custom_event

            self.dispatch_event = dispatch_custom_event
        except ImportError:

            def fallback_dispatch(event_type, data):
                logger.info(f"Progress: {event_type} - {data.get('message', '')}")

            self.dispatch_event = fallback_dispatch

    # Node implementations (using self for configuration access)

    def analyze_repository(self, state: WikiState, config: RunnableConfig) -> dict[str, Any]:
        """Analyze repository using EXISTING indexer data (no double indexing)"""

        # Configuration comes from self, not state
        repo_url = self.repository_url
        branch = self.branch

        # Validate components before starting
        if not self.indexer:
            raise ValueError("Indexer is required but not provided")
        if not self.llm:
            raise ValueError("BaseLanguageModel (LLM) is required but not provided")

        logger.info(f"Starting repository analysis for {repo_url} (branch: {branch})")
        logger.info(f"Using LLM: {type(self.llm).__name__}")
        logger.info(f"Using Indexer: {type(self.indexer).__name__}")
        if self.progress_callback:
            try:
                self.progress_callback("indexing", 0.15, "Analyzing repository structure...")
            except Exception:
                pass
        if this and getattr(this, "module", None):
            this.module.invocation_thinking(
                "I am on phase repository_analysis\nAnalyzing repository structure & README\nReasoning: Derive architectural + conceptual overview to ground subsequent structure planning.\nNext: Aggregate documents, extract tree & README, then synthesize analysis."
            )

        # self._dispatch_progress("wiki_analysis_started", {
        #     "progress_percentage": 0.0,
        #     "message": f"🔍 Starting repository analysis: {repo_url}",
        #     "phase": "repository_analysis"
        # })

        try:
            # Get documents from indexer (already indexed)
            all_documents = self.indexer.get_all_documents()
            if not all_documents:
                raise ValueError("No documents found in indexer - ensure repository is indexed first")

            # Get index statistics
            index_stats = getattr(self.indexer, "last_index_stats", {}) or {}
            logger.info(f"Retrieved {len(all_documents)} documents from indexer")
            if this and getattr(this, "module", None):
                this.module.invocation_thinking(
                    f"I am on phase repository_analysis\nCollected {len(all_documents)} documents – synthesizing analysis\nReasoning: Need holistic context vector to inform section/page allocation.\nNext: Build repository tree + extract README + run LLM synthesis."
                )

            # Build repository tree from repository file paths (avoid doc-based tree)
            repository_files = self._get_repository_file_paths()
            if not repository_files:
                repository_files = list(
                    set(doc.metadata.get("source", "") for doc in all_documents if doc.metadata.get("source"))
                )
            logger.info(f"Repository file list size: {len(repository_files)}")
            repository_tree = self._create_repository_tree(repository_files)

            # Extract full README content from repository files first (fallback to docs)
            readme_content = self._extract_full_readme_content_from_files(repository_files)
            if readme_content == "No README file found":
                readme_content = self._extract_full_readme_content_from_docs(all_documents)

            # Create basic repository analysis
            repo_analysis_obj = self._create_repository_analysis(
                repository_files, index_stats, readme_content=readme_content
            )

            # Prepare representative samples for analysis prompt
            # Prefer disk sampling (faster, more accurate) with fallback to docs
            code_samples = self._extract_representative_code_samples_from_files(repository_files)
            if not code_samples or code_samples == "No representative code samples available":
                code_samples = self._extract_representative_code_samples(all_documents)

            # Use LLM to create comprehensive analysis text (sync version)
            if self.progress_callback:
                try:
                    self.progress_callback("indexing", 0.18, f"Analyzing repository with LLM — {len(repository_files)} files, {len(all_documents)} documents...")
                except Exception:
                    pass
            repo_analysis_text = self._llm_analyze_repository(
                repository_tree, readme_content, code_samples, repository_files, config
            )

            # Store rich context in state (not configuration parameters)
            if self.progress_callback:
                try:
                    self.progress_callback("indexing", 0.25, f"Repository analysis complete — {len(repo_analysis_text)} chars of context extracted")
                except Exception:
                    pass
            if this and getattr(this, "module", None):
                this.module.invocation_thinking(
                    "I am on phase repository_analysis\nRepository analysis complete\nReasoning: Conceptual model established.\nNext: Invoke structure planning to map pages & sections."
                )
            return {
                "repository_analysis": repo_analysis_obj,
                "repository_context": repo_analysis_text,
                "repository_tree": repository_tree,
                "readme_content": readme_content,
                "current_phase": "repository_analysis_complete",
                # "messages": [f"Repository analysis complete: {len(unique_sources)} files analyzed with LLM insights"]
            }

        except Exception as e:
            logger.error(f"Repository analysis failed: {e}", exc_info=True)
            if this and getattr(this, "module", None):
                this.module.invocation_thinking(
                    "I am on phase repository_analysis\nRepository analysis failed – continuing with limited context\nReasoning: Resilient fallback to avoid full pipeline abort.\nNext: Attempt structure planning with degraded insights."
                )
            # Build a minimal fallback so structure planning still has something to work with.
            # Returning only errors with no analysis fields causes generate_wiki_structure to
            # also fail and leaves wiki_structure_spec=None, which breaks downstream nodes.
            try:
                fallback_files = self._get_repository_file_paths() or []
                fallback_analysis = self._create_repository_analysis(fallback_files, {})
            except Exception:
                fallback_analysis = None
            logger.warning(
                "Repository analysis failed; proceeding with fallback minimal analysis. "
                "Wiki quality may be reduced."
            )
            return {
                "errors": [f"Repository analysis failed: {e}"],
                "current_phase": "repository_analysis_failed",
                "repository_analysis": fallback_analysis,
                "repository_context": "",
                "repository_tree": "",
                "readme_content": "",
            }

    def generate_wiki_structure(self, state: WikiState, config: RunnableConfig) -> dict[str, Any]:
        """Generate wiki structure using LLM analysis with complete repository context"""

        # Configuration comes from self
        repo_analysis = state.get("repository_analysis")
        repository_tree = state.get("repository_tree", "")
        readme_content = state.get("readme_content", "")
        repo_context = state.get("repository_context", "")

        if not repo_analysis and not repo_context:
            error_msg = "Repository analysis missing from state"
            logger.error(error_msg)
            return {"errors": [error_msg]}

        try:
            # Validate LLM is available
            if not self.llm:
                raise ValueError("LLM is required for structure generation")

            if self.progress_callback:
                try:
                    self.progress_callback("planning", 0.28, "Planning wiki structure with LLM...")
                except Exception:
                    pass

            # ── Cluster planner path ──────────────────────────────────
            planner_type = state.get("planner_type", self.planner_type)
            exclude_tests = state.get("exclude_tests", self.exclude_tests)

            if planner_type == "cluster":
                try:
                    result = self._generate_wiki_structure_cluster(
                        state=state, config=config, exclude_tests=exclude_tests,
                    )
                    if result is not None:
                        return result
                    # None means fallback to existing planners
                    logger.warning("Cluster planner returned no result, falling back to agent planner")
                except Exception as cluster_err:
                    logger.warning(
                        "Cluster planner failed, falling back to agent planner: %s",
                        cluster_err,
                    )

            repository_files = self._get_repository_file_paths()
            use_deepagents = self._should_use_deepagents_structure_planner(
                repo_context=repo_context,
                repository_files=repository_files,
            )

            if use_deepagents:
                # ── Graph-First path (feature-flagged) ────────────────
                use_graph_first = os.getenv("WIKIS_USE_GRAPH_FIRST", "0") == "1"
                _gf_graph = getattr(self.retriever_stack, "relationship_graph", None) or getattr(
                    self.indexer, "relationship_graph", None
                )
                if use_graph_first and _gf_graph is not None:
                    logger.info("Structure planner: graph-first (WIKIS_USE_GRAPH_FIRST=1)")
                    try:
                        response = self._generate_wiki_structure_graph_first(
                            repository_files=repository_files,
                            target_audience=TARGET_AUDIENCES.get(
                                self.target_audience.value,
                                "Mixed audience with varied technical backgrounds",
                            ),
                            wiki_type=self.wiki_style.value,
                            config=config,
                        )
                        logger.info("WikiStructureSpec created successfully (graph-first)")
                        logger.info(f"Wiki structure sections: {len(response.sections)}")
                        if this and getattr(this, "module", None):
                            total_pages = sum(len(s.pages) for s in response.sections)
                            this.module.invocation_thinking(
                                f"I am on phase structure_planning\nStructure designed: {len(response.sections)} sections / {total_pages} pages (graph-first)\nReasoning: Deterministic skeleton from code graph with LLM refinement.\nNext: Parallelize page drafting with focused context windows."
                            )
                        return {
                            "wiki_structure_spec": response,
                            "structure_planning_complete": True,
                            "current_phase": "structure_complete",
                        }
                    except Exception as gf_error:
                        logger.warning(
                            "Graph-first structure planner failed, falling back to deepagents: %s",
                            gf_error,
                        )

                # ── DeepAgents path (default) ─────────────────────────
                logger.info("Structure planner: deepagents (auto)")
                try:
                    response = self._generate_wiki_structure_with_deepagents(
                        repository_tree=repository_tree,
                        readme_content=readme_content,
                        repo_analysis=repo_analysis,
                        repo_context=repo_context,
                        repository_files=repository_files,
                        target_audience=TARGET_AUDIENCES.get(
                            self.target_audience.value,
                            "Mixed audience with varied technical backgrounds",
                        ),
                        wiki_type=self.wiki_style.value,
                        config=config,
                    )
                    logger.info("WikiStructureSpec created successfully (deepagents)")
                    logger.info(f"Wiki structure sections: {len(response.sections)}")
                    if this and getattr(this, "module", None):
                        total_pages = sum(len(s.pages) for s in response.sections)
                        this.module.invocation_thinking(
                            f"I am on phase structure_planning\nStructure designed: {len(response.sections)} sections / {total_pages} pages\nReasoning: Coverage tuned to repository breadth; each page targets coherent feature or module.\nNext: Parallelize page drafting with focused context windows."
                        )
                    return {
                        "wiki_structure_spec": response,
                        "structure_planning_complete": True,
                        "current_phase": "structure_complete",
                    }
                except Exception as deepagents_error:
                    logger.warning(f"Deepagents structure planner failed, falling back to LLM: {deepagents_error}")

            logger.info(f"Starting structure generation with LLM: {type(self.llm).__name__}")

            # Create prompt with repository analysis
            structure_prompt = ChatPromptTemplate.from_messages(
                [
                    (
                        "system",
                        "You are an expert technical documentation architect specializing in comprehensive repository analysis. Your task is to create complete documentation structures that capture every significant component without omission. You must ensure comprehensive coverage of all repository elements, creating structures that scale appropriately based on actual codebase complexity and component count.",
                    ),
                    ("human", ENHANCED_WIKI_STRUCTURE_PROMPT),
                ]
            )

            # Get target audience description
            target_audience = TARGET_AUDIENCES.get(
                self.target_audience.value, "Mixed audience with varied technical backgrounds"
            )

            # Use the LLM analysis directly (not formatting needed)
            repo_analysis_str = repo_context or str(repo_analysis)

            # Get wiki type - default to comprehensive
            wiki_type = self.wiki_style.value  # Could be made configurable if needed

            logger.info("Invoking LLM for structure generation...")
            if this and getattr(this, "module", None):
                this.module.invocation_thinking(
                    "I am on phase structure_planning\nDesigning documentation structure\nReasoning: Translate analysis into hierarchical sections & pages sized to codebase complexity.\nNext: Produce structured spec then dispatch page drafting."
                )

            prompt_messages = structure_prompt.format_messages(
                repository_tree=repository_tree,
                readme_content=readme_content,
                repo_analysis=repo_analysis_str,
                target_audience=target_audience,
                wiki_type=wiki_type,
            )

            # NOTE: Avoid `with_structured_output` here.
            # Some OpenAI-compatible proxies intermittently produce invalid streaming events
            # (e.g., message.content=None / role=None) when response_format parsing is enabled.
            # We instead request JSON in the prompt and parse/validate manually.
            llm_message = self.llm.invoke(prompt_messages, config=config)
            raw_content = getattr(llm_message, "content", "")
            if raw_content is None:
                raw_text = ""
            elif isinstance(raw_content, str):
                raw_text = raw_content
            elif isinstance(raw_content, list):
                parts = []
                for item in raw_content:
                    if item is None:
                        continue
                    if isinstance(item, str):
                        parts.append(item)
                    elif isinstance(item, dict):
                        parts.append(str(item.get("text") or item.get("content") or ""))
                    else:
                        parts.append(str(item))
                raw_text = "\n".join([p for p in parts if p]).strip()
            else:
                raw_text = str(raw_content)

            data = self._parse_llm_json_response(raw_text)
            try:
                response = WikiStructureSpec.model_validate(data)
            except Exception as ve:
                if os.getenv("WIKIS_LOG_LLM_STRUCTURE_RAW") == "1":
                    raw = raw_text or ""
                    raw_len = len(raw)
                    prefix = raw[:4000].replace("\n", "\\n")
                    suffix = raw[-800:].replace("\n", "\\n") if raw_len > 4000 else ""
                    logger.warning(
                        f"WikiStructureSpec validation failed: {ve}. Raw length={raw_len}. "
                        f"Prefix(4k)='{prefix}'" + (f" Suffix(800)='{suffix}'" if suffix else "")
                    )
                logger.warning(f"WikiStructureSpec validation failed; using fallback structure. Error: {ve}")
                response = WikiStructureSpec.model_validate(self._parse_llm_json_response(""))

            logger.info("WikiStructureSpec created successfully")
            total_pages = sum(len(s.pages) for s in response.sections)
            logger.info(f"Wiki structure sections: {len(response.sections)}")
            if self.progress_callback:
                try:
                    section_names = ", ".join(s.title for s in response.sections[:5])
                    suffix = f"... and {len(response.sections) - 5} more" if len(response.sections) > 5 else ""
                    self.progress_callback("planning", 0.29, f"Structure planned — {len(response.sections)} sections, {total_pages} pages: {section_names}{suffix}")
                except Exception:
                    pass
            if this and getattr(this, "module", None):
                this.module.invocation_thinking(
                    f"I am on phase structure_planning\nStructure designed: {len(response.sections)} sections / {total_pages} pages\nReasoning: Coverage tuned to repository breadth; each page targets coherent feature or module.\nNext: Parallelize page drafting with focused context windows."
                )

            return {
                "wiki_structure_spec": response,
                "structure_planning_complete": True,
                "current_phase": "structure_complete",
            }

        except Exception as e:
            logger.error(f"Structure generation failed: {e}", exc_info=True)
            # Re-raise so generate_wiki() catches it and returns success=False.
            # Returning a dict without wiki_structure_spec would let the graph continue
            # into dispatch_page_generation with None structure, causing NoneType errors
            # in finalize_wiki and export_wiki.
            raise RuntimeError(f"Structure generation failed: {e}") from e

    # Maximum symbols per page — pages exceeding this are auto-split.
    # Scaled by repo size in _compute_max_symbols_per_page().
    MAX_SYMBOLS_PER_PAGE = 25

    @staticmethod
    def _compute_max_symbols_per_page(graph_nodes: int, file_count: int) -> int:
        """Scale the per-page symbol cap based on repository size.

        Small repos keep tight pages (20-25 symbols).  Large repos get
        broader pages (up to 50) so the planner can create chapter-level
        pages instead of hyper-granular ones.
        """
        # Use the larger of the two size signals
        size_signal = max(graph_nodes, file_count)
        if size_signal >= 5000:
            return 50
        elif size_signal >= 2000:
            return 40
        elif size_signal >= 500:
            return 30
        else:
            return 25  # default for small repos

    def dispatch_page_generation(self, state: WikiState, config: RunnableConfig) -> list[Send]:
        """Dispatch parallel page generation using Send"""

        logger.info("📄 Dispatching page generation")

        if not state.get("wiki_structure_spec"):
            # Raising here propagates cleanly through the LangGraph executor and is caught
            # by generate_wiki()'s outer try/except, which returns success=False with an
            # error message.  Returning [] silently would let the graph continue into
            # finalize_wiki and export_wiki with a None structure spec, causing
            # AttributeError on wiki_structure.wiki_title and similar crashes.
            raise RuntimeError(
                "wiki_structure_spec is missing from state — structure planning must have failed. "
                "Cannot dispatch page generation."
            )

        wiki_structure = state["wiki_structure_spec"]

        # ── Post-structure page splitting ──
        # Safety net: if the structure planner assigned too many symbols to a
        # page (>MAX_SYMBOLS_PER_PAGE), split it deterministically so that
        # each sub-page comfortably fits within the context token budget.
        self._split_overloaded_pages(wiki_structure)

        # Generate all pages at once (simplified - no batching)
        sends = []
        for section_idx, section in enumerate(wiki_structure.sections):
            for page_idx, page in enumerate(section.pages):
                page_id = f"{section_idx}#{page_idx}"
                # Use Pydantic v2 model_dump for structured models; fallback to __dict__
                page_spec_dict = (
                    page.model_dump()
                    if hasattr(page, "model_dump")
                    else (page.dict() if hasattr(page, "dict") else page.__dict__)
                )

                sends.append(
                    Send(
                        "generate_page_content",
                        {
                            "page_id": page_id,
                            "page_spec": page_spec_dict,
                            "repository_context": state["repository_context"],
                        },
                    )
                )

        self._total_pages = len(sends)
        self._pages_generated = 0
        logger.info(f"📄 Dispatching {len(sends)} page generation tasks")
        if self.progress_callback:
            try:
                self.progress_callback("generating", 0.30, f"Generating {len(sends)} wiki pages...")
            except Exception:
                pass
        if this and getattr(this, "module", None):
            this.module.invocation_thinking(
                f"I am on phase page_generation\nPlanning {len(sends)} pages for parallel drafting\nReasoning: Concurrency reduces end-to-end latency while leveraging shared repo analysis.\nNext: Draft each page with targeted context retrieval."
            )
        return sends

    def _split_overloaded_pages(self, wiki_structure) -> None:
        """Split pages that have too many target_symbols to fit the context budget.

        Groups symbols by their source file path so that related symbols stay
        together on the same sub-page.  Each sub-page inherits the parent's
        section, description, retrieval_query, target_docs, etc.

        Mutates ``wiki_structure.sections[*].pages`` in place.
        """
        from ..state.wiki_state import PageSpec

        # Compute repo-aware default cap
        graph = getattr(self.retriever_stack, "relationship_graph", None)
        graph_nodes = graph.number_of_nodes() if graph else 0
        file_count = getattr(self, "_repository_file_count", 0) or 0
        default_cap = self._compute_max_symbols_per_page(graph_nodes, file_count)
        max_syms = self._get_env_int("WIKIS_MAX_SYMBOLS_PER_PAGE", default_cap)
        total_splits = 0

        for section in wiki_structure.sections:
            new_pages: list = []
            for page in section.pages:
                symbols = getattr(page, "target_symbols", []) or []
                if len(symbols) <= max_syms:
                    new_pages.append(page)
                    continue

                # Group symbols by file path for cohesive sub-pages
                file_groups: dict = {}  # file_path -> [symbol_name]
                ungrouped: list = []
                if graph is not None:
                    name_index = getattr(graph, "_name_index", None) or {}
                    for sym in symbols:
                        fpath = None
                        nids = name_index.get(sym, [])
                        if nids:
                            nd = graph.nodes.get(nids[0], {})
                            fpath = nd.get("rel_path") or nd.get("file_path", "")
                        if fpath:
                            file_groups.setdefault(fpath, []).append(sym)
                        else:
                            ungrouped.append(sym)
                else:
                    ungrouped = list(symbols)

                # Distribute ungrouped symbols across file groups (or make their own)
                if ungrouped:
                    if file_groups:
                        # Spread across existing groups roughly evenly
                        groups_list = list(file_groups.values())
                        for i, sym in enumerate(ungrouped):
                            groups_list[i % len(groups_list)].append(sym)
                    else:
                        # No file grouping info — just chunk linearly
                        file_groups["__ungrouped__"] = ungrouped

                # Merge small groups into chunks of ≤ max_syms
                chunks: list = []
                current_chunk: list = []
                for _fpath, syms in sorted(file_groups.items()):
                    if len(current_chunk) + len(syms) > max_syms and current_chunk:
                        chunks.append(current_chunk)
                        current_chunk = []
                    current_chunk.extend(syms)
                    # If a single file group exceeds max_syms, split it
                    while len(current_chunk) > max_syms:
                        chunks.append(current_chunk[:max_syms])
                        current_chunk = current_chunk[max_syms:]
                if current_chunk:
                    chunks.append(current_chunk)

                # Create sub-pages
                for part_idx, chunk in enumerate(chunks, start=1):
                    suffix = f" (Part {part_idx})" if len(chunks) > 1 else ""
                    sub_page = PageSpec(
                        page_name=f"{page.page_name}{suffix}",
                        page_order=page.page_order * 100 + part_idx,
                        description=page.description,
                        content_focus=page.content_focus,
                        rationale=f"{page.rationale} [auto-split part {part_idx}/{len(chunks)}]",
                        target_symbols=chunk,
                        target_docs=list(getattr(page, "target_docs", []) or []),
                        target_folders=list(getattr(page, "target_folders", []) or []),
                        key_files=list(getattr(page, "key_files", []) or []),
                        retrieval_query=getattr(page, "retrieval_query", ""),
                    )
                    new_pages.append(sub_page)

                logger.info(
                    f"[PAGE_SPLIT] '{page.page_name}' had {len(symbols)} symbols → "
                    f"split into {len(chunks)} sub-pages (max {max_syms} per page)"
                )
                total_splits += len(chunks) - 1

            section.pages = new_pages

        if total_splits > 0:
            wiki_structure.total_pages = sum(len(s.pages) for s in wiki_structure.sections)
            logger.info(
                f"[PAGE_SPLIT] Total: {total_splits} extra pages created from overloaded pages. "
                f"New total: {wiki_structure.total_pages} pages"
            )

    def generate_page_content(self, state: PageGenerationState, config: RunnableConfig) -> dict[str, Any]:
        """Generate content for a single page using LLM"""

        # Configuration comes from self
        page_id = state["page_id"]
        page_spec_dict = state["page_spec"]
        repo_context = state["repository_context"]

        # Recreate PageSpec object from dict
        from ..state.wiki_state import PageSpec

        page_spec = PageSpec(**page_spec_dict) if isinstance(page_spec_dict, dict) else page_spec_dict

        # Create meaningful display ID for progress tracking
        display_page_id = f"{page_spec.page_name}"

        try:
            if this and getattr(this, "module", None):
                this.module.invocation_thinking(
                    f"I am on phase page_generation\nDrafting page: {page_spec.page_name}\nReasoning: Converting structural intent into detailed narrative leveraging repository signals.\nNext: Retrieve relevant symbols/files then invoke LLM."
                )
            # Get relevant content for this page (enhanced with location mapping)
            if self.progress_callback:
                try:
                    self.progress_callback("generating", 0.30, f"Retrieving context for: {page_spec.page_name}...")
                except Exception:
                    pass
            relevant_content = self._get_relevant_content_for_page(page_spec, self.repository_url, repo_context)

            # Retrieval diagnostics
            n_files = len(relevant_content.get("files", []))
            n_code = relevant_content.get("code_files", 0)
            n_docs = relevant_content.get("documentation_files", 0)
            n_prioritized = relevant_content.get("prioritized_sources", 0)
            content_chars = len(relevant_content.get("content", ""))
            content_tokens_est = content_chars // 4
            retrieval_failed = relevant_content.get("retrieval_failed", False)

            logger.info(
                f"[RETRIEVAL] page='{page_spec.page_name}' "
                f"files={n_files} code={n_code} docs={n_docs} "
                f"prioritized={n_prioritized} tokens≈{content_tokens_est:,} "
                f"failed={retrieval_failed}"
            )

            if self.progress_callback:
                try:
                    self.progress_callback("generating", 0.30, f"Writing: {page_spec.page_name}")
                except Exception:
                    pass

            if not n_files and not retrieval_failed:
                logger.warning(
                    f"[PAGE_GEN] 0 docs retrieved for '{page_spec.page_name}' — "
                    f"content may be generic. Check vectorstore/graph health."
                )

            if this and getattr(this, "module", None):
                this.module.invocation_thinking(
                    f"I am on phase page_generation\nRetrieved context for: {page_spec.page_name}\nReasoning: Curated symbol + file set narrows LLM focus while preserving completeness.\nNext: Generate draft content then sanitize diagrams."
                )

            # Generate page content (all pages use simple mode after ranked truncation)
            generated_content = self._generate_simple(page_spec, relevant_content, repo_context, config)

            # Post-process: sanitize Mermaid diagrams (Phase 1 heuristic sanitizer)
            try:
                from ..diagram_sanitizer import SanitizerConfig, sanitize_content

                sanitized_content, summary = sanitize_content(generated_content, SanitizerConfig())
                if summary.total:
                    logger.info(
                        f"🖼 Mermaid diagrams sanitized for page '{page_spec.page_name}': total={summary.total} valid={summary.valid} fixed={summary.fixed} failed={summary.failed}"
                    )
                    # Replace content purely with sanitized version (no extra markers)
                    generated_content = sanitized_content
                    if this and getattr(this, "module", None):
                        this.module.invocation_thinking(
                            f"I am on phase page_generation\nSanitized diagrams in: {page_spec.page_name}\nReasoning: Ensured Mermaid diagrams are syntactically valid for downstream rendering.\nNext: Finalize page draft output."
                        )
            except Exception as se:  # fail-soft: do not block page generation
                logger.warning(f"Mermaid sanitization skipped for page '{page_spec.page_name}': {se}")
            if this and getattr(this, "module", None):
                this.module.invocation_thinking(
                    f"I am on phase page_generation\nCompleted page: {page_spec.page_name}\nReasoning: Draft ready for potential quality assessment/enhancement.\nNext: Aggregate all drafts; proceed to finalization."
                )

            # Report per-page progress via SSE callback
            if self.progress_callback:
                try:
                    with self._progress_lock:
                        self._pages_generated += 1
                        done = self._pages_generated
                    total = self._total_pages or 1
                    # Map page generation to 30%–85% range
                    pct = 0.30 + (done / total) * 0.55
                    self.progress_callback(
                        "generating",
                        pct,
                        f"Generated page: {page_spec.page_name} ({done}/{total})",
                    )
                except Exception:
                    pass  # never break generation for progress

            return {
                "wiki_pages": [
                    WikiPage(
                        page_id=page_id,
                        title=page_spec.page_name,
                        content=generated_content,
                        status="completed",
                        retry_count=0,
                    )
                ],
            }

        except Exception as e:
            is_overflow = False
            try:
                from app.services.context_overflow import classify_and_wrap, is_context_overflow

                is_overflow = is_context_overflow(e)
                if is_overflow:
                    wrapped = classify_and_wrap(e)
                    logger.error(
                        f"[CONTEXT_OVERFLOW] Page generation failed for '{display_page_id}' "
                        f"due to context window overflow: {wrapped}",
                        exc_info=False,
                    )
                else:
                    logger.error(f"Page generation failed for {display_page_id}: {e}", exc_info=True)
            except Exception:
                logger.error(f"Page generation failed for {display_page_id}: {e}", exc_info=True)

            error_msg = (
                "Context overflow: input too large for model's context window. "
                "Try reducing page scope or using a larger model."
                if is_overflow
                else str(e)
            )

            if this and getattr(this, "module", None):
                this.module.invocation_thinking(
                    f"I am on phase page_generation\nPage failed: {page_spec.page_name}\nError: {error_msg}"
                )

            return {
                "wiki_pages": [
                    WikiPage(page_id=page_id, title=page_spec.page_name, content="", status="failed", retry_count=0)
                ],
                "errors": [f"Page generation failed for {display_page_id}: {e}"],
            }

    def finalize_wiki(self, state: WikiState, config: RunnableConfig) -> dict[str, Any]:
        """Finalize wiki generation and prepare for export"""

        logger.info("🏁 Finalizing wiki generation")
        if self.progress_callback:
            try:
                self.progress_callback("storing", 0.92, "Finalizing wiki — assembling pages and diagrams...")
            except Exception:
                pass
        if this and getattr(this, "module", None):
            this.module.invocation_thinking("Finalizing wiki & aggregating pages")

        try:
            # Consolidate all pages
            wiki_pages = state.get("wiki_pages", [])

            final_pages = list(wiki_pages)

            # Calculate final statistics
            wiki_structure = state.get("wiki_structure_spec")
            if wiki_structure is None:
                # This should not normally be reached because generate_wiki_structure now
                # raises on failure, but guard defensively in case the graph is invoked
                # directly in tests or unusual code paths.
                raise RuntimeError(
                    "wiki_structure_spec is None in finalize_wiki — structure planning failed "
                    "without raising, which is a bug."
                )
            total_pages = len(final_pages)
            total_diagrams = sum(page.content.count("```mermaid") for page in final_pages)

            # Create generation summary
            generation_summary = {
                "wiki_title": wiki_structure.wiki_title,
                "total_sections": len(wiki_structure.sections),
                "total_pages": total_pages,
                "total_diagrams": total_diagrams,
                "generation_timestamp": datetime.now().isoformat(),
                "execution_time": time.time() - state.get("start_time", time.time()),
            }

            return {
                "wiki_pages": final_pages,
                "generation_summary": generation_summary,
                "current_phase": "finalization_complete",
                "messages": [
                    f"Wiki finalized: {total_pages} pages, {total_diagrams} diagrams"
                ],
            }

        except Exception as e:
            logger.error(f"Wiki finalization failed: {e}")
            if this and getattr(this, "module", None):
                this.module.invocation_thinking("Finalization failed")
            return {"errors": [f"Finalization failed: {e}"], "current_phase": "finalization_failed"}

    def export_wiki(self, state: WikiState, config: RunnableConfig) -> dict[str, Any]:
        """Export wiki to artifacts with Context7-style folder structure"""

        # self._dispatch_progress("wiki_export", {
        #     "progress_percentage": 95.0,
        #     "message": "📦 Exporting wiki artifacts",
        #     "phase": "export"
        # })

        try:
            from ..artifact_export import ArtifactExporter, normalize_wiki_id

            # Generate wiki_id from repository info for folder-based storage
            wiki_id = None
            try:
                # Parse repository URL to get owner/repo
                repo_url = self.repository_url
                if repo_url:
                    # Handle various URL formats
                    # https://github.com/owner/repo.git -> owner/repo
                    # git@github.com:owner/repo.git -> owner/repo
                    if "://" in repo_url:
                        path = repo_url.split("://", 1)[1].split("/", 1)[1]
                    elif "@" in repo_url and ":" in repo_url:
                        path = repo_url.split(":", 1)[1]
                    else:
                        path = repo_url

                    # Remove .git suffix
                    path = path.rstrip("/").removesuffix(".git")

                    if "/" in path:
                        wiki_id = normalize_wiki_id(repository=path, branch=self.branch)
                        logger.info(f"Generated wiki_id: {wiki_id}")
            except Exception as e:
                logger.warning(f"Could not generate wiki_id from repository URL: {e}")

            # Create exporter with wiki_id for folder-based storage
            exporter = ArtifactExporter(wiki_id=wiki_id)

            # Convert WikiPage objects to format expected by exporter
            wiki_pages = state.get("wiki_pages", [])
            wiki_structure_spec = state.get("wiki_structure_spec")
            if wiki_structure_spec is None:
                raise RuntimeError(
                    "wiki_structure_spec is None in export_wiki — cannot export without a structure."
                )

            # Export wiki with all generated content
            artifacts = exporter.export_wiki_artifacts(
                wiki_structure=wiki_structure_spec,
                wiki_pages=wiki_pages,
                repo_metadata={
                    "repository_url": self.repository_url,
                    "branch": self.branch,
                    "generation_summary": state.get("generation_summary", {}),
                },
                export_formats=["json", "markdown"],
            )

            # self._dispatch_progress("wiki_export_complete", {
            #     "progress_percentage": 100.0,
            #     "message": f"🎉 Wiki export complete",
            #     "phase": "complete",
            #     # "details": {"artifacts": list(artifacts.keys())}
            # })

            return {
                "artifacts": artifacts,
                "wiki_id": wiki_id,  # Include wiki_id for registry registration
                "export_complete": True,
                # "messages": [f"Export complete: {len(artifacts)} artifacts created"]
            }

        except Exception as e:
            logger.error(f"Wiki export failed: {e}")
            if this and getattr(this, "module", None):
                this.module.invocation_thinking("Export failed")
            return {"errors": [f"Export failed: {e}"], "export_complete": False}

    def _build_graph(self) -> CompiledStateGraph:
        """Build optimized wiki generation code_graph with clean map-reduce chain"""

        # Create code_graph (no config schema needed since config comes from self)
        builder = StateGraph(WikiState)

        # Add nodes
        builder.add_node("analyze_repository", self.analyze_repository)
        builder.add_node("generate_wiki_structure", self.generate_wiki_structure)
        builder.add_node("generate_page_content", self.generate_page_content)
        builder.add_node("finalize_wiki", self.finalize_wiki)
        builder.add_node("export_wiki", self.export_wiki)

        # Define clean map-reduce chain workflow
        builder.add_edge(START, "analyze_repository")
        builder.add_edge("analyze_repository", "generate_wiki_structure")

        # Page Generation (map)
        builder.add_conditional_edges(
            "generate_wiki_structure", self.dispatch_page_generation, ["generate_page_content"]
        )

        # Reduce → finalize → export
        builder.add_edge("generate_page_content", "finalize_wiki")
        builder.add_edge("finalize_wiki", "export_wiki")
        builder.add_edge("export_wiki", END)

        # Compile WITHOUT checkpointing by default to save RAM
        # We don't use human-in-the-loop or need state snapshots
        # Set WIKIS_CHECKPOINT_BACKEND=memory to enable checkpointing if needed
        checkpoint_backend = os.getenv("WIKIS_CHECKPOINT_BACKEND", "none").lower()
        if checkpoint_backend in {"none", "off", "0", "false", ""}:
            return builder.compile()

        checkpointer = MemorySaver()
        return builder.compile(checkpointer=checkpointer)

    def generate_wiki(self, user_message: str) -> dict[str, Any]:
        """Generate wiki using optimized LangGraph workflow with user message"""

        code_graph = (
            getattr(self.retriever_stack, "relationship_graph", None)
            or getattr(self.indexer, "relationship_graph", None)
        )
        if code_graph is not None and code_graph.number_of_nodes() > 0:
            try:
                self._ensure_unified_db(code_graph)
            except Exception as exc:
                logger.warning("[UNIFIED_DB] Failed to prepare unified DB before workflow: %s", exc)

        # Initialize state with clean phase-based structure (only workflow data, no config)
        initial_state: WikiState = {
            "start_time": time.time(),
            "current_phase": "initialization",
            # Generation options
            "planner_type": self.planner_type,
            "exclude_tests": self.exclude_tests,
            # Repository analysis results (state data)
            "repository_analysis": None,
            "repository_context": None,
            "repository_tree": None,
            "readme_content": None,
            "repository_tree_stats": None,
            "conceptual_summary_raw": None,
            "conceptual_summary": None,
            "page_budget": None,
            "page_budget_justification": None,
            "budget_compression_recommendation": None,
            # Wiki structure results (state data)
            "wiki_structure_spec": None,
            "structure_planning_complete": False,
            # Content Generation (operator.add accumulation)
            "wiki_pages": [],
            # Final results (state data)
            "generation_summary": None,
            "artifacts": [],
            "export_complete": False,
            # Monitoring (operator.add accumulation)
            "messages": [],
            "errors": [],
        }

        # Execute workflow with user message (standard LangGraph pattern)
        try:
            runnable_config = {
                "max_concurrency": 4,
                "recursion_limit": 5000,
                "configurable": {
                    "thread_id": uuid.uuid4(),
                },
            }

            # Use sync invoke with complete initial state
            final_state = self.graph.invoke(initial_state, runnable_config)

            # Convert WikiPage objects to expected format for response
            wiki_pages = final_state.get("wiki_pages", [])
            generated_pages = {page.page_id: page.content for page in wiki_pages}

            # Convert WikiStructureSpec to dict format for JSON serialization
            wiki_structure = final_state.get("wiki_structure_spec")
            if wiki_structure is None:
                # The graph completed without raising but left wiki_structure_spec=None.
                # This means the structure planner returned an error dict rather than raising,
                # which is a bug we should surface clearly rather than returning success=True
                # with empty content.
                workflow_errors = final_state.get("errors", []) or []
                error_detail = "; ".join(workflow_errors) if workflow_errors else "wiki_structure_spec was not set"
                logger.error(f"Graph completed but wiki_structure_spec is None: {error_detail}")
                return {
                    "success": False,
                    "error": f"Structure planning produced no output: {error_detail}",
                    "execution_time": time.time() - initial_state["start_time"],
                    "errors": workflow_errors,
                }
            wiki_structure_dict = (
                wiki_structure.model_dump()
                if hasattr(wiki_structure, "model_dump")
                else (wiki_structure.dict() if hasattr(wiki_structure, "dict") else wiki_structure)
            )

            # Surface non-fatal workflow errors (e.g., page generation failures) for callers.
            workflow_errors = final_state.get("errors", []) or []
            failed_pages = [
                {
                    "page_id": page.page_id,
                    "title": page.title,
                    "status": page.status,
                }
                for page in wiki_pages
                if getattr(page, "status", None) and str(page.status).lower() == "failed"
            ]

            return {
                "success": True,
                "wiki_structure": wiki_structure_dict,
                "generated_pages": generated_pages,
                "generation_summary": final_state.get("generation_summary", {}),
                "artifacts": final_state.get("artifacts", []),
                "execution_time": time.time() - initial_state["start_time"],
                "errors": workflow_errors,
                "failed_pages": failed_pages,
                # LLM-generated repository analysis for Ask tool
                "repository_context": final_state.get("repository_context"),
            }

        except Exception as e:
            stacktrace = traceback.format_exc()
            logger.error(f"Wiki generation failed: Error parsing payload params: {stacktrace}")
            logger.error(f"Wiki generation failed: {e}")
            return {"success": False, "error": str(e), "execution_time": time.time() - initial_state["start_time"]}

    # Helper Methods

    def _create_repository_analysis(
        self, file_paths: list[str], index_stats: dict[str, Any], readme_content: str = ""
    ) -> RepositoryAnalysis:
        """Create structured repository analysis from file paths and stats"""
        unique_files = list(set(path for path in file_paths if path))

        # File type frequency
        file_types: dict[str, int] = {}
        for path in unique_files:
            if "." in path:
                ext = path.rsplit(".", 1)[-1].lower()
                if ext:
                    file_types[ext] = file_types.get(ext, 0) + 1

        # Map common extensions to language names
        code_extensions = {
            "py": "Python",
            "js": "JavaScript",
            "ts": "TypeScript",
            "java": "Java",
            "cpp": "C++",
            "cc": "C++",
            "cxx": "C++",
            "c++": "C++",
            "c": "C",
            "h": "C/C++",
            "hpp": "C++",
            "hh": "C++",
            "hxx": "C++",
            "go": "Go",
            "rs": "Rust",
            "rb": "Ruby",
            "php": "PHP",
            "cs": "C#",
        }
        languages: list[str] = []
        for ext, _count in file_types.items():
            if ext in code_extensions:
                languages.append(code_extensions[ext])

        # README detection
        readme_files = [p for p in unique_files if "readme" in p.lower()]
        readme_sample = ""
        if readme_content and readme_content != "No README file found":
            readme_sample = readme_content[:1000]

        # Rough project complexity heuristic
        complexity = "simple"
        if len(unique_files) > 50 or len(languages) > 2:
            complexity = "moderate"
        if len(unique_files) > 200 or len(languages) > 3:
            complexity = "complex"

        return RepositoryAnalysis(
            total_files=len(unique_files),
            programming_languages=languages[:5],
            primary_language=languages[0] if languages else "Unknown",
            has_readme=bool(readme_files),
            readme_content=readme_sample,
            total_symbols=index_stats.get("code_analysis", {}).get("total_symbols", 0),
            file_types=file_types,
            project_complexity=complexity,
            key_directories=self._extract_key_directories(unique_files),
            main_modules=self._extract_main_modules(unique_files),
            documentation_files=[
                p for p in unique_files if any(p.lower().endswith(ext) for ext in [".md", ".rst", ".txt"])
            ],
            configuration_files=[
                p
                for p in unique_files
                if any(p.lower().endswith(ext) for ext in [".json", ".yaml", ".yml", ".toml", ".ini", ".cfg"])
            ],
            test_files=[p for p in unique_files if "test" in p.lower() or "spec" in p.lower()],
        )

    def _extract_key_directories(self, file_paths: list[str]) -> list[str]:
        """Extract key directory names from file paths"""
        directories = set()
        for path in file_paths:
            parts = path.split("/")
            if len(parts) > 1:
                # Add top-level directories
                directories.add(parts[0])
                # Add second-level directories for better context
                if len(parts) > 2:
                    directories.add(f"{parts[0]}/{parts[1]}")

        return sorted(list(directories))[:20]  # Top 20 directories

    def _extract_main_modules(self, file_paths: list[str]) -> list[str]:
        """Extract main module names from file paths"""
        modules = set()
        for path in file_paths:
            if path.endswith(".py"):
                # Python modules
                module_name = path.replace("/", ".").replace(".py", "")
                if not module_name.startswith(".") and "test" not in module_name:
                    modules.add(module_name)
            elif any(path.endswith(ext) for ext in [".js", ".ts"]):
                # JavaScript/TypeScript modules
                if "/" in path:
                    module_name = path.split("/")[-1].split(".")[0]
                    modules.add(module_name)

        return sorted(list(modules))[:50]  # Top 50 modules

    def _get_env_int(self, name: str, default: int) -> int:
        """Read an int env var with safe fallback."""
        value = os.getenv(name)
        if value is None or value == "":
            return default
        try:
            return int(value)
        except ValueError:
            logger.warning(f"Invalid int for {name}='{value}', using default {default}")
            return default

    def _split_env_paths(self, name: str) -> list[str]:
        """Split comma-separated env paths into a list."""
        raw = os.getenv(name, "").strip()
        if not raw:
            return []
        return [item.strip() for item in raw.split(",") if item.strip()]

    def _should_use_deepagents_structure_planner(
        self,
        repo_context: str | None,
        repository_files: list[str],
    ) -> bool:
        """Decide whether to use deepagents for structure planning."""
        mode = os.getenv("WIKIS_STRUCTURE_PLANNER", "auto").strip().lower()
        if mode in {"deepagents", "agentic"}:
            return True
        if mode in {"llm", "classic", "standard"}:
            return False

        file_threshold = self._get_env_int("WIKIS_DEEPAGENTS_FILE_THRESHOLD", 2000)
        token_threshold = self._get_env_int("WIKIS_DEEPAGENTS_REPOCTX_TOKENS", 8000)

        if repository_files and len(repository_files) >= file_threshold:
            return True

        if repo_context:
            token_counter = get_token_counter()
            try:
                repo_tokens = token_counter.count(repo_context)
            except Exception:
                repo_tokens = max(len(repo_context) // 4, 1)
            if repo_tokens >= token_threshold:
                return True

        return False

    def _generate_wiki_structure_cluster(
        self,
        state: WikiState,
        config: RunnableConfig,
        exclude_tests: bool = False,
    ) -> dict[str, Any] | None:
        """Generate wiki structure using the cluster-based planner.

        Returns the standard structure dict on success, or ``None`` to signal
        that the caller should fall back to the agent-based planner.
        """
        from ..graph_clustering import run_phase3
        from ..wiki_structure_planner.cluster_planner import ClusterStructurePlanner

        code_graph = (
            getattr(self.retriever_stack, "relationship_graph", None)
            or getattr(self.indexer, "relationship_graph", None)
        )
        if code_graph is None or code_graph.number_of_nodes() == 0:
            logger.warning("Cluster planner: no code graph available — falling back")
            return None

        # ── Populate unified DB from code graph (for expansion & topology) ──
        self._ensure_unified_db(code_graph)

        if self._cluster_db is None:
            logger.warning("Cluster planner: no unified DB available — falling back")
            return None

        # ── Ensure code_graph has Phase 2 enrichment ──────────────────
        # When the indexer pre-built the DB, Phase 2 (orphan resolution,
        # doc edges, component bridging, edge weighting) was applied to
        # the indexer's copy of the graph and persisted to the DB.
        # However, the in-memory code_graph we hold here is the *raw*
        # graph from the retriever stack — it has none of those
        # enrichments.  Leiden clustering on the un-enriched graph
        # produces fragmented/disconnected results because ~55% of
        # symbol nodes are orphans and all edges have weight=1.0.
        #
        # Fix: reconstruct the graph from the DB (which has the full
        # enriched edge set with calibrated weights) and use THAT for
        # clustering.
        clustering_graph = code_graph  # default: use as-is
        if not code_graph.graph.get("phase2_enriched"):
            try:
                phase2_done = self._cluster_db.get_meta("phase2_completed")
                if phase2_done:
                    enriched_g = self._cluster_db.to_networkx()
                    if enriched_g.number_of_nodes() > 0:
                        logger.info(
                            "[CLUSTER] Using Phase 2-enriched graph from DB: "
                            "%d nodes, %d edges (raw graph had %d edges)",
                            enriched_g.number_of_nodes(),
                            enriched_g.number_of_edges(),
                            code_graph.number_of_edges(),
                        )
                        clustering_graph = enriched_g
                    else:
                        logger.warning(
                            "[CLUSTER] DB to_networkx() returned empty graph — "
                            "using raw code_graph"
                        )
                else:
                    logger.debug(
                        "[CLUSTER] Phase 2 not completed in DB — "
                        "using raw code_graph for clustering"
                    )
            except Exception as exc:
                logger.warning(
                    "[CLUSTER] Failed to load enriched graph from DB: %s — "
                    "using raw code_graph",
                    exc,
                )

        if self.progress_callback:
            try:
                self.progress_callback("planning", 0.29, "Running graph clustering...")
            except Exception:
                pass

        # ── Run Phase 3: clustering persists directly to DB ──
        try:
            from ..feature_flags import FeatureFlags

            # planner_type="cluster" means all Leiden features are ON.
            # exclude_tests comes from the UI toggle.
            flags = FeatureFlags(
                hierarchical_leiden=True,
                capability_validation=True,
                smart_expansion=True,
                coverage_ledger=True,
                language_hints=True,
                exclude_tests=exclude_tests,
            )
            phase3_stats = run_phase3(
                db=self._cluster_db,
                G=clustering_graph,
                feature_flags=flags,
            )
            logger.info("[CLUSTER] Phase 3 complete: %s", phase3_stats)
        except Exception as exc:
            logger.warning("Cluster planner: Phase 3 failed: %s — falling back", exc)
            return None

        # Verify clustering produced results
        all_clusters = self._cluster_db.get_all_clusters()
        if not all_clusters:
            logger.warning("Cluster planner: clustering produced 0 sections — falling back")
            return None

        if self.progress_callback:
            try:
                n_sections = len(all_clusters)
                n_pages = sum(len(micros) for micros in all_clusters.values())
                self.progress_callback(
                    "planning", 0.32,
                    f"Detected {n_sections} sections, {n_pages} pages — naming with LLM...",
                )
            except Exception:
                pass

        planner = ClusterStructurePlanner(
            db=self._cluster_db,
            llm=self.llm_low,
            wiki_title=None,  # Auto-derive from repo metadata
        )

        response = planner.plan_structure()
        total_pages = sum(len(s.pages) for s in response.sections)

        logger.info(
            "ClusterStructurePlanner: %d sections, %d pages",
            len(response.sections),
            total_pages,
        )

        if self.progress_callback:
            try:
                section_names = ", ".join(s.title for s in response.sections[:5])
                suffix = f"... and {len(response.sections) - 5} more" if len(response.sections) > 5 else ""
                self.progress_callback(
                    "planning", 0.38,
                    f"Structure planned — {len(response.sections)} sections, {total_pages} pages: {section_names}{suffix}",
                )
            except Exception:
                pass

        if this and getattr(this, "module", None):
            this.module.invocation_thinking(
                f"I am on phase structure_planning\n"
                f"Structure designed: {len(response.sections)} sections / {total_pages} pages (cluster planner)\n"
                f"Reasoning: Deterministic clustering from code graph with LLM naming.\n"
                f"Next: Parallelize page drafting with focused context windows."
            )

        return {
            "wiki_structure_spec": response,
            "structure_planning_complete": True,
            "current_phase": "structure_complete",
        }

    # ── Cluster-bounded page expansion ──────────────────────────────

    def _try_cluster_expansion(
        self, page_spec: PageSpec,
    ) -> dict[str, Any] | None:
        """Attempt cluster-bounded expansion for a page.

        Returns the formatted context dict (same shape as
        ``_format_simple_context``) when the page was produced by the
        cluster planner.  Returns ``None`` to signal the caller should
        fall through to the legacy graph-based path.
        """
        rationale = getattr(page_spec, "rationale", "") or ""
        if "graph clustering" not in rationale:
            return None

        metadata = getattr(page_spec, "metadata", None) or {}
        macro_id = metadata.get("section_id")
        micro_id = metadata.get("page_id")
        cluster_node_ids = metadata.get("cluster_node_ids")

        if macro_id is None:
            macro_id = self._extract_macro_id(rationale)

        db = self._open_cluster_db()
        if db is None:
            logger.warning(
                "[CLUSTER_EXPANSION] No unified DB available — "
                "falling back to legacy expansion"
            )
            return None

        target_symbols = getattr(page_spec, "target_symbols", []) or []
        if not target_symbols:
            return None

        try:
            from ..cluster_expansion import expand_for_page

            docs = expand_for_page(
                db=db,
                page_symbols=target_symbols,
                macro_id=macro_id,
                micro_id=micro_id,
                cluster_node_ids=cluster_node_ids,
                token_budget=CONTEXT_TOKEN_BUDGET,
            )
        except Exception as exc:
            logger.warning(
                "[CLUSTER_EXPANSION] Failed for page '%s': %s — "
                "falling back to legacy",
                page_spec.page_name, exc,
            )
            return None

        if not docs:
            logger.info(
                "[CLUSTER_EXPANSION] No docs for page '%s' — "
                "falling back to legacy",
                page_spec.page_name,
            )
            return None

        token_counter = get_token_counter()
        total_tokens = token_counter.count_documents(docs)

        if total_tokens <= CONTEXT_TOKEN_BUDGET:
            logger.info(
                "CLUSTER EXPANSION: %d docs for '%s' "
                "(%d/%d tokens, macro=%s)",
                len(docs), page_spec.page_name,
                total_tokens, CONTEXT_TOKEN_BUDGET, macro_id,
            )
            return self._format_simple_context(docs, page_spec)
        else:
            logger.info(
                "[CLUSTER_EXPANSION] Budget exceeded (%d/%d) — truncating",
                total_tokens, CONTEXT_TOKEN_BUDGET,
            )
            truncated = self._ranked_truncation(
                docs, page_spec, CONTEXT_TOKEN_BUDGET,
            )
            return self._format_simple_context(truncated, page_spec)

    def _ensure_unified_db(self, code_graph) -> None:
        """Locate / open the unified DB that was written during indexing.

        The indexer writes a ``{cache_key}.wiki.db`` to the cache directory
        with graph, embeddings, Phase 2 topology, and Phase 3 clustering
        already populated.  This method simply opens it.

        Falls back to creating one in-memory from the code graph if the
        pre-built DB cannot be found (e.g. unit tests, local dev).
        """
        if self._cluster_db is not None:
            return

        from ..storage import open_storage, repo_id_from_path

        # Try to find the pre-built DB from the indexer
        db_path = self._find_unified_db()
        if db_path:
            try:
                repo_id = repo_id_from_path(db_path) if db_path else None
                db = open_storage(repo_id=repo_id, db_path=db_path)
                self._cluster_db = db
                self._cluster_db_path = db_path
                from ..storage.text_index import StorageTextIndex

                self.graph_text_index = StorageTextIndex(db)
                stats = db.stats()
                backend_name = type(db).__name__
                logger.info(
                    "[UNIFIED_DB] Opened storage: %s (repo_id=%s, %d nodes)",
                    backend_name,
                    repo_id,
                    stats["node_count"],
                )
                return
            except Exception as exc:
                logger.warning("[UNIFIED_DB] Failed to open pre-built DB: %s — will create", exc)

        # Fallback: create from the NX graph (tests / local dev without indexer)
        import tempfile

        source_dir = (
            getattr(self.indexer, "source_dir", None)
            or getattr(self.indexer, "repo_path", None)
        )
        if source_dir:
            wikis_dir = os.path.join(source_dir, ".wikis")
            os.makedirs(wikis_dir, exist_ok=True)
            db_path = os.path.join(wikis_dir, "unified_wiki.db")
        else:
            tmp = tempfile.mkdtemp(prefix="wikis_udb_")
            db_path = os.path.join(tmp, "unified_wiki.db")

        try:
            repo_id = repo_id_from_path(db_path)
            db = open_storage(repo_id=repo_id, db_path=db_path)
            db.from_networkx(code_graph)
            self._cluster_db = db
            self._cluster_db_path = db_path
            from ..storage.text_index import StorageTextIndex

            self.graph_text_index = StorageTextIndex(db)
            logger.info(
                "[UNIFIED_DB] Created from code graph: %d nodes, %d edges → %s",
                code_graph.number_of_nodes(),
                code_graph.number_of_edges(),
                db_path,
            )

            # Phase 2 enrichment
            from ..config import get_settings as _get_settings

            _settings = _get_settings()
            if _settings.graph_enrichment_enabled:
                try:
                    from ..graph_topology import run_phase2

                    p2 = run_phase2(
                        db=db,
                        G=code_graph,
                        z_threshold=_settings.hub_z_threshold,
                        vec_distance_threshold=_settings.vec_distance_threshold,
                    )
                    logger.info(
                        "[UNIFIED_DB] Phase 2 enrichment: %d hubs, %d edges persisted",
                        p2.get("hubs", {}).get("count", 0),
                        p2.get("persisted_edges", 0),
                    )
                except Exception as p2_exc:
                    logger.warning("[UNIFIED_DB] Phase 2 enrichment failed: %s", p2_exc)

        except Exception as exc:
            logger.warning("[UNIFIED_DB] Failed to populate: %s", exc)

    def _open_cluster_db(self):
        """Open (or return cached) unified DB for cluster expansion."""
        if self._cluster_db is not None:
            return self._cluster_db

        db_path = self._cluster_db_path or self._find_unified_db()
        if not db_path:
            return None

        try:
            from ..storage import open_storage, repo_id_from_path

            repo_id = repo_id_from_path(db_path)
            self._cluster_db = open_storage(repo_id=repo_id, db_path=db_path)
            self._cluster_db_path = db_path
            logger.info("[CLUSTER_EXPANSION] Opened unified DB: %s", db_path)
            return self._cluster_db
        except Exception as exc:
            logger.warning("[CLUSTER_EXPANSION] Failed to open DB: %s", exc)
            return None

    def _close_cluster_db(self) -> None:
        """Close the cached unified DB connection if open."""
        if self._cluster_db is not None:
            try:
                self._cluster_db.close()
            except Exception:
                pass
            self._cluster_db = None

    def _find_unified_db(self) -> str | None:
        """Locate the unified wiki DB file (or cache key for postgres).

        Checks:
        1. Indexer's ``_unified_db_path`` attribute (set during indexing)
        2. ``{cache_dir}/*.wiki.db`` via cache_index.json
        3. Legacy ``{source_dir}/.wikis/unified_wiki.db`` location

        For postgres backend, we still return a db_path-like string
        so the caller can extract the repo_id/cache_key from it.
        """
        from ..storage import is_postgres_backend

        _pg = is_postgres_backend()

        # 1. Direct path from indexer
        indexer_db = getattr(self.indexer, "_unified_db_path", None)
        if indexer_db:
            if _pg or os.path.isfile(indexer_db):
                return indexer_db

        # 2. Look in cache directory via cache_index.json
        cache_dir = None
        graph_mgr = getattr(self.indexer, "graph_manager", None)
        if graph_mgr:
            cache_dir = str(getattr(graph_mgr, "cache_dir", None) or "")
        if not cache_dir:
            from ..config import get_settings as _get_settings
            try:
                _settings = _get_settings()
                cache_dir = _settings.cache_dir
            except Exception:
                pass

        if cache_dir:
            import json
            index_file = os.path.join(cache_dir, "cache_index.json")
            if os.path.isfile(index_file):
                try:
                    with open(index_file) as f:
                        index = json.load(f)
                    udb_section = index.get("unified_db", {})
                    for _repo_id, cache_key in udb_section.items():
                        db_path = os.path.join(cache_dir, f"{cache_key}.wiki.db")
                        if _pg or os.path.isfile(db_path):
                            return db_path
                except Exception:
                    pass

            if not _pg:
                # Glob fallback (SQLite only)
                import glob as _glob
                wiki_dbs = _glob.glob(os.path.join(cache_dir, "*.wiki.db"))
                if wiki_dbs:
                    wiki_dbs.sort(key=os.path.getmtime, reverse=True)
                    return wiki_dbs[0]

        if not _pg:
            # 3. Legacy location in source dir (SQLite only)
            source_dir = getattr(self.indexer, "source_dir", None) or getattr(self.indexer, "repo_path", None)
            if source_dir:
                candidates = [
                    os.path.join(source_dir, ".wikis", "unified_wiki.db"),
                    os.path.join(source_dir, "unified_wiki.db"),
                ]
                for path in candidates:
                    if os.path.isfile(path):
                        return path

        return None

    @staticmethod
    def _extract_macro_id(rationale: str) -> int | None:
        """Extract macro cluster ID from a page's rationale string."""
        import re as _re
        m = _re.search(r"macro[_\s]*(?:id|cluster)[:\s=]*(\d+)", rationale, _re.IGNORECASE)
        if m:
            return int(m.group(1))
        m = _re.search(r"section[_\s]*(?:id)[:\s=]*(\d+)", rationale, _re.IGNORECASE)
        if m:
            return int(m.group(1))
        return None

    def _build_repo_profile(self, repository_files: list[str]) -> dict[str, Any]:
        """Build a lean repository profile for structure planning.

        Also detects "documentation clusters" - directories where documentation
        files (md, rst, yaml, toml) outnumber code files. These need explicit
        coverage to avoid garbage output for doc-only repos.
        """
        total_files = len(repository_files)
        dir_counts: dict[str, int] = {}
        extensions: Counter = Counter()

        # Track files per directory for doc cluster detection
        dir_files: dict[str, list[str]] = {}

        for path in repository_files:
            if not path:
                continue
            if "/" in path:
                top = path.split("/", 1)[0]
                dir_counts[top] = dir_counts.get(top, 0) + 1

                # Track at depth 1-6 for doc cluster detection (covers deep doc dirs
                # like project/modules/core/docs/ and root-level doc dirs like docs/)
                parts = path.split("/")
                for depth in range(1, min(len(parts), 7)):  # Depth 1 through 6
                    dir_path = "/".join(parts[:depth])
                    if dir_path not in dir_files:
                        dir_files[dir_path] = []
                    dir_files[dir_path].append(path)
            else:
                dir_counts["/"] = dir_counts.get("/", 0) + 1
            _, ext = os.path.splitext(path)
            if ext:
                extensions[ext.lstrip(".").lower()] += 1

        top_dirs = sorted(dir_counts.items(), key=lambda x: (-x[1], x[0]))[:25]
        top_exts = extensions.most_common(25)

        # Detect documentation clusters
        doc_clusters = self._detect_doc_clusters(dir_files)

        return {
            "total_files": total_files,
            "top_level_dirs": top_dirs,
            "top_level_dir_count": len(dir_counts),
            "top_extensions": top_exts,
            "doc_clusters": doc_clusters,
        }

    def _detect_doc_clusters(self, dir_files: dict[str, list[str]]) -> list[dict[str, Any]]:
        """Detect directories that are primarily documentation.

        A doc cluster is a directory where:
        - Documentation files > 50% of files
        - Has at least 2 files
        - Parent directory has multiple such children (like config/katas/*)

        Returns list of clusters with metadata for the LLM.
        """
        # Use single source of truth from constants.py
        DOC_EXTENSIONS = DOCUMENTATION_EXTENSIONS_SET
        CODE_EXTENSIONS = {
            ".py",
            ".js",
            ".ts",
            ".java",
            ".go",
            ".rs",
            ".cpp",
            ".cc",
            ".cxx",
            ".c++",
            ".c",
            ".h",
            ".hpp",
            ".hh",
            ".hxx",
            ".rb",
            ".php",
        }

        doc_clusters = []
        parent_children: dict[str, list[dict]] = {}  # Group by parent for batch detection

        for dir_path, files in dir_files.items():
            if len(files) < 2:
                continue

            # Count doc vs code files
            doc_count = 0
            code_count = 0
            for f in files:
                ext = os.path.splitext(f)[1].lower()
                if ext in DOC_EXTENSIONS:
                    doc_count += 1
                elif ext in CODE_EXTENSIONS:
                    code_count += 1

            total = doc_count + code_count
            if total == 0:
                continue

            doc_ratio = doc_count / total

            # Is this primarily documentation?
            if doc_ratio >= 0.5 and code_count == 0:
                # Get parent directory
                parent = "/".join(dir_path.split("/")[:-1]) if "/" in dir_path else ""
                dir_name = dir_path.split("/")[-1]

                cluster_info = {
                    "path": dir_path,
                    "name": dir_name,
                    "file_count": len(files),
                    "doc_types": list(set(os.path.splitext(f)[1].lower() for f in files if os.path.splitext(f)[1])),
                }

                if parent:
                    if parent not in parent_children:
                        parent_children[parent] = []
                    parent_children[parent].append(cluster_info)
                else:
                    doc_clusters.append(cluster_info)

        # For parents with multiple doc children, group them
        for parent, children in parent_children.items():
            if len(children) >= 2:
                # This is a doc collection (like config/katas/* with 15 tutorials)
                doc_clusters.append(
                    {
                        "path": parent,
                        "name": parent.split("/")[-1] if parent else "root",
                        "type": "collection",
                        "children": children,
                        "child_count": len(children),
                        "total_files": sum(c["file_count"] for c in children),
                    }
                )
            else:
                # Single doc directory
                doc_clusters.extend(children)

        # Sort by child count (collections first) then by path
        doc_clusters.sort(key=lambda x: (-x.get("child_count", 0), x["path"]))

        if doc_clusters:
            logger.info(f"[REPO_PROFILE] Detected {len(doc_clusters)} documentation cluster(s)")

        return doc_clusters[:20]  # Limit to top 20 for prompt size

    def _get_graph_top_nodes(self, graph: Any | None, limit: int = 20) -> list[dict[str, Any]]:
        """Return top-degree nodes for lightweight graph profiling."""
        if not graph:
            return []
        try:
            degree_iter = graph.degree()
            top_nodes = heapq.nlargest(limit, degree_iter, key=lambda x: x[1])
        except Exception as e:
            logger.warning(f"Failed to compute graph top nodes: {e}")
            return []

        results = []
        for node_id, degree in top_nodes:
            node_data = {}
            if hasattr(graph, "nodes"):
                try:
                    node_data = graph.nodes[node_id]
                except Exception:
                    node_data = {}
            results.append(
                {
                    "node_id": node_id,
                    "symbol_name": node_data.get("symbol_name") or node_data.get("name") or str(node_id),
                    "file_path": node_data.get("file_path", ""),
                    "language": node_data.get("language", ""),
                    "degree": degree,
                }
            )
        return results

    def _generate_wiki_structure_with_deepagents(
        self,
        repository_tree: str,
        readme_content: str,
        repo_analysis: Any,
        repo_context: str,
        repository_files: list[str],
        target_audience: str,
        wiki_type: str,
        config: RunnableConfig,
    ) -> WikiStructureSpec:
        """Generate wiki structure using the lean deepagents-based structure planner.

        This uses the new WikiStructurePlannerEngine which:
        - Uses FilesystemMiddleware for repository exploration (ls, glob, grep, read_file)
        - Does NOT embed large context in prompts
        - Lets the agent discover structure via tools
        - Returns structured WikiStructureSpec JSON
        """
        # Get repository root path
        repo_root_getter = getattr(self.indexer, "get_repo_root", None)
        repo_root = repo_root_getter() if callable(repo_root_getter) else None

        if not repo_root or not os.path.isdir(repo_root):
            raise ValueError(f"Repository root not available or invalid: {repo_root}")

        try:
            from ..wiki_structure_planner import StructurePlannerConfig, WikiStructurePlannerEngine
        except ImportError as e:
            raise RuntimeError(f"WikiStructurePlannerEngine unavailable: {e}") from e

        # Calculate page budget
        repository_files = repository_files or self._get_repository_file_paths()
        file_count = len(repository_files)
        repo_profile = self._build_repo_profile(repository_files)

        # --- Auto-detect effective depth for deep monorepo structures ---
        # Java: src/main/java/org/springframework/boot/ (depth 6)
        # Go mono: src/go/rpk/pkg/adminapi/ (depth 5+)
        # Rust workspace: src/transform-sdk/rust/core-sys/src/ (depth 5+)
        from ..wiki_structure_planner.structure_tools import detect_effective_depth

        effective_depth = detect_effective_depth(repository_files)

        # Get graph stats for budget calculation
        graph = None
        if getattr(self.retriever_stack, "relationship_graph", None) is not None:
            graph = self.retriever_stack.relationship_graph
        elif getattr(self.indexer, "relationship_graph", None) is not None:
            graph = self.indexer.relationship_graph
        graph_nodes = graph.number_of_nodes() if graph else 0

        page_budget_env = self._get_env_int("WIKIS_PAGE_BUDGET", 0)
        if page_budget_env > 0:
            page_budget = page_budget_env
        else:
            # Broad-page budget: fewer pages with 10-15+ symbols each
            # Priority: comprehensive broad pages > many narrow pages
            # Each page should cover a full capability with all related symbols
            min_pages = self._get_env_int("WIKIS_PAGE_BUDGET_MIN", 8)
            default_max_pages = 200 if file_count >= 5000 or graph_nodes >= 100000 else 80
            max_pages = self._get_env_int("WIKIS_PAGE_BUDGET_MAX", default_max_pages)
            files_per_page = self._get_env_int("WIKIS_PAGE_FILES_PER_PAGE", 100)  # Fewer pages per file (broad pages)
            dirs_per_page = self._get_env_int("WIKIS_PAGE_DIRS_PER_PAGE", 4)  # Fewer pages per dir (broad pages)
            graph_divisor = self._get_env_int(
                "WIKIS_PAGE_GRAPH_NODES_DIVISOR", 5
            )  # Lower graph influence (broad pages)

            file_factor = (file_count / max(files_per_page, 1)) if file_count else 0
            dir_count = repo_profile.get("top_level_dir_count", 0)
            dir_factor = (dir_count / max(dirs_per_page, 1)) if dir_count else 0
            # Use sqrt of graph nodes for proportional scaling (135K nodes → ~367 → /5 = 73)
            graph_factor = (math.sqrt(graph_nodes) / max(graph_divisor, 1)) if graph_nodes else 0

            raw_budget = min_pages + file_factor + dir_factor + graph_factor
            page_budget = int(round(raw_budget))
            page_budget = max(min_pages, min(page_budget, max_pages))

            logger.info(
                "[STRUCTURE][BUDGET] dynamic=%s (files=%s, top_dirs=%s, graph_nodes=%s)",
                page_budget,
                file_count,
                dir_count,
                graph_nodes,
            )

        # Calculate dynamic max_iterations based on workload
        # Formula: page_budget * 1.5 (each page needs ~1.5 tool calls) + exploration overhead
        # For Redpanda (budget=150, dirs=453): 150*1.5 + sqrt(453)*2 = 225 + 42 = 267 → capped at 200
        # Depth increase adds iteration headroom: deeper trees need more query_graph calls
        dir_count_for_iterations = repo_profile.get("top_level_dir_count", 0)
        base_iterations = page_budget * 1.5
        exploration_overhead = math.sqrt(dir_count_for_iterations) * 2
        # Extra budget when depth is auto-increased (more dirs to explore)
        depth_bonus = (effective_depth - 5) * 10 if effective_depth > 5 else 0
        calculated_iterations = int(base_iterations + exploration_overhead + depth_bonus)
        min_iterations = 50  # Minimum for any repo
        default_iterations_cap = 600 if page_budget >= 200 or file_count >= 5000 or graph_nodes >= 100000 else 200
        max_iterations_cap = self._get_env_int("WIKIS_DEEPAGENTS_MAX_ITERATIONS_CAP", default_iterations_cap)
        dynamic_iterations = max(min_iterations, min(calculated_iterations, max_iterations_cap))

        # Allow env override, but use dynamic as default
        max_iterations_env = self._get_env_int("WIKIS_DEEPAGENTS_MAX_ITERATIONS", 0)
        max_iterations = max_iterations_env if max_iterations_env > 0 else dynamic_iterations

        logger.info(
            "[STRUCTURE][ITERATIONS] dynamic=%s (page_budget=%s, dirs=%s, calculated=%s)",
            max_iterations,
            page_budget,
            dir_count_for_iterations,
            calculated_iterations,
        )

        # Configure the structure planner engine with dynamic and generous limits
        is_large_repo = file_count >= 5000 or graph_nodes >= 100000 or dir_count_for_iterations >= 12
        max_ls_default = 40 if is_large_repo else 25
        max_glob_default = 25 if is_large_repo else 15
        max_read_default = 40 if is_large_repo else 25
        max_grep_default = 25 if is_large_repo else 15

        planner_config = StructurePlannerConfig(
            page_budget=page_budget,
            target_audience=target_audience,
            wiki_type=wiki_type,
            max_iterations=max_iterations,
            effective_depth=effective_depth,
            repository_file_count=file_count,
            max_ls_calls=self._get_env_int("WIKIS_DEEPAGENTS_MAX_LS", max_ls_default),
            max_glob_calls=self._get_env_int("WIKIS_DEEPAGENTS_MAX_GLOB", max_glob_default),
            max_read_calls=self._get_env_int("WIKIS_DEEPAGENTS_MAX_READ", max_read_default),
            max_grep_calls=self._get_env_int("WIKIS_DEEPAGENTS_MAX_GREP", max_grep_default),
            doc_clusters=repo_profile.get("doc_clusters", []),
        )

        # Create and run the engine
        # Pass the code graph for symbol-aware complexity analysis
        engine = WikiStructurePlannerEngine(
            repo_root=repo_root,
            llm_client=self.llm,
            config=planner_config,
            code_graph=graph,
            graph_text_index=self.graph_text_index,
        )

        # Get repository name from URL
        repo_name = ""
        if self.repository_url:
            repo_name = self.repository_url.rstrip("/").split("/")[-1].replace(".git", "")

        log_stream = os.getenv("WIKIS_DEEPAGENTS_LOG_STREAM", "0") == "1"

        def stream_callback(event):
            if log_stream:
                logger.debug(f"[STRUCTURE_PLANNER] Event: {type(event)}")

        logger.info(f"[STRUCTURE][DEEPAGENTS] Starting lean structure planner for {repo_name}")
        start_time = time.time()

        data = engine.plan_structure(repo_name=repo_name, stream_callback=stream_callback if log_stream else None)

        elapsed = time.time() - start_time
        logger.info(f"[STRUCTURE][DEEPAGENTS] Completed in {elapsed:.2f}s, tool_calls={len(engine.tool_calls)}")

        # Normalize and validate
        data = self._normalize_structure_payload(data)
        structure = WikiStructureSpec.model_validate(data)

        # Apply coverage check if enabled
        if os.getenv("WIKIS_DEEPAGENTS_COVERAGE_CHECK", "1") != "0":
            repo_outline = self._create_repository_tree(repository_files, max_files_per_dir=10)
            structure = self._apply_deepagents_coverage_check(
                structure=structure,
                repository_files=repository_files,
                repo_outline=repo_outline,
                page_budget=page_budget,
                target_audience=target_audience,
                wiki_type=wiki_type,
                config=config,
            )

        return structure

    # -----------------------------------------------------------------
    # Graph-First Structure Planning
    # -----------------------------------------------------------------

    def _generate_wiki_structure_graph_first(
        self,
        repository_files: list[str],
        target_audience: str,
        wiki_type: str,
        config: RunnableConfig,
    ) -> WikiStructureSpec:
        """Generate wiki structure using deterministic graph-first approach.

        Phase 1 – skeleton (zero LLM, ~15 s for large repos):
          Walk the code graph once, cluster directories by weighted
          relationships (Louvain), normalise sizes.

        Phase 2 – LLM refinement (1-3 calls, ~30 s):
          LLM receives only cluster *summaries*, outputs names/sections/
          descriptions.  Symbols injected programmatically → 100 % coverage.

        Feature-flagged by ``WIKIS_USE_GRAPH_FIRST=1``.
        """
        from ..wiki_structure_planner.structure_engine import (
            StructurePlannerConfig,
            WikiStructurePlannerEngine,
        )
        from ..wiki_structure_planner.structure_tools import detect_effective_depth

        # Resolve repo root the same way as deepagents path
        repo_root_getter = getattr(self.indexer, "get_repo_root", None)
        repo_root = repo_root_getter() if callable(repo_root_getter) else None
        if not repo_root or not os.path.isdir(repo_root):
            raise ValueError(f"Repository root not available or invalid: {repo_root}")

        repo_name = os.path.basename(repo_root.rstrip("/"))

        # Resolve code graph the same way as _generate_wiki_structure_with_deepagents
        code_graph = getattr(self.retriever_stack, "relationship_graph", None) or getattr(
            self.indexer, "relationship_graph", None
        )

        # ── compute page budget (same formula as deepagents path) ─────
        file_count = len(repository_files)
        file_factor = max(8, min(file_count // 25, 120))
        unique_dirs: set = set()
        for p in repository_files:
            parts = p.split("/")
            for d in range(1, len(parts)):
                unique_dirs.add("/".join(parts[:d]))
        dir_count = len(unique_dirs)
        dir_factor = max(0, min(dir_count // 10, 80))
        graph_factor = 0
        if code_graph:
            graph_nodes = code_graph.number_of_nodes()
            graph_factor = max(0, min(graph_nodes // 100, 60))
        raw_budget = (file_factor + dir_factor + graph_factor) // 3
        page_budget_cap = int(os.getenv("WIKIS_PAGE_BUDGET_CAP", "200"))
        page_budget = max(8, min(raw_budget, page_budget_cap))

        effective_depth = detect_effective_depth(repository_files)

        logger.info(
            "[GRAPH_FIRST] repo=%s budget=%d depth=%d files=%d dirs=%d graph_nodes=%d",
            repo_name,
            page_budget,
            effective_depth,
            file_count,
            dir_count,
            code_graph.number_of_nodes() if code_graph else 0,
        )

        planner_config = StructurePlannerConfig(
            page_budget=page_budget,
            target_audience=target_audience,
            wiki_type=wiki_type,
            effective_depth=effective_depth,
            repository_file_count=file_count,
        )

        engine = WikiStructurePlannerEngine(
            repo_root=repo_root,
            llm_client=self.llm,
            config=planner_config,
            code_graph=code_graph,
            graph_text_index=self.graph_text_index,
        )

        data = engine.plan_structure_graph_first(
            repo_name=repo_name,
            repository_files=repository_files,
        )

        # Normalise and validate
        data = self._normalize_structure_payload(data)
        structure = WikiStructureSpec.model_validate(data)

        return structure

    def _apply_deepagents_coverage_check(
        self,
        structure: WikiStructureSpec,
        repository_files: list[str],
        repo_outline: str,
        page_budget: int,
        target_audience: str,
        wiki_type: str,
        config: RunnableConfig,
    ) -> WikiStructureSpec:
        """Ensure directory coverage up to a configurable depth for deepagents structure plans.

        Handles deep monorepo structures like Spring Framework, enterprise repos.
        Uses auto-detected depth from detect_effective_depth() when available.
        """
        if not repository_files:
            return structure

        # Use auto-detected depth instead of static default
        from ..wiki_structure_planner.structure_tools import detect_effective_depth

        max_depth = detect_effective_depth(repository_files)

        # Collect directories at all depths up to MAX_DEPTH.
        # Use the SAME filtering as the structure planner to avoid
        # false "missing" dirs that the planner deliberately skipped.
        dirs_by_depth = {d: set() for d in range(1, max_depth + 1)}

        for path in repository_files:
            if not path:
                continue
            parts = path.split("/")
            for depth in range(1, min(len(parts), max_depth + 1)):
                component = parts[depth - 1]
                # Skip dirs in COVERAGE_IGNORE_DIRS (exact match)
                if component in COVERAGE_IGNORE_DIRS:
                    continue
                # Skip dot-dirs not in the whitelist
                if component.startswith(".") and component not in DOT_DIR_WHITELIST:
                    continue
                dir_path = "/".join(parts[:depth])
                dirs_by_depth[depth].add(dir_path)

        # Convert to sorted lists
        all_expected = []
        for depth in range(1, max_depth + 1):
            all_expected.extend(sorted(dirs_by_depth[depth]))

        if not all_expected:
            return structure

        # ---------------------------------------------------------------
        # Coverage detection: precise prefix matching on target_folders
        # ---------------------------------------------------------------
        covered_dirs = set()
        for section in structure.sections:
            for page in section.pages:
                # Primary: exact prefix matching on target_folders
                for tf in page.target_folders or []:
                    tf_norm = tf.strip("/").lower()
                    for dir_path in all_expected:
                        dp_lower = dir_path.lower()
                        if (
                            dp_lower == tf_norm
                            or dp_lower.startswith(tf_norm + "/")
                            or tf_norm.startswith(dp_lower + "/")
                        ):
                            covered_dirs.add(dir_path)

                # Secondary: check page_name / description for full dir paths
                hay = " ".join(
                    p
                    for p in [
                        page.page_name,
                        page.description,
                        page.content_focus,
                    ]
                    + list(page.key_files or [])
                    if p
                ).lower()
                for dir_path in all_expected:
                    if dir_path.lower() in hay:
                        covered_dirs.add(dir_path)

        # Find missing directories
        missing_all = [d for d in all_expected if d not in covered_dirs]

        # Sort: shallowest first so parent pages absorb children
        missing_all.sort(key=lambda d: (d.count("/"), d))

        # ---------------------------------------------------------------
        # Group sibling missing dirs under a common parent into one page.
        # E.g. ado/repos, ado/wiki, ado/work_item → one "ado" page.
        # ---------------------------------------------------------------
        grouped: dict = {}  # parent_path → [child_dir, ...]
        standalone: list = []
        for d in missing_all:
            parts = d.split("/")
            if len(parts) <= 1:
                standalone.append(d)
            else:
                parent = "/".join(parts[:-1])
                grouped.setdefault(parent, []).append(d)

        # -----------------------------------------------------------------
        # Build final missing list with weighted splitting.
        # When a sibling group covers directories with many code files
        # (> SPLIT_THRESHOLD), split into multiple sub-groups so each
        # reconciliation page stays manageable (~3-5 dirs per page).
        # -----------------------------------------------------------------
        SPLIT_THRESHOLD = 50  # code files per group before splitting
        CHUNK_SIZE = 4  # target dirs per sub-group after split

        # Pre-compute code file counts per directory for fast lookup
        _code_files_per_dir: dict = {}  # dir_path → code file count
        for path in repository_files:
            if not path:
                continue
            _, ext = os.path.splitext(path)
            if ext.lower() not in {
                ".py",
                ".js",
                ".ts",
                ".jsx",
                ".tsx",
                ".java",
                ".go",
                ".rs",
                ".rb",
                ".cpp",
                ".cc",
                ".cxx",
                ".c++",
                ".c",
                ".h",
                ".hpp",
                ".hh",
                ".hxx",
                ".cs",
                ".swift",
                ".kt",
                ".kts",
                ".scala",
                ".php",
                ".vue",
                ".svelte",
                ".lua",
                ".r",
                ".jl",
                ".ex",
                ".exs",
                ".erl",
                ".hs",
                ".clj",
                ".cljs",
                ".ml",
                ".fs",
                ".fsx",
                ".dart",
                ".groovy",
                ".gradle",
            }:
                continue
            parts = path.split("/")
            for depth in range(1, min(len(parts), max_depth + 1)):
                dp = "/".join(parts[:depth])
                _code_files_per_dir[dp] = _code_files_per_dir.get(dp, 0) + 1

        def _group_code_files(dirs: list) -> int:
            """Sum code files across a list of dirs (non-recursive count)."""
            return sum(_code_files_per_dir.get(d, 0) for d in dirs)

        priority_missing: list = []  # (representative_dir, folders_list)
        for d in standalone:
            priority_missing.append((d, [d]))
        for parent, children in sorted(grouped.items()):
            if len(children) >= 2 and parent not in covered_dirs:
                total_code = _group_code_files(children)
                if total_code > SPLIT_THRESHOLD and len(children) > CHUNK_SIZE:
                    # Weighted split: sort children by code-file count descending
                    # so heavy dirs don't cluster together in the same page.
                    children_sorted = sorted(
                        children,
                        key=lambda c: _code_files_per_dir.get(c, 0),
                        reverse=True,
                    )
                    for i in range(0, len(children_sorted), CHUNK_SIZE):
                        chunk = children_sorted[i : i + CHUNK_SIZE]
                        # Representative = parent + chunk index for dedup
                        rep = parent if i == 0 else f"{parent} (part {i // CHUNK_SIZE + 1})"
                        priority_missing.append((rep, chunk))
                    logger.info(
                        f"[COVERAGE][SPLIT] {parent}: {len(children)} dirs, "
                        f"{total_code} code files → "
                        f"{math.ceil(len(children) / CHUNK_SIZE)} sub-groups"
                    )
                else:
                    # Group children under parent (small enough for one page)
                    priority_missing.append((parent, children))
            else:
                for c in children:
                    priority_missing.append((c, [c]))

        if not priority_missing:
            logger.info(f"[STRUCTURE][COVERAGE] All directories covered! total={len(covered_dirs)}")
            return structure

        # Proportional limit — enough to close the gap
        max_missing_default = max(40, int(len(all_expected) * 0.4))
        max_missing = self._get_env_int("WIKIS_DEEPAGENTS_COVERAGE_MAX_DIRS", max_missing_default)
        missing_entries = priority_missing[:max_missing]
        # Flat list of dirs for LLM prompt (unique)
        missing_dirs_flat = []
        seen = set()
        for _, folders in missing_entries:
            for f in folders:
                if f not in seen:
                    missing_dirs_flat.append(f)
                    seen.add(f)

        # Coverage pages are NOT constrained by soft_budget — coverage is
        # the top priority.  We only cap by max_missing to avoid runaway.
        coverage_overflow_default = max(10, int(page_budget * 0.3)) if page_budget else 0
        page_budget_overflow = self._get_env_int("WIKIS_PAGE_BUDGET_OVERFLOW", coverage_overflow_default)
        soft_budget = page_budget + max(page_budget_overflow, 0) if page_budget else 0

        # Group by depth for logging
        depth_counts = {}
        for d in missing_dirs_flat:
            depth = d.count("/") + 1
            depth_counts[depth] = depth_counts.get(depth, 0) + 1

        logger.info(
            f"[STRUCTURE][COVERAGE] Missing dirs by depth: {depth_counts}, "
            f"total={len(missing_dirs_flat)}, grouped_entries={len(missing_entries)}"
        )

        # Don't hard-stop at page budget - coverage is more important
        # Only log a warning if over budget
        if page_budget > 0 and structure.total_pages >= page_budget:
            logger.info(
                f"[STRUCTURE][COVERAGE] Adding coverage pages (over budget but coverage is priority); "
                f"missing={len(missing_dirs_flat)}, current_pages={structure.total_pages}, "
                f"budget={page_budget}, soft_cap={soft_budget or 'unlimited'}"
            )

        if not structure.sections:
            structure.sections = [
                SectionSpec(
                    section_name="Additional Coverage",
                    section_order=1,
                    description="Auto-added coverage for uncovered directories.",
                    rationale="Coverage reconciliation",
                    pages=[],
                )
            ]

        coverage_section = structure.sections[-1]
        if coverage_section.section_name.lower() != "additional coverage":
            coverage_section = SectionSpec(
                section_name="Additional Coverage",
                section_order=len(structure.sections) + 1,
                description="Auto-added coverage for uncovered directories.",
                rationale="Coverage reconciliation",
                pages=[],
            )
            structure.sections.append(coverage_section)

        start_order = max((p.page_order for s in structure.sections for p in s.pages), default=0) + 1
        added = 0
        use_llm = os.getenv("WIKIS_DEEPAGENTS_COVERAGE_LLM", "1") != "0"
        # Coverage pages are NOT subject to soft_budget — they are always added
        max_coverage_pages = len(missing_entries)

        pages_from_llm: list[PageSpec] = []
        if use_llm and max_coverage_pages > 0:
            try:
                pages_from_llm = self._generate_coverage_pages_with_llm(
                    missing_dirs=missing_dirs_flat,
                    repo_outline=repo_outline,
                    target_audience=target_audience,
                    wiki_type=wiki_type,
                    max_pages=max_coverage_pages,
                    config=config,
                )
            except Exception as e:
                logger.warning(f"[STRUCTURE][COVERAGE] LLM coverage refinement failed: {e}")

        if pages_from_llm:
            for page in pages_from_llm:
                page.page_order = start_order + added
                if not page.target_folders:
                    page.target_folders = [page.page_name.split(" ", 1)[0]]
                coverage_section.pages.append(page)
                added += 1
        else:
            # Deterministic fallback: one page per grouped entry
            for representative, folders in missing_entries:
                dir_leaf = representative.rsplit("/", 1)[-1] if "/" in representative else representative
                page = PageSpec(
                    page_name=f"{dir_leaf.replace('_', ' ').title()} Overview",
                    page_order=start_order + added,
                    description=f"Overview of {representative} components and responsibilities.",
                    content_focus=f"Key modules, responsibilities, and workflows in {representative}.",
                    rationale="Ensure directory coverage",
                    target_folders=folders,
                    key_files=[],
                    retrieval_query=f"{representative} overview architecture core components responsibilities modules",
                )
                coverage_section.pages.append(page)
                added += 1

        if added:
            structure.total_pages += added
            sample = [e[0] for e in missing_entries[:added]]
            logger.info(
                f"[STRUCTURE][COVERAGE] Added {added} coverage page(s) "
                f"covering {len(missing_dirs_flat)} dirs: {', '.join(sample[:10])}"
            )

        return structure

    def _generate_coverage_pages_with_llm(
        self,
        missing_dirs: list[str],
        repo_outline: str,
        target_audience: str,
        wiki_type: str,
        max_pages: int,
        config: RunnableConfig,
    ) -> list[PageSpec]:
        """Use LLM to generate coverage pages for missing directories."""
        if not missing_dirs or max_pages <= 0:
            return []

        def _truncate_lines(text: str, max_lines: int) -> str:
            if not text:
                return ""
            lines = text.splitlines()
            if len(lines) <= max_lines:
                return text
            return "\n".join(lines[:max_lines]) + "\n... [truncated]"

        outline_brief = _truncate_lines(repo_outline, 200)

        prompt = ChatPromptTemplate.from_messages(
            [
                ("system", "Return ONLY valid JSON. No markdown fences. No prose."),
                (
                    "human",
                    "You are generating coverage pages for missing top-level directories.\n"
                    "Return JSON with a single key 'pages' (array). Each page must include:\n"
                    "page_name, description, content_focus, rationale, target_folders, key_files, retrieval_query.\n"
                    "Use only these directories: {missing_dirs}.\n"
                    "Limit pages to {max_pages}. Target audience: {target_audience}. Wiki style: {wiki_type}.\n"
                    "Retrieval queries must combine page topic + folder/file terms.\n\n"
                    "Repository outline (truncated):\n{repo_outline}\n",
                ),
            ]
        )

        prompt_messages = prompt.format_messages(
            missing_dirs=", ".join(missing_dirs),
            max_pages=max_pages,
            target_audience=target_audience,
            wiki_type=wiki_type,
            repo_outline=outline_brief,
        )

        llm_message = self.llm.invoke(prompt_messages, config=config)
        raw_content = getattr(llm_message, "content", "")
        if raw_content is None:
            raw_text = ""
        elif isinstance(raw_content, str):
            raw_text = raw_content
        elif isinstance(raw_content, list):
            parts = []
            for item in raw_content:
                if item is None:
                    continue
                if isinstance(item, str):
                    parts.append(item)
                elif isinstance(item, dict):
                    parts.append(str(item.get("text") or item.get("content") or ""))
                else:
                    parts.append(str(item))
            raw_text = "\n".join([p for p in parts if p]).strip()
        else:
            raw_text = str(raw_content)

        data = self._parse_llm_json_response(raw_text)
        pages_raw = data.get("pages", []) if isinstance(data, dict) else []

        pages: list[PageSpec] = []
        for idx, page_data in enumerate(pages_raw[:max_pages]):
            if not isinstance(page_data, dict):
                continue
            try:
                page = PageSpec(
                    page_name=page_data.get("page_name", f"coverage {idx + 1}"),
                    page_order=0,
                    description=page_data.get("description", "Coverage page."),
                    content_focus=page_data.get("content_focus", "Overview and responsibilities."),
                    rationale=page_data.get("rationale", "Coverage reconciliation"),
                    target_folders=page_data.get("target_folders", []),
                    key_files=page_data.get("key_files", []),
                    retrieval_query=page_data.get("retrieval_query", ""),
                )
                pages.append(page)
            except Exception:  # noqa: S112
                continue

        return pages

    def _truncate_log_text(self, text: Any, limit: int = 1200) -> str:
        """Trim log text to a safe length."""
        if text is None:
            return ""
        value = str(text)
        if len(value) <= limit:
            return value
        return value[:limit] + "... [truncated]"

    def _log_deepagents_messages(self, messages: list[Any], tool_message_cls: Any) -> None:
        """Log tool calls and tool outputs from deepagents messages."""
        for msg in messages:
            if msg is None:
                continue
            msg_type = getattr(msg, "type", None) or msg.__class__.__name__
            tool_calls = getattr(msg, "tool_calls", None)
            if tool_calls:
                for call in tool_calls:
                    name = call.get("name") or call.get("function", {}).get("name")
                    args = call.get("args") or call.get("arguments") or {}
                    logger.info(
                        "[DEEPAGENTS][TOOL_CALL] %s args=%s",
                        name,
                        self._truncate_log_text(args, 800),
                    )
            if tool_message_cls and isinstance(msg, tool_message_cls):
                tool_name = getattr(msg, "name", None)
                tool_call_id = getattr(msg, "tool_call_id", None)
                content = self._truncate_log_text(getattr(msg, "content", ""))
                logger.info(
                    "[DEEPAGENTS][TOOL_RESULT] %s id=%s\n%s",
                    tool_name or "tool",
                    tool_call_id,
                    content,
                )
            if msg_type in {"AIMessage", "ai"}:
                content = getattr(msg, "content", "")
                if content:
                    logger.info(
                        "[DEEPAGENTS][AI] %s",
                        self._truncate_log_text(content, 800),
                    )

    def _log_deepagents_event(self, event: Any, tool_message_cls: Any) -> None:
        """Log events from deepagents stream (messages + updates)."""
        if isinstance(event, dict):
            messages = event.get("messages", [])
            if messages:
                self._log_deepagents_messages(messages, tool_message_cls)
            updates = event.get("updates")
            if updates:
                logger.info(
                    "[DEEPAGENTS][UPDATE] %s",
                    self._truncate_log_text(updates, 1200),
                )

    def _parse_llm_json_response(self, response_content: str) -> dict[str, Any]:
        """Parse JSON response from LLM, handling potential formatting issues"""
        response_content = (response_content or "").strip()
        try:
            # Try direct JSON parsing first
            return json.loads(response_content)
        except json.JSONDecodeError as e:
            if os.getenv("WIKIS_LOG_LLM_STRUCTURE_RAW") == "1":
                raw = response_content or ""
                raw_len = len(raw)
                prefix = raw[:4000].replace("\n", "\\n")
                suffix = raw[-800:].replace("\n", "\\n") if raw_len > 4000 else ""
                logger.warning(
                    f"LLM structure JSON parse failed: {e}. Raw length={raw_len}. "
                    f"Prefix(4k)='{prefix}'" + (f" Suffix(800)='{suffix}'" if suffix else "")
                )
            # Try to extract JSON from fenced markdown code blocks.
            # NOTE: previous implementation incorrectly searched for literal "\\n".
            fence_match = re.search(r"```json\s*(.*?)\s*```", response_content, re.DOTALL | re.IGNORECASE)
            if fence_match:
                return json.loads(fence_match.group(1).strip())

            # Some models emit ``` (no language). Accept it if it looks like JSON.
            fence_any = re.search(r"```\s*(\{.*?\})\s*```", response_content, re.DOTALL)
            if fence_any:
                return json.loads(fence_any.group(1).strip())

            # Try to extract the first JSON object in the text.
            obj_match = re.search(r"\{.*\}", response_content, re.DOTALL)
            if obj_match:
                return json.loads(obj_match.group(0).strip())

            # Fallback: create basic structure that matches WikiStructureSpec
            logger.warning("Could not parse LLM JSON response, using fallback structure")
            return {
                "wiki_title": "Generated Wiki",
                "overview": "Documentation for the repository",
                "sections": [
                    {
                        "section_name": "Documentation",
                        "section_order": 1,
                        "description": "Main documentation",
                        "rationale": "Fallback structure due to parsing error",
                        "pages": [
                            {
                                "page_name": "Overview",
                                "page_order": 1,
                                "description": "Repository overview",
                                "content_focus": "General information",
                                "rationale": "Basic overview page for fallback structure",
                                "target_folders": [],
                                "key_files": [],
                                "retrieval_query": "repository overview introduction getting started documentation README architecture",
                            }
                        ],
                    }
                ],
                "total_pages": 1,
            }

    def _normalize_structure_payload(self, payload: Any) -> dict[str, Any]:
        """Normalize structure payloads into WikiStructureSpec shape."""
        if not isinstance(payload, dict):
            return {}

        normalized = dict(payload)

        if not normalized.get("wiki_title"):
            normalized["wiki_title"] = normalized.get("title") or normalized.get("name") or "Generated Wiki"

        if normalized.get("overview") is None:
            normalized["overview"] = normalized.get("summary") or normalized.get("introduction") or ""

        sections_raw = normalized.get("sections")
        if sections_raw is None:
            sections_raw = normalized.get("chapters") or normalized.get("structure") or []

        if not isinstance(sections_raw, list):
            sections_raw = []

        sections: list[dict[str, Any]] = []

        for s_idx, section in enumerate(sections_raw):
            if not isinstance(section, dict):
                continue

            section_name = (
                section.get("section_name") or section.get("title") or section.get("name") or f"Section {s_idx + 1}"
            )

            section_order = section.get("section_order") or section.get("order") or (s_idx + 1)

            section_desc = section.get("description") or section.get("summary") or ""

            section_rationale = section.get("rationale") or section.get("reason") or "Structure planning"

            pages_raw = section.get("pages")
            if pages_raw is None:
                pages_raw = section.get("items") or section.get("subsections") or []

            if not isinstance(pages_raw, list):
                pages_raw = []

            pages: list[dict[str, Any]] = []
            for p_idx, page in enumerate(pages_raw):
                if not isinstance(page, dict):
                    continue

                page_name = (
                    page.get("page_name") or page.get("title") or page.get("name") or f"{section_name} Page {p_idx + 1}"
                )

                page_order = page.get("page_order") or page.get("order") or (p_idx + 1)

                page_desc = page.get("description") or page.get("summary") or ""
                raw_focus = page.get("content_focus") or page.get("focus") or ""
                if isinstance(raw_focus, list):
                    content_focus = "; ".join([str(item) for item in raw_focus if item is not None])
                else:
                    content_focus = raw_focus
                rationale = page.get("rationale") or page.get("reason") or "Structure planning"

                target_folders = page.get("target_folders") or page.get("folders") or []
                key_files = page.get("key_files") or page.get("files") or []
                retrieval_query = page.get("retrieval_query") or page.get("query") or ""
                target_symbols = page.get("target_symbols") or page.get("symbols") or []
                target_docs = page.get("target_docs") or page.get("docs") or []

                pages.append(
                    {
                        "page_name": page_name,
                        "page_order": page_order,
                        "description": page_desc,
                        "content_focus": content_focus,
                        "rationale": rationale,
                        "target_symbols": target_symbols,
                        "target_docs": target_docs,
                        "target_folders": target_folders,
                        "key_files": key_files,
                        "retrieval_query": retrieval_query,
                    }
                )

            sections.append(
                {
                    "section_name": section_name,
                    "section_order": section_order,
                    "description": section_desc,
                    "rationale": section_rationale,
                    "pages": pages,
                }
            )

        normalized["sections"] = sections

        computed_pages = sum(len(s.get("pages", [])) for s in sections)
        existing_pages = normalized.get("total_pages")
        if isinstance(existing_pages, int) and existing_pages != computed_pages:
            logger.info(
                "[STRUCTURE][NORMALIZE] total_pages mismatch: reported=%s computed=%s; using computed",
                existing_pages,
                computed_pages,
            )
        normalized["total_pages"] = computed_pages

        return normalized

    def _get_docs_by_target_symbols(
        self, target_symbols: list[str], max_hops: int = 1, target_folders: list[str] = None
    ) -> list[Document]:
        """
        Pure graph-based retrieval using exact symbol names from structure planning.

        This retrieves code content directly from graph nodes - NO indexer needed.
        1. Finds exact symbol matches in the code graph
        2. Expands to direct neighbors (imports, usages) for context
        3. Creates Documents directly from graph node content

        Args:
            target_symbols: Exact class/function names from query_graph (e.g., ["WorkflowExecutor", "BaseNode"])
            max_hops: How many relationship hops to expand (default: 1 for direct connections)
            target_folders: Page's target_folders for path-scoped resolution of ambiguous symbols

        Returns:
            List of Documents created directly from graph node data
        """
        if not target_symbols:
            return []

        # Get the relationship graph
        graph = getattr(self.retriever_stack, "relationship_graph", None)
        if graph is None:
            graph = getattr(self.indexer, "relationship_graph", None)

        if graph is None:
            logger.warning("[GRAPH_RETRIEVAL] No relationship_graph available, falling back to vector search")
            return []

        # Find matching nodes in the graph
        matched_nodes = set()
        symbol_to_nodes = {}  # Track which symbols matched for logging

        # Normalize target_folders for prefix matching
        _scope_prefixes = [f.strip("/") for f in (target_folders or []) if f]

        # Compute repo-aware thresholds once
        graph_nodes_count = graph.number_of_nodes() if graph else 0
        file_count = getattr(self, "_repository_file_count", 0) or 0
        max_per_page = self._compute_max_symbols_per_page(graph_nodes_count, file_count)
        # Symbols appearing in more files than max_per_page are boilerplate
        # (logger, __all__) — skip entirely unless path-scoping narrows them down
        _frequency_threshold = max_per_page

        def _path_scope_nodes(node_ids: list[str], target_name: str) -> list[str]:
            """Prefer nodes under target_folders when a symbol matches many nodes.

            For symbols like get_tools that exist in 60+ files, this picks
            only the ones in the page's relevant directories.
            Falls back to all nodes if none match the scope.
            """
            if not _scope_prefixes or len(node_ids) <= 1:
                return node_ids

            scoped = []
            for nid in node_ids:
                nd = graph.nodes.get(nid, {})
                rel_path = (nd.get("rel_path") or nd.get("file_path", "")).strip("/")
                for prefix in _scope_prefixes:
                    if rel_path.startswith(prefix + "/") or rel_path == prefix:
                        scoped.append(nid)
                        break

            if scoped:
                if len(scoped) < len(node_ids):
                    logger.info(
                        f"[GRAPH_RETRIEVAL][PATH_SCOPE] '{target_name}': "
                        f"{len(node_ids)} → {len(scoped)} nodes "
                        f"(scoped to {_scope_prefixes[:3]})"
                    )
                return scoped
            # No nodes in scope — fall back to all
            return node_ids

        # STRATEGY 1: Use graph's _name_index if available (O(1) lookup)
        # Filter to architectural symbols to prevent methods/fields noise
        name_index = getattr(graph, "_name_index", None)
        if name_index:
            for target in target_symbols:
                # Direct lookup by exact name
                if target in name_index:
                    # Filter to arch symbols only — skip methods/constructors/fields
                    node_ids = [
                        nid
                        for nid in name_index[target]
                        if (graph.nodes.get(nid, {}).get("symbol_type") or "").lower() in ARCHITECTURAL_SYMBOLS
                    ]
                    if node_ids:
                        # PATH-SCOPED RESOLUTION: prefer nodes under target_folders
                        node_ids = _path_scope_nodes(node_ids, target)

                        # FREQUENCY FILTER: after path scoping, if still too many
                        # nodes remain, this symbol is boilerplate (logger, __all__)
                        if len(node_ids) > _frequency_threshold:
                            logger.info(
                                f"[GRAPH_RETRIEVAL][FREQ_FILTER] Skipping '{target}' — "
                                f"still {len(node_ids)} nodes after path scoping "
                                f"(threshold={_frequency_threshold}), likely boilerplate"
                            )
                            continue

                        matched_nodes.update(node_ids)
                        symbol_to_nodes[target] = node_ids
                        logger.debug(f"[GRAPH_RETRIEVAL] Index hit for '{target}': {len(node_ids)} arch nodes")

        # STRATEGY 2: If index didn't find all symbols, use FTS5 or graph scan
        unmatched_symbols = [t for t in target_symbols if t not in symbol_to_nodes]
        if unmatched_symbols:
            fts = self.graph_text_index
            if fts is not None and fts.is_open:
                # Determine majority language from already-matched nodes to
                # prevent cross-language contamination (e.g. "App" matching
                # C++/Go/Java/Python in a multilingual repo).
                lang_counts: dict = {}
                for nids in symbol_to_nodes.values():
                    for nid in nids:
                        nd = graph.nodes.get(nid, {})
                        lang = (nd.get("language") or "").strip().lower()
                        if lang:
                            lang_counts[lang] = lang_counts.get(lang, 0) + 1
                page_lang = max(lang_counts, key=lang_counts.get) if lang_counts else None

                # FTS5 O(log N) lookup — search by name, filtered to arch symbols
                # to prevent methods/constructors/fields from leaking into page context
                logger.debug(
                    f"[GRAPH_RETRIEVAL][FTS5] Resolving {len(unmatched_symbols)} "
                    f"unmatched symbols via FTS5 (lang={page_lang}): {unmatched_symbols}"
                )
                for target in unmatched_symbols:
                    # Try exact match first, then fall back to fuzzy with lower k
                    docs = fts.search_by_name(
                        target,
                        exact=True,
                        k=5,
                        symbol_types=ARCHITECTURAL_SYMBOLS,
                        language=page_lang,
                    )
                    if not docs:
                        docs = fts.search_by_name(
                            target,
                            exact=False,
                            k=5,
                            symbol_types=ARCHITECTURAL_SYMBOLS,
                            language=page_lang,
                        )
                    if docs:
                        for doc in docs:
                            nid = doc.metadata.get("node_id", "")
                            if nid:
                                matched_nodes.add(nid)
                                if target not in symbol_to_nodes:
                                    symbol_to_nodes[target] = []
                                symbol_to_nodes[target].append(nid)
                        logger.debug(f"[GRAPH_RETRIEVAL][FTS5] '{target}' → {len(docs)} hits")
                    else:
                        logger.debug(f"[GRAPH_RETRIEVAL][FTS5] '{target}' → 0 hits")

            # STRATEGY 3: Final O(N) fallback for any still-unmatched symbols
            # Runs regardless of whether FTS5 was used — catches symbols that
            # may be stored differently (namespace-qualified, hash suffixes, etc.)
            # Filtered to architectural symbols only to avoid methods/fields noise.
            still_unmatched = [t for t in target_symbols if t not in symbol_to_nodes]
            if still_unmatched:
                logger.debug(
                    f"[GRAPH_RETRIEVAL] O(N) scan for {len(still_unmatched)} still-unmatched symbols: {still_unmatched}"
                )
                for node_id, node_data in graph.nodes(data=True):
                    # Skip non-architectural symbols (methods, constructors, etc.)
                    sym_type = (node_data.get("symbol_type") or "").lower()
                    if sym_type and sym_type not in ARCHITECTURAL_SYMBOLS:
                        continue

                    # Get symbol name from node data (both comprehensive and basic modes)
                    symbol_name = node_data.get("symbol_name", "") or node_data.get("name", "")

                    # Also extract symbol name from node_id (format: language::path::SymbolName)
                    node_id_parts = str(node_id).split("::")
                    node_id_symbol = node_id_parts[-1] if len(node_id_parts) >= 3 else ""
                    # Remove any hash suffix (e.g., ClassName_a1b2)
                    if "_" in node_id_symbol and len(node_id_symbol.split("_")[-1]) == 4:
                        node_id_symbol = "_".join(node_id_symbol.split("_")[:-1])

                    # Use either source
                    if not symbol_name and node_id_symbol:
                        symbol_name = node_id_symbol

                    if not symbol_name:
                        continue

                    # Check if this symbol matches any of our still-unmatched target symbols
                    for target in still_unmatched:
                        # Match exact name or qualified suffix only — NO substring
                        # (substring "Start" would match hundreds of symbols)
                        if symbol_name == target or symbol_name.endswith(f".{target}") or node_id_symbol == target:
                            matched_nodes.add(node_id)
                            if target not in symbol_to_nodes:
                                symbol_to_nodes[target] = []
                            symbol_to_nodes[target].append(node_id)

        # ---- Cross-language post-filter ----
        # In multilingual repos, common names (Config, App, Workload) match in
        # every language.  Compute the majority language and remove outliers so
        # a Java page doesn't drag in Go/C++/Python nodes.
        if matched_nodes and len(matched_nodes) > 1:
            lang_counts: dict = {}
            node_langs: dict = {}  # node_id → language
            for nid in matched_nodes:
                nd = graph.nodes.get(nid, {})
                lang = (nd.get("language") or "").strip().lower()
                if lang:
                    lang_counts[lang] = lang_counts.get(lang, 0) + 1
                    node_langs[nid] = lang
            if lang_counts:
                majority_lang = max(lang_counts, key=lang_counts.get)
                majority_count = lang_counts[majority_lang]
                total_with_lang = sum(lang_counts.values())
                # Only filter if one language clearly dominates (>=60%) and
                # there are actually cross-language nodes to remove.
                if len(lang_counts) > 1 and majority_count / total_with_lang >= 0.6:
                    before = len(matched_nodes)
                    evicted_langs: set = set()
                    for nid in list(matched_nodes):
                        nl = node_langs.get(nid, "")
                        if nl and nl != majority_lang:
                            matched_nodes.discard(nid)
                            evicted_langs.add(nl)
                    # Also clean up symbol_to_nodes
                    for sym in list(symbol_to_nodes):
                        symbol_to_nodes[sym] = [nid for nid in symbol_to_nodes[sym] if nid in matched_nodes]
                        if not symbol_to_nodes[sym]:
                            del symbol_to_nodes[sym]
                    if before != len(matched_nodes):
                        logger.info(
                            f"[GRAPH_RETRIEVAL][LANG_FILTER] Removed "
                            f"{before - len(matched_nodes)} cross-language nodes "
                            f"(kept {majority_lang}, removed {evicted_langs})"
                        )

        # ---- Repo-aware safety cap ----
        # Prevent degenerate cases where common names match many nodes.
        # Uses the same repo-size-scaled cap as page splitting.
        if len(matched_nodes) > max_per_page:
            logger.warning(
                f"[GRAPH_RETRIEVAL] {len(matched_nodes)} matched nodes exceeds "
                f"repo-scaled cap {max_per_page} — trimming"
            )
            matched_nodes = set(list(matched_nodes)[:max_per_page])
            # Trim symbol_to_nodes to match
            for sym in list(symbol_to_nodes):
                symbol_to_nodes[sym] = [nid for nid in symbol_to_nodes[sym] if nid in matched_nodes]
                if not symbol_to_nodes[sym]:
                    del symbol_to_nodes[sym]

        logger.info(f"[GRAPH_RETRIEVAL] Matched {len(matched_nodes)} nodes for {len(target_symbols)} target symbols")
        for sym, nodes in symbol_to_nodes.items():
            logger.debug(f"[GRAPH_RETRIEVAL]   {sym}: {len(nodes)} nodes")

        if not matched_nodes:
            # Diagnostic: Log some similar symbols that DO exist in the graph
            similar_symbols = []
            sample_node_ids = []

            if name_index:
                for target in target_symbols:
                    target_lower = target.lower()
                    # Search ALL index keys (not just a sample) for substring matches
                    for indexed_name in name_index:
                        if target_lower in indexed_name.lower():
                            similar_symbols.append(indexed_name)
                        if len(similar_symbols) >= 10:
                            break
                    if len(similar_symbols) >= 10:
                        break

            # Also sample some node_ids to show format
            for node_id in list(graph.nodes())[:10]:
                node_data = graph.nodes.get(node_id, {})
                sym_name = node_data.get("symbol_name", "") or node_data.get("name", "")
                sample_node_ids.append(
                    f"{node_id[:80]}... ({sym_name})" if len(str(node_id)) > 80 else f"{node_id} ({sym_name})"
                )

            # Compute total indexed node count (name_index keys map to lists of node_ids)
            total_indexed_nodes = sum(len(v) for v in name_index.values()) if name_index else 0

            if similar_symbols:
                logger.warning(
                    f"[GRAPH_RETRIEVAL] No exact matches for {target_symbols}. "
                    f"Similar symbols in index: {similar_symbols[:5]}"
                )
            else:
                logger.warning(
                    f"[GRAPH_RETRIEVAL] No matches found for symbols: {target_symbols}. "
                    f"Graph has {graph.number_of_nodes()} nodes, "
                    f"name_index has {len(name_index) if name_index else 0} unique names "
                    f"({total_indexed_nodes} node entries)."
                )
                logger.debug(f"[GRAPH_RETRIEVAL] Sample node_ids: {sample_node_ids[:5]}")
            return []

        # Expand to neighbors — either smart (relationship-aware) or naive (legacy)
        augmentations = {}  # node_id → augmented content string
        expansion_reasons = {}  # node_id → reason string (for metadata)

        if SMART_EXPANSION_ENABLED and max_hops > 0:
            # ---- Smart Expansion: priority-based edge traversal ----
            try:
                result = expand_smart(matched_nodes, graph)
                expanded_nodes = result.expanded_nodes
                augmentations = {nid: aug.augmented_content for nid, aug in result.augmentations.items()}
                expansion_reasons = result.expansion_reasons
                logger.info(
                    f"[GRAPH_RETRIEVAL][SMART] Expanded to {len(expanded_nodes)} nodes "
                    f"(from {len(matched_nodes)} matched, {len(augmentations)} augmented)"
                )
            except Exception:
                logger.warning(
                    "[GRAPH_RETRIEVAL][SMART] Smart expansion failed, falling back to naive",
                    exc_info=True,
                )
                expanded_nodes = set(matched_nodes)
                augmentations = {}
                expansion_reasons = {}
                # Fall through to naive below
                SMART_EXPANSION_ENABLED_LOCAL = False
            else:
                SMART_EXPANSION_ENABLED_LOCAL = True
        else:
            SMART_EXPANSION_ENABLED_LOCAL = False

        if not SMART_EXPANSION_ENABLED or not locals().get("SMART_EXPANSION_ENABLED_LOCAL", False):
            # ---- Legacy naive expansion (relationship-blind) ----
            expanded_nodes = set(matched_nodes)
            if max_hops > 0:
                for node_id in matched_nodes:
                    try:
                        predecessors = [
                            n
                            for n in graph.predecessors(node_id)
                            if graph.nodes.get(n, {}).get("symbol_type") in EXPANSION_SYMBOL_TYPES
                        ]
                        successors = [
                            n
                            for n in graph.successors(node_id)
                            if graph.nodes.get(n, {}).get("symbol_type") in EXPANSION_SYMBOL_TYPES
                        ]
                        expanded_nodes.update(predecessors[:10])
                        expanded_nodes.update(successors[:10])
                    except Exception:  # noqa: S110
                        pass
            logger.info(
                f"[GRAPH_RETRIEVAL][NAIVE] Expanded to {len(expanded_nodes)} nodes (from {len(matched_nodes)} initial)"
            )

        # ---- Framework-string fallback for orphan classes ----
        # When a class (like Event) has no graph-edge relationships — no creates,
        # no calls to its methods, no inheritance — its method names may still be
        # referenced as string literals in framework/RPC code.  Use FTS5 text
        # search on method names to discover those string-based references.
        fts = self.graph_text_index
        if fts is not None and fts.is_open:
            searched_methods: set = set()  # dedup across orphan classes
            for node_id in list(matched_nodes):
                node_data = graph.nodes.get(node_id, {})
                sym_type = (node_data.get("symbol_type") or "").lower()
                if sym_type not in ("class", "interface", "struct", "enum", "trait"):
                    continue

                # Check if this class already has expansion neighbors
                # (successors/predecessors that ended up in expanded_nodes).
                # If it does, it's not an orphan — the graph gave us context.
                has_expansion = False
                for succ in list(graph.successors(node_id))[:20]:
                    if succ in expanded_nodes and succ != node_id:
                        has_expansion = True
                        break
                if not has_expansion:
                    for pred in list(graph.predecessors(node_id))[:20]:
                        if pred in expanded_nodes and pred != node_id:
                            has_expansion = True
                            break

                if has_expansion:
                    continue

                # This is an orphan class — search FTS for its method names.
                # Use has_relationship() for robust MultiDiGraph edge detection.
                # Scope search to the same language to prevent cross-language
                # contamination in multilingual repos (e.g. run/close/start
                # matching C++/Go/Java/Python indiscriminately).
                orphan_lang = (node_data.get("language") or "").strip() or None
                method_names = []
                for succ in graph.successors(node_id):
                    if has_relationship(graph, node_id, succ, "defines"):
                        succ_data = graph.nodes.get(succ, {})
                        succ_type = (succ_data.get("symbol_type") or "").lower()
                        if succ_type in ("method", "function"):
                            method_name = succ_data.get("symbol_name", "") or succ_data.get("name", "")
                            if method_name and not method_name.startswith("__"):
                                method_names.append(method_name)

                if method_names:
                    sym_name = node_data.get("symbol_name", "") or node_data.get("name", "")
                    logger.debug(
                        f"[GRAPH_RETRIEVAL][ORPHAN_FTS] Class '{sym_name}' is an orphan "
                        f"(lang={orphan_lang}), "
                        f"searching FTS for method names: {method_names[:5]}"
                    )
                    fts_hits = 0
                    for method_name in method_names[:5]:  # Cap at 5 methods
                        # Dedup: skip method names already searched by a
                        # sibling orphan (e.g. inherited run/close/start).
                        dedup_key = (method_name, orphan_lang)
                        if dedup_key in searched_methods:
                            continue
                        searched_methods.add(dedup_key)
                        # Search for the method name in source text (string refs)
                        docs = fts.search(
                            method_name,
                            k=3,
                            symbol_types=ARCHITECTURAL_SYMBOLS - frozenset({"macro"}),
                            language=orphan_lang,
                        )
                        for doc in docs:
                            nid = doc.metadata.get("node_id", "")
                            if nid and nid != node_id and nid not in expanded_nodes:
                                expanded_nodes.add(nid)
                                expansion_reasons[nid] = f"FTS string-ref to {sym_name}.{method_name}"
                                fts_hits += 1
                    if fts_hits:
                        logger.info(
                            f"[GRAPH_RETRIEVAL][ORPHAN_FTS] Found {fts_hits} FTS hits "
                            f"for orphan class '{sym_name}' method names"
                        )

        # Create Documents directly from graph node data
        documents = []
        seen_content_hashes = set()

        for node_id in expanded_nodes:
            # Use augmented content for C++ nodes if available
            augmented_content = augmentations.get(node_id)
            doc = self._node_to_document(node_id, graph, augmented_content=augmented_content)
            if doc is None:
                continue

            # Attach expansion metadata
            if node_id in expansion_reasons:
                doc.metadata["expansion_reason"] = expansion_reasons[node_id]

            content_hash = hash(doc.page_content[:500])
            if content_hash in seen_content_hashes:
                continue
            seen_content_hashes.add(content_hash)
            documents.append(doc)

        return documents

    def _get_docs_from_store_semantic(self, page_spec: "PageSpec", max_docs: int = 10) -> list[Document]:
        """Semantic doc retrieval using vector store + optional EmbeddingsFilter reranking.

        This method is used when WIKIS_DOC_SEPARATE_INDEX=1 (docs not in graph)
        or when WIKIS_DOC_SEMANTIC_RETRIEVAL=1 (prefer semantic over path-based).

        Args:
            page_spec: Page specification with retrieval_query, page_name, etc.
            max_docs: Maximum number of documentation documents to return

        Returns:
            List of Documents retrieved semantically from vector store
        """
        # Build query from page spec
        retrieval_query = getattr(page_spec, "retrieval_query", "")
        page_name = getattr(page_spec, "page_name", "")
        description = getattr(page_spec, "description", "")

        # Compose semantic query from available page info
        query_parts = [retrieval_query, page_name, description]
        query = " ".join(p for p in query_parts if p).strip()

        if not query:
            logger.warning("[SEMANTIC_DOC_RETRIEVAL] No query available for semantic doc retrieval")
            return []

        # Use centralized semantic doc retrieval from retriever stack
        # This applies EmbeddingsFilter reranking when USE_SEMANTIC_DOC_RETRIEVAL=1
        try:
            doc_candidates = self.retriever_stack.search_docs_semantic(
                query=query, k=max_docs, similarity_threshold=0.3
            )

            if doc_candidates:
                logger.info(
                    f"[SEMANTIC_DOC_RETRIEVAL] Retrieved {len(doc_candidates)} docs "
                    f"for page '{page_name}' via retriever_stack.search_docs_semantic()"
                )

            return doc_candidates

        except Exception as e:
            logger.error(f"[SEMANTIC_DOC_RETRIEVAL] Error during semantic doc retrieval: {e}")
            return []

    @staticmethod
    def _node_to_document(
        node_id: str,
        graph,
        augmented_content: str | None = None,
    ) -> Document | None:
        """Create a ``Document`` from a graph node, or ``None`` if it has no content.

        Extracts content and metadata from both basic and comprehensive analysis
        modes (direct attributes *or* nested ``symbol`` object).

        Args:
            node_id: The graph node identifier.
            graph: NetworkX graph.
            augmented_content: If provided (smart expansion C++ augmentation),
                use this instead of the node's native source text.
        """
        node_data = graph.nodes.get(node_id, {})
        if not node_data:
            return None

        # --- Content ---
        if augmented_content:
            content = augmented_content
        else:
            content = node_data.get("source_text", "")
        if not content:
            symbol_obj = node_data.get("symbol")
            if symbol_obj and hasattr(symbol_obj, "source_text"):
                content = symbol_obj.source_text or ""
        if not content:
            content = node_data.get("content", "") or node_data.get("source_code", "") or node_data.get("code", "")
        if not content:
            return None

        # --- Metadata ---
        symbol_obj = node_data.get("symbol")

        file_path = node_data.get("rel_path") or node_data.get("file_path", "")
        if not file_path and symbol_obj:
            file_path = getattr(symbol_obj, "rel_path", "") or getattr(symbol_obj, "file_path", "")

        symbol_name = node_data.get("symbol_name", "") or node_data.get("name", "")
        if not symbol_name and symbol_obj:
            symbol_name = getattr(symbol_obj, "name", "")

        symbol_type = node_data.get("symbol_type", "") or node_data.get("type", "")
        if not symbol_type and symbol_obj:
            st = getattr(symbol_obj, "symbol_type", None)
            symbol_type = st.value if hasattr(st, "value") else str(st) if st else "unknown"

        language = node_data.get("language", "")
        if not language and symbol_obj:
            language = getattr(symbol_obj, "language", "")

        start_line = node_data.get("start_line", 0)
        end_line = node_data.get("end_line", 0)
        if symbol_obj:
            start_line = start_line or getattr(symbol_obj, "start_line", 0)
            end_line = end_line or getattr(symbol_obj, "end_line", 0)

        metadata = {
            "source": file_path,
            "file_path": file_path,
            "rel_path": file_path,
            "symbol": symbol_name,
            "symbol_name": symbol_name,
            "symbol_type": symbol_type or "unknown",
            "node_type": symbol_type or "unknown",
            "chunk_type": "code",
            "language": language,
            "start_line": start_line,
            "end_line": end_line,
            "graph_retrieved": True,
        }
        return Document(page_content=content, metadata=metadata)

    # Documentation symbol types — see constants.py for single source of truth
    DOC_SYMBOL_TYPES = DOC_SYMBOL_TYPES

    def _get_doc_nodes_from_graph(
        self, target_docs: list[str] = None, page_name: str = None, target_symbols: list[str] = None, max_docs: int = 20
    ) -> list[Document]:
        """
        Retrieve documentation nodes from the graph with relevance ranking.

        Documentation files (README.md, docs/*.md, etc.) are stored in the graph with
        symbol_type like 'markdown_document', 'markdown_section', etc. This method
        retrieves those nodes for inclusion in page generation context.

        When WIKIS_DOC_SEPARATE_INDEX=1, docs are NOT in the graph (code-only).
        In that case, callers should use _get_docs_from_store_semantic() instead.

        Relevance ranking (when target_docs not specified):
        1. Docs matching target_symbols in path/name (highest)
        2. Docs matching page_name keywords (medium)
        3. README files (always included, lower priority)
        4. Other docs (lowest)

        Note: These nodes exist in the graph but typically have no edges (relationships).
        This is an architectural wart - we're using NetworkX as a fancy dictionary for docs.
        Pragmatically fine for retrieval (<1% overhead), but not true graph usage.

        Args:
            target_docs: Optional list of specific doc paths to retrieve (deterministic)
            page_name: Page name for keyword-based relevance ranking
            target_symbols: Symbol names for relevance matching
            max_docs: Maximum number of documents to return (default: 20)

        Returns:
            List of Documents created from graph documentation nodes, ranked by relevance
        """
        # When SEPARATE_DOC_INDEX is enabled, docs are NOT in the graph
        # Callers should use semantic retrieval instead
        if SEPARATE_DOC_INDEX:
            logger.info(
                "[DOC_RETRIEVAL] SEPARATE_DOC_INDEX=1: Docs not in graph. Use semantic retrieval from vector store."
            )
            return []

        # Get the relationship graph
        graph = getattr(self.retriever_stack, "relationship_graph", None)
        if graph is None:
            graph = getattr(self.indexer, "relationship_graph", None)

        if graph is None:
            logger.warning("[DOC_RETRIEVAL] No relationship_graph available for doc retrieval")
            return []

        # Collect all doc candidates with scoring
        doc_candidates = []  # [(score, file_path, node_data, content)]

        # Build keyword sets for relevance matching
        page_keywords = set()
        if page_name:
            # Extract keywords from page name (split on common separators)
            for word in page_name.lower().replace("-", " ").replace("_", " ").split():
                if len(word) > 2:  # Skip short words
                    page_keywords.add(word)

        symbol_keywords = set()
        if target_symbols:
            for sym in target_symbols:
                # Split CamelCase and snake_case
                sym_lower = sym.lower()
                symbol_keywords.add(sym_lower)
                # Split camelCase: OrderService -> order, service
                parts = re.findall(r"[A-Z]?[a-z]+|[A-Z]+(?=[A-Z][a-z]|\b)", sym)
                for part in parts:
                    if len(part) > 2:
                        symbol_keywords.add(part.lower())

        # --- FTS5 fast path: use SQLite index when available ---
        fts = self.graph_text_index
        if fts is not None and fts.is_open:
            doc_candidates = self._get_doc_nodes_fts5(
                fts, target_docs, page_name, target_symbols, page_keywords, symbol_keywords, max_docs
            )
        else:
            # --- Brute-force fallback: O(N) graph scan ---
            doc_candidates = self._get_doc_nodes_brute(
                graph, target_docs, page_name, target_symbols, page_keywords, symbol_keywords, max_docs
            )

        # Sort by score (descending) and take top max_docs
        doc_candidates.sort(key=lambda x: -x[0])

        documents = []
        for score, file_path, node_data, content, symbol_type in doc_candidates[:max_docs]:
            # Build metadata
            metadata = {
                "source": file_path,
                "symbol": node_data.get("symbol_name", "") or node_data.get("name", file_path),
                "symbol_type": symbol_type,
                "node_type": symbol_type,
                "chunk_type": "documentation",
                "language": "markdown" if "markdown" in symbol_type else "text",
                "graph_retrieved": True,
                "is_documentation": True,
                "relevance_score": score,
            }

            doc = Document(page_content=content, metadata=metadata)
            documents.append(doc)

        if documents:
            top_scores = [d.metadata.get("relevance_score", 0) for d in documents[:3]]
            logger.info(
                f"[DOC_RETRIEVAL] Retrieved {len(documents)} documentation nodes from graph "
                f"(top scores: {top_scores})"
                f"{f', filtered by {len(target_docs)} target_docs' if target_docs else ''}"
            )
        else:
            logger.info("[DOC_RETRIEVAL] No documentation nodes found in graph")

        return documents

    # ------------------------------------------------------------------
    # _get_doc_nodes helpers: FTS5 fast path + brute-force fallback
    # ------------------------------------------------------------------

    @staticmethod
    def _score_doc_candidate(
        file_path: str,
        content: str,
        target_docs,
        page_keywords: set,
        symbol_keywords: set,
        page_name: str,
    ) -> int:
        """Compute a relevance score for a doc node candidate."""
        score = 0
        path_lower = file_path.lower()
        doc_name = file_path.split("/")[-1].lower() if "/" in file_path else file_path.lower()

        if target_docs:
            return 1000  # Deterministic mode — always highest

        # Topic relevance check
        page_topic_keywords = set()
        if page_name:
            page_topic_keywords = set(page_name.lower().replace("-", " ").replace("_", " ").split())
            page_topic_keywords -= {"the", "a", "an", "and", "or", "for", "in", "of", "to"}

        doc_keywords = set(
            doc_name.replace(".md", "")
            .replace(".rst", "")
            .replace(".txt", "")
            .replace("-", " ")
            .replace("_", " ")
            .split()
        )
        topic_match = bool(page_topic_keywords & doc_keywords)
        if not topic_match and "readme" not in doc_name:
            score -= 50

        for kw in symbol_keywords:
            if kw in path_lower:
                score += 100
        for kw in page_keywords:
            if kw in path_lower:
                score += 50
        if "readme" in path_lower:
            score += 30
        if "/docs/" in path_lower or path_lower.startswith("docs/"):
            score += 20
        content_sample = content[:500].lower()
        for kw in symbol_keywords:
            if kw in content_sample:
                score += 10
        return score

    def _get_doc_nodes_fts5(
        self,
        fts,
        target_docs,
        page_name,
        target_symbols,
        page_keywords,
        symbol_keywords,
        max_docs,
    ):
        """Retrieve doc nodes via FTS5 index (O(log N))."""
        from ..constants import DOC_SYMBOL_TYPES as _DOC_TYPES

        doc_candidates = []
        seen_paths: set = set()

        if target_docs:
            # Deterministic: search by path
            for target in target_docs:
                rows = fts.search_by_path_prefix(
                    target,
                    symbol_types=_DOC_TYPES,
                    k=5,
                )
                for row in rows:
                    fp = row.get("rel_path") or row.get("file_path", "")
                    if fp in seen_paths:
                        continue
                    seen_paths.add(fp)
                    content = row.get("content") or row.get("docstring") or ""
                    if not content:
                        continue
                    score = self._score_doc_candidate(
                        fp,
                        content,
                        target_docs,
                        page_keywords,
                        symbol_keywords,
                        page_name,
                    )
                    doc_candidates.append((score, fp, row, content, row.get("symbol_type", "")))
            logger.info(f"[DOC_RETRIEVAL][FTS5] target_docs path search → {len(doc_candidates)} candidates")
        else:
            # Build a combined FTS query from page_name + target_symbols
            query_parts = []
            if page_name:
                query_parts.append(page_name)
            if target_symbols:
                query_parts.extend(target_symbols[:5])
            fts_query = " ".join(query_parts) if query_parts else "README documentation"

            docs = fts.search(fts_query, k=max_docs * 2, symbol_types=_DOC_TYPES)
            for doc in docs:
                fp = doc.metadata.get("rel_path") or doc.metadata.get("source", "")
                if fp in seen_paths:
                    continue
                seen_paths.add(fp)
                content = doc.page_content
                if not content:
                    continue
                score = self._score_doc_candidate(
                    fp,
                    content,
                    target_docs,
                    page_keywords,
                    symbol_keywords,
                    page_name,
                )
                doc_candidates.append((score, fp, doc.metadata, content, doc.metadata.get("symbol_type", "")))
            logger.info(f"[DOC_RETRIEVAL][FTS5] semantic query={fts_query!r:.60} → {len(doc_candidates)} candidates")
        return doc_candidates

    def _get_doc_nodes_brute(
        self,
        graph,
        target_docs,
        page_name,
        target_symbols,
        page_keywords,
        symbol_keywords,
        max_docs,
    ):
        """Retrieve doc nodes via O(N) graph scan (brute-force fallback)."""
        doc_candidates = []
        seen_paths: set = set()

        for _node_id, node_data in graph.nodes(data=True):
            symbol_type = node_data.get("symbol_type", "") or node_data.get("type", "")
            if hasattr(symbol_type, "value"):
                symbol_type = symbol_type.value
            symbol_type = str(symbol_type).lower()
            if symbol_type not in self.DOC_SYMBOL_TYPES:
                continue

            file_path = node_data.get("rel_path") or node_data.get("file_path", "")
            if not file_path:
                symbol_obj = node_data.get("symbol")
                if symbol_obj:
                    file_path = getattr(symbol_obj, "rel_path", "") or getattr(symbol_obj, "file_path", "")

            if target_docs:
                path_match = any(t in file_path or file_path.endswith(t) for t in target_docs)
                if not path_match:
                    continue

            if file_path in seen_paths:
                continue
            seen_paths.add(file_path)

            content = node_data.get("source_text", "")
            if not content:
                symbol_obj = node_data.get("symbol")
                if symbol_obj and hasattr(symbol_obj, "source_text"):
                    content = symbol_obj.source_text or ""
            if not content:
                content = node_data.get("content", "")
            if not content:
                continue

            score = self._score_doc_candidate(
                file_path,
                content,
                target_docs,
                page_keywords,
                symbol_keywords,
                page_name,
            )
            doc_candidates.append((score, file_path, node_data, content, symbol_type))

        logger.info(f"[DOC_RETRIEVAL][BRUTE] O(N) scan → {len(doc_candidates)} candidates")
        return doc_candidates

    def _get_docs_from_store_semantic(self, page_spec: "PageSpec", max_docs: int = 10) -> list[Document]:
        """
        Semantic doc retrieval using vector store + optional EmbeddingsFilter reranking.

        This method is used when WIKIS_DOC_SEPARATE_INDEX=1 (docs not in graph)
        or when WIKIS_DOC_SEMANTIC_RETRIEVAL=1 (prefer semantic over path-based).

        Args:
            page_spec: Page specification with retrieval_query, page_name, etc.
            max_docs: Maximum number of documentation documents to return

        Returns:
            List of Documents retrieved semantically from vector store
        """
        # Build query from page spec
        retrieval_query = getattr(page_spec, "retrieval_query", "")
        page_name = getattr(page_spec, "page_name", "")
        description = getattr(page_spec, "description", "")

        # Compose semantic query from available page info
        query_parts = [retrieval_query, page_name, description]
        query = " ".join(p for p in query_parts if p).strip()

        if not query:
            logger.warning("[SEMANTIC_DOC_RETRIEVAL] No query available for semantic doc retrieval")
            return []

        # Use centralized semantic doc retrieval from retriever stack
        # This applies EmbeddingsFilter reranking when USE_SEMANTIC_DOC_RETRIEVAL=1
        try:
            doc_candidates = self.retriever_stack.search_docs_semantic(
                query=query,
                k=max_docs,
                # Uses method default (0.7) — don't override here
            )

            if doc_candidates:
                logger.info(
                    f"[SEMANTIC_DOC_RETRIEVAL] Retrieved {len(doc_candidates)} docs "
                    f"for page '{page_name}' via retriever_stack.search_docs_semantic()"
                )

            return doc_candidates

        except Exception as e:
            logger.error(f"[SEMANTIC_DOC_RETRIEVAL] Error during semantic doc retrieval: {e}")
            return []

    # ------------------------------------------------------------------
    # Explicit target_docs fetcher (deterministic, path-based)
    # ------------------------------------------------------------------
    # Token cap for explicit target_docs on mixed pages.
    # Keeps total context (code + docs) under CONTEXT_TOKEN_BUDGET (50K).
    TARGET_DOC_TOKEN_CAP = 8_000

    def _fetch_explicit_target_docs(self, target_docs: list[str], token_cap: int = 0) -> list[Document]:
        """
        Fetch explicitly-assigned documentation files by **exact path** from
        the graph (via FTS5 or brute-force) — NOT semantic search.

        This is used on *mixed* pages that have both ``target_symbols`` and
        ``target_docs``.  Semantic search would return unpredictable docs and
        inflate token counts; path-based lookup is cheap and deterministic.

        A ``token_cap`` (default: ``TARGET_DOC_TOKEN_CAP``) limits the total
        characters collected so that the mixed page stays within budget.

        Falls back to reading the raw file from the cached repo when the doc
        is not found in the graph/FTS5 index.
        """
        if not target_docs:
            return []

        if token_cap <= 0:
            token_cap = self.TARGET_DOC_TOKEN_CAP

        # Approximate chars→tokens ratio (conservative: 1 token ≈ 3.5 chars)
        char_cap = int(token_cap * 3.5)
        collected: list[Document] = []
        total_chars = 0
        seen_paths: set = set()

        # ------ Strategy 1: FTS5 path lookup (O(log N)) ------
        fts = self.graph_text_index
        if fts is not None and fts.is_open:
            from ..constants import DOC_SYMBOL_TYPES as _DOC_TYPES

            for target in target_docs:
                if total_chars >= char_cap:
                    break
                rows = fts.search_by_path_prefix(
                    target,
                    symbol_types=_DOC_TYPES,
                    k=3,
                )
                for row in rows:
                    fp = row.get("rel_path") or row.get("file_path", "")
                    if fp in seen_paths:
                        continue
                    seen_paths.add(fp)
                    content = row.get("content") or row.get("docstring") or ""
                    if not content:
                        continue
                    # Trim if it would exceed cap
                    if total_chars + len(content) > char_cap:
                        content = content[: char_cap - total_chars]
                    total_chars += len(content)
                    collected.append(
                        Document(
                            page_content=content,
                            metadata={
                                "source": fp,
                                "file_path": fp,
                                "rel_path": fp,
                                "symbol": fp.split("/")[-1] if "/" in fp else fp,
                                "symbol_type": row.get("symbol_type", "document"),
                                "node_type": row.get("symbol_type", "document"),
                                "chunk_type": "documentation",
                                "language": "text",
                                "graph_retrieved": True,
                                "is_documentation": True,
                                "explicit_target_doc": True,
                            },
                        )
                    )

        # ------ Strategy 2: Graph brute-force for misses ------
        missed = [t for t in target_docs if not any(t in d.metadata.get("source", "") for d in collected)]
        if missed and total_chars < char_cap:
            graph = getattr(self.retriever_stack, "relationship_graph", None)
            if graph is None:
                graph = getattr(self.indexer, "relationship_graph", None)
            if graph is not None:
                for _node_id, node_data in graph.nodes(data=True):
                    if total_chars >= char_cap:
                        break
                    sym_type = str(node_data.get("symbol_type", "")).lower()
                    if hasattr(sym_type, "value"):
                        sym_type = sym_type.value
                    if sym_type not in self.DOC_SYMBOL_TYPES:
                        continue
                    fp = node_data.get("rel_path") or node_data.get("file_path", "")
                    if fp in seen_paths:
                        continue
                    if not any(t in fp or fp.endswith(t) for t in missed):
                        continue
                    seen_paths.add(fp)
                    content = (
                        node_data.get("source_text", "")
                        or (getattr(node_data.get("symbol"), "source_text", "") or "")
                        or node_data.get("content", "")
                    )
                    if not content:
                        continue
                    if total_chars + len(content) > char_cap:
                        content = content[: char_cap - total_chars]
                    total_chars += len(content)
                    collected.append(
                        Document(
                            page_content=content,
                            metadata={
                                "source": fp,
                                "file_path": fp,
                                "rel_path": fp,
                                "symbol": fp.split("/")[-1] if "/" in fp else fp,
                                "symbol_type": sym_type,
                                "node_type": sym_type,
                                "chunk_type": "documentation",
                                "language": "text",
                                "graph_retrieved": True,
                                "is_documentation": True,
                                "explicit_target_doc": True,
                            },
                        )
                    )

        # ------ Strategy 3: Raw file read from cached repo on disk ------
        # When SEPARATE_DOC_INDEX=1, docs are NOT in the graph or FTS5 — they
        # only exist in the vector store (semantic).  But for *explicit* target_docs
        # we want deterministic path-based retrieval, not semantic matching.
        # Reading the raw file from the cloned repo is the most reliable fallback.
        still_missed = [t for t in target_docs if not any(t in d.metadata.get("source", "") for d in collected)]
        if still_missed and total_chars < char_cap:
            repo_root = None
            repo_root_getter = getattr(self.indexer, "get_repo_root", None)
            if callable(repo_root_getter):
                try:
                    repo_root = repo_root_getter()
                except Exception:  # noqa: S110
                    pass
            if repo_root and os.path.isdir(repo_root):
                for target in still_missed:
                    if total_chars >= char_cap:
                        break
                    # target may be a relative path like ".github/workflows/ci.yml"
                    # or a bare filename like "docker-compose.yml"
                    candidate_paths = [os.path.join(repo_root, target)]
                    # Also try walking for bare filenames
                    if "/" not in target:
                        for root_dir, _, files in os.walk(repo_root):
                            if target in files:
                                candidate_paths.append(os.path.join(root_dir, target))
                                break  # first match is enough
                    for cpath in candidate_paths:
                        if not os.path.isfile(cpath):
                            continue
                        try:
                            with open(cpath, encoding="utf-8", errors="replace") as f:
                                content = f.read()
                        except Exception:  # noqa: S112
                            continue
                        if not content:
                            continue
                        rel = os.path.relpath(cpath, repo_root)
                        if rel in seen_paths:
                            continue
                        seen_paths.add(rel)
                        # Determine doc_type from extension
                        ext = os.path.splitext(rel)[1].lower()
                        from ..constants import DOCUMENTATION_EXTENSIONS as _DOC_EXT_MAP

                        doc_type = _DOC_EXT_MAP.get(ext, "document")
                        sym_type = f"{doc_type}_document"
                        if total_chars + len(content) > char_cap:
                            content = content[: char_cap - total_chars]
                        total_chars += len(content)
                        collected.append(
                            Document(
                                page_content=content,
                                metadata={
                                    "source": rel,
                                    "file_path": rel,
                                    "rel_path": rel,
                                    "symbol": os.path.basename(rel),
                                    "symbol_type": sym_type,
                                    "node_type": sym_type,
                                    "chunk_type": "documentation",
                                    "language": "text",
                                    "graph_retrieved": False,
                                    "is_documentation": True,
                                    "explicit_target_doc": True,
                                    "disk_read": True,
                                },
                            )
                        )
                        logger.debug(f"[EXPLICIT_DOCS][DISK] Read '{rel}' ({len(content):,} chars)")
                        break  # found this target, move to next

        if collected:
            disk_count = sum(1 for d in collected if d.metadata.get("disk_read"))
            source_info = f" ({disk_count} from disk)" if disk_count else ""
            logger.info(
                f"[EXPLICIT_DOCS] Fetched {len(collected)} target_docs{source_info} "
                f"({total_chars:,} chars / ~{total_chars // 4:,} tokens, "
                f"cap={token_cap:,} tokens): "
                f"{[d.metadata['source'] for d in collected]}"
            )
        else:
            logger.warning(
                f"[EXPLICIT_DOCS] Could not find any of {len(target_docs)} "
                f"target_docs in graph/FTS5/disk: {target_docs}"
            )

        return collected

    def _get_relevant_content_for_page(self, page_spec: PageSpec, repo_url: str, repo_context: str) -> dict[str, Any]:
        """
        Get relevant content for page generation with proper context structure.

        Three-path routing model (broad-page optimised):
        ─────────────────────────────────────────────────
        1. CODE-ONLY  — target_symbols present, NO target_docs
                        → Retrieve code symbols only.
        2. MIXED      — target_symbols AND target_docs both present
                        → Code symbols + explicit path-based doc fetch
                          (deterministic, capped at TARGET_DOC_TOKEN_CAP).
        3. DOC-ONLY   — No target_symbols (or code retrieval found nothing)
                        → Semantic (or graph-based) doc retrieval.

        Why not always mix?  With 10–30 symbols/page the code context alone is
        30–90 K tokens.  Adding 10 semantic docs (~20 K) pushes total past 50 K
        into hierarchical mode, which compresses the docs away — the page-writer
        LLM never sees them intact.  The three-path model keeps most pages in
        simple mode while still surfacing the docs the planner explicitly chose.
        """

        # ── Cluster-expansion fast path ─────────────────────────────────
        # When the cluster planner produced this page, we can retrieve
        # content directly from the unified DB with cluster-bounded
        # expansion — no NX graph needed.
        cluster_docs = self._try_cluster_expansion(page_spec)
        if cluster_docs is not None:
            return cluster_docs

        # GRAPH-FIRST RETRIEVAL: Use target_symbols for precise code lookup if available
        target_symbols = getattr(page_spec, "target_symbols", []) or []
        target_docs = getattr(page_spec, "target_docs", []) or []

        graph_docs: list[Document] = []

        # ── Path 1 & 2: pages with target_symbols ──────────────────────
        has_symbols = bool(target_symbols)
        code_found = False

        if has_symbols:
            logger.info(
                f"[CONTENT_RETRIEVAL] Using graph-first retrieval with {len(target_symbols)} target symbols: {target_symbols[:5]}..."
            )
            target_folders = getattr(page_spec, "target_folders", []) or []
            code_docs = self._get_docs_by_target_symbols(target_symbols, max_hops=1, target_folders=target_folders)
            graph_docs.extend(code_docs)
            code_found = bool(code_docs)

            if code_found and target_docs:
                # ── MIXED PAGE: code symbols + explicit target_docs ──
                # Fetch only the docs the planner explicitly assigned (by path),
                # capped so the total stays comfortably within simple-mode budget.
                explicit_docs = self._fetch_explicit_target_docs(target_docs)
                if explicit_docs:
                    graph_docs.extend(explicit_docs)
                logger.info(
                    f"[CONTENT_RETRIEVAL] Mixed page '{page_spec.page_name}': "
                    f"{len(code_docs)} code docs + {len(explicit_docs)} explicit target_docs "
                    f"from {len(target_symbols)} symbols"
                )
            elif code_found:
                # ── CODE-ONLY PAGE: symbols produced results, no target_docs ──
                logger.info(
                    f"[CONTENT_RETRIEVAL] Code-only page '{page_spec.page_name}': "
                    f"{len(code_docs)} code documents from {len(target_symbols)} symbols"
                )
            else:
                # Symbols specified but no code found — fall through to doc path
                logger.warning(
                    f"[CONTENT_RETRIEVAL] target_symbols specified but no code found for "
                    f"'{page_spec.page_name}', falling back to doc retrieval"
                )
                has_symbols = False  # Treat as doc-focused

        # ── Path 3: doc-focused pages ────────────────────────────────
        if not has_symbols or not code_found:
            # DOC-FOCUSED PAGE: No symbols or symbols produced nothing → use doc retrieval
            #
            # Prefer explicit target_docs (path-based, deterministic) over
            # semantic retrieval.  The structure planner already assigned
            # relevant doc files — use them directly instead of hoping the
            # vector store surfaces the same files.
            if target_docs:
                explicit_docs = self._fetch_explicit_target_docs(target_docs)
                if explicit_docs:
                    graph_docs.extend(explicit_docs)
                    logger.info(
                        f"[DOC_RETRIEVAL] Doc-focused page '{page_spec.page_name}': "
                        f"using {len(explicit_docs)} explicitly assigned target_docs"
                    )

            # If explicit fetch produced nothing, fall back to semantic/graph
            if not graph_docs:
                if SEPARATE_DOC_INDEX or USE_SEMANTIC_DOC_RETRIEVAL:
                    # Semantic retrieval from vector store
                    doc_nodes = self._get_docs_from_store_semantic(page_spec, max_docs=10)
                    if doc_nodes:
                        logger.info(
                            f"[DOC_RETRIEVAL] Doc-focused page: adding {len(doc_nodes)} docs from semantic retrieval"
                        )
                        graph_docs.extend(doc_nodes)
                else:
                    # Legacy: Retrieve documentation nodes from graph
                    doc_nodes = self._get_doc_nodes_from_graph(
                        target_docs=target_docs if target_docs else None,
                        page_name=page_spec.page_name,
                        target_symbols=target_symbols,
                        max_docs=10,
                    )
                    if doc_nodes:
                        logger.info(
                            f"[DOC_RETRIEVAL] Doc-focused page: adding {len(doc_nodes)} documentation nodes from graph"
                        )
                        graph_docs.extend(doc_nodes)

        if graph_docs:
            # Check if graph results fit within budget
            token_counter = get_token_counter()
            graph_tokens = token_counter.count_documents(graph_docs)

            if graph_tokens <= CONTEXT_TOKEN_BUDGET:
                utilization_pct = (graph_tokens / CONTEXT_TOKEN_BUDGET) * 100
                code_count = sum(1 for d in graph_docs if d.metadata.get("chunk_type") in ("code", "symbol"))
                doc_count = sum(1 for d in graph_docs if d.metadata.get("is_documentation"))
                logger.info(
                    f"✅ GRAPH-FIRST SUCCESS: {len(graph_docs)} documents ({code_count} code, {doc_count} docs) fit budget "
                    f"({graph_tokens:,}/{CONTEXT_TOKEN_BUDGET:,} tokens, {utilization_pct:.1f}% utilized)"
                )
                return self._format_simple_context(graph_docs, page_spec)
            else:
                logger.info(
                    f"[GRAPH_RETRIEVAL] Graph results exceed budget ({graph_tokens:,}/{CONTEXT_TOKEN_BUDGET:,} tokens), "
                    f"applying ranked truncation"
                )
                truncated_docs, _trunc_metrics = self._ranked_truncation(graph_docs, page_spec, CONTEXT_TOKEN_BUDGET)
                logger.info(
                    f"[TOKEN_USAGE] page='{page_spec.page_name}' "
                    f"docs_retrieved={_trunc_metrics['docs_retrieved']} "
                    f"docs_included={_trunc_metrics['docs_included']} "
                    f"docs_dropped={_trunc_metrics['docs_dropped']} "
                    f"tokens_before={_trunc_metrics['tokens_before']:,} "
                    f"tokens_after={_trunc_metrics['tokens_after']:,} "
                    f"budget={_trunc_metrics['token_budget']:,} "
                    f"utilization={_trunc_metrics['utilization_pct']}%"
                )
                return self._format_simple_context(truncated_docs, page_spec)
        else:
            if target_symbols:
                logger.warning(
                    f"[GRAPH_RETRIEVAL] No documents found for target_symbols={target_symbols}, falling back to vector search"
                )
            else:
                logger.info(
                    f"[CONTENT_RETRIEVAL] No target_symbols for page '{page_spec.page_name}', using vector search fallback"
                )

        # FALLBACK: Use retrieval_query for vector search (legacy behavior)
        if hasattr(page_spec, "retrieval_query") and page_spec.retrieval_query.strip():
            query = page_spec.retrieval_query
        else:
            # Fallback to component-based query building
            query_parts = [page_spec.page_name, page_spec.description, page_spec.content_focus]

            # Add location-specific terms to improve retrieval
            if hasattr(page_spec, "target_folders") and page_spec.target_folders:
                folder_terms = [folder.split("/")[-1] for folder in page_spec.target_folders if folder]
                query_parts.extend(folder_terms)

            if hasattr(page_spec, "key_files") and page_spec.key_files:
                file_terms = [
                    file.split("/")[-1].replace(".py", "").replace(".js", "") for file in page_spec.key_files if file
                ]
                query_parts.extend(file_terms)

            query = " ".join(query_parts)

        try:
            # Retrieve expanded documents from ContentExpander (may return 100-140 docs)
            relevant_docs = self.retriever_stack.search_repository(query, k=50)

            logger.info(f"Retrieved {len(relevant_docs)} expanded documents for page '{page_spec.page_name}'")

            # DECISION POINT: Simple Mode vs Hierarchical Mode based on TOKEN BUDGET
            # Use accurate tiktoken counting instead of arbitrary document count
            token_counter = get_token_counter()
            total_tokens = token_counter.count_documents(relevant_docs)

            tokens_remaining = CONTEXT_TOKEN_BUDGET - total_tokens
            utilization_pct = (total_tokens / CONTEXT_TOKEN_BUDGET) * 100

            logger.info(
                f"Total context tokens: {total_tokens:,} (budget: {CONTEXT_TOKEN_BUDGET:,}, "
                f"using {'tiktoken' if token_counter.is_accurate else 'fallback estimation'})"
            )

            if total_tokens <= CONTEXT_TOKEN_BUDGET:
                # SIMPLE MODE: All documents fit within token budget - use full content
                logger.info(
                    f"✅ ONESHOT MODE ELIGIBLE: {len(relevant_docs)} documents fit within token budget "
                    f"({total_tokens:,}/{CONTEXT_TOKEN_BUDGET:,} tokens, {utilization_pct:.1f}% utilized, "
                    f"{tokens_remaining:,} tokens remaining)"
                )
                return self._format_simple_context(relevant_docs, page_spec)
            else:
                # RANKED TRUNCATION: Include top-priority docs with full content, drop the rest
                tokens_over = total_tokens - CONTEXT_TOKEN_BUDGET
                logger.info(
                    f"⚠️ RANKED TRUNCATION: {len(relevant_docs)} documents exceed budget "
                    f"({total_tokens:,}/{CONTEXT_TOKEN_BUDGET:,} tokens, {tokens_over:,} over) - "
                    f"including top-priority docs with full content, dropping rest"
                )
                truncated_docs, _trunc_metrics = self._ranked_truncation(relevant_docs, page_spec, CONTEXT_TOKEN_BUDGET)
                logger.info(
                    f"[TOKEN_USAGE] page='{page_spec.page_name}' "
                    f"docs_retrieved={_trunc_metrics['docs_retrieved']} "
                    f"docs_included={_trunc_metrics['docs_included']} "
                    f"docs_dropped={_trunc_metrics['docs_dropped']} "
                    f"tokens_before={_trunc_metrics['tokens_before']:,} "
                    f"tokens_after={_trunc_metrics['tokens_after']:,} "
                    f"budget={_trunc_metrics['token_budget']:,} "
                    f"utilization={_trunc_metrics['utilization_pct']}%"
                )
                return self._format_simple_context(truncated_docs, page_spec)

        except Exception as e:
            page_name = page_spec.page_name if page_spec else "unknown"
            logger.error(
                f"Retrieval FAILED for page '{page_name}': {e}. "
                f"LLM will receive only repo summary — content quality will be degraded."
            )
            return {
                "content": repo_context,
                "files": [],
                "location_matches": [],
                "prioritized_sources": 0,
                "documentation_files": 0,
                "code_files": 0,
                "retrieval_failed": True,
            }

    def _dispatch_progress(self, event_type: str, data: dict[str, Any]):
        """Dispatch progress event if tracking is enabled"""
        if self.enable_progress_tracking:
            self.dispatch_event(event_type, data)

    def _extract_full_readme_content_from_docs(self, documents: list[Any]) -> str:
        """Extract full README content by combining all README chunks from documents"""
        readme_docs = [doc for doc in documents if "readme" in doc.metadata.get("source", "").lower()]

        if not readme_docs:
            return "No README file found"

        # Group by source file and sort by chunk order if available
        readme_by_file = {}
        for doc in readme_docs:
            source = doc.metadata.get("source", "")
            if source not in readme_by_file:
                readme_by_file[source] = []
            readme_by_file[source].append(doc)

        # Use the first README file found (usually README.md)
        first_readme_source = list(readme_by_file.keys())[0]
        readme_chunks = readme_by_file[first_readme_source]

        # Sort by section_id if available (for proper reconstruction)
        readme_chunks.sort(key=lambda x: x.metadata.get("section_id", 0))

        # Combine all chunks
        full_content = "\\n\\n".join([chunk.page_content for chunk in readme_chunks])
        return full_content  # Limit to first 5000 chars for LLM context

    def _extract_full_readme_content_from_files(self, file_paths: list[str]) -> str:
        """Extract full README content directly from repository files when available"""
        if not file_paths:
            return "No README file found"

        from pathlib import Path

        readme_candidates = [path for path in file_paths if Path(path).name.lower().startswith("readme")]

        if not readme_candidates:
            return "No README file found"

        readme_path = sorted(readme_candidates, key=len)[0]
        traverser = getattr(self.indexer, "repository_traverser", None)
        if not traverser:
            return "No README file found"

        content = traverser.get_file_content(readme_path)
        return content if content else "No README file found"

    def _get_repository_file_paths(self) -> list[str]:
        """Return repository file paths using indexer/traverser cache when possible"""
        if not self.indexer:
            return []

        get_files = getattr(self.indexer, "get_repository_files", None)
        if callable(get_files):
            try:
                return get_files()
            except Exception as e:
                logger.warning(f"Failed to fetch repository files from indexer: {e}")

        traverser = getattr(self.indexer, "repository_traverser", None)
        if traverser:
            try:
                return traverser.discover_files()
            except Exception as e:
                logger.warning(f"Failed to discover repository files: {e}")

        # Filesystem indexer path (local clone)
        repo_root_getter = getattr(self.indexer, "get_repo_root", None)
        repo_root = repo_root_getter() if callable(repo_root_getter) else None
        if repo_root and os.path.isdir(repo_root):
            filter_manager = getattr(self.indexer, "filter_manager", None)
            file_paths = []
            for root, dirs, files in os.walk(repo_root):
                if filter_manager:
                    dirs_to_remove = []
                    for dir_name in dirs:
                        dir_path = os.path.relpath(os.path.join(root, dir_name), repo_root)
                        if not filter_manager.should_process_directory(dir_path):
                            dirs_to_remove.append(dir_name)
                    for dir_name in dirs_to_remove:
                        dirs.remove(dir_name)

                for file_name in files:
                    rel_path = os.path.relpath(os.path.join(root, file_name), repo_root)
                    if filter_manager and not filter_manager.should_process_file(rel_path):
                        continue
                    file_paths.append(rel_path)

            if file_paths:
                return file_paths

        return []

    def _create_repository_tree(self, file_paths: list[str], max_files_per_dir: int = 25) -> str:
        """Create a repository tree with directory counts and sampled file listings"""
        if not file_paths:
            return "No files found"

        unique_paths = sorted(set(path for path in file_paths if path))
        dir_map: dict[str, list[str]] = {}
        root_files: list[str] = []

        for path in unique_paths:
            if "/" in path:
                dir_path, file_name = path.rsplit("/", 1)
                dir_map.setdefault(dir_path, []).append(file_name)
            else:
                root_files.append(path)

        tree_lines = [f"Total files: {len(unique_paths)}", f"Showing up to {max_files_per_dir} files per directory"]

        if root_files:
            tree_lines.append("📁 /")
            for file_name in sorted(root_files)[:max_files_per_dir]:
                tree_lines.append(f"  📄 {file_name}")
            if len(root_files) > max_files_per_dir:
                tree_lines.append(f"  ... (+{len(root_files) - max_files_per_dir} more)")

        for dir_path in sorted(dir_map.keys()):
            files = sorted(dir_map[dir_path])
            tree_lines.append(f"📁 {dir_path}/ ({len(files)} files)")
            for file_name in files[:max_files_per_dir]:
                tree_lines.append(f"  📄 {file_name}")
            if len(files) > max_files_per_dir:
                tree_lines.append(f"  ... (+{len(files) - max_files_per_dir} more)")

        return "\n".join(tree_lines)

    def _llm_analyze_repository(
        self,
        repository_tree: str,
        readme_content: str,
        code_samples: str,
        file_paths: list[str],
        config: RunnableConfig,
    ) -> str:
        """Use LLM to analyze repository architecture, patterns, and technologies.

        Returns:
            If USE_STRUCTURED_REPO_ANALYSIS is True: JSON string (~5-8K chars)
            If USE_STRUCTURED_REPO_ANALYSIS is False: Markdown string (~67K chars)
        """

        # Prepare file statistics
        file_stats = self._prepare_basic_file_stats(file_paths)

        # Choose prompt based on toggle
        if USE_STRUCTURED_REPO_ANALYSIS:
            # New structured JSON output - ~10x smaller, indexable
            analysis_prompt = ChatPromptTemplate.from_messages(
                [
                    (
                        "system",
                        "You are an expert repository analyst. Output ONLY valid JSON matching the specified schema. No markdown, no explanations - just the JSON object.",
                    ),
                    ("human", STRUCTURED_REPO_ANALYSIS_PROMPT),
                ]
            )
            logger.info("Using STRUCTURED_REPO_ANALYSIS_PROMPT (JSON output, ~5-8K chars)")
        else:
            # Legacy markdown output - comprehensive but verbose
            analysis_prompt = ChatPromptTemplate.from_messages(
                [
                    (
                        "system",
                        "You are an expert software architect and comprehensive repository analysis specialist. Your mission is to perform exhaustive analysis of codebases, documenting every significant component without omission. You must analyze ALL architectural patterns, technologies, and relationships present in the repository to enable complete documentation coverage.",
                    ),
                    ("human", ENHANCED_REPO_ANALYSIS_PROMPT),
                ]
            )
            logger.info("Using ENHANCED_REPO_ANALYSIS_PROMPT (Markdown output, ~67K chars)")

        logger.info("Invoking LLM for repository analysis...")

        try:
            response = self.llm.invoke(
                analysis_prompt.format_messages(
                    repository_name=self.repository_url,
                    branch_name=self.branch,
                    repository_tree=repository_tree,
                    readme_content=readme_content if readme_content else "No README found",
                    code_samples=code_samples,
                    file_stats=file_stats,
                ),
                config=config,
            )

            analysis_text = response.content

            # Log format-specific info
            if USE_STRUCTURED_REPO_ANALYSIS:
                logger.info(f"Structured JSON analysis completed: {len(analysis_text)} characters")
                # Validate JSON (log warning if invalid, but don't fail)
                try:
                    import json

                    json.loads(analysis_text)
                    logger.info("JSON validation passed")
                except json.JSONDecodeError as e:
                    logger.warning(f"JSON validation failed: {e}. LLM may have included extra text.")
            else:
                logger.info(f"LLM repository analysis completed: {len(analysis_text)} characters")

            return analysis_text

        except Exception as e:
            logger.error(f"LLM repository analysis failed: {e}")
            raise

    def _extract_representative_code_samples(self, all_documents: list[Any]) -> str:
        """Extract representative code samples for LLM analysis"""

        code_samples = []

        # Get samples from different file types
        python_files = [doc for doc in all_documents if doc.metadata.get("source", "").endswith(".py")]
        config_files = [
            doc
            for doc in all_documents
            if any(
                ext in doc.metadata.get("source", "").lower()
                for ext in ["yaml", "json", "toml", "ini", "proto", "tf", "gradle", "wsdl", "xsd"]
            )
        ]
        doc_files = [doc for doc in all_documents if doc.metadata.get("source", "").endswith(".md")]

        # Sample Python code (max 3 files)
        for doc in python_files[:3]:
            source = doc.metadata.get("source", "unknown")
            content = doc.page_content[:800]  # First 800 chars
            code_samples.append(f"=== {source} ===\\n{content}\\n")

        # Sample configuration files (max 2)
        for doc in config_files[:2]:
            source = doc.metadata.get("source", "unknown")
            content = doc.page_content[:400]  # First 400 chars
            code_samples.append(f"=== {source} ===\\n{content}\\n")

        # Sample documentation files (max 2, excluding README)
        for doc in doc_files[:2]:
            source = doc.metadata.get("source", "unknown")
            if "readme" not in source.lower():
                content = doc.page_content[:600]  # First 600 chars
                code_samples.append(f"=== {source} ===\\n{content}\\n")

        combined_samples = "\\n".join(code_samples)

        # Limit total length to avoid context overflow
        if len(combined_samples) > 4000:
            combined_samples = combined_samples[:4000] + "\\n... [truncated for context length]"

        return combined_samples if combined_samples else "No representative code samples available"

    def _extract_representative_code_samples_from_files(self, file_paths: list[str]) -> str:
        """Extract representative code samples from repository files directly."""
        if not file_paths:
            return "No representative code samples available"

        repo_root_getter = getattr(self.indexer, "get_repo_root", None)
        repo_root = repo_root_getter() if callable(repo_root_getter) else None
        traverser = getattr(self.indexer, "repository_traverser", None)

        def read_file_content(rel_path: str, max_chars: int) -> str:
            try:
                if repo_root:
                    abs_path = os.path.join(repo_root, rel_path)
                    if os.path.isfile(abs_path):
                        with open(abs_path, encoding="utf-8", errors="replace") as handle:
                            return handle.read(max_chars)
                if traverser:
                    content = traverser.get_file_content(rel_path)
                    if isinstance(content, str):
                        return content[:max_chars]
            except Exception:
                return ""
            return ""

        normalized_paths = [p for p in file_paths if isinstance(p, str) and p]

        python_files = [p for p in normalized_paths if p.lower().endswith(".py")]
        config_files = [
            p
            for p in normalized_paths
            if any(
                p.lower().endswith(ext)
                for ext in [
                    ".yaml",
                    ".yml",
                    ".json",
                    ".toml",
                    ".ini",
                    ".proto",
                    ".tf",
                    ".tfvars",
                    ".gradle",
                    ".kts",
                    ".wsdl",
                    ".xsd",
                    ".hcl",
                ]
            )
        ]
        doc_files = [p for p in normalized_paths if p.lower().endswith(".md")]

        code_samples = []

        for rel_path in sorted(python_files)[:3]:
            content = read_file_content(rel_path, 800)
            if content:
                code_samples.append(f"=== {rel_path} ===\n{content}\n")

        for rel_path in sorted(config_files)[:2]:
            content = read_file_content(rel_path, 400)
            if content:
                code_samples.append(f"=== {rel_path} ===\n{content}\n")

        for rel_path in sorted(doc_files)[:2]:
            if "readme" in rel_path.lower():
                continue
            content = read_file_content(rel_path, 600)
            if content:
                code_samples.append(f"=== {rel_path} ===\n{content}\n")

        combined_samples = "\n".join(code_samples)

        if len(combined_samples) > 4000:
            combined_samples = combined_samples[:4000] + "\n... [truncated for context length]"

        return combined_samples if combined_samples else "No representative code samples available"

    def _prepare_basic_file_stats(self, file_paths: list[str]) -> str:
        """Prepare basic file statistics for LLM analysis"""

        # Count file types
        file_types = {}
        for path in file_paths:
            if "." in path:
                ext = path.split(".")[-1].lower()
                file_types[ext] = file_types.get(ext, 0) + 1

        # Sort by count
        sorted_types = sorted(file_types.items(), key=lambda x: x[1], reverse=True)

        stats_lines = [f"Total Files: {len(file_paths)}", "File Type Distribution:"]

        for ext, count in sorted_types[:50]:  # Top 10 file types
            stats_lines.append(f"  - .{ext}: {count} files")

        return "\\n".join(stats_lines)

    def _is_documentation_file(self, file_path: str) -> bool:
        """Check if file is documentation (uses constants.py as single source of truth)"""
        doc_names = {"readme", "changelog", "license", "contributing", "docs"}

        path_lower = file_path.lower()

        # Check extension against full documentation set
        from pathlib import Path as _P

        ext = _P(file_path).suffix.lower()
        if ext in DOCUMENTATION_EXTENSIONS_SET:
            return True

        # Check known filenames (Makefile, Dockerfile, etc.)
        filename = _P(file_path).name
        if filename in KNOWN_FILENAMES:
            return True

        # Check filename stem
        if _P(file_path).stem.lower() in doc_names:
            return True

        # Check if in docs directory
        if "/docs/" in path_lower or "/documentation/" in path_lower:
            return True

        return False

    def _extract_imports_for_file(self, file_path: str, content: str) -> str:
        """Extract import statements for a file using graph data first, regex fallback.

        Priority order:
        1. Graph file_imports edges (language-agnostic, complete file-level imports)
        2. Graph symbol import edges (per-symbol imports from relationships)
        3. Regex extraction from content (legacy fallback for supported languages)
        """
        # Try graph-based extraction first (works for ALL languages)
        graph_imports = self._get_imports_from_graph(file_path)
        if graph_imports:
            return graph_imports

        # Regex fallback for when graph is unavailable
        return self._extract_imports_regex(file_path, content)

    def _get_imports_from_graph(self, file_path: str) -> str:
        """Get imports for a file from graph edges and file_imports data.

        Checks two sources:
        1. 'file_imports' edges: file-level imports stored by the parser
        2. 'imports' edges: per-symbol import relationships

        Returns formatted import string or empty string if no graph available.
        """
        graph = getattr(self.retriever_stack, "relationship_graph", None)
        if not graph:
            graph = getattr(self.indexer, "relationship_graph", None)
        if not graph:
            return ""

        imports = set()

        try:
            for node_id, data in graph.nodes(data=True):
                # Match nodes from the same file
                node_file = data.get("rel_path", "") or data.get("file_path", "")
                if not node_file or node_file != file_path:
                    continue

                # Collect imports from node's stored imports attribute
                node_imports = data.get("imports", [])
                if isinstance(node_imports, (list, set)):
                    for imp in node_imports:
                        if imp and isinstance(imp, str):
                            imports.add(imp)

                # Check outgoing edges for import relationships
                try:
                    for _, target, edge_data in graph.out_edges(node_id, data=True):
                        edge_type = edge_data.get("relationship_type", "") or edge_data.get("type", "")
                        if edge_type in ("imports", "file_imports"):
                            target_data = graph.nodes.get(target, {})
                            target_name = target_data.get("symbol_name") or target_data.get("name") or target
                            if target_name and isinstance(target_name, str):
                                imports.add(target_name)
                except Exception:  # noqa: S110
                    pass  # Skip if graph traversal fails for this node
        except Exception:
            return ""

        if imports:
            return "\n".join(sorted(imports))
        return ""

    def _extract_imports_regex(self, file_path: str, content: str) -> str:
        """Extract import statements from code content using regex patterns (legacy fallback)"""
        from pathlib import Path

        # Determine language from file extension
        extension = Path(file_path).suffix.lower()

        # Define import patterns for different languages
        import_patterns = {
            ".py": [
                r"^from\s+([^\s]+)\s+import\s+(.+)$",
                r"^import\s+([^\s,]+)(?:\s+as\s+[^\s,]+)?(?:\s*,\s*[^\s,]+(?:\s+as\s+[^\s,]+)?)*$",
            ],
            ".js": [
                r'^import\s+.*\s+from\s+[\'"]([^\'"]+)[\'"]',
                r'^const\s+.*\s+=\s+require\([\'"]([^\'"]+)[\'"]\)',
                r'^import\s+[\'"]([^\'"]+)[\'"]',
            ],
            ".jsx": [
                r'^import\s+.*\s+from\s+[\'"]([^\'"]+)[\'"]',
                r'^const\s+.*\s+=\s+require\([\'"]([^\'"]+)[\'"]\)',
                r'^import\s+[\'"]([^\'"]+)[\'"]',
            ],
            ".ts": [
                r'^import\s+.*\s+from\s+[\'"]([^\'"]+)[\'"]',
                r'^const\s+.*\s+=\s+require\([\'"]([^\'"]+)[\'"]\)',
                r'^import\s+[\'"]([^\'"]+)[\'"]',
            ],
            ".tsx": [
                r'^import\s+.*\s+from\s+[\'"]([^\'"]+)[\'"]',
                r'^const\s+.*\s+=\s+require\([\'"]([^\'"]+)[\'"]\)',
                r'^import\s+[\'"]([^\'"]+)[\'"]',
            ],
            ".java": [r"^import\s+([^\s;]+);?$", r"^import\s+static\s+([^\s;]+);?$"],
        }

        patterns = import_patterns.get(extension, [])
        if not patterns:
            return "Import patterns not supported for this file type."

        imports = []
        lines = content.split("\n")

        for line in lines:
            line = line.strip()
            if not line:
                continue

            # Check against import patterns for the language
            for pattern in patterns:
                if re.match(pattern, line):
                    imports.append(line)
                    break

        if not imports:
            return "No imports found in this file."

        return "\n".join(imports)

    # ---------------- Phase 2 Helpers: Conceptual Summary & Adaptive Page Budget ----------------

    # def _get_env_flag(self, name: str, default: bool = False) -> bool:
    #     import os
    #     val = os.getenv(name)
    #     if val is None:
    #         return default
    #     return val.lower() in {"1", "true", "yes", "on"}
    #
    # def _parse_repository_tree_stats(self, repository_tree: str) -> Dict[str, int]:
    #     """Parse stats footer produced by _create_repository_tree."""
    #     if not repository_tree or "--- TREE_STATS ---" not in repository_tree:
    #         return {}
    #     stats: Dict[str, int] = {}
    #     try:
    #         lines = repository_tree.splitlines()
    #         start = lines.index("--- TREE_STATS ---") + 1
    #         for line in lines[start:]:
    #             if not line.strip() or ':' not in line:
    #                 continue
    #             key, val = line.split(':', 1)
    #             key = key.strip()
    #             val = val.strip()
    #             try:
    #                 stats[key] = int(val)
    #             except ValueError:
    #                 # ignore non-int for numeric dict, could be extended later
    #                 pass
    #         return stats
    #     except Exception as e:
    #         logger.warning(f"Failed to parse repository tree stats: {e}")
    #         return {}
    #
    # def _generate_conceptual_summary(self, repository_tree: str, readme_content: str, repo_analysis_text: str, config: RunnableConfig):
    #     """Invoke LLM to produce conceptual repository summary (raw + parsed)."""
    #     logger.info("Generating conceptual repository summary (Phase 2)...")
    #     prompt = ChatPromptTemplate.from_messages([
    #         ("system", "You are an expert software architect producing structured conceptual JSON summary."),
    #         ("human", CONCEPTUAL_REPOSITORY_SUMMARY_PROMPT)
    #     ])
    #     response = self.llm.invoke(
    #         prompt.format_messages(
    #             repository_tree=repository_tree,
    #             readme_content=readme_content or "",
    #             repo_analysis=repo_analysis_text or ""
    #         ),
    #         config=config
    #     )
    #     raw = response.content.strip()
    #     parsed = None
    #     try:
    #         if raw.startswith('```'):
    #             raw_clean = re.sub(r'^```[a-zA-Z]*\n|```$', '', raw, flags=re.MULTILINE).strip()
    #         else:
    #             raw_clean = raw
    #         data = json.loads(raw_clean)
    #         parsed = ConceptualRepositorySummary(**data)
    #     except Exception as e:
    #         logger.warning(f"Failed to parse conceptual summary JSON: {e}")
    #     return raw, parsed
    #
    # def _compute_adaptive_page_budget(self, repo_analysis_obj: RepositoryAnalysis, tree_stats: Dict[str, int], conceptual_summary: Optional[ConceptualRepositorySummary]):
    #     """Compute adaptive page budget using LLM prompt with fallback heuristic."""
    #     logger.info("Computing adaptive page budget (Phase 2)...")
    #     min_pages = 5
    #     max_pages = 55
    #     total_files = repo_analysis_obj.total_files if repo_analysis_obj else 0
    #     total_symbols = getattr(repo_analysis_obj, 'total_symbols', 0)
    #     languages = repo_analysis_obj.programming_languages if repo_analysis_obj else []
    #     project_complexity = repo_analysis_obj.project_complexity if repo_analysis_obj else "unknown"
    #     max_depth = tree_stats.get('max_depth', 0)
    #     est_tree_tokens = tree_stats.get('est_tree_tokens', 0)
    #     has_readme = repo_analysis_obj.has_readme if repo_analysis_obj else False
    #     component_count = len(conceptual_summary.major_components) if conceptual_summary else 0
    #     data_flow_count = len(conceptual_summary.data_flows) if conceptual_summary else 0
    #
    #     prompt = ChatPromptTemplate.from_messages([
    #         ("system", "You are a documentation planning specialist"),
    #         ("human", ADAPTIVE_PAGE_BUDGET_PROMPT)
    #     ])
    #     try:
    #         response = self.llm.invoke(
    #             prompt.format_messages(
    #                 total_files=total_files,
    #                 total_symbols=total_symbols,
    #                 languages=languages,
    #                 project_complexity=project_complexity,
    #                 max_depth=max_depth,
    #                 est_tree_tokens=est_tree_tokens,
    #                 has_readme=has_readme,
    #                 component_count=component_count,
    #                 data_flow_count=data_flow_count,
    #                 min_pages=min_pages,
    #                 max_pages=max_pages
    #             )
    #         )
    #         raw = response.content.strip()
    #         if raw.startswith('```'):
    #             raw = re.sub(r'^```[a-zA-Z]*\n|```$', '', raw, flags=re.MULTILINE).strip()
    #         data = json.loads(raw)
    #         proposed = int(data.get('proposed_pages', min_pages))
    #         if proposed < min_pages:
    #             proposed = min_pages
    #         if proposed > max_pages:
    #             proposed = max_pages
    #         return proposed, data.get('justification'), data.get('compression_recommendation')
    #     except Exception as e:
    #         logger.warning(f"Adaptive page budget LLM path failed, using heuristic: {e}")
    #         import math
    #         complexity_score = 1.0
    #         if project_complexity == 'moderate':
    #             complexity_score = 1.3
    #         elif project_complexity == 'complex':
    #             complexity_score = 1.6
    #         size_factor = math.log(total_files + 1, 2) * 0.8
    #         symbol_factor = math.log(total_symbols + 1, 10) * 0.5
    #         component_factor = (component_count ** 0.5) * 0.9
    #         raw_score = (size_factor + symbol_factor + component_factor) * complexity_score
    #         proposed = int(min(max_pages, max(min_pages, round(raw_score + 4))))
    #         justification = f"Heuristic fallback: files={total_files}, components={component_count}, complexity={project_complexity}"
    #         return proposed, justification, "heuristic"
    #
    # def _apply_page_budget(self, structure: WikiStructureSpec, page_budget: int) -> WikiStructureSpec:
    #     """Merge low-scope pages if total exceeds budget, add deferred section for dropped pages."""
    #     logger.info(f"Applying page budget: target={page_budget}, current={structure.total_pages}")
    #     if structure.total_pages <= page_budget:
    #         return structure
    #     all_pages = []
    #     for section in structure.sections:
    #         for page in section.pages:
    #             all_pages.append((section, page))
    #     def page_scope(p):
    #         score = 0
    #         if len(p.description) < 60:
    #             score += 1
    #         if len(p.key_files) <= 1:
    #             score += 1
    #         if len(p.target_folders) <= 1:
    #             score += 1
    #         return score
    #     all_pages_sorted = sorted(all_pages, key=lambda sp: page_scope(sp[1]))
    #     pages_to_keep = set()
    #     for section, page in reversed(all_pages_sorted):
    #         pages_to_keep.add(page.page_name)
    #         if len(pages_to_keep) >= page_budget:
    #             break
    #     deferred_pages = []
    #     new_sections = []
    #     for section in structure.sections:
    #         kept_pages = []
    #         for page in section.pages:
    #             if page.page_name in pages_to_keep:
    #                 kept_pages.append(page)
    #             else:
    #                 deferred_pages.append(page)
    #         if kept_pages:
    #             section.pages = kept_pages
    #             new_sections.append(section)
    #     if deferred_pages:
    #         from ..state.wiki_state import SectionSpec
    #         deferred_section = SectionSpec(
    #             section_name="Proposed Additional Pages (Deferred)",
    #             section_order=len(new_sections) + 1,
    #             description="Pages deferred due to page budget; consider adding if budget increases.",
    #             rationale="Budget enforcement",
    #             pages=[p for p in deferred_pages]
    #         )
    #         new_sections.append(deferred_section)
    #     order = 1
    #     for section in new_sections:
    #         section.section_order = order
    #         page_order = 1
    #         for page in section.pages:
    #             page.page_order = page_order
    #             page_order += 1
    #         order += 1
    #     structure.sections = new_sections
    #     structure.total_pages = sum(len(s.pages) for s in new_sections if s.section_name != "Proposed Additional Pages (Deferred)")
    #     logger.info(f"Page budget applied: new_total={structure.total_pages} (+ deferred section with {len(deferred_pages)} pages)")
    #     return structure

    def _ranked_truncation(self, docs: list[Any], page_spec: PageSpec, budget: int) -> tuple:
        """
        Ranked truncation: include top-priority docs with FULL content until budget,
        silently drop the rest. No signatures, no summaries, no compression.

        This replaces hierarchical tiered compression. The model either sees complete
        implementation or doesn't see the symbol at all — producing clean output
        with no "implementation not present" artifacts.

        Args:
            docs: All retrieved documents (may exceed budget)
            page_spec: Page specification for ranking priority
            budget: Token budget to fit within

        Returns:
            Tuple of (included_docs, token_metrics) where token_metrics is a dict
            with keys: docs_retrieved, docs_included, docs_dropped, tokens_before,
            tokens_after, token_budget, utilization_pct.
        """
        from ..document_ranker import DocumentRanker

        token_counter = get_token_counter()

        # Rank documents by relevance to page
        ranker = DocumentRanker(
            code_graph=self.retriever_stack.relationship_graph
            if hasattr(self.retriever_stack, "relationship_graph")
            else None
        )
        page_spec_dict = {
            "topic": page_spec.page_name,
            "target_folders": getattr(page_spec, "target_folders", []),
            "key_files": getattr(page_spec, "key_files", []),
        }
        tiered_docs, metrics = ranker.rank_expanded_documents(
            page_spec=page_spec_dict,
            expanded_docs=docs,
            total_token_budget=budget * 2,  # Give ranker room — we do our own truncation
        )

        # Sort by tier (ascending) then score (descending) — tier 1 first, highest score first
        sorted_docs = sorted(tiered_docs, key=lambda x: (x[1], -x[2]))

        # Fill up to budget with full content
        included = []
        tokens_used = 0
        dropped = 0
        for doc, _tier, _score in sorted_docs:
            doc_tokens = token_counter.count_document(doc)
            if tokens_used + doc_tokens <= budget:
                included.append(doc)
                tokens_used += doc_tokens
            else:
                dropped += 1

        total_tokens_before = token_counter.count_documents(docs)
        drop_pct = (dropped / len(docs) * 100) if docs else 0
        log_fn = logger.warning if drop_pct > 30 else logger.info
        log_fn(
            f"[RANKED_TRUNCATION] {len(docs)} docs ({total_tokens_before:,} tokens) → "
            f"{len(included)} docs ({tokens_used:,} tokens), "
            f"{dropped} docs dropped ({drop_pct:.0f}%) to fit {budget:,} budget"
        )
        if drop_pct > 30:
            logger.warning(
                f"[RANKED_TRUNCATION][QUALITY_RISK] Page may have degraded quality — "
                f"{drop_pct:.0f}% of context was truncated. "
                f"Consider splitting this page into smaller sub-pages."
            )

        token_metrics: dict = {
            "docs_retrieved": len(docs),
            "docs_included": len(included),
            "docs_dropped": dropped,
            "tokens_before": total_tokens_before,
            "tokens_after": tokens_used,
            "token_budget": budget,
            "utilization_pct": round((tokens_used / budget * 100) if budget else 0.0, 1),
        }
        return included, token_metrics

    def _format_simple_context(self, relevant_docs: list[Any], page_spec: PageSpec) -> dict[str, Any]:
        """
        Format context using SIMPLE MODE (all documents fit within token budget).

        Uses all documents with full content - preserves current system behavior.
        No compression, no tiering, straightforward context building.
        Applied when total context tokens <= CONTEXT_TOKEN_BUDGET (50K).
        """
        # Group documents by file path for proper structuring
        docs_by_file = {}
        documentation_docs = []

        for doc in relevant_docs:
            source = doc.metadata.get("source", "unknown")

            # Separate documentation from code files
            if self._is_documentation_file(source):
                documentation_docs.append(doc)
            else:
                if source not in docs_by_file:
                    docs_by_file[source] = []
                docs_by_file[source].append(doc)

        # Prioritize files based on page specification
        prioritized_files = []
        other_files = []

        for file_path in docs_by_file.keys():
            is_priority = False

            # Check if file matches target folders or key files
            if hasattr(page_spec, "target_folders") and page_spec.target_folders:
                for folder in page_spec.target_folders:
                    if folder and folder in file_path:
                        is_priority = True
                        break

            if hasattr(page_spec, "key_files") and page_spec.key_files:
                for key_file in page_spec.key_files:
                    if key_file and key_file in file_path:
                        is_priority = True
                        break

            if is_priority:
                prioritized_files.append(file_path)
            else:
                other_files.append(file_path)

        # Build structured context
        context_parts = []

        # Documentation context section
        if documentation_docs:
            context_parts.append("Documentation Context:")
            for doc in documentation_docs:  # Include ALL documentation files
                source = doc.metadata.get("source", "unknown")
                content = doc.page_content  # FULL content (no truncation)
                context_parts.append(f"<documentation_source: {source}>")
                context_parts.append(content)
                context_parts.append("</documentation_source>")
            context_parts.append("")

        # Code context section - include ALL files (no truncation)
        if docs_by_file:
            context_parts.append("Code Context:")

            # Process prioritized files first, then others
            all_files = prioritized_files + other_files  # NO TRUNCATION

            for file_path in all_files:
                file_docs = docs_by_file[file_path]

                # Combine all content from this file
                combined_content = "\\n".join(doc.page_content for doc in file_docs)

                # Extract imports
                imports = self._extract_imports_for_file(file_path, combined_content)

                context_parts.append(f"<code_source: {file_path}>")

                # Add per-symbol line annotations — use visible markers (not HTML
                # comments) so the LLM reliably picks them up for citations.
                line_annotations = []
                for doc in file_docs:
                    start = doc.metadata.get("start_line", 0)
                    end = doc.metadata.get("end_line", 0)
                    symbol = doc.metadata.get("symbol_name", "")
                    if start and end and symbol:
                        line_annotations.append(f"  [SYMBOL] {symbol}: L{start}-L{end}")
                if line_annotations:
                    context_parts.append("<line_map>")
                    context_parts.extend(line_annotations)
                    context_parts.append("</line_map>")

                if imports:
                    context_parts.append("<imports>")
                    context_parts.append(imports)
                    context_parts.append("</imports>")
                context_parts.append("<implementation>")
                context_parts.append(combined_content)
                context_parts.append("</implementation>")
                context_parts.append("</code_source>")
                context_parts.append("")

        # Build location matches for feedback
        location_matches = []
        if hasattr(page_spec, "target_folders") and page_spec.target_folders:
            for folder in page_spec.target_folders:
                if any(folder in fp for fp in prioritized_files):
                    location_matches.append(f"📁 {folder}")

        if hasattr(page_spec, "key_files") and page_spec.key_files:
            for key_file in page_spec.key_files:
                if any(key_file in fp for fp in prioritized_files):
                    location_matches.append(f"📄 {key_file}")

        logger.info(
            f"Simple mode: {len(relevant_docs)} docs, {len(documentation_docs)} doc files, {len(docs_by_file)} code files"
        )

        return {
            "content": "\\n".join(context_parts),
            "files": list(prioritized_files) + list(other_files),
            "location_matches": list(set(location_matches)),
            "prioritized_sources": len(prioritized_files),
            "documentation_files": len(documentation_docs),
            "code_files": len(docs_by_file),
            "total_docs": len(relevant_docs),  # Add total document count
        }

    def _format_hierarchical_context(self, relevant_docs: list[Any], page_spec: PageSpec) -> dict[str, Any]:
        """
        Format context using HIERARCHICAL MODE (context exceeds token budget).

        Applies ranking + compression when total tokens > CONTEXT_TOKEN_BUDGET:
        - Tier 1: Top 30-40 docs with FULL content (~50K tokens)
        - Tier 2: Next 30-50 docs with SIGNATURES only (~30K tokens)
        - Tier 3: Remaining docs with SUMMARIES only (~10K tokens)
        """
        # Initialize compressor and ranker
        compressor = DocumentCompressor()
        ranker = DocumentRanker(code_graph=self.retriever_stack.relationship_graph)

        # Convert PageSpec to dict format expected by ranker
        page_spec_dict = {
            "topic": page_spec.page_name,
            "target_folders": getattr(page_spec, "target_folders", []),
            "key_files": getattr(page_spec, "key_files", []),
        }

        # Rank all documents
        tiered_docs, metrics = ranker.rank_expanded_documents(
            page_spec=page_spec_dict,
            expanded_docs=relevant_docs,
            total_token_budget=90000,  # 90K for initial context
        )

        logger.info(
            f"Hierarchical mode ranking: Tier 1={metrics['tier1_count']}, "
            + f"Tier 2={metrics['tier2_count']}, Tier 3={metrics['tier3_count']}"
        )

        # Build tier map from ranker output (for relationship hints)
        tier_map = {}
        for doc, tier, _score in tiered_docs:
            symbol_name = doc.metadata.get("symbol_name")
            if symbol_name:
                tier_map[symbol_name] = tier

        logger.debug(f"Built tier map: {len(tier_map)} symbols mapped")

        # Get graph for relationship hints
        graph = (
            self.retriever_stack.content_expander.graph if hasattr(self.retriever_stack, "content_expander") else None
        )

        # Process each tier with compression
        tier_1_docs = []
        tier_2_docs = []
        tier_3_docs = []

        for doc, tier, _score in tiered_docs:
            if tier == 1:
                # Tier 1: Keep full content
                tier_1_docs.append(doc)
            elif tier == 2:
                # Tier 2: Compress to signatures
                compressed = compressor.compress_document(doc, tier=2)
                tier_2_docs.append(compressed)
            else:  # tier == 3
                # Tier 3: Compress to summaries
                compressed = compressor.compress_document(doc, tier=3)
                tier_3_docs.append(compressed)

        # Build file list from all tiers
        all_files = set()
        for doc in tier_1_docs + tier_2_docs + tier_3_docs:
            source = doc.metadata.get("source", "unknown")
            if source != "unknown":
                all_files.add(source)

        logger.info(
            f"Hierarchical mode: {len(relevant_docs)} total docs → "
            + f"Tier 1: {len(tier_1_docs)}, Tier 2: {len(tier_2_docs)}, Tier 3: {len(tier_3_docs)}"
        )

        # Format separate tier content for prompts WITH relationship hints
        tier1_content = self._format_tier_with_hints(tier_1_docs, tier=1, tier_map=tier_map, graph=graph)
        tier2_content = self._format_tier_with_hints(tier_2_docs, tier=2, tier_map=tier_map, graph=graph)
        tier3_content = self._format_tier_with_hints(tier_3_docs, tier=3, tier_map=tier_map, graph=graph)

        return {
            "mode": "hierarchical",
            "tier1_content": tier1_content,
            "tier2_content": tier2_content,
            "tier3_content": tier3_content,
            "tier1_docs": tier_1_docs,  # Store for continuation fetching
            "tier2_docs": tier_2_docs,
            "tier3_docs": tier_3_docs,
            "all_expanded_docs": relevant_docs,  # Store original for continuation
            "files": list(all_files),
            "location_matches": [],  # TODO: Extract from tiered docs
            "prioritized_sources": len(tier_1_docs),
            "documentation_files": 0,  # TODO: Count doc files in tiers
            "code_files": len(all_files),
            "tier_metrics": metrics,
        }

    def _format_tier_with_hints(self, docs: list[Any], tier: int, tier_map: dict[str, int], graph) -> str:
        """
        Format a tier of documents for prompt inclusion with HYBRID relationship hints.

        HYBRID APPROACH:
        - Initially retrieved docs (~30): Forward hints (→) computed from graph traversal
          Example: "→ extends BaseService [T2]; uses UserRepo [T1]"
        - Expanded docs (80-100): Backward hints (←) from expansion metadata (FREE)
          Example: "← expanded because: composed_by UserService (via repo field)"

        This avoids double graph traversal for expanded docs while still providing
        useful context about WHY each document was included.
        """
        if not docs:
            return "No documents in this tier."

        parts = []
        tier_name = {1: "FULL DETAIL", 2: "SIGNATURES ONLY", 3: "SUMMARIES ONLY"}[tier]

        parts.append(f"TIER {tier}: {tier_name} ({len(docs)} documents)")
        parts.append("=" * 80)
        parts.append("")

        # Hints for Tier 1 and 2 only
        include_hints = tier in [1, 2]

        if tier == 1:
            # Build a consolidated <line_map> at the top of Tier 1 so the LLM
            # can find symbol→line mappings in the same format as simple mode.
            line_map_entries = []
            for doc in docs:
                s_name = doc.metadata.get("symbol_name", "")
                s_start = doc.metadata.get("start_line", 0)
                s_end = doc.metadata.get("end_line", 0)
                s_source = doc.metadata.get("source", "")
                if s_name and s_start and s_end:
                    line_map_entries.append(f"  [SYMBOL] {s_name}: L{s_start}-L{s_end}  ({s_source})")
            if line_map_entries:
                parts.append("<line_map>")
                parts.extend(line_map_entries)
                parts.append("</line_map>")
                parts.append("")

            for doc in docs:
                source = doc.metadata.get("source", "unknown")
                symbol_name = doc.metadata.get("symbol_name", "Unknown")
                file_path = doc.metadata.get("file_path", source)
                start_line = doc.metadata.get("start_line", 0)
                end_line = doc.metadata.get("end_line", 0)

                # Format header with line range when available
                source_ref = source
                if start_line and end_line:
                    source_ref = f"{source} L{start_line}-L{end_line}"
                header = f"**{symbol_name}** (`{source_ref}`)"

                # HYBRID HINTS: Forward for initially retrieved, backward for expanded
                if include_hints:
                    hint_line = self._format_hybrid_hint(doc, symbol_name, file_path, graph, tier_map)
                    if hint_line:
                        header += f"\n{hint_line}\n"

                parts.append(header)
                parts.append(doc.page_content)
                parts.append("")

        elif tier == 2:
            parts.append("_Implementation details omitted. Request full versions using NEED_CONTEXT if needed._")
            parts.append("")
            for doc in docs:
                source = doc.metadata.get("source", "unknown")
                symbol_name = doc.metadata.get("symbol_name", "Unknown")
                file_path = doc.metadata.get("file_path", source)
                start_line = doc.metadata.get("start_line", 0)
                end_line = doc.metadata.get("end_line", 0)

                # Format header with line range when available
                source_ref = source
                if start_line and end_line:
                    source_ref = f"{source} L{start_line}-L{end_line}"
                header = f"**{symbol_name}** (`{source_ref}`)"

                # HYBRID HINTS: Forward for initially retrieved, backward for expanded
                if include_hints:
                    hint_line = self._format_hybrid_hint(doc, symbol_name, file_path, graph, tier_map)
                    if hint_line:
                        header += f"\n{hint_line}\n"

                parts.append(header)
                parts.append(doc.page_content)
                parts.append("")

        else:  # tier == 3
            parts.append("_Minimal summaries for awareness. Request full versions using NEED_CONTEXT if needed._")
            parts.append("")
            # No hints for Tier 3 (too many symbols, would add noise)
            for doc in docs:
                parts.append(doc.page_content)
                parts.append("")

        return "\\n".join(parts)

    def _format_hybrid_hint(
        self, doc: Any, symbol_name: str, file_path: str, graph, tier_map: dict[str, int] = None
    ) -> str:
        """
        Format a hint line using the hybrid approach:
        - Forward hints (→) for initially retrieved documents (computed from graph)
        - Backward hints (←) for expanded documents (from expansion metadata - FREE)

        Args:
            doc: Document with metadata
            symbol_name: Name of the symbol
            file_path: File path for graph lookup
            graph: NetworkX code graph (may be None)
            tier_map: Symbol to tier mapping (None = skip tier labels, for simple mode)

        Returns:
            Formatted hint string or empty string
        """
        is_initially_retrieved = doc.metadata.get("is_initially_retrieved", False)

        if is_initially_retrieved:
            # FORWARD HINTS (→): Initially retrieved docs - compute from graph
            if graph is not None:
                hints = self._extract_relationship_hints(symbol_name, file_path, graph, tier_map)
                if hints:
                    return f"→ {hints}"
        else:
            # BACKWARD HINTS (←): Expanded docs - use metadata (FREE, no graph traversal)
            expansion_reason = doc.metadata.get("expansion_reason", "")
            expansion_source = doc.metadata.get("expansion_source", "")
            expansion_via = doc.metadata.get("expansion_via", "")

            if expansion_reason and expansion_source:
                # Format backward hint with "via" context if available
                reason_descriptions = {
                    # Inheritance relationships
                    "extended_by": "parent of",
                    "implemented_by": "interface of",
                    "extends": "child of",
                    # Composition relationships
                    "composed_by": "component of",
                    "used_by": "used by",
                    "aggregation": "aggregated by",
                    "composition": "composed by",
                    # Call relationships
                    "called_by": "called by",
                    "references": "referenced by",
                    "creates": "created by",
                    "instantiated_by": "instantiated by",
                    "calls_func": "initialization dependency of",
                    # Containment relationships
                    "defines": "defines",
                    "contains": "contained by",
                    "has_member": "member of",
                    # Type relationships
                    "return_type": "return type of",
                    "return_type_of": "return type of",
                    "param_type": "parameter type of",
                    "parameter_type_of": "parameter type of",
                    "type_of": "type of",
                    "type_annotation": "type annotation of",
                    "initializes": "initialized by",
                    # Member relationships
                    "constructor_of": "constructor of",
                    "constructor_member": "constructor of",
                    "method_of": "method of",
                    "method_member": "method of",
                    "field_of": "field of",
                    "field_member": "field of",
                    "implements_interface": "implements",
                }
                reason_text = reason_descriptions.get(expansion_reason, expansion_reason)

                hint = f"← included as {reason_text} `{expansion_source}`"
                if expansion_via:
                    hint += f" ({expansion_via})"
                return hint

        return ""

    def _generate_simple(
        self, page_spec: PageSpec, relevant_content: dict[str, Any], repo_context: str, config: RunnableConfig
    ) -> str:
        """Generate content for simple mode (context fits token budget) using V3_TONE_ADJUSTED prompt"""

        content_prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    "You are an expert technical writer creating comprehensive, enterprise-grade documentation.",
                ),
                ("human", ENHANCED_CONTENT_GENERATION_PROMPT_V3_TONE_ADJUSTED),
            ]
        )

        target_audience = TARGET_AUDIENCES.get(self.target_audience.value, "Mixed audience")

        # Get actual document count (not file count)
        total_docs = relevant_content.get("total_docs", 0)
        file_count = len(relevant_content.get("files", []))

        logger.info(
            f"📝 Generating content for '{page_spec.page_name}' in SIMPLE MODE "
            f"(V3_TONE_ADJUSTED) - {total_docs} documents across {file_count} files"
        )

        # Toggle: skip repository_context for token savings test
        effective_repo_context = "" if SKIP_REPO_CONTEXT_FOR_PAGES else repo_context
        if SKIP_REPO_CONTEXT_FOR_PAGES:
            logger.info(f"⚡ SKIP_REPO_CONTEXT_FOR_PAGES=1: Omitting {len(repo_context):,} char repository_context")

        response = self.llm.invoke(
            content_prompt.format_messages(
                section_name=page_spec.page_name.split("/")[0] if "/" in page_spec.page_name else "Main",
                page_name=page_spec.page_name,
                page_description=page_spec.description,
                content_focus=page_spec.content_focus,
                repository_url=self.repository_url,
                wiki_style=self.wiki_style.value,
                repository_context=effective_repo_context,
                relevant_content=relevant_content["content"],
                related_files="\\n".join(relevant_content["files"]),
                target_audience=target_audience,
            ),
            config=config,
        )

        return response.content

    def _extract_relationship_hints(
        self, symbol_name: str, file_path: str, graph, tier_map: dict[str, int] = None
    ) -> str:
        """
        Extract relationship hints in natural language format.

        Example (with tiers): "extends BaseService [T2]; uses UserRepo [T1]; creates Session (via create())"
        Example (no tiers): "extends `BaseService`; uses `UserRepo`; creates `Session` (via create())"

        Args:
            symbol_name: Name of the symbol
            file_path: Path to file containing symbol
            graph: NetworkX code graph
            tier_map: Mapping of symbol names to tier numbers (None = skip tier labels)

        Returns:
            Formatted relationship hint string, or empty string if no relationships
        """
        if not graph or not hasattr(graph, "nodes"):
            return ""

        # Find node in graph using O(1) index lookup
        node_id = self._find_node_in_graph(symbol_name, file_path, graph)
        if not node_id:
            return ""

        parts = []

        # INHERITANCE (direct architectural dependency)
        inherits = self._extract_inheritance_hints(node_id, graph, tier_map)
        if inherits:
            parts.append(f"extends {inherits}")

        # COMPOSITION/AGGREGATION (direct field-level dependencies)
        uses = self._extract_composition_hints(node_id, graph, tier_map)
        if uses:
            parts.append(f"uses {uses}")

        # CREATES (transitive dependency with method context)
        creates = self._extract_creates_hints(node_id, graph, tier_map)
        if creates:
            parts.append(f"creates {creates}")

        # CALLS (only high fan-out - orchestrator pattern)
        calls = self._extract_calls_hints(node_id, graph, tier_map)
        if calls:
            parts.append(f"calls {calls}")

        return "; ".join(parts) if parts else ""

    def _find_node_in_graph(self, symbol_name: str, file_path: str, graph) -> str | None:
        """Find node ID using O(1) index if available, fallback to search"""
        # Try O(1) index first
        if hasattr(graph, "_node_index") and graph._node_index:
            language = self._infer_language_from_path(file_path)
            key = (symbol_name, file_path, language)
            node_id = graph._node_index.get(key)
            if node_id:
                return node_id

        # Fallback: search (O(N) but rare)
        for node_id, node_data in graph.nodes(data=True):
            if node_data.get("symbol_name") == symbol_name and node_data.get("file_path") == file_path:
                return node_id

        return None

    def _extract_inheritance_hints(self, node_id: str, graph, tier_map: dict = None) -> str:
        """Extract inheritance: 'BaseClass [T2], Interface [T1]' or 'BaseClass, Interface' (no tiers)"""
        inherits = []

        try:
            for successor in graph.successors(node_id):
                for edge_data in graph[node_id][successor].values():
                    rel_type = edge_data.get("relationship_type", "").lower()
                    if rel_type == "inheritance":
                        parent_name = graph.nodes[successor].get("symbol_name", "")
                        if parent_name:
                            # Only add tier label if tier_map provided (hierarchical mode)
                            if tier_map:
                                tier = tier_map.get(parent_name, 3)
                                inherits.append(f"`{parent_name}` [T{tier}]")
                            else:
                                inherits.append(f"`{parent_name}`")
        except Exception as e:
            logger.debug(f"Error extracting inheritance for {node_id}: {e}")

        return ", ".join(inherits[:3])  # Max 3 parents

    def _extract_composition_hints(self, node_id: str, graph, tier_map: dict = None) -> str:
        """
        Extract composition/aggregation hints with field context.

        Format examples (with tiers):
        - 'UserRepo [T1], ConfigService [T2]' (basic)
        - 'UserRepo [T1] (via repo field)' (with field context)

        Format examples (no tiers - simple mode):
        - '`UserRepo`, `ConfigService`' (basic)
        - '`UserRepo` (via repo field)' (with field context)
        """
        uses = []
        seen = set()

        try:
            for successor in graph.successors(node_id):
                for edge_data in graph[node_id][successor].values():
                    rel_type = edge_data.get("relationship_type", "").lower()
                    if rel_type in ("composition", "aggregation"):
                        child_name = graph.nodes[successor].get("symbol_name", "")
                        if child_name and child_name not in seen:
                            seen.add(child_name)

                            # Only add tier label if tier_map provided (hierarchical mode)
                            if tier_map:
                                tier = tier_map.get(child_name, 3)
                                symbol_ref = f"`{child_name}` [T{tier}]"
                            else:
                                symbol_ref = f"`{child_name}`"

                            # Extract "via field" context
                            via_context = None

                            # 1. Check source_context (e.g., "UserService.repo")
                            source_context = edge_data.get("source_context", "")
                            if source_context and "." in source_context:
                                field_name = source_context.split(".")[-1]
                                via_context = f"via {field_name} field"

                            # 2. Check annotations
                            annotations = edge_data.get("annotations", {})
                            if not via_context and annotations.get("field_name"):
                                via_context = f"via {annotations['field_name']} field"

                            if via_context:
                                uses.append(f"{symbol_ref} ({via_context})")
                            else:
                                uses.append(symbol_ref)
        except Exception as e:
            logger.debug(f"Error extracting composition for {node_id}: {e}")

        if len(uses) > 5:
            return ", ".join(uses[:5]) + f" (+{len(uses) - 5} more)"
        return ", ".join(uses)

    def _extract_creates_hints(self, node_id: str, graph, tier_map: dict = None) -> str:
        """
        Extract CREATES hints with transitive context.

        Format examples (with tiers):
        - 'Session [T2], Token [T3]' (basic)
        - 'Session [T2] (via create_session())' (method context)

        Format examples (no tiers - simple mode):
        - '`Session`, `Token`' (basic)
        - '`Session` (via create_session())' (method context)
        """
        creates_map = {}  # {created_class: via_context}

        try:
            for successor in graph.successors(node_id):
                for edge_data in graph[node_id][successor].values():
                    rel_type = edge_data.get("relationship_type", "").lower()
                    if rel_type == "creates":
                        created_class = graph.nodes[successor].get("symbol_name", "")

                        # Extract "via" context from multiple sources
                        via_context = None

                        # 1. Check source_context (preserved from parser, e.g., "UserService.createUser")
                        source_context = edge_data.get("source_context", "")
                        if source_context and "." in source_context:
                            method_name = source_context.split(".")[-1]
                            via_context = f"`{method_name}()`"

                        # 2. Check annotations (field_name from composition edges)
                        annotations = edge_data.get("annotations", {})
                        if not via_context and annotations.get("field_name"):
                            via_context = f"{annotations['field_name']} field"

                        # 3. Legacy: Check source_method directly
                        if not via_context:
                            source_method = edge_data.get("source_method")
                            if source_method:
                                via_context = f"`{source_method}()`"

                        if created_class and created_class not in creates_map:
                            creates_map[created_class] = via_context
        except Exception as e:
            logger.debug(f"Error extracting creates for {node_id}: {e}")

        if not creates_map:
            return ""

        # Format with or without tier labels
        creates_parts = []
        for created_class, via_context in list(creates_map.items())[:4]:  # Max 4 classes
            # Only add tier label if tier_map provided (hierarchical mode)
            if tier_map:
                tier = tier_map.get(created_class, 3)
                symbol_ref = f"`{created_class}` [T{tier}]"
            else:
                symbol_ref = f"`{created_class}`"

            if via_context:
                creates_parts.append(f"{symbol_ref} (via {via_context})")
            else:
                creates_parts.append(symbol_ref)

        return ", ".join(creates_parts)

    def _extract_calls_hints(self, node_id: str, graph, tier_map: dict = None) -> str:
        """Extract CALLS (only if high fan-out >5): 'Service.method1() [T2]' or 'Service.method1()' (no tiers)"""
        calls = []

        try:
            for successor in graph.successors(node_id):
                for edge_data in graph[node_id][successor].values():
                    rel_type = edge_data.get("relationship_type", "").lower()
                    if rel_type == "calls":
                        called_symbol = graph.nodes[successor].get("symbol_name", "")
                        if called_symbol:
                            calls.append(called_symbol)
        except Exception as e:
            logger.debug(f"Error extracting calls for {node_id}: {e}")

        # Only show if high fan-out (orchestrator pattern)
        if len(calls) < 5:
            return ""

        unique_calls = list(set(calls))[:4]  # Show first 4 unique
        calls_parts = []
        for call in unique_calls:
            # Only add tier label if tier_map provided (hierarchical mode)
            if tier_map:
                # Extract class name from method call if present
                base_name = call.split(".")[0] if "." in call else call
                tier = tier_map.get(base_name, 3)
                calls_parts.append(f"`{call}()` [T{tier}]")
            else:
                calls_parts.append(f"`{call}()`")

        result = ", ".join(calls_parts)
        if len(set(calls)) > 4:
            result += f" (+{len(set(calls)) - 4} more)"

        return result

    def _infer_language_from_path(self, file_path: str) -> str:
        """Infer language from file extension"""
        import os

        ext_map = {
            ".py": "python",
            ".cpp": "cpp",
            ".cc": "cpp",
            ".cxx": "cpp",
            ".c++": "cpp",
            ".h": "cpp",
            ".hpp": "cpp",
            ".hh": "cpp",
            ".hxx": "cpp",
            ".c": "c",
            ".ts": "typescript",
            ".tsx": "typescript",
            ".js": "javascript",
            ".jsx": "javascript",
            ".java": "java",
        }
        ext = os.path.splitext(file_path)[1]
        return ext_map.get(ext, "unknown")
