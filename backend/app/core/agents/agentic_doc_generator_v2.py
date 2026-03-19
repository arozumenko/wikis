"""
Agentic Documentation Generator v2 - Planning + Tool Calling

This module combines:
1. LLM Planning Phase: Generates semantic page structure with symbol assignments
2. Tool Calling Generation: Iterates through planned sections using context swapping

Key features:
- LLM-driven section planning (semantic, not token-based)
- Human-readable section titles ("Authentication & Authorization", not "AuthService")
- 5-8 detail sections with evenly distributed symbols
- Introduction/Overview/Conclusion written during planning (full context)
- Detail sections generated via tool calling (focused context)
"""

import gc
import json
import logging
import os
import re
import time
from dataclasses import dataclass, field
from typing import Any

from langchain_core.documents import Document
from langchain_core.language_models import BaseLanguageModel
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableConfig
from pydantic import BaseModel, Field, ValidationError

from ..token_counter import get_token_counter

# Toggle to skip repository_context in page generation (test mode for token savings)
SKIP_REPO_CONTEXT_FOR_PAGES = os.getenv("WIKIS_SKIP_REPO_CONTEXT_FOR_PAGES", "0") == "1"
from ..parsers.base_parser import RelationshipType
from ..prompts.wiki_prompts_enhanced import (
    AGENTIC_INTRO_OVERVIEW_PROMPT,
    PAGE_STRUCTURE_PLANNING_PROMPT,
)
from ..state.wiki_state import OverviewSection, PageSpec, PageStructure, SymbolSummary

logger = logging.getLogger(__name__)


# =============================================================================
# ARCHITECTURAL SYMBOL TYPES
# =============================================================================

# Top-level symbols that define code structure and API surface
ARCHITECTURAL_SYMBOL_TYPES = {
    # Top-level code symbols
    "class",
    "interface",
    "struct",
    "enum",
    "trait",
    "protocol",
    "function",  # Standalone functions only - methods excluded
    "constant",  # Module-level constants
    "macro",  # C/C++ preprocessor definitions
    # Documentation symbols
    "markdown_document",
    "toml_document",
    "text_chunk",
    "text_document",
    "restructuredtext_document",
    "plaintext_document",
    "documentation",
}


# =============================================================================
# TOOL CALLING PROMPTS
# =============================================================================

AGENTIC_SYSTEM_PROMPT = """You are an expert technical documentation writer creating comprehensive, enterprise-grade documentation for a software repository.

## DOCUMENTATION STYLE
- Write clear, professional documentation suitable for {target_audience}
- Style: {wiki_style}
- Use proper markdown formatting (headers, code blocks, tables, lists)
- Include **Mermaid diagrams** for architecture, workflows, and relationships
- Reference code with backticks: `ClassName.method()`
- Explain the "why" not just the "what"

## REPOSITORY OVERVIEW
{repo_analysis}

## PAGE CONTEXT
You are documenting: **{page_name}**
Description: {page_description}
Content Focus: {content_focus}

## PAGE STRUCTURE (Pre-Planned)

Introduction and Overview sections have already been written.
Your task is to write DETAIL SECTIONS one by one.

**Detail Sections to Generate:**
{section_list}

## WORKFLOW

1. **RECEIVE** - You receive code for ONE detail section
2. **WRITE** - Write comprehensive documentation following the section's purpose
3. **COMMIT** - Call `commit_section` to save your work
4. **CONTINUE** - Call `request_next_section` to proceed (or auto-advance)

**CRITICAL**: Always call `commit_section` BEFORE `request_next_section`.

## TOOLS

### 1. commit_section
Save documentation for the current section.
- `content` (string, required): Your markdown documentation
- `summary` (string, required): One-sentence summary

### 2. request_next_section
Get the next section's code context.
- `section_name` (string, optional): Specific section to jump to

## QUALITY GUIDELINES
- Document ALL public methods, parameters, and configurations
- Include at least one Mermaid diagram per section
- Use tables for parameters and configurations
- Cross-reference other sections where relevant
- Serve both technical and non-technical audiences
- Cite sources using the exact format: `<code_source: relative/path.py>`
- Use the code_source tag inline near the relevant paragraph or in a short bullet list at the end of the section
"""

AGENTIC_USER_PROMPT = """## CURRENT SECTION: {section_name}
**Section {current_idx} of {total_sections}**

### Section Purpose
{section_description}

### Suggested Elements
{suggested_elements}

---

### PREVIOUSLY WRITTEN (Table of Contents)
{toc_block}

---

### CODE TO DOCUMENT [FULL IMPLEMENTATION]

{full_code}

---

### RELATED SYMBOLS [SIGNATURES FOR CONTEXT]
{cross_ref_note}

```
{signatures}
```

---

**TASK**: Write comprehensive documentation for **{section_name}**.

**CITATIONS**: Use the code_source format exactly: `<code_source: relative/path.py>`.
If needed, include a short **Sources** bullet list with these tags at the end of the section.

When finished:
1. Call `commit_section` with your markdown and a brief summary
2. Call `request_next_section` to proceed
"""


# =============================================================================
# DATA STRUCTURES
# =============================================================================


@dataclass
class PlannedSection:
    """A section from the planning phase"""

    section_id: str
    title: str
    description: str
    primary_symbols: list[str]
    supporting_symbols: list[str]
    symbol_paths: list[str]
    retrieval_queries: list[str]
    intent: str | None
    suggested_elements: list[str]

    # Built during generation
    full_code: str = ""
    signatures: str = ""
    section_docs: list[Any] = field(default_factory=list)


@dataclass
class DocumentationState:
    """Tracks what has been documented for coherence across sections"""

    toc: list[tuple] = field(default_factory=list)  # [(section_name, summary)]
    completed_sections: set[str] = field(default_factory=set)

    def add_section(self, section_name: str, summary: str):
        """Record a completed section"""
        self.toc.append((section_name, summary))
        self.completed_sections.add(section_name)

    def get_toc_for_prompt(self) -> str:
        """Generate ToC context for the prompt"""
        if not self.toc:
            return "*This is the first detail section - introduction and overview are above.*"

        lines = ["*Documentation written so far:*"]
        for section_name, summary in self.toc:
            lines.append(f"- **{section_name}**: {summary}")
        return "\n".join(lines)


class IntroOverviewBundle(BaseModel):
    introduction: str = ""
    overview_sections: list[OverviewSection] = Field(default_factory=list)
    conclusion: str = ""


# =============================================================================
# TOOL SCHEMAS
# =============================================================================


def get_agentic_tools():
    """Get tool definitions for agentic generation"""
    return [
        {
            "type": "function",
            "function": {
                "name": "commit_section",
                "description": "Save the documentation you just wrote for the current section.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "content": {"type": "string", "description": "The markdown documentation for this section"},
                        "summary": {"type": "string", "description": "One-sentence summary for the table of contents"},
                    },
                    "required": ["content", "summary"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "request_next_section",
                "description": "Request the next section's code context.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "section_name": {
                            "type": "string",
                            "description": "Name of specific section to jump to. Leave empty to auto-advance.",
                        }
                    },
                    "required": [],
                },
            },
        },
    ]


# =============================================================================
# MAIN GENERATOR CLASS
# =============================================================================


class AgenticDocGeneratorV2:
    """
    Planning + Tool Calling documentation generator.

    Phase 1: Planning
    - Build symbol index from documents
    - LLM generates PageStructure (intro, overview, detail specs, conclusion)
    - Semantic section titles, not symbol names

    Phase 2: Generation
    - Iterate through detail sections
    - Tool calling for each section (commit_section, request_next_section)
    - Context swapping between sections
    """

    # Token budget per section (input context)
    SECTION_TOKEN_BUDGET = 25_000

    def __init__(
        self,
        llm: BaseLanguageModel,
        expanded_docs: list[Any],
        page_spec: PageSpec,
        repo_context: str,
        repository_url: str,
        wiki_style: str,
        target_audience: str,
        graph: Any | None = None,
        retriever_stack: Any | None = None,
    ):
        self.llm = llm
        self.page_spec = page_spec
        self.repo_context = repo_context
        self.repository_url = repository_url
        self.wiki_style = wiki_style
        self.target_audience = target_audience
        self.graph = graph
        self.retriever_stack = retriever_stack

        # Build document index
        self.all_docs = {}
        self.all_rel_paths = set()

        for doc in expanded_docs:
            doc_id = self._get_doc_id(doc)
            self.all_docs[doc_id] = doc

            # Collect file paths
            rel_path = doc.metadata.get("rel_path") or doc.metadata.get("source") or doc.metadata.get("file_path", "")
            if rel_path:
                self.all_rel_paths.add(rel_path)

        logger.info(
            f"AgenticDocGeneratorV2 initialized: {len(self.all_docs)} docs, {len(self.all_rel_paths)} unique paths"
        )

        # These will be set during planning
        self.page_structure: PageStructure | None = None
        self.planned_sections: list[PlannedSection] = []
        self.symbol_index: list[SymbolSummary] = []

        # Stage B selection controls
        self.use_stage_b = os.getenv("WIKIS_AGENTIC_STAGE_B", "1") == "1"
        self.stage_b_trace = os.getenv("WIKIS_AGENTIC_STAGE_B_TRACE", "0") == "1"

        # State for generation phase
        self.doc_state = DocumentationState()
        self.committed_content: list[str] = []
        self.current_section_idx = 0

        # Tool definitions
        self.tools = get_agentic_tools()

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

    def _maybe_drop_full_doc_cache(self, stage: str) -> None:
        """Optionally drop full doc cache to reduce memory after Stage B."""
        drop_requested = self._get_env_int("WIKIS_AGENTIC_DROP_EXPANDED_DOCS", 0) == 1
        if not drop_requested:
            return

        force_drop = self._get_env_int("WIKIS_AGENTIC_DROP_EXPANDED_DOCS_FORCE", 0) == 1

        if not self.use_stage_b:
            logger.info(f"[MEM] Drop requested at {stage} but Stage B is disabled; keeping full doc cache")
            return

        if not self.planned_sections:
            logger.info(f"[MEM] Drop requested at {stage} but no planned sections; keeping full doc cache")
            return

        missing_docs = [section.title for section in self.planned_sections if not section.section_docs]
        if missing_docs and not force_drop:
            logger.info(
                f"[MEM] Not dropping full doc cache at {stage}; sections without docs: "
                f"{', '.join(missing_docs[:8])}{'...' if len(missing_docs) > 8 else ''}"
            )
            return

        dropped_docs = len(self.all_docs)
        dropped_paths = len(self.all_rel_paths)
        self.all_docs.clear()
        self.all_rel_paths.clear()
        gc.collect()
        logger.info(f"[MEM] Dropped full doc cache at {stage}: docs={dropped_docs}, paths={dropped_paths}")

    def _get_doc_source(self, doc) -> str:
        """Prefer relative paths for citations and context."""
        return doc.metadata.get("rel_path") or doc.metadata.get("source") or doc.metadata.get("file_path") or "unknown"

    def _bind_llm_max_tokens(self, max_tokens: int | None):
        """Bind max_tokens to the LLM when supported."""
        if not max_tokens or max_tokens <= 0:
            return self.llm
        if hasattr(self.llm, "bind"):
            try:
                return self.llm.bind(max_tokens=max_tokens)
            except Exception as e:
                logger.warning(f"Failed to bind max_tokens={max_tokens}: {e}")
        return self.llm

    def _bind_llm_max_tokens_on(self, llm: BaseLanguageModel, max_tokens: int | None):
        """Bind max_tokens to a provided LLM instance when supported."""
        if not max_tokens or max_tokens <= 0:
            return llm
        if hasattr(llm, "bind"):
            try:
                return llm.bind(max_tokens=max_tokens)
            except Exception as e:
                logger.warning(f"Failed to bind max_tokens={max_tokens} on bound LLM: {e}")
        return llm

    def _truncate_text_to_tokens(self, text: str, max_tokens: int) -> str:
        """Best-effort truncation using token estimation (chars/4)."""
        if not text or max_tokens <= 0:
            return ""
        max_chars = max_tokens * 4
        if len(text) <= max_chars:
            return text
        truncated = text[:max_chars].rstrip()
        return f"{truncated}\n... [truncated]"

    def _build_repo_outline(self) -> str:
        """Build a compact repo outline from known paths for planning context."""
        if not self.all_rel_paths:
            return "*No repository outline available*"

        max_dirs = self._get_env_int("WIKIS_AGENTIC_OUTLINE_DIRS_MAX", 25)
        max_files = self._get_env_int("WIKIS_AGENTIC_OUTLINE_FILES_MAX", 20)

        dir_counts: dict[str, int] = {}
        root_files = []

        for path in sorted(self.all_rel_paths):
            if not path or path.startswith("/"):
                continue
            parts = path.split("/")
            if len(parts) == 1:
                root_files.append(parts[0])
            else:
                top = parts[0]
                dir_counts[top] = dir_counts.get(top, 0) + 1

        lines = ["Top-level directories:"]
        for name, count in sorted(dir_counts.items(), key=lambda x: (-x[1], x[0]))[:max_dirs]:
            lines.append(f"- {name}/ ({count} files)")

        if root_files:
            lines.append("\nTop-level files:")
            for file_name in root_files[:max_files]:
                lines.append(f"- {file_name}")

        return "\n".join(lines)

    def _infer_section_intent(self, section: PlannedSection) -> str:
        """Infer section intent from title/description if not provided."""
        if section.intent:
            return section.intent

        title = (section.title or "").lower()
        desc = (section.description or "").lower()
        text = f"{title} {desc}"

        if any(k in text for k in ["graph", "langgraph", "workflow", "pipeline", "orchestration", "state"]):
            return "graph_construction"
        if any(k in text for k in ["config", "settings", "env", "environment", "yaml", "toml"]):
            return "config"
        if any(k in text for k in ["api", "endpoint", "route", "http", "rest", "grpc"]):
            return "api"
        if any(k in text for k in ["runtime", "execution", "run", "cli", "command", "invocation"]):
            return "runtime"
        if any(k in text for k in ["auth", "security", "token", "permission", "oauth"]):
            return "security"
        return "general"

    def _get_section_queries(self, section: PlannedSection) -> list[str]:
        """Get retrieval queries for a section with fallback heuristics."""
        queries = [q for q in (section.retrieval_queries or []) if q and q.strip()]
        if queries:
            return queries

        fallback = []
        if section.title:
            fallback.append(section.title)
        if section.description:
            fallback.append(section.description)

        # Include page context for better recall
        if self.page_spec.page_name:
            fallback.append(self.page_spec.page_name)
        if self.page_spec.content_focus:
            fallback.append(self.page_spec.content_focus)

        merged = " ".join(fallback).strip()
        return [merged] if merged else []

    def _dedupe_documents(self, documents: list[Document]) -> list[Document]:
        """Deduplicate documents by file path, symbol name, and type."""
        seen = set()
        unique_docs = []
        for doc in documents:
            symbol_name = doc.metadata.get("symbol_name", "")
            file_path = doc.metadata.get("file_path", doc.metadata.get("source", ""))
            symbol_type = doc.metadata.get("symbol_type", doc.metadata.get("type", ""))
            key = f"{file_path}::{symbol_name}::{symbol_type}"
            if key not in seen:
                seen.add(key)
                unique_docs.append(doc)
        return unique_docs

    def _get_graph_node_id(self, doc: Document) -> str | None:
        """Resolve graph node ID for a document using graph indexes when available."""
        if not self.graph:
            return None

        node_id = doc.metadata.get("node_id")
        if node_id and self.graph.has_node(node_id):
            return node_id

        symbol_name = doc.metadata.get("symbol_name", "")
        file_path = doc.metadata.get("file_path") or doc.metadata.get("source", "")
        language = doc.metadata.get("language", "")

        if hasattr(self.graph, "_node_index") and self.graph._node_index:
            key = (symbol_name, file_path, language)
            node_id = self.graph._node_index.get(key)
            if node_id:
                return node_id

        # Fallback: scan graph nodes (best effort)
        for node_id, node_data in self.graph.nodes(data=True):
            if node_data.get("symbol_name") == symbol_name and node_data.get("file_path") == file_path:
                return node_id
        return None

    def _create_document_from_graph_node(
        self, node_id: str, expansion_reason: str | None = None, source_symbol: str | None = None
    ) -> Document | None:
        """Create a Document from a graph node with minimal metadata."""
        if not self.graph or not self.graph.has_node(node_id):
            return None

        node_data = self.graph.nodes[node_id]
        symbol = node_data.get("symbol")

        if symbol and hasattr(symbol, "source_text") and symbol.source_text:
            content = symbol.source_text
        else:
            content = node_data.get("content") or ""

        if not content:
            symbol_name = node_data.get("symbol_name", node_id)
            symbol_type = node_data.get("symbol_type", "symbol")
            content = f"Symbol: {symbol_name} ({symbol_type})"

        rel_path = node_data.get("rel_path") or node_data.get("file_path", "")

        metadata = {
            "node_id": node_id,
            "symbol_name": node_data.get("symbol_name", ""),
            "symbol_type": node_data.get("symbol_type", ""),
            "file_path": node_data.get("file_path", ""),
            "rel_path": rel_path,
            "language": node_data.get("language", ""),
            "chunk_type": "symbol",
            "source": rel_path,
        }

        if expansion_reason:
            metadata["expansion_reason"] = expansion_reason
        if source_symbol:
            metadata["expansion_source"] = source_symbol

        return Document(page_content=content, metadata=metadata)

    def _get_relationship_policy(self, intent: str) -> dict[str, Any]:
        """Return expansion policy for a given section intent."""
        all_types = {rel.value for rel in RelationshipType}

        mandatory = {
            "creates",
            "instantiates",
            "inheritance",
            "implementation",
        }

        # Intent overrides
        if intent in {"graph_construction", "runtime"}:
            mandatory = mandatory | {"defines", "contains"}
        if intent in {"api", "security"}:
            mandatory = mandatory | {"implementation", "inheritance", "defines"}

        capped = all_types - mandatory

        cap_per_edge = self._get_env_int("WIKIS_AGENTIC_EXPAND_CAP_PER_EDGE", 3)
        max_hops = self._get_env_int("WIKIS_AGENTIC_EXPAND_MAX_HOPS", 1)

        return {
            "mandatory": mandatory,
            "capped": capped,
            "cap_per_edge": cap_per_edge,
            "max_hops": max_hops,
            "all_types": all_types,
        }

    def _expand_seeds_with_graph(self, seed_docs: list[Document], section: PlannedSection) -> list[Document]:
        """Expand seed docs using the relationship graph with policy caps."""
        if not self.graph or not seed_docs:
            return []

        intent = self._infer_section_intent(section)
        policy = self._get_relationship_policy(intent)
        mandatory = policy["mandatory"]
        capped = policy["capped"]
        cap_per_edge = policy["cap_per_edge"]
        max_hops = policy["max_hops"]

        expanded_docs: list[Document] = []
        visited_nodes: set[str] = set()

        def _add_neighbor(node_id: str, rel_type: str, source_symbol: str):
            if node_id in visited_nodes:
                return
            visited_nodes.add(node_id)
            doc = self._create_document_from_graph_node(node_id, expansion_reason=rel_type, source_symbol=source_symbol)
            if doc:
                expanded_docs.append(doc)

        for seed in seed_docs:
            source_symbol = seed.metadata.get("symbol_name", "") or seed.metadata.get("source", "")
            node_id = self._get_graph_node_id(seed)
            if not node_id:
                continue

            visited_nodes.add(node_id)
            queue = [(node_id, 0)]
            per_edge_counts: dict[str, int] = {}

            while queue:
                current, depth = queue.pop(0)
                if depth >= max_hops:
                    continue

                # Outgoing edges
                for _, succ, edge_data in self.graph.out_edges(current, data=True):
                    rel_type = (edge_data.get("relationship_type") or "").lower()
                    if not rel_type:
                        continue
                    if rel_type in mandatory:
                        _add_neighbor(succ, rel_type, source_symbol)
                        queue.append((succ, depth + 1))
                    elif rel_type in capped:
                        count = per_edge_counts.get(rel_type, 0)
                        if count < cap_per_edge:
                            per_edge_counts[rel_type] = count + 1
                            _add_neighbor(succ, rel_type, source_symbol)
                            queue.append((succ, depth + 1))

                # Incoming edges
                for pred, _, edge_data in self.graph.in_edges(current, data=True):
                    rel_type = (edge_data.get("relationship_type") or "").lower()
                    if not rel_type:
                        continue
                    if rel_type in mandatory:
                        _add_neighbor(pred, rel_type, source_symbol)
                        queue.append((pred, depth + 1))
                    elif rel_type in capped:
                        count = per_edge_counts.get(rel_type, 0)
                        if count < cap_per_edge:
                            per_edge_counts[rel_type] = count + 1
                            _add_neighbor(pred, rel_type, source_symbol)
                            queue.append((pred, depth + 1))

        return expanded_docs

    def _select_symbols_for_section(
        self, section: PlannedSection, seed_docs: list[Document], expanded_docs: list[Document]
    ) -> None:
        """Populate section symbol lists based on retrieval and expansion results."""
        primary_limit = self._get_env_int("WIKIS_AGENTIC_PRIMARY_LIMIT", 10)

        primary = []
        supporting = []
        symbol_paths = []

        def _symbol_name(doc: Document) -> str:
            return doc.metadata.get("symbol_name") or doc.metadata.get("source") or ""

        def _symbol_path(doc: Document) -> str:
            return doc.metadata.get("rel_path") or doc.metadata.get("source") or doc.metadata.get("file_path") or ""

        # Seeds are primary (up to limit)
        for doc in seed_docs:
            name = _symbol_name(doc)
            if not name or name in primary:
                continue
            if len(primary) < primary_limit:
                primary.append(name)
            else:
                supporting.append(name)
            path = _symbol_path(doc)
            if path:
                symbol_paths.append(path)

        # Expanded docs become supporting by default
        for doc in expanded_docs:
            name = _symbol_name(doc)
            if not name or name in primary or name in supporting:
                continue
            supporting.append(name)
            path = _symbol_path(doc)
            if path:
                symbol_paths.append(path)

        section.primary_symbols = primary
        section.supporting_symbols = supporting
        section.symbol_paths = list(dict.fromkeys(symbol_paths))

    def _build_page_context_snippets(self) -> str:
        """Build compact page-level context from section seeds."""
        per_section = self._get_env_int("WIKIS_AGENTIC_PAGE_CONTEXT_PER_SECTION", 4)
        max_sections = self._get_env_int("WIKIS_AGENTIC_PAGE_CONTEXT_SECTIONS_MAX", 8)

        lines = []
        for section in self.planned_sections[:max_sections]:
            if not section.section_docs:
                continue

            lines.append(f"## {section.title}")
            added = 0
            docs = section.section_docs
            doc_index = self._build_doc_index(docs)

            for symbol_name in section.primary_symbols:
                if added >= per_section:
                    break
                doc = self._find_doc_by_symbol(symbol_name, doc_index, docs)
                if not doc:
                    continue
                symbol_type = doc.metadata.get("symbol_type", doc.metadata.get("type", "symbol"))
                source = self._get_doc_source(doc)
                brief = ""
                if doc.metadata.get("docstring"):
                    brief = doc.metadata.get("docstring", "").split("\n")[0].strip()
                if not brief:
                    first_line = (doc.page_content or "").strip().split("\n")[:1]
                    brief = first_line[0][:160] if first_line else ""
                lines.append(f"- **{symbol_name}** ({symbol_type}) <code_context {source}>")
                if brief:
                    lines.append(f"  - {brief}")
                added += 1

        if not lines:
            return "*No page-level context available*"
        return "\n".join(lines)

    def _generate_intro_overview_conclusion(self, config: RunnableConfig) -> None:
        """Generate intro/overview/conclusion after Stage B using compact context."""
        if not self.page_structure:
            return

        generate_intro = os.getenv("WIKIS_AGENTIC_GENERATE_INTRO", "1") == "1"
        generate_conclusion = os.getenv("WIKIS_AGENTIC_GENERATE_CONCLUSION", "0") == "1"
        generate_overview = os.getenv("WIKIS_AGENTIC_GENERATE_OVERVIEW", "0") == "1"

        if not (generate_intro or generate_conclusion or generate_overview):
            return

        overview_max = self._get_env_int("WIKIS_AGENTIC_OVERVIEW_SECTIONS_MAX", 1)
        max_tokens = self._get_env_int("WIKIS_AGENTIC_INTRO_MAX_OUTPUT_TOKENS", 4096)

        section_list = "\n".join(f"- {s.title}" for s in self.planned_sections)
        page_context = self._build_page_context_snippets()

        prompt = ChatPromptTemplate.from_messages(
            [
                ("system", "Return ONLY valid JSON. No markdown fences. No prose."),
                ("human", AGENTIC_INTRO_OVERVIEW_PROMPT),
            ]
        )

        prompt_messages = prompt.format_messages(
            page_name=self.page_spec.page_name,
            page_description=self.page_spec.description,
            content_focus=self.page_spec.content_focus,
            section_list=section_list,
            page_context=page_context,
            overview_max=overview_max,
        )

        llm_for_intro = self._bind_llm_max_tokens(max_tokens)

        try:
            llm_message = llm_for_intro.invoke(prompt_messages, config=config)
            raw_text = self._normalize_llm_content_to_text(llm_message)
            data = self._extract_json_object(raw_text)
            bundle = IntroOverviewBundle.model_validate(data)

            if generate_intro:
                self.page_structure.introduction = bundle.introduction
            if generate_overview:
                self.page_structure.overview_sections = bundle.overview_sections
            else:
                self.page_structure.overview_sections = []
            if generate_conclusion:
                self.page_structure.conclusion = bundle.conclusion
            else:
                self.page_structure.conclusion = ""
        except Exception as e:
            logger.warning(f"[PLAN] Intro/overview generation failed: {e}")

    def _populate_sections_with_retrieval(self) -> None:
        """Populate section symbols using retrieval + graph expansion (no LLM)."""
        if not self.retriever_stack:
            logger.warning("[PLAN] Stage B retrieval skipped: retriever_stack unavailable")
            return

        seed_k = self._get_env_int("WIKIS_AGENTIC_SECTION_SEED_K", 15)

        for section in self.planned_sections:
            queries = self._get_section_queries(section)
            if not queries:
                logger.warning(f"[PLAN] No queries for section '{section.title}', skipping Stage B")
                continue

            seed_docs: list[Document] = []
            for query in queries:
                try:
                    docs = self.retriever_stack.search_repository(query, k=seed_k, apply_expansion=False)
                    seed_docs.extend(docs or [])
                except Exception as e:
                    logger.warning(f"[PLAN] Retrieval failed for query '{query}': {e}")

            seed_docs = self._dedupe_documents(seed_docs)
            expanded_docs = self._expand_seeds_with_graph(seed_docs, section)

            # Filter to architectural symbols only
            def _is_arch(doc: Document) -> bool:
                symbol_type = doc.metadata.get("symbol_type", doc.metadata.get("type", "")).lower()
                chunk_type = doc.metadata.get("chunk_type", "").lower()
                if chunk_type in ["text_document", "text_chunk", "documentation"]:
                    return True
                return symbol_type in ARCHITECTURAL_SYMBOL_TYPES

            expanded_docs = [doc for doc in expanded_docs if _is_arch(doc)]

            # Merge docs for section context
            combined_docs = self._dedupe_documents(seed_docs + expanded_docs)
            section.section_docs = combined_docs

            # Fill symbols and paths
            self._select_symbols_for_section(section, seed_docs, expanded_docs)

            if self.stage_b_trace:
                logger.info(
                    f"[PLAN][STAGE_B] '{section.title}': queries={len(queries)}, seeds={len(seed_docs)}, "
                    f"expanded={len(expanded_docs)}, primary={len(section.primary_symbols)}, "
                    f"supporting={len(section.supporting_symbols)}"
                )

    def _get_doc_id(self, doc) -> str:
        """Get unique identifier for a document"""
        return doc.metadata.get("node_id") or doc.metadata.get("symbol_name") or doc.metadata.get("source", "unknown")

    def _build_doc_index(self, docs: list[Any]) -> dict[str, Any]:
        """Build a document index for quick lookup within a section."""
        return {self._get_doc_id(doc): doc for doc in docs}

    # =========================================================================
    # PHASE 1: PLANNING
    # =========================================================================

    def _build_symbol_index(self, max_symbols: int | None = None, fast_estimate: bool = False) -> list[SymbolSummary]:
        """
        Build lightweight symbol index from documents for planning.

        Only includes architectural symbols (classes, functions, etc.)
        Excludes methods, variables, fields (implementation details).
        """
        start_time = time.time()
        logger.info(f"[PLAN] Building symbol index from {len(self.all_docs)} documents")

        token_counter = None if fast_estimate else get_token_counter()
        symbol_index = []
        seen_keys = set()

        for _doc_id, doc in self.all_docs.items():
            metadata = doc.metadata
            content = doc.page_content
            chunk_type = metadata.get("chunk_type", "unknown")

            # Get file path
            file_path = metadata.get("rel_path") or metadata.get("source") or metadata.get("file_path", "")

            # Handle CODE SYMBOLS (have symbol_name)
            symbol_name = metadata.get("symbol_name", "")
            if symbol_name:
                symbol_type = metadata.get("symbol_type", metadata.get("type", "unknown"))

                # Skip non-architectural types
                if symbol_type.lower() not in ARCHITECTURAL_SYMBOL_TYPES:
                    continue

                qualified_key = f"{file_path}::{symbol_name}"
                if qualified_key in seen_keys:
                    continue
                seen_keys.add(qualified_key)

                # Build brief from metadata
                brief_parts = []

                docstring = metadata.get("docstring", "")
                if docstring:
                    first_line = docstring.split("\n")[0].strip()
                    if first_line:
                        brief_parts.append(first_line[:100])

                # Add signature info
                params = metadata.get("parameters", [])
                return_type = metadata.get("return_type", "")
                if params or return_type:
                    sig = f"({', '.join(p[:20] for p in params[:5])})"
                    if return_type:
                        sig += f" -> {return_type[:30]}"
                    brief_parts.append(sig)

                brief = " | ".join(brief_parts) if brief_parts else ""

                # Relationships from expansion metadata
                relationships = self._get_relationships_from_metadata(metadata)

                estimated_tokens = metadata.get("estimated_tokens")
                if estimated_tokens is None:
                    if token_counter is None:
                        estimated_tokens = max(len(content) // 4, 1)
                    else:
                        estimated_tokens = token_counter.count(content)

                symbol_index.append(
                    SymbolSummary(
                        name=symbol_name,
                        qualified_name=qualified_key,
                        symbol_type=symbol_type,
                        file_path=file_path,
                        brief=brief[:200],
                        relationships=relationships,
                        estimated_tokens=estimated_tokens,
                    )
                )

                if max_symbols and len(symbol_index) >= max_symbols:
                    break

            # Handle TEXT DOCUMENTS
            elif chunk_type == "text" or file_path:
                qualified_key = file_path
                if qualified_key in seen_keys:
                    continue
                seen_keys.add(qualified_key)

                # Brief from first lines
                lines = content.strip().split("\n")[:3]
                brief = " ".join(line.strip() for line in lines if line.strip())[:200]

                estimated_tokens = metadata.get("estimated_tokens")
                if estimated_tokens is None:
                    if token_counter is None:
                        estimated_tokens = max(len(content) // 4, 1)
                    else:
                        estimated_tokens = token_counter.count(content)

                symbol_index.append(
                    SymbolSummary(
                        name=file_path,
                        qualified_name=qualified_key,
                        symbol_type="documentation",
                        file_path=file_path,
                        brief=brief or file_path,
                        relationships=[],
                        estimated_tokens=estimated_tokens,
                    )
                )

                if max_symbols and len(symbol_index) >= max_symbols:
                    break

        elapsed = time.time() - start_time
        logger.info(f"[PLAN] Symbol index built: {len(symbol_index)} items in {elapsed:.2f}s")

        return symbol_index

    def _get_relationships_from_metadata(self, metadata: dict[str, Any]) -> list[str]:
        """Extract relationship hints from document metadata"""
        relationships = []

        expansion_reason = metadata.get("expansion_reason", "")
        expansion_source = metadata.get("expansion_source", "")

        if expansion_reason:
            if expansion_source:
                relationships.append(f"{expansion_reason} {expansion_source}")
            else:
                relationships.append(expansion_reason)

        return relationships

    def _format_symbol_index_for_prompt(self, max_symbols: int | None = None) -> str:
        """Format symbol index for planning prompt"""
        lines = []
        symbols = self.symbol_index
        truncated = False
        if max_symbols and max_symbols > 0 and len(symbols) > max_symbols:
            symbols = symbols[:max_symbols]
            truncated = True
        for sym in symbols:
            entry = f"- **{sym.name}** ({sym.symbol_type})\n"
            entry += f"  - Path: `{sym.file_path}`\n"
            if sym.brief:
                entry += f"  - Brief: {sym.brief}\n"
            if sym.relationships:
                entry += f"  - Relationships: {', '.join(sym.relationships)}\n"
            entry += f"  - Size: ~{sym.estimated_tokens} tokens\n"
            lines.append(entry)
        if truncated:
            lines.append(f"- ... truncated ({len(self.symbol_index) - len(symbols)} more symbols)")
        return "\n".join(lines)

    def _normalize_llm_content_to_text(self, message: Any) -> str:
        """Best-effort conversion of a LangChain message content to plain text."""
        content = getattr(message, "content", message)
        if content is None:
            return ""
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            parts: list[str] = []
            for item in content:
                if item is None:
                    continue
                if isinstance(item, str):
                    parts.append(item)
                    continue
                if isinstance(item, dict):
                    # Common OpenAI content parts: {"type": "text", "text": "..."}
                    text = item.get("text") or item.get("content") or ""
                    if text:
                        parts.append(str(text))
                    continue
                parts.append(str(item))
            return "\n".join(parts).strip()
        return str(content).strip()

    def _extract_json_object(self, text: str) -> dict[str, Any]:
        """Extract a JSON object from an LLM response (supports raw JSON or fenced blocks)."""
        raw = (text or "").strip()
        if not raw:
            raise ValueError("Empty LLM response")

        # Direct parse
        try:
            parsed = json.loads(raw)
            if isinstance(parsed, dict):
                return parsed
        except Exception:  # noqa: S110
            pass

        # ```json ... ```
        fence_match = re.search(r"```json\s*(.*?)\s*```", raw, re.DOTALL | re.IGNORECASE)
        if fence_match:
            return json.loads(fence_match.group(1).strip())

        # ``` ... ``` (no language)
        fence_any = re.search(r"```\s*(\{.*?\})\s*```", raw, re.DOTALL)
        if fence_any:
            return json.loads(fence_any.group(1).strip())

        # Any {...}
        # Non-greedy to reduce the chance of capturing trailing prose that contains braces.
        obj_match = re.search(r"\{.*?\}", raw, re.DOTALL)
        if obj_match:
            return json.loads(obj_match.group(0))

        raise ValueError("Could not extract JSON object from LLM response")

    def _run_planning_phase(self, config: RunnableConfig) -> PageStructure:
        """
        Run LLM planning to generate PageStructure.

        Returns structured output with:
        - Introduction (written)
        - Overview sections (written)
        - Detail section specs (for generation)
        - Conclusion (written)
        """
        start_time = time.time()
        trace_plan = self._get_env_int("WIKIS_AGENTIC_PLAN_TRACE", 0) == 1
        logger.info(f"[PLAN] Starting planning phase for '{self.page_spec.page_name}'")

        max_symbol_index = self._get_env_int("WIKIS_AGENTIC_SYMBOL_INDEX_MAX", 250)

        plan_mode = os.getenv("WIKIS_AGENTIC_PLAN_MODE", "queries").lower()
        fast_estimate = self._get_env_int("WIKIS_AGENTIC_FAST_TOKEN_ESTIMATE", 1) == 1

        symbol_index_str = ""
        all_rel_paths_str = ""
        repo_outline = self._build_repo_outline()

        if plan_mode == "symbols":
            # Build symbol index (legacy mode)
            symbol_start = time.perf_counter()
            self.symbol_index = self._build_symbol_index(max_symbols=max_symbol_index, fast_estimate=fast_estimate)
            if trace_plan:
                logger.info(
                    f"[PLAN][TRACE] Symbol index build: {time.perf_counter() - symbol_start:.2f}s "
                    f"(symbols={len(self.symbol_index)}, fast_estimate={fast_estimate})"
                )

            # Format for prompt
            max_paths = self._get_env_int("WIKIS_AGENTIC_PATHS_MAX", 120)

            format_start = time.perf_counter()
            symbol_index_str = self._format_symbol_index_for_prompt(max_symbols=max_symbol_index)
            all_rel_paths_str = "\n".join(sorted(self.all_rel_paths)[:max_paths])
            if trace_plan:
                logger.info(
                    f"[PLAN][TRACE] Prompt format: {time.perf_counter() - format_start:.2f}s "
                    f"(symbol_index_chars={len(symbol_index_str)}, paths_chars={len(all_rel_paths_str)})"
                )
        else:
            if trace_plan:
                logger.info(
                    f"[PLAN][TRACE] Planning mode: queries (no symbol index), outline_chars={len(repo_outline)}"
                )

        # Format for prompt
        max_paths = self._get_env_int("WIKIS_AGENTIC_PATHS_MAX", 120)

        format_start = time.perf_counter()
        symbol_index_str = self._format_symbol_index_for_prompt(max_symbols=max_symbol_index)
        all_rel_paths_str = "\n".join(sorted(self.all_rel_paths)[:max_paths])
        if trace_plan:
            logger.info(
                f"[PLAN][TRACE] Prompt format: {time.perf_counter() - format_start:.2f}s "
                f"(symbol_index_chars={len(symbol_index_str)}, paths_chars={len(all_rel_paths_str)})"
            )

        if plan_mode == "symbols":
            logger.info(f"[PLAN] Symbol index: {len(self.symbol_index)} symbols")
            logger.info(f"[PLAN] File paths: {len(self.all_rel_paths)} paths")

        # Build planning prompt
        planning_prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    "You are a documentation architect creating comprehensive page structures. "
                    "Generate content that serves both technical and non-technical audiences. "
                    "Include Mermaid diagrams where they would help understanding.",
                ),
                ("human", PAGE_STRUCTURE_PLANNING_PROMPT),
            ]
        )

        logger.info("[PLAN] Invoking LLM for planning...")

        # Toggle: skip repository_context for token savings test
        effective_repo_context = "" if SKIP_REPO_CONTEXT_FOR_PAGES else self.repo_context
        if SKIP_REPO_CONTEXT_FOR_PAGES:
            logger.info(
                f"⚡ SKIP_REPO_CONTEXT_FOR_PAGES=1: Omitting {len(self.repo_context or ''):,} char repository_context (agentic planning)"
            )

        prompt_messages = planning_prompt.format_messages(
            repository_context=effective_repo_context,
            page_name=self.page_spec.page_name,
            page_description=self.page_spec.description,
            content_focus=self.page_spec.content_focus,
            key_files=", ".join(self.page_spec.key_files or []),
            target_folders=", ".join(self.page_spec.target_folders or []),
            all_rel_paths=all_rel_paths_str,
            symbol_index=symbol_index_str,
            total_symbols=len(self.symbol_index),
            repo_outline=repo_outline,
        )

        if trace_plan:
            token_counter = get_token_counter()
            prompt_text = "\n\n".join(self._normalize_llm_content_to_text(message) for message in prompt_messages)
            logger.info(
                "[PLAN][TRACE] Prompt size: chars=%s, tokens=%s (fallback=%s)",
                len(prompt_text),
                token_counter.count(prompt_text),
                not token_counter.is_accurate,
            )
            logger.info(
                "[PLAN][TRACE] Prompt parts: repo_context_chars=%s, outline_chars=%s, "
                "paths_chars=%s, symbol_index_chars=%s",
                len(self.repo_context or ""),
                len(repo_outline or ""),
                len(all_rel_paths_str or ""),
                len(symbol_index_str or ""),
            )

        last_raw_text: str = ""
        last_err: Exception | None = None

        # Some OpenAI-compatible proxies intermittently emit malformed streaming events
        # (e.g., message.content=None / role=None). Force non-streaming for planning.
        plan_max_tokens = self._get_env_int("WIKIS_AGENTIC_PLAN_MAX_OUTPUT_TOKENS", 16_000)
        llm_for_planning = self._bind_llm_max_tokens(plan_max_tokens)
        if trace_plan:
            logger.info(f"[PLAN][TRACE] plan_max_tokens={plan_max_tokens}")

        # Some OpenAI-compatible proxies intermittently break response_format + streaming.
        # We avoid `with_structured_output` here and instead parse JSON from text.
        for attempt in (1, 2):
            try:
                invoke_start = time.perf_counter()
                llm_message = llm_for_planning.invoke(prompt_messages, config=config)
                if trace_plan:
                    logger.info(
                        f"[PLAN][TRACE] LLM invoke: {time.perf_counter() - invoke_start:.2f}s (attempt={attempt})"
                    )
                raw_text = self._normalize_llm_content_to_text(llm_message)
                last_raw_text = raw_text
                parse_start = time.perf_counter()
                data = self._extract_json_object(raw_text)
                response = PageStructure.model_validate(data)
                if trace_plan:
                    logger.info(
                        f"[PLAN][TRACE] Parse+validate: {time.perf_counter() - parse_start:.2f}s "
                        f"(raw_chars={len(raw_text)})"
                    )
                break
            except (json.JSONDecodeError, ValueError, ValidationError) as e:
                last_err = e
                raw = last_raw_text or ""
                if os.getenv("WIKIS_LOG_LLM_PAGE_PLAN_RAW") == "1":
                    raw_len = len(raw)
                    prefix = raw[:4000].replace("\n", "\\n")
                    suffix = raw[-800:].replace("\n", "\\n") if raw_len > 4000 else ""
                    logger.warning(
                        f"[PLAN] Planning parse/validation failed (attempt={attempt}/2): {e}. Raw length={raw_len}. "
                        f"Prefix(4k)='{prefix}'" + (f" Suffix(800)='{suffix}'" if suffix else "")
                    )
                else:
                    prefix = raw[:500].replace("\n", "\\n")
                    logger.warning(
                        f"[PLAN] Planning parse/validation failed (attempt={attempt}/2): {e}. Response prefix: {prefix}"
                    )
                if attempt == 1:
                    # One retry with an explicit JSON-only reminder.
                    prompt_messages = [
                        SystemMessage(content="Return ONLY valid JSON. No markdown fences. No prose."),
                        *prompt_messages,
                    ]
                    continue
                raise
            except Exception as e:
                last_err = e
                prefix = (last_raw_text or "")[:500].replace("\n", "\\n")
                logger.error(
                    f"[PLAN] Planning failed with unexpected error: {e}. Response prefix: {prefix}", exc_info=True
                )
                raise

        if last_err and "response" not in locals():
            raise last_err

        elapsed = time.time() - start_time
        logger.info(f"[PLAN] Planning complete in {elapsed:.2f}s")
        logger.info(f"[PLAN]   Title: '{response.title}'")
        logger.info(f"[PLAN]   Introduction: {len(response.introduction)} chars")
        logger.info(f"[PLAN]   Overview sections: {len(response.overview_sections)}")
        logger.info(f"[PLAN]   Detail sections: {len(response.detail_sections)}")
        logger.info(f"[PLAN]   Conclusion: {len(response.conclusion)} chars")

        for i, ds in enumerate(response.detail_sections):
            logger.info(
                f"[PLAN]   Detail {i + 1}: '{ds.title}' "
                f"(primary={len(ds.primary_symbols)}, supporting={len(ds.supporting_symbols)})"
            )

        return response

    def _convert_spec_to_sections(self) -> list[PlannedSection]:
        """Convert PageStructure detail specs to PlannedSection objects"""
        sections = []

        for spec in self.page_structure.detail_sections:
            sections.append(
                PlannedSection(
                    section_id=spec.section_id,
                    title=spec.title,
                    description=spec.description,
                    primary_symbols=spec.primary_symbols,
                    supporting_symbols=spec.supporting_symbols,
                    symbol_paths=spec.symbol_paths,
                    retrieval_queries=getattr(spec, "retrieval_queries", []) or [],
                    intent=getattr(spec, "intent", None),
                    suggested_elements=spec.suggested_elements,
                )
            )

        return sections

    # =========================================================================
    # PHASE 2: GENERATION
    # =========================================================================

    def _build_section_context(self, section: PlannedSection) -> None:
        """Build full code and signatures for a planned section"""
        full_code_parts = []
        signature_parts = []
        seen_for_sig = set()

        docs = section.section_docs if section.section_docs else list(self.all_docs.values())
        if not docs:
            logger.warning(f"[GEN] No documents available for '{section.title}'. Section context will be empty.")
            section.full_code = "*No primary code available*"
            section.signatures = "*No related symbols*"
            return
        doc_index = self._build_doc_index(docs)

        token_counter = get_token_counter()
        section_budget = self._get_env_int("WIKIS_AGENTIC_SECTION_CONTEXT_TOKENS", self.SECTION_TOKEN_BUDGET)
        code_budget = self._get_env_int("WIKIS_AGENTIC_SECTION_CODE_TOKENS", int(section_budget * 0.75))
        signatures_budget = self._get_env_int(
            "WIKIS_AGENTIC_SIGNATURES_TOKEN_BUDGET", max(section_budget - code_budget, 0)
        )
        max_signatures = self._get_env_int("WIKIS_AGENTIC_SIGNATURES_MAX", 80)

        full_code_tokens = 0
        signature_tokens = 0

        # Tier 1: Full code for primary symbols
        for symbol_name in section.primary_symbols:
            doc = self._find_doc_by_symbol(symbol_name, doc_index, docs)
            if doc:
                source = self._get_doc_source(doc)
                symbol_type = doc.metadata.get("symbol_type", "symbol")
                header = f"### {symbol_name} ({symbol_type})\n<code_source: {source}>\n```\n"
                footer = "\n```"
                header_tokens = token_counter.count(header) + token_counter.count(footer)

                remaining = code_budget - full_code_tokens
                if remaining <= header_tokens:
                    logger.info(
                        f"[GEN] Code budget reached for '{section.title}'. "
                        f"Added {len(full_code_parts)} primary symbols."
                    )
                    break

                content_tokens = token_counter.count(doc.page_content)
                if header_tokens + content_tokens > remaining:
                    available_tokens = max(remaining - header_tokens, 0)
                    truncated_content = self._truncate_text_to_tokens(doc.page_content, available_tokens)
                    code_block = f"{header}{truncated_content}{footer}"
                else:
                    code_block = f"{header}{doc.page_content}{footer}"

                block_tokens = token_counter.count(code_block)
                if full_code_tokens + block_tokens > code_budget and full_code_parts:
                    logger.info(
                        f"[GEN] Code budget exceeded for '{section.title}'. "
                        f"Stopping at {len(full_code_parts)} primary symbols."
                    )
                    break

                full_code_parts.append(code_block)
                full_code_tokens += block_tokens
                seen_for_sig.add(symbol_name)

        # Tier 2: Signatures for supporting symbols
        for symbol_name in section.supporting_symbols:
            if symbol_name in seen_for_sig:
                continue

            doc = self._find_doc_by_symbol(symbol_name, doc_index, docs)
            if doc:
                if len(signature_parts) >= max_signatures:
                    break
                sig = self._extract_signature(doc)
                sig_tokens = token_counter.count(sig)
                if signature_tokens + sig_tokens > signatures_budget and signature_parts:
                    break
                signature_parts.append(sig)
                signature_tokens += sig_tokens
                seen_for_sig.add(symbol_name)

        section.full_code = "\n\n".join(full_code_parts) if full_code_parts else "*No primary code available*"
        section.signatures = "\n".join(signature_parts) if signature_parts else "*No related symbols*"

        logger.info(
            f"[GEN] Built context for '{section.title}': "
            f"{len(full_code_parts)} primary, {len(signature_parts)} supporting "
            f"(code_tokens={full_code_tokens}, sig_tokens={signature_tokens})"
        )

    def _find_doc_by_symbol(
        self, symbol_name: str, doc_index: dict[str, Any] | None = None, docs_list: list[Any] | None = None
    ) -> Any | None:
        """Find a document by symbol name (exact or partial match)."""
        docs_list = docs_list or list(self.all_docs.values())
        doc_index = doc_index or self._build_doc_index(docs_list)

        # Try exact match first
        if symbol_name in doc_index:
            return doc_index[symbol_name]

        # Try partial matches
        for doc in docs_list:
            doc_symbol = doc.metadata.get("symbol_name", "")
            if doc_symbol == symbol_name:
                return doc
            if doc_symbol.endswith(f".{symbol_name}"):
                return doc
            if symbol_name in doc_symbol:
                return doc

        # Try matching by file path for text documents
        for doc in docs_list:
            rel_path = doc.metadata.get("rel_path", doc.metadata.get("source", ""))
            if rel_path == symbol_name or rel_path.endswith(symbol_name):
                return doc

        return None

    def _extract_signature(self, doc) -> str:
        """Extract a signature/summary from a document"""
        content = doc.page_content
        symbol_name = doc.metadata.get("symbol_name", "Unknown")
        symbol_type = doc.metadata.get("symbol_type", "symbol")
        source = self._get_doc_source(doc)

        lines = content.split("\n")

        # Find function/class definition
        for _i, line in enumerate(lines[:15]):
            line_stripped = line.strip()
            if line_stripped.startswith(("def ", "async def ", "class ", "function ", "interface ")):
                file_hint = f" [{source.split('/')[-1]}]" if source else ""
                return f"{symbol_name}{file_hint}: {line_stripped[:100]}"

        # For text documents
        if symbol_type in ["markdown_document", "text_document", "documentation"]:
            for line in lines[:5]:
                if line.strip() and not line.startswith("#"):
                    return f"{symbol_name} (doc): {line.strip()[:80]}..."

        # Fallback
        file_hint = f" [{source.split('/')[-1]}]" if source else ""
        return f"{symbol_name}{file_hint} ({symbol_type})"

    def _build_system_prompt(self) -> str:
        """Build the system prompt with section list"""
        section_list = "\n".join(f"- **{s.title}**: {s.description[:100]}..." for s in self.planned_sections)

        # Toggle: skip repository_context for token savings test
        effective_repo_context = "" if SKIP_REPO_CONTEXT_FOR_PAGES else (self.repo_context or "")[:3000]

        return AGENTIC_SYSTEM_PROMPT.format(
            target_audience=self.target_audience,
            wiki_style=self.wiki_style,
            repo_analysis=effective_repo_context,
            page_name=self.page_spec.page_name,
            page_description=self.page_spec.description,
            content_focus=self.page_spec.content_focus,
            section_list=section_list,
        )

    def _build_user_prompt(self, section: PlannedSection) -> str:
        """Build the dynamic user prompt for current section"""
        toc_block = self.doc_state.get_toc_for_prompt()

        cross_ref_note = ""
        if self.doc_state.completed_sections:
            cross_ref_note = "*Previous sections available for cross-reference.*"

        return AGENTIC_USER_PROMPT.format(
            section_name=section.title,
            current_idx=self.current_section_idx + 1,
            total_sections=len(self.planned_sections),
            section_description=section.description,
            suggested_elements=", ".join(section.suggested_elements)
            if section.suggested_elements
            else "diagrams, code examples, tables",
            toc_block=toc_block,
            full_code=section.full_code,
            cross_ref_note=cross_ref_note,
            signatures=section.signatures,
        )

    def _handle_tool_calls(self, response, section: PlannedSection) -> str | None:
        """Handle tool calls from LLM response"""
        next_section_name = None

        tool_calls = getattr(response, "tool_calls", None)
        if not tool_calls:
            if hasattr(response, "additional_kwargs"):
                tool_calls = response.additional_kwargs.get("tool_calls", [])

        if not tool_calls:
            logger.warning(f"[GEN] No tool calls for section '{section.title}'")
            return None

        logger.info(f"[GEN] Processing {len(tool_calls)} tool calls for '{section.title}'")

        for tool_call in tool_calls:
            if isinstance(tool_call, dict):
                tool_name = tool_call.get("name") or tool_call.get("function", {}).get("name")
                tool_args = tool_call.get("args") or tool_call.get("arguments", {})
                if isinstance(tool_args, str):
                    import json

                    try:
                        tool_args = json.loads(tool_args)
                    except (ValueError, TypeError):
                        tool_args = {}
            else:
                tool_name = getattr(tool_call, "name", None)
                tool_args = getattr(tool_call, "args", {})

            if tool_name == "commit_section":
                content = tool_args.get("content", "")
                summary = tool_args.get("summary", "Section documented")

                if content:
                    self.committed_content.append(content)
                    self.doc_state.add_section(section.title, summary)
                    logger.info(f"[GEN] ✓ Committed '{section.title}' ({len(content)} chars)")

            elif tool_name == "request_next_section":
                requested_name = tool_args.get("section_name", "")

                if requested_name:
                    valid_sections = [
                        s.title for s in self.planned_sections if s.title not in self.doc_state.completed_sections
                    ]
                    if requested_name in valid_sections:
                        next_section_name = requested_name

        return next_section_name

    # =========================================================================
    # MAIN ENTRY POINT
    # =========================================================================

    def generate(self, config: RunnableConfig) -> str:
        """
        Main generation with planning + tool calling.

        1. Planning phase: Generate PageStructure
        2. Generation phase: Iterate through detail sections with tools
        3. Assembly: Combine intro + overview + details + conclusion
        """
        if not self.all_docs:
            logger.error("No documents to generate from")
            return ""

        logger.info(f"🚀 Starting AgenticDocGeneratorV2 for '{self.page_spec.page_name}'")

        # =================================================================
        # PHASE 1: PLANNING
        # =================================================================
        logger.info("=" * 60)
        logger.info("PHASE 1: PLANNING")
        logger.info("=" * 60)

        try:
            self.page_structure = self._run_planning_phase(config)
            self.planned_sections = self._convert_spec_to_sections()
        except Exception as e:
            logger.error(f"Planning failed: {e}", exc_info=True)
            # Fall back to simple generation
            return self._generate_simple_fallback(config)

        if self.use_stage_b and self.retriever_stack:
            self._populate_sections_with_retrieval()
        elif self.use_stage_b and not self.retriever_stack:
            logger.warning("[PLAN] Stage B enabled but retriever_stack missing; using planning symbols")

        if self.use_stage_b:
            self._generate_intro_overview_conclusion(config)
            self._maybe_drop_full_doc_cache(stage="post_stage_b")

        if not self.planned_sections:
            logger.warning("No detail sections planned - generating simple page")
            return self._assemble_without_details()

        # =================================================================
        # PHASE 2: GENERATION (Tool Calling)
        # =================================================================
        logger.info("=" * 60)
        logger.info(f"PHASE 2: GENERATING {len(self.planned_sections)} DETAIL SECTIONS")
        logger.info("=" * 60)

        system_prompt = self._build_system_prompt()
        max_iterations = len(self.planned_sections) * 2
        iteration = 0

        while self.current_section_idx < len(self.planned_sections) and iteration < max_iterations:
            iteration += 1
            section = self.planned_sections[self.current_section_idx]

            if section.title in self.doc_state.completed_sections:
                self.current_section_idx += 1
                continue

            logger.info(f"[GEN] Section {self.current_section_idx + 1}/{len(self.planned_sections)}: '{section.title}'")

            # Build context for this section
            self._build_section_context(section)

            # Build prompt
            user_prompt = self._build_user_prompt(section)

            # Invoke LLM
            messages = [SystemMessage(content=system_prompt), HumanMessage(content=user_prompt)]

            try:
                section_max_tokens = self._get_env_int("WIKIS_AGENTIC_SECTION_MAX_OUTPUT_TOKENS", 16_000)
                llm_for_sections = self.llm

                if hasattr(llm_for_sections, "bind_tools"):
                    llm_with_tools = llm_for_sections.bind_tools(self.tools)
                    llm_with_tools = self._bind_llm_max_tokens_on(llm_with_tools, section_max_tokens)
                    response = llm_with_tools.invoke(messages, config=config)
                else:
                    llm_for_sections = self._bind_llm_max_tokens_on(llm_for_sections, section_max_tokens)
                    response = llm_for_sections.invoke(messages, tools=self.tools, config=config)

                next_section = self._handle_tool_calls(response, section)

                if next_section:
                    for i, s in enumerate(self.planned_sections):
                        if s.title == next_section:
                            self.current_section_idx = i
                            break
                else:
                    self.current_section_idx += 1

            except Exception as e:
                logger.error(f"Error generating '{section.title}': {e}", exc_info=True)
                self.doc_state.completed_sections.add(section.title)
                self.current_section_idx += 1

        # =================================================================
        # PHASE 3: ASSEMBLY
        # =================================================================
        logger.info("=" * 60)
        logger.info("PHASE 3: ASSEMBLY")
        logger.info("=" * 60)

        final_content = self._assemble_final_document()

        logger.info(f"✅ Generation complete: {len(final_content)} chars")
        logger.info(f"   Sections committed: {len(self.committed_content)}")

        return final_content

    def _assemble_without_details(self) -> str:
        """Assemble page with only intro, overview, and conclusion (no details)"""
        parts = []

        # Title
        parts.append(f"# {self.page_structure.title}\n")

        # Introduction
        if self.page_structure.introduction:
            parts.append(self.page_structure.introduction)
            parts.append("\n")

        # Overview sections
        for overview in self.page_structure.overview_sections:
            parts.append(f"\n## {overview.title}\n\n")
            parts.append(overview.content)
            parts.append("\n")

        # Conclusion
        if self.page_structure.conclusion:
            parts.append("\n---\n\n## Conclusion\n\n")
            parts.append(self.page_structure.conclusion)

        return "\n".join(parts)

    def _assemble_final_document(self) -> str:
        """Assemble all parts into final document"""
        parts = []

        # Title
        parts.append(f"# {self.page_structure.title}\n")

        # Introduction
        if self.page_structure.introduction:
            parts.append(self.page_structure.introduction)
            parts.append("\n")

        # Table of Contents
        toc_lines = ["\n## Contents\n"]
        toc_idx = 1

        for overview in self.page_structure.overview_sections:
            anchor = self._make_anchor(overview.title)
            toc_lines.append(f"{toc_idx}. [{overview.title}](#{anchor})")
            toc_idx += 1

        for section_title, _ in self.doc_state.toc:
            anchor = self._make_anchor(section_title)
            toc_lines.append(f"{toc_idx}. [{section_title}](#{anchor})")
            toc_idx += 1

        toc_lines.append(f"{toc_idx}. [Conclusion](#conclusion)")
        parts.append("\n".join(toc_lines))
        parts.append("\n\n---\n")

        # Overview sections (from planning)
        for overview in self.page_structure.overview_sections:
            parts.append(f"\n## {overview.title}\n\n")
            parts.append(overview.content)
            parts.append("\n\n---\n")

        # Detail sections (from tool calling)
        for content in self.committed_content:
            parts.append("\n")
            if not content.strip().startswith("##"):
                # Find section title from ToC
                idx = self.committed_content.index(content)
                if idx < len(self.doc_state.toc):
                    title = self.doc_state.toc[idx][0]
                    parts.append(f"## {title}\n\n")
            parts.append(content)
            parts.append("\n")

        # Conclusion
        if self.page_structure.conclusion:
            parts.append("\n---\n\n## Conclusion\n\n")
            parts.append(self.page_structure.conclusion)

        return "\n".join(parts)

    def _make_anchor(self, title: str) -> str:
        """Convert title to markdown anchor"""
        return title.lower().replace(" ", "-").replace(".", "").replace(":", "")

    def _generate_simple_fallback(self, config: RunnableConfig) -> str:
        """Simple fallback if planning fails"""
        logger.warning("Falling back to simple generation (no planning)")

        # Just concatenate all docs with basic structure
        parts = [f"# {self.page_spec.page_name}\n\n"]
        parts.append(f"{self.page_spec.description}\n\n")

        for doc_id, doc in list(self.all_docs.items())[:20]:
            symbol_name = doc.metadata.get("symbol_name", doc_id)
            parts.append(f"## {symbol_name}\n\n")
            parts.append(f"```\n{doc.page_content[:2000]}\n```\n\n")

        return "".join(parts)


# =============================================================================
# INTEGRATION HELPER
# =============================================================================


def should_use_agentic_mode(expanded_docs: list[Any], token_budget: int = 50_000) -> bool:
    """
    Determine if agentic mode should be used based on document count and tokens.
    """
    token_counter = get_token_counter()
    total_tokens = token_counter.count_documents(expanded_docs)

    logger.info(f"Context size check: {total_tokens:,} tokens, budget: {token_budget:,}")

    return total_tokens > token_budget
