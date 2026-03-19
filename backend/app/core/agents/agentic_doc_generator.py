"""
Agentic Documentation Generator with Context Swapping

This module implements a section-by-section documentation generation approach
that keeps context size manageable by swapping code context between sections.

Key features:
- Constant context size (~15-20K tokens per iteration)
- Two-tool workflow: commit_section + request_next_section
- Table of Contents tracking for coherence
- Cross-reference support for already-documented symbols
"""

import logging
import os
from dataclasses import dataclass, field
from typing import Any

from langchain_core.language_models import BaseLanguageModel
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.runnables import RunnableConfig

from ..state.wiki_state import PageSpec
from ..token_counter import get_token_counter

logger = logging.getLogger(__name__)

# Toggle to skip repository_context in page generation (test mode for token savings)
SKIP_REPO_CONTEXT_FOR_PAGES = os.getenv("WIKIS_SKIP_REPO_CONTEXT_FOR_PAGES", "0") == "1"


# =============================================================================
# PROMPTS FOR AGENTIC GENERATION
# =============================================================================

AGENTIC_SYSTEM_PROMPT = """You are an expert technical documentation writer creating comprehensive, enterprise-grade documentation for a software repository.

## DOCUMENTATION STYLE
- Write clear, professional documentation suitable for {target_audience}
- Style: {wiki_style}
- Use proper markdown formatting:
  - Headers (##, ###) for structure
  - Code blocks (```) for code snippets and examples
  - Tables for comparing options, listing parameters, or structured data
  - Bullet and numbered lists for sequences and enumerations
  - **Bold** and *italic* for emphasis
- Include **Mermaid diagrams** where applicable:
  - Class diagrams to show inheritance and relationships
  - Sequence diagrams for request/response flows
  - Flowcharts for decision logic or workflows
- Use **code citations** appropriately:
  - Reference specific methods, classes, or files using backticks: `ClassName.method()`
  - Include short inline code snippets to illustrate usage
  - For longer examples, use fenced code blocks with language specifiers
- Explain the "why" not just the "what"
- Include practical examples where the code supports them

## REPOSITORY OVERVIEW
{repo_analysis}

## PAGE CONTEXT
You are documenting: **{page_name}**
Description: {page_description}
Content Focus: {content_focus}
Repository: {repository_url}

## WORKFLOW

You will document this page **section-by-section** using a context-swapping approach:

1. **RECEIVE** - I provide full code for current section + signatures of related symbols
2. **WRITE** - You write comprehensive documentation for the current section
3. **COMMIT** - You call `commit_section` to save your work
4. **REQUEST** - You call `request_next_section` to get the next context (or I auto-advance)
5. **REPEAT** - Steps 1-4 repeat until all sections are documented

**CRITICAL**: Always call `commit_section` BEFORE `request_next_section`.
When you request a new section, the current code context is REPLACED.
Any uncommitted work will be LOST.

## TOOLS

### 1. commit_section

**Purpose**: Save the documentation you just wrote.

**When to call**: After you finish writing documentation for the current section, BEFORE requesting a new section.

**Parameters**:
- `content` (string, required): Your markdown documentation for this section
- `summary` (string, required): One-sentence summary for table of contents

**What happens when called**:
- Your documentation is saved to the final output
- The summary is added to the table of contents
- The current section is marked as complete
- You receive confirmation with total sections written

### 2. request_next_section

**Purpose**: Request the next section's code context, triggering a context swap.

**When to call**: AFTER calling `commit_section`, when you're ready to move to the next section.

**Parameters**:
- `section_name` (string, optional): Name of specific section to document next. Leave empty to auto-advance.

**What happens when called**:
- Current section's full code is UNLOADED from context
- Next section's full code is LOADED into context
- You receive the new code and can begin documenting

**Behavior**:
- If `section_name` is provided: Jump to that specific section
- If `section_name` is omitted or empty: Advance to the next sequential section
- If all sections are complete: You receive a completion signal

## TOOL CALL SEQUENCE

**Correct order**:
```
[Write documentation]
  ↓
commit_section(content="...", summary="...")
  ↓
request_next_section()  OR  request_next_section(section_name="...")
  ↓
[Receive new context, repeat]
```

**WRONG** (will lose work):
```
[Write documentation]
  ↓
request_next_section()  ← ERROR: Content not saved!
```

## CONTENT GUIDELINES

### Diagrams
Include Mermaid diagrams where they add clarity:

**Class relationships**:
```mermaid
classDiagram
    BaseService <|-- AuthService
    AuthService --> TokenFactory
```

**Sequence flows**:
```mermaid
sequenceDiagram
    Client->>AuthService: authenticate()
    AuthService->>TokenFactory: createToken()
    TokenFactory-->>AuthService: token
    AuthService-->>Client: response
```

### Code Citations
- Reference code elements with backticks: `ClassName`, `method()`, `module.function()`
- Show method signatures when explaining interfaces
- Include usage examples in fenced code blocks

### Tables
Use tables to present structured information:

| Parameter | Type | Description |
|-----------|------|-------------|
| `username` | `str` | User identifier |
| `password` | `str` | User credentials |

## QUALITY GUIDELINES
- Each section should be self-contained but reference other sections where relevant
- Use cross-references: "as described in [Section Name] above" for already-documented symbols
- Don't repeat full explanations - reference previous sections
- Focus on the symbols shown in full code; signatures are for context only
- Ensure proper markdown structure with appropriate heading levels
- Place diagrams after conceptual explanations, before implementation details

## END OF PAGE
When you receive a message indicating all sections are complete, do NOT call any more tools.
Simply acknowledge completion.
"""

AGENTIC_USER_PROMPT = """## CURRENT SECTION: {section_name}
**Section {current_idx} of {total_sections}**

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

### REMAINING SECTIONS
{remaining_sections_block}

---

**TASK**: Write comprehensive documentation for **{section_name}**.

When finished:
1. Call `commit_section` with your markdown and a brief summary
2. Call `request_next_section` to proceed (or specify a section name to jump)
"""


# =============================================================================
# DATA STRUCTURES
# =============================================================================


@dataclass
class SectionContext:
    """A logical grouping of related symbols for one documentation section"""

    name: str
    anchor_ids: list[str]  # Primary symbols to document (initially_retrieved=True)
    full_code: str = ""  # Combined full code for this section
    signatures: str = ""  # Signatures of related symbols

    def __post_init__(self):
        self.anchor_ids = list(self.anchor_ids) if self.anchor_ids else []


@dataclass
class DocumentationState:
    """Tracks what has been documented for coherence across sections"""

    toc: list[tuple] = field(default_factory=list)  # [(section_name, symbols, summary)]
    symbol_to_section: dict[str, str] = field(default_factory=dict)  # symbol_id -> section_name
    completed_sections: set[str] = field(default_factory=set)

    def add_section(self, section_name: str, symbols: list[str], summary: str):
        """Record a completed section"""
        self.toc.append((section_name, symbols, summary))
        self.completed_sections.add(section_name)
        for s in symbols:
            self.symbol_to_section[s] = section_name

    def get_toc_for_prompt(self) -> str:
        """Generate ToC context for the prompt"""
        if not self.toc:
            return "*This is the first section - no previous content.*"

        lines = []
        for section, symbols, summary in self.toc:
            symbol_preview = ", ".join(f"`{s}`" for s in symbols[:3])
            if len(symbols) > 3:
                symbol_preview += f" +{len(symbols) - 3} more"
            lines.append(f"- **{section}**: {summary}")
            lines.append(f"  - Covers: {symbol_preview}")
        return "\n".join(lines)

    def get_cross_reference(self, symbol_id: str) -> str | None:
        """Get cross-reference note if symbol already documented"""
        if symbol_id in self.symbol_to_section:
            return f"# see '{self.symbol_to_section[symbol_id]}' section"
        return None


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
                "description": "Save the documentation you just wrote for the current section. MUST be called before request_next_section to avoid losing work.",
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
                "description": "Request the next section's code context. This triggers a context swap. Call AFTER commit_section.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "section_name": {
                            "type": "string",
                            "description": "Name of specific section to document next. Leave empty to auto-advance to next sequential section.",
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


class AgenticDocGenerator:
    """
    Section-based documentation generator with context swapping.

    Handles large context by documenting one section at a time,
    swapping code context between sections to keep token count manageable.
    """

    # Token budget per section (conservative to leave room for generation)
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
    ):
        self.llm = llm
        self.page_spec = page_spec
        self.repo_context = repo_context
        self.repository_url = repository_url
        self.wiki_style = wiki_style
        self.target_audience = target_audience
        self.graph = graph

        # Build document index
        self.all_docs = {}
        self.anchor_docs = []
        self.expanded_deps = []

        # Debug: log first few documents' metadata keys to understand what flags exist
        if expanded_docs and len(expanded_docs) > 0:
            sample_doc = expanded_docs[0]
            logger.debug(f"Sample document metadata keys: {list(sample_doc.metadata.keys())}")
            logger.debug(
                f"Sample metadata: is_initially_retrieved={sample_doc.metadata.get('is_initially_retrieved')}, "
                f"initially_retrieved={sample_doc.metadata.get('initially_retrieved')}"
            )

        for doc in expanded_docs:
            doc_id = self._get_doc_id(doc)
            self.all_docs[doc_id] = doc

            # Check for is_initially_retrieved flag (set by retrievers.py)
            # Also check initially_retrieved for backwards compatibility with tests
            is_anchor = doc.metadata.get("is_initially_retrieved", False) or doc.metadata.get(
                "initially_retrieved", False
            )

            if is_anchor:
                self.anchor_docs.append(doc)
            else:
                self.expanded_deps.append(doc)

        logger.info(
            f"AgenticDocGenerator initialized: {len(self.anchor_docs)} anchors, {len(self.expanded_deps)} expanded deps"
        )

        # If no anchors found, log diagnostic info
        if not self.anchor_docs and expanded_docs:
            logger.warning(f"No anchor docs found! Checking all {len(expanded_docs)} docs for flags...")
            for i, doc in enumerate(expanded_docs[:5]):  # Sample first 5
                logger.warning(
                    f"  Doc {i}: symbol={doc.metadata.get('symbol_name')}, "
                    f"is_initially_retrieved={doc.metadata.get('is_initially_retrieved')}, "
                    f"initially_retrieved={doc.metadata.get('initially_retrieved')}"
                )

        # Build sections from anchor docs
        self.sections = self._cluster_into_sections()

        # State tracking
        self.doc_state = DocumentationState()
        self.committed_content: list[str] = []
        self.current_section_idx = 0

        # Toggle: skip repository_context for token savings test
        effective_repo_context = "" if SKIP_REPO_CONTEXT_FOR_PAGES else repo_context
        if SKIP_REPO_CONTEXT_FOR_PAGES:
            logger.info(
                f"⚡ SKIP_REPO_CONTEXT_FOR_PAGES=1: Omitting {len(repo_context or ''):,} char repository_context (agentic v1)"
            )

        # Build system prompt (constant across iterations)
        self.system_prompt = AGENTIC_SYSTEM_PROMPT.format(
            target_audience=target_audience,
            wiki_style=wiki_style,
            repo_analysis=effective_repo_context,
            page_name=page_spec.page_name,
            page_description=page_spec.description,
            content_focus=page_spec.content_focus,
            repository_url=repository_url,
        )

        # Tool definitions
        self.tools = get_agentic_tools()

    def _get_doc_id(self, doc) -> str:
        """Get unique identifier for a document"""
        return doc.metadata.get("node_id") or doc.metadata.get("symbol_name") or doc.metadata.get("source", "unknown")

    def _cluster_into_sections(self) -> list[SectionContext]:
        """
        Cluster documents into logical sections for documentation.

        Strategy: Create semantic sections based on symbol types and purposes,
        with a minimum of 3-4 sections for good documentation structure.
        Uses page context to create meaningful section names.
        """
        all_docs = list(self.anchor_docs) + list(self.expanded_deps)

        if not all_docs:
            logger.warning("No documents to cluster into sections")
            return []

        # Categorize documents by type and purpose
        categories = {
            "overview": [],  # Documentation, README, guides
            "core_classes": [],  # Classes, interfaces, structs
            "functions": [],  # Standalone functions
            "types": [],  # Enums, type definitions
            "configuration": [],  # Config files, constants
            "other": [],  # Everything else
        }

        for doc in all_docs:
            symbol_type = doc.metadata.get("symbol_type", "").lower()
            symbol_name = doc.metadata.get("symbol_name", "").lower()
            source = doc.metadata.get("source", "").lower()

            # Categorize based on type and naming patterns
            if symbol_type in ["markdown_document", "text_document", "documentation", "text_chunk"]:
                categories["overview"].append(doc)
            elif symbol_type in ["class", "interface", "struct", "trait", "protocol"]:
                categories["core_classes"].append(doc)
            elif symbol_type in ["function"]:
                categories["functions"].append(doc)
            elif symbol_type in ["enum", "type", "type_alias"]:
                categories["types"].append(doc)
            elif "config" in symbol_name or "config" in source or "constant" in symbol_type:
                categories["configuration"].append(doc)
            else:
                categories["other"].append(doc)

        # Build sections from non-empty categories
        sections = []
        token_counter = get_token_counter()

        # Define section metadata with readable names
        section_templates = [
            ("overview", "Overview & Documentation", "Documentation and guides"),
            ("core_classes", "Core Components", "Main classes and interfaces"),
            ("functions", "Key Functions", "Utility and helper functions"),
            ("types", "Type Definitions", "Enums and type aliases"),
            ("configuration", "Configuration", "Settings and constants"),
            ("other", "Additional Components", "Other relevant code"),
        ]

        for cat_key, section_title, _section_desc in section_templates:
            cat_docs = categories[cat_key]
            if not cat_docs:
                continue

            # Split large categories into multiple sections if needed
            section_docs = []
            section_tokens = 0
            section_num = 1

            for doc in cat_docs:
                doc_tokens = token_counter.count(doc.page_content)

                # If adding this doc exceeds budget, finalize current section
                if section_tokens + doc_tokens > self.SECTION_TOKEN_BUDGET and section_docs:
                    name = f"{section_title}" if section_num == 1 else f"{section_title} (Part {section_num})"
                    sections.append(SectionContext(name=name, anchor_ids=[self._get_doc_id(d) for d in section_docs]))
                    section_docs = []
                    section_tokens = 0
                    section_num += 1

                section_docs.append(doc)
                section_tokens += doc_tokens

            # Don't forget the last batch
            if section_docs:
                name = f"{section_title}" if section_num == 1 else f"{section_title} (Part {section_num})"
                sections.append(SectionContext(name=name, anchor_ids=[self._get_doc_id(d) for d in section_docs]))

        # If we ended up with too few sections, subdivide the largest one
        if len(sections) < 3 and sections:
            sections = self._ensure_minimum_sections(sections, min_sections=3)

        logger.info(f"Created {len(sections)} sections from {len(all_docs)} documents")
        for i, section in enumerate(sections):
            logger.info(f"  Section {i + 1}: '{section.name}' ({len(section.anchor_ids)} docs)")

        return sections

    def _ensure_minimum_sections(self, sections: list[SectionContext], min_sections: int = 3) -> list[SectionContext]:
        """Ensure we have at least min_sections by subdividing large sections"""
        if len(sections) >= min_sections:
            return sections

        result = []
        sections_needed = min_sections - len(sections) + 1

        # Find the largest section to split
        largest_idx = max(range(len(sections)), key=lambda i: len(sections[i].anchor_ids))

        for i, section in enumerate(sections):
            if i == largest_idx and len(section.anchor_ids) >= sections_needed * 2:
                # Split this section
                docs_per_section = len(section.anchor_ids) // sections_needed
                for j in range(sections_needed):
                    start = j * docs_per_section
                    end = start + docs_per_section if j < sections_needed - 1 else len(section.anchor_ids)

                    part_name = f"{section.name} - Part {j + 1}"
                    result.append(SectionContext(name=part_name, anchor_ids=section.anchor_ids[start:end]))
            else:
                result.append(section)

        return result

    def _generate_section_name(self, docs: list[Any], section_idx: int) -> str:
        """Generate a meaningful section name from its documents"""
        if not docs:
            return f"Section {section_idx}"

        # Collect information about the documents
        symbol_names = [d.metadata.get("symbol_name", "") for d in docs if d.metadata.get("symbol_name")]
        symbol_types = [d.metadata.get("symbol_type", "").lower() for d in docs]
        sources = [d.metadata.get("source", "") or d.metadata.get("file_path", "") for d in docs]

        # Strategy 1: Find common file/module theme
        if sources:
            # Extract directory or module names
            dir_names = set()
            file_names = set()
            for source in sources:
                if source:
                    parts = source.replace("\\", "/").split("/")
                    # Get parent directory
                    if len(parts) >= 2:
                        dir_names.add(parts[-2])
                    # Get filename without extension
                    filename = parts[-1].rsplit(".", 1)[0] if parts else ""
                    if filename and not filename.startswith("_"):
                        file_names.add(filename)

            # Use most common directory as theme
            if len(dir_names) == 1:
                dir_name = list(dir_names)[0]
                # Clean up common directory names
                if dir_name not in ["src", "lib", "utils", "helpers", "core"]:
                    return self._format_section_title(dir_name)

            # Use most common filename as theme
            if len(file_names) == 1:
                return self._format_section_title(list(file_names)[0])

        # Strategy 2: Use symbol type grouping
        type_counts = {}
        for st in symbol_types:
            if st:
                type_counts[st] = type_counts.get(st, 0) + 1

        if type_counts:
            dominant_type = max(type_counts.items(), key=lambda x: x[1])[0]
            if dominant_type in ["class", "interface", "struct"]:
                return f"Core {dominant_type.title()}es"
            elif dominant_type == "function":
                return "Utility Functions"
            elif dominant_type in ["markdown_document", "text_document"]:
                return "Documentation"

        # Strategy 3: Find common prefix in symbol names
        if symbol_names and len(symbol_names) > 1:
            # Clean symbol names (remove leading underscores, etc.)
            clean_names = [s.lstrip("_") for s in symbol_names if s]
            if clean_names:
                # Find common prefix
                common = self._find_common_prefix(clean_names)
                if common and len(common) >= 3:
                    return self._format_section_title(common)

        # Strategy 4: Use first meaningful symbol name
        if symbol_names:
            for name in symbol_names:
                # Skip private/internal symbols
                if not name.startswith("_"):
                    return self._format_section_title(name)
            # Fall back to first symbol, cleaned up
            return self._format_section_title(symbol_names[0].lstrip("_"))

        return f"Section {section_idx}"

    def _format_section_title(self, name: str) -> str:
        """Format a name into a readable section title"""
        # Convert snake_case or camelCase to Title Case
        import re

        # Split on underscores and camelCase boundaries
        words = re.split(r"[_\s]|(?<=[a-z])(?=[A-Z])", name)
        words = [w for w in words if w]  # Remove empty strings

        if not words:
            return name

        # Title case each word
        return " ".join(word.capitalize() for word in words)

    def _find_common_prefix(self, strings: list[str]) -> str:
        """Find the common prefix among a list of strings"""
        if not strings:
            return ""

        prefix = strings[0]
        for s in strings[1:]:
            while not s.startswith(prefix) and prefix:
                prefix = prefix[:-1]

        # Don't return partial words - find word boundary
        if prefix and not prefix[-1].isupper():
            # Find last underscore or capital letter
            for i in range(len(prefix) - 1, -1, -1):
                if prefix[i] == "_" or prefix[i].isupper():
                    return prefix[: i + 1].rstrip("_")

        return prefix

    def _build_section_context(self, section: SectionContext) -> None:
        """Build the full code and signatures for a section, including relationship hints"""
        full_code_parts = []
        signature_parts = []
        seen_deps = set()

        logger.info(f"Building context for section '{section.name}':")
        logger.info(f"  - all_docs has {len(self.all_docs)} documents")
        logger.info(f"  - section has {len(section.anchor_ids)} anchor IDs")
        logger.info(f"  - anchor_ids: {section.anchor_ids}")

        # Tier 1: Full code for anchors with relationship hints
        for anchor_id in section.anchor_ids:
            doc = self.all_docs.get(anchor_id)
            if doc:
                source = doc.metadata.get("source", "unknown")
                symbol_name = doc.metadata.get("symbol_name", anchor_id)
                symbol_type = doc.metadata.get("symbol_type", "symbol")

                # Build relationship hints for this symbol
                hints = self._build_relationship_hints(doc, section.anchor_ids)
                hints_block = f"\n**Relationships:**\n{hints}\n" if hints else ""

                full_code_parts.append(
                    f"### {symbol_name} ({symbol_type})\n"
                    f"**File:** `{source}`{hints_block}\n"
                    f"```\n{doc.page_content}\n```"
                )
            else:
                logger.warning(f"  - Anchor '{anchor_id}' NOT FOUND in all_docs!")

        logger.info(f"  - Built {len(full_code_parts)} full_code_parts from anchors")

        # Tier 2: Signatures for ALL expanded dependencies (not just metadata-linked)
        # Include all non-anchor documents as context signatures
        skipped_anchors = 0
        added_sigs = 0

        for doc_id, doc in self.all_docs.items():
            # Skip if this is an anchor for THIS section (already showing full code)
            if doc_id in section.anchor_ids:
                skipped_anchors += 1
                continue
            if doc_id in seen_deps:
                continue

            seen_deps.add(doc_id)
            added_sigs += 1

            sig = self._extract_signature(doc)

            # Add cross-reference if already documented
            symbol_name = doc.metadata.get("symbol_name", doc_id)
            cross_ref = self.doc_state.get_cross_reference(symbol_name)
            if cross_ref:
                sig += f"  {cross_ref}"

            # Add expansion reason if available (why was this included?)
            reason = doc.metadata.get("expansion_reason", "")
            source = doc.metadata.get("expansion_source", "")
            if reason and source:
                sig += f"  # {reason} from {source}"

            signature_parts.append(sig)

        logger.info(f"  - Skipped {skipped_anchors} anchors, added {added_sigs} signatures")

        section.full_code = "\n\n".join(full_code_parts) if full_code_parts else "*No code available*"
        section.signatures = "\n".join(signature_parts) if signature_parts else "*No related symbols*"

        logger.info(f"  - Final: {len(section.full_code)} chars code, {len(section.signatures)} chars signatures")

    def _build_relationship_hints(self, doc, section_anchor_ids: list[str]) -> str:
        """Build human-readable relationship hints for a document"""
        hints = []
        symbol_name = doc.metadata.get("symbol_name", "")

        # Check metadata for relationships
        metadata = doc.metadata

        # Inheritance hints
        if metadata.get("extends"):
            extends = metadata["extends"]
            if isinstance(extends, list):
                for parent in extends:
                    if parent in section_anchor_ids:
                        hints.append(f"- Extends `{parent}` (in this section)")
                    else:
                        hints.append(f"- Extends `{parent}`")
            else:
                if extends in section_anchor_ids:
                    hints.append(f"- Extends `{extends}` (in this section)")
                else:
                    hints.append(f"- Extends `{extends}`")

        if metadata.get("implements"):
            implements = metadata["implements"]
            if isinstance(implements, list):
                for iface in implements:
                    hints.append(f"- Implements `{iface}`")
            else:
                hints.append(f"- Implements `{implements}`")

        # Composition hints (uses/has)
        if metadata.get("field_types"):
            for field_type in metadata["field_types"][:3]:  # Limit to avoid clutter
                if field_type in section_anchor_ids:
                    hints.append(f"- Uses `{field_type}` (in this section)")
                elif field_type not in ["str", "int", "bool", "float", "list", "dict", "None", "Any"]:
                    hints.append(f"- Uses `{field_type}`")

        # Call hints
        if metadata.get("calls"):
            calls = metadata["calls"]
            if isinstance(calls, list):
                for callee in calls[:3]:  # Limit
                    if callee in section_anchor_ids:
                        hints.append(f"- Calls `{callee}` (in this section)")

        # Expansion reason (why was this included?)
        if metadata.get("expansion_reason"):
            reason = metadata["expansion_reason"]
            source = metadata.get("expansion_source", "")
            if source:
                hints.append(f"- Included because: {reason} from `{source}`")

        # Check graph for additional relationships if available
        if self.graph and symbol_name:
            graph_hints = self._get_graph_relationship_hints(symbol_name, section_anchor_ids)
            hints.extend(graph_hints)

        return "\n".join(hints) if hints else ""

    def _get_graph_relationship_hints(self, symbol_name: str, section_anchor_ids: list[str]) -> list[str]:
        """Get relationship hints from the graph"""
        hints = []

        if not self.graph:
            return hints

        # Find the node for this symbol
        node_id = None
        for nid in self.graph.nodes():
            if self.graph.nodes[nid].get("symbol_name") == symbol_name:
                node_id = nid
                break

        if not node_id:
            return hints

        # Check outgoing edges (what this symbol uses/calls/extends)
        for _, target, edge_data in self.graph.out_edges(node_id, data=True):
            rel_type = edge_data.get("relationship_type", "").lower()
            target_data = self.graph.nodes.get(target, {})
            target_name = target_data.get("symbol_name", "")

            if not target_name:
                continue

            in_section = target_name in section_anchor_ids
            suffix = " (in this section)" if in_section else ""

            if rel_type == "inherits":
                hints.append(f"- Inherits from `{target_name}`{suffix}")
            elif rel_type == "implements":
                hints.append(f"- Implements `{target_name}`{suffix}")
            elif rel_type in ["uses", "composes", "has_member"]:
                hints.append(f"- Uses `{target_name}`{suffix}")
            elif rel_type == "calls":
                if in_section:
                    hints.append(f"- Calls `{target_name}` (in this section)")

        # Check incoming edges (what uses/extends this symbol) - limit these
        incoming_count = 0
        for source, _, edge_data in self.graph.in_edges(node_id, data=True):
            if incoming_count >= 2:  # Limit incoming hints
                break

            rel_type = edge_data.get("relationship_type", "").lower()
            source_data = self.graph.nodes.get(source, {})
            source_name = source_data.get("symbol_name", "")

            if not source_name or source_name == symbol_name:
                continue

            in_section = source_name in section_anchor_ids
            if in_section:
                if rel_type == "inherits":
                    hints.append(f"- Extended by `{source_name}` (in this section)")
                elif rel_type in ["uses", "calls"]:
                    hints.append(f"- Used by `{source_name}` (in this section)")
                incoming_count += 1

        return hints[:5]  # Limit total hints

    def _extract_signature(self, doc) -> str:
        """Extract a signature/summary from a document"""
        content = doc.page_content
        symbol_name = doc.metadata.get("symbol_name", "Unknown")
        symbol_type = doc.metadata.get("symbol_type", "symbol")
        source = doc.metadata.get("source", doc.metadata.get("rel_path", ""))

        # Try to extract just the signature (first few lines)
        lines = content.split("\n")

        # For functions/methods, get the def line and maybe docstring
        signature_line = None
        docstring_preview = ""

        for i, line in enumerate(lines[:15]):
            line_stripped = line.strip()

            # Find function/class definition
            if line_stripped.startswith(
                ("def ", "async def ", "class ", "function ", "interface ", "type ", "struct ", "enum ")
            ):
                signature_line = line_stripped
                # Look for docstring on next lines
                for j in range(i + 1, min(i + 5, len(lines))):
                    next_line = lines[j].strip()
                    if next_line.startswith(('"""', "'''", "//", "/*", "*")):
                        # Extract first line of docstring
                        docstring_preview = next_line.strip("\"'/' *").strip()[:80]  # noqa: B005 — intentionally stripping a set of doc-markup chars
                        if docstring_preview:
                            docstring_preview = f" - {docstring_preview}"
                        break
                break

        if signature_line:
            # Include file path for context
            file_hint = f" [{source.split('/')[-1]}]" if source else ""
            return f"{symbol_name}{file_hint}: {signature_line}{docstring_preview}"

        # For text documents, extract first meaningful line
        if symbol_type in ["markdown_document", "text_document", "documentation"]:
            for line in lines[:5]:
                line_stripped = line.strip()
                if line_stripped and not line_stripped.startswith("#"):
                    preview = line_stripped[:100]
                    return f"{symbol_name} (doc): {preview}..."

        # Fallback: symbol name, type, and file
        file_hint = f" [{source.split('/')[-1]}]" if source else ""
        return f"{symbol_name}{file_hint} ({symbol_type})"

    def _build_user_prompt(self, section: SectionContext) -> str:
        """Build the dynamic user prompt for current section"""
        # ToC block
        toc_block = "### Documentation Written So Far\n" + self.doc_state.get_toc_for_prompt()

        # Cross-reference note
        if self.doc_state.symbol_to_section:
            cross_ref_note = "*Symbols marked with `# see [Section]` have already been documented.*"
        else:
            cross_ref_note = ""

        # Remaining sections
        remaining = [s.name for s in self.sections if s.name not in self.doc_state.completed_sections]
        remaining = [r for r in remaining if r != section.name]  # Exclude current

        if remaining:
            remaining_block = "\n".join(f"- {name}" for name in remaining)
        else:
            remaining_block = "*This is the last section.*"

        return AGENTIC_USER_PROMPT.format(
            section_name=section.name,
            current_idx=self.current_section_idx + 1,
            total_sections=len(self.sections),
            toc_block=toc_block,
            full_code=section.full_code,
            cross_ref_note=cross_ref_note,
            signatures=section.signatures,
            remaining_sections_block=remaining_block,
        )

    def _handle_tool_calls(self, response, section: SectionContext) -> str | None:
        """
        Handle tool calls from LLM response.

        Returns:
            - None: Continue with next sequential section
            - str: Name of specific section to jump to
            - "DONE": All sections complete
        """
        next_section_name = None
        committed_this_turn = False

        # Check for tool calls in the response
        tool_calls = getattr(response, "tool_calls", None)
        if not tool_calls:
            # Try alternative attribute names
            if hasattr(response, "additional_kwargs"):
                tool_calls = response.additional_kwargs.get("tool_calls", [])

        if not tool_calls:
            logger.warning(f"🔧 No tool calls in response for section '{section.name}'")
            return None

        logger.info(f"🔧 Processing {len(tool_calls)} tool call(s) for section '{section.name}'")

        for idx, tool_call in enumerate(tool_calls):
            # Handle different tool call formats
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

            # Log the tool call details
            args_preview = (
                {k: (v[:100] + "..." if isinstance(v, str) and len(v) > 100 else v) for k, v in tool_args.items()}
                if tool_args
                else {}
            )
            logger.info(f"  🔧 Tool {idx + 1}: {tool_name}({args_preview})")

            if tool_name == "commit_section":
                content = tool_args.get("content", "")
                summary = tool_args.get("summary", "Section documented")

                if content:
                    self.committed_content.append(content)
                    self.doc_state.add_section(section.name, section.anchor_ids, summary)
                    committed_this_turn = True
                    logger.info(
                        f"    ✓ Committed section '{section.name}' ({len(content)} chars, summary: '{summary[:50]}...')"
                    )
                else:
                    logger.warning(f"    ⚠️ Empty content in commit_section for '{section.name}'")

            elif tool_name == "request_next_section":
                requested_name = tool_args.get("section_name", "")

                if not committed_this_turn:
                    logger.warning(f"    ⚠️ request_next_section called without committing '{section.name}'")

                if requested_name:
                    # Validate requested section
                    valid_sections = [s.name for s in self.sections if s.name not in self.doc_state.completed_sections]
                    if requested_name in valid_sections:
                        next_section_name = requested_name
                        logger.info(f"    → Jumping to requested section: '{requested_name}'")
                    else:
                        logger.warning(
                            f"    ⚠️ Invalid section requested: '{requested_name}'. Available: {valid_sections}"
                        )
                else:
                    logger.info("    → Auto-advancing to next section")

            else:
                logger.warning(f"    ⚠️ Unknown tool call: '{tool_name}'")

        return next_section_name

    def generate(self, config: RunnableConfig) -> str:
        """
        Main generation loop.

        Iterates through sections, invoking LLM with tools,
        handling context swaps between sections.
        """
        if not self.sections:
            logger.error("No sections to generate")
            return ""

        logger.info(
            f"🚀 Starting agentic generation for '{self.page_spec.page_name}' with {len(self.sections)} sections"
        )

        # Log section overview
        for i, section in enumerate(self.sections):
            logger.info(f"   📑 Section {i + 1}: '{section.name}' ({len(section.anchor_ids)} anchors)")

        max_iterations = len(self.sections) * 2  # Safety limit
        iteration = 0

        while self.current_section_idx < len(self.sections) and iteration < max_iterations:
            iteration += 1
            section = self.sections[self.current_section_idx]

            # Skip if already done
            if section.name in self.doc_state.completed_sections:
                logger.debug(f"⏭️ Skipping already completed section: '{section.name}'")
                self.current_section_idx += 1
                continue

            logger.info(f"🔄 Processing section {self.current_section_idx + 1}/{len(self.sections)}: '{section.name}'")
            logger.info(f"   Anchors: {section.anchor_ids}")

            # Build context for this section (THE SWAP)
            self._build_section_context(section)
            logger.info(
                f"   📝 Context built: {len(section.full_code)} chars code, {len(section.signatures)} chars signatures"
            )

            # Build user prompt
            user_prompt = self._build_user_prompt(section)
            logger.debug(f"   📄 User prompt: {len(user_prompt)} chars")

            # Invoke LLM with tools
            messages = [SystemMessage(content=self.system_prompt), HumanMessage(content=user_prompt)]

            logger.info(f"   🤖 Invoking LLM with {len(self.tools)} tools...")

            try:
                # Use bind_tools if available, otherwise pass tools directly
                if hasattr(self.llm, "bind_tools"):
                    llm_with_tools = self.llm.bind_tools(self.tools)
                    response = llm_with_tools.invoke(messages, config=config)
                else:
                    response = self.llm.invoke(messages, tools=self.tools, config=config)

                logger.info("   ✓ LLM response received")

                # Handle tool calls
                next_section = self._handle_tool_calls(response, section)

                # Determine next section
                if next_section:
                    # Jump to specific section
                    for i, s in enumerate(self.sections):
                        if s.name == next_section:
                            self.current_section_idx = i
                            logger.info(f"   ↪️ Jumping to section index {i}")
                            break
                else:
                    # Sequential advance
                    self.current_section_idx += 1
                    logger.info(f"   ➡️ Advancing to section index {self.current_section_idx}")

            except Exception as e:
                logger.error(f"❌ Error processing section '{section.name}': {e}", exc_info=True)
                # Mark as done to avoid infinite loop, but note the error
                self.doc_state.completed_sections.add(section.name)
                self.current_section_idx += 1

        if iteration >= max_iterations:
            logger.warning(f"⚠️ Reached max iterations ({max_iterations}) for agentic generation")

        # Assemble final document
        final_content = self._assemble_final_document()

        logger.info("✅ Agentic generation complete:")
        logger.info(f"   📊 Sections committed: {len(self.committed_content)}")
        logger.info(f"   📊 ToC entries: {len(self.doc_state.toc)}")
        logger.info(f"   📊 Total output: {len(final_content)} chars")

        return final_content

    def _assemble_final_document(self) -> str:
        """Assemble all committed sections into final document"""
        if not self.committed_content:
            return ""

        # Add a brief intro if we have multiple sections
        parts = []

        if len(self.committed_content) > 1:
            # Generate ToC
            toc_lines = ["## Contents\n"]
            for section_name, _symbols, summary in self.doc_state.toc:
                # Create anchor from section name
                anchor = section_name.lower().replace(" ", "-").replace("/", "-")
                toc_lines.append(f"- [{section_name}](#{anchor}): {summary}")
            toc_lines.append("\n---\n")
            parts.append("\n".join(toc_lines))

        # Add all committed sections
        parts.extend(self.committed_content)

        return "\n\n".join(parts)


# =============================================================================
# INTEGRATION HELPER
# =============================================================================


def should_use_agentic_mode(expanded_docs: list[Any], token_budget: int = 50_000) -> bool:
    """
    Determine if agentic mode should be used based on document count and tokens.

    Args:
        expanded_docs: List of expanded documents
        token_budget: Token budget for simple mode

    Returns:
        True if agentic mode should be used
    """
    token_counter = get_token_counter()
    total_tokens = token_counter.count_documents(expanded_docs)

    logger.info(f"Context size check: {total_tokens:,} tokens, budget: {token_budget:,}")

    return total_tokens > token_budget
