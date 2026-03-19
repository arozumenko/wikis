"""
Wiki Structure Planner Tools - Custom tools for incremental structure output

Instead of having the LLM generate one massive JSON blob at the end (slow),
we provide tools that the agent calls to define sections and pages incrementally.
Each tool call is fast, and we assemble the final structure from the calls.

This eliminates the 140+ second JSON generation bottleneck.

Coverage Tracking:
- The collector tracks discovered directories (from ls calls)
- Each define_page response includes coverage feedback
- This guides the LLM to cover all major directories without repository_context

Symbol Validation:
- Validates target_symbols against the code graph
- Case-insensitive matching with automatic correction
- Fuzzy matching suggests similar symbols for hallucinated names
"""

import logging
import os
import re
from typing import Any

from langchain_core.tools import tool
from pydantic import BaseModel, Field

from ..code_graph.graph_query_service import GraphQueryService
from ..constants import CODE_SYMBOL_TYPES, DOC_SYMBOL_TYPES, DOCUMENTATION_EXTENSIONS_SET

logger = logging.getLogger(__name__)

# =============================================================================
# Feature Flags for Doc/Code Separation
# =============================================================================
# When enabled, docs are auto-retrieved using retrieval_query during generation
# LLM doesn't need to manually specify target_docs during structure planning
AUTO_TARGET_DOCS = os.getenv("WIKIS_AUTO_TARGET_DOCS", "0") == "1"

# Log feature flag state at import time
if AUTO_TARGET_DOCS:
    logger.info("[FEATURE FLAG] WIKIS_AUTO_TARGET_DOCS=1: Docs auto-retrieved during generation")


def _get_env_int(name: str, default: int) -> int:
    value = os.getenv(name, "")
    if value == "":
        return default
    try:
        return int(value)
    except ValueError:
        logger.warning("Invalid int for %s=%s, using default=%s", name, value, default)
        return default


def _find_similar_symbols(target: str, symbol_index: dict[str, str], max_results: int = 3) -> list[tuple[str, float]]:
    """
    Find symbols similar to the target using fuzzy matching.

    Uses multiple strategies:
    1. Suffix matching (e.g., 'ApiWrapper' matches 'SomeApiWrapper')
    2. Prefix matching (e.g., 'Gmail' matches 'GmailWrapper')
    3. Substring matching (e.g., 'AWS' matches 'AWSToolConfig')
    4. Word overlap (e.g., 'BrowserApiWrapper' matches 'BrowserToolkit')
    5. Acronym matching (e.g., 'AWS' matches 'AWSS3Wrapper')

    Args:
        target: The symbol name to match
        symbol_index: lowercase -> actual_name mapping
        max_results: Maximum number of suggestions to return

    Returns:
        List of (actual_name, score) tuples, sorted by score descending
    """
    if not target or not symbol_index:
        return []

    target_lower = target.lower()
    # Extract words from camelCase/PascalCase (e.g., 'GraphQLApiWrapper' -> ['graphql', 'api', 'wrapper'])
    target_words = set(re.findall(r"[A-Z]?[a-z]+|[A-Z]+(?=[A-Z]|$)", target.lower()))
    # Extract acronyms more carefully: sequences of uppercase followed by lowercase or end
    # e.g., 'AWSApiWrapper' -> ['AWS'], 'GraphQLApiWrapper' -> ['Graph', 'QL']
    target_acronyms = set(re.findall(r"[A-Z]{2,}(?=[A-Z][a-z]|$|[^a-zA-Z])", target))

    # Common suffix patterns to look for
    common_suffixes = ["wrapper", "apiwrapper", "toolkit", "client", "config", "tool", "toolconfig"]
    target_suffix = None
    for suffix in common_suffixes:
        if target_lower.endswith(suffix):
            target_suffix = suffix
            target_prefix = target_lower[: -len(suffix)]
            break
    else:
        target_prefix = target_lower

    candidates = []

    for lower_name, actual_name in symbol_index.items():
        # Skip exact matches (already handled)
        if lower_name == target_lower:
            continue

        score = 0.0

        # Strategy 1: Prefix match (most important for API wrappers)
        # e.g., 'GmailApiWrapper' should match 'GmailWrapper'
        if target_prefix and lower_name.startswith(target_prefix):
            score += 0.6
        elif len(target_lower) >= 4 and lower_name.startswith(target_lower[:4]):
            score += 0.3

        # Strategy 2: Suffix category match
        # e.g., 'AWSApiWrapper' should consider 'AWSToolConfig' (both are AWS-related configs)
        if target_suffix:
            for suffix in common_suffixes:
                if lower_name.endswith(suffix):
                    score += 0.2
                    break

        # Strategy 3: Word overlap
        # e.g., 'BrowserApiWrapper' -> 'BrowserToolkit' (both have 'browser')
        candidate_words = set(re.findall(r"[A-Z]?[a-z]+|[A-Z]+(?=[A-Z]|$)", lower_name))
        overlap = target_words & candidate_words
        if overlap:
            # Weight by overlap proportion
            score += 0.4 * (len(overlap) / max(len(target_words), 1))

        # Strategy 4: Key substring match
        # e.g., 'GraphQL' in 'GraphQLClientWrapper'
        if len(target_prefix) >= 3 and target_prefix in lower_name:
            score += 0.3

        # Strategy 5: Acronym matching (important for short prefixes like AWS, API, SQL)
        # e.g., 'AWSApiWrapper' should match 'AWSToolConfig', 'AWSS3Wrapper'
        if target_acronyms:
            candidate_acronyms = set(re.findall(r"[A-Z]{2,}(?=[A-Z][a-z]|$|[^a-zA-Z])", actual_name))
            acronym_overlap = target_acronyms & candidate_acronyms
            if acronym_overlap:
                # Strong signal - same acronym means same domain
                score += 0.5 * (len(acronym_overlap) / max(len(target_acronyms), 1))

        # Strategy 6: Short prefix match for acronym-style names (AWS, GCP, etc.)
        # Check if target starts with short acronym that matches
        if len(target_prefix) <= 4 and len(target_prefix) >= 2:
            if lower_name.startswith(target_prefix):
                score += 0.4  # Boost for short acronym prefix match

        # Only include if there's some relevance
        if score > 0.2:
            candidates.append((actual_name, score))

    # Sort by score descending, take top results
    candidates.sort(key=lambda x: x[1], reverse=True)
    return candidates[:max_results]


# Import centralized exclusion constants (single source of truth).
# COVERAGE_IGNORE_DIRS does NOT include .github/.gitlab — those are in
# DOT_DIR_WHITELIST and should be discoverable / coverable.
from ..constants import COVERAGE_IGNORE_DIRS, DOT_DIR_WHITELIST

# Backward-compatible alias — existing code references IGNORED_DIRS.
IGNORED_DIRS = COVERAGE_IGNORE_DIRS


def _extract_node_docstring(node_data: dict) -> str:
    """Extract docstring from a graph node, handling both parser formats.

    Comprehensive parser nodes (Python, Java, JS, TS, C++) store the
    docstring inside ``data['symbol'].docstring``, while basic parser
    nodes (Go, C#, Rust, etc.) store it as a top-level attribute.
    """
    # Basic parser nodes — top-level attribute
    doc = node_data.get("docstring", "") or ""
    if doc:
        return doc
    # Comprehensive parser nodes — inside Symbol object
    symbol = node_data.get("symbol")
    if symbol and hasattr(symbol, "docstring") and symbol.docstring:
        return symbol.docstring
    return ""


def _first_line_summary(docstring: str, max_len: int = 120) -> str:
    """Return first meaningful sentence/line from a docstring.

    Strips leading whitespace, blank lines, decorator noise, and common
    docstring delimiters (triple-quotes, ``//``, ``/**``, ``#``).
    Truncates to *max_len* characters with an ellipsis if needed.
    """
    if not docstring:
        return ""
    # Strip common comment/docstring delimiters
    text = docstring.strip().strip('"').strip("'").strip("/")
    # Remove leading /** or /// or # style markers
    text = re.sub(r"^(?:\s*[/*#]+\s*)+", "", text)
    # Remove trailing */
    text = re.sub(r"\s*\*+/\s*$", "", text)
    # Take first non-empty line
    for line in text.splitlines():
        line = line.strip().lstrip("*").lstrip("#").lstrip("/").strip()
        if not line:
            continue
        # Skip lines that are just parameter docs (@param, :param, Args:, etc.)
        if re.match(r"^[@:]?(param|arg|type|returns?|raises?|throws?|see|note|todo|example|deprecated)\b", line, re.I):
            continue
        if line.endswith(":") and len(line) < 15:
            continue  # Skip section headers like "Args:", "Returns:"
        # Truncate
        if len(line) > max_len:
            return line[: max_len - 3] + "..."
        return line
    return ""


# Tokens that appear in auto-generated or trivial signatures and add no
# semantic value.  Used by _signature_brief to clean up noise.
_TRIVIAL_SIG_TOKENS = frozenset(
    {
        "self",
        "cls",
        "this",
        "public",
        "private",
        "protected",
        "static",
        "final",
        "abstract",
        "virtual",
        "override",
        "inline",
        "extern",
        "async",
        "def",
        "function",
        "fn",
        "func",
        "void",
    }
)


def _signature_brief(node_data: dict, max_len: int = 120) -> str:
    """Build a short description from signature / type information.

    Fallback chain (tries each in order, returns first non-empty result):
    1. ``symbol.signature`` (comprehensive parser) – already concise.
    2. First declaration line of ``source_text`` (universal).
    3. Synthesised from ``parameter_types``/``parameters`` + ``return_type``.

    The result is trimmed to *max_len* characters.
    """
    symbol = node_data.get("symbol")  # comprehensive node

    # --- 1. signature field (comprehensive parsers) -----------------------
    sig = ""
    if symbol and hasattr(symbol, "signature") and symbol.signature:
        sig = symbol.signature.strip()
    if sig:
        # Strip common leading keywords to keep it concise
        sig = re.sub(r"^(export\s+)?(default\s+)?", "", sig).strip()
        if len(sig) > max_len:
            sig = sig[: max_len - 3] + "..."
        return sig

    # --- 2. First line of source_text (declaration line) ------------------
    src = ""
    if symbol and hasattr(symbol, "source_text") and symbol.source_text:
        src = symbol.source_text
    else:
        src = node_data.get("source_text", "") or ""
    if src:
        first_line = src.strip().split("\n", 1)[0].strip()
        # Discard trivial one-word lines or very short fragments
        if first_line and len(first_line) > 5:
            # Remove body-start tokens
            first_line = first_line.rstrip("{").rstrip(":").strip()
            if first_line:
                if len(first_line) > max_len:
                    first_line = first_line[: max_len - 3] + "..."
                return first_line

    # --- 3. Synthesise from params + return_type --------------------------
    parts = []
    # parameter types (comprehensive) or parameter names (basic)
    param_types = []
    if symbol and hasattr(symbol, "parameter_types"):
        param_types = [p for p in (symbol.parameter_types or []) if p.lower() not in _TRIVIAL_SIG_TOKENS]
    if not param_types:
        param_types = [p for p in (node_data.get("parameters") or []) if p.lower() not in _TRIVIAL_SIG_TOKENS]

    ret = ""
    if symbol and hasattr(symbol, "return_type") and symbol.return_type:
        ret = symbol.return_type
    elif node_data.get("return_type"):
        ret = node_data["return_type"]

    if param_types:
        parts.append(f"({', '.join(param_types[:6])})")
    if ret and ret.lower() not in ("none", "void", "undefined"):
        parts.append(f"-> {ret}")

    if parts:
        result = " ".join(parts)
        if len(result) > max_len:
            result = result[: max_len - 3] + "..."
        return result

    return ""


def _get_symbol_brief(node_data: dict, max_len: int = 120) -> str:
    """Return a short human-readable description for a graph node.

    Tries, in order:
    1. First line of the docstring.
    2. Signature / declaration line / synthesised type info.

    Returns empty string when nothing useful can be extracted.
    """
    # Prefer docstring — most semantically meaningful
    brief = _first_line_summary(_extract_node_docstring(node_data), max_len)
    if brief:
        return brief
    # Fallback to signature / declaration
    return _signature_brief(node_data, max_len)


class SectionDefinition(BaseModel):
    """Schema for define_section tool input"""

    section_name: str = Field(..., description="Name of the section (e.g., 'Getting Started', 'API Reference')")
    section_order: int = Field(..., description="Order of the section (1-based)")
    description: str = Field(..., description="Brief description of what this section covers")
    rationale: str = Field(..., description="Why this section is needed in the documentation")


class PageDefinition(BaseModel):
    """Schema for define_page tool input"""

    section_name: str = Field(
        ..., description="Name of the section this page belongs to (must match a defined section)"
    )
    page_name: str = Field(
        ...,
        description=(
            "CAPABILITY-BASED page title describing WHAT users can do, NOT class/function names. "
            "GOOD: 'Order Lifecycle Management', 'Authentication Flow', 'Data Validation Pipeline'. "
            "BAD: 'OrderService & PaymentProcessor', 'AuthManager Class'. "
            "Symbol names belong ONLY in target_symbols field, NOT in the page name."
        ),
    )
    page_order: int = Field(..., description="Order within the section (1-based)")
    description: str = Field(..., description="Brief description of what this page documents")
    content_focus: str = Field(..., description="Specific topics this page should cover")
    rationale: str = Field(..., description="Why this page is needed and what unique value it provides")
    target_symbols: list[str] = Field(
        default_factory=list,
        description="REQUIRED: Exact class/function names from query_graph to document (e.g., ['WorkflowExecutor', 'BaseNode', 'TransformNode']). These are used for precise graph-based code retrieval. Put symbol names HERE, not in page_name.",
    )
    target_docs: list[str] = Field(
        default_factory=list,
        description="REQUIRED when available: Documentation files shown by query_graph's DOCUMENTATION FILES section. Select docs in the same module or matching the page topic. Example: ['src/auth/README.md', 'docs/authentication.md']. Use [] ONLY if no relevant docs exist for this module.",
    )
    target_folders: list[str] = Field(
        default_factory=list, description="List of folders this page covers (e.g., ['src/auth', 'lib/security'])"
    )
    key_files: list[str] = Field(
        default_factory=list, description="Key files to reference (e.g., ['README.md', 'config.py'])"
    )
    retrieval_query: str = Field(
        default="",
        description="Fallback search query for documentation files. target_symbols is preferred for code retrieval.",
    )


class BatchPageItem(BaseModel):
    """Single page definition for batch processing - simplified schema"""

    section_name: str = Field(..., description="Name of the section this page belongs to")
    page_name: str = Field(..., description="CAPABILITY-BASED page title (what users can do)")
    description: str = Field(..., description="Brief description of what this page documents")
    target_symbols: list[str] = Field(default_factory=list, description="Class/function names from query_graph")
    target_folders: list[str] = Field(default_factory=list, description="Folders this page covers")


class BatchPageDefinition(BaseModel):
    """Schema for batch_define_pages tool input - define multiple pages at once"""

    pages: list[BatchPageItem] = Field(
        default_factory=list,
        description=(
            "List of 3-15 page definitions to create. For LARGE repos, use this to define "
            "multiple pages per section efficiently. Each page should cover a distinct capability."
        ),
        max_length=15,
    )


class WikiMetadata(BaseModel):
    """Schema for set_wiki_metadata tool input"""

    wiki_title: str = Field(..., description="Title for the wiki documentation")
    overview: str = Field(..., description="Brief overview of what this documentation covers")


# =============================================================================
# EXPLORATION TOOLS - For deep repository discovery
# =============================================================================

# Max depth for tree exploration (configurable for deep Java monorepos)
# This is the STATIC default — overridden at runtime by detect_effective_depth()
STRUCTURE_MAX_DEPTH = _get_env_int("WIKIS_STRUCTURE_MAX_DEPTH", 5)
MAX_TREE_DEPTH = STRUCTURE_MAX_DEPTH
# Absolute ceiling — never exceed this even with auto-detection
_DEPTH_CEILING = _get_env_int("WIKIS_STRUCTURE_MAX_DEPTH_CEILING", 8)
# MAX_TREE_ENTRIES counts DIRECTORIES only (not files) to prevent early truncation
# Enterprise repos like Spring can have 500+ directories
MAX_TREE_ENTRIES = _get_env_int("WIKIS_STRUCTURE_MAX_ENTRIES", 1000)


def detect_effective_depth(
    repository_files: list[str], static_default: int = STRUCTURE_MAX_DEPTH, ceiling: int = _DEPTH_CEILING
) -> int:
    """Auto-detect the effective tree depth from where code files actually live.

    Problem: Java uses ``src/main/java/org/springframework/boot/`` (depth 6),
    Go monorepos use ``src/go/rpk/pkg/adminapi/`` (depth 5+), Rust workspaces
    ``src/transform-sdk/rust/core-sys/src/`` (depth 5+).  A fixed depth=5
    makes those directories invisible to the structure planner.

    Algorithm:
      1. For every code file (by extension), compute the depth of the
         *directory* that contains it: ``len(path.split('/')) - 1``.
      2. Take the **90th-percentile** depth (avoids outlier deeply-nested
         generated files while still capturing the main code band).
      3. Add +1 so the exploration can see immediate children *below* that
         depth.
      4. Clamp to ``[static_default, ceiling]`` — never go below the static
         default (env-var override still works) and never exceed the ceiling.

    For repos where code lives at depth 1-3 (Python, JS), this returns
    ``static_default`` (5) unchanged — zero regression.

    Args:
        repository_files: List of all file paths in the repo (relative).
        static_default: The env-var / hardcoded default depth (typically 5).
        ceiling: Absolute max depth to prevent explosion (typically 8).

    Returns:
        Effective max depth, between ``static_default`` and ``ceiling``.
    """
    if not repository_files:
        return static_default

    # Collect depths of directories containing code files
    code_depths: list[int] = []
    for path in repository_files:
        if not path:
            continue
        # Fast extension check (inline, avoids function-call overhead)
        dot_idx = path.rfind(".")
        if dot_idx == -1:
            continue
        ext = path[dot_idx:].lower()
        if ext in _CODE_EXT_SET:
            # depth = number of '/' separators = number of directory segments
            depth = path.count("/")
            if depth > 0:
                code_depths.append(depth)

    if not code_depths:
        return static_default

    # 90th-percentile depth
    code_depths.sort()
    p90_idx = int(len(code_depths) * 0.90)
    p90_depth = code_depths[min(p90_idx, len(code_depths) - 1)]

    # Add +1 headroom so explore_tree can see children at that depth
    effective = p90_depth + 1

    # Clamp
    effective = max(static_default, min(effective, ceiling))

    if effective != static_default:
        logger.info(
            "[DEPTH_AUTO] Effective depth adjusted: %d → %d (p90_code_depth=%d, files_sampled=%d, ceiling=%d)",
            static_default,
            effective,
            p90_depth,
            len(code_depths),
            ceiling,
        )

    return effective


# Frozen set of code extensions for fast O(1) lookup in detect_effective_depth()
_CODE_EXT_SET = frozenset(
    {
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
    }
)


def _is_code_file(name: str) -> bool:
    """Check if a file looks like source code (not perfect, just a heuristic)."""
    import os

    _, ext = os.path.splitext(name)
    ext = ext.lower()
    # Common programming language extensions
    # NOTE: .proto and .tf are NOT included here — the graph_builder has no
    # tree-sitter parser for them, so they produce zero code symbols and are
    # indexed as documentation (schema_document / infrastructure_document).
    # Listing them as "code" here would miscount files in the tree display.
    return ext in {
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
        ".hxx",  # C/C++ complete set
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
    }


def _is_doc_file(name: str) -> bool:
    """Check if a file is a documentation file (markdown, rst, txt, etc.)."""
    _, ext = os.path.splitext(name)
    ext = ext.lower()
    return ext in DOCUMENTATION_EXTENSIONS_SET


def _should_ignore_dir(name: str) -> bool:
    """Check if a directory should be ignored during exploration."""
    return (
        name in IGNORED_DIRS
        or name.startswith(".")
        and name not in DOT_DIR_WHITELIST
        or name.startswith("_")
        and name != "__init__.py"
        or name.endswith(".egg-info")
    )


def explore_repository_tree(
    repo_root: str, max_depth: int = MAX_TREE_DEPTH, max_entries: int = MAX_TREE_ENTRIES
) -> dict[str, Any]:
    """
    Explore the repository structure without filtering by file type.
    Shows everything relevant and lets the LLM decide what matters.

    Args:
        repo_root: Root directory of the repository
        max_depth: Maximum depth to explore
        max_entries: Maximum total entries to prevent explosion

    Returns:
        Dict with directories, their sizes, and top-level files
    """
    import os

    result = {
        "directories": [],  # All non-ignored directories with stats
        "top_level_files": [],  # Important top-level files
        "total_files": 0,
    }

    dirs_visited = 0  # Count directories only, not files

    def count_files(path: str) -> tuple:
        """Count total files, code files, doc files, and subdirs in a directory."""
        files = 0
        code_files = 0
        doc_files = 0
        subdirs = []
        try:
            for item in os.listdir(path):
                item_path = os.path.join(path, item)
                if os.path.isfile(item_path):
                    files += 1
                    if _is_code_file(item):
                        code_files += 1
                    elif _is_doc_file(item):
                        doc_files += 1
                elif os.path.isdir(item_path) and not _should_ignore_dir(item):
                    subdirs.append(item)
        except (PermissionError, OSError):
            pass
        return files, code_files, doc_files, subdirs

    def explore(path: str, rel_path: str, depth: int):
        """Recursively explore directories (breadth-first for depth 1)."""
        nonlocal dirs_visited

        # Only limit by directory count, not files (files can be 1000s)
        if depth > max_depth or dirs_visited > max_entries:
            return

        files, code_files, doc_files, subdirs = count_files(path)
        dirs_visited += len(subdirs)  # Count dirs only, not files

        if rel_path:  # Don't add root itself
            result["directories"].append(
                {
                    "path": rel_path,
                    "name": os.path.basename(rel_path),
                    "depth": depth,
                    "files": files,
                    "code_files": code_files,
                    "doc_files": doc_files,
                    "subdirs": len(subdirs),
                }
            )

        # For depth 0 (root), use breadth-first to ensure we see ALL directories
        # at depth 1 and 2 before going deeper
        if depth == 0:
            # First pass: add all top-level directories (depth 1)
            top_level_info = []
            for subdir in sorted(subdirs):
                subdir_path = os.path.join(path, subdir)
                sub_files, sub_code, sub_docs, sub_subdirs = count_files(subdir_path)
                dirs_visited += len(sub_subdirs)  # Count dirs only

                result["directories"].append(
                    {
                        "path": subdir,
                        "name": subdir,
                        "depth": 1,
                        "files": sub_files,
                        "code_files": sub_code,
                        "doc_files": sub_docs,
                        "subdirs": len(sub_subdirs),
                    }
                )
                top_level_info.append((subdir_path, subdir, sub_subdirs))

            # Second pass: add all depth 2 directories BEFORE recursing deeper
            depth2_info = []
            for subdir_path, subdir_rel, sub_subdirs in top_level_info:
                for sub_subdir in sorted(sub_subdirs):
                    sub_subdir_path = os.path.join(subdir_path, sub_subdir)
                    sub_sub_files, sub_sub_code, sub_sub_docs, sub_sub_subdirs = count_files(sub_subdir_path)
                    dirs_visited += len(sub_sub_subdirs)  # Count dirs only

                    sub_subdir_rel = f"{subdir_rel}/{sub_subdir}"
                    result["directories"].append(
                        {
                            "path": sub_subdir_rel,
                            "name": sub_subdir,
                            "depth": 2,
                            "files": sub_sub_files,
                            "code_files": sub_sub_code,
                            "doc_files": sub_sub_docs,
                            "subdirs": len(sub_sub_subdirs),
                        }
                    )
                    depth2_info.append((sub_subdir_path, sub_subdir_rel, sub_sub_subdirs))

            # Third pass: recurse into depth 3+ (with limit)
            for d2_path, d2_rel, d2_subdirs in depth2_info:
                if dirs_visited > max_entries:
                    break
                for sub in sorted(d2_subdirs):
                    sub_path = os.path.join(d2_path, sub)
                    sub_rel = f"{d2_rel}/{sub}"
                    explore(sub_path, sub_rel, 3)
        else:
            # Normal DFS for deeper levels
            for subdir in sorted(subdirs):
                subdir_path = os.path.join(path, subdir)
                subdir_rel = f"{rel_path}/{subdir}" if rel_path else subdir
                explore(subdir_path, subdir_rel, depth + 1)

    # Start exploration
    explore(repo_root, "", 0)

    # Get important top-level files
    try:
        for item in sorted(os.listdir(repo_root)):
            if os.path.isfile(os.path.join(repo_root, item)):
                # Include common important files
                item_lower = item.lower()
                if (
                    item_lower.startswith("readme")
                    or item_lower.startswith("contributing")
                    or item_lower.startswith("changelog")
                    or item_lower
                    in (
                        "license",
                        "makefile",
                        "justfile",
                        "dockerfile",
                        "docker-compose.yml",
                        "docker-compose.yaml",
                        "package.json",
                        "pyproject.toml",
                        "setup.py",
                        "requirements.txt",
                        "cargo.toml",
                        "go.mod",
                        "build.gradle",
                        "pom.xml",
                        "cmakelists.txt",
                    )
                ):
                    result["top_level_files"].append(item)
    except (PermissionError, OSError):
        pass

    result["total_files"] = sum(d["files"] for d in result["directories"])

    return result


def _format_file_breakdown(d: dict[str, Any]) -> str:
    """Format file counts with code/doc breakdown when available."""
    total = d.get("files", 0)
    code = d.get("code_files", 0)
    docs = d.get("doc_files", 0)
    other = total - code - docs
    parts = []
    if code:
        parts.append(f"{code} code")
    if docs:
        parts.append(f"{docs} doc")
    if other > 0:
        parts.append(f"{other} other")
    if parts:
        return f"{total} files ({', '.join(parts)})"
    return f"{total} files"


def format_tree_for_llm(tree_data: dict[str, Any]) -> str:
    """
    Format the tree exploration result for LLM consumption.

    **Content-only approach**: Only directories that directly contain files
    (code or doc) are shown.  Intermediate/shallow directories (0 files,
    only subdirectories) are completely omitted — their content-bearing
    descendants appear instead with full relative paths, so the hierarchy
    is visible from the path itself.  This eliminates noise and ensures
    every directory the LLM sees is a valid ``query_graph`` target.
    """
    lines = []

    lines.append("=== Repository Structure ===")
    lines.append("")

    # Top-level files
    top_files = tree_data.get("top_level_files", [])
    if top_files:
        lines.append(f"Key files: {', '.join(top_files)}")
        lines.append("")

    # Directories grouped by depth
    directories = tree_data.get("directories", [])
    if not directories:
        lines.append("No directories found.")
        return "\n".join(lines)

    # ---- TOP-LEVEL DIRECTORIES (depth 1, content-only) ----
    depth_1 = [d for d in directories if d["depth"] == 1]
    depth_1_content = [d for d in depth_1 if d["files"] > 0]
    depth_1_sorted = sorted(depth_1_content, key=lambda d: d["files"] + d["subdirs"] * 5, reverse=True)

    # Identify large dirs from ALL depth-1 (including shallow containers)
    # so we can show their content descendants in the detail section.
    large_dirs = [d for d in depth_1 if d["files"] >= 10 or d["subdirs"] >= 3]

    lines.append("=== Top-Level Directories ===")
    lines.append("")

    for d in depth_1_sorted:
        size_indicator = ""
        if d["files"] >= 10 or d["subdirs"] >= 3:
            size_indicator = " ⭐ LARGE"
        elif d["files"] >= 5 or d["subdirs"] >= 2:
            size_indicator = " 📦 MEDIUM"

        file_info = _format_file_breakdown(d)
        lines.append(f"📁 {d['name']}/ - {file_info}, {d['subdirs']} subdirs{size_indicator}")

    lines.append("")

    # ---- CONTENT DIRECTORIES INSIDE LARGE TREES ----
    # Show ALL content descendants (files > 0) under each large parent,
    # regardless of depth.  Relative paths encode the hierarchy naturally.
    if large_dirs:
        lines.append("=== Inside Large Directories ===")
        lines.append("")

        large_sorted = sorted(large_dirs, key=lambda d: d["files"] + d["subdirs"] * 5, reverse=True)
        for parent in large_sorted[:8]:
            parent_path = parent["path"]

            # ALL content descendants (files > 0) under this parent
            content_descendants = [d for d in directories if d["path"].startswith(parent_path + "/") and d["files"] > 0]

            if not content_descendants:
                continue

            lines.append(f"📁 {parent['name']}/")

            # Sort by direct file count — leaf dirs with the most code
            # come first so the LLM queries specific areas before broad
            # parents.  The old scoring (files + subdirs*5) put container
            # dirs like 'src/v' at the top, causing 16K-symbol queries.
            content_sorted = sorted(
                content_descendants,
                key=lambda d: d["files"],
                reverse=True,
            )

            # Show ALL content descendants (up to 200).  With the old
            # cap of 25 the LLM was blind to 95 % of dirs in large repos
            # like Redpanda (200+ content dirs under src/).  Each line is
            # ~3 tokens, so 200 lines ≈ 600 tokens — small vs. 128K ctx.
            shown = content_sorted[:200]
            for desc in shown:
                file_info = _format_file_breakdown(desc)
                rel = desc["path"][len(parent_path) + 1 :]  # relative to parent
                if desc["subdirs"] > 0:
                    lines.append(f"  └─ {rel}/ ({file_info}, {desc['subdirs']} subdirs)")
                else:
                    lines.append(f"  └─ {rel}/ ({file_info})")

            rest = len(content_descendants) - len(shown)
            if rest > 0:
                lines.append(f"  └─ ... and {rest} more content directories")

            # Grouping guidance — detect many sibling content dirs sharing
            # the same immediate parent directory.
            from collections import Counter

            parent_counts = Counter(d["path"].rsplit("/", 1)[0] for d in content_descendants)
            for dir_parent, count in parent_counts.most_common(3):
                if count >= 15:
                    sibling_names = sorted(
                        d["path"].rsplit("/", 1)[-1]
                        for d in content_descendants
                        if d["path"].rsplit("/", 1)[0] == dir_parent
                    )
                    lines.append("")
                    lines.append(f"  ⚠️ GROUPING REQUIRED: {dir_parent}/ has {count} content subdirectories!")
                    lines.append("  → GROUP multiple subdirs into themed pages, not one per subdirectory")
                    lines.append(f"  → Query SPECIFIC subdirs (NOT the parent): {', '.join(sibling_names[:6])}")
                    if len(sibling_names) > 6:
                        lines.append(f"  → ... and {len(sibling_names) - 6} more subdirs")

            lines.append("")

    # ---- SUMMARY ----
    total_files = tree_data.get("total_files", 0)
    total_code = sum(d.get("code_files", 0) for d in directories)
    total_docs = sum(d.get("doc_files", 0) for d in directories)
    content_dir_count = sum(1 for d in directories if d["files"] > 0)
    summary = f"TOTAL: {content_dir_count} content directories, {total_files} files"
    if total_code or total_docs:
        summary += f" ({total_code} code, {total_docs} doc, {total_files - total_code - total_docs} other)"
    lines.append(summary)
    lines.append("")

    # ---- NEXT STEPS ----
    # Only suggest LEAF-ISH dirs as query targets.  Broad parents (subdirs ≥ 15)
    # match thousands of symbols via prefix search and produce overwhelming results.
    # E.g. query_graph('src/v') on Redpanda → 16 080 symbols — useless breadth.
    # Their children (subdirs < 15) are far better query targets.
    all_content = [d for d in directories if d["files"] > 0]
    if all_content:
        # Leaf-ish = dirs with fewer than 15 subdirs → focused query results
        leaf_content = [d for d in all_content if d["subdirs"] < 15]
        # Fallback: if too few leaf dirs, relax threshold
        if len(leaf_content) < 6:
            leaf_content = sorted(all_content, key=lambda d: d["subdirs"])[:30]

        top_targets = sorted(
            leaf_content,
            key=lambda d: d["files"],
            reverse=True,
        )[:15]

        lines.append("=== NEXT STEPS (MANDATORY) ===")
        lines.append("")
        lines.append("1. Query SPECIFIC directories (NOT broad parents with 100+ subdirs):")
        lines.append("")
        for d in top_targets:
            lines.append(f"   → query_graph('{d['path']}')  # {d['files']} files")
        lines.append("")
        lines.append("2. Call think() with your PAGE PLAN before defining pages.")

    return "\n".join(lines)


class StructureCollector:
    """
    Collects structure definitions from tool calls and assembles final output.

    Also tracks directory coverage to provide feedback to the LLM about
    which directories have been covered and which are still missing.

    Provides exploration tools for deep repository discovery.
    Optionally uses code_graph (nx.DiGraph) for symbol-aware complexity analysis.

    Usage:
        collector = StructureCollector(repo_root="/path/to/repo", page_budget=20, code_graph=graph)
        tools = collector.get_tools()
        # ... run agent with tools ...
        result = collector.assemble()
    """

    def __init__(
        self,
        page_budget: int = 20,
        repo_root: str | None = None,
        code_graph: Any | None = None,
        graph_text_index: Any | None = None,
        effective_depth: int | None = None,
        repository_file_count: int = 0,
    ):
        self.page_budget = page_budget
        self.repo_root = repo_root
        self.code_graph = code_graph  # nx.DiGraph with symbol nodes (rel_path, symbol_name, symbol_type)
        self.graph_text_index = graph_text_index  # GraphTextIndex (FTS5) for O(log N) lookups
        # Auto-detected depth or static default
        self.effective_depth = effective_depth if effective_depth else STRUCTURE_MAX_DEPTH
        # Repository file count — used for dynamic FTS k scaling
        self._repository_file_count = repository_file_count
        # SPEC-1: unified query service — wraps FTS5 + in-memory dicts + traversal
        self.query_service = GraphQueryService(code_graph, fts_index=graph_text_index) if code_graph else None
        self.metadata: dict | None = None
        self.sections: dict[str, dict] = {}  # section_name -> section_data
        self.pages: list[dict] = []  # flat list of pages
        self.errors: list[str] = []

        # Coverage tracking
        self.discovered_dirs: set[str] = set()  # Directories seen from ls
        self.covered_dirs: set[str] = set()  # Directories covered by pages

        # Symbol-level coverage tracking (Phase 3)
        self._covered_symbols: set[str] = set()  # symbol names assigned to pages
        self._all_architectural_symbols: set[str] = set()  # all architectural symbols from graph
        self._symbol_to_page: dict[str, str] = {}  # symbol → page name (first assignment)
        self._architectural_types: frozenset = frozenset(
            {
                "class",
                "interface",
                "struct",
                "enum",
                "trait",
                "function",
                "constant",
                "type_alias",
            }
        )

        overflow_default = max(10, int(self.page_budget * 0.2)) if self.page_budget else 0
        self.page_budget_overflow = _get_env_int("WIKIS_PAGE_BUDGET_OVERFLOW", overflow_default)
        self.soft_page_budget = self.page_budget + max(self.page_budget_overflow, 0)

        # Compute repo-aware symbol cap (same logic as OptimizedWikiGenerationAgent._split_overloaded_pages)
        from ..agents.wiki_graph_optimized import OptimizedWikiGenerationAgent

        _gn = code_graph.number_of_nodes() if code_graph else 0
        self._max_symbols_per_page = OptimizedWikiGenerationAgent._compute_max_symbols_per_page(
            _gn, repository_file_count
        )

        # Deep exploration cache
        self._tree_analysis: dict | None = None
        self._thinking_log: list[str] = []  # Record of think tool calls
        self._graph_module_cache: dict[str, dict] = {}  # Cache for graph module analysis

    def reset(self):
        """Reset collector for new run."""
        self.metadata = None
        self.sections = {}
        self.pages = []
        self.errors = []
        self.discovered_dirs = set()
        self.covered_dirs = set()
        self._covered_symbols = set()
        self._all_architectural_symbols = set()
        self._symbol_to_page = {}
        self._tree_analysis = None
        self._thinking_log = []
        self._graph_module_cache = {}

    def register_discovered_dirs(self, dirs: list[str]):
        """Register directories discovered from ls calls.

        Called by the engine when it intercepts ls tool results.
        Stores the full cleaned path (without trailing /).

        NOTE: ls output contains relative entries (e.g., 'ado/', 'jira/').
        These leaf-only names are stored as-is. The primary discovery
        mechanism is explore_tree which registers full paths
        (e.g., 'sdk/tools/ado'). Leaf-only entries from ls may
        not match covered_dirs but they are harmless since explore_tree
        already provides the canonical full paths.
        """
        for d in dirs:
            d_clean = d.strip().rstrip("/")
            if not d_clean:
                continue
            leaf = d_clean.rsplit("/", 1)[-1] if "/" in d_clean else d_clean
            if leaf in IGNORED_DIRS or (leaf.startswith(".") and leaf not in DOT_DIR_WHITELIST):
                continue
            self.discovered_dirs.add(d_clean)

    def _get_uncovered_dirs(self) -> list[str]:
        """Get list of discovered directories not yet covered by any page.

        Only returns directories that were actually discovered during exploration.
        This prevents false negatives where we 'cover' paths that don't exist.
        """
        # Only consider dirs that are in discovered_dirs (intersection)
        valid_covered = self.covered_dirs & self.discovered_dirs
        return sorted(self.discovered_dirs - valid_covered)

    def _get_coverage_stats(self) -> tuple:
        """Get accurate coverage statistics.

        Returns:
            (covered_count, discovered_count, percentage)

        Note: Only counts dirs that are BOTH covered AND discovered.
        This prevents coverage > 100% which was confusing in logs.
        """
        valid_covered = self.covered_dirs & self.discovered_dirs
        discovered_count = len(self.discovered_dirs)
        covered_count = len(valid_covered)
        percentage = (covered_count / discovered_count * 100) if discovered_count > 0 else 0
        return covered_count, discovered_count, percentage

    def _update_coverage(self, target_folders: list[str]):
        """Update coverage based on target_folders from a page definition.

        Marks coverage in BOTH directions:

        1. **Ancestors** — for path 'src/services/orders/api/client', marks
           src, src/services, src/services/orders, etc. up to effective_depth.

        2. **Descendants** — all discovered_dirs that start with a target_folder
           prefix are also covered.  A page about 'src/v/cluster' structurally
           covers src/v/cluster/topics, src/v/cluster/partitions, etc.
           because the content expander will pull symbols from the entire subtree.

        This bidirectional marking matches the reconciliation logic in
        _apply_deepagents_coverage_check (wiki_graph_optimized.py). Previously
        only ancestors were marked, causing coverage to stay at 12-15% on big
        repos even when the LLM had created pages for major subsystems.

        Note: Coverage percentage only counts dirs that are in discovered_dirs.
        """
        # Use auto-detected depth (not the static global constant) so that
        # directories at depth 6-7 discovered by explore_tree can actually
        # be marked covered.  Previously this used STRUCTURE_MAX_DEPTH=5
        # which made depth 6+ dirs permanently uncoverable.
        max_cover_depth = getattr(self, "effective_depth", STRUCTURE_MAX_DEPTH)

        for folder in target_folders:
            folder_norm = folder.strip("/")
            parts = folder_norm.split("/")
            if not parts or not parts[0]:
                continue

            # (1) Mark all ANCESTOR directory levels up to effective depth
            for depth in range(1, min(len(parts) + 1, max_cover_depth + 1)):
                dir_path = "/".join(parts[:depth])
                if parts[depth - 1] not in IGNORED_DIRS:
                    self.covered_dirs.add(dir_path)

            # (2) Mark all DESCENDANT discovered dirs under this folder.
            # A page about 'src/v/cluster' covers its entire subtree.
            if folder_norm:
                prefix = folder_norm + "/"
                for disc_dir in self.discovered_dirs:
                    if disc_dir.startswith(prefix):
                        self.covered_dirs.add(disc_dir)

    def _format_coverage_feedback(self) -> str:
        """Format coverage feedback for tool response.

        Always suggests SPECIFIC uncovered directories, never broad parent
        paths.  Broad queries like ``query_graph('tools')`` return thousands
        of symbols, waste context, and lead to overlapping / duplicate pages.
        """
        uncovered = self._get_uncovered_dirs()
        if not uncovered:
            if self.discovered_dirs:
                return "✅ All discovered directories are covered!"
            return ""

        # Sort: shallowest first (highest impact), then alphabetical
        sorted_uncovered = sorted(uncovered, key=lambda d: (d.count("/"), d))

        # Build feedback — always show individual leaf dirs
        feedback_parts = []
        shown = sorted_uncovered[:10]
        remaining = len(sorted_uncovered) - len(shown)
        dirs_display = ", ".join(shown)
        feedback_parts.append(
            f"⚠️ STILL UNCOVERED ({len(uncovered)} dirs): {dirs_display}"
            + (f" (+{remaining} more)" if remaining > 0 else "")
        )
        feedback_parts.append("→ query_graph each SPECIFIC path above, then add pages.")

        total_msg = f"\n📊 Total uncovered: {len(uncovered)} directories → ADD MORE PAGES!"
        return "\n".join(feedback_parts) + total_msg

    def _ensure_architectural_symbols_loaded(self):
        """Lazily load all architectural symbols from the code graph.

        This builds the set once per session. Architectural symbols are classes,
        interfaces, structs, enums, traits, functions, constants, and type aliases.
        """
        if self._all_architectural_symbols or not self.code_graph:
            return

        for _node_id, node_data in self.code_graph.nodes(data=True):
            sym_type = node_data.get("symbol_type", "") or node_data.get("type", "")
            if sym_type in self._architectural_types:
                sym_name = node_data.get("symbol_name", "") or node_data.get("name", "")
                if sym_name:
                    self._all_architectural_symbols.add(sym_name)

    def _compute_symbol_coverage(self):
        """Compute symbol coverage statistics.

        Returns:
            Tuple of (covered_count, total_count, uncovered_symbol_names)
        """
        self._ensure_architectural_symbols_loaded()
        if not self._all_architectural_symbols:
            return (0, 0, [])

        total = len(self._all_architectural_symbols)
        covered = len(self._covered_symbols & self._all_architectural_symbols)
        uncovered = sorted(self._all_architectural_symbols - self._covered_symbols)
        return (covered, total, uncovered)

    def _format_symbol_coverage_feedback(self) -> str:
        """Format symbol coverage feedback for tool response.

        Shows architectural symbol coverage alongside directory coverage.
        """
        covered, total, uncovered = self._compute_symbol_coverage()
        if total == 0:
            return ""

        pct = (covered / total * 100) if total > 0 else 0

        # Always show the metric
        feedback = f"📊 Symbol coverage: {covered}/{total} ({pct:.0f}%) architectural symbols assigned to pages"

        if uncovered:
            # Show a sample of uncovered symbols (most important first)
            shown = uncovered[:6]
            remaining = len(uncovered) - len(shown)
            feedback += f"\n   Unassigned symbols: {', '.join(shown)}"
            if remaining > 0:
                feedback += f" (+{remaining} more)"

        return feedback

    def _assignment_tag(self, sym_name: str) -> str:
        """Return assignment annotation for query_graph display.

        When a symbol has already been assigned to a page, returns a short
        tag like ``  ← "Agent Graph Construction"`` so the LLM can see
        which symbols are taken and choose accordingly.
        """
        page = getattr(self, "_symbol_to_page", {}).get(sym_name)
        if page:
            return f'  ← "{page}"'
        return ""

    def _handle_set_metadata(self, wiki_title: str, overview: str) -> str:
        """Handle set_wiki_metadata tool call."""
        logger.info(f"[STRUCTURE_PLANNER][TOOL] set_wiki_metadata: {wiki_title}")
        self.metadata = {"wiki_title": wiki_title, "overview": overview}
        return "OK: Wiki metadata set"

    def _handle_define_section(self, section_name: str, section_order: int, description: str, rationale: str) -> str:
        """Handle define_section tool call."""
        logger.info(f"[STRUCTURE_PLANNER][TOOL] define_section: {section_name} (order={section_order})")

        # Check for generic names that mirror directories
        warnings = []
        generic_patterns = [
            "core",
            "service",
            "services",
            "clients",
            "utils",
            "helpers",
            "components",
            "modules",
            "framework",
            "external",
        ]
        name_lower = section_name.lower()
        for pattern in generic_patterns:
            if pattern in name_lower and len(section_name.split()) <= 2:
                warnings.append(
                    f"⚠️ '{section_name}' may be too generic. "
                    "Consider naming after CAPABILITIES (e.g., 'Database Integrations', 'AI Chat Services')."
                )
                break

        if section_name in self.sections:
            return f"WARN: Section '{section_name}' already defined, updating"

        self.sections[section_name] = {
            "section_name": section_name,
            "section_order": section_order,
            "description": description,
            "rationale": rationale,
            "pages": [],  # Will be filled during assembly
        }

        result = f"OK: Section '{section_name}' defined"
        if warnings:
            result += " " + " ".join(warnings)
        return result

    def _handle_define_page(
        self,
        section_name: str,
        page_name: str,
        page_order: int,
        description: str,
        content_focus: str,
        rationale: str,
        target_symbols: list[str],
        target_docs: list[str],
        target_folders: list[str],
        key_files: list[str],
        retrieval_query: str = "",
    ) -> str:
        """Handle define_page tool call.

        Uses soft budget: warns when over budget but doesn't block.
        Provides coverage feedback to guide the LLM.
        Validates target_symbols for better page generation.
        """
        page_num = len(self.pages) + 1
        logger.info(
            f"[STRUCTURE_PLANNER][TOOL] define_page: {page_name} in {section_name} ({page_num}/{self.page_budget})"
        )

        # Check if section exists
        if section_name not in self.sections:
            self.errors.append(f"Page '{page_name}' references undefined section '{section_name}'")

        # Warn if no target_symbols provided (should have class/function names from query_graph)
        warnings = []
        if not target_symbols:
            warnings.append(
                "⚠️ No target_symbols provided. Use class/function names from query_graph for precise code retrieval."
            )
        elif len(target_symbols) < 5:
            warnings.append(
                f"⚠️ Only {len(target_symbols)} target_symbols — pages should have 10-15+ symbols. Merge related classes, functions, constants, and helpers from the same capability into this page."
            )
        elif len(target_symbols) > self._max_symbols_per_page + 15:
            warnings.append(
                f"⚠️ Very high target_symbols count ({len(target_symbols)}). Consider splitting into 2-3 capability-focused pages with 15-{self._max_symbols_per_page} symbols each."
            )

        # CRITICAL: Validate target_docs for topic relevance
        # This prevents "Related Documentation" section from showing irrelevant docs
        # Skip validation when AUTO_TARGET_DOCS is enabled (docs auto-retrieved during generation)
        if AUTO_TARGET_DOCS:
            # New mode: docs auto-retrieved, target_docs is optional
            if target_docs:
                logger.info(
                    f"[STRUCTURE_PLANNER] AUTO_TARGET_DOCS mode: {len(target_docs)} explicit target_docs provided"
                )
            else:
                logger.debug(
                    "[STRUCTURE_PLANNER] AUTO_TARGET_DOCS mode: target_docs empty, will auto-retrieve during generation"
                )
        elif not target_docs:
            # Check if we have cached docs from query_graph for relevant folders
            docs_available = False
            for folder in target_folders or []:
                folder_norm = folder.strip("/")
                for cached_path, cached_analysis in self._graph_module_cache.items():
                    if folder_norm.startswith(cached_path) or cached_path.startswith(folder_norm):
                        if cached_analysis.get("all_docs"):
                            docs_available = True
                            break
                if docs_available:
                    break

            if docs_available:
                warnings.append(
                    "⚠️ DOCUMENTATION FILES exist for this module but target_docs is EMPTY! "
                    "Check query_graph output and add TOPIC-RELEVANT docs to target_docs. "
                    "Only include docs that directly relate to this page's topic."
                )
            else:
                warnings.append(
                    "💡 No target_docs specified. If query_graph showed DOCUMENTATION FILES for this module, add relevant ones."
                )
        else:
            # Validate target_docs are topic-relevant (basic heuristic)
            page_keywords = set(page_name.lower().replace("-", " ").replace("_", " ").split())
            irrelevant_docs = []
            for doc_path in target_docs:
                doc_name = doc_path.split("/")[-1].lower()
                # Check if doc name has any overlap with page topic
                doc_keywords = set(
                    doc_name.replace(".md", "").replace(".rst", "").replace("-", " ").replace("_", " ").split()
                )
                if not page_keywords & doc_keywords and "readme" not in doc_name:
                    # No keyword overlap and not a README - might be irrelevant
                    irrelevant_docs.append(doc_path)

            if irrelevant_docs and len(irrelevant_docs) > len(target_docs) // 2:
                warnings.append(
                    f"⚠️ Some target_docs may not be relevant to '{page_name}': {irrelevant_docs[:2]}. "
                    "Only include docs that DIRECTLY relate to this page's topic."
                )

        # Validate target_symbols exist in the graph (prevent hallucinations)
        validated_symbols = []
        unvalidated_symbols = []
        symbol_suggestions = {}  # Map unvalidated symbol -> list of suggestions

        if target_symbols and self.code_graph:
            name_index = getattr(self.code_graph, "_name_index", None)

            # Build case-insensitive index if needed (lazy, once per session)
            # Only include architectural symbol types — methods are excluded
            # to prevent them from leaking into page target_symbols assignments.
            if not hasattr(self, "_case_insensitive_symbols"):
                self._case_insensitive_symbols = {}  # lowercase -> actual_name
                self._symbol_rel_paths = {}  # lowercase -> rel_path (for folder derivation)
                for _node_id, node_data in self.code_graph.nodes(data=True):
                    sym_type = (node_data.get("symbol_type") or node_data.get("type") or "").lower()
                    # Skip non-architectural symbols (methods, constructors, etc.)
                    if sym_type and sym_type not in CODE_SYMBOL_TYPES:
                        continue
                    sym_name = node_data.get("symbol_name", "") or node_data.get("name", "")
                    if sym_name:
                        key = sym_name.lower()
                        self._case_insensitive_symbols[key] = sym_name
                        rel_path = node_data.get("rel_path", "") or node_data.get("file_path", "")
                        if rel_path and key not in self._symbol_rel_paths:
                            self._symbol_rel_paths[key] = rel_path

            for sym in target_symbols:
                found = False
                corrected_sym = sym

                # Check index first (O(1)) — only accept architectural symbols
                if name_index and sym in name_index:
                    # Verify at least one node for this name is an architectural type
                    arch_nodes = [
                        nid
                        for nid in name_index[sym]
                        if (self.code_graph.nodes.get(nid, {}).get("symbol_type") or "").lower() in CODE_SYMBOL_TYPES
                    ]
                    if arch_nodes:
                        found = True
                    else:
                        # Symbol exists but is a method/constructor — not page-assignable
                        logger.debug(
                            f"[STRUCTURE_PLANNER] Symbol '{sym}' exists in graph "
                            f"but is a method/constructor — skipping for page assignment"
                        )
                # Check case-insensitive match (already filtered to arch types)
                elif sym.lower() in self._case_insensitive_symbols:
                    found = True
                    actual_name = self._case_insensitive_symbols[sym.lower()]
                    if actual_name != sym:
                        # Log the correction for debugging
                        logger.debug(f"[STRUCTURE_PLANNER] Symbol case correction: '{sym}' -> '{actual_name}'")
                        corrected_sym = actual_name
                else:
                    # Fall back to scanning with case-insensitive match (for basic mode nodes)
                    # Only match architectural symbols — methods excluded from page assignments
                    for node_id, node_data in list(self.code_graph.nodes(data=True))[:2000]:
                        sym_type = (node_data.get("symbol_type") or node_data.get("type") or "").lower()
                        if sym_type and sym_type not in CODE_SYMBOL_TYPES:
                            continue
                        sym_name = node_data.get("symbol_name", "") or node_data.get("name", "")
                        # Also check node_id suffix
                        node_suffix = str(node_id).split("::")[-1] if "::" in str(node_id) else ""
                        # Case-insensitive exact or qualified-suffix match only
                        # (no substring — "name" must NOT match "symbol_name")
                        if (
                            sym_name.lower() == sym.lower()
                            or node_suffix.lower() == sym.lower()
                            or sym_name.lower().endswith(f".{sym.lower()}")
                        ):
                            found = True
                            corrected_sym = sym_name if sym_name else sym
                            break

                if found:
                    validated_symbols.append(corrected_sym)
                else:
                    unvalidated_symbols.append(sym)
                    # Find similar symbols to suggest
                    similar = _find_similar_symbols(sym, self._case_insensitive_symbols, max_results=3)
                    if similar:
                        symbol_suggestions[sym] = [name for name, score in similar]

            if unvalidated_symbols:
                # Build detailed warning with suggestions
                warning_parts = []
                for sym in unvalidated_symbols:
                    if sym in symbol_suggestions:
                        suggestions = symbol_suggestions[sym]
                        warning_parts.append(f"'{sym}' -> did you mean: {suggestions}?")
                    else:
                        warning_parts.append(f"'{sym}' (no similar symbols found)")

                warnings.append(f"⚠️ Symbols not found in graph: {'; '.join(warning_parts)}")
                logger.warning(
                    f"[STRUCTURE_PLANNER] target_symbols not in graph: {unvalidated_symbols}, suggestions: {symbol_suggestions}"
                )

        # ── Duplicate symbol awareness ──
        # The primary dedup mechanism is query_graph annotations: each symbol
        # already assigned shows "← Page X" so the LLM avoids picking it.
        # Here we only WARN for visibility — no stripping.  Assembly-time
        # dedup (first-assignment wins) acts as a safety net.
        if validated_symbols:
            overlap_symbols: list[str] = []
            for sym in validated_symbols:
                existing_page = self._symbol_to_page.get(sym)
                if existing_page and existing_page != page_name:
                    overlap_symbols.append(f"'{sym}' (also on '{existing_page}')")
            if overlap_symbols:
                warnings.append(
                    f"ℹ️ {len(overlap_symbols)} symbol(s) already assigned to another page: "
                    f"{'; '.join(overlap_symbols)}. "
                    f"This may cause content overlap — consider removing them "
                    f"or merging pages."
                )
                logger.info(f"[STRUCTURE_PLANNER] Overlapping symbols on '{page_name}': {overlap_symbols}")

        # CRITICAL: Derive coverage from validated symbols' paths (not just target_folders)
        # This ensures coverage accurately reflects what symbols are actually being documented
        symbol_derived_folders = []
        if validated_symbols and self.code_graph:
            for sym in validated_symbols:
                # Register the symbol as covered (Phase 3 symbol tracking)
                self._covered_symbols.add(sym)
                if sym not in self._symbol_to_page:
                    self._symbol_to_page[sym] = page_name

                # Find the symbol's rel_path in the graph
                for _node_id, node_data in self.code_graph.nodes(data=True):
                    sym_name = node_data.get("symbol_name", "") or node_data.get("name", "")
                    if sym_name == sym or sym_name.lower() == sym.lower():
                        rel_path = node_data.get("rel_path", "") or node_data.get("file_path", "")
                        if rel_path:
                            # Extract directory path from file path
                            dir_path = "/".join(rel_path.split("/")[:-1])  # Remove filename
                            if dir_path:
                                symbol_derived_folders.append(dir_path)
                        break

        # Merge symbol-derived folders with LLM-provided target_folders
        # This ensures wiki_graph_optimized.py sees the full coverage in the returned structure
        all_folders = list(set((target_folders or []) + symbol_derived_folders))

        page_data = {
            "section_name": section_name,
            "page_name": page_name,
            "page_order": page_order,
            "description": description,
            "content_focus": content_focus,
            "rationale": rationale,
            "target_symbols": validated_symbols
            if validated_symbols
            else [],  # Only validated symbols — never fall back to unvalidated LLM input
            "target_docs": target_docs or [],  # Documentation files for page context
            "target_folders": all_folders,  # Include both LLM-provided and symbol-derived paths
            "key_files": key_files or [],
            "retrieval_query": retrieval_query,
        }
        self.pages.append(page_data)

        # Update internal coverage tracking
        self._update_coverage(all_folders)

        if symbol_derived_folders:
            logger.info(f"[STRUCTURE_PLANNER][COVERAGE] Derived from symbols: {symbol_derived_folders}")

        # Build response with coverage feedback
        status_parts = [f"OK: Page '{page_name}' added ({page_num}/{self.page_budget})"]

        # Include target_symbols warnings
        if warnings:
            status_parts.extend(warnings)

        # Soft budget warning (not blocking)
        if page_num > self.page_budget:
            status_parts.append(f"(over budget by {page_num - self.page_budget})")

        # Coverage feedback
        coverage_feedback = self._format_coverage_feedback()
        if coverage_feedback:
            status_parts.append(coverage_feedback)

        # Symbol coverage feedback (Phase 3)
        symbol_feedback = self._format_symbol_coverage_feedback()
        if symbol_feedback:
            status_parts.append(symbol_feedback)

        return ". ".join(status_parts)

    def _handle_batch_define_pages(self, pages: list[Any]) -> str:
        """Handle batch_define_pages tool call - define multiple pages efficiently.

        This is optimized for large repositories where defining pages one-by-one
        through LLM thinking is too slow. Each page gets minimal validation.

        Args:
            pages: List of BatchPageItem Pydantic models (from LangChain schema validation)
                   Each has: section_name, page_name, description, target_symbols, target_folders

        Returns:
            Summary of pages created with any errors
        """
        if not pages:
            logger.warning("[STRUCTURE_PLANNER][TOOL] batch_define_pages: Empty pages list")
            return (
                "ERROR: batch_define_pages received empty 'pages' list. "
                "Provide 3-15 pages with section_name/page_name/description/target_symbols/target_folders."
            )

        logger.info(f"[STRUCTURE_PLANNER][TOOL] batch_define_pages: Processing {len(pages)} pages")

        results = []
        success_count = 0
        error_count = 0

        for i, page_def in enumerate(pages):
            # Handle both Pydantic models and dicts (for flexibility)
            if hasattr(page_def, "section_name"):
                # Pydantic model - access attributes directly
                section_name = page_def.section_name
                page_name = page_def.page_name
                description = page_def.description
                target_symbols = list(page_def.target_symbols) if page_def.target_symbols else []
                target_folders = list(page_def.target_folders) if page_def.target_folders else []
            else:
                # Dict fallback
                section_name = page_def.get("section_name", "")
                page_name = page_def.get("page_name", "")
                description = page_def.get("description", "")
                target_symbols = page_def.get("target_symbols", [])
                target_folders = page_def.get("target_folders", [])

            # Skip if missing required fields
            if not section_name or not page_name:
                results.append(f"❌ Page {i + 1}: Missing section_name or page_name")
                error_count += 1
                continue

            # Auto-create section if not exists (for efficiency in batch mode)
            if section_name not in self.sections:
                self.sections[section_name] = {
                    "section_name": section_name,
                    "section_order": len(self.sections) + 1,
                    "description": f"Section for {section_name}",
                    "rationale": "Auto-created in batch mode",
                    "pages": [],
                }

            # Get next page order in section
            existing_pages = [p for p in self.pages if p.get("section_name") == section_name]
            page_order = len(existing_pages) + 1

            # Derive coverage from symbols if we have a graph
            symbol_derived_folders = []
            validated_symbols = []
            unvalidated_symbols = []
            batch_duplicates: list[str] = []

            if target_symbols and self.code_graph:
                # Quick symbol validation + O(1) folder lookup via pre-built index
                if hasattr(self, "_case_insensitive_symbols"):
                    batch_duplicates: list[str] = []
                    for sym in target_symbols:
                        key = sym.lower()
                        if key in self._case_insensitive_symbols:
                            actual_sym = self._case_insensitive_symbols[key]
                            # Duplicate awareness — warn but don't strip.
                            # Primary dedup is query_graph annotations (← "Page X").
                            # Assembly-time dedup is the safety net.
                            existing_page = self._symbol_to_page.get(actual_sym)
                            if existing_page and existing_page != page_name:
                                batch_duplicates.append(f"'{actual_sym}' (also on '{existing_page}')")
                            validated_symbols.append(actual_sym)
                            # Register as covered (Phase 3 symbol tracking)
                            self._covered_symbols.add(actual_sym)
                            if actual_sym not in self._symbol_to_page:
                                self._symbol_to_page[actual_sym] = page_name
                            # O(1) folder derivation from pre-built path index
                            rel_path = getattr(self, "_symbol_rel_paths", {}).get(key, "")
                            if rel_path:
                                dir_path = "/".join(rel_path.split("/")[:-1])
                                if dir_path:
                                    symbol_derived_folders.append(dir_path)
                        else:
                            unvalidated_symbols.append(sym)
                    if batch_duplicates:
                        logger.warning(
                            f"[STRUCTURE_PLANNER] Stripped {len(batch_duplicates)} duplicate symbols "
                            f"from batch page '{page_name}': {batch_duplicates}"
                        )
                else:
                    validated_symbols = target_symbols  # Use as-is if no index
                    batch_duplicates = []

            # Merge folders
            all_folders = list(set((target_folders or []) + symbol_derived_folders))

            page_num = len(self.pages) + 1
            page_data = {
                "section_name": section_name,
                "page_name": page_name,
                "page_order": page_order,
                "description": description,
                "content_focus": description,  # Use description as content_focus
                "rationale": "Batch-defined page",
                "target_symbols": validated_symbols
                if validated_symbols
                else [],  # Only validated symbols — never fall back to unvalidated LLM input
                "target_docs": [],
                "target_folders": all_folders,
                "key_files": [],
                "retrieval_query": "",
            }
            self.pages.append(page_data)

            # Update coverage
            self._update_coverage(all_folders)

            success_count += 1
            page_result = f"✓ {page_name} ({page_num}/{self.page_budget})"
            # Add per-page symbol validation feedback (Phase 3)
            if unvalidated_symbols:
                page_result += f" ⚠️ not in graph: {', '.join(unvalidated_symbols[:3])}"
                if len(unvalidated_symbols) > 3:
                    page_result += f" (+{len(unvalidated_symbols) - 3} more)"
            # Duplicate symbol feedback — informational, not stripped
            if batch_duplicates:
                page_result += f" ℹ️ {len(batch_duplicates)} overlap(s): {'; '.join(batch_duplicates[:3])}"
                if len(batch_duplicates) > 3:
                    page_result += f" (+{len(batch_duplicates) - 3} more)"
            results.append(page_result)

        # Log summary
        logger.info(
            f"[STRUCTURE_PLANNER][TOOL] batch_define_pages: Created {success_count} pages, {error_count} errors"
        )

        # Calculate remaining budget (soft cap allows coverage overflow)
        total_pages = len(self.pages)
        soft_budget = self.soft_page_budget if self.page_budget else self.page_budget_overflow
        remaining = max(0, soft_budget - total_pages) if soft_budget else 0
        covered_count, discovered_count, coverage_pct = self._get_coverage_stats()

        # Get uncovered directories for suggestions
        uncovered = self._get_uncovered_dirs()

        # Build actionable uncovered-dir suggestions.
        # ALWAYS suggest the SPECIFIC uncovered leaf directories, never broad
        # parent paths.  A broad query like `query_graph('tools')` returns
        # thousands of symbols, wastes context, and leads to overlapping pages.
        # Group siblings visually for readability, but each `query_graph()`
        # suggestion MUST point to an individual uncovered directory.
        uncovered_top_dirs: list = []  # list of (display_path, count) tuples — count=1 always
        parent_groups: dict = {}  # parent → [child paths]
        for d in uncovered:
            parts = d.split("/")
            parent = "/".join(parts[:-1]) if len(parts) >= 2 else ""
            parent_groups.setdefault(parent, []).append(d)

        # Flatten ALL uncovered dirs into individual suggestions.
        # Sort each group alphabetically, then globally by depth (shallowest first).
        all_uncovered_leaf_dirs: list[str] = []
        for _parent, children in sorted(parent_groups.items()):
            all_uncovered_leaf_dirs.extend(sorted(children))
        # Prioritise shallower dirs (more impactful) then alphabetical
        all_uncovered_leaf_dirs.sort(key=lambda d: (d.count("/"), d))
        for d in all_uncovered_leaf_dirs:
            uncovered_top_dirs.append((d, 1))

        # ---------------------------------------------------------------
        # BUILD RESPONSE — put CONTINUE signal FIRST so LLM reads it
        # before page details. LLMs attend most to the beginning of
        # tool responses; burying the signal at the end caused the agent
        # to stop at 53 pages / 15% coverage.
        # ---------------------------------------------------------------
        summary_parts = []
        needs_continue = False

        # Two-tier CONTINUE logic:
        #   Tier 1 (< 90%): Full-strength signal with uncovered suggestions.
        #           Previously this was < 60%, but bidirectional coverage marking
        #           pushes coverage to 80-85% quickly (a page about src/v/cluster
        #           now covers all children).  LLM stopped at 84% because the
        #           threshold was too low.  Raised to 90% to push for full coverage.
        #   Tier 2 (≥ 90% but remaining > 10): Mild nudge.
        if coverage_pct < 90 and (remaining > 0 or self.page_budget_overflow > 0):
            needs_continue = True
            budget_info = (
                f"{remaining} pages remaining" if remaining > 0 else f"overflow budget: +{self.page_budget_overflow}"
            )
            summary_parts.append("=" * 60)
            summary_parts.append('🚨🚨🚨 STOP — DO NOT SAY "DONE" 🚨🚨🚨')
            summary_parts.append("")
            summary_parts.append(
                f"COVERAGE: {coverage_pct:.0f}% ({covered_count}/{discovered_count} dirs) — TARGET: 90%"
            )
            summary_parts.append(f"PAGES: {total_pages}/{self.page_budget} — {budget_info}")
            summary_parts.append("")
            if uncovered_top_dirs:
                top_suggestions = uncovered_top_dirs[:12]
                summary_parts.append("UNCOVERED AREAS — query_graph these SPECIFIC paths, then batch_define_pages:")
                for path, _count in top_suggestions:
                    summary_parts.append(f"   → query_graph('{path}')")
                if len(uncovered_top_dirs) > 12:
                    summary_parts.append(f"   ... and {len(uncovered_top_dirs) - 12} more areas")
                summary_parts.append("")
            summary_parts.append(
                "ACTION: Call query_graph() for uncovered areas, then batch_define_pages with 10-15 more pages."
            )
            summary_parts.append("=" * 60)
            summary_parts.append("")
        elif remaining > 10:
            needs_continue = True
            summary_parts.append(f"🚨 CONTINUE REQUIRED: {remaining} pages remaining, coverage {coverage_pct:.0f}%")
            if uncovered_top_dirs:
                top_suggestions = uncovered_top_dirs[:6]
                summary_parts.append(f"Uncovered: {', '.join(p for p, _ in top_suggestions)}")
            summary_parts.append("→ Call batch_define_pages again with 10-15 more pages")
            summary_parts.append("")

        # Page creation summary
        summary_label = f"{total_pages}/{self.page_budget} total"
        if self.page_budget_overflow > 0:
            summary_label = f"{total_pages}/{self.page_budget} base (soft cap {soft_budget})"
        summary_parts.append(f"✅ Batch created {success_count} pages ({summary_label})")

        # Coverage feedback (specific uncovered dirs) — only if not already shown above
        if not needs_continue:
            coverage_feedback = self._format_coverage_feedback()
            if coverage_feedback:
                summary_parts.append(coverage_feedback)

        # Symbol coverage feedback (Phase 3)
        symbol_feedback = self._format_symbol_coverage_feedback()
        if symbol_feedback:
            summary_parts.append(symbol_feedback)

        # ── Existing structure snapshot ──
        # After middleware eviction, the LLM loses all memory of what sections
        # and pages exist.  This compact tree (section → page names) restores
        # that context at ~100-200 tokens.  Without it the LLM creates duplicate
        # pages with similar names and misassigns pages to wrong sections.
        if self.sections and self.pages:
            summary_parts.append("")
            summary_parts.append("EXISTING STRUCTURE (do NOT duplicate these pages):")
            # Group pages by section, show page names only (no symbols — too many tokens)
            pages_by_section: dict[str, list[str]] = {}
            for p in self.pages:
                sec = p.get("section_name", "Uncategorized")
                pages_by_section.setdefault(sec, []).append(p.get("page_name", "?"))
            for sec_name in sorted(pages_by_section.keys()):
                page_names = pages_by_section[sec_name]
                names_str = ", ".join(page_names)
                summary_parts.append(f"  {sec_name}: {names_str}")
            summary_parts.append("→ Add new pages to EXISTING sections above. Do NOT create similar/duplicate pages.")

        # Add details (truncated if too many)
        if len(results) <= 10:
            summary_parts.extend(results)
        else:
            summary_parts.extend(results[:5])
            summary_parts.append(f"... and {len(results) - 5} more")

        return "\n".join(summary_parts)

    def _handle_explore_tree(self) -> str:
        """Handle explore_tree tool call - deep repository exploration."""
        if not self.repo_root:
            return "ERROR: repo_root not configured for deep exploration"

        logger.info("[STRUCTURE_PLANNER][TOOL] explore_tree: Starting deep repository exploration")

        # Use cached result if available
        if self._tree_analysis is not None:
            logger.info("[STRUCTURE_PLANNER][TOOL] explore_tree: Using cached analysis")
            return format_tree_for_llm(self._tree_analysis)

        # Perform deep exploration using auto-detected depth
        self._tree_analysis = explore_repository_tree(
            self.repo_root,
            max_depth=self.effective_depth,
        )

        # Register only "content directories" — directories that directly
        # contain ≥1 file.  Intermediate path segments (e.g. src/, src/v/)
        # that only contain subdirectories are NOT meaningful for coverage.
        MAX_DISCOVERY_DEPTH = self.effective_depth
        directories = self._tree_analysis.get("directories", [])

        # Group directories by depth for logging
        dirs_by_depth = {}
        for d in directories:
            depth = d.get("depth", 1)
            if depth <= MAX_DISCOVERY_DEPTH:
                if depth not in dirs_by_depth:
                    dirs_by_depth[depth] = []
                dirs_by_depth[depth].append(d)

        # Register only directories that have direct files
        for depth in range(1, MAX_DISCOVERY_DEPTH + 1):
            for d in dirs_by_depth.get(depth, []):
                dir_path = d.get("path", "") or d.get("name", "")
                direct_files = d.get("files", 0)
                if dir_path and direct_files > 0 and dir_path.split("/")[-1] not in IGNORED_DIRS:
                    self.discovered_dirs.add(dir_path)

        # Count directories for logging
        total_registered = len(self.discovered_dirs)
        depth_counts = {d: len(dirs_by_depth.get(d, [])) for d in range(1, MAX_DISCOVERY_DEPTH + 1)}

        # Count large directories (potential high-complexity modules)
        all_important_dirs = [d for d in directories if d.get("depth", 1) <= MAX_DISCOVERY_DEPTH]
        large_dirs = [d for d in all_important_dirs if d.get("files", 0) >= 10 or d.get("subdirs", 0) >= 3]

        logger.info(
            "[STRUCTURE_PLANNER][TOOL] explore_tree: Found %d dirs (by depth: %s), %d large, %d registered",
            len(all_important_dirs),
            depth_counts,
            len(large_dirs),
            total_registered,
        )

        return format_tree_for_llm(self._tree_analysis)

    def _handle_think(self, analysis: str) -> str:
        """Handle think tool call - strategic reflection."""
        logger.info("[STRUCTURE_PLANNER][TOOL] think: Recording strategic analysis")
        self._thinking_log.append(analysis)

        # Provide helpful feedback with exploration enforcement
        feedback_parts = ["✓ Analysis recorded."]
        warnings = []

        # CRITICAL: Show coverage status during planning (not after!)
        if self.discovered_dirs and not self.pages:
            uncovered = self._get_uncovered_dirs()
            if uncovered:
                # Group by depth for clearer display
                depth_2_uncovered = [d for d in uncovered if d.count("/") == 1]  # src/services
                depth_3_uncovered = [d for d in uncovered if d.count("/") == 2]  # src/services/orders

                if depth_2_uncovered:
                    warnings.append(f"📋 UNCOVERED DIRECTORIES (depth 2): {', '.join(sorted(depth_2_uncovered)[:8])}")
                if depth_3_uncovered:
                    warnings.append(f"📋 UNCOVERED DIRECTORIES (depth 3): {', '.join(sorted(depth_3_uncovered)[:8])}")

        # Check exploration depth using directories
        if self._tree_analysis:
            directories = self._tree_analysis.get("directories", [])
            depth_1_dirs = [d for d in directories if d.get("depth") == 1]
            # Large directories are likely high-complexity
            large_dirs = [d for d in depth_1_dirs if d.get("files", 0) >= 10 or d.get("subdirs", 0) >= 3]

            # Check if large directories were analyzed via query_graph
            analyzed_count = len(self._graph_module_cache)

            if large_dirs and analyzed_count == 0 and not self.sections:
                warnings.append(
                    f"⚠️ WARNING: You have {len(large_dirs)} LARGE directories "
                    f"but haven't used query_graph() to analyze any of them. "
                    f"Consider calling query_graph() for: {', '.join(d['name'] for d in large_dirs[:3])}"
                )
            elif large_dirs and analyzed_count < len(large_dirs) // 2 and not self.sections:
                unanalyzed = [d["name"] for d in large_dirs if d.get("path") not in self._graph_module_cache]
                if unanalyzed:
                    warnings.append(
                        f"TIP: Consider analyzing more directories before defining structure. "
                        f"Unanalyzed large directories: {', '.join(unanalyzed[:3])}"
                    )

        # Suggest next steps with coverage enforcement
        if warnings:
            feedback_parts.extend(warnings)

        if not self.sections:
            feedback_parts.append(
                "NEXT: Define sections based on CAPABILITIES you discovered, not directory names. "
                "IMPORTANT: Every directory listed above MUST be covered by at least one page."
            )
        elif self.sections and not self.pages:
            uncovered = self._get_uncovered_dirs()
            coverage_msg = (
                f"NEXT: Define SPECIFIC pages for your {len(self.sections)} sections. "
                "Each page should cover a distinct feature or component."
            )
            if uncovered:
                coverage_msg += f" ⚠️ {len(uncovered)} directories still need coverage!"
            feedback_parts.append(coverage_msg)

        return " ".join(feedback_parts)

    def _handle_analyze_module(self, module_path: str) -> str:
        """Handle analyze_module tool call - detailed module analysis."""
        import os

        if not self.repo_root:
            return "ERROR: repo_root not configured"

        # Resolve path relative to repo root
        full_path = os.path.join(self.repo_root, module_path.lstrip("/"))

        if not os.path.isdir(full_path):
            return f"ERROR: '{module_path}' is not a valid directory"

        logger.info(f"[STRUCTURE_PLANNER][TOOL] analyze_module: {module_path}")

        try:
            items = sorted(os.listdir(full_path))
        except (PermissionError, OSError) as e:
            return f"ERROR: Cannot read directory: {e}"

        # Analyze contents
        code_files = []
        doc_files = []
        subdirs = []

        for item in items:
            if _should_ignore_dir(item):
                continue

            item_path = os.path.join(full_path, item)
            if os.path.isdir(item_path):
                # Count code and doc files in subdir
                try:
                    subdir_items = os.listdir(item_path)
                    subdir_code = sum(1 for i in subdir_items if _is_code_file(i))
                    subdir_docs = sum(1 for i in subdir_items if _is_doc_file(i))
                    subdirs.append({"name": item, "code_files": subdir_code, "doc_files": subdir_docs})
                except (PermissionError, OSError):
                    subdirs.append({"name": item, "code_files": 0, "doc_files": 0})
            elif _is_code_file(item):
                code_files.append(item)
            elif _is_doc_file(item):
                doc_files.append(item)

        # Build response
        lines = [f"=== Module Analysis: {module_path} ===", ""]

        if code_files:
            lines.append(f"Code files ({len(code_files)}):")
            for f in code_files[:20]:
                lines.append(f"  📄 {f}")
            if len(code_files) > 20:
                lines.append(f"  ... and {len(code_files) - 20} more")
            lines.append("")

        if doc_files:
            lines.append(f"Documentation files ({len(doc_files)}):")
            for f in doc_files[:10]:
                lines.append(f"  📝 {f}")
            if len(doc_files) > 10:
                lines.append(f"  ... and {len(doc_files) - 10} more")
            lines.append("")

        if subdirs:
            lines.append(f"Subdirectories ({len(subdirs)}):")
            for sd in subdirs:
                parts = [f"{sd['code_files']} code"]
                if sd.get("doc_files"):
                    parts.append(f"{sd['doc_files']} doc")
                lines.append(f"  📁 {sd['name']}/ ({', '.join(parts)} files)")
            lines.append("")

        # Recommendation
        total_code = len(code_files) + sum(sd["code_files"] for sd in subdirs)
        if len(subdirs) >= 3 or total_code >= 15:
            lines.append("RECOMMENDATION: This module is complex.")
            lines.append("Consider creating multiple pages:")
            if subdirs:
                for sd in subdirs:
                    if sd["code_files"] >= 3:
                        lines.append(f"  - Page for '{sd['name']}' ({sd['code_files']} files)")
        elif total_code >= 5:
            lines.append("RECOMMENDATION: Single dedicated page is appropriate.")
        else:
            lines.append("RECOMMENDATION: Can be grouped with related modules.")

        return "\n".join(lines)

    def _handle_query_graph(self, path_prefix: str, text_filter: str = None) -> str:
        """Handle query_graph tool call - analyze symbols from the code graph.

        Queries the nx.DiGraph to find all symbols whose rel_path starts with path_prefix.
        Returns ALL symbols with system context (inheritance, usage, relationships).

        When *text_filter* is provided and the FTS5 index is available, candidates
        are intersected with BM25-ranked text matches so that only symbols whose
        name, docstring, or content match the keywords are returned — with full
        relationship context preserved.

        This provides accurate complexity data based on actual parsed code,
        not just file counts.
        """
        if not self.code_graph:
            return "Graph not available. Use filesystem analysis instead."

        # Normalize path prefix
        path_prefix = path_prefix.strip("/")
        if not path_prefix:
            return "ERROR: path_prefix is required (e.g., 'src/services/orders')"

        # Build list of prefixes to try - the path might have a repo name prefix
        # e.g., LLM asks for 'tools' but rel_path is 'my_project/tools/...'
        prefixes_to_try = [path_prefix]

        # Try to detect repo name prefix from first few nodes
        if not hasattr(self, "_detected_repo_prefix"):
            self._detected_repo_prefix = None
            sample_count = 0
            for _node_id, node_data in self.code_graph.nodes(data=True):
                rel_path = node_data.get("rel_path", "")
                if rel_path and "/" in rel_path:
                    first_segment = rel_path.split("/")[0]
                    # Common patterns: my_project, src, lib, etc.
                    if first_segment and not first_segment.endswith(".py"):
                        self._detected_repo_prefix = first_segment
                        break
                sample_count += 1
                if sample_count > 100:
                    break

        # If we detected a repo prefix, add it to search paths
        if self._detected_repo_prefix and not path_prefix.startswith(self._detected_repo_prefix):
            prefixes_to_try.append(f"{self._detected_repo_prefix}/{path_prefix}")

        # Also try with the path_prefix as a suffix match for deeply nested queries
        # e.g., 'services/orders' should match 'my_project/services/orders'

        logger.info(
            f"[STRUCTURE_PLANNER][TOOL] query_graph: Trying prefixes {prefixes_to_try} for {path_prefix}"
            f"{f' (text_filter={text_filter!r})' if text_filter else ''}"
        )

        # Check cache — include text_filter in cache key only when actually filtering
        cache_key = f"{path_prefix}|{text_filter}" if text_filter else path_prefix
        if cache_key in self._graph_module_cache:
            logger.info(f"[STRUCTURE_PLANNER][TOOL] query_graph: Using cache for {cache_key}")
            return self._format_graph_analysis(path_prefix, self._graph_module_cache[cache_key])

        logger.info(f"[STRUCTURE_PLANNER][TOOL] query_graph: Analyzing symbols under {path_prefix}")

        # Type annotation keys used by different parsers (not yet standardized)
        # TODO: See PARSER_STANDARDIZATION.md for unification plan
        TYPE_USAGE_ANNOTATIONS = {
            "usage": ["return_type", "parameter_type", "parameter"],  # C++ parser
            "reference_context": ["return_type", "parameter", "annotation"],  # Python parser
            "reference_type": ["type_usage"],  # TypeScript parser
        }

        def _get_type_references_from_node(node_id: str, max_types: int = 5) -> set:
            """Extract type references from REFERENCES edges of a node.

            Handles multiple annotation key formats from different parsers.
            Returns set of type names used by this symbol (return types, param types).
            """
            types_used = set()
            for succ in self.code_graph.successors(node_id):
                edge_data = self.code_graph.get_edge_data(node_id, succ)
                if not edge_data:
                    continue
                for edge in edge_data.values():
                    rel_type = str(edge.get("relationship_type", "")).lower()
                    if rel_type == "references":
                        annotations = edge.get("annotations", {})
                        is_type_ref = False
                        # Check all known annotation keys/values
                        for key, values in TYPE_USAGE_ANNOTATIONS.items():
                            if key in annotations:
                                if annotations[key] in values:
                                    is_type_ref = True
                                    break
                        if is_type_ref:
                            succ_data = self.code_graph.nodes.get(succ, {})
                            type_name = succ_data.get("symbol_name", "") or succ_data.get("name", "")
                            if type_name and len(types_used) < max_types:
                                types_used.add(type_name)
            return types_used

        # Query graph nodes by rel_path prefix
        symbols_by_type = {}  # type -> [symbol_names]
        symbols_by_file = {}  # file -> [symbols]
        all_classes = []  # ALL classes/interfaces/structs/traits (not just top 25)
        all_functions = []  # ALL functions with creates info
        all_enums = []  # Enums/constants
        all_type_aliases = []  # Type aliases (using/typedef)
        all_macros = []  # C/C++ preprocessor macros (architecturally significant)
        all_docs = []  # Documentation files for target_docs field
        node_ids_in_scope = {}  # symbol_name -> node_id for relationship lookup

        def matches_any_prefix(rel_path_norm: str, prefixes: list) -> bool:
            """Check if rel_path starts with any of the given directory prefixes.

            Enforces a directory boundary so that prefix 'src' does NOT match
            'srcgen/foo.py'.  The path must either start with 'prefix/' or
            equal the prefix exactly.
            """
            for prefix in prefixes:
                if (
                    rel_path_norm.startswith(prefix + "/")
                    or rel_path_norm == prefix
                    or
                    # Also check if the path contains the prefix as a segment
                    # e.g., 'my_project/services/orders' matches query for 'services/orders'
                    f"/{prefix}/" in f"/{rel_path_norm}/"
                ):
                    return True
            return False

        try:
            # --- Gather candidate nodes by path prefix ---
            # Use FTS5 index if available (O(log N)), else brute-force (O(N))
            fts = self.graph_text_index
            candidate_nodes = []  # [(node_id, node_data_dict)]

            if fts is not None and fts.is_open:
                if text_filter:
                    # -------------------------------------------------
                    # TEXT-FIRST strategy: find text matches globally,
                    # then post-filter by path prefix.
                    # This avoids the k-cap problem where path-prefix
                    # results (limited to k rows in SQL LIMIT) miss the
                    # target symbols that sit beyond the cap.
                    # -------------------------------------------------

                    # 1. BM25 ranked text search (concept / keyword match)
                    text_docs = fts.search_smart(
                        text_filter,
                        k=500,
                        intent="general",
                    )
                    # 2. SQL LIKE name search — catches CamelCase names
                    #    that FTS5 tokenisation may fragment or miss
                    name_docs = fts.search_by_name(
                        text_filter,
                        exact=False,
                        k=100,
                    )

                    # Merge by node_id — deduplicate, text_docs first
                    merged: dict = {}  # node_id → metadata dict
                    for doc in text_docs:
                        nid = doc.metadata.get("node_id", "")
                        if nid and nid not in merged:
                            merged[nid] = dict(doc.metadata)
                    for doc in name_docs:
                        nid = doc.metadata.get("node_id", "")
                        if nid and nid not in merged:
                            merged[nid] = dict(doc.metadata)

                    # Post-filter by path prefix
                    for nid, meta in merged.items():
                        rp = (meta.get("rel_path") or meta.get("file_path", "")).strip("/")
                        if any(rp.startswith(pfx + "/") or rp == pfx for pfx in prefixes_to_try):
                            candidate_nodes.append((nid, meta))

                    logger.info(
                        f"[STRUCTURE_PLANNER][TOOL] query_graph: text_filter "
                        f"{text_filter!r} found {len(candidate_nodes)} nodes "
                        f"via text-first (BM25={len(text_docs)}, name={len(name_docs)}, "
                        f"merged={len(merged)}, path-filtered={len(candidate_nodes)})"
                    )
                else:
                    # --- NO TEXT FILTER: retrieve all symbols by path ---
                    # Dynamic FTS k: scale with repo size to prevent truncation
                    # on huge directories (e.g., src/ in a 5K-file monorepo).
                    # Small repos: 5000 (default).  Large repos: up to 20000.
                    # Exclude DOC_SYMBOL_TYPES so doc nodes don't consume
                    # SQL LIMIT budget meant for code symbols.
                    fts_k = min(5000 + self._repository_file_count * 2, 20000)
                    for prefix in prefixes_to_try:
                        rows = fts.search_by_path_prefix(
                            prefix,
                            k=fts_k,
                            exclude_types=DOC_SYMBOL_TYPES,
                        )
                        for row in rows:
                            nid = row.get("node_id", "")
                            if nid:
                                candidate_nodes.append((nid, row))
                    # Deduplicate
                    seen_nids = set()
                    unique_candidates = []
                    for nid, data in candidate_nodes:
                        if nid not in seen_nids:
                            seen_nids.add(nid)
                            unique_candidates.append((nid, data))
                    candidate_nodes = unique_candidates
                    logger.info(
                        f"[STRUCTURE_PLANNER][TOOL] query_graph: FTS5 "
                        f"returned {len(candidate_nodes)} nodes for {prefixes_to_try}"
                    )
            else:
                # No FTS5 — brute-force graph scan (supports text_filter inline)
                tf_lower = text_filter.lower() if text_filter else None
                tf_tokens = [t for t in tf_lower.split() if len(t) >= 2] if tf_lower else None
                for node_id, node_data in self.code_graph.nodes(data=True):
                    rel_path = node_data.get("rel_path") or node_data.get("file_path", "")
                    if not rel_path:
                        continue
                    rel_path_normalized = rel_path.strip("/")
                    if not matches_any_prefix(rel_path_normalized, prefixes_to_try):
                        continue
                    # Apply text_filter as brute-force substring match
                    if tf_tokens:
                        sym_name = (node_data.get("symbol_name", "") or node_data.get("name", "") or "").lower()
                        if not any(tok in sym_name for tok in tf_tokens):
                            continue
                    candidate_nodes.append((node_id, dict(node_data)))
                if text_filter:
                    logger.info(
                        f"[STRUCTURE_PLANNER][TOOL] query_graph: brute-force "
                        f"found {len(candidate_nodes)} nodes for {prefixes_to_try} "
                        f"with text_filter={text_filter!r}"
                    )

            for node_id, node_data in candidate_nodes:
                # Extract rel_path for current node (works for both FTS5 rows and graph data)
                rel_path = node_data.get("rel_path") or node_data.get("file_path", "")
                rel_path_normalized = rel_path.strip("/") if rel_path else ""

                # Extract symbol info - check both 'symbol_name' (comprehensive) and 'name' (basic)
                symbol_name = node_data.get("symbol_name", "") or node_data.get("name", "")
                if not symbol_name:
                    continue

                symbol_type = (
                    node_data.get("symbol_type") or node_data.get("type") or node_data.get("node_type", "unknown")
                ).lower()

                node_ids_in_scope[symbol_name] = node_id

                # Aggregate by type
                if symbol_type not in symbols_by_type:
                    symbols_by_type[symbol_type] = []
                symbols_by_type[symbol_type].append(symbol_name)

                # Group by file for better organization
                file_key = rel_path_normalized.split("/")[-1] if "/" in rel_path_normalized else rel_path_normalized
                if file_key not in symbols_by_file:
                    symbols_by_file[file_key] = []
                symbols_by_file[file_key].append({"name": symbol_name, "type": symbol_type})

                # Count connections for importance
                connections = self.code_graph.in_degree(node_id) + self.code_graph.out_degree(node_id)

                # Extract first-line docstring summary for semantic context.
                # For FTS5-sourced candidates, look up the authoritative graph
                # node to handle comprehensive parser nodes properly.
                _graph_ndata = self.code_graph.nodes.get(node_id, {})
                _desc = _get_symbol_brief(_graph_ndata)

                # Collect ALL symbols by type
                if symbol_type in ("class", "interface", "protocol", "struct", "trait"):
                    # Get inheritance info
                    inherits_from = []
                    implemented_by = []
                    created_by = []  # What creates instances of this class (factory/DI insight)
                    specializes_templates = []  # Template bases this class specializes (C++)
                    specialized_by = []  # Classes that specialize this template (C++)

                    # C++/Java multi-language support: declaration/implementation split
                    declared_in = None  # Header file (.h, .hpp) where this is declared
                    implemented_in = []  # Implementation files (.cpp, .cc) with method bodies
                    has_out_of_line_impl = False  # Track if class has split decl/impl
                    types_used = set()  # Types used in method signatures (return/param types)

                    for succ in self.code_graph.successors(node_id):
                        edge_data = self.code_graph.get_edge_data(node_id, succ)
                        if edge_data:
                            for edge in edge_data.values():
                                rel_type = str(edge.get("relationship_type", "")).lower()
                                if rel_type in ("inheritance", "extends", "implements"):
                                    succ_data = self.code_graph.nodes.get(succ, {})
                                    parent_name = succ_data.get("symbol_name", "") or succ_data.get("name", "")
                                    if parent_name:
                                        inherits_from.append(parent_name)
                                elif rel_type == "specializes":
                                    # Template specialization: this class → template base
                                    succ_data = self.code_graph.nodes.get(succ, {})
                                    tmpl_name = succ_data.get("symbol_name", "") or succ_data.get("name", "")
                                    if tmpl_name and len(specializes_templates) < 5:
                                        specializes_templates.append(tmpl_name)
                                elif rel_type == "defines":
                                    # This class DEFINES methods - check if they have out-of-line implementations
                                    succ_data = self.code_graph.nodes.get(succ, {})
                                    succ_type = succ_data.get("symbol_type", "").lower()
                                    if succ_type in ("method", "constructor", "function"):
                                        # Collect type references from this method's signature
                                        method_types = _get_type_references_from_node(succ, max_types=3)
                                        types_used.update(method_types)

                                        # Check if this method has DEFINES_BODY (out-of-line impl)
                                        for impl_node in self.code_graph.predecessors(succ):
                                            impl_edge = self.code_graph.get_edge_data(impl_node, succ)
                                            if impl_edge:
                                                for impl_e in impl_edge.values():
                                                    if (
                                                        str(impl_e.get("relationship_type", "")).lower()
                                                        == "defines_body"
                                                    ):
                                                        impl_data = self.code_graph.nodes.get(impl_node, {})
                                                        impl_file = impl_data.get("file_path", "")
                                                        if impl_file and impl_file != rel_path:
                                                            # Found out-of-line implementation
                                                            impl_rel = impl_data.get("rel_path", impl_file)
                                                            if (
                                                                impl_rel not in implemented_in
                                                                and len(implemented_in) < 3
                                                            ):
                                                                implemented_in.append(impl_rel)
                                                            has_out_of_line_impl = True

                    # Check who implements/extends this AND who CREATES instances
                    for pred in self.code_graph.predecessors(node_id):
                        edge_data = self.code_graph.get_edge_data(pred, node_id)
                        if edge_data:
                            for edge in edge_data.values():
                                rel_type = str(edge.get("relationship_type", "")).lower()
                                if rel_type in ("inheritance", "extends", "implements"):
                                    pred_data = self.code_graph.nodes.get(pred, {})
                                    child_name = pred_data.get("symbol_name", "") or pred_data.get("name", "")
                                    if child_name and len(implemented_by) < 5:
                                        implemented_by.append(child_name)
                                elif rel_type in ("creates", "instantiates", "constructs"):
                                    # Track what creates instances of this class
                                    pred_data = self.code_graph.nodes.get(pred, {})
                                    creator_name = pred_data.get("symbol_name", "") or pred_data.get("name", "")
                                    if creator_name and len(created_by) < 5:
                                        created_by.append(creator_name)
                                elif rel_type == "specializes":
                                    # This class is a template that is specialized by pred
                                    pred_data = self.code_graph.nodes.get(pred, {})
                                    spec_name = pred_data.get("symbol_name", "") or pred_data.get("name", "")
                                    if spec_name and len(specialized_by) < 5:
                                        specialized_by.append(spec_name)
                                elif rel_type == "declares":
                                    # This class is DECLARED in a header (pred is the header/declaration)
                                    pred_data = self.code_graph.nodes.get(pred, {})
                                    decl_file = pred_data.get("rel_path", pred_data.get("file_path", ""))
                                    if decl_file:
                                        declared_in = decl_file

                    class_info = {
                        "name": symbol_name,
                        "type": symbol_type,
                        "file": file_key,
                        "connections": connections,
                        "inherits_from": inherits_from,
                        "implemented_by": implemented_by,
                        "created_by": created_by,  # Factory methods, constructors that instantiate this
                    }
                    if _desc:
                        class_info["desc"] = _desc

                    # Add template specialization info (C++ specific)
                    if specializes_templates:
                        class_info["specializes"] = specializes_templates
                    if specialized_by:
                        class_info["specialized_by"] = specialized_by

                    # Add C++/Java specific fields only if present (avoid noise for Python)
                    if declared_in:
                        class_info["declared_in"] = declared_in
                    if implemented_in:
                        class_info["implemented_in"] = implemented_in
                    if has_out_of_line_impl:
                        class_info["has_split_impl"] = True
                    # Add type dependencies (limit to 5 most relevant)
                    if types_used:
                        class_info["uses_types"] = list(types_used)[:5]

                    all_classes.append(class_info)
                elif symbol_type == "function":
                    # Get what this function CREATES (object instantiation)
                    creates = []
                    for succ in self.code_graph.successors(node_id):
                        edge_data = self.code_graph.get_edge_data(node_id, succ)
                        if edge_data:
                            for edge in edge_data.values():
                                rel_type = str(edge.get("relationship_type", "")).lower()
                                if rel_type in ("creates", "instantiates", "constructs"):
                                    succ_data = self.code_graph.nodes.get(succ, {})
                                    created_name = succ_data.get("symbol_name", "") or succ_data.get("name", "")
                                    if created_name and len(creates) < 5:
                                        creates.append(created_name)

                    # Get type references directly (return type, param types)
                    fn_types = _get_type_references_from_node(node_id, max_types=5)

                    fn_info = {
                        "name": symbol_name,
                        "file": file_key,
                        "connections": connections,
                        "creates": creates,  # What objects this function instantiates
                    }
                    if _desc:
                        fn_info["desc"] = _desc
                    if fn_types:
                        fn_info["uses_types"] = list(fn_types)
                    all_functions.append(fn_info)
                elif symbol_type == "type_alias":
                    # Track what this alias resolves to and who uses it
                    alias_of = []
                    used_by = []
                    for succ in list(self.code_graph.successors(node_id))[:5]:
                        edge_data = self.code_graph.get_edge_data(node_id, succ)
                        if edge_data:
                            for edge in edge_data.values():
                                rel_type = str(edge.get("relationship_type", "")).lower()
                                if rel_type in ("alias_of", "uses_type", "references"):
                                    succ_data = self.code_graph.nodes.get(succ, {})
                                    target_name = succ_data.get("symbol_name", "") or succ_data.get("name", "")
                                    if target_name:
                                        alias_of.append(target_name)
                    for pred in list(self.code_graph.predecessors(node_id))[:5]:
                        pred_data = self.code_graph.nodes.get(pred, {})
                        user_name = pred_data.get("symbol_name", "") or pred_data.get("name", "")
                        if user_name:
                            used_by.append(user_name)

                    ta_info = {
                        "name": symbol_name,
                        "type": symbol_type,
                        "file": file_key,
                        "alias_of": alias_of,
                        "used_by": used_by,
                    }
                    if _desc:
                        ta_info["desc"] = _desc
                    all_type_aliases.append(ta_info)
                elif symbol_type in ("enum", "constant"):
                    # Find what uses this enum/constant
                    used_by = []
                    for pred in list(self.code_graph.predecessors(node_id))[:5]:
                        pred_data = self.code_graph.nodes.get(pred, {})
                        user_name = pred_data.get("symbol_name", "") or pred_data.get("name", "")
                        if user_name:
                            used_by.append(user_name)

                    enum_info = {"name": symbol_name, "type": symbol_type, "file": file_key, "used_by": used_by}
                    if _desc:
                        enum_info["desc"] = _desc
                    all_enums.append(enum_info)
                elif symbol_type in DOC_SYMBOL_TYPES:
                    # Collect documentation files for target_docs field
                    all_docs.append(
                        {"path": rel_path_normalized, "name": symbol_name, "type": symbol_type, "file": file_key}
                    )
                elif symbol_type == "macro":
                    # C/C++ preprocessor macros — architecturally significant
                    # Find what references/uses this macro
                    used_by = []
                    references_types = []
                    for succ in list(self.code_graph.successors(node_id))[:5]:
                        edge_data = self.code_graph.get_edge_data(node_id, succ)
                        if edge_data:
                            for edge in edge_data.values():
                                rel_type = str(edge.get("relationship_type", "")).lower()
                                if rel_type in ("references", "calls"):
                                    succ_data = self.code_graph.nodes.get(succ, {})
                                    ref_name = succ_data.get("symbol_name", "") or succ_data.get("name", "")
                                    if ref_name and len(references_types) < 5:
                                        references_types.append(ref_name)
                    for pred in list(self.code_graph.predecessors(node_id))[:5]:
                        edge_data = self.code_graph.get_edge_data(pred, node_id)
                        if edge_data:
                            for edge in edge_data.values():
                                rel_type = str(edge.get("relationship_type", "")).lower()
                                if rel_type in ("references", "calls", "uses"):
                                    pred_data = self.code_graph.nodes.get(pred, {})
                                    user_name = pred_data.get("symbol_name", "") or pred_data.get("name", "")
                                    if user_name and len(used_by) < 5:
                                        used_by.append(user_name)

                    macro_info = {
                        "name": symbol_name,
                        "file": file_key,
                        "references": references_types,
                        "used_by": used_by,
                        "connections": connections,
                    }
                    if _desc:
                        macro_info["desc"] = _desc
                    all_macros.append(macro_info)
        except Exception as e:
            logger.warning(f"[STRUCTURE_PLANNER][TOOL] query_graph error: {e}")
            return f"ERROR: Failed to query graph: {e}"

        # Sort by connections (importance)
        all_classes.sort(key=lambda x: -x["connections"])
        all_functions.sort(key=lambda x: -x["connections"])

        # Calculate metrics — use full sets that match the buckets
        total_symbols = sum(len(v) for v in symbols_by_type.values())
        class_count = len(
            [
                s
                for t, syms in symbols_by_type.items()
                if t in ("class", "interface", "protocol", "struct", "trait")
                for s in syms
            ]
        )
        function_count = len(
            [s for t, syms in symbols_by_type.items() if t in ("function", "method", "def") for s in syms]
        )
        type_alias_count = len(all_type_aliases)

        # Generate grouping suggestions based on relationships
        grouping_suggestions = self._generate_grouping_suggestions(
            all_classes, all_enums, all_functions, symbols_by_file
        )

        # Store in cache - ALL symbols, not truncated
        analysis = {
            "total_symbols": total_symbols,
            "class_count": class_count,
            "function_count": function_count,
            "type_alias_count": type_alias_count,
            "symbols_by_type": {k: len(v) for k, v in symbols_by_type.items()},
            "symbols_by_file": symbols_by_file,
            "files_with_symbols": len(symbols_by_file),
            "all_classes": all_classes,  # ALL classes/interfaces/structs/traits with system context
            "all_functions": all_functions[:50],  # Limit functions (less important for docs)
            "all_enums": all_enums,  # ALL enums/constants
            "all_type_aliases": all_type_aliases,  # ALL type aliases with alias_of info
            "all_macros": all_macros,  # C/C++ preprocessor macros
            "all_docs": all_docs,  # Documentation files for target_docs
            "grouping_suggestions": grouping_suggestions,
        }
        self._graph_module_cache[cache_key] = analysis

        # Build detailed log line with all architectural types
        parts = [f"{total_symbols} symbols"]
        if class_count:
            parts.append(f"{class_count} classes/interfaces/traits")
        if function_count:
            parts.append(f"{function_count} functions")
        if len(all_enums):
            parts.append(f"{len(all_enums)} enums/constants")
        if type_alias_count:
            parts.append(f"{type_alias_count} type_aliases")
        if len(all_macros):
            parts.append(f"{len(all_macros)} macros")
        if len(all_docs):
            parts.append(f"{len(all_docs)} docs")
        logger.info(f"[STRUCTURE_PLANNER][TOOL] query_graph: {path_prefix} -> {', '.join(parts)}")

        return self._format_graph_analysis(path_prefix, analysis)

    def _generate_grouping_suggestions(
        self, classes: list[dict], enums: list[dict], functions: list[dict], symbols_by_file: dict[str, list]
    ) -> list[dict]:
        """Generate smart grouping suggestions based on relationships.

        Groups symbols that belong together:
        - Classes with their related enums/constants
        - Base classes with implementations
        - Small utilities from same file
        """
        suggestions = []
        used_symbols = set()

        # Strategy 1: Group classes with enums they use
        for cls in classes:
            cls_name = cls["name"]
            if cls_name in used_symbols:
                continue

            group = {"main": cls_name, "related": [], "rationale": ""}

            # Find enums used by this class
            for enum in enums:
                if cls_name in enum.get("used_by", []) and enum["name"] not in used_symbols:
                    group["related"].append(enum["name"])
                    used_symbols.add(enum["name"])

            # Find classes that extend this (include base + implementations together)
            for impl in cls.get("implemented_by", []):
                if impl not in used_symbols:
                    group["related"].append(impl)
                    used_symbols.add(impl)

            if group["related"]:
                group["rationale"] = f"Class with related types ({len(group['related'])} items)"
                suggestions.append(group)
                used_symbols.add(cls_name)

        # Strategy 2: Group small enums/constants from same file
        for file_name, file_symbols in symbols_by_file.items():
            ungrouped_enums = [
                s
                for s in file_symbols
                if s["type"] in ("enum", "constant", "type_alias") and s["name"] not in used_symbols
            ]
            if len(ungrouped_enums) >= 2:
                # Find a class in this file to group with
                file_classes = [s for s in file_symbols if s["type"] in ("class", "interface")]
                main_symbol = file_classes[0]["name"] if file_classes else ungrouped_enums[0]["name"]

                suggestions.append(
                    {
                        "main": main_symbol,
                        "related": [e["name"] for e in ungrouped_enums if e["name"] != main_symbol],
                        "rationale": f"Related types from {file_name}",
                    }
                )
                for e in ungrouped_enums:
                    used_symbols.add(e["name"])

        return suggestions[:10]  # Limit to 10 suggestions

    def _format_graph_analysis(self, path_prefix: str, analysis: dict) -> str:
        """Format graph analysis for LLM consumption with system context.

        Token-aware formatting:
        - Prioritizes symbols WITH relationships (more valuable for documentation)
        - Summarizes orphan symbols (no relationships) to save tokens
        - Caps output to avoid context window bloat
        - Switches to COMPACT mode when total symbols > 80 to save context window
        """
        lines = [f"=== Code Graph Analysis: {path_prefix} ===", ""]

        total = analysis.get("total_symbols", 0)
        if total == 0:
            lines.append("No symbols found in graph for this path.")
            lines.append("This may be a documentation/data directory or the path has no parseable code.")
            lines.append("TIP: Use explore_tree to list files under this directory.")
            lines.append("Documentation pages can reference files by path without requiring code symbols.")
            return "\n".join(lines)

        # Use compact format for high-density modules to save LLM context tokens
        compact = total > 80

        lines.append(f"Total symbols: {total}")
        lines.append(f"Classes/Interfaces/Traits: {analysis.get('class_count', 0)}")
        lines.append(f"Functions/Methods: {analysis.get('function_count', 0)}")
        ta_count = analysis.get("type_alias_count", 0)
        enum_count = len(analysis.get("all_enums", []))
        if ta_count:
            lines.append(f"Type Aliases: {ta_count}")
        if enum_count:
            lines.append(f"Enums/Constants: {enum_count}")
        macro_count = len(analysis.get("all_macros", []))
        if macro_count:
            lines.append(f"Macros: {macro_count}")
        lines.append(f"Files with symbols: {analysis.get('files_with_symbols', 0)}")
        if compact:
            lines.append("FORMAT: COMPACT (high symbol count — showing one-line summaries)")
        lines.append("")

        # TOKEN OPTIMIZATION: Separate classes with relationships from orphans
        all_classes = analysis.get("all_classes", [])
        # Caps for each section — reduced in compact mode
        class_detail_cap = 20 if compact else 40
        orphan_cap = 5 if compact else 10
        enum_cap = 15 if compact else 25
        ta_cap = 10 if compact else 20
        macro_cap = 15 if compact else 25
        func_cap = 8 if compact else 15

        if all_classes:
            # Classes with relationships get full detail (including C++ split impl, type deps)
            classes_with_context = [
                c
                for c in all_classes
                if c.get("inherits_from")
                or c.get("implemented_by")
                or c.get("created_by")
                or c.get("specializes")
                or c.get("specialized_by")
                or c.get("declared_in")
                or c.get("implemented_in")
                or c.get("has_split_impl")
                or c.get("uses_types")
            ]
            # Classes without relationships get summarized
            orphan_classes = [c for c in all_classes if c not in classes_with_context]

            # Sort by connections (importance)
            classes_with_context.sort(key=lambda x: -x.get("connections", 0))
            orphan_classes.sort(key=lambda x: -x.get("connections", 0))

            if classes_with_context:
                lines.append(f"CLASSES WITH RELATIONSHIPS ({len(classes_with_context)} - full detail):")
                for sym in classes_with_context[:class_detail_cap]:
                    context_parts = []
                    if sym.get("inherits_from"):
                        context_parts.append(f"extends {', '.join(sym['inherits_from'][:3])}")
                    if sym.get("implemented_by"):
                        context_parts.append(f"implemented by {', '.join(sym['implemented_by'][:3])}")
                    if sym.get("created_by"):
                        context_parts.append(f"created by {', '.join(sym['created_by'][:3])}")

                    # Template specialization (C++)
                    if sym.get("specializes"):
                        context_parts.append(f"specializes {', '.join(sym['specializes'][:3])}")
                    if sym.get("specialized_by"):
                        context_parts.append(f"specialized by {', '.join(sym['specialized_by'][:3])}")

                    # Type dependencies - aggregated from method signatures
                    if sym.get("uses_types"):
                        context_parts.append(f"uses {', '.join(sym['uses_types'][:5])}")

                    # C++/Java: Show header/implementation split
                    if sym.get("declared_in"):
                        context_parts.append(f"declared in {sym['declared_in']}")
                    if sym.get("implemented_in"):
                        impl_files = sym["implemented_in"][:2]  # Max 2 impl files
                        context_parts.append(f"impl in {', '.join(impl_files)}")

                    context_str = f" → {'; '.join(context_parts)}" if context_parts else ""
                    conn_str = f" [{sym['connections']} refs]" if sym.get("connections") else ""
                    desc_str = f" — {sym['desc']}" if sym.get("desc") else ""
                    tag = self._assignment_tag(sym["name"])
                    lines.append(f"  • {sym['name']} ({sym['type']}){conn_str}{context_str}{desc_str}{tag}")

                if len(classes_with_context) > class_detail_cap:
                    rest = classes_with_context[class_detail_cap:]
                    rest_names = [f"{c['name']}{self._assignment_tag(c['name'])}" for c in rest]
                    lines.append(f"  ALSO ({len(rest)} more classes): {', '.join(rest_names)}")
                lines.append("")

            if orphan_classes:
                # TOKEN SAVING: Summarize orphan classes in compact format
                lines.append(f"STANDALONE CLASSES ({len(orphan_classes)} - no inheritance/factories):")
                # Show top by connections, then just list names
                top_orphans = orphan_classes[:orphan_cap]
                for sym in top_orphans:
                    conn_str = f" [{sym['connections']} refs]" if sym.get("connections") else ""
                    desc_str = f" — {sym['desc']}" if sym.get("desc") else ""
                    tag = self._assignment_tag(sym["name"])
                    lines.append(f"  • {sym['name']}{conn_str}{desc_str}{tag}")
                if len(orphan_classes) > orphan_cap:
                    rest = orphan_classes[orphan_cap:]
                    rest_names = [f"{c['name']}{self._assignment_tag(c['name'])}" for c in rest]
                    lines.append(f"  ALSO ({len(rest)} more): {', '.join(rest_names)}")
                lines.append("")

        # ALL enums/constants with usage context (usually small, keep full)
        all_enums = analysis.get("all_enums", [])
        if all_enums:
            lines.append("ENUMS & CONSTANTS (with usage):")
            for enum in all_enums[:enum_cap]:
                used_by_str = ""
                if enum.get("used_by"):
                    used_by_str = f" → used by {', '.join(enum['used_by'][:3])}"
                desc_str = f" — {enum['desc']}" if enum.get("desc") else ""
                tag = self._assignment_tag(enum["name"])
                lines.append(f"  • {enum['name']} ({enum['type']}){used_by_str}{desc_str}{tag}")
            if len(all_enums) > enum_cap:
                rest = all_enums[enum_cap:]
                rest_names = [f"{e['name']}{self._assignment_tag(e['name'])}" for e in rest]
                lines.append(f"  ALSO ({len(rest)} more): {', '.join(rest_names)}")
            lines.append("")

        # TYPE ALIASES with resolution info
        all_type_aliases = analysis.get("all_type_aliases", [])
        if all_type_aliases:
            lines.append(f"TYPE ALIASES ({len(all_type_aliases)}):")
            for ta in all_type_aliases[:ta_cap]:
                parts = []
                if ta.get("alias_of"):
                    parts.append(f"= {', '.join(ta['alias_of'][:2])}")
                if ta.get("used_by"):
                    parts.append(f"used by {', '.join(ta['used_by'][:3])}")
                context_str = f" → {'; '.join(parts)}" if parts else ""
                desc_str = f" — {ta['desc']}" if ta.get("desc") else ""
                tag = self._assignment_tag(ta["name"])
                lines.append(f"  • {ta['name']}{context_str}{desc_str}{tag}")
            if len(all_type_aliases) > ta_cap:
                rest = all_type_aliases[ta_cap:]
                rest_names = [f"{t['name']}{self._assignment_tag(t['name'])}" for t in rest]
                lines.append(f"  ALSO ({len(rest)} more): {', '.join(rest_names)}")
            lines.append("")

        # MACROS — architecturally significant preprocessor definitions (C/C++)
        all_macros = analysis.get("all_macros", [])
        if all_macros:
            # Sort by connections (importance)
            all_macros.sort(key=lambda x: -x.get("connections", 0))
            lines.append(f"MACROS ({len(all_macros)}):")
            for macro in all_macros[:macro_cap]:
                context_parts = []
                if macro.get("references"):
                    context_parts.append(f"references {', '.join(macro['references'][:3])}")
                if macro.get("used_by"):
                    context_parts.append(f"used by {', '.join(macro['used_by'][:3])}")
                context_str = f" → {'; '.join(context_parts)}" if context_parts else ""
                conn_str = f" [{macro['connections']} refs]" if macro.get("connections") else ""
                desc_str = f" — {macro['desc']}" if macro.get("desc") else ""
                tag = self._assignment_tag(macro["name"])
                lines.append(f"  • {macro['name']}{conn_str}{context_str}{desc_str}{tag}")
            if len(all_macros) > macro_cap:
                rest = all_macros[macro_cap:]
                rest_names = [f"{m['name']}{self._assignment_tag(m['name'])}" for m in rest]
                lines.append(f"  ALSO ({len(rest)} more): {', '.join(rest_names)}")
            lines.append("")

        # Key functions with CREATES info - prioritize those that create objects
        all_functions = analysis.get("all_functions", [])
        if all_functions:
            # Prioritize functions that CREATE things (factory methods are doc-worthy)
            creators = [f for f in all_functions if f.get("creates")]
            # Functions with type info are also valuable (typed signatures)
            typed_funcs = [f for f in all_functions if f.get("uses_types") and not f.get("creates")]
            non_creators = [f for f in all_functions if not f.get("creates") and not f.get("uses_types")]

            if creators:
                lines.append(f"FACTORY FUNCTIONS ({len(creators)} - create objects):")
                for func in creators[:func_cap]:
                    conn_str = f" [{func['connections']} refs]" if func.get("connections") else ""
                    creates_str = f" → creates {', '.join(func['creates'][:3])}"
                    types_str = f"; uses {', '.join(func['uses_types'][:3])}" if func.get("uses_types") else ""
                    desc_str = f" — {func['desc']}" if func.get("desc") else ""
                    tag = self._assignment_tag(func["name"])
                    lines.append(f"  • {func['name']}(){conn_str}{creates_str}{types_str}{desc_str}{tag}")
                if len(creators) > func_cap:
                    rest = creators[func_cap:]
                    rest_names = [f"{f['name']}{self._assignment_tag(f['name'])}" for f in rest]
                    lines.append(f"  ALSO ({len(rest)} more): {', '.join(rest_names)}")
                lines.append("")

            if typed_funcs:
                typed_cap = func_cap if compact else 10
                lines.append(f"TYPED FUNCTIONS ({len(typed_funcs)} - with type signatures):")
                for func in typed_funcs[:typed_cap]:
                    conn_str = f" [{func['connections']} refs]" if func.get("connections") else ""
                    types_str = f" → uses {', '.join(func['uses_types'][:4])}"
                    desc_str = f" — {func['desc']}" if func.get("desc") else ""
                    tag = self._assignment_tag(func["name"])
                    lines.append(f"  • {func['name']}(){conn_str}{types_str}{desc_str}{tag}")
                if len(typed_funcs) > typed_cap:
                    rest = typed_funcs[typed_cap:]
                    rest_names = [f"{f['name']}{self._assignment_tag(f['name'])}" for f in rest]
                    lines.append(f"  ALSO ({len(rest)} more): {', '.join(rest_names)}")
                lines.append("")

            if non_creators:
                other_cap = func_cap if compact else 10
                lines.append(f"OTHER KEY FUNCTIONS ({len(non_creators)} total, top {other_cap}):")
                for func in non_creators[:other_cap]:
                    conn_str = f" [{func['connections']} refs]" if func.get("connections") else ""
                    desc_str = f" — {func['desc']}" if func.get("desc") else ""
                    tag = self._assignment_tag(func["name"])
                    lines.append(f"  • {func['name']}(){conn_str}{desc_str}{tag}")
                if len(non_creators) > other_cap:
                    rest = non_creators[other_cap:]
                    rest_names = [f"{f['name']}{self._assignment_tag(f['name'])}" for f in rest]
                    lines.append(f"  ALSO ({len(rest)} more): {', '.join(rest_names)}")
                lines.append("")

        # Grouping suggestions
        suggestions = analysis.get("grouping_suggestions", [])
        if suggestions:
            lines.append("SUGGESTED GROUPINGS (for page creation):")
            for i, sug in enumerate(suggestions[:5], 1):
                related_str = ", ".join(sug["related"][:4])
                if len(sug["related"]) > 4:
                    related_str += f" +{len(sug['related']) - 4} more"
                lines.append(f"  {i}. {sug['main']} + [{related_str}]")
                lines.append(f"     Reason: {sug['rationale']}")
            lines.append("")

        # Symbol breakdown by type (concise) — exclude methods/constructors
        # to avoid suggesting non-architectural symbols for page assignment
        by_type = analysis.get("symbols_by_type", {})
        _non_arch_display = {"method", "constructor"}
        if by_type:
            type_summary = ", ".join(
                f"{stype}: {count}"
                for stype, count in sorted(by_type.items(), key=lambda x: -x[1])[:5]
                if stype not in _non_arch_display
            )
            lines.append(f"Symbol breakdown: {type_summary}")
            lines.append("")

        # Documentation files - for target_docs field
        all_docs = analysis.get("all_docs", [])

        # Feature flag dispatch: AUTO_TARGET_DOCS changes how we display doc info
        if AUTO_TARGET_DOCS:
            # Docs auto-retrieved during generation — schema alignment:
            # - target_symbols: primary retrieval mechanism (graph-based)
            # - retrieval_query: fallback for docs not linked to symbols
            # - target_docs: optional, only for specific known doc paths
            lines.append("📄 DOCUMENTATION RETRIEVAL MODE: AUTO")
            lines.append("   Docs will be auto-retrieved based on your target_symbols during generation.")
            lines.append("   → target_symbols is PRIMARY: use exact symbol names from this analysis.")
            lines.append("   → retrieval_query is FALLBACK: used only when target_symbols miss relevant docs.")
            lines.append("   → target_docs is OPTIONAL: specify only if you know exact doc file paths.")
            if all_docs:
                lines.append(f"   ({len(all_docs)} doc files available in this module)")
            lines.append("")
        elif all_docs:
            lines.append(f"📄 DOCUMENTATION FILES ({len(all_docs)} found - SELECT for target_docs):")
            lines.append("   ⚠️ You MUST include relevant docs from this list in your define_page target_docs!")
            lines.append("")
            # Group by type for clarity
            docs_by_type = {}
            for doc in all_docs:
                dtype = doc.get("type", "unknown")
                if dtype not in docs_by_type:
                    docs_by_type[dtype] = []
                docs_by_type[dtype].append(doc)

            for dtype, docs in sorted(docs_by_type.items()):
                lines.append(f"  {dtype} ({len(docs)}):")
                for doc in docs[:5]:
                    lines.append(f"    • {doc['path']}")
                if len(docs) > 5:
                    lines.append(f"    ... and {len(docs) - 5} more")
            lines.append("")
            lines.append("→ ACTION: Copy relevant paths to target_docs field in define_page().")
            lines.append("")
        else:
            lines.append("📄 No documentation files found in this module.")
            lines.append("")

        # Complexity and page recommendations
        class_count = analysis.get("class_count", 0)
        func_count = analysis.get("function_count", 0)

        if class_count >= 10 or func_count >= 50 or total >= 100:
            complexity = "HIGH"
            pages = min(5, max(2, class_count // 3 + 1))
            lines.append(f"COMPLEXITY: {complexity} - Suggest {pages} pages")
            lines.append("NAME YOUR PAGES after the key classes/functions above, not directory names.")
            lines.append("Use the SUGGESTED GROUPINGS above to combine related symbols.")
        elif class_count >= 4 or func_count >= 20 or total >= 30:
            complexity = "MEDIUM"
            lines.append(f"COMPLEXITY: {complexity} - Suggest 1-2 pages")
            lines.append("Use the class names above for page naming.")
        else:
            complexity = "LOW"
            lines.append(f"COMPLEXITY: {complexity} - Can group with similar modules")
            lines.append("Consider combining enums/constants with their main class.")

        return "\n".join(lines)

    def _handle_search_symbols(
        self,
        query: str,
        symbol_type: str | None = None,
        path_prefix: str | None = None,
    ) -> str:
        """Handle search_symbols tool call — BM25-ranked FTS5 concept search.

        Unlike search_graph (O(N) substring match), this uses the FTS5 index
        for BM25-ranked results with keyword extraction and stop word removal.
        Falls back to search_graph if FTS5 is unavailable.
        """
        fts = self.graph_text_index
        if fts is None or not fts.is_open:
            # Graceful fallback to substring search
            logger.info("[STRUCTURE_PLANNER][TOOL] search_symbols: FTS5 unavailable, falling back to search_graph")
            return self._handle_search_graph(query, symbol_type, include_docstrings=True)

        query = query.strip()
        if not query or len(query) < 2:
            return "ERROR: query must be at least 2 characters"

        # Build type filter
        sym_types = None
        # Exclude methods/constructors from results by default — they should
        # not be assigned to pages. Only architectural symbols are page-worthy.
        _method_types = frozenset({"method", "constructor"})
        if symbol_type:
            sym_types = frozenset({symbol_type.lower()})

        logger.info(
            f"[STRUCTURE_PLANNER][TOOL] search_symbols: query={query!r}, type={symbol_type}, path={path_prefix}"
        )

        try:
            docs = fts.search_symbols(
                query,
                k=30,
                symbol_types=sym_types,
                exclude_types=DOC_SYMBOL_TYPES | _method_types,
                path_prefix=path_prefix,
            )
        except Exception as e:
            logger.warning(f"[STRUCTURE_PLANNER][TOOL] search_symbols error: {e}")
            return f"ERROR: Search failed: {e}"

        logger.info(f"[STRUCTURE_PLANNER][TOOL] search_symbols: {len(docs)} results for {query!r}")

        return self._format_search_symbols_results(
            query,
            docs,
            symbol_type,
            path_prefix,
        )

    def _format_search_symbols_results(
        self,
        query: str,
        docs: list,
        symbol_type: str | None,
        path_prefix: str | None,
    ) -> str:
        """Format search_symbols FTS5 results for LLM consumption."""
        filter_parts = []
        if symbol_type:
            filter_parts.append(f"type={symbol_type}")
        if path_prefix:
            filter_parts.append(f"path={path_prefix}")
        filter_str = f" ({', '.join(filter_parts)})" if filter_parts else ""

        lines = [f"=== Symbol Search: '{query}'{filter_str} [BM25-ranked] ===", ""]

        if not docs:
            lines.append("No symbols found matching this query.")
            lines.append("")
            lines.append("TIPS:")
            lines.append("  • Try different keywords (e.g., 'auth' instead of 'authentication')")
            lines.append("  • Try without type filter to see all matching symbols")
            lines.append("  • Use search_graph for exact substring matching")
            return "\n".join(lines)

        lines.append(f"Found {len(docs)} matching symbols (ranked by relevance):")
        lines.append("")

        # Group by type
        by_type: dict[str, list] = {}
        for doc in docs:
            t = doc.metadata.get("symbol_type", "unknown")
            if t not in by_type:
                by_type[t] = []
            by_type[t].append(doc)

        # Show in architectural order (methods excluded — not assignable to pages)
        type_order = ["class", "interface", "struct", "enum", "trait", "function", "constant", "type_alias"]
        _PLURAL = {
            "class": "CLASSES",
            "interface": "INTERFACES",
            "struct": "STRUCTS",
            "enum": "ENUMS",
            "trait": "TRAITS",
            "function": "FUNCTIONS",
            "constant": "CONSTANTS",
            "type_alias": "TYPE_ALIASES",
        }
        shown = set()
        for stype in type_order:
            if stype in by_type:
                shown.add(stype)
                syms = by_type[stype]
                label = _PLURAL.get(stype, f"{stype.upper()}S")
                lines.append(f"{label} ({len(syms)}):")
                for doc in syms[:10]:
                    name = doc.metadata.get("symbol_name", "?")
                    path = doc.metadata.get("rel_path", "")
                    score = doc.metadata.get("search_score", 0)
                    score_str = f" [score={abs(score):.1f}]" if score else ""
                    path_short = "/".join(path.split("/")[-2:]) if "/" in path else path
                    lines.append(f"  • {name}{score_str} in {path_short}")
                if len(syms) > 10:
                    lines.append(f"  ... and {len(syms) - 10} more")
                lines.append("")

        # Show remaining types
        for stype, syms in by_type.items():
            if stype in shown:
                continue
            lines.append(f"{stype.upper()}S ({len(syms)}):")
            for doc in syms[:5]:
                name = doc.metadata.get("symbol_name", "?")
                path = doc.metadata.get("rel_path", "")
                path_short = "/".join(path.split("/")[-2:]) if "/" in path else path
                lines.append(f"  • {name} in {path_short}")
            if len(syms) > 5:
                lines.append(f"  ... and {len(syms) - 5} more")
            lines.append("")

        lines.append("Use these symbol names in target_symbols when defining pages.")
        return "\n".join(lines)

    def _handle_search_graph(
        self, pattern: str, symbol_type: str | None = None, include_docstrings: bool = False
    ) -> str:
        """Handle search_graph tool call - find symbols by name pattern.

        Deterministic search over graph node attributes.
        Searches symbol names by default, optionally includes docstrings.

        By default, only returns architecturally significant symbols (classes,
        functions, constants, etc.) — methods are excluded from results unless
        explicitly requested via symbol_type='method'.

        Args:
            pattern: Substring to search for (case-insensitive)
            symbol_type: Optional filter by type (class, function, method)
            include_docstrings: Also search in docstrings (fuller text search)
        """
        if not self.code_graph:
            return "Graph not available. Use filesystem analysis instead."

        pattern = pattern.strip()
        if not pattern or len(pattern) < 2:
            return "ERROR: pattern must be at least 2 characters"

        pattern_lower = pattern.lower()
        symbol_type_filter = symbol_type.lower() if symbol_type else None

        # When no explicit type filter, exclude methods from results.
        # Methods are part of their parent class and should not be assigned
        # to pages directly. They are shown transitively via class relationships.
        exclude_methods = symbol_type_filter is None

        search_mode = "name+docstring" if include_docstrings else "name"
        logger.info(f"[STRUCTURE_PLANNER][TOOL] search_graph: Searching for '{pattern}' (mode={search_mode})")

        matches = []
        try:
            # SPEC-1: Use GraphQueryService for FTS5-backed search (O(log N))
            if self.query_service and (self.graph_text_index and self.graph_text_index.is_open):
                sym_types_filter = frozenset({symbol_type_filter}) if symbol_type_filter else None
                exclude = frozenset({"method", "constructor"}) if exclude_methods and not symbol_type_filter else None

                results = self.query_service.search(
                    query=pattern,
                    k=50,
                    symbol_types=sym_types_filter,
                    exclude_types=exclude,
                )
                for r in results:
                    # Extract brief docstring from authoritative graph node
                    _gnd = self.code_graph.nodes.get(r.node_id, {}) if r.node_id else {}
                    _ds = _get_symbol_brief(_gnd)
                    m = {
                        "name": r.symbol_name,
                        "type": r.symbol_type,
                        "path": r.rel_path or r.file_path,
                        "connections": r.connections,
                        "match_type": "fts5",
                    }
                    if _ds:
                        m["desc"] = _ds
                    matches.append(m)
                logger.info(
                    f"[STRUCTURE_PLANNER][TOOL] search_graph: FTS5 returned {len(matches)} matches for '{pattern}'"
                )
            else:
                # Legacy fallback: O(N) brute-force scan
                for node_id, node_data in self.code_graph.nodes(data=True):
                    symbol_name = node_data.get("symbol_name", "") or node_data.get("name", "")
                    if not symbol_name:
                        continue

                    # Check symbol name (always)
                    name_match = pattern_lower in symbol_name.lower()

                    # Check docstring if requested
                    docstring_match = False
                    if include_docstrings:
                        docstring = _extract_node_docstring(node_data)
                        docstring_match = pattern_lower in docstring.lower()

                    if not name_match and not docstring_match:
                        continue

                    stype = (
                        node_data.get("symbol_type") or node_data.get("type") or node_data.get("node_type", "unknown")
                    ).lower()

                    # Filter by type if specified
                    if symbol_type_filter and stype != symbol_type_filter:
                        continue

                    # Exclude methods unless explicitly requested
                    if exclude_methods and stype in ("method", "constructor"):
                        continue

                    rel_path = node_data.get("rel_path") or node_data.get("file_path", "")

                    connections = self.code_graph.in_degree(node_id) + self.code_graph.out_degree(node_id)

                    _ds = _get_symbol_brief(node_data)
                    m = {
                        "name": symbol_name,
                        "type": stype,
                        "path": rel_path,
                        "connections": connections,
                        "match_type": "name" if name_match else "docstring",
                    }
                    if _ds:
                        m["desc"] = _ds
                    matches.append(m)
        except Exception as e:
            logger.warning(f"[STRUCTURE_PLANNER][TOOL] search_graph error: {e}")
            return f"ERROR: Search failed: {e}"

        # Sort by connections (importance)
        matches.sort(key=lambda x: -x["connections"])

        logger.info(f"[STRUCTURE_PLANNER][TOOL] search_graph: Found {len(matches)} matches for '{pattern}'")

        return self._format_search_results(pattern, matches, symbol_type_filter, include_docstrings)

    def _format_search_results(
        self, pattern: str, matches: list[dict], symbol_type: str | None, include_docstrings: bool = False
    ) -> str:
        """Format search results for LLM consumption."""
        filter_str = f" (type={symbol_type})" if symbol_type else ""
        mode_str = " [name+docstring]" if include_docstrings else ""
        lines = [f"=== Search Results for '{pattern}'{filter_str}{mode_str} ===", ""]

        if not matches:
            lines.append("No symbols found matching this pattern.")
            lines.append("")
            lines.append("TIPS:")
            lines.append("  • Try a shorter pattern (e.g., 'Client' instead of 'GitHubClient')")
            lines.append("  • Try include_docstrings=true to search in documentation")
            lines.append("  • Try without type filter")
            return "\n".join(lines)

        lines.append(f"Found {len(matches)} matching symbols:")
        lines.append("")

        # Group by type for better organization
        by_type = {}
        for m in matches:
            t = m["type"]
            if t not in by_type:
                by_type[t] = []
            by_type[t].append(m)

        # Show classes first (most architecturally relevant)
        # Methods excluded from default output — they're part of parent classes
        type_order = ["class", "interface", "struct", "function", "constant", "type_alias", "enum", "trait"]
        for stype in type_order:
            if stype in by_type:
                syms = by_type.pop(stype)
                lines.append(f"{stype.upper()}ES ({len(syms)}):")
                for sym in syms[:10]:
                    conn_str = f" [{sym['connections']} refs]" if sym.get("connections") else ""
                    path_short = sym["path"].split("/")[-2] if "/" in sym["path"] else sym["path"]
                    match_indicator = " (docstring match)" if sym.get("match_type") == "docstring" else ""
                    desc_str = f" — {sym['desc']}" if sym.get("desc") else ""
                    lines.append(f"  • {sym['name']}{conn_str} in {path_short}{match_indicator}{desc_str}")
                if len(syms) > 10:
                    lines.append(f"  ... and {len(syms) - 10} more")
                lines.append("")

        # Any remaining types
        for stype, syms in by_type.items():
            lines.append(f"{stype.upper()} ({len(syms)}):")
            for sym in syms[:5]:
                desc_str = f" — {sym['desc']}" if sym.get("desc") else ""
                lines.append(
                    f"  • {sym['name']} in {sym['path'].split('/')[-2] if '/' in sym['path'] else sym['path']}{desc_str}"
                )
            if len(syms) > 5:
                lines.append(f"  ... and {len(syms) - 5} more")
            lines.append("")

        lines.append("USE THESE NAMES for page naming instead of directory names.")

        return "\n".join(lines)

    def get_tools(self) -> list:
        """Create LangChain tools bound to this collector.

        Tools are organized in suggested order of use:
        1. explore_tree - Deep exploration to understand repository structure
        2. analyze_module - Detailed analysis of specific modules
        3. query_graph - Query code graph for symbol data by path (if available)
        4. search_graph - Search for symbols by name pattern (e.g., "Agent", "Client")
        5. think - Strategic reflection before defining structure
        6. set_wiki_metadata - Set wiki title and overview
        7. define_section - Define documentation sections
        8. define_page - Define pages within sections
        """

        # ===== EXPLORATION TOOLS =====

        @tool
        def explore_tree() -> str:
            """Explore the repository deeply to find the actual code structure.

            This tool automatically:
            - Finds the real code root (e.g., src/mypackage/ instead of just src/)
            - Identifies all code modules and their complexity
            - Suggests how many pages each module needs

            Call this FIRST before defining any sections or pages.
            The results will guide your documentation structure.

            Returns a structured analysis with module complexity ratings.
            """
            return self._handle_explore_tree()

        @tool
        def analyze_module(module_path: str) -> str:
            """Analyze a specific module directory in detail.

            Use this when explore_tree shows a HIGH COMPLEXITY module
            to understand its internal structure better.

            Args:
                module_path: Relative path to the module (e.g., 'src/services/orders')

            Returns detailed breakdown of code files and subdirectories
            with recommendations for page structure.
            """
            return self._handle_analyze_module(module_path)

        @tool
        def think(analysis: str) -> str:
            """Strategic reflection for planning documentation structure.

            Use this to:
            - Synthesize what you learned from explore_tree
            - Decide how to group modules into sections
            - Plan page structure for complex modules
            - Ensure you're creating SPECIFIC, non-generic pages

            Call this AFTER explore_tree but BEFORE defining sections.

            Args:
                analysis: Your full reasoning about the documentation structure
            """
            return self._handle_think(analysis)

        @tool
        def query_graph(path_prefix: str, text_filter: str = "") -> str:
            """Query the code graph to understand symbols in a module.

            This provides accurate complexity data based on actual parsed code:
            - Number of classes, functions, methods
            - Key class names for documentation
            - Complexity assessment based on symbol density

            Use this for HIGH COMPLEXITY modules from explore_tree to get
            precise data about what needs to be documented.

            Args:
                path_prefix: Path prefix to query (e.g., 'src/services/orders')
                text_filter: Optional keyword filter to narrow results by concept
                             (e.g., "authentication", "cache", "error handling").
                             When provided, only symbols matching the keywords are
                             returned — with full relationship context.
                             Leave empty to get ALL symbols under path_prefix.

            Returns symbol counts and key classes for documentation planning.
            NOTE: Only available if code graph was indexed. Returns helpful
            message if graph is not available.
            """
            return self._handle_query_graph(path_prefix, text_filter=text_filter if text_filter else None)

        @tool
        def search_graph(pattern: str, symbol_type: str = "", include_docstrings: bool = False) -> str:
            """Search the code graph for symbols by name pattern.

            Use this to discover symbols across the codebase by keyword.
            For example, search for "Client" to find all client classes,
            or "Agent" to find agent-related code.

            This is DETERMINISTIC - same pattern always returns same results.

            Args:
                pattern: Substring to search for (case-insensitive, min 2 chars)
                         Examples: "Agent", "Client", "Service", "Handler"
                symbol_type: Optional filter by type: "class", "function", "method"
                             Leave empty to search all types.
                include_docstrings: Set to true to also search in docstrings.
                                   This finds symbols where the pattern appears in
                                   the documentation, not just the name.

            Returns matching symbols sorted by importance (connection count).
            Use these symbol names for page naming instead of directory names.

            NOTE: Only available if code graph was indexed.
            """
            return self._handle_search_graph(pattern, symbol_type if symbol_type else None, include_docstrings)

        @tool
        def search_symbols(query: str, symbol_type: str = "", path_prefix: str = "") -> str:
            """Search for symbols by concept using BM25-ranked full-text search.

            Unlike search_graph (exact substring match), this tool uses
            keyword extraction and relevance ranking to find symbols related
            to a CONCEPT.  Use this for natural-language queries like
            "authentication and token handling" or "error management".

            Examples:
              • search_symbols("authentication token")  → finds auth-related classes
              • search_symbols("error handling", symbol_type="class")  → error classes
              • search_symbols("database connection", path_prefix="src/db")  → scoped

            Args:
                query: Keywords or natural-language description of what to find.
                       Stop words are automatically removed.
                symbol_type: Optional filter: "class", "function", "method", "enum", etc.
                path_prefix: Optional directory scope (e.g., "src/services").

            Returns BM25-ranked symbols with relevance scores.
            Use these symbol names in target_symbols when defining pages.

            NOTE: Falls back to search_graph if full-text index is unavailable.
            """
            return self._handle_search_symbols(
                query,
                symbol_type if symbol_type else None,
                path_prefix if path_prefix else None,
            )

        # ===== OUTPUT TOOLS =====

        @tool(args_schema=WikiMetadata)
        def set_wiki_metadata(wiki_title: str, overview: str) -> str:
            """Set the wiki title and overview. Call this first before defining sections."""
            return self._handle_set_metadata(wiki_title, overview)

        @tool(args_schema=SectionDefinition)
        def define_section(section_name: str, section_order: int, description: str, rationale: str) -> str:
            """Define a documentation section. Call this before defining pages in the section."""
            return self._handle_define_section(section_name, section_order, description, rationale)

        @tool(args_schema=PageDefinition)
        def define_page(
            section_name: str,
            page_name: str,
            page_order: int,
            description: str,
            content_focus: str,
            rationale: str,
            target_symbols: list[str],
            target_docs: list[str],
            target_folders: list[str],
            key_files: list[str],
            retrieval_query: str = "",
        ) -> str:
            """Define a documentation page within a section.

            IMPORTANT: target_symbols MUST contain actual class/function names from query_graph.
            These enable precise graph-based code retrieval during page generation.
            Use target_docs for documentation files shown by query_graph.
            """
            return self._handle_define_page(
                section_name,
                page_name,
                page_order,
                description,
                content_focus,
                rationale,
                target_symbols,
                target_docs,
                target_folders,
                key_files,
                retrieval_query,
            )

        @tool(args_schema=BatchPageDefinition)
        def batch_define_pages(pages: list[dict[str, Any]]) -> str:
            """Define multiple documentation pages at once for EFFICIENCY.

            RECOMMENDED for large repositories (>2000 files) to speed up planning.
            Instead of calling define_page 30+ times with thinking between each,
            use this to define 5-10 pages per section in a single call.

            Each page in the list needs:
            - section_name: The section this page belongs to
            - page_name: CAPABILITY-BASED title (what users can do)
            - description: Brief description of the page
            - target_symbols: Class/function names from query_graph
            - target_folders: Folders this page covers

            Example:
            batch_define_pages(pages=[
                {"section_name": "Core Engine", "page_name": "Message Processing Pipeline",
                 "description": "How messages flow through the system",
                 "target_symbols": ["MessageProcessor", "Pipeline"], "target_folders": ["src/engine"]},
                {"section_name": "Core Engine", "page_name": "State Management",
                 "description": "Managing runtime state",
                 "target_symbols": ["StateManager", "Store"], "target_folders": ["src/state"]},
                ...
            ])
            """
            return self._handle_batch_define_pages(pages)

        return [
            explore_tree,
            analyze_module,
            think,
            query_graph,
            search_graph,
            search_symbols,
            set_wiki_metadata,
            define_section,
            define_page,
            batch_define_pages,
        ]

    def assemble(self) -> dict[str, Any]:
        """
        Assemble the final WikiStructureSpec from collected definitions.

        Returns:
            Dict ready for WikiStructureSpec.model_validate()
        """
        # Default metadata if not set
        if not self.metadata:
            self.metadata = {"wiki_title": "Documentation", "overview": "Repository documentation"}

        # Group pages by section
        for page in self.pages:
            section_name = page["section_name"]
            if section_name in self.sections:
                # Remove section_name from page dict (it's implied by parent)
                page_copy = {k: v for k, v in page.items() if k != "section_name"}
                self.sections[section_name]["pages"].append(page_copy)
            else:
                # Create section for orphan page
                self.sections[section_name] = {
                    "section_name": section_name,
                    "section_order": 99,
                    "description": f"Section for {section_name}",
                    "rationale": "Auto-created for orphan pages",
                    "pages": [{k: v for k, v in page.items() if k != "section_name"}],
                }

        # Sort sections by order
        sorted_sections = sorted(self.sections.values(), key=lambda s: s.get("section_order", 99))

        # Sort pages within each section
        for section in sorted_sections:
            section["pages"] = sorted(section["pages"], key=lambda p: p.get("page_order", 99))

        return {
            "wiki_title": self.metadata["wiki_title"],
            "overview": self.metadata["overview"],
            "sections": sorted_sections,
            "total_pages": len(self.pages),
        }

    @property
    def stats(self) -> dict[str, Any]:
        """Get current collection stats including coverage.

        Coverage is calculated as: (covered ∩ discovered) / discovered
        This ensures coverage never exceeds 100% and only counts real directories.
        """
        uncovered = self._get_uncovered_dirs()
        covered_count, discovered_count, coverage_pct = self._get_coverage_stats()

        return {
            "sections": len(self.sections),
            "pages": len(self.pages),
            "budget": self.page_budget,
            "remaining": max(0, self.page_budget - len(self.pages)),
            "over_budget": max(0, len(self.pages) - self.page_budget),
            "discovered_dirs": discovered_count,
            "covered_dirs": covered_count,
            "uncovered_dirs": uncovered,
            "coverage_pct": round(coverage_pct, 1),
        }
