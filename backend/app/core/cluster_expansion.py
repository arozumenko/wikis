"""
Cluster-Bounded Content Expansion — Phase 6 of the Unified Graph & Clustering plan.

Provides ``expand_for_page()`` which retrieves and expands page content
directly from :class:`UnifiedWikiDB`, using cluster boundaries to prevent
context pollution from unrelated symbols.

When ``WIKIS_STRUCTURE_PLANNER=cluster``, the wiki agent calls this
module instead of the legacy NX-graph based expansion
(``_get_docs_by_target_symbols`` + ``expand_smart``).

Key advantages:
- Expansion is bounded to the page's macro-cluster (no cross-cluster leaks)
- No in-memory NX graph needed at page-generation time
- Documentation files naturally included (they're cluster members via Phase 2 semantic edges)
- Token-budget aware: symbols are added in priority order until the budget is exhausted
- Result is a flat list of LangChain ``Document`` objects ready for LLM context

Typical flow::

    with UnifiedWikiDB(db_path) as db:
        docs = expand_for_page(
            db=db,
            page_symbols=["AuthService", "LoginHandler", "SessionManager"],
            macro_id=3,
            token_budget=50_000,
        )
"""

from __future__ import annotations

import logging
import os
from typing import Any, Callable, Dict, List, Optional, Set, Tuple

from langchain_core.documents import Document

from .code_graph.shared_expansion import expand_symbol_smart
from .feature_flags import get_feature_flags
from .wiki_structure_planner.language_heuristics import (
    detect_dominant_language,
    get_language_hints,
    should_include_in_expansion,
    compute_augmentation_budget_fraction,
)

logger = logging.getLogger(__name__)


def _is_excluded_test_node(node: Dict[str, Any], exclude_tests: bool) -> bool:
    """Return True if *node* should be skipped because it's a test node."""
    return exclude_tests and bool(node.get("is_test"))


# Languages that support header/implementation split augmentation.
# Values must match what graph_builder.SUPPORTED_LANGUAGES stores in
# repo_nodes.language  (e.g. 'cpp', 'c', 'go', 'rust').
_CPP_LANGUAGES = frozenset({'cpp', 'c'})   # C++ & C headers/impls
_GO_LANGUAGES  = frozenset({'go'})
_RUST_LANGUAGES = frozenset({'rust'})
_AUGMENTABLE_LANGUAGES = _CPP_LANGUAGES | _GO_LANGUAGES | _RUST_LANGUAGES

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

# Default token budget per page (matches the CONTEXT_TOKEN_BUDGET in wiki agent)
DEFAULT_TOKEN_BUDGET = 50_000

# Max neighbours to fetch per symbol during 1-hop expansion
MAX_NEIGHBORS_PER_SYMBOL = 15

# Global cap on total expansion nodes for one page
MAX_EXPANSION_TOTAL = 200

# Architectural symbol types accepted during expansion
# (mirrors constants.EXPANSION_SYMBOL_TYPES)
EXPANSION_SYMBOL_TYPES = frozenset({
    'class', 'interface', 'struct', 'enum', 'trait',
    'function', 'constant', 'type_alias', 'macro',
    'module_doc', 'file_doc',
})

# Priority relationship types: edges traversed during expansion (sorted by
# architectural importance — P0 first, P2 last).
# Only outgoing edges are expanded.  Incoming edges add too much noise for
# page-level context (the caller doesn't need the callees' own callers).
_P0_REL_TYPES = frozenset({
    'inheritance', 'implementation', 'defines_body',
    'creates', 'instantiates',
})
_P1_REL_TYPES = frozenset({
    'composition', 'aggregation', 'alias_of', 'specializes',
})
_P2_REL_TYPES = frozenset({
    'calls', 'references',
})

# Average tokens per 1 char of source — rough estimate for budget tracking.
# Measured across several real repos: 1 token ≈ 3.5 chars for code.
_CHARS_PER_TOKEN = 3.5

# ---------------------------------------------------------------------------
# Framework reference discovery — for expansion orphans
# ---------------------------------------------------------------------------
# Symbols with few/no structural graph edges (event handlers, DI-injected
# classes, signal receivers) need FTS-based discovery to find code that
# references them via string literals or framework-specific dispatch.

# Max FTS-discovered reference docs per orphan symbol.
MAX_FRAMEWORK_REFS_PER_ORPHAN = 3

# Max structural (non-synthetic) edges for a symbol to qualify as an
# "expansion orphan" that should trigger FTS framework ref search.
ORPHAN_EDGE_THRESHOLD = 1

# Minimum symbol name length for FTS search (shorter names produce noise).
_MIN_FTS_NAME_LEN = 4


# ---------------------------------------------------------------------------
# Token estimation
# ---------------------------------------------------------------------------

def _estimate_tokens(text: str) -> int:
    """Quick token estimate without importing tiktoken."""
    if not text:
        return 0
    return max(1, int(len(text) / _CHARS_PER_TOKEN))


# ---------------------------------------------------------------------------
# Framework reference discovery helpers
# ---------------------------------------------------------------------------

def _count_structural_edges(db, node_id: str) -> int:
    """Count structural (AST-derived) edges for a node.

    Synthetic edges from Phase 2 orphan / doc resolution (``edge_class`` in
    ``directory``, ``lexical``, ``semantic``, ``doc``, ``bridge``) are
    excluded.  Only edges with ``edge_class='structural'`` count.

    Uses protocol methods ``get_edges_from`` / ``get_edges_to`` so it
    works with both SQLite and PostgreSQL backends.
    """
    outgoing = db.get_edges_from(node_id) or []
    incoming = db.get_edges_to(node_id) or []
    return sum(
        1 for e in outgoing if e.get("edge_class") == "structural"
    ) + sum(
        1 for e in incoming if e.get("edge_class") == "structural"
    )


def _collect_search_terms(
    db,
    orphan_id: str,
    orphan_node: Dict[str, Any],
) -> List[str]:
    """Build FTS search terms for an orphan symbol.

    For classes/interfaces with child ``defines`` edges, returns the
    child method/field names *in addition to* the class name itself.
    This catches framework-dispatch patterns where the **method** name
    appears at the call site, not the class name::

        # Orphan class — parser knows Event.defines→configuration_created
        @web.event('configuration_created')
        class Event:
            def configuration_created(self, ctx, event, payload): ...

        # Dispatch site — mentions the method name, not "Event"
        event_manager.emit("configuration_created", data)

    For non-class orphans (functions, constants, etc.) returns just the
    symbol name.
    """
    sym_name = (orphan_node.get("symbol_name") or "").strip()
    terms: List[str] = []
    if len(sym_name) >= _MIN_FTS_NAME_LEN:
        terms.append(sym_name)

    sym_type = (orphan_node.get("symbol_type") or "").lower()
    if sym_type in ("class", "interface", "struct"):
        try:
            edges = db.get_edges_from(orphan_id, rel_types=["defines"]) or []
            child_ids = [
                e["target_id"] for e in edges
                if e.get("edge_class") == "structural"
            ]
            if child_ids:
                children = db.get_nodes_by_ids(child_ids)
                for child in children:
                    child_name = (child.get("symbol_name") or "").strip()
                    if (
                        len(child_name) >= _MIN_FTS_NAME_LEN
                        and child_name not in terms
                    ):
                        terms.append(child_name)
        except Exception as exc:
            logger.debug(
                "[CLUSTER_EXPANSION] child term collection failed for "
                "'%s': %s", orphan_id, exc,
            )

    return terms


def _find_framework_references(
    db,
    orphan_nodes: Dict[str, Dict[str, Any]],
    seen_ids: Set[str],
    macro_id: Optional[int] = None,
    limit_per_orphan: int = MAX_FRAMEWORK_REFS_PER_ORPHAN,
) -> List[Tuple[str, Dict[str, Any], str]]:
    """Find code that references orphan symbols via FTS source_text search.

    For symbols with few structural graph edges (e.g. framework-dispatched
    event handlers, DI-injected classes, signal receivers), searches the
    full-text index for other nodes whose ``source_text`` mentions the
    symbol name.  This catches patterns invisible to the code graph::

        # Handler (orphan — no callers in the graph)
        @web.event('configuration_created')
        def configuration_created(self, context, event, payload):
            ...

        # Dispatch site — found via FTS
        event_manager.emit("configuration_created", data)

    For orphan **classes**, also searches for child method/field names (via
    ``defines`` edges) because the dispatch site typically references the
    method name, not the class name (e.g. ``emit("configuration_created")``
    rather than ``emit("Event")``).

    Searches **repo-wide first**, then narrows to the macro-cluster.
    Framework dispatch callers are most likely in a *different* cluster
    from the handler.

    Uses ``db.search_fts()`` (protocol-standard) so it works with both
    SQLite (FTS5) and PostgreSQL (tsvector) backends.

    Parameters
    ----------
    db : WikiStorageProtocol
    orphan_nodes : dict
        ``{node_id: node_dict}`` for symbols identified as expansion orphans.
    seen_ids : set
        Node IDs already included in the page expansion (excluded from results).
    macro_id : int, optional
        Macro-cluster scope — used as secondary search after repo-wide.
    limit_per_orphan : int
        Maximum reference nodes per orphan symbol.

    Returns
    -------
    list of (node_id, node_dict, description_str)
    """
    results: List[Tuple[str, Dict[str, Any], str]] = []
    found_ids: Set[str] = set()

    # Pre-collect all orphan node IDs so we can skip cross-refs between
    # orphans that share the same name (e.g. 30 × get_tools functions).
    all_orphan_ids = set(orphan_nodes.keys())

    for orphan_id, orphan_node in orphan_nodes.items():
        search_terms = _collect_search_terms(db, orphan_id, orphan_node)
        if not search_terms:
            continue

        # Search repo-wide first (None), then cluster-scoped.
        # Framework dispatch callers are most likely OUTSIDE the
        # orphan's own cluster.
        scope_ids = [None, macro_id] if macro_id is not None else [None]
        hits: List[Dict[str, Any]] = []

        for term in search_terms:
            if len(hits) >= limit_per_orphan:
                break
            term_lower = term.lower()

            for scope_cid in scope_ids:
                if len(hits) >= limit_per_orphan:
                    break
                try:
                    rows = db.search_fts(
                        query=term,
                        cluster_id=scope_cid,
                        limit=limit_per_orphan * 3,
                    )

                    for node in rows:
                        nid = node.get("node_id", "")
                        if (
                            nid == orphan_id
                            or nid in seen_ids
                            or nid in found_ids
                            or nid in all_orphan_ids
                        ):
                            continue

                        # Skip same-name peer definitions.  When
                        # searching for "get_tools" we want *callers*,
                        # not 30 other get_tools() definitions.
                        hit_name = (
                            node.get("symbol_name") or ""
                        ).lower()
                        if hit_name == term_lower:
                            continue

                        # Confirm actual substring presence in
                        # source_text.  FTS tokenization can produce
                        # false positives for short tokens.
                        src = (
                            node.get("source_text") or ""
                        ).lower()
                        if term_lower not in src:
                            continue

                        hits.append(node)
                        found_ids.add(nid)
                        if len(hits) >= limit_per_orphan:
                            break

                except Exception as exc:
                    logger.debug(
                        "[CLUSTER_EXPANSION] FTS framework ref failed "
                        "for '%s': %s", term, exc,
                    )
                    continue

        for hit in hits:
            hit_name = hit.get("symbol_name", "?")
            hit_path = hit.get("rel_path", "?")
            desc = (
                f"fts_framework_ref:"
                f"{orphan_node.get('symbol_name', '?')}"
                f"→{hit_name}@{hit_path}"
            )
            results.append((hit["node_id"], hit, desc))

    return results


# ---------------------------------------------------------------------------
# Node ➜ Document conversion
# ---------------------------------------------------------------------------

def _node_to_document(
    node: Dict[str, Any],
    *,
    is_initial: bool = True,
    expanded_from: Optional[str] = None,
) -> Document:
    """Convert a unified-DB node dict to a LangChain ``Document``."""
    page_content = node.get("source_text") or ""
    metadata = {
        "node_id": node.get("node_id", ""),
        "symbol_name": node.get("symbol_name", ""),
        "symbol_type": node.get("symbol_type", ""),
        "source": node.get("rel_path", ""),
        "rel_path": node.get("rel_path", ""),
        "file_name": node.get("file_name", ""),
        "language": node.get("language", ""),
        "start_line": node.get("start_line", 0),
        "end_line": node.get("end_line", 0),
        "chunk_type": node.get("chunk_type") or node.get("symbol_type", ""),
        "docstring": node.get("docstring", ""),
        "signature": node.get("signature", ""),
        "is_architectural": bool(node.get("is_architectural", 0)),
        "is_doc": bool(node.get("is_doc", 0)),
        "is_documentation": bool(node.get("is_doc", 0)),
        "macro_cluster": node.get("macro_cluster"),
        "micro_cluster": node.get("micro_cluster"),
        "is_initially_retrieved": is_initial,
    }
    if expanded_from:
        metadata["expanded_from"] = expanded_from
    return Document(page_content=page_content, metadata=metadata)


# ---------------------------------------------------------------------------
# Cross-file augmentation (C++ header↔impl, Go receivers, Rust impl blocks)
# ---------------------------------------------------------------------------

def _augment_document(db, doc: Document) -> int:
    """Augment a Document in-place with cross-file implementation bodies.

    For C++ declarations, finds ``defines_body`` predecessors (impl → decl)
    and stitches implementation source text into the document.

    For Go structs, finds cross-file receiver methods via ``defines`` edges.

    For Rust types, finds cross-file impl methods via ``defines`` edges.

    Returns the *additional* token cost from the augmentation (0 if none).
    """
    lang = (doc.metadata.get("language") or "").lower()
    if lang not in _AUGMENTABLE_LANGUAGES:
        return 0

    node_id = doc.metadata.get("node_id", "")
    sym_type = (doc.metadata.get("symbol_type") or "").lower()
    rel_path = doc.metadata.get("rel_path") or doc.metadata.get("source", "")
    if not node_id:
        return 0

    conn = db.conn
    original_len = len(doc.page_content)
    augmented_parts: List[str] = []

    if lang in _CPP_LANGUAGES:
        augmented_parts = _augment_cpp(conn, node_id, sym_type, rel_path)
    elif lang in _GO_LANGUAGES:
        augmented_parts = _augment_go_rust(conn, node_id, sym_type, rel_path)
    elif lang in _RUST_LANGUAGES:
        augmented_parts = _augment_go_rust(conn, node_id, sym_type, rel_path)

    if augmented_parts:
        extra = "\n\n".join(augmented_parts)
        doc.page_content = doc.page_content + "\n\n" + extra
        doc.metadata["is_augmented"] = True
        extra_tokens = _estimate_tokens(extra)
        logger.debug(
            "[CLUSTER_EXPANSION] Augmented %s (%s) with %d impl parts (+%d tokens)",
            node_id, sym_type, len(augmented_parts), extra_tokens,
        )
        return extra_tokens

    return 0


def _augment_cpp(
    conn, node_id: str, sym_type: str, decl_file: str,
) -> List[str]:
    """C++ augmentation: find defines_body implementations for declarations."""
    parts: List[str] = []

    if sym_type in ('function', 'method', 'constructor'):
        # Function/method: look for incoming defines_body (impl → decl)
        rows = conn.execute(
            "SELECT n.source_text, n.rel_path FROM repo_edges e "
            "JOIN repo_nodes n ON e.source_id = n.node_id "
            "WHERE e.target_id = ? AND e.rel_type = 'defines_body'",
            (node_id,),
        ).fetchall()
        for row in rows:
            impl_text = row[0] or ""
            impl_file = row[1] or ""
            if impl_text.strip() and impl_file != decl_file:
                parts.append(
                    f"/* Implementation from {impl_file} */\n{impl_text}"
                )

    elif sym_type in ('class', 'struct'):
        # Class/struct: find methods defined by this class, then their impls
        method_rows = conn.execute(
            "SELECT e.target_id FROM repo_edges e "
            "JOIN repo_nodes n ON e.target_id = n.node_id "
            "WHERE e.source_id = ? AND e.rel_type = 'defines' "
            "AND n.symbol_type IN ('method', 'constructor', 'function')",
            (node_id,),
        ).fetchall()

        impl_by_file: Dict[str, List[str]] = {}
        for method_row in method_rows:
            method_id = method_row[0]
            impl_rows = conn.execute(
                "SELECT n.source_text, n.rel_path FROM repo_edges e "
                "JOIN repo_nodes n ON e.source_id = n.node_id "
                "WHERE e.target_id = ? AND e.rel_type = 'defines_body'",
                (method_id,),
            ).fetchall()
            for row in impl_rows:
                impl_text = row[0] or ""
                impl_file = row[1] or ""
                if impl_text.strip() and impl_file != decl_file:
                    impl_by_file.setdefault(impl_file, []).append(impl_text)

        for impl_file, impls in sorted(impl_by_file.items()):
            header = (
                f"/* Implementations from {impl_file} "
                f"({len(impls)} method{'s' if len(impls) != 1 else ''}) */"
            )
            parts.append(header + "\n" + "\n\n".join(impls))

    return parts


def _augment_go_rust(
    conn, node_id: str, sym_type: str, type_file: str,
) -> List[str]:
    """Go/Rust augmentation: find cross-file receiver/impl methods."""
    if sym_type not in ('struct', 'class', 'enum', 'trait'):
        return []

    # Find methods defined by this type in other files
    rows = conn.execute(
        "SELECT n.source_text, n.rel_path, n.symbol_type FROM repo_edges e "
        "JOIN repo_nodes n ON e.target_id = n.node_id "
        "WHERE e.source_id = ? AND e.rel_type = 'defines' "
        "AND n.symbol_type IN ('method', 'function', 'constructor') "
        "AND n.rel_path != ?",
        (node_id, type_file),
    ).fetchall()

    methods_by_file: Dict[str, List[str]] = {}
    for row in rows:
        text = row[0] or ""
        mfile = row[1] or ""
        if text.strip() and mfile:
            methods_by_file.setdefault(mfile, []).append(text)

    parts: List[str] = []
    for mfile, mtexts in sorted(methods_by_file.items()):
        header = (
            f"// Methods from {mfile} "
            f"({len(mtexts)} method{'s' if len(mtexts) != 1 else ''})"
        )
        parts.append(header + "\n" + "\n\n".join(mtexts))

    return parts


# ---------------------------------------------------------------------------
# Core API
# ---------------------------------------------------------------------------

def expand_for_page(
    db,  # UnifiedWikiDB instance (avoid import cycle)
    page_symbols: List[str],
    macro_id: Optional[int] = None,
    micro_id: Optional[int] = None,
    cluster_node_ids: Optional[List[str]] = None,
    token_budget: int = DEFAULT_TOKEN_BUDGET,
    include_docs: bool = True,
) -> List[Document]:
    """Expand page symbols with cluster-bounded context from the unified DB.

    Parameters
    ----------
    db : UnifiedWikiDB
        Opened database (readonly or read-write).
    page_symbols : list[str]
        Symbol *names* (not node_ids) assigned to this page by the
        cluster planner.  Resolved via SQL lookup on ``symbol_name``.
    macro_id : int, optional
        Macro-cluster ID for boundary enforcement.  When set, expansion
        neighbours are restricted to this cluster.
    micro_id : int, optional
        Micro-cluster ID for tighter scoping.
    cluster_node_ids : list[str], optional
        Authoritative page membership from ``PageSpec.metadata``.
        When provided, expansion neighbours are restricted to this
        exact set of node IDs (page boundary), with fallback to
        macro_id for nodes not in the set.
    token_budget : int
        Maximum estimated tokens for the returned documents.
    include_docs : bool
        If True (default), documentation nodes in the same cluster are
        included automatically alongside code symbols.

    Returns
    -------
    list[Document]
        Flat list ordered by: initial symbols → P0 expansions → P1 → P2 → docs.
    """
    if not page_symbols:
        return []

    budget_remaining = token_budget
    seen_ids: Set[str] = set()
    result_docs: List[Document] = []

    # ── Step 1: Resolve symbol names to node_ids ─────────────────
    matched_nodes = _resolve_symbols(db, page_symbols, macro_id)

    # Drop test nodes when exclude_tests is enabled
    flags = get_feature_flags()
    if flags.exclude_tests:
        matched_nodes = {
            nid: n for nid, n in matched_nodes.items()
            if not n.get("is_test")
        }

    if not matched_nodes:
        logger.warning(
            "[CLUSTER_EXPANSION] No nodes resolved for symbols %s (macro=%s)",
            page_symbols[:5], macro_id,
        )
        return []

    logger.info(
        "[CLUSTER_EXPANSION] Resolved %d/%d symbols to %d nodes (macro=%s)",
        len([s for s in page_symbols if any(
            n["symbol_name"] == s for n in matched_nodes.values()
        )]),
        len(page_symbols),
        len(matched_nodes),
        macro_id,
    )

    # ── Step 2: Add initial symbols (highest priority) ───────────
    for node_id, node in matched_nodes.items():
        doc = _node_to_document(node, is_initial=True)
        cost = _estimate_tokens(doc.page_content)
        if cost > budget_remaining:
            logger.debug(
                "[CLUSTER_EXPANSION] Budget exhausted adding initial symbol %s "
                "(%d tokens remaining, %d needed)",
                node.get("symbol_name"), budget_remaining, cost,
            )
            break
        result_docs.append(doc)
        seen_ids.add(node_id)
        budget_remaining -= cost

    # ── Step 2.5: Augment initial symbols with cross-file impls ──
    # C++ header↔impl, Go receiver methods, Rust impl blocks.
    augmented_count = 0
    for doc in result_docs:
        extra = _augment_document(db, doc)
        if extra > 0:
            budget_remaining -= extra
            augmented_count += 1
    if augmented_count:
        logger.info(
            "[CLUSTER_EXPANSION] Augmented %d/%d initial docs "
            "(budget remaining: %d)",
            augmented_count, len(result_docs), budget_remaining,
        )

    # ── Step 2.75: Framework reference discovery for orphans ─────
    #    For symbols with few structural graph edges (event handlers,
    #    DI-injected classes, signal receivers, etc.), find other code
    #    that references them via string literals or framework dispatch.
    if budget_remaining > 0:
        orphan_nodes: Dict[str, Dict[str, Any]] = {}
        for node_id in list(seen_ids):
            if node_id in matched_nodes:
                edge_count = _count_structural_edges(db, node_id)
                if edge_count <= ORPHAN_EDGE_THRESHOLD:
                    orphan_nodes[node_id] = matched_nodes[node_id]

        if orphan_nodes:
            framework_refs = _find_framework_references(
                db, orphan_nodes, seen_ids, macro_id,
            )
            fts_ref_count = 0
            for nid, node, reason in framework_refs:
                if budget_remaining <= 0 or len(result_docs) >= MAX_EXPANSION_TOTAL:
                    break
                if nid in seen_ids:
                    continue
                doc = _node_to_document(node, is_initial=False, expanded_from=reason)
                cost = _estimate_tokens(doc.page_content)
                if cost > budget_remaining:
                    continue
                result_docs.append(doc)
                seen_ids.add(nid)
                budget_remaining -= cost
                fts_ref_count += 1

            if fts_ref_count:
                logger.info(
                    "[CLUSTER_EXPANSION] Framework ref discovery: "
                    "%d orphans → %d refs (budget remaining: %d)",
                    len(orphan_nodes), fts_ref_count, budget_remaining,
                )

    # ── Step 3: 1-hop expansion within cluster boundary ──────────
    #    When cluster_node_ids is provided, use it as the authoritative
    #    page boundary (Phase 2 fix).  Fall back to macro_id boundary.
    page_boundary_ids = set(cluster_node_ids) if cluster_node_ids else None

    flags = get_feature_flags()

    if flags.smart_expansion:
        # ── Phase 4: Per-symbol-type smart expansion ─────────────
        conn = db.conn

        # ── Phase 7: Language hints (optional) ───────────────────
        lang_hints = None
        if flags.language_hints:
            seed_ids_for_lang = list(matched_nodes.keys())[:50]
            dominant_lang = detect_dominant_language(conn, seed_ids_for_lang)
            if dominant_lang:
                lang_hints = get_language_hints(dominant_lang)
                logger.info(
                    "[CLUSTER_EXPANSION] Language hints: %s (dominant=%s)",
                    lang_hints.language, dominant_lang,
                )

        extra_rels = frozenset(lang_hints.extra_expansion_rels) if lang_hints else frozenset()

        for seed_id in list(matched_nodes.keys()):
            if budget_remaining <= 0 or len(result_docs) >= MAX_EXPANSION_TOTAL:
                break
            seed_node = matched_nodes[seed_id]
            seed_type = (seed_node.get("symbol_type") or "").lower()
            neighbors = expand_symbol_smart(
                conn, seed_id, seed_type, seen_ids,
                page_boundary_ids=page_boundary_ids,
                macro_id=macro_id,
                per_symbol_budget=MAX_NEIGHBORS_PER_SYMBOL,
                extra_rel_types=extra_rels,
            )
            for nid, node, reason in neighbors:
                if budget_remaining <= 0 or len(result_docs) >= MAX_EXPANSION_TOTAL:
                    break
                if nid in seen_ids:
                    continue

                # Phase 7: Language-aware filtering
                if lang_hints:
                    n_type = (node.get("symbol_type") or "").lower()
                    n_path = node.get("rel_path") or ""
                    if not should_include_in_expansion(lang_hints, n_type, n_path):
                        continue

                doc = _node_to_document(node, is_initial=False, expanded_from=reason)
                aug_cost = _augment_document(db, doc)
                cost = _estimate_tokens(doc.page_content)
                if cost > budget_remaining:
                    continue
                result_docs.append(doc)
                seen_ids.add(nid)
                budget_remaining -= cost
    else:
        # ── Legacy: generic 1-hop expansion ──────────────────────
        expansion_pool = _collect_expansion_neighbors(
            db, list(matched_nodes.keys()), seen_ids, macro_id,
            page_boundary_ids=page_boundary_ids,
        )

        for priority_group in (_P0_REL_TYPES, _P1_REL_TYPES, _P2_REL_TYPES):
            if budget_remaining <= 0:
                break
            group_nodes = [
                (nid, node, rel) for nid, node, rel in expansion_pool
                if rel in priority_group and nid not in seen_ids
            ]
            for nid, node, rel in group_nodes:
                if budget_remaining <= 0 or len(result_docs) >= MAX_EXPANSION_TOTAL:
                    break
                doc = _node_to_document(node, is_initial=False, expanded_from=rel)
                aug_cost = _augment_document(db, doc)
                cost = _estimate_tokens(doc.page_content)
                if cost > budget_remaining:
                    continue
                result_docs.append(doc)
                seen_ids.add(nid)
                budget_remaining -= cost

    # ── Step 4: Include cluster doc nodes (if budget allows) ─────
    if include_docs and macro_id is not None and budget_remaining > 0:
        doc_nodes = _get_cluster_docs(db, macro_id, micro_id, seen_ids)
        for node in doc_nodes:
            if budget_remaining <= 0 or len(result_docs) >= MAX_EXPANSION_TOTAL:
                break
            doc = _node_to_document(node, is_initial=False, expanded_from="cluster_doc")
            cost = _estimate_tokens(doc.page_content)
            if cost > budget_remaining:
                continue
            result_docs.append(doc)
            seen_ids.add(node.get("node_id", ""))
            budget_remaining -= cost

    tokens_used = token_budget - budget_remaining
    logger.info(
        "[CLUSTER_EXPANSION] Page expansion: %d docs, ~%d tokens used "
        "(%d initial, %d expanded, macro=%s)",
        len(result_docs), tokens_used,
        sum(1 for d in result_docs if d.metadata.get("is_initially_retrieved")),
        sum(1 for d in result_docs if not d.metadata.get("is_initially_retrieved")),
        macro_id,
    )
    return result_docs


# ---------------------------------------------------------------------------
# Symbol resolution
# ---------------------------------------------------------------------------

def _resolve_symbols(
    db, symbol_names: List[str], macro_id: Optional[int] = None,
) -> Dict[str, Dict[str, Any]]:
    """Resolve symbol *names* to node dicts via SQL, optionally scoped to cluster.

    Returns ``{node_id: node_dict}`` preserving insertion order.
    """
    result: Dict[str, Dict[str, Any]] = {}
    conn = db.conn

    for name in symbol_names:
        if not name:
            continue

        # Exact match on symbol_name, optionally within cluster
        if macro_id is not None:
            rows = conn.execute(
                "SELECT * FROM repo_nodes "
                "WHERE symbol_name = ? AND macro_cluster = ? "
                "AND is_architectural = 1 "
                "ORDER BY end_line - start_line DESC "
                "LIMIT 5",
                (name, macro_id),
            ).fetchall()
        else:
            rows = conn.execute(
                "SELECT * FROM repo_nodes "
                "WHERE symbol_name = ? "
                "AND is_architectural = 1 "
                "ORDER BY end_line - start_line DESC "
                "LIMIT 5",
                (name,),
            ).fetchall()

        if not rows:
            # Fallback: FTS5 fuzzy search scoped to cluster
            rows = _fts_fallback(db, name, macro_id)

        for row in rows:
            node = dict(row)
            nid = node.get("node_id", "")
            if nid and nid not in result:
                result[nid] = node

    return result


def _fts_fallback(
    db, name: str, macro_id: Optional[int] = None,
) -> list:
    """Use FTS5 to find a symbol by name when exact SQL match fails."""
    try:
        # Escape FTS5 special characters
        safe_name = name.replace('"', '""')
        if macro_id is not None:
            rows = db.conn.execute(
                'SELECT n.* FROM repo_fts f '
                'JOIN repo_nodes n ON f.node_id = n.node_id '
                'WHERE repo_fts MATCH ? '
                'AND n.macro_cluster = ? '
                'AND n.is_architectural = 1 '
                'ORDER BY rank LIMIT 3',
                (f'symbol_name:"{safe_name}"', macro_id),
            ).fetchall()
        else:
            rows = db.conn.execute(
                'SELECT n.* FROM repo_fts f '
                'JOIN repo_nodes n ON f.node_id = n.node_id '
                'WHERE repo_fts MATCH ? '
                'AND n.is_architectural = 1 '
                'ORDER BY rank LIMIT 3',
                (f'symbol_name:"{safe_name}"',),
            ).fetchall()
        return rows
    except Exception as exc:
        logger.debug("[CLUSTER_EXPANSION] FTS5 fallback failed for '%s': %s", name, exc)
        return []


# ---------------------------------------------------------------------------
# Expansion neighbour collection
# ---------------------------------------------------------------------------

def _collect_expansion_neighbors(
    db,
    seed_ids: List[str],
    seen_ids: Set[str],
    macro_id: Optional[int] = None,
    page_boundary_ids: Optional[Set[str]] = None,
) -> List[Tuple[str, Dict[str, Any], str]]:
    """Collect 1-hop expansion neighbours for *seed_ids*, bounded to cluster.

    When *page_boundary_ids* is provided, only neighbours whose node_id
    is in that set are included (page-level boundary).  Otherwise, falls
    back to macro-cluster boundary via *macro_id*.

    Returns list of ``(node_id, node_dict, rel_type)`` tuples sorted by
    edge weight (descending, i.e. highest-weight edges first).
    """
    candidates: List[Tuple[str, Dict[str, Any], str, float]] = []
    conn = db.conn

    for seed_id in seed_ids:
        # Outgoing edges
        out_edges = conn.execute(
            "SELECT target_id, rel_type, weight FROM repo_edges WHERE source_id = ?",
            (seed_id,),
        ).fetchall()
        # Incoming edges (structurally important: who inherits/implements me)
        in_edges = conn.execute(
            "SELECT source_id, rel_type, weight FROM repo_edges WHERE target_id = ?",
            (seed_id,),
        ).fetchall()

        neighbor_ids: Set[str] = set()
        edge_info: List[Tuple[str, str, float]] = []  # (nid, rel_type, weight)

        for row in out_edges:
            tid = row[0]
            if tid not in seen_ids and tid not in neighbor_ids:
                neighbor_ids.add(tid)
                edge_info.append((tid, row[1] or "unknown", row[2] or 1.0))

        for row in in_edges:
            sid = row[0]
            if sid not in seen_ids and sid not in neighbor_ids:
                neighbor_ids.add(sid)
                edge_info.append((sid, row[1] or "unknown", row[2] or 1.0))

        if not neighbor_ids:
            continue

        # Batch-fetch node metadata
        # (SQLite doesn't have array params, but we can build an IN clause)
        id_list = list(neighbor_ids)
        placeholders = ",".join("?" * len(id_list))
        rows = conn.execute(
            f"SELECT * FROM repo_nodes WHERE node_id IN ({placeholders})",
            id_list,
        ).fetchall()
        node_map = {dict(r)["node_id"]: dict(r) for r in rows}

        for nid, rel_type, weight in edge_info:
            node = node_map.get(nid)
            if not node:
                continue
            # Only expand architectural nodes
            if not node.get("is_architectural"):
                continue
            # Skip test nodes when exclude_tests is enabled
            if _is_excluded_test_node(node, get_feature_flags().exclude_tests):
                continue
            stype = (node.get("symbol_type") or "").lower()
            if stype not in EXPANSION_SYMBOL_TYPES:
                continue
            # Cluster boundary enforcement
            if page_boundary_ids is not None:
                # Page-level boundary: only include neighbours in the page
                if nid not in page_boundary_ids:
                    continue
            elif macro_id is not None and node.get("macro_cluster") != macro_id:
                continue
            candidates.append((nid, node, rel_type, weight))

    # Deduplicate (first occurrence wins) and sort by weight descending
    seen: Set[str] = set()
    unique: List[Tuple[str, Dict[str, Any], str, float]] = []
    for nid, node, rel, w in sorted(candidates, key=lambda x: -x[3]):
        if nid not in seen and nid not in seen_ids:
            seen.add(nid)
            unique.append((nid, node, rel, w))

    # Cap total expansion candidates
    unique = unique[:MAX_EXPANSION_TOTAL]

    return [(nid, node, rel) for nid, node, rel, _ in unique]


# ---------------------------------------------------------------------------
# Documentation nodes from cluster
# ---------------------------------------------------------------------------

def _get_cluster_docs(
    db,
    macro_id: int,
    micro_id: Optional[int],
    seen_ids: Set[str],
) -> List[Dict[str, Any]]:
    """Fetch documentation nodes from the same cluster not yet in ``seen_ids``.

    Returns nodes sorted by path (deterministic ordering).
    """
    conn = db.conn
    if micro_id is not None:
        rows = conn.execute(
            "SELECT * FROM repo_nodes "
            "WHERE macro_cluster = ? AND micro_cluster = ? AND is_doc = 1 "
            "ORDER BY rel_path",
            (macro_id, micro_id),
        ).fetchall()
    else:
        rows = conn.execute(
            "SELECT * FROM repo_nodes "
            "WHERE macro_cluster = ? AND is_doc = 1 "
            "ORDER BY rel_path",
            (macro_id,),
        ).fetchall()

    result = []
    exclude = get_feature_flags().exclude_tests
    for r in rows:
        node = dict(r)
        if node.get("node_id", "") not in seen_ids:
            if not _is_excluded_test_node(node, exclude):
                result.append(node)
    return result
