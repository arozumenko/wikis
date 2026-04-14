"""
Cluster-Bounded Content Expansion.

Provides ``expand_for_page()`` which retrieves and expands page content
directly from :class:`UnifiedWikiDB`, using cluster boundaries to prevent
context pollution from unrelated symbols.

When ``planner_type="cluster"``, the wiki agent calls this module instead
of the legacy NX-graph-based expansion.

Key advantages:
- Expansion is bounded to the page's macro-cluster (no cross-cluster leaks)
- No in-memory NX graph needed at page-generation time
- Documentation files naturally included (they're cluster members)
- Token-budget aware: symbols are added in priority order until the budget is exhausted
- Result is a flat list of LangChain ``Document`` objects ready for LLM context

Ported from DeepWiki ``cluster_expansion.py``.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Set, Tuple

from langchain_core.documents import Document

from .code_graph.shared_expansion import (
    EXPANSION_WORTHY_TYPES,
    expand_symbol_smart,
)
from .cluster_constants import is_test_path
from .wiki_structure_planner.language_heuristics import (
    compute_augmentation_budget_fraction,
    detect_dominant_language,
    get_language_hints,
    should_include_in_expansion,
)

logger = logging.getLogger(__name__)


def _is_excluded_test_node(node: Dict[str, Any], exclude_tests: bool) -> bool:
    """Return True if *node* should be skipped because it's a test node.

    Since the unified DB may not have an ``is_test`` column yet, we also
    check the node's ``rel_path`` against the test-path heuristics.
    """
    if not exclude_tests:
        return False
    if node.get("is_test"):
        return True
    rel_path = node.get("rel_path", "")
    if rel_path and is_test_path(rel_path):
        return True
    return False


# Languages that support header/implementation split augmentation.
_CPP_LANGUAGES = frozenset({'cpp', 'c'})
_GO_LANGUAGES = frozenset({'go'})
_RUST_LANGUAGES = frozenset({'rust'})
_AUGMENTABLE_LANGUAGES = _CPP_LANGUAGES | _GO_LANGUAGES | _RUST_LANGUAGES

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

DEFAULT_TOKEN_BUDGET = 50_000
MAX_NEIGHBORS_PER_SYMBOL = 15
MAX_EXPANSION_TOTAL = 200

EXPANSION_SYMBOL_TYPES = frozenset({
    'class', 'interface', 'struct', 'enum', 'trait',
    'function', 'constant', 'type_alias', 'macro',
    'module_doc', 'file_doc',
})

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

_CHARS_PER_TOKEN = 3.5


def _estimate_tokens(text: str) -> int:
    """Rough token estimate from character count."""
    return max(1, int(len(text) / _CHARS_PER_TOKEN))


# ---------------------------------------------------------------------------
# Node → Document conversion
# ---------------------------------------------------------------------------

def _node_to_document(
    node: Dict[str, Any],
    is_initial: bool = False,
    expanded_from: str = "",
) -> Document:
    """Convert a unified-DB node dict to a LangChain Document.

    The ``page_content`` is built from source_text (preferred) or
    signature+docstring fallback.
    """
    source = node.get("source_text") or ""
    if not source:
        sig = node.get("signature") or ""
        doc = node.get("docstring") or ""
        source = f"{sig}\n{doc}".strip() if sig or doc else ""

    rel_path = node.get("rel_path", "")
    symbol_name = node.get("symbol_name", "")
    symbol_type = node.get("symbol_type", "")
    start_line = node.get("start_line", 0)
    end_line = node.get("end_line", 0)
    language = node.get("language", "")

    metadata = {
        "source": rel_path,
        "symbol_name": symbol_name,
        "symbol_type": symbol_type,
        "start_line": start_line,
        "end_line": end_line,
        "language": language,
        "node_id": node.get("node_id", ""),
        "is_initially_retrieved": is_initial,
        "expanded_from": expanded_from,
        "macro_cluster": node.get("macro_cluster"),
        "micro_cluster": node.get("micro_cluster"),
    }

    return Document(page_content=source, metadata=metadata)


# ---------------------------------------------------------------------------
# Cross-file augmentation (C++, Go, Rust)
# ---------------------------------------------------------------------------

def _augment_document(db, doc: Document) -> int:
    """Augment a document with cross-file implementation details.

    Returns estimated extra tokens added (for budget tracking).
    """
    lang = doc.metadata.get("language", "").lower()
    if lang not in _AUGMENTABLE_LANGUAGES:
        return 0

    node_id = doc.metadata.get("node_id", "")
    if not node_id:
        return 0

    before_len = len(doc.page_content)

    if lang in _CPP_LANGUAGES:
        _augment_cpp(db, doc, node_id)
    elif lang in _GO_LANGUAGES:
        _augment_go_rust(db, doc, node_id, direction="out")
    elif lang in _RUST_LANGUAGES:
        _augment_go_rust(db, doc, node_id, direction="out")

    added_chars = len(doc.page_content) - before_len
    return _estimate_tokens(doc.page_content[:added_chars]) if added_chars > 0 else 0


def _augment_cpp(db, doc: Document, node_id: str) -> None:
    """C/C++: stitch header declarations with implementation bodies.

    For a class declared in a .h file, find the corresponding .cpp
    methods via ``defines_body`` edges.
    """
    conn = db.conn
    rel_path = doc.metadata.get("source", "")

    # Check if this is a header file
    is_header = any(rel_path.endswith(ext) for ext in ('.h', '.hpp', '.hxx', '.hh'))

    if is_header:
        # Find implementation bodies linked via defines_body
        rows = conn.execute(
            "SELECT n.source_text, n.rel_path, n.symbol_name, n.start_line "
            "FROM repo_edges e "
            "JOIN repo_nodes n ON e.source_id = n.node_id "
            "WHERE e.target_id = ? AND e.rel_type = 'defines_body' "
            "ORDER BY n.rel_path, n.start_line "
            "LIMIT 10",
            (node_id,),
        ).fetchall()

        if rows:
            impl_parts = []
            for row in rows:
                r = dict(row)
                impl_text = r.get("source_text", "")
                impl_path = r.get("rel_path", "")
                impl_name = r.get("symbol_name", "")
                if impl_text:
                    impl_parts.append(
                        f"\n// --- Implementation: {impl_name} "
                        f"(from {impl_path}) ---\n{impl_text}"
                    )
            if impl_parts:
                doc.page_content += "\n".join(impl_parts)
    else:
        # Implementation file: find class declaration in header
        rows = conn.execute(
            "SELECT n.source_text, n.rel_path, n.symbol_name "
            "FROM repo_edges e "
            "JOIN repo_nodes n ON e.target_id = n.node_id "
            "WHERE e.source_id = ? AND e.rel_type = 'defines_body' "
            "AND n.symbol_type IN ('class', 'struct') "
            "ORDER BY n.rel_path "
            "LIMIT 3",
            (node_id,),
        ).fetchall()

        for row in rows:
            r = dict(row)
            decl_text = r.get("source_text", "")
            decl_path = r.get("rel_path", "")
            if decl_text:
                doc.page_content = (
                    f"// --- Declaration (from {decl_path}) ---\n"
                    f"{decl_text}\n\n"
                    f"// --- Implementation ---\n{doc.page_content}"
                )


def _augment_go_rust(
    db, doc: Document, node_id: str, direction: str = "out",
) -> None:
    """Go/Rust: attach receiver methods or impl block members.

    For a struct/interface, find methods defined in other files via
    ``defines_body`` or ``implementation`` edges.
    """
    conn = db.conn
    stype = doc.metadata.get("symbol_type", "").lower()

    if stype not in ('struct', 'interface', 'trait', 'enum'):
        return

    # Find associated methods/implementations
    rows = conn.execute(
        "SELECT n.source_text, n.rel_path, n.symbol_name, n.start_line "
        "FROM repo_edges e "
        "JOIN repo_nodes n ON e.target_id = n.node_id "
        "WHERE e.source_id = ? "
        "AND e.rel_type IN ('defines_body', 'implementation') "
        "AND n.rel_path != ? "
        "ORDER BY n.rel_path, n.start_line "
        "LIMIT 15",
        (node_id, doc.metadata.get("source", "")),
    ).fetchall()

    if rows:
        impl_parts = []
        for row in rows:
            r = dict(row)
            text = r.get("source_text", "")
            path = r.get("rel_path", "")
            name = r.get("symbol_name", "")
            if text:
                impl_parts.append(
                    f"\n// --- Method/Impl: {name} (from {path}) ---\n{text}"
                )
        if impl_parts:
            doc.page_content += "\n".join(impl_parts)


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def expand_for_page(
    db,
    page_symbols: List[str],
    macro_id: Optional[int] = None,
    micro_id: Optional[int] = None,
    cluster_node_ids: Optional[List[str]] = None,
    token_budget: int = DEFAULT_TOKEN_BUDGET,
    include_docs: bool = True,
    exclude_tests: bool = False,
) -> List[Document]:
    """Expand content for a single wiki page using cluster-bounded retrieval.

    Parameters
    ----------
    db : UnifiedWikiDB
        Open database instance.
    page_symbols : list[str]
        Symbol names assigned to this page by the cluster planner.
    macro_id : int, optional
        Macro-cluster ID for boundary enforcement.
    micro_id : int, optional
        Micro-cluster ID for finer doc retrieval.
    cluster_node_ids : list[str], optional
        Explicit set of node IDs belonging to the page (from planner metadata).
    token_budget : int
        Maximum tokens to include in the result.
    include_docs : bool
        Whether to include documentation nodes from the cluster.
    exclude_tests : bool
        Whether to exclude test nodes.

    Returns
    -------
    list[Document]
        Ordered list of LangChain Documents, ready for LLM context.
    """
    if not page_symbols:
        return []

    result_docs: List[Document] = []
    seen_ids: Set[str] = set()
    budget_remaining = token_budget

    # Page boundary: prefer explicit node IDs from planner metadata
    page_boundary_ids: Optional[Set[str]] = None
    if cluster_node_ids:
        page_boundary_ids = set(cluster_node_ids)

    # ── Step 1: Resolve symbols to nodes ──────────────────────────
    matched_nodes = _resolve_symbols(db, page_symbols, macro_id)

    if not matched_nodes:
        logger.info(
            "[CLUSTER_EXPANSION] No symbols resolved for page "
            "(symbols=%s, macro=%s)",
            page_symbols[:5], macro_id,
        )
        return []

    # ── Step 2: Add initial (seed) documents ──────────────────────
    for node_id, node in matched_nodes.items():
        if _is_excluded_test_node(node, exclude_tests):
            continue
        doc = _node_to_document(node, is_initial=True)
        aug_cost = _augment_document(db, doc)
        cost = _estimate_tokens(doc.page_content)
        if cost > budget_remaining:
            continue
        result_docs.append(doc)
        seen_ids.add(node_id)
        budget_remaining -= cost

    if not result_docs:
        return []

    # ── Step 3: Smart expansion per seed ──────────────────────────
    # Detect dominant language for expansion hints
    seed_ids = list(matched_nodes.keys())
    dominant_lang = detect_dominant_language(db.conn, seed_ids)
    lang_hints = get_language_hints(dominant_lang or "unknown")

    use_smart_expansion = True  # Always use smart expansion in wikis

    if use_smart_expansion:
        for seed_id in seed_ids:
            if budget_remaining <= 0 or len(result_docs) >= MAX_EXPANSION_TOTAL:
                break
            node = matched_nodes.get(seed_id)
            if not node:
                continue
            stype = (node.get("symbol_type") or "").lower()

            neighbors = expand_symbol_smart(
                conn=db.conn,
                node_id=seed_id,
                symbol_type=stype,
                seen_ids=seen_ids,
                page_boundary_ids=page_boundary_ids,
                macro_id=macro_id,
                per_symbol_budget=MAX_NEIGHBORS_PER_SYMBOL,
                extra_rel_types=lang_hints.extra_expansion_rels or None,
                exclude_tests=exclude_tests,
            )

            for nid, n_node, reason in neighbors:
                if budget_remaining <= 0 or len(result_docs) >= MAX_EXPANSION_TOTAL:
                    break
                if nid in seen_ids:
                    continue

                n_type = (n_node.get("symbol_type") or "").lower()
                n_path = n_node.get("rel_path", "")
                if not should_include_in_expansion(lang_hints, n_type, n_path):
                    continue

                doc = _node_to_document(n_node, is_initial=False, expanded_from=reason)
                _augment_document(db, doc)
                cost = _estimate_tokens(doc.page_content)
                if cost > budget_remaining:
                    continue
                result_docs.append(doc)
                seen_ids.add(nid)
                budget_remaining -= cost
    else:
        # Legacy: generic 1-hop expansion
        expansion_pool = _collect_expansion_neighbors(
            db, list(matched_nodes.keys()), seen_ids, macro_id,
            page_boundary_ids=page_boundary_ids,
            exclude_tests=exclude_tests,
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
                _augment_document(db, doc)
                cost = _estimate_tokens(doc.page_content)
                if cost > budget_remaining:
                    continue
                result_docs.append(doc)
                seen_ids.add(nid)
                budget_remaining -= cost

    # ── Step 4: Include cluster doc nodes (if budget allows) ─────
    if include_docs and macro_id is not None and budget_remaining > 0:
        doc_nodes = _get_cluster_docs(
            db, macro_id, micro_id, seen_ids, exclude_tests,
        )
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
    exclude_tests: bool = False,
) -> List[Tuple[str, Dict[str, Any], str]]:
    """Collect 1-hop expansion neighbours for *seed_ids*, bounded to cluster.

    Returns list of ``(node_id, node_dict, rel_type)`` tuples sorted by
    edge weight descending.
    """
    candidates: List[Tuple[str, Dict[str, Any], str, float]] = []
    conn = db.conn

    for seed_id in seed_ids:
        out_edges = conn.execute(
            "SELECT target_id, rel_type, weight FROM repo_edges WHERE source_id = ?",
            (seed_id,),
        ).fetchall()
        in_edges = conn.execute(
            "SELECT source_id, rel_type, weight FROM repo_edges WHERE target_id = ?",
            (seed_id,),
        ).fetchall()

        neighbor_ids: Set[str] = set()
        edge_info: List[Tuple[str, str, float]] = []

        for row in out_edges:
            tid = row[0] if isinstance(row, (tuple, list)) else row["target_id"]
            rtype = (row[1] if isinstance(row, (tuple, list)) else row["rel_type"]) or "unknown"
            w = (row[2] if isinstance(row, (tuple, list)) else row["weight"]) or 1.0
            if tid not in seen_ids and tid not in neighbor_ids:
                neighbor_ids.add(tid)
                edge_info.append((tid, rtype, w))

        for row in in_edges:
            sid = row[0] if isinstance(row, (tuple, list)) else row["source_id"]
            rtype = (row[1] if isinstance(row, (tuple, list)) else row["rel_type"]) or "unknown"
            w = (row[2] if isinstance(row, (tuple, list)) else row["weight"]) or 1.0
            if sid not in seen_ids and sid not in neighbor_ids:
                neighbor_ids.add(sid)
                edge_info.append((sid, rtype, w))

        if not neighbor_ids:
            continue

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
            if not node.get("is_architectural"):
                continue
            if _is_excluded_test_node(node, exclude_tests):
                continue
            stype = (node.get("symbol_type") or "").lower()
            if stype not in EXPANSION_SYMBOL_TYPES:
                continue
            if page_boundary_ids is not None:
                if nid not in page_boundary_ids:
                    continue
            elif macro_id is not None and node.get("macro_cluster") != macro_id:
                continue
            candidates.append((nid, node, rel_type, weight))

    # Deduplicate and sort by weight descending
    seen: Set[str] = set()
    unique: List[Tuple[str, Dict[str, Any], str, float]] = []
    for nid, node, rel, w in sorted(candidates, key=lambda x: -x[3]):
        if nid not in seen and nid not in seen_ids:
            seen.add(nid)
            unique.append((nid, node, rel, w))

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
    exclude_tests: bool = False,
) -> List[Dict[str, Any]]:
    """Fetch documentation nodes from the same cluster not yet in ``seen_ids``."""
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
    for r in rows:
        node = dict(r)
        if node.get("node_id", "") not in seen_ids:
            if not _is_excluded_test_node(node, exclude_tests):
                result.append(node)
    return result
