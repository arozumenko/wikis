"""
Shared Expansion Helpers — Phase 4.

Per-symbol-type bidirectional expansion strategies extracted from
``expansion_engine.py`` (NX-graph-based) and adapted for the
unified-DB-based cluster expansion path.

These helpers operate on raw DB connections (not NX graphs) so they
can be used by ``cluster_expansion.py`` without loading the full graph
into memory.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Set, Tuple

logger = logging.getLogger(__name__)

# Relationship type priority groups (same as expansion_engine.py)
P0_REL_TYPES = frozenset({
    'inheritance', 'implementation', 'defines_body',
    'creates', 'instantiates',
})
P1_REL_TYPES = frozenset({
    'composition', 'aggregation', 'alias_of', 'specializes',
})
P2_REL_TYPES = frozenset({
    'calls', 'references',
})

# Relationships to skip
SKIP_RELATIONSHIPS = frozenset({
    'defines', 'imports', 'decorates',
    'contains', 'annotates', 'exports', 'overrides',
    'assigns', 'returns', 'parameter',
    'reads', 'writes', 'captures', 'hides',
    'uses_type',
})

# Symbol types that are valid expansion targets
CLASS_LIKE_TYPES = frozenset({
    'class', 'interface', 'struct', 'enum', 'trait',
})

EXPANSION_WORTHY_TYPES = frozenset({
    'class', 'interface', 'struct', 'enum', 'trait',
    'function', 'constant', 'type_alias', 'macro',
    'module_doc', 'file_doc',
})


# ═════════════════════════════════════════════════════════════════════════════
# Alias Chain Resolution
# ═════════════════════════════════════════════════════════════════════════════

def resolve_alias_chain_db(
    conn, node_id: str, max_hops: int = 5,
) -> Optional[str]:
    """Follow ``alias_of`` edges from a type_alias to a concrete type.

    Operates on the unified DB connection, not an NX graph.
    Returns the concrete target node_id, or None if resolution fails.
    """
    visited = {node_id}
    current = node_id

    for _ in range(max_hops):
        rows = conn.execute(
            "SELECT target_id FROM repo_edges "
            "WHERE source_id = ? AND rel_type = 'alias_of' LIMIT 1",
            (current,),
        ).fetchall()

        if not rows:
            break

        target = rows[0][0] if isinstance(rows[0], (tuple, list)) else rows[0]["target_id"]
        if target in visited:
            break
        visited.add(target)

        # Check if target is still a type_alias
        node = conn.execute(
            "SELECT symbol_type FROM repo_nodes WHERE node_id = ?",
            (target,),
        ).fetchone()

        if not node:
            break

        stype = (node[0] if isinstance(node, (tuple, list)) else node["symbol_type"]) or ""
        if stype.lower() == "type_alias":
            current = target
        else:
            return target  # Concrete type reached

    return current if current != node_id else None


# ═════════════════════════════════════════════════════════════════════════════
# Per-Symbol-Type Expansion Strategies
# ═════════════════════════════════════════════════════════════════════════════

def expand_symbol_smart(
    conn,
    node_id: str,
    symbol_type: str,
    seen_ids: Set[str],
    page_boundary_ids: Optional[Set[str]] = None,
    macro_id: Optional[int] = None,
    per_symbol_budget: int = 10,
    extra_rel_types: Optional[frozenset] = None,
) -> List[Tuple[str, Dict[str, Any], str]]:
    """Bidirectional expansion for a single symbol, strategy by type.

    Parameters
    ----------
    extra_rel_types : frozenset, optional
        Additional relationship types to follow (from language heuristics).
        These are added to the standard P0/P1/P2 sets during expansion.

    Returns list of (neighbor_node_id, node_dict, reason) tuples.
    """
    stype = symbol_type.lower()
    candidates: List[Tuple[str, Dict[str, Any], str, float]] = []

    if stype in ('class', 'interface', 'struct', 'enum', 'trait'):
        candidates = _expand_class_db(conn, node_id, seen_ids, page_boundary_ids, macro_id)
    elif stype == 'function':
        candidates = _expand_function_db(conn, node_id, seen_ids, page_boundary_ids, macro_id)
    elif stype == 'constant':
        candidates = _expand_constant_db(conn, node_id, seen_ids, page_boundary_ids, macro_id)
    elif stype == 'type_alias':
        candidates = _expand_type_alias_db(conn, node_id, seen_ids, page_boundary_ids, macro_id)
    elif stype == 'macro':
        candidates = _expand_macro_db(conn, node_id, seen_ids, page_boundary_ids, macro_id)
    else:
        # Generic 1-hop for unknown types
        candidates = _expand_generic_db(conn, node_id, seen_ids, page_boundary_ids, macro_id)

    # Phase 7: expand via extra language-specific relationship types
    if extra_rel_types:
        extra_candidates = _collect_neighbors(
            conn, node_id, extra_rel_types, "out", seen_ids,
            page_boundary_ids, macro_id, limit=5, reason_prefix="lang_hint:",
        )
        extra_candidates.extend(_collect_neighbors(
            conn, node_id, extra_rel_types, "in", seen_ids,
            page_boundary_ids, macro_id, limit=5, reason_prefix="lang_hint:",
        ))
        candidates.extend(extra_candidates)

    # Deduplicate and cap
    result = []
    result_ids = set()
    for nid, node, reason, weight in sorted(candidates, key=lambda x: -x[3]):
        if nid in result_ids or nid in seen_ids:
            continue
        result_ids.add(nid)
        result.append((nid, node, reason))
        if len(result) >= per_symbol_budget:
            break

    return result


def _get_outgoing(conn, node_id, rel_types=None):
    """Get outgoing edges from node_id, optionally filtered by rel_type."""
    rows = conn.execute(
        "SELECT target_id, rel_type, weight FROM repo_edges WHERE source_id = ?",
        (node_id,),
    ).fetchall()
    result = []
    for row in rows:
        tid = row[0] if isinstance(row, (tuple, list)) else row["target_id"]
        rtype = (row[1] if isinstance(row, (tuple, list)) else row["rel_type"]) or ""
        w = (row[2] if isinstance(row, (tuple, list)) else row["weight"]) or 1.0
        if rel_types is None or rtype.lower() in rel_types:
            result.append((tid, rtype.lower(), w))
    return result


def _get_incoming(conn, node_id, rel_types=None):
    """Get incoming edges to node_id, optionally filtered by rel_type."""
    rows = conn.execute(
        "SELECT source_id, rel_type, weight FROM repo_edges WHERE target_id = ?",
        (node_id,),
    ).fetchall()
    result = []
    for row in rows:
        sid = row[0] if isinstance(row, (tuple, list)) else row["source_id"]
        rtype = (row[1] if isinstance(row, (tuple, list)) else row["rel_type"]) or ""
        w = (row[2] if isinstance(row, (tuple, list)) else row["weight"]) or 1.0
        if rel_types is None or rtype.lower() in rel_types:
            result.append((sid, rtype.lower(), w))
    return result


def _fetch_node(conn, node_id):
    """Fetch a node dict from DB."""
    row = conn.execute(
        "SELECT * FROM repo_nodes WHERE node_id = ?", (node_id,)
    ).fetchone()
    if not row:
        return None
    return dict(row)


def _is_valid_expansion(node, page_boundary_ids, macro_id):
    """Check if a node is valid for expansion."""
    if not node:
        return False
    if not node.get("is_architectural"):
        return False
    # Exclude test nodes when feature flag is active (checked via is_test column)
    if node.get("is_test"):
        from ..feature_flags import get_feature_flags
        if get_feature_flags().exclude_tests:
            return False
    stype = (node.get("symbol_type") or "").lower()
    if stype not in EXPANSION_WORTHY_TYPES:
        return False
    if page_boundary_ids is not None:
        return node["node_id"] in page_boundary_ids
    if macro_id is not None:
        return node.get("macro_cluster") == macro_id
    return True


def _collect_neighbors(conn, node_id, rel_types, direction, seen_ids,
                       page_boundary_ids, macro_id, limit=5, reason_prefix=""):
    """Collect validated neighbors in a given direction."""
    edges = _get_outgoing(conn, node_id, rel_types) if direction == "out" \
        else _get_incoming(conn, node_id, rel_types)
    result = []
    for nid, rtype, weight in edges:
        if nid in seen_ids or len(result) >= limit:
            continue
        node = _fetch_node(conn, nid)
        if not _is_valid_expansion(node, page_boundary_ids, macro_id):
            continue
        reason = f"{reason_prefix}{rtype}" if reason_prefix else rtype
        result.append((nid, node, reason, weight))
    return result


# ── Class/Interface/Struct expansion ─────────────────────────────────

def _expand_class_db(conn, node_id, seen_ids, page_boundary_ids, macro_id):
    """Expand class-like symbol: base classes, implementors, composed types, callers."""
    candidates = []

    # P0 forward: inheritance targets (base classes)
    candidates.extend(_collect_neighbors(
        conn, node_id, {'inheritance', 'implementation'}, "out",
        seen_ids, page_boundary_ids, macro_id, limit=3, reason_prefix="base:"))

    # P0 backward: derived classes / implementors
    candidates.extend(_collect_neighbors(
        conn, node_id, {'inheritance', 'implementation'}, "in",
        seen_ids, page_boundary_ids, macro_id, limit=3, reason_prefix="derived:"))

    # P0 forward: creates/instantiates
    candidates.extend(_collect_neighbors(
        conn, node_id, {'creates', 'instantiates'}, "out",
        seen_ids, page_boundary_ids, macro_id, limit=2, reason_prefix="creates:"))

    # P1 forward: composition/aggregation
    candidates.extend(_collect_neighbors(
        conn, node_id, {'composition', 'aggregation'}, "out",
        seen_ids, page_boundary_ids, macro_id, limit=2, reason_prefix="composes:"))

    # P1 backward: composed-by
    candidates.extend(_collect_neighbors(
        conn, node_id, {'composition', 'aggregation'}, "in",
        seen_ids, page_boundary_ids, macro_id, limit=2, reason_prefix="composed_by:"))

    # P2: references
    candidates.extend(_collect_neighbors(
        conn, node_id, {'references'}, "out",
        seen_ids, page_boundary_ids, macro_id, limit=2, reason_prefix="refs:"))

    return candidates


# ── Function expansion ───────────────────────────────────────────────

def _expand_function_db(conn, node_id, seen_ids, page_boundary_ids, macro_id):
    """Expand function: callees, callers, parameter types, return type."""
    candidates = []

    # P0 forward: calls (callees)
    candidates.extend(_collect_neighbors(
        conn, node_id, {'calls'}, "out",
        seen_ids, page_boundary_ids, macro_id, limit=3, reason_prefix="calls:"))

    # P0 backward: callers
    candidates.extend(_collect_neighbors(
        conn, node_id, {'calls'}, "in",
        seen_ids, page_boundary_ids, macro_id, limit=3, reason_prefix="called_by:"))

    # P0: creates
    candidates.extend(_collect_neighbors(
        conn, node_id, {'creates', 'instantiates'}, "out",
        seen_ids, page_boundary_ids, macro_id, limit=2, reason_prefix="creates:"))

    # P1: references (parameter/return types)
    candidates.extend(_collect_neighbors(
        conn, node_id, {'references', 'composition'}, "out",
        seen_ids, page_boundary_ids, macro_id, limit=2, reason_prefix="refs:"))

    return candidates


# ── Constant expansion ───────────────────────────────────────────────

def _expand_constant_db(conn, node_id, seen_ids, page_boundary_ids, macro_id):
    """Expand constant: referenced-by, definition chain."""
    candidates = []

    # Backward: who references this constant
    candidates.extend(_collect_neighbors(
        conn, node_id, {'references'}, "in",
        seen_ids, page_boundary_ids, macro_id, limit=3, reason_prefix="referenced_by:"))

    # Forward: alias chain if it's typed
    candidates.extend(_collect_neighbors(
        conn, node_id, {'alias_of', 'references'}, "out",
        seen_ids, page_boundary_ids, macro_id, limit=2, reason_prefix="type:"))

    return candidates


# ── Type alias expansion ─────────────────────────────────────────────

def _expand_type_alias_db(conn, node_id, seen_ids, page_boundary_ids, macro_id):
    """Expand type_alias: resolve chain, referenced-by, usage sites."""
    candidates = []

    # Follow alias chain to concrete type
    concrete = resolve_alias_chain_db(conn, node_id)
    if concrete and concrete != node_id and concrete not in seen_ids:
        node = _fetch_node(conn, concrete)
        if _is_valid_expansion(node, page_boundary_ids, macro_id):
            candidates.append((concrete, node, "alias_resolves_to", 5.0))

    # Backward: who uses this alias
    candidates.extend(_collect_neighbors(
        conn, node_id, {'references'}, "in",
        seen_ids, page_boundary_ids, macro_id, limit=3, reason_prefix="used_by:"))

    # Forward: what this alias references beyond the chain
    candidates.extend(_collect_neighbors(
        conn, node_id, {'references', 'composition'}, "out",
        seen_ids, page_boundary_ids, macro_id, limit=2, reason_prefix="refs:"))

    return candidates


# ── Macro expansion ──────────────────────────────────────────────────

def _expand_macro_db(conn, node_id, seen_ids, page_boundary_ids, macro_id):
    """Expand macro: usage sites, referenced symbols."""
    candidates = []

    # Backward: who uses this macro
    candidates.extend(_collect_neighbors(
        conn, node_id, {'references', 'calls'}, "in",
        seen_ids, page_boundary_ids, macro_id, limit=3, reason_prefix="used_by:"))

    # Forward: what the macro references
    candidates.extend(_collect_neighbors(
        conn, node_id, {'references'}, "out",
        seen_ids, page_boundary_ids, macro_id, limit=2, reason_prefix="refs:"))

    return candidates


# ── Generic expansion ────────────────────────────────────────────────

def _expand_generic_db(conn, node_id, seen_ids, page_boundary_ids, macro_id):
    """Generic 1-hop expansion for unknown symbol types."""
    candidates = []

    # Outgoing non-skip edges
    all_out = _get_outgoing(conn, node_id)
    for nid, rtype, weight in all_out:
        if rtype in SKIP_RELATIONSHIPS or nid in seen_ids:
            continue
        node = _fetch_node(conn, nid)
        if _is_valid_expansion(node, page_boundary_ids, macro_id):
            candidates.append((nid, node, rtype, weight))
        if len(candidates) >= 5:
            break

    return candidates
