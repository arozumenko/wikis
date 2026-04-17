"""
Shared Expansion Helpers — DB-based per-symbol-type bidirectional expansion.

Per-symbol-type bidirectional expansion strategies operating on the
WikiStorageProtocol so they can be used by ``cluster_expansion``
without loading the full graph into memory.

Ported from DeepWiki ``code_graph/shared_expansion.py``.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, FrozenSet, List, Optional, Set, Tuple

logger = logging.getLogger(__name__)

# Relationship type priority groups
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
    db, node_id: str, max_hops: int = 5,
) -> Optional[str]:
    """Follow ``alias_of`` edges from a type_alias to a concrete type.

    Returns the concrete target node_id, or None if resolution fails.
    """
    visited = {node_id}
    current = node_id

    for _ in range(max_hops):
        edges = db.get_edge_targets(current)
        alias_targets = [
            e.get("target_id") for e in edges
            if (e.get("rel_type") or "").lower() == "alias_of"
        ]

        if not alias_targets:
            break

        target = alias_targets[0]
        if target in visited:
            break
        visited.add(target)

        node = db.get_node(target)
        if not node:
            break

        stype = (node.get("symbol_type") or "").lower()
        if stype == "type_alias":
            current = target
        else:
            return target

    return current if current != node_id else None


# ═════════════════════════════════════════════════════════════════════════════
# Per-Symbol-Type Expansion Strategies
# ═════════════════════════════════════════════════════════════════════════════

def expand_symbol_smart(
    db,
    node_id: str,
    symbol_type: str,
    seen_ids: Set[str],
    page_boundary_ids: Optional[Set[str]] = None,
    macro_id: Optional[int] = None,
    per_symbol_budget: int = 10,
    extra_rel_types: Optional[frozenset] = None,
    exclude_tests: bool = False,
) -> List[Tuple[str, Dict[str, Any], str]]:
    """Bidirectional expansion for a single symbol, strategy by type.

    Parameters
    ----------
    db
        WikiStorageProtocol-conforming storage instance.
    extra_rel_types : frozenset, optional
        Additional relationship types to follow (from language heuristics).
    exclude_tests : bool
        When True, skip nodes marked as test nodes.

    Returns list of (neighbor_node_id, node_dict, reason) tuples.
    """
    stype = symbol_type.lower()
    candidates: List[Tuple[str, Dict[str, Any], str, float]] = []

    if stype in ('class', 'interface', 'struct', 'enum', 'trait'):
        candidates = _expand_class_db(db, node_id, seen_ids, page_boundary_ids, macro_id, exclude_tests)
    elif stype == 'function':
        candidates = _expand_function_db(db, node_id, seen_ids, page_boundary_ids, macro_id, exclude_tests)
    elif stype == 'constant':
        candidates = _expand_constant_db(db, node_id, seen_ids, page_boundary_ids, macro_id, exclude_tests)
    elif stype == 'type_alias':
        candidates = _expand_type_alias_db(db, node_id, seen_ids, page_boundary_ids, macro_id, exclude_tests)
    elif stype == 'macro':
        candidates = _expand_macro_db(db, node_id, seen_ids, page_boundary_ids, macro_id, exclude_tests)
    else:
        candidates = _expand_generic_db(db, node_id, seen_ids, page_boundary_ids, macro_id, exclude_tests)

    # Expand via extra language-specific relationship types
    if extra_rel_types:
        extra_candidates = _collect_neighbors(
            db, node_id, extra_rel_types, "out", seen_ids,
            page_boundary_ids, macro_id, exclude_tests,
            limit=5, reason_prefix="lang_hint:",
        )
        extra_candidates.extend(_collect_neighbors(
            db, node_id, extra_rel_types, "in", seen_ids,
            page_boundary_ids, macro_id, exclude_tests,
            limit=5, reason_prefix="lang_hint:",
        ))
        candidates.extend(extra_candidates)

    # Deduplicate and cap
    result = []
    result_ids: set[str] = set()
    for nid, node, reason, weight in sorted(candidates, key=lambda x: -x[3]):
        if nid in result_ids or nid in seen_ids:
            continue
        result_ids.add(nid)
        result.append((nid, node, reason))
        if len(result) >= per_symbol_budget:
            break

    return result


# ═════════════════════════════════════════════════════════════════════════════
# DB Query Helpers
# ═════════════════════════════════════════════════════════════════════════════

def _get_outgoing(db, node_id, rel_types=None):
    """Get outgoing edges from node_id, optionally filtered by rel_type."""
    rows = db.get_edge_targets(node_id)
    result = []
    for row in rows:
        tid = row.get("target_id", "")
        rtype = (row.get("rel_type") or "").lower()
        w = row.get("weight") or 1.0
        if rel_types is None or rtype in rel_types:
            result.append((tid, rtype, w))
    return result


def _get_incoming(db, node_id, rel_types=None):
    """Get incoming edges to node_id, optionally filtered by rel_type."""
    rows = db.get_edge_sources(node_id)
    result = []
    for row in rows:
        sid = row.get("source_id", "")
        rtype = (row.get("rel_type") or "").lower()
        w = row.get("weight") or 1.0
        if rel_types is None or rtype in rel_types:
            result.append((sid, rtype, w))
    return result


def _fetch_node(db, node_id):
    """Fetch a node dict from DB via protocol."""
    return db.get_node(node_id)


def _is_valid_expansion(node, page_boundary_ids, macro_id, exclude_tests: bool = False):
    """Check if a node is valid for expansion."""
    if not node:
        return False
    if not node.get("is_architectural"):
        return False
    if exclude_tests and node.get("is_test"):
        return False
    stype = (node.get("symbol_type") or "").lower()
    if stype not in EXPANSION_WORTHY_TYPES:
        return False
    if page_boundary_ids is not None:
        return node["node_id"] in page_boundary_ids
    if macro_id is not None:
        return node.get("macro_cluster") == macro_id
    return True


def _collect_neighbors(db, node_id, rel_types, direction, seen_ids,
                       page_boundary_ids, macro_id, exclude_tests: bool = False,
                       limit=5, reason_prefix=""):
    """Collect validated neighbors in a given direction."""
    edges = _get_outgoing(db, node_id, rel_types) if direction == "out" \
        else _get_incoming(db, node_id, rel_types)
    result = []
    for nid, rtype, weight in edges:
        if nid in seen_ids or len(result) >= limit:
            continue
        node = _fetch_node(db, nid)
        if not _is_valid_expansion(node, page_boundary_ids, macro_id, exclude_tests):
            continue
        reason = f"{reason_prefix}{rtype}" if reason_prefix else rtype
        result.append((nid, node, reason, weight))
    return result


# ── Class/Interface/Struct expansion ─────────────────────────────────

def _expand_class_db(db, node_id, seen_ids, page_boundary_ids, macro_id, exclude_tests):
    """Expand class-like symbol: base classes, implementors, composed types."""
    candidates = []
    candidates.extend(_collect_neighbors(
        db, node_id, {'inheritance', 'implementation'}, "out",
        seen_ids, page_boundary_ids, macro_id, exclude_tests, limit=3, reason_prefix="base:"))
    candidates.extend(_collect_neighbors(
        db, node_id, {'inheritance', 'implementation'}, "in",
        seen_ids, page_boundary_ids, macro_id, exclude_tests, limit=3, reason_prefix="derived:"))
    candidates.extend(_collect_neighbors(
        db, node_id, {'creates', 'instantiates'}, "out",
        seen_ids, page_boundary_ids, macro_id, exclude_tests, limit=2, reason_prefix="creates:"))
    candidates.extend(_collect_neighbors(
        db, node_id, {'composition', 'aggregation'}, "out",
        seen_ids, page_boundary_ids, macro_id, exclude_tests, limit=2, reason_prefix="composes:"))
    candidates.extend(_collect_neighbors(
        db, node_id, {'composition', 'aggregation'}, "in",
        seen_ids, page_boundary_ids, macro_id, exclude_tests, limit=2, reason_prefix="composed_by:"))
    candidates.extend(_collect_neighbors(
        db, node_id, {'references'}, "out",
        seen_ids, page_boundary_ids, macro_id, exclude_tests, limit=2, reason_prefix="refs:"))
    return candidates


# ── Function expansion ───────────────────────────────────────────────

def _expand_function_db(db, node_id, seen_ids, page_boundary_ids, macro_id, exclude_tests):
    """Expand function: callees, callers, parameter types, return type."""
    candidates = []
    candidates.extend(_collect_neighbors(
        db, node_id, {'calls'}, "out",
        seen_ids, page_boundary_ids, macro_id, exclude_tests, limit=3, reason_prefix="calls:"))
    candidates.extend(_collect_neighbors(
        db, node_id, {'calls'}, "in",
        seen_ids, page_boundary_ids, macro_id, exclude_tests, limit=3, reason_prefix="called_by:"))
    candidates.extend(_collect_neighbors(
        db, node_id, {'creates', 'instantiates'}, "out",
        seen_ids, page_boundary_ids, macro_id, exclude_tests, limit=2, reason_prefix="creates:"))
    candidates.extend(_collect_neighbors(
        db, node_id, {'references', 'composition'}, "out",
        seen_ids, page_boundary_ids, macro_id, exclude_tests, limit=2, reason_prefix="refs:"))
    return candidates


# ── Constant expansion ───────────────────────────────────────────────

def _expand_constant_db(db, node_id, seen_ids, page_boundary_ids, macro_id, exclude_tests):
    """Expand constant: referenced-by, definition chain."""
    candidates = []
    candidates.extend(_collect_neighbors(
        db, node_id, {'references'}, "in",
        seen_ids, page_boundary_ids, macro_id, exclude_tests, limit=3, reason_prefix="referenced_by:"))
    candidates.extend(_collect_neighbors(
        db, node_id, {'alias_of', 'references'}, "out",
        seen_ids, page_boundary_ids, macro_id, exclude_tests, limit=2, reason_prefix="type:"))
    return candidates


# ── Type alias expansion ─────────────────────────────────────────────

def _expand_type_alias_db(db, node_id, seen_ids, page_boundary_ids, macro_id, exclude_tests):
    """Expand type_alias: resolve chain, referenced-by, usage sites."""
    candidates = []
    concrete = resolve_alias_chain_db(db, node_id)
    if concrete and concrete != node_id and concrete not in seen_ids:
        node = _fetch_node(db, concrete)
        if _is_valid_expansion(node, page_boundary_ids, macro_id, exclude_tests):
            candidates.append((concrete, node, "alias_resolves_to", 5.0))
    candidates.extend(_collect_neighbors(
        db, node_id, {'references'}, "in",
        seen_ids, page_boundary_ids, macro_id, exclude_tests, limit=3, reason_prefix="used_by:"))
    candidates.extend(_collect_neighbors(
        db, node_id, {'references', 'composition'}, "out",
        seen_ids, page_boundary_ids, macro_id, exclude_tests, limit=2, reason_prefix="refs:"))
    return candidates


# ── Macro expansion ──────────────────────────────────────────────────

def _expand_macro_db(db, node_id, seen_ids, page_boundary_ids, macro_id, exclude_tests):
    """Expand macro: usage sites, referenced symbols."""
    candidates = []
    candidates.extend(_collect_neighbors(
        db, node_id, {'references', 'calls'}, "in",
        seen_ids, page_boundary_ids, macro_id, exclude_tests, limit=3, reason_prefix="used_by:"))
    candidates.extend(_collect_neighbors(
        db, node_id, {'references'}, "out",
        seen_ids, page_boundary_ids, macro_id, exclude_tests, limit=2, reason_prefix="refs:"))
    return candidates


# ── Generic expansion ────────────────────────────────────────────────

def _expand_generic_db(db, node_id, seen_ids, page_boundary_ids, macro_id, exclude_tests):
    """Generic 1-hop expansion for unknown symbol types."""
    candidates = []
    all_out = _get_outgoing(db, node_id)
    for nid, rtype, weight in all_out:
        if rtype in SKIP_RELATIONSHIPS or nid in seen_ids:
            continue
        node = _fetch_node(db, nid)
        if _is_valid_expansion(node, page_boundary_ids, macro_id, exclude_tests):
            candidates.append((nid, node, rtype, weight))
        if len(candidates) >= 5:
            break
    return candidates
