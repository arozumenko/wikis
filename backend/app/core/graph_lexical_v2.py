"""
Phase 2 (graph-quality roadmap) — IDF gating + tiered lexical T1–T4
+ REST endpoint disambiguation.

This module is consumed by :func:`graph_topology.resolve_orphans` Pass 1
when :attr:`FeatureFlags.orphan_lexical_tiered` is enabled. It is kept
in a separate file so the legacy flat lexical path in ``graph_topology``
remains unchanged and can be diffed cleanly.

Design notes
------------
* ``_compute_symbol_idf`` uses ``count_fts_matches(exact_match=True)``
  introduced in Phase 0 — both SQLite and Postgres back-ends implement
  it.
* All four tiers ultimately call ``db.search_fts5`` (SQLite) or the
  Postgres ``search_fts`` (when implemented as a thin alias). The
  returned dicts already carry ``score_norm`` thanks to Phase 0.
* The REST disambiguation rewrites the **query** only. We do **not**
  mutate the persisted node — only the in-memory graph attribute, so
  re-runs are idempotent.
"""

from __future__ import annotations

import logging
import math
import os
from typing import Any, Callable, Dict, List, Optional, Tuple

import networkx as nx

from .feature_flags import FeatureFlags, get_feature_flags

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════
# Constants — tier eligibility, REST detection, IDF thresholds
# ═══════════════════════════════════════════════════════════════════════════

#: Architectural target ``symbol_type``s for tier filtering. T1/T2 use this
#: as a hard filter; T3/T4 fall back to the broader FTS index.
ARCH_TARGETS: frozenset[str] = frozenset({
    "module", "class", "interface", "function", "method",
    "constant", "module_doc", "file_doc", "rest_endpoint",
})

#: Tiers a given source symbol may issue. Anything not in the table skips
#: lexical resolution entirely (parameters, fields, locals, etc.).
_TIER_ELIGIBILITY: Dict[str, frozenset[str]] = {
    "module":         frozenset({"T1", "T2", "T3", "T4"}),
    "class":          frozenset({"T1", "T2", "T3", "T4"}),
    "interface":      frozenset({"T1", "T2", "T3", "T4"}),
    "struct":         frozenset({"T1", "T2", "T3", "T4"}),
    "enum":           frozenset({"T1", "T2", "T3", "T4"}),
    "trait":          frozenset({"T1", "T2", "T3", "T4"}),
    "rest_endpoint":  frozenset({"T1", "T2", "T3", "T4"}),
    "function":       frozenset({"T1", "T2", "T3"}),
    "method":         frozenset({"T1", "T2"}),
}

#: IDF thresholds → eligible tiers (after intersecting with the symbol
#: type's allowed set). Tighter IDF ⇒ broader tiers.
_IDF_HIGH: float = 3.0
_IDF_LOW: float = 1.5

#: T4 minimum requirements (length + IDF).
_T4_MIN_NAME_LEN: int = 8

#: Generic class names that almost always identify a routing surface
#: rather than a unique semantic concept.
GENERIC_CLASS_NAMES: frozenset[str] = frozenset({
    "API", "Resource", "View", "ViewSet", "Endpoint",
    "Handler", "Controller", "Service", "Manager",
    "Mixin", "Base", "Abstract", "Interface",
})

#: Decorators that flag a class/function as a REST endpoint.
REST_DECORATORS: frozenset[str] = frozenset({
    "@app.route", "@router.get", "@router.post", "@router.put",
    "@router.delete", "@router.patch",
    "@api_view", "@route", "@get", "@post", "@put", "@delete", "@patch",
    "@RestController", "@RequestMapping",
    "@GetMapping", "@PostMapping", "@PutMapping",
    "@DeleteMapping", "@PatchMapping", "@expose",
})

#: Base classes that flag a class as a REST endpoint.
REST_BASE_CLASSES: frozenset[str] = frozenset({
    "APIView", "Resource", "ViewSet", "HTTPEndpoint", "MethodView",
    "Controller", "RouteHandler", "GenericAPIView",
})


# ═══════════════════════════════════════════════════════════════════════════
# IDF
# ═══════════════════════════════════════════════════════════════════════════

def _compute_symbol_idf(db, symbol_name: str) -> float:
    """Return ``log(N_total / (1 + N_matches))`` for *symbol_name*.

    Uses ``count_fts_matches(exact_match=True)`` so we get token-level
    counts rather than substring counts.

    Returns ``+inf`` when the storage back-end refuses the query
    (treat as "very specific" — let the tier table decide).
    """
    try:
        total = db.node_count()
        matches = db.count_fts_matches(symbol_name, exact_match=True)
    except Exception:  # noqa: BLE001 — defensive
        return float("inf")
    if total <= 0:
        return float("inf")
    return math.log(total / (1 + max(0, matches)))


def _allowed_tiers(symbol_type: str, idf: float) -> frozenset[str]:
    """Intersect symbol-type eligibility with the IDF gate."""
    base = _TIER_ELIGIBILITY.get(symbol_type, frozenset())
    if not base:
        return frozenset()
    if idf >= _IDF_HIGH:
        return base
    if idf >= _IDF_LOW:
        return base & frozenset({"T1", "T2"})
    return frozenset()


# ═══════════════════════════════════════════════════════════════════════════
# REST endpoint disambiguation
# ═══════════════════════════════════════════════════════════════════════════

def _detect_rest_endpoint(node_data: Dict[str, Any]) -> bool:
    """Return True iff *node_data* looks like a REST endpoint surface.

    A node is a REST endpoint when **any** of the following is true:

    * Its ``decorators`` list contains a known REST decorator (case
      sensitive on the leading ``@``-prefix; substring match so
      ``@app.route('/')`` still hits ``@app.route``).
    * Its ``base_classes`` list contains a known REST base.
    * Its raw ``source_text`` contains a REST decorator on a non-comment
      line (cheap fallback when AST metadata is missing).
    """
    decorators = node_data.get("decorators") or []
    if isinstance(decorators, (list, tuple)):
        for dec in decorators:
            if not isinstance(dec, str):
                continue
            head = dec.strip()
            if not head:
                continue
            head = head if head.startswith("@") else "@" + head
            for pattern in REST_DECORATORS:
                if head.startswith(pattern):
                    return True

    bases = node_data.get("base_classes") or node_data.get("bases") or []
    if isinstance(bases, (list, tuple)):
        for b in bases:
            if isinstance(b, str) and b.split(".")[-1] in REST_BASE_CLASSES:
                return True

    source = node_data.get("source_text") or ""
    if isinstance(source, str) and source:
        for line in source.splitlines()[:20]:  # decorators always at top
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                continue
            if not stripped.startswith("@"):
                continue
            for pattern in REST_DECORATORS:
                if stripped.startswith(pattern):
                    return True
    return False


def _file_stem(rel_path: str) -> str:
    """Return ``"users"`` for ``"src/api/users.py"`` (no extension)."""
    if not rel_path:
        return ""
    base = rel_path.rsplit("/", 1)[-1]
    return base.rsplit(".", 1)[0]


def _compute_fts_symbol_name(symbol_name: str, file_stem: str) -> str:
    """Build the disambiguated FTS query for a generic REST class.

    ``("API", "users")`` → ``"users API"``. When ``file_stem`` is empty
    the original ``symbol_name`` is returned unchanged.
    """
    if not file_stem:
        return symbol_name
    if file_stem.lower() == symbol_name.lower():
        return symbol_name
    return f"{file_stem} {symbol_name}"


def maybe_disambiguate_rest(
    node_data: Dict[str, Any],
    symbol_name: str,
    rel_path: str,
    flags: Optional[FeatureFlags] = None,
) -> Tuple[str, bool]:
    """Return (effective_query, is_rest_endpoint).

    Side-effect free — callers that want to mark the in-memory graph
    must mutate ``node_data["symbol_type"] = "rest_endpoint"`` themselves
    after inspecting the second tuple element.
    """
    flags = flags or get_feature_flags()
    if not flags.orphan_rest_disambig:
        return symbol_name, False
    if symbol_name not in GENERIC_CLASS_NAMES:
        return symbol_name, False
    if not _detect_rest_endpoint(node_data):
        return symbol_name, False
    return _compute_fts_symbol_name(symbol_name, _file_stem(rel_path)), True


# ═══════════════════════════════════════════════════════════════════════════
# Tiered lexical cascade
# ═══════════════════════════════════════════════════════════════════════════

def _node_dir(rel_path: str) -> str:
    if not rel_path:
        return ""
    parts = rel_path.rsplit("/", 1)
    return parts[0] if len(parts) == 2 else ""


def _parent_dir(rel_path: str) -> str:
    d = _node_dir(rel_path)
    if not d:
        return ""
    return d.rsplit("/", 1)[0] if "/" in d else ""


def _is_arch_target(hit: Dict[str, Any]) -> bool:
    return (hit.get("symbol_type") or "") in ARCH_TARGETS


def _filter_by_dir(hits: List[Dict[str, Any]], directory: str) -> List[Dict[str, Any]]:
    if not directory:
        return []
    out = []
    for h in hits:
        p = h.get("rel_path") or ""
        if not p:
            continue
        # Same directory or sub-directory.
        if p == directory or p.startswith(directory + "/"):
            out.append(h)
    return out


def _drop_self_and_low_score(
    hits: List[Dict[str, Any]],
    self_id: str,
    min_score_norm: float,
) -> List[Dict[str, Any]]:
    out = []
    for h in hits:
        if h.get("node_id") == self_id:
            continue
        if min_score_norm > 0:
            sn = h.get("score_norm")
            if sn is not None and float(sn) < min_score_norm:
                continue
        out.append(h)
    return out


def resolve_orphans_lexical_tiered(
    db,
    G: nx.MultiDiGraph,
    orphan_id: str,
    *,
    fts_limit: int = 8,
    max_lexical_edges: int = 2,
    flags: Optional[FeatureFlags] = None,
) -> List[Dict[str, Any]]:
    """Run the T1→T4 cascade for *orphan_id* and return the chosen hits.

    The first non-empty tier wins; results are capped by
    ``max_lexical_edges`` and stripped of self-hits + sub-threshold
    BM25 scores.

    The function does **not** call :func:`graph_topology._add_edge` —
    the caller (``graph_topology.resolve_orphans``) consumes the
    returned list, attaches provenance, and writes the edges. This
    keeps the v2 path testable in isolation.
    """
    flags = flags or get_feature_flags()

    node_data = G.nodes.get(orphan_id, {}) or {}
    db_node = db.get_node(orphan_id) if db is not None else None

    symbol_name: str = ""
    if db_node:
        symbol_name = db_node.get("symbol_name", "") or ""
    if not symbol_name:
        sym = node_data.get("symbol")
        if isinstance(sym, dict):
            symbol_name = sym.get("name", "") or ""
        elif sym is not None:
            symbol_name = getattr(sym, "name", "") or ""
        if not symbol_name:
            symbol_name = node_data.get("symbol_name", "") or ""
    if not symbol_name or len(symbol_name) < 2:
        return []

    rel_path: str = ""
    if db_node:
        rel_path = db_node.get("rel_path", "") or ""
    if not rel_path:
        rel_path = (
            (node_data.get("location") or {}).get("rel_path", "")
            or node_data.get("rel_path", "")
            or ""
        )

    symbol_type: str = ""
    if db_node:
        symbol_type = db_node.get("symbol_type", "") or ""
    if not symbol_type:
        symbol_type = node_data.get("symbol_type", "") or ""

    # ── REST disambiguation (rewrites query, may upgrade symbol_type)
    fts_query, is_rest = maybe_disambiguate_rest(
        node_data | (db_node or {}), symbol_name, rel_path, flags=flags,
    )
    if is_rest:
        # Mark the in-memory graph (does not touch the persisted row).
        node_data["symbol_type"] = "rest_endpoint"
        symbol_type = "rest_endpoint"

    # ── IDF gate
    if flags.orphan_lexical_idf_gate:
        idf = _compute_symbol_idf(db, fts_query)
    else:
        idf = float("inf")
    allowed = _allowed_tiers(symbol_type, idf)
    if not allowed:
        return []

    # All tiers ultimately read from the FTS index; we issue at most one
    # broad query and partition results in-memory by directory. This is
    # cheaper than 4 separate SQL calls and keeps the storage layer
    # untouched.
    try:
        all_hits = db.search_fts5(query=fts_query, limit=max(fts_limit, 16))
    except Exception as exc:  # noqa: BLE001
        logger.debug("tiered FTS for %r failed: %s", fts_query, exc)
        return []
    all_hits = _drop_self_and_low_score(all_hits, orphan_id, flags.fts_min_score_norm)
    if not all_hits:
        return []

    own_dir = _node_dir(rel_path)
    parent = _parent_dir(rel_path)

    # ── T1: same dir + ARCH target
    if "T1" in allowed and own_dir:
        same_dir = _filter_by_dir(all_hits, own_dir)
        t1 = [h for h in same_dir if _is_arch_target(h)]
        if t1:
            return [dict(h, _tier="T1") for h in t1[:max_lexical_edges]]

    # ── T2: parent dir
    if "T2" in allowed and parent:
        in_parent = _filter_by_dir(all_hits, parent)
        if in_parent:
            return [dict(h, _tier="T2") for h in in_parent[:max_lexical_edges]]

    # ── T3: full FTS local (everything we already fetched, filter ARCH)
    if "T3" in allowed:
        t3 = [h for h in all_hits if _is_arch_target(h)]
        if t3:
            return [dict(h, _tier="T3") for h in t3[:max_lexical_edges]]

    # ── T4: global; gated on length + IDF
    if "T4" in allowed and len(symbol_name) >= _T4_MIN_NAME_LEN and idf >= _IDF_HIGH:
        return [dict(h, _tier="T4") for h in all_hits[:max_lexical_edges]]

    return []


__all__ = [
    "ARCH_TARGETS",
    "GENERIC_CLASS_NAMES",
    "REST_DECORATORS",
    "REST_BASE_CLASSES",
    "_compute_symbol_idf",
    "_detect_rest_endpoint",
    "_compute_fts_symbol_name",
    "maybe_disambiguate_rest",
    "resolve_orphans_lexical_tiered",
]
