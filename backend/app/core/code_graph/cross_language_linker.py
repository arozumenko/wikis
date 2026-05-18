"""Cross-language linker (Phase 6 / Action 6.3).

Side-effect-free 4-level cascade that returns the list of edges to add
to the graph. The caller (Phase 1c pipeline integration, gated by
``WIKI_CROSS_LANGUAGE_LINKING``) is responsible for actually invoking
``graph.add_edge`` and persisting the edges.

Cascade
-------
* **L0** — Exact name + signature: existing
  ``cross_language_relationships`` from the parser stage are promoted
  to ``edge_class="cross_language"`` edges, ``relationship_type=
  "cross_language_L0"``.
* **L1** — API surface match: pair nodes whose ``api_surface[].surface``
  strings match. Confidence ``0.7 * locality * specificity``.
* **L2** — Hybrid FTS+Vec RRF restricted to other-language candidates.
  Confidence ``0.5 * rrf_score``. Reuses :func:`graph_orphan_hybrid.rrf_fuse`.
* **L3** — Containment + member-uses promotion. Confidence
  ``parent_weight * 0.6``.

Final per-edge weight is clamped to ``[0.3, 0.8]`` so that no L1–L3
heuristic can outweigh a real AST-derived structural edge.
"""

from __future__ import annotations

import math
from collections import defaultdict
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import networkx as nx

from .api_surface_extractor import APISurface
from ..feature_flags import FeatureFlags, get_feature_flags
from ..graph_orphan_hybrid import rrf_fuse


_WEIGHT_FLOOR = 0.3
_WEIGHT_CEIL = 0.8


def _clamp(w: float) -> float:
    return max(_WEIGHT_FLOOR, min(_WEIGHT_CEIL, w))


def _node_language(g: nx.MultiDiGraph, node_id: str) -> str:
    return (g.nodes.get(node_id, {}).get("language") or "").lower()


def _locality_factor(g: nx.MultiDiGraph, src: str, tgt: str) -> float:
    """Boost links whose endpoints live in the same top-level directory."""
    a = (g.nodes.get(src, {}).get("rel_path") or "")
    b = (g.nodes.get(tgt, {}).get("rel_path") or "")
    if not a or not b:
        return 0.5
    pa = a.split("/", 1)[0]
    pb = b.split("/", 1)[0]
    return 1.0 if pa == pb else 0.7


def _rest_surface_parts(surface: str) -> tuple[str, str] | None:
    try:
        method, path = str(surface or "").split(" ", 1)
    except ValueError:
        return None
    if not method or not path.startswith("/"):
        return None
    return method.upper(), path


def _rest_methods_by_node(
    surfaces_by_node: Dict[str, List[APISurface]],
) -> dict[tuple[str, str], set[str]]:
    methods: dict[tuple[str, str], set[str]] = defaultdict(set)
    for node_id, surfaces in surfaces_by_node.items():
        for surface in surfaces:
            if surface.get("kind") != "rest":
                continue
            parts = _rest_surface_parts(surface.get("surface", ""))
            if not parts:
                continue
            method, path = parts
            if method != "*":
                methods[(node_id, path)].add(method)
    return methods


def _skip_rest_path_only_pair(
    *,
    kind: str,
    surface: str,
    src: str,
    tgt: str,
    rest_methods: dict[tuple[str, str], set[str]],
) -> bool:
    if kind != "rest":
        return False
    parts = _rest_surface_parts(surface)
    if not parts:
        return False
    method, path = parts
    if method != "*":
        return False
    src_methods = rest_methods.get((src, path), set())
    tgt_methods = rest_methods.get((tgt, path), set())
    # The method-agnostic twin is for asymmetric contracts where one side
    # only exposes a literal path. If both sides know verbs, method-specific
    # surfaces should represent matches and prevent GET↔POST false links.
    return bool(src_methods and tgt_methods)


def _edge_specificity(attrs: dict) -> tuple[int, float]:
    provenance = attrs.get("provenance") or {}
    surface = str(provenance.get("surface") or "")
    matcher = str(provenance.get("matcher") or "")
    level = str(provenance.get("level") or "")

    if level == "L0":
        rank = 100
    elif surface.startswith("ffi:") or matcher.endswith(":ffi"):
        rank = 90
    elif surface.startswith("grpc:") or matcher.endswith(":grpc"):
        rank = 85
    elif surface.startswith("graphql:") or matcher.endswith(":graphql"):
        rank = 80
    elif surface.startswith("obj:") or matcher.endswith(":obj"):
        rank = 75
    else:
        rest_parts = _rest_surface_parts(surface)
        if rest_parts and rest_parts[0] != "*":
            rank = 70
        elif rest_parts:
            rank = 55
        elif surface.startswith("name:") or matcher.endswith(":name"):
            rank = 10
        else:
            rank = 40
    return rank, float(attrs.get("weight", 0.0) or 0.0)


# ──────────────────────────────────────────────────────────────────────
# L0 — Exact name + signature (parser-supplied cross-lang relationships)
# ──────────────────────────────────────────────────────────────────────


def link_l0_exact(
    g: nx.MultiDiGraph,
    cross_language_relationships: Sequence[dict],
) -> List[Tuple[str, str, dict]]:
    """Promote parser-supplied cross-language relationships to graph edges."""
    out: List[Tuple[str, str, dict]] = []
    for rel in cross_language_relationships or []:
        src = rel.get("source") or rel.get("source_node_id")
        tgt = rel.get("target") or rel.get("target_node_id")
        if not src or not tgt or not g.has_node(src) or not g.has_node(tgt):
            continue
        if _node_language(g, src) == _node_language(g, tgt):
            continue
        weight = _clamp(float(rel.get("confidence", 0.8)))
        out.append((src, tgt, {
            "relationship_type": "cross_language_L0",
            "edge_class": "cross_language",
            "weight": weight,
            "provenance": {
                "source": "cross_language_linker",
                "level": "L0",
                "matcher": rel.get("matcher", "parser_exact"),
            },
        }))
    return out


# ──────────────────────────────────────────────────────────────────────
# L1 — API surface match
# ──────────────────────────────────────────────────────────────────────


def link_l1_api_surface(
    g: nx.MultiDiGraph,
    surfaces_by_node: Dict[str, List[APISurface]],
) -> List[Tuple[str, str, dict]]:
    """Pair nodes whose API surfaces match across languages.

    Specificity = ``1 / log(1 + N_matches_for_this_surface)`` so that
    ubiquitous surfaces (``GET /``) yield weak edges and rare ones
    (``POST /api/admin/billing/refund``) yield strong edges.
    """
    by_surface: Dict[Tuple[str, str], List[Tuple[str, APISurface]]] = defaultdict(list)
    for node_id, surfaces in surfaces_by_node.items():
        if not g.has_node(node_id):
            continue
        for s in surfaces:
            by_surface[(s["kind"], s["surface"])].append((node_id, s))

    rest_methods = _rest_methods_by_node(surfaces_by_node)
    out: List[Tuple[str, str, dict]] = []
    for (kind, surface), nodes in by_surface.items():
        if len(nodes) < 2:
            continue
        specificity = 1.0 / math.log(1.0 + len(nodes))
        # Cross every pair of *distinct-language* nodes once.
        for i in range(len(nodes)):
            for j in range(i + 1, len(nodes)):
                src, src_s = nodes[i]
                tgt, tgt_s = nodes[j]
                if _node_language(g, src) == _node_language(g, tgt):
                    continue
                if _skip_rest_path_only_pair(
                    kind=kind,
                    surface=surface,
                    src=src,
                    tgt=tgt,
                    rest_methods=rest_methods,
                ):
                    continue
                base = 0.7 * (src_s.get("weight_hint", 0.5) + tgt_s.get("weight_hint", 0.5)) / 2.0
                weight = _clamp(base * _locality_factor(g, src, tgt) * specificity)
                attrs = {
                    "relationship_type": "cross_language_L1",
                    "edge_class": "cross_language",
                    "weight": weight,
                    "provenance": {
                        "source": "cross_language_linker",
                        "level": "L1",
                        "matcher": f"api_surface:{kind}",
                        "surface": surface,
                    },
                }
                out.append((src, tgt, attrs))
    return out


def link_api_surface_any_language(
    g: nx.MultiDiGraph,
    surfaces_by_node: Dict[str, List[APISurface]],
    *,
    edge_class: str = "cross_language",
    relationship_type: str = "api_surface_any_language",
    provenance_source: str = "cross_repo_linker",
    provenance_level: str = "L1",
) -> List[Tuple[str, str, dict]]:
    """Variant of :func:`link_l1_api_surface` that does **not** require
    the two endpoints to be in different languages.

    Used by Phase 8 cross-repo linking, where two repos in the same
    language (e.g. two Python services exposing ``POST /api/users``)
    must still be paired by their shared API surface. The caller is
    responsible for filtering out within-repo edges via the
    ``wiki_id`` node attribute.
    """
    by_surface: Dict[Tuple[str, str], List[Tuple[str, APISurface]]] = defaultdict(list)
    for node_id, surfaces in surfaces_by_node.items():
        if not g.has_node(node_id):
            continue
        for s in surfaces:
            by_surface[(s["kind"], s["surface"])].append((node_id, s))

    rest_methods = _rest_methods_by_node(surfaces_by_node)
    out: List[Tuple[str, str, dict]] = []
    for (kind, surface), nodes in by_surface.items():
        if len(nodes) < 2:
            continue
        specificity = 1.0 / math.log(1.0 + len(nodes))
        for i in range(len(nodes)):
            for j in range(i + 1, len(nodes)):
                src, src_s = nodes[i]
                tgt, tgt_s = nodes[j]
                if _skip_rest_path_only_pair(
                    kind=kind,
                    surface=surface,
                    src=src,
                    tgt=tgt,
                    rest_methods=rest_methods,
                ):
                    continue
                base = 0.7 * (src_s.get("weight_hint", 0.5) + tgt_s.get("weight_hint", 0.5)) / 2.0
                weight = _clamp(base * _locality_factor(g, src, tgt) * specificity)
                attrs = {
                    "relationship_type": relationship_type,
                    "edge_class": edge_class,
                    "weight": weight,
                    "provenance": {
                        "source": provenance_source,
                        "level": provenance_level,
                        "matcher": f"api_surface_any:{kind}",
                        "surface": surface,
                    },
                }
                out.append((src, tgt, attrs))
    return out


# ──────────────────────────────────────────────────────────────────────
# L2 — Hybrid RRF restricted to other-language candidates
# ──────────────────────────────────────────────────────────────────────


def link_l2_hybrid(
    g: nx.MultiDiGraph,
    fts_hits_per_node: Dict[str, List[dict]],
    vec_hits_per_node: Dict[str, List[dict]],
    *,
    k: int = 60,
    threshold: float = 0.02,
    top_n: int = 10,
) -> List[Tuple[str, str, dict]]:
    """Fuse FTS + Vec hits per source node, keep cross-language hits only."""
    out: List[Tuple[str, str, dict]] = []
    seen_pairs: set = set()
    for src in fts_hits_per_node.keys() | vec_hits_per_node.keys():
        if not g.has_node(src):
            continue
        src_lang = _node_language(g, src)
        fused = rrf_fuse(
            fts_hits_per_node.get(src, []),
            vec_hits_per_node.get(src, []),
            k=k,
        )
        for hit in fused[:top_n]:
            if hit["rrf_score"] < threshold:
                break
            tgt = hit.get("node_id")
            if not tgt or tgt == src or not g.has_node(tgt):
                continue
            if _node_language(g, tgt) == src_lang:
                continue
            pair = (src, tgt)
            if pair in seen_pairs:
                continue
            seen_pairs.add(pair)
            weight = _clamp(0.5 * hit["rrf_score"] * _locality_factor(g, src, tgt))
            out.append((src, tgt, {
                "relationship_type": "cross_language_L2",
                "edge_class": "cross_language",
                "weight": weight,
                "provenance": {
                    "source": "cross_language_linker",
                    "level": "L2",
                    "matcher": "hybrid_rrf",
                    "rrf_score": hit["rrf_score"],
                },
            }))
    return out


# ──────────────────────────────────────────────────────────────────────
# L3 — Containment + member-uses promotion
# ──────────────────────────────────────────────────────────────────────


def link_l3_containment(
    g: nx.MultiDiGraph,
    parent_links: Sequence[Tuple[str, str, dict]],
) -> List[Tuple[str, str, dict]]:
    """Promote contained-member uses based on already-discovered parent links.

    Inputs are L0/L1/L2 edges. For each parent link (A → B), if A
    contains member ``A.m`` and B contains member ``B.m`` (matched by
    simple symbol name), emit a child link with weight ``0.6 *
    parent_weight``.
    """
    if not parent_links:
        return []

    members_by_parent: Dict[str, Dict[str, str]] = defaultdict(dict)
    for nid, data in g.nodes(data=True):
        parent = data.get("parent_symbol")
        if parent and parent in g:
            simple = (data.get("symbol_name") or "").split(".")[-1]
            if simple:
                members_by_parent[parent][simple] = nid

    out: List[Tuple[str, str, dict]] = []
    seen_pairs: set = set()
    for src, tgt, attrs in parent_links:
        parent_w = float(attrs.get("weight", 0.5))
        src_members = members_by_parent.get(src, {})
        tgt_members = members_by_parent.get(tgt, {})
        if not src_members or not tgt_members:
            continue
        for name in src_members.keys() & tgt_members.keys():
            child_src = src_members[name]
            child_tgt = tgt_members[name]
            if child_src == child_tgt:
                continue
            if _node_language(g, child_src) == _node_language(g, child_tgt):
                continue
            pair = (child_src, child_tgt)
            if pair in seen_pairs:
                continue
            seen_pairs.add(pair)
            out.append((child_src, child_tgt, {
                "relationship_type": "cross_language_L3",
                "edge_class": "cross_language",
                "weight": _clamp(0.6 * parent_w),
                "provenance": {
                    "source": "cross_language_linker",
                    "level": "L3",
                    "matcher": "containment_member",
                    "parent_pair": [src, tgt],
                },
            }))
    return out


# ──────────────────────────────────────────────────────────────────────
# Entry point
# ──────────────────────────────────────────────────────────────────────


def run_cross_language_linker(
    g: nx.MultiDiGraph,
    *,
    cross_language_relationships: Optional[Sequence[dict]] = None,
    surfaces_by_node: Optional[Dict[str, List[APISurface]]] = None,
    fts_hits_per_node: Optional[Dict[str, List[dict]]] = None,
    vec_hits_per_node: Optional[Dict[str, List[dict]]] = None,
    flags: Optional[FeatureFlags] = None,
) -> List[Tuple[str, str, dict]]:
    """Run all four levels and return the merged edge list.

    Pure function — does NOT mutate the graph. The caller writes edges
    via ``g.add_edge(*e)`` once it has decided to apply the result.
    Duplicate (src, tgt, relationship_type) triples are dropped.
    """
    flags = flags or get_feature_flags()
    edges: List[Tuple[str, str, dict]] = []

    edges.extend(link_l0_exact(g, cross_language_relationships or []))
    edges.extend(link_l1_api_surface(g, surfaces_by_node or {}))
    edges.extend(link_l2_hybrid(
        g, fts_hits_per_node or {}, vec_hits_per_node or {}
    ))
    # L3 runs over the union of L0/L1/L2 results so far.
    edges.extend(link_l3_containment(g, list(edges)))

    best_by_key: dict[tuple[str, str, Any], Tuple[str, str, dict]] = {}
    for src, tgt, attrs in edges:
        key = (src, tgt, attrs.get("relationship_type"))
        current = best_by_key.get(key)
        if current is None or _edge_specificity(attrs) > _edge_specificity(current[2]):
            best_by_key[key] = (src, tgt, attrs)
    return list(best_by_key.values())
