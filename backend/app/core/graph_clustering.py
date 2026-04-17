"""
Graph-based community detection for wiki structure.

Deterministically partitions a codebase graph into sections (macro-clusters)
and pages (micro-clusters) using hierarchical Leiden community detection,
eliminating the need for expensive LLM-driven exploratory tool calls.

Pipeline:

1. **Architectural projection** — collapse methods/fields/variables into
   their parent class/function so community detection sees only top-level
   structure.
2. **Hub detection** — identify high-connectivity nodes (loggers, base
   classes, config singletons) that distort clustering and quarantine them.
3. **File-level contraction** — contract the projected graph to file-level
   nodes (each ``rel_path`` becomes one node) to eliminate parameter /
   variable / field noise.
4. **Leiden pass 1 (sections)** — run Leiden on the file-contracted graph
   at ``LEIDEN_FILE_SECTION_RESOLUTION`` to produce macro-clusters.
5. **Section consolidation** — merge small sections using directory
   proximity until count ≤ target (log₂-based on file count).
6. **Leiden pass 2 (pages)** — per-section Leiden on the original-node
   subgraph at ``LEIDEN_PAGE_RESOLUTION`` to produce micro-clusters.
7. **Page consolidation** — merge small pages globally using directory
   proximity until total ≤ target (√-based on node count).
8. **Hub re-integration** — plurality-vote assignment of quarantined hubs
   to most-connected cluster.

Ported from the DeepWiki plugin ``graph_clustering.py``, keeping only the
hierarchical Leiden path (legacy Louvain pipeline removed).  All DB-specific
functions (persist, load_hubs_from_db) are excluded — this module operates
purely on in-memory NetworkX graphs.

Usage::

    from app.core.graph_clustering import run_clustering

    result = run_clustering(G, exclude_tests=True)
    print(result.section_count, result.page_count)
"""

import logging
import math
from collections import Counter
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple

import networkx as nx

from .cluster_constants import (
    CENTROID_ARCHITECTURAL_MIN_PRIORITY,
    DEFAULT_HUB_Z_THRESHOLD,
    DEFAULT_MAX_PAGE_SIZE,
    DOC_DOMINANT_THRESHOLD,
    LEIDEN_FILE_SECTION_RESOLUTION,
    LEIDEN_PAGE_RESOLUTION,
    MAX_CENTROIDS,
    MERGE_THRESHOLD,
    MIN_CENTROIDS,
    MIN_PAGE_SIZE,
    SYMBOL_TYPE_PRIORITY,
    adaptive_max_page_size,
    is_test_path,
    target_section_count,
    target_total_pages,
)
from .constants import ARCHITECTURAL_SYMBOLS, DOC_SYMBOL_TYPES

try:
    import igraph as ig

    _HAS_IGRAPH = True
except ImportError:
    _HAS_IGRAPH = False

try:
    import leidenalg

    _HAS_LEIDEN = True
except ImportError:
    _HAS_LEIDEN = False

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════
# Data Structures
# ═══════════════════════════════════════════════════════════════════════════


@dataclass
class ClusterAssignment:
    """Immutable result of the clustering pipeline.

    Attributes:
        macro_clusters: Mapping ``{node_id: section_id}`` for every
            clustered node (including reintegrated hubs).
        micro_clusters: Nested mapping
            ``{section_id: {node_id: page_id}}``.
        hub_nodes: Set of node IDs identified as high-connectivity hubs.
        hub_assignments: ``{hub_id: (section_id, page_id | None)}``.
        sections: ``{section_id: {"pages": {page_id: [node_ids]},
            "centroids": {page_id: [centroid_dicts]}}}``.
        section_count: Number of final sections after consolidation.
        page_count: Total number of final pages across all sections.
        algorithm: Algorithm identifier string.
        resolution: ``{"section": γ_sec, "page": γ_pg}``.
        stats: Raw algorithm metadata and pipeline statistics.
    """

    macro_clusters: Dict[str, int]
    micro_clusters: Dict[int, Dict[str, int]]
    hub_nodes: Set[str]
    hub_assignments: Dict[str, Tuple[int, Optional[int]]]
    sections: Dict[int, Dict[str, Any]]
    section_count: int
    page_count: int
    algorithm: str
    resolution: Dict[str, float]
    stats: Dict[str, Any] = field(default_factory=dict)


# ═══════════════════════════════════════════════════════════════════════════
# Internal Constants
# ═══════════════════════════════════════════════════════════════════════════

# Symbol types that are always sub-components of a parent (never standalone
# cluster members).  If their parent is an architectural node the edge is
# "promoted" to that parent; otherwise the node is kept as-is.
_CHILD_SYMBOL_TYPES = frozenset(
    {
        "method",
        "constructor",
        "destructor",
        "field",
        "variable",
        "property",
        "getter",
        "setter",
    }
)


# ═══════════════════════════════════════════════════════════════════════════
# 0. Architectural Projection
# ═══════════════════════════════════════════════════════════════════════════


def _find_architectural_parent(
    node_id: str,
    node_data: Dict[str, Any],
    arch_node_ids: Set[str],
    arch_by_file_sym: Dict[tuple, str],
    graph: nx.MultiDiGraph,
) -> Optional[str]:
    """Resolve a child node to its parent architectural node.

    Three-strategy cascade:
    1. Node ID convention: ``lang::file::Parent.child`` → ``lang::file::Parent``
    2. ``parent_symbol`` attribute lookup via ``(file, name)`` index
    3. Returns ``None`` if no parent can be resolved

    Args:
        node_id: The child node ID.
        node_data: Node attributes dict.
        arch_node_ids: Set of known architectural node IDs.
        arch_by_file_sym: Index ``(file_name, symbol_name) → node_id``
            for architectural nodes.
        graph: The full graph (for ``has_node`` checks).

    Returns:
        Parent node ID or ``None``.
    """
    parts = node_id.split("::", 2)
    if len(parts) != 3:
        return None

    lang, fname, qualified = parts

    # Strategy 1: node_id naming convention
    if "." in qualified:
        parent_name = qualified.rsplit(".", 1)[0]
        parent_nid = f"{lang}::{fname}::{parent_name}"
        if parent_nid in arch_node_ids or graph.has_node(parent_nid):
            return parent_nid

    # Strategy 2: parent_symbol attribute
    psym = node_data.get("parent_symbol")
    if psym:
        candidate = arch_by_file_sym.get((fname, psym))
        if candidate:
            return candidate

    return None


def architectural_projection(
    G: nx.MultiDiGraph,
) -> nx.MultiDiGraph:
    """Project *G* down to architectural-only nodes for community detection.

    Methods, constructors, fields, and variables inflate the node count by
    5–15× compared to the top-level symbols that actually define code
    structure.  Running community detection on the full graph produces
    far too many communities for small-to-medium repos.

    This function:

    1. Keeps every node whose ``symbol_type`` is in
       :data:`ARCHITECTURAL_SYMBOLS`.
    2. For each *child* node (method, constructor, field, variable)
       resolves its parent architectural node using
       :func:`_find_architectural_parent`.
    3. Re-maps every edge so that child endpoints are replaced by their
       parent.  Edges that become self-loops are dropped.  Weights are
       summed.

    The result is a much smaller graph that still captures inter-module
    connectivity and produces sensible Leiden communities.
    """
    # ── 1. Classify nodes ──
    arch_nodes: Set[str] = set()
    child_to_parent: Dict[str, str] = {}

    # Build a fast lookup: (file_name, symbol_name) → node_id for arch nodes
    arch_by_file_sym: Dict[tuple, str] = {}

    for nid, data in G.nodes(data=True):
        stype = (data.get("symbol_type") or "").lower()
        if stype in ARCHITECTURAL_SYMBOLS:
            arch_nodes.add(nid)
            fname = data.get("file_name", "")
            sname = data.get("symbol_name", "")
            if fname and sname:
                arch_by_file_sym[(fname, sname)] = nid

    for nid, data in G.nodes(data=True):
        if nid in arch_nodes:
            continue
        stype = (data.get("symbol_type") or "").lower()
        if stype not in _CHILD_SYMBOL_TYPES:
            # Keep non-child, non-architectural nodes as-is (e.g. 'inferred',
            # 'unknown', 'module').  They still participate in clustering.
            arch_nodes.add(nid)
            continue

        parent_nid = _find_architectural_parent(
            nid, data, arch_nodes, arch_by_file_sym, G
        )
        if parent_nid:
            child_to_parent[nid] = parent_nid
            arch_nodes.add(parent_nid)
        else:
            # No resolvable parent — keep as standalone cluster member
            arch_nodes.add(nid)

    projected = len(child_to_parent)
    remaining = len(arch_nodes)
    logger.info(
        "Architectural projection: %d → %d nodes (%d children promoted to parents)",
        G.number_of_nodes(),
        remaining,
        projected,
    )

    # ── 2. Build projected graph ──
    P = nx.MultiDiGraph()
    for nid in arch_nodes:
        if G.has_node(nid):
            P.add_node(nid, **G.nodes[nid])

    def _remap(nid: str) -> Optional[str]:
        if nid in arch_nodes:
            return nid
        return child_to_parent.get(nid)

    for u, v, data in G.edges(data=True):
        mu = _remap(u)
        mv = _remap(v)
        if mu is None or mv is None:
            continue
        if mu == mv:
            continue  # internal edge (method→method in same class)
        P.add_edge(mu, mv, **data)

    return P


# ═══════════════════════════════════════════════════════════════════════════
# 1. Hub Detection (Z-Score on In-Degree)
# ═══════════════════════════════════════════════════════════════════════════


def detect_hubs(
    G: nx.MultiDiGraph,
    z_threshold: float = DEFAULT_HUB_Z_THRESHOLD,
) -> Set[str]:
    """Return node IDs whose in-degree Z-score exceeds *z_threshold*.

    A hub is a node that is imported/called by a disproportionate number
    of other nodes — e.g. loggers, base classes, config singletons.
    Including them in community detection distorts clusters because they
    pull unrelated nodes together.

    Args:
        G: The graph (after weighting).
        z_threshold: Standard deviations above mean to flag as hub.
            Default 3.0 (≈ top 0.13% under normal distribution).

    Returns:
        Set of hub node IDs.
    """
    if G.number_of_nodes() < 3:
        return set()

    nodes_list = list(G.nodes())
    degrees = [G.in_degree(n) for n in nodes_list]

    n = len(degrees)
    mean = sum(degrees) / n
    variance = sum((d - mean) ** 2 for d in degrees) / n
    std = math.sqrt(variance)

    if std == 0:
        return set()

    hubs = set()
    for i, deg in enumerate(degrees):
        z = (deg - mean) / std
        if z > z_threshold:
            hubs.add(nodes_list[i])

    if hubs:
        logger.info(
            "Hub detection: %d hubs found (Z > %.1f) out of %d nodes. "
            "Mean in-degree=%.1f, std=%.1f",
            len(hubs),
            z_threshold,
            G.number_of_nodes(),
            mean,
            std,
        )
    else:
        logger.info(
            "Hub detection: no hubs found (Z > %.1f). "
            "Mean in-degree=%.1f, std=%.1f",
            z_threshold,
            mean,
            std,
        )

    return hubs


# ═══════════════════════════════════════════════════════════════════════════
# 2. Graph Conversion Helpers
# ═══════════════════════════════════════════════════════════════════════════


def _to_weighted_undirected(G: nx.MultiDiGraph) -> nx.Graph:
    """Collapse directed multi-edges into an undirected weighted graph.

    For each pair (u, v), sums all parallel edge weights.
    """
    U = nx.Graph()
    U.add_nodes_from(G.nodes())

    edge_weights: Dict[tuple, float] = {}
    for u, v, data in G.edges(data=True):
        key = (min(u, v), max(u, v))
        edge_weights[key] = edge_weights.get(key, 0.0) + data.get("weight", 1.0)

    for (u, v), w in edge_weights.items():
        U.add_edge(u, v, weight=w)

    return U


def _nx_to_igraph(G_undirected: nx.Graph) -> Tuple["ig.Graph", List[str]]:
    """Convert a NetworkX undirected graph to an igraph Graph.

    Uses deterministic (sorted) node ordering so that results are
    reproducible across runs.

    Returns:
        (ig_graph, node_list) where node_list maps igraph vertex indices
        back to NetworkX node IDs.
    """
    node_list = sorted(G_undirected.nodes())
    node_to_idx = {nid: i for i, nid in enumerate(node_list)}

    ig_graph = ig.Graph(n=len(node_list), directed=False)
    edges = []
    weights = []
    for u, v, data in G_undirected.edges(data=True):
        edges.append((node_to_idx[u], node_to_idx[v]))
        weights.append(data.get("weight", 1.0))

    ig_graph.add_edges(edges)
    ig_graph.es["weight"] = weights
    return ig_graph, node_list


def _contract_to_file_graph(
    G: nx.MultiDiGraph,
) -> Tuple[nx.Graph, Dict[str, List[str]]]:
    """Contract the node graph to file-level for section clustering.

    Every unique ``rel_path`` becomes a single file-node.  For edges
    between nodes in *different* files the weight is summed into the
    file-pair edge.  Same-file edges are dropped (they are intra-node
    after contraction).

    This eliminates parameter/variable/field noise that fragments
    community detection: those nodes' cross-file edges become file-level
    edge weight without inflating the node count.

    Returns:
        (file_graph, file_to_nodes):
        - file_graph — undirected weighted ``nx.Graph``
        - file_to_nodes — ``{rel_path: [node_id, ...]}``
    """
    file_to_nodes: Dict[str, List[str]] = {}
    node_to_file: Dict[str, str] = {}

    for nid, data in G.nodes(data=True):
        rel_path = data.get("rel_path") or data.get("file_name") or "<unknown>"
        file_to_nodes.setdefault(rel_path, []).append(nid)
        node_to_file[nid] = rel_path

    # Sum edge weights between distinct file pairs
    edge_weights: Dict[Tuple[str, str], float] = {}
    for u, v, data in G.edges(data=True):
        fu = node_to_file.get(u)
        fv = node_to_file.get(v)
        if fu is None or fv is None or fu == fv:
            continue
        key = (min(fu, fv), max(fu, fv))
        edge_weights[key] = edge_weights.get(key, 0.0) + data.get("weight", 1.0)

    FG = nx.Graph()
    FG.add_nodes_from(file_to_nodes.keys())
    for (fu, fv), w in edge_weights.items():
        FG.add_edge(fu, fv, weight=w)

    return FG, file_to_nodes


# ═══════════════════════════════════════════════════════════════════════════
# 3. Directory Proximity Helpers
# ═══════════════════════════════════════════════════════════════════════════


def _dir_of_node(nid: str, G: nx.MultiDiGraph) -> str:
    """Get directory prefix from a node's rel_path."""
    data = G.nodes.get(nid, {})
    rel_path = data.get("rel_path") or data.get("file_name") or ""
    if "/" in rel_path:
        return rel_path.rsplit("/", 1)[0]
    return "<root>"


def _dir_histogram(node_ids, G: nx.MultiDiGraph) -> Counter:
    """Build a directory-frequency Counter for a set of node IDs."""
    c: Counter = Counter()
    for nid in node_ids:
        c[_dir_of_node(nid, G)] += 1
    return c


def _dir_similarity(a: Counter, b: Counter) -> int:
    """Shared directory entries (min-intersect) between two histograms."""
    return sum(min(a[d], b[d]) for d in a if d in b)


# ═══════════════════════════════════════════════════════════════════════════
# 4. Centroid Detection
# ═══════════════════════════════════════════════════════════════════════════


def _detect_page_centroids(
    G: nx.MultiDiGraph,
    page_node_ids: List[str],
    top_k: Optional[int] = None,
) -> List[Dict[str, Any]]:
    """Identify centroid nodes in a page community.

    Scoring:
        ``composite = degree_norm * 0.6  +  type_priority_norm * 0.4``

    Only architectural nodes (type priority ≥ CENTROID_ARCHITECTURAL_MIN_PRIORITY)
    are eligible.  Falls back to all nodes when none qualify.

    Returns:
        List of ``{"node_id": ..., "name": ..., "type": ..., "score": ...}``
        sorted by descending score.
    """
    if not page_node_ids:
        return []

    k = top_k or min(
        MAX_CENTROIDS,
        max(MIN_CENTROIDS, round(math.sqrt(len(page_node_ids)))),
    )

    scored = []
    for nid in page_node_ids:
        data = G.nodes.get(nid, {})
        stype = (data.get("symbol_type") or "unknown").lower()
        degree = G.in_degree(nid) + G.out_degree(nid)
        priority = SYMBOL_TYPE_PRIORITY.get(stype, 0)
        scored.append(
            {
                "node_id": nid,
                "name": data.get("symbol_name", nid),
                "type": stype,
                "degree": degree,
                "priority": priority,
            }
        )

    # Normalize degree and priority to [0, 1]
    max_deg = max((s["degree"] for s in scored), default=1) or 1
    max_pri = max((s["priority"] for s in scored), default=1) or 1

    for s in scored:
        s["score"] = 0.6 * (s["degree"] / max_deg) + 0.4 * (s["priority"] / max_pri)

    # Filter to architectural-only candidates
    arch = [s for s in scored if s["priority"] >= CENTROID_ARCHITECTURAL_MIN_PRIORITY]
    pool = arch if arch else scored
    pool.sort(key=lambda x: x["score"], reverse=True)

    result = pool[:k]

    # Re-normalize scores so the top centroid = 1.0
    top_score = result[0]["score"] if result else 1.0
    return [
        {
            "node_id": s["node_id"],
            "name": s["name"],
            "type": s["type"],
            "score": round(s["score"] / top_score, 4) if top_score else 0.0,
        }
        for s in result
    ]


def select_central_symbols(
    G: nx.MultiDiGraph,
    cluster_nodes: Set[str],
    k: int = 5,
) -> List[str]:
    """Select the top-*k* most central symbols in a cluster using PageRank.

    The returned node IDs represent the "representative" symbols that best
    characterise the cluster.

    Uses PageRank on the **undirected** projection so that hub nodes (high
    out-degree) rank equally with authority nodes (high in-degree).

    Args:
        G: Full graph (we extract the subgraph for *cluster_nodes*).
        cluster_nodes: Set of node IDs belonging to this cluster.
        k: Number of central symbols to return.

    Returns:
        List of node IDs sorted by descending PageRank, length ≤ k.
    """
    if not cluster_nodes:
        return []

    k = min(k, len(cluster_nodes))

    sub = G.subgraph(cluster_nodes)
    undirected = sub.to_undirected()

    try:
        pr = nx.pagerank(undirected, weight="weight", max_iter=100, tol=1e-6)
    except nx.PowerIterationFailedConvergence:
        pr = {n: 1.0 / len(cluster_nodes) for n in cluster_nodes}

    ranked = sorted(pr.items(), key=lambda x: x[1], reverse=True)
    return [nid for nid, _ in ranked[:k]]


# ═══════════════════════════════════════════════════════════════════════════
# 5. Doc-Cluster Detection
# ═══════════════════════════════════════════════════════════════════════════


def detect_doc_clusters(
    sections: Dict[int, Dict[str, Any]],
    G: nx.MultiDiGraph,
) -> Dict[int, float]:
    """Identify sections that are predominantly documentation.

    A section is "doc-dominant" when more than
    :data:`DOC_DOMINANT_THRESHOLD` of its nodes have a symbol type
    in :data:`DOC_SYMBOL_TYPES`.

    Args:
        sections: ``{section_id: {"pages": {page_id: [node_ids]}}}``.
        G: The graph with node attributes.

    Returns:
        ``{section_id: doc_ratio}`` for each doc-dominant section.
    """
    result: Dict[int, float] = {}

    for sec_id, sec_data in sections.items():
        total = 0
        doc_count = 0
        for node_ids in sec_data["pages"].values():
            for nid in node_ids:
                total += 1
                data = G.nodes.get(nid, {})
                stype = (data.get("symbol_type") or "").lower()
                if stype in DOC_SYMBOL_TYPES:
                    doc_count += 1

        if total > 0:
            ratio = doc_count / total
            if ratio > DOC_DOMINANT_THRESHOLD:
                result[sec_id] = ratio

    return result


# ═══════════════════════════════════════════════════════════════════════════
# 6. Hub Re-Integration
# ═══════════════════════════════════════════════════════════════════════════


def reintegrate_hubs(
    G: nx.MultiDiGraph,
    hubs: Set[str],
    macro_assignments: Dict[str, int],
    micro_assignments: Optional[Dict[int, Dict[str, int]]] = None,
) -> Dict[str, Tuple[int, Optional[int]]]:
    """Assign quarantined hubs to their most-connected cluster.

    For each hub, count edges (in + out) to each macro-cluster and
    assign to the one with the most connections (plurality wins).
    Hubs with zero edges go to the largest macro-cluster.

    When *micro_assignments* is provided, each hub also gets the
    micro-cluster within its target macro that has the most edge
    connections.

    Returns:
        ``{hub_node_id: (macro_id, micro_id | None)}``.
    """
    hub_results: Dict[str, Tuple[int, Optional[int]]] = {}

    # Fallback: largest macro cluster (by node count)
    if macro_assignments:
        cluster_sizes: Counter = Counter(macro_assignments.values())
        largest_macro = cluster_sizes.most_common(1)[0][0]
    else:
        largest_macro = 0

    for hub in hubs:
        if not G.has_node(hub):
            continue

        cluster_edges: Counter = Counter()

        # Outgoing edges
        for _, neighbor in G.out_edges(hub):
            if neighbor in macro_assignments:
                cluster_edges[macro_assignments[neighbor]] += 1

        # Incoming edges
        for predecessor, _ in G.in_edges(hub):
            if predecessor in macro_assignments:
                cluster_edges[macro_assignments[predecessor]] += 1

        if not cluster_edges:
            target_macro = largest_macro
        else:
            target_macro = cluster_edges.most_common(1)[0][0]

        # Determine best micro-cluster within the target macro
        target_micro: Optional[int] = None
        if micro_assignments and target_macro in micro_assignments:
            micro_map = micro_assignments[target_macro]
            micro_edges: Counter = Counter()

            for _, neighbor in G.out_edges(hub):
                if neighbor in micro_map:
                    micro_edges[micro_map[neighbor]] += 1
            for predecessor, _ in G.in_edges(hub):
                if predecessor in micro_map:
                    micro_edges[micro_map[predecessor]] += 1

            if micro_edges:
                target_micro = micro_edges.most_common(1)[0][0]
            else:
                # No direct micro edges — pick the largest micro in this macro
                micro_sizes: Counter = Counter(micro_map.values())
                target_micro = micro_sizes.most_common(1)[0][0]

        hub_results[hub] = (target_macro, target_micro)

    logger.info(
        "Hub re-integration: %d hubs assigned to clusters (plurality vote)",
        len(hub_results),
    )
    return hub_results


# ═══════════════════════════════════════════════════════════════════════════
# 7. Consolidation (post-Leiden merge passes)
# ═══════════════════════════════════════════════════════════════════════════


def _consolidate_sections(
    leiden_result: Dict[str, Any],
    G: nx.MultiDiGraph,
    n_files: Optional[int] = None,
) -> Dict[str, Any]:
    """Merge small Leiden sections into neighbours using directory proximity.

    Iteratively picks the smallest section and merges it into the section
    with the most shared directory prefixes.  Stops when the number of
    sections reaches the adaptive target.

    Args:
        leiden_result: Mutable dict with ``sections``, ``macro_assignments``,
            ``micro_assignments``.
        G: The graph (for directory lookups).
        n_files: File count for target calculation.  When ``None``,
            falls back to the node count.

    Mutates *leiden_result* in-place and returns it.
    """
    sections = leiden_result["sections"]
    macro_assign = leiden_result["macro_assignments"]
    micro_assign = leiden_result["micro_assignments"]

    scale = n_files if n_files is not None else len(macro_assign)
    target = target_section_count(scale)

    if len(sections) <= target:
        return leiden_result

    original_count = len(sections)

    # Pre-compute per-section directory histograms and sizes
    sec_dirs: Dict[int, Counter] = {}
    sec_sizes: Dict[int, int] = {}
    for sid, sec in sections.items():
        all_nids = [nid for pg in sec["pages"].values() for nid in pg]
        sec_dirs[sid] = _dir_histogram(all_nids, G)
        sec_sizes[sid] = len(all_nids)

    while len(sections) > target:
        smallest_id = min(sec_sizes, key=sec_sizes.get)
        dirs_a = sec_dirs[smallest_id]

        best_target = None
        best_score = -1
        for sid in sections:
            if sid == smallest_id:
                continue
            score = _dir_similarity(dirs_a, sec_dirs[sid])
            if score > best_score or (
                score == best_score
                and (best_target is None or sec_sizes[sid] < sec_sizes[best_target])
            ):
                best_score = score
                best_target = sid

        if best_target is None:
            break

        # Merge all pages from source into target
        target_pages = sections[best_target]["pages"]
        source_pages = sections[smallest_id]["pages"]
        next_pg = max(target_pages.keys(), default=-1) + 1

        for _old_pg, node_ids in source_pages.items():
            target_pages[next_pg] = node_ids
            for nid in node_ids:
                macro_assign[nid] = best_target
                micro_assign.setdefault(best_target, {})[nid] = next_pg
            next_pg += 1

        micro_assign.pop(smallest_id, None)

        # Update bookkeeping
        for d, c in dirs_a.items():
            sec_dirs[best_target][d] = sec_dirs[best_target].get(d, 0) + c
        del sec_dirs[smallest_id]
        sec_sizes[best_target] += sec_sizes.pop(smallest_id)
        del sections[smallest_id]

    logger.info(
        "Section consolidation: %d → %d sections (target %d, scale %d %s)",
        original_count,
        len(sections),
        target,
        scale,
        "files" if n_files is not None else "nodes",
    )
    return leiden_result


def _consolidate_pages(
    leiden_result: Dict[str, Any],
    G: nx.MultiDiGraph,
) -> Dict[str, Any]:
    """Merge small pages globally using directory proximity.

    Picks the smallest page across *all* sections, merges it into its
    best same-section neighbour.  Repeats until total pages ≤ target.

    Page target uses **node count** (code volume), while section target
    uses **file count** (structural domains).  Different inputs +
    different growth rates (√ vs log₂) break the proportional lock.

    Mutates *leiden_result* in-place and returns it.
    """
    sections = leiden_result["sections"]
    micro_assign = leiden_result["micro_assignments"]

    n_nodes = len(leiden_result["macro_assignments"])
    target_total = target_total_pages(n_nodes)

    total_pages = sum(len(s["pages"]) for s in sections.values())
    if total_pages <= target_total:
        return leiden_result

    original_total = total_pages

    # Build global page-size and page-dir indexes: (sid, pid) → size/dirs
    pg_sizes: Dict[tuple, int] = {}
    pg_dirs: Dict[tuple, Counter] = {}

    for sid, sec in sections.items():
        for pid, nids in sec["pages"].items():
            key = (sid, pid)
            pg_sizes[key] = len(nids)
            pg_dirs[key] = _dir_histogram(nids, G)

    while total_pages > target_total:
        candidates = sorted(pg_sizes.items(), key=lambda x: x[1])

        merged = False
        for (sid, pid), _size in candidates:
            sec_pages = sections[sid]["pages"]
            if len(sec_pages) <= 1:
                continue

            dirs_a = pg_dirs[(sid, pid)]

            best_pg = None
            best_score = -1
            for other_pid in sec_pages:
                if other_pid == pid:
                    continue
                score = _dir_similarity(dirs_a, pg_dirs[(sid, other_pid)])
                if score > best_score or (
                    score == best_score
                    and (
                        best_pg is None
                        or pg_sizes[(sid, other_pid)] < pg_sizes.get((sid, best_pg), 0)
                    )
                ):
                    best_score = score
                    best_pg = other_pid

            if best_pg is None:
                continue

            # Merge
            sec_pages[best_pg].extend(sec_pages.pop(pid))
            for nid in sec_pages[best_pg]:
                micro_assign.setdefault(sid, {})[nid] = best_pg

            for d, c in dirs_a.items():
                pg_dirs[(sid, best_pg)][d] = pg_dirs[(sid, best_pg)].get(d, 0) + c
            del pg_dirs[(sid, pid)]
            pg_sizes[(sid, best_pg)] += pg_sizes.pop((sid, pid))

            total_pages -= 1
            merged = True
            break

        if not merged:
            break

    new_total = sum(len(s["pages"]) for s in sections.values())
    logger.info(
        "Page consolidation: %d → %d pages (target %d)",
        original_total,
        new_total,
        target_total,
    )
    return leiden_result


# ═══════════════════════════════════════════════════════════════════════════
# 7b. Section Expansion (split under-count sections to reach target)
# ═══════════════════════════════════════════════════════════════════════════


def _expand_sections(
    leiden_result: Dict[str, Any],
    G: nx.MultiDiGraph,
    n_files: Optional[int] = None,
) -> Dict[str, Any]:
    """Split large sections using sub-Leiden until section count reaches target.

    Consolidation only *merges* down (too many → target).  This function
    handles the opposite case: when Leiden produces *fewer* sections than
    the adaptive target, the largest sections are split via higher-resolution
    Leiden on their file-contracted subgraph.

    Mutates *leiden_result* in-place and returns it.
    """
    if not _HAS_IGRAPH or not _HAS_LEIDEN:
        return leiden_result

    sections = leiden_result["sections"]
    macro_assign = leiden_result["macro_assignments"]
    micro_assign = leiden_result["micro_assignments"]

    scale = n_files if n_files is not None else len(macro_assign)
    target = target_section_count(scale)

    if len(sections) >= target:
        return leiden_result

    original_count = len(sections)
    next_section_id = max(sections.keys(), default=-1) + 1

    # Build per-section node sets (for quick size lookups)
    sec_node_sets: Dict[int, Set[str]] = {}
    for nid, sid in macro_assign.items():
        sec_node_sets.setdefault(sid, set()).add(nid)

    while len(sections) < target:
        # Pick the largest section (by node count) that has enough nodes to split
        splittable = [
            (sid, len(nset))
            for sid, nset in sec_node_sets.items()
            if len(nset) >= 4  # need at least 4 nodes to produce 2 meaningful parts
        ]
        if not splittable:
            break

        splittable.sort(key=lambda x: -x[1])
        split_sid, _ = splittable[0]
        split_nodes = sec_node_sets[split_sid]

        # Build file-contracted subgraph for this section
        sub_G = G.subgraph(split_nodes).copy()
        sub_file_graph, sub_file_to_nodes = _contract_to_file_graph(sub_G)

        if sub_file_graph.number_of_edges() == 0:
            # No cross-file edges — fall back to directory-based split
            split_result = _directory_based_split(split_nodes, G)
            if len(split_result) <= 1:
                break  # Can't split further
        else:
            sub_undirected = sub_file_graph
            ig_sub, file_list = _nx_to_igraph(sub_undirected)

            # Use higher resolution (2.0) to encourage more communities
            sub_partition = leidenalg.find_partition(
                ig_sub,
                leidenalg.RBConfigurationVertexPartition,
                resolution_parameter=2.0,
                weights="weight",
                seed=42,
            )

            # Group files by community
            file_communities: Dict[int, List[str]] = {}
            for idx, comm_id in enumerate(sub_partition.membership):
                file_communities.setdefault(comm_id, []).append(file_list[idx])

            if len(file_communities) <= 1:
                # Leiden couldn't split even at γ=2.0 — try directory split
                split_result = _directory_based_split(split_nodes, G)
                if len(split_result) <= 1:
                    break
            else:
                # Expand file communities to node communities
                split_result = []
                for _comm_id, files in file_communities.items():
                    nodes_in_comm: Set[str] = set()
                    for f in files:
                        nodes_in_comm.update(sub_file_to_nodes.get(f, []))
                    # Filter to only nodes in our section
                    nodes_in_comm &= split_nodes
                    if nodes_in_comm:
                        split_result.append(nodes_in_comm)

                if len(split_result) <= 1:
                    break

        # Apply the split: keep the first part in split_sid, create new sections for rest
        parts = sorted(split_result, key=lambda s: -len(s))  # Largest first
        keep_nodes = parts[0]

        # Update the original section to keep only the first part
        new_pages_for_kept: Dict[int, List[str]] = {}
        kept_micro: Dict[str, int] = {}
        pg_id = 0
        for old_pg_id, old_nids in sections[split_sid]["pages"].items():
            kept = [nid for nid in old_nids if nid in keep_nodes]
            if kept:
                new_pages_for_kept[pg_id] = kept
                for nid in kept:
                    kept_micro[nid] = pg_id
                pg_id += 1

        sections[split_sid]["pages"] = new_pages_for_kept
        if "centroids" in sections[split_sid]:
            del sections[split_sid]["centroids"]
        micro_assign[split_sid] = kept_micro
        sec_node_sets[split_sid] = keep_nodes

        # Create new sections from remaining parts
        for part_nodes in parts[1:]:
            new_sid = next_section_id
            next_section_id += 1

            # Group part_nodes by their original micro assignment
            old_micro_groups: Dict[int, List[str]] = {}
            for nid in part_nodes:
                old_mid = None
                old_sec_micro = micro_assign.get(split_sid)
                if isinstance(old_sec_micro, dict):
                    old_mid = old_sec_micro.get(nid)
                if old_mid is None:
                    old_mid = 0
                old_micro_groups.setdefault(old_mid, []).append(nid)

            new_pages: Dict[int, List[str]] = {}
            new_micro: Dict[str, int] = {}
            new_pg_id = 0
            for _old_mid, nids in old_micro_groups.items():
                new_pages[new_pg_id] = nids
                for nid in nids:
                    new_micro[nid] = new_pg_id
                    macro_assign[nid] = new_sid
                new_pg_id += 1

            sections[new_sid] = {"pages": new_pages}
            micro_assign[new_sid] = new_micro
            sec_node_sets[new_sid] = part_nodes

            if len(sections) >= target:
                break

    logger.info(
        "Section expansion: %d → %d sections (target %d, scale %d %s)",
        original_count,
        len(sections),
        target,
        scale,
        "files" if n_files is not None else "nodes",
    )
    return leiden_result


def _directory_based_split(
    nodes: Set[str],
    G: nx.MultiDiGraph,
) -> List[Set[str]]:
    """Split a set of nodes into groups by top-level directory.

    Fallback when Leiden cannot split the subgraph further.
    Returns at least 2 groups if nodes span multiple directories.
    """
    dir_groups: Dict[str, Set[str]] = {}
    for nid in nodes:
        d = _dir_of_node(nid, G)
        dir_groups.setdefault(d, set()).add(nid)

    if len(dir_groups) <= 1:
        return [nodes]

    # Return the groups sorted by size (largest first)
    return sorted(dir_groups.values(), key=lambda s: -len(s))


# ═══════════════════════════════════════════════════════════════════════════
# 7c. Page Size Enforcement (merge small, split oversized pages)
# ═══════════════════════════════════════════════════════════════════════════


def _enforce_page_sizes(
    leiden_result: Dict[str, Any],
    G: nx.MultiDiGraph,
) -> Dict[str, Any]:
    """Merge tiny pages and split oversized pages within each section.

    Uses adaptive_max_page_size() based on total node count.
    Ported from deepwiki's legacy Louvain pipeline — adapted for the
    Leiden result dict structure.

    Mutates *leiden_result* in-place and returns it.
    """
    sections = leiden_result["sections"]
    micro_assign = leiden_result["micro_assignments"]
    n_nodes = len(leiden_result["macro_assignments"])

    max_size = adaptive_max_page_size(n_nodes)
    min_size = MIN_PAGE_SIZE
    merge_thresh = MERGE_THRESHOLD

    total_before = sum(len(s["pages"]) for s in sections.values())

    for sec_id in list(sections.keys()):
        pages = sections[sec_id]["pages"]
        sec_micro = micro_assign.get(sec_id, {})

        # ── Merge small pages (only when >1 page) ──
        if len(pages) > 1:
            changed = True
            while changed:
                changed = False
                small_pids = [
                    pid for pid, nids in pages.items()
                    if len(nids) < merge_thresh and len(pages) > 1
                ]
                for small_pid in small_pids:
                    if small_pid not in pages or len(pages) <= 1:
                        break
                    small_nids = pages[small_pid]
                    small_dirs = _dir_histogram(small_nids, G)

                    best_pid = None
                    best_score = -1
                    for other_pid, other_nids in pages.items():
                        if other_pid == small_pid:
                            continue
                        score = _dir_similarity(small_dirs, _dir_histogram(other_nids, G))
                        if score > best_score:
                            best_score = score
                            best_pid = other_pid

                    if best_pid is not None:
                        pages[best_pid].extend(pages.pop(small_pid))
                        for nid in pages[best_pid]:
                            sec_micro[nid] = best_pid
                        changed = True

        # ── Split oversized pages ──
        new_pages: Dict[int, List[str]] = {}
        next_pg = 0
        for pid, nids in sorted(pages.items()):
            if len(nids) <= max_size:
                new_pages[next_pg] = nids
                for nid in nids:
                    sec_micro[nid] = next_pg
                next_pg += 1
            else:
                # Split by file-aware chunking
                chunks = _file_aware_split(nids, G, max_size)
                for chunk in chunks:
                    new_pages[next_pg] = chunk
                    for nid in chunk:
                        sec_micro[nid] = next_pg
                    next_pg += 1

        sections[sec_id]["pages"] = new_pages
        if "centroids" in sections[sec_id]:
            del sections[sec_id]["centroids"]
        micro_assign[sec_id] = sec_micro

    total_after = sum(len(s["pages"]) for s in sections.values())
    logger.info(
        "Page size enforcement: %d → %d pages (max_size=%d, min_merge=%d)",
        total_before,
        total_after,
        max_size,
        merge_thresh,
    )
    return leiden_result


def _file_aware_split(
    node_ids: List[str],
    G: nx.MultiDiGraph,
    max_size: int,
) -> List[List[str]]:
    """Split a list of nodes by file path, then chunk by declaration order.

    Keeps nodes from the same file together. If a single file exceeds
    max_size, splits by start_line order.
    """
    from collections import defaultdict

    by_file: Dict[str, List[str]] = defaultdict(list)
    for nid in node_ids:
        data = G.nodes.get(nid, {})
        fpath = data.get("rel_path") or data.get("file_path") or ""
        by_file[fpath].append(nid)

    # Sort each file's nodes by start_line
    for fpath in by_file:
        by_file[fpath].sort(
            key=lambda n: G.nodes.get(n, {}).get("start_line", 0)
        )

    result: List[List[str]] = []
    current_chunk: List[str] = []

    for fpath in sorted(by_file):
        file_nodes = by_file[fpath]
        if len(file_nodes) > max_size:
            # Flush current chunk
            if current_chunk:
                result.append(current_chunk)
                current_chunk = []
            # Split large file by line order
            for i in range(0, len(file_nodes), max_size):
                result.append(file_nodes[i:i + max_size])
        elif len(current_chunk) + len(file_nodes) > max_size:
            # Flush current chunk, start new one
            if current_chunk:
                result.append(current_chunk)
            current_chunk = list(file_nodes)
        else:
            current_chunk.extend(file_nodes)

    if current_chunk:
        result.append(current_chunk)

    return result if result else [node_ids]


# ═══════════════════════════════════════════════════════════════════════════
# 8. Hierarchical Leiden Clustering
# ═══════════════════════════════════════════════════════════════════════════


def _hierarchical_leiden(
    G_projected: nx.MultiDiGraph,
    hubs: Set[str],
    section_resolution: float = LEIDEN_FILE_SECTION_RESOLUTION,
    page_resolution: float = LEIDEN_PAGE_RESOLUTION,
    seed: int = 42,
) -> Dict[str, Any]:
    """Two-pass Leiden with file-level contraction for sections.

    **Pass 1 (sections)** — contracts the full graph to file-level nodes
    (each ``rel_path`` becomes one node, cross-file edge weights are
    summed), then runs Leiden at *section_resolution*.

    **Pass 2 (pages)** — for each section, runs Leiden at
    *page_resolution* on the **original node** subgraph.

    Args:
        G_projected: Code graph (hub nodes are excluded internally).
        hubs: Hub node IDs (excluded from clustering, reintegrated later).
        section_resolution: Leiden γ for the file-contracted section pass.
        page_resolution: Leiden γ for the per-section page pass.
        seed: Random seed for reproducibility.

    Returns:
        ``{"sections": ..., "macro_assignments": ...,
           "micro_assignments": ..., "algorithm_metadata": ...}``
    """
    if not _HAS_IGRAPH or not _HAS_LEIDEN:
        raise RuntimeError(
            "Hierarchical Leiden requires igraph and leidenalg. "
            "Install with: pip install igraph leidenalg"
        )

    # Remove hubs for clustering
    non_hub_nodes = set(G_projected.nodes()) - hubs
    if not non_hub_nodes:
        return {
            "sections": {},
            "macro_assignments": {},
            "micro_assignments": {},
            "algorithm_metadata": {
                "algorithm": "hierarchical_leiden_file_contracted",
                "section_resolution": section_resolution,
                "page_resolution": page_resolution,
                "sections": 0,
                "pages": 0,
                "note": "no non-hub nodes",
            },
        }

    G_sub = G_projected.subgraph(non_hub_nodes).copy()

    # ── Pass 1: Section-level on file-contracted graph ────────────────
    file_graph, file_to_nodes = _contract_to_file_graph(G_sub)

    connected_files = [f for f in file_graph.nodes() if file_graph.degree(f) > 0]
    isolated_files = [f for f in file_graph.nodes() if file_graph.degree(f) == 0]

    file_to_section: Dict[str, int] = {}

    if connected_files:
        file_sub = file_graph.subgraph(connected_files).copy()
        ig_file, file_list = _nx_to_igraph(file_sub)

        sec_partition = leidenalg.find_partition(
            ig_file,
            leidenalg.RBConfigurationVertexPartition,
            resolution_parameter=section_resolution,
            weights="weight",
            seed=seed,
        )

        for idx, sec_id in enumerate(sec_partition.membership):
            file_to_section[file_list[idx]] = sec_id

    # Assign isolated files to nearest section by directory proximity
    if isolated_files and file_to_section:
        sec_dirs: Dict[int, Counter] = {}
        for f, sid in file_to_section.items():
            d = f.rsplit("/", 1)[0] if "/" in f else "<root>"
            sec_dirs.setdefault(sid, Counter())[d] += 1

        for f in isolated_files:
            d = f.rsplit("/", 1)[0] if "/" in f else "<root>"
            best_sec = max(sec_dirs, key=lambda s: sec_dirs[s].get(d, 0))
            file_to_section[f] = best_sec
    elif isolated_files:
        # All files are isolated — each gets its own section so
        # consolidation can merge them by directory proximity later.
        for i, f in enumerate(isolated_files):
            file_to_section[f] = i

    # Propagate file → section to node → section
    macro_assignments: Dict[str, int] = {}
    for f, sec_id in file_to_section.items():
        for nid in file_to_nodes.get(f, []):
            macro_assignments[nid] = sec_id

    logger.info(
        "File-contracted section pass: %d files (%d connected, "
        "%d isolated) → %d sections",
        len(file_to_section),
        len(connected_files),
        len(isolated_files),
        len(set(file_to_section.values())),
    )

    # ── Pass 2: Page-level per section on original nodes ──────────────
    sec_nodes: Dict[int, Set[str]] = {}
    for nid, sec_id in macro_assignments.items():
        sec_nodes.setdefault(sec_id, set()).add(nid)

    sections: Dict[int, Dict[str, Any]] = {}
    micro_assignments: Dict[int, Dict[str, int]] = {}

    for sec_id, node_set in sec_nodes.items():
        if len(node_set) < 2:
            nid_list = list(node_set)
            sections[sec_id] = {
                "pages": {0: nid_list},
                "centroids": {0: _detect_page_centroids(G_sub, nid_list)},
            }
            micro_assignments[sec_id] = {nid: 0 for nid in nid_list}
            continue

        sec_subgraph = G_sub.subgraph(node_set)
        sec_undirected = _to_weighted_undirected(sec_subgraph)

        if sec_undirected.number_of_edges() == 0:
            nid_list = sorted(node_set)
            sections[sec_id] = {
                "pages": {0: nid_list},
                "centroids": {0: _detect_page_centroids(G_sub, nid_list)},
            }
            micro_assignments[sec_id] = {nid: 0 for nid in nid_list}
            continue

        ig_sec, sec_node_list = _nx_to_igraph(sec_undirected)

        pg_partition = leidenalg.find_partition(
            ig_sec,
            leidenalg.RBConfigurationVertexPartition,
            resolution_parameter=page_resolution,
            weights="weight",
            seed=seed,
        )

        # Build page structure
        pages: Dict[int, List[str]] = {}
        for idx, pg_id in enumerate(pg_partition.membership):
            pages.setdefault(pg_id, []).append(sec_node_list[idx])

        # Renumber pages to dense 0..N-1
        renumbered: Dict[int, List[str]] = {}
        page_centroids: Dict[int, List[Dict[str, Any]]] = {}
        micro_map: Dict[str, int] = {}
        for new_pg_id, (_, node_ids) in enumerate(sorted(pages.items())):
            renumbered[new_pg_id] = node_ids
            page_centroids[new_pg_id] = _detect_page_centroids(G_sub, node_ids)
            for nid in node_ids:
                micro_map[nid] = new_pg_id

        sections[sec_id] = {"pages": renumbered, "centroids": page_centroids}
        micro_assignments[sec_id] = micro_map

    n_sections = len(sections)
    n_pages = sum(len(s["pages"]) for s in sections.values())

    metadata = {
        "algorithm": "hierarchical_leiden_file_contracted",
        "section_resolution": section_resolution,
        "page_resolution": page_resolution,
        "sections": n_sections,
        "pages": n_pages,
        "seed": seed,
        "file_nodes": len(file_to_section),
        "connected_files": len(connected_files),
        "isolated_files": len(isolated_files),
    }

    logger.info(
        "Hierarchical Leiden (file-contracted): %d sections, %d pages "
        "(γ_sec=%.4f, γ_pg=%.2f, %d files → %d nodes)",
        n_sections,
        n_pages,
        section_resolution,
        page_resolution,
        len(file_to_section),
        len(non_hub_nodes),
    )

    return {
        "sections": sections,
        "macro_assignments": macro_assignments,
        "micro_assignments": micro_assignments,
        "algorithm_metadata": metadata,
    }


# ═══════════════════════════════════════════════════════════════════════════
# 9. Trivial Assignment (small graph fallback)
# ═══════════════════════════════════════════════════════════════════════════


def _trivial_assignment(
    G: nx.MultiDiGraph,
) -> ClusterAssignment:
    """Produce a single-section, single-page assignment for tiny graphs.

    Used when the graph has fewer than 3 non-hub nodes, making community
    detection pointless.
    """
    all_nodes = list(G.nodes())
    section_id = 0
    page_id = 0

    macro = {nid: section_id for nid in all_nodes}
    micro = {section_id: {nid: page_id for nid in all_nodes}}
    sections = {
        section_id: {
            "pages": {page_id: all_nodes},
            "centroids": {page_id: _detect_page_centroids(G, all_nodes)},
        }
    }

    return ClusterAssignment(
        macro_clusters=macro,
        micro_clusters=micro,
        hub_nodes=set(),
        hub_assignments={},
        sections=sections,
        section_count=1,
        page_count=1,
        algorithm="trivial",
        resolution={"section": 0.0, "page": 0.0},
        stats={"note": "trivial assignment for small graph"},
    )


# ═══════════════════════════════════════════════════════════════════════════
# 10. Main Entry Point
# ═══════════════════════════════════════════════════════════════════════════


def run_clustering(
    G: nx.MultiDiGraph,
    exclude_tests: bool = False,
    section_resolution: float = LEIDEN_FILE_SECTION_RESOLUTION,
    page_resolution: float = LEIDEN_PAGE_RESOLUTION,
    hub_z_threshold: float = DEFAULT_HUB_Z_THRESHOLD,
) -> ClusterAssignment:
    """Execute the complete clustering pipeline.

    This is the main entry point for graph-based wiki structure
    generation.  It replaces the legacy ``run_phase3()`` from the
    DeepWiki plugin with a clean, DB-free interface.

    Steps:
        1. Hub detection (Z-score on in-degree, full graph)
        2. Optional test-node exclusion
        3. Hierarchical Leiden clustering (file-contracted sections,
           per-section pages on ALL symbol types)
        4. Section consolidation (directory proximity, log₂ target)
        5. Page consolidation (directory proximity, √ target)
        6. Hub re-integration (plurality vote)

    Note:
        Architectural projection is intentionally NOT applied before
        hierarchical Leiden.  The file-contraction step in Leiden
        already compresses methods/fields into file-level nodes for
        section discovery, while per-section page-level Leiden benefits
        from seeing ALL symbol types (tighter communities).  The
        planner's ``_build_architectural_cluster_map`` filters to
        architectural symbols afterward — matching deepwiki's behavior.

    Args:
        G: In-memory ``nx.MultiDiGraph`` with node attributes including
            ``symbol_type``, ``rel_path``/``file_name``, ``symbol_name``.
        exclude_tests: When True, nodes whose ``rel_path`` matches test
            patterns are removed before clustering.
        section_resolution: Leiden γ for the file-contracted section pass.
        page_resolution: Leiden γ for the per-section page pass.
        hub_z_threshold: Z-score threshold for hub detection.

    Returns:
        :class:`ClusterAssignment` with all cluster assignments and
        metadata.
    """
    stats: Dict[str, Any] = {
        "graph": {
            "total_nodes": G.number_of_nodes(),
            "total_edges": G.number_of_edges(),
        }
    }

    # Step 1: Hub detection — on the FULL graph so in-degree Z-scores
    # reflect the true connectivity (methods/fields contribute).
    # This matches deepwiki's Phase 2 pipeline which detects hubs on
    # the complete weighted graph *before* any architectural filtering.
    hubs = detect_hubs(G, z_threshold=hub_z_threshold)

    stats["graph"]["hubs"] = len(hubs)
    stats["projection"] = {
        "original_nodes": G.number_of_nodes(),
        "projected_nodes": G.number_of_nodes(),
        "note": "no projection — hierarchical Leiden uses file-contraction",
    }

    # Trivial fallback for very small graphs
    non_hub_count = G.number_of_nodes() - len(hubs)
    if non_hub_count < 3:
        logger.info(
            "Graph too small for clustering (%d non-hub nodes). "
            "Using trivial assignment.",
            non_hub_count,
        )
        return _trivial_assignment(G)

    # Step 2: Optional test-node exclusion
    excluded_test_nodes: Set[str] = set()
    G_cluster = G

    if exclude_tests:
        for nid, data in G.nodes(data=True):
            rel_path = data.get("rel_path") or data.get("file_name") or ""
            if is_test_path(rel_path):
                excluded_test_nodes.add(nid)

        if excluded_test_nodes:
            non_test_nodes = set(G.nodes()) - excluded_test_nodes
            if not non_test_nodes:
                # Every node is a test file — fall back to full graph
                logger.warning(
                    "All %d nodes are test files; ignoring "
                    "exclude_tests and clustering the full graph.",
                    len(excluded_test_nodes),
                )
                excluded_test_nodes = set()
            else:
                G_cluster = G.subgraph(non_test_nodes).copy()
                logger.info(
                    "Test exclusion: removed %d test nodes from clustering "
                    "(%d remaining)",
                    len(excluded_test_nodes),
                    G_cluster.number_of_nodes(),
                )

    stats["graph"]["test_nodes_excluded"] = len(excluded_test_nodes)

    # Check again after test exclusion
    non_hub_in_cluster = G_cluster.number_of_nodes() - len(hubs & set(G_cluster.nodes()))
    if non_hub_in_cluster < 3:
        return _trivial_assignment(G_cluster)

    # Step 3: Hierarchical Leiden
    leiden_result = _hierarchical_leiden(
        G_cluster,
        hubs,
        section_resolution=section_resolution,
        page_resolution=page_resolution,
    )

    # Step 4+5: Consolidation
    # Use the ORIGINAL graph G (not G_cluster) for directory-proximity
    # lookups during consolidation — this matches deepwiki's pipeline
    # where _consolidate_sections/pages receive the full graph.
    n_files = leiden_result["algorithm_metadata"].get("file_nodes")
    leiden_result = _consolidate_sections(leiden_result, G, n_files=n_files)
    leiden_result = _consolidate_pages(leiden_result, G)

    macro_assignments = leiden_result["macro_assignments"]
    micro_assignments = leiden_result["micro_assignments"]

    stats["macro"] = {
        "cluster_count": len(leiden_result["sections"]),
        "cluster_count_raw": leiden_result["algorithm_metadata"]["sections"],
        "nodes_assigned": len(macro_assignments),
    }
    stats["micro"] = {
        "total_pages": sum(
            len(s["pages"]) for s in leiden_result["sections"].values()
        ),
        "total_pages_raw": leiden_result["algorithm_metadata"]["pages"],
    }
    stats["algorithm_metadata"] = leiden_result["algorithm_metadata"]

    # Step 6: Hub re-integration
    # Use original G so hub edges to all nodes (including test nodes)
    # participate in the plurality vote — matches deepwiki.
    hub_assignments = reintegrate_hubs(
        G, hubs, macro_assignments, micro_assignments
    )

    # Merge hubs into macro/micro AND sections so the planner sees them.
    # In deepwiki, persist_clusters() writes ALL nodes (including hubs)
    # to the DB, and the planner reads them via SQL.  Here we must
    # insert hubs into the sections dict directly.
    sections = leiden_result["sections"]
    for hub_id, (macro_id, micro_id) in hub_assignments.items():
        macro_assignments[hub_id] = macro_id
        if micro_id is not None:
            micro_assignments.setdefault(macro_id, {})[hub_id] = micro_id
        # Insert into sections page-list so _build_architectural_cluster_map
        # can see the hub when iterating sections → pages → node_ids.
        if macro_id in sections:
            pages = sections[macro_id].setdefault("pages", {})
            if micro_id is not None and micro_id in pages:
                if hub_id not in pages[micro_id]:
                    pages[micro_id].append(hub_id)
            elif micro_id is not None:
                # micro_id doesn't exist yet — pick the largest page
                if pages:
                    largest_pg = max(pages, key=lambda p: len(pages[p]))
                    pages[largest_pg].append(hub_id)
            else:
                # No micro assigned — add to the largest page
                if pages:
                    largest_pg = max(pages, key=lambda p: len(pages[p]))
                    pages[largest_pg].append(hub_id)

    stats["hubs"] = {
        "total": len(hub_assignments),
        "assigned_to_cluster": len(hub_assignments),
    }

    section_count = len(leiden_result["sections"])
    page_count = sum(len(s["pages"]) for s in leiden_result["sections"].values())

    logger.info(
        "Clustering complete: %d sections (%d raw), %d pages (%d raw), %d hubs",
        section_count,
        stats["macro"]["cluster_count_raw"],
        page_count,
        stats["micro"]["total_pages_raw"],
        len(hub_assignments),
    )

    return ClusterAssignment(
        macro_clusters=macro_assignments,
        micro_clusters=micro_assignments,
        hub_nodes=hubs,
        hub_assignments=hub_assignments,
        sections=leiden_result["sections"],
        section_count=section_count,
        page_count=page_count,
        algorithm="hierarchical_leiden_file_contracted",
        resolution={
            "section": section_resolution,
            "page": page_resolution,
        },
        stats=stats,
    )
