# Technical Design: Cluster-Based Structure Planner

**Spec**: `docs/proposals/cluster-planner-migration/spec.md`
**Created**: 2026-04-10
**Updated**: 2026-04-10
**Scope**: Phase 1 (Graph Clustering Engine) + Phase 2 (Cluster Planner + Integration). Phase 3 (Expansion Sync) noted at boundaries.

---

## 1. Module Structure

### New Files (Phase 1 — Clustering Engine)

| File | Purpose |
|------|---------|
| `backend/app/core/graph_clustering.py` | Core clustering pipeline: projection, hierarchical Leiden (leidenalg), hub detection, page sizing |
| `backend/app/core/cluster_constants.py` | Clustering-specific constants, resolution formulas, symbol type sets |

### New Files (Phase 2 — Cluster Planner)

| File | Purpose |
|------|---------|
| `backend/app/core/wiki_structure_planner/cluster_planner.py` | LLM-based naming of pre-computed clusters → `WikiStructureSpec` |
| `backend/app/core/wiki_structure_planner/cluster_prompts.py` | Section/page naming prompt templates |

### Modified Files (Phase 1)

| File | Change |
|------|--------|
| `backend/app/core/code_graph/graph_builder.py` | Add `is_architectural` flag to nodes during graph building; ensure `rel_path` on all nodes |
| `backend/app/core/constants.py` | Sync `ARCHITECTURAL_SYMBOLS` with DeepWiki; add `EXPANSION_SYMBOL_TYPES`, `PAGE_IDENTITY_SYMBOLS`, `SUPPORTING_CODE_SYMBOLS`, `SYMBOL_TYPE_PRIORITY` |

### Modified Files (Phase 2)

| File | Change |
|------|--------|
| `backend/app/core/agents/wiki_graph_optimized.py` | Add planner routing: if `planner_type == "cluster"`, use `ClusterStructurePlanner` |
| `backend/app/core/state/wiki_state.py` | Add optional `planner_type: str` field to `WikiState` |
| `backend/app/models/api.py` | Add `planner_type: str = "deepagents"` to `GenerateWikiRequest` |
| `backend/app/services/wiki_service.py` | Pass `planner_type` from request to agent config |
| `backend/app/config.py` | Add `cluster_planner_enabled`, `cluster_algorithm`, `cluster_hub_percentile`, `cluster_max_sections` settings |

### Phase 3 additions (Expansion Sync)

`backend/app/core/code_graph/expansion_engine.py` — sync priority-based expansion with DeepWiki (P0/P1/P2 relationship types, language-specific augmenters).

---

## 2. Data Model

### ClusterAssignment (Dataclass — in `graph_clustering.py`)

```python
from dataclasses import dataclass, field
from typing import Any

@dataclass
class ClusterAssignment:
    """Result of Phase 3 deterministic clustering.
    
    Maps every architectural node to a (macro_cluster, micro_cluster) pair.
    macro_cluster = wiki section ID
    micro_cluster = wiki page ID within that section
    """
    
    macro_clusters: dict[str, int]          # node_id → section_id
    micro_clusters: dict[str, int]          # node_id → page_id (section-scoped)
    hub_nodes: set[str] = field(default_factory=set)
    hub_assignments: dict[str, int] = field(default_factory=dict)
    
    # Stats for logging / SSE events
    section_count: int = 0
    page_count: int = 0
    algorithm: str = "leiden"               # Always Leiden (no Louvain)
    resolution: float = 1.0
    stats: dict[str, Any] = field(default_factory=dict)
    
    def get_cluster_map(self) -> dict[int, dict[int, list[str]]]:
        """Build {section_id: {page_id: [node_ids]}} mapping.
        
        This is the primary input for ClusterStructurePlanner.
        """
        cluster_map: dict[int, dict[int, list[str]]] = {}
        for node_id, section_id in self.macro_clusters.items():
            page_id = self.micro_clusters.get(node_id, 0)
            cluster_map.setdefault(section_id, {}).setdefault(page_id, []).append(node_id)
        return cluster_map
```

### Constants (in `cluster_constants.py`)

```python
"""Clustering constants — ported from DeepWiki plugin_implementation/constants.py"""

import math
import re

# ─── Architectural Symbols ───────────────────────────────────────────────
# Single source of truth: which symbol types get graph nodes
# NOTE: Must stay in sync with backend/app/core/constants.py ARCHITECTURAL_SYMBOLS

CODE_ARCHITECTURAL_TYPES = {
    "class", "interface", "struct", "enum", "trait",
    "function",    # Standalone only (methods excluded)
    "constant",    # Module-level only
    "type_alias",
    "macro",
}

DOC_ARCHITECTURAL_TYPES = frozenset({
    "markdown_document", "markdown_section",
    "restructuredtext_document",
    "asciidoc_document",
    "plaintext_document",
    "document_document",
    "pdf_document",
    "html_document",
    "xml_document",
    "json_document",
    "yaml_document",
    "toml_document",
    "config_document",
    "build_config_document",
    "schema_document",
    "infrastructure_document",
    "script_document",
    "text_chunk",
    "text_document",
})

ARCHITECTURAL_SYMBOLS = CODE_ARCHITECTURAL_TYPES | DOC_ARCHITECTURAL_TYPES

# Symbols that anchor pages (primary identity)
PAGE_IDENTITY_SYMBOLS = frozenset({
    "class", "interface", "struct", "enum", "trait",
    "record",
    "module", "namespace",
    "function",
})

# Symbols that support pages (secondary)
SUPPORTING_CODE_SYMBOLS = frozenset({
    "constant", "type_alias", "macro",
    "method", "property",
})

# Priority ranking for seed selection (higher = higher priority)
SYMBOL_TYPE_PRIORITY = {
    'class': 10, 'interface': 10, 'trait': 10, 'protocol': 10,
    'enum': 9, 'struct': 9, 'record': 9,
    'module': 8, 'namespace': 8,
    'function': 7,
    'constant': 6, 'type_alias': 6, 'macro': 6,
    'method': 3, 'property': 2,
    'module_doc': 1, 'file_doc': 1,
}

# ─── Clustering Resolution ───────────────────────────────────────────────

LEIDEN_FILE_SECTION_RESOLUTION = 1.0   # Standard modularity for file-contracted sections
LEIDEN_PAGE_RESOLUTION = 1.0           # Same resolution for pages within sections (matches DeepWiki)

# ─── Page Sizing ─────────────────────────────────────────────────────────

MIN_PAGE_SIZE = 5              # Merge clusters smaller than this
MERGE_THRESHOLD = 4            # Hard minimum before merge
DEFAULT_MAX_PAGE_SIZE = 25     # Fallback

def adaptive_max_page_size(total_nodes: int) -> int:
    """Scale max page size with repository size."""
    if total_nodes >= 5000:
        return 60
    elif total_nodes >= 2000:
        return 45
    elif total_nodes >= 500:
        return 35
    return 25

def target_section_count(n_files: int) -> int:
    """Logarithmic section cap on file count: max(5, min(20, ceil(1.2 × log₂(F))))"""
    if n_files < 10:
        return 5
    return max(5, min(20, math.ceil(1.2 * math.log2(max(10, n_files)))))

def target_total_pages(n_nodes: int) -> int:
    """Global page target: max(8, min(200, ceil(√(nodes/7))))"""
    return max(8, min(200, math.ceil(math.sqrt(n_nodes / 7))))

# ─── Test Detection ──────────────────────────────────────────────────────

TEST_PATH_PATTERNS = re.compile(
    r"(^|/)tests?/|/__tests__/|\.test\.|_test\.|\.spec\.|_spec\.",
    re.IGNORECASE,
)

def is_test_path(rel_path: str) -> bool:
    """Determine if a file path matches test patterns."""
    return bool(TEST_PATH_PATTERNS.search(rel_path))

# ─── Hub Detection ───────────────────────────────────────────────────────

DEFAULT_HUB_PERCENTILE = 90       # PageRank percentile for hub identification
MAX_HUB_FRACTION = 0.15           # Never classify more than 15% of nodes as hubs

# ─── Centroid Detection ──────────────────────────────────────────────────

CENTROID_ARCHITECTURAL_MIN_PRIORITY = 5   # Minimum priority for centroid-eligible nodes
MIN_CENTROIDS = 1
MAX_CENTROIDS = 10
```

---

## 3. Graph Clustering Pipeline

### `graph_clustering.py` — Core Algorithm

The file contains pure functions operating on NetworkX graphs. No class instances, no database access, no LLM calls. All state is passed explicitly.

```python
"""
Graph-based deterministic clustering for wiki structure planning.

Ported from DeepWiki plugin_implementation/graph_clustering.py.
Adapted for Wikis: operates on in-memory NetworkX graphs,
no unified_db.py dependency, returns ClusterAssignment dataclass.

Requires: leidenalg==0.11.0, python-igraph==1.0.0
"""

import logging
import math
from collections import Counter
from typing import Any

import networkx as nx
import numpy as np

from app.core.cluster_constants import (
    ARCHITECTURAL_SYMBOLS,
    CODE_ARCHITECTURAL_TYPES,
    DEFAULT_HUB_PERCENTILE,
    LEIDEN_FILE_SECTION_RESOLUTION,
    LEIDEN_PAGE_RESOLUTION,
    MAX_HUB_FRACTION,
    MERGE_THRESHOLD,
    MIN_PAGE_SIZE,
    CENTROID_ARCHITECTURAL_MIN_PRIORITY,
    SYMBOL_TYPE_PRIORITY,
    adaptive_max_page_size,
    target_section_count,
    target_total_pages,
)

logger = logging.getLogger(__name__)

# ─── Required Dependencies ───────────────────────────────────────────────

import igraph as ig
import leidenalg  # C++ Leiden community detection (REQUIRED)


def run_clustering(
    graph: nx.DiGraph,
    hub_percentile: int = DEFAULT_HUB_PERCENTILE,
) -> ClusterAssignment:
    """Main entry point: run the full hierarchical Leiden clustering pipeline.
    
    1. Architectural projection (collapse children → parents)
    2. Hub detection (PageRank + percentile)
    3. File contraction (symbols → file-level nodes)
    4. Leiden pass 1: macro-clustering on file graph → sections
       (isolated files handled by directory proximity)
    5. Leiden pass 2: micro-clustering per section on original nodes → pages
    6. Consolidate sections (directory proximity) + pages (global target + dir proximity)
    7. Hub re-integration (plurality voting)
    
    Args:
        graph: NetworkX DiGraph with code symbols as nodes.
                Each node must have: symbol_type, rel_path.
                Each edge should have: type (relationship type), weight (optional).
        hub_percentile: PageRank percentile threshold for hub detection.
    
    Returns:
        ClusterAssignment with macro/micro cluster mappings.
    """
    # Step 1: Architectural projection
    arch_graph = architectural_projection(graph)
    arch_count = arch_graph.number_of_nodes()
    
    if arch_count < 3:
        logger.warning(f"Too few architectural nodes ({arch_count}); trivial clustering")
        return _trivial_assignment(arch_graph)
    
    # Step 2: Hub detection
    hubs = detect_hubs(arch_graph, percentile=hub_percentile)
    
    # Step 3: Remove hubs for clustering
    cluster_graph = arch_graph.copy()
    cluster_graph.remove_nodes_from(hubs)
    
    # Step 4: Macro-clustering via hierarchical Leiden (sections)
    macro = _leiden_macro(cluster_graph)
    
    # Step 5: Micro-clustering via Leiden (pages per section)
    micro = _leiden_micro_all(arch_graph, macro)
    
    # Step 6: Consolidate sections (dir proximity) and pages (global target + dir proximity)
    n_files = len(set(
        (arch_graph.nodes[n].get("location", {}).get("rel_path", "") or arch_graph.nodes[n].get("rel_path", ""))
        for n in arch_graph.nodes()
    ))
    macro = _consolidate_sections(arch_graph, macro, n_files=n_files)
    micro = _consolidate_pages(arch_graph, macro, micro)
    
    # Step 7: Hub re-integration
    macro, micro = reintegrate_hubs(arch_graph, macro, micro, hubs)
    
    return ClusterAssignment(
        macro_clusters=macro,
        micro_clusters=micro,
        hub_nodes=hubs,
        hub_assignments={h: macro[h] for h in hubs if h in macro},
        section_count=len(set(macro.values())),
        page_count=len(set((macro[n], micro[n]) for n in macro)),
        algorithm="leiden",
        resolution=LEIDEN_FILE_SECTION_RESOLUTION,
        stats={
            "arch_node_count": arch_count,
            "hub_count": len(hubs),
            "edge_count": arch_graph.number_of_edges(),
        },
    )


def architectural_projection(graph: nx.DiGraph) -> nx.DiGraph:
    """Project graph to architectural-only nodes.
    
    1. Keep nodes where symbol_type in ARCHITECTURAL_SYMBOLS
    2. For child nodes (method, field, variable):
       - Find parent via graph 'contains'/'defines' edges or node_id convention
       - Remap all edges to parent
    3. Drop self-loops
    4. Sum parallel edge weights
    
    CRITICAL: This is ported from DeepWiki's graph_clustering.py.
    The wikis graph uses UUID node IDs (not lang::file::symbol composite IDs).
    Parent resolution must use graph edges (contains/defines) instead of
    string splitting on '::'.
    """
    projected = nx.DiGraph()
    
    # Map child → parent
    child_to_parent: dict[str, str] = {}
    architectural_nodes: set[str] = set()
    
    for node_id, data in graph.nodes(data=True):
        symbol_type = data.get("symbol", {}).get("kind", "") or data.get("symbol_type", "")
        if symbol_type in ARCHITECTURAL_SYMBOLS:
            architectural_nodes.add(node_id)
            projected.add_node(node_id, **data)
        else:
            # Find parent via 'contains' or 'defines' incoming edge
            parent = _find_architectural_parent(graph, node_id)
            if parent and parent in graph:
                child_to_parent[node_id] = parent
    
    # Remap edges
    for u, v, edge_data in graph.edges(data=True):
        # Resolve actual endpoints
        actual_u = child_to_parent.get(u, u)
        actual_v = child_to_parent.get(v, v)
        
        # Skip if either endpoint not in projected graph
        if actual_u not in architectural_nodes or actual_v not in architectural_nodes:
            continue
        
        # Skip self-loops (intra-class edges)
        if actual_u == actual_v:
            continue
        
        # Add or merge edge
        if projected.has_edge(actual_u, actual_v):
            existing = projected[actual_u][actual_v]
            existing["weight"] = existing.get("weight", 1.0) + edge_data.get("weight", 1.0)
        else:
            projected.add_edge(actual_u, actual_v, **edge_data)
    
    logger.info(
        f"Architectural projection: {graph.number_of_nodes()} → {projected.number_of_nodes()} nodes, "
        f"{graph.number_of_edges()} → {projected.number_of_edges()} edges"
    )
    return projected


def _find_architectural_parent(graph: nx.DiGraph, node_id: str) -> str | None:
    """Find the nearest architectural parent of a non-architectural node.
    
    Strategy cascade (adapted from DeepWiki for UUID-based graphs):
    1. Check incoming 'contains' or 'defines' edges
    2. Check node's 'parent_symbol' attribute
    3. Search for same-file class by 'location.rel_path' matching
    """
    # Strategy 1: Graph edges
    for pred in graph.predecessors(node_id):
        edge_data = graph[pred][node_id]
        edge_type = edge_data.get("type", "")
        if edge_type in ("contains", "defines", "defines_body"):
            pred_type = (graph.nodes[pred].get("symbol", {}).get("kind", "") 
                        or graph.nodes[pred].get("symbol_type", ""))
            if pred_type in ARCHITECTURAL_SYMBOLS:
                return pred
    
    # Strategy 2: Node attribute
    node_data = graph.nodes.get(node_id, {})
    parent_ref = node_data.get("parent_symbol") or node_data.get("symbol", {}).get("parent_symbol")
    if parent_ref:
        # Search for parent by name + file
        node_file = (node_data.get("location", {}).get("rel_path", "") 
                    or node_data.get("rel_path", ""))
        for candidate_id, candidate_data in graph.nodes(data=True):
            candidate_name = (candidate_data.get("symbol", {}).get("name", "") 
                            or candidate_data.get("symbol_name", ""))
            candidate_file = (candidate_data.get("location", {}).get("rel_path", "") 
                            or candidate_data.get("rel_path", ""))
            candidate_type = (candidate_data.get("symbol", {}).get("kind", "") 
                            or candidate_data.get("symbol_type", ""))
            if (candidate_name == parent_ref 
                and candidate_file == node_file 
                and candidate_type in ARCHITECTURAL_SYMBOLS):
                return candidate_id
    
    return None


def detect_hubs(
    graph: nx.DiGraph,
    percentile: int = DEFAULT_HUB_PERCENTILE,
) -> set[str]:
    """Detect hub nodes via PageRank.
    
    Hubs are nodes with PageRank above the given percentile.
    Capped at MAX_HUB_FRACTION of total nodes.
    """
    if graph.number_of_nodes() < 10:
        return set()
    
    pr = nx.pagerank(graph.to_undirected())
    values = list(pr.values())
    threshold = float(np.percentile(values, percentile))
    
    hubs = {node for node, score in pr.items() if score > threshold}
    
    # Cap at MAX_HUB_FRACTION
    max_hubs = int(graph.number_of_nodes() * MAX_HUB_FRACTION)
    if len(hubs) > max_hubs:
        sorted_hubs = sorted(hubs, key=lambda n: pr[n], reverse=True)
        hubs = set(sorted_hubs[:max_hubs])
    
    logger.info(f"Hub detection: {len(hubs)} hubs from {graph.number_of_nodes()} nodes (threshold={threshold:.6f})")
    return hubs


def _contract_to_file_graph(graph: nx.DiGraph) -> tuple[nx.Graph, dict[str, list[str]]]:
    """Contract symbol-level graph to file-level graph.
    
    Each unique rel_path becomes ONE node.
    Cross-file edge weights are summed.
    Same-file edges are dropped (intra-file noise).
    
    Returns:
        (file_graph, file_to_nodes) where file_to_nodes maps rel_path → [node_ids]
    """
    node_to_file: dict[str, str] = {}
    file_to_nodes: dict[str, list[str]] = {}
    
    for node_id, data in graph.nodes(data=True):
        rel_path = (data.get("location", {}).get("rel_path", "") 
                   or data.get("rel_path", "") 
                   or "unknown")
        node_to_file[node_id] = rel_path
        file_to_nodes.setdefault(rel_path, []).append(node_id)
    
    # Sum cross-file edges
    edge_weights: dict[tuple[str, str], float] = {}
    for u, v, data in graph.edges(data=True):
        fu = node_to_file.get(u, "")
        fv = node_to_file.get(v, "")
        if fu and fv and fu != fv:
            key = (min(fu, fv), max(fu, fv))
            edge_weights[key] = edge_weights.get(key, 0) + data.get("weight", 1.0)
    
    file_graph = nx.Graph()
    file_graph.add_nodes_from(file_to_nodes.keys())
    for (f1, f2), weight in edge_weights.items():
        file_graph.add_edge(f1, f2, weight=weight)
    
    logger.info(
        f"File contraction: {graph.number_of_nodes()} symbol-nodes → "
        f"{file_graph.number_of_nodes()} file-nodes, "
        f"{len(edge_weights)} cross-file edges"
    )
    return file_graph, file_to_nodes


def _nx_to_igraph(G: nx.Graph) -> tuple[ig.Graph, list[str]]:
    """Convert NetworkX graph to igraph for Leiden.
    
    Manually builds igraph (not ig.Graph.from_networkx()) to ensure
    deterministic node ordering and correct weight transfer.
    Ported from DeepWiki graph_clustering.py.
    
    Returns:
        (ig_graph, node_list) where node_list[i] is the name of igraph vertex i.
    """
    nodes = sorted(G.nodes())
    node_to_idx = {n: i for i, n in enumerate(nodes)}
    
    edges = []
    weights = []
    for u, v, data in G.edges(data=True):
        edges.append((node_to_idx[u], node_to_idx[v]))
        weights.append(data.get("weight", 1.0))
    
    ig_graph = ig.Graph(n=len(nodes), edges=edges, directed=False)
    ig_graph.vs["name"] = nodes
    if weights:
        ig_graph.es["weight"] = weights
    
    return ig_graph, nodes


def _leiden_macro(
    graph: nx.DiGraph,
) -> dict[str, int]:
    """Macro-clustering via Leiden on file-contracted graph.
    
    Two-pass strategy:
    1. Contract graph to file-level nodes
    2. Separate connected vs isolated files (degree-0 in file graph)
    3. Run Leiden with RBConfigurationVertexPartition at γ=1.0 on connected files
    4. Assign isolated files to nearest section by directory proximity
    5. Propagate file → section assignment back to symbol nodes
    """
    file_graph, file_to_nodes = _contract_to_file_graph(graph)
    
    if file_graph.number_of_nodes() < 2:
        return {nid: 0 for nid in graph.nodes()}
    
    # Separate connected vs isolated files
    connected_files = [f for f in file_graph.nodes() if file_graph.degree(f) > 0]
    isolated_files = [f for f in file_graph.nodes() if file_graph.degree(f) == 0]
    
    file_to_section: dict[str, int] = {}
    
    if connected_files:
        # Convert connected subgraph to weighted undirected for igraph
        file_sub = file_graph.subgraph(connected_files).copy()
        ig_file, file_list = _nx_to_igraph(file_sub)
        
        partition = leidenalg.find_partition(
            ig_file,
            leidenalg.RBConfigurationVertexPartition,
            weights="weight" if ig_file.ecount() > 0 else None,
            seed=42,
            resolution_parameter=LEIDEN_FILE_SECTION_RESOLUTION,
        )
        
        for idx, sec_id in enumerate(partition.membership):
            file_to_section[file_list[idx]] = sec_id
    
    # Assign isolated files to nearest section by directory proximity
    if isolated_files and file_to_section:
        sec_dirs: dict[int, Counter] = {}
        for f, sid in file_to_section.items():
            d = f.rsplit("/", 1)[0] if "/" in f else "<root>"
            sec_dirs.setdefault(sid, Counter())[d] += 1
        
        for f in isolated_files:
            d = f.rsplit("/", 1)[0] if "/" in f else "<root>"
            best_sec = max(sec_dirs, key=lambda s: sec_dirs[s].get(d, 0))
            file_to_section[f] = best_sec
    elif isolated_files:
        # All files are isolated — each gets its own section
        # (consolidation can merge them by directory proximity later)
        for i, f in enumerate(isolated_files):
            file_to_section[f] = i
    
    # Propagate to symbol nodes
    macro: dict[str, int] = {}
    for file_path, node_ids in file_to_nodes.items():
        section_id = file_to_section.get(file_path, 0)
        for node_id in node_ids:
            macro[node_id] = section_id
    
    section_count = len(set(macro.values()))
    logger.info(f"Leiden macro-clustering: {section_count} sections from {file_graph.number_of_nodes()} files "
                f"({len(isolated_files)} isolated)")
    return macro


def _consolidate_sections(
    graph: nx.DiGraph,
    macro: dict[str, int],
    n_files: int | None = None,
) -> dict[str, int]:
    """Merge smallest sections until count ≤ target using directory proximity.
    
    Ported from DeepWiki _consolidate_sections(). Uses _dir_similarity()
    (shared directory entries via min-intersect of directory histograms)
    to find the best merge target for the smallest section.
    
    The target section count scales with file count: target_section_count(n_files).
    """
    section_nodes: dict[int, list[str]] = {}
    for node, sid in macro.items():
        section_nodes.setdefault(sid, []).append(node)
    
    scale = n_files if n_files is not None else len(macro)
    target = target_section_count(scale)
    
    if len(section_nodes) <= target:
        # Re-index and return
        old_to_new = {old: new for new, old in enumerate(sorted(section_nodes.keys()))}
        return {n: old_to_new[s] for n, s in macro.items()}
    
    # Pre-compute per-section directory histograms
    sec_dirs: dict[int, Counter] = {}
    sec_sizes: dict[int, int] = {}
    for sid, nodes in section_nodes.items():
        sec_dirs[sid] = _dir_histogram(nodes, graph)
        sec_sizes[sid] = len(nodes)
    
    while len(sec_sizes) > target:
        # Find smallest section
        smallest_id = min(sec_sizes, key=sec_sizes.get)
        
        # Find best merge target by directory similarity (tie-break: prefer smaller)
        best_target = None
        best_sim = -1
        for sid in sec_sizes:
            if sid == smallest_id:
                continue
            sim = _dir_similarity(sec_dirs[smallest_id], sec_dirs[sid])
            if sim > best_sim or (sim == best_sim and sec_sizes[sid] < sec_sizes.get(best_target, float('inf'))):
                best_sim = sim
                best_target = sid
        
        if best_target is None:
            break
        
        # Merge smallest into best_target
        for node in section_nodes[smallest_id]:
            macro[node] = best_target
        section_nodes[best_target].extend(section_nodes.pop(smallest_id))
        sec_dirs[best_target] = _dir_histogram(section_nodes[best_target], graph)
        sec_sizes[best_target] += sec_sizes.pop(smallest_id)
        del sec_dirs[smallest_id]
    
    # Re-index sections to 0..N-1
    old_to_new = {old_id: new_id for new_id, old_id in enumerate(sorted(set(macro.values())))}
    return {node: old_to_new[section] for node, section in macro.items()}


def _leiden_micro_all(
    graph: nx.DiGraph,
    macro: dict[str, int],
) -> dict[str, int]:
    """Run Leiden micro-clustering within each macro-cluster.
    
    Each section’s original symbol nodes are clustered with Leiden
    (RBConfigurationVertexPartition, γ=1.0, seed=42) to produce pages.
    Ported from DeepWiki hierarchical_leiden_cluster() Pass 2.
    """
    micro: dict[str, int] = {}
    
    section_ids = sorted(set(macro.values()))
    for section_id in section_ids:
        section_nodes = [n for n, s in macro.items() if s == section_id]
        
        if len(section_nodes) <= MIN_PAGE_SIZE:
            # Too small to split — single page
            for node in section_nodes:
                micro[node] = 0
            continue
        
        # Extract subgraph for this section
        subgraph = graph.subgraph(section_nodes).copy()
        undirected = subgraph.to_undirected()
        
        # Convert to igraph and run Leiden
        ig_sub = _nx_to_igraph(undirected)
        
        try:
            partition = leidenalg.find_partition(
                ig_sub,
                leidenalg.RBConfigurationVertexPartition,
                weights="weight" if ig_sub.ecount() > 0 else None,
                seed=42,
                resolution_parameter=LEIDEN_PAGE_RESOLUTION,
            )
            for page_id, community in enumerate(partition):
                for node_idx in community:
                    node_name = ig_sub.vs[node_idx]["name"]
                    micro[node_name] = page_id
        except Exception:
            # Fallback: single page
            for node in section_nodes:
                micro[node] = 0
    
    return micro


def _consolidate_pages(
    graph: nx.DiGraph,
    macro: dict[str, int],
    micro: dict[str, int],
) -> dict[str, int]:
    """Consolidate pages globally using directory proximity.
    
    Ported from DeepWiki _consolidate_pages(). Picks the smallest page
    across ALL sections, merges it into its best same-section neighbour
    by _dir_similarity(). Repeats until total pages ≤ target.
    
    The global page target uses node count: target_total_pages(n_nodes).
    There is no per-section cap — a large section may keep many pages
    while a small one collapses to 1.
    
    Also enforces adaptive per-page maximum via adaptive_max_page_size().
    """
    total_nodes = len(macro)
    target = target_total_pages(total_nodes)
    max_size = adaptive_max_page_size(total_nodes)
    
    # Group by (section, page)
    pages: dict[tuple[int, int], list[str]] = {}
    for node_id in macro:
        key = (macro[node_id], micro.get(node_id, 0))
        pages.setdefault(key, []).append(node_id)
    
    total_pages = len(pages)
    if total_pages <= target:
        return micro
    
    # Build page-dir indexes
    pg_dirs: dict[tuple, Counter] = {}
    pg_sizes: dict[tuple, int] = {}
    for key, nodes in pages.items():
        pg_dirs[key] = _dir_histogram(nodes, graph)
        pg_sizes[key] = len(nodes)
    
    while len(pg_sizes) > target:
        # Find smallest page globally
        smallest_key = min(pg_sizes, key=pg_sizes.get)
        section_id = smallest_key[0]
        
        # Find best same-section neighbour by directory similarity
        best_target = None
        best_sim = -1
        for key in pg_sizes:
            if key == smallest_key or key[0] != section_id:
                continue
            sim = _dir_similarity(pg_dirs[smallest_key], pg_dirs[key])
            if sim > best_sim or (sim == best_sim and pg_sizes[key] < pg_sizes.get(best_target, float('inf'))):
                best_sim = sim
                best_target = key
        
        if best_target is None:
            break
        
        # Merge smallest into best_target
        for node in pages[smallest_key]:
            micro[node] = best_target[1]
        pages[best_target].extend(pages.pop(smallest_key))
        pg_dirs[best_target] = _dir_histogram(pages[best_target], graph)
        pg_sizes[best_target] += pg_sizes.pop(smallest_key)
        del pg_dirs[smallest_key]
    
    # Re-index pages within each section to 0..N-1
    for section_id in set(macro.values()):
        section_nodes = [n for n, s in macro.items() if s == section_id]
        old_pages = sorted(set(micro[n] for n in section_nodes))
        remap = {old: new for new, old in enumerate(old_pages)}
        for node in section_nodes:
            micro[node] = remap.get(micro[node], 0)
    
    return micro


def _dir_histogram(node_ids: list[str], graph: nx.DiGraph) -> Counter:
    """Build a histogram of directory prefixes for a set of nodes.
    
    For each node, extracts all expanding directory prefixes from its rel_path.
    E.g., "src/auth/service.py" → ["src", "src/auth"]
    """
    hist: Counter = Counter()
    for nid in node_ids:
        data = graph.nodes.get(nid, {})
        rel_path = (data.get("location", {}).get("rel_path", "")
                   or data.get("rel_path", ""))
        if not rel_path:
            continue
        # Immediate parent directory only (matches DeepWiki _dir_hist behavior)
        d = rel_path.rsplit("/", 1)[0] if "/" in rel_path else "<root>"
        hist[d] += 1
    return hist


def _dir_similarity(a: Counter, b: Counter) -> int:
    """Shared directory entries (min-intersect) between two histograms."""
    return sum(min(a[d], b[d]) for d in a if d in b)


def _to_weighted_undirected(G: nx.MultiDiGraph) -> nx.Graph:
    """Collapse directed multi-edges into an undirected weighted graph.
    
    For each pair (u, v), sums all parallel edge weights.
    Ported from DeepWiki graph_clustering.py.
    """
    U = nx.Graph()
    U.add_nodes_from(G.nodes())
    
    edge_weights: dict[tuple, float] = {}
    for u, v, data in G.edges(data=True):
        key = (min(u, v), max(u, v))
        edge_weights[key] = edge_weights.get(key, 0.0) + data.get("weight", 1.0)
    
    for (u, v), w in edge_weights.items():
        U.add_edge(u, v, weight=w)
    
    return U


def _detect_page_centroids(
    G: nx.DiGraph,
    page_node_ids: list[str],
    top_k: int | None = None,
) -> list[dict]:
    """Identify centroid nodes in a page community.
    
    Scoring: composite = degree_norm × 0.6 + type_priority_norm × 0.4
    
    Only architectural nodes (priority >= CENTROID_ARCHITECTURAL_MIN_PRIORITY)
    are eligible. Falls back to all nodes when none qualify.
    """
    if not page_node_ids:
        return []
    
    MAX_CENTROIDS = 10
    MIN_CENTROIDS = 1
    k = top_k or min(MAX_CENTROIDS, max(MIN_CENTROIDS, round(math.sqrt(len(page_node_ids)))))
    
    scored = []
    for nid in page_node_ids:
        data = G.nodes.get(nid, {})
        stype = (data.get("symbol_type") or data.get("symbol", {}).get("kind", "") or "unknown").lower()
        degree = G.in_degree(nid) + G.out_degree(nid) if G.is_directed() else G.degree(nid)
        priority = SYMBOL_TYPE_PRIORITY.get(stype, 0)
        scored.append({
            "node_id": nid,
            "name": data.get("symbol_name", "") or data.get("symbol", {}).get("name", nid),
            "type": stype,
            "degree": degree,
            "priority": priority,
        })
    
    max_deg = max((s["degree"] for s in scored), default=1) or 1
    max_pri = max((s["priority"] for s in scored), default=1) or 1
    
    for s in scored:
        s["score"] = 0.6 * (s["degree"] / max_deg) + 0.4 * (s["priority"] / max_pri)
    
    # Filter to architectural-only candidates
    arch = [s for s in scored if s["priority"] >= CENTROID_ARCHITECTURAL_MIN_PRIORITY]
    pool = arch if arch else scored
    
    pool.sort(key=lambda s: s["score"], reverse=True)
    return pool[:k]


def select_central_symbols(
    G: nx.DiGraph,
    cluster_nodes: set[str],
    k: int = 5,
) -> list[str]:
    """Select central symbols via PageRank on cluster subgraph.
    
    Extracts subgraph → converts to undirected → runs PageRank
    → returns top-k by descending rank.
    """
    subgraph = G.subgraph(cluster_nodes)
    undirected = subgraph.to_undirected()
    
    try:
        pr = nx.pagerank(undirected, weight="weight", max_iter=100, tol=1e-6)
    except nx.PowerIterationFailedConvergence:
        pr = {n: 1.0 / len(cluster_nodes) for n in cluster_nodes}
    
    ranked = sorted(pr.items(), key=lambda x: x[1], reverse=True)
    return [nid for nid, _ in ranked[:k]]


def detect_doc_clusters(
    cluster_map: dict[int, dict[int, list[str]]],
    graph: nx.DiGraph,
) -> dict[int, float]:
    """Identify macro-clusters that are predominantly documentation.
    
    A cluster is "doc-dominant" when more than DOC_DOMINANT_THRESHOLD (0.6)
    of its nodes have a doc symbol type.
    """
    DOC_DOMINANT_THRESHOLD = 0.6
    result: dict[int, float] = {}
    
    for macro_id, micro_map in cluster_map.items():
        total = 0
        doc_count = 0
        for node_ids in micro_map.values():
            for nid in node_ids:
                total += 1
                data = graph.nodes.get(nid, {})
                symbol_type = (data.get("symbol_type") or 
                              data.get("symbol", {}).get("kind", "")).lower()
                if symbol_type in DOC_ARCHITECTURAL_TYPES or data.get("is_doc"):
                    doc_count += 1
        
        if total > 0:
            ratio = doc_count / total
            if ratio >= DOC_DOMINANT_THRESHOLD:
                result[macro_id] = ratio
    
    return result


def reintegrate_hubs(
    graph: nx.DiGraph,
    macro: dict[str, int],
    micro: dict[str, int],
    hubs: set[str],
) -> tuple[dict[str, int], dict[str, int]]:
    """Re-integrate hub nodes via plurality voting.
    
    Each hub is assigned to the section it has the most edges to.
    Within that section, assigned to the page with the most edges.
    """
    for hub in hubs:
        # Section voting
        section_votes: Counter = Counter()
        for neighbor in graph.neighbors(hub):
            if neighbor in macro:
                section_votes[macro[neighbor]] += graph[hub][neighbor].get("weight", 1.0)
        
        if not section_votes:
            macro[hub] = 0
            micro[hub] = 0
            continue
        
        best_section = section_votes.most_common(1)[0][0]
        macro[hub] = best_section
        
        # Page voting within section
        page_votes: Counter = Counter()
        for neighbor in graph.neighbors(hub):
            if neighbor in macro and macro[neighbor] == best_section:
                page_votes[micro.get(neighbor, 0)] += graph[hub][neighbor].get("weight", 1.0)
        
        micro[hub] = page_votes.most_common(1)[0][0] if page_votes else 0
    
    return macro, micro


def _trivial_assignment(graph: nx.DiGraph) -> ClusterAssignment:
    """Produce a trivial assignment for very small graphs."""
    macro = {nid: 0 for nid in graph.nodes()}
    micro = {nid: 0 for nid in graph.nodes()}
    return ClusterAssignment(
        macro_clusters=macro,
        micro_clusters=micro,
        section_count=1,
        page_count=1,
        algorithm="trivial",
        stats={"arch_node_count": graph.number_of_nodes()},
    )
```

---

## 4. Cluster Structure Planner (LLM Naming)

### `cluster_planner.py` — Naming Engine

```python
"""
Cluster-based wiki structure planner.

Converts pre-computed ClusterAssignment into WikiStructureSpec
using lightweight LLM calls for capability-based naming.

Ported from DeepWiki wiki_structure_planner/cluster_planner.py.
"""

import logging
from typing import Any

from langchain_core.language_models import BaseChatModel

from app.core.cluster_constants import SYMBOL_TYPE_PRIORITY
from app.core.graph_clustering import ClusterAssignment
from app.core.state.wiki_state import PageSpec, SectionSpec, WikiStructureSpec
from app.core.wiki_structure_planner.cluster_prompts import (
    PAGE_NAMING_PROMPT,
    SECTION_NAMING_PROMPT,
)

logger = logging.getLogger(__name__)


class ClusterStructurePlanner:
    """Convert cluster assignments to named wiki structure.
    
    Two-pass naming strategy:
    1. Section naming — 1 LLM call per macro-cluster (dominant symbols)
    2. Page naming — 1 LLM call per micro-cluster (page-only symbols)
    """
    
    def __init__(
        self,
        graph: "nx.DiGraph",
        cluster_assignment: ClusterAssignment,
        llm: BaseChatModel,
        repo_name: str = "",
        progress_callback: callable | None = None,
    ):
        self.graph = graph
        self.assignment = cluster_assignment
        self.llm = llm
        self.repo_name = repo_name
        self._progress = progress_callback or (lambda *a, **kw: None)
    
    async def plan_structure(self) -> WikiStructureSpec:
        """Generate complete wiki structure from clusters.
        
        Returns WikiStructureSpec identical in schema to DeepAgents planner output.
        """
        cluster_map = self.assignment.get_cluster_map()
        sections: list[SectionSpec] = []
        
        total_sections = len(cluster_map)
        for order, (section_id, page_map) in enumerate(sorted(cluster_map.items())):
            self._progress(
                message=f"Naming section {order + 1}/{total_sections}...",
            )
            
            # Gather section-level symbols (all nodes in this section)
            section_nodes = []
            for page_nodes in page_map.values():
                section_nodes.extend(page_nodes)
            
            # Get dominant symbols for section naming
            dominant = self._get_dominant_symbols(section_nodes, limit=10)
            
            # LLM naming for section
            section_info = await self._name_section(dominant, page_map)
            
            # Name pages within section
            pages: list[PageSpec] = []
            for page_order, (page_id, node_ids) in enumerate(sorted(page_map.items())):
                # Get page-only central symbols (prevents centroid bleed)
                central = self._get_central_symbols(node_ids, limit=8)
                
                page_info = await self._name_page(
                    central, node_ids, section_info["section_name"],
                )
                
                pages.append(PageSpec(
                    page_name=page_info["page_name"],
                    page_order=page_order,
                    description=page_info.get("description", ""),
                    content_focus=page_info.get("description", ""),
                    target_symbols=self._node_ids_to_symbol_names(node_ids),
                    target_docs=self._node_ids_to_doc_paths(node_ids),
                    target_folders=self._node_ids_to_directories(node_ids),
                    retrieval_query=page_info.get("retrieval_query", ""),
                    key_files=self._node_ids_to_file_paths(node_ids),
                    metadata={
                        "macro_id": section_id,
                        "micro_id": page_id,
                        "cluster_node_ids": node_ids,
                    },
                ))
            
            sections.append(SectionSpec(
                section_name=section_info["section_name"],
                section_order=order,
                description=section_info.get("description", ""),
                rationale=section_info.get("rationale", ""),
                pages=pages,
            ))
        
        total_pages = sum(len(s.pages) for s in sections)
        
        return WikiStructureSpec(
            wiki_title=f"{self.repo_name} Documentation" if self.repo_name else "Documentation",
            overview="",  # Filled by repository analysis node
            sections=sections,
            total_pages=total_pages,
        )
    
    def _get_dominant_symbols(self, node_ids: list[str], limit: int = 10) -> list[dict]:
        """Get top symbols for section naming, ranked by PageRank + type priority."""
        # Compute PageRank on subgraph
        subgraph = self.graph.subgraph(node_ids)
        try:
            pr = nx.pagerank(subgraph.to_undirected())
        except Exception:
            pr = {n: 1.0 / len(node_ids) for n in node_ids}
        
        # Score = PageRank × type_priority (higher priority = higher weight)
        scored = []
        for nid in node_ids:
            data = self.graph.nodes.get(nid, {})
            symbol_type = data.get("symbol", {}).get("kind", "") or data.get("symbol_type", "")
            priority = SYMBOL_TYPE_PRIORITY.get(symbol_type, 0)
            scored.append((nid, pr.get(nid, 0) * priority))
        
        scored.sort(key=lambda x: x[1], reverse=True)
        
        return [self._node_to_symbol_info(nid) for nid, _ in scored[:limit]]
    
    def _get_central_symbols(self, node_ids: list[str], limit: int = 8) -> list[dict]:
        """Get central symbols for page naming — page-only (no centroid bleed)."""
        return self._get_dominant_symbols(node_ids, limit=limit)
    
    def _node_to_symbol_info(self, node_id: str) -> dict:
        """Extract symbol info dict from graph node."""
        data = self.graph.nodes.get(node_id, {})
        symbol = data.get("symbol", {})
        location = data.get("location", {})
        return {
            "name": symbol.get("name", "") or data.get("symbol_name", ""),
            "type": symbol.get("kind", "") or data.get("symbol_type", ""),
            "signature": symbol.get("signature", "") or data.get("signature", ""),
            "file": location.get("rel_path", "") or data.get("rel_path", ""),
            "docstring": (data.get("docstring", "") or "")[:200],
        }
    
    def _node_ids_to_symbol_names(self, node_ids: list[str]) -> list[str]:
        """Extract symbol names from node IDs."""
        names = []
        for nid in node_ids:
            data = self.graph.nodes.get(nid, {})
            name = data.get("symbol", {}).get("name", "") or data.get("symbol_name", "")
            if name:
                names.append(name)
        return names
    
    def _node_ids_to_doc_paths(self, node_ids: list[str]) -> list[str]:
        """Extract documentation file paths from doc-type nodes."""
        docs = []
        for nid in node_ids:
            data = self.graph.nodes.get(nid, {})
            symbol_type = data.get("symbol", {}).get("kind", "") or data.get("symbol_type", "")
            if "document" in symbol_type:
                path = data.get("location", {}).get("rel_path", "") or data.get("rel_path", "")
                if path:
                    docs.append(path)
        return docs
    
    def _node_ids_to_directories(self, node_ids: list[str]) -> list[str]:
        """Extract unique directories from node file paths."""
        dirs = set()
        for nid in node_ids:
            data = self.graph.nodes.get(nid, {})
            path = data.get("location", {}).get("rel_path", "") or data.get("rel_path", "")
            if path:
                parts = path.rsplit("/", 1)
                if len(parts) > 1:
                    dirs.add(parts[0])
        return sorted(dirs)
    
    def _node_ids_to_file_paths(self, node_ids: list[str]) -> list[str]:
        """Extract unique file paths from nodes."""
        paths = set()
        for nid in node_ids:
            data = self.graph.nodes.get(nid, {})
            path = data.get("location", {}).get("rel_path", "") or data.get("rel_path", "")
            if path:
                paths.add(path)
        return sorted(paths)
    
    async def _name_section(self, dominant_symbols: list[dict], page_map: dict) -> dict:
        """LLM call to name a macro-cluster (section)."""
        import json
        prompt = SECTION_NAMING_PROMPT.format(
            symbols_json=json.dumps(dominant_symbols, indent=2),
            page_count=len(page_map),
            repo_name=self.repo_name,
        )
        try:
            response = await self.llm.ainvoke(prompt)
            return self._parse_naming_response(response.content)
        except Exception as e:
            logger.warning(f"Section naming failed: {e}")
            # Fallback: use dominant symbol name
            fallback_name = dominant_symbols[0]["name"] if dominant_symbols else "Section"
            return {"section_name": fallback_name, "description": ""}
    
    async def _name_page(self, central_symbols: list[dict], node_ids: list[str], section_name: str) -> dict:
        """LLM call to name a micro-cluster (page)."""
        import json
        prompt = PAGE_NAMING_PROMPT.format(
            symbols_json=json.dumps(central_symbols, indent=2),
            section_name=section_name,
            node_count=len(node_ids),
            directories=", ".join(self._node_ids_to_directories(node_ids)[:5]),
        )
        try:
            response = await self.llm.ainvoke(prompt)
            return self._parse_naming_response(response.content)
        except Exception as e:
            logger.warning(f"Page naming failed: {e}")
            fallback_name = central_symbols[0]["name"] if central_symbols else "Page"
            return {"page_name": fallback_name, "description": ""}
    
    def _parse_naming_response(self, content: str) -> dict:
        """Parse LLM naming response (JSON expected)."""
        import json
        # Try to extract JSON from response
        try:
            # Handle markdown-wrapped JSON
            if "```json" in content:
                content = content.split("```json")[1].split("```")[0]
            elif "```" in content:
                content = content.split("```")[1].split("```")[0]
            return json.loads(content.strip())
        except (json.JSONDecodeError, IndexError):
            # Fallback: treat first line as name
            lines = content.strip().split("\n")
            return {"section_name": lines[0].strip(), "page_name": lines[0].strip(), "description": ""}
```

---

## 5. Integration Points

### Planner Routing in `wiki_graph_optimized.py`

The `generate_wiki_structure` node chooses the planner based on `planner_type`:

```python
# In OptimizedWikiGenerationAgent.generate_wiki_structure()

async def generate_wiki_structure(self, state, config):
    planner_type = state.get("planner_type", "deepagents")
    
    if planner_type == "cluster":
        from app.core.graph_clustering import run_clustering
        from app.core.wiki_structure_planner.cluster_planner import ClusterStructurePlanner
        
        # Run deterministic clustering
        assignment = run_clustering(
            graph=self.graph,
            hub_percentile=settings.cluster_hub_percentile,
        )
        
        # Emit progress
        self._emit_progress(f"Detected {assignment.section_count} sections, {assignment.page_count} pages")
        
        # Name clusters via LLM
        planner = ClusterStructurePlanner(
            graph=self.graph,
            cluster_assignment=assignment,
            llm=self.llm,
            repo_name=self.repo_name,
            progress_callback=self._emit_progress,
        )
        structure = await planner.plan_structure()
    else:
        # Existing DeepAgents planner path
        structure = await self._plan_with_deepagents(state, config)
    
    return {"wiki_structure_spec": structure, "structure_planning_complete": True}
```

### API Request Extension

```python
# In backend/app/models/api.py

class GenerateWikiRequest(BaseModel):
    # ... existing fields ...
    planner_type: str = "deepagents"  # "deepagents" | "cluster"
    
    @field_validator("planner_type")
    @classmethod
    def _validate_planner_type(cls, v):
        if v not in ("deepagents", "cluster"):
            raise ValueError(f"planner_type must be 'deepagents' or 'cluster', got '{v}'")
        return v
```

### Configuration Extension

```python
# In backend/app/config.py — add to Settings class

# Cluster Planner
cluster_planner_enabled: bool = False
cluster_hub_percentile: int = 90
```

---

## 6. SSE Event Contract

The cluster planner emits events through the same `invocation.emit()` mechanism. Event types are unchanged — only the message content differs:

```python
# Phase: structure planning with cluster planner
progress(invocation_id, progress=30, message="Running graph clustering (Leiden)...", phase="structure")
progress(invocation_id, progress=32, message="Detected 8 sections, 24 pages (0.3s)", phase="structure")
progress(invocation_id, progress=34, message="Naming section 1/8: analyzing symbols...", phase="structure")
progress(invocation_id, progress=36, message="Named: 'Authentication & Authorization'", phase="structure")
# ... repeat for sections 2-8 ...
progress(invocation_id, progress=44, message="Structure planning complete (24 pages)", phase="structure")
```

No new event types are needed. The frontend SSE handler already processes `progress` events with `phase` and `message` fields.

---

## 7. Dependency Changes

### `pyproject.toml` (or `requirements.txt`)

```toml
[project.optional-dependencies]
cluster = [
    "python-igraph==1.0.0",
    "leidenalg==0.11.0",
]
```

Both are **required for the cluster planner** (optional for the overall application). If not installed, the cluster planner is disabled and users must use the DeepAgents planner.

### Runtime Check

```python
# In graph_clustering.py
try:
    import igraph as ig
    import leidenalg
    LEIDEN_AVAILABLE = True
except ImportError:
    LEIDEN_AVAILABLE = False
```

---

## 8. Testing Strategy

### Unit Tests (Phase 1)

| Test | File | What |
|------|------|------|
| Architectural projection | `tests/unit/test_graph_clustering.py` | Collapse methods → parent, edge remapping, orphan removal |
| Hub detection | `tests/unit/test_graph_clustering.py` | PageRank threshold, cap at 15% |
| Macro-clustering | `tests/unit/test_graph_clustering.py` | Section count formula, section cap enforcement |
| Micro-clustering | `tests/unit/test_graph_clustering.py` | Page sizing, merge/split |
| File contraction | `tests/unit/test_graph_clustering.py` | Cross-file edge summing, same-file dropped |
| Hub re-integration | `tests/unit/test_graph_clustering.py` | Plurality voting correctness |
| Trivial assignment | `tests/unit/test_graph_clustering.py` | < 3 nodes returns single section/page |
| Constants | `tests/unit/test_cluster_constants.py` | Resolution formula, page sizing, section cap |

### Integration Tests (Phase 2)

| Test | File | What |
|------|------|------|
| Cluster planner end-to-end | `tests/integration/test_cluster_planner.py` | Graph → ClusterAssignment → WikiStructureSpec |
| Planner routing | `tests/integration/test_planner_routing.py` | `planner_type="cluster"` selects correct planner |
| Fallback to DeepAgents | `tests/integration/test_planner_routing.py` | Empty graph → falls back to DeepAgents |
| API field validation | `tests/unit/test_api_models.py` | `planner_type` validation |
