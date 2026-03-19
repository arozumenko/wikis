"""
Deterministic Structure Skeleton Builder

Builds a complete repository structure skeleton without any LLM calls.
Uses graph topology (Louvain community detection) to cluster directories
into page-sized groups, ensuring 100% directory and symbol coverage by
construction.

Phase 1 of the Graph-First approach:
  1a. build_dir_symbol_map   – walk graph, extract architectural symbols
  1b. build_dir_interaction_graph – weighted cross-dir edges
  1c. cluster_directories    – Louvain community detection
  1d. normalize_clusters     – split large / merge small
  1e. build_doc_clusters     – doc-only directories
  1f. build_skeleton         – top-level orchestrator
"""

from __future__ import annotations

import logging
import math
import os
from collections import Counter, defaultdict
from dataclasses import dataclass

import networkx as nx

from ..constants import (
    COVERAGE_IGNORE_DIRS,
    DOCUMENTATION_EXTENSIONS_SET,
    classify_symbol_layer,
)

logger = logging.getLogger(__name__)

# ── Symbol type filter ────────────────────────────────────────────────
# Architectural types only – no methods, no doc symbols.
# Macro excluded from expansion but kept in graph; we include it in the
# skeleton because it affects coupling analysis.
_ARCH_TYPES: frozenset[str] = frozenset(
    {
        "class",
        "interface",
        "struct",
        "enum",
        "trait",
        "function",
        "constant",
        "type_alias",
        "macro",
    }
)

# ── Code-file extensions (fast O(1) lookup) ───────────────────────────
_CODE_EXT_SET: frozenset[str] = frozenset(
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

# ── Weighted edge table ──────────────────────────────────────────────
# Not all RelationshipType values contribute equally to architectural
# coupling between directories.
#
# NOTE on USES_TYPE vs REFERENCES: currently REFERENCES is used broadly
# (including for type usage) with metadata.  We should migrate type-usage
# cases to USES_TYPE.  Until then both get medium weight.
_EDGE_WEIGHT: dict[str, int] = {
    # Strong coupling (3) – architectural dependency
    "inheritance": 3,
    "implementation": 3,
    "composition": 3,
    "overrides": 3,
    # Medium coupling (2) – standard cross-dir dependencies
    "imports": 2,
    "calls": 2,
    "creates": 2,
    "instantiates": 2,
    "uses_type": 2,
    "references": 2,
    # Weak coupling (1) – looser relationships
    "aggregation": 1,
    "reads": 1,
    "writes": 1,
    "captures": 1,
    "specializes": 1,
    # Everything else (defines, contains, defines_body, declares,
    # parameter, returns, annotates, decorates, assigns, exports,
    # alias_of, hides) → 0 (structural/internal, mostly same-file).
}
_DEFAULT_EDGE_WEIGHT = 0


# =====================================================================
# Data classes
# =====================================================================


@dataclass
class SymbolInfo:
    """Lightweight data for one architectural symbol."""

    name: str
    type: str  # class, function, …
    rel_path: str  # file containing the symbol
    layer: str  # entry_point / public_api / core_type / …
    connections: int  # degree in the code graph
    docstring: str  # first ≤ 200 chars


@dataclass
class DirCluster:
    """A group of directories that should become one wiki page."""

    cluster_id: int
    dirs: list[str]
    symbols: list[SymbolInfo]
    total_symbols: int
    primary_languages: list[str]
    depth_range: tuple[int, int]  # (min_depth, max_depth)


@dataclass
class DocCluster:
    """A documentation-only directory needing wiki coverage."""

    dir_path: str
    doc_files: list[str]
    doc_types: list[str]
    file_count: int


@dataclass
class StructureSkeleton:
    """Complete deterministic skeleton ready for LLM refinement."""

    code_clusters: list[DirCluster]
    doc_clusters: list[DocCluster]
    total_arch_symbols: int
    total_dirs_covered: int
    total_dirs_in_repo: int
    repo_languages: list[str]
    effective_depth: int
    repo_name: str = ""


# =====================================================================
# 1a. Build dir → symbols map
# =====================================================================


def build_dir_symbol_map(
    code_graph: nx.DiGraph,
) -> dict[str, list[SymbolInfo]]:
    """Map every directory to its architectural symbols.

    Returns:
        dict mapping directory path → list of SymbolInfo.
        Root-level files map to ``"."``.
    """
    dir_symbols: dict[str, list[SymbolInfo]] = defaultdict(list)

    for node_id, data in code_graph.nodes(data=True):
        sym_type = (data.get("symbol_type") or data.get("type") or "").lower()
        if sym_type not in _ARCH_TYPES:
            continue

        sym_name = data.get("symbol_name") or data.get("name", "")
        rel_path = data.get("rel_path") or data.get("file_path", "")
        if not sym_name or not rel_path:
            continue

        # Directory of the file containing the symbol
        parts = rel_path.split("/")
        dir_path = "/".join(parts[:-1]) if len(parts) > 1 else "."

        # Skip ignored directories
        if any(seg in COVERAGE_IGNORE_DIRS for seg in parts[:-1]):
            continue

        layer = data.get("layer", "")
        if not layer:
            layer = classify_symbol_layer(sym_name, sym_type, rel_path)

        docstring = (data.get("docstring") or "")[:200]

        dir_symbols[dir_path].append(
            SymbolInfo(
                name=sym_name,
                type=sym_type,
                rel_path=rel_path,
                layer=layer,
                connections=code_graph.degree(node_id),
                docstring=docstring,
            )
        )

    return dict(dir_symbols)


# =====================================================================
# 1b. Build directory interaction graph (weighted)
# =====================================================================


def build_dir_interaction_graph(
    code_graph: nx.DiGraph,
    dir_symbols: dict[str, list[SymbolInfo]],
) -> nx.Graph:
    """Build an undirected weighted graph of directory interactions.

    Nodes = directories that have ≥ 1 architectural symbol.
    Edges  = weighted sum of cross-directory relationships.
    """
    # node_id → directory lookup
    node_to_dir: dict[str, str] = {}
    for node_id, data in code_graph.nodes(data=True):
        rel_path = data.get("rel_path") or data.get("file_path", "")
        if not rel_path:
            continue
        parts = rel_path.split("/")
        node_to_dir[node_id] = "/".join(parts[:-1]) if len(parts) > 1 else "."

    dir_graph = nx.Graph()
    for dir_path, symbols in dir_symbols.items():
        dir_graph.add_node(dir_path, symbol_count=len(symbols))

    # Accumulate weighted cross-dir edges
    edge_weights: dict[tuple[str, str], int] = defaultdict(int)
    for src, tgt, data in code_graph.edges(data=True):
        src_dir = node_to_dir.get(src)
        tgt_dir = node_to_dir.get(tgt)
        if not src_dir or not tgt_dir or src_dir == tgt_dir:
            continue
        # Only count if both dirs have architectural symbols
        if src_dir not in dir_symbols or tgt_dir not in dir_symbols:
            continue
        rel_type = (data.get("relationship_type") or data.get("type") or "").lower()
        w = _EDGE_WEIGHT.get(rel_type, _DEFAULT_EDGE_WEIGHT)
        if w > 0:
            key = (min(src_dir, tgt_dir), max(src_dir, tgt_dir))
            edge_weights[key] += w

    for (d1, d2), weight in edge_weights.items():
        dir_graph.add_edge(d1, d2, weight=weight)

    return dir_graph


# =====================================================================
# 1c. Cluster directories
# =====================================================================


def cluster_directories(
    dir_graph: nx.Graph,
    page_budget: int,
    dir_symbols: dict[str, list[SymbolInfo]],
    min_symbols: int = 10,
) -> list[set[str]]:
    """Partition directories into clusters using Louvain community detection.

    For graphs with few/no edges falls back to directory-tree proximity
    clustering (siblings under the same parent are grouped together).

    Returns list of sets, each set = directory paths in one cluster.
    """
    if len(dir_graph.nodes) == 0:
        return []

    # ── Check if graph has meaningful edges ───────────────────────────
    total_edge_weight = sum(d.get("weight", 1) for _, _, d in dir_graph.edges(data=True))
    if total_edge_weight < len(dir_graph.nodes) * 0.3:
        # Too few edges – Louvain won't produce meaningful communities
        logger.info(
            "[SKELETON] Sparse edge graph (%d nodes, total_weight=%d) → falling back to directory-tree clustering",
            len(dir_graph.nodes),
            total_edge_weight,
        )
        return _cluster_by_directory_tree(dir_graph, dir_symbols, min_symbols)

    # ── Louvain with adaptive resolution ──────────────────────────────
    # Higher resolution → more (smaller) communities
    # Binary-search to get cluster count ≈ page_budget ± 30%
    best_communities: list[set[str]] = []
    best_diff = float("inf")
    target = page_budget

    for resolution in (0.5, 0.7, 1.0, 1.3, 1.6, 2.0, 2.5, 3.0):
        try:
            communities = list(
                nx.community.louvain_communities(
                    dir_graph,
                    weight="weight",
                    resolution=resolution,
                    seed=42,
                )
            )
        except Exception:  # noqa: S112
            continue
        diff = abs(len(communities) - target)
        if diff < best_diff:
            best_diff = diff
            best_communities = communities
        # Good enough
        if diff <= max(3, target * 0.2):
            break

    if not best_communities:
        # Louvain failed entirely – fallback
        return _cluster_by_directory_tree(dir_graph, dir_symbols, min_symbols)

    return best_communities


def _cluster_by_directory_tree(
    dir_graph: nx.Graph,
    dir_symbols: dict[str, list[SymbolInfo]],
    min_symbols: int = 10,
) -> list[set[str]]:
    """Fallback clustering: group sibling directories under same parent.

    Used when the code graph has too few cross-directory edges for
    community detection to work.
    """
    parent_groups: dict[str, list[str]] = defaultdict(list)
    for dir_path in dir_graph.nodes:
        parts = dir_path.split("/")
        parent = "/".join(parts[:-1]) if len(parts) > 1 else "<root>"
        parent_groups[parent].append(dir_path)

    clusters: list[set[str]] = []
    for _parent, children in parent_groups.items():
        group: set[str] = set()
        group_syms = 0
        for child in sorted(children):
            sym_count = len(dir_symbols.get(child, []))
            group.add(child)
            group_syms += sym_count
            if group_syms >= min_symbols * 2:
                clusters.append(group)
                group = set()
                group_syms = 0
        if group:
            clusters.append(group)

    return clusters


# =====================================================================
# 1d. Normalize cluster sizes (split large / merge small)
# =====================================================================


def normalize_clusters(
    raw_clusters: list[set[str]],
    dir_symbols: dict[str, list[SymbolInfo]],
    min_symbols: int = 10,
    max_symbols: int = 40,
    page_budget: int = 30,
) -> list[DirCluster]:
    """Split oversized clusters and merge tiny ones.

    Returns final list of ``DirCluster`` objects with IDs assigned.
    """
    # ── First pass: split large, tag small for merging ────────────────
    normal: list[set[str]] = []
    small: list[set[str]] = []  # < min_symbols

    for cluster_dirs in raw_clusters:
        total = sum(len(dir_symbols.get(d, [])) for d in cluster_dirs)
        if total > max_symbols and len(cluster_dirs) > 1:
            normal.extend(_split_large(cluster_dirs, dir_symbols, max_symbols))
        elif total < min_symbols:
            small.append(cluster_dirs)
        else:
            normal.append(cluster_dirs)

    # ── Second pass: merge small clusters into nearest neighbour ──────
    for smol in small:
        merged = False
        best_idx = -1
        best_overlap = -1
        smol_parents = {"/".join(d.split("/")[:-1]) if "/" in d else "<root>" for d in smol}
        for idx, existing in enumerate(normal):
            existing_parents = {"/".join(d.split("/")[:-1]) if "/" in d else "<root>" for d in existing}
            overlap = len(smol_parents & existing_parents)
            total_after = sum(len(dir_symbols.get(d, [])) for d in existing) + sum(
                len(dir_symbols.get(d, [])) for d in smol
            )
            if overlap > best_overlap and total_after <= max_symbols * 1.5:
                best_overlap = overlap
                best_idx = idx
        if best_idx >= 0:
            normal[best_idx] = normal[best_idx] | smol
            merged = True
        if not merged:
            normal.append(smol)  # keep as-is even if small

    # ── Build DirCluster objects ──────────────────────────────────────
    result: list[DirCluster] = []
    for cid, dirs_set in enumerate(normal, start=1):
        dirs = sorted(dirs_set)
        symbols: list[SymbolInfo] = []
        for d in dirs:
            symbols.extend(dir_symbols.get(d, []))

        if not symbols:
            continue

        # Primary languages
        ext_counts = Counter(s.rel_path.rsplit(".", 1)[-1].lower() for s in symbols if "." in s.rel_path)
        primary_langs = [ext for ext, _ in ext_counts.most_common(3)]

        # Depth range
        depths = [d.count("/") for d in dirs]
        depth_range = (min(depths), max(depths)) if depths else (0, 0)

        result.append(
            DirCluster(
                cluster_id=cid,
                dirs=dirs,
                symbols=symbols,
                total_symbols=len(symbols),
                primary_languages=primary_langs,
                depth_range=depth_range,
            )
        )

    logger.info(
        "[SKELETON] Normalized %d raw clusters → %d DirClusters (total symbols=%d)",
        len(raw_clusters),
        len(result),
        sum(c.total_symbols for c in result),
    )
    return result


def _split_large(
    cluster_dirs: set[str],
    dir_symbols: dict[str, list[SymbolInfo]],
    max_symbols: int,
) -> list[set[str]]:
    """Split an oversized cluster into 2+ sub-clusters.

    Strategy: sort dirs by symbol count descending, round-robin assign
    them to sub-groups targeting ≤ max_symbols each.
    """
    ordered = sorted(
        cluster_dirs,
        key=lambda d: len(dir_symbols.get(d, [])),
        reverse=True,
    )
    n_groups = max(2, math.ceil(sum(len(dir_symbols.get(d, [])) for d in cluster_dirs) / max_symbols))
    groups: list[list[str]] = [[] for _ in range(n_groups)]
    group_sizes = [0] * n_groups

    for d in ordered:
        # Assign to the lightest group
        idx = min(range(n_groups), key=lambda i: group_sizes[i])
        groups[idx].append(d)
        group_sizes[idx] += len(dir_symbols.get(d, []))

    return [set(g) for g in groups if g]


# =====================================================================
# 1e. Build documentation clusters
# =====================================================================


def build_doc_clusters(
    repository_files: list[str],
    dir_symbols: dict[str, list[SymbolInfo]],
    effective_depth: int,
) -> list[DocCluster]:
    """Identify directories containing only documentation files.

    These supplement code clusters for mixed repos and form the entire
    skeleton for doc-only repos.
    """
    dir_doc_files: dict[str, list[str]] = defaultdict(list)

    for path in repository_files:
        if not path:
            continue
        parts = path.split("/")
        # Skip ignored directories
        if any(seg in COVERAGE_IGNORE_DIRS for seg in parts):
            continue

        ext = os.path.splitext(path)[1].lower()
        if ext not in DOCUMENTATION_EXTENSIONS_SET:
            continue

        # Assign to deepest qualifying directory
        if len(parts) > 1:
            dir_path = "/".join(parts[:-1])
        else:
            dir_path = "."
        dir_doc_files[dir_path].append(path)

    # Filter to doc-only directories (no code symbols, ≥1 doc file)
    raw_doc_dirs: dict[str, list[str]] = {}
    for dir_path, doc_files in sorted(dir_doc_files.items()):
        if dir_path in dir_symbols and len(dir_symbols[dir_path]) > 0:
            continue
        if not doc_files:
            continue
        raw_doc_dirs[dir_path] = doc_files

    # ── Group nearby doc dirs to avoid 50+ tiny doc pages ─────────────
    # Strategy: group by top-2-level prefix (e.g. proto/redpanda/…/v1
    # and proto/redpanda/…/datalake share group prefix "proto/redpanda").
    # Root-level doc dirs each get their own cluster.
    group_map: dict[str, list[str]] = defaultdict(list)
    for dir_path in raw_doc_dirs:
        parts = dir_path.split("/")
        if len(parts) <= 2:
            group_key = dir_path  # keep shallow dirs standalone
        else:
            group_key = "/".join(parts[:2])  # group by first 2 segments
        group_map[group_key].append(dir_path)

    doc_clusters: list[DocCluster] = []
    for group_key, dir_list in sorted(group_map.items()):
        merged_files: list[str] = []
        for dp in dir_list:
            merged_files.extend(raw_doc_dirs[dp])
        doc_types = sorted(set(os.path.splitext(f)[1].lower() for f in merged_files))
        # Use the group key as the representative path
        doc_clusters.append(
            DocCluster(
                dir_path=group_key,
                doc_files=merged_files,
                doc_types=doc_types,
                file_count=len(merged_files),
            )
        )

    logger.info(
        "[SKELETON] Found %d doc-only clusters (%d total doc files, grouped from %d raw doc dirs)",
        len(doc_clusters),
        sum(dc.file_count for dc in doc_clusters),
        len(raw_doc_dirs),
    )
    return doc_clusters


# =====================================================================
# 1f. Build the complete skeleton (top-level orchestrator)
# =====================================================================


def build_skeleton(
    code_graph: nx.DiGraph,
    repository_files: list[str],
    page_budget: int,
    effective_depth: int,
    repo_name: str = "",
) -> StructureSkeleton:
    """Build the complete deterministic skeleton.

    Orchestrates steps 1a–1e and returns a StructureSkeleton ready for
    LLM refinement.  Zero LLM calls — pure graph + filesystem analysis.

    Args:
        code_graph: NetworkX graph produced by ``graph_builder``.
        repository_files: All file paths in the repo (relative).
        page_budget: Target number of wiki pages.
        effective_depth: Maximum directory depth for exploration.
        repo_name: Repository name (for metadata).

    Returns:
        StructureSkeleton with code_clusters, doc_clusters, and stats.
    """
    logger.info(
        "[SKELETON] Building skeleton for %s (budget=%d, depth=%d, files=%d)",
        repo_name or "repo",
        page_budget,
        effective_depth,
        len(repository_files),
    )

    # 1a. dir → symbols
    dir_symbols = build_dir_symbol_map(code_graph)
    total_arch = sum(len(syms) for syms in dir_symbols.values())
    logger.info(
        "[SKELETON] 1a. dir_symbol_map: %d dirs, %d arch symbols",
        len(dir_symbols),
        total_arch,
    )

    # Count only "content directories" — dirs that directly contain ≥1 file.
    # Intermediate path segments (e.g. src/, src/v/) that only contain
    # subdirectories are NOT meaningful for coverage; no one expects a
    # separate wiki page for a directory with zero files in it.
    content_dirs: set[str] = set()
    for path in repository_files:
        parts = path.split("/")
        if any(seg in COVERAGE_IGNORE_DIRS for seg in parts):
            continue
        # Only the immediate parent directory of the file
        if len(parts) > 1:
            content_dirs.add("/".join(parts[:-1]))
        else:
            content_dirs.add(".")

    # 1b. directory interaction graph
    dir_graph = build_dir_interaction_graph(code_graph, dir_symbols)
    logger.info(
        "[SKELETON] 1b. dir_interaction_graph: %d nodes, %d edges, total_weight=%d",
        dir_graph.number_of_nodes(),
        dir_graph.number_of_edges(),
        sum(d.get("weight", 1) for _, _, d in dir_graph.edges(data=True)),
    )

    # 1c. Cluster directories
    raw_clusters = cluster_directories(
        dir_graph,
        page_budget,
        dir_symbols,
    )
    logger.info(
        "[SKELETON] 1c. Louvain produced %d raw clusters",
        len(raw_clusters),
    )

    # 1d. Normalize cluster sizes
    # Scale thresholds to repo size.  With 23k symbols and budget 81
    # the old hardcoded max=40 would split 44 Louvain clusters into 273.
    avg_per_page = max(1, total_arch // max(page_budget, 1))
    dyn_max = max(120, int(avg_per_page * 2.0))  # allow 2× headroom
    dyn_min = max(5, avg_per_page // 5)  # merge if < 20% avg
    logger.info(
        "[SKELETON] 1d. normalize thresholds: avg=%d, min=%d, max=%d",
        avg_per_page,
        dyn_min,
        dyn_max,
    )
    code_clusters = normalize_clusters(
        raw_clusters,
        dir_symbols,
        min_symbols=dyn_min,
        max_symbols=dyn_max,
        page_budget=page_budget,
    )

    # 1e. Documentation clusters
    doc_clusters = build_doc_clusters(
        repository_files,
        dir_symbols,
        effective_depth,
    )

    # Repo-wide language stats
    global_ext = Counter(
        s.rel_path.rsplit(".", 1)[-1].lower() for cluster in code_clusters for s in cluster.symbols if "." in s.rel_path
    )
    repo_languages = [ext for ext, _ in global_ext.most_common(5)]

    skeleton = StructureSkeleton(
        code_clusters=code_clusters,
        doc_clusters=doc_clusters,
        total_arch_symbols=total_arch,
        total_dirs_covered=sum(len(c.dirs) for c in code_clusters),
        total_dirs_in_repo=len(content_dirs),
        repo_languages=repo_languages,
        effective_depth=effective_depth,
        repo_name=repo_name,
    )

    logger.info(
        "[SKELETON] Complete: %d code clusters, %d doc clusters, %d/%d dirs covered, %d arch symbols",
        len(code_clusters),
        len(doc_clusters),
        skeleton.total_dirs_covered,
        skeleton.total_dirs_in_repo,
        total_arch,
    )
    return skeleton
