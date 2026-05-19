"""Doc-graph clustering for markdown, Confluence, and Jira sources (#230).

Builds a document link graph for each source kind and runs Louvain community
detection to emit ``Cluster`` objects that feed the unified planner pipeline.

This is the doc-graph equivalent of the code-graph clustering in
``structure_skeleton.build_skeleton``. Three public functions are exposed
(one per source kind) rather than a single dispatcher because their edge
sources, hierarchy models, and metadata needs differ enough that a single
signature would obscure more than it clarifies.

Public API
----------
cluster_markdown_docs(artifacts, *, repo_root)    -> list[Cluster]
cluster_confluence_pages(artifacts, *, repo_root) -> list[Cluster]
cluster_jira_issues(artifacts)                    -> list[Cluster]

Algorithm (shared)
------------------
1. Build a ``networkx.Graph`` (undirected, weighted) whose nodes are
   artifact ``source_path`` values.
2. Add edges specific to the source kind:
   - markdown: internal links parsed by ``extract_links``
   - confluence: parent-child hierarchy + internal links
   - jira: epic → story/subtask membership + cross-issue links
3. Run Louvain via ``nx.community.louvain_communities``. If the graph has
   too few edges (or Louvain fails) fall back to connected-component
   partitioning so every artifact ends up in exactly one cluster.
4. Convert each community → ``Cluster`` object. ``dirs`` is derived from
   the unique directory prefixes of the member ``source_path`` values (or
   an empty list for Jira, where paths are logical keys, not directories).
"""

from __future__ import annotations

import logging
import os
from collections import defaultdict
from pathlib import Path

import networkx as nx

from ..parsers.markdown_links import extract_links
from ._evidence_utils import safe_join
from .structure_skeleton import ArtifactInfo, Cluster

logger = logging.getLogger(__name__)

# Weight assigned to a link edge; parent-child edges get a higher weight
# because hierarchy is a stronger clustering signal than an occasional inline
# link across pages.
_LINK_EDGE_WEIGHT = 1
_PARENT_CHILD_EDGE_WEIGHT = 3
_EPIC_MEMBERSHIP_WEIGHT = 3


# ── Internal graph helpers ────────────────────────────────────────────────────


def _louvain_or_components(
    graph: nx.Graph,
) -> list[set[str]]:
    """Run Louvain; fall back to connected-component partitioning.

    Returns a list of sets, each containing the ``source_path`` identifiers
    that belong to one community.
    """
    if graph.number_of_nodes() == 0:
        return []

    # If the graph has no edges at all, every node is its own community.
    if graph.number_of_edges() == 0:
        return [{n} for n in graph.nodes()]

    total_weight = sum(d.get("weight", 1) for _, _, d in graph.edges(data=True))
    # Threshold: at least 0.3 edges per node on average for Louvain to be
    # meaningful (mirrors the heuristic in structure_skeleton.cluster_directories).
    if total_weight < graph.number_of_nodes() * 0.3:
        logger.debug(
            "[DOC_CLUSTER] Sparse graph (%d nodes, weight=%d) → component fallback",
            graph.number_of_nodes(),
            total_weight,
        )
        return [set(comp) for comp in nx.connected_components(graph)]

    best: list[set[str]] = []
    best_n = None
    for resolution in (0.5, 1.0, 1.5, 2.0):
        try:
            communities = list(
                nx.community.louvain_communities(
                    graph,
                    weight="weight",
                    resolution=resolution,
                    seed=42,
                )
            )
        except Exception:
            continue
        n = len(communities)
        if best_n is None or abs(n - graph.number_of_nodes() // 3) < abs(best_n - graph.number_of_nodes() // 3):
            best = communities
            best_n = n

    if not best:
        # Louvain failed entirely — connected components
        return [set(comp) for comp in nx.connected_components(graph)]

    return best


def _communities_to_clusters(
    communities: list[set[str]],
    path_to_artifact: dict[str, ArtifactInfo],
    cluster_kind: str,
) -> list[Cluster]:
    """Convert Louvain communities to ``Cluster`` objects.

    Args:
        communities: Each set contains artifact ``source_path`` keys.
        path_to_artifact: Maps source_path → ArtifactInfo.
        cluster_kind: One of ``"doc"``, ``"confluence"``, ``"jira"``.
    """
    result: list[Cluster] = []
    for cid, community in enumerate(communities, start=1):
        artifacts = [
            path_to_artifact[path]
            for path in sorted(community)
            if path in path_to_artifact
        ]
        if not artifacts:
            continue

        # Derive ``dirs`` from the directory components of source_path values.
        # For Jira issues the paths are logical keys (e.g. "EPIC-1.md") and
        # the parent dir is "." or empty — use an empty list to signal that
        # the cluster is not filesystem-based.
        if cluster_kind == "jira":
            dirs: list[str] = []
        else:
            dir_set: set[str] = set()
            for a in artifacts:
                parent = os.path.dirname(a.source_path)
                dir_set.add(parent if parent else ".")
            dirs = sorted(dir_set)

        # Depth range: count slashes in each source_path
        depths = [a.source_path.count("/") for a in artifacts]
        depth_range = (min(depths), max(depths)) if depths else (0, 0)

        result.append(
            Cluster(
                cluster_id=cid,
                kind=cluster_kind,  # type: ignore[arg-type]
                dirs=dirs,
                artifacts=artifacts,
                total_artifacts=len(artifacts),
                primary_languages=[],
                depth_range=depth_range,
            )
        )

    return result


def _read_doc_content(source_path: str, repo_root: str) -> str | None:
    """Safely read a doc file from disk.  Returns None if path is unsafe or missing."""
    abs_path = safe_join(repo_root, source_path)
    if abs_path is None:
        return None
    try:
        return Path(abs_path).read_text(encoding="utf-8", errors="replace")
    except (OSError, IOError):
        return None


def _add_link_edges(
    graph: nx.Graph,
    source_path: str,
    repo_root: str,
    path_set: set[str],
    weight: int = _LINK_EDGE_WEIGHT,
) -> None:
    """Parse links in a doc file and add edges for each resolved internal target."""
    content = _read_doc_content(source_path, repo_root)
    if not content:
        return
    links = extract_links(content, source_path=source_path)
    for link in links:
        if link.kind not in ("internal", "wikilink"):
            continue
        target = link.resolved or link.target
        if not target:
            continue
        # Normalize to forward slashes and strip leading "./"
        target = target.replace("\\", "/").lstrip("./")
        if target in path_set:
            if graph.has_edge(source_path, target):
                graph[source_path][target]["weight"] += weight
            else:
                graph.add_edge(source_path, target, weight=weight)


# ── Public API ────────────────────────────────────────────────────────────────


def cluster_markdown_docs(
    artifacts: list[ArtifactInfo],
    *,
    repo_root: str,
) -> list[Cluster]:
    """Cluster markdown doc artifacts into ``Cluster(kind='doc')`` objects.

    Builds an undirected graph where each node is an artifact keyed by
    ``source_path`` and edges are internal/wikilinks between docs. Louvain
    community detection (or connected-component fallback) partitions the
    graph into clusters.

    Args:
        artifacts: List of ``ArtifactInfo(kind='doc_section')`` objects.
        repo_root: Absolute path to the repository root (used to resolve
            and safely read doc files).

    Returns:
        List of ``Cluster(kind='doc')`` objects, one per community. Empty
        list when ``artifacts`` is empty.
    """
    if not artifacts:
        return []

    path_to_artifact = {a.source_path: a for a in artifacts}
    path_set = set(path_to_artifact)

    graph = nx.Graph()
    for path in path_set:
        graph.add_node(path)

    for source_path in path_set:
        _add_link_edges(graph, source_path, repo_root, path_set)

    logger.debug(
        "[DOC_CLUSTER] markdown: %d nodes, %d edges",
        graph.number_of_nodes(),
        graph.number_of_edges(),
    )

    communities = _louvain_or_components(graph)
    clusters = _communities_to_clusters(communities, path_to_artifact, "doc")

    logger.info(
        "[DOC_CLUSTER] markdown → %d clusters from %d artifacts",
        len(clusters),
        len(artifacts),
    )
    return clusters


def cluster_confluence_pages(
    artifacts: list[ArtifactInfo],
    *,
    repo_root: str,
) -> list[Cluster]:
    """Cluster Confluence page artifacts into ``Cluster(kind='confluence')`` objects.

    Edges come from two sources:
    1. **Parent-child hierarchy**: every pair sharing the same ``parent_path``
       gets a weighted edge; each page also gets an edge to its parent page if
       the parent appears in the artifact set (by name matching).
    2. **Inline links**: internal links parsed from the exported markdown files
       (same as markdown clustering).

    Args:
        artifacts: List of ``ArtifactInfo(kind='confluence_page')`` objects.
        repo_root: Absolute path to the repository root.

    Returns:
        List of ``Cluster(kind='confluence')`` objects.
    """
    if not artifacts:
        return []

    path_to_artifact = {a.source_path: a for a in artifacts}
    name_to_path = {a.name: a.source_path for a in artifacts}
    path_set = set(path_to_artifact)

    graph = nx.Graph()
    for path in path_set:
        graph.add_node(path)

    # 1. Parent-child edges: group sibling pages by parent_path
    parent_groups: dict[str, list[str]] = defaultdict(list)
    for a in artifacts:
        if a.parent_path:
            parent_groups[a.parent_path].append(a.source_path)

    for parent_path_val, children in parent_groups.items():
        # Connect all siblings under the same parent to each other
        for i, c1 in enumerate(children):
            for c2 in children[i + 1 :]:
                if graph.has_edge(c1, c2):
                    graph[c1][c2]["weight"] += _PARENT_CHILD_EDGE_WEIGHT
                else:
                    graph.add_edge(c1, c2, weight=_PARENT_CHILD_EDGE_WEIGHT)
        # If the parent itself is in the artifact set (by name), connect children to it
        if parent_path_val in name_to_path:
            parent_src = name_to_path[parent_path_val]
            for child in children:
                if graph.has_edge(parent_src, child):
                    graph[parent_src][child]["weight"] += _PARENT_CHILD_EDGE_WEIGHT
                else:
                    graph.add_edge(parent_src, child, weight=_PARENT_CHILD_EDGE_WEIGHT)

    # 2. Inline link edges
    for source_path in path_set:
        _add_link_edges(graph, source_path, repo_root, path_set)

    logger.debug(
        "[DOC_CLUSTER] confluence: %d nodes, %d edges",
        graph.number_of_nodes(),
        graph.number_of_edges(),
    )

    communities = _louvain_or_components(graph)
    clusters = _communities_to_clusters(communities, path_to_artifact, "confluence")

    logger.info(
        "[DOC_CLUSTER] confluence → %d clusters from %d artifacts",
        len(clusters),
        len(artifacts),
    )
    return clusters


def cluster_jira_issues(
    artifacts: list[ArtifactInfo],
) -> list[Cluster]:
    """Cluster Jira issue artifacts into ``Cluster(kind='jira')`` objects.

    Edges come from epic membership: each story/subtask with a non-empty
    ``epic_key`` receives a weighted edge to its epic (if the epic is present
    in the artifact set).  Stories/subtasks with no ``epic_key`` form their
    own isolated nodes.

    No filesystem reads are performed; all clustering data comes from the
    artifact metadata.

    Args:
        artifacts: List of ``ArtifactInfo(kind='jira_issue')`` objects.

    Returns:
        List of ``Cluster(kind='jira')`` objects.
    """
    if not artifacts:
        return []

    path_to_artifact = {a.source_path: a for a in artifacts}
    name_to_path = {a.name: a.source_path for a in artifacts}

    graph = nx.Graph()
    for path in path_to_artifact:
        graph.add_node(path)

    # Epic → child edges
    for a in artifacts:
        if a.epic_key and a.epic_key in name_to_path:
            epic_path = name_to_path[a.epic_key]
            child_path = a.source_path
            if graph.has_edge(epic_path, child_path):
                graph[epic_path][child_path]["weight"] += _EPIC_MEMBERSHIP_WEIGHT
            else:
                graph.add_edge(epic_path, child_path, weight=_EPIC_MEMBERSHIP_WEIGHT)

    logger.debug(
        "[DOC_CLUSTER] jira: %d nodes, %d edges",
        graph.number_of_nodes(),
        graph.number_of_edges(),
    )

    communities = _louvain_or_components(graph)
    clusters = _communities_to_clusters(communities, path_to_artifact, "jira")

    logger.info(
        "[DOC_CLUSTER] jira → %d clusters from %d artifacts",
        len(clusters),
        len(artifacts),
    )
    return clusters
