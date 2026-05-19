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
   - jira: epic → story/subtask membership
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
import posixpath
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

    # Target cluster count = nodes / 3.  Empirically this lands close to the
    # "one wiki page per ~3 docs" cardinality the unified planner produces
    # downstream — fewer than that and clusters become noisy, more than that
    # and we get too many tiny wiki pages.  Sweep four Louvain resolutions and
    # keep the run whose community count is closest to the target.  Tiny
    # graphs (n ≤ 5) all hit the same target (1 or 2 clusters) regardless of
    # resolution and behave as if Louvain was a one-shot call.
    target = graph.number_of_nodes() // 3
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
        if best_n is None or abs(n - target) < abs(best_n - target):
            best = communities
            best_n = n

    if not best:
        # Louvain failed entirely — connected components
        return [set(comp) for comp in nx.connected_components(graph)]

    return best


def _communities_to_clusters(
    communities: list[set[str]],
    path_to_artifacts: dict[str, list[ArtifactInfo]],
    cluster_kind: str,
) -> list[Cluster]:
    """Convert Louvain communities to ``Cluster`` objects.

    Args:
        communities: Each set contains artifact ``source_path`` keys.
        path_to_artifacts: Maps source_path → list of ArtifactInfo. Multiple
            artifacts can share a source_path (e.g. doc_section artifacts for
            different headings in the same file); all of them are expanded
            into the resulting cluster so `sum(c.total_artifacts) ==
            len(input_artifacts)`.
        cluster_kind: One of ``"doc"``, ``"confluence"``, ``"jira"``.
    """
    result: list[Cluster] = []
    for cid, community in enumerate(communities, start=1):
        artifacts: list[ArtifactInfo] = []
        for path in sorted(community):
            artifacts.extend(path_to_artifacts.get(path, []))
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


def _last_breadcrumb_segment(breadcrumb: str) -> str:
    """Return the immediate-parent page title from a Confluence breadcrumb.

    Confluence frontmatter stores ``parent_path`` as a delimited breadcrumb
    (e.g. ``"Engineering / Security"`` or ``"Engineering > Backend > Auth"``).
    The immediate parent is the last segment after splitting on either ``/``
    or ``>``; returns the trimmed string, or ``""`` if no segments.
    """
    # Try ``>`` first (Confluence default in some scanners), then ``/``.
    sep = ">" if ">" in breadcrumb else "/"
    parts = [p.strip() for p in breadcrumb.split(sep) if p.strip()]
    return parts[-1] if parts else ""


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
        # Normalize: forward slashes, strip any fragment/anchor (`#section`),
        # drop a single "./" prefix, then normpath.  Reject targets that
        # resolve outside the cluster's path set (e.g. `../other.md` from a
        # doc whose source_path is at the root).
        target = target.replace("\\", "/")
        target = target.split("#", 1)[0]
        if not target:
            continue
        target = target.removeprefix("./")
        target = posixpath.normpath(target)
        if target.startswith("..") or target == ".":
            continue
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

    # Multiple artifacts can share a source_path (e.g. doc_section entries
    # per heading in the same file); the graph is keyed by path but every
    # artifact is preserved in the resulting clusters.
    path_to_artifacts: dict[str, list[ArtifactInfo]] = defaultdict(list)
    for a in artifacts:
        path_to_artifacts[a.source_path].append(a)
    path_set = set(path_to_artifacts)

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
    clusters = _communities_to_clusters(communities, path_to_artifacts, "doc")

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

    path_to_artifacts: dict[str, list[ArtifactInfo]] = defaultdict(list)
    for a in artifacts:
        path_to_artifacts[a.source_path].append(a)

    # Build the name → path index.  If two pages share a title (common in
    # large Confluence spaces where the same heading appears under multiple
    # parents), warn and keep the first occurrence — silent last-writer-wins
    # would misdirect parent → child edges to whichever page happened to be
    # last in iteration order.
    name_to_path: dict[str, str] = {}
    for a in artifacts:
        if a.name in name_to_path and name_to_path[a.name] != a.source_path:
            logger.warning(
                "[DOC_CLUSTER] confluence: duplicate page title %r at %r and %r; "
                "parent→child edges will target %r",
                a.name,
                name_to_path[a.name],
                a.source_path,
                name_to_path[a.name],
            )
            continue
        name_to_path[a.name] = a.source_path

    path_set = set(path_to_artifacts)

    graph = nx.Graph()
    for path in path_set:
        graph.add_node(path)

    # 1. Parent-child edges: group sibling pages by parent_path
    parent_groups: dict[str, list[str]] = defaultdict(list)
    for a in artifacts:
        if a.parent_path:
            parent_groups[a.parent_path].append(a.source_path)

    # Track synthetic parent-hub nodes so we can drop them after clustering.
    # When the real parent page is not in the artifact set (common — Confluence
    # exports often start one level below the root), we still need a single
    # connection point so siblings cohesively land in the same community.
    # A synthetic hub adds O(N) edges, never O(N²), and is removed from each
    # community before the cluster is built so it doesn't show up in output.
    synthetic_parents: set[str] = set()

    for parent_path_val, children in parent_groups.items():
        # parent_path is a breadcrumb like "Engineering / Security / Auth".
        # The immediate parent is the LAST segment after the final separator.
        # If a page named like that segment exists in the artifact set,
        # connect each child to it directly. We accept "/" or ">" as
        # separators since different scanner configurations differ.
        parent_title = _last_breadcrumb_segment(parent_path_val)
        if parent_title and parent_title in name_to_path:
            parent_src = name_to_path[parent_title]
            edge_weight = _PARENT_CHILD_EDGE_WEIGHT
        else:
            # No real parent in the artifact set — synthesize a hub node so
            # siblings still cluster together without an O(N²) edge clique.
            # The synthetic edge weight is matched to the link weight (rather
            # than the parent weight) so a cross-parent inline link can still
            # merge two synthetic communities; a REAL parent always wins.
            parent_src = f"__synthetic_parent::{parent_path_val}"
            synthetic_parents.add(parent_src)
            graph.add_node(parent_src)
            edge_weight = _LINK_EDGE_WEIGHT

        for child in children:
            if graph.has_edge(parent_src, child):
                graph[parent_src][child]["weight"] += edge_weight
            else:
                graph.add_edge(parent_src, child, weight=edge_weight)

    # 2. Inline link edges
    for source_path in path_set:
        _add_link_edges(graph, source_path, repo_root, path_set)

    logger.debug(
        "[DOC_CLUSTER] confluence: %d nodes (%d synthetic), %d edges",
        graph.number_of_nodes(),
        len(synthetic_parents),
        graph.number_of_edges(),
    )

    communities = _louvain_or_components(graph)
    # Strip synthetic hub nodes — they only existed to seed clustering and
    # have no corresponding artifact, so they would otherwise emit an empty
    # cluster (or a ghost artifact) in the output.
    if synthetic_parents:
        communities = [c - synthetic_parents for c in communities if c - synthetic_parents]
    clusters = _communities_to_clusters(communities, path_to_artifacts, "confluence")

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

    path_to_artifacts: dict[str, list[ArtifactInfo]] = defaultdict(list)
    for a in artifacts:
        path_to_artifacts[a.source_path].append(a)
    name_to_path = {a.name: a.source_path for a in artifacts}

    graph = nx.Graph()
    for path in path_to_artifacts:
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
    clusters = _communities_to_clusters(communities, path_to_artifacts, "jira")

    logger.info(
        "[DOC_CLUSTER] jira → %d clusters from %d artifacts",
        len(clusters),
        len(artifacts),
    )
    return clusters
