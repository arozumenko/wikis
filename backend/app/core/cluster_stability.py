"""Cluster ID stability across regenerations (#116 PR 4).

Leiden produces a deterministic partition given a fixed seed and an
unchanged graph, but the IDs it assigns to each cluster are arbitrary
(insertion order of the partition object). On the next run, the same
logical cluster can come back labeled 7 instead of 3 — which would shift
every ``compute_page_id`` hash and break every wikilink that points at
that page.

This module provides the Jaccard-based remapping that pins new cluster
IDs to their best-matching old cluster ID. Pure functions, no I/O, so
the algorithm can be unit-tested without touching storage. The caller
(``graph_clustering.run_phase3``) is responsible for reading the old
assignments out of ``repo_nodes`` before re-clustering and applying the
remap to the new assignments before writing them back.

Algorithm (greedy, threshold-gated):

    For each new cluster N, compute Jaccard(N.node_ids, O.node_ids)
    against every old cluster O. If the highest similarity is ≥ the
    threshold (default 0.5), N inherits O's ID. Otherwise N gets a
    fresh ID counting up from ``max(old_ids) + 1``.

The 0.5 threshold is the conventional choice — it requires that more
than half of the union is preserved. Lower than 0.5 and unrelated
clusters get matched; higher and minor membership churn breaks IDs.
PR 4 ships with 0.5; the constant is exposed for future tuning.
"""

from __future__ import annotations

import logging
from collections.abc import Iterable, Mapping

logger = logging.getLogger(__name__)


# Conventional threshold: Jaccard ≥ 0.5 means "more than half of the union
# is shared", which is the boundary where two cluster snapshots are more
# likely the same logical cluster than unrelated.
DEFAULT_JACCARD_THRESHOLD = 0.5


def jaccard(a: Iterable[str], b: Iterable[str]) -> float:
    """Jaccard similarity = |A ∩ B| / |A ∪ B|. Returns 0.0 when both are empty.

    Wrapped in a helper so call sites stay declarative and the empty-case
    behavior is documented in one place (two empty sets are arbitrarily
    treated as 0.0 rather than NaN/1.0 — the algorithm never asks).
    """
    set_a = set(a)
    set_b = set(b)
    if not set_a and not set_b:
        return 0.0
    union = set_a | set_b
    if not union:
        return 0.0
    return len(set_a & set_b) / len(union)


def compute_jaccard_remap(
    old_clusters: Mapping[int, Iterable[str]],
    new_clusters: Mapping[int, Iterable[str]],
    *,
    threshold: float = DEFAULT_JACCARD_THRESHOLD,
) -> dict[int, int]:
    """Map fresh cluster IDs onto stable IDs from the previous run.

    Args:
        old_clusters: ``{old_cluster_id: [node_ids]}`` from the previous
            partition. Keys are the cluster IDs we want to preserve when
            possible.
        new_clusters: ``{new_cluster_id: [node_ids]}`` from the fresh
            Leiden run. Keys are arbitrary integers Leiden picked.
        threshold: minimum Jaccard similarity to consider two clusters
            "the same". Anything below this threshold leaves the new
            cluster un-remapped (it gets a fresh ID, not the old one).

    Returns:
        ``{new_cluster_id: stable_id}`` for every key in ``new_clusters``.
        Stable IDs are either inherited from a matching old cluster (when
        Jaccard ≥ threshold) or freshly allocated starting from
        ``max(old_cluster_ids) + 1``.

    Algorithm (greedy):
        1. Sort new clusters by descending size — large clusters get
           first pick of old IDs, which matters when membership has
           shifted enough that two new clusters tie for one old ID.
        2. For each new cluster, compute Jaccard against every still-
           unclaimed old cluster. If the best match clears the threshold,
           claim that old ID.
        3. Remaining new clusters get fresh IDs from
           ``max(old_ids) + 1`` upward.

    Determinism: the function is deterministic given identical inputs.
    Tie-breaking on equal Jaccard prefers the lower old cluster ID so
    re-runs produce identical remaps.
    """
    if not new_clusters:
        return {}

    # Materialize as sets for fast intersection / union.
    old_sets: dict[int, set[str]] = {
        old_id: set(nodes) for old_id, nodes in old_clusters.items()
    }
    new_sets: dict[int, set[str]] = {
        new_id: set(nodes) for new_id, nodes in new_clusters.items()
    }

    # Largest-first ordering by new-cluster size; tie-break on new_id for
    # determinism.
    ordered_new_ids = sorted(
        new_sets.keys(), key=lambda nid: (-len(new_sets[nid]), nid),
    )

    remap: dict[int, int] = {}
    claimed_old: set[int] = set()
    matched_count = 0

    for new_id in ordered_new_ids:
        new_set = new_sets[new_id]
        best_old_id: int | None = None
        best_similarity = 0.0
        for old_id, old_set in old_sets.items():
            if old_id in claimed_old:
                continue
            similarity = jaccard(new_set, old_set)
            # Strictly-greater than best comparison; ties broken by lower
            # old_id (the iteration order over a dict is insertion order,
            # but we sort the candidates explicitly to be safe).
            if similarity > best_similarity or (
                similarity == best_similarity
                and best_old_id is not None
                and old_id < best_old_id
            ):
                best_similarity = similarity
                best_old_id = old_id
        if best_old_id is not None and best_similarity >= threshold:
            remap[new_id] = best_old_id
            claimed_old.add(best_old_id)
            matched_count += 1

    # Unmatched new clusters get fresh IDs starting at max(old_ids) + 1
    # (or 0 if no old clusters existed).
    next_id = (max(old_sets.keys()) + 1) if old_sets else 0
    # Skip over any old IDs we *didn't* claim — they're "dead" but
    # incrementing past them keeps fresh IDs above the historical
    # high-water mark, so wiki_pages rows from earlier-than-previous
    # regens don't accidentally re-collide.
    while next_id in old_sets and next_id not in claimed_old:
        next_id += 1

    for new_id in ordered_new_ids:
        if new_id in remap:
            continue
        while next_id in old_sets:
            next_id += 1
        remap[new_id] = next_id
        next_id += 1

    logger.info(
        "[cluster_stability] remapped %d/%d new clusters to old IDs "
        "(threshold=%.2f, fresh IDs allocated for %d)",
        matched_count,
        len(new_sets),
        threshold,
        len(new_sets) - matched_count,
    )
    return remap


def apply_remap_to_assignments(
    assignments: Mapping[str, int],
    remap: Mapping[int, int],
) -> dict[str, int]:
    """Rewrite ``{node_id: cluster_id}`` using a ``{old_id: new_id}`` map.

    Nodes whose cluster_id is not in the remap pass through unchanged,
    which means a remap that only covers a subset of clusters is safe
    (and useful for partial mappings during testing).
    """
    return {
        node_id: remap.get(cluster_id, cluster_id)
        for node_id, cluster_id in assignments.items()
    }


def stabilize_hierarchical_assignments(
    macro_assignments: Mapping[str, int],
    micro_assignments: Mapping[int, Mapping[str, int]],
    *,
    old_clusters: Mapping[int, Mapping[int, Iterable[str]]],
    threshold: float = DEFAULT_JACCARD_THRESHOLD,
) -> tuple[dict[str, int], dict[int, dict[str, int]]]:
    """Remap both macro and micro IDs to maximise stability vs the old run.

    The two levels are remapped sequentially: macros first (Jaccard over
    the union of each macro's nodes across all its micros), then per
    *remapped* macro the micros are Jaccard-matched against the old
    macro's micros.

    Args:
        macro_assignments: ``{node_id: macro_id}`` from the fresh run.
        micro_assignments: ``{macro_id: {node_id: micro_id}}`` from the
            fresh run.
        old_clusters: previous-run state, shape
            ``{old_macro_id: {old_micro_id: [node_ids]}}`` — exactly what
            ``WikiStorageProtocol.get_all_clusters()`` returns.
        threshold: Jaccard similarity gate for both passes.

    Returns:
        ``(stable_macro_assignments, stable_micro_assignments)``.

    If ``old_clusters`` is empty (no prior run) this returns the input
    assignments unchanged.
    """
    if not old_clusters:
        return dict(macro_assignments), {
            mid: dict(m) for mid, m in micro_assignments.items()
        }

    # Build new_macro_clusters from the fresh macro_assignments.
    new_macro_clusters: dict[int, list[str]] = {}
    for node_id, macro_id in macro_assignments.items():
        new_macro_clusters.setdefault(macro_id, []).append(node_id)

    # Flatten old hierarchy → {old_macro_id: [all node_ids in that macro]}.
    old_macro_clusters: dict[int, list[str]] = {
        macro_id: [nid for micro_nodes in micros.values() for nid in micro_nodes]
        for macro_id, micros in old_clusters.items()
    }

    macro_remap = compute_jaccard_remap(
        old_macro_clusters, new_macro_clusters, threshold=threshold,
    )
    stable_macro_assignments = apply_remap_to_assignments(
        macro_assignments, macro_remap,
    )

    # Per-macro micro remap: only meaningful when the macro itself was
    # matched to an old macro. Otherwise the macro is brand new and its
    # micros get fresh IDs starting from 0 (compute_jaccard_remap with
    # empty old will allocate 0..N).
    stable_micro_assignments: dict[int, dict[str, int]] = {}
    for new_macro_id, node_to_micro in micro_assignments.items():
        stable_macro_id = macro_remap[new_macro_id]

        # Group this macro's new nodes by micro for Jaccard input.
        new_micro_clusters: dict[int, list[str]] = {}
        for node_id, micro_id in node_to_micro.items():
            new_micro_clusters.setdefault(micro_id, []).append(node_id)

        # The "old" micros are those of the matched old macro, if any.
        if stable_macro_id in old_clusters:
            old_micro_clusters: dict[int, list[str]] = {
                micro_id: list(nodes)
                for micro_id, nodes in old_clusters[stable_macro_id].items()
            }
        else:
            old_micro_clusters = {}

        micro_remap = compute_jaccard_remap(
            old_micro_clusters, new_micro_clusters, threshold=threshold,
        )
        # Key the remapped micros by the *stable* macro ID, so downstream
        # consumers see one consistent {macro: {node: micro}} structure.
        stable_micro_assignments[stable_macro_id] = apply_remap_to_assignments(
            node_to_micro, micro_remap,
        )

    return stable_macro_assignments, stable_micro_assignments
