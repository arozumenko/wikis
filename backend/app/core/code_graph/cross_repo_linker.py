"""Phase 8 — Cross-repo linker.

Pure-function orchestrator that, given a precomputed relatedness
matrix and per-wiki adapters, produces the rows that should land in
``project_edges`` with ``edge_class='cross_repo'``.

For each wiki pair ``(a, b)`` whose relatedness clears the threshold:

1. Reuse Phase 6 :func:`run_cross_language_linker` cascade L0/L1/L2
   across the union of the two wikis' nodes.
2. Apply ``base_weight × cross_repo_dampening × relatedness_score``
   so a 0.7-confidence link between two only-mildly-related wikis
   cannot outrank an in-repo edge.

Side-effect free: returns a list of dicts shaped for
:meth:`ProjectStorageProtocol.upsert_project_edge`. The caller persists
them. Wiring is gated by ``flags.cross_repo_linking``.

Because Phase 7 storage backends + Phase 8 pipeline integration are
deferred, this module deliberately does not touch any storage.
"""

from __future__ import annotations

import logging
from typing import Iterable

from .cross_language_linker import (
    link_l0_exact,
    link_l1_api_surface,
    link_l2_hybrid,
)

logger = logging.getLogger(__name__)

__all__ = [
    "build_cross_repo_edges",
]


def _clamp_unit(x: float) -> float:
    if x < 0.0:
        return 0.0
    if x > 1.0:
        return 1.0
    return x


def _wiki_pair_iter(
    wiki_ids: list[str],
    relatedness: dict[tuple[str, str], float],
    *,
    threshold: float,
    max_wikis: int,
) -> Iterable[tuple[str, str, float]]:
    """Yield ``(wiki_a, wiki_b, score)`` for pairs above ``threshold``."""
    capped = wiki_ids[:max_wikis] if max_wikis > 0 else wiki_ids
    for i, a in enumerate(capped):
        for b in capped[i + 1 :]:
            key = (a, b) if a <= b else (b, a)
            score = relatedness.get(key, 0.0)
            if score >= threshold:
                yield a, b, score


def build_cross_repo_edges(
    *,
    wiki_ids: list[str],
    merged_graph,
    relatedness: dict[tuple[str, str], float],
    cross_language_relationships_by_pair: dict[tuple[str, str], list[dict]] | None = None,
    surfaces_by_node: dict[str, list[dict]] | None = None,
    fts_hits_per_node: dict[str, list[dict]] | None = None,
    vec_hits_per_node: dict[str, list[dict]] | None = None,
    threshold: float = 0.15,
    dampening: float = 0.7,
    max_wikis: int = 50,
) -> list[dict]:
    """Build cross-repo edge rows ready for ``project_edges`` upsert.

    Args:
        wiki_ids: Ordered list of wiki ids participating in the
            project. Order determines which pair appears first in
            output but does not affect the symmetric relatedness key.
        merged_graph: A ``MultiDiGraph`` whose nodes are tagged with
            a ``wiki_id`` attribute so the Phase 6 cascade helpers
            can determine cross-language pairs across the union.
        relatedness: Pair → score map keyed by alphabetically sorted
            ``(wiki_a, wiki_b)`` tuples (matches
            :class:`InMemoryProjectStorage`'s ``_pair_key``).
        cross_language_relationships_by_pair: Parser-supplied L0
            cross-repo relationships, keyed by sorted pair.
        surfaces_by_node, fts_hits_per_node, vec_hits_per_node:
            Inputs forwarded to L1/L2 for the merged graph. Defaults
            to empty dicts.
        threshold: Minimum relatedness to consider a pair.
        dampening: Multiplier applied on top of the base cascade
            weight.
        max_wikis: Hard cap on the number of wikis fed into the
            pairwise loop.

    Returns:
        List of edge dicts with keys ``source_node_id``,
        ``target_node_id``, ``edge_class``, ``weight``, ``provenance``
        ready to hand to :meth:`upsert_project_edge`.
    """
    if not wiki_ids or merged_graph is None:
        return []

    surfaces_by_node = surfaces_by_node or {}
    fts_hits_per_node = fts_hits_per_node or {}
    vec_hits_per_node = vec_hits_per_node or {}
    cross_language_relationships_by_pair = (
        cross_language_relationships_by_pair or {}
    )

    rows: list[dict] = []
    seen: set[tuple[str, str]] = set()

    for wiki_a, wiki_b, score in _wiki_pair_iter(
        wiki_ids,
        relatedness,
        threshold=threshold,
        max_wikis=max_wikis,
    ):
        pair_key = (wiki_a, wiki_b) if wiki_a <= wiki_b else (wiki_b, wiki_a)
        l0_rels = cross_language_relationships_by_pair.get(pair_key, [])

        cascade_edges: list[tuple[str, str, dict]] = []
        try:
            cascade_edges.extend(link_l0_exact(merged_graph, l0_rels))
            cascade_edges.extend(
                link_l1_api_surface(merged_graph, surfaces_by_node)
            )
            cascade_edges.extend(
                link_l2_hybrid(
                    merged_graph,
                    fts_hits_per_node,
                    vec_hits_per_node,
                )
            )
        except Exception:  # pragma: no cover — defensive
            logger.warning(
                "cross_repo_linker cascade failed for pair %s/%s",
                wiki_a,
                wiki_b,
                exc_info=True,
            )
            continue

        for src, tgt, attrs in cascade_edges:
            src_wiki = merged_graph.nodes.get(src, {}).get("wiki_id")
            tgt_wiki = merged_graph.nodes.get(tgt, {}).get("wiki_id")
            # Only keep edges that actually cross this pair.
            if not src_wiki or not tgt_wiki or src_wiki == tgt_wiki:
                continue
            if {src_wiki, tgt_wiki} != {wiki_a, wiki_b}:
                continue
            dedup_key = (src, tgt)
            if dedup_key in seen:
                continue
            seen.add(dedup_key)

            base_weight = float(attrs.get("weight", 0.0) or 0.0)
            weight = _clamp_unit(base_weight * dampening * score)
            provenance = dict(attrs.get("provenance", {}) or {})
            provenance.update(
                {
                    "wiki_a": wiki_a,
                    "wiki_b": wiki_b,
                    "relatedness": round(score, 6),
                    "base_weight": round(base_weight, 6),
                    "dampening": dampening,
                    "source_relationship_type": attrs.get("relationship_type"),
                }
            )
            rows.append(
                {
                    "source_node_id": src,
                    "target_node_id": tgt,
                    "edge_class": "cross_repo",
                    "weight": weight,
                    "provenance": provenance,
                }
            )
    return rows
