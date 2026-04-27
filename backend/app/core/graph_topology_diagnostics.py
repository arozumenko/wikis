"""
Phase 1 / graph-quality roadmap — Phase 2 observability.

Provides :class:`Phase2Stats` (a dataclass aggregating per-step
component counts + edge breakdowns) and :func:`explain_connectivity`,
which inspects a ``MultiDiGraph`` to populate edge-class / provenance
counts.

These structures are wired into :func:`graph_topology.run_phase2` so the
final stats blob persisted in ``repo_meta`` ('phase2_stats_v2') captures:

    * Component counts before / after each step
    * Edge counts grouped by ``edge_class``
    * Edge counts grouped by ``provenance.source``
    * Per-step result dicts (orphan resolution, doc edges, bridging,
      weighting, hubs)
"""

from __future__ import annotations

import logging
from collections import Counter
from dataclasses import dataclass, field, asdict
from typing import Any

import networkx as nx

logger = logging.getLogger(__name__)


@dataclass
class Phase2Stats:
    """Aggregated diagnostics for a Phase 2 run.

    All component counts are populated by :func:`run_phase2` between
    steps. Edge breakdowns are populated by
    :func:`explain_connectivity`.
    """

    components_before_orphan: int = 0
    components_after_orphan: int = 0
    components_after_doc: int = 0
    components_after_bridge: int = 0
    edges_by_class: dict[str, int] = field(default_factory=dict)
    edges_by_provenance: dict[str, int] = field(default_factory=dict)
    orphan_resolution: dict[str, Any] = field(default_factory=dict)
    doc_edges: dict[str, Any] = field(default_factory=dict)
    bridging: dict[str, Any] = field(default_factory=dict)
    weighting: dict[str, Any] = field(default_factory=dict)
    hubs: dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        """JSON-friendly serialization for ``db.set_meta``."""
        return asdict(self)


def explain_connectivity(db, G: nx.MultiDiGraph) -> Phase2Stats:
    """Return a :class:`Phase2Stats` populated from the graph alone.

    Component counts are left at 0 — callers (typically
    :func:`run_phase2`) overwrite the four ``components_*`` fields with
    snapshots taken between steps. ``db`` is currently unused but kept in
    the signature so future work (e.g. cluster-level breakdowns from the
    storage) does not break the API.
    """
    stats = Phase2Stats()

    class_counter: Counter[str] = Counter()
    prov_counter: Counter[str] = Counter()

    for _u, _v, data in G.edges(data=True):
        edge_class = str(data.get("edge_class") or "unknown")
        class_counter[edge_class] += 1

        provenance = data.get("provenance")
        if isinstance(provenance, dict):
            source = str(provenance.get("source") or "unknown")
        elif isinstance(provenance, str) and provenance:
            source = provenance
        else:
            # Fall back to created_by so old edges still group cleanly.
            source = str(data.get("created_by") or "unknown")
        prov_counter[source] += 1

    stats.edges_by_class = dict(class_counter)
    stats.edges_by_provenance = dict(prov_counter)
    return stats
