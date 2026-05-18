"""Shared confidence-filter helpers (#120/#157).

Two callsites need the same ``min_confidence`` rank-comparison logic:

* :mod:`app.core.unified_retriever` — filters edges during SQL-backed
  graph expansion in ``UnifiedRetriever._get_expansion_neighbors``.
* :mod:`app.core.code_graph.graph_query_service` — filters edges during
  in-memory NetworkX traversal in ``GraphQueryService.get_relationships``
  (the live agent path used by the codebase tools).

Both readers share the same edge-data shape (the ``confidence`` key
written by ``graph_builder._add_relationships_bulk`` is identical
across the SQL and NetworkX paths). Lifting the helpers here keeps
the rank ordering + default-to-EXTRACTED semantics in a single
canonical location.
"""

from __future__ import annotations

# Rank ordering — higher rank == stronger signal.
# ``EXTRACTED`` is the parser's explicit observation, ``INFERRED`` is
# name-only resolution, ``AMBIGUOUS`` is reserved for multi-target
# cases (no callsites today). ``min_confidence`` is the **minimum
# acceptable** label — an edge passes the filter when its label's
# rank is >= the threshold's rank.
CONFIDENCE_RANK: dict[str, int] = {
    "EXTRACTED": 3,
    "INFERRED": 2,
    "AMBIGUOUS": 1,
}


def edge_passes_confidence(edge: dict, min_confidence: str | None) -> bool:
    """``True`` when ``edge`` meets the ``min_confidence`` floor.

    ``min_confidence is None`` disables filtering (legacy behavior).
    Missing / unknown labels on the edge are treated as ``EXTRACTED``
    because that's the storage layer's default and PR #150 guarantees
    this for newly-written edges. Legacy rows with NULL confidence
    (pre-#150 dbs) also coalesce here, so they continue to surface
    at the EXTRACTED tier.

    Case-insensitive on both sides — MCP clients may pass lowercase.
    """
    if min_confidence is None:
        return True
    edge_label = (edge.get("confidence") or "EXTRACTED").upper()
    threshold = min_confidence.upper()
    return CONFIDENCE_RANK.get(edge_label, 0) >= CONFIDENCE_RANK.get(threshold, 0)


# Accepted ``min_confidence`` values — used by Pydantic / signature
# validators that want to reject typos before they silently behave
# like "no filter" (the unknown-threshold case returns 0 from
# ``CONFIDENCE_RANK.get`` and admits every edge).
ACCEPTED_CONFIDENCE_LEVELS: frozenset[str] = frozenset(CONFIDENCE_RANK.keys())
