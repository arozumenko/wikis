"""Tests for the ``min_confidence`` retriever filter (#120 Phase 2 / #157).

Drives ``UnifiedRetriever._get_expansion_neighbors`` directly with a
stub storage backend so we can assert the filter passes/rejects edges
correctly without spinning up a real ``.wiki.db``. Also pins the
helper-level threshold semantics (rank ordering, default-to-EXTRACTED
on missing/None labels).
"""

from __future__ import annotations

from typing import Any

import pytest

from app.core.unified_retriever import (
    _CONFIDENCE_RANK,
    UnifiedRetriever,
    _edge_passes_confidence,
)


# ---------------------------------------------------------------------------
# Helper: minimal storage stub
# ---------------------------------------------------------------------------


class _StubDB:
    """Just enough of the storage protocol for ``_get_expansion_neighbors``.

    The retriever calls ``get_edges_from`` / ``get_edges_to`` for graph
    expansion and ``get_node`` for the resulting nodes. Symbol types
    must be in the architectural-types set to survive the post-fetch
    filter — ``class`` is in that set across all configurations.
    """

    def __init__(
        self,
        edges_from: dict[str, list[dict]],
        edges_to: dict[str, list[dict]] | None = None,
        nodes: dict[str, dict] | None = None,
    ) -> None:
        self._edges_from = edges_from
        self._edges_to = edges_to or {}
        self._nodes = nodes or {}

    def get_edges_from(self, node_id: str) -> list[dict]:
        return list(self._edges_from.get(node_id, []))

    def get_edges_to(self, node_id: str) -> list[dict]:
        return list(self._edges_to.get(node_id, []))

    def get_node(self, node_id: str) -> dict | None:
        return self._nodes.get(node_id)


def _retriever(db: _StubDB) -> UnifiedRetriever:
    """Construct a UnifiedRetriever bypassing its real init —
    ``_get_expansion_neighbors`` only reads ``self.db``."""
    r = UnifiedRetriever.__new__(UnifiedRetriever)
    r.db = db
    return r


# ---------------------------------------------------------------------------
# _edge_passes_confidence helper
# ---------------------------------------------------------------------------


class TestEdgePassesConfidence:
    def test_none_threshold_keeps_everything(self) -> None:
        """``min_confidence=None`` is the legacy/default contract: no
        filtering, every edge passes."""
        for label in ("EXTRACTED", "INFERRED", "AMBIGUOUS"):
            assert _edge_passes_confidence({"confidence": label}, None) is True

    def test_extracted_threshold_rejects_lower_tiers(self) -> None:
        assert _edge_passes_confidence({"confidence": "EXTRACTED"}, "EXTRACTED") is True
        assert _edge_passes_confidence({"confidence": "INFERRED"}, "EXTRACTED") is False
        assert _edge_passes_confidence({"confidence": "AMBIGUOUS"}, "EXTRACTED") is False

    def test_inferred_threshold_keeps_extracted_and_inferred(self) -> None:
        """``min_confidence`` is the MINIMUM acceptable label —
        anything at-or-above the threshold's rank passes."""
        assert _edge_passes_confidence({"confidence": "EXTRACTED"}, "INFERRED") is True
        assert _edge_passes_confidence({"confidence": "INFERRED"}, "INFERRED") is True
        assert _edge_passes_confidence({"confidence": "AMBIGUOUS"}, "INFERRED") is False

    def test_ambiguous_threshold_keeps_all_three(self) -> None:
        for label in ("EXTRACTED", "INFERRED", "AMBIGUOUS"):
            assert _edge_passes_confidence({"confidence": label}, "AMBIGUOUS") is True

    def test_missing_label_defaults_to_extracted(self) -> None:
        """Legacy rows + storage defaults both land in EXTRACTED.
        Verified by the PR #150 storage-default contract."""
        assert _edge_passes_confidence({}, "EXTRACTED") is True
        assert _edge_passes_confidence({"confidence": None}, "EXTRACTED") is True

    def test_case_insensitive(self) -> None:
        """MCP clients may pass lowercase strings; the rank lookup
        normalises both sides."""
        assert _edge_passes_confidence({"confidence": "extracted"}, "inferred") is True
        assert _edge_passes_confidence({"confidence": "Inferred"}, "EXTRACTED") is False


def test_confidence_rank_ordering_is_strict() -> None:
    """Pin the rank values so a future renumbering of the enum
    doesn't silently flip the semantics ('EXTRACTED' must always
    outrank 'INFERRED')."""
    assert _CONFIDENCE_RANK["EXTRACTED"] > _CONFIDENCE_RANK["INFERRED"]
    assert _CONFIDENCE_RANK["INFERRED"] > _CONFIDENCE_RANK["AMBIGUOUS"]


# ---------------------------------------------------------------------------
# UnifiedRetriever._get_expansion_neighbors with min_confidence
# ---------------------------------------------------------------------------


_NODE_CLASS = {"node_id": "_placeholder", "symbol_type": "class"}


def _class_node(node_id: str) -> dict[str, Any]:
    return {**_NODE_CLASS, "node_id": node_id}


class TestExpansionNeighborsFilter:
    def test_no_filter_returns_all_qualifying_neighbours(self) -> None:
        """Baseline: with ``min_confidence=None`` the filter is
        transparent and every edge contributes a neighbour."""
        db = _StubDB(
            edges_from={
                "seed": [
                    {"target_id": "high", "confidence": "EXTRACTED"},
                    {"target_id": "low", "confidence": "INFERRED"},
                ],
            },
            nodes={"high": _class_node("high"), "low": _class_node("low")},
        )
        r = _retriever(db)
        results = r._get_expansion_neighbors("seed", seen_ids=set())
        names = {n["node_id"] for n in results}
        assert names == {"high", "low"}

    def test_extracted_threshold_drops_inferred_edges(self) -> None:
        """The headline filter behaviour — ``min_confidence="EXTRACTED"``
        keeps only the strongest tier."""
        db = _StubDB(
            edges_from={
                "seed": [
                    {"target_id": "high", "confidence": "EXTRACTED"},
                    {"target_id": "low", "confidence": "INFERRED"},
                ],
            },
            nodes={"high": _class_node("high"), "low": _class_node("low")},
        )
        r = _retriever(db)
        results = r._get_expansion_neighbors(
            "seed", seen_ids=set(), min_confidence="EXTRACTED",
        )
        names = {n["node_id"] for n in results}
        assert names == {"high"}
        # The kept node carries its edge-confidence in metadata.
        assert results[0]["_edge_confidence"] == "EXTRACTED"

    def test_multi_edge_node_keeps_strongest_confidence(self) -> None:
        """A node reached via BOTH an EXTRACTED and an INFERRED edge
        keeps EXTRACTED — any high-confidence path makes the node
        trustworthy. This is the rationale documented in the
        ``_record`` helper inside ``_get_expansion_neighbors``."""
        db = _StubDB(
            edges_from={
                "seed": [
                    {"target_id": "both", "confidence": "INFERRED"},  # first
                    {"target_id": "both", "confidence": "EXTRACTED"},  # second
                ],
            },
            nodes={"both": _class_node("both")},
        )
        r = _retriever(db)
        results = r._get_expansion_neighbors(
            "seed", seen_ids=set(), min_confidence="INFERRED",
        )
        assert len(results) == 1
        # Best wins regardless of edge insertion order.
        assert results[0]["_edge_confidence"] == "EXTRACTED"

    def test_incoming_edges_also_filtered(self) -> None:
        """Filter applies symmetrically to incoming + outgoing edges."""
        db = _StubDB(
            edges_from={},
            edges_to={
                "seed": [
                    {"source_id": "parent_inferred", "confidence": "INFERRED"},
                ],
            },
            nodes={"parent_inferred": _class_node("parent_inferred")},
        )
        r = _retriever(db)
        # No filter — included.
        baseline = r._get_expansion_neighbors("seed", seen_ids=set())
        assert {n["node_id"] for n in baseline} == {"parent_inferred"}
        # EXTRACTED threshold — dropped.
        filtered = r._get_expansion_neighbors(
            "seed", seen_ids=set(), min_confidence="EXTRACTED",
        )
        assert filtered == []

    def test_seen_ids_skipped_before_filter(self) -> None:
        """Already-seen IDs short-circuit before the confidence check,
        same as the pre-#157 behaviour."""
        db = _StubDB(
            edges_from={
                "seed": [{"target_id": "already_seen", "confidence": "EXTRACTED"}],
            },
            nodes={"already_seen": _class_node("already_seen")},
        )
        r = _retriever(db)
        results = r._get_expansion_neighbors(
            "seed", seen_ids={"already_seen"},
            min_confidence="EXTRACTED",
        )
        assert results == []

    def test_missing_confidence_label_treated_as_extracted(self) -> None:
        """Legacy rows (pre-#150) with no confidence column on the
        edge dict default to EXTRACTED in the filter — they survive
        the strictest threshold so existing wikis keep working."""
        db = _StubDB(
            edges_from={"seed": [{"target_id": "legacy"}]},  # no confidence key
            nodes={"legacy": _class_node("legacy")},
        )
        r = _retriever(db)
        results = r._get_expansion_neighbors(
            "seed", seen_ids=set(), min_confidence="EXTRACTED",
        )
        assert {n["node_id"] for n in results} == {"legacy"}
        assert results[0]["_edge_confidence"] == "EXTRACTED"


class TestSourceReferenceConfidence:
    """End-to-end check that the field added in #120 Phase 1 now
    actually populates from the retriever (it was always None in
    Phase 1 because the upstream wiring wasn't there)."""

    def test_confidence_flows_from_edge_to_document_metadata(self) -> None:
        """The ``_edge_confidence`` annotation on the node dict is
        propagated into ``Document.metadata["confidence"]`` so the
        downstream agent's source dict picks it up automatically."""
        from langchain_core.documents import Document

        db = _StubDB(
            edges_from={
                "seed": [
                    {"target_id": "inferred_neighbor", "confidence": "INFERRED"},
                ],
            },
            nodes={"inferred_neighbor": _class_node("inferred_neighbor")},
        )
        r = _retriever(db)
        seed_doc = Document(
            page_content="seed",
            metadata={"node_id": "seed"},
        )
        result_docs = r._expand_documents([seed_doc])
        # The expanded doc carries the edge confidence in metadata.
        expanded = [
            d for d in result_docs
            if d.metadata.get("node_id") == "inferred_neighbor"
        ]
        assert len(expanded) == 1
        assert expanded[0].metadata.get("confidence") == "INFERRED"


def test_ask_request_min_confidence_field_round_trips() -> None:
    """The Pydantic model accepts the new field and serialises it
    cleanly — what MCP / HTTP clients will rely on."""
    from app.models.api import AskRequest

    req = AskRequest(
        wiki_id="w1",
        question="how does auth work?",
        min_confidence="EXTRACTED",
    )
    assert req.min_confidence == "EXTRACTED"
    assert req.model_dump()["min_confidence"] == "EXTRACTED"

    # Backward compat — old callers don't pass it.
    req2 = AskRequest(wiki_id="w1", question="hi")
    assert req2.min_confidence is None


def test_ask_request_validator_normalises_case() -> None:
    """Lowercase input from MCP/HTTP clients is accepted and
    normalised to upper-case so downstream comparisons (the rank
    lookup) work against the canonical labels."""
    from app.models.api import AskRequest

    req = AskRequest(
        wiki_id="w1", question="hi", min_confidence="inferred",
    )
    assert req.min_confidence == "INFERRED"


def test_ask_request_validator_rejects_unknown_values() -> None:
    """Code-review I1: a typo'd ``min_confidence`` would otherwise
    silently behave like "no filter" because the rank lookup
    returns 0 for unknown keys and EVERY edge has rank >= 0. The
    Pydantic validator now rejects it with a clear 422-shaped
    error instead."""
    from app.models.api import AskRequest

    with pytest.raises(ValueError, match="min_confidence must be one of"):
        AskRequest(
            wiki_id="w1", question="hi", min_confidence="extrcted",
        )


# ---------------------------------------------------------------------------
# GraphQueryService.get_relationships — the live agent path
# ---------------------------------------------------------------------------


def _make_nx_graph_with_confidence_edges():
    """Build a NetworkX MultiDiGraph with edges carrying confidence
    labels — matches the shape ``_add_relationships_bulk`` writes
    (PR #150) and what ``GraphQueryService.get_relationships`` reads."""
    import networkx as nx

    g = nx.MultiDiGraph()
    g.add_node("root", symbol_name="Root", symbol_type="class")
    g.add_node("ext", symbol_name="ExternalSym", symbol_type="class")
    g.add_node("inf", symbol_name="InferredSym", symbol_type="class")
    g.add_node("amb", symbol_name="AmbiguousSym", symbol_type="class")
    # Three outgoing edges, each at a different confidence tier.
    g.add_edge("root", "ext", relationship_type="calls", confidence="EXTRACTED")
    g.add_edge("root", "inf", relationship_type="calls", confidence="INFERRED")
    g.add_edge("root", "amb", relationship_type="calls", confidence="AMBIGUOUS")
    return g


class TestGraphQueryServiceFilter:
    """Code-review C1 fix: ``min_confidence`` now actually filters
    edges during the live agent path's NetworkX graph traversal
    (via ``GraphQueryService.get_relationships``)."""

    @pytest.fixture
    def svc(self):
        from app.core.code_graph.graph_query_service import GraphQueryService

        graph = _make_nx_graph_with_confidence_edges()
        return GraphQueryService(graph)

    def test_no_filter_returns_all_three_tiers(self, svc) -> None:
        rels = svc.get_relationships("root", direction="outgoing", max_depth=1)
        targets = {r.target_name for r in rels}
        assert targets == {"ExternalSym", "InferredSym", "AmbiguousSym"}

    def test_extracted_threshold_drops_lower_tiers(self, svc) -> None:
        rels = svc.get_relationships(
            "root", direction="outgoing", max_depth=1,
            min_confidence="EXTRACTED",
        )
        targets = {r.target_name for r in rels}
        assert targets == {"ExternalSym"}

    def test_inferred_threshold_keeps_extracted_and_inferred(self, svc) -> None:
        rels = svc.get_relationships(
            "root", direction="outgoing", max_depth=1,
            min_confidence="INFERRED",
        )
        targets = {r.target_name for r in rels}
        assert targets == {"ExternalSym", "InferredSym"}

    def test_filter_advances_frontier_through_dropped_nodes(self) -> None:
        """When an edge is filtered out at depth 1, the traversal
        should still advance the frontier so deeper EXTRACTED edges
        can be discovered. Without this, a single INFERRED edge in
        the path would block visibility of everything beyond it."""
        import networkx as nx

        g = nx.MultiDiGraph()
        g.add_node("a", symbol_name="A", symbol_type="class")
        g.add_node("b", symbol_name="B", symbol_type="class")
        g.add_node("c", symbol_name="C", symbol_type="class")
        g.add_edge("a", "b", relationship_type="calls", confidence="INFERRED")
        g.add_edge("b", "c", relationship_type="calls", confidence="EXTRACTED")

        from app.core.code_graph.graph_query_service import GraphQueryService

        svc = GraphQueryService(g)
        rels = svc.get_relationships(
            "a", direction="outgoing", max_depth=2,
            min_confidence="EXTRACTED",
        )
        # a→b was INFERRED → dropped. b→c was EXTRACTED → kept.
        # The traversal must have advanced through b to find c.
        targets = {r.target_name for r in rels}
        assert "B" not in targets
        assert "C" in targets


def test_resolve_and_traverse_threads_min_confidence() -> None:
    """``resolve_and_traverse`` is a thin wrapper around
    ``get_relationships``; it must forward ``min_confidence`` so
    the agent tool ``get_relationships_tool`` reaches the filter.

    Mocks ``resolve_symbol`` so this test focuses on the forwarding
    contract — the underlying symbol-resolution logic has its own
    tests elsewhere.
    """
    from unittest.mock import MagicMock, patch

    from app.core.code_graph.graph_query_service import GraphQueryService

    g = _make_nx_graph_with_confidence_edges()
    svc = GraphQueryService(g)

    captured: dict = {}

    real_get_rels = svc.get_relationships

    def _spy(node_id, **kwargs):
        captured["min_confidence"] = kwargs.get("min_confidence")
        return real_get_rels(node_id, **kwargs)

    with (
        patch.object(svc, "resolve_symbol", return_value="root"),
        patch.object(svc, "get_relationships", side_effect=_spy),
    ):
        _, rels = svc.resolve_and_traverse(
            "Root", direction="outgoing", max_depth=1,
            min_confidence="EXTRACTED",
        )

    # The forwarding contract: whatever value the caller passed must
    # show up in get_relationships' kwargs.
    assert captured["min_confidence"] == "EXTRACTED"
    # And the filter actually took effect — only EXTRACTED edges.
    assert {r.target_name for r in rels} == {"ExternalSym"}
