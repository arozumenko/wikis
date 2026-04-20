"""Unit tests for MultiGraphQueryService.

Verifies:
- resolve_symbol fans out across wikis and returns first match
- search fans out and merges results sorted by score
- get_relationships delegates to the correct wiki's graph
- resolve_and_traverse returns node_id + relationships
- query fans out and de-duplicates
- stats aggregates counts and returns wiki_ids list
- _MergedGraphView / _MergedNodesView proxy correctly
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import networkx as nx
import pytest

from app.core.code_graph.graph_query_service import GraphQueryService, RelationshipResult, SymbolResult
from app.core.code_graph.multi_graph_query_service import MultiGraphQueryService


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_gqs(nodes: dict[str, dict] | None = None) -> GraphQueryService:
    """Build a minimal GraphQueryService backed by a small NetworkX graph."""
    g = nx.DiGraph()
    if nodes:
        for nid, data in nodes.items():
            g.add_node(nid, **data)
    return GraphQueryService(g, fts_index=None)


def _sym_result(node_id: str, name: str, score: float = 1.0) -> SymbolResult:
    return SymbolResult(
        node_id=node_id,
        symbol_name=name,
        symbol_type="function",
        score=score,
        connections=0,
    )


# ---------------------------------------------------------------------------
# resolve_symbol
# ---------------------------------------------------------------------------


class TestResolveSymbol:
    def test_returns_first_match(self):
        gqs_a = MagicMock(spec=GraphQueryService)
        gqs_b = MagicMock(spec=GraphQueryService)
        gqs_a.resolve_symbol.return_value = "node-a"
        gqs_b.resolve_symbol.return_value = "node-b"
        # attach graph attribute so _MergedGraphView works
        gqs_a.graph = nx.DiGraph()
        gqs_b.graph = nx.DiGraph()

        multi = MultiGraphQueryService({"wiki-a": gqs_a, "wiki-b": gqs_b})
        result = multi.resolve_symbol("SomeClass")

        assert result == "node-a"
        gqs_b.resolve_symbol.assert_not_called()

    def test_falls_back_to_second_wiki(self):
        gqs_a = MagicMock(spec=GraphQueryService)
        gqs_b = MagicMock(spec=GraphQueryService)
        gqs_a.resolve_symbol.return_value = None
        gqs_b.resolve_symbol.return_value = "node-b"
        gqs_a.graph = nx.DiGraph()
        gqs_b.graph = nx.DiGraph()

        multi = MultiGraphQueryService({"wiki-a": gqs_a, "wiki-b": gqs_b})
        result = multi.resolve_symbol("SomeClass")

        assert result == "node-b"

    def test_returns_none_when_not_found(self):
        gqs_a = MagicMock(spec=GraphQueryService)
        gqs_a.resolve_symbol.return_value = None
        gqs_a.graph = nx.DiGraph()

        multi = MultiGraphQueryService({"wiki-a": gqs_a})
        result = multi.resolve_symbol("Nonexistent")

        assert result is None


# ---------------------------------------------------------------------------
# search
# ---------------------------------------------------------------------------


class TestSearch:
    def test_merges_and_sorts_by_score(self):
        gqs_a = MagicMock(spec=GraphQueryService)
        gqs_b = MagicMock(spec=GraphQueryService)
        gqs_a.graph = nx.DiGraph()
        gqs_b.graph = nx.DiGraph()
        gqs_a.search.return_value = [_sym_result("a1", "Alpha", score=0.8)]
        gqs_b.search.return_value = [_sym_result("b1", "Beta", score=0.9)]

        multi = MultiGraphQueryService({"wiki-a": gqs_a, "wiki-b": gqs_b})
        results = multi.search("auth", k=5)

        assert len(results) == 2
        assert results[0].symbol_name == "Beta"  # higher score first
        assert results[1].symbol_name == "Alpha"

    def test_respects_k_limit(self):
        gqs_a = MagicMock(spec=GraphQueryService)
        gqs_a.graph = nx.DiGraph()
        gqs_a.search.return_value = [
            _sym_result(f"n{i}", f"Sym{i}", score=float(i)) for i in range(10)
        ]

        multi = MultiGraphQueryService({"wiki-a": gqs_a})
        results = multi.search("query", k=3)

        assert len(results) <= 3

    def test_skips_failed_wiki(self):
        gqs_ok = MagicMock(spec=GraphQueryService)
        gqs_ok.graph = nx.DiGraph()
        gqs_ok.search.return_value = [_sym_result("n1", "Good")]

        gqs_bad = MagicMock(spec=GraphQueryService)
        gqs_bad.graph = nx.DiGraph()
        gqs_bad.search.side_effect = RuntimeError("index broken")

        multi = MultiGraphQueryService({"wiki-ok": gqs_ok, "wiki-bad": gqs_bad})
        results = multi.search("query", k=5)

        assert len(results) == 1
        assert results[0].symbol_name == "Good"


# ---------------------------------------------------------------------------
# get_relationships
# ---------------------------------------------------------------------------


class TestGetRelationships:
    def test_delegates_to_correct_wiki(self):
        g_a = nx.DiGraph()
        g_a.add_node("node-a")
        g_b = nx.DiGraph()

        gqs_a = GraphQueryService(g_a, fts_index=None)
        gqs_b = MagicMock(spec=GraphQueryService)
        gqs_b.graph = g_b

        multi = MultiGraphQueryService({"wiki-a": gqs_a, "wiki-b": gqs_b})
        rels = multi.get_relationships("node-a")

        assert isinstance(rels, list)
        gqs_b.get_relationships.assert_not_called()

    def test_returns_empty_for_unknown_node(self):
        gqs_a = MagicMock(spec=GraphQueryService)
        gqs_a.graph = nx.DiGraph()  # empty graph — node not present

        multi = MultiGraphQueryService({"wiki-a": gqs_a})
        rels = multi.get_relationships("nonexistent-node")

        assert rels == []


# ---------------------------------------------------------------------------
# resolve_and_traverse
# ---------------------------------------------------------------------------


class TestResolveAndTraverse:
    def test_returns_node_and_rels(self):
        gqs_a = MagicMock(spec=GraphQueryService)
        gqs_a.graph = nx.DiGraph()
        gqs_a.resolve_symbol.return_value = "node-a"
        gqs_a.get_relationships.return_value = [
            RelationshipResult(
                source_name="A", target_name="B",
                relationship_type="calls",
            )
        ]

        multi = MultiGraphQueryService({"wiki-a": gqs_a})
        node_id, rels = multi.resolve_and_traverse("SomeFunc")

        assert node_id == "node-a"
        assert len(rels) == 1

    def test_returns_none_when_not_found(self):
        gqs_a = MagicMock(spec=GraphQueryService)
        gqs_a.graph = nx.DiGraph()
        gqs_a.resolve_symbol.return_value = None

        multi = MultiGraphQueryService({"wiki-a": gqs_a})
        node_id, rels = multi.resolve_and_traverse("Unknown")

        assert node_id is None
        assert rels == []


# ---------------------------------------------------------------------------
# query
# ---------------------------------------------------------------------------


class TestQuery:
    def test_fans_out_and_deduplicates(self):
        gqs_a = MagicMock(spec=GraphQueryService)
        gqs_b = MagicMock(spec=GraphQueryService)
        gqs_a.graph = nx.DiGraph()
        gqs_b.graph = nx.DiGraph()

        shared = _sym_result("shared-node", "SharedSym", score=0.9)
        unique_b = _sym_result("unique-b", "UniqueSym", score=0.5)

        gqs_a.query.return_value = [shared]
        gqs_b.query.return_value = [shared, unique_b]  # shared duplicated

        multi = MultiGraphQueryService({"wiki-a": gqs_a, "wiki-b": gqs_b})
        results = multi.query("type:class")

        node_ids = [r.node_id for r in results]
        # shared-node should appear exactly once
        assert node_ids.count("shared-node") == 1
        assert "unique-b" in node_ids

    def test_skips_failed_wiki(self):
        gqs_ok = MagicMock(spec=GraphQueryService)
        gqs_ok.graph = nx.DiGraph()
        gqs_ok.query.return_value = [_sym_result("n1", "OK")]

        gqs_bad = MagicMock(spec=GraphQueryService)
        gqs_bad.graph = nx.DiGraph()
        gqs_bad.query.side_effect = ValueError("bad JQL")

        multi = MultiGraphQueryService({"wiki-ok": gqs_ok, "wiki-bad": gqs_bad})
        results = multi.query("type:class")

        assert len(results) == 1


# ---------------------------------------------------------------------------
# stats
# ---------------------------------------------------------------------------


class TestStats:
    def test_aggregates_counts(self):
        gqs_a = MagicMock(spec=GraphQueryService)
        gqs_b = MagicMock(spec=GraphQueryService)
        gqs_a.graph = nx.DiGraph()
        gqs_b.graph = nx.DiGraph()
        gqs_a.stats.return_value = {
            "graph_nodes": 100, "graph_edges": 50,
            "node_index_size": 10, "name_index_size": 20,
            "full_name_index_size": 15, "suffix_index_size": 5,
            "fts5_available": True,
        }
        gqs_b.stats.return_value = {
            "graph_nodes": 200, "graph_edges": 80,
            "node_index_size": 30, "name_index_size": 40,
            "full_name_index_size": 25, "suffix_index_size": 8,
            "fts5_available": False,
        }

        multi = MultiGraphQueryService({"wiki-a": gqs_a, "wiki-b": gqs_b})
        s = multi.stats()

        assert s["graph_nodes"] == 300
        assert s["graph_edges"] == 130
        assert s["fts5_available"] is True  # True if any wiki has FTS5
        assert set(s["wiki_ids"]) == {"wiki-a", "wiki-b"}
        assert s["wiki_count"] == 2

    def test_empty_services(self):
        multi = MultiGraphQueryService({})
        s = multi.stats()

        assert s["graph_nodes"] == 0
        assert s["wiki_ids"] == []
        assert s["wiki_count"] == 0


# ---------------------------------------------------------------------------
# MergedGraphView
# ---------------------------------------------------------------------------


class TestMergedGraphView:
    def test_contains_node_from_any_graph(self):
        g_a = nx.DiGraph()
        g_a.add_node("node-a")
        g_b = nx.DiGraph()
        g_b.add_node("node-b")

        gqs_a = GraphQueryService(g_a)
        gqs_b = GraphQueryService(g_b)

        multi = MultiGraphQueryService({"wiki-a": gqs_a, "wiki-b": gqs_b})

        assert "node-a" in multi.graph
        assert "node-b" in multi.graph
        assert "missing" not in multi.graph

    def test_nodes_get_returns_correct_data(self):
        g_a = nx.DiGraph()
        g_a.add_node("node-a", symbol_name="FuncA", symbol_type="function")
        g_b = nx.DiGraph()
        g_b.add_node("node-b", symbol_name="ClassB", symbol_type="class")

        gqs_a = GraphQueryService(g_a)
        gqs_b = GraphQueryService(g_b)

        multi = MultiGraphQueryService({"wiki-a": gqs_a, "wiki-b": gqs_b})

        data_a = multi.graph.nodes.get("node-a")
        assert data_a is not None
        assert data_a["symbol_name"] == "FuncA"

        data_b = multi.graph.nodes.get("node-b")
        assert data_b is not None
        assert data_b["symbol_name"] == "ClassB"

        assert multi.graph.nodes.get("nonexistent") is None
