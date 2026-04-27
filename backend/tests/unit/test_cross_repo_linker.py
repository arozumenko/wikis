"""Phase 8 — Cross-repo linker."""

from __future__ import annotations

import networkx as nx
import pytest

from app.core.code_graph.cross_repo_linker import build_cross_repo_edges


def _merged_graph():
    """Two-wiki merged graph used across tests.

    Each node carries a ``wiki_id`` attribute so the cascade can
    determine whether an edge truly crosses a wiki boundary.
    """
    g = nx.MultiDiGraph()
    g.add_node(
        "py.handler",
        wiki_id="wiki-a",
        language="python",
        rel_path="src/api/handler.py",
        file_name="handler",
        symbol_name="create_user",
        symbol_type="function",
    )
    g.add_node(
        "ts.client",
        wiki_id="wiki-b",
        language="typescript",
        rel_path="web/src/api/users.ts",
        file_name="users",
        symbol_name="createUser",
        symbol_type="function",
    )
    g.add_node(
        "ts.unrelated",
        wiki_id="wiki-b",
        language="typescript",
        rel_path="web/src/util.ts",
        file_name="util",
        symbol_name="noop",
        symbol_type="function",
    )
    return g


class TestBasic:
    def test_empty_inputs_return_empty(self):
        assert build_cross_repo_edges(
            wiki_ids=[],
            merged_graph=_merged_graph(),
            relatedness={},
        ) == []
        assert build_cross_repo_edges(
            wiki_ids=["wiki-a"],
            merged_graph=None,
            relatedness={},
        ) == []

    def test_below_threshold_pair_skipped(self):
        rels_by_pair = {
            ("wiki-a", "wiki-b"): [
                {"source": "py.handler", "target": "ts.client", "confidence": 0.9}
            ]
        }
        # Relatedness 0.10 < default threshold 0.15.
        out = build_cross_repo_edges(
            wiki_ids=["wiki-a", "wiki-b"],
            merged_graph=_merged_graph(),
            relatedness={("wiki-a", "wiki-b"): 0.10},
            cross_language_relationships_by_pair=rels_by_pair,
        )
        assert out == []

    def test_above_threshold_emits_edge(self):
        rels_by_pair = {
            ("wiki-a", "wiki-b"): [
                {"source": "py.handler", "target": "ts.client", "confidence": 0.8}
            ]
        }
        out = build_cross_repo_edges(
            wiki_ids=["wiki-a", "wiki-b"],
            merged_graph=_merged_graph(),
            relatedness={("wiki-a", "wiki-b"): 0.5},
            cross_language_relationships_by_pair=rels_by_pair,
        )
        assert len(out) == 1
        row = out[0]
        assert row["source_node_id"] == "py.handler"
        assert row["target_node_id"] == "ts.client"
        assert row["edge_class"] == "cross_repo"
        # base_weight=clamp(0.8)=0.8; weight = 0.8 * 0.7 * 0.5 = 0.28
        assert row["weight"] == pytest.approx(0.28, abs=1e-6)
        prov = row["provenance"]
        assert prov["wiki_a"] == "wiki-a" and prov["wiki_b"] == "wiki-b"
        assert prov["relatedness"] == 0.5
        assert prov["dampening"] == 0.7
        assert prov["source_relationship_type"] == "cross_language_L0"


class TestPairKeyOrder:
    def test_relatedness_lookup_is_alphabetical(self):
        # Wiki ids passed b, a — relatedness still keyed by sorted pair.
        rels_by_pair = {
            ("wiki-a", "wiki-b"): [
                {"source": "py.handler", "target": "ts.client", "confidence": 0.6}
            ]
        }
        out = build_cross_repo_edges(
            wiki_ids=["wiki-b", "wiki-a"],
            merged_graph=_merged_graph(),
            relatedness={("wiki-a", "wiki-b"): 0.5},
            cross_language_relationships_by_pair=rels_by_pair,
        )
        assert len(out) == 1


class TestFiltering:
    def test_skips_in_repo_edges(self):
        # Same-language same-wiki relationship should never appear in cross_repo output.
        g = _merged_graph()
        g.add_node(
            "py.other",
            wiki_id="wiki-a",
            language="python",
            rel_path="src/util.py",
            file_name="util",
            symbol_name="helper",
            symbol_type="function",
        )
        rels_by_pair = {
            ("wiki-a", "wiki-b"): [
                {"source": "py.handler", "target": "py.other", "confidence": 0.9}
            ]
        }
        out = build_cross_repo_edges(
            wiki_ids=["wiki-a", "wiki-b"],
            merged_graph=g,
            relatedness={("wiki-a", "wiki-b"): 0.5},
            cross_language_relationships_by_pair=rels_by_pair,
        )
        # Same-wiki relationship is filtered out by L0's same-language guard.
        assert out == []


class TestMaxWikis:
    def test_respects_cap(self):
        # 3 wikis, cap=2 → only the first two participate (no pairs with wiki-c).
        g = _merged_graph()
        g.add_node(
            "go.svc",
            wiki_id="wiki-c",
            language="go",
            rel_path="cmd/api/main.go",
            file_name="main",
            symbol_name="HandleUser",
            symbol_type="function",
        )
        rels_by_pair = {
            ("wiki-a", "wiki-c"): [
                {"source": "py.handler", "target": "go.svc", "confidence": 0.9}
            ],
        }
        out = build_cross_repo_edges(
            wiki_ids=["wiki-a", "wiki-b", "wiki-c"],
            merged_graph=g,
            relatedness={
                ("wiki-a", "wiki-b"): 0.5,
                ("wiki-a", "wiki-c"): 0.5,
                ("wiki-b", "wiki-c"): 0.5,
            },
            cross_language_relationships_by_pair=rels_by_pair,
            max_wikis=2,
        )
        # Only wiki-a/wiki-b participate; wiki-c relationship is dropped.
        assert all(
            row["target_node_id"] != "go.svc"
            and row["source_node_id"] != "go.svc"
            for row in out
        )


class TestDedup:
    def test_repeated_relationship_dedupes_per_pair(self):
        rels_by_pair = {
            ("wiki-a", "wiki-b"): [
                # Same edge twice.
                {"source": "py.handler", "target": "ts.client", "confidence": 0.8},
                {"source": "py.handler", "target": "ts.client", "confidence": 0.5},
            ]
        }
        out = build_cross_repo_edges(
            wiki_ids=["wiki-a", "wiki-b"],
            merged_graph=_merged_graph(),
            relatedness={("wiki-a", "wiki-b"): 0.5},
            cross_language_relationships_by_pair=rels_by_pair,
        )
        # Single (src, tgt) row regardless of duplicate L0 inputs.
        assert len(out) == 1
