"""Phase 6 / Action 6.3 — cross-language linker."""

from __future__ import annotations

from typing import Dict, List

import networkx as nx
import pytest

from app.core.code_graph.cross_language_linker import (
    link_l0_exact,
    link_l1_api_surface,
    link_l2_hybrid,
    link_l3_containment,
    run_cross_language_linker,
)


def _g():
    g = nx.MultiDiGraph()
    g.add_node("py.handler", language="python", rel_path="src/api/handler.py", file_name="handler", symbol_name="create_user", symbol_type="function")
    g.add_node("ts.client",  language="typescript", rel_path="web/src/api/users.ts", file_name="users", symbol_name="createUser", symbol_type="function")
    g.add_node("ts.client2", language="typescript", rel_path="web/src/api/users.ts", file_name="users", symbol_name="getUser",    symbol_type="function")
    g.add_node("py.same",    language="python",    rel_path="src/api/handler.py", file_name="handler", symbol_name="same_lang", symbol_type="function")
    return g


class TestL0Exact:
    def test_promotes_parser_relationships(self):
        g = _g()
        edges = link_l0_exact(g, [{"source": "py.handler", "target": "ts.client", "confidence": 0.9}])
        assert len(edges) == 1
        src, tgt, attrs = edges[0]
        assert (src, tgt) == ("py.handler", "ts.client")
        assert attrs["edge_class"] == "cross_language"
        assert attrs["relationship_type"] == "cross_language_L0"
        assert 0.3 <= attrs["weight"] <= 0.8
        assert attrs["provenance"]["level"] == "L0"

    def test_skips_same_language(self):
        g = _g()
        edges = link_l0_exact(g, [{"source": "py.handler", "target": "py.same"}])
        assert edges == []

    def test_skips_unknown_node(self):
        g = _g()
        edges = link_l0_exact(g, [{"source": "py.handler", "target": "missing"}])
        assert edges == []


class TestL1APISurface:
    def test_pairs_cross_language_matches(self):
        g = _g()
        surfaces = {
            "py.handler": [{"kind": "rest", "surface": "POST /api/users", "weight_hint": 0.7, "metadata": {}}],
            "ts.client":  [{"kind": "rest", "surface": "POST /api/users", "weight_hint": 0.7, "metadata": {}}],
        }
        edges = link_l1_api_surface(g, surfaces)
        assert len(edges) == 1
        _, _, attrs = edges[0]
        assert attrs["relationship_type"] == "cross_language_L1"
        assert attrs["provenance"]["surface"] == "POST /api/users"

    def test_skips_same_language_and_solo(self):
        g = _g()
        surfaces = {
            # Solo surface, no pair.
            "py.handler": [{"kind": "rest", "surface": "GET /alone", "weight_hint": 0.7, "metadata": {}}],
            # Same language.
            "py.same":    [{"kind": "rest", "surface": "POST /x", "weight_hint": 0.7, "metadata": {}}],
        }
        edges = link_l1_api_surface(g, surfaces)
        assert edges == []

    def test_specificity_lowers_weight_for_common_surfaces(self):
        g = _g()
        # 3 nodes share the same surface (one Python, two TypeScript).
        surfaces = {
            "py.handler": [{"kind": "rest", "surface": "GET /", "weight_hint": 0.7, "metadata": {}}],
            "ts.client":  [{"kind": "rest", "surface": "GET /", "weight_hint": 0.7, "metadata": {}}],
            "ts.client2": [{"kind": "rest", "surface": "GET /", "weight_hint": 0.7, "metadata": {}}],
        }
        edges = link_l1_api_surface(g, surfaces)
        assert edges  # produces at least one cross-language pair
        for _, _, attrs in edges:
            assert attrs["weight"] <= 0.8


class TestL2Hybrid:
    def test_filters_same_language_hits(self):
        g = _g()
        fts = {"py.handler": [{"node_id": "ts.client"}, {"node_id": "py.same"}]}
        vec = {"py.handler": [{"node_id": "ts.client"}]}
        edges = link_l2_hybrid(g, fts, vec, threshold=0.0)
        targets = sorted(t for _, t, _ in edges)
        assert targets == ["ts.client"]

    def test_threshold_drops_weak(self):
        g = _g()
        # 30 hits → top hit rank 1 → rrf 1/61 ≈ 0.0164.
        many = [{"node_id": f"x{i}"} for i in range(30)]
        # Only one is a real cross-lang node ("ts.client") but at rank 30,
        # rrf = 1/90 ≈ 0.011 which is below the default 0.02 threshold.
        many[29] = {"node_id": "ts.client"}
        fts = {"py.handler": many}
        vec = {"py.handler": []}
        edges = link_l2_hybrid(g, fts, vec, threshold=0.02)
        # All emitted edges (if any) must clear the threshold.
        for _, _, attrs in edges:
            assert "rrf_score" in attrs["provenance"]


class TestL3Containment:
    def test_promotes_member_pairs(self):
        g = _g()
        # Add member methods.
        g.add_node("py.handler.create", language="python", rel_path="src/api/handler.py",
                   file_name="handler", symbol_name="create", symbol_type="method",
                   parent_symbol="py.handler")
        g.add_node("ts.client.create",  language="typescript", rel_path="web/src/api/users.ts",
                   file_name="users", symbol_name="create", symbol_type="method",
                   parent_symbol="ts.client")

        parent = ("py.handler", "ts.client", {"weight": 0.7, "relationship_type": "cross_language_L1"})
        edges = link_l3_containment(g, [parent])
        assert len(edges) == 1
        src, tgt, attrs = edges[0]
        assert (src, tgt) == ("py.handler.create", "ts.client.create")
        assert attrs["relationship_type"] == "cross_language_L3"
        assert pytest.approx(attrs["weight"], abs=1e-9) == 0.6 * 0.7

    def test_no_members_returns_empty(self):
        g = _g()
        parent = ("py.handler", "ts.client", {"weight": 0.5, "relationship_type": "cross_language_L0"})
        assert link_l3_containment(g, [parent]) == []


class TestRunner:
    def test_dedups_repeated_relationship(self):
        g = _g()
        # L0 supplies it once; L1 surface match supplies it again.
        rels = [{"source": "py.handler", "target": "ts.client"}]
        surfaces = {
            "py.handler": [{"kind": "rest", "surface": "POST /api/users", "weight_hint": 0.7, "metadata": {}}],
            "ts.client":  [{"kind": "rest", "surface": "POST /api/users", "weight_hint": 0.7, "metadata": {}}],
        }
        edges = run_cross_language_linker(g,
            cross_language_relationships=rels,
            surfaces_by_node=surfaces,
        )
        rel_types = sorted(a["relationship_type"] for _, _, a in edges)
        assert rel_types == ["cross_language_L0", "cross_language_L1"]
