"""Phase 6 / Action 6.4 — test linker."""

from __future__ import annotations

import networkx as nx
import pytest

from app.core.code_graph.test_linker import (
    link_class_proxy,
    link_same_stem,
    run_test_linker,
)


def _add(g, nid, **attrs):
    g.add_node(nid, **attrs)


class TestSameStem:
    def test_pairs_test_with_prod(self):
        g = nx.MultiDiGraph()
        _add(g, "test:auth", language="python", rel_path="tests/test_auth.py",
             file_name="test_auth", symbol_name="test_thing", symbol_type="function")
        _add(g, "src:auth",  language="python", rel_path="src/auth.py",
             file_name="auth", symbol_name="login", symbol_type="function")

        edges = link_same_stem(g)
        assert len(edges) == 1
        src, tgt, attrs = edges[0]
        assert (src, tgt) == ("test:auth", "src:auth")
        assert attrs["edge_class"] == "test_link"
        assert attrs["relationship_type"] == "test_link_same_stem"
        assert attrs["weight"] == 0.5

    def test_no_match_when_stems_differ(self):
        g = nx.MultiDiGraph()
        _add(g, "test:other", language="python", rel_path="tests/test_other.py",
             file_name="test_other", symbol_name="test_x", symbol_type="function")
        _add(g, "src:auth",   language="python", rel_path="src/auth.py",
             file_name="auth", symbol_name="login", symbol_type="function")

        assert link_same_stem(g) == []

    def test_respects_language_boundary(self):
        g = nx.MultiDiGraph()
        _add(g, "test:auth", language="python", rel_path="tests/test_auth.py",
             file_name="test_auth", symbol_name="t", symbol_type="function")
        _add(g, "ts:auth",   language="typescript", rel_path="web/src/auth.ts",
             file_name="auth", symbol_name="login", symbol_type="function")

        assert link_same_stem(g) == []  # different languages → not paired


class TestClassProxy:
    def test_pairs_test_class_with_prod_class(self):
        g = nx.MultiDiGraph()
        _add(g, "test:UserService", language="python", rel_path="tests/test_user.py",
             file_name="test_user", symbol_name="TestUserService", symbol_type="class")
        _add(g, "src:UserService",  language="python", rel_path="src/user.py",
             file_name="user", symbol_name="UserService", symbol_type="class")

        edges = link_class_proxy(g)
        assert len(edges) == 1
        _, tgt, attrs = edges[0]
        assert tgt == "src:UserService"
        assert attrs["relationship_type"] == "test_link_class_proxy"

    def test_skips_when_no_test_marker(self):
        g = nx.MultiDiGraph()
        _add(g, "src:UserService",  language="python", rel_path="src/user.py",
             file_name="user", symbol_name="UserService", symbol_type="class")
        _add(g, "src:Helper",       language="python", rel_path="src/helper.py",
             file_name="helper", symbol_name="Helper",     symbol_type="class")

        assert link_class_proxy(g) == []


class TestRunner:
    def test_dedups_overlapping_heuristics(self):
        g = nx.MultiDiGraph()
        _add(g, "test:UserService", language="python", rel_path="tests/test_user.py",
             file_name="test_user", symbol_name="TestUserService", symbol_type="class")
        _add(g, "src:UserService",  language="python", rel_path="src/user.py",
             file_name="user", symbol_name="UserService", symbol_type="class")

        edges = run_test_linker(g)
        rel_types = sorted(a["relationship_type"] for _, _, a in edges)
        # Both heuristics should fire — but emit *different* relationship types.
        assert rel_types == ["test_link_class_proxy", "test_link_same_stem"]
