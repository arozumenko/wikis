"""Phase 5 / Action 3B — qualified-name + FQN indexes."""

from __future__ import annotations

from unittest.mock import patch

import networkx as nx
import pytest

from app.core.code_graph.graph_builder import attach_graph_indexes
from app.core.code_graph.graph_query_service import GraphQueryService
from app.core.feature_flags import FeatureFlags


def _flags(qualified_name_index: bool = True):
    return FeatureFlags(
        node_id_style="rel_path",
        qualified_name_index=qualified_name_index,
    )


def _build_graph_with_two_inits():
    """Two ``__init__`` methods on different classes in different files."""
    g = nx.MultiDiGraph()
    g.add_node(
        "python::src__a_py::AppConfig.__init__",
        symbol_name="__init__",
        full_name="a.AppConfig.__init__",
        file_path="src/a.py",
        rel_path="src/a.py",
        file_name="a",
        language="python",
        symbol_type="method",
    )
    g.add_node(
        "python::src__b_py::DatabaseConfig.__init__",
        symbol_name="__init__",
        full_name="b.DatabaseConfig.__init__",
        file_path="src/b.py",
        rel_path="src/b.py",
        file_name="b",
        language="python",
        symbol_type="method",
    )
    return g


class TestAttachGraphIndexes:
    def test_qualified_index_keys_by_parent_dot_symbol(self):
        g = _build_graph_with_two_inits()
        with patch("app.core.code_graph.graph_builder.get_feature_flags", return_value=_flags(True)):
            attach_graph_indexes(g)

        idx = g._qualified_name_index
        # AppConfig.__init__ + DatabaseConfig.__init__ — distinct keys.
        assert "AppConfig.__init__" in idx
        assert "DatabaseConfig.__init__" in idx
        assert idx["AppConfig.__init__"] == ["python::src__a_py::AppConfig.__init__"]
        assert idx["DatabaseConfig.__init__"] == ["python::src__b_py::DatabaseConfig.__init__"]

    def test_fqn_index_keys_by_relpath_qualified(self):
        g = _build_graph_with_two_inits()
        with patch("app.core.code_graph.graph_builder.get_feature_flags", return_value=_flags(True)):
            attach_graph_indexes(g)

        fqn = g._fqn_index
        assert fqn["src/a.py::AppConfig.__init__"] == "python::src__a_py::AppConfig.__init__"
        assert fqn["src/b.py::DatabaseConfig.__init__"] == "python::src__b_py::DatabaseConfig.__init__"

    def test_disabled_flag_yields_empty_indexes(self):
        g = _build_graph_with_two_inits()
        with patch("app.core.code_graph.graph_builder.get_feature_flags", return_value=_flags(False)):
            attach_graph_indexes(g)

        assert g._qualified_name_index == {}
        assert g._fqn_index == {}


class TestResolveSymbolPrefersQualified:
    def test_resolves_via_fqn_first(self):
        g = _build_graph_with_two_inits()
        with patch("app.core.code_graph.graph_builder.get_feature_flags", return_value=_flags(True)):
            attach_graph_indexes(g)

        svc = GraphQueryService(g)
        # FQN form: pass qualified name + rel_path file_path hint.
        nid = svc.resolve_symbol("AppConfig.__init__", file_path="src/a.py")
        assert nid == "python::src__a_py::AppConfig.__init__"

    def test_resolves_via_qualified_when_fqn_misses(self):
        g = _build_graph_with_two_inits()
        with patch("app.core.code_graph.graph_builder.get_feature_flags", return_value=_flags(True)):
            attach_graph_indexes(g)

        svc = GraphQueryService(g)
        # No file hint — must still resolve unambiguously via qualified index.
        nid = svc.resolve_symbol("DatabaseConfig.__init__")
        assert nid == "python::src__b_py::DatabaseConfig.__init__"

    def test_legacy_lookup_still_works_when_qualified_disabled(self):
        g = _build_graph_with_two_inits()
        with patch("app.core.code_graph.graph_builder.get_feature_flags", return_value=_flags(False)):
            attach_graph_indexes(g)

        svc = GraphQueryService(g)
        # Bare ``__init__`` + file hint must still hit the simple-name index.
        nid = svc.resolve_symbol("__init__", file_path="src/a.py", language="python")
        assert nid == "python::src__a_py::AppConfig.__init__"
