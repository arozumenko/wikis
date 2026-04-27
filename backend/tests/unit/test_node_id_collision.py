"""Phase 5 / Action 3A — node-id style.

Verifies that ``WIKI_NODE_ID_STYLE=rel_path`` produces collision-free
node IDs when two same-stem files live in different directories
(e.g. ``src/auth/handler.py`` vs ``src/api/handler.py``), while the
default ``"stem"`` mode keeps the legacy hash-suffix branch.
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import networkx as nx
import pytest

from app.core.code_graph.graph_builder import EnhancedUnifiedGraphBuilder
from app.core.feature_flags import FeatureFlags


def _builder():
    """Construct a builder with parser deps mocked out (we never parse)."""
    with patch("app.core.code_graph.graph_builder.GraphAwareCodeSplitter") as mock_splitter_cls, \
         patch("app.core.code_graph.graph_builder.CppEnhancedParser"), \
         patch("app.core.code_graph.graph_builder.GoVisitorParser"), \
         patch("app.core.code_graph.graph_builder.JavaVisitorParser"), \
         patch("app.core.code_graph.graph_builder.JavaScriptVisitorParser"), \
         patch("app.core.code_graph.graph_builder.PythonParser"), \
         patch("app.core.code_graph.graph_builder.RustVisitorParser"), \
         patch("app.core.code_graph.graph_builder.TypeScriptEnhancedParser"):
        mock_splitter = MagicMock()
        mock_splitter.parsers = {}
        mock_splitter.symbol_table = {}
        mock_splitter.file_imports = {}
        mock_splitter_cls.return_value = mock_splitter
        return EnhancedUnifiedGraphBuilder(max_workers=1)


def _symbol(name: str, full_name: str | None = None, stype: str = "class"):
    """Minimal duck-typed symbol object that satisfies _process_file_symbols."""
    return SimpleNamespace(
        name=name,
        full_name=full_name,
        symbol_type=stype,                 # plain str — no .value
        range=SimpleNamespace(
            start=SimpleNamespace(line=1),
            end=SimpleNamespace(line=10),
        ),
        parent_symbol=None,
    )


def _result(symbols):
    return SimpleNamespace(symbols=symbols, relationships=[])


def _registry():
    return {"by_name": {}, "by_qualified_name": {}, "by_full_path": {}}


def _flags(**overrides):
    base = dict(node_id_style="stem", qualified_name_index=True)
    base.update(overrides)
    return FeatureFlags(**base)


# --------------------------------------------------------------------------
# Action 3A: rel_path mode is collision-free
# --------------------------------------------------------------------------


class TestNodeIdRelPath:
    def test_same_stem_different_dirs_collides_in_stem_mode(self):
        b = _builder()
        g = nx.MultiDiGraph()
        reg = _registry()

        with patch("app.core.code_graph.graph_builder.get_feature_flags", return_value=_flags(node_id_style="stem")):
            b._process_file_symbols(
                g, "/repo/src/auth/handler.py", _result([_symbol("Handler")]),
                "python", reg, repo_root="/repo",
            )
            b._process_file_symbols(
                g, "/repo/src/api/handler.py", _result([_symbol("Handler")]),
                "python", reg, repo_root="/repo",
            )

        # Stem mode keeps the hash-suffix branch — two nodes total.
        node_ids = [n for n in g.nodes() if n.startswith("python::handler::")]
        assert len(node_ids) == 2
        # Exactly one of them carries the 4-hex collision suffix.
        suffixed = [nid for nid in node_ids if nid.endswith(tuple("0123456789abcdef"))
                    and len(nid.split("_")[-1]) == 4]
        assert len(suffixed) == 1

    def test_same_stem_different_dirs_unique_in_rel_path_mode(self):
        b = _builder()
        g = nx.MultiDiGraph()
        reg = _registry()

        with patch("app.core.code_graph.graph_builder.get_feature_flags", return_value=_flags(node_id_style="rel_path")):
            b._process_file_symbols(
                g, "/repo/src/auth/handler.py", _result([_symbol("Handler")]),
                "python", reg, repo_root="/repo",
            )
            b._process_file_symbols(
                g, "/repo/src/api/handler.py", _result([_symbol("Handler")]),
                "python", reg, repo_root="/repo",
            )

        ids = sorted(g.nodes())
        assert ids == [
            "python::src__api__handler_py::Handler",
            "python::src__auth__handler_py::Handler",
        ]
        # No hash-suffix nodes at all.
        assert not any("_" in nid.split("::")[-1].split(".")[-1]
                       and len(nid.rsplit("_", 1)[-1]) == 4
                       for nid in ids)

    def test_true_same_file_duplicate_skipped_in_rel_path_mode(self):
        """Same symbol added twice from same file must collapse to one node."""
        b = _builder()
        g = nx.MultiDiGraph()
        reg = _registry()

        with patch("app.core.code_graph.graph_builder.get_feature_flags", return_value=_flags(node_id_style="rel_path")):
            b._process_file_symbols(
                g, "/repo/src/x.py", _result([_symbol("X"), _symbol("X")]),
                "python", reg, repo_root="/repo",
            )

        assert list(g.nodes()) == ["python::src__x_py::X"]
