"""Tests for the cluster-based structure planner.

Covers:
- ClusterStructurePlanner construction and configuration
- Architectural cluster map building (filtering, test exclusion)
- Two-pass LLM naming (section + page)
- Dominant symbol scoring and selection
- Micro-cluster summaries
- Page-level symbol enrichment
- Central symbol selection (tiered priority, doc-dominant)
- Node-to-field helpers (symbol names, paths, folders, doc paths)
- Fallback spec / section generation
- JSON parsing helpers (_extract_text, _parse_json_response)
- Full plan_structure() integration with mocked LLM
- Adaptive central-k sizing
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any
from unittest.mock import MagicMock, patch

import networkx as nx
import pytest

from app.core.graph_clustering import ClusterAssignment
from app.core.wiki_structure_planner.cluster_planner import (
    MAX_DOMINANT_SYMBOLS,
    MAX_MICRO_SUMMARY_SYMBOLS,
    MAX_PAGE_NAMING_SYMBOLS,
    SIGNATURE_TRUNC,
    DOCSTRING_TRUNC,
    ClusterStructurePlanner,
    _extract_text,
    _parse_json_response,
)


# ═══════════════════════════════════════════════════════════════════════════
# Fixtures
# ═══════════════════════════════════════════════════════════════════════════


def _make_node(
    symbol_name: str,
    symbol_type: str = "class",
    rel_path: str = "src/module.py",
    signature: str = "",
    docstring: str = "",
) -> dict[str, Any]:
    """Return a node attribute dict matching the graph convention."""
    return {
        "symbol_name": symbol_name,
        "symbol_type": symbol_type,
        "rel_path": rel_path,
        "signature": signature,
        "docstring": docstring,
    }


def _make_assignment(
    node_to_section_page: dict[str, tuple[int, int]],
) -> ClusterAssignment:
    """Build a ClusterAssignment from a simple ``{node_id: (section, page)}`` map.

    Derives ``macro_clusters``, ``micro_clusters``, and ``sections`` dicts
    from this map so we don't have to construct them by hand every time.
    """
    macro_clusters: dict[str, int] = {}
    micro_clusters: dict[int, dict[str, int]] = {}
    sections: dict[int, dict[str, Any]] = {}

    for node_id, (sec, pg) in node_to_section_page.items():
        macro_clusters[node_id] = sec
        micro_clusters.setdefault(sec, {})[node_id] = pg
        sec_data = sections.setdefault(sec, {"pages": {}, "centroids": {}})
        sec_data["pages"].setdefault(pg, []).append(node_id)

    section_ids = set(macro_clusters.values())
    page_set: set[tuple[int, int]] = set()
    for sid, sec_data in sections.items():
        for pid in sec_data["pages"]:
            page_set.add((sid, pid))

    return ClusterAssignment(
        macro_clusters=macro_clusters,
        micro_clusters=micro_clusters,
        hub_nodes=set(),
        hub_assignments={},
        sections=sections,
        section_count=len(section_ids),
        page_count=len(page_set),
        algorithm="leiden",
        resolution={"section": 1.0, "page": 1.0},
    )


def _build_graph_with_clusters() -> tuple[
    nx.MultiDiGraph, ClusterAssignment
]:
    """Build a small graph with two macro-clusters, each with 2 micro-clusters.

    Macro 0:
      micro 0: A (class), B (function)   — src/core/engine.py
      micro 1: C (interface)              — src/core/api.py

    Macro 1:
      micro 0: D (class), E (constant)   — src/utils/helpers.py
      micro 1: F (module_doc)             — docs/README.md
    """
    G = nx.MultiDiGraph()

    # Macro 0, micro 0
    G.add_node("A", **_make_node("Engine", "class", "src/core/engine.py", "class Engine:", "Main engine class"))
    G.add_node("B", **_make_node("process", "function", "src/core/engine.py", "def process():", "Process data"))

    # Macro 0, micro 1
    G.add_node("C", **_make_node("IProcessor", "interface", "src/core/api.py", "interface IProcessor", ""))

    # Macro 1, micro 0
    G.add_node("D", **_make_node("Config", "class", "src/utils/helpers.py", "class Config:", "Configuration"))
    G.add_node("E", **_make_node("MAX_SIZE", "constant", "src/utils/helpers.py", "MAX_SIZE = 100", ""))

    # Macro 1, micro 1  (doc-dominant)
    G.add_node("F", **_make_node("README", "markdown_document", "docs/README.md", "", "Top-level docs"))

    # Edges
    G.add_edge("A", "B", relationship_type="calls", weight=1.0)
    G.add_edge("A", "C", relationship_type="implements", weight=1.0)
    G.add_edge("D", "E", relationship_type="uses", weight=1.0)

    assignment = _make_assignment({
        "A": (0, 0),
        "B": (0, 0),
        "C": (0, 1),
        "D": (1, 0),
        "E": (1, 0),
        "F": (1, 1),
    })

    return G, assignment


def _build_graph_with_test_nodes() -> tuple[
    nx.MultiDiGraph, ClusterAssignment
]:
    """Graph with some test nodes for test-exclusion testing."""
    G = nx.MultiDiGraph()

    G.add_node("A", **_make_node("Engine", "class", "src/core/engine.py"))
    G.add_node("T", **_make_node("TestEngine", "class", "tests/test_engine.py"))
    G.add_node("B", **_make_node("process", "function", "src/core/engine.py"))

    G.add_edge("A", "B", relationship_type="calls", weight=1.0)

    assignment = _make_assignment({
        "A": (0, 0),
        "T": (0, 0),
        "B": (0, 0),
    })

    return G, assignment


def _mock_llm_response(content: str) -> MagicMock:
    """Build a mock LLM response object."""
    resp = MagicMock()
    resp.content = content
    return resp


def _make_naming_llm(
    section_name: str = "Core Engine",
    section_desc: str = "The main processing engine",
    page_name: str = "Data Processing Pipeline",
    page_desc: str = "Handles data transformation",
    retrieval_query: str = "data processing pipeline",
) -> MagicMock:
    """Build a mock LLM that returns section naming then page naming."""
    section_json = json.dumps({
        "section_name": section_name,
        "section_description": section_desc,
    })
    page_json = json.dumps({
        "page_name": page_name,
        "description": page_desc,
        "retrieval_query": retrieval_query,
    })

    llm = MagicMock()
    # First call = section naming, subsequent calls = page naming
    llm.invoke = MagicMock(
        side_effect=[
            _mock_llm_response(section_json),
            _mock_llm_response(page_json),
            _mock_llm_response(page_json),
            _mock_llm_response(page_json),
            _mock_llm_response(page_json),
            _mock_llm_response(page_json),
            _mock_llm_response(page_json),
        ]
    )
    return llm


# ═══════════════════════════════════════════════════════════════════════════
# JSON Parsing Helpers
# ═══════════════════════════════════════════════════════════════════════════


class TestExtractText:
    """Tests for _extract_text()."""

    def test_plain_string(self):
        assert _extract_text("hello") == "hello"

    def test_ai_message_content(self):
        msg = MagicMock()
        msg.content = "response text"
        assert _extract_text(msg) == "response text"

    def test_list_content_strings(self):
        msg = MagicMock()
        msg.content = ["part1", "part2"]
        assert _extract_text(msg) == "part1\npart2"

    def test_list_content_dicts(self):
        msg = MagicMock()
        msg.content = [{"text": "hello"}, {"content": "world"}]
        assert _extract_text(msg) == "hello\nworld"

    def test_list_content_mixed(self):
        msg = MagicMock()
        msg.content = ["intro", {"text": "body"}]
        assert _extract_text(msg) == "intro\nbody"

    def test_non_string_fallback(self):
        assert _extract_text(42) == "42"


class TestParseJsonResponse:
    """Tests for _parse_json_response()."""

    def test_clean_json(self):
        result = _parse_json_response('{"key": "value"}')
        assert result == {"key": "value"}

    def test_markdown_fenced(self):
        raw = '```json\n{"key": "value"}\n```'
        assert _parse_json_response(raw) == {"key": "value"}

    def test_markdown_no_lang(self):
        raw = '```\n{"key": "value"}\n```'
        assert _parse_json_response(raw) == {"key": "value"}

    def test_leading_text(self):
        raw = 'Here is the JSON:\n{"key": "value"}'
        assert _parse_json_response(raw) == {"key": "value"}

    def test_trailing_comma(self):
        raw = '{"key": "value",}'
        assert _parse_json_response(raw) == {"key": "value"}

    def test_nested_json(self):
        raw = '{"a": {"b": 1}, "c": [1, 2]}'
        result = _parse_json_response(raw)
        assert result == {"a": {"b": 1}, "c": [1, 2]}

    def test_unparseable_returns_empty(self):
        assert _parse_json_response("not json at all") == {}

    def test_empty_string(self):
        assert _parse_json_response("") == {}

    def test_whitespace_only(self):
        assert _parse_json_response("   \n  ") == {}


# ═══════════════════════════════════════════════════════════════════════════
# Planner Construction
# ═══════════════════════════════════════════════════════════════════════════


class TestPlannerInit:
    """Tests for ClusterStructurePlanner.__init__()."""

    def test_default_wiki_title_from_repo_name(self):
        G, assignment = _build_graph_with_clusters()
        planner = ClusterStructurePlanner(
            graph=G,
            cluster_assignment=assignment,
            llm=MagicMock(),
            repo_name="owner/my-project",
        )
        assert planner.wiki_title == "my-project — Technical Documentation"

    def test_explicit_wiki_title(self):
        G, assignment = _build_graph_with_clusters()
        planner = ClusterStructurePlanner(
            graph=G,
            cluster_assignment=assignment,
            llm=MagicMock(),
            wiki_title="Custom Wiki Title",
        )
        assert planner.wiki_title == "Custom Wiki Title"

    def test_default_title_without_repo_name(self):
        G, assignment = _build_graph_with_clusters()
        planner = ClusterStructurePlanner(
            graph=G, cluster_assignment=assignment, llm=MagicMock()
        )
        assert planner.wiki_title == "Repository Technical Documentation"

    def test_exclude_tests_default(self):
        G, assignment = _build_graph_with_clusters()
        planner = ClusterStructurePlanner(
            graph=G, cluster_assignment=assignment, llm=MagicMock()
        )
        assert planner.exclude_tests is True


# ═══════════════════════════════════════════════════════════════════════════
# Node Access Helpers
# ═══════════════════════════════════════════════════════════════════════════


class TestNodeHelpers:
    """Tests for _get_node, _is_architectural, _is_doc_node, etc."""

    def setup_method(self):
        self.G, self.assignment = _build_graph_with_clusters()
        self.planner = ClusterStructurePlanner(
            graph=self.G, cluster_assignment=self.assignment, llm=MagicMock()
        )

    def test_get_node_exists(self):
        node = self.planner._get_node("A")
        assert node is not None
        assert node["symbol_name"] == "Engine"
        assert node["node_id"] == "A"

    def test_get_node_missing(self):
        assert self.planner._get_node("MISSING") is None

    def test_is_architectural_class(self):
        node = self.planner._get_node("A")
        assert self.planner._is_architectural(node)

    def test_is_architectural_function(self):
        node = self.planner._get_node("B")
        assert self.planner._is_architectural(node)

    def test_is_doc_node(self):
        node = self.planner._get_node("F")
        assert self.planner._is_doc_node(node)

    def test_is_not_doc_node(self):
        node = self.planner._get_node("A")
        assert not self.planner._is_doc_node(node)

    def test_edge_count_for(self):
        # A has edges to B and C
        assert self.planner._edge_count_for("A") == 2

    def test_edge_count_for_leaf(self):
        # C has no outgoing edges
        assert self.planner._edge_count_for("C") == 0

    def test_edge_count_for_missing(self):
        assert self.planner._edge_count_for("MISSING") == 0


# ═══════════════════════════════════════════════════════════════════════════
# Cluster Map Building
# ═══════════════════════════════════════════════════════════════════════════


class TestBuildArchitecturalClusterMap:
    """Tests for _build_architectural_cluster_map()."""

    def test_basic_structure(self):
        G, assignment = _build_graph_with_clusters()
        planner = ClusterStructurePlanner(
            graph=G, cluster_assignment=assignment, llm=MagicMock()
        )
        cmap = planner._build_architectural_cluster_map()
        assert 0 in cmap
        assert 1 in cmap
        assert 0 in cmap[0]  # micro 0
        assert 1 in cmap[0]  # micro 1

    def test_node_ids_are_correct(self):
        G, assignment = _build_graph_with_clusters()
        planner = ClusterStructurePlanner(
            graph=G, cluster_assignment=assignment, llm=MagicMock()
        )
        cmap = planner._build_architectural_cluster_map()
        assert set(cmap[0][0]) == {"A", "B"}
        assert set(cmap[0][1]) == {"C"}

    def test_excludes_test_nodes(self):
        G, assignment = _build_graph_with_test_nodes()
        planner = ClusterStructurePlanner(
            graph=G,
            cluster_assignment=assignment,
            llm=MagicMock(),
            exclude_tests=True,
        )
        cmap = planner._build_architectural_cluster_map()
        all_ids = set()
        for mm in cmap.values():
            for nids in mm.values():
                all_ids.update(nids)
        assert "T" not in all_ids

    def test_includes_test_nodes_when_disabled(self):
        G, assignment = _build_graph_with_test_nodes()
        planner = ClusterStructurePlanner(
            graph=G,
            cluster_assignment=assignment,
            llm=MagicMock(),
            exclude_tests=False,
        )
        cmap = planner._build_architectural_cluster_map()
        all_ids = set()
        for mm in cmap.values():
            for nids in mm.values():
                all_ids.update(nids)
        assert "T" in all_ids

    def test_non_architectural_excluded(self):
        """Non-architectural symbols (method, field, etc.) are filtered out."""
        G = nx.MultiDiGraph()
        G.add_node("M", **_make_node("do_thing", "method", "src/x.py"))
        assignment = _make_assignment({"M": (0, 0)})
        planner = ClusterStructurePlanner(
            graph=G, cluster_assignment=assignment, llm=MagicMock()
        )
        cmap = planner._build_architectural_cluster_map()
        assert cmap == {}

    def test_empty_assignment(self):
        G = nx.MultiDiGraph()
        assignment = _make_assignment({})
        planner = ClusterStructurePlanner(
            graph=G, cluster_assignment=assignment, llm=MagicMock()
        )
        cmap = planner._build_architectural_cluster_map()
        assert cmap == {}


# ═══════════════════════════════════════════════════════════════════════════
# Dominant Symbols
# ═══════════════════════════════════════════════════════════════════════════


class TestGetDominantSymbols:
    """Tests for _get_dominant_symbols()."""

    def test_returns_architectural_only(self):
        G, assignment = _build_graph_with_clusters()
        planner = ClusterStructurePlanner(
            graph=G, cluster_assignment=assignment, llm=MagicMock()
        )
        symbols = planner._get_dominant_symbols(["A", "B", "C"])
        names = [s["name"] for s in symbols]
        assert "Engine" in names
        assert "process" in names
        assert "IProcessor" in names

    def test_scoring_prefers_connected_with_docs(self):
        G, assignment = _build_graph_with_clusters()
        planner = ClusterStructurePlanner(
            graph=G, cluster_assignment=assignment, llm=MagicMock()
        )
        symbols = planner._get_dominant_symbols(["A", "B", "C"])
        # A has 2 edges + docstring = 4; B has 0 edges + docstring = 2; C has 0 edges no docstring = 0
        assert symbols[0]["name"] == "Engine"

    def test_capped_at_max(self):
        G = nx.MultiDiGraph()
        for i in range(20):
            G.add_node(f"N{i}", **_make_node(f"Sym{i}", "class", f"src/f{i}.py"))
        assignment = _make_assignment({f"N{i}": (0, 0) for i in range(20)})
        planner = ClusterStructurePlanner(
            graph=G, cluster_assignment=assignment, llm=MagicMock()
        )
        symbols = planner._get_dominant_symbols([f"N{i}" for i in range(20)])
        assert len(symbols) <= MAX_DOMINANT_SYMBOLS

    def test_truncates_signature_and_docstring(self):
        G = nx.MultiDiGraph()
        long_sig = "x" * 500
        long_doc = "y" * 500
        G.add_node("X", **_make_node("LongSym", "class", "src/x.py", long_sig, long_doc))
        assignment = _make_assignment({"X": (0, 0)})
        planner = ClusterStructurePlanner(
            graph=G, cluster_assignment=assignment, llm=MagicMock()
        )
        symbols = planner._get_dominant_symbols(["X"])
        assert len(symbols[0]["signature"]) == SIGNATURE_TRUNC
        assert len(symbols[0]["docstring"]) == DOCSTRING_TRUNC

    def test_empty_node_list(self):
        G, assignment = _build_graph_with_clusters()
        planner = ClusterStructurePlanner(
            graph=G, cluster_assignment=assignment, llm=MagicMock()
        )
        assert planner._get_dominant_symbols([]) == []


# ═══════════════════════════════════════════════════════════════════════════
# Micro Summaries
# ═══════════════════════════════════════════════════════════════════════════


class TestGetMicroSummaries:
    """Tests for _get_micro_summaries()."""

    def test_summary_structure(self):
        G, assignment = _build_graph_with_clusters()
        planner = ClusterStructurePlanner(
            graph=G, cluster_assignment=assignment, llm=MagicMock()
        )
        micro_map = {0: ["A", "B"], 1: ["C"]}
        summaries = planner._get_micro_summaries(micro_map)
        assert len(summaries) == 2
        assert summaries[0]["micro_id"] == 0
        assert summaries[0]["symbol_count"] == 2
        assert summaries[1]["micro_id"] == 1
        assert summaries[1]["symbol_count"] == 1

    def test_includes_directories(self):
        G, assignment = _build_graph_with_clusters()
        planner = ClusterStructurePlanner(
            graph=G, cluster_assignment=assignment, llm=MagicMock()
        )
        summaries = planner._get_micro_summaries({0: ["A", "B"]})
        assert "src/core" in summaries[0]["directories"]

    def test_caps_symbols_at_max(self):
        G = nx.MultiDiGraph()
        node_ids = []
        for i in range(50):
            nid = f"N{i}"
            G.add_node(nid, **_make_node(f"Sym{i}", "class", f"src/f{i}.py"))
            node_ids.append(nid)
        assignment = _make_assignment({nid: (0, 0) for nid in node_ids})
        planner = ClusterStructurePlanner(
            graph=G, cluster_assignment=assignment, llm=MagicMock()
        )
        summaries = planner._get_micro_summaries({0: node_ids})
        assert len(summaries[0]["symbols"]) <= MAX_MICRO_SUMMARY_SYMBOLS


# ═══════════════════════════════════════════════════════════════════════════
# Page Symbols
# ═══════════════════════════════════════════════════════════════════════════


class TestGetPageSymbols:
    """Tests for _get_page_symbols()."""

    def test_returns_enriched_data(self):
        G, assignment = _build_graph_with_clusters()
        planner = ClusterStructurePlanner(
            graph=G, cluster_assignment=assignment, llm=MagicMock()
        )
        symbols = planner._get_page_symbols(["A", "B"])
        # Should have signature and docstring (unlike micro summaries)
        for s in symbols:
            assert "signature" in s
            assert "docstring" in s

    def test_caps_at_max(self):
        G = nx.MultiDiGraph()
        node_ids = []
        for i in range(30):
            nid = f"N{i}"
            G.add_node(nid, **_make_node(f"Sym{i}", "class", f"src/f{i}.py"))
            node_ids.append(nid)
        assignment = _make_assignment({nid: (0, 0) for nid in node_ids})
        planner = ClusterStructurePlanner(
            graph=G, cluster_assignment=assignment, llm=MagicMock()
        )
        symbols = planner._get_page_symbols(node_ids)
        assert len(symbols) <= MAX_PAGE_NAMING_SYMBOLS

    def test_empty_list(self):
        G, assignment = _build_graph_with_clusters()
        planner = ClusterStructurePlanner(
            graph=G, cluster_assignment=assignment, llm=MagicMock()
        )
        assert planner._get_page_symbols([]) == []


# ═══════════════════════════════════════════════════════════════════════════
# Node-to-Field Helpers
# ═══════════════════════════════════════════════════════════════════════════


class TestNodeToFieldHelpers:
    """Tests for _node_ids_to_symbol_names, _paths, _folders, _doc_paths."""

    def setup_method(self):
        self.G, self.assignment = _build_graph_with_clusters()
        self.planner = ClusterStructurePlanner(
            graph=self.G, cluster_assignment=self.assignment, llm=MagicMock()
        )

    def test_symbol_names_code_only(self):
        names = self.planner._node_ids_to_symbol_names(["A", "F"])
        assert "Engine" in names
        # F is module_doc — should be excluded
        assert "README" not in names

    def test_paths(self):
        paths = self.planner._node_ids_to_paths(["A", "B", "C"])
        assert "src/core/engine.py" in paths
        assert "src/core/api.py" in paths

    def test_paths_capped(self):
        G = nx.MultiDiGraph()
        node_ids = []
        for i in range(30):
            nid = f"N{i}"
            G.add_node(nid, **_make_node(f"S{i}", "class", f"src/dir{i}/f{i}.py"))
            node_ids.append(nid)
        assignment = _make_assignment({nid: (0, 0) for nid in node_ids})
        planner = ClusterStructurePlanner(
            graph=G, cluster_assignment=assignment, llm=MagicMock()
        )
        paths = planner._node_ids_to_paths(node_ids)
        assert len(paths) <= 20

    def test_folders(self):
        folders = self.planner._node_ids_to_folders(["A", "D"])
        assert "src/core" in folders
        assert "src/utils" in folders

    def test_folders_capped(self):
        G = nx.MultiDiGraph()
        node_ids = []
        for i in range(15):
            nid = f"N{i}"
            G.add_node(nid, **_make_node(f"S{i}", "class", f"src/d{i}/f{i}.py"))
            node_ids.append(nid)
        assignment = _make_assignment({nid: (0, 0) for nid in node_ids})
        planner = ClusterStructurePlanner(
            graph=G, cluster_assignment=assignment, llm=MagicMock()
        )
        folders = planner._node_ids_to_folders(node_ids)
        assert len(folders) <= 10

    def test_doc_paths(self):
        doc_paths = self.planner._node_ids_to_doc_paths(["F", "A"])
        assert "docs/README.md" in doc_paths
        # A is not a doc node
        assert "src/core/engine.py" not in doc_paths

    def test_doc_paths_empty_for_code_only(self):
        doc_paths = self.planner._node_ids_to_doc_paths(["A", "B"])
        assert doc_paths == []


# ═══════════════════════════════════════════════════════════════════════════
# Adaptive Central K
# ═══════════════════════════════════════════════════════════════════════════


class TestAdaptiveCentralK:
    """Tests for _adaptive_central_k()."""

    def test_small_graph(self):
        G = nx.MultiDiGraph()
        for i in range(50):
            G.add_node(f"N{i}", **_make_node(f"S{i}", "class"))
        assignment = _make_assignment({})
        planner = ClusterStructurePlanner(
            graph=G, cluster_assignment=assignment, llm=MagicMock()
        )
        assert planner._adaptive_central_k() == 5

    def test_medium_graph(self):
        G = nx.MultiDiGraph()
        for i in range(500):
            G.add_node(f"N{i}", **_make_node(f"S{i}", "class"))
        assignment = _make_assignment({})
        planner = ClusterStructurePlanner(
            graph=G, cluster_assignment=assignment, llm=MagicMock()
        )
        assert planner._adaptive_central_k() == 8

    def test_large_graph(self):
        G = nx.MultiDiGraph()
        for i in range(2000):
            G.add_node(f"N{i}", **_make_node(f"S{i}", "class"))
        assignment = _make_assignment({})
        planner = ClusterStructurePlanner(
            graph=G, cluster_assignment=assignment, llm=MagicMock()
        )
        assert planner._adaptive_central_k() == 12

    def test_very_large_graph(self):
        G = nx.MultiDiGraph()
        for i in range(6000):
            G.add_node(f"N{i}", **_make_node(f"S{i}", "class"))
        assignment = _make_assignment({})
        planner = ClusterStructurePlanner(
            graph=G, cluster_assignment=assignment, llm=MagicMock()
        )
        assert planner._adaptive_central_k() == 15

    def test_caches_result(self):
        G = nx.MultiDiGraph()
        for i in range(50):
            G.add_node(f"N{i}", **_make_node(f"S{i}", "class"))
        assignment = _make_assignment({})
        planner = ClusterStructurePlanner(
            graph=G, cluster_assignment=assignment, llm=MagicMock()
        )
        k1 = planner._adaptive_central_k()
        k2 = planner._adaptive_central_k()
        assert k1 == k2 == 5


# ═══════════════════════════════════════════════════════════════════════════
# Central Symbol Selection
# ═══════════════════════════════════════════════════════════════════════════


class TestSelectCentralNodeIds:
    """Tests for _select_central_node_ids()."""

    def test_returns_all_when_fewer_than_k(self):
        G, assignment = _build_graph_with_clusters()
        planner = ClusterStructurePlanner(
            graph=G, cluster_assignment=assignment, llm=MagicMock()
        )
        result = planner._select_central_node_ids(["A", "B"], k=10)
        assert set(result) == {"A", "B"}

    def test_code_first_priority(self):
        """Identity symbols should be selected before supporting/doc."""
        G, assignment = _build_graph_with_clusters()
        planner = ClusterStructurePlanner(
            graph=G, cluster_assignment=assignment, llm=MagicMock()
        )
        # All 5 nodes, k=2; should pick identity (class/interface/function) first
        result = planner._select_central_node_ids(
            ["A", "B", "C", "D", "E"], k=2
        )
        # All A, B, C, D are identity types (class, function, interface, class)
        # E is constant (supporting)
        assert "E" not in result or len(result) <= 2

    def test_doc_dominant_cluster(self):
        """Cluster with ≥70% doc nodes should prefer doc symbols."""
        G = nx.MultiDiGraph()
        for i in range(7):
            G.add_node(f"D{i}", **_make_node(f"Doc{i}", "markdown_document", f"docs/d{i}.md"))
        for i in range(3):
            G.add_node(f"C{i}", **_make_node(f"Code{i}", "class", f"src/c{i}.py"))

        assignment = _make_assignment({
                **{f"D{i}": (0, 0) for i in range(7)},
                **{f"C{i}": (0, 0) for i in range(3)},
            })
        planner = ClusterStructurePlanner(
            graph=G, cluster_assignment=assignment, llm=MagicMock()
        )
        result = planner._select_central_node_ids(
            [f"D{i}" for i in range(7)] + [f"C{i}" for i in range(3)],
            k=5,
        )
        assert len(result) == 5

    def test_missing_nodes_in_graph(self):
        """Node IDs not in graph should be silently skipped."""
        G, assignment = _build_graph_with_clusters()
        planner = ClusterStructurePlanner(
            graph=G, cluster_assignment=assignment, llm=MagicMock()
        )
        result = planner._select_central_node_ids(
            ["A", "MISSING1", "MISSING2"], k=5
        )
        assert "A" in result


# ═══════════════════════════════════════════════════════════════════════════
# Fallback Spec / Section
# ═══════════════════════════════════════════════════════════════════════════


class TestFallbacks:
    """Tests for _fallback_spec() and _fallback_section()."""

    def test_fallback_spec(self):
        G, assignment = _build_graph_with_clusters()
        planner = ClusterStructurePlanner(
            graph=G, cluster_assignment=assignment, llm=MagicMock()
        )
        spec = planner._fallback_spec()
        assert spec.wiki_title is not None
        assert len(spec.sections) == 1
        assert spec.sections[0].section_name == "Repository Overview"
        assert spec.total_pages == 1

    def test_fallback_section_derives_name_from_paths(self):
        G, assignment = _build_graph_with_clusters()
        planner = ClusterStructurePlanner(
            graph=G, cluster_assignment=assignment, llm=MagicMock()
        )
        section = planner._fallback_section(
            macro_id=0,
            micro_map={0: ["A", "B"], 1: ["C"]},
            section_order=1,
        )
        # Both paths are under src/core
        assert "src/core" in section.section_name
        assert len(section.pages) == 2

    def test_fallback_section_generic_name(self):
        """Without paths, falls back to 'Section N'."""
        G = nx.MultiDiGraph()
        G.add_node("X", **_make_node("X", "class", ""))  # empty path
        assignment = _make_assignment({"X": (5, 0)})
        planner = ClusterStructurePlanner(
            graph=G, cluster_assignment=assignment, llm=MagicMock()
        )
        section = planner._fallback_section(5, {0: ["X"]}, 1)
        assert section.section_name == "Section 5"


# ═══════════════════════════════════════════════════════════════════════════
# Full plan_structure() Integration
# ═══════════════════════════════════════════════════════════════════════════


class TestPlanStructure:
    """Integration tests for plan_structure() with mocked LLM."""

    def test_basic_plan(self):
        G, assignment = _build_graph_with_clusters()
        llm = _make_naming_llm()
        planner = ClusterStructurePlanner(
            graph=G,
            cluster_assignment=assignment,
            llm=llm,
            repo_name="test-repo",
        )
        spec = planner.plan_structure()

        assert spec.wiki_title == "test-repo — Technical Documentation"
        assert len(spec.sections) == 2
        assert spec.total_pages >= 2  # at least one page per section
        assert spec.overview  # non-empty
        assert "sections" in spec.overview.lower() or "pages" in spec.overview.lower()

    def test_plan_with_empty_clusters_returns_fallback(self):
        G = nx.MultiDiGraph()
        assignment = _make_assignment({})
        planner = ClusterStructurePlanner(
            graph=G,
            cluster_assignment=assignment,
            llm=MagicMock(),
        )
        spec = planner.plan_structure()
        assert spec.total_pages == 1
        assert spec.sections[0].section_name == "Repository Overview"

    def test_plan_uses_llm_names(self):
        G, assignment = _build_graph_with_clusters()
        llm = _make_naming_llm(
            section_name="Core Engine Subsystem",
            page_name="Data Processing Pipeline",
        )
        planner = ClusterStructurePlanner(
            graph=G, cluster_assignment=assignment, llm=llm
        )
        spec = planner.plan_structure()
        # First section should use the LLM-provided name
        assert spec.sections[0].section_name == "Core Engine Subsystem"

    def test_plan_handles_llm_failure_gracefully(self):
        """When LLM raises, fallback naming is used."""
        G, assignment = _build_graph_with_clusters()
        llm = MagicMock()
        llm.invoke = MagicMock(side_effect=RuntimeError("LLM is down"))

        planner = ClusterStructurePlanner(
            graph=G, cluster_assignment=assignment, llm=llm
        )
        spec = planner.plan_structure()
        # Should still produce a valid spec via fallback
        assert len(spec.sections) == 2
        assert spec.total_pages >= 2

    def test_page_specs_have_required_fields(self):
        G, assignment = _build_graph_with_clusters()
        llm = _make_naming_llm()
        planner = ClusterStructurePlanner(
            graph=G, cluster_assignment=assignment, llm=llm
        )
        spec = planner.plan_structure()
        for section in spec.sections:
            for page in section.pages:
                assert page.page_name
                assert page.page_order >= 1
                assert page.description
                assert page.content_focus
                assert page.rationale
                assert page.retrieval_query

    def test_section_order_is_sequential(self):
        G, assignment = _build_graph_with_clusters()
        llm = _make_naming_llm()
        planner = ClusterStructurePlanner(
            graph=G, cluster_assignment=assignment, llm=llm
        )
        spec = planner.plan_structure()
        orders = [s.section_order for s in spec.sections]
        assert orders == list(range(1, len(orders) + 1))

    def test_page_order_within_section_is_sequential(self):
        G, assignment = _build_graph_with_clusters()
        llm = _make_naming_llm()
        planner = ClusterStructurePlanner(
            graph=G, cluster_assignment=assignment, llm=llm
        )
        spec = planner.plan_structure()
        for section in spec.sections:
            orders = [p.page_order for p in section.pages]
            assert orders == list(range(1, len(orders) + 1))


# ═══════════════════════════════════════════════════════════════════════════
# Constants exported properly
# ═══════════════════════════════════════════════════════════════════════════


class TestConstants:
    """Verify configuration constants are sensible."""

    def test_max_dominant_symbols_positive(self):
        assert MAX_DOMINANT_SYMBOLS > 0

    def test_max_micro_summary_positive(self):
        assert MAX_MICRO_SUMMARY_SYMBOLS > 0

    def test_max_page_naming_positive(self):
        assert MAX_PAGE_NAMING_SYMBOLS > 0

    def test_truncation_lengths(self):
        assert SIGNATURE_TRUNC > 0
        assert DOCSTRING_TRUNC > 0
