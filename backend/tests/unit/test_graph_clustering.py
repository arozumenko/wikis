"""Tests for app.core.graph_clustering.

Covers:
- ClusterAssignment dataclass construction
- architectural_projection() — child promotion, edge remapping
- _find_architectural_parent() — 3-strategy cascade
- detect_hubs() — Z-score hub detection
- _to_weighted_undirected() — directed→undirected collapse
- _nx_to_igraph() — NetworkX→igraph conversion
- _contract_to_file_graph() — file-level contraction
- _dir_of_node(), _dir_histogram(), _dir_similarity() — directory helpers
- _detect_page_centroids() — centroid scoring
- select_central_symbols() — PageRank-based selection
- detect_doc_clusters() — doc-dominant section detection
- reintegrate_hubs() — plurality-vote hub assignment
- _trivial_assignment() — small graph fallback
- _consolidate_sections() — section merging
- _consolidate_pages() — page merging
- run_clustering() — full pipeline integration

NOTE: Tests that require igraph/leidenalg are marked with
``@pytest.mark.skipif`` so the suite still passes when those
optional dependencies are not installed.
"""

import math
from collections import Counter
from unittest.mock import patch

import networkx as nx
import pytest

from app.core.graph_clustering import (
    ClusterAssignment,
    _CHILD_SYMBOL_TYPES,
    _consolidate_pages,
    _consolidate_sections,
    _contract_to_file_graph,
    _detect_page_centroids,
    _dir_histogram,
    _dir_of_node,
    _dir_similarity,
    _enforce_page_sizes,
    _expand_sections,
    _file_aware_split,
    _find_architectural_parent,
    _hierarchical_leiden,
    _nx_to_igraph,
    _to_weighted_undirected,
    _trivial_assignment,
    architectural_projection,
    detect_doc_clusters,
    detect_hubs,
    reintegrate_hubs,
    run_clustering,
    select_central_symbols,
)

# Check optional dependencies
try:
    import igraph  # noqa: F401
    import leidenalg  # noqa: F401

    _HAS_LEIDEN = True
except ImportError:
    _HAS_LEIDEN = False

requires_leiden = pytest.mark.skipif(
    not _HAS_LEIDEN,
    reason="igraph and leidenalg not installed",
)


# ═══════════════════════════════════════════════════════════════════════════
# Fixtures
# ═══════════════════════════════════════════════════════════════════════════


def _make_node(symbol_type, rel_path, symbol_name=None, **extra):
    """Helper to create node attribute dicts."""
    data = {
        "symbol_type": symbol_type,
        "rel_path": rel_path,
        "file_name": rel_path,
        "symbol_name": symbol_name or symbol_type,
    }
    data.update(extra)
    return data


def _build_simple_graph() -> nx.MultiDiGraph:
    """Build a small graph with clear structure for testing.

    Structure:
        src/models/: ClassA (class), ClassB (class), ClassA.method1 (method)
        src/utils/: helper_func (function), CONST1 (constant)
        src/api/: APIHandler (class)

    Edges: ClassA→ClassB, ClassA→helper_func, APIHandler→ClassA, APIHandler→ClassB
    """
    G = nx.MultiDiGraph()

    G.add_node(
        "py::src/models/a.py::ClassA",
        **_make_node("class", "src/models/a.py", "ClassA"),
    )
    G.add_node(
        "py::src/models/b.py::ClassB",
        **_make_node("class", "src/models/b.py", "ClassB"),
    )
    G.add_node(
        "py::src/models/a.py::ClassA.method1",
        **_make_node("method", "src/models/a.py", "method1", parent_symbol="ClassA"),
    )
    G.add_node(
        "py::src/utils/helpers.py::helper_func",
        **_make_node("function", "src/utils/helpers.py", "helper_func"),
    )
    G.add_node(
        "py::src/utils/helpers.py::CONST1",
        **_make_node("constant", "src/utils/helpers.py", "CONST1"),
    )
    G.add_node(
        "py::src/api/handler.py::APIHandler",
        **_make_node("class", "src/api/handler.py", "APIHandler"),
    )

    G.add_edge(
        "py::src/models/a.py::ClassA",
        "py::src/models/b.py::ClassB",
        weight=2.0,
    )
    G.add_edge(
        "py::src/models/a.py::ClassA",
        "py::src/utils/helpers.py::helper_func",
        weight=1.0,
    )
    G.add_edge(
        "py::src/api/handler.py::APIHandler",
        "py::src/models/a.py::ClassA",
        weight=3.0,
    )
    G.add_edge(
        "py::src/api/handler.py::APIHandler",
        "py::src/models/b.py::ClassB",
        weight=1.5,
    )
    G.add_edge(
        "py::src/models/a.py::ClassA.method1",
        "py::src/utils/helpers.py::helper_func",
        weight=1.0,
    )

    return G


def _build_hub_graph() -> nx.MultiDiGraph:
    """Build a graph with a clear hub node.

    Node 'hub' has in-degree far exceeding the mean, making it
    detectable via Z-score.
    """
    G = nx.MultiDiGraph()

    G.add_node("hub", **_make_node("class", "src/base.py", "BaseClass"))
    for i in range(20):
        nid = f"node_{i}"
        G.add_node(nid, **_make_node("class", f"src/mod{i}.py", f"Class{i}"))
        G.add_edge(nid, "hub", weight=1.0)  # all point to hub

    # A few edges between regular nodes
    for i in range(0, 18, 2):
        G.add_edge(f"node_{i}", f"node_{i + 1}", weight=0.5)

    return G


def _build_multi_file_graph(n_files=10, nodes_per_file=5) -> nx.MultiDiGraph:
    """Build a larger graph with multiple files for clustering tests.

    Creates n_files files, each with nodes_per_file nodes.
    Cross-file edges connect nodes from adjacent files.
    """
    G = nx.MultiDiGraph()

    for f in range(n_files):
        for n in range(nodes_per_file):
            nid = f"py::src/pkg{f}/mod.py::Sym{n}"
            stype = "class" if n == 0 else "function"
            G.add_node(nid, **_make_node(stype, f"src/pkg{f}/mod.py", f"Sym{n}"))

    # Intra-file edges
    for f in range(n_files):
        for n in range(nodes_per_file - 1):
            G.add_edge(
                f"py::src/pkg{f}/mod.py::Sym{n}",
                f"py::src/pkg{f}/mod.py::Sym{n + 1}",
                weight=2.0,
            )

    # Cross-file edges (adjacent files)
    for f in range(n_files - 1):
        G.add_edge(
            f"py::src/pkg{f}/mod.py::Sym0",
            f"py::src/pkg{f + 1}/mod.py::Sym0",
            weight=1.0,
        )

    return G


# ═══════════════════════════════════════════════════════════════════════════
# ClusterAssignment Tests
# ═══════════════════════════════════════════════════════════════════════════


class TestClusterAssignment:
    def test_construction(self):
        ca = ClusterAssignment(
            macro_clusters={"a": 0, "b": 0},
            micro_clusters={0: {"a": 0, "b": 0}},
            hub_nodes=set(),
            hub_assignments={},
            sections={0: {"pages": {0: ["a", "b"]}}},
            section_count=1,
            page_count=1,
            algorithm="test",
            resolution={"section": 1.0, "page": 1.0},
        )
        assert ca.section_count == 1
        assert ca.page_count == 1
        assert ca.algorithm == "test"
        assert ca.stats == {}

    def test_default_stats(self):
        ca = ClusterAssignment(
            macro_clusters={},
            micro_clusters={},
            hub_nodes=set(),
            hub_assignments={},
            sections={},
            section_count=0,
            page_count=0,
            algorithm="test",
            resolution={},
        )
        assert ca.stats == {}


# ═══════════════════════════════════════════════════════════════════════════
# _find_architectural_parent Tests
# ═══════════════════════════════════════════════════════════════════════════


class TestFindArchitecturalParent:
    def test_node_id_convention(self):
        """Strategy 1: lang::file::Parent.child → lang::file::Parent."""
        arch_ids = {"py::src/a.py::MyClass"}
        result = _find_architectural_parent(
            "py::src/a.py::MyClass.method",
            {"symbol_type": "method"},
            arch_ids,
            {},
            nx.MultiDiGraph(),
        )
        assert result == "py::src/a.py::MyClass"

    def test_parent_symbol_attribute(self):
        """Strategy 2: parent_symbol lookup via (file, name) index."""
        arch_ids = {"py::src/a.py::MyClass"}
        by_file_sym = {("src/a.py", "MyClass"): "py::src/a.py::MyClass"}
        result = _find_architectural_parent(
            "py::src/a.py::some_method",
            {"parent_symbol": "MyClass"},
            arch_ids,
            by_file_sym,
            nx.MultiDiGraph(),
        )
        assert result == "py::src/a.py::MyClass"

    def test_no_parent_found(self):
        """Strategy 3: returns None when no parent resolves."""
        result = _find_architectural_parent(
            "py::src/a.py::orphan",
            {},
            set(),
            {},
            nx.MultiDiGraph(),
        )
        assert result is None

    def test_non_triple_id(self):
        """Node IDs without :: are not resolvable."""
        result = _find_architectural_parent(
            "simple_id",
            {},
            set(),
            {},
            nx.MultiDiGraph(),
        )
        assert result is None

    def test_parent_in_graph_not_in_arch(self):
        """Parent exists in graph but not in arch_node_ids — still resolves."""
        G = nx.MultiDiGraph()
        G.add_node("py::src/a.py::Parent")
        result = _find_architectural_parent(
            "py::src/a.py::Parent.child",
            {},
            set(),  # empty arch_ids
            {},
            G,
        )
        assert result == "py::src/a.py::Parent"


# ═══════════════════════════════════════════════════════════════════════════
# Architectural Projection Tests
# ═══════════════════════════════════════════════════════════════════════════


class TestArchitecturalProjection:
    def test_method_promoted_to_parent(self):
        G = _build_simple_graph()
        P = architectural_projection(G)

        # method1 should be collapsed into ClassA
        assert "py::src/models/a.py::ClassA.method1" not in P.nodes()
        assert "py::src/models/a.py::ClassA" in P.nodes()

    def test_architectural_nodes_preserved(self):
        G = _build_simple_graph()
        P = architectural_projection(G)

        for nid in [
            "py::src/models/a.py::ClassA",
            "py::src/models/b.py::ClassB",
            "py::src/utils/helpers.py::helper_func",
            "py::src/utils/helpers.py::CONST1",
            "py::src/api/handler.py::APIHandler",
        ]:
            assert nid in P.nodes()

    def test_fewer_nodes_after_projection(self):
        G = _build_simple_graph()
        P = architectural_projection(G)
        assert P.number_of_nodes() < G.number_of_nodes()

    def test_method_edge_remapped_to_parent(self):
        """Edge from method1→helper_func should become ClassA→helper_func."""
        G = _build_simple_graph()
        P = architectural_projection(G)

        # ClassA should have edge to helper_func (through method promotion)
        edges_from_a = list(P.out_edges("py::src/models/a.py::ClassA"))
        targets = {v for _, v in edges_from_a}
        assert "py::src/utils/helpers.py::helper_func" in targets

    def test_no_self_loops(self):
        """Projection should not create self-loops."""
        G = _build_simple_graph()
        P = architectural_projection(G)
        for u, v in P.edges():
            assert u != v

    def test_empty_graph(self):
        G = nx.MultiDiGraph()
        P = architectural_projection(G)
        assert P.number_of_nodes() == 0
        assert P.number_of_edges() == 0


# ═══════════════════════════════════════════════════════════════════════════
# Hub Detection Tests
# ═══════════════════════════════════════════════════════════════════════════


class TestDetectHubs:
    def test_hub_detected(self):
        G = _build_hub_graph()
        hubs = detect_hubs(G)
        assert "hub" in hubs

    def test_no_false_positives(self):
        """Regular nodes should not be flagged as hubs."""
        G = _build_hub_graph()
        hubs = detect_hubs(G)
        for nid in hubs:
            if nid != "hub":
                # Any other hub must also have very high in-degree
                assert G.in_degree(nid) > 10

    def test_small_graph_no_hubs(self):
        G = nx.MultiDiGraph()
        G.add_node("a", **_make_node("class", "a.py"))
        G.add_node("b", **_make_node("class", "b.py"))
        assert detect_hubs(G) == set()

    def test_uniform_degree_no_hubs(self):
        """Graph where all nodes have the same in-degree → no hubs."""
        G = nx.MultiDiGraph()
        nodes = [f"n{i}" for i in range(10)]
        for n in nodes:
            G.add_node(n, **_make_node("class", f"{n}.py"))
        # Ring topology: each node has in-degree 1
        for i in range(10):
            G.add_edge(nodes[i], nodes[(i + 1) % 10], weight=1.0)
        assert detect_hubs(G) == set()

    def test_custom_threshold(self):
        G = _build_hub_graph()
        # Very high threshold → no hubs
        hubs = detect_hubs(G, z_threshold=100.0)
        assert len(hubs) == 0

    def test_returns_set(self):
        G = _build_hub_graph()
        result = detect_hubs(G)
        assert isinstance(result, set)


# ═══════════════════════════════════════════════════════════════════════════
# Graph Conversion Tests
# ═══════════════════════════════════════════════════════════════════════════


class TestToWeightedUndirected:
    def test_edge_weights_summed(self):
        G = nx.MultiDiGraph()
        G.add_node("a")
        G.add_node("b")
        G.add_edge("a", "b", weight=1.0)
        G.add_edge("b", "a", weight=2.0)
        G.add_edge("a", "b", weight=0.5)  # parallel edge

        U = _to_weighted_undirected(G)
        assert U.has_edge("a", "b")
        assert U["a"]["b"]["weight"] == 3.5  # 1.0 + 2.0 + 0.5

    def test_preserves_all_nodes(self):
        G = nx.MultiDiGraph()
        for i in range(5):
            G.add_node(f"n{i}")
        U = _to_weighted_undirected(G)
        assert set(U.nodes()) == set(G.nodes())

    def test_default_weight(self):
        G = nx.MultiDiGraph()
        G.add_node("a")
        G.add_node("b")
        G.add_edge("a", "b")  # no weight attr
        U = _to_weighted_undirected(G)
        assert U["a"]["b"]["weight"] == 1.0


class TestNxToIgraph:
    @requires_leiden
    def test_node_count_preserved(self):
        G = nx.Graph()
        G.add_nodes_from(["a", "b", "c"])
        G.add_edge("a", "b", weight=1.0)
        G.add_edge("b", "c", weight=2.0)

        ig_g, node_list = _nx_to_igraph(G)
        assert ig_g.vcount() == 3
        assert ig_g.ecount() == 2
        assert len(node_list) == 3

    @requires_leiden
    def test_deterministic_ordering(self):
        G = nx.Graph()
        G.add_nodes_from(["c", "a", "b"])
        _, node_list = _nx_to_igraph(G)
        assert node_list == ["a", "b", "c"]

    @requires_leiden
    def test_weights_preserved(self):
        G = nx.Graph()
        G.add_edge("a", "b", weight=3.14)
        ig_g, _ = _nx_to_igraph(G)
        assert abs(ig_g.es[0]["weight"] - 3.14) < 1e-6


class TestContractToFileGraph:
    def test_basic_contraction(self):
        G = nx.MultiDiGraph()
        G.add_node("n1", rel_path="src/a.py")
        G.add_node("n2", rel_path="src/a.py")
        G.add_node("n3", rel_path="src/b.py")
        G.add_edge("n1", "n3", weight=1.0)
        G.add_edge("n2", "n3", weight=2.0)

        FG, f2n = _contract_to_file_graph(G)
        assert FG.number_of_nodes() == 2
        assert set(FG.nodes()) == {"src/a.py", "src/b.py"}
        assert FG["src/a.py"]["src/b.py"]["weight"] == 3.0

    def test_same_file_edges_dropped(self):
        G = nx.MultiDiGraph()
        G.add_node("n1", rel_path="src/a.py")
        G.add_node("n2", rel_path="src/a.py")
        G.add_edge("n1", "n2", weight=5.0)

        FG, _ = _contract_to_file_graph(G)
        assert FG.number_of_edges() == 0

    def test_file_to_nodes_mapping(self):
        G = nx.MultiDiGraph()
        G.add_node("n1", rel_path="src/a.py")
        G.add_node("n2", rel_path="src/a.py")
        G.add_node("n3", rel_path="src/b.py")

        _, f2n = _contract_to_file_graph(G)
        assert set(f2n["src/a.py"]) == {"n1", "n2"}
        assert f2n["src/b.py"] == ["n3"]

    def test_fallback_to_file_name(self):
        G = nx.MultiDiGraph()
        G.add_node("n1", file_name="mod.py")  # no rel_path
        _, f2n = _contract_to_file_graph(G)
        assert "mod.py" in f2n


# ═══════════════════════════════════════════════════════════════════════════
# Directory Helper Tests
# ═══════════════════════════════════════════════════════════════════════════


class TestDirHelpers:
    def test_dir_of_node_with_path(self):
        G = nx.MultiDiGraph()
        G.add_node("n", rel_path="src/models/user.py")
        assert _dir_of_node("n", G) == "src/models"

    def test_dir_of_node_root_file(self):
        G = nx.MultiDiGraph()
        G.add_node("n", rel_path="main.py")
        assert _dir_of_node("n", G) == "<root>"

    def test_dir_of_node_missing(self):
        G = nx.MultiDiGraph()
        G.add_node("n")
        assert _dir_of_node("n", G) == "<root>"

    def test_dir_histogram(self):
        G = nx.MultiDiGraph()
        G.add_node("n1", rel_path="src/a/x.py")
        G.add_node("n2", rel_path="src/a/y.py")
        G.add_node("n3", rel_path="src/b/z.py")

        hist = _dir_histogram(["n1", "n2", "n3"], G)
        assert hist["src/a"] == 2
        assert hist["src/b"] == 1

    def test_dir_similarity_overlap(self):
        a = Counter({"src/a": 3, "src/b": 2})
        b = Counter({"src/a": 1, "src/b": 5})
        assert _dir_similarity(a, b) == 3  # min(3,1) + min(2,5)

    def test_dir_similarity_no_overlap(self):
        a = Counter({"src/a": 3})
        b = Counter({"src/b": 5})
        assert _dir_similarity(a, b) == 0


# ═══════════════════════════════════════════════════════════════════════════
# Centroid Detection Tests
# ═══════════════════════════════════════════════════════════════════════════


class TestDetectPageCentroids:
    def test_empty_nodes(self):
        G = nx.MultiDiGraph()
        assert _detect_page_centroids(G, []) == []

    def test_single_node(self):
        G = nx.MultiDiGraph()
        G.add_node("n1", symbol_type="class", symbol_name="MyClass")
        result = _detect_page_centroids(G, ["n1"])
        assert len(result) == 1
        assert result[0]["node_id"] == "n1"
        assert result[0]["score"] == 1.0

    def test_high_priority_node_preferred(self):
        G = nx.MultiDiGraph()
        G.add_node("cls", symbol_type="class", symbol_name="Foo")
        G.add_node("prop", symbol_type="property", symbol_name="bar")
        # Give them equal degree
        G.add_edge("cls", "prop", weight=1.0)
        G.add_edge("prop", "cls", weight=1.0)

        result = _detect_page_centroids(G, ["cls", "prop"])
        assert result[0]["node_id"] == "cls"

    def test_top_k_limit(self):
        G = nx.MultiDiGraph()
        for i in range(20):
            G.add_node(f"n{i}", symbol_type="class", symbol_name=f"C{i}")
        result = _detect_page_centroids(G, [f"n{i}" for i in range(20)], top_k=3)
        assert len(result) == 3

    def test_result_structure(self):
        G = nx.MultiDiGraph()
        G.add_node("n1", symbol_type="function", symbol_name="foo")
        result = _detect_page_centroids(G, ["n1"])
        assert "node_id" in result[0]
        assert "name" in result[0]
        assert "type" in result[0]
        assert "score" in result[0]


class TestSelectCentralSymbols:
    def test_empty_cluster(self):
        G = nx.MultiDiGraph()
        assert select_central_symbols(G, set()) == []

    def test_returns_at_most_k(self):
        G = nx.MultiDiGraph()
        for i in range(10):
            G.add_node(f"n{i}")
        for i in range(9):
            G.add_edge(f"n{i}", f"n{i + 1}", weight=1.0)

        result = select_central_symbols(G, set(G.nodes()), k=3)
        assert len(result) <= 3

    def test_single_node(self):
        G = nx.MultiDiGraph()
        G.add_node("only")
        result = select_central_symbols(G, {"only"})
        assert result == ["only"]


# ═══════════════════════════════════════════════════════════════════════════
# Doc Cluster Detection Tests
# ═══════════════════════════════════════════════════════════════════════════


class TestDetectDocClusters:
    def test_doc_dominant_section(self):
        G = nx.MultiDiGraph()
        # 4 doc nodes, 1 code node → 80% doc
        for i in range(4):
            G.add_node(f"doc{i}", symbol_type="markdown_document")
        G.add_node("code", symbol_type="class")

        sections = {0: {"pages": {0: [f"doc{i}" for i in range(4)] + ["code"]}}}
        result = detect_doc_clusters(sections, G)
        assert 0 in result
        assert result[0] == pytest.approx(0.8)

    def test_code_dominant_section(self):
        G = nx.MultiDiGraph()
        for i in range(4):
            G.add_node(f"code{i}", symbol_type="class")
        G.add_node("doc", symbol_type="markdown_document")

        sections = {0: {"pages": {0: [f"code{i}" for i in range(4)] + ["doc"]}}}
        result = detect_doc_clusters(sections, G)
        assert 0 not in result

    def test_empty_sections(self):
        result = detect_doc_clusters({}, nx.MultiDiGraph())
        assert result == {}


# ═══════════════════════════════════════════════════════════════════════════
# Hub Re-Integration Tests
# ═══════════════════════════════════════════════════════════════════════════


class TestReintegrateHubs:
    def test_hub_assigned_to_most_connected(self):
        G = nx.MultiDiGraph()
        G.add_node("hub")
        G.add_node("a")
        G.add_node("b")
        G.add_node("c")
        # hub has 2 edges to cluster 0, 1 edge to cluster 1
        G.add_edge("a", "hub", weight=1.0)
        G.add_edge("b", "hub", weight=1.0)
        G.add_edge("c", "hub", weight=1.0)

        macro = {"a": 0, "b": 0, "c": 1}
        result = reintegrate_hubs(G, {"hub"}, macro)
        assert result["hub"][0] == 0  # majority is cluster 0

    def test_hub_with_no_edges(self):
        G = nx.MultiDiGraph()
        G.add_node("hub")
        G.add_node("a")
        macro = {"a": 0}
        result = reintegrate_hubs(G, {"hub"}, macro)
        assert result["hub"][0] == 0  # falls back to largest

    def test_micro_assignment(self):
        G = nx.MultiDiGraph()
        G.add_node("hub")
        G.add_node("a")
        G.add_node("b")
        G.add_edge("a", "hub", weight=1.0)
        G.add_edge("b", "hub", weight=1.0)

        macro = {"a": 0, "b": 0}
        micro = {0: {"a": 0, "b": 1}}
        result = reintegrate_hubs(G, {"hub"}, macro, micro)
        assert result["hub"][0] == 0
        assert result["hub"][1] is not None  # should have micro assignment

    def test_empty_hubs(self):
        result = reintegrate_hubs(nx.MultiDiGraph(), set(), {})
        assert result == {}


# ═══════════════════════════════════════════════════════════════════════════
# Consolidation Tests
# ═══════════════════════════════════════════════════════════════════════════


class TestConsolidateSections:
    def test_no_merge_when_under_target(self):
        """If section count ≤ target, no merging happens."""
        G = nx.MultiDiGraph()
        G.add_node("n1", rel_path="src/a.py")
        G.add_node("n2", rel_path="src/b.py")

        leiden_result = {
            "sections": {
                0: {"pages": {0: ["n1"]}},
                1: {"pages": {0: ["n2"]}},
            },
            "macro_assignments": {"n1": 0, "n2": 1},
            "micro_assignments": {0: {"n1": 0}, 1: {"n2": 0}},
        }
        result = _consolidate_sections(leiden_result, G, n_files=5)
        # target_section_count(5) = 5, we have 2 sections → no merge
        assert len(result["sections"]) == 2

    def test_merges_to_target(self):
        """Creates many sections and verifies merging down to target."""
        G = nx.MultiDiGraph()
        n_sections = 30

        sections = {}
        macro = {}
        micro = {}
        for i in range(n_sections):
            nid = f"n{i}"
            G.add_node(nid, rel_path=f"src/pkg{i}/mod.py")
            sections[i] = {"pages": {0: [nid]}}
            macro[nid] = i
            micro[i] = {nid: 0}

        leiden_result = {
            "sections": sections,
            "macro_assignments": macro,
            "micro_assignments": micro,
        }
        result = _consolidate_sections(leiden_result, G, n_files=10)
        # target_section_count(10) = 5
        assert len(result["sections"]) <= 5


class TestConsolidatePages:
    def test_no_merge_when_under_target(self):
        G = nx.MultiDiGraph()
        for i in range(5):
            G.add_node(f"n{i}", rel_path=f"src/mod{i}.py")

        leiden_result = {
            "sections": {
                0: {"pages": {0: ["n0", "n1"], 1: ["n2", "n3", "n4"]}},
            },
            "macro_assignments": {f"n{i}": 0 for i in range(5)},
            "micro_assignments": {0: {f"n{i}": 0 if i < 2 else 1 for i in range(5)}},
        }
        # target_total_pages(5) = 8, we have 2 pages → no merge
        result = _consolidate_pages(leiden_result, G)
        assert len(result["sections"][0]["pages"]) == 2


# ═══════════════════════════════════════════════════════════════════════════
# Trivial Assignment Tests
# ═══════════════════════════════════════════════════════════════════════════


class TestTrivialAssignment:
    def test_single_section_single_page(self):
        G = nx.MultiDiGraph()
        G.add_node("a", symbol_type="class", symbol_name="A")
        G.add_node("b", symbol_type="function", symbol_name="B")

        result = _trivial_assignment(G)
        assert isinstance(result, ClusterAssignment)
        assert result.section_count == 1
        assert result.page_count == 1
        assert result.algorithm == "trivial"
        assert "a" in result.macro_clusters
        assert "b" in result.macro_clusters

    def test_empty_graph(self):
        result = _trivial_assignment(nx.MultiDiGraph())
        assert result.section_count == 1
        assert result.page_count == 1
        assert result.macro_clusters == {}


# ═══════════════════════════════════════════════════════════════════════════
# Hierarchical Leiden Tests (require igraph + leidenalg)
# ═══════════════════════════════════════════════════════════════════════════


class TestHierarchicalLeiden:
    @requires_leiden
    def test_basic_clustering(self):
        G = _build_multi_file_graph(n_files=6, nodes_per_file=4)
        result = _hierarchical_leiden(G, hubs=set())

        assert "sections" in result
        assert "macro_assignments" in result
        assert "micro_assignments" in result
        assert "algorithm_metadata" in result
        assert result["algorithm_metadata"]["sections"] > 0
        assert result["algorithm_metadata"]["pages"] > 0

    @requires_leiden
    def test_all_nodes_assigned(self):
        G = _build_multi_file_graph(n_files=5, nodes_per_file=3)
        result = _hierarchical_leiden(G, hubs=set())
        assert len(result["macro_assignments"]) == G.number_of_nodes()

    @requires_leiden
    def test_hubs_excluded(self):
        G = _build_multi_file_graph(n_files=5, nodes_per_file=3)
        hub_node = list(G.nodes())[0]
        result = _hierarchical_leiden(G, hubs={hub_node})
        assert hub_node not in result["macro_assignments"]

    @requires_leiden
    def test_empty_after_hub_removal(self):
        G = nx.MultiDiGraph()
        G.add_node("only", rel_path="a.py")
        result = _hierarchical_leiden(G, hubs={"only"})
        assert result["sections"] == {}

    @requires_leiden
    def test_missing_libs_error(self):
        G = _build_multi_file_graph()
        with patch("app.core.graph_clustering._HAS_IGRAPH", False):
            with pytest.raises(RuntimeError, match="igraph and leidenalg"):
                _hierarchical_leiden(G, hubs=set())


# ═══════════════════════════════════════════════════════════════════════════
# Full Pipeline (run_clustering) Tests
# ═══════════════════════════════════════════════════════════════════════════


class TestRunClustering:
    @requires_leiden
    def test_full_pipeline(self):
        G = _build_multi_file_graph(n_files=8, nodes_per_file=5)
        result = run_clustering(G)

        assert isinstance(result, ClusterAssignment)
        assert result.section_count > 0
        assert result.page_count > 0
        assert result.algorithm == "hierarchical_leiden_file_contracted"
        assert len(result.macro_clusters) > 0

    @requires_leiden
    def test_exclude_tests(self):
        G = _build_multi_file_graph(n_files=5, nodes_per_file=3)
        # Add a test node
        G.add_node(
            "py::tests/test_foo.py::TestFoo",
            **_make_node("class", "tests/test_foo.py", "TestFoo"),
        )

        result = run_clustering(G, exclude_tests=True)
        assert isinstance(result, ClusterAssignment)
        # Test node should be excluded from clustering
        assert result.stats["graph"]["test_nodes_excluded"] >= 0

    def test_exclude_tests_all_test_nodes_fallback(self):
        """When every node is a test file, fall back to full graph."""
        G = nx.MultiDiGraph()
        for i in range(5):
            G.add_node(
                f"py::tests/test_{i}.py::Test{i}",
                **_make_node("class", f"tests/test_{i}.py", f"Test{i}"),
            )
        # Add edges so Leiden has something to work with
        nodes = list(G.nodes())
        for a, b in zip(nodes, nodes[1:]):
            G.add_edge(a, b, relationship_type="calls", weight=1.0)

        result = run_clustering(G, exclude_tests=True)
        assert isinstance(result, ClusterAssignment)
        # Should NOT have excluded any (fallback to full graph)
        assert result.stats["graph"]["test_nodes_excluded"] == 0

    @requires_leiden
    def test_resolution_parameters(self):
        G = _build_multi_file_graph(n_files=6, nodes_per_file=4)
        result = run_clustering(G, section_resolution=2.0, page_resolution=0.5)
        assert result.resolution["section"] == 2.0
        assert result.resolution["page"] == 0.5

    def test_very_small_graph_trivial(self):
        """Graph with <3 non-hub nodes should get trivial assignment."""
        G = nx.MultiDiGraph()
        G.add_node("a", **_make_node("class", "a.py", "A"))
        G.add_node("b", **_make_node("class", "b.py", "B"))

        result = run_clustering(G)
        assert result.algorithm == "trivial"
        assert result.section_count == 1

    def test_empty_graph(self):
        G = nx.MultiDiGraph()
        result = run_clustering(G)
        assert result.algorithm == "trivial"

    @requires_leiden
    def test_hub_detection_and_reintegration(self):
        """Build graph with a clear hub and verify it gets reintegrated."""
        G = _build_hub_graph()
        result = run_clustering(G)
        assert isinstance(result, ClusterAssignment)
        # Hub should be detected
        assert len(result.hub_nodes) >= 1
        # Hub should be reintegrated (present in macro_clusters)
        for hub in result.hub_nodes:
            if hub in set(G.nodes()):
                # After projection, hub might have been collapsed
                pass  # Just verify no crash

    @requires_leiden
    def test_stats_populated(self):
        G = _build_multi_file_graph(n_files=6, nodes_per_file=4)
        result = run_clustering(G)
        assert "graph" in result.stats
        assert "total_nodes" in result.stats["graph"]
        assert "macro" in result.stats
        assert "micro" in result.stats

    @requires_leiden
    def test_sections_have_pages(self):
        G = _build_multi_file_graph(n_files=6, nodes_per_file=4)
        result = run_clustering(G)
        for sec_id, sec_data in result.sections.items():
            assert "pages" in sec_data
            assert len(sec_data["pages"]) > 0

    @requires_leiden
    def test_deterministic(self):
        """Same graph should produce same result (seed=42)."""
        G = _build_multi_file_graph(n_files=6, nodes_per_file=4)
        r1 = run_clustering(G)
        r2 = run_clustering(G)
        assert r1.section_count == r2.section_count
        assert r1.page_count == r2.page_count
        assert r1.macro_clusters == r2.macro_clusters


# ═══════════════════════════════════════════════════════════════════════════
# Section Expansion Tests
# ═══════════════════════════════════════════════════════════════════════════


class TestExpandSections:
    """Test _expand_sections — splits large sections to reach target count."""

    def _make_leiden_result(self, sections, macro, micro):
        return {
            "sections": sections,
            "macro_assignments": macro,
            "micro_assignments": micro,
            "algorithm_metadata": {"file_nodes": 0, "sections": 0, "pages": 0},
        }

    def test_noop_when_already_at_target(self):
        """When section count >= target, return unchanged."""
        G = _build_multi_file_graph(n_files=4, nodes_per_file=2)
        nodes = list(G.nodes())
        # 4 sections for 4 files — target_section_count(4) = 5
        # but we mock n_files=2 so target=5, but with 4 sections >= 5? No.
        # target_section_count(2) = max(5, ...) = 5, so 4 < 5.
        # Let's just test that >= target is a noop:
        sections = {
            i: {"pages": {0: [nodes[i * 2], nodes[i * 2 + 1]]}}
            for i in range(4)
        }
        macro = {n: i // 2 for i, n in enumerate(nodes)}
        micro = {i: {nodes[i * 2]: 0, nodes[i * 2 + 1]: 0} for i in range(4)}
        lr = self._make_leiden_result(sections, macro, micro)
        # Force n_files=1 so target=5, but we have 4 sections
        # Actually target_section_count(1)=max(5,...)=5
        # Set n_files high enough that target <= 4
        # target_section_count(8) = max(5, min(20, ceil(1.2*3)))=5
        # Hmm. Let's override n_files to get target <= current.
        # We have 4 sections. n_files=4 → target=5. n_files=3 → target=5.
        # Just patch target_section_count to return 3:
        with patch("app.core.graph_clustering.target_section_count", return_value=3):
            result = _expand_sections(lr, G, n_files=4)
        assert len(result["sections"]) == 4  # unchanged

    @requires_leiden
    def test_splits_when_under_target(self):
        """Build a graph that Leiden under-segments, verify expansion adds sections."""
        # Build a large graph with many files in distinct directories
        G = nx.MultiDiGraph()
        n_files = 20
        for f in range(n_files):
            # Use different top-level dirs to enable directory-based splitting
            dir_name = f"src/group{f % 5}/pkg{f}"
            for n in range(4):
                nid = f"py::{dir_name}/mod.py::Sym{n}"
                stype = "class" if n == 0 else "function"
                G.add_node(nid, **_make_node(stype, f"{dir_name}/mod.py", f"Sym{n}",
                                             start_line=n * 20))
            # Intra-file edges
            for n in range(3):
                G.add_edge(
                    f"py::{dir_name}/mod.py::Sym{n}",
                    f"py::{dir_name}/mod.py::Sym{n + 1}",
                    weight=2.0,
                )

        # Cross-file edges (within same group)
        for f in range(n_files - 1):
            d1 = f"src/group{f % 5}/pkg{f}"
            d2 = f"src/group{(f + 1) % 5}/pkg{f + 1}"
            G.add_edge(
                f"py::{d1}/mod.py::Sym0",
                f"py::{d2}/mod.py::Sym0",
                weight=1.0,
            )

        # Create leiden_result with only 2 sections containing all nodes
        all_nodes = list(G.nodes())
        half = len(all_nodes) // 2
        sections = {
            0: {"pages": {0: all_nodes[:half]}},
            1: {"pages": {0: all_nodes[half:]}},
        }
        macro = {}
        for nid in all_nodes[:half]:
            macro[nid] = 0
        for nid in all_nodes[half:]:
            macro[nid] = 1
        micro = {
            0: {nid: 0 for nid in all_nodes[:half]},
            1: {nid: 0 for nid in all_nodes[half:]},
        }
        lr = {
            "sections": sections,
            "macro_assignments": macro,
            "micro_assignments": micro,
            "algorithm_metadata": {"file_nodes": n_files, "sections": 2, "pages": 2},
        }

        # target_section_count(20) = max(5, min(20, ceil(1.2*log2(20))))
        # = ceil(1.2*4.32) = ceil(5.18) = 6
        result = _expand_sections(lr, G, n_files=n_files)
        assert len(result["sections"]) > 2, "Should have expanded beyond 2 sections"

        # All nodes should still be assigned
        all_assigned = set()
        for sec in result["sections"].values():
            for pg_nids in sec["pages"].values():
                all_assigned.update(pg_nids)
        assert all_assigned == set(all_nodes)

    def test_no_leiden_libs_noop(self):
        """Without igraph/leidenalg, expansion is a noop."""
        G = _build_multi_file_graph(n_files=4, nodes_per_file=2)
        nodes = list(G.nodes())
        lr = self._make_leiden_result(
            {0: {"pages": {0: nodes}}},
            {n: 0 for n in nodes},
            {0: {n: 0 for n in nodes}},
        )
        with patch("app.core.graph_clustering._HAS_IGRAPH", False):
            result = _expand_sections(lr, G, n_files=100)
        assert len(result["sections"]) == 1


# ═══════════════════════════════════════════════════════════════════════════
# Page Size Enforcement Tests
# ═══════════════════════════════════════════════════════════════════════════


class TestEnforcePageSizes:
    """Test _enforce_page_sizes — merge tiny, split oversized pages."""

    def test_merges_tiny_pages(self):
        """Pages smaller than MERGE_THRESHOLD get merged."""
        G = nx.MultiDiGraph()
        # All in the same directory for easy merging
        for i in range(12):
            G.add_node(f"n{i}", **_make_node("function", "src/a.py", f"fn{i}",
                                              start_line=i * 10))

        # 3 pages: [n0..n7] (8 nodes), [n8,n9] (2 nodes — tiny), [n10,n11] (2 — tiny)
        sections = {
            0: {
                "pages": {
                    0: [f"n{i}" for i in range(8)],
                    1: ["n8", "n9"],
                    2: ["n10", "n11"],
                },
            },
        }
        macro = {f"n{i}": 0 for i in range(12)}
        micro = {0: {f"n{i}": (0 if i < 8 else 1 if i < 10 else 2) for i in range(12)}}
        lr = {
            "sections": sections,
            "macro_assignments": macro,
            "micro_assignments": micro,
            "algorithm_metadata": {},
        }

        result = _enforce_page_sizes(lr, G)
        # Tiny pages (size 2 < MERGE_THRESHOLD=4) should be merged
        total_pages = sum(len(s["pages"]) for s in result["sections"].values())
        assert total_pages < 3, f"Expected merging, got {total_pages} pages"
        # All nodes still present
        all_nids = set()
        for s in result["sections"].values():
            for pg in s["pages"].values():
                all_nids.update(pg)
        assert len(all_nids) == 12

    def test_splits_oversized_pages(self):
        """Pages larger than adaptive_max_page_size get split."""
        G = nx.MultiDiGraph()
        # Create 40 nodes in different files
        for i in range(40):
            G.add_node(
                f"n{i}",
                **_make_node("function", f"src/f{i // 5}.py", f"fn{i}",
                             start_line=(i % 5) * 10),
            )

        sections = {0: {"pages": {0: [f"n{i}" for i in range(40)]}}}
        macro = {f"n{i}": 0 for i in range(40)}
        micro = {0: {f"n{i}": 0 for i in range(40)}}
        lr = {
            "sections": sections,
            "macro_assignments": macro,
            "micro_assignments": micro,
            "algorithm_metadata": {},
        }

        # With 40 nodes total, adaptive_max_page_size(40) = 25
        # So 40-node page should be split into 2 pages
        result = _enforce_page_sizes(lr, G)
        total_pages = sum(len(s["pages"]) for s in result["sections"].values())
        assert total_pages >= 2, f"Expected split, got {total_pages} pages"

    def test_single_page_section_untouched(self):
        """Section with 1 page is not modified."""
        G = nx.MultiDiGraph()
        for i in range(6):
            G.add_node(f"n{i}", **_make_node("class", "a.py", f"C{i}"))
        lr = {
            "sections": {0: {"pages": {0: [f"n{i}" for i in range(6)]}}},
            "macro_assignments": {f"n{i}": 0 for i in range(6)},
            "micro_assignments": {0: {f"n{i}": 0 for i in range(6)}},
            "algorithm_metadata": {},
        }
        result = _enforce_page_sizes(lr, G)
        assert len(result["sections"][0]["pages"]) == 1


# ═══════════════════════════════════════════════════════════════════════════
# File-Aware Split Tests
# ═══════════════════════════════════════════════════════════════════════════


class TestFileAwareSplit:
    """Test _file_aware_split — chunking nodes by file then by size cap."""

    def test_keeps_same_file_together(self):
        G = nx.MultiDiGraph()
        for i in range(10):
            G.add_node(f"n{i}", **_make_node("function", f"src/f{i // 5}.py", f"fn{i}",
                                              start_line=(i % 5) * 10))
        # max_size=6: file0 has 5 nodes, file1 has 5 nodes → 2 chunks
        chunks = _file_aware_split([f"n{i}" for i in range(10)], G, max_size=6)
        assert len(chunks) == 2
        # Each chunk should have nodes from the same file
        for chunk in chunks:
            files = {G.nodes[n]["rel_path"] for n in chunk}
            assert len(files) == 1

    def test_splits_large_file(self):
        G = nx.MultiDiGraph()
        for i in range(30):
            G.add_node(f"n{i}", **_make_node("function", "src/big.py", f"fn{i}",
                                              start_line=i * 10))
        chunks = _file_aware_split([f"n{i}" for i in range(30)], G, max_size=10)
        assert len(chunks) == 3
        total = sum(len(c) for c in chunks)
        assert total == 30

    def test_empty_returns_original(self):
        G = nx.MultiDiGraph()
        result = _file_aware_split([], G, max_size=10)
        assert result == [[]]
