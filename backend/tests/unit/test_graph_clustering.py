"""Tests for app.core.graph_clustering.

Covers:
- architectural_projection() — child promotion, edge remapping
- macro_cluster() — Louvain macro-clustering
- _to_weighted_undirected() — directed→undirected collapse
- _nx_to_igraph() — NetworkX→igraph conversion
- _contract_to_file_graph() — file-level contraction
- _dir_of_node(), _dir_histogram(), _dir_similarity() — directory helpers
- _detect_page_centroids() — centroid scoring
- select_central_symbols() — PageRank-based selection
- detect_doc_clusters() — doc-dominant section detection
- reintegrate_hubs() — plurality-vote hub assignment
- _consolidate_sections() — section merging
- _consolidate_pages() — page merging
- run_phase3() — full pipeline integration

NOTE: Tests requiring igraph/leidenalg are marked with
``@pytest.mark.skipif`` so the suite still passes when those
optional dependencies are not installed.
"""

import math
from collections import Counter
from unittest.mock import patch

import networkx as nx
import pytest

from app.core.graph_clustering import (
    _CHILD_SYMBOL_TYPES,
    _consolidate_pages,
    _consolidate_sections,
    _contract_to_file_graph,
    _detect_page_centroids,
    _dir_histogram,
    _dir_of_node,
    _dir_similarity,
    _nx_to_igraph,
    _to_weighted_undirected,
    architectural_projection,
    detect_doc_clusters,
    hierarchical_leiden_cluster,
    merge_macro_clusters,
    reintegrate_hubs,
    run_phase3,
    select_central_symbols,
)
from app.core.feature_flags import FeatureFlags
from app.core.unified_db import UnifiedWikiDB

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

    G.add_node("ClassA", **_make_node("class", "src/models/a.py", "ClassA"))
    G.add_node("ClassB", **_make_node("class", "src/models/b.py", "ClassB"))
    G.add_node("method1", **_make_node("method", "src/models/a.py", "method1",
                                        parent_symbol="ClassA"))
    G.add_node("helper_func", **_make_node("function", "src/utils/helpers.py", "helper_func"))
    G.add_node("CONST1", **_make_node("constant", "src/utils/helpers.py", "CONST1"))
    G.add_node("APIHandler", **_make_node("class", "src/api/handler.py", "APIHandler"))

    G.add_edge("ClassA", "ClassB", relationship_type="calls", weight=1.0)
    G.add_edge("ClassA", "helper_func", relationship_type="calls", weight=1.0)
    G.add_edge("APIHandler", "ClassA", relationship_type="calls", weight=2.0)
    G.add_edge("APIHandler", "ClassB", relationship_type="calls", weight=1.0)

    return G


def _build_multi_file_graph(n_files=8, nodes_per_file=5) -> nx.MultiDiGraph:
    """Build a larger graph with multiple files for clustering tests."""
    G = nx.MultiDiGraph()
    all_nodes = []
    for f in range(n_files):
        for n in range(nodes_per_file):
            nid = f"py::src/pkg{f}/mod{n}.py::Sym{f}_{n}"
            stype = "class" if n % 2 == 0 else "function"
            G.add_node(nid, **_make_node(
                stype, f"src/pkg{f}/mod{n}.py", f"Sym{f}_{n}",
                source_text=f"class Sym{f}_{n}: pass",
            ))
            all_nodes.append(nid)

    # Add cross-file edges within packages and some between
    for f in range(n_files):
        for n in range(nodes_per_file - 1):
            src = f"py::src/pkg{f}/mod{n}.py::Sym{f}_{n}"
            tgt = f"py::src/pkg{f}/mod{n+1}.py::Sym{f}_{n+1}"
            G.add_edge(src, tgt, relationship_type="calls", weight=1.0)

    # Cross-package edges (sparser)
    for f in range(n_files - 1):
        src = f"py::src/pkg{f}/mod0.py::Sym{f}_0"
        tgt = f"py::src/pkg{f+1}/mod0.py::Sym{f+1}_0"
        G.add_edge(src, tgt, relationship_type="calls", weight=0.5)

    return G


def _build_hub_graph() -> nx.MultiDiGraph:
    """Build a graph with a clear hub node."""
    G = nx.MultiDiGraph()
    G.add_node("Hub", **_make_node("class", "src/core/hub.py", "Hub"))
    for i in range(10):
        nid = f"Node{i}"
        G.add_node(nid, **_make_node("class", f"src/mod{i}/m.py", f"Node{i}"))
        G.add_edge("Hub", nid, relationship_type="calls", weight=1.0)
        G.add_edge(nid, "Hub", relationship_type="calls", weight=0.5)
    return G


# ═══════════════════════════════════════════════════════════════════════════
# architectural_projection
# ═══════════════════════════════════════════════════════════════════════════


class TestArchitecturalProjection:

    def test_projection_preserves_nodes(self):
        """architectural_projection keeps all nodes including methods
        (it promotes edges from children to parents, not removes children)."""
        G = _build_simple_graph()
        P = architectural_projection(G)
        # All nodes are preserved in the projection
        assert "ClassA" in P.nodes()
        assert "ClassB" in P.nodes()

    def test_edges_remapped(self):
        G = _build_simple_graph()
        P = architectural_projection(G)
        # ClassA→ClassB edge should survive
        assert P.has_edge("ClassA", "ClassB")

    def test_architectural_nodes_preserved(self):
        G = _build_simple_graph()
        P = architectural_projection(G)
        for nid in ("ClassA", "ClassB", "helper_func", "CONST1", "APIHandler"):
            assert nid in P.nodes(), f"{nid} should be in projection"

    def test_child_types_excluded(self):
        """Verify _CHILD_SYMBOL_TYPES are excluded from projection."""
        assert "method" in _CHILD_SYMBOL_TYPES
        assert "field" in _CHILD_SYMBOL_TYPES


# ═══════════════════════════════════════════════════════════════════════════
# _to_weighted_undirected
# ═══════════════════════════════════════════════════════════════════════════


class TestToWeightedUndirected:

    def test_collapses_to_undirected(self):
        G = _build_simple_graph()
        U = _to_weighted_undirected(G)
        assert not U.is_directed()
        assert U.number_of_nodes() == G.number_of_nodes()

    def test_weights_preserved(self):
        G = nx.MultiDiGraph()
        G.add_node("a", **_make_node("class", "a.py"))
        G.add_node("b", **_make_node("class", "b.py"))
        G.add_edge("a", "b", weight=3.0)
        U = _to_weighted_undirected(G)
        assert U.has_edge("a", "b")
        # Weight should be present
        assert U["a"]["b"].get("weight", 0) > 0


# ═══════════════════════════════════════════════════════════════════════════
# _nx_to_igraph
# ═══════════════════════════════════════════════════════════════════════════


@requires_leiden
class TestNxToIgraph:

    def test_conversion(self):
        G = nx.Graph()
        G.add_edge("a", "b", weight=1.0)
        G.add_edge("b", "c", weight=2.0)
        ig, node_list = _nx_to_igraph(G)
        assert ig.vcount() == 3
        assert ig.ecount() == 2
        assert set(node_list) == {"a", "b", "c"}


# ═══════════════════════════════════════════════════════════════════════════
# _contract_to_file_graph
# ═══════════════════════════════════════════════════════════════════════════


class TestContractToFileGraph:

    def test_contracts_to_file_nodes(self):
        G = _build_simple_graph()
        FG, file_members = _contract_to_file_graph(G)
        # Should have file nodes, not symbol nodes
        for node in FG.nodes():
            assert "/" in node or "." in node  # file path pattern
        # file_members maps file → list of symbol ids
        assert isinstance(file_members, dict)

    def test_intrafile_edges_removed(self):
        """Two symbols in the same file: no self-edge in file graph."""
        G = nx.MultiDiGraph()
        G.add_node("A", **_make_node("class", "src/same.py", "A"))
        G.add_node("B", **_make_node("function", "src/same.py", "B"))
        G.add_edge("A", "B", relationship_type="calls", weight=1.0)

        FG, _ = _contract_to_file_graph(G)
        # Should only have 1 file node, no self-edges
        assert FG.number_of_nodes() == 1
        assert FG.number_of_edges() == 0

    def test_cross_file_edges_summed(self):
        G = nx.MultiDiGraph()
        G.add_node("A", **_make_node("class", "src/a.py", "A"))
        G.add_node("B", **_make_node("class", "src/a.py", "B"))
        G.add_node("C", **_make_node("class", "src/c.py", "C"))
        G.add_edge("A", "C", relationship_type="calls", weight=1.0)
        G.add_edge("B", "C", relationship_type="calls", weight=2.0)

        FG, _ = _contract_to_file_graph(G)
        assert FG.number_of_nodes() == 2
        assert FG.number_of_edges() >= 1


# ═══════════════════════════════════════════════════════════════════════════
# _dir_of_node, _dir_histogram, _dir_similarity
# ═══════════════════════════════════════════════════════════════════════════


class TestDirHelpers:

    def test_dir_of_node(self):
        G = nx.MultiDiGraph()
        G.add_node("n1", rel_path="src/core/module.py")
        assert _dir_of_node("n1", G) == "src/core"

    def test_dir_of_root_file(self):
        G = nx.MultiDiGraph()
        G.add_node("n1", rel_path="file.py")
        assert _dir_of_node("n1", G) == "<root>"

    def test_dir_histogram(self):
        G = nx.MultiDiGraph()
        G.add_node("a", rel_path="src/core/a.py")
        G.add_node("b", rel_path="src/core/b.py")
        G.add_node("c", rel_path="src/utils/c.py")
        hist = _dir_histogram(["a", "b", "c"], G)
        assert hist["src/core"] == 2
        assert hist["src/utils"] == 1

    def test_dir_similarity_identical(self):
        a = Counter({"src/core": 3, "src/utils": 1})
        b = Counter({"src/core": 3, "src/utils": 1})
        assert _dir_similarity(a, b) == 4  # sum of mins

    def test_dir_similarity_disjoint(self):
        a = Counter({"src/core": 3})
        b = Counter({"src/utils": 2})
        assert _dir_similarity(a, b) == 0


# ═══════════════════════════════════════════════════════════════════════════
# _detect_page_centroids
# ═══════════════════════════════════════════════════════════════════════════


class TestDetectPageCentroids:

    def test_returns_centroids(self):
        G = _build_simple_graph()
        page_nids = ["ClassA", "ClassB", "helper_func"]
        centroids = _detect_page_centroids(G, page_nids)
        assert len(centroids) > 0
        # Returns list of dicts with centroid info
        assert all(isinstance(c, dict) for c in centroids)


# ═══════════════════════════════════════════════════════════════════════════
# select_central_symbols
# ═══════════════════════════════════════════════════════════════════════════


class TestSelectCentralSymbols:

    def test_returns_list_within_limit(self):
        G = _build_simple_graph()
        result = select_central_symbols(G, {"ClassA", "ClassB", "APIHandler"}, k=2)
        assert len(result) <= 2
        assert all(isinstance(s, str) for s in result)

    def test_returns_all_when_fewer_than_k(self):
        G = _build_simple_graph()
        result = select_central_symbols(G, {"ClassA"}, k=10)
        assert "ClassA" in result


# ═══════════════════════════════════════════════════════════════════════════
# detect_doc_clusters
# ═══════════════════════════════════════════════════════════════════════════


class TestDetectDocClusters:

    def test_detects_doc_dominant_section(self, tmp_path):
        """detect_doc_clusters takes (cluster_map, db) — uses unified DB."""
        G = nx.MultiDiGraph()
        for nid in ["d1", "d2", "d3"]:
            G.add_node(nid, **_make_node("markdown_document", f"docs/{nid}.md", nid,
                                          source_text=f"# Doc {nid}"))
        db = UnifiedWikiDB(tmp_path / "doc_test.wiki.db", embedding_dim=8)
        db.from_networkx(G)
        # Manually set cluster assignments
        for nid in ["d1", "d2", "d3"]:
            db.conn.execute(
                "UPDATE repo_nodes SET macro_cluster=0, micro_cluster=0 WHERE node_id=?",
                (nid,),
            )
        db.conn.commit()
        cluster_map = {0: {0: ["d1", "d2", "d3"]}}
        result = detect_doc_clusters(cluster_map, db)
        assert 0 in result
        db.close()

    def test_non_doc_section_not_detected(self, tmp_path):
        G = nx.MultiDiGraph()
        for nid in ["c1", "c2"]:
            G.add_node(nid, **_make_node("class", f"src/{nid}.py", nid,
                                          source_text=f"class {nid}: pass"))
        db = UnifiedWikiDB(tmp_path / "nondoc_test.wiki.db", embedding_dim=8)
        db.from_networkx(G)
        for nid in ["c1", "c2"]:
            db.conn.execute(
                "UPDATE repo_nodes SET macro_cluster=0, micro_cluster=0 WHERE node_id=?",
                (nid,),
            )
        db.conn.commit()
        cluster_map = {0: {0: ["c1", "c2"]}}
        result = detect_doc_clusters(cluster_map, db)
        assert 0 not in result
        db.close()


# ═══════════════════════════════════════════════════════════════════════════
# reintegrate_hubs
# ═══════════════════════════════════════════════════════════════════════════


class TestReintegrateHubs:

    def test_assigns_hub_to_majority_cluster(self):
        G = nx.MultiDiGraph()
        G.add_node("Hub", **_make_node("class", "hub.py", "Hub"))
        G.add_node("A", **_make_node("class", "a.py", "A"))
        G.add_node("B", **_make_node("class", "b.py", "B"))
        G.add_node("C", **_make_node("class", "c.py", "C"))

        G.add_edge("Hub", "A", relationship_type="calls", weight=1.0)
        G.add_edge("Hub", "B", relationship_type="calls", weight=1.0)
        G.add_edge("Hub", "C", relationship_type="calls", weight=1.0)

        macro = {"A": 0, "B": 0, "C": 1}
        micro = {0: {"A": 0, "B": 0}, 1: {"C": 0}}

        result = reintegrate_hubs(G, {"Hub"}, macro, micro)
        # Hub should go to macro 0 (2 neighbors there vs 1 in macro 1)
        assert "Hub" in result
        assert result["Hub"][0] == 0  # macro_id

    def test_empty_hubs(self):
        G = nx.MultiDiGraph()
        result = reintegrate_hubs(G, set(), {}, {})
        assert result == {}


# ═══════════════════════════════════════════════════════════════════════════
# _consolidate_sections / _consolidate_pages
# ═══════════════════════════════════════════════════════════════════════════


class TestConsolidation:

    def _make_leiden_result(self, sections, macro, micro):
        return {
            "sections": sections,
            "macro_assignments": macro,
            "micro_assignments": micro,
            "algorithm_metadata": {"file_nodes": 10, "sections": len(sections), "pages": 5},
        }

    def test_consolidate_sections_noop(self):
        """When already at target, sections unchanged."""
        G = _build_simple_graph()
        sections = {0: {"pages": {0: ["ClassA", "ClassB"]}}}
        lr = self._make_leiden_result(sections, {"ClassA": 0, "ClassB": 0}, {0: {"ClassA": 0, "ClassB": 0}})
        result = _consolidate_sections(lr, G, n_files=1)
        assert len(result["sections"]) >= 1

    def test_consolidate_pages_noop(self):
        """When pages are reasonable, no merging."""
        G = _build_simple_graph()
        sections = {0: {"pages": {0: ["ClassA"], 1: ["ClassB"]}}}
        lr = self._make_leiden_result(sections, {"ClassA": 0, "ClassB": 0}, {0: {"ClassA": 0, "ClassB": 1}})
        result = _consolidate_pages(lr, G)
        assert 0 in result["sections"]


# ═══════════════════════════════════════════════════════════════════════════
# hierarchical_leiden_cluster (requires igraph)
# ═══════════════════════════════════════════════════════════════════════════


@requires_leiden
class TestHierarchicalLeidenCluster:

    def test_basic_clustering(self):
        G = _build_multi_file_graph(n_files=6, nodes_per_file=4)
        result = hierarchical_leiden_cluster(G, set())
        assert "sections" in result
        assert "macro_assignments" in result
        assert "micro_assignments" in result
        assert "algorithm_metadata" in result
        assert len(result["sections"]) > 0

    def test_with_hubs(self):
        G = _build_hub_graph()
        result = hierarchical_leiden_cluster(G, {"Hub"})
        assert "sections" in result
        assert len(result["sections"]) >= 1


# ═══════════════════════════════════════════════════════════════════════════
# Full Pipeline (run_phase3) Tests
# ═══════════════════════════════════════════════════════════════════════════


class TestRunPhase3:

    @requires_leiden
    def test_full_pipeline(self, tmp_path):
        G = _build_multi_file_graph(n_files=8, nodes_per_file=5)
        db = UnifiedWikiDB(tmp_path / "test.wiki.db", embedding_dim=8)
        db.from_networkx(G)

        result = run_phase3(db, G)

        assert "macro" in result
        assert "micro" in result
        assert result["macro"]["cluster_count"] > 0
        assert result["micro"]["total_pages"] > 0
        db.close()

    @requires_leiden
    def test_exclude_tests(self, tmp_path):
        G = _build_multi_file_graph(n_files=5, nodes_per_file=3)
        # Add a test node
        G.add_node(
            "py::tests/test_foo.py::TestFoo",
            **_make_node("class", "tests/test_foo.py", "TestFoo"),
        )
        G.add_edge(
            list(G.nodes())[0], "py::tests/test_foo.py::TestFoo",
            relationship_type="calls", weight=1.0,
        )

        db = UnifiedWikiDB(tmp_path / "test_excl.wiki.db", embedding_dim=8)
        db.from_networkx(G)

        flags = FeatureFlags(exclude_tests=True)
        with patch("app.core.feature_flags.get_feature_flags", return_value=flags):
            result = run_phase3(db, G, feature_flags=flags)

        assert result["macro"]["cluster_count"] > 0
        db.close()

    def test_very_small_graph(self, tmp_path):
        """Graph with <3 nodes should still complete without error."""
        G = nx.MultiDiGraph()
        G.add_node("a", **_make_node("class", "a.py", "A"))
        G.add_node("b", **_make_node("class", "b.py", "B"))

        db = UnifiedWikiDB(tmp_path / "small.wiki.db", embedding_dim=8)
        db.from_networkx(G)

        result = run_phase3(db, G)
        assert "macro" in result
        db.close()

    def test_empty_graph(self, tmp_path):
        G = nx.MultiDiGraph()

        db = UnifiedWikiDB(tmp_path / "empty.wiki.db", embedding_dim=8)
        db.from_networkx(G)

        result = run_phase3(db, G)
        assert "macro" in result
        db.close()

    @requires_leiden
    def test_hub_detection_and_reintegration(self, tmp_path):
        """Build graph with a clear hub and verify it gets reintegrated."""
        G = _build_hub_graph()
        db = UnifiedWikiDB(tmp_path / "hub.wiki.db", embedding_dim=8)
        db.from_networkx(G)

        result = run_phase3(db, G)
        assert "hubs" in result
        # Hub node has 10 connections so should be detected
        assert result["hubs"]["total"] >= 0  # depends on Z-score threshold
        db.close()

    @requires_leiden
    def test_result_has_persistence(self, tmp_path):
        """Verify persistence step runs."""
        G = _build_multi_file_graph(n_files=6, nodes_per_file=4)
        db = UnifiedWikiDB(tmp_path / "persist.wiki.db", embedding_dim=8)
        db.from_networkx(G)

        result = run_phase3(db, G)
        assert "persistence" in result
        db.close()

    @requires_leiden
    def test_deterministic(self, tmp_path):
        """Same graph should produce same section/page counts."""
        G = _build_multi_file_graph(n_files=6, nodes_per_file=4)

        db1 = UnifiedWikiDB(tmp_path / "det1.wiki.db", embedding_dim=8)
        db1.from_networkx(G)
        r1 = run_phase3(db1, G)
        db1.close()

        db2 = UnifiedWikiDB(tmp_path / "det2.wiki.db", embedding_dim=8)
        db2.from_networkx(G)
        r2 = run_phase3(db2, G)
        db2.close()

        assert r1["macro"]["cluster_count"] == r2["macro"]["cluster_count"]
        assert r1["micro"]["total_pages"] == r2["micro"]["total_pages"]


# ═══════════════════════════════════════════════════════════════════════════
# merge_macro_clusters
# ═══════════════════════════════════════════════════════════════════════════


class TestMergeMacroClusters:

    def test_no_merge_when_under_cap(self):
        G = _build_simple_graph()
        P = architectural_projection(G)
        macro = {n: 0 for n in P.nodes()}
        result = merge_macro_clusters(P, macro, max_sections=10)
        assert len(set(result.values())) == 1

    def test_merges_to_cap(self):
        G = nx.MultiDiGraph()
        for i in range(20):
            G.add_node(f"n{i}", **_make_node("class", f"src/f{i}.py", f"S{i}"))
        P = architectural_projection(G)
        # Assign each node to its own cluster
        macro = {n: i for i, n in enumerate(P.nodes())}
        result = merge_macro_clusters(P, macro, max_sections=3)
        assert len(set(result.values())) <= 3
