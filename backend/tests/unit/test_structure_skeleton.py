"""Unit tests for app/core/wiki_structure_planner/structure_skeleton.py.

All functions are pure/deterministic with no LLM calls.
"""

from __future__ import annotations

import pytest
import networkx as nx

from app.core.wiki_structure_planner.structure_skeleton import (
    SymbolInfo,
    DirCluster,
    DocCluster,
    StructureSkeleton,
    build_dir_symbol_map,
    build_dir_interaction_graph,
    cluster_directories,
    normalize_clusters,
    build_doc_clusters,
    build_skeleton,
    _cluster_by_directory_tree,
    _split_large,
)


# ── Helpers ────────────────────────────────────────────────────────────────────

def _make_graph_with_symbols(entries: list[dict]) -> nx.DiGraph:
    """Build a minimal nx.DiGraph with symbol nodes."""
    g = nx.DiGraph()
    for i, entry in enumerate(entries):
        g.add_node(f"node_{i}", **entry)
    return g


def _make_cross_edge_graph(entries: list[dict], edges: list[tuple]) -> nx.DiGraph:
    """Build graph with nodes and weighted edges."""
    g = _make_graph_with_symbols(entries)
    for src, tgt, rel in edges:
        g.add_edge(src, tgt, relationship_type=rel)
    return g


# ── SymbolInfo / dataclass construction ────────────────────────────────────────

class TestDataclasses:
    def test_symbol_info_fields(self):
        sym = SymbolInfo(name="Foo", type="class", rel_path="src/foo.py",
                         layer="public_api", connections=5, docstring="A class")
        assert sym.name == "Foo"
        assert sym.type == "class"
        assert sym.rel_path == "src/foo.py"
        assert sym.layer == "public_api"
        assert sym.connections == 5
        assert sym.docstring == "A class"

    def test_dir_cluster_fields(self):
        sym = SymbolInfo("X", "function", "a/b.py", "internal", 2, "")
        cluster = DirCluster(
            cluster_id=1,
            dirs=["a"],
            symbols=[sym],
            total_symbols=1,
            primary_languages=["py"],
            depth_range=(1, 1),
        )
        assert cluster.cluster_id == 1
        assert cluster.dirs == ["a"]
        assert len(cluster.symbols) == 1
        assert cluster.total_symbols == 1

    def test_doc_cluster_fields(self):
        dc = DocCluster(
            dir_path="docs",
            doc_files=["docs/README.md"],
            doc_types=[".md"],
            file_count=1,
        )
        assert dc.dir_path == "docs"
        assert len(dc.doc_files) == 1
        assert dc.file_count == 1

    def test_structure_skeleton_fields(self):
        skel = StructureSkeleton(
            code_clusters=[],
            doc_clusters=[],
            total_arch_symbols=0,
            total_dirs_covered=0,
            total_dirs_in_repo=0,
            repo_languages=[],
            effective_depth=5,
        )
        assert skel.effective_depth == 5
        assert skel.code_clusters == []


# ── build_dir_symbol_map ───────────────────────────────────────────────────────

class TestBuildDirSymbolMap:
    def test_empty_graph_returns_empty(self):
        g = nx.DiGraph()
        result = build_dir_symbol_map(g)
        assert result == {}

    def test_single_class_mapped_to_dir(self):
        g = _make_graph_with_symbols([{
            "symbol_type": "class",
            "symbol_name": "MyClass",
            "rel_path": "src/models/foo.py",
            "layer": "core_type",
            "docstring": "A model",
        }])
        result = build_dir_symbol_map(g)
        assert "src/models" in result
        syms = result["src/models"]
        assert len(syms) == 1
        assert syms[0].name == "MyClass"
        assert syms[0].type == "class"

    def test_root_level_file_maps_to_dot(self):
        g = _make_graph_with_symbols([{
            "symbol_type": "function",
            "symbol_name": "main",
            "rel_path": "main.py",
            "layer": "entry_point",
            "docstring": "",
        }])
        result = build_dir_symbol_map(g)
        assert "." in result
        assert result["."][0].name == "main"

    def test_non_arch_types_excluded(self):
        g = _make_graph_with_symbols([{
            "symbol_type": "method",
            "symbol_name": "do_thing",
            "rel_path": "src/utils.py",
            "layer": "internal",
            "docstring": "",
        }])
        result = build_dir_symbol_map(g)
        assert result == {}

    def test_ignored_dirs_excluded(self):
        g = _make_graph_with_symbols([{
            "symbol_type": "class",
            "symbol_name": "TestHelper",
            "rel_path": "__pycache__/helpers.py",
            "layer": "internal",
            "docstring": "",
        }])
        result = build_dir_symbol_map(g)
        assert result == {}

    def test_docstring_truncated_to_200_chars(self):
        long_doc = "X" * 500
        g = _make_graph_with_symbols([{
            "symbol_type": "class",
            "symbol_name": "Verbose",
            "rel_path": "pkg/a.py",
            "layer": "internal",
            "docstring": long_doc,
        }])
        result = build_dir_symbol_map(g)
        assert result["pkg"][0].docstring == long_doc[:200]

    def test_multiple_symbols_in_same_dir(self):
        entries = [
            {"symbol_type": "class", "symbol_name": "A", "rel_path": "src/mod.py", "layer": "core_type", "docstring": ""},
            {"symbol_type": "function", "symbol_name": "helper", "rel_path": "src/mod.py", "layer": "internal", "docstring": ""},
        ]
        g = _make_graph_with_symbols(entries)
        result = build_dir_symbol_map(g)
        assert len(result["src"]) == 2
        names = {s.name for s in result["src"]}
        assert names == {"A", "helper"}

    def test_connections_use_graph_degree(self):
        g = nx.DiGraph()
        g.add_node("n1", symbol_type="class", symbol_name="NodeClass", rel_path="mod/file.py",
                   layer="core_type", docstring="")
        g.add_node("n2")
        g.add_node("n3")
        g.add_edge("n2", "n1")
        g.add_edge("n3", "n1")
        result = build_dir_symbol_map(g)
        sym = result["mod"][0]
        assert sym.connections == 2  # in-degree only counts for undirected, but degree = in+out

    def test_layer_inferred_when_missing(self):
        g = _make_graph_with_symbols([{
            "symbol_type": "class",
            "symbol_name": "SomeClass",
            "rel_path": "app/services.py",
            # no "layer" key
        }])
        result = build_dir_symbol_map(g)
        # Layer should be set to something non-empty (classify_symbol_layer)
        assert result["app"][0].layer != ""


# ── build_dir_interaction_graph ────────────────────────────────────────────────

class TestBuildDirInteractionGraph:
    def test_empty_dir_symbols_gives_empty_graph(self):
        g = nx.DiGraph()
        result = build_dir_interaction_graph(g, {})
        assert result.number_of_nodes() == 0

    def test_nodes_added_for_dirs_with_symbols(self):
        syms = {"src/a": [SymbolInfo("A", "class", "src/a/a.py", "core_type", 1, "")]}
        g = nx.DiGraph()
        result = build_dir_interaction_graph(g, syms)
        assert "src/a" in result.nodes

    def test_cross_dir_import_creates_edge(self):
        g = nx.DiGraph()
        g.add_node("n1", rel_path="src/a/a.py")
        g.add_node("n2", rel_path="src/b/b.py")
        g.add_edge("n1", "n2", relationship_type="imports")
        syms = {
            "src/a": [SymbolInfo("A", "class", "src/a/a.py", "core_type", 1, "")],
            "src/b": [SymbolInfo("B", "class", "src/b/b.py", "core_type", 1, "")],
        }
        result = build_dir_interaction_graph(g, syms)
        assert result.has_edge("src/a", "src/b")
        assert result["src/a"]["src/b"]["weight"] == 2  # "imports" weight=2

    def test_same_dir_edges_not_added(self):
        g = nx.DiGraph()
        g.add_node("n1", rel_path="src/a/a.py")
        g.add_node("n2", rel_path="src/a/b.py")
        g.add_edge("n1", "n2", relationship_type="calls")
        syms = {"src/a": [
            SymbolInfo("A", "function", "src/a/a.py", "internal", 1, ""),
            SymbolInfo("B", "function", "src/a/b.py", "internal", 1, ""),
        ]}
        result = build_dir_interaction_graph(g, syms)
        assert result.number_of_edges() == 0

    def test_inheritance_edge_has_weight_3(self):
        g = nx.DiGraph()
        g.add_node("n1", rel_path="core/base.py")
        g.add_node("n2", rel_path="impl/derived.py")
        g.add_edge("n1", "n2", relationship_type="inheritance")
        syms = {
            "core": [SymbolInfo("Base", "class", "core/base.py", "public_api", 5, "")],
            "impl": [SymbolInfo("Derived", "class", "impl/derived.py", "core_type", 3, "")],
        }
        result = build_dir_interaction_graph(g, syms)
        assert result["core"]["impl"]["weight"] == 3


# ── cluster_directories ────────────────────────────────────────────────────────

class TestClusterDirectories:
    def test_empty_graph_returns_empty(self):
        g = nx.Graph()
        result = cluster_directories(g, page_budget=10, dir_symbols={})
        assert result == []

    def test_sparse_graph_uses_directory_tree_fallback(self):
        # Graph with nodes but no edges -> sparse -> directory-tree fallback
        g = nx.Graph()
        g.add_node("src/a")
        g.add_node("src/b")
        g.add_node("lib/c")
        syms = {
            "src/a": [SymbolInfo("A", "class", "src/a/a.py", "internal", 1, "")] * 5,
            "src/b": [SymbolInfo("B", "class", "src/b/b.py", "internal", 1, "")] * 5,
            "lib/c": [SymbolInfo("C", "class", "lib/c/c.py", "internal", 1, "")] * 5,
        }
        result = cluster_directories(g, page_budget=5, dir_symbols=syms)
        assert len(result) >= 1
        # Each result item is a set of dir paths
        all_dirs = set()
        for s in result:
            all_dirs.update(s)
        assert "src/a" in all_dirs or "src/b" in all_dirs

    def test_densely_connected_graph_uses_louvain(self):
        g = nx.Graph()
        dirs = [f"grp{i}/mod{j}" for i in range(3) for j in range(4)]
        syms = {}
        for d in dirs:
            syms[d] = [SymbolInfo(f"Class{d}", "class", f"{d}/f.py", "core_type", 2, "")] * 15
        for d in dirs:
            g.add_node(d, symbol_count=15)
        # Dense connections within groups
        for i in range(3):
            group = [d for d in dirs if d.startswith(f"grp{i}")]
            for a in group:
                for b in group:
                    if a != b:
                        g.add_edge(a, b, weight=5)
        result = cluster_directories(g, page_budget=3, dir_symbols=syms)
        assert len(result) >= 1


# ── normalize_clusters ────────────────────────────────────────────────────────

class TestNormalizeClusters:
    def _make_syms(self, n: int, dir_path: str) -> list[SymbolInfo]:
        return [SymbolInfo(f"S{i}", "class", f"{dir_path}/f{i}.py", "core_type", 1, "") for i in range(n)]

    def test_empty_input_returns_empty(self):
        result = normalize_clusters([], {})
        assert result == []

    def test_normal_cluster_preserved(self):
        syms = self._make_syms(20, "src/core")
        dir_syms = {"src/core": syms}
        raw = [{"src/core"}]
        result = normalize_clusters(raw, dir_syms, min_symbols=5, max_symbols=40)
        assert len(result) == 1
        assert result[0].dirs == ["src/core"]
        assert result[0].total_symbols == 20

    def test_large_cluster_is_split(self):
        # Need 2+ dirs in the cluster for _split_large to trigger
        syms_a = self._make_syms(45, "src/big_a")
        syms_b = self._make_syms(45, "src/big_b")
        dir_syms = {"src/big_a": syms_a, "src/big_b": syms_b}
        raw = [{"src/big_a", "src/big_b"}]
        result = normalize_clusters(raw, dir_syms, min_symbols=5, max_symbols=40)
        # Split into 2+ clusters
        assert len(result) >= 2

    def test_small_clusters_merged_with_sibling(self):
        syms_large = self._make_syms(20, "src/mod_a")
        syms_small = self._make_syms(3, "src/mod_b")
        dir_syms = {"src/mod_a": syms_large, "src/mod_b": syms_small}
        raw = [{"src/mod_a"}, {"src/mod_b"}]
        result = normalize_clusters(raw, dir_syms, min_symbols=5, max_symbols=40)
        # mod_b should be merged into mod_a since they're siblings
        assert len(result) == 1
        all_dirs = result[0].dirs
        assert "src/mod_a" in all_dirs
        assert "src/mod_b" in all_dirs

    def test_cluster_ids_assigned(self):
        syms_a = self._make_syms(15, "a")
        syms_b = self._make_syms(15, "b")
        dir_syms = {"a": syms_a, "b": syms_b}
        raw = [{"a"}, {"b"}]
        result = normalize_clusters(raw, dir_syms, min_symbols=5, max_symbols=40)
        ids = [c.cluster_id for c in result]
        assert ids == sorted(ids)

    def test_cluster_with_no_symbols_skipped(self):
        dir_syms = {"src/empty": []}
        raw = [{"src/empty"}]
        result = normalize_clusters(raw, dir_syms)
        assert result == []

    def test_primary_languages_computed(self):
        syms = [SymbolInfo(f"F{i}", "function", f"src/main.py", "internal", 1, "") for i in range(15)]
        dir_syms = {"src": syms}
        raw = [{"src"}]
        result = normalize_clusters(raw, dir_syms, min_symbols=5, max_symbols=40)
        assert len(result) == 1
        assert "py" in result[0].primary_languages

    def test_depth_range_computed(self):
        syms = self._make_syms(15, "a/b/c")
        dir_syms = {"a/b/c": syms}
        raw = [{"a/b/c"}]
        result = normalize_clusters(raw, dir_syms, min_symbols=5, max_symbols=40)
        assert result[0].depth_range == (2, 2)  # 2 slashes = depth 2


# ── _split_large ──────────────────────────────────────────────────────────────

class TestSplitLarge:
    def test_splits_into_two_groups(self):
        syms_a = [SymbolInfo(f"A{i}", "class", f"a/f.py", "core_type", 1, "") for i in range(30)]
        syms_b = [SymbolInfo(f"B{i}", "class", f"b/f.py", "core_type", 1, "") for i in range(30)]
        dir_syms = {"a": syms_a, "b": syms_b}
        result = _split_large({"a", "b"}, dir_syms, max_symbols=30)
        assert len(result) >= 2
        # Each group is a set
        all_dirs = set()
        for g in result:
            all_dirs.update(g)
        assert "a" in all_dirs
        assert "b" in all_dirs

    def test_single_dir_stays_in_one_group(self):
        # _split_large can't split a single directory further — it groups dirs, not files
        syms = [SymbolInfo(f"C{i}", "class", f"big/f.py", "core_type", 1, "") for i in range(100)]
        dir_syms = {"big": syms}
        result = _split_large({"big"}, dir_syms, max_symbols=30)
        # With only 1 dir, round-robin puts all dirs in the smallest group
        # n_groups = ceil(100/30) = 4, but only 1 dir → 1 non-empty group
        assert len(result) >= 1
        assert {"big"} in result


# ── _cluster_by_directory_tree ────────────────────────────────────────────────

class TestClusterByDirectoryTree:
    def test_sibling_dirs_grouped_together(self):
        g = nx.Graph()
        for d in ["src/a", "src/b", "lib/c"]:
            g.add_node(d)
        syms = {
            "src/a": [SymbolInfo(f"A{i}", "class", f"src/a/f.py", "internal", 1, "") for i in range(8)],
            "src/b": [SymbolInfo(f"B{i}", "class", f"src/b/f.py", "internal", 1, "") for i in range(8)],
            "lib/c": [SymbolInfo(f"C{i}", "class", f"lib/c/f.py", "internal", 1, "") for i in range(8)],
        }
        result = _cluster_by_directory_tree(g, syms, min_symbols=5)
        assert len(result) >= 1


# ── build_doc_clusters ────────────────────────────────────────────────────────

class TestBuildDocClusters:
    def test_empty_files_returns_empty(self):
        result = build_doc_clusters([], {}, effective_depth=3)
        assert result == []

    def test_markdown_files_in_docs_dir(self):
        files = ["docs/README.md", "docs/guide.md", "src/main.py"]
        dir_syms = {"src": [SymbolInfo("Foo", "class", "src/main.py", "internal", 1, "")]}
        result = build_doc_clusters(files, dir_syms, effective_depth=3)
        assert len(result) == 1
        assert result[0].dir_path == "docs"
        assert ".md" in result[0].doc_types
        assert result[0].file_count == 2

    def test_dirs_with_code_symbols_excluded(self):
        files = ["src/README.md"]
        dir_syms = {"src": [SymbolInfo("Foo", "class", "src/a.py", "internal", 1, "")]}
        result = build_doc_clusters(files, dir_syms, effective_depth=3)
        assert result == []

    def test_deep_doc_dirs_grouped_by_prefix(self):
        files = [
            "docs/api/v1/README.md",
            "docs/api/v2/guide.md",
        ]
        result = build_doc_clusters(files, {}, effective_depth=5)
        # Should group both under "docs/api" prefix
        assert len(result) == 1
        assert result[0].dir_path == "docs/api"

    def test_ignored_dirs_excluded(self):
        files = ["node_modules/README.md"]
        result = build_doc_clusters(files, {}, effective_depth=3)
        assert result == []


# ── build_skeleton ────────────────────────────────────────────────────────────

class TestBuildSkeleton:
    def _make_minimal_graph(self) -> nx.DiGraph:
        g = nx.DiGraph()
        entries = [
            {"symbol_type": "class", "symbol_name": "Alpha", "rel_path": "src/a/alpha.py",
             "layer": "public_api", "docstring": "Alpha class"},
            {"symbol_type": "function", "symbol_name": "beta", "rel_path": "src/b/beta.py",
             "layer": "internal", "docstring": "Beta fn"},
        ]
        for i, e in enumerate(entries):
            g.add_node(f"n{i}", **e)
        return g

    def test_returns_structure_skeleton(self):
        g = self._make_minimal_graph()
        files = ["src/a/alpha.py", "src/b/beta.py", "docs/guide.md"]
        skel = build_skeleton(g, files, page_budget=5, effective_depth=3, repo_name="test")
        assert isinstance(skel, StructureSkeleton)
        assert skel.repo_name == "test"

    def test_total_arch_symbols_counted(self):
        g = self._make_minimal_graph()
        files = ["src/a/alpha.py", "src/b/beta.py"]
        skel = build_skeleton(g, files, page_budget=5, effective_depth=3)
        assert skel.total_arch_symbols == 2

    def test_doc_clusters_detected(self):
        g = self._make_minimal_graph()
        files = ["src/a/alpha.py", "docs/guide.md", "docs/api.md"]
        skel = build_skeleton(g, files, page_budget=5, effective_depth=3)
        assert len(skel.doc_clusters) >= 1

    def test_empty_graph_produces_valid_skeleton(self):
        g = nx.DiGraph()
        skel = build_skeleton(g, [], page_budget=5, effective_depth=3)
        assert skel.total_arch_symbols == 0
        assert skel.code_clusters == []

    def test_repo_languages_from_extensions(self):
        g = nx.DiGraph()
        for i in range(5):
            g.add_node(f"n{i}", symbol_type="class", symbol_name=f"Cls{i}",
                       rel_path=f"src/mod{i}.py", layer="core_type", docstring="")
        files = [f"src/mod{i}.py" for i in range(5)]
        skel = build_skeleton(g, files, page_budget=5, effective_depth=3)
        assert "py" in skel.repo_languages

    def test_total_dirs_covered_is_non_negative(self):
        g = self._make_minimal_graph()
        files = ["src/a/alpha.py", "src/b/beta.py"]
        skel = build_skeleton(g, files, page_budget=5, effective_depth=3)
        assert skel.total_dirs_covered >= 0
        assert skel.total_dirs_in_repo >= 0
