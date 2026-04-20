"""Unit tests for app/core/wiki_structure_planner/structure_refiner.py.

LLM calls are mocked — no real network activity.
"""

from __future__ import annotations

import json
from unittest.mock import MagicMock

import pytest

from app.core.wiki_structure_planner.structure_refiner import (
    _score_symbol,
    prioritize_symbols,
    summarize_cluster,
    refine_with_llm,
    _single_llm_refinement,
    _batched_llm_refinement,
    assemble_wiki_structure,
    _build_doc_only_structure,
    _add_doc_pages,
    _format_doc_clusters_for_prompt,
    _format_doc_assignment_instruction,
    _parse_llm_json,
)
from app.core.wiki_structure_planner.structure_skeleton import (
    DirCluster,
    DocCluster,
    StructureSkeleton,
    SymbolInfo,
)


# ── Helpers ────────────────────────────────────────────────────────────────────

def _make_sym(name: str = "Foo", sym_type: str = "class", layer: str = "core_type",
              connections: int = 5, rel_path: str = "src/foo.py") -> SymbolInfo:
    return SymbolInfo(name=name, type=sym_type, rel_path=rel_path,
                      layer=layer, connections=connections, docstring="A brief doc")


def _make_cluster(cid: int = 1, n_syms: int = 10, dirs: list[str] | None = None) -> DirCluster:
    dirs = dirs or [f"src/mod{cid}"]
    syms = [_make_sym(f"Cls{i}", rel_path=f"{dirs[0]}/f{i}.py") for i in range(n_syms)]
    return DirCluster(
        cluster_id=cid,
        dirs=dirs,
        symbols=syms,
        total_symbols=n_syms,
        primary_languages=["py"],
        depth_range=(1, 1),
    )


def _make_skeleton(n_clusters: int = 3, n_doc_clusters: int = 0) -> StructureSkeleton:
    clusters = [_make_cluster(i + 1) for i in range(n_clusters)]
    doc_clusters = [
        DocCluster(
            dir_path=f"docs/section{i}",
            doc_files=[f"docs/section{i}/guide.md"],
            doc_types=[".md"],
            file_count=1,
        )
        for i in range(n_doc_clusters)
    ]
    return StructureSkeleton(
        code_clusters=clusters,
        doc_clusters=doc_clusters,
        total_arch_symbols=sum(c.total_symbols for c in clusters),
        total_dirs_covered=n_clusters,
        total_dirs_in_repo=n_clusters + 2,
        repo_languages=["py"],
        effective_depth=5,
        repo_name="test-repo",
    )


def _llm_returning(items: list[dict]) -> MagicMock:
    llm = MagicMock()
    llm.invoke.return_value.content = json.dumps(items)
    return llm


# ── _score_symbol ─────────────────────────────────────────────────────────────

class TestScoreSymbol:
    def test_entry_point_scores_higher_than_constant(self):
        ep = _make_sym(layer="entry_point", connections=0)
        const = _make_sym(layer="constant", connections=100)
        assert _score_symbol(ep) > _score_symbol(const)

    def test_connections_break_ties_within_layer(self):
        high = _make_sym(layer="core_type", connections=100)
        low = _make_sym(layer="core_type", connections=1)
        assert _score_symbol(high) > _score_symbol(low)

    def test_connections_capped_at_100(self):
        sym_100 = _make_sym(layer="internal", connections=100)
        sym_999 = _make_sym(layer="internal", connections=999)
        assert _score_symbol(sym_100) == _score_symbol(sym_999)

    def test_unknown_layer_uses_default_weight(self):
        sym = _make_sym(layer="unknown_layer")
        score = _score_symbol(sym)
        # Default weight=2, so base score = 2*200 = 400
        assert score >= 400


# ── prioritize_symbols ────────────────────────────────────────────────────────

class TestPrioritizeSymbols:
    def test_returns_names_not_objects(self):
        syms = [_make_sym(f"S{i}") for i in range(5)]
        result = prioritize_symbols(syms, max_symbols=10)
        assert all(isinstance(n, str) for n in result)

    def test_returns_all_when_below_cap(self):
        syms = [_make_sym(f"C{i}") for i in range(5)]
        result = prioritize_symbols(syms, max_symbols=10)
        assert len(result) == 5

    def test_caps_output_when_over_max(self):
        syms = [_make_sym(f"X{i}", connections=i) for i in range(30)]
        result = prioritize_symbols(syms, max_symbols=10)
        assert len(result) == 10

    def test_most_important_symbols_first(self):
        ep = _make_sym("EntryClass", layer="entry_point", connections=50)
        const = _make_sym("CONST_VAL", layer="constant", connections=0)
        syms = [const, ep]
        result = prioritize_symbols(syms, max_symbols=2)
        assert result[0] == "EntryClass"

    def test_deduplicates_names_when_over_cap(self):
        # Dedup only fires on the pruning path (len > max_symbols)
        # Create many symbols but 2 share the same name — dedup removes the duplicate
        syms = [SymbolInfo(f"Unique{i}", "class", f"src/f{i}.py", "core_type", i, "") for i in range(10)]
        syms.append(SymbolInfo("Unique0", "class", "src/other.py", "core_type", 10, ""))
        # max_symbols=10 → pruning path, Unique0 appears twice but dedup removes one
        result = prioritize_symbols(syms, max_symbols=10)
        assert result.count("Unique0") == 1

    def test_empty_symbols_returns_empty(self):
        assert prioritize_symbols([], max_symbols=10) == []


# ── summarize_cluster ─────────────────────────────────────────────────────────

class TestSummarizeCluster:
    def test_returns_string(self):
        cluster = _make_cluster(1, 5)
        result = summarize_cluster(cluster)
        assert isinstance(result, str)

    def test_contains_cluster_id(self):
        cluster = _make_cluster(42, 3)
        result = summarize_cluster(cluster)
        assert "42" in result

    def test_contains_dir_list(self):
        cluster = _make_cluster(1, 5, dirs=["src/alpha", "src/beta"])
        result = summarize_cluster(cluster)
        assert "src/alpha" in result

    def test_contains_symbol_count(self):
        cluster = _make_cluster(1, 15)
        result = summarize_cluster(cluster)
        assert "15" in result

    def test_shows_top_symbols_by_connections(self):
        syms = [_make_sym(f"Name{i}", connections=i) for i in range(5)]
        cluster = DirCluster(1, ["src"], syms, 5, ["py"], (1, 1))
        result = summarize_cluster(cluster)
        # Top by connection should appear (highest connection = Name4)
        assert "Name4" in result


# ── _parse_llm_json ────────────────────────────────────────────────────────────

class TestParseLlmJson:
    def test_empty_content_returns_empty(self):
        assert _parse_llm_json("") == []

    def test_valid_json_array_parsed(self):
        data = [{"cluster_id": 1, "page_name": "Test Page"}]
        result = _parse_llm_json(json.dumps(data))
        assert len(result) == 1
        assert result[0]["page_name"] == "Test Page"

    def test_markdown_fenced_json_stripped(self):
        data = [{"cluster_id": 2, "page_name": "Auth Service"}]
        content = f"```json\n{json.dumps(data)}\n```"
        result = _parse_llm_json(content)
        assert len(result) == 1
        assert result[0]["cluster_id"] == 2

    def test_json_embedded_in_text_extracted(self):
        data = [{"cluster_id": 3, "page_name": "Storage Layer"}]
        content = f"Here is the output: {json.dumps(data)} Hope that helps."
        result = _parse_llm_json(content)
        assert len(result) == 1
        assert result[0]["cluster_id"] == 3

    def test_dict_with_clusters_key_extracted(self):
        data = {"clusters": [{"cluster_id": 1, "page_name": "Module"}]}
        result = _parse_llm_json(json.dumps(data))
        assert len(result) == 1
        assert result[0]["cluster_id"] == 1

    def test_invalid_json_returns_empty(self):
        result = _parse_llm_json("this is not json at all")
        assert result == []

    def test_single_dict_wrapped_in_list(self):
        data = {"cluster_id": 5, "page_name": "Single Page"}
        result = _parse_llm_json(json.dumps(data))
        assert len(result) == 1
        assert result[0]["cluster_id"] == 5


# ── _format_doc_clusters_for_prompt ───────────────────────────────────────────

class TestFormatDocClusters:
    def test_empty_returns_empty_string(self):
        assert _format_doc_clusters_for_prompt([]) == ""

    def test_includes_doc_index(self):
        dc = DocCluster("docs", ["docs/guide.md"], [".md"], 1)
        result = _format_doc_clusters_for_prompt([dc])
        assert "doc_0" in result

    def test_includes_dir_path(self):
        dc = DocCluster("proto/api", ["proto/api/README.md"], [".md"], 1)
        result = _format_doc_clusters_for_prompt([dc])
        assert "proto/api" in result

    def test_multiple_clusters_indexed_sequentially(self):
        clusters = [
            DocCluster(f"docs/sec{i}", [f"docs/sec{i}/f.md"], [".md"], 1)
            for i in range(3)
        ]
        result = _format_doc_clusters_for_prompt(clusters)
        assert "doc_0" in result
        assert "doc_1" in result
        assert "doc_2" in result


# ── _format_doc_assignment_instruction ────────────────────────────────────────

class TestFormatDocAssignmentInstruction:
    def test_empty_returns_empty_string(self):
        assert _format_doc_assignment_instruction([]) == ""

    def test_non_empty_returns_instruction(self):
        dc = DocCluster("docs", ["docs/guide.md"], [".md"], 1)
        result = _format_doc_assignment_instruction([dc])
        assert "doc_" in result
        assert len(result) > 10


# ── _add_doc_pages ────────────────────────────────────────────────────────────

class TestAddDocPages:
    def test_adds_page_for_each_cluster(self):
        from collections import defaultdict
        sections: dict = defaultdict(list)
        clusters = [
            DocCluster("docs/api", ["docs/api/guide.md"], [".md"], 1),
            DocCluster("docs/tutorial", ["docs/tutorial/intro.md"], [".md"], 1),
        ]
        _add_doc_pages(sections, clusters)
        # Should have pages in at least one section
        total_pages = sum(len(v) for v in sections.values())
        assert total_pages == 2

    def test_section_name_derived_from_top_dir(self):
        from collections import defaultdict
        sections: dict = defaultdict(list)
        dc = DocCluster("guides/intro", ["guides/intro/doc.md"], [".md"], 1)
        _add_doc_pages(sections, [dc])
        assert "Guides" in sections or any("guides" in k.lower() for k in sections)


# ── assemble_wiki_structure ───────────────────────────────────────────────────

class TestAssembleWikiStructure:
    def test_returns_valid_wiki_structure(self):
        skeleton = _make_skeleton(2)
        llm_items = [
            {"cluster_id": 1, "page_name": "Core Engine", "section_name": "Core",
             "description": "Core module", "retrieval_query": "engine"},
            {"cluster_id": 2, "page_name": "Data Pipeline", "section_name": "Data",
             "description": "Data processing", "retrieval_query": "pipeline"},
        ]
        result = assemble_wiki_structure(llm_items, skeleton, "test-repo")
        assert "wiki_title" in result
        assert "sections" in result
        assert result["total_pages"] == 2

    def test_orphan_clusters_auto_named(self):
        skeleton = _make_skeleton(2)
        # Only item for cluster 1 — cluster 2 is orphan
        llm_items = [
            {"cluster_id": 1, "page_name": "Core Engine", "section_name": "Core",
             "description": "Core", "retrieval_query": "core"},
        ]
        result = assemble_wiki_structure(llm_items, skeleton, "test-repo")
        # Orphan cluster 2 should appear in "Additional Components"
        total = sum(len(s["pages"]) for s in result["sections"])
        assert total == 2

    def test_doc_clusters_assigned_from_llm(self):
        skeleton = _make_skeleton(1, n_doc_clusters=1)
        llm_items = [
            {"cluster_id": 1, "page_name": "Core", "section_name": "Core",
             "description": "Core", "retrieval_query": "core"},
            {"cluster_id": "doc_0", "page_name": "API Documentation",
             "section_name": "Documentation", "description": "API docs",
             "retrieval_query": "api"},
        ]
        result = assemble_wiki_structure(llm_items, skeleton, "test-repo")
        all_page_names = [p["page_name"] for s in result["sections"] for p in s["pages"]]
        assert "API Documentation" in all_page_names

    def test_uncovered_doc_clusters_get_fallback_pages(self):
        skeleton = _make_skeleton(1, n_doc_clusters=1)
        # LLM doesn't cover the doc cluster
        llm_items = [
            {"cluster_id": 1, "page_name": "Core", "section_name": "Core",
             "description": "Core", "retrieval_query": "core"},
        ]
        result = assemble_wiki_structure(llm_items, skeleton, "test-repo")
        total = sum(len(s["pages"]) for s in result["sections"])
        assert total == 2

    def test_sections_sorted(self):
        skeleton = _make_skeleton(2)
        llm_items = [
            {"cluster_id": 1, "page_name": "Module A", "section_name": "Zebra",
             "description": "Z", "retrieval_query": ""},
            {"cluster_id": 2, "page_name": "Module B", "section_name": "Alpha",
             "description": "A", "retrieval_query": ""},
        ]
        result = assemble_wiki_structure(llm_items, skeleton, "repo")
        section_names = [s["section_name"] for s in result["sections"]]
        assert section_names == sorted(section_names)

    def test_target_symbols_injected_from_skeleton(self):
        skeleton = _make_skeleton(1)
        llm_items = [
            {"cluster_id": 1, "page_name": "Engine", "section_name": "Core",
             "description": "Engine", "retrieval_query": "engine"},
        ]
        result = assemble_wiki_structure(llm_items, skeleton, "repo")
        page = result["sections"][0]["pages"][0]
        assert len(page["target_symbols"]) > 0  # Symbols from cluster


# ── _build_doc_only_structure ─────────────────────────────────────────────────

class TestBuildDocOnlyStructure:
    def test_returns_valid_structure(self):
        skeleton = StructureSkeleton(
            code_clusters=[],
            doc_clusters=[DocCluster("docs", ["docs/guide.md"], [".md"], 1)],
            total_arch_symbols=0,
            total_dirs_covered=0,
            total_dirs_in_repo=1,
            repo_languages=[],
            effective_depth=3,
            repo_name="docs-only-repo",
        )
        result = _build_doc_only_structure(skeleton, "docs-only-repo")
        assert result["total_pages"] >= 1
        assert "sections" in result

    def test_fallback_when_no_doc_clusters(self):
        skeleton = StructureSkeleton(
            code_clusters=[],
            doc_clusters=[],
            total_arch_symbols=0,
            total_dirs_covered=0,
            total_dirs_in_repo=0,
            repo_languages=[],
            effective_depth=3,
        )
        result = _build_doc_only_structure(skeleton, "empty-repo")
        assert result["total_pages"] == 1
        assert result["sections"][0]["pages"][0]["page_name"] == "Repository Documentation"


# ── refine_with_llm ───────────────────────────────────────────────────────────

class TestRefineWithLlm:
    def test_single_llm_call_for_small_skeleton(self):
        skeleton = _make_skeleton(3)
        llm_items = [
            {"cluster_id": i + 1, "page_name": f"Page {i + 1}",
             "section_name": "Section A", "description": "Desc", "retrieval_query": ""}
            for i in range(3)
        ]
        llm = _llm_returning(llm_items)
        result = refine_with_llm(skeleton, llm, "test-repo", page_budget=10)
        assert llm.invoke.call_count == 1
        assert result["total_pages"] == 3

    def test_doc_only_skeleton_skips_llm(self):
        skeleton = StructureSkeleton(
            code_clusters=[],
            doc_clusters=[DocCluster("docs", ["docs/guide.md"], [".md"], 1)],
            total_arch_symbols=0,
            total_dirs_covered=0,
            total_dirs_in_repo=1,
            repo_languages=[],
            effective_depth=3,
        )
        llm = MagicMock()
        result = refine_with_llm(skeleton, llm, "repo", page_budget=5)
        assert llm.invoke.call_count == 0
        assert result["total_pages"] >= 1

    def test_large_skeleton_batched(self):
        # Create skeleton with >60 clusters to trigger batching
        clusters = [_make_cluster(i + 1) for i in range(65)]
        skeleton = StructureSkeleton(
            code_clusters=clusters,
            doc_clusters=[],
            total_arch_symbols=sum(c.total_symbols for c in clusters),
            total_dirs_covered=65,
            total_dirs_in_repo=65,
            repo_languages=["py"],
            effective_depth=5,
        )
        # LLM returns items for whatever clusters it's given
        def mock_invoke(prompt):
            # Extract cluster IDs from prompt by finding "Cluster N:" patterns
            import re
            ids = [int(m) for m in re.findall(r"Cluster (\d+):", prompt)]
            items = [
                {"cluster_id": cid, "page_name": f"Page {cid}",
                 "section_name": "General", "description": "Desc", "retrieval_query": ""}
                for cid in ids
            ]
            mock = MagicMock()
            mock.content = json.dumps(items)
            return mock

        llm = MagicMock()
        llm.invoke.side_effect = mock_invoke
        result = refine_with_llm(skeleton, llm, "large-repo", page_budget=80)
        # Should have called invoke twice (2 batches: 60 + 5)
        assert llm.invoke.call_count == 2
        assert result["total_pages"] == 65

    def test_wiki_title_contains_repo_name(self):
        skeleton = _make_skeleton(1)
        llm = _llm_returning([
            {"cluster_id": 1, "page_name": "Core", "section_name": "Core",
             "description": "Core", "retrieval_query": ""}
        ])
        result = refine_with_llm(skeleton, llm, "my-project", page_budget=5)
        assert "my-project" in result["wiki_title"]
