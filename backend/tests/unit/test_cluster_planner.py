"""Tests for app.core.wiki_structure_planner.cluster_planner.

Covers:
- Pure-function helpers: ``_extract_text``, ``_parse_json_response``
- ``ClusterStructurePlanner`` with a real (in-memory) ``UnifiedWikiDB``
  and a mocked LLM
- ``_adaptive_central_k`` scaling
- ``_load_architectural_cluster_map`` with/without test exclusion
- ``_get_dominant_symbols``, ``_get_micro_summaries``, ``_get_page_symbols``
- ``_node_ids_to_*`` helper family
- ``_fallback_spec``, ``_fallback_section``
- ``plan_structure`` end-to-end with mocked LLM
"""

from __future__ import annotations

import json
from typing import Any, Dict, List
from unittest.mock import MagicMock, patch

import networkx as nx
import pytest

from app.core.feature_flags import FeatureFlags
from app.core.graph_clustering import run_phase3
from app.core.unified_db import UnifiedWikiDB
from app.core.wiki_structure_planner.cluster_planner import (
    MAX_DOMINANT_SYMBOLS,
    MAX_MICRO_SUMMARY_SYMBOLS,
    ClusterStructurePlanner,
    _extract_text,
    _parse_json_response,
)


# ═══════════════════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════════════════


def _fake_llm_response(json_dict: Dict[str, Any]) -> MagicMock:
    """Create a mock LLM response with the given JSON content."""
    msg = MagicMock()
    msg.content = json.dumps(json_dict)
    return msg


def _build_graph_for_planner(n_groups=3, per_group=4) -> nx.MultiDiGraph:
    """Build a graph with multiple 'packages' to form macro-clusters."""
    G = nx.MultiDiGraph()
    for g in range(n_groups):
        for i in range(per_group):
            nid = f"py::src/pkg{g}/mod{i}.py::Cls{g}_{i}"
            stype = "class" if i % 2 == 0 else "function"
            G.add_node(
                nid,
                rel_path=f"src/pkg{g}/mod{i}.py",
                file_name=f"mod{i}.py",
                language="python",
                symbol_name=f"Cls{g}_{i}",
                symbol_type=stype,
                source_text=f"class Cls{g}_{i}: pass",
                signature=f"class Cls{g}_{i}:",
                docstring=f"A class in package {g}",
            )
        # Intra-group edges (dense)
        for i in range(per_group - 1):
            src = f"py::src/pkg{g}/mod{i}.py::Cls{g}_{i}"
            tgt = f"py::src/pkg{g}/mod{i+1}.py::Cls{g}_{i+1}"
            G.add_edge(src, tgt, relationship_type="calls", weight=2.0)
    # Sparse cross-group edges
    for g in range(n_groups - 1):
        G.add_edge(
            f"py::src/pkg{g}/mod0.py::Cls{g}_0",
            f"py::src/pkg{g+1}/mod0.py::Cls{g+1}_0",
            relationship_type="calls",
            weight=0.3,
        )
    return G


def _populate_db_with_clusters(db: UnifiedWikiDB, G: nx.MultiDiGraph) -> None:
    """Populate DB from graph and run Phase 3 clustering."""
    db.from_networkx(G)
    # Phase 3 persists cluster assignments into the DB
    run_phase3(db, G)


# ═══════════════════════════════════════════════════════════════════════════
# _extract_text
# ═══════════════════════════════════════════════════════════════════════════


class TestExtractText:

    def test_string_content(self):
        msg = MagicMock()
        msg.content = "Hello, world"
        assert _extract_text(msg) == "Hello, world"

    def test_list_content(self):
        msg = MagicMock()
        msg.content = [
            "Part 1",
            {"text": "Part 2"},
            {"content": "Part 3"},
        ]
        result = _extract_text(msg)
        assert "Part 1" in result
        assert "Part 2" in result
        assert "Part 3" in result

    def test_plain_string(self):
        assert _extract_text("just a string") == "just a string"


# ═══════════════════════════════════════════════════════════════════════════
# _parse_json_response
# ═══════════════════════════════════════════════════════════════════════════


class TestParseJsonResponse:

    def test_plain_json(self):
        raw = '{"section_name": "Auth", "section_description": "Auth module"}'
        result = _parse_json_response(raw)
        assert result["section_name"] == "Auth"

    def test_json_in_code_fence(self):
        raw = '```json\n{"key": "value"}\n```'
        result = _parse_json_response(raw)
        assert result["key"] == "value"

    def test_json_with_leading_text(self):
        raw = 'Here is the JSON:\n{"key": "value"}'
        result = _parse_json_response(raw)
        assert result["key"] == "value"

    def test_empty_returns_empty_dict(self):
        result = _parse_json_response("")
        assert result == {}


# ═══════════════════════════════════════════════════════════════════════════
# ClusterStructurePlanner — construction
# ═══════════════════════════════════════════════════════════════════════════


class TestPlannerConstruction:

    def test_init_with_title(self, tmp_path):
        db = UnifiedWikiDB(tmp_path / "test.wiki.db", embedding_dim=8)
        llm = MagicMock()
        planner = ClusterStructurePlanner(db, llm, wiki_title="My Wiki")
        assert planner.wiki_title == "My Wiki"
        db.close()

    def test_init_derives_title(self, tmp_path):
        db = UnifiedWikiDB(tmp_path / "test.wiki.db", embedding_dim=8)
        db.set_meta("repo_identifier", "owner/my-repo")
        llm = MagicMock()
        planner = ClusterStructurePlanner(db, llm)
        assert "my-repo" in planner.wiki_title
        db.close()

    def test_exclude_tests_flag(self, tmp_path):
        db = UnifiedWikiDB(tmp_path / "test.wiki.db", embedding_dim=8)
        llm = MagicMock()

        with patch(
            "app.core.wiki_structure_planner.cluster_planner.get_feature_flags",
            return_value=FeatureFlags(exclude_tests=True),
        ):
            planner = ClusterStructurePlanner(db, llm)
            assert planner._exclude_tests is True
            assert "is_test" in planner._test_sql

        db.close()


# ═══════════════════════════════════════════════════════════════════════════
# _adaptive_central_k
# ═══════════════════════════════════════════════════════════════════════════


class TestAdaptiveCentralK:

    def test_small_repo(self, tmp_path):
        G = _build_graph_for_planner(n_groups=2, per_group=3)  # 6 nodes
        db = UnifiedWikiDB(tmp_path / "test.wiki.db", embedding_dim=8)
        _populate_db_with_clusters(db, G)

        llm = MagicMock()
        planner = ClusterStructurePlanner(db, llm, wiki_title="Test")
        k = planner._adaptive_central_k()
        assert k == 5  # < 200 nodes
        db.close()

    def test_caches_result(self, tmp_path):
        G = _build_graph_for_planner(n_groups=2, per_group=3)
        db = UnifiedWikiDB(tmp_path / "test.wiki.db", embedding_dim=8)
        _populate_db_with_clusters(db, G)

        llm = MagicMock()
        planner = ClusterStructurePlanner(db, llm, wiki_title="Test")
        k1 = planner._adaptive_central_k()
        k2 = planner._adaptive_central_k()
        assert k1 == k2
        db.close()


# ═══════════════════════════════════════════════════════════════════════════
# _load_architectural_cluster_map
# ═══════════════════════════════════════════════════════════════════════════


class TestLoadArchitecturalClusterMap:

    def test_returns_cluster_map(self, tmp_path):
        G = _build_graph_for_planner(n_groups=3, per_group=4)
        db = UnifiedWikiDB(tmp_path / "test.wiki.db", embedding_dim=8)
        _populate_db_with_clusters(db, G)

        llm = MagicMock()
        planner = ClusterStructurePlanner(db, llm, wiki_title="Test")
        cmap = planner._load_architectural_cluster_map()

        # Should have clusters
        assert len(cmap) > 0
        # Every value should have micro-clusters
        for macro_id, micro_map in cmap.items():
            assert isinstance(micro_map, dict)
            for micro_id, node_ids in micro_map.items():
                assert len(node_ids) > 0
        db.close()

    def test_excludes_test_nodes(self, tmp_path):
        G = _build_graph_for_planner(n_groups=2, per_group=3)
        # Add a test node
        G.add_node(
            "py::tests/test_foo.py::TestFoo",
            rel_path="tests/test_foo.py",
            file_name="test_foo.py",
            language="python",
            symbol_name="TestFoo",
            symbol_type="class",
            source_text="class TestFoo: pass",
        )
        G.add_edge(
            list(G.nodes())[0], "py::tests/test_foo.py::TestFoo",
            relationship_type="calls", weight=1.0,
        )
        db = UnifiedWikiDB(tmp_path / "test.wiki.db", embedding_dim=8)
        _populate_db_with_clusters(db, G)

        with patch(
            "app.core.wiki_structure_planner.cluster_planner.get_feature_flags",
            return_value=FeatureFlags(exclude_tests=True),
        ):
            llm = MagicMock()
            planner = ClusterStructurePlanner(db, llm, wiki_title="Test")
            cmap = planner._load_architectural_cluster_map()

        # Test node should not appear
        all_node_ids = [
            nid
            for mm in cmap.values()
            for nids in mm.values()
            for nid in nids
        ]
        assert "py::tests/test_foo.py::TestFoo" not in all_node_ids
        db.close()


# ═══════════════════════════════════════════════════════════════════════════
# _get_dominant_symbols
# ═══════════════════════════════════════════════════════════════════════════


class TestGetDominantSymbols:

    def test_returns_limited_symbols(self, tmp_path):
        G = _build_graph_for_planner(n_groups=2, per_group=6)
        db = UnifiedWikiDB(tmp_path / "test.wiki.db", embedding_dim=8)
        _populate_db_with_clusters(db, G)

        llm = MagicMock()
        planner = ClusterStructurePlanner(db, llm, wiki_title="Test")
        cmap = planner._load_architectural_cluster_map()

        if cmap:
            macro_id = next(iter(cmap))
            dominant = planner._get_dominant_symbols(macro_id)
            assert len(dominant) <= MAX_DOMINANT_SYMBOLS
            for sym in dominant:
                assert "name" in sym
                assert "type" in sym
                assert "path" in sym
        db.close()


# ═══════════════════════════════════════════════════════════════════════════
# _get_micro_summaries
# ═══════════════════════════════════════════════════════════════════════════


class TestGetMicroSummaries:

    def test_returns_summaries(self, tmp_path):
        G = _build_graph_for_planner(n_groups=2, per_group=4)
        db = UnifiedWikiDB(tmp_path / "test.wiki.db", embedding_dim=8)
        _populate_db_with_clusters(db, G)

        llm = MagicMock()
        planner = ClusterStructurePlanner(db, llm, wiki_title="Test")
        cmap = planner._load_architectural_cluster_map()

        if cmap:
            macro_id = next(iter(cmap))
            summaries = planner._get_micro_summaries(cmap[macro_id])
            assert len(summaries) > 0
            for s in summaries:
                assert "micro_id" in s
                assert "symbol_count" in s
                assert "symbols" in s
        db.close()


# ═══════════════════════════════════════════════════════════════════════════
# _get_page_symbols
# ═══════════════════════════════════════════════════════════════════════════


class TestGetPageSymbols:

    def test_returns_enriched_symbols(self, tmp_path):
        G = _build_graph_for_planner(n_groups=2, per_group=4)
        db = UnifiedWikiDB(tmp_path / "test.wiki.db", embedding_dim=8)
        _populate_db_with_clusters(db, G)

        llm = MagicMock()
        planner = ClusterStructurePlanner(db, llm, wiki_title="Test")
        cmap = planner._load_architectural_cluster_map()

        if cmap:
            macro_id = next(iter(cmap))
            micro_id = next(iter(cmap[macro_id]))
            node_ids = cmap[macro_id][micro_id]
            symbols = planner._get_page_symbols(node_ids)
            for sym in symbols:
                assert "name" in sym
                assert "signature" in sym  # enriched vs micro_summaries
                assert "docstring" in sym
        db.close()


# ═══════════════════════════════════════════════════════════════════════════
# Node-to-field helpers
# ═══════════════════════════════════════════════════════════════════════════


class TestNodeIdHelpers:

    def test_node_ids_to_symbol_names(self, tmp_path):
        G = _build_graph_for_planner(n_groups=1, per_group=3)
        db = UnifiedWikiDB(tmp_path / "test.wiki.db", embedding_dim=8)
        _populate_db_with_clusters(db, G)

        llm = MagicMock()
        planner = ClusterStructurePlanner(db, llm, wiki_title="Test")
        node_ids = list(G.nodes())[:3]
        names = planner._node_ids_to_symbol_names(node_ids)
        assert all(isinstance(n, str) for n in names)
        db.close()

    def test_node_ids_to_paths(self, tmp_path):
        G = _build_graph_for_planner(n_groups=1, per_group=3)
        db = UnifiedWikiDB(tmp_path / "test.wiki.db", embedding_dim=8)
        _populate_db_with_clusters(db, G)

        llm = MagicMock()
        planner = ClusterStructurePlanner(db, llm, wiki_title="Test")
        node_ids = list(G.nodes())[:3]
        paths = planner._node_ids_to_paths(node_ids)
        assert all("/" in p for p in paths)
        db.close()

    def test_node_ids_to_folders(self, tmp_path):
        G = _build_graph_for_planner(n_groups=1, per_group=3)
        db = UnifiedWikiDB(tmp_path / "test.wiki.db", embedding_dim=8)
        _populate_db_with_clusters(db, G)

        llm = MagicMock()
        planner = ClusterStructurePlanner(db, llm, wiki_title="Test")
        node_ids = list(G.nodes())[:3]
        folders = planner._node_ids_to_folders(node_ids)
        assert len(folders) > 0
        db.close()


# ═══════════════════════════════════════════════════════════════════════════
# _fallback_spec & _fallback_section
# ═══════════════════════════════════════════════════════════════════════════


class TestFallbacks:

    def test_fallback_spec(self, tmp_path):
        db = UnifiedWikiDB(tmp_path / "test.wiki.db", embedding_dim=8)
        llm = MagicMock()
        planner = ClusterStructurePlanner(db, llm, wiki_title="Test")
        spec = planner._fallback_spec()
        assert spec.wiki_title == "Test"
        assert len(spec.sections) == 1
        assert spec.total_pages == 1
        db.close()

    def test_fallback_section(self, tmp_path):
        G = _build_graph_for_planner(n_groups=1, per_group=3)
        db = UnifiedWikiDB(tmp_path / "test.wiki.db", embedding_dim=8)
        _populate_db_with_clusters(db, G)

        llm = MagicMock()
        planner = ClusterStructurePlanner(db, llm, wiki_title="Test")

        micro_map = {0: list(G.nodes())[:2], 1: list(G.nodes())[2:3]}
        section = planner._fallback_section(0, micro_map, 1)
        assert section.section_name
        assert section.section_order == 1
        assert len(section.pages) == len(micro_map)
        db.close()


# ═══════════════════════════════════════════════════════════════════════════
# plan_structure end-to-end (LLM mocked)
# ═══════════════════════════════════════════════════════════════════════════


class TestPlanStructure:

    def _mock_llm(self):
        """Create an LLM mock that returns valid JSON for both naming passes."""
        llm = MagicMock()

        call_count = [0]

        def side_effect(messages):
            call_count[0] += 1
            # First call per macro = section naming, subsequent = page naming
            if call_count[0] % 3 == 1:
                return _fake_llm_response({
                    "section_name": f"Section-{call_count[0]}",
                    "section_description": f"Section description {call_count[0]}",
                })
            else:
                return _fake_llm_response({
                    "page_name": f"Page-{call_count[0]}",
                    "description": f"Page description {call_count[0]}",
                    "retrieval_query": f"query-{call_count[0]}",
                })

        llm.invoke.side_effect = side_effect
        return llm

    def test_plan_structure_basic(self, tmp_path):
        """End-to-end: populate DB → cluster → plan → verify spec."""
        G = _build_graph_for_planner(n_groups=3, per_group=4)
        db = UnifiedWikiDB(tmp_path / "test.wiki.db", embedding_dim=8)
        _populate_db_with_clusters(db, G)

        llm = self._mock_llm()
        planner = ClusterStructurePlanner(db, llm, wiki_title="Test Wiki")
        spec = planner.plan_structure()

        assert spec.wiki_title == "Test Wiki"
        assert len(spec.sections) > 0
        assert spec.total_pages > 0

        for section in spec.sections:
            assert section.section_name
            assert len(section.pages) > 0
            for page in section.pages:
                assert page.page_name
                assert page.metadata.get("planner_mode") == "cluster"

        db.close()

    def test_plan_structure_empty_db(self, tmp_path):
        """Empty DB → fallback spec."""
        db = UnifiedWikiDB(tmp_path / "test.wiki.db", embedding_dim=8)
        llm = MagicMock()
        planner = ClusterStructurePlanner(db, llm, wiki_title="Empty")
        spec = planner.plan_structure()
        assert spec.total_pages == 1
        assert "fallback" in spec.sections[0].rationale.lower()
        db.close()

    def test_plan_structure_llm_failure_uses_fallback(self, tmp_path):
        """LLM that raises → fallback section names used."""
        G = _build_graph_for_planner(n_groups=2, per_group=4)
        db = UnifiedWikiDB(tmp_path / "test.wiki.db", embedding_dim=8)
        _populate_db_with_clusters(db, G)

        llm = MagicMock()
        llm.invoke.side_effect = Exception("LLM down")

        planner = ClusterStructurePlanner(db, llm, wiki_title="Test")
        spec = planner.plan_structure()

        # Should still produce sections (via fallback)
        assert len(spec.sections) > 0
        assert spec.total_pages > 0
        db.close()
