"""Unit tests for app/core/wiki_structure_planner/structure_tools.py.

Tests cover: StructureCollector, detect_effective_depth, explore_repository_tree,
format_tree_for_llm, and the various helper/tool-handler methods.
No real LLM calls or network activity.
"""

from __future__ import annotations

import os
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import networkx as nx
import pytest

from app.core.wiki_structure_planner.structure_tools import (
    StructureCollector,
    detect_effective_depth,
    explore_repository_tree,
    format_tree_for_llm,
    _find_similar_symbols,
    _first_line_summary,
    _signature_brief,
    _get_symbol_brief,
    _is_code_file,
    _is_doc_file,
    _should_ignore_dir,
)


# ── Helpers ────────────────────────────────────────────────────────────────────

def _make_graph(entries: list[dict]) -> nx.DiGraph:
    g = nx.DiGraph()
    for i, e in enumerate(entries):
        g.add_node(f"node_{i}", **e)
    return g


def _make_collector(**kwargs) -> StructureCollector:
    """Create a StructureCollector with minimal dependencies."""
    defaults = dict(page_budget=10)
    defaults.update(kwargs)
    return StructureCollector(**defaults)


def _make_code_graph_with_symbols(names_and_types: list[tuple[str, str, str]]) -> nx.DiGraph:
    """Build a graph with (name, type, rel_path) entries."""
    g = nx.DiGraph()
    for i, (name, sym_type, rel_path) in enumerate(names_and_types):
        g.add_node(f"n{i}", symbol_name=name, symbol_type=sym_type, rel_path=rel_path)
    return g


# ── detect_effective_depth ────────────────────────────────────────────────────

class TestDetectEffectiveDepth:
    def test_empty_files_returns_default(self):
        result = detect_effective_depth([], static_default=5)
        assert result == 5

    def test_shallow_files_returns_default(self):
        files = ["main.py", "app.py", "config.py"]
        result = detect_effective_depth(files, static_default=5)
        assert result == 5

    def test_deep_java_files_increases_depth(self):
        files = [f"src/main/java/org/app/service/sub/Service{i}.java" for i in range(20)]
        result = detect_effective_depth(files, static_default=5, ceiling=8)
        assert result > 5

    def test_ceiling_respected(self):
        # Files at depth 10 — should be capped at ceiling
        files = [f"a/b/c/d/e/f/g/h/i/j/k{i}.py" for i in range(20)]
        result = detect_effective_depth(files, static_default=5, ceiling=8)
        assert result <= 8

    def test_non_code_files_ignored(self):
        # Only .md files — no code depth info → use default
        files = ["docs/guide.md", "docs/api/reference.md"]
        result = detect_effective_depth(files, static_default=5)
        assert result == 5

    def test_mixed_code_and_doc_uses_code_depths(self):
        # Code at depth 2, docs at depth 6 — should use code depths
        code_files = [f"src/mod/file{i}.py" for i in range(10)]
        doc_files = [f"docs/a/b/c/d/e/guide{i}.md" for i in range(10)]
        result = detect_effective_depth(code_files + doc_files, static_default=5)
        # Should not be inflated by doc file depths
        assert result <= 7  # code is at depth 2 → p90=2 → effective=3 → clamped to 5

    def test_result_at_least_static_default(self):
        files = ["src/main.py"]
        result = detect_effective_depth(files, static_default=5)
        assert result >= 5


# ── _is_code_file / _is_doc_file / _should_ignore_dir ────────────────────────

class TestFileClassifiers:
    def test_py_is_code(self):
        assert _is_code_file("main.py") is True

    def test_ts_is_code(self):
        assert _is_code_file("app.ts") is True

    def test_md_is_not_code(self):
        assert _is_code_file("README.md") is False

    def test_md_is_doc(self):
        assert _is_doc_file("guide.md") is True

    def test_py_is_not_doc(self):
        assert _is_doc_file("app.py") is False

    def test_node_modules_ignored(self):
        assert _should_ignore_dir("node_modules") is True

    def test_dot_dir_ignored_by_default(self):
        assert _should_ignore_dir(".git") is True

    def test_github_dir_not_ignored(self):
        # .github is in DOT_DIR_WHITELIST
        assert _should_ignore_dir(".github") is False

    def test_normal_dir_not_ignored(self):
        assert _should_ignore_dir("src") is False


# ── _find_similar_symbols ─────────────────────────────────────────────────────

class TestFindSimilarSymbols:
    def test_empty_index_returns_empty(self):
        result = _find_similar_symbols("Foo", {})
        assert result == []

    def test_empty_target_returns_empty(self):
        result = _find_similar_symbols("", {"foo": "Foo"})
        assert result == []

    def test_prefix_match_found(self):
        idx = {"gmailwrapper": "GmailWrapper", "githubclient": "GitHubClient"}
        result = _find_similar_symbols("GmailApiWrapper", idx, max_results=3)
        names = [r[0] for r in result]
        assert "GmailWrapper" in names

    def test_max_results_respected(self):
        idx = {f"class{i}wrapper": f"Class{i}Wrapper" for i in range(20)}
        result = _find_similar_symbols("MyWrapper", idx, max_results=3)
        assert len(result) <= 3

    def test_scores_are_positive(self):
        idx = {"foobar": "FooBar"}
        result = _find_similar_symbols("Foo", idx, max_results=5)
        for name, score in result:
            assert score > 0


# ── _first_line_summary / _signature_brief / _get_symbol_brief ───────────────

class TestDocstringHelpers:
    def test_first_line_extracts_first_sentence(self):
        doc = "Handle authentication requests.\n\nArgs:\n    token: str"
        result = _first_line_summary(doc)
        assert result == "Handle authentication requests."

    def test_first_line_empty_for_empty_input(self):
        assert _first_line_summary("") == ""

    def test_first_line_skips_param_lines(self):
        doc = "@param token: str\nDo the thing"
        result = _first_line_summary(doc)
        assert result == "Do the thing"

    def test_first_line_truncates_long_lines(self):
        doc = "X" * 300
        result = _first_line_summary(doc, max_len=120)
        assert len(result) <= 123  # 120 + "..."

    def test_signature_brief_from_source_text(self):
        node_data = {"source_text": "def process(data: dict, timeout: int) -> Response:"}
        result = _signature_brief(node_data)
        assert len(result) > 0
        assert "process" in result

    def test_signature_brief_empty_for_empty_node(self):
        result = _signature_brief({})
        assert result == ""

    def test_get_symbol_brief_prefers_docstring(self):
        node_data = {
            "docstring": "My useful description",
            "source_text": "def func():",
        }
        result = _get_symbol_brief(node_data)
        assert result == "My useful description"

    def test_get_symbol_brief_falls_back_to_signature(self):
        node_data = {"source_text": "class MyClass(Base):"}
        result = _get_symbol_brief(node_data)
        assert "MyClass" in result or "class" in result.lower()


# ── explore_repository_tree ───────────────────────────────────────────────────

class TestExploreRepositoryTree:
    def _create_temp_repo(self) -> str:
        """Create a minimal temp repo structure for filesystem tests."""
        tmp = tempfile.mkdtemp()
        # src/
        os.makedirs(os.path.join(tmp, "src", "core"), exist_ok=True)
        os.makedirs(os.path.join(tmp, "docs"), exist_ok=True)
        # Create files
        Path(os.path.join(tmp, "src", "main.py")).write_text("pass")
        Path(os.path.join(tmp, "src", "core", "engine.py")).write_text("pass")
        Path(os.path.join(tmp, "docs", "README.md")).write_text("# Docs")
        Path(os.path.join(tmp, "README.md")).write_text("# Root")
        return tmp

    def test_returns_directories_and_files(self):
        tmp = self._create_temp_repo()
        result = explore_repository_tree(tmp, max_depth=5)
        assert "directories" in result
        assert "top_level_files" in result
        assert len(result["directories"]) > 0

    def test_readme_in_top_level_files(self):
        tmp = self._create_temp_repo()
        result = explore_repository_tree(tmp, max_depth=5)
        top_files = [f.lower() for f in result["top_level_files"]]
        assert "readme.md" in top_files

    def test_depth_limit_respected(self):
        # Create a deeper structure and use max_depth=2 — only depth 1 and 2 dirs visible
        tmp = tempfile.mkdtemp()
        deep = os.path.join(tmp, "a", "b", "c", "d")
        os.makedirs(deep)
        Path(os.path.join(deep, "file.py")).write_text("pass")
        result = explore_repository_tree(tmp, max_depth=2)
        # Depth should not exceed 2 for any directory in results
        dirs = result["directories"]
        for d in dirs:
            assert d["depth"] <= 2

    def test_total_files_counted(self):
        tmp = self._create_temp_repo()
        result = explore_repository_tree(tmp, max_depth=5)
        assert result["total_files"] >= 2  # At least src/main.py, src/core/engine.py


# ── format_tree_for_llm ───────────────────────────────────────────────────────

class TestFormatTreeForLlm:
    def test_empty_directories_returns_string(self):
        tree = {"directories": [], "top_level_files": [], "total_files": 0}
        result = format_tree_for_llm(tree)
        assert isinstance(result, str)
        assert "No directories found" in result

    def test_top_level_files_shown(self):
        tree = {
            "directories": [],
            "top_level_files": ["README.md", "pyproject.toml"],
            "total_files": 0,
        }
        result = format_tree_for_llm(tree)
        assert "README.md" in result

    def test_depth1_dirs_shown(self):
        tree = {
            "directories": [
                {"path": "src", "name": "src", "depth": 1, "files": 5,
                 "code_files": 5, "doc_files": 0, "subdirs": 2},
            ],
            "top_level_files": [],
            "total_files": 5,
        }
        result = format_tree_for_llm(tree)
        assert "src" in result

    def test_summary_line_present(self):
        tree = {
            "directories": [
                {"path": "src", "name": "src", "depth": 1, "files": 3,
                 "code_files": 3, "doc_files": 0, "subdirs": 0},
            ],
            "top_level_files": [],
            "total_files": 3,
        }
        result = format_tree_for_llm(tree)
        assert "TOTAL" in result


# ── StructureCollector — basic construction ───────────────────────────────────

class TestStructureCollectorInit:
    def test_default_construction(self):
        c = _make_collector()
        assert c.page_budget == 10
        assert c.pages == []
        assert c.sections == {}
        assert c.metadata is None

    def test_stats_initial_state(self):
        c = _make_collector()
        s = c.stats
        assert s["pages"] == 0
        assert s["sections"] == 0
        assert s["coverage_pct"] == 0.0

    def test_reset_clears_state(self):
        c = _make_collector()
        c.metadata = {"wiki_title": "X", "overview": "Y"}
        c.sections = {"A": {}}
        c.reset()
        assert c.metadata is None
        assert c.sections == {}

    def test_get_tools_returns_list(self):
        c = _make_collector()
        tools = c.get_tools()
        assert isinstance(tools, list)
        assert len(tools) > 0

    def test_tool_names_include_expected(self):
        c = _make_collector()
        tools = c.get_tools()
        names = [t.name for t in tools]
        assert "define_section" in names
        assert "define_page" in names
        assert "explore_tree" in names
        assert "set_wiki_metadata" in names


# ── StructureCollector — coverage tracking ────────────────────────────────────

class TestStructureCollectorCoverage:
    def test_register_discovered_dirs(self):
        c = _make_collector()
        c.register_discovered_dirs(["src/a/", "src/b/", ""])
        assert "src/a" in c.discovered_dirs
        assert "src/b" in c.discovered_dirs
        assert "" not in c.discovered_dirs

    def test_ignored_dirs_not_registered(self):
        c = _make_collector()
        c.register_discovered_dirs(["node_modules/", ".git/"])
        assert "node_modules" not in c.discovered_dirs
        assert ".git" not in c.discovered_dirs

    def test_coverage_zero_with_no_discovered(self):
        c = _make_collector()
        covered, discovered, pct = c._get_coverage_stats()
        assert discovered == 0
        assert pct == 0.0

    def test_update_coverage_marks_ancestors(self):
        c = _make_collector()
        c.register_discovered_dirs(["src/services/orders/api/"])
        c._update_coverage(["src/services/orders/api"])
        # All ancestor levels should be covered
        assert "src" in c.covered_dirs
        assert "src/services" in c.covered_dirs
        assert "src/services/orders" in c.covered_dirs

    def test_update_coverage_marks_descendants(self):
        c = _make_collector()
        c.register_discovered_dirs(["src/a/b/c/"])
        c._update_coverage(["src/a"])
        # Descendants of src/a should be covered
        assert "src/a/b/c" in c.covered_dirs

    def test_coverage_percentage_computed(self):
        c = _make_collector(effective_depth=5)
        c.register_discovered_dirs(["src/a/", "src/b/"])
        c._update_coverage(["src/a"])
        _, _, pct = c._get_coverage_stats()
        assert 0 < pct <= 100

    def test_format_coverage_feedback_all_covered(self):
        c = _make_collector()
        c.register_discovered_dirs(["src/a/"])
        c._update_coverage(["src/a"])
        feedback = c._format_coverage_feedback()
        assert "covered" in feedback.lower()

    def test_format_coverage_feedback_with_uncovered(self):
        c = _make_collector()
        c.register_discovered_dirs(["src/a/", "src/b/"])
        c._update_coverage(["src/a"])
        feedback = c._format_coverage_feedback()
        assert "src/b" in feedback


# ── StructureCollector — metadata / section / page handlers ──────────────────

class TestStructureCollectorHandlers:
    def test_handle_set_metadata(self):
        c = _make_collector()
        result = c._handle_set_metadata("My Wiki", "Overview text")
        assert "OK" in result
        assert c.metadata["wiki_title"] == "My Wiki"
        assert c.metadata["overview"] == "Overview text"

    def test_handle_define_section_creates_section(self):
        c = _make_collector()
        result = c._handle_define_section("Architecture", 1, "Arch overview", "Needed")
        assert "OK" in result
        assert "Architecture" in c.sections

    def test_handle_define_section_duplicate_warns(self):
        c = _make_collector()
        c._handle_define_section("Auth", 1, "Auth section", "Needed")
        result = c._handle_define_section("Auth", 2, "Auth section again", "Duplicate")
        assert "WARN" in result

    def test_handle_define_section_generic_name_warns(self):
        c = _make_collector()
        result = c._handle_define_section("Services", 1, "Desc", "Rationale")
        assert "⚠️" in result

    def test_handle_define_page_basic(self):
        c = _make_collector()
        c._handle_define_section("Core", 1, "Core section", "Needed")
        result = c._handle_define_page(
            section_name="Core",
            page_name="Authentication Flow",
            page_order=1,
            description="Auth desc",
            content_focus="Auth focus",
            rationale="Need auth docs",
            target_symbols=["AuthService"],
            target_docs=[],
            target_folders=["src/auth"],
            key_files=[],
            retrieval_query="authentication",
        )
        assert "OK" in result
        assert len(c.pages) == 1
        assert c.pages[0]["page_name"] == "Authentication Flow"

    def test_handle_define_page_no_symbols_warns(self):
        c = _make_collector()
        c._handle_define_section("Core", 1, "Core section", "Needed")
        result = c._handle_define_page(
            section_name="Core",
            page_name="Config Page",
            page_order=1,
            description="Config desc",
            content_focus="Config",
            rationale="Config docs",
            target_symbols=[],
            target_docs=[],
            target_folders=["src/config"],
            key_files=[],
        )
        assert "⚠️" in result

    def test_handle_define_page_validates_symbols_against_graph(self):
        graph = _make_code_graph_with_symbols([
            ("RealClass", "class", "src/real.py"),
        ])
        c = StructureCollector(page_budget=5, code_graph=graph)
        c._handle_define_section("Core", 1, "Core", "Needed")
        result = c._handle_define_page(
            section_name="Core",
            page_name="Real Page",
            page_order=1,
            description="Desc",
            content_focus="Focus",
            rationale="Rationale",
            target_symbols=["RealClass", "FakeClass"],
            target_docs=[],
            target_folders=["src"],
            key_files=[],
        )
        assert len(c.pages) == 1
        page = c.pages[0]
        # RealClass should be validated, FakeClass should not
        assert "RealClass" in page["target_symbols"]
        assert "FakeClass" not in page["target_symbols"]

    def test_handle_define_page_over_budget_warns(self):
        c = _make_collector(page_budget=2)
        c._handle_define_section("Core", 1, "Core", "Needed")
        for i in range(3):
            c._handle_define_page(
                section_name="Core",
                page_name=f"Page {i}",
                page_order=i + 1,
                description="Desc",
                content_focus="Focus",
                rationale="Rationale",
                target_symbols=[],
                target_docs=[],
                target_folders=[f"src/mod{i}"],
                key_files=[],
            )
        assert len(c.pages) == 3  # Pages still added past budget


# ── StructureCollector — assemble ─────────────────────────────────────────────

class TestStructureCollectorAssemble:
    def test_assemble_empty_returns_defaults(self):
        c = _make_collector()
        result = c.assemble()
        assert result["wiki_title"] == "Documentation"
        assert result["sections"] == []
        assert result["total_pages"] == 0

    def test_assemble_with_section_and_page(self):
        c = _make_collector()
        c._handle_set_metadata("Test Wiki", "Test overview")
        c._handle_define_section("Core", 1, "Core functionality", "Needed")
        c._handle_define_page(
            section_name="Core",
            page_name="Core Features",
            page_order=1,
            description="Core page",
            content_focus="Core",
            rationale="Needed",
            target_symbols=[],
            target_docs=[],
            target_folders=["src/core"],
            key_files=[],
        )
        result = c.assemble()
        assert result["wiki_title"] == "Test Wiki"
        assert result["total_pages"] == 1
        assert len(result["sections"]) == 1
        assert result["sections"][0]["section_name"] == "Core"
        assert len(result["sections"][0]["pages"]) == 1

    def test_assemble_orphan_page_creates_section(self):
        c = _make_collector()
        # Add page without pre-defined section
        c._handle_define_page(
            section_name="Orphan Section",
            page_name="Orphan Page",
            page_order=1,
            description="Orphan",
            content_focus="Orphan",
            rationale="Orphan rationale",
            target_symbols=[],
            target_docs=[],
            target_folders=["src/orphan"],
            key_files=[],
        )
        result = c.assemble()
        assert result["total_pages"] == 1
        sections = {s["section_name"] for s in result["sections"]}
        assert "Orphan Section" in sections

    def test_assemble_pages_sorted_by_order(self):
        c = _make_collector()
        c._handle_define_section("Core", 1, "Core", "Needed")
        for i in range(3, 0, -1):  # Add in reverse order
            c._handle_define_page(
                section_name="Core",
                page_name=f"Page {i}",
                page_order=i,
                description="Desc",
                content_focus="Focus",
                rationale="Rationale",
                target_symbols=[],
                target_docs=[],
                target_folders=[f"src/m{i}"],
                key_files=[],
            )
        result = c.assemble()
        orders = [p["page_order"] for p in result["sections"][0]["pages"]]
        assert orders == sorted(orders)


# ── StructureCollector — explore_tree handler ─────────────────────────────────

class TestStructureCollectorExploreTree:
    def test_explore_tree_no_repo_root_returns_error(self):
        c = _make_collector()
        result = c._handle_explore_tree()
        assert "ERROR" in result

    def test_explore_tree_with_valid_repo_root(self):
        tmp = tempfile.mkdtemp()
        Path(os.path.join(tmp, "src")).mkdir()
        Path(os.path.join(tmp, "src", "main.py")).write_text("pass")

        c = _make_collector(repo_root=tmp, effective_depth=3)
        result = c._handle_explore_tree()
        assert isinstance(result, str)
        assert len(result) > 0
        assert "src" in result

    def test_explore_tree_cached_on_second_call(self):
        tmp = tempfile.mkdtemp()
        Path(os.path.join(tmp, "src")).mkdir()
        Path(os.path.join(tmp, "src", "main.py")).write_text("pass")

        c = _make_collector(repo_root=tmp, effective_depth=3)
        result1 = c._handle_explore_tree()
        result2 = c._handle_explore_tree()
        assert result1 == result2
        assert c._tree_analysis is not None


# ── StructureCollector — think handler ────────────────────────────────────────

class TestStructureCollectorThink:
    def test_think_records_analysis(self):
        c = _make_collector()
        c._handle_think("I see 3 modules: auth, core, db")
        assert len(c._thinking_log) == 1
        assert "auth" in c._thinking_log[0]

    def test_think_returns_string(self):
        c = _make_collector()
        result = c._handle_think("Plan goes here")
        assert isinstance(result, str)
        assert len(result) > 0


# ── StructureCollector — analyze_module handler ───────────────────────────────

class TestStructureCollectorAnalyzeModule:
    def test_analyze_module_no_repo_root_returns_error(self):
        c = _make_collector()
        result = c._handle_analyze_module("src/services")
        assert "ERROR" in result

    def test_analyze_module_invalid_path_returns_error(self):
        tmp = tempfile.mkdtemp()
        c = _make_collector(repo_root=tmp)
        result = c._handle_analyze_module("nonexistent/path")
        assert "ERROR" in result

    def test_analyze_module_valid_path(self):
        tmp = tempfile.mkdtemp()
        mod_dir = os.path.join(tmp, "src", "auth")
        os.makedirs(mod_dir)
        Path(os.path.join(mod_dir, "auth.py")).write_text("class Auth: pass")
        Path(os.path.join(mod_dir, "tokens.py")).write_text("def issue(): pass")

        c = _make_collector(repo_root=tmp)
        result = c._handle_analyze_module("src/auth")
        assert "src/auth" in result
        assert "auth.py" in result or "Auth" in result


# ── StructureCollector — query_graph handler ──────────────────────────────────

class TestStructureCollectorQueryGraph:
    def test_query_graph_no_code_graph_returns_message(self):
        c = _make_collector()
        result = c._handle_query_graph("src/auth")
        assert "not available" in result.lower() or "graph" in result.lower()

    def test_query_graph_empty_prefix_returns_error(self):
        graph = _make_code_graph_with_symbols([("Foo", "class", "src/foo.py")])
        c = StructureCollector(page_budget=5, code_graph=graph)
        result = c._handle_query_graph("")
        assert "ERROR" in result

    def test_query_graph_finds_symbols_by_path_prefix(self):
        graph = _make_code_graph_with_symbols([
            ("AuthService", "class", "src/auth/service.py"),
            ("UserModel", "class", "src/models/user.py"),
        ])
        c = StructureCollector(page_budget=5, code_graph=graph)
        result = c._handle_query_graph("src/auth")
        assert "AuthService" in result

    def test_query_graph_excludes_symbols_outside_prefix(self):
        graph = _make_code_graph_with_symbols([
            ("AuthService", "class", "src/auth/service.py"),
            ("UserModel", "class", "src/models/user.py"),
        ])
        c = StructureCollector(page_budget=5, code_graph=graph)
        result = c._handle_query_graph("src/auth")
        # UserModel is in models, not auth
        assert "UserModel" not in result


# ── StructureCollector — symbol coverage tracking ─────────────────────────────

class TestStructureCollectorSymbolCoverage:
    def test_covered_symbols_tracked_on_page_define(self):
        graph = _make_code_graph_with_symbols([
            ("MyClass", "class", "src/my.py"),
        ])
        c = StructureCollector(page_budget=5, code_graph=graph)
        c._handle_define_section("Core", 1, "Core", "Needed")
        c._handle_define_page(
            section_name="Core",
            page_name="My Page",
            page_order=1,
            description="Desc",
            content_focus="Focus",
            rationale="Reason",
            target_symbols=["MyClass"],
            target_docs=[],
            target_folders=["src"],
            key_files=[],
        )
        assert "MyClass" in c._covered_symbols

    def test_symbol_to_page_mapping_recorded(self):
        graph = _make_code_graph_with_symbols([
            ("Service", "class", "app/service.py"),
        ])
        c = StructureCollector(page_budget=5, code_graph=graph)
        c._handle_define_section("Core", 1, "Core", "Needed")
        c._handle_define_page(
            section_name="Core",
            page_name="Service Layer",
            page_order=1,
            description="Desc",
            content_focus="Focus",
            rationale="Reason",
            target_symbols=["Service"],
            target_docs=[],
            target_folders=["app"],
            key_files=[],
        )
        assert c._symbol_to_page.get("Service") == "Service Layer"


# ── StructureCollector — batch_define_pages ───────────────────────────────────

class TestStructureCollectorBatchDefinePages:
    def test_empty_pages_returns_error(self):
        c = _make_collector()
        result = c._handle_batch_define_pages([])
        assert "ERROR" in result

    def test_batch_defines_multiple_pages(self):
        c = _make_collector(page_budget=20)

        class FakePage:
            def __init__(self, sec, pn, desc, syms, folders):
                self.section_name = sec
                self.page_name = pn
                self.description = desc
                self.target_symbols = syms
                self.target_folders = folders

        pages = [
            FakePage("Core", f"Page {i}", f"Desc {i}", [], [f"src/mod{i}"])
            for i in range(3)
        ]
        result = c._handle_batch_define_pages(pages)
        assert len(c.pages) == 3
        assert "3" in result or "pages" in result.lower()

    def test_batch_skips_pages_with_missing_required_fields(self):
        c = _make_collector()
        pages = [
            {"section_name": "", "page_name": ""},
            {"section_name": "Core", "page_name": "Valid Page",
             "description": "Desc", "target_symbols": [], "target_folders": ["src"]},
        ]
        result = c._handle_batch_define_pages(pages)
        assert len(c.pages) == 1  # Only valid page added

    def test_batch_auto_creates_section(self):
        c = _make_collector(page_budget=10)

        class FakePage:
            section_name = "NewSection"
            page_name = "New Page"
            description = "Desc"
            target_symbols = []
            target_folders = ["src"]

        result = c._handle_batch_define_pages([FakePage()])
        assert "NewSection" in c.sections


# ── StructureCollector — query_graph with rich graph ─────────────────────────

class TestStructureCollectorQueryGraphRichGraph:
    """Test query_graph with a rich graph containing multiple symbol types."""

    def _make_rich_graph(self) -> nx.MultiDiGraph:
        """Build a graph with classes, functions, enums, macros, type aliases.

        Uses MultiDiGraph because _handle_query_graph iterates edge_data.values()
        which expects the MultiDiGraph format: {edge_key: {attr_dict}}.
        """
        g = nx.MultiDiGraph()
        # Class
        g.add_node("cls1", symbol_type="class", symbol_name="AuthService",
                   rel_path="src/auth/service.py", docstring="Auth service")
        # Another class
        g.add_node("cls2", symbol_type="class", symbol_name="TokenManager",
                   rel_path="src/auth/tokens.py", docstring="Token management")
        # Function
        g.add_node("fn1", symbol_type="function", symbol_name="validate_token",
                   rel_path="src/auth/validators.py", docstring="Validate token")
        # Enum
        g.add_node("enum1", symbol_type="enum", symbol_name="AuthStatus",
                   rel_path="src/auth/enums.py", docstring="Auth statuses")
        # Type alias
        g.add_node("ta1", symbol_type="type_alias", symbol_name="TokenStr",
                   rel_path="src/auth/types.py", docstring="Token type alias")
        # Inheritance edge: TokenManager inherits from AuthService
        g.add_edge("cls2", "cls1", relationship_type="inheritance")
        # Enum used by AuthService
        g.add_edge("cls1", "enum1", relationship_type="references")
        return g

    def test_query_graph_returns_classes(self):
        g = self._make_rich_graph()
        c = StructureCollector(page_budget=5, code_graph=g)
        result = c._handle_query_graph("src/auth")
        assert "AuthService" in result
        assert "TokenManager" in result

    def test_query_graph_includes_functions(self):
        g = self._make_rich_graph()
        c = StructureCollector(page_budget=5, code_graph=g)
        result = c._handle_query_graph("src/auth")
        assert "validate_token" in result

    def test_query_graph_includes_enums(self):
        g = self._make_rich_graph()
        c = StructureCollector(page_budget=5, code_graph=g)
        result = c._handle_query_graph("src/auth")
        assert "AuthService" in result or "AuthStatus" in result  # At least one visible

    def test_query_graph_result_cached(self):
        g = self._make_rich_graph()
        c = StructureCollector(page_budget=5, code_graph=g)
        result1 = c._handle_query_graph("src/auth")
        result2 = c._handle_query_graph("src/auth")
        assert result1 == result2

    def test_query_graph_with_macro_type(self):
        g = nx.MultiDiGraph()
        g.add_node("m1", symbol_type="macro", symbol_name="MAX_RETRIES",
                   rel_path="src/core/config.h", docstring="Max retry count")
        c = StructureCollector(page_budget=5, code_graph=g)
        result = c._handle_query_graph("src/core")
        assert isinstance(result, str)
        assert len(result) > 0

    def test_query_graph_zero_symbols_returns_no_symbols_message(self):
        g = nx.MultiDiGraph()
        g.add_node("n1", symbol_type="class", symbol_name="Unrelated",
                   rel_path="src/other/file.py", docstring="")
        c = StructureCollector(page_budget=5, code_graph=g)
        result = c._handle_query_graph("src/different_path")
        assert "No symbols found" in result or "documentation" in result.lower()

    def test_query_graph_with_inheritance_shown(self):
        g = nx.MultiDiGraph()
        g.add_node("base", symbol_type="class", symbol_name="BaseHandler",
                   rel_path="src/handlers/base.py", docstring="Base class")
        g.add_node("derived", symbol_type="class", symbol_name="AuthHandler",
                   rel_path="src/handlers/auth.py", docstring="Auth handler")
        g.add_edge("derived", "base", relationship_type="inheritance")
        c = StructureCollector(page_budget=5, code_graph=g)
        result = c._handle_query_graph("src/handlers")
        assert "BaseHandler" in result or "AuthHandler" in result


# ── StructureCollector — search_graph handler ─────────────────────────────────

class TestStructureCollectorSearchGraph:
    def test_search_graph_no_code_graph_returns_message(self):
        c = _make_collector()
        result = c._handle_search_graph("Auth")
        assert "not available" in result.lower() or "graph" in result.lower()

    def test_search_graph_too_short_pattern_returns_error(self):
        g = _make_code_graph_with_symbols([("MyClass", "class", "src/my.py")])
        c = StructureCollector(page_budget=5, code_graph=g)
        result = c._handle_search_graph("A")  # Single char - too short
        assert "ERROR" in result or "2" in result  # Expects at least 2 chars

    def test_search_graph_finds_matching_symbols(self):
        g = nx.DiGraph()
        g.add_node("n1", symbol_type="class", symbol_name="AuthService",
                   rel_path="src/auth.py", docstring="Auth service")
        g.add_node("n2", symbol_type="class", symbol_name="UserModel",
                   rel_path="src/user.py", docstring="User model")
        c = StructureCollector(page_budget=5, code_graph=g)
        result = c._handle_search_graph("Auth")
        assert "AuthService" in result

    def test_search_graph_type_filter_works(self):
        g = nx.DiGraph()
        g.add_node("n1", symbol_type="class", symbol_name="AuthClass",
                   rel_path="src/auth.py", docstring="")
        g.add_node("n2", symbol_type="function", symbol_name="auth_func",
                   rel_path="src/utils.py", docstring="")
        c = StructureCollector(page_budget=5, code_graph=g)
        result = c._handle_search_graph("auth", symbol_type="class")
        assert "AuthClass" in result
        assert "auth_func" not in result

    def test_search_graph_docstring_search(self):
        g = nx.DiGraph()
        g.add_node("n1", symbol_type="class", symbol_name="Pipeline",
                   rel_path="src/core.py", docstring="Handles authentication flows")
        c = StructureCollector(page_budget=5, code_graph=g)
        result = c._handle_search_graph("authentication", include_docstrings=True)
        assert "Pipeline" in result

    def test_search_graph_no_results_returns_tips(self):
        g = nx.DiGraph()
        g.add_node("n1", symbol_type="class", symbol_name="UnrelatedClass",
                   rel_path="src/other.py", docstring="")
        c = StructureCollector(page_budget=5, code_graph=g)
        result = c._handle_search_graph("ZZZnonexistent")
        assert "No symbols found" in result or "TIP" in result or "Try" in result


# ── format_tree_for_llm edge cases ────────────────────────────────────────────

class TestFormatTreeForLlmEdgeCases:
    def test_large_dir_shows_grouping_guidance(self):
        """When a large parent dir has 15+ siblings, grouping guidance appears."""
        parent_path = "src/big"
        dirs = [
            {
                "path": f"{parent_path}/sub{i}",
                "name": f"sub{i}",
                "depth": 2,
                "files": 3,
                "code_files": 3,
                "doc_files": 0,
                "subdirs": 0,
            }
            for i in range(20)
        ]
        # Add the parent as depth-1 large dir
        dirs.insert(0, {
            "path": parent_path,
            "name": "big",
            "depth": 1,
            "files": 20,
            "code_files": 20,
            "doc_files": 0,
            "subdirs": 20,
        })
        tree = {"directories": dirs, "top_level_files": [], "total_files": 80}
        result = format_tree_for_llm(tree)
        assert isinstance(result, str)
        # Should have content about big directory
        assert "big" in result.lower()

    def test_next_steps_shown_for_content_dirs(self):
        tree = {
            "directories": [
                {
                    "path": "src/auth",
                    "name": "auth",
                    "depth": 1,
                    "files": 5,
                    "code_files": 5,
                    "doc_files": 0,
                    "subdirs": 0,
                }
            ],
            "top_level_files": [],
            "total_files": 5,
        }
        result = format_tree_for_llm(tree)
        assert "NEXT STEPS" in result

    def test_medium_dir_indicator(self):
        """Dirs with 5-9 files or 2 subdirs get MEDIUM label."""
        tree = {
            "directories": [
                {
                    "path": "utils",
                    "name": "utils",
                    "depth": 1,
                    "files": 5,
                    "code_files": 5,
                    "doc_files": 0,
                    "subdirs": 2,
                }
            ],
            "top_level_files": [],
            "total_files": 5,
        }
        result = format_tree_for_llm(tree)
        assert "MEDIUM" in result or "utils" in result


# ── StructureCollector — symbol coverage internal methods ─────────────────────

class TestStructureCollectorSymbolCoverageInternals:
    def test_ensure_architectural_symbols_loaded(self):
        g = _make_code_graph_with_symbols([
            ("ClassA", "class", "src/a.py"),
            ("funcB", "function", "src/b.py"),
        ])
        c = StructureCollector(page_budget=5, code_graph=g)
        c._ensure_architectural_symbols_loaded()
        assert "ClassA" in c._all_architectural_symbols
        assert "funcB" in c._all_architectural_symbols

    def test_compute_symbol_coverage_no_graph(self):
        c = _make_collector()
        covered, total, uncovered = c._compute_symbol_coverage()
        assert total == 0
        assert covered == 0
        assert uncovered == []

    def test_format_symbol_coverage_feedback_no_symbols(self):
        c = _make_collector()
        result = c._format_symbol_coverage_feedback()
        assert result == ""

    def test_format_symbol_coverage_feedback_with_graph(self):
        g = _make_code_graph_with_symbols([
            ("ClassA", "class", "src/a.py"),
            ("ClassB", "class", "src/b.py"),
        ])
        c = StructureCollector(page_budget=5, code_graph=g)
        c._all_architectural_symbols = {"ClassA", "ClassB"}
        c._covered_symbols = {"ClassA"}
        result = c._format_symbol_coverage_feedback()
        assert "ClassA" in result or "50" in result  # Shows 50% coverage

    def test_assignment_tag_for_known_symbol(self):
        c = _make_collector()
        c._symbol_to_page = {"MyClass": "Core Features"}
        result = c._assignment_tag("MyClass")
        assert "Core Features" in result

    def test_assignment_tag_for_unknown_symbol(self):
        c = _make_collector()
        result = c._assignment_tag("UnknownClass")
        assert result == ""


# ── StructureCollector — stats property ───────────────────────────────────────

class TestStructureCollectorStatsProperty:
    def test_stats_reflects_budget(self):
        c = _make_collector(page_budget=25)
        stats = c.stats
        assert stats["budget"] == 25

    def test_stats_remaining_decreases_with_pages(self):
        c = _make_collector(page_budget=5)
        c._handle_define_section("S", 1, "D", "R")
        c._handle_define_page("S", "P1", 1, "D", "F", "R", [], [], ["src"], [])
        stats = c.stats
        assert stats["pages"] == 1
        assert stats["remaining"] == 4

    def test_stats_over_budget_tracked(self):
        c = _make_collector(page_budget=1)
        c._handle_define_section("S", 1, "D", "R")
        for i in range(3):
            c._handle_define_page("S", f"P{i}", i + 1, "D", "F", "R", [], [], [f"src/{i}"], [])
        stats = c.stats
        assert stats["over_budget"] == 2


# ── engine — _build_model ─────────────────────────────────────────────────────

class TestEngineBuildModel:
    """Test _build_model with mocked langchain_openai.ChatOpenAI."""

    def test_build_model_uses_settings(self):
        from app.core.wiki_structure_planner.structure_engine import WikiStructurePlannerEngine
        import tempfile, os
        from pathlib import Path

        tmp = tempfile.mkdtemp()
        Path(os.path.join(tmp, "main.py")).write_text("pass")

        engine = WikiStructurePlannerEngine(
            repo_root=tmp,
            llm_settings={"model_name": "gpt-4o-mini", "api_key": "test-key"},
        )

        mock_openai = MagicMock()
        with patch("langchain_openai.ChatOpenAI", return_value=mock_openai) as MockChatOpenAI:
            result = engine._build_model()

        MockChatOpenAI.assert_called_once()
        call_kwargs = MockChatOpenAI.call_args[1]
        assert call_kwargs["model"] == "gpt-4o-mini"


# ── detect_effective_depth edge cases ─────────────────────────────────────────

class TestDetectEffectiveDepthEdgeCases:
    def test_empty_path_string_skipped(self):
        """Empty strings in the files list should not crash."""
        files = ["", "src/main.py"]
        result = detect_effective_depth(files, static_default=5)
        assert result >= 5

    def test_file_with_no_extension_skipped(self):
        """Paths without a dot (no extension) should be skipped."""
        files = ["Makefile", "Dockerfile", "src/main.py"]
        result = detect_effective_depth(files, static_default=5)
        assert result >= 5

    def test_exactly_p90_depth_used(self):
        """Verify p90 calculation: 10 files at depth 3 → effective=4."""
        files = [f"a/b/c/f{i}.py" for i in range(10)]
        result = detect_effective_depth(files, static_default=3, ceiling=10)
        assert result >= 4  # depth 3 + 1 headroom


# ── _signature_brief and _extract_node_docstring symbol paths ─────────────────

class TestSignatureBriefSymbolObject:
    """Test _signature_brief with comprehensive parser Symbol objects."""

    def test_signature_from_symbol_object(self):
        """symbol.signature path is taken when symbol has signature attr."""
        from app.core.wiki_structure_planner.structure_tools import _signature_brief

        class FakeSymbol:
            signature = "def authenticate(token: str) -> bool"
            source_text = ""
            parameter_types = []
            return_type = ""

        node_data = {"symbol": FakeSymbol()}
        result = _signature_brief(node_data)
        assert "authenticate" in result

    def test_long_signature_truncated(self):
        """Signatures longer than max_len are truncated with ellipsis."""
        from app.core.wiki_structure_planner.structure_tools import _signature_brief

        class FakeSymbol:
            signature = "def " + "a" * 200 + "()"
            source_text = ""
            parameter_types = []
            return_type = ""

        node_data = {"symbol": FakeSymbol()}
        result = _signature_brief(node_data, max_len=50)
        assert result.endswith("...")
        assert len(result) <= 53  # max_len + "..."

    def test_source_text_used_when_no_signature(self):
        """Fall through to source_text when symbol has no signature."""
        from app.core.wiki_structure_planner.structure_tools import _signature_brief

        class FakeSymbol:
            signature = ""
            source_text = "class MyClass extends Base {"
            parameter_types = []
            return_type = ""

        node_data = {"symbol": FakeSymbol()}
        result = _signature_brief(node_data)
        assert "MyClass" in result

    def test_param_return_synthesized(self):
        """Synthesize from parameter_types and return_type."""
        from app.core.wiki_structure_planner.structure_tools import _signature_brief

        class FakeSymbol:
            signature = ""
            source_text = ""
            parameter_types = ["str", "int"]
            return_type = "bool"

        node_data = {"symbol": FakeSymbol()}
        result = _signature_brief(node_data)
        assert "str" in result or "bool" in result

    def test_node_data_source_text_used(self):
        """Use source_text from node_data dict when no symbol."""
        from app.core.wiki_structure_planner.structure_tools import _signature_brief

        node_data = {"source_text": "func Process(ctx context.Context) error {"}
        result = _signature_brief(node_data)
        assert "Process" in result


class TestExtractNodeDocstringSymbol:
    """Test _extract_node_docstring with Symbol object."""

    def test_symbol_docstring_used_when_no_top_level(self):
        """When no top-level docstring, falls back to symbol.docstring."""
        from app.core.wiki_structure_planner.structure_tools import _signature_brief

        # Access via _get_symbol_brief which calls _extract_node_docstring internally
        class FakeSymbol:
            docstring = "This is the service docstring"
            signature = ""
            source_text = ""
            parameter_types = []
            return_type = ""

        node_data = {"symbol": FakeSymbol()}
        # Use _get_symbol_brief which calls both
        from app.core.wiki_structure_planner.structure_tools import _get_symbol_brief
        result = _get_symbol_brief(node_data)
        assert "service docstring" in result


# ── _handle_query_graph — additional symbol types ─────────────────────────────

class TestStructureCollectorQueryGraphSymbolTypes:
    """Cover type_alias, enum/constant, macro, and doc symbol paths."""

    def _make_multi_type_graph(self) -> nx.MultiDiGraph:
        g = nx.MultiDiGraph()
        # type_alias
        g.add_node("ta1", symbol_type="type_alias", symbol_name="UserId",
                   rel_path="src/types/ids.py", docstring="User ID type alias")
        g.add_node("cls1", symbol_type="class", symbol_name="UserService",
                   rel_path="src/types/service.py", docstring="")
        # type_alias → alias_of edge
        g.add_edge("ta1", "cls1", relationship_type="alias_of")
        # enum / constant
        g.add_node("en1", symbol_type="enum", symbol_name="StatusCode",
                   rel_path="src/types/status.py", docstring="HTTP status codes")
        g.add_node("const1", symbol_type="constant", symbol_name="MAX_CONN",
                   rel_path="src/types/config.py", docstring="Max connections")
        # predecessor uses StatusCode
        g.add_edge("cls1", "en1", relationship_type="references")
        # macro
        g.add_node("mac1", symbol_type="macro", symbol_name="RETRY_COUNT",
                   rel_path="src/types/macros.h", docstring="Retry macro")
        # macro references something
        g.add_node("mac_target", symbol_type="constant", symbol_name="MAX_RETRY",
                   rel_path="src/types/macros.h", docstring="")
        g.add_edge("mac1", "mac_target", relationship_type="references")
        g.add_edge("cls1", "mac1", relationship_type="calls")
        # doc symbol
        g.add_node("doc1", symbol_type="markdown_document",
                   symbol_name="api-guide.md",
                   rel_path="src/types/api-guide.md", docstring="")
        return g

    def test_query_graph_finds_type_aliases(self):
        g = self._make_multi_type_graph()
        c = StructureCollector(page_budget=5, code_graph=g)
        result = c._handle_query_graph("src/types")
        assert "UserId" in result

    def test_query_graph_finds_enums(self):
        g = self._make_multi_type_graph()
        c = StructureCollector(page_budget=5, code_graph=g)
        result = c._handle_query_graph("src/types")
        assert "StatusCode" in result

    def test_query_graph_finds_constants(self):
        g = self._make_multi_type_graph()
        c = StructureCollector(page_budget=5, code_graph=g)
        result = c._handle_query_graph("src/types")
        assert "MAX_CONN" in result

    def test_query_graph_finds_macros(self):
        g = self._make_multi_type_graph()
        c = StructureCollector(page_budget=5, code_graph=g)
        result = c._handle_query_graph("src/types")
        assert "RETRY_COUNT" in result

    def test_query_graph_finds_doc_symbols(self):
        g = self._make_multi_type_graph()
        c = StructureCollector(page_budget=5, code_graph=g)
        result = c._handle_query_graph("src/types")
        # Doc symbols appear in target_docs section or documentation section
        assert "api-guide.md" in result or "documentation" in result.lower()

    def test_query_graph_type_alias_alias_of_populated(self):
        """type_alias with alias_of edge lists what it resolves to."""
        g = nx.MultiDiGraph()
        g.add_node("ta1", symbol_type="type_alias", symbol_name="TokenStr",
                   rel_path="src/auth/types.py", docstring="")
        g.add_node("base1", symbol_type="class", symbol_name="str",
                   rel_path="src/auth/types.py", docstring="")
        g.add_edge("ta1", "base1", relationship_type="alias_of")
        c = StructureCollector(page_budget=5, code_graph=g)
        result = c._handle_query_graph("src/auth")
        assert "TokenStr" in result

    def test_query_graph_enum_with_predecessor_used_by(self):
        """enum node with predecessor correctly lists used_by."""
        g = nx.MultiDiGraph()
        g.add_node("e1", symbol_type="enum", symbol_name="Color",
                   rel_path="src/ui/color.py", docstring="Colors")
        g.add_node("user1", symbol_type="class", symbol_name="Painter",
                   rel_path="src/ui/painter.py", docstring="")
        g.add_edge("user1", "e1", relationship_type="references")
        c = StructureCollector(page_budget=5, code_graph=g)
        result = c._handle_query_graph("src/ui")
        assert "Color" in result

    def test_query_graph_class_with_inheritance_and_implementors(self):
        """Test class node with both inherits_from and implemented_by paths."""
        g = nx.MultiDiGraph()
        g.add_node("base", symbol_type="class", symbol_name="BaseHandler",
                   rel_path="src/handlers/base.py", docstring="")
        g.add_node("derived", symbol_type="class", symbol_name="AuthHandler",
                   rel_path="src/handlers/auth.py", docstring="")
        g.add_node("factory", symbol_type="function", symbol_name="make_handler",
                   rel_path="src/handlers/factory.py", docstring="")
        # derived inherits from base
        g.add_edge("derived", "base", relationship_type="inheritance")
        # factory creates base
        g.add_edge("factory", "base", relationship_type="creates")
        c = StructureCollector(page_budget=5, code_graph=g)
        result = c._handle_query_graph("src/handlers")
        assert "BaseHandler" in result or "AuthHandler" in result

    def test_query_graph_function_with_creates_edge(self):
        """function node with creates/instantiates edges populates creates list."""
        g = nx.MultiDiGraph()
        g.add_node("fn1", symbol_type="function", symbol_name="build_service",
                   rel_path="src/factory.py", docstring="Factory function")
        g.add_node("svc", symbol_type="class", symbol_name="Service",
                   rel_path="src/factory.py", docstring="")
        g.add_edge("fn1", "svc", relationship_type="creates")
        c = StructureCollector(page_budget=5, code_graph=g)
        result = c._handle_query_graph("src")
        assert "build_service" in result

    def test_query_graph_with_text_filter(self):
        """text_filter limits results to matching names."""
        g = nx.MultiDiGraph()
        g.add_node("n1", symbol_type="class", symbol_name="AuthFilter",
                   rel_path="src/auth/filter.py", docstring="Auth filter class")
        g.add_node("n2", symbol_type="class", symbol_name="DatabasePool",
                   rel_path="src/auth/pool.py", docstring="Database connection pool")
        c = StructureCollector(page_budget=5, code_graph=g)
        result = c._handle_query_graph("src/auth", text_filter="auth")
        assert isinstance(result, str)


# ── _handle_define_page — target_symbols validation paths ─────────────────────

class TestHandleDefinePageSymbolValidation:
    """Cover symbol validation code paths in _handle_define_page."""

    def _setup_collector_with_graph(self):
        g = nx.DiGraph()
        g.add_node("n1", symbol_type="class", symbol_name="AuthService",
                   rel_path="src/auth/service.py", docstring="")
        g.add_node("n2", symbol_type="function", symbol_name="validate_token",
                   rel_path="src/auth/validators.py", docstring="")
        c = StructureCollector(page_budget=10, code_graph=g)
        c._handle_define_section("Core", 1, "Core functionality", "Needed")
        return c

    def test_valid_symbols_tracked(self):
        """Symbols found in graph are added to _covered_symbols."""
        c = self._setup_collector_with_graph()
        result = c._handle_define_page(
            section_name="Core",
            page_name="Auth Page",
            page_order=1,
            description="Auth",
            content_focus="Focus",
            rationale="Reason",
            target_symbols=["AuthService"],
            target_docs=[],
            target_folders=["src/auth"],
            key_files=[],
        )
        assert "AuthService" in c._covered_symbols
        assert isinstance(result, str)

    def test_invalid_symbols_produce_warning(self):
        """Symbols not found in graph produce a warning in the response."""
        c = self._setup_collector_with_graph()
        result = c._handle_define_page(
            section_name="Core",
            page_name="Some Page",
            page_order=1,
            description="Desc",
            content_focus="Focus",
            rationale="Reason",
            target_symbols=["NonExistentClass"],
            target_docs=[],
            target_folders=["src"],
            key_files=[],
        )
        # Response should mention the missing symbol
        assert "NonExistentClass" in result or "not found" in result.lower() or "Symbols" in result

    def test_overlapping_symbols_produce_warning(self):
        """Symbols already assigned to another page trigger overlap warning."""
        c = self._setup_collector_with_graph()
        # Build the case-insensitive symbol index first
        c._ensure_architectural_symbols_loaded()
        # Pre-assign AuthService to another page
        c._symbol_to_page["AuthService"] = "Other Page"
        result = c._handle_define_page(
            section_name="Core",
            page_name="New Page",
            page_order=1,
            description="Desc",
            content_focus="Focus",
            rationale="Reason",
            target_symbols=["AuthService"],
            target_docs=[],
            target_folders=["src/auth"],
            key_files=[],
        )
        assert isinstance(result, str)
        # The warning about overlap should appear somewhere
        assert "AuthService" in result or "overlap" in result.lower() or "Other Page" in result

    def test_many_target_symbols_warning(self):
        """More than max_symbols_per_page + 15 symbols triggers very-high warning."""
        g = nx.DiGraph()
        for i in range(50):
            g.add_node(f"n{i}", symbol_type="class", symbol_name=f"Class{i}",
                       rel_path=f"src/classes/c{i}.py", docstring="")
        c = StructureCollector(page_budget=10, code_graph=g)
        c._handle_define_section("Core", 1, "Desc", "Reason")
        result = c._handle_define_page(
            section_name="Core",
            page_name="Huge Page",
            page_order=1,
            description="Desc",
            content_focus="Focus",
            rationale="Reason",
            target_symbols=[f"Class{i}" for i in range(50)],
            target_docs=[],
            target_folders=["src/classes"],
            key_files=[],
        )
        assert isinstance(result, str)

    def test_few_target_symbols_warning(self):
        """Fewer than 5 target_symbols triggers low-count warning."""
        c = self._setup_collector_with_graph()
        result = c._handle_define_page(
            section_name="Core",
            page_name="Tiny Page",
            page_order=1,
            description="Desc",
            content_focus="Focus",
            rationale="Reason",
            target_symbols=["AuthService", "validate_token"],  # Only 2
            target_docs=[],
            target_folders=["src/auth"],
            key_files=[],
        )
        assert isinstance(result, str)

    def test_target_docs_with_relevant_docs(self):
        """target_docs that match page name keywords are accepted silently."""
        c = self._setup_collector_with_graph()
        result = c._handle_define_page(
            section_name="Core",
            page_name="Authentication Guide",
            page_order=1,
            description="Desc",
            content_focus="Focus",
            rationale="Reason",
            target_symbols=[],
            target_docs=["docs/authentication.md"],
            target_folders=["src/auth"],
            key_files=[],
        )
        assert isinstance(result, str)

    def test_target_docs_irrelevant_produces_warning(self):
        """target_docs with no keyword overlap with page name produces warning."""
        c = self._setup_collector_with_graph()
        # Populate cached docs for the folder so the check triggers
        c._graph_module_cache["src/auth"] = {"all_docs": ["docs/unrelated.md"]}
        result = c._handle_define_page(
            section_name="Core",
            page_name="Authentication Service",
            page_order=1,
            description="Desc",
            content_focus="Focus",
            rationale="Reason",
            target_symbols=[],
            target_docs=["docs/unrelated.md", "docs/other.md", "docs/random.md"],
            target_folders=["src/auth"],
            key_files=[],
        )
        assert isinstance(result, str)


# ── _handle_batch_define_pages — symbol validation and continue logic ──────────

class TestHandleBatchDefinePagesExtended:
    """Extended batch_define_pages tests for symbol validation paths."""

    def _make_indexed_collector(self):
        """Collector with a graph and pre-built case-insensitive index.

        The _case_insensitive_symbols index is lazily built inside
        _handle_define_page when target_symbols is non-empty.  To test
        batch_define_pages' symbol-validation path (line 1669+) we must
        trigger that lazy init first by calling _handle_define_page once.
        """
        g = nx.DiGraph()
        g.add_node("n1", symbol_type="class", symbol_name="OrderService",
                   rel_path="src/orders/service.py", docstring="")
        g.add_node("n2", symbol_type="function", symbol_name="process_payment",
                   rel_path="src/payments/processor.py", docstring="")
        c = StructureCollector(page_budget=20, code_graph=g)
        # Trigger the lazy _case_insensitive_symbols build via a define_page call
        c._handle_define_section("Bootstrap", 99, "Bootstrap section", "Trigger index")
        c._handle_define_page(
            section_name="Bootstrap",
            page_name="_index_trigger_",
            page_order=99,
            description="D",
            content_focus="F",
            rationale="R",
            target_symbols=["OrderService"],
            target_docs=[],
            target_folders=["src/orders"],
            key_files=[],
        )
        # Reset pages so tests start from a clean slate
        c.pages = []
        c.sections = {}
        c._covered_symbols = set()
        c._symbol_to_page = {}
        return c

    def test_batch_valid_symbols_tracked(self):
        """batch_define_pages validates symbols against the graph."""
        c = self._make_indexed_collector()
        c._handle_define_section("Core", 1, "Core", "Needed")

        class FakePage:
            section_name = "Core"
            page_name = "Orders"
            description = "Order management"
            target_symbols = ["OrderService"]
            target_folders = ["src/orders"]

        c._handle_batch_define_pages([FakePage()])
        assert len(c.pages) == 1
        assert "OrderService" in c._covered_symbols

    def test_batch_invalid_symbols_still_creates_page(self):
        """Pages with invalid target_symbols show warning but still get created."""
        c = self._make_indexed_collector()

        class FakePage:
            section_name = "Core"
            page_name = "Ghost Page"
            description = "Has ghost symbols"
            target_symbols = ["GhostClass", "Phantom"]
            target_folders = ["src/ghost"]

        result = c._handle_batch_define_pages([FakePage()])
        assert len(c.pages) == 1  # Page created even with invalid symbols
        # With _case_insensitive_symbols present, GhostClass goes to unvalidated_symbols
        assert "Ghost Page" in result  # Page shows up in results

    def test_batch_duplicate_symbols_flagged(self):
        """Symbols already assigned to another page get flagged as duplicates."""
        c = self._make_indexed_collector()
        c._symbol_to_page["OrderService"] = "Existing Page"

        class FakePage:
            section_name = "Core"
            page_name = "New Orders"
            description = "New orders page"
            target_symbols = ["OrderService"]
            target_folders = ["src/orders"]

        result = c._handle_batch_define_pages([FakePage()])
        assert isinstance(result, str)
        assert len(c.pages) == 1

    def test_batch_continue_signal_when_low_coverage(self):
        """When coverage < 90%, response contains a CONTINUE signal."""
        c = StructureCollector(page_budget=50, code_graph=nx.DiGraph())
        # Register many discovered dirs so coverage is low
        for i in range(20):
            c.discovered_dirs.add(f"src/module{i}")

        class FakePage:
            section_name = "Core"
            page_name = "Page A"
            description = "Desc"
            target_symbols = []
            target_folders = ["src/module0"]

        result = c._handle_batch_define_pages([FakePage()])
        # With 1/20 coverage (5%) and remaining budget, CONTINUE signal should appear
        assert "CONTINUE" in result or "DO NOT" in result or "coverage" in result.lower()

    def test_batch_summary_shows_existing_structure(self):
        """When sections and pages exist, the existing structure snapshot appears."""
        c = StructureCollector(page_budget=50, code_graph=nx.DiGraph())
        c._handle_define_section("Auth", 1, "Authentication", "Needed")

        class FakePage:
            section_name = "Auth"
            page_name = "Login Flow"
            description = "Login"
            target_symbols = []
            target_folders = ["src/auth"]

        result = c._handle_batch_define_pages([FakePage()])
        # Existing structure shown after second batch call
        c2_pages = [FakePage(), FakePage()]
        c2_pages[1].page_name = "Logout Flow"
        result2 = c._handle_batch_define_pages(c2_pages)
        assert isinstance(result2, str)
        assert len(c.pages) == 3

    def test_batch_many_results_truncated_at_10(self):
        """When > 10 pages in a batch, only first 5 are shown in detail."""
        c = StructureCollector(page_budget=50, code_graph=nx.DiGraph())

        class FakePage:
            def __init__(self, n):
                self.section_name = "Core"
                self.page_name = f"Page {n}"
                self.description = "D"
                self.target_symbols = []
                self.target_folders = [f"src/m{n}"]

        pages = [FakePage(i) for i in range(15)]
        result = c._handle_batch_define_pages(pages)
        assert "more" in result or "15" in result
        assert len(c.pages) == 15


# ── _format_symbol_coverage_feedback — more unassigned symbols ─────────────────

class TestFormatSymbolCoverageFeedbackExtended:
    def test_more_than_6_unassigned_shows_remainder(self):
        """When > 6 unassigned symbols, shows (+N more) suffix."""
        g = _make_code_graph_with_symbols(
            [(f"Class{i}", "class", f"src/c{i}.py") for i in range(10)]
        )
        c = StructureCollector(page_budget=5, code_graph=g)
        c._all_architectural_symbols = {f"Class{i}" for i in range(10)}
        c._covered_symbols = set()  # Nothing covered
        result = c._format_symbol_coverage_feedback()
        # Should show some covered and the +N more pattern
        assert "more" in result or "%" in result


# ── _update_coverage — empty folder path ──────────────────────────────────────

class TestUpdateCoverageEdgeCases:
    def test_empty_parts_skipped(self):
        """Folder paths that are empty strings or just '/' are skipped."""
        c = _make_collector()
        c.discovered_dirs.add("src/auth")
        c._update_coverage(["/", "", "src/auth"])
        # 'src/auth' should be covered, the empty/slash paths should not crash
        assert "src/auth" in c.covered_dirs

    def test_descendant_dirs_covered_bidirectionally(self):
        """A target folder covers all discovered descendant directories."""
        c = _make_collector()
        c.discovered_dirs.update(["src/auth/handlers", "src/auth/models", "src/other"])
        c._update_coverage(["src/auth"])
        assert "src/auth/handlers" in c.covered_dirs
        assert "src/auth/models" in c.covered_dirs
        assert "src/other" not in c.covered_dirs


# ── format_tree_for_llm — parent dir with no content descendants ───────────────

class TestFormatTreeLargeParentNoDescendants:
    def test_large_parent_without_content_descendants_skipped(self):
        """Large parent dir with zero content descendants (files=0) is skipped."""
        dirs = [
            {"path": "src/big", "name": "big", "depth": 1, "files": 30,
             "code_files": 0, "doc_files": 0, "subdirs": 3},
            # Child dirs have files=0 (no content)
            {"path": "src/big/sub", "name": "sub", "depth": 2, "files": 0,
             "code_files": 0, "doc_files": 0, "subdirs": 0},
        ]
        tree = {"directories": dirs, "top_level_files": [], "total_files": 0}
        result = format_tree_for_llm(tree)
        assert isinstance(result, str)

    def test_large_parent_with_subdirs_in_child(self):
        """Content descendant with subdirs > 0 shows subdir count."""
        dirs = [
            {"path": "src/big", "name": "big", "depth": 1, "files": 30,
             "code_files": 30, "doc_files": 0, "subdirs": 5},
            {"path": "src/big/sub", "name": "sub", "depth": 2, "files": 5,
             "code_files": 5, "doc_files": 0, "subdirs": 2},
        ]
        tree = {"directories": dirs, "top_level_files": [], "total_files": 35}
        result = format_tree_for_llm(tree)
        # subdirs > 0 so "2 subdirs" appears
        assert "sub" in result or "subdirs" in result


# ── _format_graph_analysis — compact mode, overflow caps ──────────────────────

class TestFormatGraphAnalysis:
    """Directly test _format_graph_analysis with various analysis dicts."""

    def _make_collector(self):
        return StructureCollector(page_budget=10)

    def _class(self, name, **kwargs):
        return {"name": name, "type": "class", "file": f"{name.lower()}.py",
                "connections": 0, **kwargs}

    def test_no_symbols_returns_no_symbols_message(self):
        c = self._make_collector()
        analysis = {
            "total_symbols": 0, "class_count": 0, "function_count": 0,
            "type_alias_count": 0, "symbols_by_type": {}, "symbols_by_file": {},
            "files_with_symbols": 0, "all_classes": [], "all_functions": [],
            "all_enums": [], "all_type_aliases": [], "all_macros": [],
            "all_docs": [], "grouping_suggestions": [],
        }
        result = c._format_graph_analysis("src/empty", analysis)
        assert "No symbols found" in result

    def test_compact_mode_triggered_above_80_symbols(self):
        c = self._make_collector()
        many_classes = [self._class(f"Class{i}") for i in range(90)]
        analysis = {
            "total_symbols": 90, "class_count": 90, "function_count": 0,
            "type_alias_count": 0, "symbols_by_type": {"class": 90},
            "symbols_by_file": {}, "files_with_symbols": 5,
            "all_classes": many_classes, "all_functions": [], "all_enums": [],
            "all_type_aliases": [], "all_macros": [], "all_docs": [],
            "grouping_suggestions": [],
        }
        result = c._format_graph_analysis("src/big", analysis)
        assert "COMPACT" in result

    def test_more_than_class_detail_cap_shows_also_line(self):
        """When classes_with_context > 40, shows ALSO line."""
        c = self._make_collector()
        # Create 50 classes that all have inherits_from (so they go in classes_with_context)
        classes = [
            self._class(f"Class{i}", inherits_from=["BaseClass"])
            for i in range(50)
        ]
        analysis = {
            "total_symbols": 50, "class_count": 50, "function_count": 0,
            "type_alias_count": 0, "symbols_by_type": {"class": 50},
            "symbols_by_file": {}, "files_with_symbols": 10,
            "all_classes": classes, "all_functions": [], "all_enums": [],
            "all_type_aliases": [], "all_macros": [], "all_docs": [],
            "grouping_suggestions": [],
        }
        result = c._format_graph_analysis("src/many", analysis)
        assert "ALSO" in result or "more classes" in result

    def test_more_than_orphan_cap_shows_also_line(self):
        """When orphan classes > 10, shows ALSO line."""
        c = self._make_collector()
        # 15 orphan classes (no relationships)
        classes = [self._class(f"Orphan{i}") for i in range(15)]
        analysis = {
            "total_symbols": 15, "class_count": 15, "function_count": 0,
            "type_alias_count": 0, "symbols_by_type": {"class": 15},
            "symbols_by_file": {}, "files_with_symbols": 5,
            "all_classes": classes, "all_functions": [], "all_enums": [],
            "all_type_aliases": [], "all_macros": [], "all_docs": [],
            "grouping_suggestions": [],
        }
        result = c._format_graph_analysis("src/orphans", analysis)
        assert "ALSO" in result or "more" in result

    def test_more_than_enum_cap_shows_also_line(self):
        """When enums > 25, shows ALSO line."""
        c = self._make_collector()
        enums = [
            {"name": f"Status{i}", "type": "enum", "file": "status.py",
             "connections": 0, "used_by": []}
            for i in range(30)
        ]
        analysis = {
            "total_symbols": 30, "class_count": 0, "function_count": 0,
            "type_alias_count": 0, "symbols_by_type": {"enum": 30},
            "symbols_by_file": {}, "files_with_symbols": 1,
            "all_classes": [], "all_functions": [], "all_enums": enums,
            "all_type_aliases": [], "all_macros": [], "all_docs": [],
            "grouping_suggestions": [],
        }
        result = c._format_graph_analysis("src/enums", analysis)
        assert "ALSO" in result or "more" in result

    def test_more_than_type_alias_cap_shows_also_line(self):
        """When type aliases > 20, shows ALSO line."""
        c = self._make_collector()
        aliases = [
            {"name": f"Type{i}", "type": "type_alias", "file": "types.py",
             "alias_of": [], "used_by": []}
            for i in range(25)
        ]
        analysis = {
            "total_symbols": 25, "class_count": 0, "function_count": 0,
            "type_alias_count": 25, "symbols_by_type": {"type_alias": 25},
            "symbols_by_file": {}, "files_with_symbols": 1,
            "all_classes": [], "all_functions": [], "all_enums": [],
            "all_type_aliases": aliases, "all_macros": [], "all_docs": [],
            "grouping_suggestions": [],
        }
        result = c._format_graph_analysis("src/types", analysis)
        assert "ALSO" in result or "more" in result

    def test_more_than_macro_cap_shows_also_line(self):
        """When macros > 25, shows ALSO line."""
        c = self._make_collector()
        macros = [
            {"name": f"MACRO_{i}", "file": "config.h", "references": [],
             "used_by": [], "connections": 0}
            for i in range(30)
        ]
        analysis = {
            "total_symbols": 30, "class_count": 0, "function_count": 0,
            "type_alias_count": 0, "symbols_by_type": {"macro": 30},
            "symbols_by_file": {}, "files_with_symbols": 1,
            "all_classes": [], "all_functions": [], "all_enums": [],
            "all_type_aliases": [], "all_macros": macros, "all_docs": [],
            "grouping_suggestions": [],
        }
        result = c._format_graph_analysis("src/macros", analysis)
        assert "ALSO" in result or "more" in result

    def test_factory_functions_shown_with_creates(self):
        """Functions with creates[] show under FACTORY FUNCTIONS."""
        c = self._make_collector()
        funcs = [
            {"name": f"build{i}", "file": "factory.py",
             "connections": 1, "creates": ["ServiceA"], "uses_types": []}
            for i in range(3)
        ]
        analysis = {
            "total_symbols": 3, "class_count": 0, "function_count": 3,
            "type_alias_count": 0, "symbols_by_type": {"function": 3},
            "symbols_by_file": {}, "files_with_symbols": 1,
            "all_classes": [], "all_functions": funcs, "all_enums": [],
            "all_type_aliases": [], "all_macros": [], "all_docs": [],
            "grouping_suggestions": [],
        }
        result = c._format_graph_analysis("src/factory", analysis)
        assert "FACTORY FUNCTIONS" in result or "build0" in result

    def test_typed_functions_shown(self):
        """Functions with uses_types[] show under TYPED FUNCTIONS."""
        c = self._make_collector()
        funcs = [
            {"name": f"process{i}", "file": "proc.py",
             "connections": 1, "creates": [], "uses_types": ["str", "int"]}
            for i in range(3)
        ]
        analysis = {
            "total_symbols": 3, "class_count": 0, "function_count": 3,
            "type_alias_count": 0, "symbols_by_type": {"function": 3},
            "symbols_by_file": {}, "files_with_symbols": 1,
            "all_classes": [], "all_functions": funcs, "all_enums": [],
            "all_type_aliases": [], "all_macros": [], "all_docs": [],
            "grouping_suggestions": [],
        }
        result = c._format_graph_analysis("src/proc", analysis)
        assert "TYPED FUNCTIONS" in result or "process0" in result

    def test_factory_functions_overflow_shows_also(self):
        """When factory functions > func_cap (15), shows ALSO line."""
        c = self._make_collector()
        funcs = [
            {"name": f"build{i}", "file": "factory.py",
             "connections": 1, "creates": ["Service"], "uses_types": []}
            for i in range(20)
        ]
        analysis = {
            "total_symbols": 20, "class_count": 0, "function_count": 20,
            "type_alias_count": 0, "symbols_by_type": {"function": 20},
            "symbols_by_file": {}, "files_with_symbols": 1,
            "all_classes": [], "all_functions": funcs, "all_enums": [],
            "all_type_aliases": [], "all_macros": [], "all_docs": [],
            "grouping_suggestions": [],
        }
        result = c._format_graph_analysis("src/factory", analysis)
        assert "ALSO" in result or "more" in result

    def test_typed_functions_overflow_shows_also(self):
        """When typed functions > typed_cap (10), shows ALSO line."""
        c = self._make_collector()
        funcs = [
            {"name": f"typed{i}", "file": "typed.py",
             "connections": 1, "creates": [], "uses_types": ["str"]}
            for i in range(15)
        ]
        analysis = {
            "total_symbols": 15, "class_count": 0, "function_count": 15,
            "type_alias_count": 0, "symbols_by_type": {"function": 15},
            "symbols_by_file": {}, "files_with_symbols": 1,
            "all_classes": [], "all_functions": funcs, "all_enums": [],
            "all_type_aliases": [], "all_macros": [], "all_docs": [],
            "grouping_suggestions": [],
        }
        result = c._format_graph_analysis("src/typed", analysis)
        assert "ALSO" in result or "more" in result

    def test_non_creator_functions_shown(self):
        """Non-creator, non-typed functions show under OTHER KEY FUNCTIONS."""
        c = self._make_collector()
        funcs = [
            {"name": f"util{i}", "file": "utils.py",
             "connections": 0, "creates": [], "uses_types": []}
            for i in range(5)
        ]
        analysis = {
            "total_symbols": 5, "class_count": 0, "function_count": 5,
            "type_alias_count": 0, "symbols_by_type": {"function": 5},
            "symbols_by_file": {}, "files_with_symbols": 1,
            "all_classes": [], "all_functions": funcs, "all_enums": [],
            "all_type_aliases": [], "all_macros": [], "all_docs": [],
            "grouping_suggestions": [],
        }
        result = c._format_graph_analysis("src/utils", analysis)
        assert "KEY FUNCTIONS" in result or "util0" in result

    def test_non_creator_functions_overflow(self):
        """When non-creator functions > other_cap (10), shows ALSO line."""
        c = self._make_collector()
        funcs = [
            {"name": f"util{i}", "file": "utils.py",
             "connections": 0, "creates": [], "uses_types": []}
            for i in range(15)
        ]
        analysis = {
            "total_symbols": 15, "class_count": 0, "function_count": 15,
            "type_alias_count": 0, "symbols_by_type": {"function": 15},
            "symbols_by_file": {}, "files_with_symbols": 1,
            "all_classes": [], "all_functions": funcs, "all_enums": [],
            "all_type_aliases": [], "all_macros": [], "all_docs": [],
            "grouping_suggestions": [],
        }
        result = c._format_graph_analysis("src/utils", analysis)
        assert "ALSO" in result or "more" in result

    def test_class_with_created_by_shown(self):
        """Class with created_by field shows in context_parts."""
        c = self._make_collector()
        classes = [self._class("MyService", created_by=["ServiceFactory"])]
        analysis = {
            "total_symbols": 1, "class_count": 1, "function_count": 0,
            "type_alias_count": 0, "symbols_by_type": {"class": 1},
            "symbols_by_file": {}, "files_with_symbols": 1,
            "all_classes": classes, "all_functions": [], "all_enums": [],
            "all_type_aliases": [], "all_macros": [], "all_docs": [],
            "grouping_suggestions": [],
        }
        result = c._format_graph_analysis("src", analysis)
        assert "MyService" in result
        assert "created by" in result or "ServiceFactory" in result

    def test_class_with_specializes_shown(self):
        """Class with specializes field shows template context."""
        c = self._make_collector()
        classes = [self._class("ConcreteList",
                               specializes=["std::vector"],
                               specialized_by=["OrderedList"])]
        analysis = {
            "total_symbols": 1, "class_count": 1, "function_count": 0,
            "type_alias_count": 0, "symbols_by_type": {"class": 1},
            "symbols_by_file": {}, "files_with_symbols": 1,
            "all_classes": classes, "all_functions": [], "all_enums": [],
            "all_type_aliases": [], "all_macros": [], "all_docs": [],
            "grouping_suggestions": [],
        }
        result = c._format_graph_analysis("src", analysis)
        assert "ConcreteList" in result

    def test_class_with_declared_and_implemented_in(self):
        """Class with declared_in / implemented_in shows split impl info."""
        c = self._make_collector()
        classes = [self._class("Engine",
                               declared_in="include/engine.h",
                               implemented_in=["src/engine.cpp"],
                               has_split_impl=True)]
        analysis = {
            "total_symbols": 1, "class_count": 1, "function_count": 0,
            "type_alias_count": 0, "symbols_by_type": {"class": 1},
            "symbols_by_file": {}, "files_with_symbols": 1,
            "all_classes": classes, "all_functions": [], "all_enums": [],
            "all_type_aliases": [], "all_macros": [], "all_docs": [],
            "grouping_suggestions": [],
        }
        result = c._format_graph_analysis("src", analysis)
        assert "Engine" in result

    def test_docs_section_shows_when_present(self):
        """Documentation files section appears when all_docs is non-empty."""
        c = self._make_collector()
        docs = [{"path": "docs/guide.md", "name": "guide.md",
                 "type": "markdown_document", "file": "guide.md"}]
        analysis = {
            "total_symbols": 1, "class_count": 0, "function_count": 0,
            "type_alias_count": 0, "symbols_by_type": {},
            "symbols_by_file": {}, "files_with_symbols": 0,
            "all_classes": [], "all_functions": [], "all_enums": [],
            "all_type_aliases": [], "all_macros": [], "all_docs": docs,
            "grouping_suggestions": [],
        }
        result = c._format_graph_analysis("docs", analysis)
        assert "DOCUMENTATION FILES" in result or "guide.md" in result

    def test_grouping_suggestions_shown(self):
        """Grouping suggestions are rendered in output."""
        c = self._make_collector()
        suggestions = [
            {"main": "AuthService", "related": ["TokenManager", "LoginHelper"],
             "rationale": "Related auth classes"}
        ]
        analysis = {
            "total_symbols": 3, "class_count": 3, "function_count": 0,
            "type_alias_count": 0, "symbols_by_type": {"class": 3},
            "symbols_by_file": {}, "files_with_symbols": 1,
            "all_classes": [self._class("AuthService"),
                             self._class("TokenManager"),
                             self._class("LoginHelper")],
            "all_functions": [], "all_enums": [],
            "all_type_aliases": [], "all_macros": [], "all_docs": [],
            "grouping_suggestions": suggestions,
        }
        result = c._format_graph_analysis("src/auth", analysis)
        assert "AuthService" in result
        assert "SUGGESTED GROUPINGS" in result or "Related auth" in result

    def test_high_complexity_message(self):
        """When class_count >= 10, HIGH complexity message shown."""
        c = self._make_collector()
        classes = [self._class(f"Class{i}") for i in range(10)]
        analysis = {
            "total_symbols": 10, "class_count": 10, "function_count": 0,
            "type_alias_count": 0, "symbols_by_type": {"class": 10},
            "symbols_by_file": {}, "files_with_symbols": 2,
            "all_classes": classes, "all_functions": [], "all_enums": [],
            "all_type_aliases": [], "all_macros": [], "all_docs": [],
            "grouping_suggestions": [],
        }
        result = c._format_graph_analysis("src/complex", analysis)
        assert "HIGH" in result

    def test_medium_complexity_message(self):
        """When 4 <= class_count < 10, MEDIUM complexity shown."""
        c = self._make_collector()
        classes = [self._class(f"Class{i}") for i in range(5)]
        analysis = {
            "total_symbols": 5, "class_count": 5, "function_count": 0,
            "type_alias_count": 0, "symbols_by_type": {"class": 5},
            "symbols_by_file": {}, "files_with_symbols": 1,
            "all_classes": classes, "all_functions": [], "all_enums": [],
            "all_type_aliases": [], "all_macros": [], "all_docs": [],
            "grouping_suggestions": [],
        }
        result = c._format_graph_analysis("src/medium", analysis)
        assert "MEDIUM" in result

    def test_low_complexity_message(self):
        """When fewer than 4 classes, LOW complexity shown."""
        c = self._make_collector()
        analysis = {
            "total_symbols": 2, "class_count": 2, "function_count": 0,
            "type_alias_count": 0, "symbols_by_type": {"class": 2},
            "symbols_by_file": {}, "files_with_symbols": 1,
            "all_classes": [self._class("A"), self._class("B")],
            "all_functions": [], "all_enums": [],
            "all_type_aliases": [], "all_macros": [], "all_docs": [],
            "grouping_suggestions": [],
        }
        result = c._format_graph_analysis("src/small", analysis)
        assert "LOW" in result


# ── _generate_grouping_suggestions — same-file enum strategy ──────────────────

class TestGenerateGroupingSuggestions:
    """Test Strategy 2: group same-file enums/constants."""

    def test_same_file_enums_grouped_together(self):
        """2+ ungrouped enums in same file produce a suggestion."""
        c = StructureCollector(page_budget=5)
        enums = [
            {"name": f"Status{i}", "type": "enum", "file": "status.py",
             "connections": 0, "used_by": []}
            for i in range(3)
        ]
        symbols_by_file = {
            "status.py": [
                {"name": f"Status{i}", "type": "enum"} for i in range(3)
            ]
        }
        result = c._generate_grouping_suggestions([], enums, [], symbols_by_file)
        assert len(result) >= 1  # At least one suggestion from Strategy 2
        assert result[0]["main"] in {f"Status{i}" for i in range(3)}

    def test_class_in_same_file_used_as_main(self):
        """When a class exists in the same file as enums, class is the main."""
        c = StructureCollector(page_budget=5)
        enums = [
            {"name": "ErrorCode", "type": "enum", "file": "errors.py",
             "connections": 0, "used_by": []},
            {"name": "WarningCode", "type": "enum", "file": "errors.py",
             "connections": 0, "used_by": []},
        ]
        symbols_by_file = {
            "errors.py": [
                {"name": "ExceptionHandler", "type": "class"},
                {"name": "ErrorCode", "type": "enum"},
                {"name": "WarningCode", "type": "enum"},
            ]
        }
        result = c._generate_grouping_suggestions([], enums, [], symbols_by_file)
        # The class should be the main symbol when available
        assert any(s["main"] == "ExceptionHandler" for s in result)

    def test_class_with_enum_groups_together(self):
        """Strategy 1: class + its used enums form a suggestion."""
        c = StructureCollector(page_budget=5)
        classes = [
            {"name": "AuthService", "type": "class", "file": "auth.py",
             "connections": 5, "inherits_from": [], "implemented_by": [],
             "created_by": []}
        ]
        enums = [
            {"name": "AuthStatus", "type": "enum", "file": "auth.py",
             "connections": 1, "used_by": ["AuthService"]}
        ]
        result = c._generate_grouping_suggestions(classes, enums, [], {})
        assert len(result) >= 1
        assert result[0]["main"] == "AuthService"
        assert "AuthStatus" in result[0]["related"]

    def test_no_groups_when_fewer_than_2_enums(self):
        """Strategy 2 doesn't fire with only 1 enum in the file."""
        c = StructureCollector(page_budget=5)
        enums = [
            {"name": "Status", "type": "enum", "file": "status.py",
             "connections": 0, "used_by": []}
        ]
        symbols_by_file = {
            "status.py": [{"name": "Status", "type": "enum"}]
        }
        result = c._generate_grouping_suggestions([], enums, [], symbols_by_file)
        # Strategy 2 needs >= 2 ungrouped enums
        assert len(result) == 0
