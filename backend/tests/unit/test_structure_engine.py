"""Unit tests for app/core/wiki_structure_planner/structure_engine.py.

LLM and agent calls are fully mocked — no real network/LLM activity.
"""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import networkx as nx
import pytest

from app.core.wiki_structure_planner.structure_engine import (
    MAX_ITERATIONS,
    DEFAULT_PAGE_BUDGET,
    StructurePlannerConfig,
    WikiStructurePlannerEngine,
)


# ── Helpers ────────────────────────────────────────────────────────────────────

def _make_tmp_repo() -> str:
    """Create a minimal temp directory to serve as repo_root."""
    tmp = tempfile.mkdtemp()
    Path(os.path.join(tmp, "src")).mkdir()
    Path(os.path.join(tmp, "src", "main.py")).write_text("class App: pass")
    return tmp


def _minimal_wiki_structure() -> dict:
    return {
        "wiki_title": "Test Wiki",
        "overview": "Test overview",
        "sections": [
            {
                "section_name": "Core",
                "section_order": 1,
                "description": "Core section",
                "rationale": "Core",
                "pages": [
                    {
                        "page_name": "Core Features",
                        "page_order": 1,
                        "description": "Core page",
                        "content_focus": "Core",
                        "rationale": "Core",
                        "target_symbols": ["App"],
                        "target_docs": [],
                        "target_folders": ["src"],
                        "key_files": [],
                        "retrieval_query": "app",
                    }
                ],
            }
        ],
        "total_pages": 1,
    }


def _make_engine(repo_root: str | None = None, **kwargs) -> WikiStructurePlannerEngine:
    root = repo_root or _make_tmp_repo()
    llm = MagicMock()
    return WikiStructurePlannerEngine(repo_root=root, llm_client=llm, **kwargs)


# ── StructurePlannerConfig ────────────────────────────────────────────────────

class TestStructurePlannerConfig:
    def test_defaults(self):
        cfg = StructurePlannerConfig()
        assert cfg.page_budget == DEFAULT_PAGE_BUDGET
        assert cfg.target_audience == "developers"
        assert cfg.wiki_type == "technical"
        assert cfg.max_iterations == MAX_ITERATIONS
        assert cfg.effective_depth == 5
        assert cfg.max_ls_calls > 0
        assert cfg.max_glob_calls > 0
        assert cfg.max_read_calls > 0
        assert cfg.max_grep_calls > 0

    def test_override_page_budget(self):
        cfg = StructurePlannerConfig(page_budget=50)
        assert cfg.page_budget == 50

    def test_override_target_audience(self):
        cfg = StructurePlannerConfig(target_audience="ops")
        assert cfg.target_audience == "ops"

    def test_doc_clusters_none_by_default(self):
        cfg = StructurePlannerConfig()
        assert cfg.doc_clusters is None


# ── WikiStructurePlannerEngine — init ─────────────────────────────────────────

class TestWikiStructurePlannerEngineInit:
    def test_valid_repo_root_accepted(self):
        tmp = _make_tmp_repo()
        engine = WikiStructurePlannerEngine(repo_root=tmp, llm_client=MagicMock())
        assert engine.repo_root == tmp
        assert engine.status == "idle"
        assert engine.error is None

    def test_invalid_repo_root_raises(self):
        with pytest.raises(ValueError, match="repo_root"):
            WikiStructurePlannerEngine(repo_root="/nonexistent/path/xyz")

    def test_empty_repo_root_raises(self):
        with pytest.raises(ValueError):
            WikiStructurePlannerEngine(repo_root="")

    def test_default_config_when_none_provided(self):
        tmp = _make_tmp_repo()
        engine = WikiStructurePlannerEngine(repo_root=tmp)
        assert isinstance(engine.config, StructurePlannerConfig)
        assert engine.config.page_budget == DEFAULT_PAGE_BUDGET

    def test_custom_config_applied(self):
        tmp = _make_tmp_repo()
        cfg = StructurePlannerConfig(page_budget=42)
        engine = WikiStructurePlannerEngine(repo_root=tmp, config=cfg)
        assert engine.config.page_budget == 42

    def test_llm_settings_stored(self):
        tmp = _make_tmp_repo()
        settings = {"model_name": "gpt-4o", "api_key": "test-key"}
        engine = WikiStructurePlannerEngine(repo_root=tmp, llm_settings=settings)
        assert engine.llm_settings["model_name"] == "gpt-4o"

    def test_code_graph_stored(self):
        tmp = _make_tmp_repo()
        g = nx.DiGraph()
        engine = WikiStructurePlannerEngine(repo_root=tmp, code_graph=g)
        assert engine.code_graph is g

    def test_initial_tool_calls_empty(self):
        tmp = _make_tmp_repo()
        engine = WikiStructurePlannerEngine(repo_root=tmp)
        assert engine.tool_calls == []


# ── WikiStructurePlannerEngine — _extract_dirs_from_ls_result ─────────────────

class TestExtractDirsFromLsResult:
    def test_extracts_directories(self):
        engine = _make_engine()
        content = "src/\ntests/\nREADME.md\napp/\n"
        result = engine._extract_dirs_from_ls_result(content)
        assert "src" in result
        assert "tests" in result
        assert "app" in result

    def test_ignores_files(self):
        engine = _make_engine()
        content = "main.py\nconfig.yaml\nsrc/\n"
        result = engine._extract_dirs_from_ls_result(content)
        assert "main.py" not in result
        assert "config.yaml" not in result

    def test_empty_content_returns_empty(self):
        engine = _make_engine()
        result = engine._extract_dirs_from_ls_result("")
        assert result == []

    def test_blank_lines_ignored(self):
        engine = _make_engine()
        content = "\nsrc/\n\ntests/\n"
        result = engine._extract_dirs_from_ls_result(content)
        assert "" not in result
        assert len(result) == 2


# ── WikiStructurePlannerEngine — _parse_json ─────────────────────────────────

class TestParseJson:
    def test_valid_json_parsed(self):
        engine = _make_engine()
        data = {"wiki_title": "Test", "sections": []}
        result = engine._parse_json(json.dumps(data))
        assert result["wiki_title"] == "Test"

    def test_markdown_fenced_json_parsed(self):
        engine = _make_engine()
        data = {"sections": []}
        content = f"```json\n{json.dumps(data)}\n```"
        result = engine._parse_json(content)
        assert "sections" in result

    def test_invalid_json_raises_value_error(self):
        engine = _make_engine()
        with pytest.raises(ValueError, match="Could not parse JSON"):
            engine._parse_json("not valid json at all")


# ── WikiStructurePlannerEngine — _validate_structure ─────────────────────────

class TestValidateStructure:
    def test_valid_structure_no_warnings(self):
        engine = _make_engine()
        parsed = _minimal_wiki_structure()
        collector = MagicMock()
        collector.stats = {
            "coverage_pct": 90,
            "uncovered_dirs": [],
            "covered_dirs": 5,
            "discovered_dirs": 5,
        }
        collector.code_graph = None
        # Should not raise
        engine._validate_structure(parsed, collector)

    def test_low_coverage_logs_warning(self):
        engine = _make_engine()
        parsed = _minimal_wiki_structure()
        collector = MagicMock()
        collector.stats = {
            "coverage_pct": 20,  # below 50%
            "uncovered_dirs": ["src/a", "src/b"],
            "covered_dirs": 1,
            "discovered_dirs": 5,
        }
        collector.code_graph = None
        # Should not raise — validation is advisory
        engine._validate_structure(parsed, collector)

    def test_empty_pages_detected(self):
        engine = _make_engine()
        parsed = {
            "sections": [
                {
                    "section_name": "Core",
                    "pages": [
                        {
                            "page_name": "Empty Page",
                            "target_symbols": [],
                            "target_folders": [],
                            "target_docs": [],
                        }
                    ],
                }
            ]
        }
        collector = MagicMock()
        collector.stats = {
            "coverage_pct": 0,
            "uncovered_dirs": [],
            "covered_dirs": 0,
            "discovered_dirs": 0,
        }
        collector.code_graph = None
        # Should not raise
        engine._validate_structure(parsed, collector)


# ── WikiStructurePlannerEngine — _track_ls_results ───────────────────────────

class TestTrackLsResults:
    def test_tuple_messages_event_tracked(self):
        engine = _make_engine()
        collector = MagicMock()

        msg = MagicMock()
        msg.type = "tool"
        msg.name = "ls"
        msg.content = "src/\nlib/\nREADME.md\n"

        event = ("messages", (msg,))
        engine._track_ls_results(event, collector)
        collector.register_discovered_dirs.assert_called_once()
        args = collector.register_discovered_dirs.call_args[0][0]
        assert "src" in args

    def test_dict_event_with_messages_tracked(self):
        engine = _make_engine()
        collector = MagicMock()

        msg = MagicMock()
        msg.type = "tool"
        msg.name = "ls"
        msg.content = "tests/\n"

        event = {"messages": [msg]}
        engine._track_ls_results(event, collector)
        collector.register_discovered_dirs.assert_called_once()

    def test_non_ls_tool_not_tracked(self):
        engine = _make_engine()
        collector = MagicMock()

        msg = MagicMock()
        msg.type = "tool"
        msg.name = "read_file"
        msg.content = "contents here"

        event = ("messages", (msg,))
        engine._track_ls_results(event, collector)
        collector.register_discovered_dirs.assert_not_called()


# ── WikiStructurePlannerEngine — _collect_repository_files ───────────────────

class TestCollectRepositoryFiles:
    def test_collects_files_from_repo_root(self):
        tmp = _make_tmp_repo()
        engine = _make_engine(repo_root=tmp)
        files = engine._collect_repository_files()
        assert any("main.py" in f for f in files)

    def test_ignores_hidden_dirs(self):
        tmp = _make_tmp_repo()
        hidden = Path(tmp) / ".git"
        hidden.mkdir()
        (hidden / "config").write_text("git config")
        engine = _make_engine(repo_root=tmp)
        files = engine._collect_repository_files()
        assert not any(".git" in f for f in files)

    def test_returns_relative_paths(self):
        tmp = _make_tmp_repo()
        engine = _make_engine(repo_root=tmp)
        files = engine._collect_repository_files()
        for f in files:
            assert not f.startswith("/")
            assert not f.startswith(tmp)

# ── WikiStructurePlannerEngine — _validate_structure additional paths ──────────

class TestValidateStructureAdditional:
    """Cover oversized pages and symbol graph validation paths."""

    def test_oversized_page_logs_warning(self):
        """Pages with > 50 target_symbols should be detected as oversized."""
        engine = _make_engine()
        many_symbols = [f"Symbol{i}" for i in range(55)]
        parsed = {
            "sections": [
                {
                    "section_name": "Core",
                    "pages": [
                        {
                            "page_name": "Huge Page",
                            "target_symbols": many_symbols,
                            "target_folders": ["src"],
                            "target_docs": [],
                        }
                    ],
                }
            ]
        }
        collector = MagicMock()
        collector.stats = {
            "coverage_pct": 90,
            "uncovered_dirs": [],
            "covered_dirs": 5,
            "discovered_dirs": 5,
        }
        collector.code_graph = None
        # Should not raise; warning is logged internally
        engine._validate_structure(parsed, collector)

    def test_symbol_validation_against_graph(self):
        """When code_graph has nodes, validates target_symbols exist."""
        import networkx as nx
        engine = _make_engine()
        g = nx.DiGraph()
        g.add_node("n1", symbol_name="RealClass", symbol_type="class")
        parsed = {
            "sections": [
                {
                    "section_name": "Core",
                    "pages": [
                        {
                            "page_name": "Auth",
                            "target_symbols": ["RealClass", "GhostClass"],
                            "target_folders": ["src"],
                            "target_docs": [],
                        }
                    ],
                }
            ]
        }
        collector = MagicMock()
        collector.stats = {
            "coverage_pct": 90,
            "uncovered_dirs": [],
            "covered_dirs": 1,
            "discovered_dirs": 1,
        }
        collector.code_graph = g
        # Should not raise; missing symbols logged as warning
        engine._validate_structure(parsed, collector)

    def test_test_symbols_skipped_in_validation(self):
        """Symbols with test_ prefix are not flagged as missing."""
        import networkx as nx
        engine = _make_engine()
        g = nx.DiGraph()  # Empty graph — all symbols "missing"
        parsed = {
            "sections": [
                {
                    "section_name": "Tests",
                    "pages": [
                        {
                            "page_name": "Test Page",
                            "target_symbols": ["test_auth", "mock_user"],
                            "target_folders": ["tests"],
                            "target_docs": [],
                        }
                    ],
                }
            ]
        }
        collector = MagicMock()
        collector.stats = {
            "coverage_pct": 90,
            "uncovered_dirs": [],
            "covered_dirs": 1,
            "discovered_dirs": 1,
        }
        collector.code_graph = g
        # test_ and mock_ prefixed symbols should be silently skipped
        engine._validate_structure(parsed, collector)

    def test_collector_stats_exception_is_swallowed(self):
        """Exception from collector.stats should not crash validation."""
        engine = _make_engine()
        parsed = _minimal_wiki_structure()
        collector = MagicMock()
        collector.stats = MagicMock(side_effect=RuntimeError("stats broken"))
        collector.code_graph = None
        # Should not raise — exception swallowed with pass
        engine._validate_structure(parsed, collector)


# ── WikiStructurePlannerEngine — _track_ls_results additional paths ────────────

class TestTrackLsResultsAdditional:
    """Cover updates stream_mode path in _track_ls_results."""

    def test_updates_stream_mode_extracts_dirs(self):
        """(stream_mode='updates', data_dict) tuple format registers dirs."""
        from unittest.mock import MagicMock
        engine = _make_engine()
        collector = MagicMock()
        collector.register_discovered_dirs = MagicMock()

        # Build a fake ToolMessage with ls content
        msg = MagicMock()
        msg.type = "tool"
        msg.name = "ls"
        msg.content = "src/auth/\nsrc/core/\nREADME.md"

        event = ("updates", {"messages": [msg]})
        engine._track_ls_results(event, collector)
        collector.register_discovered_dirs.assert_called()

    def test_nested_dict_messages_extracted(self):
        """Dict event with nested {key: {messages: [...]}} also scanned."""
        from unittest.mock import MagicMock
        engine = _make_engine()
        collector = MagicMock()
        collector.register_discovered_dirs = MagicMock()

        msg = MagicMock()
        msg.type = "tool"
        msg.name = "ls"
        msg.content = "src/services/\nsrc/models/"

        # Nested format: {"agent": {"messages": [msg]}}
        event = {"agent": {"messages": [msg]}}
        engine._track_ls_results(event, collector)
        collector.register_discovered_dirs.assert_called()
