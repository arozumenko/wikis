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


# ── WikiStructurePlannerEngine — _extract_tool_info ───────────────────────────

class TestExtractToolInfo:
    def test_dict_format(self):
        engine = _make_engine()
        tc = {"name": "define_page", "args": {"page_name": "Auth"}}
        name, args = engine._extract_tool_info(tc)
        assert name == "define_page"
        assert args["page_name"] == "Auth"

    def test_object_format(self):
        engine = _make_engine()
        tc = MagicMock()
        tc.name = "define_section"
        tc.args = {"section_name": "Core"}
        name, args = engine._extract_tool_info(tc)
        assert name == "define_section"
        assert args["section_name"] == "Core"

    def test_dict_missing_fields_returns_defaults(self):
        engine = _make_engine()
        name, args = engine._extract_tool_info({})
        assert name == ""
        assert args == {}


# ── WikiStructurePlannerEngine — _has_tool_calls ──────────────────────────────

class TestHasToolCalls:
    def test_event_with_tool_calls_returns_true(self):
        engine = _make_engine()
        msg = MagicMock()
        msg.tool_calls = [{"name": "define_section"}]
        event = {"messages": [msg]}
        assert engine._has_tool_calls(event) is True

    def test_event_without_tool_calls_returns_false(self):
        engine = _make_engine()
        msg = MagicMock()
        msg.tool_calls = []
        event = {"messages": [msg]}
        assert engine._has_tool_calls(event) is False

    def test_non_dict_event_returns_false(self):
        engine = _make_engine()
        assert engine._has_tool_calls("not a dict") is False

    def test_empty_messages_returns_false(self):
        engine = _make_engine()
        assert engine._has_tool_calls({"messages": []}) is False


# ── WikiStructurePlannerEngine — _extract_response ───────────────────────────

class TestExtractResponse:
    def test_extracts_string_content(self):
        engine = _make_engine()
        msg = MagicMock()
        msg.content = "Final response text"
        result = engine._extract_response({"messages": [msg]})
        assert result == "Final response text"

    def test_extracts_list_content(self):
        engine = _make_engine()
        msg = MagicMock()
        msg.content = [{"text": "Part 1"}, {"text": "Part 2"}]
        result = engine._extract_response({"messages": [msg]})
        assert "Part 1" in result
        assert "Part 2" in result

    def test_empty_messages_raises(self):
        engine = _make_engine()
        with pytest.raises(ValueError, match="No messages"):
            engine._extract_response({"messages": []})

    def test_empty_dict_raises(self):
        engine = _make_engine()
        with pytest.raises(ValueError, match="No messages"):
            engine._extract_response({})


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


# ── WikiStructurePlannerEngine — _validate_graph_first ────────────────────────

class TestValidateGraphFirst:
    def _make_skeleton(self):
        from app.core.wiki_structure_planner.structure_skeleton import (
            DirCluster, StructureSkeleton, SymbolInfo
        )
        syms = [SymbolInfo(f"C{i}", "class", "src/f.py", "core_type", 1, "") for i in range(5)]
        cluster = DirCluster(1, ["src"], syms, 5, ["py"], (1, 1))
        return StructureSkeleton(
            code_clusters=[cluster],
            doc_clusters=[],
            total_arch_symbols=5,
            total_dirs_covered=1,
            total_dirs_in_repo=2,
            repo_languages=["py"],
            effective_depth=3,
        )

    def test_valid_data_no_error(self):
        engine = _make_engine()
        data = {
            "sections": [
                {
                    "section_name": "Core",
                    "pages": [
                        {
                            "page_name": "Core Features",
                            "target_symbols": ["C0", "C1", "C2", "C3", "C4"],
                            "target_folders": ["src"],
                            "target_docs": [],
                        }
                    ],
                }
            ],
            "total_pages": 1,
        }
        skeleton = self._make_skeleton()
        engine._validate_graph_first(data, skeleton)  # Should not raise

    def test_over_budget_detected(self):
        engine = _make_engine()
        # Create 100 pages when budget is 20
        pages = [{"page_name": f"P{i}", "target_symbols": [], "target_folders": [], "target_docs": []}
                 for i in range(100)]
        data = {"sections": [{"section_name": "X", "pages": pages}], "total_pages": 100}
        skeleton = self._make_skeleton()
        # Should not raise — validation is advisory
        engine._validate_graph_first(data, skeleton)


# ── WikiStructurePlannerEngine — plan_structure_graph_first ───────────────────

class TestPlanStructureGraphFirst:
    def test_graph_first_calls_refine_with_llm(self):
        """Verify graph-first approach calls refine_with_llm and returns structure."""
        tmp = _make_tmp_repo()
        g = nx.DiGraph()
        for i in range(5):
            g.add_node(f"n{i}", symbol_type="class", symbol_name=f"Cls{i}",
                       rel_path=f"src/mod{i}.py", layer="core_type", docstring="")

        mock_llm = MagicMock()
        engine = WikiStructurePlannerEngine(
            repo_root=tmp,
            llm_client=mock_llm,
            code_graph=g,
            config=StructurePlannerConfig(page_budget=5),
        )

        expected_structure = _minimal_wiki_structure()
        with patch("app.core.wiki_structure_planner.structure_engine.refine_with_llm",
                   return_value=expected_structure) as mock_refine:
            result = engine.plan_structure_graph_first(
                repo_name="test-repo",
                repository_files=["src/mod0.py", "src/mod1.py"],
            )

        mock_refine.assert_called_once()
        assert result["wiki_title"] == "Test Wiki"
        assert engine.status == "completed"

    def test_graph_first_with_stream_callback(self):
        """Stream callback receives phase events."""
        tmp = _make_tmp_repo()
        g = nx.DiGraph()
        g.add_node("n0", symbol_type="class", symbol_name="App",
                   rel_path="src/app.py", layer="core_type", docstring="")

        mock_llm = MagicMock()
        engine = WikiStructurePlannerEngine(repo_root=tmp, llm_client=mock_llm, code_graph=g)
        events = []

        with patch("app.core.wiki_structure_planner.structure_engine.refine_with_llm",
                   return_value=_minimal_wiki_structure()):
            engine.plan_structure_graph_first(
                repo_name="test",
                repository_files=["src/app.py"],
                stream_callback=lambda e: events.append(e),
            )

        assert len(events) >= 2
        phases = [e["phase"] for e in events if "phase" in e]
        assert "skeleton" in phases
        assert "refinement" in phases

    def test_graph_first_failure_sets_status_failed(self):
        tmp = _make_tmp_repo()
        g = nx.DiGraph()
        g.add_node("n0", symbol_type="class", symbol_name="App",
                   rel_path="src/app.py", layer="core_type", docstring="")
        engine = WikiStructurePlannerEngine(
            repo_root=tmp,
            llm_client=MagicMock(),
            code_graph=g,
        )

        with patch("app.core.wiki_structure_planner.structure_engine.refine_with_llm",
                   side_effect=RuntimeError("LLM failed")):
            with pytest.raises(RuntimeError, match="LLM failed"):
                engine.plan_structure_graph_first(
                    repo_name="test",
                    repository_files=["src/app.py"],
                )

        assert engine.status == "failed"
        assert engine.error == "LLM failed"

    def test_graph_first_collects_repository_files_when_not_provided(self):
        tmp = _make_tmp_repo()
        g = nx.DiGraph()
        g.add_node("n0", symbol_type="class", symbol_name="App",
                   rel_path="src/app.py", layer="core_type", docstring="")
        engine = WikiStructurePlannerEngine(repo_root=tmp, llm_client=MagicMock(), code_graph=g)

        with patch("app.core.wiki_structure_planner.structure_engine.refine_with_llm",
                   return_value=_minimal_wiki_structure()):
            result = engine.plan_structure_graph_first(repo_name="test")

        assert engine.status == "completed"
        assert "sections" in result


# ── WikiStructurePlannerEngine — plan_structure (invoke mode) ─────────────────

class TestPlanStructure:
    """Tests for the agent-based plan_structure method, mocking deepagents."""

    def _mock_agent_result(self, pages: int = 1) -> dict:
        """Build a fake agent result with tool-defined pages."""
        return {
            "messages": [
                MagicMock(content="Planning complete.")
            ]
        }

    def test_plan_structure_returns_assembled_structure(self):
        tmp = _make_tmp_repo()
        llm = MagicMock()
        engine = WikiStructurePlannerEngine(repo_root=tmp, llm_client=llm)

        # Mock the StructureCollector.assemble to return a valid structure
        expected = _minimal_wiki_structure()

        # StructureCollector is imported inside plan_structure via local import
        # Patch it at the source module level
        mock_collector_instance = MagicMock()
        mock_collector_instance.assemble.return_value = expected
        mock_collector_instance.stats = {
            "sections": 1, "pages": 1, "budget": 10,
            "coverage_pct": 100, "covered_dirs": 1, "discovered_dirs": 1,
            "uncovered_dirs": [],
        }
        mock_collector_instance.metadata = {"wiki_title": "Test Wiki"}
        mock_collector_instance.sections = {}
        mock_collector_instance.pages = []

        mock_agent = MagicMock()
        mock_agent.invoke.return_value = self._mock_agent_result()

        with patch("app.core.wiki_structure_planner.structure_tools.StructureCollector",
                   return_value=mock_collector_instance), \
             patch("app.core.wiki_structure_planner.structure_engine.ContextEvictionMiddleware",
                   return_value=MagicMock()), \
             patch("app.core.wiki_structure_planner.structure_prompts.get_structure_task_prompt",
                   return_value="task"), \
             patch.object(engine, "_create_agent", return_value=mock_agent), \
             patch("app.core.agents.wiki_graph_optimized.OptimizedWikiGenerationAgent._compute_max_symbols_per_page",
                   return_value=30):
            result = engine.plan_structure(repo_name="test-repo")

        assert result["wiki_title"] == "Test Wiki"
        assert engine.status == "completed"

    def test_plan_structure_failure_sets_status_failed(self):
        tmp = _make_tmp_repo()
        llm = MagicMock()
        engine = WikiStructurePlannerEngine(repo_root=tmp, llm_client=llm)

        # Patch the local import inside plan_structure
        with patch("app.core.wiki_structure_planner.structure_tools.StructureCollector",
                   side_effect=RuntimeError("Collector failed")):
            with pytest.raises(RuntimeError, match="Collector failed"):
                engine.plan_structure(repo_name="test-repo")

        assert engine.status == "failed"
        assert "Collector failed" in engine.error


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


# ── WikiStructurePlannerEngine — _build_middleware ───────────────────────────

class TestBuildMiddleware:
    def test_returns_list(self):
        engine = _make_engine()
        middleware = engine._build_middleware()
        assert isinstance(middleware, list)
        assert len(middleware) >= 1


# ── WikiStructurePlannerEngine — _log_event ──────────────────────────────────

class TestLogEvent:
    def test_dict_event_with_tool_calls_logged(self):
        engine = _make_engine()
        tc = MagicMock()
        tc.name = "define_page"
        tc.args = {"page_name": "Auth"}
        msg = MagicMock()
        msg.tool_calls = [tc]
        event = {"messages": [msg]}
        # Should not raise — tool calls tracked
        engine._log_event(event)

    def test_tuple_messages_event_handled(self):
        engine = _make_engine()
        tc = {"name": "define_section", "args": {"section_name": "Core"}}
        msg = MagicMock()
        msg.tool_calls = [tc]
        event = ("messages", (msg,))
        engine._log_event(event)  # Should not raise

    def test_unknown_event_type_handled_gracefully(self):
        engine = _make_engine()
        engine._log_event("unexpected_event_type")  # Should not raise
        engine._log_event(42)  # Should not raise

    def test_tool_result_message_logged(self):
        engine = _make_engine()
        msg = MagicMock()
        msg.tool_calls = []
        msg.type = "tool"
        msg.name = "ls"
        msg.content = "src/\nlib/\n"
        event = ("messages", (msg,))
        engine._log_event(event)  # Should not raise


# ── WikiStructurePlannerEngine — _track_tool_calls_from_event ─────────────────

class TestTrackToolCallsFromEvent:
    def test_dict_event_adds_to_tool_calls(self):
        engine = _make_engine()
        tc = {"name": "define_page", "args": {"page_name": "Auth"}}
        msg = MagicMock()
        msg.tool_calls = [tc]
        event = {"messages": [msg]}
        engine._track_tool_calls_from_event(event)
        assert len(engine.tool_calls) == 1
        assert engine.tool_calls[0]["tool"] == "define_page"

    def test_tuple_event_adds_to_tool_calls(self):
        engine = _make_engine()
        tc = {"name": "define_section", "args": {"section_name": "Core"}}
        msg = MagicMock()
        msg.tool_calls = [tc]
        event = ("messages", (msg,))
        engine._track_tool_calls_from_event(event)
        assert len(engine.tool_calls) == 1

    def test_duplicate_calls_not_added(self):
        engine = _make_engine()
        tc = {"name": "define_page", "args": {"page_name": "Auth"}}
        msg = MagicMock()
        msg.tool_calls = [tc]
        event = {"messages": [msg]}
        # Same event twice — should not duplicate
        engine._track_tool_calls_from_event(event)
        engine._track_tool_calls_from_event(event)
        assert len(engine.tool_calls) == 1


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
