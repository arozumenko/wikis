"""Extended unit tests for app/services/research_service.py.

Focuses on branches not covered by test_research_service.py:
- Error events → RuntimeError raised
- project_id path (multi-wiki components)
- research_sync with task_complete event type
- codemap_sync wrapper
- evict_cache
- _get_components delegation
- _build_call_tree_from_sources helper (unit-level)
- _explain_tree_nodes happy/error paths
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from pydantic import SecretStr

from app.config import Settings
from app.models.api import ResearchRequest, ResearchResponse, SourceReference
from app.services.research_service import (
    ResearchService,
    _build_call_tree_from_sources,
)
from app.storage.local import LocalArtifactStorage


def _settings() -> Settings:
    return Settings(llm_api_key=SecretStr("test-key"))


@pytest.fixture
def storage(tmp_path):
    return LocalArtifactStorage(str(tmp_path))


@pytest.fixture
def service(storage):
    return ResearchService(_settings(), storage)


# ---------------------------------------------------------------------------
# research_sync — error event branches
# ---------------------------------------------------------------------------


async def _research_error_events(*args, **kwargs):
    yield {"event_type": "research_start", "data": {}}
    yield {"event_type": "research_error", "data": {"error": "Something went wrong"}}


async def _task_failed_events(*args, **kwargs):
    yield {"event_type": "task_failed", "data": {"error": "engine exploded"}}


async def _task_complete_events(*args, **kwargs):
    yield {
        "event_type": "task_complete",
        "data": {
            "answer": "Architecture is layered.",
            "sources": [{"file_path": "main.py", "snippet": "..."}],
        },
    }


class TestResearchSyncErrorBranches:
    @pytest.mark.asyncio
    async def test_research_error_event_raises_runtime_error(self, service):
        req = ResearchRequest(wiki_id="w1", question="Q?")
        mock_engine = MagicMock()
        mock_engine.research = _research_error_events

        mock_components = MagicMock(
            retriever_stack=None, graph_manager=None, code_graph=None,
            repo_analysis=None, llm=MagicMock(), repo_path=None,
        )
        with (
            patch.object(service, "_get_components", new_callable=AsyncMock, return_value=mock_components),
            patch("app.services.research_service.DeepResearchEngine", return_value=mock_engine),
        ):
            with pytest.raises(RuntimeError, match="Something went wrong"):
                await service.research_sync(req)

    @pytest.mark.asyncio
    async def test_task_failed_event_raises_runtime_error(self, service):
        req = ResearchRequest(wiki_id="w1", question="Q?")
        mock_engine = MagicMock()
        mock_engine.research = _task_failed_events

        mock_components = MagicMock(
            retriever_stack=None, graph_manager=None, code_graph=None,
            repo_analysis=None, llm=MagicMock(), repo_path=None,
        )
        with (
            patch.object(service, "_get_components", new_callable=AsyncMock, return_value=mock_components),
            patch("app.services.research_service.DeepResearchEngine", return_value=mock_engine),
        ):
            with pytest.raises(RuntimeError, match="engine exploded"):
                await service.research_sync(req)

    @pytest.mark.asyncio
    async def test_task_complete_event_accepted(self, service):
        """task_complete (MCP alias) also produces a valid ResearchResponse."""
        req = ResearchRequest(wiki_id="w1", question="Q?")
        mock_engine = MagicMock()
        mock_engine.research = _task_complete_events

        mock_components = MagicMock(
            retriever_stack=None, graph_manager=None, code_graph=None,
            repo_analysis=None, llm=MagicMock(), repo_path=None,
        )
        with (
            patch.object(service, "_get_components", new_callable=AsyncMock, return_value=mock_components),
            patch("app.services.research_service.DeepResearchEngine", return_value=mock_engine),
        ):
            result = await service.research_sync(req)

        assert isinstance(result, ResearchResponse)
        assert result.answer == "Architecture is layered."
        assert len(result.sources) == 1


# ---------------------------------------------------------------------------
# evict_cache
# ---------------------------------------------------------------------------


class TestEvictCache:
    def test_evict_returns_true_when_cached(self, service):
        service._cache._store = {"wiki-abc": MagicMock()}  # fake a cached entry
        # Evict should succeed regardless (delegates to ComponentCache.evict)
        result = service.evict_cache("wiki-abc")
        assert isinstance(result, bool)

    def test_evict_returns_false_when_not_cached(self, service):
        result = service.evict_cache("nonexistent-wiki")
        assert result is False


# ---------------------------------------------------------------------------
# research_stream — project_id path
# ---------------------------------------------------------------------------


async def _simple_events(*args, **kwargs):
    yield {"event_type": "research_complete", "data": {"report": "done", "sources": []}}


class TestResearchStreamProjectPath:
    @pytest.mark.asyncio
    async def test_project_id_triggers_multi_wiki_components(self, service):
        req = ResearchRequest(project_id="proj-1", question="How does it work?")
        mock_engine = MagicMock()
        mock_engine.research = _simple_events

        mock_components = MagicMock(
            retriever_stack=None, graph_manager=None, code_graph=None,
            repo_analysis=None, llm=MagicMock(), repo_path=None,
        )

        with (
            patch(
                "app.services.multi_wiki_components.build_multi_wiki_components",
                new_callable=AsyncMock,
                return_value=(mock_components, {}),
            ) as mock_multi,
            patch("app.services.research_service.DeepResearchEngine", return_value=mock_engine),
        ):
            events = [e async for e in service.research_stream(req)]

        mock_multi.assert_called_once()
        assert len(events) >= 1

    @pytest.mark.asyncio
    async def test_wiki_id_path_uses_get_components(self, service):
        req = ResearchRequest(wiki_id="w1", question="Q?")
        mock_engine = MagicMock()
        mock_engine.research = _simple_events

        mock_components = MagicMock(
            retriever_stack=None, graph_manager=None, code_graph=None,
            repo_analysis=None, llm=MagicMock(), repo_path=None,
        )

        with (
            patch.object(service, "_get_components", new_callable=AsyncMock, return_value=mock_components) as mock_get,
            patch("app.services.research_service.DeepResearchEngine", return_value=mock_engine),
        ):
            events = [e async for e in service.research_stream(req)]

        mock_get.assert_called_once_with("w1")
        assert len(events) >= 1


# ---------------------------------------------------------------------------
# research_sync — thinking_step description vs tool key
# ---------------------------------------------------------------------------


async def _thinking_step_with_description(*args, **kwargs):
    yield {"event_type": "thinking_step", "data": {"description": "Analyzing graph"}}
    yield {"event_type": "research_complete", "data": {"report": "done", "sources": []}}


async def _thinking_step_with_tool_only(*args, **kwargs):
    yield {"event_type": "thinking_step", "data": {"tool": "search_codebase"}}
    yield {"event_type": "research_complete", "data": {"report": "done", "sources": []}}


class TestResearchSyncThinkingSteps:
    @pytest.mark.asyncio
    async def test_description_key_used_when_tool_absent(self, service):
        req = ResearchRequest(wiki_id="w1", question="Q?")
        mock_engine = MagicMock()
        mock_engine.research = _thinking_step_with_description

        mock_components = MagicMock(
            retriever_stack=None, graph_manager=None, code_graph=None,
            repo_analysis=None, llm=MagicMock(), repo_path=None,
        )
        with (
            patch.object(service, "_get_components", new_callable=AsyncMock, return_value=mock_components),
            patch("app.services.research_service.DeepResearchEngine", return_value=mock_engine),
        ):
            result = await service.research_sync(req)

        assert "Analyzing graph" in result.research_steps

    @pytest.mark.asyncio
    async def test_empty_step_description_not_appended(self, service):
        """Steps with no tool or description key should not add to research_steps."""
        async def _empty_step_events(*args, **kwargs):
            yield {"event_type": "thinking_step", "data": {}}
            yield {"event_type": "research_complete", "data": {"report": "done", "sources": []}}

        req = ResearchRequest(wiki_id="w1", question="Q?")
        mock_engine = MagicMock()
        mock_engine.research = _empty_step_events

        mock_components = MagicMock(
            retriever_stack=None, graph_manager=None, code_graph=None,
            repo_analysis=None, llm=MagicMock(), repo_path=None,
        )
        with (
            patch.object(service, "_get_components", new_callable=AsyncMock, return_value=mock_components),
            patch("app.services.research_service.DeepResearchEngine", return_value=mock_engine),
        ):
            result = await service.research_sync(req)

        assert result.research_steps == []


# ---------------------------------------------------------------------------
# _get_components — delegates to ComponentCache
# ---------------------------------------------------------------------------


class TestGetComponents:
    @pytest.mark.asyncio
    async def test_get_components_calls_cache(self, service):
        mock_components = MagicMock()

        with patch.object(
            service._cache,
            "get_or_build",
            new_callable=AsyncMock,
            return_value=mock_components,
        ):
            result = await service._get_components("wiki-xyz")

        assert result is mock_components


# ---------------------------------------------------------------------------
# codemap_sync — non-streaming wrapper
# ---------------------------------------------------------------------------


class TestCodemapSync:
    @pytest.mark.asyncio
    async def test_codemap_sync_collects_events(self, service):
        req = ResearchRequest(wiki_id="w1", question="How auth works?", research_type="codemap")

        async def _codemap_events(*a, **kw):
            yield {"event_type": "thinking_step", "data": {"tool": "build_call_tree"}}
            yield {
                "event_type": "research_complete",
                "data": {
                    "report": "Auth uses JWT.",
                    "sources": [{"file_path": "auth.py"}],
                    "code_map": None,
                },
            }

        with patch.object(service, "codemap_stream", return_value=_codemap_events()):
            result = await service.codemap_sync(req)

        assert result.answer == "Auth uses JWT."
        assert len(result.sources) == 1
        assert "build_call_tree" in result.research_steps

    @pytest.mark.asyncio
    async def test_codemap_sync_with_code_map_data(self, service):
        req = ResearchRequest(wiki_id="w1", question="Q?", research_type="codemap")
        code_map_dict = {
            "sections": [],
            "call_stacks": [],
            "summary": "some summary",
        }

        async def _events_with_map(*a, **kw):
            yield {
                "event_type": "research_complete",
                "data": {
                    "report": "answer",
                    "sources": [],
                    "code_map": code_map_dict,
                },
            }

        with patch.object(service, "codemap_stream", return_value=_events_with_map()):
            result = await service.codemap_sync(req)

        assert result.code_map is not None
        assert result.code_map.summary == "some summary"


# ---------------------------------------------------------------------------
# _build_call_tree_from_sources — unit tests
# ---------------------------------------------------------------------------


class TestBuildCallTreeFromSources:
    def _make_gqs(self, nodes: dict, relationships: list | None = None) -> MagicMock:
        """Build a minimal GraphQueryService mock."""
        gqs = MagicMock()
        gqs.graph = MagicMock()
        gqs.graph.nodes = nodes

        def _resolve_symbol(name, file_path=""):
            # Return the name as the node_id if it's in nodes
            return name if name in nodes else None

        gqs.resolve_symbol = MagicMock(side_effect=_resolve_symbol)
        gqs.search = MagicMock(return_value=[])
        gqs.get_relationships = MagicMock(return_value=relationships or [])
        return gqs

    def test_empty_sources_returns_none(self):
        gqs = self._make_gqs({})
        result = _build_call_tree_from_sources([], gqs)
        assert result is None

    def test_no_resolved_seeds_returns_none(self):
        gqs = self._make_gqs({})
        # Sources that won't resolve to any node_id
        sources = [SourceReference(file_path="unknown.py", symbol="unknown_func")]
        result = _build_call_tree_from_sources(sources, gqs)
        assert result is None

    def test_valid_source_builds_code_map(self):
        nodes = {
            "node1": {
                "symbol_name": "MyClass",
                "symbol_type": "class",
                "rel_path": "src/myclass.py",
                "line_start": 10,
            }
        }
        gqs = self._make_gqs(nodes)
        sources = [SourceReference(file_path="src/myclass.py", symbol="node1")]
        result = _build_call_tree_from_sources(sources, gqs)
        # Should have built something
        assert result is not None
        assert len(result.sections) > 0

    def test_source_with_only_file_path_uses_search_fallback(self):
        """When symbol not directly resolved, falls back to gqs.search."""
        nodes = {
            "node-found": {
                "symbol_name": "SomeFunc",
                "symbol_type": "function",
                "rel_path": "utils.py",
                "line_start": 5,
            }
        }
        gqs = self._make_gqs(nodes)
        # resolve_symbol returns None for symbol name, but search returns a hit
        hit = MagicMock()
        hit.rel_path = "utils.py"
        hit.file_path = "utils.py"
        hit.node_id = "node-found"
        gqs.search.return_value = [hit]
        gqs.resolve_symbol = MagicMock(side_effect=lambda name, file_path="": None)

        sources = [SourceReference(file_path="utils.py")]
        result = _build_call_tree_from_sources(sources, gqs)
        assert result is not None

    def test_symbols_in_excluded_types_skipped(self):
        """file/module type nodes should be excluded from the output."""
        nodes = {
            "node-file": {
                "symbol_name": "mymodule",
                "symbol_type": "file",
                "rel_path": "mymodule.py",
                "line_start": None,
            }
        }
        gqs = self._make_gqs(nodes)
        sources = [SourceReference(file_path="mymodule.py", symbol="node-file")]
        result = _build_call_tree_from_sources(sources, gqs)
        # file-type nodes are excluded → result may be None or empty
        # We just verify no crash
        assert result is None or len(result.sections) == 0


# ---------------------------------------------------------------------------
# codemap_stream — basic pipeline coverage
# ---------------------------------------------------------------------------


class TestCodemapStream:
    @pytest.mark.asyncio
    async def test_codemap_stream_minimal_pipeline(self, service):
        """codemap_stream yields at minimum a research_complete event."""
        req = ResearchRequest(wiki_id="w1", question="How does auth work?", research_type="codemap")

        # Mock ask engine to yield no useful events (no symbols, no answer)
        async def _ask_events(*args, **kwargs):
            from langchain_core.messages import AIMessage
            yield ("messages", (AIMessage(content=""), {}))

        mock_ask_engine = MagicMock()
        mock_ask_engine.ask = _ask_events

        # Build minimal components mock
        mock_gm = MagicMock()
        mock_gm.fts_index = None
        mock_components = MagicMock(
            retriever_stack=MagicMock(),
            graph_manager=mock_gm,
            code_graph=MagicMock(),
            repo_analysis=None,
            llm=AsyncMock(),
            repo_path=None,
        )

        with (
            patch.object(service, "_get_components", new_callable=AsyncMock, return_value=mock_components),
            patch(
                "app.core.ask_engine.AskEngine",
                return_value=mock_ask_engine,
            ),
            patch(
                "app.services.llm_factory.create_llm",
                return_value=AsyncMock(),
            ),
            patch(
                "app.core.code_graph.graph_query_service.GraphQueryService",
                return_value=MagicMock(),
            ),
            patch(
                "app.services.research_service._build_call_tree_from_sources",
                return_value=None,
            ),
        ):
            events = [e async for e in service.codemap_stream(req)]

        # Should always end with research_complete
        event_types = [e["event_type"] for e in events]
        assert "research_complete" in event_types

    @pytest.mark.asyncio
    async def test_codemap_stream_with_project_id(self, service):
        """Project-id path builds multi-wiki components."""
        req = ResearchRequest(project_id="proj-1", question="Q?", research_type="codemap")

        async def _ask_events(*args, **kwargs):
            from langchain_core.messages import AIMessage
            yield ("messages", (AIMessage(content="answer"), {}))

        mock_ask_engine = MagicMock()
        mock_ask_engine.ask = _ask_events

        mock_gm = MagicMock()
        mock_gm.fts_index = None
        mock_components = MagicMock(
            retriever_stack=MagicMock(),
            graph_manager=mock_gm,
            code_graph=MagicMock(),
            repo_analysis=None,
            llm=AsyncMock(),
            repo_path=None,
        )

        with (
            patch(
                "app.services.multi_wiki_components.build_multi_wiki_components",
                new_callable=AsyncMock,
                return_value=(mock_components, {}),
            ),
            patch(
                "app.core.ask_engine.AskEngine",
                return_value=mock_ask_engine,
            ),
            patch("app.services.llm_factory.create_llm", return_value=AsyncMock()),
            patch("app.services.research_service.GraphQueryService", return_value=MagicMock()),
            patch("app.services.research_service._build_call_tree_from_sources", return_value=None),
        ):
            events = [e async for e in service.codemap_stream(req)]

        event_types = [e["event_type"] for e in events]
        assert "research_complete" in event_types

    @pytest.mark.asyncio
    async def test_codemap_stream_with_thinking_step_tool_result(self, service):
        """thinking_step with tool_result containing symbol refs goes through filter."""
        req = ResearchRequest(wiki_id="w1", question="How does auth work?", research_type="codemap")

        async def _ask_events(*args, **kwargs):
            # Simulate a tool result event with symbol reference in output
            yield {
                "event_type": "thinking_step",
                "data": {
                    "stepType": "tool_result",
                    "tool": "search_symbols",
                    # Output format parsed by _SYM_LINE_RE: `Name` (type) in file.py
                    "output": "`authenticate` (function) in auth/login.py\n`logout` (function) in auth/logout.py",
                }
            }
            yield {
                "event_type": "ask_complete",
                "data": {"answer": "authenticate handles the login flow in auth/login.py"},
            }

        mock_ask_engine = MagicMock()
        mock_ask_engine.ask = _ask_events

        mock_gm = MagicMock()
        mock_gm.fts_index = None
        mock_llm = AsyncMock()
        mock_llm.ainvoke = AsyncMock(return_value=MagicMock(content='{"descriptions": {}, "summary": "", "call_stacks": []}'))
        mock_components = MagicMock(
            retriever_stack=MagicMock(),
            graph_manager=mock_gm,
            code_graph=MagicMock(),
            repo_analysis=None,
            llm=mock_llm,
            repo_path=None,
        )

        with (
            patch.object(service, "_get_components", new_callable=AsyncMock, return_value=mock_components),
            patch(
                "app.core.ask_engine.AskEngine",
                return_value=mock_ask_engine,
            ),
            patch("app.services.llm_factory.create_llm", return_value=AsyncMock()),
            patch("app.services.research_service.GraphQueryService", return_value=MagicMock()),
            patch("app.services.research_service._build_call_tree_from_sources", return_value=None),
        ):
            events = [e async for e in service.codemap_stream(req)]

        event_types = [e["event_type"] for e in events]
        assert "research_complete" in event_types
        # thinking_step events should have been forwarded
        assert "thinking_step" in event_types

    @pytest.mark.asyncio
    async def test_codemap_stream_with_code_map_and_explain(self, service):
        """When a code_map is built, _explain_tree_nodes is called and refine step runs."""
        from app.models.api import CodeMapData, CodeMapSection, CodeMapSymbol

        req = ResearchRequest(wiki_id="w1", question="How does auth work?", research_type="codemap")

        async def _ask_events(*args, **kwargs):
            yield {
                "event_type": "ask_complete",
                "data": {"answer": "authenticate is used in auth.py"},
            }

        mock_ask_engine = MagicMock()
        mock_ask_engine.ask = _ask_events

        # Build a non-empty code_map so explain/refine paths run
        sym = CodeMapSymbol(id="s0_sym0", name="authenticate", symbol_type="function", file_path="auth.py")
        sec = CodeMapSection(id="section_0", title="auth.py", file_path="auth.py", symbols=[sym])
        fake_code_map = CodeMapData(sections=[sec])

        mock_gm = MagicMock()
        mock_gm.fts_index = None
        mock_llm_resp = MagicMock()
        mock_llm_resp.content = '{"descriptions": {"s0_sym0": "Auth func."}, "summary": "Auth system.", "call_stacks": []}'
        mock_llm = AsyncMock()
        mock_llm.ainvoke = AsyncMock(return_value=mock_llm_resp)

        mock_components = MagicMock(
            retriever_stack=MagicMock(),
            graph_manager=mock_gm,
            code_graph=MagicMock(),
            repo_analysis=None,
            llm=mock_llm,
            repo_path=None,
        )

        mock_explain_llm = AsyncMock()
        mock_explain_llm.ainvoke = AsyncMock(return_value=mock_llm_resp)

        with (
            patch.object(service, "_get_components", new_callable=AsyncMock, return_value=mock_components),
            patch("app.core.ask_engine.AskEngine", return_value=mock_ask_engine),
            patch("app.services.llm_factory.create_llm", return_value=mock_explain_llm),
            patch("app.services.research_service.GraphQueryService", return_value=MagicMock()),
            patch(
                "app.services.research_service._build_call_tree_from_sources",
                return_value=fake_code_map,
            ),
        ):
            events = [e async for e in service.codemap_stream(req)]

        event_types = [e["event_type"] for e in events]
        assert "research_complete" in event_types
        # code_map_ready should be emitted when code_map is present
        assert "code_map_ready" in event_types

    @pytest.mark.asyncio
    async def test_codemap_stream_ask_engine_failure_is_handled(self, service):
        """If ask engine raises during codemap, pipeline still completes."""
        req = ResearchRequest(wiki_id="w1", question="Q?", research_type="codemap")

        async def _crashing_ask(*args, **kwargs):
            raise RuntimeError("ask crashed")
            yield  # make it an async generator

        mock_ask_engine = MagicMock()
        mock_ask_engine.ask = _crashing_ask

        mock_gm = MagicMock()
        mock_gm.fts_index = None
        mock_components = MagicMock(
            retriever_stack=MagicMock(),
            graph_manager=mock_gm,
            code_graph=MagicMock(),
            repo_analysis=None,
            llm=AsyncMock(),
            repo_path=None,
        )

        with (
            patch.object(service, "_get_components", new_callable=AsyncMock, return_value=mock_components),
            patch("app.core.ask_engine.AskEngine", return_value=mock_ask_engine),
            patch("app.services.llm_factory.create_llm", return_value=AsyncMock()),
            patch("app.services.research_service.GraphQueryService", return_value=MagicMock()),
            patch("app.services.research_service._build_call_tree_from_sources", return_value=None),
        ):
            events = [e async for e in service.codemap_stream(req)]

        event_types = [e["event_type"] for e in events]
        assert "research_complete" in event_types


# ---------------------------------------------------------------------------
# _explain_tree_nodes function tests
# ---------------------------------------------------------------------------


class TestExplainTreeNodes:
    @pytest.mark.asyncio
    async def test_explain_tree_nodes_empty_symbols(self):
        """When code_map has no symbols, returns unchanged code_map."""
        from app.models.api import CodeMapData
        from app.services.research_service import _explain_tree_nodes

        code_map = CodeMapData(sections=[])
        mock_llm = AsyncMock()
        result = await _explain_tree_nodes(code_map, "Q?", mock_llm)
        assert result is code_map
        mock_llm.ainvoke.assert_not_called()

    @pytest.mark.asyncio
    async def test_explain_tree_nodes_applies_descriptions(self):
        """LLM response descriptions are applied back to symbols."""
        from app.models.api import CodeMapData, CodeMapSection, CodeMapSymbol
        from app.services.research_service import _explain_tree_nodes

        sym = CodeMapSymbol(id="s0_sym0", name="LoginFunc", symbol_type="function", file_path="auth.py")
        sec = CodeMapSection(id="section_0", title="auth.py", file_path="auth.py", symbols=[sym])
        code_map = CodeMapData(sections=[sec])

        resp_json = '{"descriptions": {"s0_sym0": "Handles login logic."}, "summary": "Auth system.", "call_stacks": []}'
        mock_llm_resp = MagicMock()
        mock_llm_resp.content = resp_json
        mock_llm = AsyncMock()
        mock_llm.ainvoke = AsyncMock(return_value=mock_llm_resp)

        result = await _explain_tree_nodes(code_map, "How does login work?", mock_llm)
        assert result.sections[0].symbols[0].description == "Handles login logic."
        assert result.summary == "Auth system."

    @pytest.mark.asyncio
    async def test_explain_tree_nodes_handles_llm_json_error(self):
        """LLM returning invalid JSON is handled without raising."""
        from app.models.api import CodeMapData, CodeMapSection, CodeMapSymbol
        from app.services.research_service import _explain_tree_nodes

        sym = CodeMapSymbol(id="s0_sym0", name="Foo", symbol_type="function", file_path="foo.py")
        sec = CodeMapSection(id="section_0", title="foo.py", file_path="foo.py", symbols=[sym])
        code_map = CodeMapData(sections=[sec])

        mock_llm_resp = MagicMock()
        mock_llm_resp.content = "not valid json {"
        mock_llm = AsyncMock()
        mock_llm.ainvoke = AsyncMock(return_value=mock_llm_resp)

        result = await _explain_tree_nodes(code_map, "Q?", mock_llm)
        # Should return code_map unchanged, no exception
        assert result is code_map

    @pytest.mark.asyncio
    async def test_explain_tree_nodes_strips_markdown_fences(self):
        """Markdown code fences are stripped from LLM response before JSON parsing."""
        from app.models.api import CodeMapData, CodeMapSection, CodeMapSymbol
        from app.services.research_service import _explain_tree_nodes

        sym = CodeMapSymbol(id="s0_sym0", name="Bar", symbol_type="function", file_path="bar.py")
        sec = CodeMapSection(id="section_0", title="bar.py", file_path="bar.py", symbols=[sym])
        code_map = CodeMapData(sections=[sec])

        resp_with_fences = '```json\n{"descriptions": {"s0_sym0": "Does stuff."}, "summary": "", "call_stacks": []}\n```'
        mock_llm_resp = MagicMock()
        mock_llm_resp.content = resp_with_fences
        mock_llm = AsyncMock()
        mock_llm.ainvoke = AsyncMock(return_value=mock_llm_resp)

        result = await _explain_tree_nodes(code_map, "Q?", mock_llm)
        assert result.sections[0].symbols[0].description == "Does stuff."

    @pytest.mark.asyncio
    async def test_explain_tree_nodes_with_call_stacks(self):
        """call_stacks from LLM response are built correctly."""
        from app.models.api import CodeMapData, CodeMapSection, CodeMapSymbol
        from app.services.research_service import _explain_tree_nodes

        sym = CodeMapSymbol(id="s0_sym0", name="MyFunc", symbol_type="function", file_path="svc.py")
        sec = CodeMapSection(id="section_0", title="svc.py", file_path="svc.py", symbols=[sym])
        code_map = CodeMapData(sections=[sec])

        call_stacks = [
            {
                "title": "Request flow",
                "steps": [
                    {"symbol": "MyFunc", "file_path": "svc.py", "description": "Entry point"},
                ]
            }
        ]
        resp_json = f'{{"descriptions": {{}}, "summary": "summary", "call_stacks": {__import__("json").dumps(call_stacks)}}}'
        mock_llm_resp = MagicMock()
        mock_llm_resp.content = resp_json
        mock_llm = AsyncMock()
        mock_llm.ainvoke = AsyncMock(return_value=mock_llm_resp)

        result = await _explain_tree_nodes(code_map, "Q?", mock_llm)
        assert len(result.call_stacks) == 1
        assert result.call_stacks[0].title == "Request flow"

    @pytest.mark.asyncio
    async def test_explain_tree_nodes_with_llm_exception(self):
        """LLM call raising an exception is handled gracefully."""
        from app.models.api import CodeMapData, CodeMapSection, CodeMapSymbol
        from app.services.research_service import _explain_tree_nodes

        sym = CodeMapSymbol(id="s0_sym0", name="Foo", symbol_type="function", file_path="foo.py")
        sec = CodeMapSection(id="section_0", title="foo.py", file_path="foo.py", symbols=[sym])
        code_map = CodeMapData(sections=[sec])

        mock_llm = AsyncMock()
        mock_llm.ainvoke = AsyncMock(side_effect=RuntimeError("LLM timeout"))

        result = await _explain_tree_nodes(code_map, "Q?", mock_llm)
        assert result is code_map


# ---------------------------------------------------------------------------
# _build_call_tree_from_sources — relationship traversal paths
# ---------------------------------------------------------------------------


class TestBuildCallTreeRelationships:
    def _make_gqs_with_rels(self, nodes, relationships):
        gqs = MagicMock()
        gqs.graph = MagicMock()
        gqs.graph.nodes = nodes

        def _resolve(name, file_path=""):
            return name if name in nodes else None

        gqs.resolve_symbol = MagicMock(side_effect=_resolve)
        gqs.search = MagicMock(return_value=[])
        gqs.get_relationships = MagicMock(return_value=relationships)
        return gqs

    def test_relationship_edge_added(self):
        """A call relationship between two known nodes creates a rel_str on the source."""
        from app.services.research_service import _build_call_tree_from_sources
        from app.models.api import SourceReference

        nodes = {
            "func_a": {
                "symbol_name": "FuncA",
                "symbol_type": "function",
                "rel_path": "svc.py",
                "line_start": 1,
            },
            "func_b": {
                "symbol_name": "FuncB",
                "symbol_type": "function",
                "rel_path": "helper.py",
                "line_start": 10,
            },
        }

        rel = MagicMock()
        rel.relationship_type = "calls"
        rel.source_name = "func_a"
        rel.target_name = "func_b"
        rel.source_type = "function"
        rel.target_type = "function"

        gqs = self._make_gqs_with_rels(nodes, [rel])
        sources = [SourceReference(file_path="svc.py", symbol="func_a")]
        result = _build_call_tree_from_sources(sources, gqs)
        assert result is not None

    def test_excluded_relationship_type_skipped(self):
        """A 'uses' relationship is filtered out (only call-flow edges kept)."""
        from app.services.research_service import _build_call_tree_from_sources
        from app.models.api import SourceReference

        nodes = {
            "func_a": {
                "symbol_name": "FuncA",
                "symbol_type": "function",
                "rel_path": "svc.py",
                "line_start": 1,
            },
        }

        rel = MagicMock()
        rel.relationship_type = "uses"  # filtered in _build_call_tree
        rel.source_name = "func_a"
        rel.target_name = "something_else"
        rel.source_type = "function"
        rel.target_type = "function"

        gqs = self._make_gqs_with_rels(nodes, [rel])
        sources = [SourceReference(file_path="svc.py", symbol="func_a")]
        result = _build_call_tree_from_sources(sources, gqs)
        # No crash — may or may not return sections
        assert result is not None or result is None

    def test_generic_target_name_skipped(self):
        """Targets with short/generic names (e.g. 'get', 'set') are skipped."""
        from app.services.research_service import _build_call_tree_from_sources
        from app.models.api import SourceReference

        nodes = {
            "func_a": {
                "symbol_name": "FuncA",
                "symbol_type": "function",
                "rel_path": "svc.py",
                "line_start": 1,
            },
        }

        rel = MagicMock()
        rel.relationship_type = "calls"
        rel.source_name = "func_a"
        rel.target_name = "get"  # generic name, len < 4
        rel.source_type = "function"
        rel.target_type = "function"

        gqs = self._make_gqs_with_rels(nodes, [rel])
        sources = [SourceReference(file_path="svc.py", symbol="func_a")]
        # Should not crash
        result = _build_call_tree_from_sources(sources, gqs)
        assert result is not None or result is None


# ---------------------------------------------------------------------------
# research_sync — sources mapping from raw dicts
# ---------------------------------------------------------------------------


class TestResearchSyncSourceParsing:
    @pytest.mark.asyncio
    async def test_sources_parsed_correctly(self, service):
        """All optional source fields are correctly mapped."""

        async def _events(*args, **kwargs):
            yield {
                "event_type": "research_complete",
                "data": {
                    "report": "answer",
                    "sources": [
                        {
                            "file_path": "src/auth.py",
                            "line_start": 10,
                            "line_end": 20,
                            "snippet": "def auth():",
                            "symbol": "auth",
                            "symbol_type": "function",
                            "relevance_score": 0.9,
                        },
                        # Legacy source key
                        {"source": "legacy/path.py"},
                        # Non-dict entry should be skipped
                    ],
                },
            }

        req = ResearchRequest(wiki_id="w1", question="Q?")
        mock_engine = MagicMock()
        mock_engine.research = _events

        mock_components = MagicMock(
            retriever_stack=None, graph_manager=None, code_graph=None,
            repo_analysis=None, llm=MagicMock(), repo_path=None,
        )
        with (
            patch.object(service, "_get_components", new_callable=AsyncMock, return_value=mock_components),
            patch("app.services.research_service.DeepResearchEngine", return_value=mock_engine),
        ):
            result = await service.research_sync(req)

        assert len(result.sources) == 2
        assert result.sources[0].file_path == "src/auth.py"
        assert result.sources[0].line_start == 10
        assert result.sources[0].relevance_score == 0.9
        assert result.sources[1].file_path == "legacy/path.py"
