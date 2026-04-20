"""Tests for deep research service and endpoint."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from httpx import ASGITransport, AsyncClient
from pydantic import SecretStr

from app.config import Settings
from app.main import create_app
from app.models.api import ResearchRequest, ResearchResponse
from app.services.research_service import ResearchService
from app.storage.local import LocalArtifactStorage


def _settings() -> Settings:
    return Settings(llm_api_key=SecretStr("test-key"))


@pytest.fixture
def storage(tmp_path):
    return LocalArtifactStorage(str(tmp_path))


@pytest.fixture
def service(storage):
    return ResearchService(_settings(), storage)


async def _mock_research_events(*args, **kwargs):
    """Simulate DeepResearchEngine.research() yielding events."""
    yield {"event_type": "research_start", "data": {"session_id": "s1"}}
    yield {"event_type": "thinking_step", "data": {"tool": "search_symbols", "result": "found 5 symbols"}}
    yield {"event_type": "thinking_step", "data": {"tool": "get_relationships_tool", "result": "3 deps"}}
    yield {
        "event_type": "research_complete",
        "data": {
            "answer": "The system uses a layered architecture with dependency injection.",
            "sources": [{"file_path": "src/core.py", "snippet": "class Container:"}],
        },
    }


class TestResearchServiceUnit:
    @pytest.mark.asyncio
    async def test_research_sync_returns_response(self, service):
        req = ResearchRequest(wiki_id="w1", question="Architecture overview")
        mock_engine = MagicMock()
        mock_engine.research = _mock_research_events

        with (
            patch.object(
                service,
                "_get_components",
                new_callable=AsyncMock,
                return_value=MagicMock(
                    retriever_stack=None, graph_manager=None, code_graph=None, repo_analysis=None, llm=MagicMock()
                ),
            ),
            patch("app.services.research_service.DeepResearchEngine", return_value=mock_engine),
        ):
            result = await service.research_sync(req)

        assert isinstance(result, ResearchResponse)
        assert "layered architecture" in result.answer
        assert len(result.sources) == 1
        assert len(result.research_steps) == 2

    @pytest.mark.asyncio
    async def test_research_stream_yields_events(self, service):
        req = ResearchRequest(wiki_id="w1", question="Architecture")
        mock_engine = MagicMock()
        mock_engine.research = _mock_research_events

        with (
            patch.object(
                service,
                "_get_components",
                new_callable=AsyncMock,
                return_value=MagicMock(
                    retriever_stack=None, graph_manager=None, code_graph=None, repo_analysis=None, llm=MagicMock()
                ),
            ),
            patch("app.services.research_service.DeepResearchEngine", return_value=mock_engine),
        ):
            events = []
            async for event in service.research_stream(req):
                events.append(event)

        assert len(events) == 4
        assert events[0]["event_type"] == "research_start"
        assert events[-1]["event_type"] == "research_complete"

    @pytest.mark.asyncio
    async def test_research_steps_collected(self, service):
        req = ResearchRequest(wiki_id="w1", question="How?")
        mock_engine = MagicMock()
        mock_engine.research = _mock_research_events

        with (
            patch.object(
                service,
                "_get_components",
                new_callable=AsyncMock,
                return_value=MagicMock(
                    retriever_stack=None, graph_manager=None, code_graph=None, repo_analysis=None, llm=MagicMock()
                ),
            ),
            patch("app.services.research_service.DeepResearchEngine", return_value=mock_engine),
        ):
            result = await service.research_sync(req)

        assert "search_symbols" in result.research_steps
        assert "get_relationships_tool" in result.research_steps


class TestResearchTypePassthrough:
    """Tests that research_type from the API request reaches the engine."""

    @pytest.mark.asyncio
    async def test_research_stream_passes_research_type_to_engine(self, service):
        """DeepResearchEngine must be constructed with the research_type from the request."""
        req = ResearchRequest(wiki_id="w1", question="How does auth work?", research_type="security")
        mock_engine = MagicMock()
        mock_engine.research = _mock_research_events

        mock_components = MagicMock(
            retriever_stack=None,
            graph_manager=None,
            code_graph=None,
            repo_analysis=None,
            llm=MagicMock(),
        )

        with (
            patch.object(service, "_get_components", new_callable=AsyncMock, return_value=mock_components),
            patch("app.services.research_service.DeepResearchEngine", return_value=mock_engine) as mock_cls,
        ):
            async for _ in service.research_stream(req):
                pass

        mock_cls.assert_called_once()
        _, ctor_kwargs = mock_cls.call_args
        assert "config" in ctor_kwargs, "config kwarg was not passed to DeepResearchEngine"
        assert ctor_kwargs["config"].research_type == "security"

    @pytest.mark.asyncio
    async def test_research_stream_default_research_type_is_general(self, service):
        """When no research_type is given, the engine should receive 'general'."""
        req = ResearchRequest(wiki_id="w1", question="Overview")
        mock_engine = MagicMock()
        mock_engine.research = _mock_research_events

        mock_components = MagicMock(
            retriever_stack=None,
            graph_manager=None,
            code_graph=None,
            repo_analysis=None,
            llm=MagicMock(),
        )

        with (
            patch.object(service, "_get_components", new_callable=AsyncMock, return_value=mock_components),
            patch("app.services.research_service.DeepResearchEngine", return_value=mock_engine) as mock_cls,
        ):
            async for _ in service.research_stream(req):
                pass

        _, ctor_kwargs = mock_cls.call_args
        assert ctor_kwargs["config"].research_type == "general"

    def test_different_research_types_produce_different_prompts(self):
        """get_research_prompt must embed type-specific guidance in its output."""
        from app.core.deep_research.research_prompts import get_research_prompt

        general_prompt = get_research_prompt(research_type="general", topic="How does auth work?")
        security_prompt = get_research_prompt(research_type="security", topic="How does auth work?")
        architecture_prompt = get_research_prompt(research_type="architecture", topic="How does auth work?")

        # All three have the same topic but different guidance sections
        assert general_prompt != security_prompt
        assert general_prompt != architecture_prompt
        assert security_prompt != architecture_prompt

        # Security prompt must mention security-specific concepts
        assert "authentication" in security_prompt.lower() or "authorization" in security_prompt.lower()
        # Architecture prompt must mention architecture-specific concepts
        assert "component" in architecture_prompt.lower() or "scalability" in architecture_prompt.lower()


class TestResearchConfigImport:
    """Tests for #431: the ResearchConfig exported from app.core.deep_research must be the engine's class."""

    def test_research_config_import_comes_from_engine(self):
        """Importing ResearchConfig from the package gives the engine implementation."""
        from app.core.deep_research import ResearchConfig
        from app.core.deep_research.research_engine import ResearchConfig as EngineConfig

        assert ResearchConfig is EngineConfig

    def test_research_config_has_correct_defaults(self):
        """Engine ResearchConfig defaults: max_iterations=15, enable_subagents=False."""
        from app.core.deep_research import ResearchConfig

        cfg = ResearchConfig()
        assert cfg.max_iterations == 15
        assert cfg.enable_subagents is False

    def test_research_config_research_type_override(self):
        """research_type can be overridden at construction time."""
        from app.core.deep_research import ResearchConfig

        cfg = ResearchConfig(research_type="security")
        assert cfg.research_type == "security"
        # Other defaults remain unchanged
        assert cfg.max_iterations == 15
        assert cfg.enable_subagents is False

    def test_research_request_has_no_enable_subagents_field(self):
        """enable_subagents was removed — SubAgentMiddleware is not in the engine."""
        req = ResearchRequest(wiki_id="w1", question="Q?")
        assert not hasattr(req, "enable_subagents")


class TestResearchSimilarityThreshold:
    """Tests for #432: research similarity_threshold configurable via settings."""

    @pytest.fixture
    def service_custom_threshold(self):
        s = Settings(llm_api_key=SecretStr("k"), research_similarity_threshold=0.6)
        storage = LocalArtifactStorage(base_path="/tmp/test-artifacts-thresh")
        return ResearchService(settings=s, storage=storage)

    @pytest.mark.asyncio
    async def test_similarity_threshold_passed_to_engine(self, service_custom_threshold):
        """ResearchConfig.similarity_threshold must reflect settings."""
        req = ResearchRequest(wiki_id="w1", question="Q?")
        mock_engine = MagicMock()
        mock_engine.research = _mock_research_events
        mock_components = MagicMock(
            retriever_stack=None,
            graph_manager=None,
            code_graph=None,
            repo_analysis=None,
            llm=MagicMock(),
        )
        with (
            patch.object(
                service_custom_threshold, "_get_components", new_callable=AsyncMock, return_value=mock_components
            ),
            patch("app.services.research_service.DeepResearchEngine", return_value=mock_engine) as mock_cls,
        ):
            async for _ in service_custom_threshold.research_stream(req):
                pass

        _, ctor_kwargs = mock_cls.call_args
        assert ctor_kwargs["config"].similarity_threshold == 0.6

    @pytest.mark.asyncio
    async def test_default_similarity_threshold_is_zero(self, service):
        """Default settings produce similarity_threshold=0.0 (disabled)."""
        req = ResearchRequest(wiki_id="w1", question="Q?")
        mock_engine = MagicMock()
        mock_engine.research = _mock_research_events
        mock_components = MagicMock(
            retriever_stack=None,
            graph_manager=None,
            code_graph=None,
            repo_analysis=None,
            llm=MagicMock(),
        )
        with (
            patch.object(service, "_get_components", new_callable=AsyncMock, return_value=mock_components),
            patch("app.services.research_service.DeepResearchEngine", return_value=mock_engine) as mock_cls,
        ):
            async for _ in service.research_stream(req):
                pass

        _, ctor_kwargs = mock_cls.call_args
        assert ctor_kwargs["config"].similarity_threshold == 0.0

    def test_research_config_similarity_threshold_field_exists(self):
        from app.core.deep_research import ResearchConfig

        cfg = ResearchConfig()
        assert cfg.similarity_threshold == 0.0
        cfg2 = ResearchConfig(similarity_threshold=0.8)
        assert cfg2.similarity_threshold == 0.8


class TestReadSourceFile:
    """Tests for #433: read_source_file tool and repo_path plumbing."""

    @pytest.mark.asyncio
    async def test_repo_path_passed_to_engine(self, service):
        """repo_path from EngineComponents must reach DeepResearchEngine."""
        req = ResearchRequest(wiki_id="w1", question="Q?")
        mock_engine = MagicMock()
        mock_engine.research = _mock_research_events
        mock_components = MagicMock(
            retriever_stack=None,
            graph_manager=None,
            code_graph=None,
            repo_analysis=None,
            llm=MagicMock(),
            repo_path="/tmp/fake-repo",
        )
        with (
            patch.object(service, "_get_components", new_callable=AsyncMock, return_value=mock_components),
            patch("app.services.research_service.DeepResearchEngine", return_value=mock_engine) as mock_cls,
        ):
            async for _ in service.research_stream(req):
                pass

        _, ctor_kwargs = mock_cls.call_args
        assert ctor_kwargs["repo_path"] == "/tmp/fake-repo"

    @pytest.mark.asyncio
    async def test_repo_path_none_when_not_available(self, service):
        """When components.repo_path is None, engine gets None."""
        req = ResearchRequest(wiki_id="w1", question="Q?")
        mock_engine = MagicMock()
        mock_engine.research = _mock_research_events
        mock_components = MagicMock(
            retriever_stack=None,
            graph_manager=None,
            code_graph=None,
            repo_analysis=None,
            llm=MagicMock(),
            repo_path=None,
        )
        with (
            patch.object(service, "_get_components", new_callable=AsyncMock, return_value=mock_components),
            patch("app.services.research_service.DeepResearchEngine", return_value=mock_engine) as mock_cls,
        ):
            async for _ in service.research_stream(req):
                pass

        _, ctor_kwargs = mock_cls.call_args
        assert ctor_kwargs["repo_path"] is None

    def test_read_source_file_tool_reads_file(self, tmp_path):
        """read_source_file tool reads files from repo_path."""
        (tmp_path / "hello.py").write_text("line1\nline2\nline3\n")
        from app.core.deep_research.research_tools import create_codebase_tools

        tools = create_codebase_tools(
            retriever_stack=None,
            graph_manager=None,
            code_graph=None,
            repo_path=str(tmp_path),
        )
        tool_names = [t.name for t in tools]
        assert "read_source_file" in tool_names
        read_tool = next(t for t in tools if t.name == "read_source_file")
        result = read_tool.invoke({"file_path": "hello.py"})
        assert "line1" in result
        assert "line2" in result

    def test_read_source_file_blocks_path_traversal(self, tmp_path):
        """read_source_file must reject paths that escape the repo root."""
        (tmp_path / "ok.py").write_text("safe")
        from app.core.deep_research.research_tools import create_codebase_tools

        tools = create_codebase_tools(
            retriever_stack=None,
            graph_manager=None,
            code_graph=None,
            repo_path=str(tmp_path),
        )
        read_tool = next(t for t in tools if t.name == "read_source_file")
        result = read_tool.invoke({"file_path": "../../etc/passwd"})
        assert "outside the repository" in result.lower() or "error" in result.lower()

    def test_read_source_file_missing_file(self, tmp_path):
        """read_source_file returns error for non-existent files."""
        from app.core.deep_research.research_tools import create_codebase_tools

        tools = create_codebase_tools(
            retriever_stack=None,
            graph_manager=None,
            code_graph=None,
            repo_path=str(tmp_path),
        )
        read_tool = next(t for t in tools if t.name == "read_source_file")
        result = read_tool.invoke({"file_path": "nonexistent.py"})
        assert "not found" in result.lower()

    def test_list_repo_files_tool(self, tmp_path):
        """list_repo_files lists directory contents."""
        (tmp_path / "a.py").write_text("")
        (tmp_path / "b.py").write_text("")
        sub = tmp_path / "sub"
        sub.mkdir()
        from app.core.deep_research.research_tools import create_codebase_tools

        tools = create_codebase_tools(
            retriever_stack=None,
            graph_manager=None,
            code_graph=None,
            repo_path=str(tmp_path),
        )
        list_tool = next(t for t in tools if t.name == "list_repo_files")
        result = list_tool.invoke({"directory": "."})
        assert "a.py" in result
        assert "b.py" in result
        assert "sub/" in result

    def test_tools_omitted_when_no_repo_path(self):
        """File tools must not be included when repo_path is None."""
        from app.core.deep_research.research_tools import create_codebase_tools

        tools = create_codebase_tools(
            retriever_stack=None,
            graph_manager=None,
            code_graph=None,
            repo_path=None,
        )
        tool_names = [t.name for t in tools]
        assert "read_source_file" not in tool_names
        assert "list_repo_files" not in tool_names

    def test_read_source_file_offset_limit(self, tmp_path):
        """read_source_file respects offset and limit parameters."""
        content = "\n".join(f"line{i}" for i in range(1, 21))
        (tmp_path / "big.py").write_text(content)
        from app.core.deep_research.research_tools import create_codebase_tools

        tools = create_codebase_tools(
            retriever_stack=None,
            graph_manager=None,
            code_graph=None,
            repo_path=str(tmp_path),
        )
        read_tool = next(t for t in tools if t.name == "read_source_file")
        result = read_tool.invoke({"file_path": "big.py", "offset": 5, "limit": 3})
        assert "line6" in result
        assert "line8" in result
        assert "line9" not in result


class TestFindRepoClone:
    """Tests for _find_repo_clone helper in toolkit_bridge."""

    def test_find_with_commit_hash(self, tmp_path):
        """Finds clone dir when commit hash matches."""
        from app.services.toolkit_bridge import _find_repo_clone

        clone_dir = tmp_path / "org_repo_main_abc12345"
        clone_dir.mkdir()
        result = _find_repo_clone(str(tmp_path), "https://github.com/org/repo", "main", "abc12345deadbeef")
        assert result == str(clone_dir)

    def test_find_with_glob_fallback(self, tmp_path):
        """Falls back to glob when commit hash doesn't match exactly."""
        from app.services.toolkit_bridge import _find_repo_clone

        clone_dir = tmp_path / "org_repo_main_deadbeef"
        clone_dir.mkdir()
        result = _find_repo_clone(str(tmp_path), "https://github.com/org/repo", "main", "different_hash")
        assert result == str(clone_dir)

    def test_returns_none_when_not_found(self, tmp_path):
        """Returns None when no matching clone exists."""
        from app.services.toolkit_bridge import _find_repo_clone

        result = _find_repo_clone(str(tmp_path), "https://github.com/org/repo", "main", "abc12345")
        assert result is None

    def test_strips_git_suffix(self, tmp_path):
        """Handles .git suffix in repo URL."""
        from app.services.toolkit_bridge import _find_repo_clone

        clone_dir = tmp_path / "org_repo_main_abc12345"
        clone_dir.mkdir()
        result = _find_repo_clone(str(tmp_path), "https://github.com/org/repo.git", "main", "abc12345xxxx")
        assert result == str(clone_dir)


class TestResearchEndpoint:
    @pytest.fixture
    async def client(self):
        app = create_app()
        async with app.router.lifespan_context(app):
            async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as c:
                yield c, app

    @pytest.mark.asyncio
    async def test_research_json_mode(self, client):
        c, app = client
        mock_response = ResearchResponse(answer="Layered arch.", sources=[], research_steps=["search", "analyze"])
        with patch.object(
            app.state.research_service,
            "research_sync",
            new_callable=AsyncMock,
            return_value=mock_response,
        ):
            resp = await c.post("/api/v1/research", json={"wiki_id": "w1", "question": "Arch?"})
            assert resp.status_code == 200
            data = resp.json()
            assert data["answer"] == "Layered arch."
            assert len(data["research_steps"]) == 2

    @pytest.mark.asyncio
    async def test_research_validation_error(self, client):
        c, _ = client
        resp = await c.post("/api/v1/research", json={})
        assert resp.status_code == 422
