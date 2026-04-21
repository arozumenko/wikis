"""Unit tests for description injection into the codemap refinement prompt.

Covers:
- When repo_analysis has a description, prompt contains '## Project/Wiki Context'
  and the description text.
- When description is None/absent, the prefix is not added.
- Descriptions longer than 500 chars are truncated to 500 in the prompt.
"""
from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from pydantic import SecretStr

from app.config import Settings
from app.models.api import (
    CodeMapData,
    CodeMapSection,
    CodeMapSymbol,
    ResearchRequest,
)
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


def _make_fake_code_map() -> CodeMapData:
    sym = CodeMapSymbol(
        id="s0_sym0",
        name="authenticate",
        symbol_type="function",
        file_path="auth.py",
    )
    sec = CodeMapSection(
        id="section_0",
        title="auth.py",
        file_path="auth.py",
        symbols=[sym],
    )
    return CodeMapData(sections=[sec])


def _make_components(repo_analysis: dict | None, llm_ainvoke_resp_content: str) -> MagicMock:
    mock_gm = MagicMock()
    mock_gm.fts_index = None

    mock_llm_resp = MagicMock()
    mock_llm_resp.content = llm_ainvoke_resp_content

    mock_llm = AsyncMock()
    mock_llm.ainvoke = AsyncMock(return_value=mock_llm_resp)

    return MagicMock(
        retriever_stack=MagicMock(),
        graph_manager=mock_gm,
        code_graph=MagicMock(),
        repo_analysis=repo_analysis,
        llm=mock_llm,
        repo_path=None,
    )


async def _ask_events_with_answer(*args, **kwargs):
    yield {
        "event_type": "ask_complete",
        "data": {"answer": "authenticate is the entry point."},
    }


class TestCodemapRefinementDescriptionInjection:
    @pytest.mark.asyncio
    async def test_description_present_in_refinement_prompt(self, service):
        """When repo_analysis has a description, the refinement prompt includes
        '## Project/Wiki Context' and the description text."""
        req = ResearchRequest(
            wiki_id="w1",
            question="How does auth work?",
            research_type="codemap",
        )
        fake_code_map = _make_fake_code_map()
        describe_resp = (
            '{"descriptions": {"s0_sym0": "Handles auth."}, '
            '"summary": "Auth system.", "call_stacks": []}'
        )
        refine_resp = "Refined answer about auth."

        # The explain LLM and the refine LLM are both mock_llm.ainvoke.
        # ainvoke is called twice: first for explain_nodes, then for refine.
        # We return the describe_resp for the first call and refine_resp for the second.
        mock_gm = MagicMock()
        mock_gm.fts_index = None

        explain_resp = MagicMock()
        explain_resp.content = describe_resp
        refine_resp_mock = MagicMock()
        refine_resp_mock.content = refine_resp

        mock_llm = AsyncMock()
        mock_llm.ainvoke = AsyncMock(return_value=refine_resp_mock)

        mock_explain_llm = AsyncMock()
        mock_explain_llm.ainvoke = AsyncMock(return_value=explain_resp)

        mock_components = MagicMock(
            retriever_stack=MagicMock(),
            graph_manager=mock_gm,
            code_graph=MagicMock(),
            repo_analysis={"description": "A wiki about the authentication system."},
            llm=mock_llm,
            repo_path=None,
        )

        mock_ask_engine = MagicMock()
        mock_ask_engine.ask = _ask_events_with_answer

        with (
            patch.object(
                service, "_get_components", new_callable=AsyncMock, return_value=mock_components
            ),
            patch("app.core.ask_engine.AskEngine", return_value=mock_ask_engine),
            patch("app.services.llm_factory.create_llm", return_value=mock_explain_llm),
            patch("app.services.research_service.GraphQueryService", return_value=MagicMock()),
            patch(
                "app.services.research_service._build_call_tree_from_sources",
                return_value=fake_code_map,
            ),
        ):
            events = [e async for e in service.codemap_stream(req)]

        # Ensure the pipeline completed
        event_types = [e["event_type"] for e in events]
        assert "research_complete" in event_types

        # Verify the refine LLM was invoked
        mock_llm.ainvoke.assert_called_once()
        call_args = mock_llm.ainvoke.call_args
        messages = call_args[0][0]  # positional first arg is list of messages
        prompt_text = messages[0].content

        assert "## Project/Wiki Context" in prompt_text
        assert "A wiki about the authentication system." in prompt_text

    @pytest.mark.asyncio
    async def test_no_description_no_prefix_in_prompt(self, service):
        """When repo_analysis has no description, '## Project/Wiki Context' is absent."""
        req = ResearchRequest(
            wiki_id="w1",
            question="How does auth work?",
            research_type="codemap",
        )
        fake_code_map = _make_fake_code_map()
        describe_resp = (
            '{"descriptions": {"s0_sym0": "Handles auth."}, '
            '"summary": "Auth system.", "call_stacks": []}'
        )

        mock_gm = MagicMock()
        mock_gm.fts_index = None

        explain_resp = MagicMock()
        explain_resp.content = describe_resp
        refine_resp_mock = MagicMock()
        refine_resp_mock.content = "Refined answer."

        mock_llm = AsyncMock()
        mock_llm.ainvoke = AsyncMock(return_value=refine_resp_mock)

        mock_explain_llm = AsyncMock()
        mock_explain_llm.ainvoke = AsyncMock(return_value=explain_resp)

        mock_components = MagicMock(
            retriever_stack=MagicMock(),
            graph_manager=mock_gm,
            code_graph=MagicMock(),
            repo_analysis=None,  # No description
            llm=mock_llm,
            repo_path=None,
        )

        mock_ask_engine = MagicMock()
        mock_ask_engine.ask = _ask_events_with_answer

        with (
            patch.object(
                service, "_get_components", new_callable=AsyncMock, return_value=mock_components
            ),
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

        mock_llm.ainvoke.assert_called_once()
        call_args = mock_llm.ainvoke.call_args
        messages = call_args[0][0]
        prompt_text = messages[0].content

        assert "## Project/Wiki Context" not in prompt_text
        # Prompt starts with the standard assistant instruction
        assert prompt_text.startswith("You are a code analysis assistant.")

    @pytest.mark.asyncio
    async def test_description_absent_key_no_prefix_in_prompt(self, service):
        """When repo_analysis exists but has no 'description' key, prefix is absent."""
        req = ResearchRequest(
            wiki_id="w1",
            question="How does auth work?",
            research_type="codemap",
        )
        fake_code_map = _make_fake_code_map()
        describe_resp = (
            '{"descriptions": {"s0_sym0": "Handles auth."}, '
            '"summary": "Auth system.", "call_stacks": []}'
        )

        mock_gm = MagicMock()
        mock_gm.fts_index = None

        explain_resp = MagicMock()
        explain_resp.content = describe_resp
        refine_resp_mock = MagicMock()
        refine_resp_mock.content = "Refined answer."

        mock_llm = AsyncMock()
        mock_llm.ainvoke = AsyncMock(return_value=refine_resp_mock)

        mock_explain_llm = AsyncMock()
        mock_explain_llm.ainvoke = AsyncMock(return_value=explain_resp)

        mock_components = MagicMock(
            retriever_stack=MagicMock(),
            graph_manager=mock_gm,
            code_graph=MagicMock(),
            repo_analysis={"other_key": "some value"},  # description key missing
            llm=mock_llm,
            repo_path=None,
        )

        mock_ask_engine = MagicMock()
        mock_ask_engine.ask = _ask_events_with_answer

        with (
            patch.object(
                service, "_get_components", new_callable=AsyncMock, return_value=mock_components
            ),
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

        mock_llm.ainvoke.assert_called_once()
        messages = mock_llm.ainvoke.call_args[0][0]
        prompt_text = messages[0].content
        assert "## Project/Wiki Context" not in prompt_text

    @pytest.mark.asyncio
    async def test_long_description_truncated_to_500_chars(self, service):
        """Descriptions longer than 500 chars are truncated to exactly 500 in the prompt."""
        req = ResearchRequest(
            wiki_id="w1",
            question="How does auth work?",
            research_type="codemap",
        )
        fake_code_map = _make_fake_code_map()
        long_description = "X" * 800  # 800 chars, should be truncated to 500
        describe_resp = (
            '{"descriptions": {"s0_sym0": "Handles auth."}, '
            '"summary": "Auth system.", "call_stacks": []}'
        )

        mock_gm = MagicMock()
        mock_gm.fts_index = None

        explain_resp = MagicMock()
        explain_resp.content = describe_resp
        refine_resp_mock = MagicMock()
        refine_resp_mock.content = "Refined answer."

        mock_llm = AsyncMock()
        mock_llm.ainvoke = AsyncMock(return_value=refine_resp_mock)

        mock_explain_llm = AsyncMock()
        mock_explain_llm.ainvoke = AsyncMock(return_value=explain_resp)

        mock_components = MagicMock(
            retriever_stack=MagicMock(),
            graph_manager=mock_gm,
            code_graph=MagicMock(),
            repo_analysis={"description": long_description},
            llm=mock_llm,
            repo_path=None,
        )

        mock_ask_engine = MagicMock()
        mock_ask_engine.ask = _ask_events_with_answer

        with (
            patch.object(
                service, "_get_components", new_callable=AsyncMock, return_value=mock_components
            ),
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

        mock_llm.ainvoke.assert_called_once()
        messages = mock_llm.ainvoke.call_args[0][0]
        prompt_text = messages[0].content

        assert "## Project/Wiki Context" in prompt_text
        # The truncated description (500 X's) must appear, but the full 800 must not
        assert "X" * 500 in prompt_text
        assert "X" * 501 not in prompt_text
