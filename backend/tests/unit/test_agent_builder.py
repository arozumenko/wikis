"""Unit tests for :func:`build_agent_for_incremental_refresh` (#142).

Closes the gap PR #139 left open: the structural-regime handler now
gets a real :class:`OptimizedWikiGenerationAgent` instead of the stub
that always returns False.

The tests verify:

* The builder constructs an agent against a real ``.wiki.db``.
* The constructed agent's ``regenerate_single_page`` reads the
  persisted ``page_spec_json`` and produces markdown (via stubbed
  ``_generate_simple`` so we don't need a real LLM).
* The stub indexer doesn't break the agent's defensive
  ``getattr(..., None)`` reads.
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import patch

import pytest

from app.core.unified_db import UnifiedWikiDB
from app.services.agent_builder import (
    _StubIncrementalIndexer,
    build_agent_for_incremental_refresh,
)


# ---------------------------------------------------------------------------
# _StubIncrementalIndexer
# ---------------------------------------------------------------------------


class TestStubIndexer:
    def test_exposes_required_attributes_as_none(self) -> None:
        stub = _StubIncrementalIndexer()
        # Defensive getattr patterns in the agent return None for these,
        # which steers it into fallback branches that don't need disk access.
        assert stub.relationship_graph is None
        assert stub.source_dir is None
        assert stub.repo_path is None
        assert stub.graph_manager is None

    def test_get_repo_root_returns_none(self) -> None:
        # Triggers the agent's "no repo on disk" fallback in
        # _get_relevant_content_for_page strategy 3.
        assert _StubIncrementalIndexer().get_repo_root() is None

    def test_get_all_documents_returns_empty(self) -> None:
        # Only used in full-regen analyze_repository; single-page path
        # never calls this, but defensive.
        assert _StubIncrementalIndexer().get_all_documents() == []

    def test_unified_db_path_round_trip(self) -> None:
        stub = _StubIncrementalIndexer(unified_db_path="/tmp/foo.wiki.db")
        assert stub._unified_db_path == "/tmp/foo.wiki.db"


# ---------------------------------------------------------------------------
# build_agent_for_incremental_refresh
# ---------------------------------------------------------------------------


@pytest.fixture()
def fixture(tmp_path):
    """Seed a wiki DB + return all the inputs the builder needs."""
    db = UnifiedWikiDB(tmp_path / "agent_builder.wiki.db", embedding_dim=8)
    # Seed a page with page_spec_json so regenerate_single_page has data.
    db.upsert_wiki_page_with_symbols(
        {
            "page_id": "page-1",
            "wiki_id": "w",
            "title": "Test Page",
            "anchor_slug": "test-page",
            "page_spec_json": (
                '{"page_name": "Test Page", '
                '"page_order": 0, '
                '"description": "d", '
                '"content_focus": "c", '
                '"rationale": "r"}'
            ),
        },
        [],
    )

    wiki_record = SimpleNamespace(
        id="w",
        repo_url="https://github.com/example/widgets",
        branch="main",
        title="W",
        page_count=1,
        status="complete",
    )

    settings = SimpleNamespace(
        cache_dir=str(tmp_path),
        llm_provider="openai",
        llm_model="gpt-4o-mini",
    )

    class _StubLLM:
        def invoke(self, *_args, **_kwargs):
            from langchain_core.messages import AIMessage
            return AIMessage(content="# Test Page\n\nRegenerated content.")

    yield db, wiki_record, _StubLLM(), settings
    db.close()


class TestBuildAgent:
    def test_constructs_agent_against_real_db(self, fixture) -> None:
        db, wiki_record, llm, settings = fixture
        # Patch create_embeddings to skip the LLM-provider init dance
        # (would otherwise try to load real Anthropic/OpenAI clients).
        with patch(
            "app.services.llm_factory.create_embeddings",
            side_effect=Exception("test: embeddings unavailable"),
        ):
            agent = build_agent_for_incremental_refresh(
                wiki_record, db, llm, settings,
            )

        assert agent is not None
        # The agent holds onto the stub indexer + a real retriever.
        assert isinstance(agent.indexer, _StubIncrementalIndexer)
        assert agent.retriever_stack is not None
        # repo_url + branch flowed through.
        assert agent.repository_url == "https://github.com/example/widgets"
        assert agent.branch == "main"

    def test_builder_pre_wires_cluster_db(self, fixture) -> None:
        """Regression for the Critical finding in the PR #143 review:
        without _cluster_db pre-wiring, regenerate_single_page falls
        through to _find_unified_db() which is fragile (cache_index
        drift, glob fallback). The builder must wire it directly."""
        db, wiki_record, llm, settings = fixture
        with patch(
            "app.services.llm_factory.create_embeddings",
            side_effect=Exception("test"),
        ):
            agent = build_agent_for_incremental_refresh(
                wiki_record, db, llm, settings,
            )

        assert agent is not None
        assert agent._cluster_db is db
        assert agent._cluster_db_path == str(db.db_path)

    def test_returns_none_when_retriever_construction_fails(
        self, fixture,
    ) -> None:
        db, wiki_record, llm, settings = fixture
        with patch(
            "app.core.unified_retriever.UnifiedRetriever",
            side_effect=RuntimeError("simulated retriever boom"),
        ):
            agent = build_agent_for_incremental_refresh(
                wiki_record, db, llm, settings,
            )
        assert agent is None

    def test_constructed_agent_can_regenerate_single_page(self, fixture) -> None:
        """End-to-end: builder → agent → regenerate_single_page →
        markdown. Stubs _generate_simple + _get_relevant_content_for_page
        so we don't need real retrieval + LLM round-trips."""
        db, wiki_record, llm, settings = fixture
        with patch(
            "app.services.llm_factory.create_embeddings",
            side_effect=Exception("test"),
        ):
            agent = build_agent_for_incremental_refresh(
                wiki_record, db, llm, settings,
            )

        assert agent is not None
        # No manual _cluster_db wiring needed — the builder pre-wires it
        # to the storage handle the caller passed in. This test exercises
        # the same code path production hits.
        assert agent._cluster_db is db

        # Stub the two heavy methods regenerate_single_page calls so we
        # don't need a real LLM or retrieval. The wiring under test is
        # everything around them: storage read of page_spec_json →
        # PageSpec deserialize → method dispatch → return value.
        with patch.object(
            agent, "_get_relevant_content_for_page",
            return_value={"files": [], "content": "x", "total_docs": 0},
        ), patch.object(
            agent, "_generate_simple",
            return_value="# Test Page\n\nStubbed regenerated body.",
        ):
            result = agent.regenerate_single_page(
                "page-1", repository_context="ctx",
            )

        assert result is not None
        assert "Stubbed regenerated body" in result
