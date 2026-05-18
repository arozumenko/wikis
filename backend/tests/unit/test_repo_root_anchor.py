"""#181 Bug B — repo-root anchor docs as last-resort context.

The motivating bug: a Node.js wiki's "Development Environment Setup"
page received no retrieved docs (no ``target_symbols``, vector search
empty), so the page-content agent saw only a generic repo summary and
wrote an LLM-composed "no source supplied" warning instead of actual
documentation.

These tests pin the new fallback behavior: when retrieval comes up
empty and the cached clone is on disk, surface top-level build / config
/ deploy files as Documents so the LLM has real anchors to cite.
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import patch

import pytest

from app.core.unified_db import UnifiedWikiDB
from app.services.agent_builder import build_agent_for_incremental_refresh


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def agent_fixture(tmp_path):
    """Build a real :class:`OptimizedWikiGenerationAgent` plus a fake
    repo-root directory the agent's indexer points at."""
    repo_root = tmp_path / "repo"
    repo_root.mkdir()

    db_path = tmp_path / "anchor.wiki.db"
    db = UnifiedWikiDB(db_path, embedding_dim=8)
    db.upsert_wiki_page_with_symbols(
        {
            "page_id": "page-anchor",
            "wiki_id": "w-anchor",
            "title": "Dev Setup",
            "anchor_slug": "dev-setup",
            "page_spec_json": (
                '{"page_name": "Dev Setup", "page_order": 0, '
                '"description": "d", "content_focus": "c", "rationale": "r"}'
            ),
        },
        [],
    )

    wiki_record = SimpleNamespace(
        id="w-anchor",
        repo_url="https://github.com/example/node-app",
        branch="main",
        title="Node App",
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

            return AIMessage(content="# stub\n")

    with patch("app.services.llm_factory.create_embeddings", return_value=None):
        agent = build_agent_for_incremental_refresh(
            wiki_record, db, _StubLLM(), settings
        )
    if agent is None:
        pytest.skip("agent builder returned None — environment can't run this test")

    # Point the stub indexer at the fake repo root so the anchor helper
    # has files to surface.
    agent.indexer.get_repo_root = lambda: str(repo_root)

    yield agent, repo_root
    db.close()


# ---------------------------------------------------------------------------
# _get_repo_root_anchor_docs
# ---------------------------------------------------------------------------


class TestRepoRootAnchorDocs:
    def test_surfaces_priority_files_first(self, agent_fixture):
        """Setup-relevant files (README, Dockerfile, package.json) must
        show up before random files when both exist."""
        agent, repo_root = agent_fixture
        (repo_root / "random.txt").write_text("noise\n" * 5)
        (repo_root / "README.md").write_text("# Project\n\nIntro\n")
        (repo_root / "Dockerfile.prod").write_text("FROM node:20\nRUN npm ci\n")
        (repo_root / "package.json").write_text('{"name":"x","version":"1.0.0"}')

        docs = agent._get_repo_root_anchor_docs()

        sources = [d.metadata["source"] for d in docs]
        assert "README.md" in sources
        assert "Dockerfile.prod" in sources
        assert "package.json" in sources
        # Priority files appear before the random file.
        readme_idx = sources.index("README.md")
        random_idx = sources.index("random.txt") if "random.txt" in sources else 999
        assert readme_idx < random_idx

    def test_returns_empty_when_repo_root_unavailable(self, agent_fixture):
        agent, _ = agent_fixture
        agent.indexer.get_repo_root = lambda: None

        assert agent._get_repo_root_anchor_docs() == []

    def test_returns_empty_when_repo_root_path_missing_on_disk(self, agent_fixture):
        agent, _ = agent_fixture
        agent.indexer.get_repo_root = lambda: "/nonexistent/path/xyzzy"

        assert agent._get_repo_root_anchor_docs() == []

    def test_skips_empty_files(self, agent_fixture):
        agent, repo_root = agent_fixture
        (repo_root / "empty.txt").write_text("")
        (repo_root / "README.md").write_text("real content")

        sources = [d.metadata["source"] for d in agent._get_repo_root_anchor_docs()]

        assert "README.md" in sources
        assert "empty.txt" not in sources

    def test_skips_oversized_files(self, agent_fixture):
        """Lockfiles / minified bundles must not flood the context."""
        agent, repo_root = agent_fixture
        (repo_root / "package.json").write_text('{"name":"x"}')
        (repo_root / "huge.lock").write_text("X" * 200_000)

        sources = [d.metadata["source"] for d in agent._get_repo_root_anchor_docs()]

        assert "package.json" in sources
        assert "huge.lock" not in sources

    def test_respects_char_cap(self, agent_fixture):
        """Total char count across all returned docs must stay under cap."""
        agent, repo_root = agent_fixture
        for i in range(20):
            (repo_root / f"file_{i:02d}.md").write_text("x" * 1000)

        docs = agent._get_repo_root_anchor_docs(char_cap=3000)

        assert sum(len(d.page_content) for d in docs) <= 3000

    def test_metadata_marks_anchor_documents(self, agent_fixture):
        """Downstream formatters need to identify these docs."""
        agent, repo_root = agent_fixture
        (repo_root / "README.md").write_text("content")

        docs = agent._get_repo_root_anchor_docs()

        assert docs, "expected at least one anchor doc"
        for doc in docs:
            assert doc.metadata.get("repo_root_anchor") is True
            assert doc.metadata.get("is_documentation") is True

    def test_node_repo_files_all_surface(self, agent_fixture):
        """Mirror of the onetest-ai/core layout (the actual reproducer)."""
        agent, repo_root = agent_fixture
        (repo_root / "README.md").write_text("# OneTest\n")
        (repo_root / "package.json").write_text('{"name":"onetest"}')
        (repo_root / "Dockerfile.prod").write_text("FROM node:20\n")
        (repo_root / "nginx.conf").write_text("server { listen 80; }\n")
        (repo_root / "build-for-deploy.sh").write_text("#!/bin/bash\nset -e\n")
        (repo_root / "vite.config.js").write_text("export default {}\n")

        sources = [d.metadata["source"] for d in agent._get_repo_root_anchor_docs()]

        # All 6 must appear — that's the bundle the page-content agent
        # was missing in production.
        for expected in (
            "README.md",
            "package.json",
            "Dockerfile.prod",
            "nginx.conf",
            "build-for-deploy.sh",
            "vite.config.js",
        ):
            assert expected in sources, f"missing {expected} from {sources}"


# ---------------------------------------------------------------------------
# _safely_get_repo_root
# ---------------------------------------------------------------------------


class TestSafelyGetRepoRoot:
    def test_returns_indexer_repo_root(self, agent_fixture):
        agent, repo_root = agent_fixture
        assert agent._safely_get_repo_root() == str(repo_root)

    def test_swallows_exceptions(self, agent_fixture):
        agent, _ = agent_fixture

        def _boom():
            raise RuntimeError("indexer offline")

        agent.indexer.get_repo_root = _boom
        assert agent._safely_get_repo_root() is None

    def test_handles_missing_method(self, agent_fixture):
        agent, _ = agent_fixture
        # Replace indexer with one that has no get_repo_root at all.
        agent.indexer = SimpleNamespace()
        assert agent._safely_get_repo_root() is None
