"""Integration test for #116 PR 1: wiki page metadata persistence.

End-to-end exercise of the slice that PR 1 adds to the wiki generation
flow — dispatch builds stable page IDs, the per-page persist hook writes
``wiki_pages`` + ``page_symbols`` rows, and a second run on identical
inputs produces identical IDs.

We can't run the full ``OptimizedWikiGenerationAgent`` graph (it would
need a real LLM, retriever, and indexer). Instead we construct the agent
via ``__new__`` and exercise only the surface PR 1 changes:

* ``_resolve_wiki_id``
* ``dispatch_page_generation`` (which only allocates IDs + builds Sends)
* ``_persist_page_metadata`` (which writes the new tables)

This is the smallest unit that proves the end-to-end contract: planner
output → IDs → DB rows survive a regeneration.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from app.core.agents.wiki_graph_optimized import OptimizedWikiGenerationAgent
from app.core.state.wiki_state import (
    PageSpec,
    SectionSpec,
    WikiState,
    WikiStructureSpec,
)
from app.core.storage.incremental import compute_page_id
from app.core.unified_db import UnifiedWikiDB


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_agent(tmp_path) -> tuple[OptimizedWikiGenerationAgent, UnifiedWikiDB]:
    """Build a bare-bones agent that exposes only the surface PR 1 touches.

    Skips the heavy ``__init__`` (indexer/retriever/LLM/graph build) — we're
    not generating a wiki, only exercising the persistence hook + dispatch
    ID allocator.
    """
    db = UnifiedWikiDB(tmp_path / "agent_test.wiki.db", embedding_dim=8)
    # Seed a few nodes so the FK from page_symbols.node_id has real targets.
    # (FKs aren't enforced on the join table itself, but it makes the test
    # truer to production behaviour.)
    for nid in ("auth.AuthService", "auth.login", "cache.CacheManager"):
        db.upsert_node(
            nid,
            rel_path=f"{nid.split('.')[0]}/file.py",
            file_name="file.py",
            language="python",
            symbol_name=nid.split(".")[-1],
            symbol_type="class" if nid.endswith("Service") or nid.endswith("Manager") else "function",
            source_text=f"def {nid.split('.')[-1]}(): pass",
        )

    agent = OptimizedWikiGenerationAgent.__new__(OptimizedWikiGenerationAgent)
    agent.repository_url = "https://github.com/example/widgets.git"
    agent.branch = "main"
    agent._resolved_wiki_id = None
    agent._cluster_db = db
    agent._cluster_db_path = str(db.db_path)
    # _persist_page_metadata expects no other attrs.
    return agent, db


def _make_structure() -> WikiStructureSpec:
    return WikiStructureSpec(
        wiki_title="Widgets Wiki",
        overview="Test wiki.",
        sections=[
            SectionSpec(
                section_name="Authentication",
                section_order=0,
                description="Auth bits",
                rationale="Auth is critical",
                pages=[
                    PageSpec(
                        page_name="Auth Service",
                        page_order=0,
                        description="...",
                        content_focus="...",
                        rationale="...",
                        target_symbols=["AuthService", "login"],
                        metadata={
                            "planner_mode": "cluster",
                            "section_id": 0,
                            "page_id": 1,
                            "cluster_node_ids": ["auth.AuthService", "auth.login"],
                        },
                    ),
                ],
            ),
            SectionSpec(
                section_name="Caching",
                section_order=1,
                description="Cache bits",
                rationale="Cache speeds things up",
                pages=[
                    PageSpec(
                        page_name="Cache Manager",
                        page_order=0,
                        description="...",
                        content_focus="...",
                        rationale="...",
                        target_symbols=["CacheManager"],
                        metadata={
                            "planner_mode": "cluster",
                            "section_id": 2,
                            "page_id": 3,
                            "cluster_node_ids": ["cache.CacheManager"],
                        },
                    ),
                ],
            ),
        ],
        total_pages=2,
    )


# ---------------------------------------------------------------------------
# _resolve_wiki_id
# ---------------------------------------------------------------------------


class TestResolveWikiId:
    def test_https_url(self, tmp_path) -> None:
        agent, _ = _make_agent(tmp_path)
        assert agent._resolve_wiki_id() == "example--widgets--main"

    def test_caches_result(self, tmp_path) -> None:
        agent, _ = _make_agent(tmp_path)
        first = agent._resolve_wiki_id()
        # Mutate repository_url; cached value should win.
        agent.repository_url = "https://github.com/other/repo.git"
        assert agent._resolve_wiki_id() == first

    def test_ssh_url(self, tmp_path) -> None:
        agent, _ = _make_agent(tmp_path)
        agent.repository_url = "git@github.com:owner/repo.git"
        agent._resolved_wiki_id = None
        assert agent._resolve_wiki_id() == "owner--repo--main"

    def test_unparseable_url_returns_none(self, tmp_path) -> None:
        agent, _ = _make_agent(tmp_path)
        agent.repository_url = "not-a-url"
        agent._resolved_wiki_id = None
        assert agent._resolve_wiki_id() is None


# ---------------------------------------------------------------------------
# dispatch_page_generation — stable IDs in Send payloads
# ---------------------------------------------------------------------------


class TestDispatchStableIds:
    def test_allocates_stable_ids_for_resolvable_wiki(self, tmp_path) -> None:
        agent, _ = _make_agent(tmp_path)
        # Stubs that dispatch_page_generation reads but doesn't otherwise use.
        agent.retriever_stack = SimpleNamespace(relationship_graph=None)
        agent._total_pages = 0
        agent._pages_generated = 0
        agent.progress_callback = None

        structure = _make_structure()
        state = WikiState(repository_context="x", wiki_structure_spec=structure)
        sends = agent.dispatch_page_generation(state, config={})

        assert len(sends) == 2
        # Each Send carries a stable page_id keyed off the planner metadata.
        page_ids = [s.arg["page_id"] for s in sends]
        wiki_id = agent._resolve_wiki_id()
        expected_auth = compute_page_id(wiki_id, 0, 1, "auth.AuthService", "Auth Service")
        expected_cache = compute_page_id(wiki_id, 2, 3, "cache.CacheManager", "Cache Manager")
        assert page_ids == [expected_auth, expected_cache]

        # Each Send carries persistence context for the persist hook.
        for s in sends:
            persist = s.arg["page_persist"]
            assert persist["wiki_id"] == wiki_id
            assert persist["id_scheme"] == "stable_v1"
            assert persist["anchor_slug"]  # populated

    def test_falls_back_to_legacy_when_wiki_id_unresolvable(self, tmp_path) -> None:
        agent, _ = _make_agent(tmp_path)
        agent.repository_url = "not-a-url"
        agent._resolved_wiki_id = None
        agent.retriever_stack = SimpleNamespace(relationship_graph=None)
        agent._total_pages = 0
        agent._pages_generated = 0
        agent.progress_callback = None

        structure = _make_structure()
        state = WikiState(repository_context="x", wiki_structure_spec=structure)
        sends = agent.dispatch_page_generation(state, config={})

        page_ids = [s.arg["page_id"] for s in sends]
        # Legacy {section_idx}#{page_idx} format.
        assert page_ids == ["0#0", "1#0"]
        for s in sends:
            assert s.arg["page_persist"]["id_scheme"] == "legacy"

    def test_slug_collision_resolved(self, tmp_path) -> None:
        # Two pages titled "Overview" must get different slugs within a wiki.
        agent, _ = _make_agent(tmp_path)
        agent.retriever_stack = SimpleNamespace(relationship_graph=None)
        agent._total_pages = 0
        agent._pages_generated = 0
        agent.progress_callback = None

        structure = WikiStructureSpec(
            wiki_title="W",
            overview="O",
            sections=[
                SectionSpec(
                    section_name="S",
                    section_order=0,
                    description="d",
                    rationale="r",
                    pages=[
                        PageSpec(
                            page_name="Overview",
                            page_order=0,
                            description="d",
                            content_focus="c",
                            rationale="r",
                            metadata={"section_id": 0, "page_id": 1, "cluster_node_ids": ["n1"]},
                        ),
                        PageSpec(
                            page_name="Overview",
                            page_order=1,
                            description="d",
                            content_focus="c",
                            rationale="r",
                            metadata={"section_id": 0, "page_id": 2, "cluster_node_ids": ["n2"]},
                        ),
                    ],
                ),
            ],
            total_pages=2,
        )
        state = WikiState(repository_context="x", wiki_structure_spec=structure)
        sends = agent.dispatch_page_generation(state, config={})

        slugs = [s.arg["page_persist"]["anchor_slug"] for s in sends]
        assert slugs == ["overview", "overview-2"]


# ---------------------------------------------------------------------------
# _persist_page_metadata — end-to-end DB write
# ---------------------------------------------------------------------------


class TestPersistPageMetadata:
    def test_writes_wiki_page_and_symbols(self, tmp_path) -> None:
        agent, db = _make_agent(tmp_path)
        page_spec = PageSpec(
            page_name="Auth Service",
            page_order=0,
            description="d",
            content_focus="c",
            rationale="r",
            target_symbols=["AuthService", "login"],
        )
        agent._persist_page_metadata(
            page_id="page-1",
            page_spec=page_spec,
            generated_content="# Auth Service\n\nDoc body.",
            persist={
                "wiki_id": "example--widgets--main",
                "id_scheme": "stable_v1",
                "anchor_slug": "auth-service",
                "macro_cluster": 0,
                "micro_cluster": 1,
                "primary_symbol_id": "auth.AuthService",
                "section_index": 0,
                "page_index": 0,
                "cluster_node_ids": ["auth.AuthService", "auth.login"],
            },
        )

        # wiki_pages row exists.
        row = db.get_wiki_page("page-1")
        assert row is not None
        assert row["title"] == "Auth Service"
        assert row["anchor_slug"] == "auth-service"
        assert row["wiki_id"] == "example--widgets--main"
        assert row["id_scheme"] == "stable_v1"
        assert row["macro_cluster"] == 0
        assert row["primary_symbol_id"] == "auth.AuthService"
        assert row["content_hash"]  # populated, sha256 hex

        # page_symbols rows exist — primary + related.
        rows = db.get_page_symbols("page-1")
        kinds = {(r["node_id"], r["citation_kind"]) for r in rows}
        assert ("auth.AuthService", "primary") in kinds
        assert ("auth.login", "related") in kinds

        # Reverse lookup works — the join table is the source→page index.
        assert db.get_pages_citing_node("auth.AuthService") == ["page-1"]

    def test_skips_persistence_when_wiki_id_missing(self, tmp_path) -> None:
        agent, db = _make_agent(tmp_path)
        agent._persist_page_metadata(
            page_id="page-x",
            page_spec=PageSpec(
                page_name="X",
                page_order=0,
                description="d",
                content_focus="c",
                rationale="r",
            ),
            generated_content="...",
            persist={},  # no wiki_id
        )
        # No row should have been written.
        assert db.get_wiki_page("page-x") is None

    def test_failure_does_not_raise(self, tmp_path) -> None:
        # The persist hook must be fail-soft — a DB hiccup must not break
        # page generation. We provoke a duplicate-slug IntegrityError on
        # the second call and assert no exception bubbles up.
        agent, _ = _make_agent(tmp_path)

        def call(page_id: str) -> None:
            agent._persist_page_metadata(
                page_id=page_id,
                page_spec=PageSpec(
                    page_name="Same Title",
                    page_order=0,
                    description="d",
                    content_focus="c",
                    rationale="r",
                ),
                generated_content="...",
                persist={
                    "wiki_id": "w",
                    "id_scheme": "stable_v1",
                    "anchor_slug": "same-slug",  # both pages reuse this
                    "cluster_node_ids": [],
                },
            )

        call("p1")
        call("p2")  # second insert with same wiki+slug — must not raise


# ---------------------------------------------------------------------------
# Cross-run stability — same plan → same IDs and rows
# ---------------------------------------------------------------------------


def test_page_ids_stable_across_runs(tmp_path) -> None:
    """The promise of stable IDs: regenerating an unchanged wiki yields the
    same page_id values, so wikilinks survive."""
    agent_a, db_a = _make_agent(tmp_path / "a")
    agent_b, db_b = _make_agent(tmp_path / "b")
    for agent in (agent_a, agent_b):
        agent.retriever_stack = SimpleNamespace(relationship_graph=None)
        agent._total_pages = 0
        agent._pages_generated = 0
        agent.progress_callback = None

    structure = _make_structure()
    state = WikiState(repository_context="x", wiki_structure_spec=structure)

    sends_a = agent_a.dispatch_page_generation(state, config={})
    sends_b = agent_b.dispatch_page_generation(state, config={})

    ids_a = [s.arg["page_id"] for s in sends_a]
    ids_b = [s.arg["page_id"] for s in sends_b]
    assert ids_a == ids_b
