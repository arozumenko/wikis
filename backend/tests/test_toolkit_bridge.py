"""Tests for toolkit bridge."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from app.config import Settings
from app.services.toolkit_bridge import (
    ComponentCache,
    EngineComponents,
    _load_cached_artifacts,
    _load_legacy_artifacts,
    build_engine_components,
)
from app.storage.local import LocalArtifactStorage
from pydantic import SecretStr


def _settings():
    return Settings(llm_api_key=SecretStr("test-key"))


def _make_cached_unified_db(tmp_path, db_name: str, repository_analysis: str | None = None):
    from app.core.unified_db import UnifiedWikiDB

    db_path = tmp_path / f"{db_name}.wiki.db"
    db = UnifiedWikiDB(str(db_path), embedding_dim=4)
    try:
        if repository_analysis is not None:
            db.set_meta("repository_analysis", repository_analysis)
    finally:
        db.close()
    return db_path


class TestBuildEngineComponents:
    @pytest.mark.asyncio
    async def test_wiki_not_found_raises(self, tmp_path):
        storage = LocalArtifactStorage(str(tmp_path))
        with pytest.raises(FileNotFoundError, match="Wiki not found"):
            await build_engine_components("nope", storage, _settings())

    @pytest.mark.asyncio
    async def test_returns_components_with_graph(self, tmp_path):
        """build_engine_components loads DB/storage + artifacts and returns EngineComponents."""
        storage = LocalArtifactStorage(str(tmp_path))
        # Write a dummy artifact so the wiki is found via storage fallback
        # (DB is not initialised in tests, so the code falls back to storage.list_artifacts)
        await storage.upload("wiki_artifacts", "w1/dummy.txt", b"placeholder")

        with (
            patch("app.services.llm_factory.create_llm", return_value=MagicMock()),
            patch("app.services.toolkit_bridge._load_cached_artifacts") as mock_load,
        ):
            components = await build_engine_components("w1", storage, _settings())

        assert isinstance(components, EngineComponents)
        assert components.code_graph is not None  # empty DiGraph fallback
        assert components.llm is not None
        mock_load.assert_called_once()

    @pytest.mark.asyncio
    async def test_components_cached(self, tmp_path):
        """AskService should cache components per wiki_id."""
        from app.services.ask_service import AskService

        storage = LocalArtifactStorage(str(tmp_path))
        service = AskService(_settings(), storage)

        mock_components = EngineComponents(llm=MagicMock())
        with patch("app.services.ask_service.build_engine_components", return_value=mock_components) as mock_build:
            c1 = await service._get_components("w1")
            c2 = await service._get_components("w1")
            assert c1 is c2
            mock_build.assert_called_once()


class TestComponentCache:
    def test_get_miss(self):
        cache = ComponentCache(max_size=3)
        assert cache.get("missing") is None

    def test_put_and_get(self):
        cache = ComponentCache(max_size=3)
        c = EngineComponents(llm=MagicMock())
        cache.put("w1", c)
        assert cache.get("w1") is c

    def test_lru_eviction(self):
        cache = ComponentCache(max_size=2)
        c1 = EngineComponents(llm=MagicMock())
        c2 = EngineComponents(llm=MagicMock())
        c3 = EngineComponents(llm=MagicMock())
        cache.put("w1", c1)
        cache.put("w2", c2)
        # Access w1 so w2 becomes LRU
        cache.get("w1")
        cache.put("w3", c3)
        # w2 should be evicted (LRU), w1 and w3 remain
        assert cache.get("w2") is None
        assert cache.get("w1") is c1
        assert cache.get("w3") is c3

    def test_ttl_expiration(self):
        import time

        cache = ComponentCache(max_size=5, ttl_seconds=0)
        c = EngineComponents(llm=MagicMock())
        cache.put("w1", c)
        time.sleep(0.01)
        assert cache.get("w1") is None

    def test_len(self):
        cache = ComponentCache(max_size=5)
        assert len(cache) == 0
        cache.put("w1", EngineComponents())
        cache.put("w2", EngineComponents())
        assert len(cache) == 2

    def test_evict(self):
        cache = ComponentCache(max_size=5)
        cache.put("w1", EngineComponents())
        assert cache.evict("w1") is True
        assert cache.get("w1") is None
        assert cache.evict("w1") is False

    @pytest.mark.asyncio
    async def test_get_or_build_caches(self):
        cache = ComponentCache(max_size=5)
        build_count = 0

        async def factory():
            nonlocal build_count
            build_count += 1
            return EngineComponents(llm=MagicMock())

        c1 = await cache.get_or_build("w1", factory)
        c2 = await cache.get_or_build("w1", factory)
        assert c1 is c2
        assert build_count == 1

    @pytest.mark.asyncio
    async def test_get_or_build_concurrent(self):
        """Two concurrent get_or_build calls should only build once."""
        import asyncio

        cache = ComponentCache(max_size=5)
        build_count = 0

        async def slow_factory():
            nonlocal build_count
            build_count += 1
            await asyncio.sleep(0.05)
            return EngineComponents(llm=MagicMock())

        c1, c2 = await asyncio.gather(
            cache.get_or_build("w1", slow_factory),
            cache.get_or_build("w1", slow_factory),
        )
        assert c1 is c2
        assert build_count == 1


class TestLoadCachedArtifactsRepoAnalysis:
    def test_populates_repo_analysis_when_analysis_exists(self, tmp_path):
        """_load_cached_artifacts sets components.repo_analysis from unified DB metadata."""
        _make_cached_unified_db(tmp_path, "repo-analysis", "This repo does important things.")

        components = EngineComponents()
        _load_cached_artifacts(components, str(tmp_path), "wiki-123", "owner/repo:main")

        assert components.repo_analysis is not None
        assert "summary" in components.repo_analysis
        assert components.repo_analysis["summary"] == "This repo does important things."

    def test_no_analysis_file_leaves_repo_analysis_none(self, tmp_path):
        """_load_cached_artifacts does not raise and leaves repo_analysis None when no file exists."""
        components = EngineComponents()
        _load_cached_artifacts(components, str(tmp_path), "wiki-456", "owner/missing:main")

        assert components.repo_analysis is None

    def test_store_exception_does_not_propagate(self, tmp_path):
        """If unified DB metadata loading raises, _load_cached_artifacts swallows it."""
        _make_cached_unified_db(tmp_path, "broken-db")
        components = EngineComponents()
        mock_db = MagicMock()
        mock_db.get_meta.side_effect = RuntimeError("disk failure")

        with patch("app.core.storage.open_storage", return_value=mock_db):
            # Must not raise
            _load_cached_artifacts(components, str(tmp_path), "wiki-789", "owner/repo:main")

        assert components.repo_analysis is None


def _make_db_session_mock(fake_record):
    """Build a mock async session factory that returns fake_record on query.

    The code under test does:
        async with session_factory() as session:
            result = await session.execute(...)
            record = result.scalar_one_or_none()

    ``session_factory()`` must return an async context manager whose ``__aenter__``
    yields an object with an async ``execute`` method.
    """
    mock_result = MagicMock()
    mock_result.scalar_one_or_none.return_value = fake_record

    # The object that will be bound to `session` inside `async with ... as session`
    inner_session = AsyncMock()
    inner_session.execute.return_value = mock_result

    # The context manager returned by session_factory()
    ctx_manager = AsyncMock()
    ctx_manager.__aenter__.return_value = inner_session
    ctx_manager.__aexit__.return_value = False

    mock_session_factory = MagicMock()
    mock_session_factory.return_value = ctx_manager
    return mock_session_factory


def _write_legacy_cache(
    tmp_path,
    cache_key: str,
    repo_identifier: str,
    *,
    register_index: bool = True,
    nodes: list[tuple[str, dict]] | None = None,
):
    """Create a pre-UnifiedDB layout (.code_graph.gz + .fts5.db) on disk.

    Used to verify that ``_load_legacy_artifacts`` rehydrates ask/research
    engine state when no ``.wiki.db`` exists for a wiki — the failure mode
    that motivated #237.
    """
    import gzip
    import json
    import pickle
    from pathlib import Path

    import networkx as nx

    cache_dir = Path(tmp_path)
    graph = nx.DiGraph()
    if nodes is None:
        nodes = [("pkg.module.func", {"name": "func", "type": "function"})]
    for node_id, attrs in nodes:
        graph.add_node(node_id, **attrs)

    graph_file = cache_dir / f"{cache_key}.code_graph.gz"
    with gzip.open(graph_file, "wb") as fh:
        pickle.dump(graph, fh, protocol=pickle.HIGHEST_PROTOCOL)

    if register_index:
        index_path = cache_dir / "cache_index.json"
        index = {
            "graphs": {f"{repo_identifier}:combined": cache_key},
            "refs": {},
        }
        # Bookkeep ``refs`` for the bare ``repo:branch`` pointer when the
        # identifier is commit-scoped, mirroring real cache_index.json.
        parts = repo_identifier.split(":")
        if len(parts) == 3:
            index["refs"][f"{parts[0]}:{parts[1]}"] = repo_identifier
        index_path.write_text(json.dumps(index))

    return graph_file


class TestLoadLegacyArtifacts:
    """#237 — legacy ``.code_graph.gz`` fallback when no ``.wiki.db`` exists."""

    def test_loads_graph_and_query_service_from_legacy_artifacts(self, tmp_path):
        _write_legacy_cache(tmp_path, "abc123", "onetest-ai/core:main:ca9addbd")

        components = EngineComponents()
        ok = _load_legacy_artifacts(
            components, tmp_path, "onetest-ai/core:main:ca9addbd"
        )

        assert ok is True
        assert components.code_graph is not None
        assert components.code_graph.number_of_nodes() == 1
        assert components.query_service is not None
        assert components.graph_manager is not None

    def test_returns_false_when_no_legacy_artifacts_exist(self, tmp_path):
        components = EngineComponents()
        ok = _load_legacy_artifacts(components, tmp_path, "owner/repo:main")

        assert ok is False
        assert components.code_graph is None
        assert components.query_service is None

    def test_returns_false_for_empty_graph(self, tmp_path):
        """Empty pickled graph should not satisfy the fallback."""
        _write_legacy_cache(
            tmp_path,
            "empty999",
            "owner/empty:main:deadbeef",
            nodes=[],  # no nodes
        )

        components = EngineComponents()
        ok = _load_legacy_artifacts(
            components, tmp_path, "owner/empty:main:deadbeef"
        )

        assert ok is False
        assert components.code_graph is None

    def test_load_cached_artifacts_falls_back_when_no_wiki_db(self, tmp_path):
        """The public entry point ``_load_cached_artifacts`` should invoke
        the legacy fallback when cache_index has no ``unified_db`` entry.

        This is the exact scenario reported for onetest-ai/core: index lists
        only ``graphs`` + ``refs`` + ``fts5``, no ``unified_db`` key.
        """
        _write_legacy_cache(tmp_path, "abc123", "onetest-ai/core:main:ca9addbd")

        components = EngineComponents()
        _load_cached_artifacts(
            components,
            str(tmp_path),
            "wiki-onetest",
            "onetest-ai/core:main:ca9addbd",
        )

        # Legacy path wired up — ask/research tools now have a working
        # query_service even though no unified DB ever existed for this
        # wiki.
        assert components.code_graph is not None
        assert components.code_graph.number_of_nodes() == 1
        assert components.query_service is not None
        # No retriever_stack (UnifiedRetriever requires .wiki.db).  This
        # asymmetry is intentional and documented in the helper.
        assert components.retriever_stack is None
        assert components.storage is None

    def test_falls_back_when_refs_point_at_different_commit_than_on_disk(
        self, tmp_path
    ):
        """Rio's finding #2: real onetest-ai/core scenario — cache_index
        ``refs`` says ``8c210362`` but ``graphs`` only registers
        ``ca9addbd``.  ``load_graph_by_repo_name`` returns None because the
        canonicalized key doesn't match any ``graphs`` entry; the cache-
        index scan must rescue us.
        """
        import json

        # Write artifacts under the OLD commit hash
        _write_legacy_cache(
            tmp_path,
            "old-commit-key",
            "owner/repo:main:ca9addbd",
            register_index=False,
        )
        # Cache index points the bare ref at a DIFFERENT commit
        (tmp_path / "cache_index.json").write_text(
            json.dumps(
                {
                    "graphs": {"owner/repo:main:ca9addbd:combined": "old-commit-key"},
                    "refs": {"owner/repo:main": "owner/repo:main:8c210362"},
                }
            )
        )

        components = EngineComponents()
        # _load_cached_artifacts is called with the bare ref (matches
        # production: ``_extract_repo_identifier`` returns
        # ``owner/repo:branch``).
        _load_cached_artifacts(
            components, str(tmp_path), "wiki-x", "owner/repo:main"
        )

        assert components.code_graph is not None
        assert components.code_graph.number_of_nodes() == 1
        assert components.query_service is not None

    def test_fts_index_passed_to_query_service_only_when_open(self, tmp_path):
        """Rio's finding #1: if FTS load fails inside
        ``load_graph_by_repo_name`` the index instance exists but
        ``is_open=False``.  We must not hand that to GraphQueryService."""
        _write_legacy_cache(tmp_path, "no-fts-key", "owner/repo:main:abcd1234")
        # NB: ``_write_legacy_cache`` writes only ``.code_graph.gz`` — no
        # ``.fts5.db`` companion — so the FTS load inside GraphManager
        # silently fails and leaves ``is_open=False``.

        components = EngineComponents()
        ok = _load_legacy_artifacts(
            components, tmp_path, "owner/repo:main:abcd1234"
        )
        assert ok is True
        # The query_service was created — confirm its fts attribute is
        # None rather than a closed GraphTextIndex.
        assert components.query_service is not None
        assert components.query_service.fts is None

    def test_load_cached_artifacts_falls_back_when_wiki_db_missing_on_disk(
        self, tmp_path
    ):
        """If cache_index.unified_db points at a wiki.db that's been
        cleaned off disk, the loader should still fall back to legacy
        artifacts rather than leaving the wiki broken."""
        import json

        # Write legacy graph artifacts
        _write_legacy_cache(
            tmp_path,
            "legacy-key",
            "owner/repo:main:abcd1234",
            register_index=False,  # we'll write a custom index below
        )

        # Custom index: unified_db points to a key whose .wiki.db file does
        # NOT exist on disk (cache eviction scenario).
        index_path = tmp_path / "cache_index.json"
        index_path.write_text(
            json.dumps(
                {
                    "graphs": {"owner/repo:main:abcd1234:combined": "legacy-key"},
                    "refs": {"owner/repo:main": "owner/repo:main:abcd1234"},
                    "unified_db": {"owner/repo:main:abcd1234": "nonexistent-key"},
                }
            )
        )

        components = EngineComponents()
        _load_cached_artifacts(
            components, str(tmp_path), "wiki-x", "owner/repo:main:abcd1234"
        )

        # Fell through to legacy fallback
        assert components.code_graph is not None
        assert components.code_graph.number_of_nodes() == 1
        assert components.query_service is not None


class TestBuildEngineComponentsDescription:
    """Tests that wiki description is injected into repo_analysis by build_engine_components."""

    @pytest.mark.asyncio
    async def test_description_injected_when_set(self, tmp_path):
        """When WikiRecord.description is set it appears in components.repo_analysis['description']."""
        storage = LocalArtifactStorage(str(tmp_path))
        await storage.upload("wiki_artifacts", "w-desc/dummy.txt", b"placeholder")

        fake_record = MagicMock()
        fake_record.repo_url = "https://github.com/owner/repo"
        fake_record.branch = "main"
        fake_record.commit_hash = None
        fake_record.description = "A project that does something useful."

        with (
            patch("app.services.llm_factory.create_llm", return_value=MagicMock()),
            patch("app.services.toolkit_bridge._load_cached_artifacts"),
            patch("app.db.get_session_factory", return_value=_make_db_session_mock(fake_record)),
        ):
            components = await build_engine_components("w-desc", storage, _settings())

        assert components.repo_analysis is not None
        assert components.repo_analysis["description"] == "A project that does something useful."

    @pytest.mark.asyncio
    async def test_description_not_injected_when_none(self, tmp_path):
        """When WikiRecord.description is None no 'description' key is added to repo_analysis."""
        storage = LocalArtifactStorage(str(tmp_path))
        await storage.upload("wiki_artifacts", "w-nodesc/dummy.txt", b"placeholder")

        fake_record = MagicMock()
        fake_record.repo_url = "https://github.com/owner/repo"
        fake_record.branch = "main"
        fake_record.commit_hash = None
        fake_record.description = None

        with (
            patch("app.services.llm_factory.create_llm", return_value=MagicMock()),
            patch("app.services.toolkit_bridge._load_cached_artifacts"),
            patch("app.db.get_session_factory", return_value=_make_db_session_mock(fake_record)),
        ):
            components = await build_engine_components("w-nodesc", storage, _settings())

        # No description key — repo_analysis may be None or a dict without 'description'
        if components.repo_analysis is not None:
            assert "description" not in components.repo_analysis

    @pytest.mark.asyncio
    async def test_description_does_not_overwrite_existing_keys(self, tmp_path):
        """Injecting description must not remove pre-existing repo_analysis keys like 'summary'."""
        storage = LocalArtifactStorage(str(tmp_path))
        await storage.upload("wiki_artifacts", "w-both/dummy.txt", b"placeholder")

        fake_record = MagicMock()
        fake_record.repo_url = "https://github.com/owner/repo"
        fake_record.branch = "main"
        fake_record.commit_hash = None
        fake_record.description = "User description."

        def _load_with_summary(components, cache_dir, wiki_id, repo_identifier, settings=None):
            components.repo_analysis = {"summary": "Auto-generated summary."}

        with (
            patch("app.services.llm_factory.create_llm", return_value=MagicMock()),
            patch("app.services.toolkit_bridge._load_cached_artifacts", side_effect=_load_with_summary),
            patch("app.db.get_session_factory", return_value=_make_db_session_mock(fake_record)),
        ):
            components = await build_engine_components("w-both", storage, _settings())

        assert components.repo_analysis is not None
        assert components.repo_analysis["summary"] == "Auto-generated summary."
        assert components.repo_analysis["description"] == "User description."
