"""Tests for toolkit bridge."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from app.config import Settings
from app.services.toolkit_bridge import (
    ComponentCache,
    EngineComponents,
    _load_cached_artifacts,
    build_engine_components,
)
from app.storage.local import LocalArtifactStorage
from pydantic import SecretStr


def _settings():
    return Settings(llm_api_key=SecretStr("test-key"))


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
        """_load_cached_artifacts sets components.repo_analysis when analysis is stored."""
        from app.core.repository_analysis_store import RepositoryAnalysisStore

        repo_identifier = "owner/repo:main"
        store = RepositoryAnalysisStore(cache_dir=str(tmp_path))
        store.save_analysis(
            repo_identifier=repo_identifier,
            analysis="This repo does important things.",
            commit_hash=None,
        )

        components = EngineComponents()
        _load_cached_artifacts(components, str(tmp_path), "wiki-123", repo_identifier)

        assert components.repo_analysis is not None
        assert "summary" in components.repo_analysis
        assert components.repo_analysis["summary"] == "This repo does important things."

    def test_no_analysis_file_leaves_repo_analysis_none(self, tmp_path):
        """_load_cached_artifacts does not raise and leaves repo_analysis None when no file exists."""
        components = EngineComponents()
        _load_cached_artifacts(components, str(tmp_path), "wiki-456", "owner/missing:main")

        assert components.repo_analysis is None

    def test_store_exception_does_not_propagate(self, tmp_path):
        """If RepositoryAnalysisStore.get_analysis_for_prompt raises, _load_cached_artifacts swallows it."""
        components = EngineComponents()

        with patch(
            "app.core.repository_analysis_store.RepositoryAnalysisStore.get_analysis_for_prompt",
            side_effect=RuntimeError("disk failure"),
        ):
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
