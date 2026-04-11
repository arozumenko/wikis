"""Tests for toolkit bridge."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
from pydantic import SecretStr

from app.config import Settings
from app.services.toolkit_bridge import ComponentCache, EngineComponents, _load_cached_artifacts, build_engine_components
from app.storage.local import LocalArtifactStorage


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
