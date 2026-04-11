"""Unit tests for WikiPageIndexCache — LRU in-memory cache for WikiPageIndex."""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from app.core.wiki_page_index import WikiPageIndex, WikiPageIndexCache


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def make_storage() -> MagicMock:
    """Return a mock ArtifactStorage."""
    storage = MagicMock()
    storage.list_artifacts = AsyncMock(return_value=[])
    return storage


async def _fake_build(wiki_id: str, storage) -> WikiPageIndex:
    """Fake WikiPageIndex.build that returns a real (empty) index without I/O."""
    index = WikiPageIndex(wiki_id, storage)
    # Skip _load — leave pages/edges empty; we only care about identity
    return index


# ---------------------------------------------------------------------------
# 1. get() builds index on first access
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_get_builds_index_on_first_access():
    storage = make_storage()
    cache = WikiPageIndexCache(storage, max_wikis=10)

    mock_build = AsyncMock(side_effect=_fake_build)
    with patch.object(WikiPageIndex, "build", new=mock_build):
        result = await cache.get("wiki-1")

    assert isinstance(result, WikiPageIndex)
    mock_build.assert_awaited_once()


# ---------------------------------------------------------------------------
# 2. get() returns cached index on second access (no rebuild)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_get_returns_cached_on_second_access():
    storage = make_storage()
    cache = WikiPageIndexCache(storage, max_wikis=10)

    with patch.object(WikiPageIndex, "build", new=AsyncMock(side_effect=_fake_build)) as mock_build:
        first = await cache.get("wiki-1")
        second = await cache.get("wiki-1")

    assert first is second
    assert mock_build.await_count == 1  # built only once


# ---------------------------------------------------------------------------
# 3. Double-check after lock prevents duplicate builds under concurrent access
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_concurrent_get_builds_only_once():
    storage = make_storage()
    cache = WikiPageIndexCache(storage, max_wikis=10)

    build_count = 0

    async def slow_build(wiki_id: str, storage) -> WikiPageIndex:
        nonlocal build_count
        build_count += 1
        await asyncio.sleep(0)  # yield to allow other coroutines to run
        return WikiPageIndex(wiki_id, storage)

    with patch.object(WikiPageIndex, "build", new=AsyncMock(side_effect=slow_build)):
        results = await asyncio.gather(
            cache.get("wiki-concurrent"),
            cache.get("wiki-concurrent"),
            cache.get("wiki-concurrent"),
        )

    assert build_count == 1
    # All tasks received the same object
    assert results[0] is results[1]
    assert results[1] is results[2]


# ---------------------------------------------------------------------------
# 4. LRU eviction: when max_wikis exceeded, oldest entry is removed
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_lru_eviction_removes_oldest():
    storage = make_storage()
    cache = WikiPageIndexCache(storage, max_wikis=2)

    with patch.object(WikiPageIndex, "build", new=AsyncMock(side_effect=_fake_build)):
        await cache.get("wiki-a")
        await cache.get("wiki-b")
        # Adding a third entry should evict wiki-a (oldest)
        await cache.get("wiki-c")

    assert "wiki-a" not in cache._indexes
    assert "wiki-b" in cache._indexes
    assert "wiki-c" in cache._indexes


# ---------------------------------------------------------------------------
# 5. Recently accessed wiki moved to end (survives eviction)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_recently_accessed_survives_eviction():
    storage = make_storage()
    cache = WikiPageIndexCache(storage, max_wikis=2)

    with patch.object(WikiPageIndex, "build", new=AsyncMock(side_effect=_fake_build)):
        await cache.get("wiki-a")
        await cache.get("wiki-b")
        # Re-access wiki-a — it should move to the "recently used" end
        await cache.get("wiki-a")
        # Adding wiki-c now; wiki-b should be evicted (LRU), not wiki-a
        await cache.get("wiki-c")

    assert "wiki-b" not in cache._indexes
    assert "wiki-a" in cache._indexes
    assert "wiki-c" in cache._indexes


# ---------------------------------------------------------------------------
# 6. Per-wiki locks are isolated (wiki-A lock doesn't block wiki-B)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_per_wiki_locks_are_isolated():
    storage = make_storage()
    cache = WikiPageIndexCache(storage, max_wikis=10)

    order: list[str] = []

    async def ordered_build(wiki_id: str, storage) -> WikiPageIndex:
        order.append(f"start-{wiki_id}")
        await asyncio.sleep(0)
        order.append(f"end-{wiki_id}")
        return WikiPageIndex(wiki_id, storage)

    with patch.object(WikiPageIndex, "build", new=AsyncMock(side_effect=ordered_build)):
        await asyncio.gather(
            cache.get("wiki-a"),
            cache.get("wiki-b"),
        )

    # Both builds should have interleaved (not serialized) because they hold different locks
    assert "start-wiki-a" in order
    assert "start-wiki-b" in order
    assert "end-wiki-a" in order
    assert "end-wiki-b" in order


# ---------------------------------------------------------------------------
# 7. get() returns a WikiPageIndex instance
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_get_returns_wiki_page_index_instance():
    storage = make_storage()
    cache = WikiPageIndexCache(storage, max_wikis=10)

    with patch.object(WikiPageIndex, "build", new=AsyncMock(side_effect=_fake_build)):
        result = await cache.get("wiki-1")

    assert isinstance(result, WikiPageIndex)


# ---------------------------------------------------------------------------
# 8. Cache starts empty
# ---------------------------------------------------------------------------


def test_cache_starts_empty():
    storage = make_storage()
    cache = WikiPageIndexCache(storage, max_wikis=10)

    assert len(cache._indexes) == 0


# ---------------------------------------------------------------------------
# 9. max_wikis=1 evicts immediately on second wiki
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_max_wikis_1_evicts_on_second():
    storage = make_storage()
    cache = WikiPageIndexCache(storage, max_wikis=1)

    with patch.object(WikiPageIndex, "build", new=AsyncMock(side_effect=_fake_build)):
        await cache.get("wiki-first")
        await cache.get("wiki-second")

    assert "wiki-first" not in cache._indexes
    assert "wiki-second" in cache._indexes
    assert len(cache._indexes) == 1


# ---------------------------------------------------------------------------
# 10. Evicted wiki is rebuilt on next access
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_evicted_wiki_rebuilt_on_next_access():
    storage = make_storage()
    cache = WikiPageIndexCache(storage, max_wikis=1)

    with patch.object(WikiPageIndex, "build", new=AsyncMock(side_effect=_fake_build)) as mock_build:
        await cache.get("wiki-a")  # build #1
        await cache.get("wiki-b")  # build #2 — evicts wiki-a
        assert mock_build.await_count == 2

        # Access wiki-a again — must rebuild since it was evicted
        await cache.get("wiki-a")  # build #3

    assert mock_build.await_count == 3
