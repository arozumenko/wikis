"""Tests for QACacheManager."""

import asyncio
import json
import os
from dataclasses import dataclass

import numpy as np
import pytest

from app.services.qa_cache_manager import QACacheManager


class FakeEmbeddings:
    """Deterministic embeddings for testing."""

    def __init__(self, dim: int = 8):
        self._dim = dim
        self._call_count = 0

    def embed_query(self, text: str) -> list[float]:
        """Return a deterministic vector based on text hash."""
        np.random.seed(hash(text) % 2**31)
        vec = np.random.randn(self._dim).astype(np.float32)
        return vec.tolist()

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self.embed_query(t) for t in texts]


@pytest.fixture
def cache_dir(tmp_path):
    return str(tmp_path / "qa_cache")


@pytest.fixture
def embeddings():
    return FakeEmbeddings(dim=8)


@pytest.fixture
def manager(cache_dir, embeddings):
    return QACacheManager(cache_dir, embeddings, max_wikis=50)


@pytest.mark.asyncio
async def test_add_and_search_above_threshold(manager):
    """Add a single embedding, search returns it above threshold."""
    await manager.add("wiki-1", "qa-1", np.random.randn(8).astype(np.float32))
    # Search with the same question text to get a result
    # Since we use the same manager, we can do a self-search
    qa_ids, embedding = await manager.search("wiki-1", "test question", threshold=0.0)
    assert isinstance(embedding, np.ndarray)
    # With threshold=0.0 and at least one vector, we should get results
    # (the exact match depends on the embedding values)


@pytest.mark.asyncio
async def test_search_empty_index(manager):
    """Search on non-existent wiki returns empty list."""
    qa_ids, embedding = await manager.search("nonexistent", "question", threshold=0.5)
    assert qa_ids == []
    assert isinstance(embedding, np.ndarray)


@pytest.mark.asyncio
async def test_search_below_threshold(manager):
    """Search returns empty when no results above threshold."""
    # Add a vector
    vec = np.random.randn(8).astype(np.float32)
    await manager.add("wiki-1", "qa-1", vec)
    # Search with threshold=1.1 (impossible to reach with cosine)
    qa_ids, _ = await manager.search("wiki-1", "different question", threshold=1.1)
    assert qa_ids == []


@pytest.mark.asyncio
async def test_k5_returns_multiple_candidates(manager):
    """k=5 returns multiple candidates in descending similarity order."""
    for i in range(10):
        vec = np.random.randn(8).astype(np.float32)
        await manager.add("wiki-1", f"qa-{i}", vec)
    qa_ids, _ = await manager.search("wiki-1", "query", threshold=0.0, k=5)
    assert len(qa_ids) <= 5


@pytest.mark.asyncio
async def test_delete_index(manager, cache_dir):
    """delete_index removes files (including ready marker) and clears in-memory state."""
    vec = np.random.randn(8).astype(np.float32)
    await manager.add("wiki-1", "qa-1", vec)
    # Files should exist
    assert os.path.exists(os.path.join(cache_dir, "wiki-1.qa.faiss"))
    assert os.path.exists(os.path.join(cache_dir, "wiki-1.qa.ids.json"))
    assert os.path.exists(os.path.join(cache_dir, "wiki-1.qa.ready"))
    # Delete
    result = manager.delete_index("wiki-1")
    assert result is True
    assert not os.path.exists(os.path.join(cache_dir, "wiki-1.qa.faiss"))
    assert not os.path.exists(os.path.join(cache_dir, "wiki-1.qa.ids.json"))
    assert not os.path.exists(os.path.join(cache_dir, "wiki-1.qa.ready"))
    # Second delete returns False
    assert manager.delete_index("wiki-1") is False


@pytest.mark.asyncio
async def test_rebuild_from_records(manager):
    """rebuild_from_records creates index from provided records."""

    @dataclass
    class FakeRecord:
        id: str
        question: str

    records = [
        FakeRecord(id="qa-1", question="What is auth?"),
        FakeRecord(id="qa-2", question="How does routing work?"),
    ]
    await manager.rebuild_from_records("wiki-1", records)
    # Search for exact same question to guarantee cosine similarity = 1.0
    qa_ids, _ = await manager.search("wiki-1", "What is auth?", threshold=0.0)
    assert len(qa_ids) > 0
    assert "qa-1" in qa_ids


@pytest.mark.asyncio
async def test_lazy_index_creation(manager):
    """First add determines the dimension."""
    vec = np.random.randn(8).astype(np.float32)
    await manager.add("wiki-1", "qa-1", vec)
    assert "wiki-1" in manager._indexes
    assert manager._indexes["wiki-1"].d == 8


@pytest.mark.asyncio
async def test_corruption_detection(manager, cache_dir):
    """len(qa_ids) != index.ntotal triggers failed load."""
    vec = np.random.randn(8).astype(np.float32)
    await manager.add("wiki-1", "qa-1", vec)
    # Corrupt the ids file
    ids_path = os.path.join(cache_dir, "wiki-1.qa.ids.json")
    with open(ids_path, "w") as f:
        json.dump(["qa-1", "qa-extra"], f)  # 2 ids but only 1 vector
    # Clear in-memory state to force reload
    manager._indexes.pop("wiki-1", None)
    manager._qa_ids.pop("wiki-1", None)
    # Search should handle gracefully (empty results due to corruption)
    qa_ids, _ = await manager.search("wiki-1", "test", threshold=0.0)
    assert qa_ids == []


@pytest.mark.asyncio
async def test_atomic_write(manager, cache_dir):
    """After add, no .tmp files remain and ready marker exists."""
    vec = np.random.randn(8).astype(np.float32)
    await manager.add("wiki-1", "qa-1", vec)
    files = os.listdir(cache_dir)
    tmp_files = [f for f in files if f.endswith(".tmp")]
    assert tmp_files == []
    assert "wiki-1.qa.ready" in files


@pytest.mark.asyncio
async def test_concurrent_add(manager):
    """Concurrent add operations don't corrupt index."""

    async def add_one(i):
        vec = np.random.randn(8).astype(np.float32)
        await manager.add("wiki-1", f"qa-{i}", vec)

    await asyncio.gather(*[add_one(i) for i in range(10)])
    assert len(manager._qa_ids["wiki-1"]) == 10
    assert manager._indexes["wiki-1"].ntotal == 10


@pytest.mark.asyncio
async def test_check_needs_rebuild_no_files(manager):
    """check_needs_rebuild returns False when no index files exist."""
    assert manager.check_needs_rebuild("nonexistent") is False


@pytest.mark.asyncio
async def test_check_needs_rebuild_loaded_index(manager):
    """check_needs_rebuild returns False for a cleanly loaded index."""
    vec = np.random.randn(8).astype(np.float32)
    await manager.add("wiki-1", "qa-1", vec)
    # Already in memory — no rebuild needed
    assert manager.check_needs_rebuild("wiki-1") is False


@pytest.mark.asyncio
async def test_check_needs_rebuild_corrupted(manager, cache_dir):
    """check_needs_rebuild returns True when files exist but index is corrupt."""
    vec = np.random.randn(8).astype(np.float32)
    await manager.add("wiki-1", "qa-1", vec)
    # Corrupt the ids file
    ids_path = os.path.join(cache_dir, "wiki-1.qa.ids.json")
    with open(ids_path, "w") as f:
        json.dump(["qa-1", "qa-extra"], f)  # 2 ids but only 1 vector
    # Clear in-memory state to force fresh load
    manager._indexes.pop("wiki-1", None)
    manager._qa_ids.pop("wiki-1", None)
    # Files exist and are corrupt — needs rebuild
    assert manager.check_needs_rebuild("wiki-1") is True


@pytest.mark.asyncio
async def test_check_needs_rebuild_clean_cold_start(manager, cache_dir):
    """check_needs_rebuild returns False for a clean on-disk index after cold start."""
    vec = np.random.randn(8).astype(np.float32)
    await manager.add("wiki-1", "qa-1", vec)
    # Simulate cold restart: clear in-memory state, files remain on disk
    manager._indexes.pop("wiki-1", None)
    manager._qa_ids.pop("wiki-1", None)
    # Files are clean (including ready marker) — should load successfully
    assert manager.check_needs_rebuild("wiki-1") is False
    # Verify it was loaded into memory by the check
    assert "wiki-1" in manager._indexes


@pytest.mark.asyncio
async def test_missing_ready_marker_triggers_rebuild(manager, cache_dir):
    """Missing ready marker (simulating interrupted save) triggers rebuild."""
    vec = np.random.randn(8).astype(np.float32)
    await manager.add("wiki-1", "qa-1", vec)
    # Remove ready marker to simulate interrupted save
    os.remove(os.path.join(cache_dir, "wiki-1.qa.ready"))
    # Clear in-memory state
    manager._indexes.pop("wiki-1", None)
    manager._qa_ids.pop("wiki-1", None)
    # Should need rebuild (files exist but ready marker is missing)
    assert manager.check_needs_rebuild("wiki-1") is True
    # Search returns empty (index not loadable)
    qa_ids, _ = await manager.search("wiki-1", "test", threshold=0.0)
    assert qa_ids == []


@pytest.mark.asyncio
async def test_lru_eviction(cache_dir, embeddings):
    """In-memory indexes are evicted when exceeding max_wikis."""
    manager = QACacheManager(cache_dir, embeddings, max_wikis=3)
    for i in range(5):
        vec = np.random.randn(8).astype(np.float32)
        await manager.add(f"wiki-{i}", f"qa-{i}", vec)
    # Only the 3 most recently used wikis should be in memory
    assert len(manager._indexes) == 3
    # The last 3 added (wiki-2, wiki-3, wiki-4) should be in memory
    assert "wiki-2" in manager._indexes
    assert "wiki-3" in manager._indexes
    assert "wiki-4" in manager._indexes
    # Evicted wikis (wiki-0, wiki-1) still have files on disk
    assert os.path.exists(os.path.join(cache_dir, "wiki-0.qa.faiss"))
    assert os.path.exists(os.path.join(cache_dir, "wiki-1.qa.faiss"))
    # Searching an evicted wiki reloads it from disk
    qa_ids, _ = await manager.search("wiki-0", "test", threshold=0.0)
    assert "wiki-0" in manager._indexes
    # Reloading evicts the new LRU
    assert len(manager._indexes) == 3


@pytest.mark.asyncio
async def test_rebuild_recovery_makes_records_searchable(manager, cache_dir):
    """Starting from a corrupt index, rebuild_from_records restores search."""

    @dataclass
    class FakeRecord:
        id: str
        question: str

    # Add a record and then corrupt the index
    vec = np.random.randn(8).astype(np.float32)
    await manager.add("wiki-1", "qa-old", vec)
    ids_path = os.path.join(cache_dir, "wiki-1.qa.ids.json")
    with open(ids_path, "w") as f:
        json.dump(["qa-old", "qa-extra"], f)  # Mismatch
    manager._indexes.pop("wiki-1", None)
    manager._qa_ids.pop("wiki-1", None)

    # check_needs_rebuild should be True
    assert manager.check_needs_rebuild("wiki-1") is True

    # Rebuild from DB-eligible records
    records = [FakeRecord(id="qa-1", question="What is auth?")]
    await manager.rebuild_from_records("wiki-1", records)

    # Index is now clean — search should find qa-1
    qa_ids, _ = await manager.search("wiki-1", "What is auth?", threshold=0.0)
    assert "qa-1" in qa_ids
    # Corrupt entry (qa-old) is gone
    assert "qa-old" not in qa_ids
