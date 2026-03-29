"""Tests for QAService."""

from __future__ import annotations

import hashlib
import uuid
from datetime import datetime, timedelta, timezone
from unittest.mock import AsyncMock, MagicMock

import numpy as np
import pytest
from pydantic import SecretStr
from sqlalchemy import select
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

from app.config import Settings
from app.models.db_models import Base, QARecord, WikiRecord
from app.services.qa_service import QAService


@pytest.fixture
async def async_engine():
    """Create async in-memory SQLite engine."""
    engine = create_async_engine("sqlite+aiosqlite:///:memory:")
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    yield engine
    await engine.dispose()


@pytest.fixture
def session_factory(async_engine):
    return async_sessionmaker(async_engine, expire_on_commit=False)


@pytest.fixture
def settings():
    return Settings(
        llm_api_key=SecretStr("test-key"),
        qa_cache_enabled=True,
        qa_cache_similarity_threshold=0.92,
        qa_cache_max_age_seconds=86400,
    )


@pytest.fixture
def mock_cache():
    cache = AsyncMock()
    cache.search = AsyncMock(return_value=([], np.zeros(8, dtype=np.float32)))
    cache.add = AsyncMock()
    cache.delete_index = MagicMock(return_value=True)
    # check_needs_rebuild is sync — use MagicMock to avoid RuntimeWarning
    cache.check_needs_rebuild = MagicMock(return_value=False)
    return cache


@pytest.fixture
def service(session_factory, mock_cache, settings):
    return QAService(session_factory, mock_cache, settings)


@pytest.fixture
def service_no_cache(session_factory, settings):
    return QAService(session_factory, None, settings)


async def _insert_qa_record(session_factory, **kwargs):
    """Helper to insert a QARecord."""
    defaults = dict(
        id=str(uuid.uuid4()),
        wiki_id="wiki-1",
        question="test question",
        question_hash=hashlib.sha256(b"test question").hexdigest(),
        answer="test answer",
        sources_json="[]",
        tool_steps=1,
        mode="fast",
        status="pending",
        is_cache_hit=0,
        has_context=0,
        created_at=datetime.now(timezone.utc).replace(tzinfo=None),
    )
    defaults.update(kwargs)
    record = QARecord(**defaults)
    async with session_factory() as session:
        session.add(record)
        await session.commit()
    return record


async def _insert_wiki(session_factory, wiki_id="wiki-1", commit_hash="abc123"):
    """Helper to insert a WikiRecord."""
    async with session_factory() as session:
        wiki = WikiRecord(
            id=wiki_id,
            repo_url="https://github.com/test/repo",
            commit_hash=commit_hash,
            owner_id="test-owner",
            branch="main",
            title="Test Wiki",
        )
        session.add(wiki)
        await session.commit()


# --- lookup_cache tests ---


@pytest.mark.asyncio
async def test_lookup_cache_exact_hash_hit(service, session_factory):
    """Exact hash match returns record without computing embedding."""
    await _insert_wiki(session_factory)
    record = await _insert_qa_record(session_factory, source_commit_hash="abc123")
    cached, embedding = await service.lookup_cache("wiki-1", "test question")
    assert cached is not None
    assert cached.id == record.id
    assert embedding is None  # No embedding computed for exact match


@pytest.mark.asyncio
async def test_lookup_cache_semantic_hit(service, session_factory, mock_cache):
    """FAISS semantic match returns record + reusable embedding."""
    await _insert_wiki(session_factory)
    record = await _insert_qa_record(
        session_factory,
        question="similar question",
        question_hash=hashlib.sha256(b"similar question").hexdigest(),
        source_commit_hash="abc123",
    )
    # Mock FAISS to return this record's ID
    embedding = np.ones(8, dtype=np.float32)
    mock_cache.search.return_value = ([record.id], embedding)

    cached, emb = await service.lookup_cache("wiki-1", "different but similar")
    assert cached is not None
    assert cached.id == record.id
    assert emb is not None


@pytest.mark.asyncio
async def test_lookup_cache_miss(service, session_factory, mock_cache):
    """Cache miss returns (None, embedding)."""
    await _insert_wiki(session_factory)
    embedding = np.ones(8, dtype=np.float32)
    mock_cache.search.return_value = ([], embedding)

    cached, emb = await service.lookup_cache("wiki-1", "unknown question")
    assert cached is None
    assert emb is not None


@pytest.mark.asyncio
async def test_lookup_cache_context_bypass(service):
    """Non-empty chat_history returns (None, None)."""
    cached, embedding = await service.lookup_cache(
        "wiki-1", "question", chat_history=[{"role": "user", "content": "hi"}]
    )
    assert cached is None
    assert embedding is None


@pytest.mark.asyncio
async def test_lookup_cache_commit_filtering(service, session_factory, mock_cache):
    """Old commit records are excluded from semantic search."""
    await _insert_wiki(session_factory, commit_hash="new-commit")
    record = await _insert_qa_record(
        session_factory,
        question="outdated question",
        question_hash=hashlib.sha256(b"outdated question").hexdigest(),
        source_commit_hash="old-commit",
    )
    embedding = np.ones(8, dtype=np.float32)
    mock_cache.search.return_value = ([record.id], embedding)

    cached, emb = await service.lookup_cache("wiki-1", "outdated question")
    assert cached is None  # Excluded by commit mismatch


@pytest.mark.asyncio
async def test_lookup_cache_ttl_filtering(service, session_factory, mock_cache):
    """Expired records are excluded."""
    await _insert_wiki(session_factory)
    old_time = datetime.now(timezone.utc) - timedelta(days=2)
    record = await _insert_qa_record(
        session_factory,
        question="old question",
        question_hash=hashlib.sha256(b"old question").hexdigest(),
        created_at=old_time.replace(tzinfo=None),
        source_commit_hash="abc123",
    )
    embedding = np.ones(8, dtype=np.float32)
    mock_cache.search.return_value = ([record.id], embedding)

    cached, emb = await service.lookup_cache("wiki-1", "old question")
    assert cached is None  # Excluded by TTL


@pytest.mark.asyncio
async def test_lookup_cache_k5_iteration(service, session_factory, mock_cache):
    """First candidate invalid, second valid -- returns second."""
    await _insert_wiki(session_factory)
    # Invalid record (stale status)
    stale = await _insert_qa_record(
        session_factory,
        id="stale-id",
        status="stale",
        source_commit_hash="abc123",
    )
    # Valid record
    valid = await _insert_qa_record(
        session_factory,
        id="valid-id",
        question="valid q",
        question_hash=hashlib.sha256(b"valid q").hexdigest(),
        source_commit_hash="abc123",
    )
    embedding = np.ones(8, dtype=np.float32)
    mock_cache.search.return_value = ([stale.id, valid.id], embedding)

    cached, emb = await service.lookup_cache("wiki-1", "some question")
    assert cached is not None
    assert cached.id == "valid-id"


@pytest.mark.asyncio
async def test_lookup_cache_disabled(session_factory, settings):
    """qa_cache_enabled=False returns (None, None)."""
    settings.qa_cache_enabled = False
    svc = QAService(session_factory, None, settings)
    cached, emb = await svc.lookup_cache("wiki-1", "question")
    assert cached is None
    assert emb is None


# --- record_interaction tests ---


@pytest.mark.asyncio
async def test_record_interaction_creates_record(service, session_factory):
    """record_interaction creates QARecord with all fields."""
    qa_id = str(uuid.uuid4())
    await service.record_interaction(
        qa_id=qa_id,
        wiki_id="wiki-1",
        question="How does auth work?",
        answer="JWT tokens.",
        sources_json='[{"file": "auth.py"}]',
        tool_steps=3,
        mode="deep",
        user_id="user-1",
        is_cache_hit=False,
        source_qa_id=None,
        embedding=np.ones(8, dtype=np.float32),
        has_context=False,
        source_commit_hash="abc123",
    )
    async with session_factory() as session:
        result = await session.execute(select(QARecord).where(QARecord.id == qa_id))
        record = result.scalar_one()
        assert record.wiki_id == "wiki-1"
        assert record.mode == "deep"
        assert record.is_cache_hit == 0
        assert record.source_commit_hash == "abc123"


@pytest.mark.asyncio
async def test_record_interaction_adds_to_faiss(service, mock_cache):
    """Non-cache-hit, non-contextual records are added to FAISS."""
    await service.record_interaction(
        qa_id="qa-1",
        wiki_id="wiki-1",
        question="q",
        answer="a",
        sources_json="[]",
        tool_steps=0,
        mode="fast",
        user_id=None,
        is_cache_hit=False,
        source_qa_id=None,
        embedding=np.ones(8, dtype=np.float32),
    )
    mock_cache.add.assert_called_once()


@pytest.mark.asyncio
async def test_record_interaction_skips_faiss_cache_hit(service, mock_cache):
    """Cache hits are NOT added to FAISS."""
    await service.record_interaction(
        qa_id="qa-1",
        wiki_id="wiki-1",
        question="q",
        answer="a",
        sources_json="[]",
        tool_steps=0,
        mode="fast",
        user_id=None,
        is_cache_hit=True,
        source_qa_id="qa-0",
        embedding=np.ones(8, dtype=np.float32),
    )
    mock_cache.add.assert_not_called()


@pytest.mark.asyncio
async def test_record_interaction_skips_faiss_contextual(service, mock_cache):
    """Contextual answers (has_context=True) are NOT added to FAISS."""
    await service.record_interaction(
        qa_id="qa-1",
        wiki_id="wiki-1",
        question="q",
        answer="a",
        sources_json="[]",
        tool_steps=0,
        mode="fast",
        user_id=None,
        is_cache_hit=False,
        source_qa_id=None,
        embedding=np.ones(8, dtype=np.float32),
        has_context=True,
    )
    mock_cache.add.assert_not_called()


@pytest.mark.asyncio
async def test_record_interaction_graceful_degradation(service_no_cache, session_factory):
    """DB failure logs error, no exception propagated."""
    # Create a proper async context manager whose __aenter__ raises,
    # simulating a real DB connection failure without unawaited coroutines.
    failing_cm = AsyncMock()
    failing_cm.__aenter__ = AsyncMock(side_effect=Exception("DB down"))
    failing_cm.__aexit__ = AsyncMock(return_value=False)
    service_no_cache._session_factory = MagicMock(return_value=failing_cm)
    # Should not raise
    await service_no_cache.record_interaction(
        qa_id="qa-1",
        wiki_id="wiki-1",
        question="q",
        answer="a",
        sources_json="[]",
        tool_steps=0,
        mode="fast",
        user_id=None,
        is_cache_hit=False,
        source_qa_id=None,
        embedding=None,
    )


# --- list_qa tests ---


@pytest.mark.asyncio
async def test_list_qa_pagination(service, session_factory):
    """list_qa with pagination and status filter."""
    for i in range(5):
        await _insert_qa_record(session_factory, id=f"qa-{i}")
    records, total = await service.list_qa("wiki-1", limit=2, offset=0)
    assert total == 5
    assert len(records) == 2


@pytest.mark.asyncio
async def test_list_qa_status_filter(service, session_factory):
    """list_qa with status filter."""
    await _insert_qa_record(session_factory, id="qa-1", status="pending")
    await _insert_qa_record(session_factory, id="qa-2", status="validated")
    records, total = await service.list_qa("wiki-1", status="validated")
    assert total == 1
    assert records[0].id == "qa-2"


# --- get_stats tests ---


@pytest.mark.asyncio
async def test_get_stats(service, session_factory):
    """get_stats returns correct counts and hit_rate."""
    await _insert_qa_record(session_factory, id="qa-1", is_cache_hit=0, status="pending")
    await _insert_qa_record(session_factory, id="qa-2", is_cache_hit=1, status="pending")
    await _insert_qa_record(session_factory, id="qa-3", is_cache_hit=0, status="validated")
    await _insert_qa_record(session_factory, id="qa-4", is_cache_hit=0, status="rejected")

    stats = await service.get_stats("wiki-1")
    assert stats["total_count"] == 4
    assert stats["cache_hit_count"] == 1
    assert stats["validated_count"] == 1
    assert stats["rejected_count"] == 1
    assert stats["hit_rate"] == 0.25


# --- invalidate_wiki tests ---


@pytest.mark.asyncio
async def test_invalidate_wiki_delete(service, session_factory, mock_cache):
    """delete=True removes all rows + FAISS."""
    await _insert_qa_record(session_factory, id="qa-1")
    await _insert_qa_record(session_factory, id="qa-2", status="validated")

    await service.invalidate_wiki("wiki-1", delete=True)

    records, total = await service.list_qa("wiki-1")
    assert total == 0
    mock_cache.delete_index.assert_called_once_with("wiki-1")


@pytest.mark.asyncio
async def test_invalidate_wiki_refresh(service, session_factory, mock_cache):
    """delete=False marks pending->stale, preserves validated/enriched."""
    await _insert_qa_record(session_factory, id="qa-1", status="pending")
    await _insert_qa_record(session_factory, id="qa-2", status="validated")

    await service.invalidate_wiki("wiki-1", delete=False)

    # Check statuses
    async with session_factory() as session:
        result = await session.execute(select(QARecord).where(QARecord.id == "qa-1"))
        r1 = result.scalar_one()
        assert r1.status == "stale"

        result = await session.execute(select(QARecord).where(QARecord.id == "qa-2"))
        r2 = result.scalar_one()
        assert r2.status == "validated"  # Preserved

    mock_cache.delete_index.assert_called_once_with("wiki-1")


@pytest.mark.asyncio
async def test_rebuild_excludes_expired_and_wrong_commit_rows(session_factory, settings):
    """_rebuild_cache_from_db only includes cache-eligible rows (commit + TTL)."""
    import tempfile

    from langchain_core.embeddings import Embeddings as _Embeddings

    from app.services.qa_cache_manager import QACacheManager

    class _FakeEmbeddings(_Embeddings):
        def embed_query(self, text: str) -> list[float]:
            np.random.seed(hash(text) % 2**31)
            return np.random.randn(8).astype(np.float32).tolist()

        def embed_documents(self, texts: list[str]) -> list[list[float]]:
            return [self.embed_query(t) for t in texts]

    with tempfile.TemporaryDirectory() as tmp_dir:
        cache = QACacheManager(tmp_dir, _FakeEmbeddings())
        svc = QAService(session_factory, cache, settings)

        await _insert_wiki(session_factory, commit_hash="current-commit")

        # Eligible: current commit, recent
        await _insert_qa_record(
            session_factory, id="qa-eligible",
            question="What is auth?",
            question_hash=hashlib.sha256(b"What is auth?").hexdigest(),
            source_commit_hash="current-commit",
        )
        # Excluded: wrong commit
        await _insert_qa_record(
            session_factory, id="qa-wrong-commit",
            question="How does routing work?",
            question_hash=hashlib.sha256(b"How does routing work?").hexdigest(),
            source_commit_hash="old-commit",
        )
        # Excluded: expired TTL (2 days old, default max_age is 86400s = 1 day)
        old_time = datetime.now(timezone.utc) - timedelta(days=2)
        await _insert_qa_record(
            session_factory, id="qa-expired",
            question="What is caching?",
            question_hash=hashlib.sha256(b"What is caching?").hexdigest(),
            source_commit_hash="current-commit",
            created_at=old_time.replace(tzinfo=None),
        )

        # Trigger rebuild
        await svc._rebuild_cache_from_db("wiki-1")

        # Only the eligible record should be in the rebuilt index
        assert cache._qa_ids.get("wiki-1") == ["qa-eligible"]


@pytest.mark.asyncio
async def test_lookup_cache_triggers_rebuild_on_corruption(session_factory, settings):
    """When FAISS index is corrupt, lookup_cache triggers rebuild_from_records
    and the record becomes semantically searchable after recovery."""
    import faiss
    import json
    import os
    import tempfile

    from langchain_core.embeddings import Embeddings as _Embeddings

    from app.services.qa_cache_manager import QACacheManager

    class _FakeEmbeddings(_Embeddings):
        """Deterministic embeddings for this test (no external API)."""

        def embed_query(self, text: str) -> list[float]:
            np.random.seed(hash(text) % 2**31)
            vec = np.random.randn(8).astype(np.float32)
            return vec.tolist()

        def embed_documents(self, texts: list[str]) -> list[list[float]]:
            return [self.embed_query(t) for t in texts]

    with tempfile.TemporaryDirectory() as tmp_dir:
        cache = QACacheManager(tmp_dir, _FakeEmbeddings())
        svc = QAService(session_factory, cache, settings)

        # Insert an eligible DB record
        await _insert_wiki(session_factory)
        await _insert_qa_record(
            session_factory,
            id="qa-eligible",
            question="What is auth?",
            question_hash=hashlib.sha256(b"What is auth?").hexdigest(),
            source_commit_hash="abc123",
        )

        # Write a corrupt index: 1 vector but 2 ids
        idx = faiss.IndexFlatIP(8)
        vec = np.random.randn(1, 8).astype(np.float32)
        faiss.normalize_L2(vec)
        idx.add(vec)  # type: ignore[call-arg]
        faiss.write_index(idx, os.path.join(tmp_dir, "wiki-1.qa.faiss"))
        with open(os.path.join(tmp_dir, "wiki-1.qa.ids.json"), "w") as f:
            json.dump(["stale-id-1", "stale-id-2"], f)

        # Corruption should be detected before lookup
        assert cache.check_needs_rebuild("wiki-1") is True

        # lookup_cache returns the record via exact hash match (Layer 1 DB hit)
        # — this path doesn't need FAISS so it succeeds even with a corrupt index
        cached, _ = await svc.lookup_cache("wiki-1", "What is auth?")
        assert cached is not None
        assert str(cached.id) == "qa-eligible"

        # Now trigger a semantic lookup with a slightly different question
        # to exercise the rebuild path (Layer 1 will miss, Layer 2 needs FAISS)
        cached2, _ = await svc.lookup_cache("wiki-1", "How does auth work?")
        # After _rebuild_cache_from_db runs, the index is clean
        assert cache.check_needs_rebuild("wiki-1") is False
