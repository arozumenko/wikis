"""Tests for #116 PR 4 embedding reuse — populate_embeddings must skip
nodes whose source hasn't changed since the last embed.

The contract:

* First call embeds every architectural node, stamps embedding_content_hash.
* Second call with no changes embeds zero nodes (no calls to embedding_fn).
* After mutating a node's source_text (and therefore content_hash), only
  that node gets re-embedded on the next call.
* Pre-PR4 rows with NULL embedding_content_hash are re-embedded once on
  the next populate_embeddings, then stamped.
"""

from __future__ import annotations

import pytest

from app.core.unified_db import UnifiedWikiDB


@pytest.fixture()
def db(tmp_path, monkeypatch):
    # The skip-filter we care about runs inside the SELECT — independent
    # of whether sqlite-vec is loadable. To keep these tests portable we
    # pretend vec is available and replace upsert_embeddings_batch with a
    # no-op so populate_embeddings runs end-to-end without writing real
    # vectors. The test gates on which texts the embedder *sees*, not on
    # repo_vec contents.
    d = UnifiedWikiDB(tmp_path / "embed.wiki.db", embedding_dim=8)
    d._vec_available = True
    monkeypatch.setattr(d, "upsert_embeddings_batch", lambda pairs: None)
    yield d
    d.close()


def _seed(db: UnifiedWikiDB, node_id: str, source: str) -> None:
    db.upsert_node(
        node_id,
        rel_path="x.py",
        file_name="x.py",
        language="python",
        symbol_name=node_id,
        symbol_type="function",
        is_architectural=1,
        source_text=source,
    )


class _CountingEmbedder:
    """Records every (text→vector) call so tests can assert no-op behavior."""

    def __init__(self, dim: int = 8) -> None:
        self.calls: list[list[str]] = []
        self._dim = dim

    def __call__(self, texts: list[str]) -> list[list[float]]:
        self.calls.append(list(texts))
        return [[0.0] * self._dim for _ in texts]

    @property
    def total_texts(self) -> int:
        return sum(len(c) for c in self.calls)


def _read_embedding_hashes(db: UnifiedWikiDB) -> dict[str, str | None]:
    rows = db.conn.execute(
        "SELECT node_id, embedding_content_hash FROM repo_nodes"
    ).fetchall()
    return {r[0]: r[1] for r in rows}


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestEmbeddingReuse:
    def test_first_run_embeds_every_node_and_stamps_hash(
        self, db: UnifiedWikiDB,
    ) -> None:
        _seed(db, "a", "def a(): pass")
        _seed(db, "b", "def b(): pass")
        embedder = _CountingEmbedder()

        n = db.populate_embeddings(embedder, batch_size=10)
        assert n == 2
        assert embedder.total_texts == 2

        hashes = _read_embedding_hashes(db)
        # Every node has a non-NULL embedding_content_hash now.
        assert hashes["a"] is not None
        assert hashes["b"] is not None

    def test_second_run_with_no_changes_is_a_noop(
        self, db: UnifiedWikiDB,
    ) -> None:
        _seed(db, "a", "def a(): pass")
        _seed(db, "b", "def b(): pass")

        # First run — embed everything.
        db.populate_embeddings(_CountingEmbedder(), batch_size=10)

        # Second run with a fresh embedder — should be called zero times.
        embedder = _CountingEmbedder()
        n = db.populate_embeddings(embedder, batch_size=10)
        assert n == 0
        assert embedder.total_texts == 0
        assert embedder.calls == []

    def test_modified_node_is_reembedded_others_skipped(
        self, db: UnifiedWikiDB,
    ) -> None:
        _seed(db, "a", "def a(): pass")
        _seed(db, "b", "def b(): pass")

        db.populate_embeddings(_CountingEmbedder(), batch_size=10)

        # Change a's source — content_hash will be recomputed at upsert.
        _seed(db, "a", "def a(): return 1")

        embedder = _CountingEmbedder()
        n = db.populate_embeddings(embedder, batch_size=10)
        assert n == 1
        assert embedder.total_texts == 1
        # The one re-embedded node was "a", not "b".
        assert embedder.calls == [["def a(): return 1"]]

    def test_pre_pr4_null_embedding_hash_is_picked_up_then_stamped(
        self, db: UnifiedWikiDB,
    ) -> None:
        _seed(db, "a", "def a(): pass")

        # Simulate a pre-PR4 row: clear embedding_content_hash.
        db.conn.execute(
            "UPDATE repo_nodes SET embedding_content_hash = NULL WHERE node_id = ?",
            ("a",),
        )
        db.conn.commit()
        assert _read_embedding_hashes(db)["a"] is None

        embedder = _CountingEmbedder()
        n = db.populate_embeddings(embedder, batch_size=10)
        assert n == 1
        # Now stamped — next run is a no-op.
        assert _read_embedding_hashes(db)["a"] is not None
        n2 = db.populate_embeddings(_CountingEmbedder(), batch_size=10)
        assert n2 == 0
