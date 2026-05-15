"""Tests for the storage row-cap behaviour.

When the bounded fetchers (``get_architectural_nodes``,
``get_nodes_by_cluster``, ``get_architectural_node_ids``,
``get_all_edges``) hit their LIMIT, they emit a WARNING log so the
truncation is visible.  Callers that need every row should pass
``limit=None`` to bypass the LIMIT entirely.

Both behaviours are tested here against the SQLite backend.  The
Postgres backend implements the same protocol — its semantics are
exercised in ``tests/integration/test_postgres_storage_e2e.py`` once
Postgres is available.
"""

from __future__ import annotations

import logging

import pytest

from app.core.unified_db import UnifiedWikiDB


@pytest.fixture()
def db(tmp_path):
    d = UnifiedWikiDB(tmp_path / "trunc.wiki.db", embedding_dim=8)
    yield d
    d.close()


def _insert_arch_node(db: UnifiedWikiDB, node_id: str, **extra) -> None:
    db.upsert_node(
        node_id,
        rel_path=f"src/{node_id}.py",
        symbol_name=node_id,
        symbol_type="function",
        language="python",
        is_architectural=1,
        source_text=f"def {node_id}(): pass",
        **extra,
    )


def _insert_edge(db: UnifiedWikiDB, src: str, tgt: str, rel_type: str = "calls") -> None:
    db.upsert_edge(src, tgt, rel_type, weight=1.0)


# ═══════════════════════════════════════════════════════════════════════════
# limit=None disables the row cap
# ═══════════════════════════════════════════════════════════════════════════


class TestLimitNone:
    def test_get_architectural_nodes_none_returns_all(self, db):
        for i in range(25):
            _insert_arch_node(db, f"sym_{i}")
        db.commit()

        # Default limit would already include all 25, but the call must
        # not error out with limit=None.
        rows = db.get_architectural_nodes(limit=None)
        assert len(rows) == 25

    def test_get_architectural_nodes_default_limit_caps(self, db):
        # Default limit is 5000; inserting 5001 would be slow.  Force a
        # small limit instead and verify the cap.
        for i in range(30):
            _insert_arch_node(db, f"sym_{i}")
        db.commit()

        rows = db.get_architectural_nodes(limit=10)
        assert len(rows) == 10

    def test_get_architectural_node_ids_none_returns_all(self, db):
        for i in range(25):
            _insert_arch_node(db, f"sym_{i}")
        db.commit()

        ids = db.get_architectural_node_ids(limit=None)
        assert len(ids) == 25
        assert "sym_0" in ids
        assert "sym_24" in ids

    def test_get_all_edges_none_returns_all(self, db):
        for i in range(10):
            _insert_arch_node(db, f"sym_{i}")
        db.commit()
        for i in range(9):
            _insert_edge(db, f"sym_{i}", f"sym_{i + 1}")
        db.commit()

        edges = db.get_all_edges(limit=None)
        assert len(edges) == 9

    def test_get_nodes_by_cluster_none_returns_all(self, db):
        for i in range(25):
            _insert_arch_node(db, f"sym_{i}", macro_cluster=1)
        db.commit()

        rows = db.get_nodes_by_cluster(macro=1, limit=None)
        assert len(rows) == 25


# ═══════════════════════════════════════════════════════════════════════════
# Warning log fires when a finite limit is hit
# ═══════════════════════════════════════════════════════════════════════════


class TestTruncationWarning:
    def test_warns_when_architectural_nodes_capped(self, db, caplog):
        for i in range(20):
            _insert_arch_node(db, f"sym_{i}")
        db.commit()

        with caplog.at_level(logging.WARNING, logger="app.core.storage.sqlite"):
            rows = db.get_architectural_nodes(limit=10)

        assert len(rows) == 10
        warning_messages = [r.message for r in caplog.records if r.levelno == logging.WARNING]
        assert any(
            "get_architectural_nodes" in m and "limit=10" in m
            for m in warning_messages
        ), f"Expected truncation warning, got: {warning_messages}"

    def test_does_not_warn_when_under_limit(self, db, caplog):
        for i in range(5):
            _insert_arch_node(db, f"sym_{i}")
        db.commit()

        with caplog.at_level(logging.WARNING, logger="app.core.storage.sqlite"):
            rows = db.get_architectural_nodes(limit=100)

        assert len(rows) == 5
        warning_messages = [r.message for r in caplog.records if r.levelno == logging.WARNING]
        assert not any("hit row cap" in m for m in warning_messages), (
            f"Did not expect truncation warning when under cap, got: {warning_messages}"
        )

    def test_does_not_warn_when_limit_is_none(self, db, caplog):
        for i in range(20):
            _insert_arch_node(db, f"sym_{i}")
        db.commit()

        with caplog.at_level(logging.WARNING, logger="app.core.storage.sqlite"):
            db.get_architectural_nodes(limit=None)

        warning_messages = [r.message for r in caplog.records if r.levelno == logging.WARNING]
        assert not any("hit row cap" in m for m in warning_messages), (
            f"limit=None must never trigger the truncation warning, got: {warning_messages}"
        )

    def test_warning_includes_context(self, db, caplog):
        for i in range(10):
            _insert_arch_node(db, f"sym_{i}", macro_cluster=42)
        db.commit()

        with caplog.at_level(logging.WARNING, logger="app.core.storage.sqlite"):
            db.get_nodes_by_cluster(macro=42, limit=5)

        warning_messages = [r.message for r in caplog.records if r.levelno == logging.WARNING]
        # Macro id should appear in the contextual suffix so operators
        # can pin down which call truncated.
        assert any("macro=42" in m for m in warning_messages), (
            f"Expected macro context in warning, got: {warning_messages}"
        )
