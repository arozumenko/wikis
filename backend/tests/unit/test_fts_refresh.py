"""Verify the FTS refresh contract for #116 PR 4.

SQLite's FTS5 virtual table has no triggers on ``repo_nodes``, so
partial upserts leave the index stale until ``refresh_fts_index()`` is
called. Postgres uses a BEFORE INSERT OR UPDATE trigger and stays in
sync automatically.

These tests pin down both behaviors so PR 3's incremental writer knows
which backend needs the explicit refresh call.
"""

from __future__ import annotations

import pytest

from app.core.unified_db import UnifiedWikiDB


@pytest.fixture()
def db(tmp_path):
    d = UnifiedWikiDB(tmp_path / "fts.wiki.db", embedding_dim=8)
    yield d
    d.close()


def _seed_and_index(db: UnifiedWikiDB, node_id: str, *, source: str) -> None:
    db.upsert_node(
        node_id,
        rel_path="x.py",
        file_name="x.py",
        language="python",
        symbol_name=node_id,
        symbol_type="function",
        source_text=source,
    )


def _fts_row_for(db: UnifiedWikiDB, node_id: str) -> dict | None:
    row = db.conn.execute(
        "SELECT * FROM repo_fts WHERE node_id = ?",
        (node_id,),
    ).fetchone()
    return dict(row) if row else None


class TestRefreshFtsIndex:
    def test_partial_upsert_alone_leaves_fts_stale(
        self, db: UnifiedWikiDB,
    ) -> None:
        # First seed + explicit rebuild — FTS now has the original row.
        _seed_and_index(db, "a", source="def a(): pass")
        db.refresh_fts_index()
        original = _fts_row_for(db, "a")
        assert original is not None
        assert "pass" in original["source_text"]

        # Partial upsert: change source_text but DON'T call refresh.
        _seed_and_index(db, "a", source="def a(): return 1")
        # FTS row still reflects the old content — that's the gap PR 3
        # will close by always calling refresh after incremental writes.
        stale = _fts_row_for(db, "a")
        assert stale is not None
        assert "pass" in stale["source_text"]
        assert "return 1" not in stale["source_text"]

    def test_refresh_fts_index_picks_up_changes(self, db: UnifiedWikiDB) -> None:
        _seed_and_index(db, "a", source="def a(): pass")
        db.refresh_fts_index()
        _seed_and_index(db, "a", source="def a(): return 1")
        db.refresh_fts_index()
        # FTS now reflects the new source.
        row = _fts_row_for(db, "a")
        assert row is not None
        assert "return 1" in row["source_text"]

    def test_refresh_drops_rows_for_deleted_nodes(
        self, db: UnifiedWikiDB,
    ) -> None:
        # A full rebuild reflects deletions too — repo_fts mirrors
        # repo_nodes after every refresh.
        _seed_and_index(db, "a", source="def a(): pass")
        _seed_and_index(db, "b", source="def b(): pass")
        db.refresh_fts_index()
        assert _fts_row_for(db, "a") is not None
        assert _fts_row_for(db, "b") is not None

        db.conn.execute("DELETE FROM repo_nodes WHERE node_id = ?", ("b",))
        db.conn.commit()
        db.refresh_fts_index()
        assert _fts_row_for(db, "a") is not None
        assert _fts_row_for(db, "b") is None

    def test_refresh_is_idempotent(self, db: UnifiedWikiDB) -> None:
        _seed_and_index(db, "a", source="def a(): pass")
        db.refresh_fts_index()
        first = _fts_row_for(db, "a")
        db.refresh_fts_index()
        second = _fts_row_for(db, "a")
        assert first == second
