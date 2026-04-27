"""Tests for the SQLite FTS5 setup in :mod:`app.db._ensure_wiki_page_fts`.

The function is responsible for keeping ``wiki_page_fts`` (an external-content
FTS5 virtual table) in sync with ``wiki_page``.  It used to use a
``SELECT ... WHERE rowid NOT IN (SELECT rowid FROM wiki_page_fts)`` backfill
which was a silent no-op (external-content tables expose source rowids by
definition).  The current implementation uses the SQLite-recommended
``INSERT INTO wiki_page_fts(wiki_page_fts) VALUES('rebuild')`` pattern,
isolated in its own transaction so a rare rebuild failure does not roll back
the trigger DDL.
"""

from __future__ import annotations

import pytest
from sqlalchemy import text
from sqlalchemy.ext.asyncio import create_async_engine

from app.db import _ensure_wiki_page_fts
from app.models.db_models import Base


@pytest.fixture
async def engine():
    """In-memory async SQLite engine with the wiki_page table created."""
    eng = create_async_engine("sqlite+aiosqlite:///:memory:")
    async with eng.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    try:
        yield eng
    finally:
        await eng.dispose()


async def _fts_match_count(engine, query: str) -> int:
    async with engine.begin() as conn:
        result = await conn.execute(
            text(f"SELECT count(*) FROM wiki_page_fts WHERE wiki_page_fts MATCH '{query}'")
        )
        return int(result.scalar_one())


async def _insert_orphan_row(engine, page_id: str, content: str) -> None:
    """Insert a wiki_page row without firing the AFTER INSERT trigger.

    Simulates the state of a database that pre-existed the FTS5 migration,
    or one whose FTS index has drifted out of sync for any other reason.
    """
    async with engine.begin() as conn:
        await conn.execute(text("DROP TRIGGER IF EXISTS wiki_page_ai"))
        await conn.execute(
            text(
                "INSERT INTO wiki_page (id, wiki_id, page_title, description, content) "
                "VALUES (:id, 'w1', :id, '', :c)"
            ),
            {"id": page_id, "c": content},
        )


async def test_ensure_wiki_page_fts_creates_vtable_and_triggers(engine):
    """First call sets up the FTS5 vtable plus all three sync triggers."""
    await _ensure_wiki_page_fts(engine)

    async with engine.begin() as conn:
        result = await conn.execute(
            text(
                "SELECT name FROM sqlite_master "
                "WHERE name IN ('wiki_page_fts', 'wiki_page_ai', 'wiki_page_ad', 'wiki_page_au') "
                "ORDER BY name"
            )
        )
        names = [row[0] for row in result.fetchall()]

    assert names == ["wiki_page_ad", "wiki_page_ai", "wiki_page_au", "wiki_page_fts"]


async def test_rebuild_recovers_orphaned_rows(engine):
    """The rebuild backfill recovers FTS entries for rows whose insert bypassed the trigger.

    This is the regression test for the silent-no-op bug — without the
    ``rebuild`` command, an external-content FTS5 table never re-indexes
    pre-existing rows because ``WHERE rowid NOT IN (SELECT rowid FROM
    wiki_page_fts)`` is always empty.
    """
    await _ensure_wiki_page_fts(engine)
    await _insert_orphan_row(engine, "p1", "uniquemarkerorange1234")

    # Confirm the orphan state: row exists, FTS index does not contain it.
    async with engine.begin() as conn:
        wp_count = (await conn.execute(text("SELECT count(*) FROM wiki_page"))).scalar_one()
    assert wp_count == 1
    assert await _fts_match_count(engine, "uniquemarkerorange1234") == 0

    # Re-run the FTS setup; the rebuild command should pick up the orphan.
    await _ensure_wiki_page_fts(engine)

    assert await _fts_match_count(engine, "uniquemarkerorange1234") == 1


async def test_rebuild_is_idempotent_on_consistent_index(engine):
    """Calling ``_ensure_wiki_page_fts`` repeatedly is safe.

    Triggers use ``IF NOT EXISTS`` and the rebuild command is documented as
    idempotent — a no-op on an already-consistent index.
    """
    await _ensure_wiki_page_fts(engine)
    async with engine.begin() as conn:
        await conn.execute(
            text(
                "INSERT INTO wiki_page (id, wiki_id, page_title, description, content) "
                "VALUES ('p1', 'w1', 'p1', '', 'uniquemarkerteal5678')"
            )
        )
    assert await _fts_match_count(engine, "uniquemarkerteal5678") == 1

    # Second call should not raise and should not change the result.
    await _ensure_wiki_page_fts(engine)
    assert await _fts_match_count(engine, "uniquemarkerteal5678") == 1
