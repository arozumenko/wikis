"""Unit tests for ``app.services.change_detector``.

Covers the four change kinds (added / modified / deleted / moved), the
unchanged-omission rule, the affected-pages reverse lookup, and the
edge cases that matter for #116 PR 2:

* Unindexed wiki → every parsed node is ``added``
* All-deleted (parsed input empty)
* Pre-#116 rows with ``content_hash = NULL`` are treated as modified
* Page that cites multiple changed nodes shows up once with all changes
"""

from __future__ import annotations

import pytest

from app.core.unified_db import UnifiedWikiDB
from app.services.change_detector import (
    AffectedPage,
    ChangeDetector,
    ChangeKind,
    ChangeSet,
    NodeChange,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def db(tmp_path):
    d = UnifiedWikiDB(tmp_path / "change_detect.wiki.db", embedding_dim=8)
    yield d
    d.close()


@pytest.fixture()
def detector(db: UnifiedWikiDB) -> ChangeDetector:
    return ChangeDetector(db)


def _seed_node(db: UnifiedWikiDB, node_id: str, *, source: str, path: str = "x.py") -> None:
    """Insert a node — storage auto-hashes source_text into content_hash."""
    db.upsert_node(
        node_id,
        rel_path=path,
        file_name=path.rsplit("/", 1)[-1],
        language="python",
        symbol_name=node_id,
        symbol_type="function",
        source_text=source,
    )


def _parsed(node_id: str, *, hash: str | None, path: str = "x.py") -> dict:
    return {"node_id": node_id, "content_hash": hash, "rel_path": path}


def _hash(text: str) -> str:
    from app.core.storage.incremental import compute_content_hash

    return compute_content_hash(text)


# ---------------------------------------------------------------------------
# detect_changes — basic kinds
# ---------------------------------------------------------------------------


class TestDetectChanges:
    def test_unindexed_wiki_marks_every_node_added(
        self, detector: ChangeDetector,
    ) -> None:
        # Empty DB → every parsed node is brand new.
        cs = detector.detect_changes([
            _parsed("a", hash=_hash("def a(): pass")),
            _parsed("b", hash=_hash("def b(): pass")),
        ])
        assert len(cs.added) == 2
        assert cs.modified == [] and cs.deleted == [] and cs.moved == []
        assert {c.node_id for c in cs.added} == {"a", "b"}

    def test_unchanged_node_omitted(
        self, db: UnifiedWikiDB, detector: ChangeDetector,
    ) -> None:
        _seed_node(db, "a", source="def a(): pass")
        cs = detector.detect_changes([_parsed("a", hash=_hash("def a(): pass"))])
        assert cs.is_empty
        assert cs.total == 0

    def test_modified_when_hash_differs(
        self, db: UnifiedWikiDB, detector: ChangeDetector,
    ) -> None:
        _seed_node(db, "a", source="def a(): pass")
        cs = detector.detect_changes(
            [_parsed("a", hash=_hash("def a(): return 1"))]
        )
        assert len(cs.modified) == 1
        change = cs.modified[0]
        assert change.kind == ChangeKind.MODIFIED
        assert change.node_id == "a"
        assert change.old_hash == _hash("def a(): pass")
        assert change.new_hash == _hash("def a(): return 1")

    def test_deleted_when_parsed_set_omits_node(
        self, db: UnifiedWikiDB, detector: ChangeDetector,
    ) -> None:
        _seed_node(db, "a", source="def a(): pass")
        _seed_node(db, "b", source="def b(): pass")
        cs = detector.detect_changes([_parsed("a", hash=_hash("def a(): pass"))])
        assert len(cs.deleted) == 1
        assert cs.deleted[0].node_id == "b"
        assert cs.deleted[0].kind == ChangeKind.DELETED

    def test_moved_when_path_changes_but_hash_same(
        self, db: UnifiedWikiDB, detector: ChangeDetector,
    ) -> None:
        _seed_node(db, "a", source="def a(): pass", path="old.py")
        cs = detector.detect_changes([
            _parsed("a", hash=_hash("def a(): pass"), path="new.py")
        ])
        assert len(cs.moved) == 1
        change = cs.moved[0]
        assert change.kind == ChangeKind.MOVED
        assert change.old_path == "old.py"
        assert change.new_path == "new.py"
        # Modified column stays empty — a pure move is not a content change.
        assert cs.modified == []

    def test_pre_pr1_null_hash_treated_as_modified(
        self, db: UnifiedWikiDB, detector: ChangeDetector,
    ) -> None:
        # Simulate a node indexed before PR 1: NULL content_hash. We have to
        # bypass the auto-hash in upsert by inserting an empty source_text
        # then NULLing the hash directly.
        db.upsert_node(
            "legacy",
            rel_path="x.py",
            file_name="x.py",
            language="python",
            symbol_name="legacy",
            symbol_type="function",
            source_text="",  # auto-hash returns "" → stored as ""
        )
        db.conn.execute(
            "UPDATE repo_nodes SET content_hash = NULL WHERE node_id = ?",
            ("legacy",),
        )
        db.conn.commit()

        cs = detector.detect_changes(
            [_parsed("legacy", hash=_hash("anything new"))]
        )
        assert len(cs.modified) == 1
        assert cs.modified[0].old_hash is None
        assert cs.modified[0].new_hash == _hash("anything new")


class TestDetectChangesMixed:
    def test_mixed_change_kinds_in_one_call(
        self, db: UnifiedWikiDB, detector: ChangeDetector,
    ) -> None:
        # Seed: a (unchanged), b (will modify), c (will delete), d (will move)
        _seed_node(db, "a", source="def a(): pass")
        _seed_node(db, "b", source="def b(): pass")
        _seed_node(db, "c", source="def c(): pass")
        _seed_node(db, "d", source="def d(): pass", path="old.py")

        parsed = [
            _parsed("a", hash=_hash("def a(): pass")),                       # unchanged
            _parsed("b", hash=_hash("def b(): return 1")),                   # modified
            _parsed("d", hash=_hash("def d(): pass"), path="new.py"),        # moved
            _parsed("e", hash=_hash("def e(): pass")),                       # added
            # "c" omitted → deleted
        ]
        cs = detector.detect_changes(parsed)

        assert {c.node_id for c in cs.added} == {"e"}
        assert {c.node_id for c in cs.modified} == {"b"}
        assert {c.node_id for c in cs.moved} == {"d"}
        assert {c.node_id for c in cs.deleted} == {"c"}
        assert cs.total == 4

    def test_all_deleted_when_parsed_is_empty(
        self, db: UnifiedWikiDB, detector: ChangeDetector,
    ) -> None:
        _seed_node(db, "a", source="def a(): pass")
        _seed_node(db, "b", source="def b(): pass")
        cs = detector.detect_changes([])
        assert {c.node_id for c in cs.deleted} == {"a", "b"}
        assert cs.added == cs.modified == cs.moved == []

    def test_change_set_total_and_is_empty(self) -> None:
        cs = ChangeSet()
        assert cs.is_empty and cs.total == 0
        cs.added.append(NodeChange(kind=ChangeKind.ADDED, node_id="x"))
        assert not cs.is_empty and cs.total == 1


# ---------------------------------------------------------------------------
# affected_pages
# ---------------------------------------------------------------------------


def _seed_page(db: UnifiedWikiDB, page_id: str, symbols: list[tuple[str, str]]) -> None:
    db.upsert_wiki_page_with_symbols(
        {
            "page_id": page_id,
            "wiki_id": "w",
            "title": page_id.upper(),
            "anchor_slug": page_id,
        },
        symbols,
    )


class TestAffectedPageMaxKind:
    """PR 3's three-regime classifier reads .max_kind off each affected
    page to decide trivial / edit / structural routing — see severity
    ordering at the top of the module."""

    def test_returns_none_when_no_changes(self) -> None:
        # Invariant violation in practice, but the helper must be safe.
        from app.services.change_detector import AffectedPage

        assert AffectedPage(page_id="p1", changes=[]).max_kind is None

    def test_single_change_returns_its_kind(self) -> None:
        from app.services.change_detector import AffectedPage

        page = AffectedPage(
            page_id="p1",
            changes=[NodeChange(kind=ChangeKind.MODIFIED, node_id="a")],
        )
        assert page.max_kind == ChangeKind.MODIFIED

    def test_picks_highest_severity_when_multiple(self) -> None:
        # Severity: MOVED < MODIFIED < ADDED < DELETED.
        from app.services.change_detector import AffectedPage

        page = AffectedPage(
            page_id="p1",
            changes=[
                NodeChange(kind=ChangeKind.MOVED, node_id="a"),
                NodeChange(kind=ChangeKind.DELETED, node_id="b"),
                NodeChange(kind=ChangeKind.MODIFIED, node_id="c"),
            ],
        )
        # DELETED wins.
        assert page.max_kind == ChangeKind.DELETED

    def test_moved_only_yields_moved(self) -> None:
        # PR 3 routes MOVED-only pages to the no-LLM trivial regime.
        from app.services.change_detector import AffectedPage

        page = AffectedPage(
            page_id="p1",
            changes=[
                NodeChange(kind=ChangeKind.MOVED, node_id="a"),
                NodeChange(kind=ChangeKind.MOVED, node_id="b"),
            ],
        )
        assert page.max_kind == ChangeKind.MOVED


class TestAffectedPages:
    def test_empty_change_set_returns_empty(
        self, detector: ChangeDetector,
    ) -> None:
        assert detector.affected_pages(ChangeSet()) == []

    def test_groups_changes_by_page(
        self, db: UnifiedWikiDB, detector: ChangeDetector,
    ) -> None:
        _seed_node(db, "a", source="def a(): pass")
        _seed_node(db, "b", source="def b(): pass")
        _seed_page(db, "page-1", [("a", "primary"), ("b", "related")])

        cs = detector.detect_changes([
            _parsed("a", hash=_hash("def a(): return 1")),
            _parsed("b", hash=_hash("def b(): return 2")),
        ])
        affected = detector.affected_pages(cs)

        assert len(affected) == 1
        assert affected[0].page_id == "page-1"
        # Both changes attributed to the one page.
        assert {c.node_id for c in affected[0].changes} == {"a", "b"}

    def test_node_cited_by_multiple_pages_fans_out(
        self, db: UnifiedWikiDB, detector: ChangeDetector,
    ) -> None:
        _seed_node(db, "shared", source="def shared(): pass")
        _seed_page(db, "page-1", [("shared", "primary")])
        _seed_page(db, "page-2", [("shared", "referenced")])

        cs = detector.detect_changes(
            [_parsed("shared", hash=_hash("def shared(): return 1"))]
        )
        affected = detector.affected_pages(cs)
        assert {p.page_id for p in affected} == {"page-1", "page-2"}
        for page in affected:
            assert len(page.changes) == 1
            assert page.changes[0].node_id == "shared"

    def test_uncited_change_yields_no_affected_pages(
        self, db: UnifiedWikiDB, detector: ChangeDetector,
    ) -> None:
        # A node changes but no wiki page cites it → no affected pages.
        _seed_node(db, "orphan", source="def orphan(): pass")
        cs = detector.detect_changes(
            [_parsed("orphan", hash=_hash("def orphan(): return 1"))]
        )
        assert detector.affected_pages(cs) == []
