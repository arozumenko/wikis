"""SQLite-backend tests for the wiki_pages + page_symbols methods (#116 PR 1).

The protocol promises both backends behave identically; we test SQLite here
because it needs no infra. PostgreSQL behaviour is verified separately
through the existing storage-protocol integration suite.
"""

from __future__ import annotations

import pytest

from app.core.unified_db import UnifiedWikiDB


@pytest.fixture()
def db(tmp_path):
    d = UnifiedWikiDB(tmp_path / "wiki_pages.wiki.db", embedding_dim=8)
    yield d
    d.close()


# ---------------------------------------------------------------------------
# wiki_pages
# ---------------------------------------------------------------------------


def _page(page_id: str, wiki_id: str = "wiki-1", **overrides) -> dict:
    base = {
        "page_id": page_id,
        "wiki_id": wiki_id,
        "title": "Overview",
        "anchor_slug": page_id,  # unique-per-wiki guaranteed by caller
        "macro_cluster": 0,
        "micro_cluster": 1,
        "primary_symbol_id": "sym-1",
        "section_index": 0,
        "page_index": 0,
    }
    base.update(overrides)
    return base


class TestUpsertWikiPage:
    def test_round_trip(self, db: UnifiedWikiDB) -> None:
        db.upsert_wiki_page(_page("p1", title="Auth", anchor_slug="auth"))
        row = db.get_wiki_page("p1")
        assert row is not None
        assert row["page_id"] == "p1"
        assert row["wiki_id"] == "wiki-1"
        assert row["title"] == "Auth"
        assert row["anchor_slug"] == "auth"
        assert row["id_scheme"] == "stable_v1"  # default

    def test_replace_updates_in_place(self, db: UnifiedWikiDB) -> None:
        db.upsert_wiki_page(_page("p1", title="Auth", anchor_slug="auth"))
        db.upsert_wiki_page(
            _page("p1", title="Authentication", anchor_slug="authentication")
        )
        row = db.get_wiki_page("p1")
        assert row["title"] == "Authentication"
        assert row["anchor_slug"] == "authentication"

    def test_requires_page_id_and_wiki_id(self, db: UnifiedWikiDB) -> None:
        with pytest.raises(ValueError):
            db.upsert_wiki_page({"page_id": "p1"})
        with pytest.raises(ValueError):
            db.upsert_wiki_page({"wiki_id": "w1"})

    def test_anchor_slug_unique_per_wiki(self, db: UnifiedWikiDB) -> None:
        # Same slug allowed across wikis; collision within one wiki must error.
        db.upsert_wiki_page(_page("p1", wiki_id="w1", anchor_slug="overview"))
        db.upsert_wiki_page(_page("p2", wiki_id="w2", anchor_slug="overview"))
        with pytest.raises(Exception):  # IntegrityError from UNIQUE index
            db.upsert_wiki_page(_page("p3", wiki_id="w1", anchor_slug="overview"))


class TestListAndDelete:
    def test_get_wiki_pages_orders_by_indices(self, db: UnifiedWikiDB) -> None:
        # Insert out of order; expect (section_index, page_index) sort on read.
        db.upsert_wiki_page(_page("p_late", section_index=1, page_index=2, anchor_slug="late"))
        db.upsert_wiki_page(_page("p_early", section_index=0, page_index=0, anchor_slug="early"))
        db.upsert_wiki_page(_page("p_mid", section_index=0, page_index=1, anchor_slug="mid"))

        ids = [r["page_id"] for r in db.get_wiki_pages("wiki-1")]
        assert ids == ["p_early", "p_mid", "p_late"]

    def test_get_wiki_pages_scopes_by_wiki(self, db: UnifiedWikiDB) -> None:
        db.upsert_wiki_page(_page("p1", wiki_id="w1", anchor_slug="w1-overview"))
        db.upsert_wiki_page(_page("p2", wiki_id="w2", anchor_slug="w2-overview"))
        assert {r["page_id"] for r in db.get_wiki_pages("w1")} == {"p1"}
        assert {r["page_id"] for r in db.get_wiki_pages("w2")} == {"p2"}

    def test_delete_wiki_pages_cascades_to_symbols(self, db: UnifiedWikiDB) -> None:
        db.upsert_wiki_page(_page("p1"))
        db.record_page_symbols("p1", [("n1", "primary"), ("n2", "referenced")])
        # Sanity: rows present.
        assert db.get_pages_citing_node("n1") == ["p1"]

        deleted = db.delete_wiki_pages("wiki-1")
        assert deleted == 1
        assert db.get_wiki_page("p1") is None
        # FK cascade — page_symbols rows gone too.
        assert db.get_pages_citing_node("n1") == []


# ---------------------------------------------------------------------------
# page_symbols
# ---------------------------------------------------------------------------


class TestRecordPageSymbols:
    def test_basic_insert(self, db: UnifiedWikiDB) -> None:
        db.upsert_wiki_page(_page("p1"))
        db.record_page_symbols(
            "p1",
            [("sym-1", "primary"), ("sym-2", "referenced"), ("sym-3", "related")],
        )
        rows = db.get_page_symbols("p1")
        assert {(r["node_id"], r["citation_kind"]) for r in rows} == {
            ("sym-1", "primary"),
            ("sym-2", "referenced"),
            ("sym-3", "related"),
        }

    def test_replace_true_clears_old_rows(self, db: UnifiedWikiDB) -> None:
        db.upsert_wiki_page(_page("p1"))
        db.record_page_symbols("p1", [("sym-old", "referenced")])
        db.record_page_symbols("p1", [("sym-new", "referenced")])  # default replace=True

        rows = db.get_page_symbols("p1")
        assert len(rows) == 1
        assert rows[0]["node_id"] == "sym-new"

    def test_replace_false_appends(self, db: UnifiedWikiDB) -> None:
        db.upsert_wiki_page(_page("p1"))
        db.record_page_symbols("p1", [("sym-1", "referenced")], replace=False)
        db.record_page_symbols("p1", [("sym-2", "referenced")], replace=False)

        rows = db.get_page_symbols("p1")
        assert {r["node_id"] for r in rows} == {"sym-1", "sym-2"}

    def test_replace_false_idempotent_on_duplicate(self, db: UnifiedWikiDB) -> None:
        # The PK is (page_id, node_id, citation_kind). Re-recording the same
        # tuple must not raise — ON CONFLICT DO NOTHING / INSERT OR IGNORE.
        db.upsert_wiki_page(_page("p1"))
        db.record_page_symbols("p1", [("sym-1", "primary")], replace=False)
        db.record_page_symbols("p1", [("sym-1", "primary")], replace=False)
        rows = db.get_page_symbols("p1")
        assert len(rows) == 1

    def test_filter_by_citation_kind(self, db: UnifiedWikiDB) -> None:
        db.upsert_wiki_page(_page("p1"))
        db.record_page_symbols(
            "p1",
            [("sym-1", "primary"), ("sym-2", "referenced"), ("sym-3", "referenced")],
        )

        primary = db.get_page_symbols("p1", citation_kind="primary")
        referenced = db.get_page_symbols("p1", citation_kind="referenced")
        assert {r["node_id"] for r in primary} == {"sym-1"}
        assert {r["node_id"] for r in referenced} == {"sym-2", "sym-3"}


class TestReverseLookup:
    def test_get_pages_citing_node_dedupes_across_kinds(self, db: UnifiedWikiDB) -> None:
        # One node may appear as primary on one page and referenced on another.
        db.upsert_wiki_page(_page("p1", anchor_slug="p1"))
        db.upsert_wiki_page(_page("p2", anchor_slug="p2"))
        db.record_page_symbols("p1", [("sym-1", "primary")])
        db.record_page_symbols("p2", [("sym-1", "referenced")])

        cited_by = db.get_pages_citing_node("sym-1")
        assert sorted(cited_by) == ["p1", "p2"]

    def test_get_pages_citing_node_returns_empty_for_unknown(
        self, db: UnifiedWikiDB,
    ) -> None:
        assert db.get_pages_citing_node("missing") == []
