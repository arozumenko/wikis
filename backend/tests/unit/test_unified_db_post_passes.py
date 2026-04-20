"""Tests for UnifiedWikiDB post-pass methods (§11.7, §11.8, §11.9).

Covers:
- P6: case-insensitive find_nodes_by_name
- B2: compute_god_nodes returns top-N degree symbols + cross-file edge files
- P4: demote_local_constants only demotes constants whose every reference is in
  the same file as the constant; unused constants are kept; cross-file refs keep
  the constant architectural.
"""

from __future__ import annotations

import pytest

from app.core.unified_db import UnifiedWikiDB


# ───────────────────────────────────────────────────────────────────────
# Fixtures
# ───────────────────────────────────────────────────────────────────────


@pytest.fixture()
def db(tmp_path):
    d = UnifiedWikiDB(tmp_path / "post_pass.db", embedding_dim=8)
    yield d
    d.close()


def _make_node(
    db: UnifiedWikiDB,
    node_id: str,
    *,
    rel_path: str,
    symbol_type: str = "function",
    symbol_name: str | None = None,
    is_architectural: int = 1,
) -> None:
    db.upsert_node(
        node_id,
        rel_path=rel_path,
        file_name=rel_path.rsplit("/", 1)[-1],
        language="python",
        symbol_name=symbol_name or node_id,
        symbol_type=symbol_type,
        is_architectural=is_architectural,
    )


# ═══════════════════════════════════════════════════════════════════════
# P6 — case-insensitive find_nodes_by_name (§11.9)
# ═══════════════════════════════════════════════════════════════════════


class TestFindNodesByNameCaseInsensitive:
    def test_exact_match_still_works(self, db):
        _make_node(db, "n1", rel_path="src/a.py", symbol_name="EliteAClient", symbol_type="class")
        db.conn.commit()

        rows = db.find_nodes_by_name("EliteAClient")
        assert len(rows) == 1
        assert rows[0]["symbol_name"] == "EliteAClient"

    def test_case_insensitive_lookup(self, db):
        _make_node(db, "n1", rel_path="src/a.py", symbol_name="EliteAClient", symbol_type="class")
        db.conn.commit()

        # Lower-case query must match
        rows = db.find_nodes_by_name("eliteaclient")
        assert len(rows) == 1
        assert rows[0]["symbol_name"] == "EliteAClient"

        # Mixed-case query must also match
        rows = db.find_nodes_by_name("ELITEACLIENT")
        assert len(rows) == 1


# ═══════════════════════════════════════════════════════════════════════
# B2 — compute_god_nodes (§11.7)
# ═══════════════════════════════════════════════════════════════════════


class TestComputeGodNodes:
    def test_empty_graph(self, db):
        result = db.compute_god_nodes(top_n=5)
        assert result == {"by_symbol_type": [], "by_file": []}

    def test_top_symbols_ranked_by_degree(self, db):
        # hub: 3 outgoing + 2 incoming = degree 5
        _make_node(db, "hub", rel_path="src/hub.py", symbol_type="class")
        _make_node(db, "leaf1", rel_path="src/a.py")
        _make_node(db, "leaf2", rel_path="src/b.py")
        _make_node(db, "leaf3", rel_path="src/c.py")
        _make_node(db, "caller1", rel_path="src/x.py")
        _make_node(db, "caller2", rel_path="src/y.py")
        db.upsert_edge("hub", "leaf1", "calls")
        db.upsert_edge("hub", "leaf2", "calls")
        db.upsert_edge("hub", "leaf3", "calls")
        db.upsert_edge("caller1", "hub", "calls")
        db.upsert_edge("caller2", "hub", "calls")
        db.conn.commit()

        result = db.compute_god_nodes(top_n=3)
        sym = result["by_symbol_type"]
        assert sym, "expected at least one ranked symbol"
        assert sym[0]["symbol_id"] == "hub"
        assert sym[0]["degree"] == 5

    def test_top_files_by_external_edges(self, db):
        _make_node(db, "a", rel_path="src/a.py")
        _make_node(db, "b", rel_path="src/b.py")
        _make_node(db, "c", rel_path="src/c.py")
        # a → b and a → c are both *external* (different rel_path)
        db.upsert_edge("a", "b", "calls")
        db.upsert_edge("a", "c", "calls")
        db.conn.commit()

        result = db.compute_god_nodes(top_n=5)
        files = result["by_file"]
        rel_paths = [f["rel_path"] for f in files]
        assert "src/a.py" in rel_paths
        a_row = next(f for f in files if f["rel_path"] == "src/a.py")
        assert a_row["external_edges"] >= 2


# ═══════════════════════════════════════════════════════════════════════
# P4 — demote_local_constants (§11.8)
# ═══════════════════════════════════════════════════════════════════════


class TestDemoteLocalConstants:
    def test_local_only_constant_is_demoted(self, db):
        _make_node(db, "C_LOCAL", rel_path="src/x.py", symbol_type="constant")
        _make_node(db, "user_x", rel_path="src/x.py")
        db.upsert_edge("user_x", "C_LOCAL", "uses")
        db.conn.commit()

        demoted = db.demote_local_constants()
        assert demoted == 1
        row = db.get_node("C_LOCAL")
        assert row["is_architectural"] == 0

    def test_cross_file_referenced_constant_is_kept(self, db):
        _make_node(db, "C_PUB", rel_path="src/x.py", symbol_type="constant")
        _make_node(db, "user_x", rel_path="src/x.py")
        _make_node(db, "user_y", rel_path="src/y.py")
        db.upsert_edge("user_x", "C_PUB", "uses")
        db.upsert_edge("user_y", "C_PUB", "uses")
        db.conn.commit()

        demoted = db.demote_local_constants()
        assert demoted == 0
        row = db.get_node("C_PUB")
        assert row["is_architectural"] == 1

    def test_unused_constant_is_kept(self, db):
        # Could be unused public API — must NOT be demoted blindly.
        _make_node(db, "C_UNUSED", rel_path="src/x.py", symbol_type="constant")
        db.conn.commit()

        demoted = db.demote_local_constants()
        assert demoted == 0
        row = db.get_node("C_UNUSED")
        assert row["is_architectural"] == 1

    def test_does_not_touch_non_constants(self, db):
        _make_node(db, "f_local", rel_path="src/x.py", symbol_type="function")
        _make_node(db, "user_x", rel_path="src/x.py")
        db.upsert_edge("user_x", "f_local", "calls")
        db.conn.commit()

        demoted = db.demote_local_constants()
        assert demoted == 0
        row = db.get_node("f_local")
        assert row["is_architectural"] == 1


# ═══════════════════════════════════════════════════════════════════════
# P7 — demote_vendored_code (§11.12)
# ═══════════════════════════════════════════════════════════════════════


class TestDemoteVendoredCode:
    def test_third_party_top_level_demoted(self, db):
        _make_node(db, "tp1", rel_path="third_party/x/lib.cc")
        _make_node(db, "tp2", rel_path="third-party/y/lib.cc")
        _make_node(db, "tp3", rel_path="vendor/z/lib.cc")
        _make_node(db, "tp4", rel_path="external/w/lib.cc")
        _make_node(db, "src", rel_path="src/main.cc")
        db.conn.commit()

        demoted = db.demote_vendored_code()
        assert demoted == 4
        for nid in ("tp1", "tp2", "tp3", "tp4"):
            assert db.get_node(nid)["is_architectural"] == 0
        # First-party untouched
        assert db.get_node("src")["is_architectural"] == 1

    def test_bundled_gtest_under_test_demoted(self, db):
        # The fmt repo's gtest lives at test/gtest/* — must be caught.
        _make_node(db, "g1", rel_path="test/gtest/gtest/gtest.h", symbol_type="macro")
        _make_node(db, "g2", rel_path="test/gtest/gmock-gtest-all.cc")
        _make_node(db, "g3", rel_path="test/gtest/gmock/gmock.h")
        _make_node(db, "real_test", rel_path="test/format-test.cc")
        db.conn.commit()

        demoted = db.demote_vendored_code()
        assert demoted == 3
        for nid in ("g1", "g2", "g3"):
            assert db.get_node(nid)["is_architectural"] == 0
        # First-party tests are NOT vendored
        assert db.get_node("real_test")["is_architectural"] == 1

    def test_googletest_segment_anywhere_demoted(self, db):
        _make_node(db, "gt", rel_path="some/nested/googletest/foo.h")
        _make_node(db, "gm", rel_path="some/nested/googlemock/foo.h")
        db.conn.commit()

        demoted = db.demote_vendored_code()
        assert demoted == 2

    def test_idempotent(self, db):
        _make_node(db, "tp", rel_path="vendor/x.cc")
        db.conn.commit()

        first = db.demote_vendored_code()
        second = db.demote_vendored_code()
        assert first == 1
        assert second == 0  # already demoted, nothing left to do


# ═══════════════════════════════════════════════════════════════════════
# §11.6 — edge confidence column (B1)
# ═══════════════════════════════════════════════════════════════════════


class TestEdgeConfidenceColumn:
    def test_column_exists_with_default(self, db):
        cols = {row[1] for row in db.conn.execute("PRAGMA table_info(repo_edges)")}
        assert "confidence" in cols

    def test_default_value_is_extracted(self, db):
        _make_node(db, "a", rel_path="src/a.py")
        _make_node(db, "b", rel_path="src/b.py")
        db.upsert_edge("a", "b", "calls")
        db.conn.commit()
        row = db.conn.execute(
            "SELECT confidence FROM repo_edges WHERE source_id='a'"
        ).fetchone()
        assert row[0] == "EXTRACTED"

    def test_inferred_kwarg_propagates(self, db):
        _make_node(db, "a", rel_path="src/a.py")
        _make_node(db, "b", rel_path="src/b.py")
        db.upsert_edge("a", "b", "member_uses", confidence="INFERRED", weight=3.0)
        db.conn.commit()
        row = db.conn.execute(
            "SELECT confidence, weight FROM repo_edges WHERE source_id='a'"
        ).fetchone()
        assert row[0] == "INFERRED"
        assert row[1] == 3.0

    def test_index_exists(self, db):
        idx = db.conn.execute(
            "SELECT name FROM sqlite_master WHERE type='index' AND name='idx_edges_confidence'"
        ).fetchone()
        assert idx is not None


# ═══════════════════════════════════════════════════════════════════════
# §11.4 — promote_class_members (P1)
# ═══════════════════════════════════════════════════════════════════════


class TestPromoteClassMembers:
    """Class C in src/c.py defines method M that calls T in src/t.py.
    Expect one synthetic member_uses edge (C, T) with confidence=INFERRED.
    """

    def _setup_class_with_member_call(self, db):
        _make_node(db, "C", rel_path="src/c.py", symbol_type="class")
        _make_node(db, "M", rel_path="src/c.py", symbol_type="method")
        _make_node(db, "T", rel_path="src/t.py", symbol_type="function")
        db.upsert_edge("C", "M", "defines")
        db.upsert_edge("M", "T", "calls")
        db.conn.commit()

    def test_basic_promotion(self, db):
        self._setup_class_with_member_call(db)
        n = db.promote_class_members()
        assert n == 1
        row = db.conn.execute(
            "SELECT rel_type, confidence, weight, created_by "
            "FROM repo_edges WHERE source_id='C' AND target_id='T'"
        ).fetchone()
        assert row["rel_type"] == "member_uses"
        assert row["confidence"] == "INFERRED"
        assert row["weight"] == 1.0
        assert row["created_by"] == "augmentation:11.4"

    def test_evidence_count_aggregated(self, db):
        # Class with two methods both calling the same target T.
        _make_node(db, "C", rel_path="src/c.py", symbol_type="class")
        _make_node(db, "M1", rel_path="src/c.py", symbol_type="method")
        _make_node(db, "M2", rel_path="src/c.py", symbol_type="method")
        _make_node(db, "T", rel_path="src/t.py", symbol_type="function")
        db.upsert_edge("C", "M1", "defines")
        db.upsert_edge("C", "M2", "defines")
        db.upsert_edge("M1", "T", "calls")
        db.upsert_edge("M2", "T", "calls")
        db.conn.commit()

        n = db.promote_class_members()
        assert n == 1
        weight = db.conn.execute(
            "SELECT weight FROM repo_edges WHERE source_id='C' AND target_id='T'"
        ).fetchone()[0]
        assert weight == 2.0

    def test_same_file_target_not_promoted(self, db):
        _make_node(db, "C", rel_path="src/c.py", symbol_type="class")
        _make_node(db, "M", rel_path="src/c.py", symbol_type="method")
        _make_node(db, "T_local", rel_path="src/c.py", symbol_type="function")
        db.upsert_edge("C", "M", "defines")
        db.upsert_edge("M", "T_local", "calls")
        db.conn.commit()

        n = db.promote_class_members()
        assert n == 0

    def test_self_loop_excluded(self, db):
        # Method calling its own class (recursion via class) — must not loop.
        _make_node(db, "C", rel_path="src/c.py", symbol_type="class")
        _make_node(db, "M", rel_path="src/c.py", symbol_type="method")
        db.upsert_edge("C", "M", "defines")
        db.upsert_edge("M", "C", "references")
        # Cross-file would be required — make M live in different file
        db.conn.execute(
            "UPDATE repo_nodes SET rel_path = 'src/m.py' WHERE node_id = 'M'"
        )
        db.conn.commit()

        n = db.promote_class_members()
        # Even with cross-file M, target is C itself → excluded by T.node_id != C.node_id
        assert n == 0

    def test_skip_when_existing_non_defines_edge(self, db):
        # If parser already linked C → T via 'inherits', do not double-count.
        _make_node(db, "C", rel_path="src/c.py", symbol_type="class")
        _make_node(db, "M", rel_path="src/c.py", symbol_type="method")
        _make_node(db, "T", rel_path="src/t.py", symbol_type="class")
        db.upsert_edge("C", "M", "defines")
        db.upsert_edge("C", "T", "inherits")
        db.upsert_edge("M", "T", "calls")
        db.conn.commit()

        n = db.promote_class_members()
        assert n == 0  # skipped — C already linked to T directly

    def test_existing_defines_does_not_block(self, db):
        # `defines` doesn't count as a "real" relationship; M is defined by
        # C trivially, so a member call from M back to a sibling defined
        # by C should still fire.
        _make_node(db, "C", rel_path="src/c.py", symbol_type="class")
        _make_node(db, "M", rel_path="src/c.py", symbol_type="method")
        _make_node(db, "T", rel_path="src/t.py", symbol_type="function")
        db.upsert_edge("C", "M", "defines")
        db.upsert_edge("M", "T", "calls")
        # Pre-existing 'defines' from C → T (rare but legal): must not block.
        db.upsert_edge("C", "T", "defines")
        db.conn.commit()

        n = db.promote_class_members()
        assert n == 1

    def test_only_class_like_sources(self, db):
        # A free function with cross-file calls must NOT trigger promotion.
        _make_node(db, "F", rel_path="src/f.py", symbol_type="function")
        _make_node(db, "M", rel_path="src/f.py", symbol_type="method")
        _make_node(db, "T", rel_path="src/t.py", symbol_type="function")
        db.upsert_edge("F", "M", "defines")
        db.upsert_edge("M", "T", "calls")
        db.conn.commit()

        n = db.promote_class_members()
        assert n == 0

    def test_field_member_also_counts(self, db):
        # Fields can reference cross-file types — that's a structural signal too.
        _make_node(db, "C", rel_path="src/c.py", symbol_type="class")
        _make_node(db, "fld", rel_path="src/c.py", symbol_type="field")
        _make_node(db, "T", rel_path="src/t.py", symbol_type="class")
        db.upsert_edge("C", "fld", "defines")
        db.upsert_edge("fld", "T", "references")
        db.conn.commit()

        n = db.promote_class_members()
        assert n == 1
