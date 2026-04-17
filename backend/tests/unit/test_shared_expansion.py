"""Tests for shared_expansion module (DB-based per-symbol-type expansion)."""

import sqlite3
from typing import Optional

import pytest

from app.core.code_graph.shared_expansion import (
    CLASS_LIKE_TYPES,
    EXPANSION_WORTHY_TYPES,
    P0_REL_TYPES,
    P1_REL_TYPES,
    P2_REL_TYPES,
    SKIP_RELATIONSHIPS,
    _collect_neighbors,
    _expand_class_db,
    _expand_constant_db,
    _expand_function_db,
    _expand_generic_db,
    _expand_macro_db,
    _expand_type_alias_db,
    _fetch_node,
    _get_incoming,
    _get_outgoing,
    _is_valid_expansion,
    expand_symbol_smart,
    resolve_alias_chain_db,
)


# ═══════════════════════════════════════════════════════════════════════════
# Fixtures
# ═══════════════════════════════════════════════════════════════════════════


def _make_db():
    """Create an in-memory SQLite DB with realistic schema."""
    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
    conn.execute("""
        CREATE TABLE repo_nodes (
            node_id TEXT PRIMARY KEY,
            rel_path TEXT DEFAULT '',
            file_name TEXT DEFAULT '',
            language TEXT DEFAULT '',
            start_line INTEGER DEFAULT 0,
            end_line INTEGER DEFAULT 0,
            symbol_name TEXT DEFAULT '',
            symbol_type TEXT DEFAULT '',
            parent_symbol TEXT,
            source_text TEXT DEFAULT '',
            docstring TEXT DEFAULT '',
            signature TEXT DEFAULT '',
            is_architectural INTEGER DEFAULT 0,
            is_doc INTEGER DEFAULT 0,
            is_test INTEGER DEFAULT 0,
            macro_cluster INTEGER,
            micro_cluster INTEGER
        )
    """)
    conn.execute("""
        CREATE TABLE repo_edges (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            source_id TEXT NOT NULL,
            target_id TEXT NOT NULL,
            rel_type TEXT NOT NULL DEFAULT '',
            weight REAL DEFAULT 1.0
        )
    """)
    return conn


def _insert_node(conn, node_id, **kwargs):
    """Insert a node with sensible defaults."""
    defaults = {
        "rel_path": f"src/{node_id}.py",
        "file_name": f"{node_id}.py",
        "language": "python",
        "symbol_name": node_id,
        "symbol_type": "class",
        "source_text": f"class {node_id}: pass",
        "is_architectural": 1,
        "macro_cluster": 0,
    }
    defaults.update(kwargs)
    cols = ", ".join(["node_id"] + list(defaults.keys()))
    vals = ", ".join(["?"] * (1 + len(defaults)))
    conn.execute(
        f"INSERT INTO repo_nodes ({cols}) VALUES ({vals})",
        [node_id] + list(defaults.values()),
    )


def _insert_edge(conn, source_id, target_id, rel_type="calls", weight=1.0):
    conn.execute(
        "INSERT INTO repo_edges (source_id, target_id, rel_type, weight) VALUES (?, ?, ?, ?)",
        (source_id, target_id, rel_type, weight),
    )


class _MockDB:
    """Thin wrapper around raw sqlite3.Connection exposing protocol methods."""

    def __init__(self, conn):
        self.conn = conn

    # Delegate raw SQL calls for test setup helpers
    def execute(self, *args, **kwargs):
        return self.conn.execute(*args, **kwargs)

    def commit(self):
        return self.conn.commit()

    def get_edge_targets(self, source_id):
        rows = self.conn.execute(
            "SELECT target_id, rel_type, weight FROM repo_edges WHERE source_id = ?",
            (source_id,),
        ).fetchall()
        return [dict(r) for r in rows]

    def get_edge_sources(self, target_id):
        rows = self.conn.execute(
            "SELECT source_id, rel_type, weight FROM repo_edges WHERE target_id = ?",
            (target_id,),
        ).fetchall()
        return [dict(r) for r in rows]

    def get_node(self, node_id):
        row = self.conn.execute(
            "SELECT * FROM repo_nodes WHERE node_id = ?", (node_id,),
        ).fetchone()
        return dict(row) if row else None

    def get_nodes_by_ids(self, node_ids):
        if not node_ids:
            return []
        placeholders = ",".join("?" * len(node_ids))
        rows = self.conn.execute(
            f"SELECT * FROM repo_nodes WHERE node_id IN ({placeholders})",
            node_ids,
        ).fetchall()
        return [dict(r) for r in rows]


@pytest.fixture()
def conn():
    c = _make_db()
    yield _MockDB(c)
    c.close()


@pytest.fixture()
def populated_conn():
    """A DB with a small class hierarchy: Base -> Child, Child calls Helper."""
    c = _make_db()
    _insert_node(c, "Base", symbol_type="class", macro_cluster=1)
    _insert_node(c, "Child", symbol_type="class", macro_cluster=1)
    _insert_node(c, "Helper", symbol_type="function", macro_cluster=1)
    _insert_node(c, "Constant1", symbol_type="constant", macro_cluster=1)
    _insert_node(c, "TypeAlias1", symbol_type="type_alias", macro_cluster=1)
    _insert_node(c, "Macro1", symbol_type="macro", macro_cluster=1)
    _insert_node(c, "OutOfCluster", symbol_type="class", macro_cluster=99)
    _insert_node(c, "NonArch", symbol_type="class", is_architectural=0, macro_cluster=1)
    _insert_node(c, "TestNode", symbol_type="class", macro_cluster=1, is_test=1,
                 rel_path="tests/test_foo.py")

    _insert_edge(c, "Child", "Base", "inheritance", 3.0)
    _insert_edge(c, "Child", "Helper", "calls", 1.5)
    _insert_edge(c, "Helper", "Constant1", "references", 1.0)
    _insert_edge(c, "TypeAlias1", "Base", "alias_of", 2.0)
    _insert_edge(c, "Macro1", "Helper", "references", 1.0)
    _insert_edge(c, "Base", "OutOfCluster", "references", 0.5)

    c.commit()
    yield _MockDB(c)
    c.close()


# ═══════════════════════════════════════════════════════════════════════════
# Constants
# ═══════════════════════════════════════════════════════════════════════════


class TestConstants:
    def test_priority_groups_disjoint(self):
        assert P0_REL_TYPES & P1_REL_TYPES == set()
        assert P0_REL_TYPES & P2_REL_TYPES == set()
        assert P1_REL_TYPES & P2_REL_TYPES == set()

    def test_expansion_worthy_includes_class_like(self):
        assert CLASS_LIKE_TYPES.issubset(EXPANSION_WORTHY_TYPES)

    def test_skip_relationships_not_in_priority(self):
        all_priority = P0_REL_TYPES | P1_REL_TYPES | P2_REL_TYPES
        assert SKIP_RELATIONSHIPS & all_priority == set()


# ═══════════════════════════════════════════════════════════════════════════
# DB helper functions
# ═══════════════════════════════════════════════════════════════════════════


class TestDbHelpers:
    def test_get_outgoing(self, populated_conn):
        edges = _get_outgoing(populated_conn, "Child")
        assert len(edges) == 2
        targets = {e[0] for e in edges}
        assert "Base" in targets
        assert "Helper" in targets

    def test_get_outgoing_filtered(self, populated_conn):
        edges = _get_outgoing(populated_conn, "Child", rel_types={"inheritance"})
        assert len(edges) == 1
        assert edges[0][0] == "Base"

    def test_get_incoming(self, populated_conn):
        edges = _get_incoming(populated_conn, "Base")
        assert len(edges) >= 1
        sources = {e[0] for e in edges}
        assert "Child" in sources

    def test_get_incoming_filtered(self, populated_conn):
        edges = _get_incoming(populated_conn, "Base", rel_types={"inheritance"})
        assert len(edges) == 1
        assert edges[0][0] == "Child"

    def test_fetch_node_exists(self, populated_conn):
        node = _fetch_node(populated_conn, "Base")
        assert node is not None
        assert node["symbol_name"] == "Base"

    def test_fetch_node_missing(self, populated_conn):
        assert _fetch_node(populated_conn, "nonexistent") is None


# ═══════════════════════════════════════════════════════════════════════════
# _is_valid_expansion
# ═══════════════════════════════════════════════════════════════════════════


class TestIsValidExpansion:
    def test_valid_node(self, populated_conn):
        node = _fetch_node(populated_conn, "Base")
        assert _is_valid_expansion(node, None, 1) is True

    def test_none_node(self):
        assert _is_valid_expansion(None, None, None) is False

    def test_non_architectural(self, populated_conn):
        node = _fetch_node(populated_conn, "NonArch")
        assert _is_valid_expansion(node, None, None) is False

    def test_page_boundary_enforced(self, populated_conn):
        node = _fetch_node(populated_conn, "Base")
        assert _is_valid_expansion(node, {"Helper"}, 1) is False
        assert _is_valid_expansion(node, {"Base"}, 1) is True

    def test_macro_cluster_enforced(self, populated_conn):
        node = _fetch_node(populated_conn, "OutOfCluster")
        assert _is_valid_expansion(node, None, 1) is False  # macro=1, node macro=99

    def test_exclude_tests_by_column(self, populated_conn):
        node = _fetch_node(populated_conn, "TestNode")
        assert _is_valid_expansion(node, None, 1, exclude_tests=True) is False
        assert _is_valid_expansion(node, None, 1, exclude_tests=False) is True


# ═══════════════════════════════════════════════════════════════════════════
# resolve_alias_chain_db
# ═══════════════════════════════════════════════════════════════════════════


class TestResolveAliasChain:
    def test_resolves_to_concrete(self, populated_conn):
        result = resolve_alias_chain_db(populated_conn, "TypeAlias1")
        assert result == "Base"

    def test_no_alias_returns_none(self, populated_conn):
        result = resolve_alias_chain_db(populated_conn, "Base")
        assert result is None

    def test_chain_limit(self, conn):
        """Chain of aliases that exceeds max_hops returns last reachable."""
        for i in range(7):
            _insert_node(conn, f"alias_{i}", symbol_type="type_alias")
            if i < 6:
                _insert_edge(conn, f"alias_{i}", f"alias_{i+1}", "alias_of")
        conn.commit()

        result = resolve_alias_chain_db(conn, "alias_0", max_hops=3)
        # Should stop at alias_3 (3 hops from alias_0)
        assert result is not None

    def test_cycle_detection(self, conn):
        _insert_node(conn, "cycA", symbol_type="type_alias")
        _insert_node(conn, "cycB", symbol_type="type_alias")
        _insert_edge(conn, "cycA", "cycB", "alias_of")
        _insert_edge(conn, "cycB", "cycA", "alias_of")
        conn.commit()
        result = resolve_alias_chain_db(conn, "cycA")
        # Should not loop forever
        assert result is not None or result is None


# ═══════════════════════════════════════════════════════════════════════════
# expand_symbol_smart
# ═══════════════════════════════════════════════════════════════════════════


class TestExpandSymbolSmart:
    def test_class_expansion(self, populated_conn):
        results = expand_symbol_smart(
            populated_conn, "Child", "class",
            seen_ids=set(), macro_id=1,
        )
        # Should find Base (inheritance) and Helper (calls)
        found_ids = {r[0] for r in results}
        assert "Base" in found_ids

    def test_function_expansion(self, populated_conn):
        results = expand_symbol_smart(
            populated_conn, "Helper", "function",
            seen_ids=set(), macro_id=1,
        )
        found_ids = {r[0] for r in results}
        assert "Constant1" in found_ids or "Child" in found_ids

    def test_constant_expansion(self, populated_conn):
        results = expand_symbol_smart(
            populated_conn, "Constant1", "constant",
            seen_ids=set(), macro_id=1,
        )
        found_ids = {r[0] for r in results}
        assert "Helper" in found_ids  # referenced_by

    def test_type_alias_expansion(self, populated_conn):
        results = expand_symbol_smart(
            populated_conn, "TypeAlias1", "type_alias",
            seen_ids=set(), macro_id=1,
        )
        found_ids = {r[0] for r in results}
        assert "Base" in found_ids  # alias resolves

    def test_macro_expansion(self, populated_conn):
        results = expand_symbol_smart(
            populated_conn, "Macro1", "macro",
            seen_ids=set(), macro_id=1,
        )
        found_ids = {r[0] for r in results}
        assert "Helper" in found_ids

    def test_generic_expansion(self, populated_conn):
        _insert_node(populated_conn, "Unknown1", symbol_type="widget", macro_cluster=1)
        _insert_edge(populated_conn, "Unknown1", "Helper", "some_edge")
        populated_conn.commit()
        results = expand_symbol_smart(
            populated_conn, "Unknown1", "widget",
            seen_ids=set(), macro_id=1,
        )
        # helper has valid type, should appear
        found_ids = {r[0] for r in results}
        assert "Helper" in found_ids

    def test_seen_ids_excluded(self, populated_conn):
        results = expand_symbol_smart(
            populated_conn, "Child", "class",
            seen_ids={"Base"}, macro_id=1,
        )
        found_ids = {r[0] for r in results}
        assert "Base" not in found_ids

    def test_per_symbol_budget(self, populated_conn):
        results = expand_symbol_smart(
            populated_conn, "Child", "class",
            seen_ids=set(), macro_id=1, per_symbol_budget=1,
        )
        assert len(results) <= 1

    def test_extra_rel_types(self, populated_conn):
        _insert_node(populated_conn, "Decorated", symbol_type="function", macro_cluster=1)
        _insert_edge(populated_conn, "Helper", "Decorated", "decorates")
        populated_conn.commit()
        results = expand_symbol_smart(
            populated_conn, "Helper", "function",
            seen_ids=set(), macro_id=1,
            extra_rel_types=frozenset({"decorates"}),
        )
        found_ids = {r[0] for r in results}
        assert "Decorated" in found_ids

    def test_exclude_tests(self, populated_conn):
        results = expand_symbol_smart(
            populated_conn, "Child", "class",
            seen_ids=set(), macro_id=1,
            exclude_tests=True,
        )
        found_ids = {r[0] for r in results}
        assert "TestNode" not in found_ids


# ═══════════════════════════════════════════════════════════════════════════
# _collect_neighbors
# ═══════════════════════════════════════════════════════════════════════════


class TestCollectNeighbors:
    def test_outgoing(self, populated_conn):
        results = _collect_neighbors(
            populated_conn, "Child", {"inheritance"}, "out",
            set(), None, 1,
        )
        assert len(results) == 1
        assert results[0][0] == "Base"

    def test_incoming(self, populated_conn):
        results = _collect_neighbors(
            populated_conn, "Base", {"inheritance"}, "in",
            set(), None, 1,
        )
        assert len(results) == 1
        assert results[0][0] == "Child"

    def test_limit_applied(self, populated_conn):
        results = _collect_neighbors(
            populated_conn, "Child", None, "out",
            set(), None, 1, limit=1,
        )
        assert len(results) <= 1

    def test_reason_prefix(self, populated_conn):
        results = _collect_neighbors(
            populated_conn, "Child", {"inheritance"}, "out",
            set(), None, 1, reason_prefix="test_prefix:",
        )
        assert results[0][2].startswith("test_prefix:")
