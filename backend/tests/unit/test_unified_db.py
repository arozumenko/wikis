"""Tests for UnifiedWikiDB — is_test column, migration, backfill, stale cleanup."""

import sqlite3
import tempfile
from pathlib import Path

import networkx as nx
import pytest

from app.core.code_graph.unified_graph_text_index import UnifiedGraphTextIndex
from app.core.unified_db import UnifiedWikiDB


# ═══════════════════════════════════════════════════════════════════════════
# Fixtures
# ═══════════════════════════════════════════════════════════════════════════


@pytest.fixture()
def db_path(tmp_path):
    return tmp_path / "test.wiki.db"


@pytest.fixture()
def db(db_path):
    d = UnifiedWikiDB(db_path, embedding_dim=8)
    yield d
    d.close()


def _make_graph(*nodes):
    """Build a small MultiDiGraph from (node_id, rel_path, symbol_type) tuples."""
    G = nx.MultiDiGraph()
    for nid, rel_path, stype in nodes:
        G.add_node(
            nid,
            rel_path=rel_path,
            file_name=Path(rel_path).name,
            language="python",
            symbol_name=nid,
            symbol_type=stype,
            source_text=f"def {nid}(): pass",
        )
    return G


# ═══════════════════════════════════════════════════════════════════════════
# Step 2: is_test column exists in schema
# ═══════════════════════════════════════════════════════════════════════════


class TestIsTestColumn:
    def test_column_exists(self, db):
        cols = {
            row[1] for row in db.conn.execute("PRAGMA table_info(repo_nodes)")
        }
        assert "is_test" in cols

    def test_insert_and_retrieve(self, db):
        db.upsert_node("n1", rel_path="src/main.py", is_test=0)
        db.upsert_node("n2", rel_path="tests/test.py", is_test=1)
        db.conn.commit()

        n1 = db.get_node("n1")
        n2 = db.get_node("n2")
        assert n1["is_test"] == 0
        assert n2["is_test"] == 1

    def test_partial_index_exists(self, db):
        indexes = db.conn.execute(
            "SELECT name FROM sqlite_master WHERE type='index' AND name='idx_nodes_test'"
        ).fetchone()
        assert indexes is not None

    def test_default_is_zero(self, db):
        db.upsert_node("n1", rel_path="src/main.py")
        db.conn.commit()
        node = db.get_node("n1")
        assert node["is_test"] == 0


# ═══════════════════════════════════════════════════════════════════════════
# Step 3: is_test_path() wired into _upsert_nodes_batch via from_networkx
# ═══════════════════════════════════════════════════════════════════════════


class TestIsTestFromNetworkx:
    def test_prod_node_is_test_0(self, db):
        G = _make_graph(("main_fn", "src/main.py", "function"))
        db.from_networkx(G)

        node = db.get_node("main_fn")
        assert node["is_test"] == 0

    def test_test_node_is_test_1(self, db):
        G = _make_graph(("test_fn", "tests/test_main.py", "function"))
        db.from_networkx(G)

        node = db.get_node("test_fn")
        assert node["is_test"] == 1

    def test_jest_test_is_test_1(self, db):
        G = _make_graph(("comp", "__tests__/Component.test.jsx", "function"))
        db.from_networkx(G)

        node = db.get_node("comp")
        assert node["is_test"] == 1

    def test_fixtures_is_test_1(self, db):
        G = _make_graph(("fix", "testdata/sample.json", "constant"))
        db.from_networkx(G)

        node = db.get_node("fix")
        assert node["is_test"] == 1

    def test_mixed_graph(self, db):
        G = _make_graph(
            ("prod", "src/auth.py", "class"),
            ("test", "tests/test_auth.py", "function"),
            ("mock", "mocks/auth_mock.py", "function"),
        )
        db.from_networkx(G)

        assert db.get_node("prod")["is_test"] == 0
        assert db.get_node("test")["is_test"] == 1
        assert db.get_node("mock")["is_test"] == 1


# ═══════════════════════════════════════════════════════════════════════════
# Step 4: Migration + backfill on pre-existing DBs
# ═══════════════════════════════════════════════════════════════════════════


class TestMigrateSchema:
    def _make_legacy_db(self, db_path):
        """Create a DB without the is_test column (simulating legacy schema)."""
        conn = sqlite3.connect(str(db_path))
        conn.execute("""
            CREATE TABLE repo_nodes (
                node_id TEXT PRIMARY KEY,
                rel_path TEXT NOT NULL DEFAULT '',
                file_name TEXT NOT NULL DEFAULT '',
                language TEXT NOT NULL DEFAULT '',
                start_line INTEGER DEFAULT 0,
                end_line INTEGER DEFAULT 0,
                symbol_name TEXT NOT NULL DEFAULT '',
                symbol_type TEXT NOT NULL DEFAULT '',
                parent_symbol TEXT DEFAULT NULL,
                analysis_level TEXT DEFAULT 'comprehensive',
                source_text TEXT DEFAULT '',
                docstring TEXT DEFAULT '',
                signature TEXT DEFAULT '',
                parameters TEXT DEFAULT '',
                return_type TEXT DEFAULT '',
                is_architectural INTEGER DEFAULT 0,
                is_doc INTEGER DEFAULT 0,
                chunk_type TEXT DEFAULT NULL,
                macro_cluster INTEGER DEFAULT NULL,
                micro_cluster INTEGER DEFAULT NULL,
                is_hub INTEGER DEFAULT 0,
                hub_assignment TEXT DEFAULT NULL,
                indexed_at TEXT DEFAULT (datetime('now'))
            )
        """)
        # Insert some rows
        conn.execute(
            "INSERT INTO repo_nodes (node_id, rel_path, symbol_name, symbol_type) "
            "VALUES ('prod1', 'src/main.py', 'main', 'function')"
        )
        conn.execute(
            "INSERT INTO repo_nodes (node_id, rel_path, symbol_name, symbol_type) "
            "VALUES ('test1', 'tests/test_main.py', 'test_main', 'function')"
        )
        conn.execute(
            "INSERT INTO repo_nodes (node_id, rel_path, symbol_name, symbol_type) "
            "VALUES ('test2', 'mocks/auth_mock.py', 'mock_auth', 'function')"
        )
        # Need edges and FTS tables too for UnifiedWikiDB to open without error
        conn.execute("""
            CREATE TABLE IF NOT EXISTS repo_edges (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                source_id TEXT, target_id TEXT, rel_type TEXT DEFAULT '',
                edge_class TEXT DEFAULT 'structural', analysis_level TEXT DEFAULT 'comprehensive',
                weight REAL DEFAULT 1.0, raw_similarity REAL,
                source_file TEXT DEFAULT '', target_file TEXT DEFAULT '',
                language TEXT DEFAULT '', annotations TEXT DEFAULT '',
                created_by TEXT DEFAULT 'ast'
            )
        """)
        conn.execute("""
            CREATE VIRTUAL TABLE IF NOT EXISTS repo_fts USING fts5(
                node_id UNINDEXED, symbol_name, signature, docstring, source_text,
                tokenize='porter unicode61'
            )
        """)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS wiki_meta (key TEXT PRIMARY KEY, value TEXT)
        """)
        conn.commit()
        conn.close()

    def test_migration_adds_column(self, tmp_path):
        db_path = tmp_path / "legacy.wiki.db"
        self._make_legacy_db(db_path)

        # Open with UnifiedWikiDB — should trigger migration
        db = UnifiedWikiDB(db_path, embedding_dim=8)
        cols = {row[1] for row in db.conn.execute("PRAGMA table_info(repo_nodes)")}
        assert "is_test" in cols
        db.close()

    def test_backfill_sets_is_test(self, tmp_path):
        db_path = tmp_path / "legacy.wiki.db"
        self._make_legacy_db(db_path)

        db = UnifiedWikiDB(db_path, embedding_dim=8)
        prod = db.get_node("prod1")
        test1 = db.get_node("test1")
        test2 = db.get_node("test2")

        assert prod["is_test"] == 0
        assert test1["is_test"] == 1
        assert test2["is_test"] == 1
        db.close()

    def test_fresh_db_no_error(self, tmp_path):
        """A brand new DB should not fail migration (column already exists)."""
        db_path = tmp_path / "fresh.wiki.db"
        db = UnifiedWikiDB(db_path, embedding_dim=8)
        cols = {row[1] for row in db.conn.execute("PRAGMA table_info(repo_nodes)")}
        assert "is_test" in cols
        db.close()


# ═══════════════════════════════════════════════════════════════════════════
# Step 4b: Stale-node cleanup in from_networkx
# ═══════════════════════════════════════════════════════════════════════════


class TestStaleNodeCleanup:
    def test_stale_nodes_removed(self, db):
        G1 = _make_graph(
            ("a", "src/a.py", "function"),
            ("b", "src/b.py", "function"),
            ("c", "src/c.py", "function"),
        )
        G1.add_edge("a", "b", relationship_type="calls")
        db.from_networkx(G1)
        assert db.node_count() == 3

        G2 = _make_graph(
            ("b", "src/b.py", "function"),
            ("c", "src/c.py", "function"),
            ("d", "src/d.py", "function"),
        )
        db.from_networkx(G2)

        assert db.node_count() == 3
        assert db.get_node("a") is None
        assert db.get_node("b") is not None
        assert db.get_node("d") is not None


class TestUnifiedGraphTextIndex:
    def test_search_by_name_uses_unified_db(self, db, db_path):
        graph = nx.MultiDiGraph()
        graph.add_node(
            "auth-service",
            rel_path="src/auth/service.py",
            file_name="service.py",
            language="python",
            symbol_name="AuthService",
            symbol_type="class",
            source_text="class AuthService:\n    \"\"\"Authentication entry point.\"\"\"\n",
            docstring="Authentication entry point.",
        )
        db.from_networkx(graph)

        index = UnifiedGraphTextIndex(db_path)
        docs = index.search_by_name("AuthService", exact=True, k=5)

        assert len(docs) == 1
        assert docs[0].metadata["node_id"] == "auth-service"
        assert docs[0].metadata["rel_path"] == "src/auth/service.py"

    def test_search_symbols_filters_by_path_prefix(self, db, db_path):
        graph = nx.MultiDiGraph()
        graph.add_node(
            "auth-service",
            rel_path="src/auth/service.py",
            file_name="service.py",
            language="python",
            symbol_name="AuthService",
            symbol_type="class",
            source_text="Authentication service handles login tokens.",
            docstring="Authentication service.",
        )
        graph.add_node(
            "auth-doc",
            rel_path="docs/authentication.md",
            file_name="authentication.md",
            language="markdown",
            symbol_name="AuthenticationGuide",
            symbol_type="documentation",
            source_text="Authentication service overview for users.",
            docstring="",
        )
        db.from_networkx(graph)

        index = UnifiedGraphTextIndex(db_path)
        docs = index.search_symbols("authentication service", k=10, path_prefix="src")

        assert [doc.metadata["node_id"] for doc in docs] == ["auth-service"]

    def test_search_by_layer_recomputes_layer_from_repo_nodes(self, db, db_path):
        graph = nx.MultiDiGraph()
        graph.add_node(
            "main",
            rel_path="src/main.py",
            file_name="main.py",
            language="python",
            symbol_name="main",
            symbol_type="function",
            source_text="def main():\n    return 0\n",
            docstring="Application entry point.",
        )
        db.from_networkx(graph)

        index = UnifiedGraphTextIndex(db_path)
        docs = index.search_by_layer("entry_point", k=5)

        assert len(docs) == 1
        assert docs[0].metadata["node_id"] == "main"
        assert docs[0].metadata["layer"] == "entry_point"

    def test_stale_edges_removed(self, db):
        G1 = _make_graph(
            ("a", "src/a.py", "function"),
            ("b", "src/b.py", "function"),
        )
        G1.add_edge("a", "b", relationship_type="calls")
        db.from_networkx(G1)
        assert db.edge_count() == 1

        G2 = _make_graph(("b", "src/b.py", "function"))
        db.from_networkx(G2)

        # Edge referencing stale node 'a' should be gone
        assert db.edge_count() == 0

    def test_no_stale_is_noop(self, db):
        G = _make_graph(("a", "src/a.py", "function"))
        db.from_networkx(G)
        initial_count = db.node_count()

        db.from_networkx(G)  # re-import same graph
        assert db.node_count() == initial_count
