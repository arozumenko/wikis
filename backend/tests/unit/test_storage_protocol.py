"""
Unit tests for the storage abstraction layer.

Tests:
- WikiStorageProtocol — structural conformance for SQLite backend
- create_storage() factory — dispatching, validation, keyword args
- UnifiedWikiDB protocol methods — the 10 new methods added for
  cluster_expansion support
"""

import sqlite3
import tempfile
from pathlib import Path

import networkx as nx
import pytest

from app.core.storage import WikiStorageProtocol, create_storage
from app.core.storage.factory import create_storage as factory_fn
from app.core.unified_db import UnifiedWikiDB


# ═══════════════════════════════════════════════════════════════════════════
# Fixtures
# ═══════════════════════════════════════════════════════════════════════════


@pytest.fixture()
def tmp_db_path(tmp_path):
    return str(tmp_path / "test.wiki.db")


@pytest.fixture()
def db(tmp_db_path):
    """Fresh UnifiedWikiDB (read-write) with a small graph baked in."""
    d = UnifiedWikiDB(tmp_db_path, embedding_dim=8)

    G = nx.MultiDiGraph()
    G.add_node("ClassA", **{
        "symbol_name": "ClassA", "symbol_type": "class",
        "rel_path": "src/auth.py", "start_line": 1, "end_line": 30,
        "language": "python", "source_text": "class ClassA: pass",
    })
    G.add_node("func_b", **{
        "symbol_name": "func_b", "symbol_type": "function",
        "rel_path": "src/auth.py", "start_line": 32, "end_line": 45,
        "language": "python", "source_text": "def func_b(): ...",
    })
    G.add_node("README", **{
        "symbol_name": "README", "symbol_type": "module_doc",
        "rel_path": "README.md", "start_line": 0, "end_line": 10,
        "language": "markdown", "source_text": "# Readme",
    })
    G.add_node("TestX", **{
        "symbol_name": "TestX", "symbol_type": "class",
        "rel_path": "tests/test_x.py", "start_line": 1, "end_line": 10,
        "language": "python", "source_text": "class TestX: pass",
    })
    G.add_node("OuterClass", **{
        "symbol_name": "OuterClass", "symbol_type": "class",
        "rel_path": "src/outer.py", "start_line": 1, "end_line": 5,
        "language": "python", "source_text": "class OuterClass: pass",
    })
    # Edges use `relationship_type` (what _nx_edge_to_dict expects)
    G.add_edge("ClassA", "func_b", key=0, relationship_type="calls", weight=2.0)
    G.add_edge("func_b", "ClassA", key=0, relationship_type="references", weight=1.0)
    G.add_edge("ClassA", "OuterClass", key=0, relationship_type="imports", weight=0.5)

    d.from_networkx(G)

    # Assign clusters after import (mirrors real pipeline: graph → cluster → query)
    d.set_clusters_batch([
        ("ClassA", 1, 0),
        ("func_b", 1, 0),
        ("README", 1, 0),
        ("TestX", 1, 0),
        ("OuterClass", 99, 0),
    ])
    d.conn.commit()

    yield d
    d.close()


# ═══════════════════════════════════════════════════════════════════════════
# Protocol structural conformance
# ═══════════════════════════════════════════════════════════════════════════


class TestProtocolConformance:
    """UnifiedWikiDB must be a structural subtype of WikiStorageProtocol."""

    def test_isinstance_check(self, db):
        assert isinstance(db, WikiStorageProtocol)

    def test_has_required_properties(self, db):
        assert hasattr(db, "db_path")
        assert hasattr(db, "vec_available")
        assert hasattr(db, "embedding_dim")

    def test_has_lifecycle_methods(self, db):
        assert callable(getattr(db, "close", None))
        assert callable(getattr(db, "__enter__", None))
        assert callable(getattr(db, "__exit__", None))

    def test_has_node_methods(self, db):
        for m in ("upsert_node", "upsert_nodes_batch", "get_node",
                   "get_nodes_by_ids", "get_all_nodes", "node_count"):
            assert callable(getattr(db, m, None)), f"missing {m}"

    def test_has_edge_methods(self, db):
        for m in ("upsert_edge", "upsert_edges_batch", "get_edges_from",
                   "get_edges_to", "edge_count"):
            assert callable(getattr(db, m, None)), f"missing {m}"

    def test_has_search_methods(self, db):
        for m in ("search_fts", "search_vec", "search_hybrid"):
            assert callable(getattr(db, m, None)), f"missing {m}"

    def test_has_cluster_methods(self, db):
        for m in ("set_cluster", "get_cluster_nodes", "get_all_clusters"):
            assert callable(getattr(db, m, None)), f"missing {m}"

    def test_has_graph_io(self, db):
        for m in ("from_networkx", "to_networkx"):
            assert callable(getattr(db, m, None)), f"missing {m}"

    def test_has_cluster_expansion_methods(self, db):
        for m in ("find_nodes_by_name", "search_fts_by_symbol_name",
                   "get_edge_targets", "get_edge_sources", "get_nodes_by_ids",
                   "find_related_nodes", "get_doc_nodes_by_cluster", "commit"):
            assert callable(getattr(db, m, None)), f"missing {m}"


# ═══════════════════════════════════════════════════════════════════════════
# Factory
# ═══════════════════════════════════════════════════════════════════════════


class TestFactory:
    def test_creates_sqlite(self, tmp_db_path):
        db = create_storage(backend="sqlite", db_path=tmp_db_path)
        assert isinstance(db, UnifiedWikiDB)
        db.close()

    def test_sqlite_requires_db_path(self):
        with pytest.raises(ValueError, match="db_path"):
            create_storage(backend="sqlite")

    def test_postgres_requires_dsn(self):
        with pytest.raises(ValueError, match="dsn"):
            create_storage(backend="postgres", repo_id="abc")

    def test_postgres_requires_repo_id(self):
        with pytest.raises(ValueError, match="repo_id"):
            create_storage(backend="postgres", dsn="postgresql://x")

    def test_unknown_backend(self):
        with pytest.raises(ValueError, match="Unknown storage backend"):
            create_storage(backend="mysql", db_path="/tmp/x.db")

    def test_case_insensitive(self, tmp_db_path):
        db = create_storage(backend="  SQLite  ", db_path=tmp_db_path)
        assert isinstance(db, UnifiedWikiDB)
        db.close()


# ═══════════════════════════════════════════════════════════════════════════
# UnifiedWikiDB protocol methods — find_nodes_by_name
# ═══════════════════════════════════════════════════════════════════════════


class TestFindNodesByName:
    def test_exact_match(self, db):
        rows = db.find_nodes_by_name("ClassA")
        assert len(rows) == 1
        assert rows[0]["symbol_name"] == "ClassA"

    def test_no_match(self, db):
        rows = db.find_nodes_by_name("DoesNotExist")
        assert rows == []

    def test_cluster_scoped(self, db):
        rows = db.find_nodes_by_name("OuterClass", macro_cluster=1)
        assert rows == []  # OuterClass is in cluster 99

    def test_cluster_match(self, db):
        rows = db.find_nodes_by_name("OuterClass", macro_cluster=99)
        assert len(rows) == 1

    def test_architectural_only(self, db):
        # TestX has symbol_type="class" which IS architectural,
        # but it's in tests/ path so is_test=1.
        # Use a non-architectural type to test this filter.
        rows = db.find_nodes_by_name("README", architectural_only=True)
        # README is module_doc — considered architectural by _is_doc_symbol
        assert len(rows) >= 1

    def test_not_architectural_only(self, db):
        rows = db.find_nodes_by_name("TestX", architectural_only=False)
        assert len(rows) == 1


# ═══════════════════════════════════════════════════════════════════════════
# search_fts_by_symbol_name
# ═══════════════════════════════════════════════════════════════════════════


class TestSearchFtsBySymbolName:
    def test_finds_symbol(self, db):
        rows = db.search_fts_by_symbol_name("ClassA")
        assert len(rows) >= 1
        assert any(r["symbol_name"] == "ClassA" for r in rows)

    def test_not_found(self, db):
        rows = db.search_fts_by_symbol_name("TotallyNonexistent")
        assert rows == []

    def test_cluster_bounded(self, db):
        rows = db.search_fts_by_symbol_name("OuterClass", macro_cluster=1)
        assert rows == []


# ═══════════════════════════════════════════════════════════════════════════
# get_edge_targets / get_edge_sources
# ═══════════════════════════════════════════════════════════════════════════


class TestEdgeQueries:
    def test_get_edge_targets(self, db):
        targets = db.get_edge_targets("ClassA")
        ids = {t["target_id"] for t in targets}
        assert "func_b" in ids

    def test_get_edge_sources(self, db):
        sources = db.get_edge_sources("ClassA")
        ids = {s["source_id"] for s in sources}
        assert "func_b" in ids  # func_b -> ClassA references edge

    def test_no_targets(self, db):
        # README has no outgoing edges
        assert db.get_edge_targets("README") == []

    def test_no_sources(self, db):
        # OuterClass has no incoming edges from graph construction
        # (ClassA -> OuterClass exists though)
        sources = db.get_edge_sources("OuterClass")
        ids = {s["source_id"] for s in sources}
        assert "ClassA" in ids


# ═══════════════════════════════════════════════════════════════════════════
# get_nodes_by_ids
# ═══════════════════════════════════════════════════════════════════════════


class TestGetNodesByIds:
    def test_batch_fetch(self, db):
        rows = db.get_nodes_by_ids(["ClassA", "func_b"])
        names = {r["symbol_name"] for r in rows}
        assert names == {"ClassA", "func_b"}

    def test_empty_list(self, db):
        assert db.get_nodes_by_ids([]) == []

    def test_unknown_ids(self, db):
        rows = db.get_nodes_by_ids(["no_such_id"])
        assert rows == []


# ═══════════════════════════════════════════════════════════════════════════
# find_related_nodes
# ═══════════════════════════════════════════════════════════════════════════


class TestFindRelatedNodes:
    def test_outgoing(self, db):
        nodes = db.find_related_nodes("ClassA", ["calls"], direction="out")
        names = {n["symbol_name"] for n in nodes}
        assert "func_b" in names

    def test_incoming(self, db):
        nodes = db.find_related_nodes("ClassA", ["references"], direction="in")
        names = {n["symbol_name"] for n in nodes}
        assert "func_b" in names

    def test_target_symbol_types(self, db):
        nodes = db.find_related_nodes("ClassA", ["calls"], direction="out",
                                       target_symbol_types=["class"])
        # func_b is a function, not a class
        assert nodes == []

    def test_exclude_path(self, db):
        nodes = db.find_related_nodes("ClassA", ["imports"], direction="out",
                                       exclude_path="src/outer.py")
        assert nodes == []

    def test_nonexistent_node(self, db):
        assert db.find_related_nodes("nope", ["calls"]) == []


# ═══════════════════════════════════════════════════════════════════════════
# get_doc_nodes_by_cluster
# ═══════════════════════════════════════════════════════════════════════════


class TestGetDocNodesByCluster:
    def test_finds_docs(self, db):
        docs = db.get_doc_nodes_by_cluster(1)
        assert len(docs) >= 1
        assert any(d["symbol_name"] == "README" for d in docs)

    def test_empty_cluster(self, db):
        docs = db.get_doc_nodes_by_cluster(999)
        assert docs == []

    def test_micro_filter(self, db):
        docs = db.get_doc_nodes_by_cluster(1, micro=0)
        assert len(docs) >= 1


# ═══════════════════════════════════════════════════════════════════════════
# get_all_nodes
# ═══════════════════════════════════════════════════════════════════════════


class TestGetAllNodes:
    def test_returns_all(self, db):
        nodes = db.get_all_nodes()
        assert len(nodes) == 5  # ClassA, func_b, README, TestX, OuterClass


# ═══════════════════════════════════════════════════════════════════════════
# commit
# ═══════════════════════════════════════════════════════════════════════════


class TestCommit:
    def test_commit_does_not_error(self, db):
        db.commit()  # Should be a no-op or flush — must not raise
