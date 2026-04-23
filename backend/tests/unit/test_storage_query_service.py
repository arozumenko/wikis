"""
Tests for :class:`StorageQueryService` — the storage-native replacement for
``GraphQueryService`` used by the Ask and Deep-Research engines.

These tests verify the core public surface (resolve_symbol, search,
get_relationships, resolve_and_traverse, query) against a small in-memory
UnifiedWikiDB fixture.  The same fixture is mirrored into a NetworkX graph
so we can spot-check parity with :class:`GraphQueryService` for the
resolver cascade and relationship traversal.
"""

from __future__ import annotations

import networkx as nx
import pytest

from app.core.code_graph.graph_query_service import GraphQueryService
from app.core.code_graph.storage_query_service import StorageQueryService
from app.core.storage.text_index import StorageTextIndex
from app.core.unified_db import UnifiedWikiDB


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _node_rows() -> list[dict]:
    return [
        {
            "node_id": "auth::AuthService",
            "rel_path": "src/auth/service.py",
            "file_name": "service.py",
            "language": "python",
            "start_line": 10,
            "end_line": 80,
            "symbol_name": "AuthService",
            "symbol_type": "class",
            "source_text": "class AuthService:\n    def login(self): ...",
            "docstring": "Handles authentication",
            "signature": "class AuthService",
            "is_architectural": 1,
            "is_doc": 0,
        },
        {
            "node_id": "auth::TokenManager",
            "rel_path": "src/auth/tokens.py",
            "file_name": "tokens.py",
            "language": "python",
            "start_line": 5,
            "end_line": 45,
            "symbol_name": "TokenManager",
            "symbol_type": "class",
            "source_text": "class TokenManager:\n    def create_token(self): ...",
            "docstring": "Manages JWT tokens",
            "signature": "class TokenManager",
            "is_architectural": 1,
            "is_doc": 0,
        },
        {
            "node_id": "auth::login",
            "rel_path": "src/auth/service.py",
            "file_name": "service.py",
            "language": "python",
            "start_line": 15,
            "end_line": 30,
            "symbol_name": "login",
            "symbol_type": "method",
            "source_text": "def login(self, username, password): ...",
            "docstring": "Login a user",
            "signature": "def login(self, username, password)",
            "is_architectural": 0,
            "is_doc": 0,
            "parent_symbol": "AuthService",
        },
        {
            "node_id": "db::DatabasePool",
            "rel_path": "src/db/pool.py",
            "file_name": "pool.py",
            "language": "python",
            "start_line": 1,
            "end_line": 60,
            "symbol_name": "DatabasePool",
            "symbol_type": "class",
            "source_text": "class DatabasePool:\n    def get_connection(self): ...",
            "docstring": "Connection pool for databases",
            "signature": "class DatabasePool",
            "is_architectural": 1,
            "is_doc": 0,
        },
    ]


def _edges() -> list[tuple[str, str, str]]:
    return [
        ("auth::AuthService", "auth::TokenManager", "calls"),
        ("auth::AuthService", "auth::login", "defines"),
        ("auth::TokenManager", "db::DatabasePool", "calls"),
    ]


@pytest.fixture
def storage(tmp_path):
    db = UnifiedWikiDB(str(tmp_path / "test.wiki.db"), embedding_dim=4)
    for n in _node_rows():
        db._upsert_nodes_batch([n])
    for src, tgt, rel in _edges():
        db.upsert_edge(src, tgt, rel)
    db.conn.commit()
    db._populate_fts5()
    yield db
    db.close()


@pytest.fixture
def nx_graph():
    g = nx.MultiDiGraph()
    for row in _node_rows():
        g.add_node(row["node_id"], **row)
    for src, tgt, rel in _edges():
        g.add_edge(src, tgt, relationship_type=rel)
    return g


@pytest.fixture
def storage_service(storage):
    return StorageQueryService(storage=storage, text_index=StorageTextIndex(storage))


@pytest.fixture
def graph_service(nx_graph):
    return GraphQueryService(nx_graph)


# ---------------------------------------------------------------------------
# resolve_symbol
# ---------------------------------------------------------------------------


def test_resolve_symbol_exact(storage_service):
    nid = storage_service.resolve_symbol("AuthService")
    assert nid == "auth::AuthService"


def test_resolve_symbol_with_file_and_lang(storage_service):
    nid = storage_service.resolve_symbol(
        "AuthService", file_path="src/auth/service.py", language="python"
    )
    assert nid == "auth::AuthService"


def test_resolve_symbol_unknown(storage_service):
    assert storage_service.resolve_symbol("NoSuchSymbol") is None


# ---------------------------------------------------------------------------
# search
# ---------------------------------------------------------------------------


def test_search_returns_symbols(storage_service):
    results = storage_service.search("AuthService", k=5)
    names = {r.symbol_name for r in results}
    assert "AuthService" in names


def test_search_with_type_filter(storage_service):
    results = storage_service.search("Service", k=5, symbol_types=frozenset({"class"}))
    kinds = {r.symbol_type for r in results}
    assert kinds <= {"class"}


def test_search_empty_query_returns_empty(storage_service):
    assert storage_service.search("", k=5) == []


# ---------------------------------------------------------------------------
# get_relationships
# ---------------------------------------------------------------------------


def test_get_relationships_outgoing(storage_service, graph_service):
    rels_storage = storage_service.get_relationships(
        "auth::AuthService", direction="outgoing", max_depth=1
    )
    rels_graph = graph_service.get_relationships(
        "auth::AuthService", direction="outgoing", max_depth=1
    )

    storage_targets = {r.target_name for r in rels_storage}
    graph_targets = {r.target_name for r in rels_graph}
    assert storage_targets == graph_targets
    assert "TokenManager" in storage_targets


def test_get_relationships_incoming(storage_service):
    rels = storage_service.get_relationships(
        "auth::TokenManager", direction="incoming", max_depth=1
    )
    sources = {r.source_name for r in rels}
    assert "AuthService" in sources


def test_get_relationships_both_multi_hop(storage_service):
    rels = storage_service.get_relationships(
        "auth::AuthService", direction="outgoing", max_depth=2
    )
    targets = {r.target_name for r in rels}
    # Hop-2: TokenManager -> DatabasePool
    assert "DatabasePool" in targets


# ---------------------------------------------------------------------------
# resolve_and_traverse
# ---------------------------------------------------------------------------


def test_resolve_and_traverse(storage_service):
    node_id, rels = storage_service.resolve_and_traverse(
        "AuthService", direction="outgoing", max_depth=1
    )
    assert node_id == "auth::AuthService"
    assert any(r.target_name == "TokenManager" for r in rels)


# ---------------------------------------------------------------------------
# query (JQL)
# ---------------------------------------------------------------------------


def test_query_type_filter(storage_service):
    results = storage_service.query("type:class")
    names = {r.symbol_name for r in results}
    assert {"AuthService", "TokenManager", "DatabasePool"} <= names


def test_query_name_exact(storage_service):
    results = storage_service.query("name:AuthService")
    assert any(r.symbol_name == "AuthService" for r in results)


def test_query_file_prefix(storage_service):
    results = storage_service.query("file:src/auth/")
    paths = {r.rel_path for r in results}
    assert all(p.startswith("src/auth/") for p in paths)


def test_query_has_rel(storage_service):
    results = storage_service.query("has_rel:calls")
    names = {r.symbol_name for r in results}
    # AuthService and TokenManager both have outgoing 'calls' edges
    assert {"AuthService", "TokenManager"} <= names


def test_query_related(storage_service):
    results = storage_service.query('related:"AuthService" dir:outgoing')
    names = {r.symbol_name for r in results}
    assert "TokenManager" in names


def test_query_connections_gt(storage_service):
    # AuthService has 2 outgoing edges, TokenManager has 1, DatabasePool has 0
    results = storage_service.query("connections:>=1")
    names = {r.symbol_name for r in results}
    assert "AuthService" in names


def test_query_limit(storage_service):
    results = storage_service.query("type:class limit:1")
    assert len(results) <= 1


# ---------------------------------------------------------------------------
# stats
# ---------------------------------------------------------------------------


def test_stats_returns_dict(storage_service):
    stats = storage_service.stats()
    assert isinstance(stats, dict)
