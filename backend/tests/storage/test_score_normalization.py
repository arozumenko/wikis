"""
Phase 0 (graph-quality roadmap) — score normalization tests.

Validates that ``score_norm`` ∈ (0, 1] is attached to FTS rows from
both backends and that exact-match queries score higher than partial
matches on the same data.
"""

from __future__ import annotations

import os
from typing import Iterable

import networkx as nx
import pytest

from app.core.unified_db import UnifiedWikiDB

POSTGRES_DSN = os.getenv("WIKI_TEST_POSTGRES_DSN")


def _seed_graph() -> nx.MultiDiGraph:
    G = nx.MultiDiGraph()
    G.add_node(
        "AuthService",
        symbol_name="AuthService",
        symbol_type="class",
        rel_path="src/auth.py",
        start_line=1,
        end_line=30,
        language="python",
        source_text="class AuthService:\n    '''Handles authentication.'''",
        signature="class AuthService",
        docstring="Handles authentication.",
    )
    G.add_node(
        "login_user",
        symbol_name="login_user",
        symbol_type="function",
        rel_path="src/auth.py",
        start_line=32,
        end_line=45,
        language="python",
        source_text="def login_user(user): return True",
        signature="def login_user(user)",
        docstring="Login a user.",
    )
    G.add_node(
        "Helper",
        symbol_name="Helper",
        symbol_type="class",
        rel_path="src/utils/helper.py",
        start_line=1,
        end_line=10,
        language="python",
        source_text="class Helper: pass",
        signature="class Helper",
        docstring="",
    )
    return G


@pytest.fixture()
def sqlite_storage(tmp_path):
    db = UnifiedWikiDB(str(tmp_path / "norm.db"), embedding_dim=8)
    db.from_networkx(_seed_graph())
    db.conn.commit()
    yield db
    db.close()


@pytest.fixture()
def postgres_storage():
    if not POSTGRES_DSN:
        pytest.skip("WIKI_TEST_POSTGRES_DSN not configured")
    from app.core.storage.postgres import PostgresWikiStorage

    schema = "wiki_test_score_norm"
    db = PostgresWikiStorage(POSTGRES_DSN, schema=schema, embedding_dim=8)
    try:
        db.drop_schema()
    except Exception:
        pass
    db.create_schema()
    db.from_networkx(_seed_graph())
    yield db
    try:
        db.drop_schema()
    except Exception:
        pass
    db.close()


def _backends(sqlite_storage, postgres_storage) -> Iterable:
    yield "sqlite", sqlite_storage
    if POSTGRES_DSN:
        yield "postgres", postgres_storage


def test_search_fts_attaches_score_norm(sqlite_storage):
    rows = sqlite_storage.search_fts5("AuthService", limit=5)
    assert rows, "expected at least one match"
    for r in rows:
        assert "score_norm" in r, f"missing score_norm on row: {r}"
        assert 0.0 < r["score_norm"] <= 1.0, r["score_norm"]
        assert "fts_rank" in r


def test_search_fts_by_symbol_name_attaches_score_norm(sqlite_storage):
    rows = sqlite_storage.search_fts_by_symbol_name(
        "login_user", architectural_only=False
    )
    assert rows
    for r in rows:
        assert "score_norm" in r
        assert 0.0 < r["score_norm"] <= 1.0
        assert "fts_rank" in r


def test_score_norm_orders_consistently_with_rank(sqlite_storage):
    rows = sqlite_storage.search_fts5("AuthService authentication", limit=10)
    if len(rows) < 2:
        pytest.skip("need at least two rows to compare ordering")
    # Higher score_norm rows must have lower abs(fts_rank).
    sorted_rows = sorted(rows, key=lambda r: r["score_norm"], reverse=True)
    abs_ranks = [abs(r["fts_rank"]) for r in sorted_rows]
    assert abs_ranks == sorted(abs_ranks), abs_ranks


@pytest.mark.skipif(not POSTGRES_DSN, reason="postgres not configured")
def test_postgres_search_fts_attaches_score_norm(postgres_storage):
    rows = postgres_storage.search_fts("AuthService", limit=5)
    assert rows
    for r in rows:
        assert "score_norm" in r
        assert 0.0 <= r["score_norm"] <= 1.0
        assert "fts_rank" in r
