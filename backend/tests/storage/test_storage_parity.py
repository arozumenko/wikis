"""
Phase 0 (graph-quality roadmap) — storage parity tests.

Validates that the new methods introduced by Phase 0 exist on both
backends and return consistently-shaped data:

    * count_fts_matches
    * search_fts_by_column
    * search_fts_with_path
    * get_embedding_by_id
    * batch_similarity_search
"""

from __future__ import annotations

import os

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
        source_text="class AuthService: 'Handles authentication.'",
        signature="class AuthService",
        docstring="Handles authentication for the API.",
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
    db = UnifiedWikiDB(str(tmp_path / "parity.db"), embedding_dim=8)
    db.from_networkx(_seed_graph())
    # Add embeddings for two of the nodes
    db.upsert_embeddings_batch([
        ("AuthService", [0.1] * 8),
        ("login_user", [0.2] * 8),
    ])
    db.conn.commit()
    yield db
    db.close()


@pytest.fixture()
def postgres_storage():
    if not POSTGRES_DSN:
        pytest.skip("WIKI_TEST_POSTGRES_DSN not configured")
    from app.core.storage.postgres import PostgresWikiStorage

    schema = "wiki_test_parity"
    db = PostgresWikiStorage(POSTGRES_DSN, schema=schema, embedding_dim=8)
    try:
        db.drop_schema()
    except Exception:
        pass
    db.create_schema()
    db.from_networkx(_seed_graph())
    db.upsert_embeddings_batch([
        ("AuthService", [0.1] * 8),
        ("login_user", [0.2] * 8),
    ])
    yield db
    try:
        db.drop_schema()
    except Exception:
        pass
    db.close()


# ──────────────────────────────────────────────────────────────────────────
# count_fts_matches
# ──────────────────────────────────────────────────────────────────────────


def test_count_fts_matches_basic(sqlite_storage):
    n = sqlite_storage.count_fts_matches("AuthService")
    assert n >= 1
    assert sqlite_storage.count_fts_matches("") == 0
    assert sqlite_storage.count_fts_matches("zzz_no_such_token_zzz") == 0


def test_count_fts_matches_exact(sqlite_storage):
    # Phrase that exists exactly in source
    n_exact = sqlite_storage.count_fts_matches("class AuthService", exact_match=True)
    n_loose = sqlite_storage.count_fts_matches("class AuthService", exact_match=False)
    assert n_exact >= 1
    # Loose should be ≥ exact (or equal when corpus is small)
    assert n_loose >= n_exact


# ──────────────────────────────────────────────────────────────────────────
# search_fts_by_column
# ──────────────────────────────────────────────────────────────────────────


def test_search_fts_by_column_validates_column(sqlite_storage):
    with pytest.raises(ValueError):
        sqlite_storage.search_fts_by_column("foo", "not_a_column")


def test_search_fts_by_column_symbol_name(sqlite_storage):
    rows = sqlite_storage.search_fts_by_column("AuthService", "symbol_name")
    assert rows
    assert all(r["symbol_name"] == "AuthService" for r in rows)
    for r in rows:
        assert "score_norm" in r


def test_search_fts_by_column_docstring(sqlite_storage):
    rows = sqlite_storage.search_fts_by_column("authentication", "docstring")
    assert rows
    assert any("Auth" in r["symbol_name"] for r in rows)


# ──────────────────────────────────────────────────────────────────────────
# search_fts_with_path
# ──────────────────────────────────────────────────────────────────────────


def test_search_fts_with_path_requires_prefix(sqlite_storage):
    with pytest.raises(ValueError):
        sqlite_storage.search_fts_with_path("AuthService", "")


def test_search_fts_with_path_filters(sqlite_storage):
    rows = sqlite_storage.search_fts_with_path(
        "AuthService", path_prefix="src"
    )
    assert rows
    assert all(r["rel_path"].startswith("src/") for r in rows)

    rows_other = sqlite_storage.search_fts_with_path(
        "AuthService", path_prefix="tests"
    )
    assert rows_other == []


# ──────────────────────────────────────────────────────────────────────────
# get_embedding_by_id
# ──────────────────────────────────────────────────────────────────────────


def test_get_embedding_by_id_present(sqlite_storage):
    emb = sqlite_storage.get_embedding_by_id("AuthService")
    if emb is None:
        pytest.skip("sqlite-vec extension not available in this environment")
    assert isinstance(emb, list)
    assert len(emb) == 8


def test_get_embedding_by_id_missing(sqlite_storage):
    assert sqlite_storage.get_embedding_by_id("does_not_exist") is None
    assert sqlite_storage.get_embedding_by_id("") is None


# ──────────────────────────────────────────────────────────────────────────
# batch_similarity_search
# ──────────────────────────────────────────────────────────────────────────


def test_batch_similarity_search_shape(sqlite_storage):
    if not sqlite_storage.vec_available:
        pytest.skip("sqlite-vec not available")
    out = sqlite_storage.batch_similarity_search(
        [
            ("q1", [0.1] * 8),
            ("q2", [0.5] * 8),
        ],
        k=3,
        distance_threshold=10.0,  # permissive so we always get rows
    )
    assert set(out.keys()) == {"q1", "q2"}
    for qid, hits in out.items():
        assert isinstance(hits, list)
        for h in hits:
            assert "vec_distance" in h


def test_batch_similarity_search_empty_input(sqlite_storage):
    assert sqlite_storage.batch_similarity_search([]) == {}


# ──────────────────────────────────────────────────────────────────────────
# Postgres parity — only run when a DSN is configured.
# ──────────────────────────────────────────────────────────────────────────


@pytest.mark.skipif(not POSTGRES_DSN, reason="postgres not configured")
class TestPostgresParity:
    def test_count_fts_matches(self, postgres_storage):
        assert postgres_storage.count_fts_matches("AuthService") >= 1
        assert postgres_storage.count_fts_matches("") == 0

    def test_search_fts_by_column(self, postgres_storage):
        rows = postgres_storage.search_fts_by_column("AuthService", "symbol_name")
        assert rows
        for r in rows:
            assert "score_norm" in r

    def test_search_fts_with_path(self, postgres_storage):
        with pytest.raises(ValueError):
            postgres_storage.search_fts_with_path("x", "")
        rows = postgres_storage.search_fts_with_path(
            "AuthService", path_prefix="src"
        )
        assert all(r["rel_path"].startswith("src/") for r in rows)

    def test_get_embedding_by_id(self, postgres_storage):
        emb = postgres_storage.get_embedding_by_id("AuthService")
        if emb is None:
            pytest.skip("pgvector embeddings not stored")
        assert isinstance(emb, list)
        assert len(emb) == 8

    def test_batch_similarity_search_shape(self, postgres_storage):
        out = postgres_storage.batch_similarity_search(
            [("q1", [0.1] * 8), ("q2", [0.5] * 8)],
            k=3,
            distance_threshold=10.0,
        )
        assert set(out.keys()) >= {"q1", "q2"}
