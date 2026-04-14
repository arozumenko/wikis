"""
Tests for UnifiedRetriever — the single retrieval path backed by UnifiedWikiDB.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from app.core.unified_retriever import (
    DEFAULT_FTS_WEIGHT,
    DEFAULT_VEC_WEIGHT,
    FTS_POOL,
    MAX_EXPANDED_DOCS,
    MAX_EXPANSION_NEIGHBORS,
    UNIFIED_RETRIEVER_ENABLED,
    VEC_POOL,
    UnifiedRetriever,
    _ContentExpanderShim,
    _EmbeddingsShim,
    _node_to_document,
)
from app.core.unified_db import UnifiedWikiDB


# ═══════════════════════════════════════════════════════════════════════════
# Fixtures
# ═══════════════════════════════════════════════════════════════════════════


@pytest.fixture
def tmp_db(tmp_path):
    """Create a temporary UnifiedWikiDB with sample data."""
    db_path = tmp_path / "test.wiki.db"
    db = UnifiedWikiDB(str(db_path), embedding_dim=4)

    nodes = [
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
            "macro_cluster": 0,
            "micro_cluster": 0,
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
            "macro_cluster": 0,
            "micro_cluster": 1,
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
            "macro_cluster": 0,
            "micro_cluster": 0,
        },
        {
            "node_id": "docs::readme",
            "rel_path": "README.md",
            "file_name": "README.md",
            "language": "markdown",
            "symbol_name": "README",
            "symbol_type": "markdown_document",
            "source_text": "# Project\nThis is the project readme.",
            "docstring": "",
            "is_architectural": 0,
            "is_doc": 1,
            "macro_cluster": None,
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
            "macro_cluster": 1,
            "micro_cluster": 0,
        },
    ]

    for n in nodes:
        db._upsert_nodes_batch([n])

    # Insert edges: AuthService -> TokenManager, AuthService -> login (child)
    db.upsert_edge("auth::AuthService", "auth::TokenManager", "calls")
    db.upsert_edge("auth::AuthService", "auth::login", "contains")
    db.upsert_edge("auth::TokenManager", "db::DatabasePool", "calls")

    db.conn.commit()
    db._populate_fts5()

    yield db
    db.close()


@pytest.fixture
def retriever(tmp_db):
    """Create a UnifiedRetriever with a mock embedding function."""

    def mock_embed(text):
        n = len(text) % 10
        return [float(n) / 10.0, 0.5, 0.3, 0.1]

    return UnifiedRetriever(
        db=tmp_db,
        embedding_fn=mock_embed,
        embeddings=MagicMock(embed_query=mock_embed),
    )


@pytest.fixture
def retriever_no_embed(tmp_db):
    """Create a UnifiedRetriever without embedding function (FTS5-only)."""
    return UnifiedRetriever(db=tmp_db, embedding_fn=None)


# ═══════════════════════════════════════════════════════════════════════════
# Test: _node_to_document
# ═══════════════════════════════════════════════════════════════════════════


class TestNodeToDocument:
    def test_basic_conversion(self):
        node = {
            "node_id": "mod::Foo",
            "symbol_name": "Foo",
            "symbol_type": "class",
            "rel_path": "src/foo.py",
            "file_name": "foo.py",
            "language": "python",
            "start_line": 1,
            "end_line": 50,
            "source_text": "class Foo:\n    pass",
            "docstring": "Foo class",
            "signature": "class Foo",
            "is_architectural": 1,
            "is_doc": 0,
            "macro_cluster": 3,
            "micro_cluster": 1,
        }
        doc = _node_to_document(node)
        assert doc.page_content == "class Foo:\n    pass"
        assert doc.metadata["node_id"] == "mod::Foo"
        assert doc.metadata["symbol_name"] == "Foo"
        assert doc.metadata["symbol_type"] == "class"
        assert doc.metadata["source"] == "src/foo.py"
        assert doc.metadata["macro_cluster"] == 3
        assert doc.metadata["is_architectural"] is True

    def test_with_combined_score(self):
        node = {
            "node_id": "a",
            "combined_score": 0.42,
            "source_text": "x",
        }
        doc = _node_to_document(node)
        assert doc.metadata["combined_score"] == 0.42

    def test_empty_source_text(self):
        node = {"node_id": "b"}
        doc = _node_to_document(node)
        assert doc.page_content == ""

    def test_fts_rank_and_vec_distance(self):
        node = {"node_id": "c", "fts_rank": -5.3, "vec_distance": 0.12}
        doc = _node_to_document(node)
        assert doc.metadata["fts_rank"] == -5.3
        assert doc.metadata["vec_distance"] == 0.12


# ═══════════════════════════════════════════════════════════════════════════
# Test: search_repository
# ═══════════════════════════════════════════════════════════════════════════


class TestSearchRepository:
    def test_basic_search_returns_docs(self, retriever):
        docs = retriever.search_repository("AuthService login", k=5)
        assert len(docs) > 0
        names = [d.metadata["symbol_name"] for d in docs]
        assert "AuthService" in names

    def test_initially_retrieved_flag(self, retriever):
        docs = retriever.search_repository("authentication", k=5, apply_expansion=False)
        for doc in docs:
            assert doc.metadata.get("is_initially_retrieved") is True

    def test_no_expansion(self, retriever):
        docs_no_exp = retriever.search_repository("AuthService", k=5, apply_expansion=False)
        docs_with_exp = retriever.search_repository("AuthService", k=5, apply_expansion=True)
        assert len(docs_with_exp) >= len(docs_no_exp)

    def test_expansion_marks_expanded_docs(self, retriever):
        docs = retriever.search_repository("AuthService", k=5, apply_expansion=True)
        initial = [d for d in docs if d.metadata.get("is_initially_retrieved")]
        expanded = [d for d in docs if not d.metadata.get("is_initially_retrieved")]
        assert len(initial) > 0
        for d in expanded:
            assert "expanded_from" in d.metadata

    def test_empty_query(self, retriever):
        assert retriever.search_repository("") == []
        assert retriever.search_repository("   ") == []

    def test_path_prefix_filter(self, retriever):
        docs = retriever.search_repository(
            "class", k=10, apply_expansion=False, path_prefix="src/auth"
        )
        for doc in docs:
            assert doc.metadata["rel_path"].startswith("src/auth")

    def test_cluster_id_filter(self, retriever):
        docs = retriever.search_repository(
            "class", k=10, apply_expansion=False, cluster_id=1
        )
        for doc in docs:
            assert doc.metadata.get("macro_cluster") == 1

    def test_fts_only_when_no_embedding(self, retriever_no_embed):
        docs = retriever_no_embed.search_repository("AuthService", k=5)
        assert len(docs) > 0
        names = [d.metadata["symbol_name"] for d in docs]
        assert "AuthService" in names


# ═══════════════════════════════════════════════════════════════════════════
# Test: search_docs_semantic
# ═══════════════════════════════════════════════════════════════════════════


class TestSearchDocsSemantic:
    def test_returns_only_docs(self, retriever):
        docs = retriever.search_docs_semantic("project readme", k=5)
        for doc in docs:
            stype = doc.metadata.get("symbol_type", "")
            is_doc = doc.metadata.get("is_doc", False)
            assert is_doc or stype.endswith("_document") or stype in (
                "module_doc", "file_doc", "documentation"
            )

    def test_semantic_retrieved_flag(self, retriever):
        docs = retriever.search_docs_semantic("readme", k=5)
        for doc in docs:
            assert doc.metadata.get("semantic_retrieved") is True
            assert doc.metadata.get("is_documentation") is True

    def test_empty_query(self, retriever):
        assert retriever.search_docs_semantic("") == []

    def test_no_docs_in_corpus(self, tmp_path):
        """DB with only code symbols returns empty doc search."""
        db_path = tmp_path / "nocode.wiki.db"
        db = UnifiedWikiDB(str(db_path), embedding_dim=4)
        db._upsert_nodes_batch([{
            "node_id": "a::Foo",
            "rel_path": "a.py",
            "symbol_name": "Foo",
            "symbol_type": "class",
            "source_text": "class Foo: pass",
            "is_doc": 0,
        }])
        db.conn.commit()
        db._populate_fts5()

        ret = UnifiedRetriever(db=db, embedding_fn=None)
        docs = ret.search_docs_semantic("Foo", k=5)
        assert docs == []
        db.close()


# ═══════════════════════════════════════════════════════════════════════════
# Test: Graph expansion
# ═══════════════════════════════════════════════════════════════════════════


class TestGraphExpansion:
    def test_expand_finds_neighbours(self, retriever):
        """Expansion from AuthService should find TokenManager (calls edge)."""
        from langchain_core.documents import Document

        initial_doc = Document(
            page_content="class AuthService",
            metadata={
                "node_id": "auth::AuthService",
                "symbol_name": "AuthService",
                "symbol_type": "class",
                "is_initially_retrieved": True,
            },
        )
        expanded = retriever._expand_documents([initial_doc])
        ids = [d.metadata["node_id"] for d in expanded]
        assert "auth::AuthService" in ids
        assert "auth::TokenManager" in ids

    def test_expansion_filters_non_architectural(self, retriever):
        """login is a method — expansion should skip it."""
        from langchain_core.documents import Document

        initial_doc = Document(
            page_content="class AuthService",
            metadata={
                "node_id": "auth::AuthService",
                "symbol_name": "AuthService",
                "symbol_type": "class",
                "is_initially_retrieved": True,
            },
        )
        expanded = retriever._expand_documents([initial_doc])
        ids = [d.metadata["node_id"] for d in expanded]
        assert "auth::login" not in ids

    def test_expansion_respects_cluster_boundary(self, retriever):
        """When cluster_id=0, expansion should not cross to cluster 1."""
        from langchain_core.documents import Document

        initial_doc = Document(
            page_content="class TokenManager",
            metadata={
                "node_id": "auth::TokenManager",
                "symbol_name": "TokenManager",
                "symbol_type": "class",
                "is_initially_retrieved": True,
            },
        )
        expanded = retriever._expand_documents([initial_doc], cluster_id=0)
        ids = [d.metadata["node_id"] for d in expanded]
        assert "db::DatabasePool" not in ids

    def test_expansion_without_cluster_boundary(self, retriever):
        """Without cluster_id, cross-cluster expansion is allowed."""
        from langchain_core.documents import Document

        initial_doc = Document(
            page_content="class TokenManager",
            metadata={
                "node_id": "auth::TokenManager",
                "symbol_name": "TokenManager",
                "symbol_type": "class",
                "is_initially_retrieved": True,
            },
        )
        expanded = retriever._expand_documents([initial_doc])
        ids = [d.metadata["node_id"] for d in expanded]
        assert "db::DatabasePool" in ids

    def test_expansion_cap(self, retriever):
        """Expansion stops adding neighbours once MAX_EXPANDED_DOCS is reached."""
        from langchain_core.documents import Document

        initial_doc = Document(
            page_content="class AuthService",
            metadata={
                "node_id": "auth::AuthService",
                "symbol_name": "AuthService",
                "symbol_type": "class",
                "is_initially_retrieved": True,
            },
        )
        expanded = retriever._expand_documents([initial_doc])
        assert len(expanded) <= MAX_EXPANDED_DOCS

    def test_expansion_preserves_all_initial_docs(self, retriever):
        """All initially-retrieved docs are kept even when > MAX_EXPANDED_DOCS."""
        from langchain_core.documents import Document

        docs = []
        for i in range(200):
            docs.append(
                Document(
                    page_content=f"symbol_{i}",
                    metadata={
                        "node_id": f"fake::{i}",
                        "symbol_name": f"Symbol{i}",
                        "symbol_type": "class",
                        "is_initially_retrieved": True,
                    },
                )
            )
        expanded = retriever._expand_documents(docs)
        assert len(expanded) >= 200

    def test_no_duplicate_nodes(self, retriever):
        """Expansion should not produce duplicate node_ids."""
        from langchain_core.documents import Document

        initial_doc = Document(
            page_content="class AuthService",
            metadata={
                "node_id": "auth::AuthService",
                "symbol_name": "AuthService",
                "symbol_type": "class",
                "is_initially_retrieved": True,
            },
        )
        expanded = retriever._expand_documents([initial_doc])
        ids = [d.metadata["node_id"] for d in expanded if d.metadata.get("node_id")]
        assert len(ids) == len(set(ids))


# ═══════════════════════════════════════════════════════════════════════════
# Test: Compatibility shims
# ═══════════════════════════════════════════════════════════════════════════


class TestShims:
    def test_embeddings_shim_with_model(self):
        mock_emb = MagicMock()
        mock_emb.embed_query.return_value = [0.1, 0.2]
        shim = _EmbeddingsShim(mock_emb)
        assert shim.embeddings is mock_emb
        assert shim.embed_query("test") == [0.1, 0.2]

    def test_embeddings_shim_without_model(self):
        shim = _EmbeddingsShim(None)
        assert shim.embeddings is None
        with pytest.raises(RuntimeError, match="No embeddings"):
            shim.embed_query("test")

    def test_content_expander_shim(self):
        shim = _ContentExpanderShim()
        assert shim.graph is None
        docs = [MagicMock()]
        assert shim.expand_retrieved_documents(docs) is docs

    def test_retriever_relationship_graph_is_none(self, retriever):
        assert retriever.relationship_graph is None

    def test_retriever_content_expander_graph_is_none(self, retriever):
        assert retriever.content_expander.graph is None

    def test_vectorstore_manager_embeddings_accessible(self, retriever):
        assert retriever.vectorstore_manager.embeddings is not None


# ═══════════════════════════════════════════════════════════════════════════
# Test: LangChain interface
# ═══════════════════════════════════════════════════════════════════════════


class TestLangChainInterface:
    def test_invoke(self, retriever):
        docs = retriever.invoke("authentication")
        assert isinstance(docs, list)
        assert all(hasattr(d, "page_content") for d in docs)

    def test_get_relevant_documents(self, retriever):
        docs = retriever.get_relevant_documents("authentication")
        assert isinstance(docs, list)


# ═══════════════════════════════════════════════════════════════════════════
# Test: Embedding helper
# ═══════════════════════════════════════════════════════════════════════════


class TestEmbeddingHelper:
    def test_embed_with_fn(self, retriever):
        vec = retriever._embed("hello")
        assert isinstance(vec, list)
        assert len(vec) == 4

    def test_embed_without_fn(self, retriever_no_embed):
        assert retriever_no_embed._embed("hello") is None

    def test_embed_exception_returns_none(self, tmp_db):
        def bad_embed(text):
            raise RuntimeError("embedding service down")

        ret = UnifiedRetriever(db=tmp_db, embedding_fn=bad_embed)
        assert ret._embed("hello") is None


# ═══════════════════════════════════════════════════════════════════════════
# Test: Feature flag
# ═══════════════════════════════════════════════════════════════════════════


class TestFeatureFlag:
    def test_flag_always_true(self):
        """UNIFIED_RETRIEVER_ENABLED is always True (no env var gating)."""
        assert UNIFIED_RETRIEVER_ENABLED is True


# ═══════════════════════════════════════════════════════════════════════════
# Test: Edge cases
# ═══════════════════════════════════════════════════════════════════════════


class TestEdgeCases:
    def test_expansion_with_nonexistent_node(self, retriever):
        """Expansion from a node not in the DB should not crash."""
        from langchain_core.documents import Document

        doc = Document(
            page_content="ghost",
            metadata={
                "node_id": "does_not_exist::Ghost",
                "symbol_name": "Ghost",
                "symbol_type": "class",
                "is_initially_retrieved": True,
            },
        )
        expanded = retriever._expand_documents([doc])
        assert len(expanded) >= 1

    def test_search_with_special_chars(self, retriever):
        """Queries with special FTS5 characters shouldn't crash."""
        docs = retriever.search_repository('foo "bar" baz*', k=5)
        assert isinstance(docs, list)

    def test_search_with_no_matches(self, retriever):
        """Query that matches nothing returns empty list."""
        docs = retriever.search_repository(
            "zzzznonexistenttokenzzz", k=5, apply_expansion=False
        )
        assert docs == []

    def test_multiple_searches_same_retriever(self, retriever):
        """Multiple searches reuse the same DB connection."""
        r1 = retriever.search_repository("auth", k=3, apply_expansion=False)
        r2 = retriever.search_repository("database", k=3, apply_expansion=False)
        assert isinstance(r1, list)
        assert isinstance(r2, list)

    def test_retriever_with_readonly_db(self, tmp_path):
        """UnifiedRetriever works with a readonly DB."""
        db_path = tmp_path / "ro.wiki.db"
        db = UnifiedWikiDB(str(db_path), embedding_dim=4)
        db._upsert_nodes_batch([{
            "node_id": "a::X",
            "rel_path": "a.py",
            "symbol_name": "X",
            "symbol_type": "class",
            "source_text": "class X: pass",
            "is_architectural": 1,
        }])
        db.conn.commit()
        db._populate_fts5()
        db.close()

        ro_db = UnifiedWikiDB(str(db_path), readonly=True)
        ret = UnifiedRetriever(db=ro_db, embedding_fn=None)
        docs = ret.search_repository("class X", k=5, apply_expansion=False)
        assert len(docs) > 0
        ro_db.close()

    def test_incoming_edge_expansion(self, retriever):
        """Expansion should also follow incoming edges."""
        from langchain_core.documents import Document

        doc = Document(
            page_content="class TokenManager",
            metadata={
                "node_id": "auth::TokenManager",
                "symbol_name": "TokenManager",
                "symbol_type": "class",
                "is_initially_retrieved": True,
            },
        )
        expanded = retriever._expand_documents([doc])
        ids = [d.metadata["node_id"] for d in expanded]
        assert "auth::AuthService" in ids
