"""Unit tests for app/core/unified_retriever.py"""

from unittest.mock import MagicMock, patch

import pytest
from langchain_core.documents import Document

from app.core.unified_retriever import (
    UnifiedRetriever,
    _ContentExpanderShim,
    _EmbeddingsShim,
    _node_to_document,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_node(
    node_id: str = "node-1",
    symbol_name: str = "MyClass",
    symbol_type: str = "class",
    source_text: str = "class MyClass: pass",
    rel_path: str = "src/my.py",
    is_doc: int = 0,
    combined_score: float | None = None,
    macro_cluster: int | None = None,
) -> dict:
    node = {
        "node_id": node_id,
        "symbol_name": symbol_name,
        "symbol_type": symbol_type,
        "source_text": source_text,
        "rel_path": rel_path,
        "file_name": rel_path.split("/")[-1],
        "language": "python",
        "start_line": 1,
        "end_line": 5,
        "chunk_type": symbol_type,
        "docstring": "",
        "signature": "",
        "is_architectural": 1,
        "is_doc": is_doc,
        "macro_cluster": macro_cluster,
        "micro_cluster": None,
    }
    if combined_score is not None:
        node["combined_score"] = combined_score
    return node


def _make_db(search_results: list | None = None, edges_from: list | None = None, edges_to: list | None = None):
    """Build a mock UnifiedWikiDB."""
    db = MagicMock()
    db.vec_available = True
    db.search_hybrid.return_value = search_results or []
    db.get_edges_from.return_value = edges_from or []
    db.get_edges_to.return_value = edges_to or []
    db.get_node.return_value = None
    return db


def _make_retriever(
    search_results: list | None = None,
    edges_from: list | None = None,
    edges_to: list | None = None,
    embedding_fn=None,
) -> UnifiedRetriever:
    db = _make_db(search_results, edges_from, edges_to)
    return UnifiedRetriever(db=db, embedding_fn=embedding_fn)


# ---------------------------------------------------------------------------
# _node_to_document
# ---------------------------------------------------------------------------


class TestNodeToDocument:
    def test_returns_document_instance(self):
        node = _make_node()
        doc = _node_to_document(node)
        assert isinstance(doc, Document)

    def test_page_content_is_source_text(self):
        node = _make_node(source_text="some code here")
        doc = _node_to_document(node)
        assert doc.page_content == "some code here"

    def test_metadata_contains_node_id(self):
        node = _make_node(node_id="abc-123")
        doc = _node_to_document(node)
        assert doc.metadata["node_id"] == "abc-123"

    def test_metadata_contains_symbol_type(self):
        node = _make_node(symbol_type="function")
        doc = _node_to_document(node)
        assert doc.metadata["symbol_type"] == "function"

    def test_metadata_source_is_rel_path(self):
        node = _make_node(rel_path="lib/utils.py")
        doc = _node_to_document(node)
        assert doc.metadata["source"] == "lib/utils.py"

    def test_combined_score_included_when_present(self):
        node = _make_node(combined_score=0.85)
        doc = _node_to_document(node)
        assert "combined_score" in doc.metadata
        assert doc.metadata["combined_score"] == 0.85

    def test_combined_score_absent_when_not_in_node(self):
        node = _make_node()  # no combined_score
        doc = _node_to_document(node)
        assert "combined_score" not in doc.metadata

    def test_empty_source_text_produces_empty_content(self):
        node = _make_node(source_text="")
        doc = _node_to_document(node)
        assert doc.page_content == ""

    def test_is_doc_flag_cast_to_bool(self):
        node = _make_node(is_doc=1)
        doc = _node_to_document(node)
        assert doc.metadata["is_doc"] is True


# ---------------------------------------------------------------------------
# UnifiedRetriever construction
# ---------------------------------------------------------------------------


class TestUnifiedRetrieverConstruction:
    def test_relationship_graph_is_none(self):
        r = _make_retriever()
        assert r.relationship_graph is None

    def test_content_expander_is_shim(self):
        r = _make_retriever()
        assert isinstance(r.content_expander, _ContentExpanderShim)

    def test_vectorstore_manager_is_embeddings_shim(self):
        r = _make_retriever()
        assert isinstance(r.vectorstore_manager, _EmbeddingsShim)

    def test_fts_weight_default(self):
        r = _make_retriever()
        assert r.fts_weight == pytest.approx(0.4)

    def test_vec_weight_default(self):
        r = _make_retriever()
        assert r.vec_weight == pytest.approx(0.6)

    def test_custom_weights_stored(self):
        db = _make_db()
        r = UnifiedRetriever(db=db, fts_weight=0.3, vec_weight=0.7)
        assert r.fts_weight == pytest.approx(0.3)
        assert r.vec_weight == pytest.approx(0.7)


# ---------------------------------------------------------------------------
# search_repository
# ---------------------------------------------------------------------------


class TestSearchRepository:
    def test_returns_empty_for_blank_query(self):
        r = _make_retriever()
        result = r.search_repository("", k=10)
        assert result == []

    def test_returns_empty_for_whitespace_query(self):
        r = _make_retriever()
        result = r.search_repository("   ", k=10)
        assert result == []

    def test_returns_list_of_documents(self):
        nodes = [_make_node("n1"), _make_node("n2")]
        r = _make_retriever(search_results=nodes)
        results = r.search_repository("some query", k=5, apply_expansion=False)
        assert isinstance(results, list)
        assert len(results) == 2

    def test_results_are_document_instances(self):
        nodes = [_make_node("n1")]
        r = _make_retriever(search_results=nodes)
        results = r.search_repository("query", k=5, apply_expansion=False)
        for doc in results:
            assert isinstance(doc, Document)

    def test_initially_retrieved_flag_set(self):
        nodes = [_make_node("n1"), _make_node("n2")]
        r = _make_retriever(search_results=nodes)
        results = r.search_repository("query", k=5, apply_expansion=False)
        for doc in results:
            assert doc.metadata["is_initially_retrieved"] is True

    def test_no_expansion_when_disabled(self):
        nodes = [_make_node("n1")]
        db = _make_db(search_results=nodes)
        r = UnifiedRetriever(db=db)
        results = r.search_repository("query", k=5, apply_expansion=False)
        db.get_edges_from.assert_not_called()

    def test_expansion_called_when_enabled(self):
        nodes = [_make_node(node_id="n1")]
        db = _make_db(search_results=nodes)
        # Return no edges so expansion doesn't add anything
        db.get_edges_from.return_value = []
        db.get_edges_to.return_value = []
        r = UnifiedRetriever(db=db)
        results = r.search_repository("query", k=5, apply_expansion=True)
        db.get_edges_from.assert_called()

    def test_calls_db_search_hybrid(self):
        db = _make_db()
        r = UnifiedRetriever(db=db)
        r.search_repository("test query", k=10, apply_expansion=False)
        db.search_hybrid.assert_called_once()

    def test_path_prefix_forwarded_to_db(self):
        db = _make_db()
        r = UnifiedRetriever(db=db)
        r.search_repository("query", k=5, apply_expansion=False, path_prefix="src/")
        call_kwargs = db.search_hybrid.call_args[1]
        assert call_kwargs["path_prefix"] == "src/"

    def test_cluster_id_forwarded_to_db(self):
        db = _make_db()
        r = UnifiedRetriever(db=db)
        r.search_repository("query", k=5, apply_expansion=False, cluster_id=3)
        call_kwargs = db.search_hybrid.call_args[1]
        assert call_kwargs["cluster_id"] == 3


# ---------------------------------------------------------------------------
# search_docs_semantic
# ---------------------------------------------------------------------------


class TestSearchDocsSemantic:
    def test_empty_query_returns_empty(self):
        r = _make_retriever()
        assert r.search_docs_semantic("") == []

    def test_whitespace_query_returns_empty(self):
        r = _make_retriever()
        assert r.search_docs_semantic("   ") == []

    def test_only_doc_nodes_returned(self):
        nodes = [
            _make_node("d1", symbol_type="markdown_document", is_doc=1, combined_score=1.0),
            _make_node("c1", symbol_type="class", is_doc=0, combined_score=1.0),
        ]
        r = _make_retriever(search_results=nodes)
        # Use threshold=0 to bypass score filtering
        results = r.search_docs_semantic("documentation", k=10, similarity_threshold=0.0)
        assert all(doc.metadata.get("is_documentation") for doc in results)
        assert len(results) == 1

    def test_semantic_retrieved_flag_set(self):
        nodes = [_make_node("d1", symbol_type="markdown_document", is_doc=1, combined_score=1.0)]
        r = _make_retriever(search_results=nodes)
        results = r.search_docs_semantic("docs", k=10, similarity_threshold=0.0)
        for doc in results:
            assert doc.metadata.get("semantic_retrieved") is True

    def test_respects_k_limit(self):
        nodes = [
            _make_node(node_id=f"d{i}", symbol_type="markdown_document", is_doc=1, combined_score=1.0)
            for i in range(20)
        ]
        r = _make_retriever(search_results=nodes)
        results = r.search_docs_semantic("docs", k=5, similarity_threshold=0.0)
        assert len(results) <= 5

    def test_no_docs_returns_empty(self):
        nodes = [_make_node("c1", symbol_type="class", is_doc=0, combined_score=1.0)]
        r = _make_retriever(search_results=nodes)
        results = r.search_docs_semantic("query", k=10, similarity_threshold=0.0)
        assert results == []

    def test_similarity_threshold_applied(self):
        nodes = [
            _make_node("d1", symbol_type="markdown_document", is_doc=1, combined_score=0.015),
            _make_node("d2", symbol_type="markdown_document", is_doc=1, combined_score=0.001),
        ]
        r = _make_retriever(search_results=nodes)
        # With threshold=0.9, the tail doc should be filtered (score < max*(1-0.9))
        results_high = r.search_docs_semantic("query", k=10, similarity_threshold=0.9)
        results_zero = r.search_docs_semantic("query", k=10, similarity_threshold=0.0)
        # High threshold should have fewer or equal results
        assert len(results_high) <= len(results_zero)


# ---------------------------------------------------------------------------
# _embed
# ---------------------------------------------------------------------------


class TestEmbed:
    def test_returns_none_when_no_embedding_fn(self):
        r = _make_retriever()
        result = r._embed("some query")
        assert result is None

    def test_calls_embedding_fn(self):
        fn = MagicMock(return_value=[0.1, 0.2, 0.3])
        r = _make_retriever(embedding_fn=fn)
        result = r._embed("query text")
        fn.assert_called_once_with("query text")
        assert result == [0.1, 0.2, 0.3]

    def test_returns_none_on_embedding_fn_exception(self):
        fn = MagicMock(side_effect=RuntimeError("embedding failed"))
        r = _make_retriever(embedding_fn=fn)
        result = r._embed("query")
        assert result is None


# ---------------------------------------------------------------------------
# invoke / get_relevant_documents
# ---------------------------------------------------------------------------


class TestInvokeAndGetRelevantDocuments:
    def test_invoke_delegates_to_search_repository(self):
        nodes = [_make_node("n1")]
        r = _make_retriever(search_results=nodes)
        result = r.invoke("hello")
        assert isinstance(result, list)
        assert len(result) == 1

    def test_get_relevant_documents_returns_same_as_search(self):
        nodes = [_make_node("n1")]
        r = _make_retriever(search_results=nodes)
        a = r.get_relevant_documents("hello")
        b = r.search_repository("hello", apply_expansion=False)
        # Both call search_repository — just check same type/count after fresh calls
        assert len(a) == len(b)


# ---------------------------------------------------------------------------
# _expand_documents
# ---------------------------------------------------------------------------


class TestExpandDocuments:
    def test_initial_docs_preserved(self):
        nodes = [_make_node("n1"), _make_node("n2")]
        r = _make_retriever()
        docs = [_node_to_document(n) for n in nodes]
        expanded = r._expand_documents(docs)
        assert len(expanded) >= 2

    def test_expansion_adds_neighbors(self):
        db = _make_db()
        # Return one neighbor from get_edges_from
        neighbor_node = _make_node("neighbor-1", symbol_type="function")
        db.get_edges_from.return_value = [{"target_id": "neighbor-1"}]
        db.get_edges_to.return_value = []
        db.get_node.return_value = neighbor_node

        r = UnifiedRetriever(db=db)
        initial_doc = _node_to_document(_make_node("n1"))
        initial_doc.metadata["node_id"] = "n1"
        expanded = r._expand_documents([initial_doc])

        # Should have initial + neighbor
        node_ids = [d.metadata.get("node_id") for d in expanded]
        assert "neighbor-1" in node_ids

    def test_expanded_docs_have_expanded_from_set(self):
        db = _make_db()
        neighbor = _make_node("nb1", symbol_type="function")
        db.get_edges_from.return_value = [{"target_id": "nb1"}]
        db.get_edges_to.return_value = []
        db.get_node.return_value = neighbor

        r = UnifiedRetriever(db=db)
        initial_doc = _node_to_document(_make_node("n1"))
        initial_doc.metadata["node_id"] = "n1"
        expanded = r._expand_documents([initial_doc])

        expanded_docs = [d for d in expanded if d.metadata.get("expanded_from")]
        assert len(expanded_docs) == 1
        assert expanded_docs[0].metadata["expanded_from"] == "n1"


# ---------------------------------------------------------------------------
# _EmbeddingsShim
# ---------------------------------------------------------------------------


class TestEmbeddingsShim:
    def test_embeddings_attribute_accessible(self):
        shim = _EmbeddingsShim(embeddings=MagicMock())
        assert shim.embeddings is not None

    def test_embed_query_delegates(self):
        mock_emb = MagicMock()
        mock_emb.embed_query.return_value = [0.1, 0.2]
        shim = _EmbeddingsShim(embeddings=mock_emb)
        result = shim.embed_query("hello")
        assert result == [0.1, 0.2]

    def test_embed_query_raises_when_no_embeddings(self):
        shim = _EmbeddingsShim(embeddings=None)
        with pytest.raises(RuntimeError, match="No embeddings model"):
            shim.embed_query("hello")


# ---------------------------------------------------------------------------
# _ContentExpanderShim
# ---------------------------------------------------------------------------


class TestContentExpanderShim:
    def test_graph_is_none(self):
        shim = _ContentExpanderShim()
        assert shim.graph is None

    def test_expand_returns_input_unchanged(self):
        shim = _ContentExpanderShim()
        docs = [Document(page_content="test", metadata={})]
        result = shim.expand_retrieved_documents(docs)
        assert result is docs
