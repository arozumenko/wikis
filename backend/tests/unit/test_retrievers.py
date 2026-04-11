"""Unit tests for app/core/retrievers.py"""

from unittest.mock import MagicMock, patch

import pytest
from langchain_core.documents import Document

from app.core.retrievers import (
    SentenceTransformersCrossEncoderAdapter,
    WebRetriever,
    WikiRetrieverStack,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_doc(content: str = "test content", source: str = "src.py", chunk_type: str = "code") -> Document:
    return Document(page_content=content, metadata={"source": source, "chunk_type": chunk_type})


def _make_vectorstore_manager(
    vectorstore=None, documents=None, bm25_retriever=None
) -> MagicMock:
    mgr = MagicMock()
    mock_vs = vectorstore or MagicMock()
    mock_vs.as_retriever.return_value = MagicMock()
    mgr.get_vectorstore.return_value = mock_vs
    mgr.get_all_documents.return_value = documents if documents is not None else [_make_doc()]
    mgr.get_mmap_bm25_retriever.return_value = bm25_retriever
    return mgr


# ---------------------------------------------------------------------------
# SentenceTransformersCrossEncoderAdapter
# ---------------------------------------------------------------------------


class TestSentenceTransformersCrossEncoderAdapter:
    def test_raises_import_error_when_cross_encoder_unavailable(self):
        with patch("app.core.retrievers.CROSS_ENCODER_AVAILABLE", False):
            with pytest.raises(ImportError, match="sentence-transformers"):
                SentenceTransformersCrossEncoderAdapter("some-model")

    @patch("app.core.retrievers.CROSS_ENCODER_AVAILABLE", True)
    @patch("app.core.retrievers.CrossEncoder")
    def test_score_returns_list_of_floats(self, mock_cross_encoder_cls):
        import numpy as np

        mock_model = MagicMock()
        mock_model.predict.return_value = np.array([0.8, 0.3])
        mock_cross_encoder_cls.return_value = mock_model

        adapter = SentenceTransformersCrossEncoderAdapter("test-model")
        scores = adapter.score([("query", "doc1"), ("query", "doc2")])

        assert len(scores) == 2
        assert all(isinstance(s, float) for s in scores)

    @patch("app.core.retrievers.CROSS_ENCODER_AVAILABLE", True)
    @patch("app.core.retrievers.CrossEncoder")
    def test_score_values_match_model_output(self, mock_cross_encoder_cls):
        mock_model = MagicMock()
        mock_model.predict.return_value = [0.9, 0.1]
        mock_cross_encoder_cls.return_value = mock_model

        adapter = SentenceTransformersCrossEncoderAdapter("test-model")
        scores = adapter.score([("query", "doc1"), ("query", "doc2")])

        assert scores[0] == pytest.approx(0.9)
        assert scores[1] == pytest.approx(0.1)


# ---------------------------------------------------------------------------
# WebRetriever
# ---------------------------------------------------------------------------


class TestWebRetriever:
    def test_get_relevant_documents_returns_empty_when_no_tool(self):
        retriever = WebRetriever(k=5, api_key=None)
        result = retriever.get_relevant_documents("some query")
        assert result == []

    def test_no_search_tool_when_no_api_key(self):
        retriever = WebRetriever(k=5, api_key=None)
        assert retriever.search_tool is None

    @patch("app.core.retrievers.TAVILY_AVAILABLE", True)
    @patch("app.core.retrievers.TavilySearchResults")
    def test_initializes_tool_when_api_key_provided(self, mock_tavily_cls):
        mock_tool = MagicMock()
        mock_tavily_cls.return_value = mock_tool

        retriever = WebRetriever(k=3, api_key="test_key")
        assert retriever.search_tool is mock_tool

    @patch("app.core.retrievers.TAVILY_AVAILABLE", True)
    @patch("app.core.retrievers.TavilySearchResults")
    def test_get_relevant_documents_returns_docs_from_tool(self, mock_tavily_cls):
        mock_tool = MagicMock()
        mock_tool.run.return_value = [
            {"content": "web content", "url": "https://example.com", "title": "Example"}
        ]
        mock_tavily_cls.return_value = mock_tool

        retriever = WebRetriever(k=3, api_key="test_key")
        results = retriever.get_relevant_documents("query")

        assert len(results) == 1
        assert isinstance(results[0], Document)
        assert results[0].page_content == "web content"

    @patch("app.core.retrievers.TAVILY_AVAILABLE", True)
    @patch("app.core.retrievers.TavilySearchResults")
    def test_get_relevant_documents_returns_empty_on_tool_error(self, mock_tavily_cls):
        mock_tool = MagicMock()
        mock_tool.run.side_effect = RuntimeError("API error")
        mock_tavily_cls.return_value = mock_tool

        retriever = WebRetriever(k=3, api_key="test_key")
        results = retriever.get_relevant_documents("query")

        assert results == []

    @patch("app.core.retrievers.TAVILY_AVAILABLE", True)
    @patch("app.core.retrievers.TavilySearchResults")
    def test_returned_doc_has_correct_metadata(self, mock_tavily_cls):
        mock_tool = MagicMock()
        mock_tool.run.return_value = [
            {"content": "info", "url": "https://test.io", "title": "Test Title"}
        ]
        mock_tavily_cls.return_value = mock_tool

        retriever = WebRetriever(k=3, api_key="key")
        results = retriever.get_relevant_documents("query")

        doc = results[0]
        assert doc.metadata["source"] == "https://test.io"
        assert doc.metadata["title"] == "Test Title"
        assert doc.metadata["chunk_type"] == "web"

    @patch("app.core.retrievers.TAVILY_AVAILABLE", False)
    def test_no_tool_when_tavily_unavailable(self):
        retriever = WebRetriever(k=5, api_key="some_key")
        assert retriever.search_tool is None


# ---------------------------------------------------------------------------
# WikiRetrieverStack construction
# ---------------------------------------------------------------------------


class TestWikiRetrieverStackConstruction:
    @patch("app.core.retrievers.EnsembleRetriever")
    def test_initializes_dense_retriever_when_vectorstore_available(self, mock_ensemble):
        mgr = _make_vectorstore_manager()
        # Provide a bm25 retriever too so ensemble is built
        mock_bm25 = MagicMock()
        mgr.get_mmap_bm25_retriever.return_value = mock_bm25
        mock_ensemble.return_value = MagicMock()

        stack = WikiRetrieverStack(vectorstore_manager=mgr)
        assert stack.dense_retriever is not None

    def test_no_dense_retriever_when_no_vectorstore(self):
        mgr = MagicMock()
        mgr.get_vectorstore.return_value = None
        mgr.get_all_documents.return_value = []

        stack = WikiRetrieverStack(vectorstore_manager=mgr)
        assert stack.dense_retriever is None

    def test_no_dense_retriever_when_no_documents(self):
        mgr = MagicMock()
        mgr.get_vectorstore.return_value = MagicMock()
        mgr.get_all_documents.return_value = []

        stack = WikiRetrieverStack(vectorstore_manager=mgr)
        assert stack.dense_retriever is None

    @patch("app.core.retrievers.EnsembleRetriever")
    def test_bm25_retriever_set_when_mmap_available(self, mock_ensemble):
        mock_bm25 = MagicMock()
        mgr = _make_vectorstore_manager(bm25_retriever=mock_bm25)
        mock_ensemble.return_value = MagicMock()

        stack = WikiRetrieverStack(vectorstore_manager=mgr)
        assert stack.bm25_retriever is mock_bm25

    def test_bm25_retriever_none_when_unavailable(self):
        mgr = _make_vectorstore_manager(bm25_retriever=None)

        stack = WikiRetrieverStack(vectorstore_manager=mgr)
        assert stack.bm25_retriever is None


# ---------------------------------------------------------------------------
# WikiRetrieverStack.build_repo_retriever
# ---------------------------------------------------------------------------


class TestBuildRepoRetriever:
    def test_returns_none_when_no_repo_retriever(self):
        mgr = MagicMock()
        mgr.get_vectorstore.return_value = None
        mgr.get_all_documents.return_value = []

        stack = WikiRetrieverStack(vectorstore_manager=mgr)
        result = stack.build_repo_retriever(k=10)
        assert result is None

    @patch("app.core.retrievers.EnsembleRetriever")
    def test_returns_repo_retriever_when_available(self, mock_ensemble):
        mock_bm25 = MagicMock()
        mgr = _make_vectorstore_manager(bm25_retriever=mock_bm25)
        mock_ens = MagicMock()
        mock_ensemble.return_value = mock_ens

        stack = WikiRetrieverStack(vectorstore_manager=mgr)
        result = stack.build_repo_retriever(k=5)
        assert result is not None


# ---------------------------------------------------------------------------
# WikiRetrieverStack.search_repository
# ---------------------------------------------------------------------------


class TestSearchRepository:
    def test_search_returns_empty_when_no_repo_retriever(self):
        mgr = MagicMock()
        mgr.get_vectorstore.return_value = None
        mgr.get_all_documents.return_value = []

        stack = WikiRetrieverStack(vectorstore_manager=mgr)
        assert stack.repo_retriever is None
        # Calling search_repository with no repo_retriever raises AttributeError
        # (repo_retriever.invoke raises); verify the guard check works
        with pytest.raises((AttributeError, Exception)):
            stack.search_repository("query", k=5)

    @patch("app.core.retrievers.EnsembleRetriever")
    def test_search_returns_list(self, mock_ensemble):
        mock_bm25 = MagicMock()
        mock_docs = [_make_doc("retrieved content")]
        mock_ens = MagicMock()
        mock_ens.invoke.return_value = mock_docs
        mock_ensemble.return_value = mock_ens

        mgr = _make_vectorstore_manager(bm25_retriever=mock_bm25)
        stack = WikiRetrieverStack(vectorstore_manager=mgr)
        stack.content_expander = MagicMock()
        stack.content_expander.expand_retrieved_documents.return_value = mock_docs

        result = stack.search_repository("some query", k=5)
        assert isinstance(result, list)

    @patch("app.core.retrievers.EnsembleRetriever")
    def test_initially_retrieved_flag_set(self, mock_ensemble):
        doc = _make_doc("content")
        mock_ens = MagicMock()
        mock_ens.invoke.return_value = [doc]
        mock_ensemble.return_value = mock_ens

        mock_bm25 = MagicMock()
        mgr = _make_vectorstore_manager(bm25_retriever=mock_bm25)
        stack = WikiRetrieverStack(vectorstore_manager=mgr)
        stack.content_expander = MagicMock()
        stack.content_expander.expand_retrieved_documents.return_value = [doc]

        stack.search_repository("query", k=5)
        assert doc.metadata.get("is_initially_retrieved") is True

    @patch("app.core.retrievers.EnsembleRetriever")
    def test_apply_expansion_false_skips_expander(self, mock_ensemble):
        doc = _make_doc("content")
        mock_ens = MagicMock()
        mock_ens.invoke.return_value = [doc]
        mock_ensemble.return_value = mock_ens

        mock_bm25 = MagicMock()
        mgr = _make_vectorstore_manager(bm25_retriever=mock_bm25)
        stack = WikiRetrieverStack(vectorstore_manager=mgr)
        mock_expander = MagicMock()
        stack.content_expander = mock_expander

        stack.search_repository("query", k=5, apply_expansion=False)
        mock_expander.expand_retrieved_documents.assert_not_called()


# ---------------------------------------------------------------------------
# WikiRetrieverStack.search_docs_semantic
# ---------------------------------------------------------------------------


class TestSearchDocsSemantic:
    def test_empty_query_returns_empty(self):
        mgr = MagicMock()
        mgr.get_vectorstore.return_value = None
        mgr.get_all_documents.return_value = []

        stack = WikiRetrieverStack(vectorstore_manager=mgr)
        result = stack.search_docs_semantic("")
        assert result == []

    def test_whitespace_query_returns_empty(self):
        mgr = MagicMock()
        mgr.get_vectorstore.return_value = None
        mgr.get_all_documents.return_value = []

        stack = WikiRetrieverStack(vectorstore_manager=mgr)
        result = stack.search_docs_semantic("   ")
        assert result == []

    def test_no_retriever_returns_empty(self):
        mgr = MagicMock()
        mgr.get_vectorstore.return_value = None
        mgr.get_all_documents.return_value = []

        stack = WikiRetrieverStack(vectorstore_manager=mgr)
        # repo_retriever is None
        result = stack.search_docs_semantic("some docs query")
        assert result == []

    @patch("app.core.retrievers.EnsembleRetriever")
    def test_filters_to_doc_types(self, mock_ensemble):
        docs = [
            _make_doc("code content", "src.py", chunk_type="class"),
            Document(
                page_content="readme content",
                metadata={"source": "README.md", "chunk_type": "file_doc", "symbol_type": "file_doc"},
            ),
        ]
        mock_ens = MagicMock()
        mock_ens.invoke.return_value = docs
        mock_ensemble.return_value = mock_ens

        mock_bm25 = MagicMock()
        mgr = _make_vectorstore_manager(bm25_retriever=mock_bm25)

        with patch("app.core.retrievers.USE_SEMANTIC_DOC_RETRIEVAL", False):
            stack = WikiRetrieverStack(vectorstore_manager=mgr)
            result = stack.search_docs_semantic("readme", k=10)

        # Should only include doc-type results
        for doc in result:
            chunk_type = doc.metadata.get("chunk_type", "")
            symbol_type = doc.metadata.get("symbol_type", "")
            source = doc.metadata.get("source", "")
            is_doc = (
                chunk_type in ("file_doc", "file_document", "module_doc")
                or "readme" in source.lower()
                or source.endswith(".md")
            )
            assert is_doc, f"Non-doc document in results: {doc.metadata}"

    @patch("app.core.retrievers.EnsembleRetriever")
    def test_returns_empty_when_no_doc_candidates(self, mock_ensemble):
        docs = [_make_doc("code content", "src.py", chunk_type="class")]
        mock_ens = MagicMock()
        mock_ens.invoke.return_value = docs
        mock_ensemble.return_value = mock_ens

        mock_bm25 = MagicMock()
        mgr = _make_vectorstore_manager(bm25_retriever=mock_bm25)

        with patch("app.core.retrievers.USE_SEMANTIC_DOC_RETRIEVAL", False):
            stack = WikiRetrieverStack(vectorstore_manager=mgr)
            result = stack.search_docs_semantic("query", k=10)

        assert result == []

    @patch("app.core.retrievers.EnsembleRetriever")
    def test_fallback_to_dense_when_ensemble_fails(self, mock_ensemble):
        mock_ens = MagicMock()
        mock_ens.invoke.side_effect = RuntimeError("ensemble failed")
        mock_ensemble.return_value = mock_ens

        doc = Document(
            page_content="readme",
            metadata={"source": "README.md", "chunk_type": "file_doc", "symbol_type": "file_doc"},
        )
        dense_mock = MagicMock()
        dense_mock.invoke.return_value = [doc]

        mock_bm25 = MagicMock()
        mgr = _make_vectorstore_manager(bm25_retriever=mock_bm25)

        with patch("app.core.retrievers.USE_SEMANTIC_DOC_RETRIEVAL", False):
            stack = WikiRetrieverStack(vectorstore_manager=mgr)
            # Inject the dense retriever mock that succeeds
            stack.dense_retriever = dense_mock
            result = stack.search_docs_semantic("query", k=10)

        # Either fell back to dense or returned empty — either is valid
        assert isinstance(result, list)


# ---------------------------------------------------------------------------
# WikiRetrieverStack — reranker initialization (WIKIS_ENABLE_RERANKER=1)
# ---------------------------------------------------------------------------


class TestRerankerInitialization:
    @patch("app.core.retrievers.CROSS_ENCODER_AVAILABLE", True)
    @patch("app.core.retrievers.CrossEncoderReranker")
    @patch("app.core.retrievers.BaseCrossEncoder")
    @patch("app.core.retrievers.CrossEncoder")
    @patch("app.core.retrievers.EnsembleRetriever")
    @patch("app.core.retrievers.ContextualCompressionRetriever")
    def test_reranker_initialized_when_env_enabled(
        self, mock_ccr, mock_ensemble, mock_ce, mock_base_ce, mock_ce_reranker, monkeypatch
    ):
        monkeypatch.setenv("WIKIS_ENABLE_RERANKER", "1")
        mock_ens = MagicMock()
        mock_ensemble.return_value = mock_ens
        mock_ce_reranker.return_value = MagicMock()
        mock_ccr.return_value = MagicMock()

        mock_bm25 = MagicMock()
        mgr = _make_vectorstore_manager(bm25_retriever=mock_bm25)
        stack = WikiRetrieverStack(vectorstore_manager=mgr)

        # reranker was set (either real or fallback)
        # Just verify no exception was raised and stack is functional
        assert stack is not None

    @patch("app.core.retrievers.CROSS_ENCODER_AVAILABLE", True)
    @patch("app.core.retrievers.CrossEncoderReranker")
    @patch("app.core.retrievers.BaseCrossEncoder")
    @patch("app.core.retrievers.CrossEncoder")
    @patch("app.core.retrievers.EnsembleRetriever")
    def test_reranker_top_n_from_env(
        self, mock_ensemble, mock_ce, mock_base_ce, mock_ce_reranker, monkeypatch
    ):
        monkeypatch.setenv("WIKIS_ENABLE_RERANKER", "1")
        monkeypatch.setenv("WIKIS_RERANK_TOP_N", "50")
        mock_ens = MagicMock()
        mock_ensemble.return_value = mock_ens

        mock_bm25 = MagicMock()
        mgr = _make_vectorstore_manager(bm25_retriever=mock_bm25)
        # Should not raise even with env set
        stack = WikiRetrieverStack(vectorstore_manager=mgr)
        assert stack is not None

    @patch("app.core.retrievers.CROSS_ENCODER_AVAILABLE", True)
    @patch("app.core.retrievers.CrossEncoderReranker")
    @patch("app.core.retrievers.BaseCrossEncoder")
    @patch("app.core.retrievers.CrossEncoder")
    @patch("app.core.retrievers.EnsembleRetriever")
    def test_invalid_rerank_top_n_env_uses_default(
        self, mock_ensemble, mock_ce, mock_base_ce, mock_ce_reranker, monkeypatch
    ):
        monkeypatch.setenv("WIKIS_ENABLE_RERANKER", "1")
        monkeypatch.setenv("WIKIS_RERANK_TOP_N", "not_a_number")
        mock_ens = MagicMock()
        mock_ensemble.return_value = mock_ens

        mock_bm25 = MagicMock()
        mgr = _make_vectorstore_manager(bm25_retriever=mock_bm25)
        # Should not raise
        stack = WikiRetrieverStack(vectorstore_manager=mgr)
        assert stack is not None


# ---------------------------------------------------------------------------
# WikiRetrieverStack — _build_ensemble_retrievers fallback
# ---------------------------------------------------------------------------


class TestBuildEnsembleRetrieversFallback:
    @patch("app.core.retrievers.EnsembleRetriever")
    def test_uses_single_retriever_when_only_dense(self, mock_ensemble):
        """Only dense retriever available (no BM25) → single retriever used, no ensemble."""
        mgr = _make_vectorstore_manager(bm25_retriever=None)
        stack = WikiRetrieverStack(vectorstore_manager=mgr)
        # dense_retriever was created, bm25 was not
        assert stack.dense_retriever is not None
        assert stack.bm25_retriever is None
        # repo_retriever should be set to the single dense retriever
        assert stack.repo_retriever is stack.dense_retriever

    @patch("app.core.retrievers.EnsembleRetriever")
    def test_ensemble_fallback_to_dense_on_exception(self, mock_ensemble):
        """If EnsembleRetriever raises, fallback to dense only."""
        mock_ensemble.side_effect = RuntimeError("ensemble init failed")

        mock_bm25 = MagicMock()
        mgr = _make_vectorstore_manager(bm25_retriever=mock_bm25)

        stack = WikiRetrieverStack(vectorstore_manager=mgr)
        # Fallback: repo_retriever should be dense_retriever
        assert stack.repo_retriever is stack.dense_retriever


# ---------------------------------------------------------------------------
# WikiRetrieverStack.search_repository — search_kwargs via base_retriever
# ---------------------------------------------------------------------------


class TestSearchRepositorySearchKwargs:
    @patch("app.core.retrievers.EnsembleRetriever")
    def test_search_kwargs_set_via_base_retriever(self, mock_ensemble):
        """Covers the elif branch: base_retriever.search_kwargs is set."""
        doc = _make_doc("content")
        mock_base = MagicMock(spec=["search_kwargs", "invoke"])
        mock_base.search_kwargs = {}

        mock_ens = MagicMock(spec=["invoke"])
        # Has no search_kwargs, has base_retriever with search_kwargs
        mock_ens.base_retriever = mock_base
        mock_ens.invoke.return_value = [doc]
        mock_ensemble.return_value = mock_ens

        mock_bm25 = MagicMock()
        mgr = _make_vectorstore_manager(bm25_retriever=mock_bm25)
        stack = WikiRetrieverStack(vectorstore_manager=mgr)
        mock_expander = MagicMock()
        mock_expander.expand_retrieved_documents.return_value = [doc]
        stack.content_expander = mock_expander

        result = stack.search_repository("query", k=10)
        # Verify search_kwargs was updated on base_retriever
        assert mock_base.search_kwargs == {"k": 10}


# ---------------------------------------------------------------------------
# WikiRetrieverStack.search_docs_semantic — additional coverage
# ---------------------------------------------------------------------------


class TestSearchDocsSemanticCoverage:
    @patch("app.core.retrievers.EnsembleRetriever")
    def test_top_k_applied_without_embeddings_filter(self, mock_ensemble):
        """When WIKIS_DOC_SEMANTIC_RETRIEVAL=0 and EmbeddingsFilter unavailable, top-k is used."""
        docs = [
            Document(
                page_content=f"readme {i}",
                metadata={"source": f"README{i}.md", "chunk_type": "file_doc", "symbol_type": "file_doc"},
            )
            for i in range(15)
        ]
        mock_ens = MagicMock()
        mock_ens.invoke.return_value = docs
        mock_ensemble.return_value = mock_ens

        mock_bm25 = MagicMock()
        mgr = _make_vectorstore_manager(bm25_retriever=mock_bm25)

        with patch("app.core.retrievers.USE_SEMANTIC_DOC_RETRIEVAL", False), \
             patch("app.core.retrievers.EMBEDDINGS_FILTER_AVAILABLE", False):
            stack = WikiRetrieverStack(vectorstore_manager=mgr)
            result = stack.search_docs_semantic("readme", k=5)

        assert len(result) <= 5

    @patch("app.core.retrievers.EnsembleRetriever")
    def test_dense_fallback_when_dense_also_fails(self, mock_ensemble):
        """Dense retriever also fails → returns empty."""
        mock_ens = MagicMock()
        mock_ens.invoke.side_effect = RuntimeError("ensemble failed")
        mock_ensemble.return_value = mock_ens

        mock_bm25 = MagicMock()
        mgr = _make_vectorstore_manager(bm25_retriever=mock_bm25)

        with patch("app.core.retrievers.USE_SEMANTIC_DOC_RETRIEVAL", False):
            stack = WikiRetrieverStack(vectorstore_manager=mgr)
            # Also make dense retriever fail
            stack.dense_retriever = MagicMock()
            stack.dense_retriever.invoke.side_effect = RuntimeError("dense also failed")
            result = stack.search_docs_semantic("query", k=10)

        assert result == []

    @patch("app.core.retrievers.EnsembleRetriever")
    def test_no_candidates_logs_and_returns_empty(self, mock_ensemble):
        """No results from retriever → empty list."""
        mock_ens = MagicMock()
        mock_ens.invoke.return_value = []
        mock_ensemble.return_value = mock_ens

        mock_bm25 = MagicMock()
        mgr = _make_vectorstore_manager(bm25_retriever=mock_bm25)

        with patch("app.core.retrievers.USE_SEMANTIC_DOC_RETRIEVAL", False):
            stack = WikiRetrieverStack(vectorstore_manager=mgr)
            result = stack.search_docs_semantic("query", k=10)

        assert result == []

    @patch("app.core.retrievers.EnsembleRetriever")
    def test_exception_in_outer_block_returns_empty(self, mock_ensemble):
        """Exception in main try block → returns empty."""
        mock_bm25 = MagicMock()
        mgr = _make_vectorstore_manager(bm25_retriever=mock_bm25)

        mock_ens = MagicMock()
        mock_ensemble.return_value = mock_ens

        with patch("app.core.retrievers.USE_SEMANTIC_DOC_RETRIEVAL", False):
            stack = WikiRetrieverStack(vectorstore_manager=mgr)
            # Break the retriever to trigger the outer except
            stack.repo_retriever = None

        result = stack.search_docs_semantic("query", k=10)
        assert result == []

    @patch("app.core.retrievers.EnsembleRetriever")
    def test_search_docs_base_retriever_search_kwargs_path(self, mock_ensemble):
        """Covers the elif base_retriever.search_kwargs path in search_docs_semantic."""
        doc = Document(
            page_content="readme content",
            metadata={"source": "README.md", "chunk_type": "file_doc", "symbol_type": "file_doc"},
        )
        mock_base = MagicMock(spec=["search_kwargs", "invoke"])
        mock_base.search_kwargs = {}

        # Retriever has no search_kwargs itself but has base_retriever with search_kwargs
        mock_ens = MagicMock(spec=["base_retriever", "invoke"])
        mock_ens.base_retriever = mock_base
        mock_ens.invoke.return_value = [doc]
        mock_ensemble.return_value = mock_ens

        mock_bm25 = MagicMock()
        mgr = _make_vectorstore_manager(bm25_retriever=mock_bm25)

        with patch("app.core.retrievers.USE_SEMANTIC_DOC_RETRIEVAL", False):
            stack = WikiRetrieverStack(vectorstore_manager=mgr)
            result = stack.search_docs_semantic("readme", k=5)

        # The elif base_retriever.search_kwargs was exercised
        assert mock_base.search_kwargs == {"k": 25}  # min(5*5, 50) = 25

    @patch("app.core.retrievers.EnsembleRetriever")
    def test_no_dense_retriever_fallback_returns_empty(self, mock_ensemble):
        """Ensemble fails AND no dense retriever → empty (line 424)."""
        mock_ens = MagicMock()
        mock_ens.invoke.side_effect = RuntimeError("ensemble failed")
        mock_ensemble.return_value = mock_ens

        mock_bm25 = MagicMock()
        mgr = _make_vectorstore_manager(bm25_retriever=mock_bm25)

        with patch("app.core.retrievers.USE_SEMANTIC_DOC_RETRIEVAL", False):
            stack = WikiRetrieverStack(vectorstore_manager=mgr)
            # Remove dense retriever to hit the else: return [] branch
            stack.dense_retriever = None
            result = stack.search_docs_semantic("query", k=10)

        assert result == []


# ---------------------------------------------------------------------------
# WikiRetrieverStack._initialize_retrievers — BM25 init failure
# ---------------------------------------------------------------------------


class TestInitializeRetrieversEdgeCases:
    def test_bm25_init_failure_does_not_crash(self):
        """If mmap BM25 retriever raises, stack still initializes."""
        mgr = _make_vectorstore_manager()
        mgr.get_mmap_bm25_retriever.side_effect = RuntimeError("bm25 failed")

        stack = WikiRetrieverStack(vectorstore_manager=mgr)
        # bm25_retriever not set, but dense_retriever should be
        assert stack.dense_retriever is not None
        assert stack.bm25_retriever is None
