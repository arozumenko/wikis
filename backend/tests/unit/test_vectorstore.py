"""Unit tests for app/core/vectorstore.py — mocks FAISS and disk I/O."""

import json
import os
import pickle
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from langchain_core.documents import Document

from app.core.vectorstore import DummyEmbeddings, VectorStoreManager


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_embeddings() -> DummyEmbeddings:
    return DummyEmbeddings()


def _make_docs(count: int = 3, content_prefix: str = "doc") -> list[Document]:
    return [
        Document(
            page_content=f"{content_prefix} content {i}",
            metadata={"uuid": f"uuid-{i}", "source": f"file_{i}.py"},
        )
        for i in range(count)
    ]


def _make_manager(tmp_path: Path) -> VectorStoreManager:
    return VectorStoreManager(cache_dir=str(tmp_path), embeddings=_make_embeddings())


# ---------------------------------------------------------------------------
# DummyEmbeddings
# ---------------------------------------------------------------------------


class TestDummyEmbeddings:
    def test_embed_documents_returns_list_of_vectors(self):
        emb = DummyEmbeddings()
        result = emb.embed_documents(["hello", "world"])
        assert len(result) == 2
        assert all(isinstance(v, list) for v in result)

    def test_embed_documents_vector_length_is_384(self):
        emb = DummyEmbeddings()
        result = emb.embed_documents(["any text"])
        assert len(result[0]) == 384

    def test_embed_query_returns_vector_of_length_384(self):
        emb = DummyEmbeddings()
        result = emb.embed_query("some query")
        assert len(result) == 384

    def test_embed_query_all_zeros(self):
        emb = DummyEmbeddings()
        result = emb.embed_query("irrelevant")
        assert all(v == 0.0 for v in result)


# ---------------------------------------------------------------------------
# VectorStoreManager construction
# ---------------------------------------------------------------------------


class TestVectorStoreManagerConstruction:
    def test_raises_when_no_embeddings(self, tmp_path):
        with pytest.raises(ValueError, match="embeddings must be provided"):
            VectorStoreManager(cache_dir=str(tmp_path), embeddings=None)

    def test_creates_cache_dir(self, tmp_path):
        cache_dir = tmp_path / "sub" / "dir"
        VectorStoreManager(cache_dir=str(cache_dir), embeddings=_make_embeddings())
        assert cache_dir.exists()

    def test_default_cache_dir_used_when_none(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        mgr = VectorStoreManager(embeddings=_make_embeddings())
        assert mgr.cache_dir is not None

    def test_embedding_batch_size_default(self, tmp_path):
        mgr = _make_manager(tmp_path)
        assert mgr.embedding_batch_size == 20

    def test_embedding_batch_size_custom(self, tmp_path):
        mgr = VectorStoreManager(
            cache_dir=str(tmp_path),
            embeddings=_make_embeddings(),
            embedding_batch_size=50,
        )
        assert mgr.embedding_batch_size == 50

    def test_embedding_batch_size_from_env(self, tmp_path, monkeypatch):
        monkeypatch.setenv("WIKI_EMBED_BATCH_SIZE", "64")
        mgr = VectorStoreManager(cache_dir=str(tmp_path), embeddings=_make_embeddings())
        assert mgr.embedding_batch_size == 64

    def test_invalid_env_batch_size_uses_default(self, tmp_path, monkeypatch):
        monkeypatch.setenv("WIKI_EMBED_BATCH_SIZE", "not_a_number")
        mgr = VectorStoreManager(cache_dir=str(tmp_path), embeddings=_make_embeddings())
        assert mgr.embedding_batch_size == 20


# ---------------------------------------------------------------------------
# _generate_repo_hash
# ---------------------------------------------------------------------------


class TestGenerateRepoHash:
    def test_same_path_same_hash(self, tmp_path):
        mgr = _make_manager(tmp_path)
        h1 = mgr._generate_repo_hash("/repo/path", [], "abc123")
        h2 = mgr._generate_repo_hash("/repo/path", [], "abc123")
        assert h1 == h2

    def test_different_commit_different_hash(self, tmp_path):
        mgr = _make_manager(tmp_path)
        h1 = mgr._generate_repo_hash("/repo/path", [], "abc123")
        h2 = mgr._generate_repo_hash("/repo/path", [], "def456")
        assert h1 != h2

    def test_different_path_different_hash(self, tmp_path):
        mgr = _make_manager(tmp_path)
        h1 = mgr._generate_repo_hash("/repo/a", [], "abc")
        h2 = mgr._generate_repo_hash("/repo/b", [], "abc")
        assert h1 != h2

    def test_hash_is_hex_string(self, tmp_path):
        mgr = _make_manager(tmp_path)
        h = mgr._generate_repo_hash("/repo", [], "abc")
        int(h.split("_")[0], 16)  # raises ValueError if not hex


# ---------------------------------------------------------------------------
# has_cache
# ---------------------------------------------------------------------------


class TestHasCache:
    def test_returns_false_when_no_files(self, tmp_path):
        mgr = _make_manager(tmp_path)
        assert mgr.has_cache("nonexistent") is False

    def test_returns_false_when_only_faiss_file(self, tmp_path):
        mgr = _make_manager(tmp_path)
        (tmp_path / "mykey.faiss").write_bytes(b"fake")
        assert mgr.has_cache("mykey") is False

    def test_returns_true_with_faiss_and_legacy_pkl(self, tmp_path):
        mgr = _make_manager(tmp_path)
        (tmp_path / "mykey.faiss").write_bytes(b"fake")
        (tmp_path / "mykey.docs.pkl").write_bytes(b"fake")
        assert mgr.has_cache("mykey") is True

    def test_returns_true_with_faiss_and_lazy_docstore(self, tmp_path):
        mgr = _make_manager(tmp_path)
        (tmp_path / "mykey.faiss").write_bytes(b"fake")
        (tmp_path / "mykey.docstore.bin").write_bytes(b"fake")
        (tmp_path / "mykey.doc_index.json").write_text("{}", encoding="utf-8")
        assert mgr.has_cache("mykey") is True


# ---------------------------------------------------------------------------
# register_cache / _load_cache_index / _save_cache_index
# ---------------------------------------------------------------------------


class TestCacheIndex:
    def test_register_and_load_roundtrip(self, tmp_path):
        mgr = _make_manager(tmp_path)
        mgr.register_cache("owner/repo:main", "abc123hash")

        index = mgr._load_cache_index()
        assert index.get("owner/repo:main") == "abc123hash"

    def test_load_returns_empty_dict_when_missing(self, tmp_path):
        mgr = _make_manager(tmp_path)
        result = mgr._load_cache_index()
        assert result == {}

    def test_register_docstore_cache(self, tmp_path):
        mgr = _make_manager(tmp_path)
        mgr.register_docstore_cache("owner/repo:main", "dshash")

        index = mgr._load_cache_index()
        assert isinstance(index.get("docs"), dict)
        assert index["docs"].get("owner/repo:main") == "dshash"

    def test_malformed_cache_index_returns_empty(self, tmp_path):
        mgr = _make_manager(tmp_path)
        (tmp_path / "cache_index.json").write_text("INVALID JSON", encoding="utf-8")
        result = mgr._load_cache_index()
        assert result == {}


# ---------------------------------------------------------------------------
# load_by_cache_key
# ---------------------------------------------------------------------------


class TestLoadByCacheKey:
    def test_returns_false_when_no_files(self, tmp_path):
        mgr = _make_manager(tmp_path)
        assert mgr.load_by_cache_key("nonexistent") is False

    def test_returns_false_when_faiss_missing(self, tmp_path):
        mgr = _make_manager(tmp_path)
        (tmp_path / "k.docs.pkl").write_bytes(b"fake")
        assert mgr.load_by_cache_key("k") is False


# ---------------------------------------------------------------------------
# get_vectorstore / get_all_documents
# ---------------------------------------------------------------------------


class TestGetVectorstoreAndDocuments:
    def test_get_vectorstore_returns_none_before_load(self, tmp_path):
        mgr = _make_manager(tmp_path)
        assert mgr.get_vectorstore() is None

    def test_get_all_documents_returns_empty_before_load(self, tmp_path):
        mgr = _make_manager(tmp_path)
        assert mgr.get_all_documents() == []

    def test_get_vectorstore_returns_set_value(self, tmp_path):
        mgr = _make_manager(tmp_path)
        fake_vs = MagicMock()
        mgr.vectorstore = fake_vs
        assert mgr.get_vectorstore() is fake_vs

    def test_get_all_documents_returns_set_list(self, tmp_path):
        mgr = _make_manager(tmp_path)
        docs = _make_docs(2)
        mgr.documents = docs
        assert mgr.get_all_documents() == docs


# ---------------------------------------------------------------------------
# _build_and_save (with FAISS mocked)
# ---------------------------------------------------------------------------


class TestBuildAndSave:
    def test_raises_on_empty_documents(self, tmp_path):
        mgr = _make_manager(tmp_path)
        cache_file = tmp_path / "k.faiss"
        docs_file = tmp_path / "k.docs.pkl"
        with pytest.raises(ValueError, match="No documents to index"):
            mgr._build_and_save([], cache_file, docs_file)

    def test_raises_when_all_docs_empty_content(self, tmp_path):
        mgr = _make_manager(tmp_path)
        docs = [Document(page_content="   ", metadata={})]
        cache_file = tmp_path / "k.faiss"
        docs_file = tmp_path / "k.docs.pkl"
        with pytest.raises(ValueError, match="No valid documents"):
            mgr._build_and_save(docs, cache_file, docs_file)

    @patch("app.core.vectorstore.FAISS.from_texts")
    def test_builds_vector_store_with_valid_docs(self, mock_from_texts, tmp_path):
        fake_vs = MagicMock()
        fake_vs.save_local = MagicMock()
        mock_from_texts.return_value = fake_vs

        mgr = _make_manager(tmp_path)
        docs = _make_docs(2)
        cache_file = tmp_path / "k.faiss"
        docs_file = tmp_path / "k.docs.pkl"

        vs, returned_docs = mgr._build_and_save(docs, cache_file, docs_file)

        assert vs is fake_vs
        assert len(returned_docs) == 2

    @patch("app.core.vectorstore.FAISS.from_texts")
    def test_assigns_uuids_to_docs_without_them(self, mock_from_texts, tmp_path):
        fake_vs = MagicMock()
        fake_vs.save_local = MagicMock()
        mock_from_texts.return_value = fake_vs

        mgr = _make_manager(tmp_path)
        docs = [Document(page_content="some content", metadata={})]
        cache_file = tmp_path / "k.faiss"
        docs_file = tmp_path / "k.docs.pkl"

        mgr._build_and_save(docs, cache_file, docs_file)

        assert "uuid" in docs[0].metadata
        assert docs[0].metadata["uuid"] != ""

    @patch("app.core.vectorstore.FAISS.from_texts")
    def test_filters_empty_content_documents(self, mock_from_texts, tmp_path):
        fake_vs = MagicMock()
        fake_vs.save_local = MagicMock()
        mock_from_texts.return_value = fake_vs

        mgr = _make_manager(tmp_path)
        docs = [
            Document(page_content="valid content", metadata={"uuid": "v1"}),
            Document(page_content="", metadata={"uuid": "v2"}),
        ]
        cache_file = tmp_path / "k.faiss"
        docs_file = tmp_path / "k.docs.pkl"

        _, returned_docs = mgr._build_and_save(docs, cache_file, docs_file)
        assert len(returned_docs) == 1
        assert returned_docs[0].metadata["uuid"] == "v1"


# ---------------------------------------------------------------------------
# get_mmap_bm25_retriever
# ---------------------------------------------------------------------------


class TestGetMmapBm25Retriever:
    def test_returns_none_when_no_cache_key(self, tmp_path):
        mgr = _make_manager(tmp_path)
        result = mgr.get_mmap_bm25_retriever()
        assert result is None

    def test_returns_none_when_no_bm25_file(self, tmp_path):
        mgr = _make_manager(tmp_path)
        mgr.cache_key = "nonexistent"
        result = mgr.get_mmap_bm25_retriever()
        assert result is None


# ---------------------------------------------------------------------------
# _maybe_copy_documents
# ---------------------------------------------------------------------------


class TestMaybeCopyDocuments:
    def test_returns_list_without_deepcopy_by_default(self, tmp_path):
        mgr = _make_manager(tmp_path)
        docs = _make_docs(2)
        result = mgr._maybe_copy_documents(docs)
        assert result == docs

    def test_returns_deepcopy_when_env_set(self, tmp_path, monkeypatch):
        monkeypatch.setenv("WIKIS_DEEPCOPY_DOCS", "1")
        mgr = _make_manager(tmp_path)
        docs = _make_docs(2)
        result = mgr._maybe_copy_documents(docs)
        assert result == docs
        assert result is not docs  # deepcopy returns new list


# ---------------------------------------------------------------------------
# _rebuild_document_mapping / _save_document_mapping / _load_document_mapping
# ---------------------------------------------------------------------------


class TestDocumentMapping:
    def test_rebuild_creates_uuid_index_map(self, tmp_path):
        mgr = _make_manager(tmp_path)
        docs = _make_docs(3)
        mgr.documents = docs
        mgr._rebuild_document_mapping()
        for i, doc in enumerate(docs):
            assert mgr.document_ids[doc.metadata["uuid"]] == i

    def test_rebuild_skips_docs_without_uuid(self, tmp_path):
        mgr = _make_manager(tmp_path)
        mgr.documents = [Document(page_content="no uuid", metadata={})]
        mgr._rebuild_document_mapping()
        assert mgr.document_ids == {}

    def test_save_and_load_mapping_roundtrip(self, tmp_path):
        mgr = _make_manager(tmp_path)
        mgr.documents = _make_docs(3)
        mgr._rebuild_document_mapping()
        mgr._save_document_mapping(tmp_path, "k")

        mgr2 = _make_manager(tmp_path)
        result = mgr2._load_document_mapping(tmp_path, "k")
        assert result is True
        assert mgr2.document_ids == mgr.document_ids

    def test_load_returns_false_when_file_missing(self, tmp_path):
        mgr = _make_manager(tmp_path)
        result = mgr._load_document_mapping(tmp_path, "absent_key")
        assert result is False

    def test_save_handles_write_error_gracefully(self, tmp_path, monkeypatch):
        mgr = _make_manager(tmp_path)
        mgr.document_ids = {"uuid1": 0}
        # Make path unwritable by using a file where a dir is expected
        (tmp_path / "bad_key.docids.json").mkdir()
        # Should not raise
        mgr._save_document_mapping(tmp_path, "bad_key")


# ---------------------------------------------------------------------------
# add_documents
# ---------------------------------------------------------------------------


class TestAddDocuments:
    def test_raises_when_vectorstore_not_initialized(self, tmp_path):
        mgr = _make_manager(tmp_path)
        with pytest.raises(ValueError, match="Vector store not initialized"):
            mgr.add_documents(_make_docs(1))

    def test_adds_documents_and_returns_ids(self, tmp_path):
        mgr = _make_manager(tmp_path)
        fake_vs = MagicMock()
        fake_vs.add_texts.return_value = None
        mgr.vectorstore = fake_vs
        mgr.documents = _make_docs(2)
        mgr.document_ids = {}

        new_docs = [Document(page_content="new content", metadata={"uuid": "new-1"})]
        result = mgr.add_documents(new_docs)

        assert result == ["new-1"]
        assert len(mgr.documents) == 3

    def test_assigns_uuid_to_new_docs(self, tmp_path):
        mgr = _make_manager(tmp_path)
        fake_vs = MagicMock()
        mgr.vectorstore = fake_vs
        mgr.documents = []
        mgr.document_ids = {}

        new_docs = [Document(page_content="content without uuid", metadata={})]
        ids = mgr.add_documents(new_docs)

        assert len(ids) == 1
        assert ids[0] != ""
        assert ids[0] == new_docs[0].metadata["uuid"]


# ---------------------------------------------------------------------------
# delete_documents
# ---------------------------------------------------------------------------


class TestDeleteDocuments:
    def test_raises_when_vectorstore_not_initialized(self, tmp_path):
        mgr = _make_manager(tmp_path)
        with pytest.raises(ValueError, match="Vector store not initialized"):
            mgr.delete_documents(["some-uuid"])

    def test_does_nothing_when_no_matching_docs(self, tmp_path):
        mgr = _make_manager(tmp_path)
        fake_vs = MagicMock()
        fake_vs.index_to_docstore_id = {0: "docstore-1"}
        mgr.vectorstore = fake_vs
        mgr.documents = [Document(page_content="doc", metadata={"uuid": "uuid-1"})]
        mgr.document_ids = {"uuid-1": 0}

        mgr.delete_documents(["nonexistent-uuid"])
        fake_vs.delete.assert_not_called()

    def test_deletes_matching_document(self, tmp_path):
        mgr = _make_manager(tmp_path)
        fake_vs = MagicMock()
        fake_vs.index_to_docstore_id = {0: "docstore-id-1", 1: "docstore-id-2"}
        mgr.vectorstore = fake_vs
        mgr.documents = [
            Document(page_content="doc0", metadata={"uuid": "uuid-0"}),
            Document(page_content="doc1", metadata={"uuid": "uuid-1"}),
        ]
        mgr.document_ids = {"uuid-0": 0, "uuid-1": 1}

        mgr.delete_documents(["uuid-0"])
        fake_vs.delete.assert_called_once_with(["docstore-id-1"])
        assert len(mgr.documents) == 1
        assert mgr.documents[0].metadata["uuid"] == "uuid-1"


# ---------------------------------------------------------------------------
# search / search_with_score / search_by_type / as_retriever
# ---------------------------------------------------------------------------


class TestSearchMethods:
    def test_search_returns_empty_when_no_vectorstore(self, tmp_path):
        mgr = _make_manager(tmp_path)
        result = mgr.search("query")
        assert result == []

    def test_search_calls_similarity_search(self, tmp_path):
        mgr = _make_manager(tmp_path)
        fake_vs = MagicMock()
        fake_vs.similarity_search.return_value = [_make_docs(1)[0]]
        mgr.vectorstore = fake_vs

        result = mgr.search("query", k=5)
        fake_vs.similarity_search.assert_called_once_with("query", k=5)
        assert len(result) == 1

    def test_search_with_filter(self, tmp_path):
        mgr = _make_manager(tmp_path)
        fake_vs = MagicMock()
        fake_vs.similarity_search.return_value = []
        mgr.vectorstore = fake_vs

        mgr.search("query", filter_dict={"chunk_type": {"$eq": "code"}})
        fake_vs.similarity_search.assert_called_once()
        call_kwargs = fake_vs.similarity_search.call_args[1]
        assert "filter" in call_kwargs

    def test_search_with_score_returns_empty_when_no_vectorstore(self, tmp_path):
        mgr = _make_manager(tmp_path)
        result = mgr.search_with_score("query")
        assert result == []

    def test_search_with_score_returns_float_scores(self, tmp_path):
        import numpy as np

        mgr = _make_manager(tmp_path)
        fake_vs = MagicMock()
        fake_vs.similarity_search_with_score.return_value = [
            (_make_docs(1)[0], np.float32(0.85))
        ]
        mgr.vectorstore = fake_vs

        results = mgr.search_with_score("query", k=3)
        assert len(results) == 1
        doc, score = results[0]
        assert isinstance(score, float)
        assert score == pytest.approx(0.85)

    def test_search_by_type_passes_filter(self, tmp_path):
        mgr = _make_manager(tmp_path)
        fake_vs = MagicMock()
        fake_vs.similarity_search.return_value = []
        mgr.vectorstore = fake_vs

        mgr.search_by_type("query", "code")
        fake_vs.similarity_search.assert_called_once()

    def test_as_retriever_raises_when_no_vectorstore(self, tmp_path):
        mgr = _make_manager(tmp_path)
        with pytest.raises(ValueError, match="Vector store not initialized"):
            mgr.as_retriever()

    def test_as_retriever_returns_retriever(self, tmp_path):
        mgr = _make_manager(tmp_path)
        fake_vs = MagicMock()
        fake_retriever = MagicMock()
        fake_vs.as_retriever.return_value = fake_retriever
        mgr.vectorstore = fake_vs

        result = mgr.as_retriever()
        assert result is fake_retriever


# ---------------------------------------------------------------------------
# clear_cache
# ---------------------------------------------------------------------------


class TestClearCache:
    def test_clear_specific_cache(self, tmp_path):
        mgr = _make_manager(tmp_path)
        # Create fake cache files
        repo_hash = mgr._generate_repo_hash("/repo/path", [], "commit")
        faiss_file = tmp_path / f"{repo_hash}.faiss"
        docs_file = tmp_path / f"{repo_hash}.docs.pkl"
        faiss_file.write_bytes(b"fake")
        docs_file.write_bytes(b"fake")

        mgr.clear_cache(repo_path="/repo/path", commit_hash="commit")

        assert not faiss_file.exists()
        assert not docs_file.exists()

    def test_clear_all_cache(self, tmp_path):
        mgr = _make_manager(tmp_path)
        # Create fake cache files
        (tmp_path / "abc.faiss").write_bytes(b"fake")
        (tmp_path / "abc.docs.pkl").write_bytes(b"fake")

        mgr.clear_cache()

        assert not (tmp_path / "abc.faiss").exists()
        assert not (tmp_path / "abc.docs.pkl").exists()


# ---------------------------------------------------------------------------
# get_mmap_bm25_retriever (env flag)
# ---------------------------------------------------------------------------


class TestGetMmapBm25RetrieverEnv:
    def test_returns_none_when_env_disabled(self, tmp_path, monkeypatch):
        monkeypatch.setenv("WIKIS_MMAP_BM25", "0")
        mgr = _make_manager(tmp_path)
        mgr.cache_key = "somekey"
        result = mgr.get_mmap_bm25_retriever()
        assert result is None


# ---------------------------------------------------------------------------
# load_by_repo_name
# ---------------------------------------------------------------------------


class TestLoadByRepoName:
    def test_returns_none_when_no_cache_index(self, tmp_path):
        mgr = _make_manager(tmp_path)
        result = mgr.load_by_repo_name("owner/repo:main")
        assert result is None

    def test_uses_single_faiss_file_when_no_index(self, tmp_path):
        mgr = _make_manager(tmp_path)
        # Register a cache to create the index
        mgr.register_cache("owner/repo:main", "abc123")

        # The key exists but no actual faiss/docs files -> load_by_cache_key returns False
        result = mgr.load_by_repo_name("owner/repo:main")
        # load_by_cache_key returns False (files missing) so result is None
        assert result is None

    def test_uses_single_available_faiss_when_no_entry(self, tmp_path):
        """Fallback: if only one .faiss file exists, it's used directly."""
        mgr = _make_manager(tmp_path)
        # No cache index, but one faiss file
        (tmp_path / "onlyone.faiss").write_bytes(b"fake")
        (tmp_path / "onlyone.docs.pkl").write_bytes(b"fake")

        # load_by_repo_name will find single .faiss and try to load it
        # It will fail to deserialize the fake bytes, returning None
        result = mgr.load_by_repo_name("unknown/repo:main")
        assert result is None  # Load failed due to fake faiss content


# ---------------------------------------------------------------------------
# _build_and_save_from_iterable (mocking FAISS internals)
# ---------------------------------------------------------------------------


class TestBuildAndSaveFromIterable:
    @patch("app.core.vectorstore.FAISS")
    @patch("app.core.vectorstore.dependable_faiss_import")
    def test_raises_when_no_valid_documents(self, mock_faiss_import, mock_faiss_cls, tmp_path):
        mgr = _make_manager(tmp_path)
        cache_file = tmp_path / "k.faiss"
        docs_file = tmp_path / "k.docs.pkl"

        # Iterator with only empty-content documents
        docs_iter = iter([Document(page_content="   ", metadata={})])

        mock_faiss_index = MagicMock()
        mock_faiss_import.return_value = mock_faiss_index

        with pytest.raises(ValueError, match="No valid documents"):
            mgr._build_and_save_from_iterable(docs_iter, cache_file, docs_file)

    @patch("app.core.vectorstore.dependable_faiss_import")
    def test_assigns_uuid_to_docs_without_one(self, mock_faiss_import, tmp_path):
        import numpy as np

        mock_faiss_module = MagicMock()
        mock_index = MagicMock()
        mock_faiss_module.IndexFlatL2.return_value = mock_index
        mock_faiss_import.return_value = mock_faiss_module

        mgr = _make_manager(tmp_path)
        # Override embed_documents to return fixed vectors
        mgr.embeddings = MagicMock()
        mgr.embeddings.embed_documents.return_value = [[0.1] * 4]

        doc = Document(page_content="some content", metadata={})
        cache_file = tmp_path / "k.faiss"
        docs_file = tmp_path / "k.docs.pkl"

        with patch("app.core.vectorstore.FAISS") as mock_faiss_cls:
            mock_vs = MagicMock()
            mock_faiss_cls.return_value = mock_vs
            mgr._build_and_save_from_iterable(iter([doc]), cache_file, docs_file)

        assert "uuid" in doc.metadata
        assert doc.metadata["uuid"] != ""

    @patch("app.core.vectorstore.dependable_faiss_import")
    def test_builds_vectorstore_from_stream(self, mock_faiss_import, tmp_path):
        import numpy as np

        mock_faiss_module = MagicMock()
        mock_index = MagicMock()
        mock_faiss_module.IndexFlatL2.return_value = mock_index
        mock_faiss_import.return_value = mock_faiss_module

        mgr = _make_manager(tmp_path)
        mgr.embeddings = MagicMock()
        mgr.embeddings.embed_documents.return_value = [[0.1] * 4, [0.2] * 4]

        docs = [
            Document(page_content="content one", metadata={"uuid": "u1"}),
            Document(page_content="content two", metadata={"uuid": "u2"}),
        ]
        cache_file = tmp_path / "k.faiss"
        docs_file = tmp_path / "k.docs.pkl"

        with patch("app.core.vectorstore.FAISS") as mock_faiss_cls:
            mock_vs = MagicMock()
            mock_faiss_cls.return_value = mock_vs
            vs, returned_docs = mgr._build_and_save_from_iterable(iter(docs), cache_file, docs_file)

        assert vs is mock_vs
        assert len(returned_docs) == 2


# ---------------------------------------------------------------------------
# load_or_build_from_iterable
# ---------------------------------------------------------------------------


class TestLoadOrBuildFromIterable:
    def test_builds_when_no_cache(self, tmp_path):
        mgr = _make_manager(tmp_path)
        mgr.embeddings = MagicMock()
        mgr.embeddings.embed_documents.return_value = [[0.1] * 4]

        docs = [Document(page_content="hello world", metadata={"uuid": "u1"})]

        import numpy as np

        with patch("app.core.vectorstore.dependable_faiss_import") as mock_fi, \
             patch("app.core.vectorstore.FAISS") as mock_faiss_cls:
            mock_fm = MagicMock()
            mock_index = MagicMock()
            mock_fm.IndexFlatL2.return_value = mock_index
            mock_fi.return_value = mock_fm
            mock_vs = MagicMock()
            mock_faiss_cls.return_value = mock_vs

            vs, returned_docs = mgr.load_or_build_from_iterable(
                iter(docs), repo_path="/test/repo", commit_hash="abc"
            )

        assert vs is mock_vs


# ---------------------------------------------------------------------------
# _load_from_cache — lazy docstore path
# ---------------------------------------------------------------------------


class TestLoadFromCache:
    @patch("app.core.vectorstore.FAISS")
    @patch("app.core.vectorstore.dependable_faiss_import")
    def test_loads_with_lazy_docstore(self, mock_fi, mock_faiss_cls, tmp_path):
        """Exercises the lazy docstore path in _load_from_cache."""
        from app.core.docstore import build_docstore_cache

        # Build a real docstore
        docs = _make_docs(2)
        build_docstore_cache(docs, tmp_path, "testkey")

        # Create a fake faiss file
        faiss_file = tmp_path / "testkey.faiss"
        faiss_file.write_bytes(b"fake")
        docs_file = tmp_path / "testkey.docs.pkl"

        mgr = _make_manager(tmp_path)

        # Mock faiss import to return an index
        mock_faiss_module = MagicMock()
        mock_index = MagicMock()
        mock_faiss_module.read_index.return_value = mock_index
        mock_fi.return_value = mock_faiss_module

        # Mock FAISS constructor
        mock_vs = MagicMock()
        mock_faiss_cls.return_value = mock_vs

        with patch("app.core.vectorstore.LAZY_DOCSTORE_ENABLED", True):
            vs, loaded_docs = mgr._load_from_cache(faiss_file, docs_file)

        assert vs is mock_vs
        assert len(loaded_docs) == 2

    @patch("app.core.vectorstore.LAZY_DOCSTORE_ENABLED", False)
    @patch("app.core.vectorstore.FAISS.load_local")
    def test_loads_legacy_pickle_path(self, mock_load_local, tmp_path):
        """Exercises the legacy pickle path in _load_from_cache."""
        import pickle

        docs = _make_docs(2)
        docs_file = tmp_path / "testkey.docs.pkl"
        with open(docs_file, "wb") as f:
            pickle.dump(docs, f)

        fake_vs = MagicMock()
        mock_load_local.return_value = fake_vs

        mgr = _make_manager(tmp_path)
        faiss_file = tmp_path / "testkey.faiss"
        faiss_file.write_bytes(b"content")

        vs, loaded_docs = mgr._load_from_cache(faiss_file, docs_file)

        assert vs is fake_vs
        assert len(loaded_docs) == 2


# ---------------------------------------------------------------------------
# _build_and_save — large batch path (>batch_size documents)
# ---------------------------------------------------------------------------


class TestBuildAndSaveLargeBatch:
    @patch("app.core.vectorstore.FAISS.from_texts")
    def test_large_batch_embedding_calls_add_texts(self, mock_from_texts, tmp_path):
        """When docs > batch_size, uses batched embedding (add_texts calls)."""
        fake_vs = MagicMock()
        fake_vs.save_local = MagicMock()
        fake_vs.add_texts = MagicMock()
        mock_from_texts.return_value = fake_vs

        mgr = VectorStoreManager(
            cache_dir=str(tmp_path),
            embeddings=_make_embeddings(),
            embedding_batch_size=2,  # Small batch
        )

        # Create 5 docs (more than batch_size=2)
        docs = [Document(page_content=f"content {i}", metadata={"uuid": f"u{i}"}) for i in range(5)]
        cache_file = tmp_path / "k.faiss"
        docs_file = tmp_path / "k.docs.pkl"

        vs, returned_docs = mgr._build_and_save(docs, cache_file, docs_file)

        # add_texts should have been called for subsequent batches
        assert fake_vs.add_texts.called or mock_from_texts.call_count > 1
        assert len(returned_docs) == 5

    @patch("app.core.vectorstore.FAISS.from_texts")
    def test_content_truncation_when_exceeds_max_chars(self, mock_from_texts, tmp_path, monkeypatch):
        """Content longer than WIKI_MAX_EMBED_CHARS gets truncated."""
        monkeypatch.setenv("WIKI_MAX_EMBED_CHARS", "10")
        fake_vs = MagicMock()
        fake_vs.save_local = MagicMock()
        mock_from_texts.return_value = fake_vs

        mgr = _make_manager(tmp_path)
        docs = [Document(page_content="a" * 100, metadata={"uuid": "long-1"})]
        cache_file = tmp_path / "k.faiss"
        docs_file = tmp_path / "k.docs.pkl"

        mgr._build_and_save(docs, cache_file, docs_file)

        # Verify from_texts was called with truncated content
        called_texts = mock_from_texts.call_args[1]["texts"]
        assert len(called_texts[0]) <= 10


# ---------------------------------------------------------------------------
# register_cache — commit-scoped identifier path
# ---------------------------------------------------------------------------


class TestRegisterCacheCommitScoped:
    def test_register_cache_with_commit_scoped_id(self, tmp_path):
        """Exercises split_repo_identifier path in register_cache."""
        mgr = _make_manager(tmp_path)

        with patch("app.core.vectorstore.VectorStoreManager.register_cache", wraps=mgr.register_cache):
            # Use a commit-scoped identifier format
            mgr.register_cache("owner/repo:main:abc12345", "cachehash")

        index = mgr._load_cache_index()
        assert index.get("owner/repo:main:abc12345") == "cachehash"


# ---------------------------------------------------------------------------
# _generate_repo_hash — mtime fallback (no commit hash)
# ---------------------------------------------------------------------------


class TestGenerateRepoHashMtime:
    def test_uses_mtime_when_no_commit_hash(self, tmp_path):
        """When commit_hash is None, falls back to mtime."""
        mgr = _make_manager(tmp_path)
        # Create a real directory so getmtime works
        test_dir = tmp_path / "repo"
        test_dir.mkdir()
        h = mgr._generate_repo_hash(str(test_dir), [], None)
        # Should return a hex hash
        assert len(h) > 0

    def test_nonexistent_path_no_crash(self, tmp_path):
        """When repo path doesn't exist, OSError is silently ignored."""
        mgr = _make_manager(tmp_path)
        h = mgr._generate_repo_hash("/nonexistent/path/xyz", [], None)
        assert len(h) > 0


# ---------------------------------------------------------------------------
# load_or_build — cache hit path
# ---------------------------------------------------------------------------


class TestLoadOrBuildCacheHit:
    @patch("app.core.vectorstore.LAZY_DOCSTORE_ENABLED", False)
    @patch("app.core.vectorstore.FAISS")
    def test_loads_from_cache_when_files_exist(self, mock_faiss_cls, tmp_path):
        """Exercises load_or_build cache hit branch (lines 140-145)."""
        import pickle

        docs = _make_docs(2)
        mgr = _make_manager(tmp_path)
        repo_hash = mgr._generate_repo_hash("/test/repo", docs, "abc")

        # Create cache files
        faiss_file = tmp_path / f"{repo_hash}.faiss"
        docs_file = tmp_path / f"{repo_hash}.docs.pkl"
        faiss_file.write_bytes(b"fake")
        with open(docs_file, "wb") as f:
            pickle.dump(docs, f)

        fake_vs = MagicMock()
        mock_faiss_cls.load_local.return_value = fake_vs

        vs, loaded_docs = mgr.load_or_build(docs, repo_path="/test/repo", commit_hash="abc")

        assert vs is fake_vs
        assert len(loaded_docs) == 2

    @patch("app.core.vectorstore.LAZY_DOCSTORE_ENABLED", False)
    @patch("app.core.vectorstore.FAISS")
    def test_rebuilds_when_cache_load_fails(self, mock_faiss_cls, tmp_path):
        """When cache load fails, falls back to rebuild (lines 144-149)."""
        import pickle

        docs = _make_docs(2)
        mgr = _make_manager(tmp_path)
        repo_hash = mgr._generate_repo_hash("/test/repo", docs, "xyz")

        # Create cache files
        faiss_file = tmp_path / f"{repo_hash}.faiss"
        docs_file = tmp_path / f"{repo_hash}.docs.pkl"
        faiss_file.write_bytes(b"bad content")
        with open(docs_file, "wb") as f:
            pickle.dump(docs, f)

        # Make FAISS.load_local raise to trigger rebuild
        fake_vs = MagicMock()
        fake_vs.save_local = MagicMock()
        # First call (load_local) raises; second (from_texts) returns fake_vs
        mock_faiss_cls.load_local.side_effect = Exception("corrupt cache")
        mock_faiss_cls.from_texts.return_value = fake_vs

        vs, returned_docs = mgr.load_or_build(docs, repo_path="/test/repo", commit_hash="xyz")
        assert vs is fake_vs


# ---------------------------------------------------------------------------
# load_or_build_from_iterable — cache hit path
# ---------------------------------------------------------------------------


class TestLoadOrBuildFromIterableCacheHit:
    @patch("app.core.vectorstore.LAZY_DOCSTORE_ENABLED", False)
    @patch("app.core.vectorstore.FAISS")
    def test_loads_from_cache_when_exists(self, mock_faiss_cls, tmp_path):
        """Exercises load_or_build_from_iterable cache hit branch (lines 163-168)."""
        import pickle

        docs = _make_docs(2)
        mgr = _make_manager(tmp_path)
        repo_hash = mgr._generate_repo_hash("/test/repo", [], "commit1")

        faiss_file = tmp_path / f"{repo_hash}.faiss"
        docs_file = tmp_path / f"{repo_hash}.docs.pkl"
        faiss_file.write_bytes(b"fake")
        with open(docs_file, "wb") as f:
            pickle.dump(docs, f)

        fake_vs = MagicMock()
        mock_faiss_cls.load_local.return_value = fake_vs

        vs, loaded_docs = mgr.load_or_build_from_iterable(
            iter(docs), repo_path="/test/repo", commit_hash="commit1"
        )
        assert vs is fake_vs


# ---------------------------------------------------------------------------
# _load_from_cache with LAZY_DOCSTORE_ENABLED=False (legacy path)
# ---------------------------------------------------------------------------


class TestLoadFromCacheLegacy:
    @patch("app.core.vectorstore.LAZY_DOCSTORE_ENABLED", False)
    @patch("app.core.vectorstore.FAISS")
    def test_loads_via_legacy_pickle(self, mock_faiss_cls, tmp_path):
        import pickle

        docs = _make_docs(2)
        docs_file = tmp_path / "legacykey.docs.pkl"
        with open(docs_file, "wb") as f:
            pickle.dump(docs, f)

        faiss_file = tmp_path / "legacykey.faiss"
        faiss_file.write_bytes(b"fake_content")

        fake_vs = MagicMock()
        mock_faiss_cls.load_local.return_value = fake_vs

        mgr = _make_manager(tmp_path)
        vs, loaded_docs = mgr._load_from_cache(faiss_file, docs_file)

        assert vs is fake_vs
        assert len(loaded_docs) == 2
