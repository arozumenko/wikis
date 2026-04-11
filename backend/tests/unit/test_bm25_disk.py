"""Unit tests for app/core/bm25_disk.py"""

import pickle
import sqlite3
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from langchain_core.documents import Document

from app.core.bm25_disk import (
    DEFAULT_BM25_B,
    DEFAULT_BM25_K1,
    BM25SqliteIndex,
    MMapBM25Retriever,
    _default_tokenizer,
    _initialize_schema,
    build_bm25_index,
    load_or_build_bm25_index,
)
from app.core.docstore import build_docstore_cache


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_sqlite_db(tmp_path: Path, doc_ids: list[str], docs_text: list[str]) -> Path:
    """Build a minimal BM25 SQLite index directly and return its path."""
    db_path = tmp_path / "test.bm25.sqlite"
    conn = sqlite3.connect(str(db_path))
    _initialize_schema(conn)

    total_len = 0
    for idx, (doc_id, text) in enumerate(zip(doc_ids, docs_text)):
        tokens = text.split()
        total_len += len(tokens)
        conn.execute(
            "INSERT INTO docs (doc_idx, doc_id, length) VALUES (?, ?, ?)",
            (idx, doc_id, len(tokens)),
        )
        from collections import Counter

        for term, tf in Counter(tokens).items():
            conn.execute(
                "INSERT INTO postings (term, doc_idx, tf) VALUES (?, ?, ?)",
                (term, idx, tf),
            )

    conn.execute("INSERT INTO terms (term, df) SELECT term, COUNT(*) FROM postings GROUP BY term")
    avgdl = total_len / len(doc_ids) if doc_ids else 1.0
    conn.executemany(
        "INSERT INTO meta (key, value) VALUES (?, ?)",
        [
            ("schema_version", "1"),
            ("doc_count", str(len(doc_ids))),
            ("avgdl", str(avgdl)),
            ("k1", str(DEFAULT_BM25_K1)),
            ("b", str(DEFAULT_BM25_B)),
        ],
    )
    conn.commit()
    conn.close()
    return db_path


def _make_documents(count: int = 3, text_prefix: str = "token") -> list[Document]:
    return [
        Document(
            page_content=f"{text_prefix} word{i} content",
            metadata={"uuid": f"uuid-{i}"},
        )
        for i in range(count)
    ]


# ---------------------------------------------------------------------------
# _default_tokenizer
# ---------------------------------------------------------------------------


class TestDefaultTokenizer:
    def test_splits_on_whitespace(self):
        tokens = _default_tokenizer("hello world foo")
        assert "hello" in tokens
        assert "world" in tokens
        assert "foo" in tokens

    def test_empty_string_returns_empty_list(self):
        tokens = _default_tokenizer("")
        assert tokens == []

    def test_strips_empty_tokens(self):
        tokens = _default_tokenizer("  leading  trailing  ")
        assert "" not in tokens
        assert len(tokens) > 0


# ---------------------------------------------------------------------------
# BM25SqliteIndex.load
# ---------------------------------------------------------------------------


class TestBM25SqliteIndexLoad:
    def test_returns_none_when_db_missing(self, tmp_path):
        result = BM25SqliteIndex.load(tmp_path / "absent.sqlite", _default_tokenizer)
        assert result is None

    def test_loads_valid_db(self, tmp_path):
        db_path = _make_sqlite_db(tmp_path, ["id-1", "id-2"], ["hello world", "foo bar"])
        result = BM25SqliteIndex.load(db_path, _default_tokenizer)

        assert result is not None
        assert result.doc_count == 2

    def test_doc_count_matches_indexed_docs(self, tmp_path):
        db_path = _make_sqlite_db(tmp_path, ["a", "b", "c"], ["alpha", "beta", "gamma"])
        idx = BM25SqliteIndex.load(db_path, _default_tokenizer)
        assert idx.doc_count == 3


# ---------------------------------------------------------------------------
# BM25SqliteIndex.search
# ---------------------------------------------------------------------------


class TestBM25SqliteIndexSearch:
    def _build(self, tmp_path: Path) -> BM25SqliteIndex:
        texts = [
            "python programming language tutorial",
            "javascript web development framework",
            "python data science machine learning",
        ]
        ids = ["doc-0", "doc-1", "doc-2"]
        db_path = _make_sqlite_db(tmp_path, ids, texts)
        return BM25SqliteIndex.load(db_path, str.split)

    def test_returns_list(self, tmp_path):
        idx = self._build(tmp_path)
        result = idx.search("python", k=5)
        assert isinstance(result, list)

    def test_python_query_returns_python_docs(self, tmp_path):
        idx = self._build(tmp_path)
        results = idx.search("python", k=3)
        doc_ids = [r[0] for r in results]
        assert "doc-0" in doc_ids or "doc-2" in doc_ids

    def test_returns_empty_for_blank_query(self, tmp_path):
        idx = self._build(tmp_path)
        assert idx.search("", k=5) == []

    def test_respects_k_limit(self, tmp_path):
        idx = self._build(tmp_path)
        results = idx.search("python", k=1)
        assert len(results) <= 1

    def test_empty_index_returns_empty(self, tmp_path):
        idx = BM25SqliteIndex(
            db_path=tmp_path / "fake.sqlite",
            doc_ids=[],
            doc_lengths=[],
            avgdl=1.0,
            k1=DEFAULT_BM25_K1,
            b=DEFAULT_BM25_B,
            tokenizer=str.split,
        )
        result = idx.search("anything", k=5)
        assert result == []

    def test_unknown_term_returns_empty(self, tmp_path):
        idx = self._build(tmp_path)
        result = idx.search("zzzzunknownzzzzterm", k=5)
        assert result == []

    def test_scores_are_floats(self, tmp_path):
        idx = self._build(tmp_path)
        results = idx.search("python", k=3)
        for _, score in results:
            assert isinstance(score, float)

    def test_scores_are_positive(self, tmp_path):
        idx = self._build(tmp_path)
        results = idx.search("python", k=3)
        for _, score in results:
            assert score > 0


# ---------------------------------------------------------------------------
# build_bm25_index
# ---------------------------------------------------------------------------


class TestBuildBm25Index:
    def test_returns_none_when_no_docstore(self, tmp_path):
        result = build_bm25_index(tmp_path, "missing_key")
        assert result is None

    def test_builds_index_from_docstore(self, tmp_path):
        docs = _make_documents(3)
        build_docstore_cache(docs, tmp_path, "bk")

        result = build_bm25_index(tmp_path, "bk")

        assert result is not None
        assert result.exists()

    def test_returns_existing_db_without_rebuild(self, tmp_path):
        docs = _make_documents(2)
        build_docstore_cache(docs, tmp_path, "ck")
        build_bm25_index(tmp_path, "ck")

        # Second call should return cached file without rebuild
        db_path = tmp_path / "ck.bm25.sqlite"
        mtime_before = db_path.stat().st_mtime

        result = build_bm25_index(tmp_path, "ck", rebuild=False)
        assert result is not None
        assert result.stat().st_mtime == mtime_before

    def test_rebuild_flag_recreates_index(self, tmp_path):
        docs = _make_documents(2)
        build_docstore_cache(docs, tmp_path, "rk")
        build_bm25_index(tmp_path, "rk")

        db_path = tmp_path / "rk.bm25.sqlite"
        mtime_before = db_path.stat().st_mtime

        import time
        time.sleep(0.05)

        result = build_bm25_index(tmp_path, "rk", rebuild=True)
        assert result is not None
        assert result.stat().st_mtime >= mtime_before


# ---------------------------------------------------------------------------
# load_or_build_bm25_index
# ---------------------------------------------------------------------------


class TestLoadOrBuildBm25Index:
    def test_returns_none_when_no_docstore(self, tmp_path):
        result = load_or_build_bm25_index(tmp_path, "absent")
        assert result is None

    def test_returns_index_after_build(self, tmp_path):
        docs = _make_documents(3)
        build_docstore_cache(docs, tmp_path, "lk")

        result = load_or_build_bm25_index(tmp_path, "lk")

        assert result is not None
        assert isinstance(result, BM25SqliteIndex)

    def test_loads_existing_index(self, tmp_path):
        docs = _make_documents(2)
        build_docstore_cache(docs, tmp_path, "ek")
        # Build first
        load_or_build_bm25_index(tmp_path, "ek")
        # Load second (should not rebuild)
        result = load_or_build_bm25_index(tmp_path, "ek", rebuild=False)

        assert result is not None
        assert result.doc_count == 2


# ---------------------------------------------------------------------------
# MMapBM25Retriever
# ---------------------------------------------------------------------------


class TestMMapBM25Retriever:
    def _build_retriever(self, tmp_path: Path) -> MMapBM25Retriever:
        docs = [
            Document(page_content="python programming tutorial", metadata={"uuid": "p1"}),
            Document(page_content="javascript web framework spa", metadata={"uuid": "p2"}),
            Document(page_content="machine learning data science python", metadata={"uuid": "p3"}),
        ]
        build_docstore_cache(docs, tmp_path, "rk")
        return MMapBM25Retriever.from_cache(tmp_path, "rk", k=5)

    def test_from_cache_returns_none_when_no_index(self, tmp_path):
        result = MMapBM25Retriever.from_cache(tmp_path, "absent", k=5)
        assert result is None

    def test_from_cache_returns_retriever(self, tmp_path):
        retriever = self._build_retriever(tmp_path)
        assert retriever is not None
        assert isinstance(retriever, MMapBM25Retriever)

    def test_get_relevant_documents_returns_list(self, tmp_path):
        retriever = self._build_retriever(tmp_path)
        results = retriever._get_relevant_documents("python")
        assert isinstance(results, list)

    def test_results_are_document_instances(self, tmp_path):
        retriever = self._build_retriever(tmp_path)
        results = retriever._get_relevant_documents("python")
        for doc in results:
            assert isinstance(doc, Document)

    def test_results_have_non_empty_content(self, tmp_path):
        retriever = self._build_retriever(tmp_path)
        results = retriever._get_relevant_documents("python")
        assert len(results) > 0
        for doc in results:
            assert doc.page_content != ""

    def test_unknown_query_returns_empty(self, tmp_path):
        retriever = self._build_retriever(tmp_path)
        results = retriever._get_relevant_documents("zzunknownzz")
        assert results == []
