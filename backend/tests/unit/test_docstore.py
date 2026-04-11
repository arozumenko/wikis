"""Unit tests for app/core/docstore.py"""

import json
import pickle
from pathlib import Path

import pytest
from langchain_core.documents import Document

from app.core.docstore import (
    DOCSTORE_SCHEMA_VERSION,
    DocstoreIndex,
    MMapDocstore,
    build_docstore_cache,
    load_docstore_cache,
    migrate_docstore_from_docs_pickle,
    write_docstore_index,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_documents(count: int = 3) -> list[Document]:
    return [
        Document(
            page_content=f"Content of document {i}",
            metadata={"uuid": f"uuid-{i}", "source": f"file_{i}.py"},
        )
        for i in range(count)
    ]


# ---------------------------------------------------------------------------
# DocstoreIndex.load
# ---------------------------------------------------------------------------


class TestDocstoreIndexLoad:
    def test_returns_none_when_index_missing(self, tmp_path):
        result = DocstoreIndex.load(tmp_path, "nonexistent_key")
        assert result is None

    def test_returns_none_when_json_malformed(self, tmp_path):
        index_path = tmp_path / "bad_key.doc_index.json"
        index_path.write_text("not valid json", encoding="utf-8")
        result = DocstoreIndex.load(tmp_path, "bad_key")
        assert result is None

    def test_returns_none_when_entries_missing(self, tmp_path):
        index_path = tmp_path / "k.doc_index.json"
        index_path.write_text(json.dumps({"doc_ids": [], "docstore_file": "k.docstore.bin"}), encoding="utf-8")
        result = DocstoreIndex.load(tmp_path, "k")
        assert result is None

    def test_returns_none_when_docstore_file_absent(self, tmp_path):
        index_path = tmp_path / "k.doc_index.json"
        data = {
            "entries": {},
            "doc_ids": [],
            "docstore_file": "missing.bin",
        }
        index_path.write_text(json.dumps(data), encoding="utf-8")
        result = DocstoreIndex.load(tmp_path, "k")
        assert result is None

    def test_loads_valid_index(self, tmp_path):
        bin_file = tmp_path / "k.docstore.bin"
        bin_file.write_bytes(b"hello")

        data = {
            "entries": {"uuid-0": {"offset": 0, "length": 5, "metadata": {}}},
            "doc_ids": ["uuid-0"],
            "docstore_file": "k.docstore.bin",
        }
        (tmp_path / "k.doc_index.json").write_text(json.dumps(data), encoding="utf-8")

        result = DocstoreIndex.load(tmp_path, "k")

        assert result is not None
        assert result.cache_key == "k"
        assert result.doc_ids == ["uuid-0"]
        assert "uuid-0" in result.entries


# ---------------------------------------------------------------------------
# MMapDocstore
# ---------------------------------------------------------------------------


class TestMMapDocstore:
    def _build(self, tmp_path: Path) -> MMapDocstore:
        content = b"hello world"
        bin_file = tmp_path / "store.bin"
        bin_file.write_bytes(content)
        entries = {
            "doc-1": {"offset": 0, "length": 5, "metadata": {"source": "a.py"}},
            "doc-2": {"offset": 6, "length": 5, "metadata": {"source": "b.py"}},
        }
        return MMapDocstore(bin_file, entries)

    def test_search_returns_document_for_known_id(self, tmp_path):
        store = self._build(tmp_path)
        result = store.search("doc-1")
        assert isinstance(result, Document)
        assert result.page_content == "hello"
        store.close()

    def test_search_returns_second_document_correctly(self, tmp_path):
        store = self._build(tmp_path)
        result = store.search("doc-2")
        assert isinstance(result, Document)
        assert result.page_content == "world"
        store.close()

    def test_search_returns_error_string_for_unknown_id(self, tmp_path):
        store = self._build(tmp_path)
        result = store.search("not-there")
        assert isinstance(result, str)
        assert "not found" in result.lower()
        store.close()

    def test_search_preserves_metadata(self, tmp_path):
        store = self._build(tmp_path)
        result = store.search("doc-1")
        assert isinstance(result, Document)
        assert result.metadata["source"] == "a.py"
        store.close()

    def test_search_zero_length_entry_returns_empty_content(self, tmp_path):
        bin_file = tmp_path / "store.bin"
        bin_file.write_bytes(b"data")
        store = MMapDocstore(bin_file, {"empty": {"offset": 0, "length": 0, "metadata": {}}})
        result = store.search("empty")
        assert isinstance(result, Document)
        assert result.page_content == ""
        store.close()

    def test_close_is_idempotent(self, tmp_path):
        store = self._build(tmp_path)
        store.close()
        store.close()  # Should not raise


# ---------------------------------------------------------------------------
# build_docstore_cache
# ---------------------------------------------------------------------------


class TestBuildDocstoreCache:
    def test_returns_none_for_empty_document_list(self, tmp_path):
        result = build_docstore_cache([], tmp_path, "key")
        assert result is None

    def test_creates_bin_and_index_files(self, tmp_path):
        docs = _make_documents(2)
        result = build_docstore_cache(docs, tmp_path, "mykey")

        assert result is not None
        assert (tmp_path / "mykey.docstore.bin").exists()
        assert (tmp_path / "mykey.doc_index.json").exists()

    def test_index_contains_correct_doc_count(self, tmp_path):
        docs = _make_documents(3)
        build_docstore_cache(docs, tmp_path, "ck")

        with open(tmp_path / "ck.doc_index.json", encoding="utf-8") as f:
            data = json.load(f)

        assert len(data["doc_ids"]) == 3
        assert len(data["entries"]) == 3

    def test_document_content_round_trips(self, tmp_path):
        docs = [Document(page_content="round trip content", metadata={"uuid": "rt-1"})]
        build_docstore_cache(docs, tmp_path, "rk")

        index = DocstoreIndex.load(tmp_path, "rk")
        store = MMapDocstore(index.docstore_path, index.entries)
        doc = store.search("rt-1")

        assert isinstance(doc, Document)
        assert doc.page_content == "round trip content"
        store.close()

    def test_documents_without_uuid_get_one_assigned(self, tmp_path):
        docs = [Document(page_content="no uuid doc", metadata={})]
        build_docstore_cache(docs, tmp_path, "nk")

        with open(tmp_path / "nk.doc_index.json", encoding="utf-8") as f:
            data = json.load(f)

        assert len(data["doc_ids"]) == 1
        assigned_id = data["doc_ids"][0]
        assert assigned_id != ""

    def test_schema_version_written(self, tmp_path):
        docs = _make_documents(1)
        build_docstore_cache(docs, tmp_path, "sv")

        with open(tmp_path / "sv.doc_index.json", encoding="utf-8") as f:
            data = json.load(f)

        assert data["schema_version"] == DOCSTORE_SCHEMA_VERSION


# ---------------------------------------------------------------------------
# write_docstore_index
# ---------------------------------------------------------------------------


class TestWriteDocstoreIndex:
    def test_writes_json_index(self, tmp_path):
        bin_file = tmp_path / "store.bin"
        bin_file.write_bytes(b"")
        entries = {"id-1": {"offset": 0, "length": 0, "metadata": {}}}

        result = write_docstore_index(tmp_path, "wk", bin_file, entries, ["id-1"])

        assert result is not None
        assert result.exists()

    def test_written_index_is_loadable(self, tmp_path):
        bin_file = tmp_path / "store.bin"
        bin_file.write_bytes(b"abc")
        entries = {"id-1": {"offset": 0, "length": 3, "metadata": {}}}

        write_docstore_index(tmp_path, "wk2", bin_file, entries, ["id-1"])
        loaded = DocstoreIndex.load(tmp_path, "wk2")

        assert loaded is not None
        assert loaded.doc_ids == ["id-1"]


# ---------------------------------------------------------------------------
# load_docstore_cache
# ---------------------------------------------------------------------------


class TestLoadDocstoreCache:
    def test_returns_none_when_no_index(self, tmp_path):
        result = load_docstore_cache(tmp_path, "absent")
        assert result is None

    def test_returns_tuple_on_success(self, tmp_path):
        docs = _make_documents(2)
        build_docstore_cache(docs, tmp_path, "lk")

        result = load_docstore_cache(tmp_path, "lk")

        assert result is not None
        store, index_map, docs_list = result
        assert isinstance(store, MMapDocstore)
        assert isinstance(index_map, dict)
        assert isinstance(docs_list, list)
        store.close()

    def test_index_map_has_correct_length(self, tmp_path):
        docs = _make_documents(4)
        build_docstore_cache(docs, tmp_path, "lk2")
        store, index_map, _ = load_docstore_cache(tmp_path, "lk2")

        assert len(index_map) == 4
        store.close()

    def test_docs_list_length_matches_input(self, tmp_path):
        docs = _make_documents(3)
        build_docstore_cache(docs, tmp_path, "lk3")
        store, _, docs_list = load_docstore_cache(tmp_path, "lk3")

        assert len(docs_list) == 3
        store.close()


# ---------------------------------------------------------------------------
# migrate_docstore_from_docs_pickle
# ---------------------------------------------------------------------------


class TestMigrateDocstoreFromDocsPickle:
    def test_returns_false_when_pickle_missing(self, tmp_path):
        result = migrate_docstore_from_docs_pickle(tmp_path, "k", tmp_path / "absent.pkl")
        assert result is False

    def test_migrates_valid_pickle(self, tmp_path):
        docs = _make_documents(2)
        pkl_file = tmp_path / "k.docs.pkl"
        with open(pkl_file, "wb") as f:
            pickle.dump(docs, f)

        result = migrate_docstore_from_docs_pickle(tmp_path, "k", pkl_file)

        assert result is True
        assert (tmp_path / "k.doc_index.json").exists()
        assert (tmp_path / "k.docstore.bin").exists()

    def test_returns_false_for_non_list_pickle(self, tmp_path):
        pkl_file = tmp_path / "bad.pkl"
        with open(pkl_file, "wb") as f:
            pickle.dump({"not": "a list"}, f)

        result = migrate_docstore_from_docs_pickle(tmp_path, "bk", pkl_file)
        assert result is False
