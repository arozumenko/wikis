"""Disk-backed document store with mmap-backed lazy reads."""

from __future__ import annotations

import json
import logging
import mmap
import os
from pathlib import Path
from typing import Any

from langchain_community.docstore.base import Docstore
from langchain_core.documents import Document

logger = logging.getLogger(__name__)

DOCSTORE_SCHEMA_VERSION = 1


class DocstoreIndex:
    """Represents a disk-backed docstore index."""

    def __init__(
        self,
        cache_key: str,
        docstore_path: Path,
        entries: dict[str, dict[str, Any]],
        doc_ids: list[str],
    ) -> None:
        self.cache_key = cache_key
        self.docstore_path = docstore_path
        self.entries = entries
        self.doc_ids = doc_ids

    @classmethod
    def load(cls, cache_dir: Path, cache_key: str) -> DocstoreIndex | None:
        index_path = cache_dir / f"{cache_key}.doc_index.json"
        if not index_path.exists():
            return None

        try:
            with open(index_path, encoding="utf-8") as f:
                data = json.load(f)
        except Exception as e:
            logger.warning(f"Failed to load docstore index {index_path}: {e}")
            return None

        if not isinstance(data, dict):
            return None

        entries = data.get("entries")
        doc_ids = data.get("doc_ids")
        docstore_file = data.get("docstore_file")

        if not isinstance(entries, dict) or not isinstance(doc_ids, list) or not isinstance(docstore_file, str):
            return None

        docstore_path = cache_dir / docstore_file
        if not docstore_path.exists():
            logger.warning(f"Docstore file missing for cache_key={cache_key}: {docstore_path}")
            return None

        return cls(cache_key=cache_key, docstore_path=docstore_path, entries=entries, doc_ids=doc_ids)


class MMapDocstore(Docstore):
    """Docstore that reads document bodies from an mmap-backed file."""

    def __init__(self, docstore_path: Path, entries: dict[str, dict[str, Any]]) -> None:
        self._docstore_path = docstore_path
        self._entries = entries
        self._file = open(docstore_path, "rb")
        self._mmap = mmap.mmap(self._file.fileno(), length=0, access=mmap.ACCESS_READ)

    def search(self, search: str):  # type: ignore[override]
        entry = self._entries.get(search)
        if not entry:
            return f"ID {search} not found."

        offset = int(entry.get("offset", 0))
        length = int(entry.get("length", 0))
        metadata = entry.get("metadata", {}) or {}

        if length <= 0:
            content = ""
        else:
            content = self._mmap[offset : offset + length].decode("utf-8", errors="replace")

        return Document(page_content=content, metadata=metadata)

    def close(self) -> None:
        try:
            self._mmap.close()
        except Exception:  # noqa: S110
            pass
        try:
            self._file.close()
        except Exception:  # noqa: S110
            pass


def build_docstore_cache(
    documents: list[Document],
    cache_dir: Path,
    cache_key: str,
) -> Path | None:
    """Write docstore.bin + doc_index.json for lazy loading."""
    if not documents:
        return None

    cache_dir.mkdir(parents=True, exist_ok=True)
    docstore_path = cache_dir / f"{cache_key}.docstore.bin"
    index_path = cache_dir / f"{cache_key}.doc_index.json"

    entries: dict[str, dict[str, Any]] = {}
    doc_ids: list[str] = []

    try:
        with open(docstore_path, "wb") as f:
            for doc in documents:
                doc_id = doc.metadata.get("uuid")
                if not doc_id:
                    doc_id = os.urandom(16).hex()
                    doc.metadata["uuid"] = doc_id

                encoded = (doc.page_content or "").encode("utf-8")
                offset = f.tell()
                f.write(encoded)
                length = len(encoded)

                entries[str(doc_id)] = {
                    "offset": offset,
                    "length": length,
                    "metadata": doc.metadata,
                }
                doc_ids.append(str(doc_id))

        data = {
            "schema_version": DOCSTORE_SCHEMA_VERSION,
            "cache_key": cache_key,
            "docstore_file": docstore_path.name,
            "doc_ids": doc_ids,
            "entries": entries,
        }

        with open(index_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

        logger.info(f"Saved docstore cache (docs={len(doc_ids)}) to {docstore_path.name} / {index_path.name}")
        return index_path

    except Exception as e:
        logger.warning(f"Failed to build docstore cache for {cache_key}: {e}")
        return None


def write_docstore_index(
    cache_dir: Path,
    cache_key: str,
    docstore_path: Path,
    entries: dict[str, dict[str, Any]],
    doc_ids: list[str],
) -> Path | None:
    """Persist docstore index json for an existing docstore file."""
    index_path = cache_dir / f"{cache_key}.doc_index.json"
    try:
        data = {
            "schema_version": DOCSTORE_SCHEMA_VERSION,
            "cache_key": cache_key,
            "docstore_file": docstore_path.name,
            "doc_ids": doc_ids,
            "entries": entries,
        }
        with open(index_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        return index_path
    except Exception as e:
        logger.warning(f"Failed to write docstore index for {cache_key}: {e}")
        return None


def load_docstore_cache(
    cache_dir: Path,
    cache_key: str,
) -> tuple[MMapDocstore, dict[int, str], list[Document]] | None:
    """Load mmap-backed docstore + index_to_docstore_id + metadata-only docs."""
    index = DocstoreIndex.load(cache_dir, cache_key)
    if not index:
        return None

    docstore = MMapDocstore(index.docstore_path, index.entries)
    index_to_docstore_id = {i: doc_id for i, doc_id in enumerate(index.doc_ids)}

    # Build lightweight document list (metadata only)
    documents: list[Document] = []
    for doc_id in index.doc_ids:
        entry = index.entries.get(doc_id, {})
        metadata = entry.get("metadata", {}) or {}
        documents.append(Document(page_content="", metadata=metadata))

    return docstore, index_to_docstore_id, documents


def migrate_docstore_from_docs_pickle(
    cache_dir: Path,
    cache_key: str,
    docs_file: Path,
) -> bool:
    """Build docstore cache from legacy docs.pkl and register in cache index."""
    if not docs_file.exists():
        return False

    try:
        import pickle

        with open(docs_file, "rb") as f:
            documents = pickle.load(f)  # noqa: S301 — pickle used for docstore cache, data is self-generated

        if not isinstance(documents, list):
            return False

        index_path = build_docstore_cache(documents, cache_dir, cache_key)
        if not index_path:
            return False

        # Update cache_index.json to include docs mapping if possible
        try:
            from .repo_resolution import load_cache_index, save_cache_index_atomic

            index = load_cache_index(cache_dir)
            if not isinstance(index, dict):
                index = {}

            docs_map = index.get("docs")
            if not isinstance(docs_map, dict):
                docs_map = {}
                index["docs"] = docs_map

            # Reverse lookup repo_identifier by cache_key
            repo_identifier = None
            for key, value in index.items():
                if key in {"graphs", "refs", "docs"}:
                    continue
                if value == cache_key:
                    repo_identifier = key
                    break

            if repo_identifier:
                docs_map[repo_identifier] = cache_key
                save_cache_index_atomic(cache_dir, index)
        except Exception as e:
            logger.warning(f"Failed to update cache index for docstore migration: {e}")

        logger.info(f"Migrated docstore cache for key={cache_key} using {docs_file.name}")
        return True
    except Exception as e:
        logger.warning(f"Docstore migration failed for key={cache_key}: {e}")
        return False
