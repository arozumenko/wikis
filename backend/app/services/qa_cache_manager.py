"""Per-wiki QA FAISS index manager for semantic cache lookups."""

from __future__ import annotations

import asyncio
import json
import logging
import os
from pathlib import Path

import faiss
import numpy as np
from langchain_core.embeddings import Embeddings

logger = logging.getLogger(__name__)


class QACacheManager:
    """Manages per-wiki QA FAISS IndexFlatIP indexes for semantic cache.

    Each wiki gets its own FAISS index stored as:
      {cache_dir}/{wiki_id}.qa.faiss  — FAISS index binary
      {cache_dir}/{wiki_id}.qa.ids.json — parallel list of qa_id strings

    Uses raw FAISS IndexFlatIP with L2-normalized embeddings (cosine similarity).
    """

    def __init__(self, cache_dir: str, embeddings: Embeddings) -> None:
        self._cache_dir = cache_dir
        self._embeddings = embeddings
        self._indexes: dict[str, faiss.IndexFlatIP] = {}
        self._qa_ids: dict[str, list[str]] = {}
        self._locks: dict[str, asyncio.Lock] = {}
        os.makedirs(cache_dir, exist_ok=True)

    def _get_lock(self, wiki_id: str) -> asyncio.Lock:
        return self._locks.setdefault(wiki_id, asyncio.Lock())

    def _index_path(self, wiki_id: str) -> str:
        return os.path.join(self._cache_dir, f"{wiki_id}.qa.faiss")

    def _ids_path(self, wiki_id: str) -> str:
        return os.path.join(self._cache_dir, f"{wiki_id}.qa.ids.json")

    def _load_index(self, wiki_id: str) -> bool:
        """Load index from disk if not already in memory. Returns True if loaded."""
        if wiki_id in self._indexes:
            return True
        index_path = self._index_path(wiki_id)
        ids_path = self._ids_path(wiki_id)
        if not os.path.exists(index_path) or not os.path.exists(ids_path):
            return False
        try:
            index = faiss.read_index(index_path)
            with open(ids_path) as f:
                qa_ids = json.load(f)
            if len(qa_ids) != index.ntotal:
                logger.warning(
                    "QA index corruption for wiki %s: %d ids vs %d vectors — needs rebuild",
                    wiki_id,
                    len(qa_ids),
                    index.ntotal,
                )
                return False
            self._indexes[wiki_id] = index
            self._qa_ids[wiki_id] = qa_ids
            return True
        except Exception:
            logger.warning("Failed to load QA index for wiki %s", wiki_id, exc_info=True)
            return False

    def check_needs_rebuild(self, wiki_id: str) -> bool:
        """Return True if index files exist on disk but cannot be loaded.

        Used by QAService to trigger rebuild_from_records when the index is
        unusable rather than simply missing.
        """
        # _load_index returns True if already in memory or loaded successfully
        if self._load_index(wiki_id):
            return False
        # Files exist on disk but load failed (corrupt/mismatched)
        index_path = self._index_path(wiki_id)
        ids_path = self._ids_path(wiki_id)
        return os.path.exists(index_path) and os.path.exists(ids_path)

    def _save_index(self, wiki_id: str) -> None:
        """Write-ahead with rename (per-file atomic, not atomic across both files)."""
        index = self._indexes[wiki_id]
        qa_ids = self._qa_ids[wiki_id]
        index_path = self._index_path(wiki_id)
        ids_path = self._ids_path(wiki_id)
        tmp_index = index_path + ".tmp"
        tmp_ids = ids_path + ".tmp"
        try:
            faiss.write_index(index, tmp_index)
            with open(tmp_ids, "w") as f:
                json.dump(qa_ids, f)
            os.rename(tmp_index, index_path)
            os.rename(tmp_ids, ids_path)
        except Exception:
            # Clean up temp files on failure
            for tmp in (tmp_index, tmp_ids):
                if os.path.exists(tmp):
                    os.remove(tmp)
            raise

    async def search(self, wiki_id: str, question: str, threshold: float, k: int = 5) -> tuple[list[str], np.ndarray]:
        """Search for similar questions. Returns (qa_ids above threshold, L2-normalized embedding)."""
        embedding = await asyncio.to_thread(self._embeddings.embed_query, question)
        vec = np.array([embedding], dtype=np.float32)
        faiss.normalize_L2(vec)

        async with self._get_lock(wiki_id):
            if not self._load_index(wiki_id):
                return ([], vec[0])

            index = self._indexes[wiki_id]
            qa_ids = self._qa_ids[wiki_id]
            actual_k = min(k, index.ntotal)
            if actual_k == 0:
                return ([], vec[0])

            scores, indices = index.search(vec, actual_k)  # type: ignore[call-arg]
            result_ids = []
            for score, idx in zip(scores[0], indices[0]):
                if idx < 0:
                    continue
                if score >= threshold:
                    result_ids.append(qa_ids[idx])
            return (result_ids, vec[0])

    async def add(self, wiki_id: str, qa_id: str, embedding: np.ndarray) -> None:
        """Add a pre-computed, L2-normalized embedding to the wiki's index."""
        vec = np.array([embedding], dtype=np.float32)
        faiss.normalize_L2(vec)

        async with self._get_lock(wiki_id):
            if wiki_id not in self._indexes:
                # Lazy creation — dimension from first embedding
                self._load_index(wiki_id)
            if wiki_id not in self._indexes:
                dim = vec.shape[1]
                self._indexes[wiki_id] = faiss.IndexFlatIP(dim)
                self._qa_ids[wiki_id] = []

            self._indexes[wiki_id].add(vec)  # type: ignore[call-arg]
            self._qa_ids[wiki_id].append(qa_id)
            self._save_index(wiki_id)

    def delete_index(self, wiki_id: str) -> bool:
        """Delete a wiki's QA index from disk and memory."""
        self._indexes.pop(wiki_id, None)
        self._qa_ids.pop(wiki_id, None)
        self._locks.pop(wiki_id, None)
        deleted = False
        for path in (self._index_path(wiki_id), self._ids_path(wiki_id)):
            if os.path.exists(path):
                os.remove(path)
                deleted = True
        return deleted

    async def rebuild_from_records(self, wiki_id: str, records: list) -> None:
        """Rebuild index from provided records. Caller MUST pre-filter to cache-eligible only."""
        if not records:
            self.delete_index(wiki_id)
            return

        texts = [r.question for r in records]
        embeddings_list = await asyncio.to_thread(self._embeddings.embed_documents, texts)
        vecs = np.array(embeddings_list, dtype=np.float32)
        faiss.normalize_L2(vecs)

        async with self._get_lock(wiki_id):
            dim = vecs.shape[1]
            index = faiss.IndexFlatIP(dim)
            index.add(vecs)  # type: ignore[call-arg]
            self._indexes[wiki_id] = index
            self._qa_ids[wiki_id] = [r.id for r in records]
            self._save_index(wiki_id)
