"""Multi-wiki retriever — fans out retrieval across N WikiRetrieverStack instances.

Exposes the same interface as WikiRetrieverStack so AskEngine requires zero changes.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any

from .constants import DOC_SYMBOL_TYPES

logger = logging.getLogger(__name__)


class MultiWikiRetrieverStack:
    """Fans out retrieval across N wiki retriever stacks and merges results.

    Accepts a list of (wiki_id, WikiRetrieverStack) tuples and exposes
    ``search_repository`` / ``aretrieve`` compatible with the single-wiki
    WikiRetrieverStack used by AskEngine tools.
    """

    def __init__(self, wiki_stacks: list[tuple[str, Any]]) -> None:
        # wiki_stacks: list of (wiki_id, WikiRetrieverStack)
        self._stacks = wiki_stacks

    # ------------------------------------------------------------------
    # Async retrieval
    # ------------------------------------------------------------------

    async def aretrieve(self, query: str, k: int = 15) -> list:
        """Parallel retrieval across all wikis with per-wiki score normalisation."""
        if not self._stacks:
            return []

        async def retrieve_one(wiki_id: str, stack: Any) -> list:
            try:
                # WikiRetrieverStack.search_repository is sync — run in thread
                docs = await asyncio.to_thread(stack.search_repository, query, k)
                for doc in docs:
                    doc.metadata["source_wiki_id"] = wiki_id
                return docs
            except Exception as exc:
                logger.warning("Retrieval failed for wiki %s: %s", wiki_id, exc)
                return []

        results = await asyncio.gather(
            *[retrieve_one(wid, stack) for wid, stack in self._stacks]
        )

        # Per-wiki min-max score normalisation then global sort
        all_docs: list = []
        for wiki_docs in results:
            if not wiki_docs:
                continue
            scores = [d.metadata.get("score", 0.0) for d in wiki_docs]
            min_s, max_s = min(scores), max(scores)
            rng = max_s - min_s or 1.0
            for doc in wiki_docs:
                raw = doc.metadata.get("score", 0.0)
                doc.metadata["normalized_score"] = (raw - min_s) / rng
            all_docs.extend(wiki_docs)

        all_docs.sort(key=lambda d: d.metadata.get("normalized_score", 0.0), reverse=True)
        # Return top k*2 overall — AskEngine will trim further
        return all_docs[: k * 2]

    # ------------------------------------------------------------------
    # Sync wrapper (matches WikiRetrieverStack.search_repository signature)
    # ------------------------------------------------------------------

    def search_repository(self, query: str, k: int = 15, apply_expansion: bool = True) -> list:
        """Synchronous fan-out retrieval — raises if called from a running event loop.

        Use aretrieve() directly in async contexts.
        """
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = None

        if loop is not None and loop.is_running():
            raise RuntimeError(
                "MultiWikiRetrieverStack.search_repository() cannot be called from an async context. "
                "Use await aretrieve() instead."
            )
        return asyncio.run(self.aretrieve(query, k=k))

    async def asearch_docs_semantic(
        self,
        query: str,
        k: int = 10,
        similarity_threshold: float = 0.7,
    ) -> list:
        """Parallel documentation-only retrieval across all wikis."""
        if not self._stacks:
            return []

        async def retrieve_docs_one(wiki_id: str, stack: Any) -> list:
            try:
                doc_search = getattr(stack, "search_docs_semantic", None)
                if doc_search is not None:
                    docs = await asyncio.to_thread(
                        doc_search,
                        query,
                        k,
                        similarity_threshold,
                    )
                else:
                    docs = await asyncio.to_thread(stack.search_repository, query, max(k * 3, 10), False)
                    docs = [doc for doc in docs if self._is_document_hit(doc)]
                for doc in docs:
                    doc.metadata["source_wiki_id"] = wiki_id
                return docs[:k]
            except Exception as exc:
                logger.warning("Documentation retrieval failed for wiki %s: %s", wiki_id, exc)
                return []

        results = await asyncio.gather(
            *[retrieve_docs_one(wid, stack) for wid, stack in self._stacks]
        )
        all_docs = [doc for docs in results for doc in docs]
        all_docs.sort(key=lambda d: d.metadata.get("score", d.metadata.get("normalized_score", 0.0)), reverse=True)
        return all_docs[: k * 2]

    def search_docs_semantic(
        self,
        query: str,
        k: int = 10,
        similarity_threshold: float = 0.7,
    ) -> list:
        """Synchronous documentation-only fan-out retrieval."""
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = None

        if loop is not None and loop.is_running():
            raise RuntimeError(
                "MultiWikiRetrieverStack.search_docs_semantic() cannot be called from an async context. "
                "Use await asearch_docs_semantic() instead."
            )
        return asyncio.run(self.asearch_docs_semantic(query, k=k, similarity_threshold=similarity_threshold))

    def retrieve(self, query: str, k: int = 15) -> list:
        """Sync retrieve — raises if called from a running event loop.

        Use aretrieve() directly in async contexts.
        """
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = None

        if loop is not None and loop.is_running():
            raise RuntimeError(
                "MultiWikiRetrieverStack.retrieve() cannot be called from an async context. "
                "Use await aretrieve() instead."
            )
        return asyncio.run(self.aretrieve(query, k=k))

    @staticmethod
    def _is_document_hit(doc) -> bool:
        metadata = getattr(doc, "metadata", {}) or {}
        symbol_type = (metadata.get("symbol_type") or metadata.get("type") or "").lower()
        return bool(
            metadata.get("is_documentation")
            or metadata.get("semantic_retrieved")
            or metadata.get("is_doc")
            or symbol_type in DOC_SYMBOL_TYPES
            or symbol_type.endswith("_document")
        )
