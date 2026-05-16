"""Multi-wiki retriever — fans out retrieval across N WikiRetrieverStack instances.

Exposes the same interface as WikiRetrieverStack so AskEngine requires zero changes.
"""

from __future__ import annotations

import asyncio
import logging
from functools import partial
from typing import Any

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

    async def aretrieve(
        self,
        query: str,
        k: int = 15,
        min_confidence: str | None = None,
    ) -> list:
        """Parallel retrieval across all wikis with per-wiki score normalisation."""
        if not self._stacks:
            return []

        async def retrieve_one(wiki_id: str, stack: Any) -> list:
            try:
                # WikiRetrieverStack.search_repository is sync — run in thread.
                #
                # apply_expansion=True (default for UnifiedRetriever)
                # is what the per-wiki retriever wants when going
                # through the multi-wiki fan-out. The wrapper used to
                # drop ``apply_expansion`` silently when reconstructing
                # the call; pass it explicitly so the per-wiki retriever
                # gets the same expansion behaviour as a direct call.
                #
                # ``min_confidence`` passes through as a kwarg only
                # when set, so older stacks that don't accept it
                # continue to work for the default no-filter case.
                # Newer stacks see the kwarg and apply the filter.
                if min_confidence is not None:
                    call = partial(
                        stack.search_repository,
                        query, k,
                        apply_expansion=True,
                        min_confidence=min_confidence,
                    )
                else:
                    call = partial(
                        stack.search_repository,
                        query, k,
                        apply_expansion=True,
                    )
                docs = await asyncio.to_thread(call)
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

    def search_repository(
        self,
        query: str,
        k: int = 15,
        apply_expansion: bool = True,  # noqa: ARG002 — kept for signature parity
        min_confidence: str | None = None,
    ) -> list:
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
        return asyncio.run(self.aretrieve(query, k=k, min_confidence=min_confidence))

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
