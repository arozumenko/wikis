"""Project search engine — fans out WikiSearchEngine.search() across all wikis in a project.

This module is pure Python: no LLM, no FAISS.
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field
from typing import Callable

from app.core.wiki_search_engine import ResultItem, WikiSearchEngine, WikiSummaryItem

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Response dataclasses
# ---------------------------------------------------------------------------


@dataclass
class ProjectSearchResult:
    """Top-level project search result container.

    Attributes:
        query: Original search query string.
        results: Merged and re-ranked list of result pages from all wikis,
            sorted by score descending, capped at top_k.
        wiki_summary: Per-wiki statistics — always present for every wiki
            that was searched, even those with zero hits.
    """

    query: str
    results: list[ResultItem] = field(default_factory=list)
    wiki_summary: list[WikiSummaryItem] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Engine
# ---------------------------------------------------------------------------


class ProjectSearchEngine:
    """Fans out search across all wikis in a project and merges results.

    Performs parallel fan-out using asyncio.gather, collects results from
    each WikiSearchEngine, merges them, sorts by score descending, and
    caps to top_k. Zero-hit wikis are excluded from merged results but
    always included in wiki_summary.

    Args:
        wiki_search_engine_factory: Callable that takes ``(wiki_id, wiki_name)``
            and returns a configured WikiSearchEngine instance. Allows test
            injection and decouples this class from DB/storage details.
    """

    def __init__(
        self,
        wiki_search_engine_factory: Callable[[str, str], WikiSearchEngine],
    ) -> None:
        self._factory = wiki_search_engine_factory

    async def search(
        self,
        query: str,
        wikis: list[tuple[str, str]],  # list of (wiki_id, wiki_name)
        hop_depth: int = 1,
        top_k: int = 10,
    ) -> ProjectSearchResult:
        """Search all wikis in parallel and return merged, ranked results.

        Args:
            query: Free-text search query.
            wikis: List of ``(wiki_id, wiki_name)`` tuples to search.
            hop_depth: Graph expansion depth forwarded to each WikiSearchEngine.
                Defaults to 1.
            top_k: Maximum number of results to return after merging.
                Defaults to 10.

        Returns:
            ProjectSearchResult with merged results sorted by score descending
            and per-wiki summary for every wiki that was searched.
        """
        if not wikis:
            logger.debug("ProjectSearchEngine: no wikis provided — returning empty result")
            return ProjectSearchResult(query=query, results=[], wiki_summary=[])

        # Build one engine per wiki via the injected factory.
        engines = [(wiki_id, wiki_name, self._factory(wiki_id, wiki_name)) for wiki_id, wiki_name in wikis]

        # Fan out in parallel.
        wiki_results = await asyncio.gather(
            *[engine.search(query, hop_depth=hop_depth, top_k=top_k) for _, _, engine in engines]
        )

        # Merge results and wiki_summary.
        merged_results: list[ResultItem] = []
        merged_summary: list[WikiSummaryItem] = []

        for wiki_result in wiki_results:
            # Include ALL wikis in summary (even zero-hit ones).
            merged_summary.extend(wiki_result.wiki_summary)

            # Only include wikis with at least one result in the merged list.
            if wiki_result.results:
                merged_results.extend(wiki_result.results)

        # Sort merged results by score descending and cap at top_k.
        merged_results.sort(key=lambda r: r.score, reverse=True)
        merged_results = merged_results[:top_k]

        logger.debug(
            "ProjectSearchEngine: query=%r wikis=%d → %d merged results",
            query,
            len(wikis),
            len(merged_results),
        )

        return ProjectSearchResult(
            query=query,
            results=merged_results,
            wiki_summary=merged_summary,
        )
