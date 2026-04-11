"""Wiki page search adapter — wraps FTS5 full-text search with normalized scores.

This module is pure Python: no LLM, no FAISS.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

from app.core.unified_db import UnifiedWikiDB
from app.core.wiki_page_index import WikiPageIndex

logger = logging.getLogger(__name__)


@dataclass
class PageHit:
    """A single search result with a normalized relevance score.

    Attributes:
        page_title: Title of the matching wiki page (from the FTS5 result).
        base_score: Relevance score min-max normalized to [0.0, 1.0] within
            this search call.  Higher is better.
        snippet: Short description of the page from WikiPageIndex.
    """

    page_title: str
    base_score: float
    snippet: str


class WikiPageSearchAdapter:
    """Thin adapter that queries the FTS5 index and returns normalized PageHits.

    Args:
        wiki_id: Identifier of the wiki (used for logging).
        db: UnifiedWikiDB instance to query.
        page_index: WikiPageIndex used to look up page descriptions.
    """

    def __init__(
        self,
        wiki_id: str,
        db: UnifiedWikiDB,
        page_index: WikiPageIndex,
    ) -> None:
        self._wiki_id = wiki_id
        self._db = db
        self._page_index = page_index

    def search(self, query: str, top_k: int = 10) -> list[PageHit]:
        """Search wiki pages using FTS5 and return normalized hits.

        Scores are min-max normalized per call so the best result has
        base_score=1.0 and the worst has base_score=0.0.  When all scores
        are equal (or there is only one result) every hit gets 1.0.

        Args:
            query: Free-text search query.
            top_k: Maximum number of results to return. Defaults to 10.

        Returns:
            List of PageHit sorted by base_score descending.  Empty list
            when the FTS5 search returns no results.
        """
        raw = self._db.search_fts5(query, limit=top_k)

        if not raw:
            logger.debug("WikiPageSearchAdapter: no FTS5 results for wiki=%s query=%r", self._wiki_id, query)
            return []

        # Extract raw scores (fts_rank from FTS5).
        scores = [float(row["fts_rank"]) for row in raw]
        min_score = min(scores)
        max_score = max(scores)

        if max_score == min_score:
            # All equal (including single result) — normalize to 1.0.
            normalized = [1.0] * len(scores)
        else:
            span = max_score - min_score
            normalized = [(s - min_score) / span for s in scores]

        hits: list[PageHit] = []
        for row, norm_score in zip(raw, normalized):
            title = row["symbol_name"]
            snippet = self._page_index.get_description(title)
            hits.append(PageHit(page_title=title, base_score=norm_score, snippet=snippet))

        hits.sort(key=lambda h: h.base_score, reverse=True)

        logger.debug(
            "WikiPageSearchAdapter: wiki=%s query=%r → %d hits",
            self._wiki_id,
            query,
            len(hits),
        )
        return hits
