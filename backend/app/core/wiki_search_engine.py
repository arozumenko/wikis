"""Wiki search engine — FTS5 hits + graph expansion + composite re-ranking.

This module is pure Python: no LLM, no FAISS.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

from app.core.unified_db import UnifiedWikiDB
from app.core.wiki_page_index import WikiPageIndex
from app.core.wiki_page_search import WikiPageSearchAdapter

logger = logging.getLogger(__name__)

# Maximum neighbors returned per ResultItem (mirrors WikiPageIndex._MAX_NEIGHBORS).
_MAX_NEIGHBORS_PER_RESULT = 50

# Over-fetch multiplier — fetch 3× top_k candidates before re-ranking.
_OVERFETCH_FACTOR = 3


# ---------------------------------------------------------------------------
# Response dataclasses
# ---------------------------------------------------------------------------


@dataclass
class NeighborItem:
    """A single neighbor page linked to or from a result page.

    Attributes:
        title: Title of the neighboring page.
        rel: Relationship direction — ``"links_to"`` when the result page
            links *to* this neighbor; ``"linked_from"`` when this neighbor
            links *to* the result page.
    """

    title: str
    rel: str  # "links_to" | "linked_from"


@dataclass
class ResultItem:
    """A single ranked result page.

    Attributes:
        wiki_id: Identifier of the wiki that contains this page.
        wiki_name: Human-readable name of the wiki.
        page_title: Title of the matching wiki page.
        snippet: Short description sourced from the page's frontmatter.
        score: Composite re-ranking score (higher is better).
        neighbors: Direct forward links and backlinks, capped at 50.
    """

    wiki_id: str
    wiki_name: str
    page_title: str
    snippet: str
    score: float
    neighbors: list[NeighborItem] = field(default_factory=list)


@dataclass
class WikiSummaryItem:
    """Per-wiki aggregated search statistics.

    Attributes:
        wiki_id: Identifier of the wiki.
        wiki_name: Human-readable name of the wiki.
        match_count: Number of pages returned by the FTS5 search (before expansion).
        relevance: ``hit_count / total_pages`` for this wiki; ``0.0`` when the
            wiki has no pages.
    """

    wiki_id: str
    wiki_name: str
    match_count: int
    relevance: float


@dataclass
class WikiSearchResult:
    """Top-level search result container.

    Attributes:
        query: Original search query string.
        results: Re-ranked list of result pages, sorted by score descending.
        wiki_summary: Per-wiki statistics — always present even when
            ``results`` is empty.
    """

    query: str
    results: list[ResultItem] = field(default_factory=list)
    wiki_summary: list[WikiSummaryItem] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Engine
# ---------------------------------------------------------------------------


class WikiSearchEngine:
    """Core search pipeline: FTS5 hits → graph expansion → composite re-ranking.

    Re-ranking formula::

        score(p) = base_score
                   × (1 + graph_centrality + neighborhood_density)
                   × wiki_relevance

        graph_centrality     = backlink_count(p) / total_pages(wiki)
        neighborhood_density = matched_neighbors(p) / total_neighbors(p)
                               (0.0 if p has no neighbors)
        wiki_relevance       = hit_count(wiki) / total_pages(wiki)

    Args:
        wiki_id: Identifier of the wiki to search.
        wiki_name: Human-readable name of the wiki.
        db: UnifiedWikiDB instance used by WikiPageSearchAdapter.
        page_index: WikiPageIndex for graph traversal and description look-ups.
    """

    def __init__(
        self,
        wiki_id: str,
        wiki_name: str,
        db: UnifiedWikiDB,
        page_index: WikiPageIndex,
    ) -> None:
        self._wiki_id = wiki_id
        self._wiki_name = wiki_name
        self._page_index = page_index
        self._adapter = WikiPageSearchAdapter(wiki_id, db, page_index)

    async def search(
        self,
        query: str,
        hop_depth: int = 1,
        top_k: int = 10,
    ) -> WikiSearchResult:
        """Search wiki pages and return re-ranked results with graph neighbors.

        Args:
            query: Free-text search query.
            hop_depth: Depth of BFS graph expansion for neighbor look-up.
                Defaults to 1.
            top_k: Maximum number of results to return after re-ranking.
                Defaults to 10.

        Returns:
            WikiSearchResult with ranked results and per-wiki summary.
        """
        # --- 1. FTS5 search (over-fetch for re-ranking headroom) ---
        candidates = self._adapter.search(query, top_k=top_k * _OVERFETCH_FACTOR)

        total_pages = len(self._page_index.pages)
        hit_count = len(candidates)

        # --- 2. Build wiki summary (always present) ---
        if total_pages > 0:
            relevance = hit_count / total_pages
        else:
            relevance = 0.0

        wiki_summary = WikiSummaryItem(
            wiki_id=self._wiki_id,
            wiki_name=self._wiki_name,
            match_count=hit_count,
            relevance=relevance,
        )

        if not candidates:
            return WikiSearchResult(
                query=query,
                results=[],
                wiki_summary=[wiki_summary],
            )

        # --- 3. Build FTS5 hit set for neighborhood_density calculation ---
        hit_titles: set[str] = {hit.page_title for hit in candidates}

        # --- 4. Re-rank ---
        ranked: list[ResultItem] = []
        for hit in candidates:
            title = hit.page_title

            # graph_centrality
            backlink_count = len(self._page_index.backlinks(title))
            if total_pages > 0:
                graph_centrality = backlink_count / total_pages
            else:
                graph_centrality = 0.0

            # neighborhood_density — use hop_depth for neighbor expansion
            direct_neighbors = self._page_index.neighbors(title, hop_depth)
            total_neighbors = len(direct_neighbors)
            if total_neighbors > 0:
                matched_neighbors = sum(
                    1 for n in direct_neighbors if n.title in hit_titles
                )
                neighborhood_density = matched_neighbors / total_neighbors
            else:
                neighborhood_density = 0.0

            # composite score
            score = (
                hit.base_score
                * (1 + graph_centrality + neighborhood_density)
                * relevance
            )

            # --- 5. Build neighbor list (forward links + backlinks, cap 50) ---
            neighbors: list[NeighborItem] = []

            # Forward links (pages that this page links TO)
            for neighbor_meta in self._page_index.neighbors(title, 1):
                if len(neighbors) >= _MAX_NEIGHBORS_PER_RESULT:
                    break
                neighbors.append(NeighborItem(title=neighbor_meta.title, rel="links_to"))

            # Backlinks (pages that link TO this page) — fill remaining slots
            for backlink_title in self._page_index.backlinks(title):
                if len(neighbors) >= _MAX_NEIGHBORS_PER_RESULT:
                    break
                neighbors.append(NeighborItem(title=backlink_title, rel="linked_from"))

            ranked.append(
                ResultItem(
                    wiki_id=self._wiki_id,
                    wiki_name=self._wiki_name,
                    page_title=title,
                    snippet=hit.snippet,
                    score=score,
                    neighbors=neighbors,
                )
            )

        # --- 6. Sort and cap ---
        ranked.sort(key=lambda r: r.score, reverse=True)
        ranked = ranked[:top_k]

        logger.debug(
            "WikiSearchEngine: wiki=%s query=%r → %d results (from %d candidates)",
            self._wiki_id,
            query,
            len(ranked),
            hit_count,
        )

        return WikiSearchResult(
            query=query,
            results=ranked,
            wiki_summary=[wiki_summary],
        )
