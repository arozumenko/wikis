"""Wiki page search adapter — PostgreSQL full-text search with normalized scores.

This module is pure Python: no LLM, no FAISS.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from app.core.wiki_page_index import WikiPageIndex

logger = logging.getLogger(__name__)


@dataclass
class PageHit:
    """A single search result with a normalized relevance score.

    Attributes:
        page_title: Title of the matching wiki page.
        base_score: Relevance score min-max normalized to [0.0, 1.0] within
            this search call.  Higher is better.
        snippet: Short description of the page from WikiPageIndex.
    """

    page_title: str
    base_score: float
    snippet: str


class WikiPageSearchAdapter:
    """Queries the PostgreSQL wiki_page table and returns normalized PageHits.

    Supports two backends:
    - **PostgreSQL**: Uses ``tsvector``/``tsquery`` with ``ts_rank`` for ranked
      full-text search (weighted: title A, description B, content C).
    - **SQLite** (test/dev fallback): Uses ``LIKE`` matching — no ranking,
      all hits get score 1.0.

    Args:
        wiki_id: Identifier of the wiki (used for filtering and logging).
        session_factory: SQLAlchemy async session factory.
        page_index: WikiPageIndex used to look up page descriptions.
    """

    def __init__(
        self,
        wiki_id: str,
        session_factory: async_sessionmaker[AsyncSession],
        page_index: WikiPageIndex,
        *,
        is_postgres: bool | None = None,
    ) -> None:
        self._wiki_id = wiki_id
        self._session_factory = session_factory
        self._page_index = page_index
        # Auto-detect from engine URL if not explicitly provided.
        if is_postgres is None:
            try:
                engine = session_factory.kw.get("bind") or getattr(session_factory, "bind", None)
                self._is_postgres = engine is not None and "postgresql" in str(engine.url)
            except Exception:
                self._is_postgres = False
        else:
            self._is_postgres = is_postgres

    async def search(self, query: str, top_k: int = 10) -> list[PageHit]:
        """Search wiki pages and return normalized hits.

        Scores are min-max normalized per call so the best result has
        base_score=1.0 and the worst has base_score=0.0.  When all scores
        are equal (or there is only one result) every hit gets 1.0.

        Args:
            query: Free-text search query.
            top_k: Maximum number of results to return. Defaults to 10.

        Returns:
            List of PageHit sorted by base_score descending.  Empty list
            when the search returns no results.
        """
        async with self._session_factory() as session:
            raw = await self._execute_search(session, query, top_k)

        if not raw:
            logger.debug("WikiPageSearchAdapter: no results for wiki=%s query=%r", self._wiki_id, query)
            return []

        # Extract raw scores.
        scores = [row[1] for row in raw]
        min_score = min(scores)
        max_score = max(scores)

        if max_score == min_score:
            normalized = [1.0] * len(scores)
        else:
            span = max_score - min_score
            normalized = [(s - min_score) / span for s in scores]

        hits: list[PageHit] = []
        for (title, _score), norm_score in zip(raw, normalized):
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

    async def _execute_search(
        self,
        session: AsyncSession,
        query: str,
        top_k: int,
    ) -> list[tuple[str, float]]:
        """Run the FTS query and return ``(page_title, rank)`` tuples."""
        if self._is_postgres:
            return await self._search_postgres(session, query, top_k)
        return await self._search_sqlite(session, query, top_k)

    async def _search_postgres(
        self,
        session: AsyncSession,
        query: str,
        top_k: int,
    ) -> list[tuple[str, float]]:
        """PostgreSQL full-text search with ts_rank."""
        result = await session.execute(
            text(
                "SELECT page_title, ts_rank(search_vector, websearch_to_tsquery('english', :q)) AS rank "
                "FROM wiki_page "
                "WHERE wiki_id = :wiki_id "
                "  AND search_vector @@ websearch_to_tsquery('english', :q) "
                "ORDER BY rank DESC "
                "LIMIT :top_k"
            ),
            {"q": query, "wiki_id": self._wiki_id, "top_k": top_k},
        )
        return [(row[0], float(row[1])) for row in result.fetchall()]

    async def _search_sqlite(
        self,
        session: AsyncSession,
        query: str,
        top_k: int,
    ) -> list[tuple[str, float]]:
        """SQLite full-text search via the FTS5 ``wiki_page_fts`` virtual table.

        Uses ``bm25(wiki_page_fts, 10.0, 5.0, 1.0)`` so weights mirror the
        PostgreSQL A/B/C scaling (title > description > content). bm25 returns
        negative values where lower means better; we negate so callers get
        higher-is-better scores compatible with the PG ts_rank path.

        Falls back to LIKE matching if the FTS5 vtable is missing (e.g. legacy
        databases that pre-date the migration in :func:`app.db._ensure_wiki_page_fts`).
        """
        match_expr = _build_fts5_match(query)
        if not match_expr:
            return []

        try:
            result = await session.execute(
                text(
                    "SELECT wp.page_title, "
                    "       -bm25(wiki_page_fts, 10.0, 5.0, 1.0) AS rank "
                    "FROM wiki_page_fts "
                    "JOIN wiki_page wp ON wp.rowid = wiki_page_fts.rowid "
                    "WHERE wiki_page_fts MATCH :match "
                    "  AND wp.wiki_id = :wiki_id "
                    "ORDER BY rank DESC "
                    "LIMIT :top_k"
                ),
                {"match": match_expr, "wiki_id": self._wiki_id, "top_k": top_k},
            )
            return [(row[0], float(row[1])) for row in result.fetchall()]
        except Exception as e:
            # FTS5 vtable missing or malformed query — fall back to LIKE.
            logger.debug(
                "WikiPageSearchAdapter: FTS5 search failed (%s) — falling back to LIKE",
                e,
            )
            like_pattern = f"%{query}%"
            result = await session.execute(
                text(
                    "SELECT page_title, 1.0 AS rank "
                    "FROM wiki_page "
                    "WHERE wiki_id = :wiki_id "
                    "  AND (page_title LIKE :q OR content LIKE :q OR description LIKE :q) "
                    "LIMIT :top_k"
                ),
                {"q": like_pattern, "wiki_id": self._wiki_id, "top_k": top_k},
            )
            return [(row[0], float(row[1])) for row in result.fetchall()]


def _build_fts5_match(query: str) -> str:
    """Convert free-text query into a safe FTS5 MATCH expression.

    Each whitespace-separated token is wrapped in double quotes to escape any
    FTS5 syntax characters, then joined with OR so partial matches still rank.
    Returns the empty string when no usable tokens remain.
    """
    tokens = [t for t in query.replace('"', " ").split() if t]
    if not tokens:
        return ""
    quoted = [f'"{t}"' for t in tokens]
    return " OR ".join(quoted)
