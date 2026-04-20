"""Unit tests for WikiPageSearchAdapter — normalized PostgreSQL FTS search hits."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from app.core.wiki_page_search import PageHit, WikiPageSearchAdapter


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_page_index(descriptions: dict[str, str]) -> MagicMock:
    """Build a mock WikiPageIndex where get_description returns from *descriptions*."""
    page_index = MagicMock()
    page_index.get_description.side_effect = lambda title: descriptions.get(title, "")
    return page_index


def _make_session_factory(rows: list[tuple[str, float]]):
    """Build a mock async session factory that returns *rows* from execute().

    Each row is (page_title, rank).
    """
    mock_result = MagicMock()
    mock_result.fetchall.return_value = rows

    mock_session = AsyncMock()
    mock_session.execute = AsyncMock(return_value=mock_result)
    mock_session.get_bind.return_value = MagicMock(url=MagicMock(__str__=lambda self: "sqlite+aiosqlite:///test"))

    # Build an async context manager that yields mock_session
    factory = MagicMock()
    ctx = AsyncMock()
    ctx.__aenter__ = AsyncMock(return_value=mock_session)
    ctx.__aexit__ = AsyncMock(return_value=False)
    factory.return_value = ctx

    return factory


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestWikiPageSearchAdapterNormalization:
    @pytest.mark.asyncio
    async def test_multiple_results_normalized_correctly(self):
        """Min score → 0.0, max score → 1.0 after normalization."""
        rows = [("Page A", 0.1), ("Page B", 0.5), ("Page C", 1.0)]
        factory = _make_session_factory(rows)
        descriptions = {"Page A": "Desc A", "Page B": "Desc B", "Page C": "Desc C"}
        page_index = _make_page_index(descriptions)

        adapter = WikiPageSearchAdapter("wiki1", factory, page_index)
        hits = await adapter.search("test query")

        assert len(hits) == 3
        scores = {h.page_title: h.base_score for h in hits}
        assert scores["Page A"] == pytest.approx(0.0)
        assert scores["Page C"] == pytest.approx(1.0)
        # Page B: (0.5 - 0.1) / (1.0 - 0.1) = 0.4/0.9
        assert scores["Page B"] == pytest.approx(0.4 / 0.9)

    @pytest.mark.asyncio
    async def test_single_result_normalized_to_one(self):
        """Single result should always have base_score of 1.0."""
        factory = _make_session_factory([("Only Page", 0.7)])
        page_index = _make_page_index({"Only Page": "Only description"})

        adapter = WikiPageSearchAdapter("wiki1", factory, page_index)
        hits = await adapter.search("query")

        assert len(hits) == 1
        assert hits[0].base_score == pytest.approx(1.0)

    @pytest.mark.asyncio
    async def test_all_equal_scores_normalized_to_one(self):
        """When all scores are equal, all base_scores should be 1.0."""
        rows = [("Page X", 0.5), ("Page Y", 0.5), ("Page Z", 0.5)]
        factory = _make_session_factory(rows)
        descriptions = {"Page X": "X desc", "Page Y": "Y desc", "Page Z": "Z desc"}
        page_index = _make_page_index(descriptions)

        adapter = WikiPageSearchAdapter("wiki1", factory, page_index)
        hits = await adapter.search("query")

        assert len(hits) == 3
        for hit in hits:
            assert hit.base_score == pytest.approx(1.0)

    @pytest.mark.asyncio
    async def test_empty_results_returns_empty_list(self):
        """When search returns nothing, adapter returns []."""
        factory = _make_session_factory([])
        page_index = _make_page_index({})

        adapter = WikiPageSearchAdapter("wiki1", factory, page_index)
        hits = await adapter.search("no match")

        assert hits == []


class TestWikiPageSearchAdapterSnippet:
    @pytest.mark.asyncio
    async def test_snippet_comes_from_page_index(self):
        """Snippet must always be from WikiPageIndex.get_description."""
        factory = _make_session_factory([("My Page", 0.8)])
        page_index = _make_page_index({"My Page": "Correct description from index"})

        adapter = WikiPageSearchAdapter("wiki1", factory, page_index)
        hits = await adapter.search("query")

        assert len(hits) == 1
        assert hits[0].snippet == "Correct description from index"

    @pytest.mark.asyncio
    async def test_snippet_empty_when_page_not_in_index(self):
        """If page_index has no description for a title, snippet is empty string."""
        factory = _make_session_factory([("Unknown Page", 0.5)])
        page_index = _make_page_index({})

        adapter = WikiPageSearchAdapter("wiki1", factory, page_index)
        hits = await adapter.search("query")

        assert len(hits) == 1
        assert hits[0].snippet == ""


class TestWikiPageSearchAdapterOrdering:
    @pytest.mark.asyncio
    async def test_results_ordered_by_score_descending(self):
        """Hits must be sorted by base_score descending (best match first)."""
        rows = [("Low Match", 0.1), ("High Match", 1.0), ("Mid Match", 0.5)]
        factory = _make_session_factory(rows)
        descriptions = {
            "Low Match": "low",
            "High Match": "high",
            "Mid Match": "mid",
        }
        page_index = _make_page_index(descriptions)

        adapter = WikiPageSearchAdapter("wiki1", factory, page_index)
        hits = await adapter.search("query")

        titles = [h.page_title for h in hits]
        assert titles == ["High Match", "Mid Match", "Low Match"]
        assert hits[0].base_score >= hits[1].base_score >= hits[2].base_score


class TestPageHitDataclass:
    def test_page_hit_fields(self):
        """PageHit dataclass has the required fields."""
        hit = PageHit(page_title="Test", base_score=0.5, snippet="A snippet")
        assert hit.page_title == "Test"
        assert hit.base_score == 0.5
        assert hit.snippet == "A snippet"
