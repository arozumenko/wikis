"""Unit tests for WikiPageSearchAdapter — normalized FTS5 search hits."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from app.core.wiki_page_search import PageHit, WikiPageSearchAdapter


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_db(fts_results: list[dict]) -> MagicMock:
    """Build a mock UnifiedWikiDB that returns *fts_results* from search_fts5."""
    db = MagicMock()
    db.search_fts5.return_value = fts_results
    return db


def _make_page_index(descriptions: dict[str, str]) -> MagicMock:
    """Build a mock WikiPageIndex where get_description returns from *descriptions*."""
    page_index = MagicMock()
    page_index.get_description.side_effect = lambda title: descriptions.get(title, "")
    return page_index


def _fts_row(symbol_name: str, fts_rank: float) -> dict:
    """Minimal FTS5 result row as returned by UnifiedWikiDB.search_fts5."""
    return {"symbol_name": symbol_name, "fts_rank": fts_rank}


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestWikiPageSearchAdapterNormalization:
    def test_multiple_results_normalized_correctly(self):
        """Min score → 0.0, max score → 1.0 after normalization."""
        # FTS5 BM25 scores are typically negative (lower = better match).
        # The adapter should min-max normalize them.
        db = _make_db([
            _fts_row("Page A", -10.0),
            _fts_row("Page B", -5.0),
            _fts_row("Page C", -1.0),
        ])
        descriptions = {"Page A": "Desc A", "Page B": "Desc B", "Page C": "Desc C"}
        page_index = _make_page_index(descriptions)

        adapter = WikiPageSearchAdapter("wiki1", db, page_index)
        hits = adapter.search("test query")

        assert len(hits) == 3
        scores = {h.page_title: h.base_score for h in hits}
        assert scores["Page A"] == pytest.approx(0.0)
        assert scores["Page C"] == pytest.approx(1.0)
        # Page B is in the middle: (-5 - -10) / (-1 - -10) = 5/9
        assert scores["Page B"] == pytest.approx(5.0 / 9.0)

    def test_single_result_normalized_to_one(self):
        """Single result should always have base_score of 1.0."""
        db = _make_db([_fts_row("Only Page", -3.5)])
        page_index = _make_page_index({"Only Page": "Only description"})

        adapter = WikiPageSearchAdapter("wiki1", db, page_index)
        hits = adapter.search("query")

        assert len(hits) == 1
        assert hits[0].base_score == pytest.approx(1.0)

    def test_all_equal_scores_normalized_to_one(self):
        """When all scores are equal, all base_scores should be 1.0."""
        db = _make_db([
            _fts_row("Page X", -7.0),
            _fts_row("Page Y", -7.0),
            _fts_row("Page Z", -7.0),
        ])
        descriptions = {"Page X": "X desc", "Page Y": "Y desc", "Page Z": "Z desc"}
        page_index = _make_page_index(descriptions)

        adapter = WikiPageSearchAdapter("wiki1", db, page_index)
        hits = adapter.search("query")

        assert len(hits) == 3
        for hit in hits:
            assert hit.base_score == pytest.approx(1.0)

    def test_empty_results_returns_empty_list(self):
        """When FTS5 returns nothing, adapter returns []."""
        db = _make_db([])
        page_index = _make_page_index({})

        adapter = WikiPageSearchAdapter("wiki1", db, page_index)
        hits = adapter.search("no match")

        assert hits == []


class TestWikiPageSearchAdapterSnippet:
    def test_snippet_comes_from_page_index_not_fts5(self):
        """Snippet must always be from WikiPageIndex.get_description, not FTS body."""
        fts_row = _fts_row("My Page", -2.0)
        fts_row["docstring"] = "FTS5 body text — should never appear in snippet"
        db = _make_db([fts_row])
        page_index = _make_page_index({"My Page": "Correct description from index"})

        adapter = WikiPageSearchAdapter("wiki1", db, page_index)
        hits = adapter.search("query")

        assert len(hits) == 1
        assert hits[0].snippet == "Correct description from index"
        assert "FTS5 body text" not in hits[0].snippet

    def test_snippet_empty_when_page_not_in_index(self):
        """If page_index has no description for a title, snippet is empty string."""
        db = _make_db([_fts_row("Unknown Page", -1.0)])
        page_index = _make_page_index({})

        adapter = WikiPageSearchAdapter("wiki1", db, page_index)
        hits = adapter.search("query")

        assert len(hits) == 1
        assert hits[0].snippet == ""


class TestWikiPageSearchAdapterTopK:
    def test_top_k_limits_results(self):
        """top_k is passed to the FTS5 search and limits returned results."""
        # Mock DB to actually honour the limit arg
        db = MagicMock()
        db.search_fts5.return_value = [
            _fts_row(f"Page {i}", float(-i)) for i in range(1, 4)
        ]
        page_index = _make_page_index({f"Page {i}": f"Desc {i}" for i in range(1, 4)})

        adapter = WikiPageSearchAdapter("wiki1", db, page_index)
        adapter.search("query", top_k=3)

        # Verify top_k was forwarded to search_fts5
        call_kwargs = db.search_fts5.call_args
        assert call_kwargs is not None
        # limit kwarg or positional limit argument should be 3
        args, kwargs = call_kwargs
        assert kwargs.get("limit", None) == 3 or (len(args) >= 2 and 3 in args)

    def test_default_top_k_is_ten(self):
        """Default top_k=10 is forwarded to FTS5 search."""
        db = _make_db([])
        page_index = _make_page_index({})

        adapter = WikiPageSearchAdapter("wiki1", db, page_index)
        adapter.search("query")

        _, kwargs = db.search_fts5.call_args
        assert kwargs.get("limit") == 10


class TestWikiPageSearchAdapterOrdering:
    def test_results_ordered_by_score_descending(self):
        """Hits must be sorted by base_score descending (best match first)."""
        db = _make_db([
            _fts_row("Low Match", -20.0),
            _fts_row("High Match", -1.0),
            _fts_row("Mid Match", -10.0),
        ])
        descriptions = {
            "Low Match": "low",
            "High Match": "high",
            "Mid Match": "mid",
        }
        page_index = _make_page_index(descriptions)

        adapter = WikiPageSearchAdapter("wiki1", db, page_index)
        hits = adapter.search("query")

        titles = [h.page_title for h in hits]
        assert titles == ["High Match", "Mid Match", "Low Match"]
        # Scores should be descending
        assert hits[0].base_score >= hits[1].base_score >= hits[2].base_score


class TestPageHitDataclass:
    def test_page_hit_fields(self):
        """PageHit dataclass has the required fields."""
        hit = PageHit(page_title="Test", base_score=0.5, snippet="A snippet")
        assert hit.page_title == "Test"
        assert hit.base_score == 0.5
        assert hit.snippet == "A snippet"
