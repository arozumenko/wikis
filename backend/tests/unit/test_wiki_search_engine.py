"""Unit tests for WikiSearchEngine — graph expansion and composite re-ranking."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from app.core.wiki_page_index import PageMeta, WikiPageIndex
from app.core.wiki_page_search import PageHit, WikiPageSearchAdapter
from app.core.wiki_search_engine import (
    NeighborItem,
    ResultItem,
    WikiSearchEngine,
    WikiSearchResult,
    WikiSummaryItem,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_page_meta(title: str) -> PageMeta:
    return PageMeta(title=title, description=f"Desc of {title}", file_path=f"{title}.md")


def _make_page_index(
    titles: list[str],
    forward: dict[str, list[str]] | None = None,
    backward: dict[str, list[str]] | None = None,
) -> WikiPageIndex:
    """Build a WikiPageIndex mock with controlled pages / neighbors / backlinks."""
    index = MagicMock(spec=WikiPageIndex)
    index.pages = {t: _make_page_meta(t) for t in titles}

    _forward = forward or {}
    _backward = backward or {}

    def _neighbors(title: str, hop_depth: int = 1) -> list[PageMeta]:
        neighbors_titles = _forward.get(title, [])
        return [_make_page_meta(t) for t in neighbors_titles if t in index.pages]

    def _backlinks(title: str) -> list[str]:
        return list(_backward.get(title, []))

    def _get_description(title: str) -> str:
        return f"Desc of {title}"

    index.neighbors.side_effect = _neighbors
    index.backlinks.side_effect = _backlinks
    index.get_description.side_effect = _get_description
    return index


def _make_engine(
    titles: list[str],
    hits: list[tuple[str, float]],  # (page_title, base_score) pairs
    forward: dict[str, list[str]] | None = None,
    backward: dict[str, list[str]] | None = None,
    wiki_id: str = "wiki1",
    wiki_name: str = "Wiki One",
) -> WikiSearchEngine:
    """Build a WikiSearchEngine with a mocked adapter and page index."""
    page_index = _make_page_index(titles, forward, backward)
    db = MagicMock()

    engine = WikiSearchEngine(wiki_id, wiki_name, db, page_index)

    # Patch the adapter's search method directly on the instance.
    def _search(query: str, top_k: int = 10) -> list[PageHit]:
        return [
            PageHit(
                page_title=title,
                base_score=score,
                snippet=f"Desc of {title}",
            )
            for title, score in hits[:top_k]
        ]

    engine._adapter.search = _search
    return engine


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_basic_search_returns_wiki_search_result():
    """Basic search returns a WikiSearchResult instance."""
    engine = _make_engine(
        titles=["Auth", "API"],
        hits=[("Auth", 1.0), ("API", 0.5)],
    )
    result = await engine.search("authentication")
    assert isinstance(result, WikiSearchResult)
    assert result.query == "authentication"


@pytest.mark.asyncio
async def test_wiki_summary_always_present_even_on_empty_results():
    """wiki_summary is returned even when FTS5 produces no hits."""
    engine = _make_engine(titles=["Auth"], hits=[])
    result = await engine.search("nothing")
    assert len(result.wiki_summary) == 1
    summary = result.wiki_summary[0]
    assert isinstance(summary, WikiSummaryItem)
    assert summary.match_count == 0
    assert summary.relevance == 0.0
    assert result.results == []


@pytest.mark.asyncio
async def test_reranking_high_backlink_count_increases_score():
    """Pages with more backlinks get a higher graph_centrality bonus."""
    # Two pages: "Popular" has 3 backlinks, "Unknown" has 0.
    # Both have base_score=1.0 to isolate the backlink effect.
    engine = _make_engine(
        titles=["Popular", "Unknown", "A", "B", "C"],
        hits=[("Popular", 1.0), ("Unknown", 1.0)],
        backward={"Popular": ["A", "B", "C"], "Unknown": []},
    )
    result = await engine.search("query")
    scores = {r.page_title: r.score for r in result.results}
    assert scores["Popular"] > scores["Unknown"]


@pytest.mark.asyncio
async def test_reranking_matched_neighbors_increase_score():
    """Pages whose neighbors appear in the FTS5 hit set score higher."""
    # "Hub" links to "Auth" and "API" which are also hits → high density.
    # "Isolated" has no neighbors → density = 0.
    engine = _make_engine(
        titles=["Hub", "Isolated", "Auth", "API"],
        hits=[("Hub", 1.0), ("Isolated", 1.0), ("Auth", 0.8), ("API", 0.8)],
        forward={"Hub": ["Auth", "API"], "Isolated": []},
    )
    result = await engine.search("query")
    scores = {r.page_title: r.score for r in result.results}
    assert scores["Hub"] > scores["Isolated"]


@pytest.mark.asyncio
async def test_reranking_wiki_relevance_factor():
    """wiki_relevance = hit_count / total_pages scales all scores."""
    # 2 hits out of 10 total pages → relevance = 0.2
    titles = [f"Page{i}" for i in range(10)]
    engine = _make_engine(
        titles=titles,
        hits=[("Page0", 1.0), ("Page1", 1.0)],
    )
    result = await engine.search("query")
    # relevance = 2/10 = 0.2; base_score=1.0, no centrality or density →
    # score = 1.0 × (1 + 0 + 0) × 0.2 = 0.2
    for item in result.results:
        assert abs(item.score - 0.2) < 1e-9
    assert result.wiki_summary[0].relevance == pytest.approx(0.2)


@pytest.mark.asyncio
async def test_results_capped_at_top_k():
    """Only top_k results are returned after re-ranking."""
    titles = [f"P{i}" for i in range(20)]
    hits = [(f"P{i}", 1.0 - i * 0.01) for i in range(20)]
    engine = _make_engine(titles=titles, hits=hits)
    result = await engine.search("query", top_k=5)
    assert len(result.results) <= 5


@pytest.mark.asyncio
async def test_neighbor_cap_at_50_per_result():
    """neighbors list is capped at 50 items per ResultItem."""
    # Create a page with 40 forward links and 20 backlinks = 60 total.
    forward_titles = [f"Fwd{i}" for i in range(40)]
    backward_titles = [f"Back{i}" for i in range(20)]
    all_titles = ["Hub"] + forward_titles + backward_titles
    engine = _make_engine(
        titles=all_titles,
        hits=[("Hub", 1.0)],
        forward={"Hub": forward_titles},
        backward={"Hub": backward_titles},
    )
    result = await engine.search("query")
    assert len(result.results) == 1
    assert len(result.results[0].neighbors) <= 50


@pytest.mark.asyncio
async def test_rel_links_to_for_forward_links():
    """Forward links (pages the result page links TO) have rel='links_to'."""
    engine = _make_engine(
        titles=["Hub", "Target"],
        hits=[("Hub", 1.0)],
        forward={"Hub": ["Target"]},
    )
    result = await engine.search("query")
    hub_result = result.results[0]
    links_to = [n for n in hub_result.neighbors if n.rel == "links_to"]
    assert any(n.title == "Target" for n in links_to)


@pytest.mark.asyncio
async def test_rel_linked_from_for_backlinks():
    """Backlinks (pages that link TO the result page) have rel='linked_from'."""
    engine = _make_engine(
        titles=["Target", "Linker"],
        hits=[("Target", 1.0)],
        backward={"Target": ["Linker"]},
    )
    result = await engine.search("query")
    target_result = result.results[0]
    linked_from = [n for n in target_result.neighbors if n.rel == "linked_from"]
    assert any(n.title == "Linker" for n in linked_from)


@pytest.mark.asyncio
async def test_hop_depth_passed_through_to_page_index():
    """hop_depth is forwarded to page_index.neighbors() calls."""
    page_index = _make_page_index(titles=["A", "B"])
    db = MagicMock()
    engine = WikiSearchEngine("wiki1", "Wiki One", db, page_index)
    engine._adapter.search = lambda q, top_k=10: [PageHit("A", 1.0, "Desc of A")]

    await engine.search("query", hop_depth=3)

    # neighbors() must have been called with hop_depth=3 at some point
    calls = page_index.neighbors.call_args_list
    assert any(call.args[1] == 3 or call.kwargs.get("hop_depth") == 3 for call in calls)


@pytest.mark.asyncio
async def test_pages_with_zero_neighbors_no_division_by_zero():
    """Pages with no forward neighbors produce neighborhood_density=0.0 safely."""
    engine = _make_engine(
        titles=["Isolated"],
        hits=[("Isolated", 1.0)],
        forward={"Isolated": []},
        backward={},
    )
    # Should not raise ZeroDivisionError
    result = await engine.search("query")
    assert len(result.results) == 1
    assert result.results[0].score >= 0.0


@pytest.mark.asyncio
async def test_empty_fts5_results_empty_results_list_wiki_summary_present():
    """Empty FTS5 results → empty results list, but wiki_summary is still returned."""
    engine = _make_engine(titles=["Auth", "API"], hits=[])
    result = await engine.search("nothing")
    assert result.results == []
    assert len(result.wiki_summary) == 1
    assert result.wiki_summary[0].wiki_id == "wiki1"
    assert result.wiki_summary[0].match_count == 0


@pytest.mark.asyncio
async def test_results_sorted_by_score_descending():
    """Results are returned in descending score order."""
    engine = _make_engine(
        titles=["A", "B", "C"],
        hits=[("A", 0.3), ("B", 1.0), ("C", 0.6)],
    )
    result = await engine.search("query")
    scores = [r.score for r in result.results]
    assert scores == sorted(scores, reverse=True)


@pytest.mark.asyncio
async def test_wiki_summary_contains_correct_wiki_id_and_name():
    """wiki_summary reflects the wiki_id and wiki_name passed to the engine."""
    engine = _make_engine(
        titles=["Page"],
        hits=[("Page", 1.0)],
        wiki_id="my-wiki",
        wiki_name="My Awesome Wiki",
    )
    result = await engine.search("query")
    assert result.wiki_summary[0].wiki_id == "my-wiki"
    assert result.wiki_summary[0].wiki_name == "My Awesome Wiki"


@pytest.mark.asyncio
async def test_snippet_sourced_from_page_hit():
    """ResultItem.snippet comes from the PageHit returned by the adapter."""
    engine = _make_engine(
        titles=["DocPage"],
        hits=[("DocPage", 1.0)],
    )
    result = await engine.search("query")
    assert result.results[0].snippet == "Desc of DocPage"


@pytest.mark.asyncio
async def test_reranking_formula_full_arithmetic():
    """Verify the complete re-ranking formula: base × (1 + centrality + density) × relevance.

    Setup:
      - 4 total pages: "Target", "Linker", "Neighbor", "Other"
      - hits: [("Target", 0.5), ("Neighbor", 0.5)]  → hit_count=2, relevance=2/4=0.5
      - "Target" has 1 backlink ("Linker") → centrality = 1/4 = 0.25
      - "Target" has 1 forward neighbor ("Neighbor") which IS in hit_titles
        → density = 1/1 = 1.0
      - Expected score for "Target":
          0.5 × (1 + 0.25 + 1.0) × 0.5 = 0.5 × 2.25 × 0.5 = 0.5625
    """
    engine = _make_engine(
        titles=["Target", "Linker", "Neighbor", "Other"],
        hits=[("Target", 0.5), ("Neighbor", 0.5)],
        forward={"Target": ["Neighbor"]},
        backward={"Target": ["Linker"]},
    )
    result = await engine.search("query")

    target = next(r for r in result.results if r.page_title == "Target")
    assert target.score == pytest.approx(0.5625, rel=1e-6)
