"""Unit tests for ProjectSearchEngine — parallel fan-out and result merging."""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from app.core.project_search_engine import ProjectSearchEngine, ProjectSearchResult
from app.core.wiki_search_engine import ResultItem, WikiSearchResult, WikiSummaryItem


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_result_item(
    wiki_id: str = "wiki1",
    wiki_name: str = "Wiki One",
    page_title: str = "Page",
    score: float = 1.0,
) -> ResultItem:
    return ResultItem(
        wiki_id=wiki_id,
        wiki_name=wiki_name,
        page_title=page_title,
        snippet=f"Desc of {page_title}",
        score=score,
        neighbors=[],
    )


def _make_wiki_summary(
    wiki_id: str = "wiki1",
    wiki_name: str = "Wiki One",
    match_count: int = 1,
    relevance: float = 0.5,
) -> WikiSummaryItem:
    return WikiSummaryItem(
        wiki_id=wiki_id,
        wiki_name=wiki_name,
        match_count=match_count,
        relevance=relevance,
    )


def _make_wiki_search_result(
    query: str = "test",
    results: list[ResultItem] | None = None,
    wiki_id: str = "wiki1",
    wiki_name: str = "Wiki One",
    match_count: int | None = None,
    relevance: float = 0.5,
) -> WikiSearchResult:
    if results is None:
        results = []
    if match_count is None:
        match_count = len(results)
    summary = _make_wiki_summary(
        wiki_id=wiki_id,
        wiki_name=wiki_name,
        match_count=match_count,
        relevance=relevance if match_count > 0 else 0.0,
    )
    return WikiSearchResult(query=query, results=results, wiki_summary=[summary])


def _make_engine_mock(wiki_search_result: WikiSearchResult) -> MagicMock:
    """Return a mock WikiSearchEngine whose search() returns the given result."""
    engine = MagicMock()
    engine.search = AsyncMock(return_value=wiki_search_result)
    return engine


def _make_project_engine(factory_map: dict[str, MagicMock]) -> tuple[ProjectSearchEngine, MagicMock]:
    """Build a ProjectSearchEngine with a tracking factory callable."""
    factory = MagicMock(side_effect=lambda wiki_id, wiki_name: factory_map[(wiki_id, wiki_name)])
    engine = ProjectSearchEngine(wiki_search_engine_factory=factory)
    return engine, factory


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_single_wiki_fan_out_returns_merged_results():
    """Single wiki fan-out returns results from that wiki in ProjectSearchResult."""
    result_item = _make_result_item(wiki_id="w1", page_title="Auth", score=0.9)
    wiki_result = _make_wiki_search_result(results=[result_item], wiki_id="w1", wiki_name="Wiki A")
    engine_mock = _make_engine_mock(wiki_result)

    proj_engine, _ = _make_project_engine({("w1", "Wiki A"): engine_mock})
    proj_result = await proj_engine.search("auth", wikis=[("w1", "Wiki A")])

    assert isinstance(proj_result, ProjectSearchResult)
    assert proj_result.query == "auth"
    assert len(proj_result.results) == 1
    assert proj_result.results[0].page_title == "Auth"


@pytest.mark.asyncio
async def test_multi_wiki_fan_out_merges_and_sorts_by_score():
    """Multi-wiki fan-out merges results from all wikis, sorted by score descending."""
    item_a = _make_result_item(wiki_id="w1", page_title="PageA", score=0.3)
    item_b = _make_result_item(wiki_id="w2", page_title="PageB", score=0.9)
    item_c = _make_result_item(wiki_id="w2", page_title="PageC", score=0.6)

    wiki_result_1 = _make_wiki_search_result(results=[item_a], wiki_id="w1", wiki_name="Wiki A")
    wiki_result_2 = _make_wiki_search_result(results=[item_b, item_c], wiki_id="w2", wiki_name="Wiki B")

    engine1 = _make_engine_mock(wiki_result_1)
    engine2 = _make_engine_mock(wiki_result_2)

    proj_engine, _ = _make_project_engine({
        ("w1", "Wiki A"): engine1,
        ("w2", "Wiki B"): engine2,
    })
    proj_result = await proj_engine.search(
        "query", wikis=[("w1", "Wiki A"), ("w2", "Wiki B")]
    )

    assert len(proj_result.results) == 3
    scores = [r.score for r in proj_result.results]
    assert scores == sorted(scores, reverse=True)
    assert scores[0] == pytest.approx(0.9)


@pytest.mark.asyncio
async def test_zero_hit_wiki_excluded_from_results_but_in_wiki_summary():
    """Zero-hit wiki results excluded from merged results, but present in wiki_summary."""
    item_a = _make_result_item(wiki_id="w1", page_title="PageA", score=0.8)
    wiki_result_1 = _make_wiki_search_result(results=[item_a], wiki_id="w1", wiki_name="Wiki A")
    wiki_result_2 = _make_wiki_search_result(
        results=[], wiki_id="w2", wiki_name="Wiki B", match_count=0, relevance=0.0
    )

    engine1 = _make_engine_mock(wiki_result_1)
    engine2 = _make_engine_mock(wiki_result_2)

    proj_engine, _ = _make_project_engine({
        ("w1", "Wiki A"): engine1,
        ("w2", "Wiki B"): engine2,
    })
    proj_result = await proj_engine.search(
        "query", wikis=[("w1", "Wiki A"), ("w2", "Wiki B")]
    )

    # results only contain items from w1
    assert all(r.wiki_id == "w1" for r in proj_result.results)

    # wiki_summary has entries for both wikis
    summary_ids = {s.wiki_id for s in proj_result.wiki_summary}
    assert "w1" in summary_ids
    assert "w2" in summary_ids

    # zero-hit wiki has match_count=0 and relevance=0.0
    zero_summary = next(s for s in proj_result.wiki_summary if s.wiki_id == "w2")
    assert zero_summary.match_count == 0
    assert zero_summary.relevance == pytest.approx(0.0)


@pytest.mark.asyncio
async def test_top_k_caps_final_result_count():
    """top_k caps the final number of merged results returned."""
    items = [_make_result_item(page_title=f"Page{i}", score=float(i)) for i in range(15)]
    wiki_result = _make_wiki_search_result(results=items, wiki_id="w1", wiki_name="Wiki A")
    engine_mock = _make_engine_mock(wiki_result)

    proj_engine, _ = _make_project_engine({("w1", "Wiki A"): engine_mock})
    proj_result = await proj_engine.search("query", wikis=[("w1", "Wiki A")], top_k=5)

    assert len(proj_result.results) <= 5


@pytest.mark.asyncio
async def test_wiki_summary_present_even_if_all_wikis_zero_hits():
    """wiki_summary is always present even when all wikis return zero hits."""
    wiki_result_1 = _make_wiki_search_result(
        results=[], wiki_id="w1", wiki_name="Wiki A", match_count=0, relevance=0.0
    )
    wiki_result_2 = _make_wiki_search_result(
        results=[], wiki_id="w2", wiki_name="Wiki B", match_count=0, relevance=0.0
    )

    engine1 = _make_engine_mock(wiki_result_1)
    engine2 = _make_engine_mock(wiki_result_2)

    proj_engine, _ = _make_project_engine({
        ("w1", "Wiki A"): engine1,
        ("w2", "Wiki B"): engine2,
    })
    proj_result = await proj_engine.search(
        "nothing", wikis=[("w1", "Wiki A"), ("w2", "Wiki B")]
    )

    assert proj_result.results == []
    assert len(proj_result.wiki_summary) == 2


@pytest.mark.asyncio
async def test_fan_out_is_parallel_asyncio_gather():
    """asyncio.gather is called with all wiki engine search coroutines simultaneously."""
    wiki_result = _make_wiki_search_result(results=[], wiki_id="w1", wiki_name="Wiki A")
    engine1 = _make_engine_mock(wiki_result)
    wiki_result2 = _make_wiki_search_result(results=[], wiki_id="w2", wiki_name="Wiki B")
    engine2 = _make_engine_mock(wiki_result2)

    proj_engine, _ = _make_project_engine({
        ("w1", "Wiki A"): engine1,
        ("w2", "Wiki B"): engine2,
    })

    with patch("app.core.project_search_engine.asyncio.gather", wraps=asyncio.gather) as gather_mock:
        await proj_engine.search("query", wikis=[("w1", "Wiki A"), ("w2", "Wiki B")])

    # asyncio.gather must have been called exactly once with two awaitables.
    gather_mock.assert_called_once()
    args = gather_mock.call_args[0]
    assert len(args) == 2


@pytest.mark.asyncio
async def test_hop_depth_passed_through_to_each_wiki_engine():
    """hop_depth argument is forwarded to each WikiSearchEngine.search() call."""
    wiki_result = _make_wiki_search_result(results=[], wiki_id="w1", wiki_name="Wiki A")
    wiki_result2 = _make_wiki_search_result(results=[], wiki_id="w2", wiki_name="Wiki B")
    engine1 = _make_engine_mock(wiki_result)
    engine2 = _make_engine_mock(wiki_result2)

    proj_engine, _ = _make_project_engine({
        ("w1", "Wiki A"): engine1,
        ("w2", "Wiki B"): engine2,
    })
    await proj_engine.search(
        "query", wikis=[("w1", "Wiki A"), ("w2", "Wiki B")], hop_depth=3
    )

    engine1.search.assert_called_once()
    _, kwargs1 = engine1.search.call_args
    assert kwargs1.get("hop_depth") == 3

    engine2.search.assert_called_once()
    _, kwargs2 = engine2.search.call_args
    assert kwargs2.get("hop_depth") == 3


@pytest.mark.asyncio
async def test_results_sorted_by_score_descending_across_wikis():
    """Results from multiple wikis are merged and sorted by score descending."""
    # Deliberately order items so cross-wiki interleaving is required.
    item_low = _make_result_item(wiki_id="w1", page_title="Low", score=0.1)
    item_high = _make_result_item(wiki_id="w2", page_title="High", score=0.95)
    item_mid = _make_result_item(wiki_id="w1", page_title="Mid", score=0.5)

    wiki_result_1 = _make_wiki_search_result(results=[item_low, item_mid], wiki_id="w1", wiki_name="Wiki A")
    wiki_result_2 = _make_wiki_search_result(results=[item_high], wiki_id="w2", wiki_name="Wiki B")

    engine1 = _make_engine_mock(wiki_result_1)
    engine2 = _make_engine_mock(wiki_result_2)

    proj_engine, _ = _make_project_engine({
        ("w1", "Wiki A"): engine1,
        ("w2", "Wiki B"): engine2,
    })
    proj_result = await proj_engine.search(
        "query", wikis=[("w1", "Wiki A"), ("w2", "Wiki B")]
    )

    titles = [r.page_title for r in proj_result.results]
    assert titles == ["High", "Mid", "Low"]


@pytest.mark.asyncio
async def test_empty_wikis_list_returns_empty_results_and_summary():
    """Empty wikis list returns empty results and empty wiki_summary."""
    factory = MagicMock()
    proj_engine = ProjectSearchEngine(wiki_search_engine_factory=factory)

    proj_result = await proj_engine.search("query", wikis=[])

    assert proj_result.results == []
    assert proj_result.wiki_summary == []
    factory.assert_not_called()


@pytest.mark.asyncio
async def test_factory_called_once_per_wiki():
    """The factory is called exactly once for each wiki in the list."""
    wiki_result_1 = _make_wiki_search_result(results=[], wiki_id="w1", wiki_name="Wiki A")
    wiki_result_2 = _make_wiki_search_result(results=[], wiki_id="w2", wiki_name="Wiki B")
    wiki_result_3 = _make_wiki_search_result(results=[], wiki_id="w3", wiki_name="Wiki C")

    engine1 = _make_engine_mock(wiki_result_1)
    engine2 = _make_engine_mock(wiki_result_2)
    engine3 = _make_engine_mock(wiki_result_3)

    factory_map = {
        ("w1", "Wiki A"): engine1,
        ("w2", "Wiki B"): engine2,
        ("w3", "Wiki C"): engine3,
    }
    proj_engine, factory = _make_project_engine(factory_map)

    await proj_engine.search(
        "query",
        wikis=[("w1", "Wiki A"), ("w2", "Wiki B"), ("w3", "Wiki C")],
    )

    assert factory.call_count == 3
    factory.assert_any_call("w1", "Wiki A")
    factory.assert_any_call("w2", "Wiki B")
    factory.assert_any_call("w3", "Wiki C")
