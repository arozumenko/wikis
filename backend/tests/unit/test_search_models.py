"""Tests for Pydantic response models in app.models.search."""

from __future__ import annotations

import json

import pytest
from app.models.search import (
    PageListItem,
    PageNeighbor,
    PageNeighborsResponse,
    ProjectSearchResponse,
    SearchResultItem,
    WikiPageResponse,
    WikiSearchResponse,
    WikiSummaryItem,
)
from pydantic import ValidationError

# ---------------------------------------------------------------------------
# Helper factories
# ---------------------------------------------------------------------------


def _make_neighbor(**kwargs) -> PageNeighbor:
    defaults = {"title": "Overview", "rel": "links_to"}
    return PageNeighbor(**{**defaults, **kwargs})


def _make_result_item(**kwargs) -> SearchResultItem:
    defaults = {
        "wiki_id": "wid-1",
        "wiki_name": "My Wiki",
        "page_title": "Architecture",
        "snippet": "This module handles auth.",
        "score": 0.87,
        "neighbors": [_make_neighbor()],
    }
    return SearchResultItem(**{**defaults, **kwargs})


def _make_wiki_summary(**kwargs) -> WikiSummaryItem:
    defaults = {
        "wiki_id": "wid-1",
        "wiki_name": "My Wiki",
        "match_count": 3,
        "relevance": 0.91,
    }
    return WikiSummaryItem(**{**defaults, **kwargs})


# ---------------------------------------------------------------------------
# 1. WikiSearchResponse can be instantiated and serialised to dict
# ---------------------------------------------------------------------------


def test_wiki_search_response_instantiation_and_serialization():
    """WikiSearchResponse can be built and round-trips through .model_dump()."""
    response = WikiSearchResponse(
        query="auth flow",
        results=[_make_result_item()],
        wiki_summary=[_make_wiki_summary()],
    )
    data = response.model_dump()

    assert data["query"] == "auth flow"
    assert len(data["results"]) == 1
    assert data["results"][0]["page_title"] == "Architecture"
    assert len(data["wiki_summary"]) == 1
    assert data["wiki_summary"][0]["wiki_id"] == "wid-1"


# ---------------------------------------------------------------------------
# 2. ProjectSearchResponse structure matches WikiSearchResponse
# ---------------------------------------------------------------------------


def test_project_search_response_same_fields_as_wiki_search_response():
    """ProjectSearchResponse exposes the same top-level fields as WikiSearchResponse."""
    wiki_fields = set(WikiSearchResponse.model_fields.keys())
    project_fields = set(ProjectSearchResponse.model_fields.keys())
    assert wiki_fields == project_fields


def test_project_search_response_instantiation():
    """ProjectSearchResponse can be instantiated with the same data shape."""
    response = ProjectSearchResponse(
        query="database",
        results=[_make_result_item()],
        wiki_summary=[_make_wiki_summary()],
    )
    assert response.query == "database"
    assert len(response.results) == 1


# ---------------------------------------------------------------------------
# 3. PageNeighbor.rel rejects values outside the Literal
# ---------------------------------------------------------------------------


def test_page_neighbor_valid_rel_links_to():
    neighbor = PageNeighbor(title="Home", rel="links_to")
    assert neighbor.rel == "links_to"


def test_page_neighbor_valid_rel_linked_from():
    neighbor = PageNeighbor(title="Home", rel="linked_from")
    assert neighbor.rel == "linked_from"


def test_page_neighbor_invalid_rel_raises_validation_error():
    """PageNeighbor rejects a rel value not in the Literal."""
    with pytest.raises(ValidationError):
        PageNeighbor(title="Home", rel="references")  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# 4. PageListItem.section defaults to None
# ---------------------------------------------------------------------------


def test_page_list_item_section_defaults_to_none():
    """PageListItem.section is None when not provided."""
    item = PageListItem(page_title="Overview", description="High-level overview.")
    assert item.section is None


def test_page_list_item_section_accepts_string():
    """PageListItem.section accepts an explicit string value."""
    item = PageListItem(page_title="Overview", description="Overview.", section="Intro")
    assert item.section == "Intro"


# ---------------------------------------------------------------------------
# 5. SearchResultItem round-trips through JSON
# ---------------------------------------------------------------------------


def test_search_result_item_json_round_trip():
    """SearchResultItem serialises to JSON and deserialises back without data loss."""
    original = _make_result_item(
        score=0.75,
        neighbors=[
            _make_neighbor(title="Quickstart", rel="linked_from"),
        ],
    )
    json_str = original.model_dump_json()
    restored = SearchResultItem.model_validate(json.loads(json_str))

    assert restored.wiki_id == original.wiki_id
    assert restored.score == original.score
    assert restored.neighbors[0].title == "Quickstart"
    assert restored.neighbors[0].rel == "linked_from"


# ---------------------------------------------------------------------------
# 6. WikiPageResponse.sections is a list of strings
# ---------------------------------------------------------------------------


def test_wiki_page_response_sections_is_list_of_strings():
    """WikiPageResponse.sections stores and returns a list of strings."""
    page = WikiPageResponse(
        wiki_id="wid-1",
        page_title="Architecture",
        content="# Architecture\n\nDetails here.",
        sections=["Overview", "Components", "Data Flow"],
    )
    assert isinstance(page.sections, list)
    assert all(isinstance(s, str) for s in page.sections)
    assert page.sections == ["Overview", "Components", "Data Flow"]


def test_wiki_page_response_empty_sections():
    """WikiPageResponse accepts an empty sections list."""
    page = WikiPageResponse(
        wiki_id="wid-1",
        page_title="Empty",
        content="",
        sections=[],
    )
    assert page.sections == []


# ---------------------------------------------------------------------------
# 7. PageNeighborsResponse has separate links_to and linked_from lists
# ---------------------------------------------------------------------------


def test_page_neighbors_response_separate_link_lists():
    """PageNeighborsResponse keeps links_to and linked_from as independent lists."""
    response = PageNeighborsResponse(
        wiki_id="wid-1",
        page_title="Auth Module",
        links_to=["JWT Service", "Session Store"],
        linked_from=["Login Page", "Middleware"],
    )
    assert response.links_to == ["JWT Service", "Session Store"]
    assert response.linked_from == ["Login Page", "Middleware"]
    assert set(response.links_to).isdisjoint(set(response.linked_from))


def test_page_neighbors_response_empty_lists():
    """PageNeighborsResponse accepts empty neighbor lists."""
    response = PageNeighborsResponse(
        wiki_id="wid-1",
        page_title="Isolated",
        links_to=[],
        linked_from=[],
    )
    assert response.links_to == []
    assert response.linked_from == []


# ---------------------------------------------------------------------------
# 8. WikiSummaryItem.relevance accepts float values
# ---------------------------------------------------------------------------


def test_wiki_summary_item_relevance_accepts_float():
    """WikiSummaryItem.relevance stores the provided float value."""
    item = _make_wiki_summary(relevance=0.63)
    assert item.relevance == pytest.approx(0.63)


def test_wiki_summary_item_relevance_accepts_zero():
    """WikiSummaryItem.relevance accepts 0.0 as a valid float."""
    item = _make_wiki_summary(relevance=0.0)
    assert item.relevance == 0.0


def test_wiki_summary_item_relevance_accepts_one():
    """WikiSummaryItem.relevance accepts 1.0 as a valid float."""
    item = _make_wiki_summary(relevance=1.0)
    assert item.relevance == 1.0
