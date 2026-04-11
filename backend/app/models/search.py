"""Pydantic response models for search endpoints and MCP tools."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel

# ---------------------------------------------------------------------------
# Shared building blocks
# ---------------------------------------------------------------------------


class PageNeighbor(BaseModel):
    """A neighboring wiki page with its relationship direction."""

    title: str
    rel: Literal["links_to", "linked_from"]


class SearchResultItem(BaseModel):
    """A single search result item from a wiki page."""

    wiki_id: str
    wiki_name: str
    page_title: str
    snippet: str
    score: float
    neighbors: list[PageNeighbor]


class WikiSummaryItem(BaseModel):
    """Per-wiki aggregate summary within a search response."""

    wiki_id: str
    wiki_name: str
    match_count: int
    relevance: float


# ---------------------------------------------------------------------------
# Search responses
# ---------------------------------------------------------------------------


class WikiSearchResponse(BaseModel):
    """Response for a search scoped to a single wiki."""

    query: str
    results: list[SearchResultItem]
    wiki_summary: list[WikiSummaryItem]


class ProjectSearchResponse(BaseModel):
    """Response for a search scoped to all wikis in a project."""

    query: str
    results: list[SearchResultItem]
    wiki_summary: list[WikiSummaryItem]


# ---------------------------------------------------------------------------
# Page list / page content responses
# ---------------------------------------------------------------------------


class PageListItem(BaseModel):
    """A brief description of a single wiki page, used in list responses."""

    page_title: str
    description: str
    section: str | None = None


class WikiPageListResponse(BaseModel):
    """Response listing all pages in a wiki."""

    wiki_id: str
    pages: list[PageListItem]


class WikiPageResponse(BaseModel):
    """Response containing full content of a single wiki page."""

    wiki_id: str
    page_title: str
    content: str
    sections: list[str]


class PageNeighborsResponse(BaseModel):
    """Response containing the neighbor graph for a single wiki page."""

    wiki_id: str
    page_title: str
    links_to: list[str]
    linked_from: list[str]
