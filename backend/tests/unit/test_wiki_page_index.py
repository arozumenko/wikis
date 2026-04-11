"""Unit tests for WikiPageIndex — wikilink graph parsing."""

from __future__ import annotations

from unittest.mock import AsyncMock

import pytest
from app.core.wiki_page_index import WikiPageIndex, _extract_wikilink_targets, _parse_frontmatter

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_FRONTMATTER_TEMPLATE = """\
---
title: {title}
description: {description}
---
{body}
"""


def _make_storage(pages: dict[str, str], wiki_id: str = "wiki1") -> AsyncMock:
    """Build a mock ArtifactStorage from a mapping of artifact_key → content."""
    storage = AsyncMock()

    async def _list_artifacts(bucket: str, prefix: str = "") -> list[str]:
        return [k for k in pages if k.startswith(prefix)]

    async def _download(bucket: str, name: str) -> bytes:
        return pages[name].encode("utf-8")

    storage.list_artifacts.side_effect = _list_artifacts
    storage.download.side_effect = _download
    return storage


def _artifact_key(wiki_id: str, page_name: str) -> str:
    return f"{wiki_id}/wiki_pages/{page_name}.md"


# ---------------------------------------------------------------------------
# _parse_frontmatter unit tests
# ---------------------------------------------------------------------------


def test_parse_frontmatter_extracts_title_and_description():
    content = "---\ntitle: My Page\ndescription: A summary\n---\nBody text"
    title, desc = _parse_frontmatter(content)
    assert title == "My Page"
    assert desc == "A summary"


def test_parse_frontmatter_quoted_values():
    content = '---\ntitle: "Quoted Title"\ndescription: "Quoted desc"\n---\n'
    title, desc = _parse_frontmatter(content)
    assert title == "Quoted Title"
    assert desc == "Quoted desc"


def test_parse_frontmatter_no_frontmatter():
    content = "Just plain text, no YAML block"
    title, desc = _parse_frontmatter(content)
    assert title == ""
    assert desc == ""


def test_parse_frontmatter_missing_description():
    content = "---\ntitle: Only Title\n---\nBody"
    title, desc = _parse_frontmatter(content)
    assert title == "Only Title"
    assert desc == ""


# ---------------------------------------------------------------------------
# _extract_wikilink_targets unit tests
# ---------------------------------------------------------------------------


def test_extract_simple_wikilink():
    targets = _extract_wikilink_targets("See [[Architecture Overview]] for details.")
    assert "Architecture Overview" in targets


def test_extract_wikilink_with_display_text():
    targets = _extract_wikilink_targets("See [[Architecture Overview|the overview]] for details.")
    assert "Architecture Overview" in targets


def test_extract_source_link_excluded():
    targets = _extract_wikilink_targets("See [[source/path/to/file.py]] for the code.")
    assert not targets


def test_extract_source_link_with_display_excluded():
    targets = _extract_wikilink_targets("See [[source/path/to/file.py|file.py]] for the code.")
    assert not targets


def test_extract_multiple_links():
    content = "See [[Page A]] and [[Page B|B]] and [[source/x.py]]."
    targets = _extract_wikilink_targets(content)
    assert targets == {"Page A", "Page B"}


# ---------------------------------------------------------------------------
# WikiPageIndex integration tests (with mocked storage)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_empty_wiki_does_not_crash():
    storage = _make_storage({})
    index = await WikiPageIndex.build("empty_wiki", storage)
    assert index.pages == {}
    assert index.neighbors("Anything") == []
    assert index.backlinks("Anything") == []
    assert index.get_description("Anything") == ""


@pytest.mark.asyncio
async def test_pages_loaded_correctly():
    wiki_id = "wiki1"
    pages = {
        _artifact_key(wiki_id, "overview"): _FRONTMATTER_TEMPLATE.format(
            title="Overview", description="High-level summary", body="No links here."
        ),
    }
    index = await WikiPageIndex.build(wiki_id, _make_storage(pages, wiki_id))
    assert "Overview" in index.pages
    meta = index.pages["Overview"]
    assert meta.title == "Overview"
    assert meta.description == "High-level summary"
    assert "wiki_pages/overview.md" in meta.file_path


@pytest.mark.asyncio
async def test_standard_wikilink_parsed_as_edge():
    wiki_id = "wiki1"
    pages = {
        _artifact_key(wiki_id, "overview"): _FRONTMATTER_TEMPLATE.format(
            title="Overview", description="", body="See [[Architecture]] for details."
        ),
        _artifact_key(wiki_id, "architecture"): _FRONTMATTER_TEMPLATE.format(
            title="Architecture", description="", body="No outgoing links."
        ),
    }
    index = await WikiPageIndex.build(wiki_id, _make_storage(pages, wiki_id))
    neighbors = index.neighbors("Overview")
    assert any(m.title == "Architecture" for m in neighbors)


@pytest.mark.asyncio
async def test_wikilink_with_display_text_parsed_as_edge():
    wiki_id = "wiki1"
    pages = {
        _artifact_key(wiki_id, "overview"): _FRONTMATTER_TEMPLATE.format(
            title="Overview", description="", body="See [[Architecture|the arch]] for details."
        ),
        _artifact_key(wiki_id, "architecture"): _FRONTMATTER_TEMPLATE.format(
            title="Architecture", description="", body="No outgoing links."
        ),
    }
    index = await WikiPageIndex.build(wiki_id, _make_storage(pages, wiki_id))
    neighbors = index.neighbors("Overview")
    assert any(m.title == "Architecture" for m in neighbors)


@pytest.mark.asyncio
async def test_source_wikilink_excluded_from_edges():
    wiki_id = "wiki1"
    pages = {
        _artifact_key(wiki_id, "overview"): _FRONTMATTER_TEMPLATE.format(
            title="Overview", description="", body="See [[source/app/main.py]] for code."
        ),
        _artifact_key(wiki_id, "architecture"): _FRONTMATTER_TEMPLATE.format(
            title="Architecture", description="", body="No links."
        ),
    }
    index = await WikiPageIndex.build(wiki_id, _make_storage(pages, wiki_id))
    neighbors = index.neighbors("Overview")
    assert neighbors == []


@pytest.mark.asyncio
async def test_source_wikilink_with_display_excluded():
    wiki_id = "wiki1"
    pages = {
        _artifact_key(wiki_id, "overview"): _FRONTMATTER_TEMPLATE.format(
            title="Overview", description="", body="See [[source/app/main.py|main.py]]."
        ),
    }
    index = await WikiPageIndex.build(wiki_id, _make_storage(pages, wiki_id))
    assert index.neighbors("Overview") == []


@pytest.mark.asyncio
async def test_backlinks_reverse_map():
    wiki_id = "wiki1"
    pages = {
        _artifact_key(wiki_id, "overview"): _FRONTMATTER_TEMPLATE.format(
            title="Overview", description="", body="Links to [[Architecture]]."
        ),
        _artifact_key(wiki_id, "architecture"): _FRONTMATTER_TEMPLATE.format(
            title="Architecture", description="", body="No outgoing links."
        ),
    }
    index = await WikiPageIndex.build(wiki_id, _make_storage(pages, wiki_id))
    bl = index.backlinks("Architecture")
    assert "Overview" in bl


@pytest.mark.asyncio
async def test_backlinks_empty_when_no_links_to_page():
    wiki_id = "wiki1"
    pages = {
        _artifact_key(wiki_id, "overview"): _FRONTMATTER_TEMPLATE.format(
            title="Overview", description="", body="No links."
        ),
        _artifact_key(wiki_id, "architecture"): _FRONTMATTER_TEMPLATE.format(
            title="Architecture", description="", body="No links."
        ),
    }
    index = await WikiPageIndex.build(wiki_id, _make_storage(pages, wiki_id))
    assert index.backlinks("Overview") == []


@pytest.mark.asyncio
async def test_neighbors_bfs_stops_at_hop_depth():
    """A → B → C; neighbors(A, hop_depth=1) should return [B] only, not C."""
    wiki_id = "wiki1"
    pages = {
        _artifact_key(wiki_id, "a"): _FRONTMATTER_TEMPLATE.format(title="A", description="", body="Links to [[B]]."),
        _artifact_key(wiki_id, "b"): _FRONTMATTER_TEMPLATE.format(title="B", description="", body="Links to [[C]]."),
        _artifact_key(wiki_id, "c"): _FRONTMATTER_TEMPLATE.format(title="C", description="", body="No links."),
    }
    index = await WikiPageIndex.build(wiki_id, _make_storage(pages, wiki_id))
    neighbors = index.neighbors("A", hop_depth=1)
    titles = {m.title for m in neighbors}
    assert titles == {"B"}


@pytest.mark.asyncio
async def test_neighbors_bfs_two_hops():
    """A → B → C; neighbors(A, hop_depth=2) should return both B and C."""
    wiki_id = "wiki1"
    pages = {
        _artifact_key(wiki_id, "a"): _FRONTMATTER_TEMPLATE.format(title="A", description="", body="Links to [[B]]."),
        _artifact_key(wiki_id, "b"): _FRONTMATTER_TEMPLATE.format(title="B", description="", body="Links to [[C]]."),
        _artifact_key(wiki_id, "c"): _FRONTMATTER_TEMPLATE.format(title="C", description="", body="No links."),
    }
    index = await WikiPageIndex.build(wiki_id, _make_storage(pages, wiki_id))
    neighbors = index.neighbors("A", hop_depth=2)
    titles = {m.title for m in neighbors}
    assert titles == {"B", "C"}


@pytest.mark.asyncio
async def test_neighbors_does_not_include_seed():
    wiki_id = "wiki1"
    pages = {
        _artifact_key(wiki_id, "a"): _FRONTMATTER_TEMPLATE.format(
            title="A", description="", body="Links to [[B]] and back to [[A]]."
        ),
        _artifact_key(wiki_id, "b"): _FRONTMATTER_TEMPLATE.format(title="B", description="", body="No links."),
    }
    index = await WikiPageIndex.build(wiki_id, _make_storage(pages, wiki_id))
    neighbors = index.neighbors("A")
    titles = [m.title for m in neighbors]
    assert "A" not in titles


@pytest.mark.asyncio
async def test_neighbors_caps_at_50():
    """Build a star graph with 60 leaf nodes; neighbors() must return at most 50."""
    wiki_id = "wiki1"
    leaf_titles = [f"Leaf{i}" for i in range(60)]
    body = " ".join(f"[[{t}]]" for t in leaf_titles)
    pages: dict[str, str] = {
        _artifact_key(wiki_id, "hub"): _FRONTMATTER_TEMPLATE.format(title="Hub", description="", body=body),
    }
    for t in leaf_titles:
        pages[_artifact_key(wiki_id, t.lower())] = _FRONTMATTER_TEMPLATE.format(
            title=t, description="", body="No links."
        )
    index = await WikiPageIndex.build(wiki_id, _make_storage(pages, wiki_id))
    neighbors = index.neighbors("Hub")
    assert len(neighbors) <= 50


@pytest.mark.asyncio
async def test_neighbors_returns_empty_for_unknown_title():
    wiki_id = "wiki1"
    pages = {
        _artifact_key(wiki_id, "overview"): _FRONTMATTER_TEMPLATE.format(
            title="Overview", description="", body="No links."
        ),
    }
    index = await WikiPageIndex.build(wiki_id, _make_storage(pages, wiki_id))
    assert index.neighbors("Does Not Exist") == []


@pytest.mark.asyncio
async def test_get_description_returns_description():
    wiki_id = "wiki1"
    pages = {
        _artifact_key(wiki_id, "overview"): _FRONTMATTER_TEMPLATE.format(
            title="Overview", description="My description", body=""
        ),
    }
    index = await WikiPageIndex.build(wiki_id, _make_storage(pages, wiki_id))
    assert index.get_description("Overview") == "My description"


@pytest.mark.asyncio
async def test_get_description_returns_empty_for_unknown():
    wiki_id = "wiki1"
    pages = {
        _artifact_key(wiki_id, "overview"): _FRONTMATTER_TEMPLATE.format(title="Overview", description="", body=""),
    }
    index = await WikiPageIndex.build(wiki_id, _make_storage(pages, wiki_id))
    assert index.get_description("Non Existent Page") == ""
