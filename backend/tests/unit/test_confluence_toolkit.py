"""Unit tests for ConfluenceToolkit (issue #187)."""

from __future__ import annotations

from datetime import datetime, timezone

import httpx
import pytest
import respx

from app.core.sources import registry
from app.core.sources.confluence_toolkit import ConfluenceToolkit, _sanitize_path_component

_BASE = "https://example.atlassian.net"

_MIN_CONFIG = {
    "base_url": _BASE,
    "access_token": "tok-abc",
}

_FULL_CONFIG = {
    "base_url": _BASE,
    "access_token": "tok-abc",
    "refresh_token": "ref-xyz",
    "client_id": "cid-123",
    "space_keys": ["ENG", "DOCS"],
}


# ---------------------------------------------------------------------------
# 1. from_config with all fields
# ---------------------------------------------------------------------------


def test_from_config_full():
    tk = ConfluenceToolkit.from_config(_FULL_CONFIG)
    assert tk._base_url == _BASE
    assert tk._access_token == "tok-abc"
    assert tk._refresh_token == "ref-xyz"
    assert tk._client_id == "cid-123"
    assert tk._space_keys == ["ENG", "DOCS"]


# ---------------------------------------------------------------------------
# 2. from_config with space_keys as comma-separated string
# ---------------------------------------------------------------------------


def test_from_config_space_keys_string():
    cfg = {**_MIN_CONFIG, "space_keys": "ENG, DOCS, OPS"}
    tk = ConfluenceToolkit.from_config(cfg)
    assert tk._space_keys == ["ENG", "DOCS", "OPS"]


# ---------------------------------------------------------------------------
# 3. from_config with no space_keys
# ---------------------------------------------------------------------------


def test_from_config_no_space_keys():
    tk = ConfluenceToolkit.from_config(_MIN_CONFIG)
    assert tk._space_keys == []


# ---------------------------------------------------------------------------
# 4. test_connection happy path
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
@respx.mock
async def test_test_connection_happy():
    respx.get(f"{_BASE}/wiki/rest/api/space").mock(
        return_value=httpx.Response(200, json={"results": []})
    )
    tk = ConfluenceToolkit.from_config(_MIN_CONFIG)
    msg = await tk.test_connection()
    assert "Connected to Confluence" in msg
    assert _BASE in msg


# ---------------------------------------------------------------------------
# 5. list_files paginates correctly via _links.next
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
@respx.mock
async def test_list_files_paginates_across_links_next():
    """Two pages: first has _links.next, second does not."""

    def _make_page(page_id: str, title: str, space_key: str = "ENG", version: int = 1) -> dict:
        return {
            "id": page_id,
            "title": title,
            "space": {"key": space_key},
            "version": {"number": version},
        }

    page1_response = {
        "results": [_make_page("100", "Page A"), _make_page("101", "Page B")],
        "_links": {"next": "/wiki/rest/api/content/search?start=50"},
    }
    page2_response = {
        "results": [_make_page("102", "Page C")],
        "_links": {},  # no 'next' — last page
    }

    route = respx.get(f"{_BASE}/wiki/rest/api/content/search").mock(
        side_effect=[
            httpx.Response(200, json=page1_response),
            httpx.Response(200, json=page2_response),
        ]
    )

    tk = ConfluenceToolkit.from_config(_MIN_CONFIG)
    files = [fi async for fi in tk.list_files()]

    assert len(files) == 3
    assert route.call_count == 2
    paths = {fi.path for fi in files}
    assert "ENG/Page A" in paths
    assert "ENG/Page B" in paths
    assert "ENG/Page C" in paths


# ---------------------------------------------------------------------------
# 6. list_files does NOT terminate on len(results) < limit
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
@respx.mock
async def test_list_files_does_not_terminate_on_partial_page():
    """A page with fewer results than limit but _links.next must keep paginating.

    Permission filtering can return fewer results than limit even when more
    pages exist, so len(results) < limit is an unreliable termination signal.
    """
    partial_page = {
        "results": [{"id": "1", "title": "P", "space": {"key": "ENG"}, "version": {"number": 1}}],
        "_links": {"next": "/wiki/rest/api/content/search?start=50"},
    }
    final_page = {
        "results": [{"id": "2", "title": "Q", "space": {"key": "ENG"}, "version": {"number": 1}}],
        "_links": {},
    }

    route = respx.get(f"{_BASE}/wiki/rest/api/content/search").mock(
        side_effect=[
            httpx.Response(200, json=partial_page),
            httpx.Response(200, json=final_page),
        ]
    )

    tk = ConfluenceToolkit.from_config(_MIN_CONFIG)
    files = [fi async for fi in tk.list_files()]

    # Must have fetched both pages even though first had only 1 result (< 50 limit)
    assert route.call_count == 2
    assert len(files) == 2


# ---------------------------------------------------------------------------
# 7. list_files includes CQL space filter when space_keys present
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
@respx.mock
async def test_list_files_cql_includes_space_filter():
    route = respx.get(f"{_BASE}/wiki/rest/api/content/search").mock(
        return_value=httpx.Response(200, json={"results": [], "_links": {}})
    )
    cfg = {**_MIN_CONFIG, "space_keys": ["ENG", "DOCS"]}
    tk = ConfluenceToolkit.from_config(cfg)
    _ = [fi async for fi in tk.list_files()]

    request = route.calls[0].request
    from urllib.parse import parse_qs, urlparse

    qs = parse_qs(urlparse(str(request.url)).query)
    cql = qs["cql"][0]
    assert "space.key in" in cql
    assert '"ENG"' in cql
    assert '"DOCS"' in cql


# ---------------------------------------------------------------------------
# 8. fetch_content converts HTML to markdown
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
@respx.mock
async def test_fetch_content_converts_html_to_markdown():
    page_id = "999"
    respx.get(f"{_BASE}/wiki/api/v2/pages/{page_id}").mock(
        return_value=httpx.Response(
            200,
            json={"body": {"storage": {"value": "<h1>Hi</h1><p>Body text</p>"}}},
        )
    )
    tk = ConfluenceToolkit.from_config(_MIN_CONFIG)
    from app.core.sources.base import OriginPointer

    ptr = OriginPointer(
        source_type="confluence",
        ref=page_id,
        url=f"{_BASE}/wiki/pages/viewpage.action?pageId={page_id}",
        ingested_at=datetime(2026, 1, 1, tzinfo=timezone.utc),
    )
    fc = await tk.fetch_content(ptr)
    assert "# Hi" in fc.content
    assert "Body text" in fc.content
    assert fc.info.language == "markdown"


# ---------------------------------------------------------------------------
# 9. build_origin_pointer with numeric page_id
# ---------------------------------------------------------------------------


def test_build_origin_pointer_numeric_id():
    tk = ConfluenceToolkit.from_config(_MIN_CONFIG)
    ptr = tk.build_origin_pointer("12345", revision="3")
    assert "pageId=12345" in ptr.url
    assert ptr.source_type == "confluence"
    assert ptr.ref == "12345"
    assert ptr.revision == "3"


def test_build_origin_pointer_path_form():
    tk = ConfluenceToolkit.from_config(_MIN_CONFIG)
    ptr = tk.build_origin_pointer("ENG/My Page", revision=None)
    # Title must be URL-encoded (spaces → %20) in the backlink URL.
    assert "/wiki/spaces/ENG/pages/My%20Page" in ptr.url
    assert ptr.ref == "ENG/My Page"


# ---------------------------------------------------------------------------
# 10. Registry integration
# ---------------------------------------------------------------------------


def test_registry_create_returns_confluence_toolkit():
    tk = registry.create("confluence", _FULL_CONFIG)
    assert isinstance(tk, ConfluenceToolkit)


# ---------------------------------------------------------------------------
# 11. _sanitize_path_component — path traversal and special chars
# ---------------------------------------------------------------------------


def test_sanitize_removes_slashes():
    result = _sanitize_path_component("Page / with .. and \\ chars")
    assert "/" not in result
    assert "\\" not in result
    assert ".." not in result.split("-")


def test_sanitize_dotdot_not_a_segment():
    """'..' as a standalone traversal segment must be eliminated."""
    result = _sanitize_path_component("../../../etc/passwd")
    for segment in result.split("-"):
        assert segment != ".."


def test_sanitize_empty_becomes_untitled():
    assert _sanitize_path_component("...") == "untitled"


# ---------------------------------------------------------------------------
# 12. list_files — path sanitization end-to-end for title and space_key
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
@respx.mock
async def test_list_files_sanitizes_title_and_space_key():
    """Title with path traversal chars must produce a safe synthetic path."""
    respx.get(f"{_BASE}/wiki/rest/api/content/search").mock(
        return_value=httpx.Response(
            200,
            json={
                "results": [
                    {
                        "id": "77",
                        "title": "Page / with .. and \\ chars",
                        "space": {"key": "ENG"},
                        "version": {"number": 1},
                    }
                ],
                "_links": {},
            },
        )
    )
    tk = ConfluenceToolkit.from_config(_MIN_CONFIG)
    files = [fi async for fi in tk.list_files()]
    assert len(files) == 1
    path = files[0].path
    # The path is {space}/{title} — only one literal "/" separating them.
    parts = path.split("/", 1)
    assert len(parts) == 2, f"Expected exactly one '/' separator in path: {path!r}"
    title_part = parts[1]
    assert "/" not in title_part, f"Unexpected '/' in title component: {title_part!r}"
    assert "\\" not in title_part
    for seg in title_part.split("-"):
        assert seg != ".."


# ---------------------------------------------------------------------------
# 13. build_origin_pointer — URL-encoding of path components
# ---------------------------------------------------------------------------


def test_build_origin_pointer_url_encodes_space_in_title():
    """Spaces in the title must be percent-encoded in the backlink URL."""
    tk = ConfluenceToolkit.from_config(_MIN_CONFIG)
    ptr = tk.build_origin_pointer("ENG/My Page Title", revision=None)
    assert "My%20Page%20Title" in ptr.url


def test_build_origin_pointer_url_encodes_special_chars():
    """Special characters in title should be percent-encoded."""
    tk = ConfluenceToolkit.from_config(_MIN_CONFIG)
    ptr = tk.build_origin_pointer("ENG/Hello World & More", revision=None)
    assert " " not in ptr.url
    assert "&" not in ptr.url


# ---------------------------------------------------------------------------
# 14. list_files — include/exclude glob filtering
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
@respx.mock
async def test_list_files_include_filter_by_space():
    """include=['ENG/*'] keeps only ENG pages."""

    def _page(pid: str, title: str, space: str) -> dict:
        return {"id": pid, "title": title, "space": {"key": space}, "version": {"number": 1}}

    respx.get(f"{_BASE}/wiki/rest/api/content/search").mock(
        return_value=httpx.Response(
            200,
            json={
                "results": [_page("1", "Intro", "ENG"), _page("2", "Ops Guide", "OPS")],
                "_links": {},
            },
        )
    )
    tk = ConfluenceToolkit.from_config(_MIN_CONFIG)
    files = [fi async for fi in tk.list_files(include=["ENG/*"])]
    assert len(files) == 1
    assert files[0].path.startswith("ENG/")


@pytest.mark.asyncio
@respx.mock
async def test_list_files_exclude_filter_by_space():
    """exclude=['OPS/*'] drops OPS pages."""

    def _page(pid: str, title: str, space: str) -> dict:
        return {"id": pid, "title": title, "space": {"key": space}, "version": {"number": 1}}

    respx.get(f"{_BASE}/wiki/rest/api/content/search").mock(
        return_value=httpx.Response(
            200,
            json={
                "results": [_page("1", "Intro", "ENG"), _page("2", "Ops Guide", "OPS")],
                "_links": {},
            },
        )
    )
    tk = ConfluenceToolkit.from_config(_MIN_CONFIG)
    files = [fi async for fi in tk.list_files(exclude=["OPS/*"])]
    assert len(files) == 1
    assert files[0].path.startswith("ENG/")
