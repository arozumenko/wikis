"""Unit tests for JiraToolkit (issue #187)."""

from __future__ import annotations

from datetime import datetime, timezone

import httpx
import pytest
import respx

from app.core.sources import registry
from app.core.sources.jira_toolkit import JiraToolkit, _DEFAULT_JQL, _sanitize_path_component

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
    "jql": "project = ENG ORDER BY created DESC",
}


# ---------------------------------------------------------------------------
# 1. from_config with all fields
# ---------------------------------------------------------------------------


def test_from_config_full():
    tk = JiraToolkit.from_config(_FULL_CONFIG)
    assert tk._base_url == _BASE
    assert tk._access_token == "tok-abc"
    assert tk._refresh_token == "ref-xyz"
    assert tk._client_id == "cid-123"
    assert tk._jql == "project = ENG ORDER BY created DESC"


# ---------------------------------------------------------------------------
# 2. from_config defaults jql when missing
# ---------------------------------------------------------------------------


def test_from_config_default_jql():
    tk = JiraToolkit.from_config(_MIN_CONFIG)
    assert tk._jql == _DEFAULT_JQL


# ---------------------------------------------------------------------------
# 3. test_connection happy path — message includes displayName
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
@respx.mock
async def test_test_connection_happy():
    respx.get(f"{_BASE}/rest/api/3/myself").mock(
        return_value=httpx.Response(200, json={"displayName": "Alice Eng"})
    )
    tk = JiraToolkit.from_config(_MIN_CONFIG)
    msg = await tk.test_connection()
    assert "Connected to Jira" in msg
    assert "Alice Eng" in msg
    assert _BASE in msg


# ---------------------------------------------------------------------------
# 4. list_files paginates correctly using startAt + total
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
@respx.mock
async def test_list_files_paginates_with_total():
    def _issue(key: str, summary: str) -> dict:
        return {
            "key": key,
            "fields": {"summary": summary, "status": {"name": "Open"}, "issuetype": {"name": "Bug"}},
        }

    page1 = {
        "issues": [_issue("ENG-1", "First bug"), _issue("ENG-2", "Second bug")],
        "total": 3,
    }
    page2 = {
        "issues": [_issue("ENG-3", "Third bug")],
        "total": 3,
    }

    route = respx.get(f"{_BASE}/rest/api/3/search").mock(
        side_effect=[
            httpx.Response(200, json=page1),
            httpx.Response(200, json=page2),
        ]
    )

    tk = JiraToolkit.from_config(_MIN_CONFIG)
    files = [fi async for fi in tk.list_files()]

    assert len(files) == 3
    assert route.call_count == 2
    paths = {fi.path for fi in files}
    assert "ENG-1: First bug" in paths
    assert "ENG-3: Third bug" in paths


# ---------------------------------------------------------------------------
# 5. list_files terminates when issues is empty
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
@respx.mock
async def test_list_files_terminates_on_empty_issues():
    route = respx.get(f"{_BASE}/rest/api/3/search").mock(
        return_value=httpx.Response(200, json={"issues": [], "total": 100})
    )
    tk = JiraToolkit.from_config(_MIN_CONFIG)
    files = [fi async for fi in tk.list_files()]
    assert files == []
    assert route.call_count == 1


# ---------------------------------------------------------------------------
# 6. fetch_content produces correct markdown
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
@respx.mock
async def test_fetch_content_produces_markdown():
    respx.get(f"{_BASE}/rest/api/3/issue/ENG-42").mock(
        return_value=httpx.Response(
            200,
            json={
                "key": "ENG-42",
                "fields": {
                    "summary": "Widget is broken",
                    "status": {"name": "In Progress"},
                    "issuetype": {"name": "Bug"},
                },
                "renderedFields": {
                    "description": "<p>Detailed description here</p>",
                },
            },
        )
    )
    tk = JiraToolkit.from_config(_MIN_CONFIG)
    from app.core.sources.base import OriginPointer

    ptr = OriginPointer(
        source_type="jira",
        ref="ENG-42",
        url=f"{_BASE}/browse/ENG-42",
        ingested_at=datetime(2026, 1, 1, tzinfo=timezone.utc),
    )
    fc = await tk.fetch_content(ptr)
    assert "# ENG-42: Widget is broken" in fc.content
    assert "**Type:** Bug" in fc.content
    assert "**Status:** In Progress" in fc.content
    assert "Detailed description here" in fc.content
    assert fc.info.language == "markdown"


# ---------------------------------------------------------------------------
# 7. fetch_content falls back to _No description_ when missing
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
@respx.mock
async def test_fetch_content_no_description():
    respx.get(f"{_BASE}/rest/api/3/issue/ENG-1").mock(
        return_value=httpx.Response(
            200,
            json={
                "key": "ENG-1",
                "fields": {
                    "summary": "Empty issue",
                    "status": {"name": "Open"},
                    "issuetype": {"name": "Task"},
                },
                "renderedFields": {},
            },
        )
    )
    tk = JiraToolkit.from_config(_MIN_CONFIG)
    from app.core.sources.base import OriginPointer

    ptr = OriginPointer(
        source_type="jira",
        ref="ENG-1",
        url=f"{_BASE}/browse/ENG-1",
        ingested_at=datetime(2026, 1, 1, tzinfo=timezone.utc),
    )
    fc = await tk.fetch_content(ptr)
    assert "_No description_" in fc.content


# ---------------------------------------------------------------------------
# 8. build_origin_pointer produces /browse/{key} URL
# ---------------------------------------------------------------------------


def test_build_origin_pointer():
    tk = JiraToolkit.from_config(_MIN_CONFIG)
    ptr = tk.build_origin_pointer("PROJ-123")
    assert ptr.url == f"{_BASE}/browse/PROJ-123"
    assert ptr.ref == "PROJ-123"
    assert ptr.source_type == "jira"


def test_build_origin_pointer_with_revision():
    tk = JiraToolkit.from_config(_MIN_CONFIG)
    ptr = tk.build_origin_pointer("PROJ-1", revision="v2")
    assert ptr.revision == "v2"


# ---------------------------------------------------------------------------
# 9. Registry integration
# ---------------------------------------------------------------------------


def test_registry_create_returns_jira_toolkit():
    tk = registry.create("jira", _FULL_CONFIG)
    assert isinstance(tk, JiraToolkit)


# ---------------------------------------------------------------------------
# 10. _sanitize_path_component — path traversal sanitization
# ---------------------------------------------------------------------------


def test_sanitize_path_component_traversal_chars():
    """Summaries with ../..  path traversal chars produce a safe component."""
    result = _sanitize_path_component("../../../etc/passwd")
    assert "/" not in result
    assert ".." not in result.split("-")  # no ".." segment remains


def test_sanitize_path_component_backslash():
    result = _sanitize_path_component("C:\\Windows\\System32")
    assert "\\" not in result


def test_sanitize_path_component_control_chars():
    result = _sanitize_path_component("bad\x00name\x1f")
    assert "\x00" not in result
    assert "\x1f" not in result


def test_sanitize_path_component_empty_result_becomes_untitled():
    result = _sanitize_path_component("...")
    assert result == "untitled"


def test_sanitize_path_component_truncates_to_80():
    long_s = "a" * 200
    assert len(_sanitize_path_component(long_s)) <= 80


# ---------------------------------------------------------------------------
# 11. list_files — path sanitization end-to-end
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
@respx.mock
async def test_list_files_sanitizes_traversal_summary():
    """Issue with summary containing path traversal chars gets a safe path."""
    respx.get(f"{_BASE}/rest/api/3/search").mock(
        return_value=httpx.Response(
            200,
            json={
                "issues": [
                    {
                        "key": "ENG-99",
                        "fields": {
                            "summary": "../../../etc/passwd",
                            "status": {"name": "Open"},
                            "issuetype": {"name": "Bug"},
                        },
                    }
                ],
                "total": 1,
            },
        )
    )
    tk = JiraToolkit.from_config(_MIN_CONFIG)
    files = [fi async for fi in tk.list_files()]
    assert len(files) == 1
    path = files[0].path
    assert "/" not in path.split(": ", 1)[1], f"slash found in path component: {path!r}"
    # No '..' segment should survive sanitization
    for segment in path.split("-"):
        assert segment != ".."


# ---------------------------------------------------------------------------
# 12. list_files — include/exclude glob filtering
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
@respx.mock
async def test_list_files_include_filter_by_project():
    """include=['ENG-*'] keeps only ENG issues."""

    def _issue(key: str, summary: str) -> dict:
        return {
            "key": key,
            "fields": {"summary": summary, "status": {"name": "Open"}, "issuetype": {"name": "Bug"}},
        }

    respx.get(f"{_BASE}/rest/api/3/search").mock(
        return_value=httpx.Response(
            200,
            json={
                "issues": [_issue("ENG-1", "Bug one"), _issue("OPS-2", "Ops issue")],
                "total": 2,
            },
        )
    )
    tk = JiraToolkit.from_config(_MIN_CONFIG)
    files = [fi async for fi in tk.list_files(include=["ENG-*"])]
    assert len(files) == 1
    assert files[0].path.startswith("ENG-1")


@pytest.mark.asyncio
@respx.mock
async def test_list_files_exclude_filter():
    """exclude=['OPS-*'] drops OPS issues."""

    def _issue(key: str, summary: str) -> dict:
        return {
            "key": key,
            "fields": {"summary": summary, "status": {"name": "Open"}, "issuetype": {"name": "Bug"}},
        }

    respx.get(f"{_BASE}/rest/api/3/search").mock(
        return_value=httpx.Response(
            200,
            json={
                "issues": [_issue("ENG-1", "Bug one"), _issue("OPS-2", "Ops issue")],
                "total": 2,
            },
        )
    )
    tk = JiraToolkit.from_config(_MIN_CONFIG)
    files = [fi async for fi in tk.list_files(exclude=["OPS-*"])]
    assert len(files) == 1
    assert files[0].path.startswith("ENG-1")
