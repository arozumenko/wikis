"""SourceToolkit implementation for Atlassian Jira Cloud.

Uses OAuth 2.0 Bearer auth via AtlassianClient (shared transport layer).
"""

from __future__ import annotations

import fnmatch
import logging
import re
from datetime import datetime, timezone
from typing import AsyncIterator

from markdownify import markdownify as _md  # type: ignore[import-untyped]

from app.core.sources._atlassian import AtlassianClient
from app.core.sources.base import FileContent, FileInfo, OriginPointer, SourceToolkit

logger = logging.getLogger(__name__)

_DEFAULT_JQL = "ORDER BY created DESC"

# Characters not allowed in synthetic path components.
_CONTROL_CHARS_RE = re.compile(r"[\x00-\x1f\x7f]")
# Collapse runs of dashes.
_MULTI_DASH_RE = re.compile(r"-{2,}")


def _sanitize_path_component(s: str) -> str:
    """Sanitize a user-supplied string for use in a synthetic file path.

    - Replaces path separators (``/``, ``\\``), control characters, and the
      literal two-dot sequence ``..`` with ``-``.
    - Strips leading/trailing whitespace and dots.
    - Collapses runs of ``-`` into a single ``-``.
    - Truncates to 80 characters.
    - Returns ``"untitled"`` when the result would otherwise be empty.
    """
    result = s.replace("/", "-").replace("\\", "-")
    # Replace ".." as a whole token wherever it appears
    result = result.replace("..", "-")
    result = _CONTROL_CHARS_RE.sub("-", result)
    result = result.strip(". \t\r\n")
    result = _MULTI_DASH_RE.sub("-", result)
    result = result[:80].strip("-")
    return result or "untitled"


class JiraToolkit(SourceToolkit):
    """Source adapter for Atlassian Jira Cloud (REST API v3).

    The synthetic path for each issue is ``"{key}: {sanitized_summary}"``.
    ``include``/``exclude`` glob patterns in :meth:`list_files` are matched
    against this synthetic path, not against real-world Jira hierarchy.

    Args:
        base_url: Tenant root URL, e.g. ``"https://example.atlassian.net"``.
        access_token: OAuth 2.0 access token (mutually exclusive with email/api_token).
        refresh_token: Optional refresh token for automatic re-auth.
        client_id: OAuth 2.0 client ID (required when *refresh_token* set).
        email: Atlassian account email — pair with *api_token* for HTTP Basic auth.
        api_token: API token from id.atlassian.com (paired with *email*).
        jql: JQL query to filter issues (default: ``"ORDER BY created DESC"``).
    """

    source_type = "jira"

    def __init__(
        self,
        base_url: str,
        access_token: str | None = None,
        refresh_token: str | None = None,
        client_id: str | None = None,
        *,
        email: str | None = None,
        api_token: str | None = None,
        jql: str | None = None,
    ) -> None:
        self._base_url = base_url.rstrip("/")
        self._access_token = access_token
        self._refresh_token = refresh_token
        self._client_id = client_id
        self._email = email
        self._api_token = api_token
        self._jql = jql or _DEFAULT_JQL

    @classmethod
    def from_config(cls, config: dict) -> "JiraToolkit":
        """Instantiate from a plain credentials dict.

        Args:
            config: ``base_url`` (required); plus either ``access_token``
                (OAuth) or ``email`` + ``api_token`` (basic auth);
                ``refresh_token``, ``client_id``, ``jql`` are optional.
        """
        return cls(
            base_url=config["base_url"],
            access_token=config.get("access_token"),
            refresh_token=config.get("refresh_token"),
            client_id=config.get("client_id"),
            email=config.get("email"),
            api_token=config.get("api_token"),
            jql=config.get("jql"),
        )

    def _make_client(self) -> AtlassianClient:
        return AtlassianClient(
            base_url=self._base_url,
            access_token=self._access_token,
            refresh_token=self._refresh_token,
            client_id=self._client_id,
            email=self._email,
            api_token=self._api_token,
        )

    async def test_connection(self) -> str:
        async with self._make_client() as client:
            data = await client.get("/rest/api/3/myself")
        display_name = data.get("displayName", "unknown")
        return f"Connected to Jira at {self._base_url} as {display_name}"

    async def list_files(
        self,
        include: list[str] | None = None,
        exclude: list[str] | None = None,
    ) -> AsyncIterator[FileInfo]:  # type: ignore[override]
        start_at = 0
        max_results = 50

        async with self._make_client() as client:
            while True:
                data = await client.get(
                    "/rest/api/3/search",
                    params={
                        "jql": self._jql,
                        "startAt": start_at,
                        "maxResults": max_results,
                        "fields": "summary,status,issuetype,project",
                    },
                )
                issues = data.get("issues", [])
                for issue in issues:
                    key = issue["key"]
                    raw_summary = issue.get("fields", {}).get("summary", key)
                    sanitized = _sanitize_path_component(raw_summary)
                    path = f"{key}: {sanitized}"

                    # Apply include/exclude glob filters on the synthetic path.
                    if include and not any(fnmatch.fnmatch(path, pat) for pat in include):
                        continue
                    if exclude and any(fnmatch.fnmatch(path, pat) for pat in exclude):
                        continue

                    pointer = self.build_origin_pointer(key)
                    yield FileInfo(
                        path=path,
                        language="markdown",
                        size_bytes=0,
                        origin=pointer,
                    )

                total = data.get("total", 0)
                start_at += len(issues)
                if start_at >= total or len(issues) == 0:
                    break

    async def fetch_content(self, pointer: OriginPointer) -> FileContent:
        issue_key = pointer.ref
        async with self._make_client() as client:
            data = await client.get(
                f"/rest/api/3/issue/{issue_key}",
                params={"expand": "renderedFields"},
            )

        fields = data.get("fields", {})
        rendered = data.get("renderedFields", {})

        summary = fields.get("summary", issue_key)
        issue_type = fields.get("issuetype", {}).get("name", "Issue")
        status = fields.get("status", {}).get("name", "Unknown")

        description_html = rendered.get("description") or ""
        description_md = (
            _md(description_html, heading_style="ATX") if description_html else "_No description_"
        )

        content = (
            f"# {issue_key}: {summary}\n\n"
            f"**Type:** {issue_type}\n"
            f"**Status:** {status}\n\n"
            f"{description_md}"
        )

        return FileContent(
            info=FileInfo(
                path=issue_key,
                language="markdown",
                size_bytes=len(content.encode()),
                origin=pointer,
            ),
            content=content,
        )

    def build_origin_pointer(
        self,
        path: str,
        revision: str | None = None,
        line_start: int | None = None,
        line_end: int | None = None,
    ) -> OriginPointer:
        """Construct an OriginPointer for a Jira issue.

        Args:
            path: Issue key, e.g. ``"ENG-123"``.
            revision: Unused (Jira issues don't have revision identifiers).
            line_start: Unused.
            line_end: Unused.
        """
        return OriginPointer(
            source_type=self.source_type,
            ref=path,
            url=f"{self._base_url}/browse/{path}",
            ingested_at=datetime.now(tz=timezone.utc),
            revision=revision,
            line_start=line_start,
            line_end=line_end,
        )
