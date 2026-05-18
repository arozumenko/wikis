"""SourceToolkit implementation for Atlassian Jira Cloud.

Uses OAuth 2.0 Bearer auth via AtlassianClient (shared transport layer).
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any, AsyncIterator

from app.core.sources._atlassian import AtlassianClient
from app.core.sources.base import FileContent, FileInfo, OriginPointer, SourceToolkit

logger = logging.getLogger(__name__)

try:
    from markdownify import markdownify as _md  # type: ignore[import-untyped]
except ImportError:  # pragma: no cover — markdownify is a core dep; this is a safety net

    def _md(html: str, **kwargs: Any) -> str:  # type: ignore[misc]
        return html


_DEFAULT_JQL = "ORDER BY created DESC"


class JiraToolkit(SourceToolkit):
    """Source adapter for Atlassian Jira Cloud (REST API v3).

    Args:
        base_url: Tenant root URL, e.g. ``"https://example.atlassian.net"``.
        access_token: OAuth 2.0 access token.
        refresh_token: Optional refresh token for automatic re-auth.
        client_id: OAuth 2.0 client ID (required when *refresh_token* set).
        jql: JQL query to filter issues (default: ``"ORDER BY created DESC"``).
    """

    source_type = "jira"

    def __init__(
        self,
        base_url: str,
        access_token: str,
        refresh_token: str | None = None,
        client_id: str | None = None,
        jql: str | None = None,
    ) -> None:
        self._base_url = base_url.rstrip("/")
        self._access_token = access_token
        self._refresh_token = refresh_token
        self._client_id = client_id
        self._jql = jql or _DEFAULT_JQL

    @classmethod
    def from_config(cls, config: dict) -> "JiraToolkit":
        """Instantiate from a plain credentials dict.

        Args:
            config: Keys: ``base_url``, ``access_token`` (required);
                ``refresh_token``, ``client_id``, ``jql`` (optional).
        """
        return cls(
            base_url=config["base_url"],
            access_token=config["access_token"],
            refresh_token=config.get("refresh_token"),
            client_id=config.get("client_id"),
            jql=config.get("jql"),
        )

    def _make_client(self) -> AtlassianClient:
        return AtlassianClient(
            base_url=self._base_url,
            access_token=self._access_token,
            refresh_token=self._refresh_token,
            client_id=self._client_id,
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
                    summary = issue.get("fields", {}).get("summary", key)
                    # Truncate long summaries so path stays readable
                    if len(summary) > 80:
                        summary = summary[:80]
                    path = f"{key}: {summary}"
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
