"""SourceToolkit implementation for Atlassian Confluence Cloud.

Uses OAuth 2.0 Bearer auth via AtlassianClient (shared transport layer).
Authentication is token-based — no email/basic-auth — aligning with the
Atlassian OAuth 2.0 scopes granted together with JiraToolkit.
"""

from __future__ import annotations

import fnmatch
import logging
import re
from datetime import datetime, timezone
from typing import AsyncIterator
from urllib.parse import quote

from markdownify import markdownify as _md  # type: ignore[import-untyped]

from app.core.sources._atlassian import AtlassianClient
from app.core.sources.base import FileContent, FileInfo, OriginPointer, SourceToolkit

logger = logging.getLogger(__name__)

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
    result = result.replace("..", "-")
    result = _CONTROL_CHARS_RE.sub("-", result)
    result = result.strip(". \t\r\n")
    result = _MULTI_DASH_RE.sub("-", result)
    result = result[:80].strip("-")
    return result or "untitled"


class ConfluenceToolkit(SourceToolkit):
    """Source adapter for Atlassian Confluence Cloud (REST API v1 + v2).

    The synthetic path for each page is ``"{space_key}/{title}"`` where both
    components are sanitized.  ``include``/``exclude`` glob patterns in
    :meth:`list_files` are matched against this synthetic path, not against
    real-world Confluence hierarchy.

    Args:
        base_url: Tenant root URL, e.g. ``"https://example.atlassian.net"``.
        access_token: OAuth 2.0 access token.
        refresh_token: Optional refresh token for automatic re-auth.
        client_id: OAuth 2.0 client ID (required when *refresh_token* set).
        space_keys: Optional list of space keys to filter pages.
    """

    source_type = "confluence"

    def __init__(
        self,
        base_url: str,
        access_token: str,
        refresh_token: str | None = None,
        client_id: str | None = None,
        space_keys: list[str] | None = None,
    ) -> None:
        self._base_url = base_url.rstrip("/")
        self._access_token = access_token
        self._refresh_token = refresh_token
        self._client_id = client_id
        self._space_keys = space_keys or []

    @classmethod
    def from_config(cls, config: dict) -> "ConfluenceToolkit":
        """Instantiate from a plain credentials dict.

        Args:
            config: Keys: ``base_url``, ``access_token`` (required);
                ``refresh_token``, ``client_id``, ``space_keys`` (optional).
                ``space_keys`` may be a list or comma-separated string.
        """
        space_keys_raw = config.get("space_keys")
        if isinstance(space_keys_raw, str):
            space_keys = [k.strip() for k in space_keys_raw.split(",") if k.strip()]
        elif space_keys_raw is None:
            space_keys = []
        else:
            space_keys = list(space_keys_raw)

        return cls(
            base_url=config["base_url"],
            access_token=config["access_token"],
            refresh_token=config.get("refresh_token"),
            client_id=config.get("client_id"),
            space_keys=space_keys,
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
            await client.get("/wiki/rest/api/space", params={"limit": 1})
        return f"Connected to Confluence at {self._base_url}"

    async def list_files(
        self,
        include: list[str] | None = None,
        exclude: list[str] | None = None,
    ) -> AsyncIterator[FileInfo]:  # type: ignore[override]
        cql_parts = ["type=page"]
        if self._space_keys:
            keys_str = ",".join(f'"{k}"' for k in self._space_keys)
            cql_parts.append(f"space.key in ({keys_str})")
        cql = " AND ".join(cql_parts)

        start = 0
        limit = 50
        now = datetime.now(tz=timezone.utc)

        async with self._make_client() as client:
            while True:
                data = await client.get(
                    "/wiki/rest/api/content/search",
                    params={
                        "cql": cql,
                        "start": start,
                        "limit": limit,
                        "expand": "space,version",
                    },
                )
                results = data.get("results", [])
                for page in results:
                    page_id = str(page["id"])
                    raw_space_key = page.get("space", {}).get("key", "")
                    raw_title = page.get("title", page_id)
                    path = f"{_sanitize_path_component(raw_space_key)}/{_sanitize_path_component(raw_title)}"

                    # Apply include/exclude glob filters on the synthetic path.
                    if include and not any(fnmatch.fnmatch(path, pat) for pat in include):
                        continue
                    if exclude and any(fnmatch.fnmatch(path, pat) for pat in exclude):
                        continue

                    version_number = str(page.get("version", {}).get("number", ""))
                    pointer = OriginPointer(
                        source_type=self.source_type,
                        ref=page_id,  # numeric ID used by fetch_content
                        url=f"{self._base_url}/wiki/pages/viewpage.action?pageId={page_id}",
                        ingested_at=now,
                        revision=version_number,
                    )
                    yield FileInfo(
                        path=path,
                        language="markdown",
                        size_bytes=0,
                        origin=pointer,
                    )

                # Use _links.next for pagination — not len(results) < limit.
                # Permission filtering can return fewer results than limit even
                # when more pages exist, so the length check is unreliable.
                links = data.get("_links", {})
                if "next" not in links:
                    break
                start += limit

    async def fetch_content(self, pointer: OriginPointer) -> FileContent:
        page_id = pointer.ref
        async with self._make_client() as client:
            data = await client.get(
                f"/wiki/api/v2/pages/{page_id}",
                params={"body-format": "storage"},
            )

        html = data.get("body", {}).get("storage", {}).get("value", "")
        content = _md(html, heading_style="ATX")

        return FileContent(
            info=FileInfo(
                path=pointer.ref,
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
        """Construct an OriginPointer for a Confluence page.

        Args:
            path: Numeric page ID (when called from list_files) or
                ``"{space_key}/{title}"`` from external callers.
            revision: Confluence version number.
            line_start: Unused for Confluence (no line-level anchors).
            line_end: Unused for Confluence.
        """
        if path.isdigit():
            url = f"{self._base_url}/wiki/pages/viewpage.action?pageId={path}"
        else:
            parts = path.split("/", 1)
            space_key = parts[0] if len(parts) > 1 else ""
            title_slug = parts[1] if len(parts) > 1 else path
            url = (
                f"{self._base_url}/wiki/spaces/"
                f"{quote(space_key, safe='')}/pages/{quote(title_slug, safe='')}"
            )

        return OriginPointer(
            source_type=self.source_type,
            ref=path,
            url=url,
            ingested_at=datetime.now(tz=timezone.utc),
            revision=revision,
            line_start=line_start,
            line_end=line_end,
        )
