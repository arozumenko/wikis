"""Shared HTTP client for Atlassian Cloud APIs (Confluence + Jira).

Handles OAuth 2.0 token refresh and rate-limit retry so the individual
toolkit modules stay focused on API shape, not transport concerns.
"""

from __future__ import annotations

import asyncio
import json
import logging
from typing import Any

import httpx

from app.core.sources.exceptions import (
    SourceAuthError,
    SourceConnectionError,
    SourceNotFoundError,
    SourceUnavailableError,
)

logger = logging.getLogger(__name__)

_REFRESH_URL = "https://auth.atlassian.com/oauth/token"
_MAX_RATE_LIMIT_RETRIES = 3
_DEFAULT_RETRY_AFTER = 5  # seconds


def _parse_retry_after(value: str | None, default: int) -> int:
    """Parse a Retry-After header value into an integer number of seconds.

    Handles both integer seconds and HTTP-date forms.  Clamps to [0, 60] so a
    malicious header cannot force an arbitrarily long sleep.
    """
    if not value:
        return default
    try:
        return max(0, min(60, int(value.strip())))
    except ValueError:
        # HTTP-date form, e.g. "Mon, 18 May 2026 12:34:56 GMT"
        try:
            from datetime import datetime, timezone

            from email.utils import parsedate_to_datetime

            target = parsedate_to_datetime(value)
            if target is None:
                return default
            delta = (target - datetime.now(timezone.utc)).total_seconds()
            return max(0, min(60, int(delta)))
        except (TypeError, ValueError):
            return default


class AtlassianClient:
    """Async HTTP client for Atlassian Cloud REST APIs.

    Wraps httpx.AsyncClient with:

    - **OAuth 2.0 Bearer auth** when constructed with an ``access_token``
      (with optional ``refresh_token`` + ``client_id`` for automatic
      re-auth on HTTP 401).
    - **HTTP Basic auth (email + API token)** when constructed with
      ``email`` + ``api_token`` — the shape used by
      ``atlassian-python-api`` clients.  No refresh logic applies; an
      HTTP 401 maps directly to ``SourceAuthError``.
    - Automatic retry on HTTP 429 (up to 3 attempts, honouring Retry-After)
      for both auth modes.
    - Consistent exception mapping across all status codes.

    Usage::

        # OAuth
        async with AtlassianClient(base_url, access_token=tok) as client:
            data = await client.get("/wiki/rest/api/space", params={"limit": 1})

        # API token
        async with AtlassianClient(base_url, email=e, api_token=t) as client:
            ...

    Args:
        base_url: Tenant URL, e.g. ``"https://example.atlassian.net"`` (no trailing slash).
        access_token: OAuth 2.0 access token (mutually exclusive with email/api_token).
        refresh_token: Refresh token for automatic re-auth (OAuth only).
        client_id: OAuth 2.0 client ID — required when *refresh_token* is set.
        email: Atlassian account email (API-token mode).
        api_token: API token issued at id.atlassian.com/manage-profile/security/api-tokens.
    """

    def __init__(
        self,
        base_url: str,
        access_token: str | None = None,
        refresh_token: str | None = None,
        client_id: str | None = None,
        *,
        email: str | None = None,
        api_token: str | None = None,
    ) -> None:
        # Enforce exactly one auth shape at construction so callers can't
        # accidentally pass both.
        oauth_supplied = access_token is not None
        basic_supplied = email is not None and api_token is not None
        if oauth_supplied and basic_supplied:
            raise ValueError(
                "AtlassianClient: pass either access_token or "
                "(email, api_token), not both"
            )
        if not oauth_supplied and not basic_supplied:
            raise ValueError(
                "AtlassianClient: an access_token or (email, api_token) "
                "pair is required"
            )

        self._base_url = base_url.rstrip("/")
        self._access_token = access_token
        self._refresh_token = refresh_token
        self._client_id = client_id
        self._email = email
        self._api_token = api_token
        self._client: httpx.AsyncClient | None = None

    @property
    def _uses_basic_auth(self) -> bool:
        return self._email is not None and self._api_token is not None

    async def __aenter__(self) -> "AtlassianClient":
        self._client = httpx.AsyncClient(timeout=30.0)
        return self

    async def __aexit__(self, *_: object) -> None:
        if self._client is not None:
            await self._client.aclose()
            self._client = None

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _auth_headers(self) -> dict[str, str]:
        if self._uses_basic_auth:
            import base64  # noqa: PLC0415

            token = base64.b64encode(
                f"{self._email}:{self._api_token}".encode("utf-8")
            ).decode("ascii")
            return {
                "Authorization": f"Basic {token}",
                "Accept": "application/json",
            }
        return {
            "Authorization": f"Bearer {self._access_token}",
            "Accept": "application/json",
        }

    async def _refresh(self) -> None:
        """POST to Atlassian token endpoint and update stored tokens.

        Atlassian rotates the refresh token on every use — store the new one.
        Raises SourceAuthError on any failure.
        """
        if not self._refresh_token or not self._client_id:
            raise SourceAuthError("Token refresh requires refresh_token and client_id")

        try:
            resp = await self._client.post(  # type: ignore[union-attr]
                _REFRESH_URL,
                data={
                    "grant_type": "refresh_token",
                    "client_id": self._client_id,
                    "refresh_token": self._refresh_token,
                },
            )
        except httpx.TimeoutException as exc:
            raise SourceAuthError("Atlassian token refresh failed: request timed out") from exc
        except httpx.RequestError as exc:
            raise SourceAuthError("Atlassian token refresh failed: network error") from exc

        if resp.status_code != 200:
            raise SourceAuthError(
                f"Atlassian token refresh failed: HTTP {resp.status_code}"
            )

        try:
            body = resp.json()
        except json.JSONDecodeError as exc:
            raise SourceAuthError("Atlassian token refresh returned invalid JSON") from exc

        try:
            self._access_token = body["access_token"]
        except KeyError as exc:
            raise SourceAuthError("Atlassian token refresh response missing access_token") from exc

        # Atlassian rotates refresh tokens — keep the new one
        if "refresh_token" in body:
            self._refresh_token = body["refresh_token"]

    async def _raw_get(self, path: str, params: dict[str, Any] | None) -> httpx.Response:
        url = f"{self._base_url}{path}"
        return await self._client.get(  # type: ignore[union-attr]
            url,
            params=params,
            headers=self._auth_headers(),
        )

    @staticmethod
    def _map_error(resp: httpx.Response, path: str) -> None:
        """Raise the appropriate SourceError for any non-2xx status code.

        This method always raises for non-2xx responses so callers never
        see an error response in the success path.
        """
        status = resp.status_code
        if status < 300:
            return  # 2xx — success, nothing to do
        if status >= 500:
            raise SourceUnavailableError(
                f"Atlassian server error {status} for {path}"
            )
        if status == 404:
            raise SourceNotFoundError(path)
        if status in (401, 403):
            raise SourceAuthError(f"Atlassian permission denied for {path}")
        if 400 <= status < 500:
            raise SourceConnectionError(
                f"Atlassian {status} for {path}"
            )

    # ------------------------------------------------------------------
    # Public
    # ------------------------------------------------------------------

    async def get(self, path: str, params: dict[str, Any] | None = None) -> dict[str, Any]:
        """GET ``{base_url}{path}`` and return the decoded JSON body.

        Retry up to 3 times on HTTP 429 (rate limited), honouring Retry-After.
        Refresh tokens automatically on HTTP 401, then retry once.

        Args:
            path: API path, e.g. ``"/wiki/rest/api/space"``.
            params: Optional query parameters.

        Returns:
            Decoded JSON response body.

        Raises:
            SourceAuthError: 401 that cannot be resolved by token refresh.
            SourceNotFoundError: 404.
            SourceUnavailableError: 429 exhausted or 5xx.
            SourceConnectionError: 400 or network error.
        """
        assert self._client is not None, "AtlassianClient must be used as an async context manager"  # noqa: S101

        # --- rate-limit retry loop ---
        for attempt in range(_MAX_RATE_LIMIT_RETRIES):
            try:
                resp = await self._raw_get(path, params)
            except (httpx.ConnectError, httpx.TimeoutException, httpx.RequestError) as exc:
                raise SourceConnectionError(f"Network error reaching Atlassian: {exc}") from exc

            if resp.status_code == 429:
                if attempt == _MAX_RATE_LIMIT_RETRIES - 1:
                    raise SourceUnavailableError(
                        f"Atlassian rate limit exceeded after {_MAX_RATE_LIMIT_RETRIES} attempts for {path}"
                    )
                retry_after = _parse_retry_after(resp.headers.get("Retry-After"), _DEFAULT_RETRY_AFTER)
                logger.warning("Rate limited by Atlassian; waiting %ds (attempt %d)", retry_after, attempt + 1)
                await asyncio.sleep(retry_after)
                continue

            if resp.status_code == 401:
                # Try refresh exactly once.  Basic auth (email + API token)
                # has no refresh path — fail fast with a clearer message
                # pointing at the likely real cause: an invalid or revoked
                # API token.
                can_refresh = bool(self._refresh_token and self._client_id)
                if not can_refresh:
                    if self._uses_basic_auth:
                        raise SourceAuthError(
                            "Atlassian rejected the API token. Check the "
                            "email + api_token pair and that the token "
                            "has not been revoked."
                        )
                    raise SourceAuthError(
                        "Atlassian authentication failed and no refresh token available"
                    )
                try:
                    await self._refresh()
                except SourceAuthError:
                    raise
                # Retry with new token
                try:
                    resp = await self._raw_get(path, params)
                except (httpx.ConnectError, httpx.TimeoutException, httpx.RequestError) as exc:
                    raise SourceConnectionError(f"Network error reaching Atlassian: {exc}") from exc
                if resp.status_code == 401:
                    raise SourceAuthError("Atlassian authentication failed after token refresh")
                # Fall through to error-mapping below

            self._map_error(resp, path)
            # 2xx — success
            return resp.json()

        # Unreachable — loop exits via return or raise above
        raise SourceUnavailableError(f"Atlassian request failed after retries for {path}")
