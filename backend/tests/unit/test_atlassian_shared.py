"""Unit tests for backend/app/core/sources/_atlassian.py (issue #187)."""

from __future__ import annotations

import logging
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest
import respx

from app.core.sources._atlassian import AtlassianClient, _REFRESH_URL
from app.core.sources.exceptions import (
    SourceAuthError,
    SourceConnectionError,
    SourceNotFoundError,
    SourceUnavailableError,
)
from app.core.sources._logging import TokenRedactionFilter, install as install_redaction_filter


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_BASE = "https://example.atlassian.net"


def _client(
    access_token: str = "tok-abc",
    refresh_token: str | None = None,
    client_id: str | None = None,
) -> AtlassianClient:
    return AtlassianClient(
        base_url=_BASE,
        access_token=access_token,
        refresh_token=refresh_token,
        client_id=client_id,
    )


# ---------------------------------------------------------------------------
# 1. Happy path — 200 returns decoded JSON
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
@respx.mock
async def test_get_happy_path():
    respx.get(f"{_BASE}/wiki/rest/api/space").mock(
        return_value=httpx.Response(200, json={"results": [{"key": "ENG"}]})
    )
    async with _client() as c:
        data = await c.get("/wiki/rest/api/space")
    assert data["results"][0]["key"] == "ENG"


# ---------------------------------------------------------------------------
# 2. 429 + Retry-After: retries and succeeds
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
@respx.mock
async def test_get_429_retries_and_succeeds():
    route = respx.get(f"{_BASE}/test-path").mock(
        side_effect=[
            httpx.Response(429, headers={"Retry-After": "1"}, json={}),
            httpx.Response(200, json={"ok": True}),
        ]
    )
    with patch("app.core.sources._atlassian.asyncio.sleep", new_callable=AsyncMock) as mock_sleep:
        async with _client() as c:
            data = await c.get("/test-path")
    assert data["ok"] is True
    mock_sleep.assert_awaited_once_with(1)
    assert route.call_count == 2


# ---------------------------------------------------------------------------
# 3. 429 exhausted — raises SourceUnavailableError after 3 attempts
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
@respx.mock
async def test_get_429_exhausted_raises():
    respx.get(f"{_BASE}/test-path").mock(
        return_value=httpx.Response(429, headers={"Retry-After": "1"}, json={})
    )
    with patch("app.core.sources._atlassian.asyncio.sleep", new_callable=AsyncMock):
        with pytest.raises(SourceUnavailableError):
            async with _client() as c:
                await c.get("/test-path")


# ---------------------------------------------------------------------------
# 4. 401 with refresh — retries with new token
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
@respx.mock
async def test_get_401_with_refresh_retries():
    """First 401 triggers refresh; retry uses new access token."""
    # The main endpoint: first call 401, second call 200
    main_route = respx.get(f"{_BASE}/test-path").mock(
        side_effect=[
            httpx.Response(401, json={}),
            httpx.Response(200, json={"data": "ok"}),
        ]
    )
    # Token refresh endpoint
    respx.post(_REFRESH_URL).mock(
        return_value=httpx.Response(
            200,
            json={"access_token": "new-tok", "refresh_token": "new-ref"},
        )
    )

    async with _client(
        access_token="old-tok",
        refresh_token="ref-tok",
        client_id="client-123",
    ) as c:
        data = await c.get("/test-path")

    assert data["data"] == "ok"
    assert c._access_token == "new-tok"
    assert c._refresh_token == "new-ref"
    # Second call must carry the new token
    second_request = main_route.calls[1].request
    assert "new-tok" in second_request.headers["authorization"]


# ---------------------------------------------------------------------------
# 5. 401 without refresh_token — raises SourceAuthError immediately
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
@respx.mock
async def test_get_401_without_refresh_raises():
    respx.get(f"{_BASE}/test-path").mock(return_value=httpx.Response(401, json={}))
    with pytest.raises(SourceAuthError):
        async with _client(access_token="tok", refresh_token=None) as c:
            await c.get("/test-path")


# ---------------------------------------------------------------------------
# 6. 401 after refresh-then-401 — still raises SourceAuthError
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
@respx.mock
async def test_get_401_after_refresh_still_raises():
    respx.get(f"{_BASE}/test-path").mock(return_value=httpx.Response(401, json={}))
    respx.post(_REFRESH_URL).mock(
        return_value=httpx.Response(
            200,
            json={"access_token": "new-tok", "refresh_token": "new-ref"},
        )
    )
    with pytest.raises(SourceAuthError):
        async with _client(
            access_token="old-tok",
            refresh_token="ref-tok",
            client_id="client-123",
        ) as c:
            await c.get("/test-path")


# ---------------------------------------------------------------------------
# 7a. 404 → SourceNotFoundError
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
@respx.mock
async def test_get_404_raises_not_found():
    respx.get(f"{_BASE}/missing").mock(return_value=httpx.Response(404, json={}))
    with pytest.raises(SourceNotFoundError):
        async with _client() as c:
            await c.get("/missing")


# ---------------------------------------------------------------------------
# 7b. 5xx → SourceUnavailableError
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
@respx.mock
async def test_get_5xx_raises_unavailable():
    respx.get(f"{_BASE}/broken").mock(return_value=httpx.Response(500, json={}))
    with pytest.raises(SourceUnavailableError):
        async with _client() as c:
            await c.get("/broken")


# ---------------------------------------------------------------------------
# 7c. 400 → SourceConnectionError
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
@respx.mock
async def test_get_400_raises_connection_error():
    respx.get(f"{_BASE}/bad").mock(return_value=httpx.Response(400, json={}))
    with pytest.raises(SourceConnectionError):
        async with _client() as c:
            await c.get("/bad")


# ---------------------------------------------------------------------------
# 7d. Network error → SourceConnectionError
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
@respx.mock
async def test_get_network_error_raises_connection_error():
    respx.get(f"{_BASE}/unreachable").mock(side_effect=httpx.ConnectError("refused"))
    with pytest.raises(SourceConnectionError):
        async with _client() as c:
            await c.get("/unreachable")


# ---------------------------------------------------------------------------
# 8. Token redaction via TokenRedactionFilter
# ---------------------------------------------------------------------------


def test_token_redaction_filter_masks_access_token(caplog):
    """access_token values must appear as *** in log output."""
    install_redaction_filter()
    log = logging.getLogger("test_atlassian_redaction")
    log.setLevel(logging.DEBUG)
    fltr = TokenRedactionFilter()
    caplog.handler.addFilter(fltr)
    log.addFilter(fltr)

    with caplog.at_level(logging.DEBUG, logger="test_atlassian_redaction"):
        log.info("request body: %s", {"access_token": "supersecret"})

    combined = "\n".join(r.getMessage() for r in caplog.records)
    assert "supersecret" not in combined
    assert "***" in combined
