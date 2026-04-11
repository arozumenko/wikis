"""Unit tests for app/auth.py — JWT validation, JWKS caching, PAT validation."""

from __future__ import annotations

import time
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi import HTTPException
from jose import jwt
from pydantic import SecretStr

import app.auth as auth_module
from app.auth import CurrentUser, JWKSClient, _validate_pat, get_current_user
from app.config import Settings


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _settings(auth_enabled: bool = True) -> Settings:
    return Settings(
        llm_api_key=SecretStr("test-key"),
        auth_enabled=auth_enabled,
        auth_jwks_url="http://auth.test/.well-known/jwks.json",
    )


def _make_jwks():
    """Return a minimal JWKS dict with one RSA-like key stub."""
    return {
        "keys": [
            {
                "kid": "test-kid-1",
                "kty": "RSA",
                "alg": "RS256",
                "use": "sig",
                "n": "abc",
                "e": "AQAB",
            }
        ]
    }


# ---------------------------------------------------------------------------
# CurrentUser model
# ---------------------------------------------------------------------------


class TestCurrentUser:
    def test_defaults(self):
        u = CurrentUser(id="u1", email="a@b.com")
        assert u.name == ""
        assert u.provider == ""

    def test_full_fields(self):
        u = CurrentUser(id="u2", email="x@y.com", name="Alice", provider="github")
        assert u.name == "Alice"
        assert u.provider == "github"


# ---------------------------------------------------------------------------
# JWKSClient
# ---------------------------------------------------------------------------


class TestJWKSClient:
    @pytest.mark.asyncio
    async def test_fetch_on_first_call(self):
        client = JWKSClient("http://example.com/jwks")
        mock_resp = MagicMock()
        mock_resp.json.return_value = _make_jwks()
        mock_resp.raise_for_status = MagicMock()

        with patch("httpx.AsyncClient") as mock_cls:
            mock_http = AsyncMock()
            mock_http.__aenter__ = AsyncMock(return_value=mock_http)
            mock_http.__aexit__ = AsyncMock(return_value=False)
            mock_http.get = AsyncMock(return_value=mock_resp)
            mock_cls.return_value = mock_http

            result = await client.get_signing_keys()

        assert "keys" in result
        assert result["keys"][0]["kid"] == "test-kid-1"

    @pytest.mark.asyncio
    async def test_cache_used_within_ttl(self):
        client = JWKSClient("http://example.com/jwks", cache_ttl=3600)
        client._cache = _make_jwks()
        client._cached_at = time.time()  # just now

        # Should not call _fetch_jwks at all
        with patch.object(client, "_fetch_jwks", new_callable=AsyncMock) as mock_fetch:
            result = await client.get_signing_keys()
            mock_fetch.assert_not_called()

        assert "keys" in result

    @pytest.mark.asyncio
    async def test_cache_refreshed_after_ttl(self):
        client = JWKSClient("http://example.com/jwks", cache_ttl=1)
        client._cache = {"keys": []}
        client._cached_at = time.time() - 10  # expired

        fresh = _make_jwks()
        with patch.object(client, "_fetch_jwks", new_callable=AsyncMock, return_value=fresh):
            result = await client.get_signing_keys()

        assert result["keys"][0]["kid"] == "test-kid-1"

    @pytest.mark.asyncio
    async def test_get_signing_key_found(self):
        client = JWKSClient("http://example.com/jwks")
        client._cache = _make_jwks()
        client._cached_at = time.time()

        key = await client.get_signing_key("test-kid-1")
        assert key is not None
        assert key["kid"] == "test-kid-1"

    @pytest.mark.asyncio
    async def test_get_signing_key_not_found(self):
        client = JWKSClient("http://example.com/jwks")
        client._cache = _make_jwks()
        client._cached_at = time.time()

        key = await client.get_signing_key("nonexistent-kid")
        assert key is None

    @pytest.mark.asyncio
    async def test_get_signing_key_empty_jwks(self):
        client = JWKSClient("http://example.com/jwks")
        client._cache = {"keys": []}
        client._cached_at = time.time()

        key = await client.get_signing_key("any-kid")
        assert key is None


# ---------------------------------------------------------------------------
# get_current_user — AUTH_ENABLED=false path
# ---------------------------------------------------------------------------


class TestGetCurrentUserAuthDisabled:
    @pytest.mark.asyncio
    async def test_returns_dev_user_when_auth_disabled(self):
        settings = _settings(auth_enabled=False)
        user = await get_current_user(authorization=None, settings=settings)
        assert user.id == "dev-user"
        assert user.email == "dev@localhost"

    @pytest.mark.asyncio
    async def test_ignores_token_when_auth_disabled(self):
        settings = _settings(auth_enabled=False)
        user = await get_current_user(authorization="Bearer some-token", settings=settings)
        assert user.id == "dev-user"


# ---------------------------------------------------------------------------
# get_current_user — missing / malformed Authorization header
# ---------------------------------------------------------------------------


class TestGetCurrentUserMissingHeader:
    @pytest.mark.asyncio
    async def test_missing_header_raises_401(self):
        settings = _settings(auth_enabled=True)
        with pytest.raises(HTTPException) as exc:
            await get_current_user(authorization=None, settings=settings)
        assert exc.value.status_code == 401

    @pytest.mark.asyncio
    async def test_non_bearer_header_raises_401(self):
        settings = _settings(auth_enabled=True)
        with pytest.raises(HTTPException) as exc:
            await get_current_user(authorization="Basic dXNlcjpwYXNz", settings=settings)
        assert exc.value.status_code == 401

    @pytest.mark.asyncio
    async def test_empty_string_header_raises_401(self):
        settings = _settings(auth_enabled=True)
        with pytest.raises(HTTPException) as exc:
            await get_current_user(authorization="", settings=settings)
        assert exc.value.status_code == 401


# ---------------------------------------------------------------------------
# get_current_user — JWT validation paths
# ---------------------------------------------------------------------------


class TestGetCurrentUserJWT:
    @pytest.mark.asyncio
    async def test_invalid_jwt_raises_401(self):
        settings = _settings(auth_enabled=True)
        with pytest.raises(HTTPException) as exc:
            await get_current_user(authorization="Bearer not.a.jwt", settings=settings)
        assert exc.value.status_code == 401

    @pytest.mark.asyncio
    async def test_jwt_with_kid_unknown_raises_401(self):
        """Token with unknown kid → signing key not found → 401."""
        settings = _settings(auth_enabled=True)

        # Build a token with a kid claim (content doesn't matter, we mock JWKS)
        fake_token = "eyJhbGciOiJSUzI1NiIsImtpZCI6InVua25vd24ta2lkIn0.eyJzdWIiOiJ1MSJ9.sig"

        mock_jwks_client = AsyncMock()
        mock_jwks_client.get_signing_key = AsyncMock(return_value=None)

        with (
            patch("app.auth._get_jwks_client", return_value=mock_jwks_client),
            patch("app.auth.jwt.get_unverified_header", return_value={"kid": "unknown-kid", "alg": "RS256"}),
        ):
            with pytest.raises(HTTPException) as exc:
                await get_current_user(authorization=f"Bearer {fake_token}", settings=settings)

        assert exc.value.status_code == 401
        assert "Signing key not found" in str(exc.value.detail)

    @pytest.mark.asyncio
    async def test_jwt_with_kid_valid_returns_user(self):
        """Valid JWT with kid → user extracted from payload."""
        settings = _settings(auth_enabled=True)
        fake_key = {"kid": "test-kid-1", "kty": "RSA"}
        fake_payload = {
            "sub": "user-123",
            "email": "alice@example.com",
            "name": "Alice",
            "provider": "github",
        }
        fake_token = "header.payload.signature"

        mock_jwks_client = AsyncMock()
        mock_jwks_client.get_signing_key = AsyncMock(return_value=fake_key)

        with (
            patch("app.auth._get_jwks_client", return_value=mock_jwks_client),
            patch("app.auth.jwt.get_unverified_header", return_value={"kid": "test-kid-1", "alg": "RS256"}),
            patch("app.auth.jwt.decode", return_value=fake_payload),
        ):
            user = await get_current_user(authorization=f"Bearer {fake_token}", settings=settings)

        assert user.id == "user-123"
        assert user.email == "alice@example.com"
        assert user.name == "Alice"
        assert user.provider == "github"

    @pytest.mark.asyncio
    async def test_jwt_without_kid_uses_full_jwks(self):
        """JWT with no kid → falls back to full JWKS decode."""
        settings = _settings(auth_enabled=True)
        fake_jwks = _make_jwks()
        fake_payload = {"sub": "u2", "email": "bob@example.com", "name": "Bob", "provider": ""}
        fake_token = "header.payload.signature"

        mock_jwks_client = AsyncMock()
        mock_jwks_client.get_signing_keys = AsyncMock(return_value=fake_jwks)

        with (
            patch("app.auth._get_jwks_client", return_value=mock_jwks_client),
            patch("app.auth.jwt.get_unverified_header", return_value={"alg": "RS256"}),
            patch("app.auth.jwt.decode", return_value=fake_payload),
        ):
            user = await get_current_user(authorization=f"Bearer {fake_token}", settings=settings)

        assert user.id == "u2"
        assert user.email == "bob@example.com"

    @pytest.mark.asyncio
    async def test_jwt_decode_raises_jwterror_gives_401(self):
        """JWTError during decode → 401."""
        from jose import JWTError

        settings = _settings(auth_enabled=True)
        fake_key = {"kid": "k1"}
        fake_token = "header.payload.signature"

        mock_jwks_client = AsyncMock()
        mock_jwks_client.get_signing_key = AsyncMock(return_value=fake_key)

        with (
            patch("app.auth._get_jwks_client", return_value=mock_jwks_client),
            patch("app.auth.jwt.get_unverified_header", return_value={"kid": "k1", "alg": "RS256"}),
            patch("app.auth.jwt.decode", side_effect=JWTError("expired")),
        ):
            with pytest.raises(HTTPException) as exc:
                await get_current_user(authorization=f"Bearer {fake_token}", settings=settings)

        assert exc.value.status_code == 401
        assert "expired" in str(exc.value.detail)


# ---------------------------------------------------------------------------
# get_current_user — PAT path
# ---------------------------------------------------------------------------


class TestGetCurrentUserPAT:
    @pytest.mark.asyncio
    async def test_pat_valid_returns_user(self):
        settings = _settings(auth_enabled=True)
        pat_token = "wikis_abc123valid"

        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {
            "valid": True,
            "userId": "pat-user-1",
            "email": "pat@example.com",
            "name": "Pat User",
        }

        with patch("httpx.AsyncClient") as mock_cls:
            mock_http = AsyncMock()
            mock_http.__aenter__ = AsyncMock(return_value=mock_http)
            mock_http.__aexit__ = AsyncMock(return_value=False)
            mock_http.post = AsyncMock(return_value=mock_resp)
            mock_cls.return_value = mock_http

            user = await get_current_user(authorization=f"Bearer {pat_token}", settings=settings)

        assert user.id == "pat-user-1"
        assert user.email == "pat@example.com"
        assert user.provider == "api_key"

    @pytest.mark.asyncio
    async def test_pat_invalid_raises_401(self):
        settings = _settings(auth_enabled=True)
        pat_token = "wikis_badtoken"

        mock_resp = MagicMock()
        mock_resp.status_code = 401

        with patch("httpx.AsyncClient") as mock_cls:
            mock_http = AsyncMock()
            mock_http.__aenter__ = AsyncMock(return_value=mock_http)
            mock_http.__aexit__ = AsyncMock(return_value=False)
            mock_http.post = AsyncMock(return_value=mock_resp)
            mock_cls.return_value = mock_http

            with pytest.raises(HTTPException) as exc:
                await get_current_user(authorization=f"Bearer {pat_token}", settings=settings)

        assert exc.value.status_code == 401

    @pytest.mark.asyncio
    async def test_pat_valid_false_raises_401(self):
        """Auth service returns 200 but valid=False → 401."""
        settings = _settings(auth_enabled=True)
        pat_token = "wikis_revoked"

        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {"valid": False}

        with patch("httpx.AsyncClient") as mock_cls:
            mock_http = AsyncMock()
            mock_http.__aenter__ = AsyncMock(return_value=mock_http)
            mock_http.__aexit__ = AsyncMock(return_value=False)
            mock_http.post = AsyncMock(return_value=mock_resp)
            mock_cls.return_value = mock_http

            with pytest.raises(HTTPException) as exc:
                await get_current_user(authorization=f"Bearer {pat_token}", settings=settings)

        assert exc.value.status_code == 401

    @pytest.mark.asyncio
    async def test_pat_http_error_raises_401(self):
        """Network error during PAT validation → 401."""
        import httpx

        settings = _settings(auth_enabled=True)
        pat_token = "wikis_netfail"

        with patch("httpx.AsyncClient") as mock_cls:
            mock_http = AsyncMock()
            mock_http.__aenter__ = AsyncMock(return_value=mock_http)
            mock_http.__aexit__ = AsyncMock(return_value=False)
            mock_http.post = AsyncMock(side_effect=httpx.ConnectError("refused"))
            mock_cls.return_value = mock_http

            with pytest.raises(HTTPException) as exc:
                await get_current_user(authorization=f"Bearer {pat_token}", settings=settings)

        assert exc.value.status_code == 401

    @pytest.mark.asyncio
    async def test_pat_uses_user_id_fallback_key(self):
        """Falls back to user_id key when userId is missing."""
        settings = _settings(auth_enabled=True)
        pat_token = "wikis_fallback"

        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {
            "valid": True,
            "user_id": "fallback-uid",
            "email": "f@test.com",
            "name": "",
        }

        with patch("httpx.AsyncClient") as mock_cls:
            mock_http = AsyncMock()
            mock_http.__aenter__ = AsyncMock(return_value=mock_http)
            mock_http.__aexit__ = AsyncMock(return_value=False)
            mock_http.post = AsyncMock(return_value=mock_resp)
            mock_cls.return_value = mock_http

            user = await get_current_user(authorization=f"Bearer {pat_token}", settings=settings)

        assert user.id == "fallback-uid"


# ---------------------------------------------------------------------------
# _get_jwks_client — singleton initialisation
# ---------------------------------------------------------------------------


class TestGetJwksClient:
    def test_creates_client_on_first_call(self):
        original = auth_module._jwks_client
        try:
            auth_module._jwks_client = None
            settings = _settings(auth_enabled=True)
            client = auth_module._get_jwks_client(settings)
            assert isinstance(client, JWKSClient)
        finally:
            auth_module._jwks_client = original

    def test_reuses_existing_client(self):
        existing = JWKSClient("http://old.url/jwks")
        original = auth_module._jwks_client
        try:
            auth_module._jwks_client = existing
            settings = _settings(auth_enabled=True)
            client = auth_module._get_jwks_client(settings)
            assert client is existing
        finally:
            auth_module._jwks_client = original
