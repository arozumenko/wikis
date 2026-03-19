"""JWT validation middleware with JWKS fetching."""

from __future__ import annotations

import logging
import time

import httpx
from fastapi import Depends, Header, HTTPException
from jose import JWTError, jwt
from pydantic import BaseModel

from app.config import Settings, get_settings

logger = logging.getLogger(__name__)


class CurrentUser(BaseModel):
    """Authenticated user context extracted from JWT."""

    id: str
    email: str
    name: str = ""
    provider: str = ""


class JWKSClient:
    """Fetches and caches JWKS from auth service."""

    def __init__(self, jwks_url: str, cache_ttl: int = 3600) -> None:
        self._jwks_url = jwks_url
        self._cache: dict | None = None
        self._cached_at: float = 0
        self._cache_ttl = cache_ttl

    async def _fetch_jwks(self) -> dict:
        """Fetch JWKS from auth service."""
        async with httpx.AsyncClient(timeout=10) as client:
            resp = await client.get(self._jwks_url)
            resp.raise_for_status()
            return resp.json()

    async def get_signing_keys(self) -> dict:
        """Return JWKS, using cache if fresh."""
        now = time.time()
        if self._cache and (now - self._cached_at) < self._cache_ttl:
            return self._cache
        self._cache = await self._fetch_jwks()
        self._cached_at = now
        logger.info(f"JWKS refreshed from {self._jwks_url}")
        return self._cache

    async def get_signing_key(self, kid: str) -> dict | None:
        """Find a specific key by kid."""
        jwks = await self.get_signing_keys()
        for key in jwks.get("keys", []):
            if key.get("kid") == kid:
                return key
        return None


# Module-level JWKS client (initialized lazily)
_jwks_client: JWKSClient | None = None


def _get_jwks_client(settings: Settings) -> JWKSClient:
    global _jwks_client
    if _jwks_client is None:
        _jwks_client = JWKSClient(settings.auth_jwks_url)
    return _jwks_client


async def get_current_user(
    authorization: str | None = Header(None),
    settings: Settings = Depends(get_settings),
) -> CurrentUser:
    """FastAPI dependency — validates JWT, returns user context.

    When auth_enabled=False, returns a dev-mode placeholder user.
    """
    if not settings.auth_enabled:
        return CurrentUser(id="dev-user", email="dev@localhost", name="Dev User", provider="dev")

    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(401, "Missing or invalid Authorization header")

    token = authorization.split(" ", 1)[1]

    # PAT validation (wikis_ prefix)
    if token.startswith("wikis_"):
        return await _validate_pat(token, settings)

    # JWT validation
    try:
        unverified_header = jwt.get_unverified_header(token)
        kid = unverified_header.get("kid")

        jwks_client = _get_jwks_client(settings)

        if kid:
            key = await jwks_client.get_signing_key(kid)
            if not key:
                raise HTTPException(401, "Signing key not found")
            payload = jwt.decode(token, key, algorithms=["RS256"])
        else:
            jwks = await jwks_client.get_signing_keys()
            payload = jwt.decode(token, jwks, algorithms=["RS256"])

        return CurrentUser(
            id=payload.get("sub", ""),
            email=payload.get("email", ""),
            name=payload.get("name", ""),
            provider=payload.get("provider", ""),
        )
    except JWTError as e:
        raise HTTPException(401, f"Invalid token: {e}") from e


async def _validate_pat(token: str, settings: Settings) -> CurrentUser:
    """Validate a Personal Access Token via Better Auth's API key verify endpoint."""
    verify_url = settings.auth_jwks_url.replace("/jwks", "/api-key/verify")
    try:
        async with httpx.AsyncClient(timeout=10) as client:
            resp = await client.post(verify_url, json={"key": token})
            if resp.status_code != 200:
                raise HTTPException(401, "Invalid or expired API key")
            data = resp.json()
            if not data.get("valid", False):
                raise HTTPException(401, "Invalid or revoked API key")
            return CurrentUser(
                id=data.get("userId", data.get("user_id", "")),
                email=data.get("email", ""),
                name=data.get("name", ""),
                provider="api_key",
            )
    except httpx.HTTPError as e:
        logger.warning(f"PAT validation failed: {e}")
        raise HTTPException(401, f"Auth service unreachable: {e}") from e
