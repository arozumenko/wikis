"""Tests for app.auth — JWT validation and user extraction."""

from __future__ import annotations

import pytest
from fastapi import HTTPException
from pydantic import SecretStr

from app.auth import CurrentUser, get_current_user
from app.config import Settings


@pytest.fixture
def settings_auth_disabled() -> Settings:
    return Settings(
        llm_api_key=SecretStr("test-key"),
        llm_provider="openai",
        llm_model="gpt-4o-mini",
        storage_backend="local",
        auth_enabled=False,
    )


@pytest.fixture
def settings_auth_enabled() -> Settings:
    return Settings(
        llm_api_key=SecretStr("test-key"),
        llm_provider="openai",
        llm_model="gpt-4o-mini",
        storage_backend="local",
        auth_enabled=True,
        auth_jwks_url="http://app:3000/api/auth/jwks",
    )


# --------------------------------------------------------------------------- #
# CurrentUser model validation
# --------------------------------------------------------------------------- #


def test_current_user_required_fields():
    user = CurrentUser(id="u1", email="u@example.com")
    assert user.id == "u1"
    assert user.email == "u@example.com"
    assert user.name == ""
    assert user.provider == ""


def test_current_user_all_fields():
    user = CurrentUser(id="u2", email="a@b.com", name="Alice", provider="github")
    assert user.name == "Alice"
    assert user.provider == "github"


def test_current_user_missing_required_raises():
    with pytest.raises(Exception):
        CurrentUser(id="u1")  # email is required


# --------------------------------------------------------------------------- #
# get_current_user — auth disabled
# --------------------------------------------------------------------------- #


async def test_get_current_user_returns_dev_user_when_auth_disabled(settings_auth_disabled):
    user = await get_current_user(authorization=None, settings=settings_auth_disabled)
    assert user.id == "dev-user"
    assert user.email == "dev@localhost"
    assert user.provider == "dev"


async def test_get_current_user_ignores_token_when_auth_disabled(settings_auth_disabled):
    """Even with a token, auth_disabled returns dev user."""
    user = await get_current_user(authorization="Bearer some-token", settings=settings_auth_disabled)
    assert user.id == "dev-user"


# --------------------------------------------------------------------------- #
# get_current_user — auth enabled, missing/invalid token
# --------------------------------------------------------------------------- #


async def test_get_current_user_raises_401_no_header(settings_auth_enabled):
    with pytest.raises(HTTPException) as exc_info:
        await get_current_user(authorization=None, settings=settings_auth_enabled)
    assert exc_info.value.status_code == 401


async def test_get_current_user_raises_401_empty_header(settings_auth_enabled):
    with pytest.raises(HTTPException) as exc_info:
        await get_current_user(authorization="", settings=settings_auth_enabled)
    assert exc_info.value.status_code == 401


async def test_get_current_user_raises_401_non_bearer(settings_auth_enabled):
    with pytest.raises(HTTPException) as exc_info:
        await get_current_user(authorization="Basic abc123", settings=settings_auth_enabled)
    assert exc_info.value.status_code == 401


async def test_get_current_user_raises_401_invalid_jwt(settings_auth_enabled):
    """Bearer token present but not a valid JWT -> 401."""
    # Reset the module-level client so our mock takes effect
    import app.auth as auth_module

    original = auth_module._jwks_client
    auth_module._jwks_client = None
    try:
        with pytest.raises(HTTPException) as exc_info:
            await get_current_user(
                authorization="Bearer not-a-real-jwt",
                settings=settings_auth_enabled,
            )
        assert exc_info.value.status_code == 401
    finally:
        auth_module._jwks_client = original
