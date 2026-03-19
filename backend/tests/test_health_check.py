"""Tests for app.services.health_check — provider connectivity checks."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from pydantic import SecretStr

from app.config import Settings
from app.services.health_check import ProviderHealth, check_providers, validate_credentials


@pytest.fixture
def settings_health_enabled() -> Settings:
    return Settings(
        llm_api_key=SecretStr("test-key"),
        llm_provider="openai",
        llm_model="gpt-4o-mini",
        storage_backend="local",
        auth_enabled=False,
        health_check_on_startup=True,
    )


@pytest.fixture
def settings_health_disabled() -> Settings:
    return Settings(
        llm_api_key=SecretStr("test-key"),
        llm_provider="openai",
        llm_model="gpt-4o-mini",
        storage_backend="local",
        auth_enabled=False,
        health_check_on_startup=False,
    )


async def test_health_check_returns_ok_when_all_healthy(settings_health_enabled):
    """Both LLM and embeddings healthy -> both 'ok'."""
    mock_llm = AsyncMock()
    mock_llm.ainvoke.return_value = "OK"

    mock_emb = MagicMock()
    mock_emb.aembed_query = AsyncMock(return_value=[0.1, 0.2])

    with (
        patch("app.services.llm_factory.create_llm", return_value=mock_llm),
        patch("app.services.llm_factory.create_embeddings", return_value=mock_emb),
    ):
        health = await check_providers(settings_health_enabled)

    assert health.llm == "ok"
    assert health.embeddings == "ok"


async def test_health_check_degraded_when_llm_fails(settings_health_enabled):
    """LLM raises -> llm is 'unreachable: ...'."""
    mock_emb = MagicMock()
    mock_emb.aembed_query = AsyncMock(return_value=[0.1])

    with (
        patch("app.services.llm_factory.create_llm", side_effect=RuntimeError("connection refused")),
        patch("app.services.llm_factory.create_embeddings", return_value=mock_emb),
    ):
        health = await check_providers(settings_health_enabled)

    assert health.llm.startswith("unreachable:")
    assert "connection refused" in health.llm
    assert health.embeddings == "ok"


async def test_health_check_degraded_when_embeddings_fail(settings_health_enabled):
    """Embeddings raise -> embeddings is 'unreachable: ...'."""
    mock_llm = AsyncMock()
    mock_llm.ainvoke.return_value = "OK"

    with (
        patch("app.services.llm_factory.create_llm", return_value=mock_llm),
        patch("app.services.llm_factory.create_embeddings", side_effect=RuntimeError("no API key")),
    ):
        health = await check_providers(settings_health_enabled)

    assert health.llm == "ok"
    assert health.embeddings.startswith("unreachable:")
    assert "no API key" in health.embeddings


async def test_health_check_skipped_when_disabled(settings_health_disabled):
    """health_check_on_startup=False -> both fields stay 'unchecked'."""
    health = await check_providers(settings_health_disabled)

    assert health.llm == "unchecked"
    assert health.embeddings == "unchecked"


async def test_provider_health_defaults():
    """ProviderHealth dataclass has correct defaults."""
    h = ProviderHealth()
    assert h.llm == "unchecked"
    assert h.embeddings == "unchecked"


# --- validate_credentials tests ---


def test_validate_credentials_openai_missing_key():
    s = Settings(llm_provider="openai", llm_api_key=SecretStr(""), auth_enabled=False)
    result = validate_credentials(s)
    assert result is not None
    assert "LLM_API_KEY" in result


def test_validate_credentials_openai_with_key():
    s = Settings(llm_provider="openai", llm_api_key=SecretStr("sk-test"), auth_enabled=False)
    assert validate_credentials(s) is None


def test_validate_credentials_anthropic_missing_key():
    s = Settings(llm_provider="anthropic", llm_api_key=SecretStr(""), auth_enabled=False)
    result = validate_credentials(s)
    assert result is not None
    assert "LLM_API_KEY" in result


def test_validate_credentials_gemini_missing_key():
    s = Settings(llm_provider="gemini", llm_api_key=SecretStr(""), auth_enabled=False)
    result = validate_credentials(s)
    assert result is not None
    assert "GEMINI_API_KEY" in result


def test_validate_credentials_gemini_with_key():
    s = Settings(llm_provider="gemini", gemini_api_key=SecretStr("key"), auth_enabled=False)
    assert validate_credentials(s) is None


def test_validate_credentials_ollama_no_creds_needed():
    s = Settings(llm_provider="ollama", llm_api_key=SecretStr(""), auth_enabled=False)
    assert validate_credentials(s) is None


def test_validate_credentials_bedrock_partial_aws_key():
    s = Settings(
        llm_provider="bedrock",
        aws_access_key_id="AKIA123",
        aws_secret_access_key=None,
        auth_enabled=False,
    )
    result = validate_credentials(s)
    assert result is not None
    assert "AWS_SECRET_ACCESS_KEY" in result


def test_validate_credentials_bedrock_iam_role():
    """Bedrock with no explicit keys uses IAM role chain — valid."""
    s = Settings(llm_provider="bedrock", auth_enabled=False)
    assert validate_credentials(s) is None


async def test_check_providers_logs_error_on_missing_credentials(settings_health_enabled, caplog):
    """Missing LLM credentials log error and mark LLM unreachable, but embeddings still checked."""
    import logging

    no_key_settings = Settings(
        llm_provider="openai",
        llm_api_key=SecretStr(""),
        auth_enabled=False,
        health_check_on_startup=True,
    )
    mock_emb = MagicMock()
    mock_emb.aembed_query = AsyncMock(return_value=[0.1, 0.2])

    with (
        caplog.at_level(logging.ERROR, logger="app.services.health_check"),
        patch("app.services.llm_factory.create_embeddings", return_value=mock_emb),
    ):
        health = await check_providers(no_key_settings)

    assert health.llm.startswith("unreachable:")
    # Embeddings checked independently — not blocked by LLM credential error
    assert health.embeddings == "ok"
    assert any("LLM provider openai configured but credentials missing" in r.message for r in caplog.records)


async def test_provider_health_healthy_property():
    h = ProviderHealth(llm="ok", embeddings="ok")
    assert h.healthy is True


async def test_provider_health_unchecked_is_healthy():
    """unchecked (HEALTH_CHECK_ON_STARTUP=false) must not trigger 503."""
    h = ProviderHealth()  # llm="unchecked", embeddings="unchecked"
    assert h.healthy is True


async def test_provider_health_unhealthy_property():
    h = ProviderHealth(llm="unreachable: no key", embeddings="ok")
    assert h.healthy is False
