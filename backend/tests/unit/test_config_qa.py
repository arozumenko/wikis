"""Tests for QA cache configuration settings."""

from pydantic import SecretStr

from app.config import Settings


def test_qa_cache_defaults():
    """Verify default values for QA cache settings."""
    settings = Settings(llm_api_key=SecretStr("test-key"))
    assert settings.qa_cache_enabled is True
    assert settings.qa_cache_similarity_threshold == 0.92
    assert settings.qa_cache_max_age_seconds == 86400
    assert settings.qa_cache_max_wikis == 50


def test_qa_cache_env_override(monkeypatch):
    """Verify QA cache settings can be overridden via env vars."""
    monkeypatch.setenv("QA_CACHE_ENABLED", "false")
    monkeypatch.setenv("QA_CACHE_SIMILARITY_THRESHOLD", "0.85")
    monkeypatch.setenv("QA_CACHE_MAX_AGE_SECONDS", "3600")
    monkeypatch.setenv("QA_CACHE_MAX_WIKIS", "100")
    settings = Settings(llm_api_key=SecretStr("test-key"))
    assert settings.qa_cache_enabled is False
    assert settings.qa_cache_similarity_threshold == 0.85
    assert settings.qa_cache_max_age_seconds == 3600
    assert settings.qa_cache_max_wikis == 100
