"""Tests for LLM/embedding factory."""

from __future__ import annotations

import importlib

import pytest
from pydantic import SecretStr

from app.config import Settings
from app.services.llm_factory import create_embeddings, create_llm

_has_ollama = importlib.util.find_spec("langchain_ollama") is not None
_has_bedrock = importlib.util.find_spec("langchain_aws") is not None


def _settings(**overrides) -> Settings:
    defaults = {
        "llm_provider": "openai",
        "llm_api_key": SecretStr("test-key"),
        "llm_model": "gpt-4o-mini",
        "embedding_model": "text-embedding-3-large",
        "llm_max_retries": 0,  # disable retry wrapper in tests
    }
    defaults.update(overrides)
    return Settings(**defaults)


class TestCreateLLM:
    def test_openai_provider(self):
        llm = create_llm(_settings())
        assert type(llm).__name__ == "ChatOpenAI"
        assert llm.model_name == "gpt-4o-mini"

    def test_anthropic_provider(self):
        llm = create_llm(_settings(llm_provider="anthropic", llm_model="claude-sonnet-4-20250514"))
        assert type(llm).__name__ == "ChatAnthropic"

    def test_custom_provider_uses_openai_with_base_url(self):
        s = _settings(llm_provider="custom", llm_api_base="http://localhost:11434/v1")
        llm = create_llm(s)
        assert type(llm).__name__ == "ChatOpenAI"

    def test_temperature_heuristic_reasoning_model(self):
        llm = create_llm(_settings(llm_model="o1-preview"))
        assert llm.temperature == 1.0

    def test_temperature_heuristic_standard_model(self):
        llm = create_llm(_settings(llm_model="gpt-4o-mini"))
        assert llm.temperature == 0.1

    def test_temperature_heuristic_namespaced_model(self):
        """Namespaced models like openai/gpt-4o should not trigger reasoning temp."""
        llm = create_llm(_settings(llm_model="openai/gpt-4o"))
        assert llm.temperature == 0.1

    def test_override_kwargs(self):
        llm = create_llm(_settings(), temperature=0.5, max_tokens=1024)
        assert llm.temperature == 0.5
        assert llm.max_tokens == 1024

    def test_unknown_provider_raises(self):
        with pytest.raises(ValueError, match="Unknown LLM provider"):
            create_llm(_settings(llm_provider="unknown"))

    @pytest.mark.skipif(not _has_ollama, reason="langchain-ollama not installed")
    def test_ollama_provider(self):
        llm = create_llm(_settings(llm_provider="ollama", llm_model="llama3"))
        assert type(llm).__name__ == "ChatOllama"

    def test_gemini_provider(self):
        s = _settings(llm_provider="gemini", llm_model="gemini-pro", gemini_api_key=SecretStr("gk"))
        llm = create_llm(s)
        assert type(llm).__name__ == "ChatGoogleGenerativeAI"

    def test_gemini_missing_key_raises(self):
        with pytest.raises(ValueError, match="GEMINI_API_KEY"):
            create_llm(_settings(llm_provider="gemini", gemini_api_key=None))

    def test_copilot_provider(self):
        llm = create_llm(_settings(llm_provider="copilot", llm_model="gpt-4o"))
        assert type(llm).__name__ == "ChatOpenAI"
        assert llm.openai_api_base == "https://api.githubcopilot.com"

    def test_github_provider_openai_model(self):
        llm = create_llm(_settings(llm_provider="github", llm_model="openai/gpt-4o"))
        assert type(llm).__name__ == "ChatOpenAI"
        assert llm.openai_api_base == "https://models.inference.ai.azure.com"

    def test_github_provider_claude_model(self):
        llm = create_llm(_settings(llm_provider="github", llm_model="claude-sonnet-4-5"))
        assert type(llm).__name__ == "ChatAnthropic"

    def test_github_provider_anthropic_prefixed_model(self):
        llm = create_llm(_settings(llm_provider="github", llm_model="anthropic/claude-sonnet-4-5"))
        assert type(llm).__name__ == "ChatAnthropic"

    @pytest.mark.skipif(not _has_bedrock, reason="langchain-aws not installed")
    def test_bedrock_provider(self):
        llm = create_llm(_settings(llm_provider="bedrock", llm_model="anthropic.claude-3-sonnet"))
        assert type(llm).__name__ == "ChatBedrock"


class TestCreateEmbeddings:
    def test_openai_embeddings_with_key(self):
        emb = create_embeddings(_settings())
        assert type(emb).__name__ == "OpenAIEmbeddings"

    def test_falls_back_to_llm_key(self):
        s = _settings(embedding_api_key=None)
        emb = create_embeddings(s)
        assert type(emb).__name__ == "OpenAIEmbeddings"

    def test_uses_dedicated_embedding_key(self):
        s = _settings(embedding_api_key=SecretStr("embed-key"))
        emb = create_embeddings(s)
        assert type(emb).__name__ == "OpenAIEmbeddings"

    def test_no_key_raises(self):
        s = _settings(llm_api_key=SecretStr(""), embedding_api_key=None)
        with pytest.raises(ValueError, match="No API key"):
            create_embeddings(s)

    def test_model_override(self):
        emb = create_embeddings(_settings(), model="text-embedding-ada-002")
        assert emb.model == "text-embedding-ada-002"

    @pytest.mark.skipif(not _has_ollama, reason="langchain-ollama not installed")
    def test_ollama_embeddings(self):
        emb = create_embeddings(_settings(llm_provider="ollama"))
        assert type(emb).__name__ == "OllamaEmbeddings"

    def test_gemini_embeddings(self):
        emb = create_embeddings(_settings(llm_provider="gemini", gemini_api_key=SecretStr("gk")))
        assert type(emb).__name__ == "GoogleGenerativeAIEmbeddings"

    def test_gemini_embeddings_missing_key_raises(self):
        with pytest.raises(ValueError, match="GEMINI_API_KEY"):
            create_embeddings(_settings(llm_provider="gemini", gemini_api_key=None))

    @pytest.mark.skipif(not _has_bedrock, reason="langchain-aws not installed")
    def test_bedrock_embeddings(self):
        emb = create_embeddings(_settings(llm_provider="bedrock"))
        assert type(emb).__name__ == "BedrockEmbeddings"

    def test_anthropic_raises(self):
        with pytest.raises(ValueError, match="no embedding API"):
            create_embeddings(_settings(llm_provider="anthropic"))

    def test_copilot_raises(self):
        with pytest.raises(ValueError, match="no embedding API"):
            create_embeddings(_settings(llm_provider="copilot"))

    def test_github_embeddings(self):
        emb = create_embeddings(_settings(llm_provider="github"))
        assert type(emb).__name__ == "OpenAIEmbeddings"
