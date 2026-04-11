from __future__ import annotations

from pydantic import SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # LLM
    llm_provider: str = "openai"  # openai | anthropic | custom | ollama | gemini | bedrock | copilot | github
    llm_api_key: SecretStr = SecretStr("")
    llm_api_base: str | None = None
    llm_model: str = "gpt-4o-mini"  # HIGH tier (page generation, analysis)
    llm_model_low: str | None = None  # LOW tier (quality check, enhancement) — falls back to llm_model

    # Embeddings
    embedding_provider: str | None = None  # defaults to llm_provider; set to "openai" when using anthropic LLM
    embedding_model: str = "text-embedding-3-large"
    embedding_api_key: SecretStr | None = None  # falls back to llm_api_key
    embedding_api_base: str | None = None

    # Storage
    storage_backend: str = "local"  # local | s3
    storage_path: str = "./data/artifacts"
    cache_dir: str = "./data/cache"

    # Provider-specific
    ollama_base_url: str = "http://localhost:11434"
    ollama_num_ctx: int = 32768  # context window for Ollama models
    gemini_api_key: SecretStr | None = None
    aws_region: str = "us-east-1"
    aws_access_key_id: str | None = None
    aws_secret_access_key: SecretStr | None = None
    github_copilot_base_url: str = "https://api.githubcopilot.com"
    github_models_base_url: str = "https://models.inference.ai.azure.com"

    # Auth
    auth_enabled: bool = True  # set False for local dev without auth service
    auth_jwks_url: str = "http://app:3000/api/auth/jwks"

    # Local repo support
    allowed_local_paths: str | None = None  # comma-separated base paths

    # Health
    health_check_on_startup: bool = True

    # LLM retry + fallback
    llm_max_retries: int = 5
    llm_timeout: int = 900  # seconds (15 min — LLM analysis calls can be very long)
    llm_fallback_models: str | None = None  # comma-separated, e.g. "claude-sonnet-4-6,gpt-4o-mini"
    llm_fallback_provider: str | None = None  # provider for fallback models if different

    # Ask engine
    ask_similarity_threshold: float = 0.75  # EmbeddingsFilter threshold for agentic tools (0.0 = disabled)
    ask_cache_max_wikis: int = 10
    ask_cache_ttl_seconds: int = 3600

    # Research engine
    research_similarity_threshold: float = 0.0  # EmbeddingsFilter threshold (0.0 = disabled, FAISS handles ranking)

    # Database (shared with web app's Prisma; empty = default SQLite)
    database_url: str = ""

    # Wiki page index cache
    wiki_index_cache_max_wikis: int = 50

    # Q&A Cache
    qa_cache_enabled: bool = True
    qa_cache_similarity_threshold: float = 0.92
    qa_cache_max_age_seconds: int = 86400  # 24 hours
    qa_cache_max_wikis: int = 50  # LRU eviction threshold for in-memory FAISS indexes

    # App
    log_level: str = "INFO"


def get_settings() -> Settings:
    return Settings()
