"""
Configurable LLM and embedding model factory.

Creates LangChain-compatible LLM and embedding instances from app settings,
supporting multiple providers (OpenAI, Anthropic, custom OpenAI-compatible).
"""

from __future__ import annotations

import logging
from typing import Any

from langchain_core.embeddings import Embeddings
from langchain_core.language_models import BaseChatModel

from app.config import Settings

logger = logging.getLogger(__name__)


def _bedrock_credentials(settings: Settings) -> dict[str, str]:
    """Build explicit boto3 credentials from settings. Raises if missing."""
    creds = {}
    if settings.aws_access_key_id:
        creds["aws_access_key_id"] = settings.aws_access_key_id
    if settings.aws_secret_access_key:
        creds["aws_secret_access_key"] = settings.aws_secret_access_key.get_secret_value()
    if not creds:
        logger.warning("Bedrock: no explicit AWS creds — falling back to boto3 default chain")
    return creds


def create_llm(settings: Settings, tier: str = "high", **overrides: Any) -> BaseChatModel:
    """Create an LLM instance from settings.

    Supports providers: openai, anthropic, custom, ollama, gemini, bedrock.
    Tier: "high" (default, page generation) or "low" (quality check, enhancement).
    Overrides: model, temperature, max_tokens, streaming.
    """
    from langchain_openai import ChatOpenAI

    skip_retry = overrides.pop("skip_retry", False)
    provider = overrides.pop("provider", settings.llm_provider)

    # Resolve model from tier
    if "model" in overrides:
        model = overrides.pop("model")
    elif tier == "low" and settings.llm_model_low:
        model = settings.llm_model_low
    else:
        model = settings.llm_model
    api_key = settings.llm_api_key.get_secret_value()
    base_url = overrides.pop("base_url", settings.llm_api_base)

    # Temperature heuristic: reasoning models (o*) need temperature=1.0
    default_temp = 1.0 if model.startswith("o") else 0.1
    temperature = overrides.pop("temperature", default_temp)
    max_tokens = overrides.pop("max_tokens", 64000)
    streaming = overrides.pop("streaming", False)

    if provider == "anthropic":
        from langchain_anthropic import ChatAnthropic

        llm = ChatAnthropic(
            model=model,
            api_key=api_key,
            temperature=temperature,
            max_tokens=max_tokens,
            streaming=streaming,
            **overrides,
        )
    elif provider in ("openai", "custom"):
        llm = ChatOpenAI(
            model=model,
            api_key=api_key,
            base_url=base_url,
            temperature=temperature,
            max_tokens=max_tokens,
            streaming=streaming,
            **overrides,
        )
    elif provider == "ollama":
        from langchain_ollama import ChatOllama

        llm = ChatOllama(
            model=model,
            base_url=settings.ollama_base_url,
            temperature=temperature,
            num_ctx=settings.ollama_num_ctx,
            **overrides,
        )
    elif provider == "gemini":
        from langchain_google_genai import ChatGoogleGenerativeAI

        gemini_key = settings.gemini_api_key
        if not gemini_key:
            raise ValueError("GEMINI_API_KEY required for gemini provider")
        llm = ChatGoogleGenerativeAI(
            model=model,
            google_api_key=gemini_key.get_secret_value(),
            temperature=temperature,
            max_output_tokens=max_tokens,
            **overrides,
        )
    elif provider == "bedrock":
        import botocore.config
        from langchain_aws import ChatBedrock

        bedrock_kwargs = _bedrock_credentials(settings)
        bedrock_config = botocore.config.Config(
            read_timeout=settings.llm_timeout, connect_timeout=30, retries={"max_attempts": 3}
        )
        llm = ChatBedrock(
            model_id=model,
            region_name=settings.aws_region,
            model_kwargs={"temperature": temperature, "max_tokens": max_tokens},
            streaming=streaming,
            config=bedrock_config,
            **bedrock_kwargs,
            **overrides,
        )
    else:
        raise ValueError(f"Unknown LLM provider: {provider}")

    logger.info(f"Created LLM: provider={provider}, model={model}, base_url={base_url}")

    # Add retry with exponential backoff for transient errors
    # Skip for ask/research engines — RunnableRetry breaks _llm_type access
    max_retries = settings.llm_max_retries
    if max_retries > 0 and not skip_retry:
        llm = llm.with_retry(
            retry_if_exception_type=(Exception,),
            wait_exponential_jitter=True,
            stop_after_attempt=max_retries,
        )
        logger.info(f"LLM retry enabled: max_retries={max_retries}")

    # Add fallback chain if configured
    if settings.llm_fallback_models:
        fallback_models = [m.strip() for m in settings.llm_fallback_models.split(",") if m.strip()]
        fallback_provider = settings.llm_fallback_provider or provider
        fallbacks = []
        for fb_model in fallback_models:
            try:
                fb_llm = create_llm(
                    Settings(
                        **{
                            **settings.model_dump(),
                            "llm_provider": fallback_provider,
                            "llm_model": fb_model,
                            "llm_fallback_models": None,  # prevent recursion
                            "llm_max_retries": min(max_retries, 2),  # reduced retries for fallbacks
                        }
                    ),
                )
                fallbacks.append(fb_llm)
                logger.info(f"Fallback model added: {fallback_provider}/{fb_model}")
            except Exception as e:
                logger.warning(f"Failed to create fallback {fb_model}: {e}")
        if fallbacks:
            llm = llm.with_fallbacks(fallbacks)
            logger.info(f"Fallback chain: {len(fallbacks)} models")

    return llm


def create_embeddings(settings: Settings, **overrides: Any) -> Embeddings:
    """Create an embeddings instance from settings.

    Supports providers: openai, custom, ollama, gemini, bedrock.
    Raises ValueError for unsupported providers or missing credentials.
    """
    model = overrides.pop("model", settings.embedding_model)
    provider = overrides.pop("provider", settings.llm_provider)

    # Anthropic has no embedding API
    if provider == "anthropic":
        raise ValueError("Anthropic has no embedding API — use a dedicated embedding provider")

    # Resolve embedding API key (falls back to LLM key)
    embedding_key = settings.embedding_api_key
    if embedding_key:
        api_key = embedding_key.get_secret_value()
    else:
        api_key = settings.llm_api_key.get_secret_value()

    base_url = overrides.pop("base_url", settings.embedding_api_base)

    if provider == "ollama":
        from langchain_ollama import OllamaEmbeddings

        embeddings = OllamaEmbeddings(model=model or "nomic-embed-text", base_url=settings.ollama_base_url, **overrides)
        logger.info(f"Created embeddings: Ollama model={model}")
    elif provider == "gemini":
        from langchain_google_genai import GoogleGenerativeAIEmbeddings

        gemini_key = settings.gemini_api_key
        if not gemini_key:
            raise ValueError("GEMINI_API_KEY required for gemini embeddings")
        embeddings = GoogleGenerativeAIEmbeddings(
            model=model or "models/embedding-001", google_api_key=gemini_key.get_secret_value(), **overrides
        )
        logger.info(f"Created embeddings: Gemini model={model}")
    elif provider == "bedrock":
        from langchain_aws import BedrockEmbeddings

        bedrock_kwargs = _bedrock_credentials(settings)
        embeddings = BedrockEmbeddings(
            model_id=model or "amazon.titan-embed-text-v1",
            region_name=settings.aws_region,
            **bedrock_kwargs,
            **overrides,
        )
        logger.info(f"Created embeddings: Bedrock model={model}")
    elif api_key:
        from langchain_openai.embeddings import OpenAIEmbeddings

        embeddings = OpenAIEmbeddings(model=model, api_key=api_key, base_url=base_url, **overrides)
        logger.info(f"Created embeddings: OpenAI model={model}, base_url={base_url}")
    else:
        raise ValueError("No API key available for embeddings — configure llm_api_key or embedding_api_key")
    return embeddings
