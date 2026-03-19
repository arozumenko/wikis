"""Context overflow detection and chunking utilities."""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)

# Context window limits (tokens) per model
CONTEXT_LIMITS: dict[str, int] = {
    "gpt-4o": 128000,
    "gpt-4o-mini": 128000,
    "gpt-4-turbo": 128000,
    "gpt-4": 8192,
    "gpt-3.5-turbo": 16385,
    "claude-opus-4-6": 200000,
    "claude-sonnet-4-6": 200000,
    "claude-haiku-4-5": 200000,
    "claude-3-5-sonnet-20241022": 200000,
    "claude-3-sonnet-20240229": 200000,
    "llama3": 8192,
    "llama3.1": 128000,
    "gemini-2.0-flash": 1048576,
    "gemini-pro": 32760,
    "anthropic.claude-3-sonnet-20240229-v1:0": 200000,
    "anthropic.claude-3-haiku-20240307-v1:0": 200000,
}

# Embedding token limits per model
EMBEDDING_LIMITS: dict[str, int] = {
    "text-embedding-3-large": 8191,
    "text-embedding-3-small": 8191,
    "text-embedding-ada-002": 8191,
    "amazon.titan-embed-text-v1": 8192,
    "amazon.titan-embed-text-v2:0": 8192,
    "nomic-embed-text": 8192,
    "models/embedding-001": 2048,
}

DEFAULT_CONTEXT_LIMIT = 128000
DEFAULT_EMBEDDING_LIMIT = 8191


def estimate_tokens(text: str) -> int:
    """Estimate token count using chars/4 heuristic."""
    return len(text) // 4


def get_context_limit(model: str) -> int:
    """Get context window limit for a model."""
    if model in CONTEXT_LIMITS:
        return CONTEXT_LIMITS[model]
    # Fuzzy match: check if model starts with a known prefix
    for known, limit in CONTEXT_LIMITS.items():
        if model.startswith(known):
            return limit
    return DEFAULT_CONTEXT_LIMIT


def get_embedding_limit(model: str) -> int:
    """Get embedding token limit for a model."""
    if model in EMBEDDING_LIMITS:
        return EMBEDDING_LIMITS[model]
    for known, limit in EMBEDDING_LIMITS.items():
        if model.startswith(known):
            return limit
    return DEFAULT_EMBEDDING_LIMIT


def truncate_for_context(
    text: str,
    model: str,
    system_prompt: str = "",
    reserve_tokens: int = 4096,
) -> str:
    """Truncate text to fit within model context window.

    Uses middle-out strategy: preserves system prompt + start of text + end of text.
    """
    limit = get_context_limit(model)
    system_tokens = estimate_tokens(system_prompt)
    available = limit - system_tokens - reserve_tokens
    text_tokens = estimate_tokens(text)

    if text_tokens <= available:
        return text

    # Middle-out: keep first 60% and last 40% of available chars
    available_chars = available * 4
    head_chars = int(available_chars * 0.6)
    tail_chars = int(available_chars * 0.4)

    truncated = text[:head_chars] + "\n\n[... content truncated for context limit ...]\n\n" + text[-tail_chars:]
    logger.warning(
        f"Truncated input from ~{text_tokens} to ~{estimate_tokens(truncated)} tokens (model={model}, limit={limit})"
    )
    return truncated


def chunk_for_embedding(
    text: str,
    model: str,
    overlap_tokens: int = 200,
) -> list[str]:
    """Split text into chunks that fit the embedding model's token limit."""
    limit = get_embedding_limit(model)
    text_tokens = estimate_tokens(text)

    if text_tokens <= limit:
        return [text]

    chunk_chars = limit * 4
    overlap_chars = overlap_tokens * 4
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_chars
        chunks.append(text[start:end])
        start = end - overlap_chars

    logger.info(f"Split text ({text_tokens} tokens) into {len(chunks)} chunks for embedding (model={model})")
    return chunks
