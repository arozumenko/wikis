"""Tests for app.services.context_limits — token estimation and limit checks."""

from __future__ import annotations

from app.services.context_limits import (
    DEFAULT_CONTEXT_LIMIT,
    DEFAULT_EMBEDDING_LIMIT,
    chunk_for_embedding,
    estimate_tokens,
    get_context_limit,
    get_embedding_limit,
    truncate_for_context,
)

# --------------------------------------------------------------------------- #
# estimate_tokens
# --------------------------------------------------------------------------- #


def test_estimate_tokens_empty():
    assert estimate_tokens("") == 0


def test_estimate_tokens_short():
    # 12 chars -> 3 tokens
    assert estimate_tokens("hello world!") == 3


def test_estimate_tokens_exact_multiple():
    assert estimate_tokens("abcd") == 1


def test_estimate_tokens_longer():
    text = "a" * 400
    assert estimate_tokens(text) == 100


# --------------------------------------------------------------------------- #
# get_context_limit
# --------------------------------------------------------------------------- #


def test_get_context_limit_known_model():
    assert get_context_limit("gpt-4o") == 128000


def test_get_context_limit_claude():
    assert get_context_limit("claude-opus-4-6") == 200000


def test_get_context_limit_unknown_model_returns_default():
    assert get_context_limit("some-unknown-model-xyz") == DEFAULT_CONTEXT_LIMIT


def test_get_context_limit_prefix_match():
    """Model name starting with a known prefix should match."""
    assert get_context_limit("gpt-4-turbo-2024-04-09") == 128000


# --------------------------------------------------------------------------- #
# get_embedding_limit
# --------------------------------------------------------------------------- #


def test_get_embedding_limit_known():
    assert get_embedding_limit("text-embedding-3-large") == 8191


def test_get_embedding_limit_unknown():
    assert get_embedding_limit("unknown-embed") == DEFAULT_EMBEDDING_LIMIT


def test_get_embedding_limit_prefix_match():
    assert get_embedding_limit("nomic-embed-text-v2") == 8192


# --------------------------------------------------------------------------- #
# truncate_for_context
# --------------------------------------------------------------------------- #


def test_truncate_short_text_unchanged():
    """Text that fits should be returned as-is."""
    text = "a" * 100
    result = truncate_for_context(text, "gpt-4o")
    assert result == text


def test_truncate_long_text():
    """Text exceeding the limit should be truncated with marker."""
    # gpt-4 has 8192 token limit. Use 40000 chars (~10000 tokens) with 4096 reserve
    text = "x" * 40000
    result = truncate_for_context(text, "gpt-4")
    assert "[... content truncated for context limit ...]" in result
    assert len(result) < len(text)


def test_truncate_respects_system_prompt():
    """System prompt tokens should reduce available space."""
    # gpt-4: 8192 limit, reserve 4096 -> 4096 available
    # With a large system prompt consuming most remaining space
    system = "s" * 12000  # ~3000 tokens
    text = "x" * 8000  # ~2000 tokens -> should exceed (4096 - 3000 = 1096)
    result = truncate_for_context(text, "gpt-4", system_prompt=system)
    assert "[... content truncated for context limit ...]" in result


def test_truncate_preserves_head_and_tail():
    """Truncated text should start with original start and end with original end."""
    text = "START" + "m" * 40000 + "END"
    result = truncate_for_context(text, "gpt-4")
    assert result.startswith("START")
    assert result.endswith("END")


# --------------------------------------------------------------------------- #
# chunk_for_embedding
# --------------------------------------------------------------------------- #


def test_chunk_short_text_single_chunk():
    """Text fitting in limit should return single chunk."""
    text = "hello world"
    chunks = chunk_for_embedding(text, "text-embedding-3-large")
    assert len(chunks) == 1
    assert chunks[0] == text


def test_chunk_long_text_multiple_chunks():
    """Text exceeding limit should produce multiple chunks."""
    # text-embedding-3-large: 8191 tokens -> ~32764 chars
    text = "a" * 80000
    chunks = chunk_for_embedding(text, "text-embedding-3-large")
    assert len(chunks) > 1
    # All text should be covered
    for chunk in chunks:
        assert len(chunk) > 0


def test_chunk_overlap():
    """Chunks should overlap."""
    text = "a" * 80000
    chunks = chunk_for_embedding(text, "text-embedding-3-large", overlap_tokens=200)
    assert len(chunks) >= 2
    # With overlap, total chars across chunks > original
    total_chars = sum(len(c) for c in chunks)
    assert total_chars > len(text)
