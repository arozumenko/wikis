"""Context overflow detection and error classification for LLM API errors."""

from __future__ import annotations

import re

# ---------------------------------------------------------------------------
# Regex patterns for known LLM context overflow error messages
# ---------------------------------------------------------------------------

CONTEXT_OVERFLOW_PATTERNS: list[re.Pattern[str]] = [
    # OpenAI
    re.compile(r"maximum context length", re.IGNORECASE),
    re.compile(r"context_length_exceeded", re.IGNORECASE),
    re.compile(r"This model's maximum context length is \d+", re.IGNORECASE),
    re.compile(r"maximum token limit", re.IGNORECASE),
    # Anthropic
    re.compile(r"prompt is too long", re.IGNORECASE),
    re.compile(r"too many tokens", re.IGNORECASE),
    re.compile(r"Input is too long", re.IGNORECASE),
    re.compile(r"input_tokens.*exceeds.*context", re.IGNORECASE),
    # Generic / other providers
    re.compile(r"token limit exceeded", re.IGNORECASE),
    re.compile(r"context window exceeded", re.IGNORECASE),
    re.compile(r"context window full", re.IGNORECASE),
    re.compile(r"exceeds.*token.*limit", re.IGNORECASE),
    re.compile(r"request too large.*token", re.IGNORECASE),
    re.compile(r"payload too large.*token", re.IGNORECASE),
    re.compile(r"reduce the length of the messages", re.IGNORECASE),
    # Ollama / llama.cpp
    re.compile(r"n_ctx.*exceeded", re.IGNORECASE),
    re.compile(r"context size.*exceeded", re.IGNORECASE),
    # Gemini
    re.compile(r"Input token limit exceeded", re.IGNORECASE),
    re.compile(r"input_token_limit_exceeded", re.IGNORECASE),
]

# ---------------------------------------------------------------------------
# Helpers for extracting structured info from error messages
# ---------------------------------------------------------------------------

# Match the limit: "maximum context length is 128000", "200000 token limit", "limit of 128000"
_LIMIT_PATTERN = re.compile(
    r"(?:"
    r"(?:length|limit)\s+(?:is|of)\s+(\d{4,})"  # "length is 128000" or "limit of 128000"
    r"|(\d{4,})\s*(?:-?\s*)?token\s*limit"  # "200000 token limit"
    r")",
    re.IGNORECASE,
)
# Match the used count: "resulted in 210000 tokens" or "your messages resulted in 210000"
_USED_PATTERN = re.compile(r"(?:resulted in|requested|however.*?you|input.*?used?)\s+(\d{4,})", re.IGNORECASE)


def _extract_model_limit(message: str) -> int | None:
    """Extract the model's reported token limit from an error message."""
    m = _LIMIT_PATTERN.search(message)
    if not m:
        return None
    # Either group(1) or group(2) will have the number
    return int(m.group(1) or m.group(2))


def _extract_estimated_tokens(message: str) -> int | None:
    """Extract the reported token usage from an error message."""
    m = _USED_PATTERN.search(message)
    return int(m.group(1)) if m else None


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


class ContextOverflowError(Exception):
    """Raised when an LLM call fails because the context window was exceeded.

    Wraps the original exception with structured diagnostics so the caller
    can surface a helpful message to the user.
    """

    DEFAULT_SUGGESTIONS: list[str] = [
        "Switch to a model with a larger context window (e.g. claude-sonnet-4-6 with 200K tokens).",
        "Enable 'force_rebuild_index=True' to re-index with stricter file filters.",
        "Reduce repository scope by excluding large generated files or build artefacts.",
        "Split large pages into smaller, more focused topics.",
    ]

    def __init__(
        self,
        original: Exception,
        model_limit: int | None = None,
        estimated_tokens: int | None = None,
        suggestions: list[str] | None = None,
    ) -> None:
        self.original = original
        self.model_limit = model_limit
        self.estimated_tokens = estimated_tokens
        self.suggestions = suggestions or self.DEFAULT_SUGGESTIONS

        # Build a human-readable message
        parts: list[str] = ["Context window exceeded"]
        if model_limit:
            parts.append(f"(model limit: {model_limit:,} tokens)")
        if estimated_tokens:
            parts.append(f"— input used ~{estimated_tokens:,} tokens")
        super().__init__(" ".join(parts))

    def __str__(self) -> str:
        base = super().__str__()
        hints = "; ".join(self.suggestions[:2])
        return f"{base}. Suggestions: {hints}"


def is_context_overflow(error: Exception) -> bool:
    """Return True if *error* represents a context/token limit exceeded error.

    Checks the string representation of the exception (message + type name)
    against the known patterns.
    """
    text = f"{type(error).__name__}: {error}"
    return any(p.search(text) for p in CONTEXT_OVERFLOW_PATTERNS)


def classify_and_wrap(error: Exception, model: str = "") -> Exception:
    """Classify *error* and wrap it as :class:`ContextOverflowError` if appropriate.

    Non-overflow errors are returned unchanged.

    Args:
        error: The original exception.
        model: Optional model name (used only for richer logging; limit is
               extracted from the error message itself if present).

    Returns:
        :class:`ContextOverflowError` wrapping *error*, or *error* as-is.
    """
    if not is_context_overflow(error):
        return error

    msg = str(error)
    model_limit = _extract_model_limit(msg)
    estimated_tokens = _extract_estimated_tokens(msg)

    return ContextOverflowError(
        original=error,
        model_limit=model_limit,
        estimated_tokens=estimated_tokens,
    )
