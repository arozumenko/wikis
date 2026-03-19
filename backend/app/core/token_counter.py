"""
Token counting utility using tiktoken for accurate context budget management.

Provides accurate token counting for LLM context windows, with fallback to
character-based estimation when tiktoken is unavailable.
"""

import logging
from functools import lru_cache
from typing import Any

logger = logging.getLogger(__name__)

# Try to import tiktoken, fall back to estimation if unavailable
try:
    import tiktoken

    TIKTOKEN_AVAILABLE = True
except ImportError:
    TIKTOKEN_AVAILABLE = False
    logger.warning("tiktoken not available, using character-based token estimation (chars/4)")


# Context budget constants
import os as _os

CONTEXT_TOKEN_BUDGET = int(
    _os.getenv("WIKIS_CONTEXT_TOKEN_BUDGET", "80000")
)  # Max tokens for one-shot generation (full content)
# Legacy tier constants kept for backward compatibility but no longer used
HIERARCHICAL_TIER1_BUDGET = CONTEXT_TOKEN_BUDGET
HIERARCHICAL_TIER2_BUDGET = 0
HIERARCHICAL_TIER3_BUDGET = 0
TOTAL_HIERARCHICAL_BUDGET = CONTEXT_TOKEN_BUDGET


class TokenCounter:
    """
    Token counting using tiktoken for accurate budget management.

    Uses o200k_base encoding (gpt-4o) by default, with graceful fallback
    to character-based estimation (chars/4) when tiktoken is unavailable.

    The encoder is cached for performance - encoding is expensive to initialize.

    Usage:
        counter = TokenCounter()
        tokens = counter.count("Hello world")

        # For documents
        doc_tokens = counter.count_document(doc)

        # Batch counting
        total = counter.count_documents(docs)
    """

    # Default model for tokenization (gpt-4o uses o200k_base)
    DEFAULT_MODEL = "gpt-4o"

    # Fallback ratio when tiktoken unavailable (conservative estimate)
    FALLBACK_CHARS_PER_TOKEN = 4

    def __init__(self, model: str = None):
        """
        Initialize token counter.

        Args:
            model: Model name for tokenizer selection. Defaults to gpt-4o.
                   Supported: gpt-4o, gpt-4, gpt-3.5-turbo, claude-* (falls back to gpt-4o)
        """
        self.model = model or self.DEFAULT_MODEL
        self._encoder = None
        self._using_fallback = False

        if TIKTOKEN_AVAILABLE:
            try:
                self._encoder = self._get_encoder(self.model)
                logger.debug(f"TokenCounter initialized with {self._encoder.name} encoding")
            except Exception as e:
                logger.warning(f"Failed to get encoder for {self.model}: {e}, using fallback")
                self._using_fallback = True
        else:
            self._using_fallback = True

    @staticmethod
    @lru_cache(maxsize=4)
    def _get_encoder(model: str):
        """Get cached encoder for model (expensive to initialize)."""
        try:
            return tiktoken.encoding_for_model(model)
        except KeyError:
            # Model not found, use o200k_base (most recent, works for Claude-like models)
            logger.debug(f"No specific encoder for {model}, using o200k_base")
            return tiktoken.get_encoding("o200k_base")

    def count(self, text: str) -> int:
        """
        Count tokens in text.

        Args:
            text: Text to count tokens for

        Returns:
            Token count (accurate with tiktoken, estimated with fallback)
        """
        if not text:
            return 0

        if self._using_fallback:
            return len(text) // self.FALLBACK_CHARS_PER_TOKEN

        try:
            return len(self._encoder.encode(text))
        except Exception as e:
            logger.warning(f"Token encoding failed: {e}, using fallback")
            return len(text) // self.FALLBACK_CHARS_PER_TOKEN

    def count_document(self, doc: Any, include_metadata: bool = True) -> int:
        """
        Count tokens in a document (page_content + optional metadata overhead).

        Args:
            doc: Document with page_content attribute
            include_metadata: Whether to estimate metadata token overhead

        Returns:
            Token count for the document
        """
        content = getattr(doc, "page_content", "") or ""
        tokens = self.count(content)

        if include_metadata:
            # Estimate metadata overhead (symbol name, file path, hints, etc.)
            metadata = getattr(doc, "metadata", {}) or {}

            # Count key metadata fields that get formatted into context
            metadata_text_parts = []
            for key in [
                "symbol_name",
                "file_path",
                "symbol_type",
                "expansion_reason",
                "expansion_source",
                "expansion_via",
            ]:
                value = metadata.get(key)
                if value:
                    metadata_text_parts.append(str(value))

            if metadata_text_parts:
                metadata_text = " ".join(metadata_text_parts)
                tokens += self.count(metadata_text)
                # Add fixed overhead for formatting (headers, separators, etc.)
                tokens += 20  # ~20 tokens for "### Symbol (type)\n**File**: path\n"

        return tokens

    def count_documents(self, docs: list[Any], include_metadata: bool = True) -> int:
        """
        Count total tokens across multiple documents.

        Args:
            docs: List of documents
            include_metadata: Whether to estimate metadata overhead

        Returns:
            Total token count
        """
        return sum(self.count_document(doc, include_metadata) for doc in docs)

    def fits_budget(self, docs: list[Any], budget: int = CONTEXT_TOKEN_BUDGET) -> bool:
        """
        Check if documents fit within token budget.

        Args:
            docs: List of documents
            budget: Token budget (default: CONTEXT_TOKEN_BUDGET)

        Returns:
            True if total tokens <= budget
        """
        return self.count_documents(docs) <= budget

    def select_docs_for_budget(
        self, docs: list[Any], budget: int = CONTEXT_TOKEN_BUDGET, include_metadata: bool = True
    ) -> tuple:
        """
        Select documents that fit within budget (in order).

        Args:
            docs: List of documents (assumed to be in priority order)
            budget: Token budget
            include_metadata: Whether to count metadata overhead

        Returns:
            Tuple of (selected_docs, remaining_docs, total_tokens)
        """
        selected = []
        remaining = []
        total_tokens = 0

        for doc in docs:
            doc_tokens = self.count_document(doc, include_metadata)

            if total_tokens + doc_tokens <= budget:
                selected.append(doc)
                total_tokens += doc_tokens
            else:
                remaining.append(doc)

        return selected, remaining, total_tokens

    @property
    def is_accurate(self) -> bool:
        """Whether using accurate tiktoken counting (vs fallback estimation)."""
        return not self._using_fallback

    @property
    def encoding_name(self) -> str:
        """Name of the encoding being used."""
        if self._encoder:
            return self._encoder.name
        return f"fallback (chars/{self.FALLBACK_CHARS_PER_TOKEN})"


# Singleton instance for convenience
_default_counter: TokenCounter | None = None


def get_token_counter(model: str = None) -> TokenCounter:
    """
    Get a TokenCounter instance (cached singleton for default model).

    Args:
        model: Model name. If None, returns cached default counter.

    Returns:
        TokenCounter instance
    """
    global _default_counter

    if model is None:
        if _default_counter is None:
            _default_counter = TokenCounter()
        return _default_counter

    return TokenCounter(model)


def count_tokens(text: str) -> int:
    """Convenience function to count tokens using default counter."""
    return get_token_counter().count(text)


def count_document_tokens(doc: Any) -> int:
    """Convenience function to count document tokens using default counter."""
    return get_token_counter().count_document(doc)
