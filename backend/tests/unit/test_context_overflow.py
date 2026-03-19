"""Unit tests for app.services.context_overflow — classification and wrapping."""

from __future__ import annotations

from app.models.events import ErrorEvent, TokenUsageEvent
from app.services.context_overflow import (
    CONTEXT_OVERFLOW_PATTERNS,
    ContextOverflowError,
    classify_and_wrap,
    is_context_overflow,
)

# ---------------------------------------------------------------------------
# is_context_overflow
# ---------------------------------------------------------------------------


class TestIsContextOverflow:
    """is_context_overflow correctly identifies known overflow error messages."""

    def test_openai_maximum_context_length(self):
        err = Exception("This model's maximum context length is 128000 tokens.")
        assert is_context_overflow(err) is True

    def test_openai_context_length_exceeded(self):
        err = Exception("context_length_exceeded: your request used 200000 tokens")
        assert is_context_overflow(err) is True

    def test_anthropic_prompt_too_long(self):
        err = Exception("prompt is too long: 210000 tokens exceeds the limit")
        assert is_context_overflow(err) is True

    def test_anthropic_too_many_tokens(self):
        err = Exception("too many tokens in request")
        assert is_context_overflow(err) is True

    def test_generic_token_limit_exceeded(self):
        err = ValueError("token limit exceeded — reduce input size")
        assert is_context_overflow(err) is True

    def test_generic_context_window_exceeded(self):
        err = RuntimeError("context window exceeded for the current model")
        assert is_context_overflow(err) is True

    def test_reduce_length_message(self):
        err = Exception("Please reduce the length of the messages.")
        assert is_context_overflow(err) is True

    def test_gemini_input_token_limit(self):
        err = Exception("Input token limit exceeded: 1200000 > 1048576")
        assert is_context_overflow(err) is True

    def test_ordinary_runtime_error_not_overflow(self):
        err = RuntimeError("connection refused")
        assert is_context_overflow(err) is False

    def test_key_error_not_overflow(self):
        err = KeyError("some_field")
        assert is_context_overflow(err) is False

    def test_value_error_not_overflow(self):
        err = ValueError("invalid input: must be a string")
        assert is_context_overflow(err) is False

    def test_empty_message_not_overflow(self):
        err = Exception("")
        assert is_context_overflow(err) is False

    def test_request_too_large_without_token_not_overflow(self):
        """Plain HTTP 413 'request too large' should NOT match (false positive guard)."""
        err = Exception("request too large")
        assert is_context_overflow(err) is False

    def test_request_too_large_with_token_is_overflow(self):
        err = Exception("request too large: token count exceeded limit")
        assert is_context_overflow(err) is True

    def test_patterns_list_not_empty(self):
        assert len(CONTEXT_OVERFLOW_PATTERNS) > 0


# ---------------------------------------------------------------------------
# ContextOverflowError
# ---------------------------------------------------------------------------


class TestContextOverflowError:
    """ContextOverflowError carries structured diagnostic fields."""

    def test_basic_construction(self):
        original = Exception("prompt is too long")
        exc = ContextOverflowError(original=original)
        assert exc.original is original
        assert isinstance(exc.suggestions, list)
        assert len(exc.suggestions) > 0

    def test_with_model_limit(self):
        original = Exception("token limit exceeded")
        exc = ContextOverflowError(original=original, model_limit=128000)
        assert exc.model_limit == 128000
        assert "128,000" in str(exc)

    def test_with_estimated_tokens(self):
        original = Exception("token limit exceeded")
        exc = ContextOverflowError(original=original, model_limit=8192, estimated_tokens=12000)
        assert exc.estimated_tokens == 12000
        assert "12,000" in str(exc)

    def test_str_includes_suggestions(self):
        original = Exception("context window exceeded")
        exc = ContextOverflowError(original=original)
        result = str(exc)
        assert "Suggestions:" in result

    def test_custom_suggestions(self):
        original = Exception("token limit exceeded")
        custom = ["Use a smaller model."]
        exc = ContextOverflowError(original=original, suggestions=custom)
        assert exc.suggestions == custom

    def test_default_suggestions_non_empty(self):
        exc = ContextOverflowError(original=Exception("context window exceeded"))
        assert len(exc.DEFAULT_SUGGESTIONS) >= 3

    def test_is_exception_subclass(self):
        exc = ContextOverflowError(original=Exception("overflow"))
        assert isinstance(exc, Exception)


# ---------------------------------------------------------------------------
# classify_and_wrap
# ---------------------------------------------------------------------------


class TestClassifyAndWrap:
    """classify_and_wrap wraps overflow errors and passes through others."""

    def test_wraps_overflow_error(self):
        err = Exception("maximum context length is 128000 tokens")
        result = classify_and_wrap(err)
        assert isinstance(result, ContextOverflowError)
        assert result.original is err

    def test_passes_through_ordinary_error(self):
        err = RuntimeError("network timeout")
        result = classify_and_wrap(err)
        assert result is err

    def test_passes_through_key_error(self):
        err = KeyError("missing_key")
        result = classify_and_wrap(err)
        assert result is err

    def test_extracts_limit_from_message(self):
        err = Exception("maximum context length is 128000 tokens, but you requested 150000")
        result = classify_and_wrap(err)
        assert isinstance(result, ContextOverflowError)
        assert result.model_limit == 128000

    def test_extracts_limit_not_used_count_anthropic(self):
        """Anthropic messages have used count before limit — must extract the limit."""
        err = Exception("prompt is too long: 210000 tokens exceeds the 200000 token limit")
        result = classify_and_wrap(err)
        assert isinstance(result, ContextOverflowError)
        assert result.model_limit == 200000

    def test_model_param_accepted_without_error(self):
        err = Exception("prompt is too long")
        result = classify_and_wrap(err, model="gpt-4")
        assert isinstance(result, ContextOverflowError)

    def test_no_limit_in_message_limit_is_none(self):
        err = Exception("context window exceeded")
        result = classify_and_wrap(err)
        assert isinstance(result, ContextOverflowError)
        assert result.model_limit is None


# ---------------------------------------------------------------------------
# ErrorEvent — new optional fields
# ---------------------------------------------------------------------------


class TestErrorEventExtensions:
    """ErrorEvent serialises correctly with new optional fields."""

    def test_basic_error_event_unchanged(self):
        ev = ErrorEvent(error="something failed")
        assert ev.error == "something failed"
        assert ev.recoverable is False
        assert ev.error_type is None
        assert ev.model_limit is None
        assert ev.suggested_actions is None

    def test_context_overflow_error_event(self):
        ev = ErrorEvent(
            error="context window exceeded",
            recoverable=True,
            error_type="context_overflow",
            model_limit=128000,
            suggested_actions=["Switch to a larger model."],
        )
        assert ev.error_type == "context_overflow"
        assert ev.model_limit == 128000
        assert ev.suggested_actions == ["Switch to a larger model."]
        assert ev.recoverable is True

    def test_error_event_serialisation(self):
        ev = ErrorEvent(
            error="overflow",
            error_type="context_overflow",
            model_limit=8192,
            suggested_actions=["Reduce input."],
        )
        data = ev.model_dump()
        assert data["error_type"] == "context_overflow"
        assert data["model_limit"] == 8192
        assert data["suggested_actions"] == ["Reduce input."]

    def test_error_event_serialises_none_fields(self):
        ev = ErrorEvent(error="generic")
        data = ev.model_dump()
        assert data["error_type"] is None
        assert data["model_limit"] is None
        assert data["suggested_actions"] is None


# ---------------------------------------------------------------------------
# TokenUsageEvent
# ---------------------------------------------------------------------------


class TestTokenUsageEvent:
    """TokenUsageEvent serialises correctly."""

    def test_basic_construction(self):
        ev = TokenUsageEvent(
            page_id="overview",
            page_title="Overview",
            docs_retrieved=50,
            docs_included=30,
            docs_dropped=20,
            tokens_before=90000,
            tokens_after=60000,
            token_budget=80000,
            utilization_pct=75.0,
        )
        assert ev.event == "token_usage"
        assert ev.page_id == "overview"
        assert ev.docs_dropped == 20
        assert ev.utilization_pct == 75.0

    def test_serialisation_includes_all_fields(self):
        ev = TokenUsageEvent(
            page_id="api-reference",
            page_title="API Reference",
            docs_retrieved=10,
            docs_included=10,
            docs_dropped=0,
            tokens_before=40000,
            tokens_after=40000,
            token_budget=80000,
            utilization_pct=50.0,
        )
        data = ev.model_dump()
        assert data["event"] == "token_usage"
        assert data["docs_dropped"] == 0
        assert data["utilization_pct"] == 50.0

    def test_zero_drop_still_valid(self):
        ev = TokenUsageEvent(
            page_id="p1",
            page_title="P1",
            docs_retrieved=5,
            docs_included=5,
            docs_dropped=0,
            tokens_before=10000,
            tokens_after=10000,
            token_budget=80000,
            utilization_pct=12.5,
        )
        assert ev.docs_dropped == 0
        assert ev.utilization_pct == 12.5
