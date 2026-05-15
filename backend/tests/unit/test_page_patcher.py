"""Unit tests for the surgical LLM ``PagePatcher`` (#116 PR 3).

We don't test prompt-quality (an LLM behavior question). We DO test:

* The quality gate accepts low-diff outputs and rejects high-diff ones.
* The patcher falls back gracefully when the LLM errors out or returns
  empty output.
* The "nothing to do" short-circuit (no symbol diffs, no moved paths)
  returns the original content without invoking the LLM.
* The ``compute_diff_ratio`` token-Jaccard distance behaves as
  documented at the boundary values.
"""

from __future__ import annotations

import pytest
from langchain_core.messages import AIMessage

from app.core.agents.page_patcher import (
    DEFAULT_DIFF_THRESHOLD,
    PagePatcher,
    PatchResult,
    SymbolDiff,
    compute_diff_ratio,
)


class _StubLLM:
    """Minimal LangChain-compatible LLM stub that returns a canned output.

    Records every ``invoke`` call so tests can assert prompt assembly.
    """

    def __init__(self, output: str | Exception = "") -> None:
        self._output = output
        self.calls: list[list] = []

    def invoke(self, messages, **_kwargs):  # type: ignore[no-untyped-def]
        self.calls.append(list(messages))
        if isinstance(self._output, Exception):
            raise self._output
        return AIMessage(content=self._output)


# ---------------------------------------------------------------------------
# compute_diff_ratio
# ---------------------------------------------------------------------------


class TestDiffRatio:
    def test_identical_yields_zero(self) -> None:
        assert compute_diff_ratio("foo bar baz", "foo bar baz") == 0.0

    def test_disjoint_yields_one(self) -> None:
        assert compute_diff_ratio("foo bar", "xyz quux") == 1.0

    def test_partial_overlap(self) -> None:
        # {a,b,c} vs {b,c,d} → |∩|=2, |∪|=4, distance = 1 - 2/4 = 0.5
        assert compute_diff_ratio("a b c", "b c d") == 0.5

    def test_both_empty_yields_zero(self) -> None:
        assert compute_diff_ratio("", "") == 0.0

    def test_one_empty_yields_one(self) -> None:
        # Disjoint token sets → max distance.
        assert compute_diff_ratio("foo bar", "") == 1.0


# ---------------------------------------------------------------------------
# PagePatcher quality gate
# ---------------------------------------------------------------------------


def _diff(name: str = "Foo", old: str = "def foo(): pass", new: str = "def foo(): return 1") -> SymbolDiff:
    return SymbolDiff(symbol_name=name, node_id=name.lower(), old_source=old, new_source=new)


class TestQualityGate:
    def test_low_diff_output_is_accepted(self) -> None:
        # LLM returns content that shares most tokens with the original
        # → diff_ratio well below the default threshold (0.6) → accepted.
        original = (
            "# Authentication\n\n"
            "The AuthService coordinates login. It exposes one method, "
            "authenticate, which takes credentials and returns a token."
        )
        patched = (
            "# Authentication\n\n"
            "The AuthService coordinates login. It exposes one method, "
            "authenticate, which takes credentials and returns a JWT."
        )
        patcher = PagePatcher(_StubLLM(patched))
        result = patcher.patch_page(
            page_id="p1",
            page_title="Authentication",
            primary_symbol_id="auth.AuthService",
            current_content=original,
            symbol_diffs=[_diff()],
            moved_paths={},
        )
        assert result.accepted
        assert result.new_content == patched
        assert result.diff_ratio < DEFAULT_DIFF_THRESHOLD

    def test_high_diff_output_is_rejected(self) -> None:
        # LLM rewrote everything → caller should fall back to structural.
        original = "The Foo class coordinates the Bar widget."
        runaway_output = "quux baz wibble wobble whatever entirely different"
        patcher = PagePatcher(_StubLLM(runaway_output))

        result = patcher.patch_page(
            page_id="p1",
            page_title="Foo",
            primary_symbol_id="foo.Foo",
            current_content=original,
            symbol_diffs=[_diff()],
            moved_paths={},
        )

        assert not result.accepted
        assert result.diff_ratio > DEFAULT_DIFF_THRESHOLD
        # Returns original content + original hash so caller can fall back.
        assert result.new_content == original

    def test_custom_threshold_is_honored(self) -> None:
        # Same prompts, but a tighter threshold rejects what default accepts.
        original = "foo bar baz quux"
        patched = "foo bar baz different"  # ~25% diff
        loose = PagePatcher(_StubLLM(patched), diff_threshold=0.5)
        strict = PagePatcher(_StubLLM(patched), diff_threshold=0.1)

        common_kwargs = dict(
            page_id="p1",
            page_title="t",
            primary_symbol_id=None,
            current_content=original,
            symbol_diffs=[_diff()],
            moved_paths={},
        )
        assert loose.patch_page(**common_kwargs).accepted
        assert not strict.patch_page(**common_kwargs).accepted


# ---------------------------------------------------------------------------
# Fail-soft paths
# ---------------------------------------------------------------------------


class TestFailSoft:
    def test_llm_exception_is_caught(self) -> None:
        patcher = PagePatcher(_StubLLM(RuntimeError("provider down")))
        result = patcher.patch_page(
            page_id="p1",
            page_title="t",
            primary_symbol_id=None,
            current_content="orig",
            symbol_diffs=[_diff()],
            moved_paths={},
        )
        assert not result.accepted
        assert "LLM error" in result.reason
        assert result.new_content == "orig"

    def test_empty_llm_output_is_rejected(self) -> None:
        patcher = PagePatcher(_StubLLM("   "))  # whitespace only
        result = patcher.patch_page(
            page_id="p1",
            page_title="t",
            primary_symbol_id=None,
            current_content="orig",
            symbol_diffs=[_diff()],
            moved_paths={},
        )
        assert not result.accepted
        assert result.reason == "empty LLM output"


# ---------------------------------------------------------------------------
# Short-circuit when nothing to patch
# ---------------------------------------------------------------------------


def test_no_diffs_and_no_paths_short_circuits_without_llm_call() -> None:
    stub = _StubLLM("DO NOT CALL ME")
    patcher = PagePatcher(stub)

    result = patcher.patch_page(
        page_id="p1",
        page_title="t",
        primary_symbol_id=None,
        current_content="just prose",
        symbol_diffs=[],
        moved_paths={},
    )

    assert result.accepted
    assert result.new_content == "just prose"
    assert stub.calls == []  # LLM was never invoked


# ---------------------------------------------------------------------------
# Prompt assembly smoke
# ---------------------------------------------------------------------------


def test_prompt_includes_symbol_diffs_and_moved_paths() -> None:
    stub = _StubLLM("foo")  # short output rejected → we just want the call recorded
    patcher = PagePatcher(stub)
    patcher.patch_page(
        page_id="p1",
        page_title="Auth",
        primary_symbol_id="auth.AuthService",
        current_content="orig content",
        symbol_diffs=[_diff("AuthService", old="class A: ...", new="class A: pass")],
        moved_paths={"old.py": "new.py"},
    )
    assert len(stub.calls) == 1
    # Inspect the user message.
    user_msg = stub.calls[0][1].content
    assert "AuthService" in user_msg
    assert "auth.AuthService" in user_msg
    assert "old.py" in user_msg and "new.py" in user_msg
    assert "orig content" in user_msg
