"""Tests for wiki_content_writer/citation_verifier.py (#239).

Covers:
- Paragraph with no citations → citation_missing, no LLM call.
- Paragraph with valid citation that grounds the claim → supported, kept.
- Paragraph with citation that contradicts → absent, stripped.
- Paragraph with partial overlap → partial, kept, warning logged.
- Batch of 3 paragraphs → 1 LLM call, 3 verdicts in order.
- Citation pointing to nonexistent file → strip with citation_missing + reason.
- Citation pointing to out-of-range lines → clamp gracefully + reason notes.
- LLM returns malformed JSON → all paragraphs default to partial (lenient).
- Empty cited_claims → empty result + zero LLM calls.
"""

from __future__ import annotations

import json
import logging
import os
import tempfile
from dataclasses import dataclass
from typing import Any

import pytest

from app.core.wiki_content_writer.citation_extractor import Citation, CitedClaim
from app.core.wiki_content_writer.citation_verifier import (
    VerifyReport,
    VerifyVerdict,
    _extract_json_array,
    verify_citations,
)


# ── FakeLLM ───────────────────────────────────────────────────────────────────


@dataclass
class _FakeMessage:
    content: str


class FakeLLM:
    """Minimal LangChain BaseChatModel stub for tests.

    ``responses`` is a list of strings returned in sequence.  If the list
    is exhausted, the last response is repeated.
    """

    def __init__(self, responses: list[str]) -> None:
        self._responses = responses
        self._call_count = 0

    def invoke(self, messages: Any) -> _FakeMessage:  # noqa: ANN401
        idx = min(self._call_count, len(self._responses) - 1)
        self._call_count += 1
        return _FakeMessage(content=self._responses[idx])

    @property
    def call_count(self) -> int:
        return self._call_count


# ── Helpers ───────────────────────────────────────────────────────────────────


def _claim(
    text: str,
    paragraph_index: int = 0,
    citations: list[Citation] | None = None,
) -> CitedClaim:
    return CitedClaim(
        paragraph_index=paragraph_index,
        paragraph_text=text,
        citations=citations or [],
    )


def _cite(path: str, start: int = 1, end: int = 1) -> Citation:
    return Citation(path=path, start_line=start, end_line=end)


def _make_repo(files: dict[str, str]) -> str:
    """Create a temp directory with the given files and return its path."""
    tmp = tempfile.mkdtemp()
    for rel_path, content in files.items():
        full = os.path.join(tmp, rel_path)
        os.makedirs(os.path.dirname(full), exist_ok=True)
        with open(full, "w") as fh:
            fh.write(content)
    return tmp


def _verdict_json(verdict: str, reason: str = "ok") -> str:
    return json.dumps([{"verdict": verdict, "reason": reason}])


def _batch_verdict_json(items: list[tuple[str, str]]) -> str:
    return json.dumps([{"verdict": v, "reason": r} for v, r in items])


# ── citation_missing (no LLM call) ────────────────────────────────────────────


class TestCitationMissing:
    def test_no_citations_returns_citation_missing_no_llm(self):
        llm = FakeLLM([])
        claim = _claim("This paragraph has no citations.", paragraph_index=0)
        kept, report = verify_citations([claim], llm=llm, repo_root="/tmp")

        assert kept == []
        assert llm.call_count == 0
        assert len(report.verdicts) == 1
        assert report.verdicts[0].verdict == "citation_missing"
        assert report.verdicts[0].citations_checked == 0

    def test_citation_missing_verdict_strips_paragraph(self):
        llm = FakeLLM([])
        claim = _claim("Uncited content.", paragraph_index=0)
        kept, report = verify_citations([claim], llm=llm, repo_root="/tmp")

        assert len(kept) == 0
        assert report.paragraphs_stripped == 1
        assert report.paragraphs_kept == 0

    def test_citation_missing_increments_stripped_count(self):
        llm = FakeLLM([])
        claims = [
            _claim("No citations here.", paragraph_index=0),
            _claim("Also no citations.", paragraph_index=1),
        ]
        kept, report = verify_citations(claims, llm=llm, repo_root="/tmp")

        assert len(kept) == 0
        assert report.paragraphs_stripped == 2
        assert llm.call_count == 0


# ── supported verdict → kept ─────────────────────────────────────────────────


class TestSupportedVerdict:
    def test_supported_verdict_keeps_paragraph(self):
        repo = _make_repo({"api/auth.py": "def validate_token(token):\n    return True\n"})
        llm = FakeLLM([_verdict_json("supported", "Span contains validate_token definition.")])
        claim = _claim(
            "The validate_token function checks the token.",
            paragraph_index=0,
            citations=[_cite("api/auth.py", 1, 2)],
        )
        kept, report = verify_citations([claim], llm=llm, repo_root=repo)

        assert len(kept) == 1
        assert kept[0].paragraph_index == 0
        assert report.verdicts[0].verdict == "supported"
        assert report.paragraphs_kept == 1
        assert report.paragraphs_stripped == 0
        assert llm.call_count == 1

    def test_supported_verdict_llm_call_count_one(self):
        repo = _make_repo({"src/utils.py": "def helper():\n    pass\n"})
        llm = FakeLLM([_verdict_json("supported")])
        claim = _claim(
            "The helper utility is defined here.",
            citations=[_cite("src/utils.py", 1, 2)],
        )
        verify_citations([claim], llm=llm, repo_root=repo)

        assert llm.call_count == 1


# ── absent verdict → stripped ─────────────────────────────────────────────────


class TestAbsentVerdict:
    def test_absent_verdict_strips_paragraph(self):
        repo = _make_repo({"db/models.py": "class User:\n    pass\n"})
        llm = FakeLLM([_verdict_json("absent", "Span contains User class, not the claimed logic.")])
        claim = _claim(
            "The authentication service validates JWT tokens.",
            paragraph_index=0,
            citations=[_cite("db/models.py", 1, 2)],
        )
        kept, report = verify_citations([claim], llm=llm, repo_root=repo)

        assert len(kept) == 0
        assert report.verdicts[0].verdict == "absent"
        assert report.paragraphs_stripped == 1

    def test_absent_verdict_logged_at_info(self, caplog):
        repo = _make_repo({"x.py": "pass\n"})
        llm = FakeLLM([_verdict_json("absent", "No support found.")])
        claim = _claim(
            "The system does Y.",
            paragraph_index=3,
            citations=[_cite("x.py")],
        )
        with caplog.at_level(logging.INFO, logger="app.core.wiki_content_writer.citation_verifier"):
            verify_citations([claim], llm=llm, repo_root=repo)

        assert any("Stripped" in r.message and "3" in r.message for r in caplog.records)


# ── partial verdict → kept + warning ─────────────────────────────────────────


class TestPartialVerdict:
    def test_partial_verdict_keeps_paragraph(self):
        repo = _make_repo({"core/engine.py": "def run():\n    print('running')\n"})
        llm = FakeLLM([_verdict_json("partial", "run() is present but scheduler details absent.")])
        claim = _claim(
            "The engine runs tasks and manages a scheduler.",
            citations=[_cite("core/engine.py", 1, 2)],
        )
        kept, report = verify_citations([claim], llm=llm, repo_root=repo)

        assert len(kept) == 1
        assert report.verdicts[0].verdict == "partial"
        assert report.paragraphs_partial == 1
        assert report.paragraphs_kept == 1

    def test_partial_verdict_increments_partial_count(self):
        repo = _make_repo({"f.py": "x = 1\n"})
        llm = FakeLLM([_verdict_json("partial")])
        claim = _claim("Some partial content.", citations=[_cite("f.py")])
        _, report = verify_citations([claim], llm=llm, repo_root=repo)

        assert report.paragraphs_partial == 1

    def test_partial_verdict_logged_as_warning(self, caplog):
        repo = _make_repo({"g.py": "pass\n"})
        llm = FakeLLM([_verdict_json("partial", "Only half supported.")])
        claim = _claim("Claims X and Y.", citations=[_cite("g.py")])
        with caplog.at_level(logging.WARNING, logger="app.core.wiki_content_writer.citation_verifier"):
            verify_citations([claim], llm=llm, repo_root=repo)

        assert any("Partial" in r.message or "partial" in r.message for r in caplog.records)


# ── Batch processing ─────────────────────────────────────────────────────────


class TestBatchProcessing:
    def test_batch_of_three_single_llm_call(self):
        repo = _make_repo(
            {
                "a.py": "def foo(): pass\n",
                "b.py": "class Bar: pass\n",
                "c.py": "x = 42\n",
            }
        )
        llm = FakeLLM(
            [
                _batch_verdict_json(
                    [
                        ("supported", "a ok"),
                        ("absent", "b contradicts"),
                        ("partial", "c partial"),
                    ]
                )
            ]
        )
        claims = [
            _claim("Foo function defined.", paragraph_index=0, citations=[_cite("a.py")]),
            _claim("Bar class does X.", paragraph_index=1, citations=[_cite("b.py")]),
            _claim("x is set here.", paragraph_index=2, citations=[_cite("c.py")]),
        ]
        kept, report = verify_citations(claims, llm=llm, repo_root=repo, batch_size=4)

        assert llm.call_count == 1
        assert len(report.verdicts) == 3
        # supported → kept; absent → stripped; partial → kept
        assert len(kept) == 2
        kept_indices = {c.paragraph_index for c in kept}
        assert 0 in kept_indices
        assert 2 in kept_indices
        assert 1 not in kept_indices

    def test_batch_size_one_makes_separate_calls(self):
        repo = _make_repo({"a.py": "pass\n", "b.py": "pass\n"})
        llm = FakeLLM([
            _verdict_json("supported"),
            _verdict_json("supported"),
        ])
        claims = [
            _claim("Para A.", paragraph_index=0, citations=[_cite("a.py")]),
            _claim("Para B.", paragraph_index=1, citations=[_cite("b.py")]),
        ]
        verify_citations(claims, llm=llm, repo_root=repo, batch_size=1)

        assert llm.call_count == 2

    def test_verdicts_in_order_across_batches(self):
        repo = _make_repo({"f.py": "pass\n"})
        llm = FakeLLM([
            _verdict_json("supported"),
            _verdict_json("absent"),
        ])
        claims = [
            _claim("P0.", paragraph_index=0, citations=[_cite("f.py")]),
            _claim("P1.", paragraph_index=1, citations=[_cite("f.py")]),
        ]
        _, report = verify_citations(claims, llm=llm, repo_root=repo, batch_size=1)

        assert report.verdicts[0].paragraph_index == 0
        assert report.verdicts[1].paragraph_index == 1


# ── Nonexistent file ─────────────────────────────────────────────────────────


class TestNonexistentFile:
    def test_nonexistent_file_strips_with_citation_missing(self):
        """A citation pointing to a file that does not exist → citation_missing."""
        llm = FakeLLM([])  # should not be called
        claim = _claim(
            "The handler does X.",
            citations=[_cite("does/not/exist.py", 1, 5)],
        )
        # Use a real temp dir so safe_join resolves properly
        repo = _make_repo({})
        kept, report = verify_citations([claim], llm=llm, repo_root=repo)

        assert len(kept) == 0
        verdict = report.verdicts[0]
        assert verdict.verdict == "citation_missing"
        assert "does/not/exist.py" in verdict.reason
        assert llm.call_count == 0

    def test_nonexistent_file_reason_mentions_path(self):
        repo = _make_repo({})
        llm = FakeLLM([])
        claim = _claim(
            "Content.",
            citations=[_cite("missing/file.py")],
        )
        _, report = verify_citations([claim], llm=llm, repo_root=repo)

        assert "missing/file.py" in report.verdicts[0].reason


# ── Out-of-range lines (clamping) ─────────────────────────────────────────────


class TestLineClamping:
    def test_out_of_range_end_line_clamped(self, caplog):
        """end_line beyond file length is clamped; a warning is logged."""
        repo = _make_repo({"short.py": "line1\nline2\n"})  # 2 lines
        llm = FakeLLM([_verdict_json("supported")])
        claim = _claim(
            "short.py content.",
            citations=[_cite("short.py", 1, 999)],  # way beyond EOF
        )
        with caplog.at_level(logging.WARNING, logger="app.core.wiki_content_writer.citation_verifier"):
            kept, report = verify_citations([claim], llm=llm, repo_root=repo)

        # Clamp warning logged
        assert any("clamped" in r.message.lower() for r in caplog.records)
        # Paragraph still processed (not silently dropped)
        assert llm.call_count == 1

    def test_clamped_span_still_provides_content(self):
        """Even with out-of-range end_line, span text is still read."""
        repo = _make_repo({"f.py": "alpha\nbeta\n"})  # 2 lines
        llm = FakeLLM([_verdict_json("supported", "content found")])
        claim = _claim(
            "File content.",
            citations=[_cite("f.py", 1, 100)],
        )
        kept, report = verify_citations([claim], llm=llm, repo_root=repo)

        assert llm.call_count == 1
        assert report.verdicts[0].verdict == "supported"


# ── Malformed LLM JSON ────────────────────────────────────────────────────────


class TestMalformedLLMJson:
    def test_malformed_json_defaults_to_partial(self):
        """If LLM returns garbage JSON, fall back to partial — don't strip."""
        repo = _make_repo({"ok.py": "x = 1\n"})
        llm = FakeLLM(["this is not json at all"])
        claim = _claim("Content.", citations=[_cite("ok.py")])
        kept, report = verify_citations([claim], llm=llm, repo_root=repo)

        assert len(kept) == 1
        assert report.verdicts[0].verdict == "partial"
        assert report.verdicts[0].reason == "verifier-output-malformed"

    def test_malformed_json_batch_all_default_partial(self):
        """Malformed JSON for a 3-paragraph batch → all 3 become partial."""
        repo = _make_repo({"a.py": "pass\n", "b.py": "pass\n", "c.py": "pass\n"})
        llm = FakeLLM(["not valid json"])
        claims = [
            _claim("P0.", paragraph_index=0, citations=[_cite("a.py")]),
            _claim("P1.", paragraph_index=1, citations=[_cite("b.py")]),
            _claim("P2.", paragraph_index=2, citations=[_cite("c.py")]),
        ]
        kept, report = verify_citations(claims, llm=llm, repo_root=repo, batch_size=4)

        assert len(kept) == 3
        for v in report.verdicts:
            assert v.verdict == "partial"
            assert v.reason == "verifier-output-malformed"

    def test_wrong_count_verdicts_defaults_extras_to_partial(self):
        """LLM returns N-1 verdicts for N paragraphs → missing ones become partial."""
        repo = _make_repo({"a.py": "pass\n", "b.py": "pass\n"})
        # Only 1 verdict for 2 paragraphs
        llm = FakeLLM([json.dumps([{"verdict": "supported", "reason": "ok"}])])
        claims = [
            _claim("P0.", paragraph_index=0, citations=[_cite("a.py")]),
            _claim("P1.", paragraph_index=1, citations=[_cite("b.py")]),
        ]
        kept, report = verify_citations(claims, llm=llm, repo_root=repo, batch_size=4)

        verdicts_by_idx = {v.paragraph_index: v for v in report.verdicts}
        assert verdicts_by_idx[0].verdict == "supported"
        assert verdicts_by_idx[1].verdict == "partial"
        assert verdicts_by_idx[1].reason == "verifier-output-malformed"


# ── Empty input ───────────────────────────────────────────────────────────────


class TestEmptyInput:
    def test_empty_claims_returns_empty_no_llm(self):
        llm = FakeLLM([])
        kept, report = verify_citations([], llm=llm, repo_root="/tmp")

        assert kept == []
        assert llm.call_count == 0
        assert report.llm_calls == 0
        assert report.paragraphs_total == 0
        assert report.verdicts == []

    def test_empty_returns_verify_report(self):
        llm = FakeLLM([])
        _, report = verify_citations([], llm=llm, repo_root="/tmp")

        assert isinstance(report, VerifyReport)


# ── Path safety (traversal rejection) ────────────────────────────────────────


class TestPathSafety:
    def test_traversal_path_treated_as_citation_missing(self):
        """A citation with ../.. traversal must not read outside repo_root."""
        repo = _make_repo({})
        llm = FakeLLM([])
        claim = _claim(
            "Some claim.",
            citations=[_cite("../../etc/passwd", 1, 1)],
        )
        kept, report = verify_citations([claim], llm=llm, repo_root=repo)

        assert len(kept) == 0
        assert report.verdicts[0].verdict == "citation_missing"
        assert llm.call_count == 0

    def test_absolute_path_treated_as_citation_missing(self):
        """An absolute path citation must not be followed."""
        repo = _make_repo({})
        llm = FakeLLM([])
        claim = _claim(
            "Some claim.",
            citations=[_cite("/etc/passwd", 1, 1)],
        )
        kept, report = verify_citations([claim], llm=llm, repo_root=repo)

        assert len(kept) == 0
        assert report.verdicts[0].verdict == "citation_missing"


# ── VerifyReport shape ────────────────────────────────────────────────────────


class TestVerifyReportShape:
    def test_report_fields_present(self):
        llm = FakeLLM([])
        _, report = verify_citations([], llm=llm, repo_root="/tmp")

        assert hasattr(report, "verdicts")
        assert hasattr(report, "paragraphs_total")
        assert hasattr(report, "paragraphs_kept")
        assert hasattr(report, "paragraphs_stripped")
        assert hasattr(report, "paragraphs_partial")
        assert hasattr(report, "llm_calls")

    def test_llm_calls_counted_correctly(self):
        repo = _make_repo({"a.py": "x\n", "b.py": "y\n", "c.py": "z\n"})
        llm = FakeLLM([
            _batch_verdict_json([("supported", "ok"), ("supported", "ok")]),
            _verdict_json("supported"),
        ])
        claims = [
            _claim("P0.", paragraph_index=0, citations=[_cite("a.py")]),
            _claim("P1.", paragraph_index=1, citations=[_cite("b.py")]),
            _claim("P2.", paragraph_index=2, citations=[_cite("c.py")]),
        ]
        _, report = verify_citations(claims, llm=llm, repo_root=repo, batch_size=2)

        assert report.llm_calls == 2


# ── _extract_json_array unit tests ────────────────────────────────────────────


class TestExtractJsonArray:
    def test_parses_clean_array(self):
        result = _extract_json_array('[{"verdict": "supported", "reason": "ok"}]')
        assert result is not None
        assert result[0]["verdict"] == "supported"

    def test_strips_markdown_fences(self):
        result = _extract_json_array('```json\n[{"verdict": "absent"}]\n```')
        assert result is not None
        assert result[0]["verdict"] == "absent"

    def test_trailing_comma_cleanup(self):
        result = _extract_json_array('[{"verdict": "partial",}]')
        assert result is not None
        assert result[0]["verdict"] == "partial"

    def test_returns_none_on_object(self):
        result = _extract_json_array('{"verdict": "supported"}')
        # object not array → None
        assert result is None

    def test_returns_none_on_garbage(self):
        result = _extract_json_array("this is not json")
        assert result is None

    def test_extracts_embedded_array(self):
        result = _extract_json_array('Some preamble\n[{"verdict": "supported"}]\nMore text')
        assert result is not None
        assert len(result) == 1

    def test_empty_array(self):
        result = _extract_json_array("[]")
        assert result == []
