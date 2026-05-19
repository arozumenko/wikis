"""Tests for wiki_content_writer/citation_extractor.py (#238).

Covers all citation parsing edge cases:
- Single-line citation [path:N]
- Range citation [path:lo-hi]
- Multiple citations on one claim
- Multiple paragraphs, each cited
- Paragraph with no citation → CitedClaim with empty citations
- Citations inside fenced code blocks → ignored
- Citations inside inline code → ignored
- Malformed citations → gracefully ignored
"""

from __future__ import annotations

from app.core.wiki_content_writer.citation_extractor import (
    Citation,
    CitedClaim,
    extract_citations,
)

# ── Single-claim, single-line citation ──────────────────────────────────────


class TestSingleLineCitation:
    def test_single_line_cite_start_equals_end(self):
        md = "The handler validates the token [api/handler.py:42]."
        result = extract_citations(md)

        assert len(result) == 1
        claim = result[0]
        assert len(claim.citations) == 1
        cite = claim.citations[0]
        assert cite.path == "api/handler.py"
        assert cite.start_line == 42
        assert cite.end_line == 42

    def test_single_line_cite_paragraph_index_zero(self):
        md = "Entry point is here [main.py:1]."
        result = extract_citations(md)

        assert result[0].paragraph_index == 0

    def test_single_line_cite_paragraph_text_preserved(self):
        md = "Entry point is here [main.py:1]."
        result = extract_citations(md)

        assert result[0].paragraph_text == md

    def test_single_line_cite_nested_path(self):
        md = "Config loading happens here [app/config/loader.py:10]."
        result = extract_citations(md)

        cite = result[0].citations[0]
        assert cite.path == "app/config/loader.py"
        assert cite.start_line == 10
        assert cite.end_line == 10


# ── Single-claim, range citation ────────────────────────────────────────────


class TestRangeCitation:
    def test_range_cite_lo_hi_parsed(self):
        md = "The token validation logic runs from setup to finish [api/auth.py:42-58]."
        result = extract_citations(md)

        assert len(result) == 1
        cite = result[0].citations[0]
        assert cite.path == "api/auth.py"
        assert cite.start_line == 42
        assert cite.end_line == 58

    def test_range_cite_start_less_than_end(self):
        md = "See the initializer [models/user.py:100-200]."
        result = extract_citations(md)

        cite = result[0].citations[0]
        assert cite.start_line == 100
        assert cite.end_line == 200

    def test_range_cite_single_digit_lo_hi(self):
        md = "Short function [utils.py:1-5]."
        result = extract_citations(md)

        cite = result[0].citations[0]
        assert cite.start_line == 1
        assert cite.end_line == 5


# ── Multiple citations on one claim ─────────────────────────────────────────


class TestMultipleCitationsOneClaim:
    def test_two_citations_comma_separated(self):
        md = "Validated in handler and model [api/handler.py:10-15, models/item.py:5]."
        result = extract_citations(md)

        assert len(result) == 1
        claim = result[0]
        assert len(claim.citations) == 2

    def test_two_citations_paths_correct(self):
        md = "Cross-module logic [a.py:1-2, b.py:3]."
        result = extract_citations(md)

        cites = result[0].citations
        assert cites[0].path == "a.py"
        assert cites[0].start_line == 1
        assert cites[0].end_line == 2
        assert cites[1].path == "b.py"
        assert cites[1].start_line == 3
        assert cites[1].end_line == 3

    def test_three_citations_all_parsed(self):
        md = "Used throughout [x.py:1, y.py:2-4, z.py:10-20]."
        result = extract_citations(md)

        cites = result[0].citations
        assert len(cites) == 3
        assert cites[0].path == "x.py"
        assert cites[1].path == "y.py"
        assert cites[2].path == "z.py"

    def test_citation_token_with_spaces_around_comma(self):
        """Spaces around the comma separator are allowed."""
        md = "See [a.py:1 , b.py:2]."
        result = extract_citations(md)

        # Both citations should be parsed
        cites = result[0].citations
        assert len(cites) == 2


# ── Multiple paragraphs ──────────────────────────────────────────────────────


class TestMultipleParagraphs:
    def test_two_paragraphs_indexed_correctly(self):
        md = "First claim [a.py:1].\n\nSecond claim [b.py:2]."
        result = extract_citations(md)

        assert len(result) == 2
        assert result[0].paragraph_index == 0
        assert result[1].paragraph_index == 1

    def test_each_paragraph_has_correct_citations(self):
        md = "Auth flow [auth.py:10-20].\n\nToken check [token.py:5]."
        result = extract_citations(md)

        assert result[0].citations[0].path == "auth.py"
        assert result[1].citations[0].path == "token.py"

    def test_three_paragraphs_all_emitted(self):
        md = "Para one [a.py:1].\n\nPara two [b.py:2].\n\nPara three [c.py:3]."
        result = extract_citations(md)

        assert len(result) == 3


# ── Uncited paragraphs ───────────────────────────────────────────────────────


class TestUncitedParagraphs:
    def test_paragraph_with_no_citation_emitted_with_empty_list(self):
        md = "This paragraph has no citation at all."
        result = extract_citations(md)

        assert len(result) == 1
        assert result[0].citations == []

    def test_mixed_cited_and_uncited_paragraphs(self):
        md = "Cited paragraph [a.py:1].\n\nUncited paragraph.\n\nAnother cited [b.py:2]."
        result = extract_citations(md)

        assert len(result) == 3
        assert len(result[0].citations) == 1
        assert len(result[1].citations) == 0
        assert len(result[2].citations) == 1

    def test_all_uncited_returns_claims_with_empty_citations(self):
        md = "First paragraph.\n\nSecond paragraph."
        result = extract_citations(md)

        assert len(result) == 2
        for claim in result:
            assert claim.citations == []


# ── Fenced code block masking ────────────────────────────────────────────────


class TestFencedCodeBlockMasking:
    def test_citation_inside_fenced_block_ignored(self):
        md = "```python\nresult = something()  # [secret.py:99]\n```"
        result = extract_citations(md)

        # The entire input is one "paragraph" but the citation is inside a fence
        assert len(result) == 1
        assert result[0].citations == []

    def test_citation_before_fence_is_extracted_after_is_not(self):
        md = "Real claim [real.py:1].\n\n```\ncode [fake.py:99]\n```"
        result = extract_citations(md)

        # First paragraph has the real citation
        # The fenced block paragraph has no parseable citations
        # Find the paragraph with a citation
        cited = [c for c in result if c.citations]
        assert len(cited) == 1
        assert cited[0].citations[0].path == "real.py"

    def test_tilde_fenced_block_citation_ignored(self):
        md = "~~~\n[inside.py:1]\n~~~"
        result = extract_citations(md)

        assert result[0].citations == []

    def test_citation_after_closed_fence_is_extracted(self):
        md = "```\ncode block\n```\n\nReal claim [real.py:10-20]."
        result = extract_citations(md)

        cited = [c for c in result if c.citations]
        assert len(cited) == 1
        assert cited[0].citations[0].path == "real.py"


# ── Inline code masking ──────────────────────────────────────────────────────


class TestInlineCodeMasking:
    def test_citation_inside_inline_code_ignored(self):
        md = "Call `function [fake.py:5]` to do something."
        result = extract_citations(md)

        assert len(result) == 1
        assert result[0].citations == []

    def test_real_citation_after_inline_code_extracted(self):
        md = "Call `func()` to process data [real.py:42]."
        result = extract_citations(md)

        assert len(result[0].citations) == 1
        assert result[0].citations[0].path == "real.py"

    def test_inline_code_with_citation_like_string_ignored(self):
        md = "Use `[path:1-2]` as format [actual.py:10]."
        result = extract_citations(md)

        # Only the actual citation outside inline code should be parsed
        cites = result[0].citations
        assert len(cites) == 1
        assert cites[0].path == "actual.py"


# ── Malformed citations gracefully ignored ───────────────────────────────────


class TestMalformedCitations:
    def test_no_colon_ignored(self):
        """[foo] — no colon, not a citation."""
        md = "Something [foo] happened."
        result = extract_citations(md)

        assert result[0].citations == []

    def test_non_numeric_line_number_ignored(self):
        """[foo:bar] — line number not numeric."""
        md = "Something [foo:bar] happened."
        result = extract_citations(md)

        assert result[0].citations == []

    def test_triple_range_ignored(self):
        """[foo:1-2-3] — extra dash, ambiguous range."""
        md = "Something [foo:1-2-3] happened."
        result = extract_citations(md)

        assert result[0].citations == []

    def test_empty_path_ignored(self):
        """[:1] — no path."""
        md = "Something [:1] happened."
        result = extract_citations(md)

        assert result[0].citations == []

    def test_negative_line_number_ignored(self):
        """[foo:-1] — negative line not valid."""
        md = "Something [foo:-1] happened."
        result = extract_citations(md)

        assert result[0].citations == []

    def test_zero_line_number_ignored(self):
        """[foo:0] — line numbers are 1-indexed."""
        md = "Something [foo:0] happened."
        result = extract_citations(md)

        assert result[0].citations == []

    def test_valid_and_malformed_in_same_token_parses_valid_only(self):
        """[a.py:1, :bad, b.py:2] — the malformed part is skipped."""
        md = "Claim [a.py:1, :bad, b.py:2]."
        result = extract_citations(md)

        cites = result[0].citations
        # Should have 2 valid citations, malformed one ignored
        assert len(cites) == 2
        paths = {c.path for c in cites}
        assert "a.py" in paths
        assert "b.py" in paths

    def test_plain_markdown_link_not_confused_for_citation(self):
        """[text](url) — standard markdown link, not a citation."""
        md = "See [the docs](https://example.com/docs) for more."
        result = extract_citations(md)

        assert result[0].citations == []


# ── Dataclass structure ──────────────────────────────────────────────────────


class TestDataclassStructure:
    def test_citation_dataclass_fields(self):
        cite = Citation(path="app/main.py", start_line=1, end_line=5)
        assert cite.path == "app/main.py"
        assert cite.start_line == 1
        assert cite.end_line == 5

    def test_cited_claim_dataclass_fields(self):
        claim = CitedClaim(
            paragraph_index=0,
            paragraph_text="some text",
            citations=[Citation("f.py", 1, 1)],
        )
        assert claim.paragraph_index == 0
        assert claim.paragraph_text == "some text"
        assert len(claim.citations) == 1

    def test_extract_citations_returns_list(self):
        result = extract_citations("")
        assert isinstance(result, list)

    def test_empty_string_returns_empty_list(self):
        result = extract_citations("")
        assert result == []

    def test_whitespace_only_returns_empty_list(self):
        result = extract_citations("   \n\n   ")
        assert result == []


# ── Paragraph splitting edge cases ───────────────────────────────────────────


class TestParagraphSplitting:
    def test_blank_line_separates_paragraphs(self):
        md = "Para one.\n\nPara two."
        result = extract_citations(md)

        assert len(result) == 2

    def test_single_newline_does_not_split_paragraph(self):
        """A single newline is a soft wrap — same paragraph."""
        md = "Line one [a.py:1].\nLine two [b.py:2]."
        result = extract_citations(md)

        # Both citations in the same paragraph
        assert len(result) == 1
        assert len(result[0].citations) == 2

    def test_multiple_blank_lines_treated_as_one_split(self):
        """Multiple blank lines between paragraphs is still one split."""
        md = "Para one [a.py:1].\n\n\n\nPara two [b.py:2]."
        result = extract_citations(md)

        assert len(result) == 2

    def test_heading_lines_are_included_in_paragraph(self):
        """Headings are not special-cased — they are part of their paragraph."""
        md = "## Section Heading\n\nContent paragraph [a.py:1]."
        result = extract_citations(md)

        # Two "paragraphs": the heading and the content
        assert len(result) == 2
        # The content paragraph has the citation
        cited = [c for c in result if c.citations]
        assert cited[0].citations[0].path == "a.py"
