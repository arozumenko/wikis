"""Unit tests for wiki_structure_planner/_evidence_utils.py (#253, #261).

Tests the consolidated helper functions at the helper level, covering the same
behaviour verified by the per-pack test suites (path traversal, fence masking,
cap enforcement, deduplication) but without requiring cluster/pack fixtures.

Also tests ``parse_frontmatter`` — the unified frontmatter parser that
replaces the three private ``_parse_frontmatter`` implementations that lived
in evidence_md.py, evidence_confluence.py, and evidence_jira.py (#261).

These are fast, no-disk tests except for safe_join path-safety checks
which write a tiny temp tree.
"""

from __future__ import annotations

import pytest

from app.core.parsers.markdown_chunker import Chunk
from app.core.wiki_structure_planner._evidence_utils import (
    _parse_inline_list,
    collect_attachments,
    extract_first_paragraph,
    extract_toc_from_text,
    mask_code_fences,
    parse_frontmatter,
    safe_join,
)

# ── safe_join ─────────────────────────────────────────────────────────────────


class TestSafeJoin:
    def test_normal_relative_path_allowed(self, tmp_path):
        result = safe_join(str(tmp_path), "src/main.py")
        assert result is not None
        assert result.startswith(str(tmp_path))

    def test_traversal_rejected(self, tmp_path):
        result = safe_join(str(tmp_path), "../../etc/passwd")
        assert result is None

    def test_absolute_path_rejected(self, tmp_path):
        result = safe_join(str(tmp_path), "/etc/passwd")
        assert result is None

    def test_dot_path_equals_root_allowed(self, tmp_path):
        # safe_join(root, ".") should return root itself
        result = safe_join(str(tmp_path), ".")
        assert result is not None

    def test_nested_relative_path_allowed(self, tmp_path):
        result = safe_join(str(tmp_path), "a/b/c/d.py")
        assert result is not None
        assert result == str(tmp_path / "a" / "b" / "c" / "d.py")

    def test_traversal_that_resolves_inside_root_allowed(self, tmp_path):
        # a/b/../c.py resolves to a/c.py — still inside root
        result = safe_join(str(tmp_path), "a/b/../c.py")
        assert result is not None
        assert "c.py" in result

    def test_symlink_escape_rejected(self, tmp_path):
        link = tmp_path / "evil.py"
        try:
            link.symlink_to("/etc/passwd")
        except (OSError, NotImplementedError):
            pytest.skip("symlinks not supported on this platform")
        result = safe_join(str(tmp_path), "evil.py")
        # Resolves to /etc/passwd which is outside tmp_path → None
        assert result is None

    def test_returns_none_for_empty_root_with_abs_path(self, tmp_path):
        # An absolute rel_path should always be rejected regardless of root
        result = safe_join(str(tmp_path), "/absolute/path")
        assert result is None


# ── mask_code_fences ──────────────────────────────────────────────────────────


class TestMaskCodeFences:
    def test_preserves_length(self):
        md = "Hello\n```\ncode here\n```\nWorld"
        masked = mask_code_fences(md)
        assert len(masked) == len(md)

    def test_fenced_content_replaced_with_spaces(self):
        md = "Outside\n```bash\n# install\necho hi\n```\nAfter"
        masked = mask_code_fences(md)
        # Non-fence text preserved
        assert "Outside" in masked
        assert "After" in masked
        # Fence body must be blanked — `# install` should not survive as a
        # potential heading match.
        assert "# install" not in masked
        # Offsets are preserved so positions inside the fence are still valid
        # to splice against the original text.
        opening = md.index("```")
        closing = md.rindex("```") + 3
        assert masked[opening:closing].strip() == ""

    def test_unclosed_fence_masks_to_eof(self):
        # LLM-truncated input: fence with no closing.
        md = "Real heading\n```\n# fake heading inside fence\nno close..."
        masked = mask_code_fences(md)
        assert "Real heading" in masked
        # Everything after the opening fence is masked.
        opening = md.index("```")
        assert masked[opening:].strip() == ""

    def test_tilde_fences_handled(self):
        md = "Before\n~~~\n# inside\n~~~\nAfter"
        masked = mask_code_fences(md)
        assert "Before" in masked
        assert "After" in masked
        assert "# inside" not in masked

    def test_no_fences_is_identity(self):
        md = "# Real Heading\n\nProse with no code blocks at all."
        assert mask_code_fences(md) == md


# ── extract_toc_from_text ──────────────────────────────────────────────────────


class TestExtractTocFromText:
    def test_h1_headings_extracted(self):
        md = "# Alpha\n\nSome text.\n\n# Beta\n\nMore text.\n"
        toc = extract_toc_from_text(md)
        assert "Alpha" in toc
        assert "Beta" in toc

    def test_h2_headings_extracted(self):
        md = "# Main\n\n## Sub A\n\nText.\n\n## Sub B\n\nMore.\n"
        toc = extract_toc_from_text(md)
        assert "Main" in toc
        assert "Sub A" in toc
        assert "Sub B" in toc

    def test_h3_and_below_excluded(self):
        md = "# One\n\n## Two\n\n### Three\n\n#### Four\n"
        toc = extract_toc_from_text(md)
        assert "One" in toc
        assert "Two" in toc
        assert "Three" not in toc
        assert "Four" not in toc

    def test_document_order_preserved(self):
        md = "# Alpha\n\n## Beta\n\n# Gamma\n\n## Delta\n"
        toc = extract_toc_from_text(md)
        assert toc.index("Alpha") < toc.index("Beta")
        assert toc.index("Beta") < toc.index("Gamma")
        assert toc.index("Gamma") < toc.index("Delta")

    def test_headings_inside_fenced_block_excluded(self):
        md = "# Real\n\n```bash\n# Not a heading\n## Also not\n```\n\n## Also Real\n"
        toc = extract_toc_from_text(md)
        assert "Real" in toc
        assert "Also Real" in toc
        assert "Not a heading" not in toc
        assert "Also not" not in toc

    def test_tildes_fence_also_masked(self):
        md = "# Before\n\n~~~\n# Inside tilde fence\n~~~\n\n# After\n"
        toc = extract_toc_from_text(md)
        assert "Before" in toc
        assert "After" in toc
        assert "Inside tilde fence" not in toc

    def test_empty_markdown_returns_empty_toc(self):
        assert extract_toc_from_text("") == []

    def test_no_headings_returns_empty_toc(self):
        md = "Just plain prose.\n\nAnother paragraph.\n"
        assert extract_toc_from_text(md) == []

    def test_max_entries_default_is_20(self):
        md = "\n".join(f"# Heading {i}" for i in range(30))
        toc = extract_toc_from_text(md)
        assert len(toc) == 20

    def test_max_entries_configurable(self):
        md = "\n".join(f"# Heading {i}" for i in range(30))
        toc = extract_toc_from_text(md, max_entries=5)
        assert len(toc) == 5

    def test_duplicate_heading_names_preserved(self):
        # Two '## Overview' under different H1s — both must appear.
        md = "# Auth\n\n## Overview\n\n# Storage\n\n## Overview\n"
        toc = extract_toc_from_text(md)
        assert toc.count("Overview") == 2

    def test_trailing_hashes_stripped(self):
        # ATX-style closing hashes: `## Heading ##`
        md = "## Heading ##\n"
        toc = extract_toc_from_text(md)
        assert "Heading" in toc
        # The closing hashes must not appear in the result
        assert not any("##" in h for h in toc)


# ── extract_first_paragraph ───────────────────────────────────────────────────


def _make_chunk(body: str, heading_path: list[str] | None = None, attachments: list[str] | None = None) -> Chunk:
    """Build a minimal Chunk for testing."""
    return Chunk(
        chunk_id="test-0",
        doc_path="test.md",
        heading_path=heading_path or [],
        body=body,
        token_count=len(body) // 4,
        frontmatter=None,
        attachments=set(attachments or []),
    )


class TestExtractFirstParagraph:
    def test_first_non_empty_chunk_returned(self):
        chunks = [_make_chunk(""), _make_chunk("Hello world")]
        result = extract_first_paragraph(chunks)
        assert result == "Hello world"

    def test_whitespace_only_chunk_skipped(self):
        chunks = [_make_chunk("   \n\n"), _make_chunk("Actual content")]
        result = extract_first_paragraph(chunks)
        assert result == "Actual content"

    def test_empty_chunks_returns_empty_string(self):
        result = extract_first_paragraph([])
        assert result == ""

    def test_all_empty_chunks_returns_empty_string(self):
        chunks = [_make_chunk(""), _make_chunk("  ")]
        result = extract_first_paragraph(chunks)
        assert result == ""

    def test_default_cap_200_chars(self):
        long_body = "A" * 500
        chunks = [_make_chunk(long_body)]
        result = extract_first_paragraph(chunks)
        assert len(result) == 200

    def test_custom_cap_applied(self):
        chunks = [_make_chunk("A" * 300)]
        result = extract_first_paragraph(chunks, cap=100)
        assert len(result) == 100

    def test_short_body_not_truncated(self):
        chunks = [_make_chunk("Short text")]
        result = extract_first_paragraph(chunks)
        assert result == "Short text"

    def test_body_stripped_before_cap(self):
        # Leading/trailing whitespace stripped before content check and cap
        chunks = [_make_chunk("  content  ")]
        result = extract_first_paragraph(chunks)
        assert result == "content"


# ── collect_attachments ───────────────────────────────────────────────────────


class TestCollectAttachments:
    def test_single_chunk_attachments_returned(self):
        chunks = [_make_chunk("body", attachments=["diagram.png", "flow.svg"])]
        result = collect_attachments(chunks)
        assert "diagram.png" in result
        assert "flow.svg" in result

    def test_attachments_across_chunks_merged(self):
        chunks = [
            _make_chunk("body1", attachments=["a.png"]),
            _make_chunk("body2", attachments=["b.svg"]),
        ]
        result = collect_attachments(chunks)
        assert "a.png" in result
        assert "b.svg" in result

    def test_duplicate_attachments_deduplicated(self):
        chunks = [
            _make_chunk("body1", attachments=["logo.png"]),
            _make_chunk("body2", attachments=["logo.png"]),
        ]
        result = collect_attachments(chunks)
        assert result.count("logo.png") == 1

    def test_result_is_sorted(self):
        chunks = [_make_chunk("body", attachments=["z.png", "a.png", "m.png"])]
        result = collect_attachments(chunks)
        assert result == sorted(result)

    def test_empty_chunks_returns_empty_list(self):
        result = collect_attachments([])
        assert result == []

    def test_chunks_with_no_attachments_returns_empty(self):
        chunks = [_make_chunk("body with no attachments")]
        result = collect_attachments(chunks)
        assert result == []

    def test_mixed_chunks_some_empty_attachments(self):
        chunks = [
            _make_chunk("body1", attachments=[]),
            _make_chunk("body2", attachments=["photo.jpg"]),
            _make_chunk("body3", attachments=[]),
        ]
        result = collect_attachments(chunks)
        assert result == ["photo.jpg"]


# ── parse_frontmatter ─────────────────────────────────────────────────────────
#
# Tests for the unified frontmatter parser (#261).  Each test covers one
# behavioural axis; the per-caller flag table is:
#
#   evidence_md.py        inline_lists=False  strip_quotes=False
#   evidence_confluence   inline_lists=False  strip_quotes=False
#   evidence_jira.py      inline_lists=True   strip_quotes=True


class TestParseFrontmatterNoFrontmatter:
    """No frontmatter block at all → ({}, original_text)."""

    def test_plain_text_returns_empty_dict_and_original(self):
        text = "# Heading\n\nSome body text.\n"
        fm, body = parse_frontmatter(text, inline_lists=False, strip_quotes=False)
        assert fm == {}
        assert body == text

    def test_empty_string_returns_empty_dict_and_original(self):
        fm, body = parse_frontmatter("", inline_lists=False, strip_quotes=False)
        assert fm == {}
        assert body == ""

    def test_text_starting_with_dashes_but_not_frontmatter(self):
        # A line like "--- something ---" is not a valid frontmatter opener.
        text = "--- not a fence\nSome text.\n"
        fm, body = parse_frontmatter(text, inline_lists=False, strip_quotes=False)
        assert fm == {}
        assert body == text


class TestParseFrontmatterEmpty:
    """Empty frontmatter block ``---\\n---\\n`` → ({}, body)."""

    def test_empty_frontmatter_block(self):
        text = "---\n---\n\nBody text.\n"
        fm, body = parse_frontmatter(text, inline_lists=False, strip_quotes=False)
        assert fm == {}
        assert body == "\nBody text.\n"


class TestParseFrontmatterScalars:
    """Scalar key: value pairs."""

    def test_single_scalar(self):
        text = "---\ntitle: My Document\n---\n\nBody.\n"
        fm, body = parse_frontmatter(text, inline_lists=False, strip_quotes=False)
        assert fm["title"] == "My Document"
        assert body == "\nBody.\n"

    def test_multiple_scalars(self):
        text = "---\ntitle: Doc\nauthor: Alice\nspace_key: ENG\n---\n\nBody.\n"
        fm, body = parse_frontmatter(text, inline_lists=False, strip_quotes=False)
        assert fm["title"] == "Doc"
        assert fm["author"] == "Alice"
        assert fm["space_key"] == "ENG"

    def test_scalar_value_stripped_of_whitespace(self):
        text = "---\ntitle:   Padded Value   \n---\n\nBody.\n"
        fm, _ = parse_frontmatter(text, inline_lists=False, strip_quotes=False)
        assert fm["title"] == "Padded Value"


class TestParseFrontmatterMultiLineLists:
    """Multi-line YAML list syntax (``  - item``)."""

    def test_list_after_empty_value(self):
        text = "---\nlabels:\n  - infra\n  - security\n---\n\nBody.\n"
        fm, _ = parse_frontmatter(text, inline_lists=False, strip_quotes=False)
        assert fm["labels"] == ["infra", "security"]

    def test_list_items_stripped(self):
        text = "---\nlabels:\n  -  padded item  \n---\n\nBody.\n"
        fm, _ = parse_frontmatter(text, inline_lists=False, strip_quotes=False)
        assert fm["labels"] == ["padded item"]

    def test_multiple_lists(self):
        text = "---\nlabels:\n  - a\n  - b\nparents:\n  - X\n  - Y\n---\n\nBody.\n"
        fm, _ = parse_frontmatter(text, inline_lists=False, strip_quotes=False)
        assert fm["labels"] == ["a", "b"]
        assert fm["parents"] == ["X", "Y"]

    def test_mixed_scalars_and_lists(self):
        text = "---\ntitle: My Page\nlabels:\n  - infra\n  - backend\nspace_key: ENG\n---\n\nBody.\n"
        fm, _ = parse_frontmatter(text, inline_lists=False, strip_quotes=False)
        assert fm["title"] == "My Page"
        assert fm["labels"] == ["infra", "backend"]
        assert fm["space_key"] == "ENG"

    def test_any_positive_indent_accepted(self):
        # Tab-indented and deeply-indented items must be accepted.
        text = "---\nlabels:\n\t- tab-item\n      - deep-item\n  - normal\n---\n\nBody.\n"
        fm, _ = parse_frontmatter(text, inline_lists=False, strip_quotes=False)
        assert "tab-item" in fm["labels"]
        assert "deep-item" in fm["labels"]
        assert "normal" in fm["labels"]


class TestParseFrontmatterScalarToListUpgrade:
    """Scalar value followed by ``- item`` lines upgrades to a list.

    Preserves the pre-#261 Confluence/md parser behaviour (Copilot review).
    """

    def test_scalar_then_list_items_upgrades_to_list(self):
        text = "---\nlabels: oldscalar\n  - first\n  - second\n---\n\nBody.\n"
        fm, _ = parse_frontmatter(text, inline_lists=False, strip_quotes=False)
        # Scalar is dropped; list wins.
        assert fm.get("labels") == ["first", "second"]

    def test_scalar_with_no_following_list_stays_scalar(self):
        text = "---\nlabels: just_a_scalar\ntitle: Doc\n---\n\nBody.\n"
        fm, _ = parse_frontmatter(text, inline_lists=False, strip_quotes=False)
        assert fm.get("labels") == "just_a_scalar"
        assert fm.get("title") == "Doc"

    def test_inline_list_does_not_accept_following_items(self):
        # Inline lists are complete on their line — subsequent ``- item``
        # lines are orphaned (current_key cleared after inline parse).
        text = "---\nlabels: ['a', 'b']\n  - c\n---\n\nBody.\n"
        fm, _ = parse_frontmatter(text, inline_lists=True, strip_quotes=True)
        # The inline list wins; "c" is orphaned.
        assert fm.get("labels") == ["a", "b"]


class TestParseFrontmatterBlankLineHandling:
    """Blank line mid-list terminates the list (Rio #252 fix)."""

    def test_blank_line_terminates_list(self):
        # Items "a" and "b" committed before the blank line; "c" after the
        # blank is orphaned (current_key is None after the flush).
        text = "---\nlabels:\n  - a\n  - b\n\n  - c\n---\n\nBody.\n"
        fm, _ = parse_frontmatter(text, inline_lists=False, strip_quotes=False)
        # "a" and "b" must be present; "c" is orphaned and dropped.
        assert "a" in fm["labels"]
        assert "b" in fm["labels"]
        assert "c" not in fm.get("labels", [])

    def test_scalar_after_blank_line_is_new_key(self):
        # After a blank line clears current_key, a new key: value pair should
        # start a fresh entry (not be confused with the previous list).
        text = "---\nlabels:\n  - x\n\ntitle: After Blank\n---\n\nBody.\n"
        fm, _ = parse_frontmatter(text, inline_lists=False, strip_quotes=False)
        assert fm.get("labels") == ["x"]
        assert fm.get("title") == "After Blank"


class TestParseFrontmatterNoClosingFence:
    """No closing ``---`` → ({}, original_text) — graceful fallback."""

    def test_unclosed_frontmatter_returns_empty_dict(self):
        text = "---\ntitle: Broken\nlabels:\n  - item1\n\n# Real Heading\n\nBody.\n"
        fm, body = parse_frontmatter(text, inline_lists=False, strip_quotes=False)
        assert fm == {}
        assert body == text


class TestParseFrontmatterInlineLists:
    """``inline_lists=True`` recognises ``key: [a, "b"]`` syntax."""

    def test_inline_list_false_treats_as_scalar(self):
        # Without inline_lists, ``[a, b]`` is stored as a literal string.
        text = "---\nlabels: ['a', \"b\"]\n---\n\nBody.\n"
        fm, _ = parse_frontmatter(text, inline_lists=False, strip_quotes=False)
        assert isinstance(fm["labels"], str)
        assert fm["labels"] == "['a', \"b\"]"

    def test_inline_list_true_parses_list(self):
        text = "---\nlabels: ['infra', \"security\"]\n---\n\nBody.\n"
        fm, _ = parse_frontmatter(text, inline_lists=True, strip_quotes=True)
        assert fm["labels"] == ["infra", "security"]

    def test_inline_list_bare_items(self):
        text = "---\ncomponents: [api, core, auth]\n---\n\nBody.\n"
        fm, _ = parse_frontmatter(text, inline_lists=True, strip_quotes=False)
        assert fm["components"] == ["api", "core", "auth"]

    def test_inline_list_empty_brackets(self):
        text = "---\nlabels: []\n---\n\nBody.\n"
        fm, _ = parse_frontmatter(text, inline_lists=True, strip_quotes=False)
        assert fm["labels"] == []

    def test_inline_list_comma_inside_quotes_not_split(self):
        # A comma inside a quoted string must NOT split the item.
        text = '---\nlabels: ["payment,billing", security]\n---\n\nBody.\n'
        fm, _ = parse_frontmatter(text, inline_lists=True, strip_quotes=True)
        assert "payment,billing" in fm["labels"]
        assert "security" in fm["labels"]
        assert len(fm["labels"]) == 2


class TestParseFrontmatterStripQuotes:
    """``strip_quotes=True`` strips matching enclosing quotes from scalars."""

    def test_strip_quotes_false_preserves_raw(self):
        text = '---\nsummary: "Quoted Summary"\n---\n\nBody.\n'
        fm, _ = parse_frontmatter(text, inline_lists=False, strip_quotes=False)
        assert fm["summary"] == '"Quoted Summary"'

    def test_strip_quotes_true_removes_double_quotes(self):
        text = '---\nsummary: "Quoted Summary"\n---\n\nBody.\n'
        fm, _ = parse_frontmatter(text, inline_lists=False, strip_quotes=True)
        assert fm["summary"] == "Quoted Summary"

    def test_strip_quotes_true_removes_single_quotes(self):
        text = "---\nstatus: 'In Progress'\n---\n\nBody.\n"
        fm, _ = parse_frontmatter(text, inline_lists=False, strip_quotes=True)
        assert fm["status"] == "In Progress"

    def test_strip_quotes_only_matching_pairs(self):
        # Mismatched quotes are NOT stripped (opening and closing differ).
        text = "---\nstatus: 'mixed\"\n---\n\nBody.\n"
        fm, _ = parse_frontmatter(text, inline_lists=False, strip_quotes=True)
        assert fm["status"] == "'mixed\""

    def test_strip_quotes_multiline_list_items(self):
        text = "---\nlabels:\n  - 'first'\n  - \"second\"\n  - third\n---\n\nBody.\n"
        fm, _ = parse_frontmatter(text, inline_lists=False, strip_quotes=True)
        assert fm["labels"] == ["first", "second", "third"]

    def test_strip_quotes_false_multiline_list_items_preserved(self):
        text = "---\nlabels:\n  - 'quoted'\n  - plain\n---\n\nBody.\n"
        fm, _ = parse_frontmatter(text, inline_lists=False, strip_quotes=False)
        assert fm["labels"] == ["'quoted'", "plain"]


class TestParseFrontmatterJiraMode:
    """Full Jira mode: inline_lists=True, strip_quotes=True."""

    def test_jira_typical_epic_frontmatter(self):
        text = (
            "---\n"
            "summary: Platform Modernisation\n"
            "status: In Progress\n"
            "issue_type: Epic\n"
            "labels:\n"
            "  - backend\n"
            "  - platform\n"
            "components:\n"
            "  - Core API\n"
            "  - Auth Service\n"
            "fix_versions:\n"
            "  - 3.0.0\n"
            "---\n"
            "\n"
            "# Epic Body\n"
        )
        fm, body = parse_frontmatter(text, inline_lists=True, strip_quotes=True)
        assert fm["summary"] == "Platform Modernisation"
        assert fm["status"] == "In Progress"
        assert fm["labels"] == ["backend", "platform"]
        assert fm["components"] == ["Core API", "Auth Service"]
        assert fm["fix_versions"] == ["3.0.0"]
        assert "# Epic Body" in body

    def test_jira_quoted_scalar(self):
        text = "---\nsummary: \"Quoted Summary\"\nstatus: 'In Progress'\nissue_type: epic\n---\n\nBody.\n"
        fm, _ = parse_frontmatter(text, inline_lists=True, strip_quotes=True)
        assert fm["summary"] == "Quoted Summary"
        assert fm["status"] == "In Progress"

    def test_jira_inline_list_with_quoted_items(self):
        text = "---\nlabels: ['infra', \"security\", 'auth']\ncomponents: [api, \"core\"]\n---\n\nBody.\n"
        fm, _ = parse_frontmatter(text, inline_lists=True, strip_quotes=True)
        assert fm["labels"] == ["infra", "security", "auth"]
        assert fm["components"] == ["api", "core"]


# ── _parse_inline_list — quote-aware parser (#256) ────────────────────────────


class TestParseInlineList:
    """Regression tests for #256: comma inside quoted string must not split."""

    def test_comma_inside_double_quotes_preserved(self):
        """AC-1: ["payment,billing", security] → ["payment,billing", "security"]."""
        result = _parse_inline_list('["payment,billing", security]')
        assert result == ["payment,billing", "security"]

    def test_both_quote_types_preserved(self):
        """AC-2: ['a,b', "c,d"] → ["a,b", "c,d"]."""
        result = _parse_inline_list("['a,b', \"c,d\"]")
        assert result == ["a,b", "c,d"]

    def test_unquoted_items_regression(self):
        """AC-3: [a, b, c] (no quotes) → ["a", "b", "c"]."""
        result = _parse_inline_list("[a, b, c]")
        assert result == ["a", "b", "c"]

    def test_mixed_quote_types_both_preserved(self):
        """AC-4: mixed quote types with commas inside are both preserved."""
        result = _parse_inline_list("[\"mixed,quote\", 'comma,inside']")
        assert result == ["mixed,quote", "comma,inside"]

    def test_empty_list(self):
        """AC-5: [] → []."""
        result = _parse_inline_list("[]")
        assert result == []

    def test_not_a_list_returns_none(self):
        """AC-6: non-list input → None."""
        result = _parse_inline_list("not a list")
        assert result is None

    def test_double_quoted_item_with_comma(self):
        """Single item with comma inside double quotes → one-element list."""
        result = _parse_inline_list('["a,b"]')
        assert result == ["a,b"]

    def test_mixed_quoted_and_unquoted(self):
        """Quoted and unquoted items interleaved."""
        result = _parse_inline_list("[\"x,y\", plain, 'z,w']")
        assert result == ["x,y", "plain", "z,w"]

    def test_whitespace_trimmed_from_unquoted(self):
        """Whitespace around unquoted items is stripped."""
        result = _parse_inline_list("[ a ,  b , c ]")
        assert result == ["a", "b", "c"]

    def test_empty_quoted_string_preserved(self):
        """`[""]` — YAML flow sequences can carry empty-string elements."""
        result = _parse_inline_list('[""]')
        assert result == [""]

    def test_empty_quoted_string_among_others(self):
        """Empty-quoted item between two real ones."""
        result = _parse_inline_list('["a", "", "b"]')
        assert result == ["a", "", "b"]

    def test_empty_single_quoted_string_preserved(self):
        """Single-quoted variant of empty string."""
        result = _parse_inline_list("['']")
        assert result == [""]

    def test_bare_empty_slot_dropped(self):
        """`[a, , b]` — unquoted whitespace between commas is not an item."""
        result = _parse_inline_list("[a, , b]")
        assert result == ["a", "b"]

    def test_not_a_list_plain_scalar_returns_none(self):
        """Plain scalar value (no brackets) returns None."""
        result = _parse_inline_list("payment-service")
        assert result is None
