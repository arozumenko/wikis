"""Unit tests for wiki_structure_planner/_evidence_utils.py (#253).

Tests the four consolidated helper functions at the helper level,
covering the same behaviour verified by the per-pack test suites
(path traversal, fence masking, cap enforcement, deduplication)
but without requiring cluster/pack fixtures.

These are fast, no-disk tests except for safe_join path-safety checks
which write a tiny temp tree.
"""

from __future__ import annotations

import pytest

from app.core.parsers.markdown_chunker import Chunk
from app.core.wiki_structure_planner._evidence_utils import (
    collect_attachments,
    extract_first_paragraph,
    extract_toc_from_text,
    mask_code_fences,
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
