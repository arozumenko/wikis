"""Tests for the incremental-regen helpers in ``app.core.storage.incremental``.

These cover the three pure functions that PR 1 of #116 introduces:
``compute_content_hash``, ``compute_page_id``, ``compute_anchor_slug`` — and
their shared normalization rules. Pure-function tests so they run fast and
without any DB or LLM setup.
"""

from __future__ import annotations

import pytest

from app.core.storage.incremental import (
    PAGE_ID_LENGTH,
    compute_anchor_slug,
    compute_content_hash,
    compute_page_id,
    normalize_for_hash,
)


# ---------------------------------------------------------------------------
# normalize_for_hash
# ---------------------------------------------------------------------------


class TestNormalizeForHash:
    def test_empty_input_returns_empty(self) -> None:
        assert normalize_for_hash("") == ""
        assert normalize_for_hash("\n\n\n") == ""

    def test_trailing_whitespace_is_stripped_per_line(self) -> None:
        # Trailing whitespace differences shouldn't shift the hash.
        assert normalize_for_hash("foo   \nbar  ") == normalize_for_hash("foo\nbar")

    def test_line_endings_are_normalized(self) -> None:
        # CRLF, CR, and LF all collapse to LF — common editor-config drift.
        crlf = "line1\r\nline2\r\nline3"
        cr = "line1\rline2\rline3"
        lf = "line1\nline2\nline3"
        assert normalize_for_hash(crlf) == normalize_for_hash(lf)
        assert normalize_for_hash(cr) == normalize_for_hash(lf)

    def test_leading_and_trailing_blank_lines_dropped(self) -> None:
        assert normalize_for_hash("\n\nfoo\n\n") == "foo"


# ---------------------------------------------------------------------------
# compute_content_hash
# ---------------------------------------------------------------------------


class TestComputeContentHash:
    def test_empty_input_returns_empty_string(self) -> None:
        # Empty input → empty string so callers can store NULL without branching.
        assert compute_content_hash("") == ""

    def test_deterministic_across_calls(self) -> None:
        h1 = compute_content_hash("def foo(): pass")
        h2 = compute_content_hash("def foo(): pass")
        assert h1 == h2

    def test_hex_digest_shape(self) -> None:
        h = compute_content_hash("def foo(): pass")
        assert len(h) == 64  # sha256 → 64 hex chars
        assert all(c in "0123456789abcdef" for c in h)

    def test_whitespace_normalization(self) -> None:
        # Same logical code, different incidental whitespace → same hash.
        assert compute_content_hash("def foo():\n    pass\n") == compute_content_hash(
            "def foo():   \n    pass"
        )

    def test_semantic_change_shifts_hash(self) -> None:
        # Real code edit → hash must change.
        assert compute_content_hash("def foo(): pass") != compute_content_hash(
            "def foo(): return 1"
        )


# ---------------------------------------------------------------------------
# compute_page_id
# ---------------------------------------------------------------------------


class TestComputePageId:
    def test_length(self) -> None:
        page_id = compute_page_id("wiki-1", 0, 1, "node-x", "Title")
        assert len(page_id) == PAGE_ID_LENGTH

    def test_deterministic(self) -> None:
        a = compute_page_id("wiki-1", 0, 1, "node-x", "Title")
        b = compute_page_id("wiki-1", 0, 1, "node-x", "Title")
        assert a == b

    def test_title_casing_does_not_shift_id(self) -> None:
        # Editorial recase ("API" → "Api") shouldn't break wikilinks.
        a = compute_page_id("wiki-1", 0, 1, "node-x", "API Overview")
        b = compute_page_id("wiki-1", 0, 1, "node-x", "api overview")
        assert a == b

    def test_different_inputs_yield_different_ids(self) -> None:
        base = compute_page_id("wiki-1", 0, 1, "node-x", "Title")
        # Each component contributes entropy.
        assert base != compute_page_id("wiki-2", 0, 1, "node-x", "Title")
        assert base != compute_page_id("wiki-1", 9, 1, "node-x", "Title")
        assert base != compute_page_id("wiki-1", 0, 9, "node-x", "Title")
        assert base != compute_page_id("wiki-1", 0, 1, "node-y", "Title")
        assert base != compute_page_id("wiki-1", 0, 1, "node-x", "Different")

    def test_none_clusters_are_accepted(self) -> None:
        # Top-level / un-clustered pages still get stable IDs.
        page_id = compute_page_id("wiki-1", None, None, None, "Overview")
        assert len(page_id) == PAGE_ID_LENGTH


# ---------------------------------------------------------------------------
# compute_anchor_slug
# ---------------------------------------------------------------------------


class TestComputeAnchorSlug:
    def test_basic_slug(self) -> None:
        assert compute_anchor_slug("API Overview") == "api-overview"

    def test_unicode_diacritics_stripped(self) -> None:
        assert compute_anchor_slug("Café résumé") == "cafe-resume"

    def test_runs_of_punctuation_collapse(self) -> None:
        assert compute_anchor_slug("foo!!??bar") == "foo-bar"

    def test_leading_trailing_separators_trimmed(self) -> None:
        assert compute_anchor_slug("---foo---") == "foo"

    def test_empty_title_falls_back_to_page(self) -> None:
        # Pages with effectively-empty titles still need a usable slug.
        assert compute_anchor_slug("") == "page"
        assert compute_anchor_slug("!!!") == "page"

    def test_collision_appends_numeric_suffix(self) -> None:
        existing = {"overview"}
        assert compute_anchor_slug("Overview", existing) == "overview-2"

    def test_collision_skips_taken_suffixes(self) -> None:
        existing = {"overview", "overview-2", "overview-3"}
        assert compute_anchor_slug("Overview", existing) == "overview-4"

    def test_no_collision_when_slug_unique(self) -> None:
        existing = {"other-page"}
        assert compute_anchor_slug("Overview", existing) == "overview"


# ---------------------------------------------------------------------------
# Cross-helper invariants
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "title_a,title_b",
    [
        ("API Overview", "api overview"),  # case
        ("API   Overview", "API Overview"),  # whitespace collapse
    ],
)
def test_page_id_stable_under_title_normalization(title_a: str, title_b: str) -> None:
    """Page IDs must not shift on cosmetic title diffs."""
    assert compute_page_id("w", 0, 0, "s", title_a) == compute_page_id(
        "w", 0, 0, "s", title_b
    )
