"""Tests for wiki_content_writer/link_resolver.py (#241).

Covers all link-resolution actions: matched, ambiguous, missing.
Class-based pytest grouping.
"""

from __future__ import annotations

import pytest

from app.core.wiki_content_writer.link_resolver import (
    LinkAction,
    LinkResolution,
    PageReport,
    ResolverReport,
    resolve_wikilinks,
)


# ── Helpers ──────────────────────────────────────────────────────────────────


def _single_page(body: str, page_index: list[str]) -> ResolverReport:
    return resolve_wikilinks({"My Page": body}, page_index)


# ── TestMatchedLinks ─────────────────────────────────────────────────────────


class TestMatchedLinks:
    def test_exact_match_kept_unchanged(self):
        pages = {"Page A": "See [[Auth Guide]] for details."}
        index = ["Auth Guide", "API Reference"]
        report = resolve_wikilinks(pages, index)

        page_report = report.pages["Page A"]
        assert page_report.rewritten == "See [[Auth Guide]] for details."
        assert len(page_report.links) == 1
        link = page_report.links[0]
        assert link.action == LinkAction.MATCHED
        assert link.target == "Auth Guide"

    def test_matched_link_produces_no_replacement_text_change(self):
        pages = {"P": "[[API Reference]] is here."}
        index = ["API Reference"]
        report = resolve_wikilinks(pages, index)

        assert report.pages["P"].rewritten == "[[API Reference]] is here."

    def test_multiple_matched_links_in_one_page(self):
        body = "See [[Auth Guide]] and [[API Reference]]."
        pages = {"P": body}
        index = ["Auth Guide", "API Reference"]
        report = resolve_wikilinks(pages, index)

        page_report = report.pages["P"]
        actions = {l.target: l.action for l in page_report.links}
        assert actions["Auth Guide"] == LinkAction.MATCHED
        assert actions["API Reference"] == LinkAction.MATCHED
        assert page_report.rewritten == body


# ── TestMissingLinks ─────────────────────────────────────────────────────────


class TestMissingLinks:
    def test_missing_link_strips_brackets_preserves_text(self):
        pages = {"P": "Refer to [[Bogus Page]] here."}
        index = ["Auth Guide"]
        report = resolve_wikilinks(pages, index)

        page_report = report.pages["P"]
        assert "[[" not in page_report.rewritten
        assert "Bogus Page" in page_report.rewritten
        assert page_report.rewritten == "Refer to Bogus Page here."

    def test_missing_link_action_is_missing(self):
        pages = {"P": "[[Nonexistent]]"}
        index = []
        report = resolve_wikilinks(pages, index)

        link = report.pages["P"].links[0]
        assert link.action == LinkAction.MISSING
        assert link.target == "Nonexistent"

    def test_empty_index_all_links_missing(self):
        body = "See [[Page A]] and [[Page B]]."
        pages = {"P": body}
        report = resolve_wikilinks(pages, [])

        page_report = report.pages["P"]
        assert all(l.action == LinkAction.MISSING for l in page_report.links)
        assert "[[" not in page_report.rewritten
        assert "Page A" in page_report.rewritten
        assert "Page B" in page_report.rewritten

    def test_pipe_display_text_used_as_anchor_when_missing(self):
        # [[Target Page|show this]] with no match → "show this"
        pages = {"P": "Click [[Target Page|show this]] to continue."}
        index = []
        report = resolve_wikilinks(pages, index)

        page_report = report.pages["P"]
        assert "[[" not in page_report.rewritten
        assert "show this" in page_report.rewritten
        link = page_report.links[0]
        assert link.action == LinkAction.MISSING


# ── TestAmbiguousLinks ───────────────────────────────────────────────────────


class TestAmbiguousLinks:
    def test_case_insensitive_match_is_ambiguous(self):
        # "auth guide" doesn't exactly match "Auth Guide" → ambiguous
        pages = {"P": "See [[auth guide]] for details."}
        index = ["Auth Guide"]
        report = resolve_wikilinks(pages, index)

        link = report.pages["P"].links[0]
        assert link.action == LinkAction.AMBIGUOUS
        assert link.best_match == "Auth Guide"

    def test_ambiguous_link_rewrites_to_best_match(self):
        pages = {"P": "See [[auth guide]] for details."}
        index = ["Auth Guide"]
        report = resolve_wikilinks(pages, index)

        rewritten = report.pages["P"].rewritten
        # Link markup kept, target corrected to canonical title
        assert "[[Auth Guide]]" in rewritten

    def test_slug_distance_picks_closest_candidate(self):
        # "api-reference" should slug-match "API Reference" over "API Guide"
        pages = {"P": "[[api-reference]] docs."}
        index = ["API Reference", "API Guide"]
        report = resolve_wikilinks(pages, index)

        link = report.pages["P"].links[0]
        assert link.action == LinkAction.AMBIGUOUS
        assert link.best_match == "API Reference"

    def test_ambiguous_link_records_candidates(self):
        pages = {"P": "[[auth guide]]"}
        index = ["Auth Guide", "Auth Manual"]
        report = resolve_wikilinks(pages, index)

        link = report.pages["P"].links[0]
        assert link.action == LinkAction.AMBIGUOUS
        assert link.best_match is not None


# ── TestMixedLinks ───────────────────────────────────────────────────────────


class TestMixedLinks:
    def test_mixed_matched_ambiguous_missing_in_one_page(self):
        body = (
            "Start [[Auth Guide]] then [[auth guide]] "
            "and finally [[Totally Bogus Page]]."
        )
        pages = {"P": body}
        index = ["Auth Guide", "API Reference"]
        report = resolve_wikilinks(pages, index)

        page_report = report.pages["P"]
        actions = {l.target: l.action for l in page_report.links}

        assert actions["Auth Guide"] == LinkAction.MATCHED
        assert actions["auth guide"] == LinkAction.AMBIGUOUS
        assert actions["Totally Bogus Page"] == LinkAction.MISSING

        rewritten = page_report.rewritten
        # Matched stays as-is
        assert "[[Auth Guide]]" in rewritten
        # Ambiguous rewritten to canonical
        assert "[[Auth Guide]]" in rewritten
        # Missing stripped to plain text
        assert "[[Totally Bogus Page]]" not in rewritten
        assert "Totally Bogus Page" in rewritten

    def test_multiple_pages_each_get_own_report(self):
        pages = {
            "Page 1": "[[Valid Page]]",
            "Page 2": "[[Missing Page]]",
        }
        index = ["Valid Page"]
        report = resolve_wikilinks(pages, index)

        assert report.pages["Page 1"].links[0].action == LinkAction.MATCHED
        assert report.pages["Page 2"].links[0].action == LinkAction.MISSING


# ── TestEdgeCases ────────────────────────────────────────────────────────────


class TestEdgeCases:
    def test_empty_pages_dict_returns_empty_report(self):
        report = resolve_wikilinks({}, ["Some Page"])
        assert report.pages == {}

    def test_page_with_no_wikilinks_unchanged(self):
        body = "No links here, just prose."
        report = resolve_wikilinks({"P": body}, ["Some Page"])
        assert report.pages["P"].rewritten == body
        assert report.pages["P"].links == []

    def test_empty_body_returns_empty_links(self):
        report = resolve_wikilinks({"P": ""}, ["Some Page"])
        assert report.pages["P"].links == []
        assert report.pages["P"].rewritten == ""

    def test_wikilink_inside_code_fence_ignored(self):
        body = "Text.\n```\n[[Not A Link]]\n```\nEnd."
        report = resolve_wikilinks({"P": body}, [])
        # extract_links strips code fences, so this should not appear as a link
        assert report.pages["P"].links == []
        assert report.pages["P"].rewritten == body

    def test_source_prefix_links_ignored(self):
        # [[source/path/to/file]] are code references, not wiki pages
        body = "See [[source/app/main.py]] for details."
        report = resolve_wikilinks({"P": body}, ["source/app/main.py"])
        # extract_links filters source/ links out
        assert report.pages["P"].links == []


# ── TestReportShape ──────────────────────────────────────────────────────────


class TestReportShape:
    def test_report_has_pages_dict(self):
        report = resolve_wikilinks({"P": "[[X]]"}, [])
        assert isinstance(report, ResolverReport)
        assert isinstance(report.pages, dict)

    def test_page_report_has_required_fields(self):
        report = resolve_wikilinks({"P": "[[X]]"}, [])
        page_report = report.pages["P"]
        assert isinstance(page_report, PageReport)
        assert hasattr(page_report, "links")
        assert hasattr(page_report, "rewritten")

    def test_link_resolution_has_required_fields(self):
        report = resolve_wikilinks({"P": "[[X]]"}, [])
        link = report.pages["P"].links[0]
        assert isinstance(link, LinkResolution)
        assert hasattr(link, "target")
        assert hasattr(link, "action")
        assert hasattr(link, "best_match")

    def test_matched_count_in_report(self):
        pages = {"P": "[[A]] and [[B]] and [[C]]."}
        index = ["A", "B"]
        report = resolve_wikilinks(pages, index)
        page = report.pages["P"]
        matched = [l for l in page.links if l.action == LinkAction.MATCHED]
        missing = [l for l in page.links if l.action == LinkAction.MISSING]
        assert len(matched) == 2
        assert len(missing) == 1

    def test_summary_totals_across_pages(self):
        pages = {
            "P1": "[[A]] [[B]]",
            "P2": "[[C]] [[Missing]]",
        }
        index = ["A", "B", "C"]
        report = resolve_wikilinks(pages, index)
        assert report.total_matched == 3
        assert report.total_missing == 1
