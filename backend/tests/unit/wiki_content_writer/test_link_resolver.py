"""Tests for wiki_content_writer/link_resolver.py (#241).

Covers all link-resolution actions: matched, ambiguous, missing.
Class-based pytest grouping.
"""

from __future__ import annotations

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
        actions = {lr.target: lr.action for lr in page_report.links}
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
        assert all(lr.action == LinkAction.MISSING for lr in page_report.links)
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
        body = "Start [[Auth Guide]] then [[auth guide]] and finally [[Totally Bogus Page]]."
        pages = {"P": body}
        index = ["Auth Guide", "API Reference"]
        report = resolve_wikilinks(pages, index)

        page_report = report.pages["P"]
        actions = {lr.target: lr.action for lr in page_report.links}

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
        matched = [lr for lr in page.links if lr.action == LinkAction.MATCHED]
        missing = [lr for lr in page.links if lr.action == LinkAction.MISSING]
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


# ── TestAmbiguousPreservation (Copilot-flagged) ──────────────────────────────


class TestAmbiguousPreservation:
    def test_pipe_display_preserved_through_ambiguous_rewrite(self):
        # [[target|display]] should retain |display when target is normalised.
        body = "Read [[auth guide|the auth doc]] now."
        report = _single_page(body, ["Auth Guide"])
        rewritten = report.pages["My Page"].rewritten
        assert "[[Auth Guide|the auth doc]]" in rewritten
        assert "the auth doc" in rewritten

    def test_anchor_preserved_through_ambiguous_rewrite(self):
        body = "Jump to [[auth guide#oauth-flow]] section."
        report = _single_page(body, ["Auth Guide"])
        rewritten = report.pages["My Page"].rewritten
        assert "[[Auth Guide#oauth-flow]]" in rewritten

    def test_anchor_and_display_both_preserved(self):
        body = "See [[auth guide#oauth-flow|OAuth section]]."
        report = _single_page(body, ["Auth Guide"])
        rewritten = report.pages["My Page"].rewritten
        assert "[[Auth Guide#oauth-flow|OAuth section]]" in rewritten

    def test_score_field_populated_for_ambiguous(self):
        body = "[[auth gide]]"  # typo
        report = _single_page(body, ["Auth Guide"])
        link = report.pages["My Page"].links[0]
        assert link.action == LinkAction.AMBIGUOUS
        assert 0.0 < link.score <= 1.0


# ── TestDocumentOrder (Copilot-flagged) ──────────────────────────────────────


class TestDocumentOrder:
    def test_resolutions_returned_in_document_order(self):
        body = "First [[Zebra]], then [[Apple]], then [[Banana]]."
        index = ["Zebra", "Apple", "Banana"]
        report = _single_page(body, index)
        targets_in_order = [r.target for r in report.pages["My Page"].links]
        assert targets_in_order == ["Zebra", "Apple", "Banana"]

    def test_duplicates_keep_occurrence_order(self):
        body = "[[A]] [[B]] [[A]] [[C]] [[B]]"
        index = ["A", "B", "C"]
        report = _single_page(body, index)
        targets_in_order = [r.target for r in report.pages["My Page"].links]
        assert targets_in_order == ["A", "B", "A", "C", "B"]


# ── TestCodeFenceBleed (Copilot-flagged) ─────────────────────────────────────


class TestCodeFenceBleed:
    def test_wikilink_in_code_fence_does_not_pull_real_link_into_fence(self):
        # The same target appears in a code block AND in real prose.
        # The rewriter must rewrite only the real-prose occurrence,
        # leaving the code-block text untouched.
        body = "Example:\n```\n[[Auth Guide]]\n```\nReal link: [[auth guide]]."
        report = _single_page(body, ["Auth Guide"])
        rewritten = report.pages["My Page"].rewritten
        # Code block must be unchanged.
        assert "```\n[[Auth Guide]]\n```" in rewritten
        # Real link must be normalised to canonical title.
        assert "Real link: [[Auth Guide]]." in rewritten

    def test_inline_code_wikilink_not_rewritten(self):
        body = "Use `[[foo]]` syntax. Real link: [[Foo]]."
        report = _single_page(body, ["Foo"])
        rewritten = report.pages["My Page"].rewritten
        # Inline-code occurrence stays.
        assert "`[[foo]]`" in rewritten
        # Real-prose occurrence kept as MATCHED.
        assert "[[Foo]]" in rewritten
        assert report.pages["My Page"].links[0].action == LinkAction.MATCHED

    def test_unclosed_fence_does_not_leak_wikilinks(self):
        # Rio 1st-review case: LLM output truncated mid-code-block.
        # The wikilink inside the un-terminated fence must NOT be rewritten.
        body = "Prose [[Real Link]].\n```\n[[Fake Link]]\n(no closing fence)"
        report = _single_page(body, ["Real Link"])
        rewritten = report.pages["My Page"].rewritten
        assert "Prose [[Real Link]]." in rewritten
        assert "[[Fake Link]]" in rewritten  # preserved verbatim

    def test_unclosed_tilde_fence_does_not_leak_wikilinks(self):
        body = "Real [[Foo]].\n~~~\n[[Bar]]\n"  # no closing ~~~
        report = _single_page(body, ["Foo"])
        rewritten = report.pages["My Page"].rewritten
        assert "Real [[Foo]]." in rewritten
        assert "[[Bar]]" in rewritten


# ── TestMissingAnchorReplacement (Rio FYI) ───────────────────────────────────


class TestMissingAnchorReplacement:
    def test_missing_link_with_anchor_uses_clean_target(self):
        # [[Page#section]] missing → "Page" in prose (not "Page#section").
        body = "See [[Deleted Page#overview]] for context."
        report = _single_page(body, [])
        rewritten = report.pages["My Page"].rewritten
        assert "Deleted Page" in rewritten
        assert "#overview" not in rewritten

    def test_missing_link_with_pipe_display_uses_display(self):
        body = "See [[Some Page|the doc]] for context."
        report = _single_page(body, [])
        rewritten = report.pages["My Page"].rewritten
        assert "the doc" in rewritten
        # Target name does not leak into prose when display was provided.
        assert "Some Page" not in rewritten

    def test_missing_link_with_anchor_and_pipe_uses_display(self):
        body = "See [[Gone#section|the gone page]] for context."
        report = _single_page(body, [])
        rewritten = report.pages["My Page"].rewritten
        assert "the gone page" in rewritten
        assert "#section" not in rewritten
