"""Tests for the markdown link extractor (#228)."""

from __future__ import annotations

from app.core.parsers.markdown_links import Link, extract_links


class TestInternalLinks:
    def test_simple_relative_link(self):
        links = extract_links("[See more](other.md)")
        assert len(links) == 1
        assert links[0].kind == "internal"
        assert links[0].target == "other.md"
        assert links[0].text == "See more"
        assert links[0].anchor is None

    def test_link_with_anchor(self):
        links = extract_links("[Auth flow](api.md#auth)")
        assert len(links) == 1
        assert links[0].kind == "internal"
        assert links[0].target == "api.md"
        assert links[0].anchor == "auth"

    def test_link_with_subdir(self):
        links = extract_links("[Guide](docs/setup.md)")
        assert links[0].target == "docs/setup.md"

    def test_link_with_parent_dir(self):
        links = extract_links("[Up](../README.md)", source_path="docs/intro.md")
        assert links[0].kind == "internal"
        assert links[0].resolved == "README.md"

    def test_resolved_against_source_path(self):
        links = extract_links("[Sibling](other.md)", source_path="docs/intro.md")
        assert links[0].resolved == "docs/other.md"

    def test_absolute_path_resolution(self):
        links = extract_links("[Root](/README.md)", source_path="docs/x.md")
        assert links[0].kind == "internal"
        assert links[0].resolved == "README.md"

    def test_link_title_attribute_ignored(self):
        links = extract_links('[Click](other.md "tooltip text")')
        assert len(links) == 1
        assert links[0].target == "other.md"

    def test_internal_without_md_extension_still_internal(self):
        links = extract_links("[Doc](some_doc)")
        assert links[0].kind == "internal"
        assert links[0].target == "some_doc"


class TestExternalLinks:
    def test_https_link(self):
        links = extract_links("[GitHub](https://github.com/example/repo)")
        assert links[0].kind == "external"
        assert links[0].target == "https://github.com/example/repo"

    def test_http_link(self):
        links = extract_links("[Site](http://example.com)")
        assert links[0].kind == "external"

    def test_mailto_treated_as_external(self):
        links = extract_links("[Email](mailto:hi@example.com)")
        assert links[0].kind == "external"


class TestWikilinks:
    def test_simple_wikilink(self):
        links = extract_links("See [[Page Title]] for details.")
        assert len(links) == 1
        assert links[0].kind == "wikilink"
        assert links[0].target == "Page Title"
        assert links[0].text == "Page Title"

    def test_wikilink_with_display_text(self):
        links = extract_links("See [[Page Title|the docs]] for details.")
        assert links[0].kind == "wikilink"
        assert links[0].target == "Page Title"
        assert links[0].text == "the docs"

    def test_wikilink_with_anchor(self):
        links = extract_links("[[Page Title#section]]")
        assert links[0].target == "Page Title"
        assert links[0].anchor == "section"


class TestAttachments:
    def test_image_link(self):
        links = extract_links("![alt text](diagram.png)")
        assert links[0].kind == "attachment"
        assert links[0].target == "diagram.png"
        assert links[0].text == "alt text"

    def test_attachment_in_subdir(self):
        links = extract_links("![](attachments/foo.pdf)")
        assert links[0].kind == "attachment"
        assert links[0].target == "attachments/foo.pdf"

    def test_attachment_placeholder_pattern_recognized(self):
        # [[attachment: filename]] is the placeholder format used by markdown chunking (#231).
        links = extract_links("See [[attachment: foo.pdf]] for the spec.")
        assert links[0].kind == "attachment"
        assert links[0].target == "foo.pdf"


class TestCodeBlockExclusion:
    def test_links_in_fenced_block_ignored(self):
        md = """
Real link: [keep](keep.md)

```python
[ignore](ignore.md)
```
"""
        links = extract_links(md)
        targets = {link.target for link in links}
        assert "keep.md" in targets
        assert "ignore.md" not in targets

    def test_links_in_tilde_fenced_block_ignored(self):
        md = """
[keep](keep.md)

~~~
[ignore](ignore.md)
~~~
"""
        targets = {link.target for link in extract_links(md)}
        assert "ignore.md" not in targets

    def test_links_in_inline_code_ignored(self):
        md = "Plain `[ignore](ignore.md)` here but [keep](keep.md) too."
        targets = {link.target for link in extract_links(md)}
        assert "keep.md" in targets
        assert "ignore.md" not in targets

    def test_wikilinks_in_code_blocks_ignored(self):
        md = """
[[Real Page]]

```
[[Fake Page]]
```
"""
        targets = {link.target for link in extract_links(md)}
        assert "Real Page" in targets
        assert "Fake Page" not in targets


class TestConfluenceJiraURLs:
    def test_external_url_with_match_resolves_to_internal(self):
        url = "https://confluence.example.com/display/SPACE/Auth+Guide"
        index = {url: "exported/auth-guide.md"}
        links = extract_links(f"[Auth]({url})", url_to_file_index=index)
        assert links[0].kind == "internal"
        assert links[0].target == "exported/auth-guide.md"
        assert links[0].resolved == "exported/auth-guide.md"

    def test_external_url_no_match_stays_external(self):
        links = extract_links("[External](https://example.com)", url_to_file_index={})
        assert links[0].kind == "external"

    def test_external_url_with_fragment_splits_anchor(self):
        links = extract_links("[Section](https://example.com/page#frag)")
        assert links[0].kind == "external"
        assert links[0].target == "https://example.com/page"
        assert links[0].anchor == "frag"

    def test_url_with_fragment_localized_via_base_url(self):
        base = "https://confluence.example.com/display/SPACE/Auth+Guide"
        index = {base: "exported/auth-guide.md"}
        links = extract_links(f"[Auth]({base}#section)", url_to_file_index=index)
        assert links[0].kind == "internal"
        assert links[0].target == "exported/auth-guide.md"
        assert links[0].anchor == "section"


class TestImageAnchor:
    def test_image_target_strips_fragment(self):
        links = extract_links("![alt](diagram.png#region)")
        assert links[0].kind == "attachment"
        assert links[0].target == "diagram.png"
        assert links[0].anchor == "region"


class TestReferenceStyleLinks:
    def test_full_reference_link(self):
        md = "See [the docs][docs] for details.\n\n[docs]: https://example.com/docs\n"
        links = extract_links(md)
        assert len(links) == 1
        assert links[0].kind == "external"
        assert links[0].target == "https://example.com/docs"
        assert links[0].text == "the docs"

    def test_collapsed_reference_link(self):
        md = "Read the [Auth Guide][] for setup.\n\n[Auth Guide]: auth.md\n"
        links = extract_links(md)
        assert len(links) == 1
        assert links[0].kind == "internal"
        assert links[0].target == "auth.md"

    def test_undefined_reference_ignored(self):
        md = "See [the docs][nowhere] for details.\n"
        assert extract_links(md) == []

    def test_reference_label_case_insensitive(self):
        md = "See [docs][DOCS].\n\n[Docs]: https://example.com\n"
        links = extract_links(md)
        assert len(links) == 1
        assert links[0].target == "https://example.com"


class TestIndentedCodeBlocks:
    def test_indented_block_after_blank_line_ignored(self):
        md = """
Real link: [keep](keep.md)

    [ignore](ignore.md)
    [also-ignore](other.md)

End paragraph.
"""
        targets = {link.target for link in extract_links(md)}
        assert targets == {"keep.md"}

    def test_tab_indented_block_ignored(self):
        md = "Top text.\n\n\t[ignore](ignore.md)\n\nKeep [this](this.md).\n"
        targets = {link.target for link in extract_links(md)}
        assert "ignore.md" not in targets
        assert "this.md" in targets


class TestEdgeCases:
    def test_empty_input(self):
        assert extract_links("") == []

    def test_plain_text_no_links(self):
        assert extract_links("Just some prose with no links.") == []

    def test_multiple_links_same_paragraph(self):
        md = "See [a](a.md) and [b](b.md) and [[Wiki Page]]."
        links = extract_links(md)
        assert len(links) == 3
        kinds = {link.kind for link in links}
        assert kinds == {"internal", "wikilink"}

    def test_link_without_source_path_skips_resolution(self):
        links = extract_links("[Sibling](other.md)")
        assert links[0].kind == "internal"
        assert links[0].resolved is None

    def test_link_dataclass_fields(self):
        link = Link(
            kind="internal",
            target="x.md",
            anchor=None,
            text="x",
            source_path=None,
            resolved=None,
        )
        assert link.kind == "internal"

    def test_links_returned_in_document_order(self):
        md = "First [[Wiki A]], then ![img](a.png), then [link](other.md), then [[Wiki B]]."
        links = extract_links(md)
        kinds_in_order = [link.kind for link in links]
        assert kinds_in_order == ["wikilink", "attachment", "internal", "wikilink"]

    def test_source_code_wikilinks_excluded(self):
        # [[source/...]] is reserved for source-code refs in generated wikis;
        # source markdown should not produce these but if encountered, skip.
        links = extract_links("[[source/foo.py]]")
        assert links == []
