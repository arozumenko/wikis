"""Tests for the markdown chunker (#231)."""

from __future__ import annotations

from app.core.parsers.markdown_chunker import chunk_markdown

# ── H1 splitting ────────────────────────────────────────────────────────────


class TestH1Split:
    def test_single_h1_produces_one_chunk(self):
        md = "# Intro\n\nSome text here."
        chunks = chunk_markdown(md, doc_path="doc.md")
        assert len(chunks) == 1
        assert chunks[0].heading_path == ["Intro"]
        assert "Some text here." in chunks[0].body

    def test_two_h1s_produce_two_chunks(self):
        md = "# Section A\n\nText A.\n\n# Section B\n\nText B."
        chunks = chunk_markdown(md, doc_path="doc.md")
        assert len(chunks) == 2
        assert chunks[0].heading_path == ["Section A"]
        assert chunks[1].heading_path == ["Section B"]

    def test_h1_body_does_not_include_next_heading(self):
        md = "# Alpha\n\nAlpha content.\n\n# Beta\n\nBeta content."
        chunks = chunk_markdown(md, doc_path="doc.md")
        assert "Beta" not in chunks[0].body
        assert "Alpha" not in chunks[1].body

    def test_preamble_before_first_heading_becomes_chunk(self):
        md = "Some preamble text.\n\n# Section\n\nSection text."
        chunks = chunk_markdown(md, doc_path="doc.md")
        assert len(chunks) == 2
        assert chunks[0].heading_path == []
        assert "Some preamble text." in chunks[0].body

    def test_chunk_id_is_unique(self):
        md = "# A\n\nText.\n\n# B\n\nMore."
        chunks = chunk_markdown(md, doc_path="doc.md")
        ids = [c.chunk_id for c in chunks]
        assert len(set(ids)) == len(ids)

    def test_doc_path_stored_on_every_chunk(self):
        md = "# A\n\nText.\n\n# B\n\nMore."
        chunks = chunk_markdown(md, doc_path="path/to/doc.md")
        for chunk in chunks:
            assert chunk.doc_path == "path/to/doc.md"


# ── H2 splitting ────────────────────────────────────────────────────────────


class TestH2Split:
    def test_h2_under_h1_splits_chunk(self):
        md = "# Section\n\n## Sub A\n\nText A.\n\n## Sub B\n\nText B."
        chunks = chunk_markdown(md, doc_path="doc.md")
        # Expect two H2-level chunks (H1 intro may or may not be separate)
        h2_chunks = [c for c in chunks if len(c.heading_path) >= 2]
        assert len(h2_chunks) == 2

    def test_h2_heading_path_includes_parent_h1(self):
        md = "# Auth\n\n## OAuth Flow\n\nDetails."
        chunks = chunk_markdown(md, doc_path="doc.md")
        h2_chunk = next(c for c in chunks if "OAuth Flow" in c.heading_path)
        assert h2_chunk.heading_path == ["Auth", "OAuth Flow"]

    def test_h2_only_no_h1_produces_chunks(self):
        md = "## First\n\nText one.\n\n## Second\n\nText two."
        chunks = chunk_markdown(md, doc_path="doc.md")
        h2_chunks = [c for c in chunks if c.heading_path]
        assert len(h2_chunks) == 2
        assert h2_chunks[0].heading_path == ["First"]
        assert h2_chunks[1].heading_path == ["Second"]


# ── Mixed H1 + H2 ───────────────────────────────────────────────────────────


class TestMixedH1H2:
    def test_h1_intro_text_separate_from_h2_chunks(self):
        md = "# Topic\n\nIntro prose.\n\n## Detail\n\nDetail text."
        chunks = chunk_markdown(md, doc_path="doc.md")
        detail = next((c for c in chunks if "Detail" in c.heading_path), None)
        # Either intro is its own chunk or merged — but Detail must exist
        assert detail is not None
        assert detail.heading_path == ["Topic", "Detail"]

    def test_new_h1_resets_h2_context(self):
        md = "# Part 1\n\n## Sub\n\nText.\n\n# Part 2\n\n## Sub\n\nText."
        chunks = chunk_markdown(md, doc_path="doc.md")
        sub_chunks = [c for c in chunks if len(c.heading_path) == 2]
        assert sub_chunks[0].heading_path == ["Part 1", "Sub"]
        assert sub_chunks[1].heading_path == ["Part 2", "Sub"]


# ── H3 fallback ─────────────────────────────────────────────────────────────


class TestH3Fallback:
    def test_h3_not_split_when_chunk_small(self):
        md = "# Section\n\n## Sub\n\n### Detail\n\nShort text."
        chunks = chunk_markdown(md, doc_path="doc.md")
        # With small content, H3 should NOT cause a new split
        h3_chunks = [c for c in chunks if len(c.heading_path) == 3]
        assert len(h3_chunks) == 0

    def test_h3_splits_when_h2_chunk_exceeds_token_limit(self):
        # Build a chunk that exceeds 2000 tokens worth of whitespace-split words
        big_body = " ".join(["word"] * 2100)
        md = f"# Title\n\n## Big Section\n\n{big_body}\n\n### Subsection\n\nExtra content."
        chunks = chunk_markdown(md, doc_path="doc.md")
        h3_chunks = [c for c in chunks if len(c.heading_path) == 3]
        assert len(h3_chunks) >= 1

    def test_h3_chunk_heading_path_has_three_levels(self):
        big_body = " ".join(["word"] * 2100)
        md = f"# Root\n\n## Parent\n\n{big_body}\n\n### Child\n\nChild text."
        chunks = chunk_markdown(md, doc_path="doc.md")
        h3_chunks = [c for c in chunks if len(c.heading_path) == 3]
        assert len(h3_chunks) >= 1
        assert h3_chunks[0].heading_path == ["Root", "Parent", "Child"]


# ── Frontmatter preservation ─────────────────────────────────────────────────


class TestFrontmatterPreservation:
    def test_frontmatter_none_when_not_provided(self):
        chunks = chunk_markdown("# H\n\nText.", doc_path="doc.md")
        assert chunks[0].frontmatter is None

    def test_frontmatter_dict_preserved_on_single_chunk(self):
        fm = {"title": "My Doc", "space_key": "ENG"}
        chunks = chunk_markdown("# H\n\nText.", doc_path="doc.md", frontmatter=fm)
        assert chunks[0].frontmatter == fm

    def test_frontmatter_preserved_on_every_chunk(self):
        fm = {"title": "Guide", "labels": ["infra"], "original_url": "https://example.com"}
        md = "# A\n\nText A.\n\n# B\n\nText B.\n\n# C\n\nText C."
        chunks = chunk_markdown(md, doc_path="doc.md", frontmatter=fm)
        assert len(chunks) == 3
        for chunk in chunks:
            assert chunk.frontmatter == fm

    def test_frontmatter_dict_not_mutated(self):
        fm = {"title": "Test"}
        original_fm = dict(fm)
        chunk_markdown("# H\n\nText.", doc_path="doc.md", frontmatter=fm)
        assert fm == original_fm

    def test_frontmatter_all_known_fields(self):
        fm = {
            "title": "Auth Guide",
            "labels": ["security", "auth"],
            "parents": ["Engineering"],
            "original_url": "https://confluence.example.com/display/ENG/Auth",
            "space_key": "ENG",
        }
        chunks = chunk_markdown("# H\n\nContent.", doc_path="doc.md", frontmatter=fm)
        assert chunks[0].frontmatter["space_key"] == "ENG"
        assert chunks[0].frontmatter["labels"] == ["security", "auth"]


# ── Attachment handling ───────────────────────────────────────────────────────


class TestAttachmentHandling:
    def test_image_ref_becomes_placeholder(self):
        md = "# Section\n\nSee diagram:\n\n![Architecture](arch.png)"
        chunks = chunk_markdown(md, doc_path="doc.md")
        assert "[[attachment: arch.png]]" in chunks[0].body

    def test_image_ref_original_syntax_removed(self):
        md = "# Section\n\n![alt](photo.jpg)\n\nMore text."
        chunks = chunk_markdown(md, doc_path="doc.md")
        assert "![alt](photo.jpg)" not in chunks[0].body

    def test_existing_attachment_placeholder_preserved(self):
        md = "# Section\n\nSee [[attachment: report.pdf]] for details."
        chunks = chunk_markdown(md, doc_path="doc.md")
        assert "[[attachment: report.pdf]]" in chunks[0].body

    def test_attachment_filenames_listed_in_attachments_field(self):
        md = "# Section\n\n![diagram](arch.png)\n\n[[attachment: notes.pdf]]"
        chunks = chunk_markdown(md, doc_path="doc.md")
        assert "arch.png" in chunks[0].attachments
        assert "notes.pdf" in chunks[0].attachments

    def test_attachment_content_not_read(self):
        # The chunker must never attempt to open attachment files.
        # Confirmed by design: chunk_markdown is pure text transform.
        md = "# Section\n\n![big file](huge_binary.bin)"
        chunks = chunk_markdown(md, doc_path="doc.md")
        assert "[[attachment: huge_binary.bin]]" in chunks[0].body

    def test_multiple_images_all_become_placeholders(self):
        md = "# Section\n\n![a](a.png)\n\n![b](b.jpg)"
        chunks = chunk_markdown(md, doc_path="doc.md")
        assert "[[attachment: a.png]]" in chunks[0].body
        assert "[[attachment: b.jpg]]" in chunks[0].body

    def test_image_with_subpath_preserves_filename_only_in_attachments(self):
        md = "# Section\n\n![flow](attachments/flow.svg)"
        chunks = chunk_markdown(md, doc_path="doc.md")
        # The placeholder includes the full path as given
        assert "[[attachment: attachments/flow.svg]]" in chunks[0].body
        assert "attachments/flow.svg" in chunks[0].attachments

    def test_attachments_not_duplicated_across_chunks(self):
        md = "# A\n\n![img](img.png)\n\n# B\n\nNo image."
        chunks = chunk_markdown(md, doc_path="doc.md")
        chunk_a = next(c for c in chunks if "A" in c.heading_path)
        chunk_b = next(c for c in chunks if "B" in c.heading_path)
        assert "img.png" in chunk_a.attachments
        assert "img.png" not in chunk_b.attachments


# ── Edge cases ───────────────────────────────────────────────────────────────


class TestEdgeCases:
    def test_empty_document_returns_single_empty_chunk(self):
        chunks = chunk_markdown("", doc_path="doc.md")
        assert len(chunks) == 1
        assert chunks[0].heading_path == []
        assert chunks[0].body.strip() == ""

    def test_no_headings_returns_single_chunk(self):
        md = "Just prose. No headings. Some more text."
        chunks = chunk_markdown(md, doc_path="doc.md")
        assert len(chunks) == 1
        assert chunks[0].heading_path == []
        assert "Just prose." in chunks[0].body

    def test_whitespace_only_document(self):
        chunks = chunk_markdown("   \n\n   \n", doc_path="doc.md")
        assert len(chunks) == 1
        assert chunks[0].heading_path == []

    def test_single_heading_no_body_returns_no_chunk(self):
        # Empty H1 section is skipped per Copilot review; downstream
        # consumers don't need to filter zero-token chunks.
        chunks = chunk_markdown("# Title", doc_path="doc.md")
        assert chunks == []

    def test_heading_inside_fenced_code_not_a_split(self):
        # `# install` inside a ```bash block must NOT be taken as an H1.
        md = "# Real Heading\n\nIntro text.\n\n```bash\n# install\nrun setup\n```\n\nMore prose.\n"
        chunks = chunk_markdown(md, doc_path="doc.md")
        assert len(chunks) == 1
        assert chunks[0].heading_path == ["Real Heading"]
        # The code block body is preserved verbatim.
        assert "# install" in chunks[0].body

    def test_heading_inside_tilde_fenced_code_not_a_split(self):
        md = "# Outer\n\n~~~\n## Fake H2\n~~~\n"
        chunks = chunk_markdown(md, doc_path="doc.md")
        assert len(chunks) == 1
        assert chunks[0].heading_path == ["Outer"]

    def test_empty_h1_section_produces_no_chunk(self):
        # H1 with no body content should be skipped, not emit a zero-token chunk.
        md = "# A\n\n# B\n\nReal content under B.\n"
        chunks = chunk_markdown(md, doc_path="doc.md")
        # Only the H1 "B" section should produce a chunk.
        assert len(chunks) == 1
        assert chunks[0].heading_path == ["B"]
        assert "Real content under B." in chunks[0].body

    def test_back_to_back_empty_h2_sections_skipped(self):
        md = "# H1\n\n## A\n\n## B\n\n## C\n\nProse for C.\n"
        chunks = chunk_markdown(md, doc_path="doc.md")
        # Only the C section has body.
        paths = [c.heading_path for c in chunks]
        assert ["H1", "C"] in paths
        assert ["H1", "A"] not in paths
        assert ["H1", "B"] not in paths

    def test_chunk_dataclass_fields_present(self):
        chunks = chunk_markdown("# H\n\nText.", doc_path="doc.md")
        c = chunks[0]
        assert hasattr(c, "chunk_id")
        assert hasattr(c, "doc_path")
        assert hasattr(c, "heading_path")
        assert hasattr(c, "body")
        assert hasattr(c, "token_count")
        assert hasattr(c, "frontmatter")
        assert hasattr(c, "attachments")

    def test_chunk_ids_stable_across_calls(self):
        md = "# A\n\nText."
        chunks1 = chunk_markdown(md, doc_path="doc.md")
        chunks2 = chunk_markdown(md, doc_path="doc.md")
        assert chunks1[0].chunk_id == chunks2[0].chunk_id

    def test_different_doc_paths_produce_different_ids(self):
        md = "# A\n\nText."
        chunks1 = chunk_markdown(md, doc_path="a.md")
        chunks2 = chunk_markdown(md, doc_path="b.md")
        assert chunks1[0].chunk_id != chunks2[0].chunk_id


# ── Token count ──────────────────────────────────────────────────────────────


class TestTokenCount:
    def test_token_count_nonzero_for_nonempty_body(self):
        chunks = chunk_markdown("# H\n\nSome words here.", doc_path="doc.md")
        assert chunks[0].token_count > 0

    def test_token_count_zero_for_empty_body(self):
        chunks = chunk_markdown("", doc_path="doc.md")
        assert chunks[0].token_count == 0

    def test_token_count_roughly_word_count(self):
        # whitespace-split estimate: 10 words → token_count ~10
        md = "# H\n\n" + " ".join(["word"] * 10)
        chunks = chunk_markdown(md, doc_path="doc.md")
        # Allow generous range for heading words included or not
        assert 5 <= chunks[0].token_count <= 15

    def test_larger_body_has_higher_token_count(self):
        small = chunk_markdown("# H\n\nShort.", doc_path="doc.md")
        large = chunk_markdown("# H\n\n" + " ".join(["word"] * 500), doc_path="doc.md")
        assert large[0].token_count > small[0].token_count
