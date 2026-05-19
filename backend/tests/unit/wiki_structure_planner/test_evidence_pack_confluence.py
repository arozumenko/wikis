"""Unit tests for wiki_structure_planner/evidence_confluence.py (#234).

Covers:
- Pack construction from fabricated Confluence-exported .md files with YAML frontmatter
- space_key + parent_path propagated from ArtifactInfo
- Labels parsed from frontmatter (single + list forms)
- Heading TOC extracted (H1/H2 only, matches raw-text pattern from evidence_md.py)
- First paragraph captured (≤200 chars)
- Attachment filenames listed (no content read)
- Total token budget respected
- Path safety: reject inputs escaping repo_root
- Missing file / malformed frontmatter handled gracefully
"""

from __future__ import annotations

import textwrap
from pathlib import Path

import pytest

from app.core.wiki_structure_planner.evidence_confluence import (
    ConfluenceEvidencePack,
    build_confluence_pack,
)
from app.core.wiki_structure_planner.structure_skeleton import (
    ArtifactInfo,
    Cluster,
)


# ── Helpers ────────────────────────────────────────────────────────────────────


def _make_confluence_cluster(
    artifacts: list[ArtifactInfo],
    dirs: list[str] | None = None,
    cluster_id: int = 1,
) -> Cluster:
    return Cluster(
        cluster_id=cluster_id,
        kind="confluence",
        dirs=dirs or ["confluence"],
        artifacts=artifacts,
        total_artifacts=len(artifacts),
        primary_languages=[],
        depth_range=(1, 2),
    )


def _confluence_artifact(
    name: str,
    source_path: str,
    space_key: str = "ENG",
    parent_path: str = "",
    summary: str = "",
) -> ArtifactInfo:
    return ArtifactInfo(
        kind="confluence_page",
        name=name,
        source_path=source_path,
        space_key=space_key,
        parent_path=parent_path,
        summary=summary,
    )


def _write_md(tmp_path: Path, rel_path: str, content: str) -> None:
    p = tmp_path / rel_path
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(content)


# ── Sample Confluence-exported markdown with frontmatter ─────────────────────

_SIMPLE_PAGE = textwrap.dedent("""\
    ---
    title: Getting Started
    labels:
      - onboarding
      - infra
    parents:
      - Engineering
    original_url: https://confluence.example.com/pages/123
    space_key: ENG
    ---

    # Getting Started

    This is the introductory paragraph for the guide.

    ## Prerequisites

    You need Python 3.11+.

    ## Installation

    Run `pip install wikis`.
""")

_PAGE_WITH_ATTACHMENTS = textwrap.dedent("""\
    ---
    title: Architecture Diagrams
    labels:
      - architecture
    space_key: ARCH
    ---

    # Architecture

    See the diagram below.

    ![System Diagram](system-diagram.png)

    ## Components

    See also ![Component Map](components.png).
""")

_PAGE_NO_FRONTMATTER = textwrap.dedent("""\
    # No Frontmatter Page

    This page has no YAML frontmatter.

    ## Section A

    Some content here.
""")

_PAGE_SINGLE_LABEL = textwrap.dedent("""\
    ---
    title: Security Policy
    labels: security
    space_key: SEC
    ---

    # Security Policy

    The security policy document.
""")


# ── ConfluenceEvidencePack dataclass ──────────────────────────────────────────


class TestConfluenceEvidencePackDataclass:
    def test_constructs_with_defaults(self):
        pack = ConfluenceEvidencePack(cluster_id=1, kind="confluence")
        assert pack.cluster_id == 1
        assert pack.kind == "confluence"
        assert pack.page_entries == []

    def test_serialize_returns_string(self):
        pack = ConfluenceEvidencePack(cluster_id=1, kind="confluence")
        result = pack.serialize()
        assert isinstance(result, str)

    def test_serialize_empty_pack_carries_cluster_metadata(self):
        pack = ConfluenceEvidencePack(cluster_id=7, kind="confluence")
        result = pack.serialize()
        assert "cluster=7" in result
        assert "kind=confluence" in result

    def test_serialize_includes_page_title(self):
        pack = ConfluenceEvidencePack(
            cluster_id=1,
            kind="confluence",
            page_entries=[
                {
                    "source_path": "confluence/page.md",
                    "title": "My Confluence Page",
                    "space_key": "ENG",
                    "parent_path": "Engineering > Backend",
                    "labels": ["infra"],
                    "toc": ["Overview", "Details"],
                    "first_paragraph": "This page describes...",
                    "attachments": [],
                }
            ],
        )
        result = pack.serialize()
        assert "My Confluence Page" in result

    def test_serialize_includes_space_key(self):
        pack = ConfluenceEvidencePack(
            cluster_id=1,
            kind="confluence",
            page_entries=[
                {
                    "source_path": "confluence/page.md",
                    "title": "Some Page",
                    "space_key": "MYSPC",
                    "parent_path": "",
                    "labels": [],
                    "toc": [],
                    "first_paragraph": "",
                    "attachments": [],
                }
            ],
        )
        result = pack.serialize()
        assert "MYSPC" in result

    def test_serialize_includes_labels(self):
        pack = ConfluenceEvidencePack(
            cluster_id=1,
            kind="confluence",
            page_entries=[
                {
                    "source_path": "confluence/page.md",
                    "title": "Labeled Page",
                    "space_key": "ENG",
                    "parent_path": "",
                    "labels": ["security", "infra"],
                    "toc": [],
                    "first_paragraph": "",
                    "attachments": [],
                }
            ],
        )
        result = pack.serialize()
        assert "security" in result
        assert "infra" in result

    def test_serialize_includes_toc(self):
        pack = ConfluenceEvidencePack(
            cluster_id=1,
            kind="confluence",
            page_entries=[
                {
                    "source_path": "confluence/page.md",
                    "title": "Page With TOC",
                    "space_key": "ENG",
                    "parent_path": "",
                    "labels": [],
                    "toc": ["Overview", "Getting Started"],
                    "first_paragraph": "",
                    "attachments": [],
                }
            ],
        )
        result = pack.serialize()
        assert "Overview" in result
        assert "Getting Started" in result

    def test_serialize_includes_attachments(self):
        pack = ConfluenceEvidencePack(
            cluster_id=1,
            kind="confluence",
            page_entries=[
                {
                    "source_path": "confluence/page.md",
                    "title": "Page",
                    "space_key": "ENG",
                    "parent_path": "",
                    "labels": [],
                    "toc": [],
                    "first_paragraph": "",
                    "attachments": ["diagram.png", "data.csv"],
                }
            ],
        )
        result = pack.serialize()
        assert "diagram.png" in result
        assert "data.csv" in result


# ── build_confluence_pack — basic construction ────────────────────────────────


class TestBuildConfluencePackBasic:
    def test_returns_confluence_evidence_pack(self, tmp_path):
        _write_md(tmp_path, "confluence/page.md", _SIMPLE_PAGE)
        cluster = _make_confluence_cluster(
            [_confluence_artifact("Getting Started", "confluence/page.md")]
        )
        pack = build_confluence_pack(cluster, repo_root=str(tmp_path))
        assert isinstance(pack, ConfluenceEvidencePack)

    def test_pack_cluster_id_matches(self, tmp_path):
        _write_md(tmp_path, "confluence/page.md", _SIMPLE_PAGE)
        cluster = _make_confluence_cluster(
            [_confluence_artifact("Getting Started", "confluence/page.md")],
            cluster_id=42,
        )
        pack = build_confluence_pack(cluster, repo_root=str(tmp_path))
        assert pack.cluster_id == 42

    def test_pack_kind_is_confluence(self, tmp_path):
        _write_md(tmp_path, "confluence/page.md", _SIMPLE_PAGE)
        cluster = _make_confluence_cluster(
            [_confluence_artifact("Getting Started", "confluence/page.md")]
        )
        pack = build_confluence_pack(cluster, repo_root=str(tmp_path))
        assert pack.kind == "confluence"

    def test_empty_cluster_returns_valid_pack(self, tmp_path):
        cluster = _make_confluence_cluster([])
        pack = build_confluence_pack(cluster, repo_root=str(tmp_path))
        assert isinstance(pack, ConfluenceEvidencePack)
        assert pack.page_entries == []

    def test_non_confluence_artifacts_ignored(self, tmp_path):
        """Only confluence_page artifacts are processed; others are silently skipped."""
        _write_md(tmp_path, "docs/guide.md", _SIMPLE_PAGE)
        cluster = Cluster(
            cluster_id=1,
            kind="confluence",
            dirs=["confluence"],
            artifacts=[
                ArtifactInfo(
                    kind="doc_section",
                    name="Guide",
                    source_path="docs/guide.md",
                ),
                ArtifactInfo(
                    kind="symbol",
                    name="MyClass",
                    source_path="src/my_class.py",
                    layer="public_api",
                ),
            ],
            total_artifacts=2,
            primary_languages=[],
            depth_range=(1, 2),
        )
        pack = build_confluence_pack(cluster, repo_root=str(tmp_path))
        assert pack.page_entries == []

    def test_two_pages_in_pack(self, tmp_path):
        _write_md(tmp_path, "confluence/page1.md", _SIMPLE_PAGE)
        _write_md(tmp_path, "confluence/page2.md", _PAGE_SINGLE_LABEL)
        cluster = _make_confluence_cluster(
            [
                _confluence_artifact("Getting Started", "confluence/page1.md"),
                _confluence_artifact("Security Policy", "confluence/page2.md"),
            ]
        )
        pack = build_confluence_pack(cluster, repo_root=str(tmp_path))
        assert len(pack.page_entries) == 2


# ── build_confluence_pack — title extraction ──────────────────────────────────


class TestTitleExtraction:
    def test_title_from_frontmatter(self, tmp_path):
        _write_md(tmp_path, "confluence/page.md", _SIMPLE_PAGE)
        cluster = _make_confluence_cluster(
            [_confluence_artifact("Getting Started", "confluence/page.md")]
        )
        pack = build_confluence_pack(cluster, repo_root=str(tmp_path))
        assert pack.page_entries[0]["title"] == "Getting Started"

    def test_title_falls_back_to_first_h1(self, tmp_path):
        _write_md(tmp_path, "confluence/page.md", _PAGE_NO_FRONTMATTER)
        cluster = _make_confluence_cluster(
            [_confluence_artifact("No Frontmatter Page", "confluence/page.md")]
        )
        pack = build_confluence_pack(cluster, repo_root=str(tmp_path))
        assert pack.page_entries[0]["title"] == "No Frontmatter Page"

    def test_title_falls_back_to_filename(self, tmp_path):
        # Page with no frontmatter title and no H1
        content = "Just some text without headings.\n"
        _write_md(tmp_path, "confluence/my-page.md", content)
        cluster = _make_confluence_cluster(
            [_confluence_artifact("my-page.md", "confluence/my-page.md")]
        )
        pack = build_confluence_pack(cluster, repo_root=str(tmp_path))
        assert pack.page_entries[0]["title"] == "my-page.md"


# ── build_confluence_pack — space_key + parent_path ───────────────────────────


class TestSpaceKeyAndParentPath:
    def test_space_key_propagated_from_artifact(self, tmp_path):
        _write_md(tmp_path, "confluence/page.md", _SIMPLE_PAGE)
        cluster = _make_confluence_cluster(
            [_confluence_artifact("Page", "confluence/page.md", space_key="CUSTOM")]
        )
        pack = build_confluence_pack(cluster, repo_root=str(tmp_path))
        assert pack.page_entries[0]["space_key"] == "CUSTOM"

    def test_parent_path_propagated_from_artifact(self, tmp_path):
        _write_md(tmp_path, "confluence/page.md", _SIMPLE_PAGE)
        cluster = _make_confluence_cluster(
            [
                _confluence_artifact(
                    "Page",
                    "confluence/page.md",
                    parent_path="Engineering > Backend > Services",
                )
            ]
        )
        pack = build_confluence_pack(cluster, repo_root=str(tmp_path))
        assert pack.page_entries[0]["parent_path"] == "Engineering > Backend > Services"

    def test_empty_parent_path_preserved(self, tmp_path):
        _write_md(tmp_path, "confluence/page.md", _SIMPLE_PAGE)
        cluster = _make_confluence_cluster(
            [_confluence_artifact("Page", "confluence/page.md", parent_path="")]
        )
        pack = build_confluence_pack(cluster, repo_root=str(tmp_path))
        assert pack.page_entries[0]["parent_path"] == ""


# ── build_confluence_pack — labels ────────────────────────────────────────────


class TestLabelParsing:
    def test_labels_parsed_as_list_from_frontmatter(self, tmp_path):
        _write_md(tmp_path, "confluence/page.md", _SIMPLE_PAGE)
        cluster = _make_confluence_cluster(
            [_confluence_artifact("Getting Started", "confluence/page.md")]
        )
        pack = build_confluence_pack(cluster, repo_root=str(tmp_path))
        labels = pack.page_entries[0]["labels"]
        assert "onboarding" in labels
        assert "infra" in labels

    def test_single_label_as_scalar_parsed(self, tmp_path):
        """labels: security (not a list) should still parse as ["security"]."""
        _write_md(tmp_path, "confluence/sec.md", _PAGE_SINGLE_LABEL)
        cluster = _make_confluence_cluster(
            [_confluence_artifact("Security Policy", "confluence/sec.md")]
        )
        pack = build_confluence_pack(cluster, repo_root=str(tmp_path))
        labels = pack.page_entries[0]["labels"]
        assert labels == ["security"]

    def test_no_labels_in_frontmatter_returns_empty(self, tmp_path):
        content = textwrap.dedent("""\
            ---
            title: No Labels Page
            space_key: ENG
            ---

            # No Labels Page

            Content here.
        """)
        _write_md(tmp_path, "confluence/page.md", content)
        cluster = _make_confluence_cluster(
            [_confluence_artifact("No Labels Page", "confluence/page.md")]
        )
        pack = build_confluence_pack(cluster, repo_root=str(tmp_path))
        assert pack.page_entries[0]["labels"] == []

    def test_no_frontmatter_returns_empty_labels(self, tmp_path):
        _write_md(tmp_path, "confluence/plain.md", _PAGE_NO_FRONTMATTER)
        cluster = _make_confluence_cluster(
            [_confluence_artifact("No Frontmatter Page", "confluence/plain.md")]
        )
        pack = build_confluence_pack(cluster, repo_root=str(tmp_path))
        assert pack.page_entries[0]["labels"] == []


# ── build_confluence_pack — heading TOC ──────────────────────────────────────


class TestHeadingToc:
    def test_h1_and_h2_headings_in_toc(self, tmp_path):
        _write_md(tmp_path, "confluence/page.md", _SIMPLE_PAGE)
        cluster = _make_confluence_cluster(
            [_confluence_artifact("Getting Started", "confluence/page.md")]
        )
        pack = build_confluence_pack(cluster, repo_root=str(tmp_path))
        toc = pack.page_entries[0]["toc"]
        assert "Getting Started" in toc
        assert "Prerequisites" in toc
        assert "Installation" in toc

    def test_h3_headings_excluded_from_toc(self, tmp_path):
        content = textwrap.dedent("""\
            ---
            title: Deep Page
            space_key: ENG
            ---

            # Main

            ## Section

            ### Subsection

            #### Deep Subsection
        """)
        _write_md(tmp_path, "confluence/deep.md", content)
        cluster = _make_confluence_cluster(
            [_confluence_artifact("Deep Page", "confluence/deep.md")]
        )
        pack = build_confluence_pack(cluster, repo_root=str(tmp_path))
        toc = pack.page_entries[0]["toc"]
        assert "Main" in toc
        assert "Section" in toc
        assert "Subsection" not in toc
        assert "Deep Subsection" not in toc

    def test_code_fence_headings_excluded_from_toc(self, tmp_path):
        content = textwrap.dedent("""\
            ---
            title: Code Page
            space_key: ENG
            ---

            # Real Heading

            ```bash
            # This is a comment, not a heading
            ## Also not a heading
            ```

            ## Another Real Heading
        """)
        _write_md(tmp_path, "confluence/code.md", content)
        cluster = _make_confluence_cluster(
            [_confluence_artifact("Code Page", "confluence/code.md")]
        )
        pack = build_confluence_pack(cluster, repo_root=str(tmp_path))
        toc = pack.page_entries[0]["toc"]
        assert "Real Heading" in toc
        assert "Another Real Heading" in toc
        assert "This is a comment, not a heading" not in toc
        assert "Also not a heading" not in toc

    def test_no_headings_returns_empty_toc(self, tmp_path):
        content = textwrap.dedent("""\
            ---
            title: Flat Page
            space_key: ENG
            ---

            This page has no headings, just plain text.
        """)
        _write_md(tmp_path, "confluence/flat.md", content)
        cluster = _make_confluence_cluster(
            [_confluence_artifact("Flat Page", "confluence/flat.md")]
        )
        pack = build_confluence_pack(cluster, repo_root=str(tmp_path))
        assert pack.page_entries[0]["toc"] == []


# ── build_confluence_pack — first paragraph ───────────────────────────────────


class TestFirstParagraph:
    def test_first_paragraph_extracted(self, tmp_path):
        _write_md(tmp_path, "confluence/page.md", _SIMPLE_PAGE)
        cluster = _make_confluence_cluster(
            [_confluence_artifact("Getting Started", "confluence/page.md")]
        )
        pack = build_confluence_pack(cluster, repo_root=str(tmp_path))
        first_para = pack.page_entries[0]["first_paragraph"]
        assert first_para  # non-empty
        assert "introductory paragraph" in first_para

    def test_first_paragraph_capped_at_200_chars(self, tmp_path):
        long_text = "A" * 500
        content = textwrap.dedent(f"""\
            ---
            title: Long Page
            space_key: ENG
            ---

            # Long Page

            {long_text}
        """)
        _write_md(tmp_path, "confluence/long.md", content)
        cluster = _make_confluence_cluster(
            [_confluence_artifact("Long Page", "confluence/long.md")]
        )
        pack = build_confluence_pack(cluster, repo_root=str(tmp_path))
        first_para = pack.page_entries[0]["first_paragraph"]
        assert len(first_para) <= 200

    def test_empty_page_first_paragraph_is_empty(self, tmp_path):
        content = textwrap.dedent("""\
            ---
            title: Empty Page
            space_key: ENG
            ---
        """)
        _write_md(tmp_path, "confluence/empty.md", content)
        cluster = _make_confluence_cluster(
            [_confluence_artifact("Empty Page", "confluence/empty.md")]
        )
        pack = build_confluence_pack(cluster, repo_root=str(tmp_path))
        # Should not raise; first_paragraph can be empty string
        assert isinstance(pack.page_entries[0]["first_paragraph"], str)


# ── build_confluence_pack — attachments ──────────────────────────────────────


class TestAttachments:
    def test_image_attachment_filenames_collected(self, tmp_path):
        _write_md(tmp_path, "confluence/page.md", _PAGE_WITH_ATTACHMENTS)
        cluster = _make_confluence_cluster(
            [_confluence_artifact("Architecture Diagrams", "confluence/page.md")]
        )
        pack = build_confluence_pack(cluster, repo_root=str(tmp_path))
        attachments = pack.page_entries[0]["attachments"]
        assert "system-diagram.png" in attachments
        assert "components.png" in attachments

    def test_attachment_content_not_read(self, tmp_path):
        """The attachment file should not exist; we only record filenames."""
        _write_md(tmp_path, "confluence/page.md", _PAGE_WITH_ATTACHMENTS)
        # Intentionally do NOT write system-diagram.png or components.png
        cluster = _make_confluence_cluster(
            [_confluence_artifact("Architecture Diagrams", "confluence/page.md")]
        )
        # Should not raise even though attachment files don't exist
        pack = build_confluence_pack(cluster, repo_root=str(tmp_path))
        assert isinstance(pack, ConfluenceEvidencePack)

    def test_no_attachments_returns_empty_list(self, tmp_path):
        _write_md(tmp_path, "confluence/page.md", _SIMPLE_PAGE)
        cluster = _make_confluence_cluster(
            [_confluence_artifact("Getting Started", "confluence/page.md")]
        )
        pack = build_confluence_pack(cluster, repo_root=str(tmp_path))
        assert pack.page_entries[0]["attachments"] == []

    def test_attachments_deduplicated(self, tmp_path):
        content = textwrap.dedent("""\
            ---
            title: Dup Attachments
            space_key: ENG
            ---

            # Dup Attachments

            ![Same](diagram.png)

            ## Section

            ![Same again](diagram.png)
        """)
        _write_md(tmp_path, "confluence/dup.md", content)
        cluster = _make_confluence_cluster(
            [_confluence_artifact("Dup Attachments", "confluence/dup.md")]
        )
        pack = build_confluence_pack(cluster, repo_root=str(tmp_path))
        attachments = pack.page_entries[0]["attachments"]
        assert attachments.count("diagram.png") == 1


# ── build_confluence_pack — token budget ─────────────────────────────────────


class TestTokenBudget:
    def test_budget_respected_with_many_pages(self, tmp_path):
        # Write many pages to force truncation
        for i in range(20):
            content = textwrap.dedent(f"""\
                ---
                title: Page {i}
                labels:
                  - label{i}
                space_key: ENG
                ---

                # Page {i}

                {"This is content for page " * 50}{i}.

                ## Section A

                {"More content here. " * 100}

                ## Section B

                {"Even more content. " * 100}
            """)
            _write_md(tmp_path, f"confluence/page{i}.md", content)

        artifacts = [
            _confluence_artifact(f"Page {i}", f"confluence/page{i}.md")
            for i in range(20)
        ]
        cluster = _make_confluence_cluster(artifacts)
        pack = build_confluence_pack(cluster, repo_root=str(tmp_path), token_budget=1000)
        serialized = pack.serialize()
        # Rough token estimate: 4 chars per token; allow 2x slack
        approx_tokens = len(serialized) / 4
        assert approx_tokens < 1000 * 2

    def test_at_least_one_entry_even_if_over_budget(self, tmp_path):
        # Single huge page — budget cannot be met, but pack must not be empty
        content = textwrap.dedent("""\
            ---
            title: Huge Page
            space_key: ENG
            ---

            # Huge Page

        """) + "## Section\n\nContent.\n" * 500
        _write_md(tmp_path, "confluence/huge.md", content)
        cluster = _make_confluence_cluster(
            [_confluence_artifact("Huge Page", "confluence/huge.md")]
        )
        pack = build_confluence_pack(cluster, repo_root=str(tmp_path), token_budget=100)
        # Must still produce at least one entry
        assert len(pack.page_entries) >= 1


# ── build_confluence_pack — path safety ──────────────────────────────────────


class TestPathSafety:
    def test_traversal_in_source_path_rejected(self, tmp_path):
        cluster = _make_confluence_cluster(
            [_confluence_artifact("Evil", "../../etc/passwd")]
        )
        # Must not raise; traversal path simply skipped
        pack = build_confluence_pack(cluster, repo_root=str(tmp_path))
        source_paths = [e["source_path"] for e in pack.page_entries]
        assert "../../etc/passwd" not in source_paths

    def test_absolute_source_path_rejected(self, tmp_path):
        cluster = _make_confluence_cluster(
            [_confluence_artifact("AbsPath", "/etc/passwd")]
        )
        pack = build_confluence_pack(cluster, repo_root=str(tmp_path))
        source_paths = [e["source_path"] for e in pack.page_entries]
        assert "/etc/passwd" not in source_paths

    def test_symlink_escape_rejected(self, tmp_path):
        link = tmp_path / "confluence" / "evil_link.md"
        link.parent.mkdir(parents=True, exist_ok=True)
        try:
            link.symlink_to("/etc/passwd")
        except (OSError, NotImplementedError):
            pytest.skip("symlinks not supported on this platform")

        cluster = _make_confluence_cluster(
            [_confluence_artifact("Evil", "confluence/evil_link.md")]
        )
        pack = build_confluence_pack(cluster, repo_root=str(tmp_path))
        # If entry was somehow included, it must not contain /etc/passwd content
        for entry in pack.page_entries:
            assert "root:" not in entry.get("first_paragraph", "")


# ── build_confluence_pack — error handling ────────────────────────────────────


class TestErrorHandling:
    def test_missing_file_skipped_gracefully(self, tmp_path):
        """File referenced by artifact does not exist — should not raise."""
        cluster = _make_confluence_cluster(
            [_confluence_artifact("Ghost Page", "confluence/ghost.md")]
        )
        pack = build_confluence_pack(cluster, repo_root=str(tmp_path))
        # Missing file is silently skipped; pack is still valid
        assert isinstance(pack, ConfluenceEvidencePack)

    def test_malformed_frontmatter_handled_gracefully(self, tmp_path):
        """Truncated / malformed YAML frontmatter should not raise."""
        content = textwrap.dedent("""\
            ---
            title: Broken Page
            labels:
              - item1
            # Missing closing ---

            # Broken Page

            Some content.
        """)
        _write_md(tmp_path, "confluence/broken.md", content)
        cluster = _make_confluence_cluster(
            [_confluence_artifact("Broken Page", "confluence/broken.md")]
        )
        pack = build_confluence_pack(cluster, repo_root=str(tmp_path))
        assert isinstance(pack, ConfluenceEvidencePack)

    def test_duplicate_source_paths_deduplicated(self, tmp_path):
        """Same source_path appearing twice in artifacts yields one page entry."""
        _write_md(tmp_path, "confluence/page.md", _SIMPLE_PAGE)
        cluster = _make_confluence_cluster(
            [
                _confluence_artifact("Page Section 1", "confluence/page.md"),
                _confluence_artifact("Page Section 2", "confluence/page.md"),
            ]
        )
        pack = build_confluence_pack(cluster, repo_root=str(tmp_path))
        source_paths = [e["source_path"] for e in pack.page_entries]
        assert source_paths.count("confluence/page.md") == 1

    def test_empty_file_handled(self, tmp_path):
        _write_md(tmp_path, "confluence/empty.md", "")
        cluster = _make_confluence_cluster(
            [_confluence_artifact("Empty", "confluence/empty.md")]
        )
        pack = build_confluence_pack(cluster, repo_root=str(tmp_path))
        assert isinstance(pack, ConfluenceEvidencePack)
