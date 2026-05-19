"""Unit tests for wiki_structure_planner/evidence_md.py — markdown cluster evidence packs (#233).

Covers:
- Pack construction from a small fabricated doc cluster (2-3 .md files on disk)
- Heading TOC extracted (H1/H2 only, top 2 levels)
- First paragraph captured per doc (truncated to ~200 chars)
- Attachments collected from chunk metadata, listed as filenames only (no content)
- Total token budget respected (one fat README + small docs)
- Path safety: reject inputs that would escape repo_root
- Skips files that don't exist; doesn't blow up
"""

from __future__ import annotations

import textwrap
from pathlib import Path

import pytest

from app.core.wiki_structure_planner.evidence_md import (
    MarkdownEvidencePack,
    build_md_pack,
)
from app.core.wiki_structure_planner.structure_skeleton import (
    ArtifactInfo,
    Cluster,
)

# ── Helpers ────────────────────────────────────────────────────────────────────


def _make_doc_cluster(
    artifacts: list[ArtifactInfo],
    dirs: list[str] | None = None,
    cluster_id: int = 1,
) -> Cluster:
    return Cluster(
        cluster_id=cluster_id,
        kind="doc",
        dirs=dirs or ["docs"],
        artifacts=artifacts,
        total_artifacts=len(artifacts),
        primary_languages=[],
        depth_range=(1, 2),
    )


def _doc_artifact(
    name: str,
    source_path: str,
    summary: str = "",
) -> ArtifactInfo:
    return ArtifactInfo(
        kind="doc_section",
        name=name,
        source_path=source_path,
        summary=summary,
    )


def _write_md(tmp_path: Path, rel_path: str, content: str) -> None:
    p = tmp_path / rel_path
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(content)


# ── MarkdownEvidencePack dataclass ─────────────────────────────────────────────


class TestMarkdownEvidencePackDataclass:
    def test_constructs_with_defaults(self):
        pack = MarkdownEvidencePack(cluster_id=1, kind="doc")
        assert pack.cluster_id == 1
        assert pack.kind == "doc"
        assert pack.doc_entries == []

    def test_serialize_returns_string(self):
        pack = MarkdownEvidencePack(cluster_id=1, kind="doc")
        result = pack.serialize()
        assert isinstance(result, str)

    def test_serialize_empty_pack_carries_cluster_metadata(self):
        pack = MarkdownEvidencePack(cluster_id=5, kind="doc")
        result = pack.serialize()
        assert "cluster=5" in result
        assert "kind=doc" in result

    def test_serialize_includes_doc_entry_title(self):
        pack = MarkdownEvidencePack(
            cluster_id=1,
            kind="doc",
            doc_entries=[
                {
                    "source_path": "docs/guide.md",
                    "title": "User Guide",
                    "toc": ["Overview", "Getting Started"],
                    "first_paragraph": "This guide explains the system.",
                    "attachments": [],
                }
            ],
        )
        result = pack.serialize()
        assert "guide.md" in result
        assert "User Guide" in result

    def test_serialize_includes_toc_headings(self):
        pack = MarkdownEvidencePack(
            cluster_id=1,
            kind="doc",
            doc_entries=[
                {
                    "source_path": "docs/api.md",
                    "title": "API Reference",
                    "toc": ["Authentication", "Endpoints"],
                    "first_paragraph": "API docs.",
                    "attachments": [],
                }
            ],
        )
        result = pack.serialize()
        assert "Authentication" in result
        assert "Endpoints" in result

    def test_serialize_includes_first_paragraph(self):
        pack = MarkdownEvidencePack(
            cluster_id=1,
            kind="doc",
            doc_entries=[
                {
                    "source_path": "docs/intro.md",
                    "title": "Introduction",
                    "toc": [],
                    "first_paragraph": "Welcome to the platform.",
                    "attachments": [],
                }
            ],
        )
        result = pack.serialize()
        assert "Welcome to the platform." in result

    def test_serialize_includes_attachment_names(self):
        pack = MarkdownEvidencePack(
            cluster_id=1,
            kind="doc",
            doc_entries=[
                {
                    "source_path": "docs/arch.md",
                    "title": "Architecture",
                    "toc": [],
                    "first_paragraph": "System diagram below.",
                    "attachments": ["diagram.png", "flow.svg"],
                }
            ],
        )
        result = pack.serialize()
        assert "diagram.png" in result
        assert "flow.svg" in result

    def test_serialize_attachments_no_content(self):
        """Attachment list must be bare filenames only — no content is included."""
        pack = MarkdownEvidencePack(
            cluster_id=1,
            kind="doc",
            doc_entries=[
                {
                    "source_path": "docs/arch.md",
                    "title": "Architecture",
                    "toc": [],
                    "first_paragraph": "Diagram below.",
                    "attachments": ["diagram.png"],
                }
            ],
        )
        result = pack.serialize()
        # Must not attempt to embed image content or binary data
        assert "PNG" not in result  # no raw binary marker
        assert "diagram.png" in result  # but filename present


# ── build_md_pack — basic construction ────────────────────────────────────────


class TestBuildMdPackBasic:
    def test_returns_markdown_evidence_pack(self, tmp_path):
        _write_md(tmp_path, "docs/intro.md", "# Intro\n\nWelcome!\n")
        cluster = _make_doc_cluster([_doc_artifact("Intro", "docs/intro.md", summary="Welcome!")])
        pack = build_md_pack(cluster, repo_root=str(tmp_path))
        assert isinstance(pack, MarkdownEvidencePack)

    def test_pack_cluster_id_matches(self, tmp_path):
        _write_md(tmp_path, "docs/intro.md", "# Intro\n\nHello.\n")
        cluster = _make_doc_cluster(
            [_doc_artifact("Intro", "docs/intro.md")],
            cluster_id=7,
        )
        pack = build_md_pack(cluster, repo_root=str(tmp_path))
        assert pack.cluster_id == 7

    def test_pack_kind_is_doc(self, tmp_path):
        _write_md(tmp_path, "docs/guide.md", "# Guide\n\nGuide content.\n")
        cluster = _make_doc_cluster([_doc_artifact("Guide", "docs/guide.md")])
        pack = build_md_pack(cluster, repo_root=str(tmp_path))
        assert pack.kind == "doc"

    def test_empty_artifacts_returns_valid_pack(self, tmp_path):
        cluster = _make_doc_cluster([])
        pack = build_md_pack(cluster, repo_root=str(tmp_path))
        assert isinstance(pack, MarkdownEvidencePack)
        assert pack.doc_entries == []

    def test_three_doc_files_all_included(self, tmp_path):
        _write_md(tmp_path, "docs/a.md", "# Alpha\n\nFirst doc.\n")
        _write_md(tmp_path, "docs/b.md", "# Beta\n\nSecond doc.\n")
        _write_md(tmp_path, "docs/c.md", "# Gamma\n\nThird doc.\n")
        cluster = _make_doc_cluster(
            [
                _doc_artifact("Alpha", "docs/a.md"),
                _doc_artifact("Beta", "docs/b.md"),
                _doc_artifact("Gamma", "docs/c.md"),
            ]
        )
        pack = build_md_pack(cluster, repo_root=str(tmp_path))
        paths = [e["source_path"] for e in pack.doc_entries]
        assert "docs/a.md" in paths
        assert "docs/b.md" in paths
        assert "docs/c.md" in paths


# ── build_md_pack — heading TOC ───────────────────────────────────────────────


class TestHeadingTOC:
    def test_h1_headings_appear_in_toc(self, tmp_path):
        content = textwrap.dedent("""\
            # Introduction
            Intro text.

            # Setup
            Setup text.
        """)
        _write_md(tmp_path, "docs/guide.md", content)
        cluster = _make_doc_cluster([_doc_artifact("Guide", "docs/guide.md")])
        pack = build_md_pack(cluster, repo_root=str(tmp_path))
        entry = next(e for e in pack.doc_entries if e["source_path"] == "docs/guide.md")
        assert "Introduction" in entry["toc"]
        assert "Setup" in entry["toc"]

    def test_h2_headings_appear_in_toc(self, tmp_path):
        content = textwrap.dedent("""\
            # Overview

            ## Getting Started
            Some text.

            ## Advanced Usage
            More text.
        """)
        _write_md(tmp_path, "docs/guide.md", content)
        cluster = _make_doc_cluster([_doc_artifact("Guide", "docs/guide.md")])
        pack = build_md_pack(cluster, repo_root=str(tmp_path))
        entry = next(e for e in pack.doc_entries if e["source_path"] == "docs/guide.md")
        assert "Getting Started" in entry["toc"]
        assert "Advanced Usage" in entry["toc"]

    def test_h3_and_below_excluded_from_toc(self, tmp_path):
        content = textwrap.dedent("""\
            # Top Level

            ## Section

            ### Deep Section
            Deep content.

            #### Even Deeper
            More.
        """)
        _write_md(tmp_path, "docs/deep.md", content)
        cluster = _make_doc_cluster([_doc_artifact("Deep", "docs/deep.md")])
        pack = build_md_pack(cluster, repo_root=str(tmp_path))
        entry = next(e for e in pack.doc_entries if e["source_path"] == "docs/deep.md")
        assert "Top Level" in entry["toc"]
        assert "Section" in entry["toc"]
        assert "Deep Section" not in entry["toc"]
        assert "Even Deeper" not in entry["toc"]

    def test_toc_preserves_document_order(self, tmp_path):
        content = textwrap.dedent("""\
            # Alpha

            ## Beta

            # Gamma

            ## Delta
        """)
        _write_md(tmp_path, "docs/ordered.md", content)
        cluster = _make_doc_cluster([_doc_artifact("Ordered", "docs/ordered.md")])
        pack = build_md_pack(cluster, repo_root=str(tmp_path))
        entry = next(e for e in pack.doc_entries if e["source_path"] == "docs/ordered.md")
        toc = entry["toc"]
        assert toc.index("Alpha") < toc.index("Beta")
        assert toc.index("Beta") < toc.index("Gamma")
        assert toc.index("Gamma") < toc.index("Delta")

    def test_no_headings_returns_empty_toc(self, tmp_path):
        content = "Just plain text with no headings at all.\n"
        _write_md(tmp_path, "docs/plain.md", content)
        cluster = _make_doc_cluster([_doc_artifact("Plain", "docs/plain.md")])
        pack = build_md_pack(cluster, repo_root=str(tmp_path))
        entry = next(e for e in pack.doc_entries if e["source_path"] == "docs/plain.md")
        assert entry["toc"] == []


# ── build_md_pack — first paragraph ───────────────────────────────────────────


class TestFirstParagraph:
    def test_first_paragraph_extracted(self, tmp_path):
        content = textwrap.dedent("""\
            # Title

            This is the first paragraph of content.
        """)
        _write_md(tmp_path, "docs/doc.md", content)
        cluster = _make_doc_cluster([_doc_artifact("Doc", "docs/doc.md")])
        pack = build_md_pack(cluster, repo_root=str(tmp_path))
        entry = next(e for e in pack.doc_entries if e["source_path"] == "docs/doc.md")
        assert "first paragraph" in entry["first_paragraph"]

    def test_first_paragraph_truncated_to_200_chars(self, tmp_path):
        long_para = "A" * 500
        content = f"# Title\n\n{long_para}\n"
        _write_md(tmp_path, "docs/long.md", content)
        cluster = _make_doc_cluster([_doc_artifact("Long", "docs/long.md")])
        pack = build_md_pack(cluster, repo_root=str(tmp_path))
        entry = next(e for e in pack.doc_entries if e["source_path"] == "docs/long.md")
        assert len(entry["first_paragraph"]) <= 200

    def test_preamble_before_first_heading_used_as_first_paragraph(self, tmp_path):
        content = textwrap.dedent("""\
            This preamble comes before any heading.

            # Heading One

            Body text.
        """)
        _write_md(tmp_path, "docs/preamble.md", content)
        cluster = _make_doc_cluster([_doc_artifact("Preamble", "docs/preamble.md")])
        pack = build_md_pack(cluster, repo_root=str(tmp_path))
        entry = next(e for e in pack.doc_entries if e["source_path"] == "docs/preamble.md")
        assert "preamble" in entry["first_paragraph"].lower()

    def test_empty_doc_returns_empty_first_paragraph(self, tmp_path):
        _write_md(tmp_path, "docs/empty.md", "")
        cluster = _make_doc_cluster([_doc_artifact("Empty", "docs/empty.md")])
        pack = build_md_pack(cluster, repo_root=str(tmp_path))
        entry = next(e for e in pack.doc_entries if e["source_path"] == "docs/empty.md")
        assert entry["first_paragraph"] == ""


# ── build_md_pack — attachments ───────────────────────────────────────────────


class TestAttachments:
    def test_image_references_collected_as_attachments(self, tmp_path):
        content = textwrap.dedent("""\
            # Architecture

            See the diagram below.

            ![System Diagram](architecture.png)
        """)
        _write_md(tmp_path, "docs/arch.md", content)
        cluster = _make_doc_cluster([_doc_artifact("Arch", "docs/arch.md")])
        pack = build_md_pack(cluster, repo_root=str(tmp_path))
        entry = next(e for e in pack.doc_entries if e["source_path"] == "docs/arch.md")
        assert "architecture.png" in entry["attachments"]

    def test_multiple_attachments_all_collected(self, tmp_path):
        content = textwrap.dedent("""\
            # Docs

            ![Flow](flow.svg)
            ![Diagram](diagram.png)
        """)
        _write_md(tmp_path, "docs/multi.md", content)
        cluster = _make_doc_cluster([_doc_artifact("Multi", "docs/multi.md")])
        pack = build_md_pack(cluster, repo_root=str(tmp_path))
        entry = next(e for e in pack.doc_entries if e["source_path"] == "docs/multi.md")
        assert "flow.svg" in entry["attachments"]
        assert "diagram.png" in entry["attachments"]

    def test_attachments_across_chunks_deduplicated(self, tmp_path):
        # Same image referenced in both H1 sections — should appear once.
        content = textwrap.dedent("""\
            # Section A

            ![Logo](logo.png)

            # Section B

            ![Logo](logo.png)
        """)
        _write_md(tmp_path, "docs/dup.md", content)
        cluster = _make_doc_cluster([_doc_artifact("Dup", "docs/dup.md")])
        pack = build_md_pack(cluster, repo_root=str(tmp_path))
        entry = next(e for e in pack.doc_entries if e["source_path"] == "docs/dup.md")
        assert entry["attachments"].count("logo.png") == 1

    def test_no_attachments_returns_empty_list(self, tmp_path):
        content = "# Simple\n\nNo images here.\n"
        _write_md(tmp_path, "docs/simple.md", content)
        cluster = _make_doc_cluster([_doc_artifact("Simple", "docs/simple.md")])
        pack = build_md_pack(cluster, repo_root=str(tmp_path))
        entry = next(e for e in pack.doc_entries if e["source_path"] == "docs/simple.md")
        assert entry["attachments"] == []

    def test_attachment_content_not_read(self, tmp_path):
        """Evidence pack must list attachment filenames only — no disk read."""
        content = "# Docs\n\n![Secret](secret_file.png)\n"
        _write_md(tmp_path, "docs/ref.md", content)
        # secret_file.png is NOT created on disk — the pack must not error or read it
        cluster = _make_doc_cluster([_doc_artifact("Ref", "docs/ref.md")])
        pack = build_md_pack(cluster, repo_root=str(tmp_path))
        entry = next(e for e in pack.doc_entries if e["source_path"] == "docs/ref.md")
        # Filename is listed despite file not existing on disk
        assert "secret_file.png" in entry["attachments"]


# ── build_md_pack — token budget ──────────────────────────────────────────────


class TestTokenBudget:
    def test_fat_readme_stays_within_budget(self, tmp_path):
        """A very large doc file must be trimmed so total pack respects budget."""
        big_content = "# README\n\n" + ("Word " * 5000) + "\n"
        _write_md(tmp_path, "docs/README.md", big_content)
        # Small companion docs
        _write_md(tmp_path, "docs/intro.md", "# Intro\n\nShort intro.\n")
        _write_md(tmp_path, "docs/faq.md", "# FAQ\n\nFrequently asked questions.\n")
        cluster = _make_doc_cluster(
            [
                _doc_artifact("README", "docs/README.md"),
                _doc_artifact("Intro", "docs/intro.md"),
                _doc_artifact("FAQ", "docs/faq.md"),
            ]
        )
        pack = build_md_pack(cluster, repo_root=str(tmp_path), token_budget=1000)
        # Tight check against the char budget (token_budget * 4 chars/token).
        # Allow a small constant overhead for per-entry headers, but not
        # the 3× slack the previous version had — that masked trimming
        # regressions (Copilot review of #250).
        char_budget = 1000 * 4
        overhead = 200  # ~50 tokens of headers across multiple entries
        assert len(pack.serialize()) <= char_budget + overhead

    def test_small_cluster_well_under_budget(self, tmp_path):
        _write_md(tmp_path, "docs/quick.md", "# Quick\n\nA short note.\n")
        cluster = _make_doc_cluster([_doc_artifact("Quick", "docs/quick.md")])
        pack = build_md_pack(cluster, repo_root=str(tmp_path), token_budget=1000)
        approx_tokens = len(pack.serialize()) / 4
        assert approx_tokens < 1000

    def test_per_doc_first_paragraph_capped_at_200_chars(self, tmp_path):
        """Each doc's first_paragraph must be at most 200 chars regardless of budget."""
        long_text = "B" * 1000
        _write_md(tmp_path, "docs/verbose.md", f"# Verbose\n\n{long_text}\n")
        cluster = _make_doc_cluster([_doc_artifact("Verbose", "docs/verbose.md")])
        pack = build_md_pack(cluster, repo_root=str(tmp_path), token_budget=5000)
        entry = next(e for e in pack.doc_entries if e["source_path"] == "docs/verbose.md")
        assert len(entry["first_paragraph"]) <= 200


# ── build_md_pack — missing files ─────────────────────────────────────────────


class TestMissingFiles:
    def test_missing_file_skipped_gracefully(self, tmp_path):
        cluster = _make_doc_cluster([_doc_artifact("Ghost", "docs/ghost.md")])
        # ghost.md doesn't exist — must not raise
        pack = build_md_pack(cluster, repo_root=str(tmp_path))
        assert isinstance(pack, MarkdownEvidencePack)
        # Entry is skipped (file not readable)
        paths = [e["source_path"] for e in pack.doc_entries]
        assert "docs/ghost.md" not in paths

    def test_partial_missing_files_partial_result(self, tmp_path):
        _write_md(tmp_path, "docs/real.md", "# Real\n\nContent here.\n")
        cluster = _make_doc_cluster(
            [
                _doc_artifact("Real", "docs/real.md"),
                _doc_artifact("Ghost", "docs/missing.md"),
            ]
        )
        pack = build_md_pack(cluster, repo_root=str(tmp_path))
        paths = [e["source_path"] for e in pack.doc_entries]
        assert "docs/real.md" in paths
        assert "docs/missing.md" not in paths


# ── build_md_pack — path safety ───────────────────────────────────────────────


class TestPathSafety:
    def test_traversal_in_source_path_rejected(self, tmp_path):
        cluster = _make_doc_cluster([_doc_artifact("Evil", "../../etc/passwd")])
        pack = build_md_pack(cluster, repo_root=str(tmp_path))
        paths = [e["source_path"] for e in pack.doc_entries]
        assert "../../etc/passwd" not in paths

    def test_absolute_source_path_rejected(self, tmp_path):
        cluster = _make_doc_cluster([_doc_artifact("AbsEvil", "/etc/passwd")])
        pack = build_md_pack(cluster, repo_root=str(tmp_path))
        paths = [e["source_path"] for e in pack.doc_entries]
        assert "/etc/passwd" not in paths

    def test_symlink_escape_rejected(self, tmp_path):
        link = tmp_path / "docs" / "evil.md"
        link.parent.mkdir(parents=True, exist_ok=True)
        try:
            link.symlink_to("/etc/passwd")
        except (OSError, NotImplementedError):
            pytest.skip("symlinks not supported on this platform")
        cluster = _make_doc_cluster([_doc_artifact("Evil", "docs/evil.md")])
        pack = build_md_pack(cluster, repo_root=str(tmp_path))
        # The symlink target escapes repo_root — safe_join must reject it entirely,
        # producing no doc entries at all rather than silently reading /etc/passwd.
        assert pack.doc_entries == []

    def test_safe_paths_work_normally(self, tmp_path):
        _write_md(tmp_path, "docs/safe.md", "# Safe\n\nSafe content.\n")
        cluster = _make_doc_cluster([_doc_artifact("Safe", "docs/safe.md")])
        pack = build_md_pack(cluster, repo_root=str(tmp_path))
        paths = [e["source_path"] for e in pack.doc_entries]
        assert "docs/safe.md" in paths


# ── build_md_pack — deduplication ─────────────────────────────────────────────


class TestDeduplication:
    def test_same_source_path_multiple_artifacts_deduplicated(self, tmp_path):
        """Multiple doc_section artifacts pointing to the same file produce one entry."""
        _write_md(tmp_path, "docs/guide.md", "# Guide\n\n## Setup\n\nSetup text.\n")
        cluster = _make_doc_cluster(
            [
                _doc_artifact("Guide / Overview", "docs/guide.md"),
                _doc_artifact("Guide / Setup", "docs/guide.md"),
            ]
        )
        pack = build_md_pack(cluster, repo_root=str(tmp_path))
        paths = [e["source_path"] for e in pack.doc_entries]
        assert paths.count("docs/guide.md") == 1
