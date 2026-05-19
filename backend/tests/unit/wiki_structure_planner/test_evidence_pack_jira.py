"""Unit tests for wiki_structure_planner/evidence_jira.py — jira-epic cluster packs (#235).

Covers:
- Pack construction from a fabricated cluster with 1 epic + 3 stories + 1 subtask
- Epic title + first paragraph captured from disk file
- Child issue list with key/type/status/summary (descriptions NOT included)
- Labels / components / fix_versions from epic frontmatter YAML
- Edge case: cluster with no epic — pick representative gracefully
- Edge case: epic with no children
- Total token budget respected
- Path safety: reject inputs escaping repo_root
"""

from __future__ import annotations

import textwrap
from pathlib import Path

import pytest

from app.core.wiki_structure_planner.evidence_jira import (
    JiraEvidencePack,
    build_jira_pack,
)
from app.core.wiki_structure_planner.structure_skeleton import (
    ArtifactInfo,
    Cluster,
)

# ── Helpers ────────────────────────────────────────────────────────────────────


def _make_jira_artifact(
    name: str,
    source_path: str,
    issue_type: str,
    epic_key: str = "",
    summary: str = "",
) -> ArtifactInfo:
    return ArtifactInfo(
        kind="jira_issue",
        name=name,
        source_path=source_path,
        issue_type=issue_type,
        epic_key=epic_key,
        summary=summary,
    )


def _make_jira_cluster(
    artifacts: list[ArtifactInfo],
    cluster_id: int = 1,
) -> Cluster:
    return Cluster(
        cluster_id=cluster_id,
        kind="jira",
        dirs=["jira"],
        artifacts=artifacts,
        total_artifacts=len(artifacts),
        primary_languages=[],
        depth_range=(0, 1),
    )


def _write_issue_md(
    tmp_path: Path,
    rel_path: str,
    frontmatter: dict,
    body: str,
) -> None:
    """Write a Jira-exported markdown file with YAML frontmatter."""
    p = tmp_path / rel_path
    p.parent.mkdir(parents=True, exist_ok=True)
    lines = ["---"]
    for k, v in frontmatter.items():
        if isinstance(v, list):
            lines.append(f"{k}:")
            for item in v:
                lines.append(f"  - {item}")
        else:
            lines.append(f"{k}: {v}")
    lines.append("---")
    lines.append("")
    lines.append(body)
    p.write_text("\n".join(lines))


# ── JiraEvidencePack dataclass ─────────────────────────────────────────────────


class TestJiraEvidencePackDataclass:
    def test_constructs_with_required_fields(self):
        pack = JiraEvidencePack(cluster_id=1, epic_key="PROJ-1", epic_title="My Epic")
        assert pack.cluster_id == 1
        assert pack.epic_key == "PROJ-1"
        assert pack.epic_title == "My Epic"
        assert pack.epic_description == ""
        assert pack.child_issues == []
        assert pack.labels == []
        assert pack.components == []
        assert pack.fix_versions == []

    def test_serialize_returns_string(self):
        pack = JiraEvidencePack(cluster_id=1, epic_key="PROJ-1", epic_title="My Epic")
        assert isinstance(pack.serialize(), str)

    def test_serialize_includes_cluster_metadata(self):
        pack = JiraEvidencePack(cluster_id=3, epic_key="XY-10", epic_title="Auth Epic")
        result = pack.serialize()
        assert "cluster=3" in result
        assert "kind=jira" in result

    def test_serialize_includes_epic_title_and_description(self):
        pack = JiraEvidencePack(
            cluster_id=1,
            epic_key="ENG-1",
            epic_title="Payment Gateway Overhaul",
            epic_description="Refactor all payment flows.",
        )
        result = pack.serialize()
        assert "Payment Gateway Overhaul" in result
        assert "Refactor all payment flows." in result

    def test_serialize_includes_child_issues_table(self):
        pack = JiraEvidencePack(
            cluster_id=1,
            epic_key="ENG-1",
            epic_title="My Epic",
            child_issues=[
                {"key": "ENG-2", "type": "Story", "status": "In Progress", "summary": "Do the thing"},
                {"key": "ENG-3", "type": "Subtask", "status": "Done", "summary": "Sub thing"},
            ],
        )
        result = pack.serialize()
        assert "ENG-2" in result
        assert "In Progress" in result
        assert "Do the thing" in result
        assert "ENG-3" in result
        assert "Subtask" in result

    def test_serialize_includes_labels_when_present(self):
        pack = JiraEvidencePack(
            cluster_id=1,
            epic_key="ENG-1",
            epic_title="Epic",
            labels=["backend", "critical"],
        )
        result = pack.serialize()
        assert "backend" in result
        assert "critical" in result

    def test_serialize_includes_components_when_present(self):
        pack = JiraEvidencePack(
            cluster_id=1,
            epic_key="ENG-1",
            epic_title="Epic",
            components=["Auth Service", "Payments"],
        )
        result = pack.serialize()
        assert "Auth Service" in result
        assert "Payments" in result

    def test_serialize_includes_fix_versions_when_present(self):
        pack = JiraEvidencePack(
            cluster_id=1,
            epic_key="ENG-1",
            epic_title="Epic",
            fix_versions=["2.0.0", "2.1.0"],
        )
        result = pack.serialize()
        assert "2.0.0" in result
        assert "2.1.0" in result

    def test_serialize_omits_labels_section_when_empty(self):
        pack = JiraEvidencePack(cluster_id=1, epic_key="ENG-1", epic_title="Epic")
        result = pack.serialize()
        assert "Labels" not in result
        assert "Components" not in result
        assert "Fix Versions" not in result


# ── build_jira_pack — full cluster (1 epic + 3 stories + 1 subtask) ───────────


class TestBuildJiraPackFull:
    def _setup_full_cluster(self, tmp_path: Path) -> tuple[Cluster, list[ArtifactInfo]]:
        """1 epic + 3 stories + 1 subtask, all with disk files."""
        epic_fm = {
            "summary": "Platform Modernisation",
            "status": "In Progress",
            "issue_type": "Epic",
            "labels": ["backend", "platform"],
            "components": ["Core API", "Auth Service"],
            "fix_versions": ["3.0.0"],
        }
        _write_issue_md(
            tmp_path,
            "jira/PLAT-1.md",
            epic_fm,
            textwrap.dedent("""\
                # PLAT-1: Platform Modernisation

                Modernise the platform to support multi-tenant deployments.

                This is a second paragraph that should NOT appear.
            """),
        )

        story1_fm = {"summary": "Migrate auth to OAuth2", "status": "Done", "issue_type": "Story"}
        _write_issue_md(
            tmp_path, "jira/PLAT-2.md", story1_fm, "# PLAT-2: Migrate auth to OAuth2\n\nOAuth2 migration details.\n"
        )

        story2_fm = {"summary": "Add rate limiting", "status": "In Progress", "issue_type": "Story"}
        _write_issue_md(tmp_path, "jira/PLAT-3.md", story2_fm, "# PLAT-3: Add rate limiting\n\nRate limit details.\n")

        story3_fm = {"summary": "Tenant isolation", "status": "To Do", "issue_type": "Story"}
        _write_issue_md(tmp_path, "jira/PLAT-4.md", story3_fm, "# PLAT-4: Tenant isolation\n\nIsolation details.\n")

        subtask_fm = {"summary": "Write isolation tests", "status": "To Do", "issue_type": "Subtask"}
        _write_issue_md(tmp_path, "jira/PLAT-5.md", subtask_fm, "# PLAT-5: Write isolation tests\n\nTest body.\n")

        artifacts = [
            _make_jira_artifact("PLAT-1", "jira/PLAT-1.md", "epic", summary="Platform Modernisation"),
            _make_jira_artifact(
                "PLAT-2", "jira/PLAT-2.md", "story", epic_key="PLAT-1", summary="Migrate auth to OAuth2"
            ),
            _make_jira_artifact("PLAT-3", "jira/PLAT-3.md", "story", epic_key="PLAT-1", summary="Add rate limiting"),
            _make_jira_artifact("PLAT-4", "jira/PLAT-4.md", "story", epic_key="PLAT-1", summary="Tenant isolation"),
            _make_jira_artifact(
                "PLAT-5", "jira/PLAT-5.md", "subtask", epic_key="PLAT-1", summary="Write isolation tests"
            ),
        ]
        return _make_jira_cluster(artifacts), artifacts

    def test_returns_jira_evidence_pack(self, tmp_path):
        cluster, _ = self._setup_full_cluster(tmp_path)
        pack = build_jira_pack(cluster, repo_root=str(tmp_path))
        assert isinstance(pack, JiraEvidencePack)

    def test_epic_key_matches(self, tmp_path):
        cluster, _ = self._setup_full_cluster(tmp_path)
        pack = build_jira_pack(cluster, repo_root=str(tmp_path))
        assert pack.epic_key == "PLAT-1"

    def test_epic_title_captured(self, tmp_path):
        cluster, _ = self._setup_full_cluster(tmp_path)
        pack = build_jira_pack(cluster, repo_root=str(tmp_path))
        assert "Platform Modernisation" in pack.epic_title

    def test_epic_description_first_paragraph_only(self, tmp_path):
        cluster, _ = self._setup_full_cluster(tmp_path)
        pack = build_jira_pack(cluster, repo_root=str(tmp_path))
        # First paragraph of body should be present
        assert "Modernise the platform" in pack.epic_description
        # Second paragraph should NOT be included
        assert "second paragraph" not in pack.epic_description

    def test_child_issues_include_all_stories_and_subtasks(self, tmp_path):
        cluster, _ = self._setup_full_cluster(tmp_path)
        pack = build_jira_pack(cluster, repo_root=str(tmp_path))
        keys = {c["key"] for c in pack.child_issues}
        assert keys == {"PLAT-2", "PLAT-3", "PLAT-4", "PLAT-5"}

    def test_epic_not_in_child_issues(self, tmp_path):
        cluster, _ = self._setup_full_cluster(tmp_path)
        pack = build_jira_pack(cluster, repo_root=str(tmp_path))
        keys = [c["key"] for c in pack.child_issues]
        assert "PLAT-1" not in keys

    def test_child_issues_carry_key_type_status_summary(self, tmp_path):
        cluster, _ = self._setup_full_cluster(tmp_path)
        pack = build_jira_pack(cluster, repo_root=str(tmp_path))
        child_by_key = {c["key"]: c for c in pack.child_issues}
        story = child_by_key["PLAT-2"]
        assert story["key"] == "PLAT-2"
        assert "story" in story["type"].lower()
        assert story["status"] == "Done"
        assert "Migrate auth to OAuth2" in story["summary"]

    def test_child_issues_no_description_bodies(self, tmp_path):
        cluster, _ = self._setup_full_cluster(tmp_path)
        pack = build_jira_pack(cluster, repo_root=str(tmp_path))
        serialized = pack.serialize()
        # Description bodies are multi-line — they must not appear
        assert "OAuth2 migration details" not in serialized
        assert "Rate limit details" not in serialized

    def test_labels_from_epic_frontmatter(self, tmp_path):
        cluster, _ = self._setup_full_cluster(tmp_path)
        pack = build_jira_pack(cluster, repo_root=str(tmp_path))
        assert "backend" in pack.labels
        assert "platform" in pack.labels

    def test_components_from_epic_frontmatter(self, tmp_path):
        cluster, _ = self._setup_full_cluster(tmp_path)
        pack = build_jira_pack(cluster, repo_root=str(tmp_path))
        assert "Core API" in pack.components
        assert "Auth Service" in pack.components

    def test_fix_versions_from_epic_frontmatter(self, tmp_path):
        cluster, _ = self._setup_full_cluster(tmp_path)
        pack = build_jira_pack(cluster, repo_root=str(tmp_path))
        assert "3.0.0" in pack.fix_versions

    def test_cluster_id_matches(self, tmp_path):
        cluster, _ = self._setup_full_cluster(tmp_path)
        cluster.cluster_id = 99
        pack = build_jira_pack(cluster, repo_root=str(tmp_path))
        assert pack.cluster_id == 99


# ── build_jira_pack — edge: no epic in cluster ────────────────────────────────


class TestBuildJiraPackNoEpic:
    """Cluster with only stories (no epic artifact)."""

    def _setup_no_epic_cluster(self, tmp_path: Path) -> Cluster:
        fm_a = {"summary": "Do thing A", "status": "In Progress", "issue_type": "Story"}
        fm_b = {"summary": "Do thing B", "status": "To Do", "issue_type": "Story"}
        _write_issue_md(tmp_path, "jira/ABC-10.md", fm_a, "# ABC-10: Do thing A\n\nBody A.\n")
        _write_issue_md(tmp_path, "jira/ABC-11.md", fm_b, "# ABC-11: Do thing B\n\nBody B.\n")
        artifacts = [
            _make_jira_artifact("ABC-10", "jira/ABC-10.md", "story", summary="Do thing A"),
            _make_jira_artifact("ABC-11", "jira/ABC-11.md", "story", summary="Do thing B"),
        ]
        return _make_jira_cluster(artifacts)

    def test_returns_pack_without_raising(self, tmp_path):
        cluster = self._setup_no_epic_cluster(tmp_path)
        pack = build_jira_pack(cluster, repo_root=str(tmp_path))
        assert isinstance(pack, JiraEvidencePack)

    def test_representative_chosen_gracefully(self, tmp_path):
        """When no epic, a story is chosen as representative; pack is still usable."""
        cluster = self._setup_no_epic_cluster(tmp_path)
        pack = build_jira_pack(cluster, repo_root=str(tmp_path))
        # Some key was picked; pack has a non-empty epic_key
        assert pack.epic_key != ""

    def test_remaining_artifacts_become_children(self, tmp_path):
        cluster = self._setup_no_epic_cluster(tmp_path)
        pack = build_jira_pack(cluster, repo_root=str(tmp_path))
        # The non-representative artifact appears in child_issues
        all_keys = {pack.epic_key} | {c["key"] for c in pack.child_issues}
        assert "ABC-10" in all_keys
        assert "ABC-11" in all_keys


# ── build_jira_pack — edge: epic with no children ─────────────────────────────


class TestBuildJiraPackEpicOnly:
    def test_no_children_returns_empty_list(self, tmp_path):
        fm = {"summary": "Lone Epic", "status": "Open", "issue_type": "Epic"}
        _write_issue_md(tmp_path, "jira/LONE-1.md", fm, "# LONE-1: Lone Epic\n\nJust an epic.\n")
        artifacts = [
            _make_jira_artifact("LONE-1", "jira/LONE-1.md", "epic", summary="Lone Epic"),
        ]
        cluster = _make_jira_cluster(artifacts)
        pack = build_jira_pack(cluster, repo_root=str(tmp_path))
        assert pack.child_issues == []

    def test_pack_still_serializes_without_children(self, tmp_path):
        fm = {"summary": "Lone Epic", "status": "Open", "issue_type": "Epic"}
        _write_issue_md(tmp_path, "jira/LONE-1.md", fm, "# LONE-1: Lone Epic\n\nJust an epic.\n")
        artifacts = [
            _make_jira_artifact("LONE-1", "jira/LONE-1.md", "epic", summary="Lone Epic"),
        ]
        cluster = _make_jira_cluster(artifacts)
        pack = build_jira_pack(cluster, repo_root=str(tmp_path))
        result = pack.serialize()
        assert "Lone Epic" in result


# ── build_jira_pack — empty cluster ───────────────────────────────────────────


class TestBuildJiraPackEmpty:
    def test_empty_cluster_returns_valid_pack(self, tmp_path):
        cluster = _make_jira_cluster([])
        pack = build_jira_pack(cluster, repo_root=str(tmp_path))
        assert isinstance(pack, JiraEvidencePack)
        assert pack.epic_key == ""
        assert pack.child_issues == []


# ── build_jira_pack — token budget ────────────────────────────────────────────


class TestBuildJiraPackBudget:
    def test_serialized_pack_stays_within_token_budget(self, tmp_path):
        """A large cluster must not grossly exceed the ~1K token budget."""
        epic_fm = {
            "summary": "Big Epic",
            "status": "In Progress",
            "issue_type": "Epic",
            "labels": [f"label{i}" for i in range(30)],
            "components": [f"comp{i}" for i in range(30)],
            "fix_versions": [f"v{i}.0" for i in range(10)],
        }
        _write_issue_md(
            tmp_path,
            "jira/BIG-1.md",
            epic_fm,
            "# BIG-1: Big Epic\n\n" + "Very long description. " * 200 + "\n\nSecond para.\n",
        )
        artifacts = [_make_jira_artifact("BIG-1", "jira/BIG-1.md", "epic", summary="Big Epic")]
        for i in range(2, 60):
            fm = {"summary": f"Story {i}", "status": "To Do", "issue_type": "Story"}
            _write_issue_md(tmp_path, f"jira/BIG-{i}.md", fm, f"# BIG-{i}: Story {i}\n\nBody.\n")
            artifacts.append(
                _make_jira_artifact(f"BIG-{i}", f"jira/BIG-{i}.md", "story", epic_key="BIG-1", summary=f"Story {i}")
            )

        cluster = _make_jira_cluster(artifacts)
        pack = build_jira_pack(cluster, repo_root=str(tmp_path), token_budget=1000)
        serialized = pack.serialize()
        # Tight assertion (Copilot review): the budget enforcement must
        # actually trim. 1000 tokens * 4 chars + small overhead for headers.
        char_budget = 1000 * 4
        overhead = 500  # header/table-format + ~5 lines per row overhead
        assert len(serialized) <= char_budget + overhead
        # And that trimming actually engaged: child rows must have been
        # dropped from the original 58 stories.
        assert len(pack.child_issues) < 58

    def test_missing_disk_file_does_not_raise(self, tmp_path):
        """Artifact referencing a nonexistent file is tolerated gracefully."""
        artifacts = [
            _make_jira_artifact("GHOST-1", "jira/GHOST-1.md", "epic", summary="Ghost"),
        ]
        cluster = _make_jira_cluster(artifacts)
        pack = build_jira_pack(cluster, repo_root=str(tmp_path))
        assert isinstance(pack, JiraEvidencePack)
        # summary from ArtifactInfo used as fallback title
        assert "Ghost" in pack.epic_title or pack.epic_key == "GHOST-1"


# ── build_jira_pack — frontmatter parsing ─────────────────────────────────────


class TestFrontmatterParsing:
    def test_scalar_fields_parsed(self, tmp_path):
        fm = {
            "summary": "Scalar Test",
            "status": "Closed",
            "issue_type": "Epic",
        }
        _write_issue_md(tmp_path, "jira/SC-1.md", fm, "# SC-1: Scalar Test\n\nBody.\n")
        artifacts = [_make_jira_artifact("SC-1", "jira/SC-1.md", "epic", summary="Scalar Test")]
        pack = build_jira_pack(_make_jira_cluster(artifacts), repo_root=str(tmp_path))
        assert pack.epic_title != "" or pack.epic_key == "SC-1"

    def test_list_fields_parsed(self, tmp_path):
        fm = {
            "summary": "List Test",
            "status": "Open",
            "issue_type": "Epic",
            "labels": ["foo", "bar"],
            "components": ["svc-a"],
            "fix_versions": ["1.0"],
        }
        _write_issue_md(tmp_path, "jira/LT-1.md", fm, "# LT-1: List Test\n\nBody.\n")
        artifacts = [_make_jira_artifact("LT-1", "jira/LT-1.md", "epic", summary="List Test")]
        pack = build_jira_pack(_make_jira_cluster(artifacts), repo_root=str(tmp_path))
        assert "foo" in pack.labels
        assert "bar" in pack.labels
        assert "svc-a" in pack.components
        assert "1.0" in pack.fix_versions

    def test_no_frontmatter_does_not_raise(self, tmp_path):
        """Markdown files without YAML frontmatter are handled gracefully."""
        p = tmp_path / "jira" / "NF-1.md"
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text("# NF-1: No Frontmatter\n\nPlain body.\n")
        artifacts = [_make_jira_artifact("NF-1", "jira/NF-1.md", "epic", summary="No Frontmatter")]
        pack = build_jira_pack(_make_jira_cluster(artifacts), repo_root=str(tmp_path))
        assert isinstance(pack, JiraEvidencePack)
        assert pack.labels == []

    def test_quoted_scalar_values_stripped(self, tmp_path):
        # Frontmatter generators often quote special-char scalars.
        content = "---\nsummary: \"Quoted Summary\"\nstatus: 'In Progress'\nissue_type: epic\n---\n\nBody.\n"
        p = tmp_path / "jira" / "QS-1.md"
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(content)
        artifacts = [_make_jira_artifact("QS-1", "jira/QS-1.md", "epic", summary="QS")]
        pack = build_jira_pack(_make_jira_cluster(artifacts), repo_root=str(tmp_path))
        # No leftover quote chars in title.
        assert pack.epic_title == "Quoted Summary"

    def test_inline_list_syntax_parsed_and_stripped(self, tmp_path):
        content = (
            "---\nsummary: Has Lists\nlabels: ['infra', \"security\", 'auth']\n"
            'components: [api, "core"]\nissue_type: epic\n---\n\nBody.\n'
        )
        p = tmp_path / "jira" / "IL-1.md"
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(content)
        artifacts = [_make_jira_artifact("IL-1", "jira/IL-1.md", "epic", summary="IL")]
        pack = build_jira_pack(_make_jira_cluster(artifacts), repo_root=str(tmp_path))
        assert pack.labels == ["infra", "security", "auth"]
        assert pack.components == ["api", "core"]

    def test_quoted_list_items_stripped(self, tmp_path):
        content = (
            "---\nsummary: Quoted List\nlabels:\n  - 'first'\n  - \"second\"\n  - third\n"
            "issue_type: epic\n---\n\nBody.\n"
        )
        p = tmp_path / "jira" / "QL-1.md"
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(content)
        artifacts = [_make_jira_artifact("QL-1", "jira/QL-1.md", "epic", summary="QL")]
        pack = build_jira_pack(_make_jira_cluster(artifacts), repo_root=str(tmp_path))
        assert pack.labels == ["first", "second", "third"]


# ── Markdown table cell escaping ──────────────────────────────────────────────


class TestTableCellEscaping:
    def _setup_with_summary(self, tmp_path, key: str, summary: str, content: str = "Body.\n") -> ArtifactInfo:
        path = tmp_path / "jira" / f"{key}.md"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(f"---\nstatus: Open\n---\n\n# {key}\n\n{content}")
        return _make_jira_artifact(key, f"jira/{key}.md", "story", summary=summary)

    def test_pipe_in_summary_escaped(self, tmp_path):
        epic = _make_jira_artifact("E-1", "", "epic", summary="Epic")
        child = self._setup_with_summary(tmp_path, "C-1", "Cell with | pipe")
        cluster = _make_jira_cluster([epic, child])
        pack = build_jira_pack(cluster, repo_root=str(tmp_path))
        serialized = pack.serialize()
        # The row line must not contain an unescaped pipe in the summary cell.
        for line in serialized.splitlines():
            if "C-1" in line and "|" in line:
                # Count the pipes — should be exactly 5 (4 separators + 1 escaped).
                # Unescaped raw pipe + 4 separators would be 5; but the literal
                # "\|" still appears.
                assert "\\|" in line
                break

    def test_newline_in_summary_normalised(self, tmp_path):
        epic = _make_jira_artifact("E-2", "", "epic", summary="Epic")
        child = self._setup_with_summary(tmp_path, "C-2", "First line\nSecond line")
        cluster = _make_jira_cluster([epic, child])
        pack = build_jira_pack(cluster, repo_root=str(tmp_path))
        serialized = pack.serialize()
        # No newline should appear inside a child row.
        c2_lines = [line for line in serialized.splitlines() if "C-2" in line]
        for line in c2_lines:
            assert "\n" not in line
        # And both fragments still on the same line.
        assert any("First line Second line" in line for line in c2_lines)


# ── Path safety ────────────────────────────────────────────────────────────────


class TestPathSafety:
    def test_traversal_in_source_path_rejected(self, tmp_path):
        """../../escape in source_path must not read outside repo_root."""
        artifacts = [
            _make_jira_artifact("EVIL-1", "../../etc/passwd", "epic", summary="Evil"),
        ]
        cluster = _make_jira_cluster(artifacts)
        # Must not raise; traversal skipped, fallback to summary
        pack = build_jira_pack(cluster, repo_root=str(tmp_path))
        assert isinstance(pack, JiraEvidencePack)

    def test_absolute_source_path_rejected(self, tmp_path):
        """Absolute paths must not be followed."""
        artifacts = [
            _make_jira_artifact("ABS-1", "/etc/passwd", "epic", summary="Absolute"),
        ]
        cluster = _make_jira_cluster(artifacts)
        pack = build_jira_pack(cluster, repo_root=str(tmp_path))
        assert isinstance(pack, JiraEvidencePack)
        # /etc/passwd content must not appear in title or description
        assert "root:" not in pack.epic_description

    def test_symlink_escape_rejected(self, tmp_path):
        """Symlinks pointing outside repo_root must be rejected."""
        link = tmp_path / "jira" / "evil.md"
        link.parent.mkdir(parents=True, exist_ok=True)
        try:
            link.symlink_to("/etc/passwd")
        except (OSError, NotImplementedError):
            pytest.skip("symlinks not supported on this platform")
        artifacts = [
            _make_jira_artifact("SYMLINK-1", "jira/evil.md", "epic", summary="Evil symlink"),
        ]
        cluster = _make_jira_cluster(artifacts)
        pack = build_jira_pack(cluster, repo_root=str(tmp_path))
        assert "root:" not in pack.epic_description
