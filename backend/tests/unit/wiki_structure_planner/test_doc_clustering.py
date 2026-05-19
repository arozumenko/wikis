"""Unit tests for wiki_structure_planner/doc_clustering.py (#230).

Covers:
- 5-doc markdown set with two clear clusters (link-graph component split)
- Confluence pages under two different parents → two clusters
- Jira with two epics + their stories → two clusters (one per epic)
- Singleton artifact (no edges) becomes its own cluster
- Empty input → empty cluster list
- The returned Cluster.kind matches the source kind
"""

from __future__ import annotations

import pytest

from app.core.wiki_structure_planner.doc_clustering import (
    cluster_confluence_pages,
    cluster_jira_issues,
    cluster_markdown_docs,
)
from app.core.wiki_structure_planner.structure_skeleton import (
    ArtifactInfo,
    Cluster,
)


# ── Helpers ─────────────────────────────────────────────────────────────────


def _md_artifact(name: str, source_path: str, summary: str = "") -> ArtifactInfo:
    return ArtifactInfo(
        kind="doc_section",
        name=name,
        source_path=source_path,
        summary=summary,
    )


def _confluence_artifact(
    name: str,
    source_path: str,
    space_key: str = "DOCS",
    parent_path: str = "",
) -> ArtifactInfo:
    return ArtifactInfo(
        kind="confluence_page",
        name=name,
        source_path=source_path,
        space_key=space_key,
        parent_path=parent_path,
    )


def _jira_artifact(
    name: str,
    source_path: str,
    issue_type: str = "story",
    epic_key: str = "",
) -> ArtifactInfo:
    return ArtifactInfo(
        kind="jira_issue",
        name=name,
        source_path=source_path,
        issue_type=issue_type,
        epic_key=epic_key,
    )


# ── cluster_markdown_docs ────────────────────────────────────────────────────


class TestClusterMarkdownDocs:
    """Tests for cluster_markdown_docs."""

    def test_empty_input_returns_empty(self, tmp_path):
        result = cluster_markdown_docs([], repo_root=str(tmp_path))
        assert result == []

    def test_singleton_no_links_is_own_cluster(self, tmp_path):
        """A single doc with no outgoing links becomes its own cluster."""
        doc = tmp_path / "solo.md"
        doc.write_text("# Solo Doc\n\nNo links here.\n")
        arts = [_md_artifact("Solo Doc", "solo.md")]

        result = cluster_markdown_docs(arts, repo_root=str(tmp_path))

        assert len(result) == 1
        assert result[0].kind == "doc"
        assert result[0].total_artifacts == 1

    def test_cluster_kind_is_doc(self, tmp_path):
        """All clusters returned from cluster_markdown_docs have kind='doc'."""
        (tmp_path / "a.md").write_text("# A\n")
        arts = [_md_artifact("A", "a.md")]

        result = cluster_markdown_docs(arts, repo_root=str(tmp_path))

        assert all(c.kind == "doc" for c in result)

    def test_two_isolated_components_produce_two_clusters(self, tmp_path):
        """Docs A–B linked together and docs C–D linked together → 2 clusters.

        Graph has two connected components so Louvain must return at least 2
        communities (might produce more if it splits within components, but
        connected components always stay together).
        """
        # Component 1: A links to B
        (tmp_path / "a.md").write_text("# A\n\nSee [B](b.md).\n")
        (tmp_path / "b.md").write_text("# B\n\nSee [A](a.md).\n")
        # Component 2: C links to D
        (tmp_path / "c.md").write_text("# C\n\nSee [D](d.md).\n")
        (tmp_path / "d.md").write_text("# D\n\nSee [C](c.md).\n")

        arts = [
            _md_artifact("A", "a.md"),
            _md_artifact("B", "b.md"),
            _md_artifact("C", "c.md"),
            _md_artifact("D", "d.md"),
        ]

        result = cluster_markdown_docs(arts, repo_root=str(tmp_path))

        # Must produce at least 2 clusters
        assert len(result) >= 2
        # A and B must be in the same cluster; C and D must be in the same cluster
        a_cluster = next(c for c in result if any(a.source_path == "a.md" for a in c.artifacts))
        b_cluster = next(c for c in result if any(a.source_path == "b.md" for a in c.artifacts))
        c_cluster = next(c for c in result if any(a.source_path == "c.md" for a in c.artifacts))
        d_cluster = next(c for c in result if any(a.source_path == "d.md" for a in c.artifacts))
        assert a_cluster.cluster_id == b_cluster.cluster_id
        assert c_cluster.cluster_id == d_cluster.cluster_id
        # And the two groups must be different clusters
        assert a_cluster.cluster_id != c_cluster.cluster_id

    def test_five_docs_two_clusters(self, tmp_path):
        """5-doc markdown set with two dense cross-linked groups → ≥2 clusters.

        Docs 1-2-3 are mutually linked; docs 4-5 are mutually linked; no
        edges cross the two groups.  The connected-component fallback or
        Louvain must separate them.
        """
        (tmp_path / "d1.md").write_text("# D1\n\nLinks: [D2](d2.md) [D3](d3.md).\n")
        (tmp_path / "d2.md").write_text("# D2\n\nLinks: [D1](d1.md) [D3](d3.md).\n")
        (tmp_path / "d3.md").write_text("# D3\n\nLinks: [D1](d1.md) [D2](d2.md).\n")
        (tmp_path / "d4.md").write_text("# D4\n\nSee [D5](d5.md).\n")
        (tmp_path / "d5.md").write_text("# D5\n\nSee [D4](d4.md).\n")

        arts = [_md_artifact(f"D{i}", f"d{i}.md") for i in range(1, 6)]

        result = cluster_markdown_docs(arts, repo_root=str(tmp_path))

        assert len(result) >= 2
        # Group 1 stays together
        group1_ids = {
            c.cluster_id
            for c in result
            if any(a.source_path in {"d1.md", "d2.md", "d3.md"} for a in c.artifacts)
        }
        # Group 2 stays together
        group2_ids = {
            c.cluster_id
            for c in result
            if any(a.source_path in {"d4.md", "d5.md"} for a in c.artifacts)
        }
        assert len(group1_ids) == 1, "docs d1–d3 must be in the same cluster"
        assert len(group2_ids) == 1, "docs d4–d5 must be in the same cluster"
        assert group1_ids != group2_ids

    def test_clusters_cover_all_artifacts(self, tmp_path):
        """Every artifact appears in exactly one cluster."""
        for i in range(1, 4):
            (tmp_path / f"x{i}.md").write_text(f"# X{i}\n")
        arts = [_md_artifact(f"X{i}", f"x{i}.md") for i in range(1, 4)]

        result = cluster_markdown_docs(arts, repo_root=str(tmp_path))

        all_paths = {a.source_path for c in result for a in c.artifacts}
        assert all_paths == {"x1.md", "x2.md", "x3.md"}

    def test_missing_file_still_clustered(self, tmp_path):
        """Artifacts whose files don't exist are still represented in output."""
        arts = [_md_artifact("Ghost", "ghost.md")]

        result = cluster_markdown_docs(arts, repo_root=str(tmp_path))

        # Should produce one cluster even though ghost.md doesn't exist
        assert len(result) == 1

    def test_cluster_ids_are_unique(self, tmp_path):
        """Each cluster has a distinct cluster_id."""
        (tmp_path / "p.md").write_text("# P\n\nSee [Q](q.md).\n")
        (tmp_path / "q.md").write_text("# Q\n")
        (tmp_path / "r.md").write_text("# R\n")
        arts = [_md_artifact(n, f"{n.lower()}.md") for n in ("P", "Q", "R")]

        result = cluster_markdown_docs(arts, repo_root=str(tmp_path))

        ids = [c.cluster_id for c in result]
        assert len(ids) == len(set(ids))


# ── cluster_confluence_pages ─────────────────────────────────────────────────


class TestClusterConfluencePages:
    """Tests for cluster_confluence_pages."""

    def test_empty_input_returns_empty(self, tmp_path):
        result = cluster_confluence_pages([], repo_root=str(tmp_path))
        assert result == []

    def test_cluster_kind_is_confluence(self, tmp_path):
        """All returned clusters have kind='confluence'."""
        (tmp_path / "p1.md").write_text("# Page 1\n")
        arts = [_confluence_artifact("Page 1", "p1.md")]

        result = cluster_confluence_pages(arts, repo_root=str(tmp_path))

        assert all(c.kind == "confluence" for c in result)

    def test_singleton_becomes_own_cluster(self, tmp_path):
        (tmp_path / "lonely.md").write_text("# Lonely\n")
        arts = [_confluence_artifact("Lonely", "lonely.md", parent_path="")]

        result = cluster_confluence_pages(arts, repo_root=str(tmp_path))

        assert len(result) == 1
        assert result[0].total_artifacts == 1

    def test_two_parents_produce_two_clusters(self, tmp_path):
        """Pages under two different parents land in separate clusters."""
        # Parent A's children
        for i in range(1, 4):
            (tmp_path / f"a{i}.md").write_text(f"# A{i}\n")
        # Parent B's children
        for i in range(1, 4):
            (tmp_path / f"b{i}.md").write_text(f"# B{i}\n")

        arts = [
            _confluence_artifact(f"A{i}", f"a{i}.md", parent_path="Parent A")
            for i in range(1, 4)
        ] + [
            _confluence_artifact(f"B{i}", f"b{i}.md", parent_path="Parent B")
            for i in range(1, 4)
        ]

        result = cluster_confluence_pages(arts, repo_root=str(tmp_path))

        assert len(result) >= 2
        # All A pages in same cluster
        a_clusters = {
            c.cluster_id
            for c in result
            if any(a.source_path.startswith("a") for a in c.artifacts)
        }
        b_clusters = {
            c.cluster_id
            for c in result
            if any(a.source_path.startswith("b") for a in c.artifacts)
        }
        assert len(a_clusters) == 1, "A pages must share a cluster"
        assert len(b_clusters) == 1, "B pages must share a cluster"
        assert a_clusters != b_clusters

    def test_link_edges_join_pages(self, tmp_path):
        """A cross-parent inline link creates an edge that can merge clusters."""
        (tmp_path / "pa1.md").write_text("# PA1\n\nSee [PB1](pb1.md).\n")
        (tmp_path / "pb1.md").write_text("# PB1\n")

        arts = [
            _confluence_artifact("PA1", "pa1.md", parent_path="Parent A"),
            _confluence_artifact("PB1", "pb1.md", parent_path="Parent B"),
        ]

        result = cluster_confluence_pages(arts, repo_root=str(tmp_path))

        # Both linked → same cluster (single component)
        assert len(result) == 1

    def test_clusters_cover_all_artifacts(self, tmp_path):
        for i in range(1, 5):
            (tmp_path / f"cp{i}.md").write_text(f"# CP{i}\n")
        arts = [_confluence_artifact(f"CP{i}", f"cp{i}.md") for i in range(1, 5)]

        result = cluster_confluence_pages(arts, repo_root=str(tmp_path))

        all_paths = {a.source_path for c in result for a in c.artifacts}
        assert all_paths == {f"cp{i}.md" for i in range(1, 5)}

    def test_cluster_ids_unique(self, tmp_path):
        for i in range(1, 5):
            (tmp_path / f"u{i}.md").write_text(f"# U{i}\n")
        arts = [
            _confluence_artifact(f"U{i}", f"u{i}.md", parent_path=f"Parent{i%2}")
            for i in range(1, 5)
        ]

        result = cluster_confluence_pages(arts, repo_root=str(tmp_path))

        ids = [c.cluster_id for c in result]
        assert len(ids) == len(set(ids))


# ── cluster_jira_issues ──────────────────────────────────────────────────────


class TestClusterJiraIssues:
    """Tests for cluster_jira_issues."""

    def test_empty_input_returns_empty(self):
        result = cluster_jira_issues([])
        assert result == []

    def test_cluster_kind_is_jira(self):
        """All returned clusters have kind='jira'."""
        arts = [
            _jira_artifact("PROJ-1", "PROJ-1.md", issue_type="epic"),
        ]

        result = cluster_jira_issues(arts)

        assert all(c.kind == "jira" for c in result)

    def test_singleton_epic_is_own_cluster(self):
        arts = [_jira_artifact("EPIC-1", "EPIC-1.md", issue_type="epic")]

        result = cluster_jira_issues(arts)

        assert len(result) == 1
        assert result[0].total_artifacts == 1

    def test_two_epics_produce_two_clusters(self):
        """Stories under epic A and epic B land in separate clusters."""
        arts = [
            _jira_artifact("EPIC-A", "EPIC-A.md", issue_type="epic"),
            _jira_artifact("STORY-1", "STORY-1.md", issue_type="story", epic_key="EPIC-A"),
            _jira_artifact("STORY-2", "STORY-2.md", issue_type="story", epic_key="EPIC-A"),
            _jira_artifact("EPIC-B", "EPIC-B.md", issue_type="epic"),
            _jira_artifact("STORY-3", "STORY-3.md", issue_type="story", epic_key="EPIC-B"),
            _jira_artifact("STORY-4", "STORY-4.md", issue_type="story", epic_key="EPIC-B"),
        ]

        result = cluster_jira_issues(arts)

        assert len(result) >= 2
        # Epic A and its stories share a cluster
        a_clusters = {
            c.cluster_id
            for c in result
            if any(a.source_path in {"EPIC-A.md", "STORY-1.md", "STORY-2.md"} for a in c.artifacts)
        }
        # Epic B and its stories share a cluster
        b_clusters = {
            c.cluster_id
            for c in result
            if any(a.source_path in {"EPIC-B.md", "STORY-3.md", "STORY-4.md"} for a in c.artifacts)
        }
        assert len(a_clusters) == 1, "Epic A group must be in one cluster"
        assert len(b_clusters) == 1, "Epic B group must be in one cluster"
        assert a_clusters != b_clusters

    def test_orphan_story_without_epic_gets_own_cluster(self):
        """A story with no epic_key is not dropped — it forms its own cluster."""
        arts = [
            _jira_artifact("EPIC-1", "EPIC-1.md", issue_type="epic"),
            _jira_artifact("STORY-OK", "STORY-OK.md", issue_type="story", epic_key="EPIC-1"),
            _jira_artifact("STORY-ORPHAN", "STORY-ORPHAN.md", issue_type="story", epic_key=""),
        ]

        result = cluster_jira_issues(arts)

        all_paths = {a.source_path for c in result for a in c.artifacts}
        assert "STORY-ORPHAN.md" in all_paths

    def test_clusters_cover_all_artifacts(self):
        arts = [
            _jira_artifact("EPIC-X", "EPIC-X.md", issue_type="epic"),
            _jira_artifact("STORY-X1", "STORY-X1.md", issue_type="story", epic_key="EPIC-X"),
            _jira_artifact("EPIC-Y", "EPIC-Y.md", issue_type="epic"),
            _jira_artifact("STORY-Y1", "STORY-Y1.md", issue_type="story", epic_key="EPIC-Y"),
        ]

        result = cluster_jira_issues(arts)

        all_paths = {a.source_path for c in result for a in c.artifacts}
        assert all_paths == {"EPIC-X.md", "STORY-X1.md", "EPIC-Y.md", "STORY-Y1.md"}

    def test_cluster_ids_unique(self):
        arts = [
            _jira_artifact("EPIC-A", "EPIC-A.md", issue_type="epic"),
            _jira_artifact("EPIC-B", "EPIC-B.md", issue_type="epic"),
        ]

        result = cluster_jira_issues(arts)

        ids = [c.cluster_id for c in result]
        assert len(ids) == len(set(ids))

    def test_total_artifacts_correct(self):
        arts = [
            _jira_artifact("EPIC-1", "EPIC-1.md", issue_type="epic"),
            _jira_artifact("STORY-1", "STORY-1.md", issue_type="story", epic_key="EPIC-1"),
            _jira_artifact("STORY-2", "STORY-2.md", issue_type="story", epic_key="EPIC-1"),
        ]

        result = cluster_jira_issues(arts)

        total = sum(c.total_artifacts for c in result)
        assert total == 3


# ── Duplicate source_path preservation ───────────────────────────────────────


class TestDuplicateSourcePath:
    """Multiple artifacts sharing a source_path must all survive clustering.

    This guards against a regression where the path→artifact map collapsed
    duplicates: doc_section artifacts (one per heading) often share the same
    file path, and dropping any of them silently loses page content.
    """

    def test_markdown_duplicate_source_path_preserved(self, tmp_path):
        doc = tmp_path / "shared.md"
        doc.write_text("# Shared\n\n## A\n\nText.\n\n## B\n\nMore.\n")
        arts = [
            _md_artifact("Shared#A", "shared.md", "Heading A section"),
            _md_artifact("Shared#B", "shared.md", "Heading B section"),
        ]

        result = cluster_markdown_docs(arts, repo_root=str(tmp_path))

        total = sum(c.total_artifacts for c in result)
        assert total == len(arts)

    def test_confluence_duplicate_source_path_preserved(self, tmp_path):
        page = tmp_path / "page.md"
        page.write_text("# Page\n\nContent.\n")
        arts = [
            _confluence_artifact("Page#Intro", "page.md", parent_path="Eng"),
            _confluence_artifact("Page#Body", "page.md", parent_path="Eng"),
        ]

        result = cluster_confluence_pages(arts, repo_root=str(tmp_path))

        total = sum(c.total_artifacts for c in result)
        assert total == len(arts)

    def test_jira_duplicate_source_path_preserved(self):
        arts = [
            _jira_artifact("EPIC-1#a", "EPIC-1.md", issue_type="epic"),
            _jira_artifact("EPIC-1#b", "EPIC-1.md", issue_type="epic"),
        ]

        result = cluster_jira_issues(arts)

        total = sum(c.total_artifacts for c in result)
        assert total == len(arts)
