"""Tests for ArtifactInfo and Cluster types (#229 — source-kind unification).

Covers:
- ArtifactInfo construction for each kind
- Cluster.kind discriminator
- Cluster.artifacts API
- Backwards-compat: DirCluster alias still importable + constructible
- StructureSkeleton.clusters heterogeneous list
"""

from __future__ import annotations

import pytest

from app.core.wiki_structure_planner.structure_skeleton import (
    ArtifactInfo,
    Cluster,
    DirCluster,
    StructureSkeleton,
    SymbolInfo,
)

# ── ArtifactInfo construction ─────────────────────────────────────────────────


class TestArtifactInfoConstruction:
    def test_symbol_kind(self):
        a = ArtifactInfo(
            kind="symbol",
            name="MyClass",
            source_path="src/models/foo.py",
            layer="core_type",
            connections=5,
            summary="A class",
        )
        assert a.kind == "symbol"
        assert a.name == "MyClass"
        assert a.source_path == "src/models/foo.py"
        assert a.layer == "core_type"
        assert a.connections == 5
        assert a.summary == "A class"

    def test_doc_section_kind(self):
        a = ArtifactInfo(
            kind="doc_section",
            name="Getting Started",
            source_path="docs/intro.md",
            layer="",
            connections=0,
            summary="Introduction chapter",
        )
        assert a.kind == "doc_section"
        assert a.source_path == "docs/intro.md"
        assert a.layer == ""

    def test_confluence_page_kind(self):
        a = ArtifactInfo(
            kind="confluence_page",
            name="Auth Flow",
            source_path="confluence/AUTH/auth-flow.md",
            layer="",
            connections=3,
            summary="Confluence auth guide",
        )
        assert a.kind == "confluence_page"

    def test_jira_issue_kind(self):
        a = ArtifactInfo(
            kind="jira_issue",
            name="PROJ-123",
            source_path="jira/PROJ/PROJ-123.md",
            layer="",
            connections=0,
            summary="Epic: unified planner",
        )
        assert a.kind == "jira_issue"

    def test_invalid_kind_raises(self):
        with pytest.raises((ValueError, TypeError)):
            ArtifactInfo(
                kind="invalid_kind",  # type: ignore[arg-type]
                name="X",
                source_path="x.py",
                layer="",
                connections=0,
                summary="",
            )

    def test_optional_fields_have_defaults(self):
        """layer, connections, and summary should have sensible defaults."""
        a = ArtifactInfo(kind="doc_section", name="Intro", source_path="README.md")
        assert a.layer == ""
        assert a.connections == 0
        assert a.summary == ""

    def test_confluence_metadata_fields(self):
        a = ArtifactInfo(
            kind="confluence_page",
            name="Auth Guide",
            source_path="exported/auth.md",
            space_key="ENG",
            parent_path="Engineering / Security",
        )
        assert a.space_key == "ENG"
        assert a.parent_path == "Engineering / Security"
        # Jira-only fields stay empty for confluence
        assert a.issue_type == ""
        assert a.epic_key == ""

    def test_jira_metadata_fields(self):
        a = ArtifactInfo(
            kind="jira_issue",
            name="PROJ-123",
            source_path="exported/PROJ-123.md",
            issue_type="story",
            epic_key="PROJ-42",
        )
        assert a.issue_type == "story"
        assert a.epic_key == "PROJ-42"
        assert a.space_key == ""
        assert a.parent_path == ""

    def test_metadata_fields_default_to_empty_for_code(self):
        a = ArtifactInfo(kind="symbol", name="Foo", source_path="src/foo.py")
        assert a.space_key == ""
        assert a.parent_path == ""
        assert a.issue_type == ""
        assert a.epic_key == ""


# ── Cluster construction ──────────────────────────────────────────────────────


class TestClusterConstruction:
    def _make_symbol_artifact(self, name: str = "Foo", path: str = "src/foo.py") -> ArtifactInfo:
        return ArtifactInfo(
            kind="symbol",
            name=name,
            source_path=path,
            layer="core_type",
            connections=2,
            summary="",
        )

    def _make_doc_artifact(self, name: str = "Guide", path: str = "docs/guide.md") -> ArtifactInfo:
        return ArtifactInfo(
            kind="doc_section",
            name=name,
            source_path=path,
            layer="",
            connections=0,
            summary="",
        )

    def test_code_cluster_kind(self):
        arts = [self._make_symbol_artifact()]
        c = Cluster(
            cluster_id=1,
            kind="code",
            dirs=["src"],
            artifacts=arts,
            total_artifacts=1,
            primary_languages=["py"],
            depth_range=(0, 0),
        )
        assert c.kind == "code"
        assert c.artifacts == arts
        assert c.total_artifacts == 1

    def test_doc_cluster_kind(self):
        arts = [self._make_doc_artifact()]
        c = Cluster(
            cluster_id=2,
            kind="doc",
            dirs=["docs"],
            artifacts=arts,
            total_artifacts=1,
            primary_languages=[],
            depth_range=(0, 0),
        )
        assert c.kind == "doc"

    def test_confluence_cluster_kind(self):
        arts = [self._make_doc_artifact(path="confluence/space/page.md")]
        c = Cluster(
            cluster_id=3,
            kind="confluence",
            dirs=["confluence/space"],
            artifacts=arts,
            total_artifacts=1,
            primary_languages=[],
            depth_range=(1, 1),
        )
        assert c.kind == "confluence"

    def test_jira_cluster_kind(self):
        arts = [
            ArtifactInfo(
                kind="jira_issue",
                name="PROJ-1",
                source_path="jira/PROJ-1.md",
                layer="",
                connections=0,
                summary="",
            )
        ]
        c = Cluster(
            cluster_id=4,
            kind="jira",
            dirs=["jira"],
            artifacts=arts,
            total_artifacts=1,
            primary_languages=[],
            depth_range=(0, 0),
        )
        assert c.kind == "jira"

    def test_cluster_dirs_preserved(self):
        arts = [self._make_symbol_artifact()]
        c = Cluster(
            cluster_id=1,
            kind="code",
            dirs=["a", "b"],
            artifacts=arts,
            total_artifacts=1,
            primary_languages=["py"],
            depth_range=(0, 1),
        )
        assert c.dirs == ["a", "b"]

    def test_cluster_depth_range(self):
        arts = [self._make_symbol_artifact()]
        c = Cluster(
            cluster_id=1,
            kind="code",
            dirs=["a/b/c"],
            artifacts=arts,
            total_artifacts=1,
            primary_languages=["py"],
            depth_range=(2, 2),
        )
        assert c.depth_range == (2, 2)

    def test_heterogeneous_artifacts_allowed(self):
        """A single cluster can hold artifacts of mixed kinds."""
        arts = [
            self._make_symbol_artifact(),
            self._make_doc_artifact(),
        ]
        c = Cluster(
            cluster_id=5,
            kind="code",
            dirs=["src"],
            artifacts=arts,
            total_artifacts=2,
            primary_languages=["py"],
            depth_range=(0, 0),
        )
        kinds = {a.kind for a in c.artifacts}
        assert "symbol" in kinds
        assert "doc_section" in kinds

    def test_invalid_cluster_kind_raises(self):
        arts = [self._make_symbol_artifact()]
        with pytest.raises(ValueError, match="Invalid Cluster.kind"):
            Cluster(
                cluster_id=99,
                kind="typo",  # type: ignore[arg-type]
                dirs=["src"],
                artifacts=arts,
                total_artifacts=1,
                primary_languages=["py"],
                depth_range=(0, 0),
            )


# ── DirCluster backwards-compat alias ────────────────────────────────────────


class TestDirClusterBackwardsCompat:
    """DirCluster must still be importable and constructible for callers
    that haven't been migrated to Cluster yet."""

    def _make_sym(self) -> SymbolInfo:
        return SymbolInfo(
            name="Foo",
            type="class",
            rel_path="src/foo.py",
            layer="core_type",
            connections=2,
            docstring="",
        )

    def test_dircluster_is_importable(self):
        assert DirCluster is not None

    def test_dircluster_construction_with_symbols(self):
        sym = self._make_sym()
        dc = DirCluster(
            cluster_id=1,
            dirs=["src"],
            symbols=[sym],
            total_symbols=1,
            primary_languages=["py"],
            depth_range=(0, 0),
        )
        assert dc.cluster_id == 1
        assert dc.symbols == [sym]
        assert dc.total_symbols == 1

    def test_dircluster_is_cluster_subtype(self):
        """DirCluster instances should be usable wherever Cluster is expected."""
        sym = self._make_sym()
        dc = DirCluster(
            cluster_id=1,
            dirs=["src"],
            symbols=[sym],
            total_symbols=1,
            primary_languages=["py"],
            depth_range=(0, 0),
        )
        assert isinstance(dc, Cluster)

    def test_dircluster_kind_is_code(self):
        sym = self._make_sym()
        dc = DirCluster(
            cluster_id=1,
            dirs=["src"],
            symbols=[sym],
            total_symbols=1,
            primary_languages=["py"],
            depth_range=(0, 0),
        )
        assert dc.kind == "code"

    def test_dircluster_artifacts_mirrors_symbols(self):
        """DirCluster.artifacts should return ArtifactInfo wrappers around symbols."""
        sym = self._make_sym()
        dc = DirCluster(
            cluster_id=1,
            dirs=["src"],
            symbols=[sym],
            total_symbols=1,
            primary_languages=["py"],
            depth_range=(0, 0),
        )
        assert len(dc.artifacts) == 1
        art = dc.artifacts[0]
        assert art.kind == "symbol"
        assert art.name == sym.name
        assert art.source_path == sym.rel_path
        assert art.layer == sym.layer
        assert art.connections == sym.connections


# ── StructureSkeleton.clusters ────────────────────────────────────────────────


class TestStructureSkeletonClusters:
    def _make_code_cluster(self, cid: int = 1) -> Cluster:
        art = ArtifactInfo(
            kind="symbol",
            name=f"Cls{cid}",
            source_path=f"src/mod{cid}.py",
            layer="core_type",
            connections=1,
            summary="",
        )
        return Cluster(
            cluster_id=cid,
            kind="code",
            dirs=[f"src/mod{cid}"],
            artifacts=[art],
            total_artifacts=1,
            primary_languages=["py"],
            depth_range=(1, 1),
        )

    def _make_doc_cluster(self, cid: int = 10) -> Cluster:
        art = ArtifactInfo(
            kind="doc_section",
            name="Intro",
            source_path=f"docs/intro{cid}.md",
            layer="",
            connections=0,
            summary="",
        )
        return Cluster(
            cluster_id=cid,
            kind="doc",
            dirs=["docs"],
            artifacts=[art],
            total_artifacts=1,
            primary_languages=[],
            depth_range=(0, 0),
        )

    def test_skeleton_accepts_clusters_list(self):
        code = self._make_code_cluster(1)
        doc = self._make_doc_cluster(2)
        skel = StructureSkeleton(
            code_clusters=[],
            doc_clusters=[],
            clusters=[code, doc],
            total_arch_symbols=1,
            total_dirs_covered=1,
            total_dirs_in_repo=2,
            repo_languages=["py"],
            effective_depth=3,
        )
        assert len(skel.clusters) == 2

    def test_skeleton_clusters_mixed_kinds(self):
        code = self._make_code_cluster(1)
        doc = self._make_doc_cluster(2)
        skel = StructureSkeleton(
            code_clusters=[],
            doc_clusters=[],
            clusters=[code, doc],
            total_arch_symbols=1,
            total_dirs_covered=1,
            total_dirs_in_repo=2,
            repo_languages=["py"],
            effective_depth=3,
        )
        kinds = {c.kind for c in skel.clusters}
        assert kinds == {"code", "doc"}

    def test_skeleton_backward_compat_no_clusters(self):
        """Existing code that doesn't pass clusters= still works."""
        skel = StructureSkeleton(
            code_clusters=[],
            doc_clusters=[],
            total_arch_symbols=0,
            total_dirs_covered=0,
            total_dirs_in_repo=0,
            repo_languages=[],
            effective_depth=3,
        )
        assert skel.clusters == []

    def test_skeleton_code_clusters_still_works(self):
        """Existing code that passes code_clusters= still works."""
        sym = SymbolInfo(
            name="X",
            type="class",
            rel_path="src/x.py",
            layer="core_type",
            connections=1,
            docstring="",
        )
        dc = DirCluster(
            cluster_id=1,
            dirs=["src"],
            symbols=[sym],
            total_symbols=1,
            primary_languages=["py"],
            depth_range=(0, 0),
        )
        skel = StructureSkeleton(
            code_clusters=[dc],
            doc_clusters=[],
            total_arch_symbols=1,
            total_dirs_covered=1,
            total_dirs_in_repo=1,
            repo_languages=["py"],
            effective_depth=3,
        )
        assert len(skel.code_clusters) == 1
        assert skel.code_clusters[0] is dc
