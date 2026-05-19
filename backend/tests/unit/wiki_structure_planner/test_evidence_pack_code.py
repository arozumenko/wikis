"""Unit tests for wiki_structure_planner/evidence.py — code cluster evidence packs (#232).

Covers:
- Pack construction with a small fabricated cluster
- Top-K signature selection sorted by (layer priority, connections desc)
- File-head reading restricted to entry_point / public_api files
- README excerpt from cluster dirs
- CREATE TABLE extraction from .sql files
- Truncation by layer priority when over budget
- Path safety: reject inputs that would escape repo_root
"""

from __future__ import annotations

import textwrap
from pathlib import Path

import pytest

from app.core.wiki_structure_planner.evidence import (
    EvidencePack,
    build_pack,
)
from app.core.wiki_structure_planner.structure_skeleton import (
    ArtifactInfo,
    Cluster,
)

# ── Helpers ────────────────────────────────────────────────────────────────────


def _make_cluster(
    artifacts: list[ArtifactInfo],
    dirs: list[str] | None = None,
    cluster_id: int = 1,
    kind: str = "code",
) -> Cluster:
    return Cluster(
        cluster_id=cluster_id,
        kind=kind,  # type: ignore[arg-type]
        dirs=dirs or ["src/core"],
        artifacts=artifacts,
        total_artifacts=len(artifacts),
        primary_languages=["py"],
        depth_range=(1, 2),
    )


def _artifact(
    name: str,
    source_path: str,
    layer: str,
    connections: int = 0,
    summary: str = "",
) -> ArtifactInfo:
    return ArtifactInfo(
        kind="symbol",
        name=name,
        source_path=source_path,
        layer=layer,
        connections=connections,
        summary=summary,
    )


# ── EvidencePack dataclass ─────────────────────────────────────────────────────


class TestEvidencePackDataclass:
    def test_constructs_with_defaults(self):
        pack = EvidencePack(cluster_id=1, kind="code")
        assert pack.cluster_id == 1
        assert pack.kind == "code"
        assert pack.signatures == []
        assert pack.file_heads == []
        assert pack.readme_excerpt == ""
        assert pack.sql_blocks == []

    def test_serialize_returns_string(self):
        pack = EvidencePack(cluster_id=1, kind="code")
        result = pack.serialize()
        assert isinstance(result, str)

    def test_serialize_empty_pack_carries_cluster_metadata(self):
        # Empty packs must still emit the cluster header so downstream
        # consumers can identify which cluster the pack is for.
        pack = EvidencePack(cluster_id=2, kind="code")
        result = pack.serialize()
        assert "cluster=2" in result
        assert "kind=code" in result

    def test_serialize_includes_signatures(self):
        pack = EvidencePack(
            cluster_id=1,
            kind="code",
            signatures=[("MyClass", "src/core/thing.py", "public_api", "A class")],
        )
        result = pack.serialize()
        assert "MyClass" in result

    def test_serialize_includes_file_head(self):
        pack = EvidencePack(
            cluster_id=1,
            kind="code",
            file_heads=[("src/main.py", "def main():\n    pass\n")],
        )
        result = pack.serialize()
        assert "src/main.py" in result
        assert "def main()" in result

    def test_serialize_includes_readme_excerpt(self):
        pack = EvidencePack(cluster_id=1, kind="code", readme_excerpt="# Auth Module\nHandles auth.")
        result = pack.serialize()
        assert "Auth Module" in result

    def test_serialize_includes_sql_blocks(self):
        sql = "CREATE TABLE users (id INTEGER PRIMARY KEY);"
        pack = EvidencePack(cluster_id=1, kind="code", sql_blocks=[sql])
        result = pack.serialize()
        assert "CREATE TABLE" in result


# ── build_pack — basic construction ───────────────────────────────────────────


class TestBuildPackBasic:
    def test_returns_evidence_pack(self, tmp_path):
        cluster = _make_cluster([_artifact("Foo", "src/core/foo.py", "public_api", connections=3)])
        pack = build_pack(cluster, repo_root=str(tmp_path))
        assert isinstance(pack, EvidencePack)

    def test_pack_cluster_id_matches(self, tmp_path):
        cluster = _make_cluster(
            [_artifact("Bar", "src/bar.py", "core_type")],
            cluster_id=42,
        )
        pack = build_pack(cluster, repo_root=str(tmp_path))
        assert pack.cluster_id == 42

    def test_pack_kind_matches(self, tmp_path):
        cluster = _make_cluster(
            [_artifact("Baz", "src/baz.py", "infrastructure")],
        )
        pack = build_pack(cluster, repo_root=str(tmp_path))
        assert pack.kind == "code"

    def test_empty_artifacts_returns_valid_pack(self, tmp_path):
        cluster = _make_cluster([])
        pack = build_pack(cluster, repo_root=str(tmp_path))
        assert isinstance(pack, EvidencePack)
        assert pack.signatures == []


# ── build_pack — signature selection ──────────────────────────────────────────


class TestSignatureSelection:
    def test_signatures_extracted_from_artifacts(self, tmp_path):
        cluster = _make_cluster(
            [
                _artifact("Foo", "src/foo.py", "public_api", connections=5, summary="Does foo"),
                _artifact("Bar", "src/bar.py", "core_type", connections=2, summary="Does bar"),
            ]
        )
        pack = build_pack(cluster, repo_root=str(tmp_path))
        names = [s[0] for s in pack.signatures]
        assert "Foo" in names
        assert "Bar" in names

    def test_entry_point_ranked_first(self, tmp_path):
        cluster = _make_cluster(
            [
                _artifact("CoreUtil", "src/util.py", "core_type", connections=10),
                _artifact("MainEntry", "src/main.py", "entry_point", connections=1),
                _artifact("PubApi", "src/api.py", "public_api", connections=5),
            ]
        )
        pack = build_pack(cluster, repo_root=str(tmp_path))
        names = [s[0] for s in pack.signatures]
        assert names[0] == "MainEntry"

    def test_public_api_ranked_before_core_type(self, tmp_path):
        cluster = _make_cluster(
            [
                _artifact("CoreThing", "src/core.py", "core_type", connections=20),
                _artifact("ApiHandler", "src/api.py", "public_api", connections=1),
            ]
        )
        pack = build_pack(cluster, repo_root=str(tmp_path))
        names = [s[0] for s in pack.signatures]
        assert names.index("ApiHandler") < names.index("CoreThing")

    def test_within_same_layer_higher_connections_ranked_first(self, tmp_path):
        cluster = _make_cluster(
            [
                _artifact("LowConn", "src/a.py", "public_api", connections=1),
                _artifact("HighConn", "src/b.py", "public_api", connections=50),
            ]
        )
        pack = build_pack(cluster, repo_root=str(tmp_path))
        names = [s[0] for s in pack.signatures]
        assert names[0] == "HighConn"

    def test_top_k_limits_signatures(self, tmp_path):
        artifacts = [_artifact(f"Sym{i}", f"src/sym{i}.py", "core_type", connections=i) for i in range(30)]
        cluster = _make_cluster(artifacts)
        pack = build_pack(cluster, repo_root=str(tmp_path), top_k=10)
        assert len(pack.signatures) <= 10

    def test_signature_tuple_has_expected_fields(self, tmp_path):
        cluster = _make_cluster(
            [
                _artifact("Foo", "src/foo.py", "public_api", connections=3, summary="Does foo"),
            ]
        )
        pack = build_pack(cluster, repo_root=str(tmp_path))
        assert len(pack.signatures) >= 1
        sig = pack.signatures[0]
        # (name, source_path, layer, summary)
        assert sig[0] == "Foo"
        assert sig[1] == "src/foo.py"
        assert sig[2] == "public_api"
        assert sig[3] == "Does foo"


# ── build_pack — file head reading ────────────────────────────────────────────


class TestFileHeadReading:
    def _setup_files(self, tmp_path: Path, files: dict[str, str]) -> None:
        for rel, content in files.items():
            p = tmp_path / rel
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text(content)

    def test_reads_entry_point_file_head(self, tmp_path):
        content = "\n".join(f"line {i}" for i in range(60))
        self._setup_files(tmp_path, {"src/main.py": content})
        cluster = _make_cluster(
            [
                _artifact("Main", "src/main.py", "entry_point", connections=1),
            ]
        )
        pack = build_pack(cluster, repo_root=str(tmp_path))
        assert len(pack.file_heads) >= 1
        head_paths = [fh[0] for fh in pack.file_heads]
        assert "src/main.py" in head_paths

    def test_reads_public_api_file_head(self, tmp_path):
        content = "def api_func():\n    pass\n"
        self._setup_files(tmp_path, {"src/api.py": content})
        cluster = _make_cluster(
            [
                _artifact("ApiFunc", "src/api.py", "public_api", connections=2),
            ]
        )
        pack = build_pack(cluster, repo_root=str(tmp_path))
        head_paths = [fh[0] for fh in pack.file_heads]
        assert "src/api.py" in head_paths

    def test_skips_core_type_file_head(self, tmp_path):
        content = "class CoreThing:\n    pass\n"
        self._setup_files(tmp_path, {"src/core.py": content})
        cluster = _make_cluster(
            [
                _artifact("CoreThing", "src/core.py", "core_type", connections=3),
            ]
        )
        pack = build_pack(cluster, repo_root=str(tmp_path))
        head_paths = [fh[0] for fh in pack.file_heads]
        assert "src/core.py" not in head_paths

    def test_head_limited_to_40_lines(self, tmp_path):
        content = "\n".join(f"line {i}" for i in range(100))
        self._setup_files(tmp_path, {"src/main.py": content})
        cluster = _make_cluster(
            [
                _artifact("Main", "src/main.py", "entry_point"),
            ]
        )
        pack = build_pack(cluster, repo_root=str(tmp_path))
        head_text = next(fh[1] for fh in pack.file_heads if fh[0] == "src/main.py")
        assert len(head_text.splitlines()) <= 40

    def test_deduplicates_same_file_multiple_symbols(self, tmp_path):
        content = "def foo():\n    pass\ndef bar():\n    pass\n"
        self._setup_files(tmp_path, {"src/api.py": content})
        cluster = _make_cluster(
            [
                _artifact("Foo", "src/api.py", "public_api", connections=1),
                _artifact("Bar", "src/api.py", "public_api", connections=2),
            ]
        )
        pack = build_pack(cluster, repo_root=str(tmp_path))
        head_paths = [fh[0] for fh in pack.file_heads]
        assert head_paths.count("src/api.py") == 1

    def test_missing_file_omitted_gracefully(self, tmp_path):
        # File referenced by artifact doesn't exist on disk — should not raise.
        cluster = _make_cluster(
            [
                _artifact("Ghost", "src/ghost.py", "entry_point"),
            ]
        )
        pack = build_pack(cluster, repo_root=str(tmp_path))
        # Should not raise; file simply absent from heads
        assert isinstance(pack, EvidencePack)


# ── build_pack — README excerpt ───────────────────────────────────────────────


class TestReadmeExcerpt:
    def test_readme_found_in_cluster_dir(self, tmp_path):
        readme = tmp_path / "src" / "core" / "README.md"
        readme.parent.mkdir(parents=True, exist_ok=True)
        readme.write_text("# Core Module\nThis is the core.\n")
        cluster = _make_cluster(
            [_artifact("Foo", "src/core/foo.py", "public_api")],
            dirs=["src/core"],
        )
        pack = build_pack(cluster, repo_root=str(tmp_path))
        assert "Core Module" in pack.readme_excerpt

    def test_readme_limited_to_60_lines(self, tmp_path):
        content = "\n".join(f"line {i}" for i in range(100))
        readme = tmp_path / "src" / "core" / "README.md"
        readme.parent.mkdir(parents=True, exist_ok=True)
        readme.write_text(content)
        cluster = _make_cluster(
            [_artifact("Foo", "src/core/foo.py", "public_api")],
            dirs=["src/core"],
        )
        pack = build_pack(cluster, repo_root=str(tmp_path))
        assert len(pack.readme_excerpt.splitlines()) <= 60

    def test_no_readme_returns_empty_string(self, tmp_path):
        cluster = _make_cluster(
            [_artifact("Foo", "src/core/foo.py", "public_api")],
            dirs=["src/core"],
        )
        pack = build_pack(cluster, repo_root=str(tmp_path))
        assert pack.readme_excerpt == ""

    def test_readme_case_insensitive(self, tmp_path):
        readme = tmp_path / "src" / "readme.md"
        readme.parent.mkdir(parents=True, exist_ok=True)
        readme.write_text("# README lowercase\n")
        cluster = _make_cluster(
            [_artifact("Foo", "src/foo.py", "public_api")],
            dirs=["src"],
        )
        pack = build_pack(cluster, repo_root=str(tmp_path))
        assert "README lowercase" in pack.readme_excerpt

    def test_picks_first_cluster_dir_readme(self, tmp_path):
        # When multiple dirs have READMEs, the first dir in cluster.dirs wins.
        for d in ["src/a", "src/b"]:
            readme = tmp_path / d / "README.md"
            readme.parent.mkdir(parents=True, exist_ok=True)
            readme.write_text(f"# README in {d}\n")
        cluster = _make_cluster(
            [
                _artifact("Foo", "src/a/foo.py", "public_api"),
                _artifact("Bar", "src/b/bar.py", "public_api"),
            ],
            dirs=["src/a", "src/b"],
        )
        pack = build_pack(cluster, repo_root=str(tmp_path))
        # Determinism: src/a is listed first in cluster.dirs → its README wins.
        assert "README in src/a" in pack.readme_excerpt
        assert "README in src/b" not in pack.readme_excerpt


# ── build_pack — SQL CREATE TABLE extraction ──────────────────────────────────


class TestSqlExtraction:
    def _setup_sql(self, tmp_path: Path, rel_path: str, content: str) -> None:
        p = tmp_path / rel_path
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(content)

    def test_extracts_create_table_block(self, tmp_path):
        sql = textwrap.dedent("""\
            -- schema
            CREATE TABLE users (
                id INTEGER PRIMARY KEY,
                name TEXT NOT NULL
            );
            """)
        self._setup_sql(tmp_path, "src/schema.sql", sql)
        cluster = _make_cluster(
            [_artifact("UserModel", "src/models.py", "core_type")],
            dirs=["src"],
        )
        pack = build_pack(cluster, repo_root=str(tmp_path))
        assert any("CREATE TABLE" in block for block in pack.sql_blocks)
        assert any("users" in block for block in pack.sql_blocks)

    def test_multiple_create_tables_extracted(self, tmp_path):
        sql = textwrap.dedent("""\
            CREATE TABLE users (id INTEGER PRIMARY KEY);
            CREATE TABLE posts (id INTEGER PRIMARY KEY, user_id INTEGER);
            """)
        self._setup_sql(tmp_path, "src/schema.sql", sql)
        cluster = _make_cluster(
            [_artifact("Foo", "src/foo.py", "core_type")],
            dirs=["src"],
        )
        pack = build_pack(cluster, repo_root=str(tmp_path))
        combined = " ".join(pack.sql_blocks)
        assert "users" in combined
        assert "posts" in combined

    def test_no_sql_file_returns_empty_blocks(self, tmp_path):
        cluster = _make_cluster(
            [_artifact("Foo", "src/foo.py", "public_api")],
            dirs=["src"],
        )
        pack = build_pack(cluster, repo_root=str(tmp_path))
        assert pack.sql_blocks == []

    def test_ignores_non_sql_files(self, tmp_path):
        p = tmp_path / "src" / "migrations.py"
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text("CREATE TABLE users (id INTEGER);")  # Python, not SQL
        cluster = _make_cluster(
            [_artifact("Foo", "src/foo.py", "core_type")],
            dirs=["src"],
        )
        pack = build_pack(cluster, repo_root=str(tmp_path))
        # Python file not scanned for SQL blocks
        assert pack.sql_blocks == []


# ── build_pack — truncation by layer priority ─────────────────────────────────


class TestTruncation:
    def test_truncation_keeps_under_token_budget(self, tmp_path):
        # Create a cluster with many artifacts to force truncation.
        artifacts = [
            _artifact(f"Sym{i}", f"src/sym{i}.py", "infrastructure", connections=i, summary="x" * 200)
            for i in range(100)
        ]
        cluster = _make_cluster(artifacts)
        pack = build_pack(cluster, repo_root=str(tmp_path), token_budget=1500)
        result = pack.serialize()
        # Rough token estimate: ~4 chars per token
        approx_tokens = len(result) / 4
        # Allow some slack (up to 2x) for edge cases in estimation
        assert approx_tokens < 1500 * 2

    def test_truncation_preserves_entry_point_over_infrastructure(self, tmp_path):
        # Mix of high-priority and low-priority symbols; truncation should keep entry_point.
        artifacts = [
            _artifact(f"Infra{i}", f"src/infra{i}.py", "infrastructure", connections=100 + i, summary="y" * 300)
            for i in range(20)
        ] + [_artifact("EntryMain", "src/main.py", "entry_point", connections=1, summary="main")]
        cluster = _make_cluster(artifacts)
        pack = build_pack(cluster, repo_root=str(tmp_path), token_budget=300)
        names = [s[0] for s in pack.signatures]
        assert "EntryMain" in names

    def test_total_pack_respects_budget_with_large_readme(self, tmp_path):
        # Large README + file heads + SQL — total pack must stay near budget.
        (tmp_path / "src").mkdir()
        (tmp_path / "src" / "README.md").write_text("README line.\n" * 500)
        (tmp_path / "src" / "schema.sql").write_text("CREATE TABLE t1 (id INT);\n" * 200)
        (tmp_path / "src" / "main.py").write_text("def main():\n    pass\n" * 200)
        cluster = _make_cluster(
            [_artifact("Main", "src/main.py", "entry_point")],
            dirs=["src"],
        )
        pack = build_pack(cluster, repo_root=str(tmp_path), token_budget=400)
        # Allow modest overshoot (1.5x) for token-estimation slack but
        # enforce that giant inputs no longer bloat the pack to 10x+.
        assert len(pack.serialize()) / 4 < 400 * 1.5


# ── build_pack — heterogeneous cluster filter ─────────────────────────────────


class TestSymbolFilter:
    def test_doc_artifacts_excluded_from_code_pack(self, tmp_path):
        # A mixed cluster (symbol + doc_section) must produce a code pack
        # with ONLY symbol signatures — doc artifacts don't belong here.
        from app.core.wiki_structure_planner.structure_skeleton import ArtifactInfo

        cluster = _make_cluster(
            [
                _artifact("RealCode", "src/code.py", "public_api"),
                ArtifactInfo(
                    kind="doc_section",
                    name="Docs Page",
                    source_path="docs/intro.md",
                    summary="intro",
                ),
                ArtifactInfo(
                    kind="jira_issue",
                    name="PROJ-123",
                    source_path="jira/PROJ-123.md",
                    issue_type="story",
                ),
            ]
        )
        pack = build_pack(cluster, repo_root=str(tmp_path))
        names = [s[0] for s in pack.signatures]
        assert "RealCode" in names
        assert "Docs Page" not in names
        assert "PROJ-123" not in names


# ── Path safety ───────────────────────────────────────────────────────────────


class TestPathSafety:
    def test_traversal_in_source_path_rejected(self, tmp_path):
        # Artifact pointing outside repo_root via ../traversal.
        cluster = _make_cluster(
            [
                _artifact("Evil", "../../etc/passwd", "entry_point"),
            ]
        )
        # Should not raise; traversal path simply skipped.
        pack = build_pack(cluster, repo_root=str(tmp_path))
        head_paths = [fh[0] for fh in pack.file_heads]
        assert "../../etc/passwd" not in head_paths

    def test_absolute_source_path_rejected(self, tmp_path):
        cluster = _make_cluster(
            [
                _artifact("AbsPath", "/etc/passwd", "public_api"),
            ]
        )
        pack = build_pack(cluster, repo_root=str(tmp_path))
        head_paths = [fh[0] for fh in pack.file_heads]
        assert "/etc/passwd" not in head_paths

    def test_dir_traversal_in_readme_rejected(self, tmp_path):
        # Cluster with a dir that resolves outside repo_root.
        cluster = Cluster(
            cluster_id=1,
            kind="code",
            dirs=["../../etc"],
            artifacts=[_artifact("Foo", "src/foo.py", "public_api")],
            total_artifacts=1,
            primary_languages=["py"],
            depth_range=(0, 1),
        )
        # Should not raise; readme search simply finds nothing outside root.
        pack = build_pack(cluster, repo_root=str(tmp_path))
        assert pack.readme_excerpt == ""

    def test_symlink_escape_rejected(self, tmp_path):
        # Create a symlink inside tmp_path pointing outside.
        link = tmp_path / "src" / "evil_link.py"
        link.parent.mkdir(parents=True, exist_ok=True)
        try:
            link.symlink_to("/etc/passwd")
        except (OSError, NotImplementedError):
            pytest.skip("symlinks not supported on this platform")
        cluster = _make_cluster(
            [
                _artifact("Evil", "src/evil_link.py", "entry_point"),
            ]
        )
        pack = build_pack(cluster, repo_root=str(tmp_path))
        # File head should not contain /etc/passwd content.
        for _, head in pack.file_heads:
            assert "root:" not in head  # /etc/passwd starts with "root:"
