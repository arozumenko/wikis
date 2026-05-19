"""End-to-end integration tests for the unified wiki pipeline (#243).

Tests run the full pipeline (planner → writer → gate → verifier) against tiny
fixture repos in backend/tests/fixtures/unified_pipeline/.  All LLM calls use
a deterministic ``FakeLLM`` that returns well-formed JSON for both the planner
and the writer roles, detected via the system-prompt prefix.

Fixture repos
-------------
tiny_code_repo/     — 5-file Python auth module (code cluster)
tiny_doc_repo/      — 4-file Markdown doc repo with internal links
tiny_confluence_export/ — 3 Confluence-style markdown pages
tiny_jira_export/   — 2 epics with 1 story each

Each fixture test asserts:
- ``len(result.pages) > 0``
- Every page has non-empty ``title`` and ``markdown``
- No ``target_*`` metadata field is entirely absent (list may be empty for doc repos)

Adversarial tests
-----------------
- Code repo: writer hallucinates a symbol not in any trace →
  gate or verifier strips the paragraph → final markdown does NOT contain
  the hallucinated symbol name.
"""

from __future__ import annotations

import json
import os
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage
from langchain_core.outputs import ChatGeneration, ChatResult

from app.core.agents.unified_wiki_pipeline import (
    UnifiedPipelineResult,
    run_unified_pipeline,
)
from app.core.wiki_structure_planner.evidence import EvidencePack
from app.core.wiki_structure_planner.structure_skeleton import (
    ArtifactInfo,
    Cluster,
    StructureSkeleton,
)

# ── Fixtures path ─────────────────────────────────────────────────────────────

_FIXTURES_DIR = Path(__file__).parent.parent / "fixtures" / "unified_pipeline"


# ── FakeLLM ───────────────────────────────────────────────────────────────────


class _FakeMsg:
    def __init__(self, content: str) -> None:
        self.content = content


class RoleAwareFakeLLM(BaseChatModel):
    """FakeLLM that inspects the system prompt to decide which role is calling.

    Role detection (in order):
    1. System prompt starts with "You are a technical documentation writer"
       → writer role: return ``writer_response``.
    2. System prompt contains "verifier" (case-insensitive)
       → verifier role: return ``verifier_response``.
    3. Otherwise → planner role: return ``planner_response``.

    Pydantic fields required for BaseChatModel subclass.
    """

    planner_response: str = ""
    writer_response: str = ""
    verifier_response: str = ""
    record_calls: list[dict] = []
    model_name: str = "role-aware-fake"

    @property
    def _llm_type(self) -> str:
        return "role-aware-fake"

    def _generate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: Any = None,
        **kwargs: Any,
    ) -> ChatResult:
        self.record_calls.append({"messages": messages})

        # Detect role from system prompt
        system_content = ""
        for msg in messages:
            if hasattr(msg, "type") and msg.type == "system":
                system_content = msg.content if isinstance(msg.content, str) else str(msg.content)
                break
            if msg.__class__.__name__ == "SystemMessage":
                system_content = msg.content if isinstance(msg.content, str) else str(msg.content)
                break

        if system_content.startswith("You are a technical documentation writer"):
            text = self.writer_response
        elif "verifier" in system_content.lower():
            text = self.verifier_response
        else:
            text = self.planner_response

        return ChatResult(generations=[ChatGeneration(message=AIMessage(content=text))])

    def _stream(self, *args: Any, **kwargs: Any) -> Iterator[Any]:
        raise NotImplementedError


# ── Helpers ───────────────────────────────────────────────────────────────────


def _planner_json(
    title: str = "Module",
    description: str = "Covers the module.",
    retrieval_query: str = "module components",
) -> str:
    return json.dumps({"title": title, "description": description, "retrieval_query": retrieval_query})


def _writer_md(content: str, cite_path: str = "src/auth/service.py") -> str:
    """Return markdown with a citation token so the verifier can run."""
    return f"{content} [{cite_path}:1-5]"


def _verifier_supported(n: int = 1) -> str:
    return json.dumps([{"verdict": "supported", "reason": "Grounded."} for _ in range(n)])


def _make_cluster(
    cluster_id: int,
    kind: str,
    dirs: list[str],
    artifact_name: str,
    artifact_path: str,
    artifact_kind: str,
    layer: str = "public_api",
) -> Cluster:
    artifact = ArtifactInfo(
        kind=artifact_kind,  # type: ignore[arg-type]
        name=artifact_name,
        source_path=artifact_path,
        layer=layer,
    )
    return Cluster(
        cluster_id=cluster_id,
        kind=kind,  # type: ignore[arg-type]
        dirs=dirs,
        artifacts=[artifact],
        total_artifacts=1,
        primary_languages=["py"],
        depth_range=(1, 2),
    )


def _make_skeleton(clusters: list[Cluster]) -> StructureSkeleton:
    return StructureSkeleton(
        code_clusters=[],
        doc_clusters=[],
        total_arch_symbols=0,
        total_dirs_covered=len(clusters),
        total_dirs_in_repo=len(clusters),
        repo_languages=["py"],
        effective_depth=3,
        clusters=clusters,
    )


def _make_pack(cluster_id: int, kind: str = "code") -> EvidencePack:
    return EvidencePack(cluster_id=cluster_id, kind=kind)


def _run(
    skeleton: StructureSkeleton,
    packs: dict[int, EvidencePack],
    repo_root: str,
    llm: RoleAwareFakeLLM,
    gate_mode: dict[str, str] | None = None,
) -> UnifiedPipelineResult:
    return run_unified_pipeline(
        skeleton,
        packs,
        llm=llm,
        verifier_llm=llm,
        repo_root=repo_root,
        gate_mode=gate_mode,
    )


# ── tiny_code_repo ─────────────────────────────────────────────────────────────


class TestTinyCodeRepo:
    _REPO = str(_FIXTURES_DIR / "tiny_code_repo")

    def _llm(self) -> RoleAwareFakeLLM:
        return RoleAwareFakeLLM(
            planner_response=_planner_json(title="Auth Module", description="Authentication logic."),
            writer_response=_writer_md("AuthService manages login.", "src/auth/service.py"),
            verifier_response=_verifier_supported(1),
        )

    def _skeleton(self) -> StructureSkeleton:
        cluster = _make_cluster(
            cluster_id=1, kind="code",
            dirs=["src/auth"],
            artifact_name="AuthService",
            artifact_path="src/auth/service.py",
            artifact_kind="symbol",
        )
        return _make_skeleton([cluster])

    def test_produces_at_least_one_page(self):
        result = _run(self._skeleton(), {1: _make_pack(1)}, self._REPO, self._llm())
        assert len(result.pages) > 0

    def test_every_page_has_non_empty_title(self):
        result = _run(self._skeleton(), {1: _make_pack(1)}, self._REPO, self._llm())
        for page in result.pages:
            assert page.title.strip(), f"Page {page.page_id} has empty title"

    def test_every_page_has_non_empty_markdown(self):
        result = _run(self._skeleton(), {1: _make_pack(1)}, self._REPO, self._llm())
        for page in result.pages:
            assert page.markdown.strip(), f"Page {page.page_id} has empty markdown"

    def test_target_symbols_populated_from_skeleton(self):
        result = _run(self._skeleton(), {1: _make_pack(1)}, self._REPO, self._llm())
        page = result.pages[0]
        # target_symbols comes from skeleton artifacts with layer in {entry_point, public_api}
        # our fixture uses layer=public_api so it should be populated
        assert "target_symbols" in page.metadata
        assert "target_folders" in page.metadata
        assert "target_docs" in page.metadata


class TestTinyCodeRepoAdversarial:
    """Writer hallucinates a symbol; gate/verifier should strip that paragraph."""

    _REPO = str(_FIXTURES_DIR / "tiny_code_repo")

    HALLUCINATED_SYMBOL = "HallucinatedDatabaseManager"

    def _llm(self) -> RoleAwareFakeLLM:
        # Writer emits two paragraphs: one real, one with a hallucinated class name.
        # We use a citation that points to a nonexistent file so the verifier strips.
        writer_md = (
            f"AuthService manages login. [src/auth/service.py:1-5]\n\n"
            f"The {self.HALLUCINATED_SYMBOL} handles caching. [does_not_exist.py:1]"
        )
        return RoleAwareFakeLLM(
            planner_response=_planner_json(title="Auth Module"),
            writer_response=writer_md,
            # First paragraph: supported.  Second: absent (hallucinated).
            verifier_response=json.dumps([
                {"verdict": "supported", "reason": "Grounded."},
                {"verdict": "absent", "reason": "No evidence for HallucinatedDatabaseManager."},
            ]),
        )

    def _skeleton(self) -> StructureSkeleton:
        cluster = _make_cluster(
            cluster_id=1, kind="code",
            dirs=["src/auth"],
            artifact_name="AuthService",
            artifact_path="src/auth/service.py",
            artifact_kind="symbol",
        )
        return _make_skeleton([cluster])

    def test_hallucinated_symbol_not_in_final_markdown(self):
        result = _run(
            self._skeleton(),
            {1: _make_pack(1)},
            self._REPO,
            self._llm(),
            gate_mode={"readme": "flag", "sql": "flag", "identifier": "flag"},
        )
        assert len(result.pages) > 0
        final_md = result.pages[0].markdown
        assert self.HALLUCINATED_SYMBOL not in final_md, (
            f"Hallucinated symbol {self.HALLUCINATED_SYMBOL!r} survived into final markdown"
        )

    def test_verifier_stripped_count_positive(self):
        result = _run(
            self._skeleton(),
            {1: _make_pack(1)},
            self._REPO,
            self._llm(),
            gate_mode={"readme": "flag", "sql": "flag", "identifier": "flag"},
        )
        total_stripped = sum(r.paragraphs_stripped for r in result.verify_reports.values())
        assert total_stripped >= 1


# ── tiny_doc_repo ─────────────────────────────────────────────────────────────


class TestTinyDocRepo:
    _REPO = str(_FIXTURES_DIR / "tiny_doc_repo")

    def _llm(self) -> RoleAwareFakeLLM:
        return RoleAwareFakeLLM(
            planner_response=_planner_json(title="System Docs", description="System documentation overview."),
            writer_response=_writer_md("This system handles requests.", "docs/overview.md"),
            verifier_response=_verifier_supported(1),
        )

    def _skeleton(self) -> StructureSkeleton:
        cluster = _make_cluster(
            cluster_id=2, kind="doc",
            dirs=["docs"],
            artifact_name="overview",
            artifact_path="docs/overview.md",
            artifact_kind="doc_section",
            layer="",
        )
        return _make_skeleton([cluster])

    def test_produces_at_least_one_page(self):
        result = _run(self._skeleton(), {2: _make_pack(2, "doc")}, self._REPO, self._llm())
        assert len(result.pages) > 0

    def test_every_page_has_non_empty_title(self):
        result = _run(self._skeleton(), {2: _make_pack(2, "doc")}, self._REPO, self._llm())
        for page in result.pages:
            assert page.title.strip()

    def test_every_page_has_non_empty_markdown(self):
        result = _run(self._skeleton(), {2: _make_pack(2, "doc")}, self._REPO, self._llm())
        for page in result.pages:
            assert page.markdown.strip()

    def test_target_docs_in_metadata(self):
        result = _run(self._skeleton(), {2: _make_pack(2, "doc")}, self._REPO, self._llm())
        page = result.pages[0]
        assert "target_docs" in page.metadata


class TestTinyDocRepoAdversarial:
    """Writer claims README content without reading it; gate strips the paragraph."""

    _REPO = str(_FIXTURES_DIR / "tiny_doc_repo")

    def _llm(self) -> RoleAwareFakeLLM:
        writer_md = (
            "The README describes this component in detail.\n\n"
            "System documentation for deployment. [docs/deployment.md:1-5]"
        )
        return RoleAwareFakeLLM(
            planner_response=_planner_json(title="System Docs"),
            writer_response=writer_md,
            verifier_response=_verifier_supported(1),
        )

    def _skeleton(self) -> StructureSkeleton:
        cluster = _make_cluster(
            cluster_id=2, kind="doc",
            dirs=["docs"],
            artifact_name="deployment",
            artifact_path="docs/deployment.md",
            artifact_kind="doc_section",
            layer="",
        )
        return _make_skeleton([cluster])

    def test_readme_paragraph_stripped_by_gate(self):
        result = _run(
            self._skeleton(),
            {2: _make_pack(2, "doc")},
            self._REPO,
            self._llm(),
            gate_mode={"readme": "strip", "sql": "strip", "identifier": "flag"},
        )
        assert len(result.pages) > 0
        page = result.pages[0]
        assert "README" not in page.markdown


# ── tiny_confluence_export ────────────────────────────────────────────────────


class TestTinyConfluenceExport:
    _REPO = str(_FIXTURES_DIR / "tiny_confluence_export")

    def _llm(self) -> RoleAwareFakeLLM:
        return RoleAwareFakeLLM(
            planner_response=_planner_json(
                title="Engineering Confluence",
                description="Engineering team documentation.",
            ),
            writer_response=_writer_md(
                "Engineering processes documented here.",
                "spaces/ENG/auth_design.md",
            ),
            verifier_response=_verifier_supported(1),
        )

    def _skeleton(self) -> StructureSkeleton:
        cluster = _make_cluster(
            cluster_id=3, kind="confluence",
            dirs=["spaces/ENG"],
            artifact_name="Authentication Design",
            artifact_path="spaces/ENG/auth_design.md",
            artifact_kind="confluence_page",
            layer="",
        )
        return _make_skeleton([cluster])

    def test_produces_at_least_one_page(self):
        result = _run(self._skeleton(), {3: _make_pack(3, "confluence")}, self._REPO, self._llm())
        assert len(result.pages) > 0

    def test_every_page_has_non_empty_title(self):
        result = _run(self._skeleton(), {3: _make_pack(3, "confluence")}, self._REPO, self._llm())
        for page in result.pages:
            assert page.title.strip()

    def test_every_page_has_non_empty_markdown(self):
        result = _run(self._skeleton(), {3: _make_pack(3, "confluence")}, self._REPO, self._llm())
        for page in result.pages:
            assert page.markdown.strip()


class TestTinyConfluenceAdversarial:
    """Writer hallucinates a SQL schema without reading a .sql file.  Gate strips it."""

    _REPO = str(_FIXTURES_DIR / "tiny_confluence_export")

    HALLUCINATED_SQL = "CREATE TABLE users (id INTEGER PRIMARY KEY)"

    def _llm(self) -> RoleAwareFakeLLM:
        writer_md = (
            f"{self.HALLUCINATED_SQL};\n\n"
            "Auth design documented. [spaces/ENG/auth_design.md:1-5]"
        )
        return RoleAwareFakeLLM(
            planner_response=_planner_json(title="Engineering Confluence"),
            writer_response=writer_md,
            verifier_response=_verifier_supported(1),
        )

    def _skeleton(self) -> StructureSkeleton:
        cluster = _make_cluster(
            cluster_id=3, kind="confluence",
            dirs=["spaces/ENG"],
            artifact_name="auth_design",
            artifact_path="spaces/ENG/auth_design.md",
            artifact_kind="confluence_page",
            layer="",
        )
        return _make_skeleton([cluster])

    def test_sql_paragraph_stripped(self):
        result = _run(
            self._skeleton(),
            {3: _make_pack(3, "confluence")},
            self._REPO,
            self._llm(),
            gate_mode={"readme": "strip", "sql": "strip", "identifier": "flag"},
        )
        assert len(result.pages) > 0
        page = result.pages[0]
        assert "CREATE TABLE" not in page.markdown


# ── tiny_jira_export ─────────────────────────────────────────────────────────


class TestTinyJiraExport:
    _REPO = str(_FIXTURES_DIR / "tiny_jira_export")

    def _llm(self) -> RoleAwareFakeLLM:
        return RoleAwareFakeLLM(
            planner_response=_planner_json(
                title="Auth Epic",
                description="Authentication system epics and stories.",
            ),
            writer_response=_writer_md(
                "Authentication epic AUTH-1 captures the full auth scope.",
                "epics/AUTH-1.md",
            ),
            verifier_response=_verifier_supported(1),
        )

    def _skeleton(self) -> StructureSkeleton:
        cluster = _make_cluster(
            cluster_id=4, kind="jira",
            dirs=["epics"],
            artifact_name="AUTH-1",
            artifact_path="epics/AUTH-1.md",
            artifact_kind="jira_issue",
            layer="",
        )
        return _make_skeleton([cluster])

    def test_produces_at_least_one_page(self):
        result = _run(self._skeleton(), {4: _make_pack(4, "jira")}, self._REPO, self._llm())
        assert len(result.pages) > 0

    def test_every_page_has_non_empty_title(self):
        result = _run(self._skeleton(), {4: _make_pack(4, "jira")}, self._REPO, self._llm())
        for page in result.pages:
            assert page.title.strip()

    def test_every_page_has_non_empty_markdown(self):
        result = _run(self._skeleton(), {4: _make_pack(4, "jira")}, self._REPO, self._llm())
        for page in result.pages:
            assert page.markdown.strip()


class TestTinyJiraAdversarial:
    """Writer references a nonexistent path; verifier strips the citation_missing paragraph."""

    _REPO = str(_FIXTURES_DIR / "tiny_jira_export")

    HALLUCINATED_PATH = "stories/AUTH-999.md"  # does not exist

    def _llm(self) -> RoleAwareFakeLLM:
        writer_md = (
            f"Story AUTH-999 defines the token refresh flow. [{self.HALLUCINATED_PATH}:1]\n\n"
            "Epic AUTH-1 covers authentication. [epics/AUTH-1.md:1-5]"
        )
        # The verifier processes AUTH-999 as citation_missing (file not readable) BEFORE
        # calling the LLM.  Only AUTH-1 paragraph is sent to the LLM verifier — as a
        # single-element batch.  So the canned response needs exactly 1 verdict.
        return RoleAwareFakeLLM(
            planner_response=_planner_json(title="Auth Epic"),
            writer_response=writer_md,
            verifier_response=json.dumps([
                {"verdict": "supported", "reason": "Grounded in AUTH-1."},
            ]),
        )

    def _skeleton(self) -> StructureSkeleton:
        cluster = _make_cluster(
            cluster_id=4, kind="jira",
            dirs=["epics"],
            artifact_name="AUTH-1",
            artifact_path="epics/AUTH-1.md",
            artifact_kind="jira_issue",
            layer="",
        )
        return _make_skeleton([cluster])

    def test_hallucinated_path_not_in_final_markdown(self):
        result = _run(
            self._skeleton(),
            {4: _make_pack(4, "jira")},
            self._REPO,
            self._llm(),
            gate_mode={"readme": "flag", "sql": "flag", "identifier": "flag"},
        )
        assert len(result.pages) > 0
        final_md = result.pages[0].markdown
        # The paragraph that cited the nonexistent path should be stripped
        assert "AUTH-999" not in final_md, (
            "Hallucinated story reference AUTH-999 survived into final markdown"
        )

    def test_at_least_one_paragraph_kept(self):
        """The supported paragraph (AUTH-1) should remain after stripping."""
        result = _run(
            self._skeleton(),
            {4: _make_pack(4, "jira")},
            self._REPO,
            self._llm(),
            gate_mode={"readme": "flag", "sql": "flag", "identifier": "flag"},
        )
        page = result.pages[0]
        assert "AUTH-1" in page.markdown, (
            "The legitimate AUTH-1 paragraph should survive after stripping"
        )
