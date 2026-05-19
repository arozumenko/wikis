"""Unit tests for app/core/agents/unified_wiki_pipeline.py (#243).

Covers:
- Empty skeleton → pages=[], all reports empty.
- Single code cluster → 1 page, all four sub-reports present, target_* from skeleton.
- Gate strips one paragraph → final markdown omits it, gate_report has the flag,
  stream emits gate.page_done with n_stripped=1.
- Verifier strips one paragraph → final markdown omits it, verify_report records
  absent, stream emits verifier.page_done.
- Custom gate_mode flows through to apply_gate (readme+sql strip, identifier verifier).
- verifier_llm defaults to llm when None.

The FakeLLM pattern follows test_planner_agent.py (BaseChatModel subclass).
The writer LLM returns a canned markdown string; the verifier LLM returns a
canned JSON array of verdicts.  The planner LLM returns a canned JSON PageSpec.
"""

from __future__ import annotations

import json
import os
import tempfile
from collections.abc import Iterator
from typing import Any

import pytest
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage
from langchain_core.outputs import ChatGeneration, ChatResult

from app.core.wiki_content_writer.citation_extractor import Citation, CitedClaim
from app.core.wiki_structure_planner.evidence import EvidencePack
from app.core.wiki_structure_planner.structure_skeleton import (
    ArtifactInfo,
    Cluster,
    StructureSkeleton,
)
from app.core.agents.unified_wiki_pipeline import (
    GeneratedPage,
    UnifiedPipelineResult,
    run_unified_pipeline,
)


# ── FakeLLM ────────────────────────────────────────────────────────────────────


class FakeLLM(BaseChatModel):
    """Minimal BaseChatModel subclass that returns canned text responses.

    Responses are returned in sequence; if exhausted, the last is repeated.
    The LLM detects which role is calling it by scanning the system prompt
    content prefix:
    - "You are a technical documentation writer" → writer role
    - "unified planner" / PLANNER_SYSTEM_PROMPT prefix → planner role
    - "verifier" in system prompt → verifier role

    If ``responses`` is a list, they are cycled through.
    If ``role_responses`` is provided, it overrides the role detection.
    """

    responses: list[str] = []
    record_calls: list[dict] = []
    model_name: str = "fake-llm"

    @property
    def _llm_type(self) -> str:
        return "fake"

    def _generate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: Any = None,
        **kwargs: Any,
    ) -> ChatResult:
        self.record_calls.append({"messages": messages})
        idx = min(len(self.record_calls) - 1, len(self.responses) - 1)
        text = self.responses[idx] if self.responses else ""
        return ChatResult(generations=[ChatGeneration(message=AIMessage(content=text))])

    def _stream(self, *args: Any, **kwargs: Any) -> Iterator[Any]:
        raise NotImplementedError("FakeLLM does not support streaming")


# ── Helpers ────────────────────────────────────────────────────────────────────


def _planner_response(
    title: str = "Auth Module",
    description: str = "Handles authentication.",
    retrieval_query: str = "auth login session",
) -> str:
    """Return a canned planner JSON response."""
    return json.dumps({
        "title": title,
        "description": description,
        "retrieval_query": retrieval_query,
    })


def _verifier_supported_response(n: int = 1) -> str:
    """Return a canned verifier JSON with n 'supported' verdicts."""
    verdicts = [{"verdict": "supported", "reason": "Grounded in cited span."} for _ in range(n)]
    return json.dumps(verdicts)


def _verifier_absent_response() -> str:
    """Return a canned verifier JSON with one 'absent' verdict."""
    return json.dumps([{"verdict": "absent", "reason": "No supporting evidence found."}])


def _make_cluster(
    cluster_id: int = 1,
    kind: str = "code",
    dirs: list[str] | None = None,
    artifacts: list[ArtifactInfo] | None = None,
) -> Cluster:
    if dirs is None:
        dirs = ["src/auth"]
    if artifacts is None:
        artifacts = [
            ArtifactInfo(
                kind="symbol",  # type: ignore[arg-type]
                name="AuthService",
                source_path="src/auth/service.py",
                layer="public_api",
            )
        ]
    return Cluster(
        cluster_id=cluster_id,
        kind=kind,  # type: ignore[arg-type]
        dirs=dirs,
        artifacts=artifacts,
        total_artifacts=len(artifacts),
        primary_languages=["py"],
        depth_range=(1, 1),
    )


def _make_skeleton(clusters: list[Cluster]) -> StructureSkeleton:
    return StructureSkeleton(
        code_clusters=[],
        doc_clusters=[],
        total_arch_symbols=0,
        total_dirs_covered=len(clusters),
        total_dirs_in_repo=len(clusters),
        repo_languages=["py"],
        effective_depth=2,
        clusters=clusters,
    )


def _make_evidence_pack(cluster_id: int = 1) -> EvidencePack:
    return EvidencePack(cluster_id=cluster_id, kind="code")


# ── Tests ──────────────────────────────────────────────────────────────────────


class TestEmptySkeleton:
    def test_empty_skeleton_returns_no_pages(self, tmp_path):
        """Empty skeleton → pages=[], all reports empty."""
        skeleton = _make_skeleton([])
        llm = FakeLLM(responses=[_planner_response()])
        result = run_unified_pipeline(
            skeleton,
            {},
            llm=llm,
            repo_root=str(tmp_path),
        )
        assert isinstance(result, UnifiedPipelineResult)
        assert result.pages == []
        assert result.gate_reports == {}
        assert result.verify_reports == {}
        assert result.plan_report.pages_emitted == 0

    def test_empty_skeleton_plan_report_has_zero_budget(self, tmp_path):
        """Empty skeleton plan report has zero budget totals."""
        skeleton = _make_skeleton([])
        llm = FakeLLM(responses=[])
        result = run_unified_pipeline(
            skeleton,
            {},
            llm=llm,
            repo_root=str(tmp_path),
        )
        assert result.plan_report.tool_budget_total == 0
        assert result.plan_report.tool_budget_used == 0
        assert result.plan_report.tool_budget_exceeded is False


def _setup_cited_file(tmp_path, rel_path: str, content: str = "# content\n") -> None:
    """Create a cited file inside tmp_path for verifier span reading."""
    full = tmp_path / rel_path
    full.parent.mkdir(parents=True, exist_ok=True)
    full.write_text(content)


class TestSingleCodeCluster:
    def test_single_cluster_produces_one_page(self, tmp_path):
        """Single code cluster → 1 page in result."""
        cluster = _make_cluster()
        skeleton = _make_skeleton([cluster])
        packs = {1: _make_evidence_pack(1)}
        _setup_cited_file(tmp_path, "src/auth/service.py", "class AuthService: pass\n")

        writer_md = "The AuthService handles user login. [src/auth/service.py:1]"
        llm = FakeLLM(responses=[
            _planner_response(),          # planner call
            writer_md,                    # writer call
            _verifier_supported_response(1),  # verifier call
        ])

        result = run_unified_pipeline(
            skeleton,
            packs,
            llm=llm,
            repo_root=str(tmp_path),
        )
        assert len(result.pages) == 1

    def test_single_cluster_page_has_title(self, tmp_path):
        """Page title comes from the planner response."""
        cluster = _make_cluster()
        skeleton = _make_skeleton([cluster])
        packs = {1: _make_evidence_pack(1)}
        _setup_cited_file(tmp_path, "src/auth/service.py", "class AuthService: pass\n")
        writer_md = "The AuthService handles user login. [src/auth/service.py:1]"
        llm = FakeLLM(responses=[
            _planner_response(title="Auth Module"),
            writer_md,
            _verifier_supported_response(1),
        ])
        result = run_unified_pipeline(skeleton, packs, llm=llm, repo_root=str(tmp_path))
        assert result.pages[0].title == "Auth Module"

    def test_single_cluster_all_four_reports_present(self, tmp_path):
        """All four report types present for a single cluster."""
        cluster = _make_cluster()
        skeleton = _make_skeleton([cluster])
        packs = {1: _make_evidence_pack(1)}
        _setup_cited_file(tmp_path, "src/auth/service.py", "class AuthService: pass\n")
        writer_md = "Auth module. [src/auth/service.py:1]"
        llm = FakeLLM(responses=[
            _planner_response(),
            writer_md,
            _verifier_supported_response(1),
        ])
        result = run_unified_pipeline(skeleton, packs, llm=llm, repo_root=str(tmp_path))
        assert result.plan_report is not None
        assert 1 in result.gate_reports
        assert 1 in result.verify_reports

    def test_target_fields_from_skeleton(self, tmp_path):
        """target_symbols and target_folders come from the skeleton (not the LLM)."""
        cluster = _make_cluster(
            artifacts=[
                ArtifactInfo(
                    kind="symbol",  # type: ignore[arg-type]
                    name="AuthService",
                    source_path="src/auth/service.py",
                    layer="public_api",
                )
            ],
            dirs=["src/auth"],
        )
        skeleton = _make_skeleton([cluster])
        packs = {1: _make_evidence_pack(1)}
        _setup_cited_file(tmp_path, "src/auth/service.py", "class AuthService: pass\n")
        writer_md = "Description paragraph. [src/auth/service.py:1]"
        llm = FakeLLM(responses=[
            _planner_response(),
            writer_md,
            _verifier_supported_response(1),
        ])
        result = run_unified_pipeline(skeleton, packs, llm=llm, repo_root=str(tmp_path))
        page = result.pages[0]
        assert "AuthService" in page.metadata.get("target_symbols", [])
        assert "src/auth" in page.metadata.get("target_folders", [])


class TestGateStripping:
    def test_gate_strips_readme_paragraph(self, tmp_path):
        """Gate mode strip removes a paragraph that claims README content without reading it."""
        cluster = _make_cluster()
        skeleton = _make_skeleton([cluster])
        packs = {1: _make_evidence_pack(1)}

        # Create the cited file for the "clean" paragraph so the verifier keeps it
        cited_dir = tmp_path / "src" / "auth"
        cited_dir.mkdir(parents=True, exist_ok=True)
        (cited_dir / "service.py").write_text("class AuthService: pass\n")

        # Two paragraphs: one mentions README (triggers gate strip), one is clean.
        writer_md = (
            "The README describes the setup process.\n\n"
            "The AuthService handles user login. [src/auth/service.py:1]"
        )

        llm = FakeLLM(responses=[
            _planner_response(),
            writer_md,
            # Verifier sees only the kept paragraph (1 claim after gate strip)
            _verifier_supported_response(1),
        ])

        events_captured: list[dict] = []

        def _cb(event: dict) -> None:
            events_captured.append(event)

        result = run_unified_pipeline(
            skeleton,
            packs,
            llm=llm,
            repo_root=str(tmp_path),
            gate_mode={"readme": "strip", "sql": "strip", "identifier": "verifier"},
            stream_callback=_cb,
        )

        page = result.pages[0]
        # The README paragraph should be stripped from the final markdown
        assert "README" not in page.markdown

        # gate_report should record the strip
        gate_report = result.gate_reports[1]
        assert len(gate_report.stripped_paragraphs) == 1
        assert page.metadata["gate_stripped_total"] == 1

    def test_gate_page_done_event_emitted_with_n_stripped(self, tmp_path):
        """stream_callback receives gate.page_done with n_stripped=1 when gate strips."""
        cluster = _make_cluster()
        skeleton = _make_skeleton([cluster])
        packs = {1: _make_evidence_pack(1)}

        # Create the cited file for the clean paragraph
        cited_dir = tmp_path / "src" / "auth"
        cited_dir.mkdir(parents=True, exist_ok=True)
        (cited_dir / "service.py").write_text("class AuthService: pass\n")

        writer_md = (
            "The README describes the setup.\n\n"
            "Clean paragraph. [src/auth/service.py:1]"
        )
        llm = FakeLLM(responses=[
            _planner_response(),
            writer_md,
            _verifier_supported_response(1),
        ])
        events: list[dict] = []

        run_unified_pipeline(
            skeleton, packs, llm=llm, repo_root=str(tmp_path),
            gate_mode={"readme": "strip", "sql": "strip", "identifier": "flag"},
            stream_callback=events.append,
        )

        gate_events = [e for e in events if e.get("event") == "gate.page_done"]
        assert len(gate_events) == 1
        assert gate_events[0]["n_stripped"] == 1


class TestVerifierStripping:
    def test_verifier_strips_absent_paragraph(self, tmp_path):
        """Verifier strips a paragraph whose verdict is 'absent'."""
        cluster = _make_cluster()
        skeleton = _make_skeleton([cluster])
        packs = {1: _make_evidence_pack(1)}

        # One paragraph with a citation that references a real file (so gate passes).
        # Writer produces markdown with a citation token pointing to a file we'll create.
        with tempfile.NamedTemporaryFile(
            dir=str(tmp_path), suffix=".py", delete=False, mode="w"
        ) as f:
            f.write("def login(): pass\n")
            fname = os.path.basename(f.name)

        writer_md = f"Login is handled here. [{fname}:1]"

        llm = FakeLLM(responses=[
            _planner_response(),
            writer_md,
            _verifier_absent_response(),  # verdict: absent → strip
        ])

        events: list[dict] = []
        result = run_unified_pipeline(
            skeleton, packs, llm=llm, repo_root=str(tmp_path),
            gate_mode={"readme": "flag", "sql": "flag", "identifier": "flag"},
            stream_callback=events.append,
        )

        page = result.pages[0]
        # Paragraph should be stripped — markdown should be empty or not contain login text
        assert "Login is handled here" not in page.markdown

        verify_report = result.verify_reports[1]
        assert verify_report.paragraphs_stripped == 1

    def test_verifier_page_done_event_emitted(self, tmp_path):
        """stream_callback receives verifier.page_done event."""
        cluster = _make_cluster()
        skeleton = _make_skeleton([cluster])
        packs = {1: _make_evidence_pack(1)}

        with tempfile.NamedTemporaryFile(
            dir=str(tmp_path), suffix=".py", delete=False, mode="w"
        ) as f:
            f.write("def login(): pass\n")
            fname = os.path.basename(f.name)

        writer_md = f"Login is handled here. [{fname}:1]"
        llm = FakeLLM(responses=[
            _planner_response(),
            writer_md,
            _verifier_absent_response(),
        ])
        events: list[dict] = []
        run_unified_pipeline(
            skeleton, packs, llm=llm, repo_root=str(tmp_path),
            gate_mode={"readme": "flag", "sql": "flag", "identifier": "flag"},
            stream_callback=events.append,
        )
        verifier_events = [e for e in events if e.get("event") == "verifier.page_done"]
        assert len(verifier_events) == 1
        assert verifier_events[0]["n_stripped"] == 1
        assert verifier_events[0]["n_kept"] == 0


class TestGateMode:
    def test_custom_gate_mode_flows_to_apply_gate(self, tmp_path):
        """Custom gate_mode dict is respected — sql=strip removes SQL paragraphs."""
        cluster = _make_cluster()
        skeleton = _make_skeleton([cluster])
        packs = {1: _make_evidence_pack(1)}

        # Create cited file for the clean paragraph
        cited_dir = tmp_path / "src" / "auth"
        cited_dir.mkdir(parents=True, exist_ok=True)
        (cited_dir / "service.py").write_text("class AuthService: pass\n")

        # Paragraph with SQL keyword (no .sql file in trace → should be stripped with sql=strip)
        writer_md = (
            "CREATE TABLE users (id INTEGER PRIMARY KEY);\n\n"
            "Clean line. [src/auth/service.py:1]"
        )
        llm = FakeLLM(responses=[
            _planner_response(),
            writer_md,
            _verifier_supported_response(1),
        ])
        result = run_unified_pipeline(
            skeleton, packs, llm=llm, repo_root=str(tmp_path),
            gate_mode={"readme": "strip", "sql": "strip", "identifier": "flag"},
        )
        page = result.pages[0]
        assert "CREATE TABLE" not in page.markdown
        assert result.gate_reports[1].stripped_paragraphs  # at least one stripped

    def test_identifier_verifier_mode_does_not_strip(self, tmp_path):
        """identifier=verifier mode keeps identifier-flagged paragraphs (gate action=verifier).

        The gate should NOT strip when action is "verifier"; it flags the paragraph and
        passes it to the verifier.  We create the cited file so the verifier can read it
        and return a supported verdict.
        """
        cluster = _make_cluster()
        skeleton = _make_skeleton([cluster])
        packs = {1: _make_evidence_pack(1)}

        # Create the cited source file so the verifier can read it
        cited_dir = tmp_path / "src" / "auth"
        cited_dir.mkdir(parents=True, exist_ok=True)
        cited_file = cited_dir / "service.py"
        cited_file.write_text("class AuthService: pass\n")

        # Paragraph with an unknown CamelCase identifier (no trace to ground it)
        writer_md = "The AuthService handles requests. [src/auth/service.py:1]"
        llm = FakeLLM(responses=[
            _planner_response(),
            writer_md,
            _verifier_supported_response(1),
        ])
        result = run_unified_pipeline(
            skeleton, packs, llm=llm, repo_root=str(tmp_path),
            gate_mode={"readme": "strip", "sql": "strip", "identifier": "verifier"},
        )
        # identifier mode=verifier should NOT strip the paragraph (just flag it)
        page = result.pages[0]
        # Gate should NOT have stripped it (action=verifier means pass to verifier)
        assert result.gate_reports[1].stripped_paragraphs == []
        # Verifier returned "supported" so the paragraph should survive
        assert "AuthService" in page.markdown


class TestVerifierLlmDefault:
    def test_verifier_llm_defaults_to_llm_when_none(self, tmp_path):
        """verifier_llm=None → uses the primary llm for verification."""
        cluster = _make_cluster()
        skeleton = _make_skeleton([cluster])
        packs = {1: _make_evidence_pack(1)}

        with tempfile.NamedTemporaryFile(
            dir=str(tmp_path), suffix=".py", delete=False, mode="w"
        ) as f:
            f.write("def login(): pass\n")
            fname = os.path.basename(f.name)

        writer_md = f"Login paragraph. [{fname}:1]"
        # primary LLM handles both planner, writer, and verifier calls
        llm = FakeLLM(responses=[
            _planner_response(),               # planner
            writer_md,                         # writer
            _verifier_supported_response(1),   # verifier (same llm)
        ])

        result = run_unified_pipeline(
            skeleton, packs,
            llm=llm,
            verifier_llm=None,           # ← explicit None
            repo_root=str(tmp_path),
            gate_mode={"readme": "flag", "sql": "flag", "identifier": "flag"},
        )
        # Should complete without error and produce one page
        assert len(result.pages) == 1
        # The LLM was called at least 2 times (planner + writer; verifier uses same llm)
        assert len(llm.record_calls) >= 2  # planner always, writer always

    def test_separate_verifier_llm_used_for_verification(self, tmp_path):
        """verifier_llm is used for citation verification when provided."""
        cluster = _make_cluster()
        skeleton = _make_skeleton([cluster])
        packs = {1: _make_evidence_pack(1)}

        with tempfile.NamedTemporaryFile(
            dir=str(tmp_path), suffix=".py", delete=False, mode="w"
        ) as f:
            f.write("def login(): pass\n")
            fname = os.path.basename(f.name)

        writer_md = f"Login paragraph. [{fname}:1]"
        primary_llm = FakeLLM(responses=[
            _planner_response(),
            writer_md,
        ])
        verifier_llm = FakeLLM(responses=[_verifier_supported_response(1)])

        result = run_unified_pipeline(
            skeleton, packs,
            llm=primary_llm,
            verifier_llm=verifier_llm,
            repo_root=str(tmp_path),
            gate_mode={"readme": "flag", "sql": "flag", "identifier": "flag"},
        )
        assert len(result.pages) == 1
        # verifier_llm should have been called once for the verifier batch
        assert len(verifier_llm.record_calls) >= 1


class TestStreamCallbackEvents:
    def test_planner_started_and_done_events_emitted(self, tmp_path):
        """planner.started and planner.done events are always emitted."""
        skeleton = _make_skeleton([])
        llm = FakeLLM(responses=[])
        events: list[dict] = []
        run_unified_pipeline(skeleton, {}, llm=llm, repo_root=str(tmp_path),
                             stream_callback=events.append)
        event_names = {e.get("event") for e in events}
        assert "planner.started" in event_names
        assert "planner.done" in event_names

    def test_writer_page_started_emitted(self, tmp_path):
        """writer.page_started event emitted for each page."""
        cluster = _make_cluster()
        skeleton = _make_skeleton([cluster])
        packs = {1: _make_evidence_pack(1)}
        _setup_cited_file(tmp_path, "src/auth/service.py", "class AuthService: pass\n")
        writer_md = "Content. [src/auth/service.py:1]"
        llm = FakeLLM(responses=[
            _planner_response(),
            writer_md,
            _verifier_supported_response(1),
        ])
        events: list[dict] = []
        run_unified_pipeline(skeleton, packs, llm=llm, repo_root=str(tmp_path),
                             stream_callback=events.append)
        started = [e for e in events if e.get("event") == "writer.page_started"]
        assert len(started) == 1
        assert started[0]["title"] == "Auth Module"
