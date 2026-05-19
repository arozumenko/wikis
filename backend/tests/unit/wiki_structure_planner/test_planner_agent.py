"""Unit tests for wiki_structure_planner/planner_agent.py (#236).

Covers:
- Empty skeleton → empty list
- One code cluster + evidence pack → PageSpec with LLM title, skeleton target_*
- Mixed-kind skeleton (code + confluence) → both clusters get specs
- Budget exhaustion → tool_budget_exceeded=True in PlanReport
- Cluster missing evidence pack → fallback spec + note
- Malformed LLM JSON → graceful fallback spec

The ``FakeLLM`` subclass of ``BaseChatModel`` returns canned responses and
records tool invocations; no real API keys are required.
"""

from __future__ import annotations

import json
from collections.abc import Iterator
from typing import Any

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage
from langchain_core.outputs import ChatGeneration, ChatResult

from app.core.wiki_structure_planner.evidence import EvidencePack
from app.core.wiki_structure_planner.planner_agent import (
    run_planner_with_report,
)
from app.core.wiki_structure_planner.planner_prompts import (
    PLANNER_SYSTEM_PROMPT,
    _kind_prompt_fragment,  # noqa: PLC2701 – tested directly
    build_user_prompt,
)
from app.core.wiki_structure_planner.structure_skeleton import (
    ArtifactInfo,
    Cluster,
    DirCluster,
    StructureSkeleton,
    SymbolInfo,
)

# ── FakeLLM ───────────────────────────────────────────────────────────────────


class FakeLLM(BaseChatModel):
    """Minimal BaseChatModel subclass that returns canned text responses.

    Parameters
    ----------
    responses : list[str]
        Responses to return in order.  If the list is exhausted, the last
        response is repeated.
    record_calls : list[dict]
        Mutable list; each invocation appends ``{"messages": [...]}``.
    """

    responses: list[str] = []
    record_calls: list[dict] = []
    # Pydantic fields must be declared for langchain 0.2+
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


# ── Cluster / skeleton builders ───────────────────────────────────────────────


def _make_artifact(
    kind: str = "symbol",
    name: str = "MyClass",
    source_path: str = "src/module.py",
    layer: str = "public_api",
    connections: int = 5,
) -> ArtifactInfo:
    return ArtifactInfo(
        kind=kind,  # type: ignore[arg-type]
        name=name,
        source_path=source_path,
        layer=layer,
        connections=connections,
    )


def _make_cluster(
    cluster_id: int = 1,
    kind: str = "code",
    dirs: list[str] | None = None,
    artifacts: list[ArtifactInfo] | None = None,
) -> Cluster:
    arts = artifacts or [_make_artifact()]
    return Cluster(
        cluster_id=cluster_id,
        kind=kind,  # type: ignore[arg-type]
        dirs=dirs or ["src/module"],
        artifacts=arts,
        total_artifacts=len(arts),
        primary_languages=["py"],
        depth_range=(1, 2),
    )


def _make_skeleton(clusters: list[Cluster]) -> StructureSkeleton:
    return StructureSkeleton(
        code_clusters=[],
        doc_clusters=[],
        total_arch_symbols=0,
        total_dirs_covered=0,
        total_dirs_in_repo=0,
        repo_languages=["py"],
        effective_depth=3,
        repo_name="test-repo",
        clusters=clusters,
    )


def _make_pack(cluster_id: int = 1, kind: str = "code") -> EvidencePack:
    return EvidencePack(
        cluster_id=cluster_id,
        kind=kind,
        signatures=[("MyClass", "src/module.py", "public_api", "Main class.")],
        file_heads=[("src/module.py", "class MyClass:\n    pass")],
        readme_excerpt="",
        sql_blocks=[],
    )


def _json_response(
    cluster_id: int,
    title: str = "My Feature",
    description: str = "Does something.",
    retrieval_query: str = "feature module",
) -> str:
    return json.dumps(
        {
            "cluster_id": cluster_id,
            "title": title,
            "description": description,
            "retrieval_query": retrieval_query,
        }
    )


# ── Tests: empty skeleton ─────────────────────────────────────────────────────


class TestEmptySkeleton:
    def test_empty_clusters_returns_empty_list(self, tmp_path):
        skeleton = _make_skeleton([])
        llm = FakeLLM(responses=[], record_calls=[])
        pages, report = run_planner_with_report(
            skeleton, {}, llm=llm, repo_root=str(tmp_path)
        )
        assert pages == []
        assert report.pages_emitted == 0
        assert report.tool_budget_total == 0
        assert not report.tool_budget_exceeded

    def test_empty_clusters_makes_no_llm_calls(self, tmp_path):
        skeleton = _make_skeleton([])
        calls: list[dict] = []
        llm = FakeLLM(responses=[], record_calls=calls)
        run_planner_with_report(skeleton, {}, llm=llm, repo_root=str(tmp_path))
        assert calls == []


# ── Tests: single code cluster ────────────────────────────────────────────────


class TestSingleCodeCluster:
    def test_returns_one_page_spec(self, tmp_path):
        cluster = _make_cluster(cluster_id=1, kind="code")
        skeleton = _make_skeleton([cluster])
        pack = _make_pack(cluster_id=1)
        llm = FakeLLM(
            responses=[_json_response(1, title="Authentication Service")],
            record_calls=[],
        )
        pages, report = run_planner_with_report(
            skeleton, {1: pack}, llm=llm, repo_root=str(tmp_path)
        )
        assert len(pages) == 1
        assert report.pages_emitted == 1

    def test_title_comes_from_llm(self, tmp_path):
        cluster = _make_cluster(cluster_id=1, kind="code")
        skeleton = _make_skeleton([cluster])
        pack = _make_pack(cluster_id=1)
        llm = FakeLLM(
            responses=[_json_response(1, title="Authentication Service")],
            record_calls=[],
        )
        pages, _ = run_planner_with_report(
            skeleton, {1: pack}, llm=llm, repo_root=str(tmp_path)
        )
        assert pages[0].title == "Authentication Service"

    def test_target_symbols_from_skeleton(self, tmp_path):
        """target_symbols must be injected from skeleton, not from LLM."""
        arts = [
            _make_artifact(name="PublicClass", layer="public_api", source_path="src/a.py"),
            _make_artifact(name="InternalHelper", layer="internal", source_path="src/b.py"),
            _make_artifact(name="Entry", layer="entry_point", source_path="src/c.py"),
        ]
        cluster = _make_cluster(cluster_id=1, artifacts=arts)
        skeleton = _make_skeleton([cluster])
        pack = _make_pack(cluster_id=1)
        llm = FakeLLM(
            responses=[_json_response(1)],
            record_calls=[],
        )
        pages, _ = run_planner_with_report(
            skeleton, {1: pack}, llm=llm, repo_root=str(tmp_path)
        )
        # Only entry_point and public_api layers in target_symbols
        assert "PublicClass" in pages[0].target_symbols
        assert "Entry" in pages[0].target_symbols
        assert "InternalHelper" not in pages[0].target_symbols

    def test_target_folders_sorted_from_skeleton(self, tmp_path):
        cluster = _make_cluster(
            cluster_id=1,
            dirs=["src/z", "src/a", "src/m"],
        )
        skeleton = _make_skeleton([cluster])
        pack = _make_pack(cluster_id=1)
        llm = FakeLLM(responses=[_json_response(1)], record_calls=[])
        pages, _ = run_planner_with_report(
            skeleton, {1: pack}, llm=llm, repo_root=str(tmp_path)
        )
        assert pages[0].target_folders == sorted(["src/z", "src/a", "src/m"])

    def test_target_docs_from_skeleton_confluence_artifacts(self, tmp_path):
        arts = [
            _make_artifact(kind="confluence_page", name="Setup Guide", source_path="docs/setup.md", layer=""),
            _make_artifact(kind="symbol", name="MyClass", source_path="src/x.py", layer="public_api"),
        ]
        cluster = _make_cluster(cluster_id=1, artifacts=arts)
        skeleton = _make_skeleton([cluster])
        pack = _make_pack(cluster_id=1)
        llm = FakeLLM(responses=[_json_response(1)], record_calls=[])
        pages, _ = run_planner_with_report(
            skeleton, {1: pack}, llm=llm, repo_root=str(tmp_path)
        )
        assert "docs/setup.md" in pages[0].target_docs

    def test_cluster_id_preserved(self, tmp_path):
        cluster = _make_cluster(cluster_id=42)
        skeleton = _make_skeleton([cluster])
        pack = _make_pack(cluster_id=42)
        llm = FakeLLM(responses=[_json_response(42)], record_calls=[])
        pages, _ = run_planner_with_report(
            skeleton, {42: pack}, llm=llm, repo_root=str(tmp_path)
        )
        assert pages[0].cluster_id == 42


# ── Tests: mixed-kind skeleton ────────────────────────────────────────────────


class TestMixedKindSkeleton:
    def test_both_clusters_get_pages(self, tmp_path):
        code_cluster = _make_cluster(cluster_id=1, kind="code")
        confluence_cluster = _make_cluster(
            cluster_id=2,
            kind="confluence",
            dirs=["docs/confluence"],
            artifacts=[
                _make_artifact(kind="confluence_page", name="API Docs", source_path="docs/api.md", layer="")
            ],
        )
        skeleton = _make_skeleton([code_cluster, confluence_cluster])
        packs = {
            1: _make_pack(cluster_id=1, kind="code"),
            2: EvidencePack(cluster_id=2, kind="confluence"),
        }
        llm = FakeLLM(
            responses=[
                _json_response(1, title="Auth Module"),
                _json_response(2, title="API Documentation"),
            ],
            record_calls=[],
        )
        pages, report = run_planner_with_report(
            skeleton, packs, llm=llm, repo_root=str(tmp_path)
        )
        assert len(pages) == 2
        assert report.pages_emitted == 2

    def test_code_cluster_title_from_llm(self, tmp_path):
        code_cluster = _make_cluster(cluster_id=1, kind="code")
        conf_cluster = _make_cluster(
            cluster_id=2,
            kind="confluence",
            dirs=["docs"],
            artifacts=[_make_artifact(kind="confluence_page", name="Overview", source_path="docs/overview.md", layer="")],
        )
        skeleton = _make_skeleton([code_cluster, conf_cluster])
        packs = {1: _make_pack(1), 2: EvidencePack(cluster_id=2, kind="confluence")}
        llm = FakeLLM(
            responses=[
                _json_response(1, title="Code Feature"),
                _json_response(2, title="Project Overview"),
            ],
            record_calls=[],
        )
        pages, _ = run_planner_with_report(
            skeleton, packs, llm=llm, repo_root=str(tmp_path)
        )
        by_id = {p.cluster_id: p for p in pages}
        assert by_id[1].title == "Code Feature"
        assert by_id[2].title == "Project Overview"


# ── Tests: budget exhaustion ──────────────────────────────────────────────────


class TestBudgetExhaustion:
    def test_no_tool_calls_means_budget_not_exceeded(self, tmp_path):
        """Happy path: an LLM that answers from the evidence pack alone
        leaves the tool budget untouched and ``tool_budget_exceeded`` stays
        False.  The actual budget-exhaustion path is exercised by
        ``test_tool_budget_exceeded_when_forced`` below."""
        cluster = _make_cluster(cluster_id=1, kind="code")
        skeleton = _make_skeleton([cluster])
        pack = _make_pack(cluster_id=1)

        llm = FakeLLM(responses=[_json_response(1)], record_calls=[])
        pages, report = run_planner_with_report(
            skeleton, {1: pack}, llm=llm, repo_root=str(tmp_path)
        )
        assert not report.tool_budget_exceeded
        assert report.tool_budget_used == 0

    def test_budget_total_equals_3x_clusters(self, tmp_path):
        clusters = [_make_cluster(cluster_id=i) for i in range(1, 6)]
        skeleton = _make_skeleton(clusters)
        packs = {c.cluster_id: _make_pack(c.cluster_id) for c in clusters}
        llm = FakeLLM(
            responses=[_json_response(i) for i in range(1, 6)],
            record_calls=[],
        )
        _, report = run_planner_with_report(
            skeleton, packs, llm=llm, repo_root=str(tmp_path)
        )
        assert report.tool_budget_total == 3 * 5  # 15

    def test_tool_budget_exceeded_when_forced(self, tmp_path):
        """Simulate budget exhaustion by having budget start at zero via a
        skeleton with zero clusters but force the tracker condition."""
        # The only way to exhaust with a real run is to have LLM request tools.
        # We test the _plan_cluster path that consumes budget by using a
        # FakeLLM subclass that includes tool_calls in its response.

        class ToolCallingFakeLLM(BaseChatModel):
            model_name: str = "fake-tool-caller"
            call_count: int = 0

            @property
            def _llm_type(self) -> str:
                return "fake-tool-caller"

            def _generate(
                self,
                messages: list[BaseMessage],
                stop: Any = None,
                run_manager: Any = None,
                **kwargs: Any,
            ) -> ChatResult:
                self.call_count += 1
                if self.call_count <= 10:
                    # Return a tool call every time
                    ai_msg = AIMessage(
                        content="",
                        tool_calls=[
                            {
                                "name": "grep",
                                "args": {"pattern": "test"},
                                "id": f"call_{self.call_count}",
                            }
                        ],
                    )
                    return ChatResult(generations=[ChatGeneration(message=ai_msg)])
                # Eventually return a real answer
                return ChatResult(
                    generations=[
                        ChatGeneration(
                            message=AIMessage(
                                content=_json_response(1)
                            )
                        )
                    ]
                )

            def _stream(self, *args: Any, **kwargs: Any) -> Iterator[Any]:
                raise NotImplementedError

        cluster = _make_cluster(cluster_id=1, kind="code", dirs=["src"])
        skeleton = _make_skeleton([cluster])
        pack = _make_pack(cluster_id=1)

        llm = ToolCallingFakeLLM()
        pages, report = run_planner_with_report(
            skeleton, {1: pack}, llm=llm, repo_root=str(tmp_path)
        )

        # With 1 cluster, budget = 3. LLM calls tools repeatedly.
        # The budget should be consumed (≥ 3 tool calls attempted).
        assert report.tool_budget_used >= report.tool_budget_total
        assert report.tool_budget_exceeded


# ── Tests: missing evidence pack ─────────────────────────────────────────────


class TestMissingEvidencePack:
    def test_missing_pack_produces_fallback_with_note(self, tmp_path):
        cluster = _make_cluster(cluster_id=1)
        skeleton = _make_skeleton([cluster])
        # No pack provided
        llm = FakeLLM(
            responses=[_json_response(1, title="Module Components")],
            record_calls=[],
        )
        pages, report = run_planner_with_report(
            skeleton, {}, llm=llm, repo_root=str(tmp_path)
        )
        # Should still produce a page (either LLM or fallback)
        assert len(pages) == 1
        # Note should mention missing pack
        assert any("no evidence pack" in n for n in report.notes)

    def test_missing_pack_page_has_skeleton_fields(self, tmp_path):
        arts = [_make_artifact(name="PubSym", layer="public_api", source_path="src/a.py")]
        cluster = _make_cluster(cluster_id=1, artifacts=arts, dirs=["src/api"])
        skeleton = _make_skeleton([cluster])
        llm = FakeLLM(responses=[_json_response(1, title="API Layer")], record_calls=[])
        pages, _ = run_planner_with_report(
            skeleton, {}, llm=llm, repo_root=str(tmp_path)
        )
        assert "PubSym" in pages[0].target_symbols
        assert "src/api" in pages[0].target_folders


# ── Tests: malformed LLM JSON ─────────────────────────────────────────────────


class TestMalformedLLMResponse:
    def test_malformed_json_produces_fallback_spec(self, tmp_path):
        cluster = _make_cluster(cluster_id=1, dirs=["src/auth"])
        skeleton = _make_skeleton([cluster])
        pack = _make_pack(cluster_id=1)
        # Return garbage that is not valid JSON
        llm = FakeLLM(responses=["This is not JSON at all!"], record_calls=[])
        pages, report = run_planner_with_report(
            skeleton, {1: pack}, llm=llm, repo_root=str(tmp_path)
        )
        assert len(pages) == 1
        # Fallback title should be derived from directory
        assert pages[0].cluster_id == 1
        # Note should mention malformed JSON
        assert any("malformed" in n or "fallback" in n for n in report.notes)

    def test_fallback_spec_has_skeleton_fields(self, tmp_path):
        arts = [_make_artifact(name="PubClass", layer="public_api", source_path="lib/c.py")]
        cluster = _make_cluster(cluster_id=3, artifacts=arts, dirs=["lib/core"])
        skeleton = _make_skeleton([cluster])
        pack = _make_pack(cluster_id=3)
        llm = FakeLLM(responses=["definitely not json"], record_calls=[])
        pages, _ = run_planner_with_report(
            skeleton, {3: pack}, llm=llm, repo_root=str(tmp_path)
        )
        # target_* must be from skeleton regardless
        assert "PubClass" in pages[0].target_symbols
        assert "lib/core" in pages[0].target_folders

    def test_empty_string_response_produces_fallback(self, tmp_path):
        cluster = _make_cluster(cluster_id=1)
        skeleton = _make_skeleton([cluster])
        pack = _make_pack(cluster_id=1)
        llm = FakeLLM(responses=[""], record_calls=[])
        pages, report = run_planner_with_report(
            skeleton, {1: pack}, llm=llm, repo_root=str(tmp_path)
        )
        assert len(pages) == 1
        assert report.pages_emitted == 1


# ── Tests: prompt variants by cluster kind ────────────────────────────────────


class TestKindPromptFragments:
    """_kind_prompt_fragment returns distinct text for each cluster kind."""

    def test_code_fragment_mentions_source_code(self):
        frag = _kind_prompt_fragment("code")
        assert "source code" in frag.lower()

    def test_doc_fragment_mentions_documentation(self):
        frag = _kind_prompt_fragment("doc")
        assert "documentation" in frag.lower()

    def test_confluence_fragment_mentions_confluence(self):
        frag = _kind_prompt_fragment("confluence")
        assert "confluence" in frag.lower()

    def test_jira_fragment_mentions_jira(self):
        frag = _kind_prompt_fragment("jira")
        assert "jira" in frag.lower()

    def test_unknown_kind_returns_generic_fragment(self):
        frag = _kind_prompt_fragment("unknown_kind")
        assert isinstance(frag, str)
        assert len(frag) > 0

    def test_all_four_kinds_produce_distinct_fragments(self):
        fragments = {k: _kind_prompt_fragment(k) for k in ("code", "doc", "confluence", "jira")}
        assert len(set(fragments.values())) == 4


# ── Tests: build_user_prompt ──────────────────────────────────────────────────


class TestBuildUserPrompt:
    def test_prompt_contains_cluster_id(self):
        cluster = _make_cluster(cluster_id=99)
        pack = _make_pack(cluster_id=99)
        prompt = build_user_prompt(cluster, pack, tool_budget_remaining=6, tool_budget_total=6)
        assert "99" in prompt

    def test_prompt_contains_dirs(self):
        cluster = _make_cluster(dirs=["src/special_dir"])
        pack = _make_pack()
        prompt = build_user_prompt(cluster, pack, tool_budget_remaining=3, tool_budget_total=3)
        assert "src/special_dir" in prompt

    def test_prompt_contains_budget_info(self):
        cluster = _make_cluster()
        pack = _make_pack()
        prompt = build_user_prompt(cluster, pack, tool_budget_remaining=5, tool_budget_total=9)
        assert "5" in prompt
        assert "9" in prompt

    def test_prompt_handles_none_pack(self):
        cluster = _make_cluster()
        prompt = build_user_prompt(cluster, None, tool_budget_remaining=3, tool_budget_total=3)
        assert "no evidence pack" in prompt.lower() or "no evidence" in prompt.lower()

    def test_system_prompt_is_non_empty(self):
        assert len(PLANNER_SYSTEM_PROMPT) > 100


# ── Tests: validation pass ────────────────────────────────────────────────────


class TestValidationPass:
    """Every cluster in the skeleton must have exactly one PageSpec."""

    def test_all_clusters_covered_in_output(self, tmp_path):
        clusters = [_make_cluster(cluster_id=i) for i in range(1, 4)]
        skeleton = _make_skeleton(clusters)
        packs = {c.cluster_id: _make_pack(c.cluster_id) for c in clusters}
        llm = FakeLLM(
            responses=[_json_response(i) for i in range(1, 4)],
            record_calls=[],
        )
        pages, report = run_planner_with_report(
            skeleton, packs, llm=llm, repo_root=str(tmp_path)
        )
        assert len(pages) == 3
        assert {p.cluster_id for p in pages} == {1, 2, 3}

    def test_pages_emitted_matches_page_count(self, tmp_path):
        clusters = [_make_cluster(cluster_id=i) for i in range(1, 6)]
        skeleton = _make_skeleton(clusters)
        packs = {c.cluster_id: _make_pack(c.cluster_id) for c in clusters}
        llm = FakeLLM(
            responses=[_json_response(i) for i in range(1, 6)],
            record_calls=[],
        )
        pages, report = run_planner_with_report(
            skeleton, packs, llm=llm, repo_root=str(tmp_path)
        )
        assert report.pages_emitted == len(pages) == 5


# ── Tests: backwards-compat with code_clusters ───────────────────────────────


class TestBackwardsCompatCodeClusters:
    """run_planner_with_report falls back to code_clusters when clusters=[]."""

    def test_uses_code_clusters_when_unified_list_empty(self, tmp_path):
        sym = SymbolInfo(
            name="Foo", type="class", rel_path="src/foo.py", layer="public_api", connections=2, docstring=""
        )
        dir_cluster = DirCluster(
            cluster_id=1,
            dirs=["src"],
            symbols=[sym],
            total_symbols=1,
            primary_languages=["py"],
            depth_range=(1, 1),
        )
        skeleton = StructureSkeleton(
            code_clusters=[dir_cluster],
            doc_clusters=[],
            total_arch_symbols=1,
            total_dirs_covered=1,
            total_dirs_in_repo=1,
            repo_languages=["py"],
            effective_depth=2,
            repo_name="test",
            # clusters is intentionally left as [] (default)
        )
        pack = _make_pack(cluster_id=1)
        llm = FakeLLM(responses=[_json_response(1, title="Foo Layer")], record_calls=[])
        pages, report = run_planner_with_report(
            skeleton, {1: pack}, llm=llm, repo_root=str(tmp_path)
        )
        assert len(pages) == 1
        assert report.pages_emitted == 1


# ── LLM raises (Rio #268 1st pass) ───────────────────────────────────────────


class TestLLMException:
    """If the LLM invocation raises, the planner must catch it and synthesise
    a fallback PageSpec rather than propagating the exception.  Without this
    test the broad ``except Exception`` in _run_planner_core was uncovered."""

    def test_llm_exception_yields_fallback_spec_with_note(self, tmp_path):
        class RaisingLLM(FakeLLM):
            def _generate(self, *args: Any, **kwargs: Any) -> Any:
                raise RuntimeError("simulated upstream LLM failure")

        cluster = _make_cluster(cluster_id=1, kind="code", dirs=["src/auth"])
        skeleton = _make_skeleton([cluster])
        pack = _make_pack(cluster_id=1)
        llm = RaisingLLM(responses=[], record_calls=[])

        pages, report = run_planner_with_report(
            skeleton, {1: pack}, llm=llm, repo_root=str(tmp_path)
        )

        assert len(pages) == 1
        spec = pages[0]
        assert spec.cluster_id == 1
        assert spec.title  # non-empty title from fallback
        assert spec.description
        assert spec.retrieval_query
        notes = report.notes
        assert any("LLM call failed" in n for n in notes), (
            f"expected an 'LLM call failed' note, got: {notes}"
        )


# ── Fallback spec wording (Rio #268 1st pass) ────────────────────────────────


class TestFallbackSpecAlwaysHasContent:
    """``_fallback_spec`` must never emit an empty description or empty
    retrieval_query — both are load-bearing for the downstream writer."""

    def test_fallback_with_empty_dirs_uses_artifact_names(self, tmp_path):
        # No dirs (e.g. jira cluster); has artifact names.
        art = _make_artifact(name="EPIC-1", source_path="EPIC-1.md")
        cluster = _make_cluster(cluster_id=7, kind="jira", dirs=[], artifacts=[art])
        skeleton = _make_skeleton([cluster])

        class FailLLM(FakeLLM):
            def _generate(self, *args: Any, **kwargs: Any) -> Any:
                raise RuntimeError("force fallback")

        llm = FailLLM(responses=[], record_calls=[])
        pages, _report = run_planner_with_report(
            skeleton, {}, llm=llm, repo_root=str(tmp_path)
        )
        assert len(pages) == 1
        spec = pages[0]
        assert spec.description and spec.description.strip()
        assert spec.retrieval_query and spec.retrieval_query.strip()

    def test_fallback_with_no_dirs_and_no_named_artifacts(self, tmp_path):
        art = _make_artifact(name="", source_path="x.md")
        cluster = _make_cluster(cluster_id=9, kind="confluence", dirs=[], artifacts=[art])
        skeleton = _make_skeleton([cluster])

        class FailLLM(FakeLLM):
            def _generate(self, *args: Any, **kwargs: Any) -> Any:
                raise RuntimeError("force fallback")

        llm = FailLLM(responses=[], record_calls=[])
        pages, _report = run_planner_with_report(
            skeleton, {}, llm=llm, repo_root=str(tmp_path)
        )
        assert pages[0].description and pages[0].description.strip()
        assert pages[0].retrieval_query and pages[0].retrieval_query.strip()


# ── Empty-field LLM response (Rio #268 1st pass) ─────────────────────────────


class TestEmptyFieldRejection:
    """When the LLM returns valid JSON but with empty title/description/
    retrieval_query, the planner must fall back rather than emit a half-empty
    spec to the writer."""

    def test_empty_retrieval_query_triggers_fallback(self, tmp_path):
        cluster = _make_cluster(cluster_id=1, kind="code", dirs=["src/x"])
        skeleton = _make_skeleton([cluster])
        pack = _make_pack(cluster_id=1)

        bad_response = '{"title": "X Components", "description": "Does X.", "retrieval_query": ""}'
        llm = FakeLLM(responses=[bad_response], record_calls=[])
        pages, report = run_planner_with_report(
            skeleton, {1: pack}, llm=llm, repo_root=str(tmp_path)
        )

        assert len(pages) == 1
        spec = pages[0]
        # Fallback retrieval_query must be non-empty.
        assert spec.retrieval_query.strip()
        assert any("missing" in n and "retrieval_query" in n for n in report.notes), (
            f"expected a missing-field note: {report.notes}"
        )

    def test_empty_description_triggers_fallback(self, tmp_path):
        cluster = _make_cluster(cluster_id=1, kind="code", dirs=["src/x"])
        skeleton = _make_skeleton([cluster])
        pack = _make_pack(cluster_id=1)

        bad_response = '{"title": "X", "description": "", "retrieval_query": "x y z"}'
        llm = FakeLLM(responses=[bad_response], record_calls=[])
        pages, report = run_planner_with_report(
            skeleton, {1: pack}, llm=llm, repo_root=str(tmp_path)
        )

        assert pages[0].description.strip()
        assert any("missing" in n and "description" in n for n in report.notes)
