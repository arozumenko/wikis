"""Unified wiki pipeline orchestrator (#243, default path since #242).

Wires the unified planner → writer → source-kind gate → citation verifier
into a single end-to-end pipeline.  This is the sole structure-planning
path; the legacy graph-first, deepagents, and cluster-based paths were
removed in #242.

Public API
----------
``run_unified_pipeline(skeleton, evidence_packs, *, llm, verifier_llm,
                        repo_root, name_index, stream_callback)``
    → ``UnifiedPipelineResult``

``UnifiedPipelineResult``
    ``pages``          – list of ``GeneratedPage`` (final, post-verified)
    ``plan_report``    – ``PlanReport`` from the unified planner
    ``gate_reports``   – ``dict[int, GateReport]`` keyed by ``page_id``
    ``verify_reports`` – ``dict[int, VerifyReport]`` keyed by ``page_id``

``GeneratedPage``
    ``page_id``        – ``cluster_id`` from the originating ``PageSpec``
    ``title``          – page title (from planner or fallback)
    ``description``    – 1-2 sentence description
    ``markdown``       – final stripped / verified markdown
    ``cited_claims``   – kept ``CitedClaim`` list (post-verifier)
    ``metadata``       – dict with target_* fields + gate/verifier flags
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Callable

from langchain_core.language_models import BaseChatModel

from app.core.wiki_content_writer.citation_extractor import CitedClaim, extract_citations
from app.core.wiki_content_writer.citation_verifier import VerifyReport, verify_citations
from app.core.wiki_content_writer.source_gate import GateReport, ToolCall, apply_gate
from app.core.wiki_structure_planner.evidence import EvidencePack
from app.core.wiki_structure_planner.planner_agent import (
    PageSpec,
    PlanReport,
    run_planner_with_report,
)
from app.core.wiki_structure_planner.structure_skeleton import StructureSkeleton

logger = logging.getLogger(__name__)

# Default gate mode: README + SQL use strip, identifier defers to the verifier.
_DEFAULT_GATE_MODE: dict[str, str] = {
    "readme": "strip",
    "sql": "strip",
    "identifier": "verifier",
}

# Batch size for the citation verifier LLM calls.
_VERIFIER_BATCH_SIZE = 4


# ── Result dataclasses ────────────────────────────────────────────────────────


@dataclass
class GeneratedPage:
    """A single wiki page produced by the unified pipeline.

    Attributes
    ----------
    page_id : int
        Cluster ID of the originating ``PageSpec``.
    title : str
        Page title from the unified planner (or deterministic fallback).
    description : str
        1-2 sentence description.
    markdown : str
        Final markdown content after gate and verifier stripping.
    cited_claims : list[CitedClaim]
        Kept ``CitedClaim`` objects (post-verifier) for downstream use.
    metadata : dict
        Carries ``target_symbols``, ``target_folders``, ``target_docs`` from
        the skeleton plus ``gate_flags_total``, ``gate_stripped_total``,
        ``verify_stripped_total``, ``verify_partial_total`` for observability.
    """

    page_id: int
    title: str
    description: str
    markdown: str
    cited_claims: list[CitedClaim] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class UnifiedPipelineResult:
    """Result of :func:`run_unified_pipeline`.

    Attributes
    ----------
    pages : list[GeneratedPage]
        Final generated pages (one per cluster).
    plan_report : PlanReport
        Budget and coverage summary from the unified planner.
    gate_reports : dict[int, GateReport]
        Per-page gate reports keyed by ``cluster_id``.
    verify_reports : dict[int, VerifyReport]
        Per-page verifier reports keyed by ``cluster_id``.
    """

    pages: list[GeneratedPage]
    plan_report: PlanReport
    gate_reports: dict[int, GateReport] = field(default_factory=dict)
    verify_reports: dict[int, VerifyReport] = field(default_factory=dict)


# ── Writer stub ───────────────────────────────────────────────────────────────
# The full WikiContentWriter.generate_page is implemented in #243's follow-up.
# For now we call the LLM directly with a minimal prompt so the pipeline
# end-to-end contract is exercisable.


def _generate_page_markdown(
    spec: PageSpec,
    llm: BaseChatModel,
) -> tuple[str, list[ToolCall]]:
    """Generate markdown for *spec* via a direct LLM call.

    Returns
    -------
    (markdown, tool_trace)
        ``markdown`` – generated page content (may contain citation tokens).
        ``tool_trace`` – list of ``ToolCall`` objects from the writer agent.
        In this stub the tool trace is always empty because no tools are
        actually invoked; the gate falls back to flag-only mode for the
        identifier rule and can still strip README / SQL claims.
    """
    from langchain_core.messages import HumanMessage, SystemMessage  # noqa: PLC0415

    system = (
        "You are a technical documentation writer. "
        "Write a detailed wiki page in Markdown based on the specification below. "
        "Be accurate and concise. Do NOT invent symbol names or file paths. "
        "Inline citation tokens like [path/to/file.py:1-10] after each claim."
    )
    user = (
        f"## Page Title\n{spec.title}\n\n"
        f"## Description\n{spec.description}\n\n"
        f"## Retrieval Query\n{spec.retrieval_query}\n\n"
        f"## Target Symbols\n{', '.join(spec.target_symbols) or 'none'}\n\n"
        f"## Target Folders\n{', '.join(spec.target_folders) or 'none'}\n\n"
        f"## Target Docs\n{', '.join(spec.target_docs) or 'none'}\n\n"
        "Write the wiki page now."
    )

    try:
        response = llm.invoke(
            [SystemMessage(content=system), HumanMessage(content=user)]
        )
        content = response.content if isinstance(response.content, str) else str(response.content)
    except Exception as exc:
        logger.error("Writer LLM call failed for page %r: %s", spec.title, exc)
        content = f"# {spec.title}\n\n{spec.description}\n"

    return content, []


# ── Per-page pipeline ─────────────────────────────────────────────────────────


def _run_page(
    spec: PageSpec,
    *,
    llm: BaseChatModel,
    verifier_llm: BaseChatModel,
    repo_root: str,
    name_index: set[str] | None,
    gate_mode: dict[str, str],
    stream_callback: Callable[[dict], None] | None,
) -> tuple[GeneratedPage, GateReport, VerifyReport]:
    """Run the writer → gate → verifier pipeline for one ``PageSpec``.

    Args:
        spec: Planner output for this page.
        llm: Primary LLM used for content generation.
        verifier_llm: LLM used for citation verification (may equal ``llm``).
        repo_root: Absolute path to the checked-out repository root.
        name_index: Set of known symbol names for identifier grounding.
        gate_mode: Per-rule gate actions (passed through to ``apply_gate``).
        stream_callback: Optional SSE event emitter.

    Returns:
        ``(page, gate_report, verify_report)``
    """
    page_id = spec.cluster_id

    # ── 1. Emit writer.page_started ───────────────────────────────────
    if stream_callback:
        try:
            stream_callback({
                "event": "writer.page_started",
                "page_id": page_id,
                "title": spec.title,
            })
        except Exception as cb_exc:
            logger.debug("stream_callback error (writer.page_started): %s", cb_exc)

    # ── 2. Generate content ───────────────────────────────────────────
    logger.info("[UNIFIED] writer: page_id=%d title=%r", page_id, spec.title)
    raw_markdown, tool_trace = _generate_page_markdown(spec, llm)

    # ── 3. Emit writer.page_done ──────────────────────────────────────
    cited_claims_raw = extract_citations(raw_markdown)
    if stream_callback:
        try:
            stream_callback({
                "event": "writer.page_done",
                "page_id": page_id,
                "n_paragraphs": len(cited_claims_raw),
            })
        except Exception as cb_exc:
            logger.debug("stream_callback error (writer.page_done): %s", cb_exc)

    # ── 4. Source-kind gate ───────────────────────────────────────────
    logger.info(
        "[UNIFIED] gate: page_id=%d paragraphs=%d",
        page_id,
        len(cited_claims_raw),
    )
    kept_after_gate, gate_report = apply_gate(
        cited_claims_raw,
        tool_trace,
        known_symbols=name_index,
        mode=gate_mode,
    )

    n_flagged = len(gate_report.flags)
    n_gate_stripped = len(gate_report.stripped_paragraphs)
    logger.info(
        "[UNIFIED] gate done: page_id=%d flagged=%d stripped=%d",
        page_id,
        n_flagged,
        n_gate_stripped,
    )

    if n_gate_stripped:
        _emit_stripped(
            stream_callback,
            event="gate.page_done",
            page_id=page_id,
            n_flagged=n_flagged,
            n_stripped=n_gate_stripped,
        )
    else:
        if stream_callback:
            try:
                stream_callback({
                    "event": "gate.page_done",
                    "page_id": page_id,
                    "n_flagged": n_flagged,
                    "n_stripped": 0,
                })
            except Exception as cb_exc:
                logger.debug("stream_callback error (gate.page_done): %s", cb_exc)

    # ── 5. Citation verifier ──────────────────────────────────────────
    logger.info(
        "[UNIFIED] verifier: page_id=%d paragraphs=%d",
        page_id,
        len(kept_after_gate),
    )
    kept_after_verify, verify_report = verify_citations(
        kept_after_gate,
        llm=verifier_llm,
        repo_root=repo_root,
        batch_size=_VERIFIER_BATCH_SIZE,
    )

    n_verify_stripped = verify_report.paragraphs_stripped
    n_verify_partial = verify_report.paragraphs_partial

    logger.info(
        "[UNIFIED] verifier done: page_id=%d kept=%d stripped=%d partial=%d",
        page_id,
        verify_report.paragraphs_kept,
        n_verify_stripped,
        n_verify_partial,
    )

    if stream_callback:
        try:
            stream_callback({
                "event": "verifier.page_done",
                "page_id": page_id,
                "n_kept": verify_report.paragraphs_kept,
                "n_stripped": n_verify_stripped,
                "n_partial": n_verify_partial,
            })
        except Exception as cb_exc:
            logger.debug("stream_callback error (verifier.page_done): %s", cb_exc)

    # Log partial-verdict warnings
    for verdict in verify_report.verdicts:
        if verdict.verdict == "partial":
            logger.warning(
                "[UNIFIED] partial citation in page_id=%d para=%d: %s",
                page_id,
                verdict.paragraph_index,
                verdict.reason,
            )

    # ── 6. Reconstruct final markdown from kept paragraphs ────────────
    final_markdown = _claims_to_markdown(kept_after_verify)

    page = GeneratedPage(
        page_id=page_id,
        title=spec.title,
        description=spec.description,
        markdown=final_markdown,
        cited_claims=kept_after_verify,
        metadata={
            "target_symbols": spec.target_symbols,
            "target_folders": spec.target_folders,
            "target_docs": spec.target_docs,
            "gate_flags_total": n_flagged,
            "gate_stripped_total": n_gate_stripped,
            "verify_stripped_total": n_verify_stripped,
            "verify_partial_total": n_verify_partial,
        },
    )
    return page, gate_report, verify_report


# ── Markdown reconstruction ───────────────────────────────────────────────────


def _claims_to_markdown(claims: list[CitedClaim]) -> str:
    """Reconstruct markdown from kept ``CitedClaim`` objects.

    Paragraphs are joined with a blank line between them (CommonMark paragraph
    separator).  If the claim list is empty, an empty string is returned.
    """
    if not claims:
        return ""
    return "\n\n".join(c.paragraph_text for c in claims)


# ── Audit event helper ────────────────────────────────────────────────────────


def _emit_stripped(
    stream_callback: Callable[[dict], None] | None,
    *,
    event: str,
    page_id: int,
    **kwargs: Any,
) -> None:
    """Emit a stripped-paragraph audit event via *stream_callback*.

    Swallows all exceptions so audit events never break the pipeline.
    """
    if stream_callback is None:
        return
    payload: dict[str, Any] = {"event": event, "page_id": page_id, **kwargs}
    try:
        stream_callback(payload)
    except Exception as exc:
        logger.debug("stream_callback error (%s): %s", event, exc)


# ── Public entrypoint ─────────────────────────────────────────────────────────


def run_unified_pipeline(
    skeleton: StructureSkeleton,
    evidence_packs: dict[int, EvidencePack],
    *,
    llm: BaseChatModel,
    verifier_llm: BaseChatModel | None = None,
    repo_root: str,
    name_index: set[str] | None = None,
    gate_mode: dict[str, str] | None = None,
    stream_callback: Callable[[dict], None] | None = None,
) -> UnifiedPipelineResult:
    """Run the unified planner → writer → gate → verifier pipeline.

    This is the single public entrypoint for the unified wiki generation
    pipeline.  It is deliberately synchronous — async orchestration is the
    responsibility of the caller (``OptimizedWikiGenerationAgent`` or a test
    harness).

    Parameters
    ----------
    skeleton :
        Completed ``StructureSkeleton`` with ``clusters`` (preferred) or
        ``code_clusters`` (backwards-compat fallback).
    evidence_packs :
        Pre-built evidence packs keyed by ``cluster_id``.  Missing packs
        cause the planner to use a fallback spec with a note in the report.
    llm :
        Primary LangChain ``BaseChatModel`` used for both the planner and the
        writer content generation.
    verifier_llm :
        LangChain ``BaseChatModel`` used for citation verification.  Defaults
        to *llm* if ``None``.
    repo_root :
        Absolute path to the checked-out repository root.  Passed to the
        planner (tool IO), the gate (identifier grounding), and the verifier
        (span reading).
    name_index :
        Set of known symbol names from the code graph.  Used by the source-
        kind gate (rule 3 — identifier grounding).  Pass ``None`` to rely
        solely on the tool trace for grounding.
    gate_mode :
        Per-rule action override for :func:`apply_gate`.  Defaults to
        ``{"readme": "strip", "sql": "strip", "identifier": "verifier"}``.
    stream_callback :
        Optional function called with a ``dict`` event payload for each
        pipeline milestone.  Events: ``planner.started``, ``planner.done``,
        ``writer.page_started``, ``writer.page_done``, ``gate.page_done``,
        ``verifier.page_done``.  All callbacks are wrapped in try/except so
        a misbehaving callback never breaks the pipeline.

    Returns
    -------
    UnifiedPipelineResult
        ``pages`` — one ``GeneratedPage`` per cluster (empty when skeleton
        has no clusters).
        ``plan_report`` — planner budget / coverage summary.
        ``gate_reports`` — per-page gate reports (empty dict when no pages).
        ``verify_reports`` — per-page verifier reports (empty dict when no pages).

    Raises
    ------
    ValueError
        If *repo_root* is not a valid directory (propagated from the planner).
    """
    effective_verifier_llm = verifier_llm if verifier_llm is not None else llm
    effective_gate_mode = gate_mode if gate_mode is not None else _DEFAULT_GATE_MODE

    # ── Planner phase ─────────────────────────────────────────────────
    if stream_callback:
        try:
            stream_callback({"event": "planner.started"})
        except Exception as cb_exc:
            logger.debug("stream_callback error (planner.started): %s", cb_exc)

    logger.info(
        "[UNIFIED] planner: clusters=%d evidence_packs=%d",
        len(skeleton.clusters or skeleton.code_clusters),
        len(evidence_packs),
    )

    page_specs, plan_report = run_planner_with_report(
        skeleton,
        evidence_packs,
        llm=llm,
        repo_root=repo_root,
    )

    logger.info(
        "[UNIFIED] planner done: pages=%d tool_calls=%d/%d exceeded=%s",
        plan_report.pages_emitted,
        plan_report.tool_budget_used,
        plan_report.tool_budget_total,
        plan_report.tool_budget_exceeded,
    )

    if stream_callback:
        try:
            stream_callback({
                "event": "planner.done",
                "pages_emitted": plan_report.pages_emitted,
                "tool_budget_used": plan_report.tool_budget_used,
                "tool_budget_total": plan_report.tool_budget_total,
                "tool_budget_exceeded": plan_report.tool_budget_exceeded,
                "notes": plan_report.notes,
            })
        except Exception as cb_exc:
            logger.debug("stream_callback error (planner.done): %s", cb_exc)

    if not page_specs:
        logger.info("[UNIFIED] no page specs produced — returning empty result")
        return UnifiedPipelineResult(
            pages=[],
            plan_report=plan_report,
            gate_reports={},
            verify_reports={},
        )

    # ── Per-page writer → gate → verifier ────────────────────────────
    pages: list[GeneratedPage] = []
    gate_reports: dict[int, GateReport] = {}
    verify_reports: dict[int, VerifyReport] = {}

    for spec in page_specs:
        try:
            page, gate_report, verify_report = _run_page(
                spec,
                llm=llm,
                verifier_llm=effective_verifier_llm,
                repo_root=repo_root,
                name_index=name_index,
                gate_mode=effective_gate_mode,
                stream_callback=stream_callback,
            )
            pages.append(page)
            gate_reports[spec.cluster_id] = gate_report
            verify_reports[spec.cluster_id] = verify_report
        except Exception as exc:
            logger.error(
                "[UNIFIED] page pipeline failed for cluster_id=%d title=%r: %s",
                spec.cluster_id,
                spec.title,
                exc,
                exc_info=True,
            )
            # Emit a minimal page so the caller always gets one entry per spec.
            fallback_markdown = f"# {spec.title}\n\n{spec.description}\n"
            pages.append(
                GeneratedPage(
                    page_id=spec.cluster_id,
                    title=spec.title,
                    description=spec.description,
                    markdown=fallback_markdown,
                    metadata={
                        "target_symbols": spec.target_symbols,
                        "target_folders": spec.target_folders,
                        "target_docs": spec.target_docs,
                        "pipeline_error": str(exc),
                    },
                )
            )
            gate_reports[spec.cluster_id] = GateReport()
            from app.core.wiki_content_writer.citation_verifier import VerifyReport  # noqa: PLC0415

            verify_reports[spec.cluster_id] = VerifyReport()

    logger.info(
        "[UNIFIED] pipeline complete: pages=%d gate_stripped_total=%d verify_stripped_total=%d",
        len(pages),
        sum(len(r.stripped_paragraphs) for r in gate_reports.values()),
        sum(r.paragraphs_stripped for r in verify_reports.values()),
    )

    return UnifiedPipelineResult(
        pages=pages,
        plan_report=plan_report,
        gate_reports=gate_reports,
        verify_reports=verify_reports,
    )
