"""Citation verifier — LLM second pass (#239).

After ``extract_citations`` (#238) parses inline ``[path:lo-hi]`` tokens and
``source_gate.apply_gate`` (#240) pre-filters obvious hallucinations, this
module reads the cited file spans from disk and asks a cheap LLM whether each
paragraph faithfully reflects those spans.

Per-paragraph state machine
---------------------------
supported         → keep (any one valid citation faithfully grounds the claim)
partial           → keep + warn (some content not grounded but overlap exists)
absent            → strip (no citation supports the paragraph)
citation_missing  → strip (paragraph has no citations at all)

Public API
----------
VerifyVerdict  — per-paragraph verdict record.
VerifyReport   — aggregate result of ``verify_citations``.
verify_citations(cited_claims, *, llm, repo_root, batch_size) → (kept, report)

Implementation notes
--------------------
* Path safety via ``safe_join`` from ``_evidence_utils`` — any citation that
  escapes ``repo_root`` is treated as ``citation_missing``.
* Lines are 1-indexed throughout (same model as ``extract_citations``).
* Span size cap: hard cap at 200 lines.  Larger spans are truncated and the
  verdict reason notes the truncation.
* No regex JSON parsing: ``_extract_json_array`` does trailing-comma cleanup
  then ``json.loads``.  If the LLM returns the wrong number of verdicts, all
  mismatched positions default to ``partial`` with reason
  ``"verifier-output-malformed"``.
* LLM calls are batched: ``batch_size`` paragraphs per call.
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field
from typing import Any

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage

from app.core.wiki_content_writer.citation_extractor import CitedClaim
from app.core.wiki_content_writer.verifier_prompts import (
    VERIFIER_SYSTEM_PROMPT,
    ClaimWithSpans,
    SpanText,
    build_verifier_user_prompt,
)
from app.core.wiki_structure_planner._evidence_utils import safe_join

logger = logging.getLogger(__name__)

# ── Constants ─────────────────────────────────────────────────────────────────

_SPAN_LINE_CAP = 200
_VALID_VERDICTS: frozenset[str] = frozenset({"supported", "partial", "absent"})
_FALLBACK_VERDICT = "partial"
_FALLBACK_REASON = "verifier-output-malformed"

# ── Public dataclasses ────────────────────────────────────────────────────────


@dataclass
class VerifyVerdict:
    """Per-paragraph verdict from the LLM verifier.

    Attributes
    ----------
    paragraph_index : int
        0-based index of the paragraph in the original ``CitedClaim`` list.
    verdict : str
        One of ``"supported"``, ``"partial"``, ``"absent"``,
        ``"citation_missing"``.
    reason : str
        One-line human-readable explanation.
    citations_checked : int
        Number of citation spans sent to the LLM for this paragraph (0 for
        ``citation_missing``).
    """

    paragraph_index: int
    verdict: str
    reason: str
    citations_checked: int


@dataclass
class VerifyReport:
    """Aggregate result of :func:`verify_citations`.

    Attributes
    ----------
    verdicts : list[VerifyVerdict]
        All per-paragraph verdicts in paragraph order.
    paragraphs_total : int
        Total number of input paragraphs.
    paragraphs_kept : int
        Number of paragraphs in the returned ``kept_claims`` list.
    paragraphs_stripped : int
        Number of paragraphs stripped (``absent`` or ``citation_missing``).
    paragraphs_partial : int
        Number of paragraphs with ``partial`` verdict (kept with a warning).
    llm_calls : int
        Number of batched LLM invocations made.
    """

    verdicts: list[VerifyVerdict] = field(default_factory=list)
    paragraphs_total: int = 0
    paragraphs_kept: int = 0
    paragraphs_stripped: int = 0
    paragraphs_partial: int = 0
    llm_calls: int = 0


# ── JSON extraction ───────────────────────────────────────────────────────────

_JSON_ARR_RE = re.compile(r"\[.*\]", re.DOTALL)


def _extract_json_array(text: str) -> list[Any] | None:
    """Extract the first JSON array from *text*.

    Strips markdown fences, tries to parse the whole string as JSON,
    falls back to a regex scan for the first ``[...]`` block, and on each
    attempt applies trailing-comma cleanup for common LLM sloppiness like
    ``[{"verdict": "supported",}]``.  Returns ``None`` on failure.
    """
    stripped = re.sub(r"```(?:json)?\s*", "", text).strip()

    def _try_parse(candidate: str) -> list[Any] | None:
        try:
            obj = json.loads(candidate)
        except json.JSONDecodeError:
            cleaned = re.sub(r",\s*([}\]])", r"\1", candidate)
            if cleaned == candidate:
                return None
            try:
                obj = json.loads(cleaned)
            except json.JSONDecodeError:
                return None
        return obj if isinstance(obj, list) else None

    direct = _try_parse(stripped)
    if direct is not None:
        return direct

    for m in _JSON_ARR_RE.finditer(stripped):
        candidate = _try_parse(m.group(0))
        if candidate is not None:
            return candidate

    return None


# ── Span reading ──────────────────────────────────────────────────────────────


def _read_span(repo_root: str, citation_path: str, start_line: int, end_line: int) -> SpanText:
    """Read lines ``[start_line, end_line]`` from *citation_path* under *repo_root*.

    Lines are 1-indexed.  If ``end_line`` exceeds the file length, the span is
    clamped to EOF and a warning is logged.  Spans larger than ``_SPAN_LINE_CAP``
    are truncated to that cap and marked ``truncated=True``.

    If the path is unsafe or the file cannot be read, returns a ``SpanText``
    with empty ``text``.
    """
    safe_path = safe_join(repo_root, citation_path)
    if safe_path is None:
        return SpanText(
            citation_path=citation_path,
            start_line=start_line,
            end_line=end_line,
            text="",
        )

    # Read only the lines we actually need: skip up to start_line, then take
    # min(span_length, _SPAN_LINE_CAP) lines.  This keeps a citation pointing
    # into a huge generated/bundled file (lockfile, compiled JS, etc.) from
    # forcing the verifier to load megabytes just to discard them.
    lo_1based = max(1, start_line)
    requested = max(0, end_line - lo_1based + 1)
    capped = min(requested, _SPAN_LINE_CAP)
    truncated = requested > _SPAN_LINE_CAP

    try:
        from itertools import islice  # noqa: PLC0415

        with open(safe_path, encoding="utf-8", errors="replace") as fh:
            # Skip to start_line - 1, then take up to `capped` lines.
            selected = list(islice(fh, lo_1based - 1, lo_1based - 1 + capped))
    except OSError:
        return SpanText(
            citation_path=citation_path,
            start_line=start_line,
            end_line=end_line,
            text="",
        )

    # Determine clamped_end from how many lines we actually read.  If we
    # received fewer than ``capped`` it means we hit EOF before end_line.
    actual_lines = len(selected)
    clamped_end = lo_1based + actual_lines - 1 if actual_lines else lo_1based - 1
    if clamped_end < end_line:
        logger.warning(
            "Citation span clamped to EOF: %s:%d-%d → ends at line %d",
            citation_path,
            start_line,
            end_line,
            clamped_end,
        )

    span_text = "".join(selected)

    return SpanText(
        citation_path=citation_path,
        start_line=start_line,
        end_line=clamped_end,
        text=span_text,
        truncated=truncated,
    )


# ── Claim preparation ─────────────────────────────────────────────────────────


def _prepare_claim(claim: CitedClaim, repo_root: str) -> ClaimWithSpans:
    """Read all cited spans for one claim and bundle them into a ClaimWithSpans."""
    spans: list[SpanText] = []
    for cite in claim.citations:
        span = _read_span(repo_root, cite.path, cite.start_line, cite.end_line)
        spans.append(span)
    return ClaimWithSpans(claim=claim, spans=spans)


# ── LLM batch call ────────────────────────────────────────────────────────────


def _call_llm_batch(
    batch: list[ClaimWithSpans],
    llm: BaseChatModel,
) -> list[VerifyVerdict]:
    """Send one batch to the LLM and parse N verdicts.

    Returns a list of ``VerifyVerdict`` of length ``len(batch)``.  If the LLM
    response is malformed or returns the wrong number of verdicts, the affected
    positions default to ``partial`` with reason ``"verifier-output-malformed"``.
    """
    user_prompt = build_verifier_user_prompt(batch)
    messages = [
        SystemMessage(content=VERIFIER_SYSTEM_PROMPT),
        HumanMessage(content=user_prompt),
    ]
    response = llm.invoke(messages)
    raw_text = response.content if hasattr(response, "content") else str(response)

    parsed = _extract_json_array(raw_text)

    verdicts: list[VerifyVerdict] = []
    for i, item in enumerate(batch):
        claim = item.claim
        citations_checked = len(item.spans)

        # Try to get the parsed verdict for position i
        raw_verdict: dict[str, Any] | None = None
        if parsed is not None and i < len(parsed) and isinstance(parsed[i], dict):
            raw_verdict = parsed[i]

        if raw_verdict is None:
            verdicts.append(
                VerifyVerdict(
                    paragraph_index=claim.paragraph_index,
                    verdict=_FALLBACK_VERDICT,
                    reason=_FALLBACK_REASON,
                    citations_checked=citations_checked,
                )
            )
            continue

        verdict_str = str(raw_verdict.get("verdict", "")).strip().lower()
        reason_str = str(raw_verdict.get("reason", "")).strip()

        if verdict_str not in _VALID_VERDICTS:
            verdict_str = _FALLBACK_VERDICT
            reason_str = _FALLBACK_REASON

        verdicts.append(
            VerifyVerdict(
                paragraph_index=claim.paragraph_index,
                verdict=verdict_str,
                reason=reason_str,
                citations_checked=citations_checked,
            )
        )

    return verdicts


# ── Public API ────────────────────────────────────────────────────────────────


def verify_citations(
    cited_claims: list[CitedClaim],
    *,
    llm: BaseChatModel,
    repo_root: str,
    batch_size: int = 4,
) -> tuple[list[CitedClaim], VerifyReport]:
    """Verify cited claims against their source spans using an LLM.

    Parameters
    ----------
    cited_claims :
        Paragraphs from the writer agent (output of ``extract_citations``
        after optional ``apply_gate`` filtering).
    llm :
        Any LangChain ``BaseChatModel``.  The caller selects the model;
        tests use a ``FakeLLM`` stub.
    repo_root :
        Absolute path to the repository root.  All cited paths are resolved
        relative to this directory via ``safe_join``.
    batch_size :
        Maximum number of paragraphs per LLM call.  Default 4.

    Returns
    -------
    (kept_claims, report) :
        ``kept_claims`` — paragraphs whose verdict was ``supported`` or
        ``partial`` (paragraphs with ``absent`` or ``citation_missing`` are
        excluded).
        ``report`` — all verdicts plus aggregate counts and LLM call count.
    """
    if batch_size <= 0:
        raise ValueError(f"batch_size must be a positive integer, got {batch_size!r}")

    report = VerifyReport(paragraphs_total=len(cited_claims))

    if not cited_claims:
        return [], report

    kept: list[CitedClaim] = []

    # ── Pass 1: handle citation_missing without LLM ────────────────────────
    needs_llm: list[CitedClaim] = []
    for claim in cited_claims:
        if not claim.citations:
            verdict = VerifyVerdict(
                paragraph_index=claim.paragraph_index,
                verdict="citation_missing",
                reason="Paragraph has no citation tokens.",
                citations_checked=0,
            )
            report.verdicts.append(verdict)
            report.paragraphs_stripped += 1
            logger.info(
                "Stripped paragraph %d (citation_missing): no citations; paths=[]",
                claim.paragraph_index,
            )
        else:
            needs_llm.append(claim)

    # ── Pass 2: LLM verification for paragraphs that have citations ────────
    # Prepare all claims (read spans) before batching
    prepared: list[ClaimWithSpans] = []
    for claim in needs_llm:
        cws = _prepare_claim(claim, repo_root)
        # Check: if all spans failed to read (empty text, safe_join returned None)
        all_unreadable = all(not s.text for s in cws.spans) if cws.spans else False
        if all_unreadable:
            # Treat as citation_missing — no LLM call
            verdict = VerifyVerdict(
                paragraph_index=claim.paragraph_index,
                verdict="citation_missing",
                reason=(
                    f"All cited paths could not be read: "
                    f"{[c.path for c in claim.citations]}"
                ),
                citations_checked=len(cws.spans),
            )
            report.verdicts.append(verdict)
            report.paragraphs_stripped += 1
            cited_paths = [c.path for c in claim.citations]
            logger.info(
                "Stripped paragraph %d (citation_missing): unreadable paths; paths=%s",
                claim.paragraph_index,
                cited_paths,
            )
        else:
            prepared.append(cws)

    # Batch the remaining prepared claims
    for batch_start in range(0, len(prepared), batch_size):
        batch = prepared[batch_start : batch_start + batch_size]
        batch_verdicts = _call_llm_batch(batch, llm)
        report.llm_calls += 1

        for item, verdict in zip(batch, batch_verdicts):
            claim = item.claim
            report.verdicts.append(verdict)

            if verdict.verdict in ("absent", "citation_missing"):
                report.paragraphs_stripped += 1
                cited_paths = [c.path for c in claim.citations]
                logger.info(
                    "Stripped paragraph %d (%s): %s; paths=%s",
                    claim.paragraph_index,
                    verdict.verdict,
                    verdict.reason,
                    cited_paths,
                )
            else:
                kept.append(claim)
                report.paragraphs_kept += 1
                if verdict.verdict == "partial":
                    report.paragraphs_partial += 1
                    logger.warning(
                        "Partial grounding for paragraph %d: %s",
                        claim.paragraph_index,
                        verdict.reason,
                    )

    # ── Sort verdicts by paragraph_index for deterministic output ──────────
    report.verdicts.sort(key=lambda v: v.paragraph_index)

    # kept_claims preserves original order from cited_claims
    # Re-order kept to match input order (prepared may have interleaved citation_missing)
    para_idx_to_claim = {c.paragraph_index: c for c in cited_claims}
    kept_indices = {c.paragraph_index for c in kept}
    kept = [para_idx_to_claim[idx] for idx in sorted(kept_indices)]

    report.paragraphs_kept = len(kept)

    return kept, report
