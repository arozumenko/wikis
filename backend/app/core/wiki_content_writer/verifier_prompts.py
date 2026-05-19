"""Prompts for the citation verifier (#239).

The verifier compares each paragraph's claim against the cited file spans
and returns one of four verdicts:

  supported          — the cited span faithfully grounds the paragraph
  partial            — some overlap exists but the paragraph goes beyond the span
  absent             — no cited span supports the paragraph
  citation_missing   — paragraph has no citations at all (handled before LLM call)

Public API
----------
VERIFIER_SYSTEM_PROMPT     — system message for the verifier call.
ClaimWithSpans             — container bundling a CitedClaim + read span texts.
build_verifier_user_prompt — builds the user message for one batch of claims.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from app.core.wiki_content_writer.citation_extractor import CitedClaim

# ── Data containers ───────────────────────────────────────────────────────────


@dataclass
class SpanText:
    """One cited file span that has been read from disk.

    Attributes
    ----------
    citation_path : str
        Original repo-relative path from the Citation token.
    start_line : int
        First line (1-indexed, inclusive) of the span.
    end_line : int
        Last line (1-indexed, inclusive) of the span.
    text : str
        Content of the span.  May be empty if the file could not be read or
        the lines were out of range (clamped to EOF).
    truncated : bool
        True when the span was hard-capped at 200 lines and text is incomplete.
    """

    citation_path: str
    start_line: int
    end_line: int
    text: str
    truncated: bool = False


@dataclass
class ClaimWithSpans:
    """A paragraph bundled with all its cited file spans.

    Attributes
    ----------
    claim : CitedClaim
        The paragraph and its citation tokens.
    spans : list[SpanText]
        Span texts corresponding to each valid citation.  Spans for
        unreadable paths have ``text=""`` and the caller stores a reason
        on the verdict directly (the LLM is not called for those).
    """

    claim: CitedClaim
    spans: list[SpanText] = field(default_factory=list)


# ── System prompt ─────────────────────────────────────────────────────────────

VERIFIER_SYSTEM_PROMPT = """\
You are a citation verifier for a technical documentation system.

Your only task: for each paragraph in the batch, decide whether the cited \
source spans actually support what the paragraph claims.

Return ONLY a JSON array — one element per paragraph, in the same order as \
given. Each element must have exactly two keys:
  "verdict" : one of "supported", "partial", "absent"
  "reason"  : a single sentence explaining why

Verdict definitions
-------------------
supported  — every non-trivial claim in the paragraph is directly grounded by \
at least one of the provided code spans. Minor phrasing differences are fine \
as long as the meaning is accurate.
partial    — the code spans overlap with some but not all claims in the \
paragraph. The parts that are present are accurate, but the paragraph also \
makes claims not found in the spans.
absent     — the code spans do not support the paragraph's claims. Either the \
spans contradict the paragraph, the spans are about something else entirely, \
or no relevant information appears in the spans.

Rules
-----
- If spans are empty or unreadable, treat that citation as absent for that item.
- Do NOT hallucinate additional knowledge. Base your verdict solely on what \
the paragraph says and what the code spans contain.
- Do NOT invent "citation_missing" — that is handled outside your call.
- Return ONLY valid JSON. No markdown fences, no commentary, no extra keys.
- The array MUST contain exactly N elements for N paragraphs.
"""

# ── User prompt builder ───────────────────────────────────────────────────────

_ITEM_TEMPLATE = """\
--- PARAGRAPH {idx} ---
{paragraph}

--- CITED SPANS FOR PARAGRAPH {idx} ---
{spans}
"""

_SPAN_TEMPLATE = """\
[{path}:{start}-{end}]{truncation_note}
```
{text}
```
"""


def build_verifier_user_prompt(batch: list[ClaimWithSpans]) -> str:
    """Build the user message for one batch of claims.

    Parameters
    ----------
    batch : list[ClaimWithSpans]
        A list of claim+span bundles to verify in one LLM call.
        The order in the returned prompt matches the order of `batch`,
        so verdict[i] corresponds to batch[i].

    Returns
    -------
    str
        The user message to send to the LLM.
    """
    parts: list[str] = [
        f"Verify the following {len(batch)} paragraph(s). "
        f"Return a JSON array with exactly {len(batch)} element(s).\n"
    ]

    for idx, item in enumerate(batch):
        span_blocks: list[str] = []
        for span in item.spans:
            trunc_note = " [TRUNCATED to 200 lines]" if span.truncated else ""
            span_text = span.text if span.text else "(file could not be read)"
            span_blocks.append(
                _SPAN_TEMPLATE.format(
                    path=span.citation_path,
                    start=span.start_line,
                    end=span.end_line,
                    truncation_note=trunc_note,
                    text=span_text,
                )
            )

        spans_section = "\n".join(span_blocks) if span_blocks else "(no spans available)"
        parts.append(
            _ITEM_TEMPLATE.format(
                idx=idx,
                paragraph=item.claim.paragraph_text,
                spans=spans_section,
            )
        )

    parts.append(
        "Respond with a JSON array of exactly "
        f"{len(batch)} verdict object(s), in order."
    )
    return "\n".join(parts)
