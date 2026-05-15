"""Surgical LLM page editor for #116 PR 3 incremental wiki regeneration.

When the regime classifier picks ``edit`` for a page, the
:class:`PagePatcher` rewrites the page via a focused LLM call rather
than the full :class:`OptimizedWikiGenerationAgent` flow. The savings
come from:

* Smaller context — only the changed symbols' before/after, not the
  full repository.
* Smaller output — same page length, not a from-scratch redraft.
* Same model — uses the wiki's already-configured LLM.

A **quality gate** guards correctness: after the LLM returns, the
patcher compares the new content to the original via a token-Jaccard
similarity. When the diff is too large (default >60% of tokens
changed), the result is rejected and the caller is signaled to fall
back to the structural regime instead. Worst case is a performance
regression, never a correctness regression.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

from langchain_core.language_models import BaseLanguageModel
from langchain_core.messages import HumanMessage, SystemMessage

from app.core.prompts.surgical_edit_prompts import (
    SURGICAL_EDIT_SYSTEM,
    SURGICAL_EDIT_USER_TEMPLATE,
    format_symbol_diff,
)
from app.core.storage.incremental import compute_content_hash
from app.services.change_detector import ChangeKind, NodeChange

logger = logging.getLogger(__name__)


# Default quality-gate threshold: if more than this fraction of the
# token set changed, reject and fall back to structural. 0.6 is a
# pragmatic ceiling — true surgical edits sit well below it (<0.2);
# anything above usually means the LLM rewrote sections it shouldn't.
DEFAULT_DIFF_THRESHOLD = 0.6


@dataclass(frozen=True)
class PatchResult:
    """Outcome of a surgical patch attempt.

    ``accepted=True`` means the new content cleared the quality gate
    and the caller should write it to storage. ``accepted=False`` means
    the diff was too large; the caller should re-route the page to the
    structural regime.
    """

    page_id: str
    accepted: bool
    new_content: str
    new_content_hash: str
    diff_ratio: float
    reason: str = ""


@dataclass(frozen=True)
class SymbolDiff:
    """Per-symbol context passed to the surgical-edit prompt.

    Kept separate from :class:`NodeChange` because the patcher needs
    *content* (old source, new source) which the change detector's
    dataclass doesn't carry — only hashes.
    """

    symbol_name: str
    node_id: str
    old_source: str
    new_source: str
    signature_change: str | None = None


class PagePatcher:
    """Surgical-edit handler. One instance per wiki refresh; reused for
    every page routed to the ``edit`` regime."""

    def __init__(
        self,
        llm: BaseLanguageModel,
        *,
        diff_threshold: float = DEFAULT_DIFF_THRESHOLD,
    ) -> None:
        self._llm = llm
        self._diff_threshold = diff_threshold

    def patch_page(
        self,
        *,
        page_id: str,
        page_title: str,
        primary_symbol_id: str | None,
        current_content: str,
        symbol_diffs: list[SymbolDiff],
        moved_paths: dict[str, str],
    ) -> PatchResult:
        """Run one surgical edit, with quality gate.

        Args:
            page_id: target wiki page (for telemetry).
            page_title: original page title — locked by the prompt; the
                LLM mustn't change it (would shift the anchor slug).
            primary_symbol_id: optional; included in the prompt for
                LLM context.
            current_content: the page's existing markdown.
            symbol_diffs: per-symbol before/after content. The prompt
                instructs the LLM to revise prose only for these.
            moved_paths: ``{old_path: new_path}`` for ``<code_context>``
                blocks; the LLM rewrites the path attribute only.

        Returns:
            :class:`PatchResult`. When ``accepted=False`` the caller
            should treat the page as structural and discard the new
            content.
        """
        if not symbol_diffs and not moved_paths:
            # Nothing for the LLM to do — return passthrough with hash
            # recomputed (in case the caller's bookkeeping needs it).
            return PatchResult(
                page_id=page_id,
                accepted=True,
                new_content=current_content,
                new_content_hash=compute_content_hash(current_content),
                diff_ratio=0.0,
                reason="no diffs to apply",
            )

        prompt_user = SURGICAL_EDIT_USER_TEMPLATE.format(
            page_title=page_title,
            primary_symbol_id=primary_symbol_id or "(none)",
            symbol_diffs=self._format_symbol_diffs(symbol_diffs),
            moved_paths=self._format_moved_paths(moved_paths),
            current_content=current_content,
        )

        try:
            response = self._llm.invoke(
                [
                    SystemMessage(content=SURGICAL_EDIT_SYSTEM),
                    HumanMessage(content=prompt_user),
                ]
            )
            new_content = self._extract_text(response).strip()
        except Exception as exc:  # noqa: BLE001 — patcher is fail-soft
            logger.warning(
                "[page_patcher] page=%s LLM call failed; falling back to "
                "structural: %s",
                page_id,
                exc,
            )
            return PatchResult(
                page_id=page_id,
                accepted=False,
                new_content=current_content,
                new_content_hash=compute_content_hash(current_content),
                diff_ratio=1.0,
                reason=f"LLM error: {exc}",
            )

        if not new_content:
            return PatchResult(
                page_id=page_id,
                accepted=False,
                new_content=current_content,
                new_content_hash=compute_content_hash(current_content),
                diff_ratio=1.0,
                reason="empty LLM output",
            )

        diff_ratio = compute_diff_ratio(current_content, new_content)
        accepted = diff_ratio <= self._diff_threshold
        reason = (
            f"diff_ratio={diff_ratio:.2f} within threshold"
            if accepted
            else f"diff_ratio={diff_ratio:.2f} exceeds {self._diff_threshold:.2f}"
        )
        logger.info(
            "[page_patcher] page=%s accepted=%s %s",
            page_id, accepted, reason,
        )

        return PatchResult(
            page_id=page_id,
            accepted=accepted,
            new_content=new_content if accepted else current_content,
            new_content_hash=compute_content_hash(
                new_content if accepted else current_content
            ),
            diff_ratio=diff_ratio,
            reason=reason,
        )

    # ------------------------------------------------------------------
    # internals
    # ------------------------------------------------------------------

    @staticmethod
    def _format_symbol_diffs(diffs: list[SymbolDiff]) -> str:
        if not diffs:
            return "(none — only file paths moved)"
        return "\n\n".join(
            format_symbol_diff(
                symbol_name=d.symbol_name,
                old_source=d.old_source,
                new_source=d.new_source,
                signature_change=d.signature_change,
            )
            for d in diffs
        )

    @staticmethod
    def _format_moved_paths(paths: dict[str, str]) -> str:
        if not paths:
            return "(none)"
        return "\n".join(f"- `{old}` → `{new}`" for old, new in sorted(paths.items()))

    @staticmethod
    def _extract_text(response: Any) -> str:
        """LangChain message types vary — pull the string content out."""
        if hasattr(response, "content"):
            content = response.content
        else:
            content = response
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            # Some providers return [{type: text, text: "..."}, ...]
            return "".join(
                part.get("text", "") if isinstance(part, dict) else str(part)
                for part in content
            )
        return str(content)


# ---------------------------------------------------------------------------
# Quality gate
# ---------------------------------------------------------------------------


def compute_diff_ratio(old: str, new: str) -> float:
    """Estimate how much of ``old`` was rewritten in ``new``.

    Token-set Jaccard distance: tokenize both on whitespace,
    case-normalize, treat as multisets, compute
    ``1 - |intersection| / |union|``. Returns a value in [0, 1] where
    0 means identical and 1 means disjoint.

    Cheap and language-agnostic. Doesn't catch reorderings (a paragraph
    swap looks like 0 diff) but those aren't the failure mode we worry
    about — the failure mode is "LLM rewrote stuff it shouldn't have",
    which always introduces new tokens.

    #135: tokens are lower-cased before comparison so casing tweaks
    in body prose ("API" → "api") don't inflate the diff ratio. We
    explicitly DO NOT want this for code identifiers (where case is
    significant), but body-prose comparison is the only place this
    metric is used.
    """
    old_tokens = [t.lower() for t in old.split()]
    new_tokens = [t.lower() for t in new.split()]
    if not old_tokens and not new_tokens:
        return 0.0
    old_set = set(old_tokens)
    new_set = set(new_tokens)
    union = old_set | new_set
    if not union:
        return 0.0
    intersection = old_set & new_set
    return 1.0 - (len(intersection) / len(union))
