"""Three-regime classifier for #116 PR 3 incremental wiki regeneration.

PR 2's :class:`ChangeDetector` produces a :class:`ChangeSet` (added /
modified / moved / deleted nodes) and an :class:`AffectedPage` list
(pages whose ``page_symbols`` reverse index intersects the changes).
This module routes each affected page into one of four buckets:

* **unchanged** — page has no overlapping change after filtering; skip
  entirely. (Affected pages with only false-positive entries can land
  here too if the override logic demotes them.)

* **trivial** — only ``MOVED`` changes touch this page. The node IDs and
  semantic content are unchanged; only ``rel_path`` shifted. PR 3's
  trivial handler rewrites ``<code_context path="...">`` blocks in the
  page body and re-upserts. No LLM.

* **edit** — only ``MODIFIED`` changes touch this page, AND the page's
  primary symbol is still present. PR 3's surgical LLM edit rewrites
  the affected sections only, with a quality gate that falls back to
  structural when the diff is too large.

* **structural** — ``ADDED`` or ``DELETED`` changes touch this page, OR
  the page's primary symbol was deleted, OR signature-shape changes
  imply a re-cluster. PR 3 falls back to full page regen via the
  existing :class:`OptimizedWikiGenerationAgent` path.

Routing happens **before** any LLM call so the dispatcher can report
how many pages went each way (PR 5 telemetry).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from enum import Enum

from app.core.storage.protocol import WikiStorageProtocol
from app.services.change_detector import (
    AffectedPage,
    ChangeKind,
    ChangeSet,
    NodeChange,
)

logger = logging.getLogger(__name__)


class Regime(str, Enum):
    """Which incremental path the dispatcher will take for a page."""

    UNCHANGED = "unchanged"
    TRIVIAL = "trivial"
    EDIT = "edit"
    STRUCTURAL = "structural"


@dataclass(frozen=True)
class PageRegime:
    """One page's regime assignment.

    Carries enough state for the per-regime handler to act without
    re-querying the change set: the affected page itself (with its
    ``changes`` list), plus the optional reason that locked in the
    regime (used for telemetry + debugging).
    """

    page_id: str
    regime: Regime
    changes: tuple[NodeChange, ...]
    reason: str = ""


@dataclass
class RegimePlan:
    """All pages, grouped by regime. Output of :func:`classify_pages`.

    The four lists are disjoint — every affected page appears in exactly
    one bucket. ``unchanged`` is for pages that the override logic
    demoted (rare but possible — e.g. when the only NodeChange touching
    a page turned out to be filtered out by the primary-symbol check).
    """

    unchanged: list[PageRegime] = field(default_factory=list)
    trivial: list[PageRegime] = field(default_factory=list)
    edit: list[PageRegime] = field(default_factory=list)
    structural: list[PageRegime] = field(default_factory=list)

    @property
    def total(self) -> int:
        return (
            len(self.unchanged)
            + len(self.trivial)
            + len(self.edit)
            + len(self.structural)
        )

    def counts(self) -> dict[str, int]:
        """Telemetry-friendly breakdown for log lines and SSE events."""
        return {
            "unchanged": len(self.unchanged),
            "trivial": len(self.trivial),
            "edit": len(self.edit),
            "structural": len(self.structural),
            "total": self.total,
        }


# ---------------------------------------------------------------------------
# Classifier
# ---------------------------------------------------------------------------


def classify_pages(
    change_set: ChangeSet,
    affected_pages: list[AffectedPage],
    storage: WikiStorageProtocol,
) -> RegimePlan:
    """Route every affected page into a regime bucket.

    Args:
        change_set: the full :class:`ChangeSet` from
            :meth:`ChangeDetector.detect_changes`. Used to look up the
            ``deleted`` node set for the primary-symbol-deleted override.
        affected_pages: result of :meth:`ChangeDetector.affected_pages`
            — pages already filtered to those that cite at least one
            changed node.
        storage: read-only handle to ``wiki_pages`` so the classifier
            can look up each page's ``primary_symbol_id``. The override
            "primary symbol deleted → structural" depends on this.

    Returns:
        A :class:`RegimePlan` with disjoint regime buckets.

    Routing rules (in priority order):

    1. **Primary symbol deleted** → ``structural``. The page is about
       a symbol that no longer exists; surgical editing makes no sense.

    2. **Any DELETED or ADDED change** → ``structural``. Page topology
       changed (a cited symbol is gone, or a new sibling appeared);
       cluster membership for this page might have shifted.

    3. **Only MODIFIED changes** → ``edit``. Signatures may or may not
       have changed; the surgical patcher reads the diff and decides
       per-section. Falls back to structural at the patcher's quality
       gate (out of scope for the classifier).

    4. **Only MOVED changes** → ``trivial``. No semantic change; just
       update ``<code_context path="...">`` blocks in the body.

    5. **Empty change list** → ``unchanged`` (defensive — shouldn't
       happen for items in ``affected_pages`` but guard anyway).
    """
    plan = RegimePlan()
    deleted_node_ids = {c.node_id for c in change_set.deleted}

    for affected in affected_pages:
        regime, reason = _classify_one(affected, deleted_node_ids, storage)
        page_regime = PageRegime(
            page_id=affected.page_id,
            regime=regime,
            changes=tuple(affected.changes),
            reason=reason,
        )
        getattr(plan, regime.value).append(page_regime)

    logger.info(
        "[incremental_regen] classified %d affected pages: %s",
        plan.total,
        plan.counts(),
    )
    return plan


# ---------------------------------------------------------------------------
# internals
# ---------------------------------------------------------------------------


def _classify_one(
    affected: AffectedPage,
    deleted_node_ids: set[str],
    storage: WikiStorageProtocol,
) -> tuple[Regime, str]:
    """Apply the four-rule routing table to a single page."""
    if not affected.changes:
        return Regime.UNCHANGED, "no changes after filtering"

    # Rule 1: primary symbol deleted → structural. The page is "about"
    # a symbol that no longer exists; surgical editing can't recover.
    page_row = storage.get_wiki_page(affected.page_id)
    primary_symbol_id = (page_row or {}).get("primary_symbol_id")
    if primary_symbol_id and primary_symbol_id in deleted_node_ids:
        return Regime.STRUCTURAL, f"primary symbol {primary_symbol_id!r} deleted"

    kinds = {change.kind for change in affected.changes}

    # Rule 2: ADDED or DELETED → structural.
    if ChangeKind.ADDED in kinds or ChangeKind.DELETED in kinds:
        present = sorted(k.value for k in kinds if k in {ChangeKind.ADDED, ChangeKind.DELETED})
        return Regime.STRUCTURAL, f"topology change ({', '.join(present)})"

    # Rule 3: only MODIFIED → edit.
    if kinds == {ChangeKind.MODIFIED}:
        return Regime.EDIT, "modified only"

    # Rule 4: only MOVED → trivial.
    if kinds == {ChangeKind.MOVED}:
        return Regime.TRIVIAL, "moved only"

    # Mixed MODIFIED + MOVED → edit (surgical handler can patch citations
    # for the moved nodes and edit prose for the modified ones).
    if kinds == {ChangeKind.MODIFIED, ChangeKind.MOVED}:
        return Regime.EDIT, "modified + moved (mixed)"

    # Defensive fall-through: any combination we didn't anticipate goes
    # to structural since that's always safe.
    return Regime.STRUCTURAL, f"unexpected kind set {sorted(k.value for k in kinds)}"
