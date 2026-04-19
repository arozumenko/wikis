"""
Page Validator — Phase 5.

Five-check quality pipeline for candidate pages.  Each check produces
a severity (``pass``, ``warn``, ``fail``) and an optional shaping action.

The final ``shape_decision`` is one of:
- ``keep``          — page is valid as-is
- ``merge_with``    — merge into a sibling page (too small / no identity)
- ``split_by``      — split by sub-capability (too broad / incoherent)
- ``promote_docs``  — promote to docs-only page (doc-dominant)
- ``demote``        — demote to context-only (no value as standalone)

Gated behind ``FeatureFlags.capability_validation``.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import List, Optional

from .candidate_builder import (
    CLASS_BRIDGE,
    CLASS_CODE,
    CLASS_DOCS,
    CLASS_MIXED,
    CandidateRecord,
)

logger = logging.getLogger(__name__)

# ── Shaping actions ─────────────────────────────────────────────────
KEEP = "keep"
MERGE_WITH = "merge_with"
SPLIT_BY = "split_by"
PROMOTE_DOCS = "promote_docs"
DEMOTE = "demote"

# ── Thresholds ──────────────────────────────────────────────────────
#: Minimum code-identity score to pass the identity check.
MIN_IDENTITY_SCORE = 0.1

#: Minimum number of nodes for a page to be viable.
MIN_PAGE_SIZE = 2

#: Default maximum number of nodes before considering a split.
#: Dynamically scaled in ``validate_all`` based on average candidate size.
MAX_PAGE_SIZE = 60

#: Maximum utility contamination before flagging.
MAX_UTILITY_CONTAMINATION = 0.5

#: Minimum implementation evidence for code pages.
MIN_IMPL_EVIDENCE = 0.2

#: Docs-dominance threshold for promote_docs action.
DOCS_PROMOTE_THRESHOLD = 0.7


@dataclass
class CheckResult:
    """Result of a single quality check."""
    name: str
    severity: str  # "pass", "warn", "fail"
    message: str = ""
    suggested_action: str = ""


@dataclass
class ValidationResult:
    """Complete validation result for a candidate."""
    candidate: CandidateRecord
    checks: List[CheckResult] = field(default_factory=list)
    shape_decision: str = KEEP
    merge_target: Optional[int] = None  # micro_id to merge with


def validate_candidate(
    candidate: CandidateRecord,
    sibling_candidates: Optional[List[CandidateRecord]] = None,
    max_page_size: int = MAX_PAGE_SIZE,
) -> ValidationResult:
    """Run the 5-check pipeline on a candidate page.

    Parameters
    ----------
    candidate : CandidateRecord
        The candidate to validate.
    sibling_candidates : list[CandidateRecord], optional
        Other candidates in the same macro-cluster (for merge decisions).
    max_page_size : int, optional
        Override for the maximum page size threshold used in the
        coherence check.  Defaults to the module-level ``MAX_PAGE_SIZE``.

    Returns
    -------
    ValidationResult
        Complete validation result with shape decision.
    """
    result = ValidationResult(candidate=candidate)

    # Run all checks
    result.checks.append(_check_identity(candidate))
    result.checks.append(_check_coherence(candidate, max_page_size=max_page_size))
    result.checks.append(_check_grounding(candidate))
    result.checks.append(_check_coverage(candidate))
    result.checks.append(_check_shape(candidate))

    # Determine final shape decision from check results
    result.shape_decision = _decide_shape(
        candidate, result.checks, sibling_candidates,
    )

    return result


def validate_all(
    candidates: List[CandidateRecord],
) -> List[ValidationResult]:
    """Validate all candidates, providing sibling context within each section.

    Dynamically scales ``max_page_size`` based on the average candidate
    size so that large repos don't trigger split_by on every page.

    Returns
    -------
    list[ValidationResult]
        One result per candidate, in the same order.
    """
    # Scale max_page_size to avoid mass-splitting in large repos.
    # Threshold = max(default=60, 1.5 × average candidate size).
    if candidates:
        total_nodes = sum(len(c.node_ids) for c in candidates)
        avg_size = total_nodes / len(candidates)
        effective_max = max(MAX_PAGE_SIZE, int(avg_size * 1.5))
    else:
        effective_max = MAX_PAGE_SIZE

    # Group by macro_id for sibling context
    by_macro: dict[int, List[CandidateRecord]] = {}
    for c in candidates:
        by_macro.setdefault(c.macro_id, []).append(c)

    results: List[ValidationResult] = []
    for candidate in candidates:
        siblings = [
            s for s in by_macro.get(candidate.macro_id, [])
            if s.micro_id != candidate.micro_id
        ]
        results.append(validate_candidate(
            candidate, siblings, max_page_size=effective_max,
        ))

    # Summary logging
    actions = {}
    for r in results:
        actions[r.shape_decision] = actions.get(r.shape_decision, 0) + 1
    logger.info(
        "[PAGE_VALIDATOR] Validated %d candidates (max_page_size=%d): %s",
        len(results), effective_max, dict(actions),
    )

    return results


# ═════════════════════════════════════════════════════════════════════
# Individual checks
# ═════════════════════════════════════════════════════════════════════

def _check_identity(candidate: CandidateRecord) -> CheckResult:
    """Check 1: Does this page have viable code identity seeds?"""
    if candidate.classification == CLASS_DOCS:
        return CheckResult(
            name="identity", severity="pass",
            message="Docs page — identity check N/A",
        )

    if candidate.code_identity_score >= MIN_IDENTITY_SCORE:
        return CheckResult(
            name="identity", severity="pass",
            message=f"Identity score {candidate.code_identity_score:.2f} OK",
        )

    if candidate.code_identity_score > 0:
        return CheckResult(
            name="identity", severity="warn",
            message=f"Low identity score {candidate.code_identity_score:.2f}",
            suggested_action=MERGE_WITH,
        )

    return CheckResult(
        name="identity", severity="fail",
        message="No code identity symbols",
        suggested_action=DEMOTE,
    )


def _check_coherence(candidate: CandidateRecord, max_page_size: int = MAX_PAGE_SIZE) -> CheckResult:
    """Check 2: Is this page coherent (one capability, not a utility sink)?"""
    if candidate.utility_contamination >= MAX_UTILITY_CONTAMINATION:
        return CheckResult(
            name="coherence", severity="warn",
            message=f"High utility contamination {candidate.utility_contamination:.2f}",
            suggested_action=SPLIT_BY,
        )

    if len(candidate.node_ids) > max_page_size:
        return CheckResult(
            name="coherence", severity="warn",
            message=f"Large page ({len(candidate.node_ids)} nodes > {max_page_size}), may need split",
            suggested_action=SPLIT_BY,
        )

    return CheckResult(
        name="coherence", severity="pass",
        message="Coherence OK",
    )


def _check_grounding(candidate: CandidateRecord) -> CheckResult:
    """Check 3: Does this page have implementation evidence (not declarations-only)?"""
    if candidate.classification == CLASS_DOCS:
        return CheckResult(
            name="grounding", severity="pass",
            message="Docs page — grounding check N/A",
        )

    if candidate.implementation_evidence >= MIN_IMPL_EVIDENCE:
        return CheckResult(
            name="grounding", severity="pass",
            message=f"Implementation evidence {candidate.implementation_evidence:.2f} OK",
        )

    return CheckResult(
        name="grounding", severity="warn",
        message=f"Low implementation evidence {candidate.implementation_evidence:.2f}",
        suggested_action=MERGE_WITH,
    )


def _check_coverage(candidate: CandidateRecord) -> CheckResult:
    """Check 4: Does this page add unique coverage value?"""
    if len(candidate.node_ids) < MIN_PAGE_SIZE:
        return CheckResult(
            name="coverage", severity="fail",
            message=f"Too few nodes ({len(candidate.node_ids)})",
            suggested_action=MERGE_WITH,
        )

    return CheckResult(
        name="coverage", severity="pass",
        message=f"Coverage OK ({len(candidate.node_ids)} nodes)",
    )


def _check_shape(candidate: CandidateRecord) -> CheckResult:
    """Check 5: Shape decision based on classification."""
    if candidate.classification == CLASS_DOCS:
        return CheckResult(
            name="shape", severity="pass",
            message="Docs-dominant cluster",
            suggested_action=PROMOTE_DOCS,
        )

    if candidate.classification == CLASS_BRIDGE:
        return CheckResult(
            name="shape", severity="warn",
            message="Bridge cluster — high file spread",
            suggested_action=SPLIT_BY,
        )

    return CheckResult(
        name="shape", severity="pass",
        message=f"Classification: {candidate.classification}",
    )


# ═════════════════════════════════════════════════════════════════════
# Shape decision
# ═════════════════════════════════════════════════════════════════════

def _decide_shape(
    candidate: CandidateRecord,
    checks: List[CheckResult],
    siblings: Optional[List[CandidateRecord]],
) -> str:
    """Decide the final shape action based on check results."""
    # Collect all suggested actions from failing/warning checks
    actions = [
        c.suggested_action for c in checks
        if c.suggested_action and c.severity in ("fail", "warn")
    ]

    # Fail actions take priority over warnings
    fail_actions = [
        c.suggested_action for c in checks
        if c.suggested_action and c.severity == "fail"
    ]

    if fail_actions:
        # If identity fails → demote; if coverage fails → merge
        if DEMOTE in fail_actions:
            return DEMOTE
        if MERGE_WITH in fail_actions:
            return MERGE_WITH

    # Docs pages → promote
    if candidate.classification == CLASS_DOCS:
        return PROMOTE_DOCS

    # Warn-level: split beats merge
    if SPLIT_BY in actions:
        return SPLIT_BY

    if MERGE_WITH in actions:
        # Only merge if there's a sibling to merge with
        if siblings:
            return MERGE_WITH

    return KEEP
