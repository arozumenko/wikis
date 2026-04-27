"""Phase 8 — Cross-repo relatedness scorer.

Pure functions that score (wiki_a, wiki_b) similarity from cheap,
language-agnostic signals:

* ``lang``   — Jaccard over languages present.
* ``dep``    — Jaccard over dependency identifiers.
* ``api``    — Overlap of API surface keys (Phase 6 output).
* ``naming`` — Jaccard over the top-K most common symbol names.

The composite score is

    score = 0.3*lang + 0.2*dep + 0.3*api + 0.2*naming

with a hard floor of ``0.0`` and ceiling of ``1.0``. Pre-filter:
returns ``(0.0, breakdown)`` with ``"skipped": "no_overlap"`` when
``len(languages_a ∩ languages_b) == 0`` AND ``api_overlap == 0`` so
the caller can drop the pair without scoring naming/deps.

Side-effect free; the caller wires it into the pipeline behind
``flags.cross_repo_linking`` and persists results via
:class:`ProjectStorageProtocol.upsert_repo_relatedness`.
"""

from __future__ import annotations

from collections import Counter
from typing import Iterable

__all__ = [
    "jaccard",
    "score_repo_pair",
]


# ----------------------------------------------------------------------
# Primitives
# ----------------------------------------------------------------------


def jaccard(a: Iterable[str], b: Iterable[str]) -> float:
    """Return the Jaccard index of two iterables of strings."""
    set_a = {x for x in a if x}
    set_b = {x for x in b if x}
    if not set_a and not set_b:
        return 0.0
    inter = len(set_a & set_b)
    union = len(set_a | set_b)
    return inter / union if union else 0.0


def _api_overlap(api_a: Iterable[str], api_b: Iterable[str]) -> float:
    """API surface overlap ratio over the smaller side.

    Using ``min`` rather than ``union`` so that a small wiki sharing
    most of its surface with a much larger one still scores high.
    """
    set_a = {x for x in api_a if x}
    set_b = {x for x in api_b if x}
    if not set_a or not set_b:
        return 0.0
    inter = len(set_a & set_b)
    denom = min(len(set_a), len(set_b))
    return inter / denom if denom else 0.0


def _top_names(symbol_names: Iterable[str], top_k: int) -> set[str]:
    """Return the ``top_k`` most-frequent symbol names."""
    if top_k <= 0:
        return set()
    counter = Counter(s for s in symbol_names if s)
    return {name for name, _count in counter.most_common(top_k)}


# ----------------------------------------------------------------------
# Public — score_repo_pair
# ----------------------------------------------------------------------


def score_repo_pair(
    *,
    languages_a: Iterable[str],
    languages_b: Iterable[str],
    deps_a: Iterable[str] | None = None,
    deps_b: Iterable[str] | None = None,
    api_a: Iterable[str] | None = None,
    api_b: Iterable[str] | None = None,
    symbol_names_a: Iterable[str] | None = None,
    symbol_names_b: Iterable[str] | None = None,
    top_k: int = 200,
) -> tuple[float, dict]:
    """Score a wiki pair and return ``(score, breakdown)``.

    Args:
        languages_a/b: Set of programming languages in each wiki.
        deps_a/b: Optional iterables of dependency identifiers
            (package names, module imports, etc.).
        api_a/b: Optional iterables of API surface keys produced by
            :mod:`api_surface_extractor`.
        symbol_names_a/b: Optional iterables of *all* symbol names
            (the function tops them off internally).
        top_k: How many top symbol names to compare for naming jaccard.

    Returns:
        ``(score, breakdown)`` where ``breakdown`` carries each
        component plus a ``"skipped"`` reason when the pre-filter fires.
    """
    langs_a = list(languages_a or ())
    langs_b = list(languages_b or ())
    api_list_a = list(api_a or ())
    api_list_b = list(api_b or ())

    # Pre-filter: no language overlap AND no API surface overlap → skip.
    lang_inter = {x for x in langs_a if x} & {x for x in langs_b if x}
    api_inter = {x for x in api_list_a if x} & {x for x in api_list_b if x}
    if not lang_inter and not api_inter:
        return 0.0, {
            "lang": 0.0,
            "dep": 0.0,
            "api": 0.0,
            "naming": 0.0,
            "skipped": "no_overlap",
        }

    lang_score = jaccard(langs_a, langs_b)
    dep_score = jaccard(deps_a or (), deps_b or ())
    api_score = _api_overlap(api_list_a, api_list_b)

    top_a = _top_names(symbol_names_a or (), top_k)
    top_b = _top_names(symbol_names_b or (), top_k)
    naming_score = jaccard(top_a, top_b)

    score = (
        0.30 * lang_score
        + 0.20 * dep_score
        + 0.30 * api_score
        + 0.20 * naming_score
    )
    score = max(0.0, min(1.0, score))
    breakdown = {
        "lang": round(lang_score, 6),
        "dep": round(dep_score, 6),
        "api": round(api_score, 6),
        "naming": round(naming_score, 6),
    }
    return score, breakdown
