"""
Feature flags for clustering improvements.

All flags default to **enabled** — the improved Leiden-based pipeline
runs by default.  Set the corresponding ``WIKIS_CLUSTER_*`` environment
variable to ``0`` / ``false`` / ``no`` to disable individual flags.

When the UI selects the "cluster" planner type, all Leiden-related
flags are implicitly ON.  The ``exclude_tests`` flag is derived from
the UI toggle.

Usage::

    from app.core.feature_flags import get_feature_flags

    flags = get_feature_flags()
    if flags.hierarchical_leiden:
        ...  # new path
    else:
        ...  # legacy path
"""

import os
from dataclasses import dataclass

__all__ = ["FeatureFlags", "get_feature_flags"]


def _env_bool(name: str, default: bool = True) -> bool:
    """Read a boolean from an environment variable (``1`` / ``true`` / ``yes``)."""
    val = os.environ.get(name, "").strip().lower()
    if not val:
        return default
    return val in ("1", "true", "yes")


def _env_float(name: str, default: float) -> float:
    """Read a float from an environment variable; fall back to *default* on parse error."""
    val = os.environ.get(name, "").strip()
    if not val:
        return default
    try:
        return float(val)
    except ValueError:
        return default


def _env_int(name: str, default: int) -> int:
    """Read an int from an environment variable; fall back to *default* on parse error."""
    val = os.environ.get(name, "").strip()
    if not val:
        return default
    try:
        return int(val)
    except ValueError:
        return default


@dataclass(frozen=True)
class FeatureFlags:
    """Immutable snapshot of clustering feature flags.

    Every flag is ``True`` by default — the improved Leiden pipeline
    runs unless explicitly disabled via environment variable.
    """

    #: Replace Louvain two-pass pipeline with hierarchical Leiden.
    hierarchical_leiden: bool = True

    #: Enable deterministic candidate builder + page-quality validator.
    capability_validation: bool = True

    #: Enable shared smart-expansion layer in cluster expansion.
    smart_expansion: bool = True

    #: Enable explicit coverage tracking via the coverage ledger.
    coverage_ledger: bool = True

    #: Enable language-specific heuristics for page shaping.
    language_hints: bool = True

    #: Exclude test code from wiki structure (clustering / page formation).
    #: Test nodes are still indexed and available for vector retrieval,
    #: but they do not participate in clustering or form wiki pages.
    exclude_tests: bool = True

    # ── Phase 0 — storage parity (graph quality roadmap) ──────────────

    #: Cap used to normalize PostgreSQL ``ts_rank_cd`` scores into ``[0, 1]``.
    #: Score normalization formula: ``score_norm = min(fts_rank / cap, 1.0)``.
    pg_ts_rank_cap: float = 1.0

    #: When True, PostgreSQL FTS uses ``ts_rank_cd(..., 32)`` (length-norm bit
    #: flag). Disable to reproduce pre-Phase-0 ranking behaviour for rollback.
    pg_ts_rank_normalize: bool = True

    #: Concurrency for SQLite ``batch_similarity_search`` (sqlite-vec has no
    #: native batch KNN; the implementation loops per embedding).
    vec_batch_concurrency: int = 4


def get_feature_flags() -> FeatureFlags:
    """Build a ``FeatureFlags`` instance from the current environment."""
    return FeatureFlags(
        hierarchical_leiden=_env_bool("WIKIS_CLUSTER_HIERARCHICAL_LEIDEN"),
        capability_validation=_env_bool("WIKIS_CLUSTER_CAPABILITY_VALIDATION"),
        smart_expansion=_env_bool("WIKIS_CLUSTER_SMART_EXPANSION"),
        coverage_ledger=_env_bool("WIKIS_CLUSTER_COVERAGE_LEDGER"),
        language_hints=_env_bool("WIKIS_CLUSTER_LANGUAGE_HINTS"),
        exclude_tests=_env_bool("WIKIS_EXCLUDE_TESTS"),
        pg_ts_rank_cap=_env_float("WIKI_TS_RANK_CAP", 1.0),
        pg_ts_rank_normalize=_env_bool("WIKI_PG_TS_RANK_NORM", True),
        vec_batch_concurrency=_env_int("WIKI_VEC_BATCH_CONCURRENCY", 4),
    )
