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

    # ── Phase 1 — Phase-2 observability + safe quick wins ─────────────

    #: Capture and persist :class:`Phase2Stats` to ``repo_meta`` under
    #: ``phase2_stats_v2`` (cheap; default on).
    phase2_observability: bool = True

    #: Skip duplicate ``(source, target, rel_type)`` edges when injecting
    #: synthetic edges in Phase 2 helpers.
    edge_dedup: bool = True

    #: Drop FTS hits whose ``score_norm`` is below this threshold during
    #: Phase 2 lexical lookups.
    fts_min_score_norm: float = 0.15

    #: Refuse to issue FTS lookups for stop-word-only queries below
    #: ``min_fts_query_len`` characters.
    fts_stopword_gate: bool = True

    # ── Phase 2 — IDF gating + tiered lexical + REST disambig ─────────

    #: Replace flat Pass-1 lexical lookup with the T1–T4 tiered cascade
    #: (same-dir → parent-dir → local-FTS → global). Default off until
    #: Phase 3 lands the cascade reorder; flip via env.
    orphan_lexical_tiered: bool = False

    #: Apply the IDF gate to lexical Pass 1 (drops common names like
    #: ``User`` / ``API`` from ever issuing T3/T4 queries).
    orphan_lexical_idf_gate: bool = True

    #: Detect REST endpoint classes (generic name + REST decorator/base)
    #: and rewrite their FTS query to ``"<file_stem> <symbol_name>"``.
    orphan_rest_disambig: bool = True

    # ── Phase 3 — cascade reorder + explicit-ref Pass 1 + embedding reuse

    #: Enable the 4-pass v2 orphan cascade (explicit-refs → hybrid →
    #: tiered lexical → directory). Default off until Phase 4 hybrid
    #: lands and the integration tests stabilise.
    orphan_cascade_v2: bool = False

    #: Pre-fetch persisted embeddings for orphans before Pass 2 (vec)
    #: so we never re-embed already-stored vectors. Cheap + safe.
    orphan_reuse_embeddings: bool = True

    # ── Phase 4 — hybrid RRF Pass 2 ───────────────────────────────────

    #: Replace pure-vector Pass 2 with the FTS+Vec hybrid using RRF
    #: fusion. Default off until cascade-v2 is ready to flip.
    orphan_hybrid_search: bool = False

    #: RRF constant ``k`` — larger flattens the contribution of
    #: top-ranked items. TREC standard = 60.
    orphan_rrf_k: int = 60

    #: Minimum fused ``rrf_score`` required for a hit to become an edge.
    orphan_rrf_threshold: float = 0.02

    #: Per-orphan top-N cap on hybrid hits (caps both per-list FTS/vec
    #: candidates and the post-fusion result).
    orphan_hybrid_top_n: int = 20

    #: Reserved for the linear-fusion path (``alpha * vec + (1-alpha)
    #: * fts``). Only used when ``orphan_hybrid_strategy == "linear"``.
    orphan_hybrid_alpha: float = 0.5

    #: Fusion strategy: ``"rrf"`` (default) or ``"linear"`` (reserved
    #: for follow-up tuning work; falls back to RRF until implemented).
    orphan_hybrid_strategy: str = "rrf"

    # ── Phase 5 — Node disambiguation (3A / 3B / 3C) ──────────────────

    #: Node ID generation strategy. ``"stem"`` (legacy) keys nodes by
    #: ``language::file_stem::local_qualified_name`` and silently
    #: collides between same-stem files in different directories,
    #: falling back to a 16-bit file-hash suffix. ``"rel_path"``
    #: replaces the stem with a path-safe ``rel_path`` slug, which
    #: removes the collision class entirely (no hash suffix). Default
    #: ``"stem"`` for one release; flip to ``"rel_path"`` next.
    #: NOT retroactive — existing wikis stay on the build-time scheme
    #: until a full rebuild.
    node_id_style: str = "stem"

    #: Build the qualified-name (``Parent.symbol``) and FQN
    #: (``rel_path::Parent.symbol``) lookup indexes on top of the
    #: existing simple-name index. Cheap; default on. Consumed by
    #: :class:`GraphQueryService.resolve_symbol` and the explicit-ref
    #: Pass 1 resolver.
    qualified_name_index: bool = True

    # ── Phase 6 — Cross-language linker + new edge classes ───────────

    #: Run the 4-level cross-language linker (L0 exact → L1 API surface
    #: → L2 hybrid RRF → L3 containment) and inject ``cross_language``
    #: edges into the graph during Phase 1c. Default off until the
    #: per-language matchers stabilise.
    cross_language_linking: bool = False

    #: Run the test↔code linker (same-stem heuristic, shared decorator,
    #: test-framework imports) and inject ``test_link`` edges. Default
    #: off; cheap, but its weight floor affects clustering.
    test_linker: bool = False

    #: Extract API surfaces (REST / gRPC / GraphQL / FFI / BDD / CLI)
    #: from parser output during Phase 1c and persist them on the
    #: ``repo_nodes.api_surface`` column when present. Cheap; default
    #: on so the column is populated even when ``cross_language_linking``
    #: is gated off.
    api_surface_extraction: bool = True


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
        phase2_observability=_env_bool("WIKI_PHASE2_OBSERVABILITY", True),
        edge_dedup=_env_bool("WIKI_EDGE_DEDUP", True),
        fts_min_score_norm=_env_float("WIKI_FTS_MIN_SCORE_NORM", 0.15),
        fts_stopword_gate=_env_bool("WIKI_FTS_STOPWORD_GATE", True),
        orphan_lexical_tiered=_env_bool("WIKI_ORPHAN_LEXICAL_TIERED", False),
        orphan_lexical_idf_gate=_env_bool("WIKI_ORPHAN_LEXICAL_IDF_GATE", True),
        orphan_rest_disambig=_env_bool("WIKI_ORPHAN_REST_DISAMBIG", True),
        orphan_cascade_v2=_env_bool("WIKI_ORPHAN_CASCADE_V2", False),
        orphan_reuse_embeddings=_env_bool("WIKI_ORPHAN_REUSE_EMBEDDINGS", True),
        orphan_hybrid_search=_env_bool("WIKI_ORPHAN_HYBRID_SEARCH", False),
        orphan_rrf_k=_env_int("WIKI_ORPHAN_RRF_K", 60),
        orphan_rrf_threshold=_env_float("WIKI_ORPHAN_RRF_THRESHOLD", 0.02),
        orphan_hybrid_top_n=_env_int("WIKI_ORPHAN_HYBRID_TOP_N", 20),
        orphan_hybrid_alpha=_env_float("WIKI_ORPHAN_HYBRID_ALPHA", 0.5),
        orphan_hybrid_strategy=os.environ.get("WIKI_ORPHAN_HYBRID_STRATEGY", "rrf").strip() or "rrf",
        node_id_style=(os.environ.get("WIKI_NODE_ID_STYLE", "stem").strip().lower() or "stem"),
        qualified_name_index=_env_bool("WIKI_QUALIFIED_NAME_INDEX", True),
        cross_language_linking=_env_bool("WIKI_CROSS_LANGUAGE_LINKING", False),
        test_linker=_env_bool("WIKI_TEST_LINKER", False),
        api_surface_extraction=_env_bool("WIKI_API_SURFACE_EXTRACTION", True),
    )
