"""Phase 2 (graph-quality roadmap) — IDF gating + tiered lexical T1–T4."""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional

import networkx as nx
import pytest

from app.core import graph_lexical_v2 as lex
from app.core.feature_flags import FeatureFlags


# ──────────────────────────────────────────────────────────────────────────
# Fakes
# ──────────────────────────────────────────────────────────────────────────


class _FakeDB:
    """Minimal DB stand-in implementing the surface used by the cascade."""

    def __init__(
        self,
        nodes: Dict[str, Dict[str, Any]],
        fts_results: Dict[str, List[Dict[str, Any]]],
        total_nodes: int,
        match_counts: Dict[str, int],
    ) -> None:
        self._nodes = nodes
        self._fts = fts_results
        self._total = total_nodes
        self._matches = match_counts

    def get_node(self, node_id: str) -> Optional[Dict[str, Any]]:
        return self._nodes.get(node_id)

    def search_fts5(self, *, query: str, limit: int = 8) -> List[Dict[str, Any]]:
        return list(self._fts.get(query, []))[:limit]

    def node_count(self) -> int:
        return self._total

    def count_fts_matches(self, query: str, *, exact_match: bool = False) -> int:  # noqa: ARG002
        return self._matches.get(query, 0)


def _flags(**overrides: Any) -> FeatureFlags:
    """Build a FeatureFlags instance with Phase 2 defaults explicit."""
    base = dict(
        orphan_lexical_tiered=True,
        orphan_lexical_idf_gate=True,
        orphan_rest_disambig=True,
        fts_min_score_norm=0.0,
        fts_stopword_gate=False,
    )
    base.update(overrides)
    return FeatureFlags(**base)


# ──────────────────────────────────────────────────────────────────────────
# IDF
# ──────────────────────────────────────────────────────────────────────────


class TestIDF:
    def test_compute_symbol_idf_specific_name(self):
        db = _FakeDB(nodes={}, fts_results={}, total_nodes=1000, match_counts={"AuthHandler": 1})
        idf = lex._compute_symbol_idf(db, "AuthHandler")
        # log(1000 / (1 + 1)) ≈ 6.21
        assert idf > 5.0

    def test_compute_symbol_idf_generic_name(self):
        db = _FakeDB(nodes={}, fts_results={}, total_nodes=1000, match_counts={"User": 200})
        idf = lex._compute_symbol_idf(db, "User")
        # log(1000 / 201) ≈ 1.6
        assert idf < 2.0

    def test_allowed_tiers_high_idf_class(self):
        assert lex._allowed_tiers("class", 5.0) == frozenset({"T1", "T2", "T3", "T4"})

    def test_allowed_tiers_mid_idf_class(self):
        assert lex._allowed_tiers("class", 2.0) == frozenset({"T1", "T2"})

    def test_allowed_tiers_low_idf(self):
        assert lex._allowed_tiers("class", 1.0) == frozenset()

    def test_allowed_tiers_method_capped_at_t2(self):
        # Even with very high IDF, methods top out at T2.
        assert lex._allowed_tiers("method", 10.0) == frozenset({"T1", "T2"})

    def test_allowed_tiers_field_skipped(self):
        assert lex._allowed_tiers("field", 10.0) == frozenset()


# ──────────────────────────────────────────────────────────────────────────
# REST disambiguation
# ──────────────────────────────────────────────────────────────────────────


class TestRESTDisambiguation:
    def test_detect_rest_endpoint_via_decorator(self):
        node = {"decorators": ["@router.get('/users')"]}
        assert lex._detect_rest_endpoint(node) is True

    def test_detect_rest_endpoint_via_base_class(self):
        node = {"base_classes": ["APIView"]}
        assert lex._detect_rest_endpoint(node) is True

    def test_detect_rest_endpoint_via_source_text(self):
        node = {"source_text": "@app.route('/api/v1/users')\nclass API:\n    pass"}
        assert lex._detect_rest_endpoint(node) is True

    def test_detect_rest_endpoint_negative(self):
        node = {"decorators": ["@dataclass"], "base_classes": ["BaseModel"]}
        assert lex._detect_rest_endpoint(node) is False

    def test_compute_fts_symbol_name(self):
        assert lex._compute_fts_symbol_name("API", "users") == "users API"

    def test_compute_fts_symbol_name_same(self):
        # Stem and symbol identical → no rewrite.
        assert lex._compute_fts_symbol_name("api", "api") == "api"

    def test_compute_fts_symbol_name_empty_stem(self):
        assert lex._compute_fts_symbol_name("API", "") == "API"

    def test_maybe_disambiguate_rest_rewrites_query(self):
        node = {"decorators": ["@router.get('/users')"]}
        q, is_rest = lex.maybe_disambiguate_rest(
            node, "API", "src/api/users.py", flags=_flags(),
        )
        assert is_rest is True
        assert q == "users API"

    def test_maybe_disambiguate_rest_skips_non_generic(self):
        node = {"decorators": ["@router.get('/users')"]}
        q, is_rest = lex.maybe_disambiguate_rest(
            node, "AuthService", "src/api/users.py", flags=_flags(),
        )
        assert is_rest is False
        assert q == "AuthService"

    def test_maybe_disambiguate_rest_disabled(self):
        node = {"decorators": ["@router.get('/users')"]}
        q, is_rest = lex.maybe_disambiguate_rest(
            node, "API", "src/api/users.py",
            flags=_flags(orphan_rest_disambig=False),
        )
        assert is_rest is False
        assert q == "API"


# ──────────────────────────────────────────────────────────────────────────
# Tiered cascade
# ──────────────────────────────────────────────────────────────────────────


def _hit(node_id: str, *, rel_path: str, symbol_type: str = "class", score_norm: float = 0.5) -> Dict[str, Any]:
    return {
        "node_id": node_id,
        "rel_path": rel_path,
        "symbol_type": symbol_type,
        "score_norm": score_norm,
    }


class TestTieredCascade:
    def test_t1_same_dir_arch_target(self):
        # Orphan in src/auth/handlers.py — match in src/auth/users.py wins T1.
        nodes = {
            "orphan": {
                "symbol_name": "AuthHandler",
                "symbol_type": "class",
                "rel_path": "src/auth/handlers.py",
            },
        }
        fts = {
            "AuthHandler": [
                _hit("same_dir", rel_path="src/auth/users.py", symbol_type="class"),
                _hit("other_dir", rel_path="src/db/conn.py", symbol_type="class"),
            ],
        }
        db = _FakeDB(
            nodes=nodes, fts_results=fts,
            total_nodes=1000, match_counts={"AuthHandler": 1},
        )
        G = nx.MultiDiGraph()
        G.add_node("orphan")

        hits = lex.resolve_orphans_lexical_tiered(
            db, G, "orphan", flags=_flags(),
        )
        assert len(hits) == 1
        assert hits[0]["node_id"] == "same_dir"
        assert hits[0]["_tier"] == "T1"

    def test_t4_skipped_for_short_name(self):
        # Short name + only-global, non-architectural hits → T1/T2 fail
        # (different dir), T3 fails (not ARCH target), T4 is the only
        # remaining tier and is gated on len(name) >= 8.
        nodes = {
            "orphan": {
                "symbol_name": "Foo",  # length 3 < 8
                "symbol_type": "class",
                "rel_path": "src/x/foo.py",
            },
        }
        fts = {"Foo": [_hit("global_hit", rel_path="other/dir/y.py", symbol_type="parameter")]}
        db = _FakeDB(
            nodes=nodes, fts_results=fts,
            total_nodes=10000, match_counts={"Foo": 1},
        )
        G = nx.MultiDiGraph()
        G.add_node("orphan")

        hits = lex.resolve_orphans_lexical_tiered(db, G, "orphan", flags=_flags())
        assert hits == []

    def test_idf_low_skips_lexical_entirely(self):
        nodes = {
            "orphan": {
                "symbol_name": "User",
                "symbol_type": "class",
                "rel_path": "src/models/user.py",
            },
        }
        fts = {
            "User": [
                _hit("a", rel_path="src/models/foo.py", symbol_type="class"),
                _hit("b", rel_path="src/models/bar.py", symbol_type="class"),
            ],
        }
        # 200/1000 → IDF ≈ 1.6 → still T1+T2 — push lower to skip entirely.
        db = _FakeDB(
            nodes=nodes, fts_results=fts,
            total_nodes=1000, match_counts={"User": 800},  # IDF ≈ log(1000/801) ≈ 0.22
        )
        G = nx.MultiDiGraph()
        G.add_node("orphan")

        hits = lex.resolve_orphans_lexical_tiered(db, G, "orphan", flags=_flags())
        assert hits == []

    def test_field_symbol_type_skipped(self):
        nodes = {
            "orphan": {
                "symbol_name": "user_id",
                "symbol_type": "field",
                "rel_path": "src/models/user.py",
            },
        }
        fts = {"user_id": [_hit("a", rel_path="src/models/user.py", symbol_type="class")]}
        db = _FakeDB(nodes=nodes, fts_results=fts, total_nodes=1000, match_counts={"user_id": 1})
        G = nx.MultiDiGraph()
        G.add_node("orphan")

        hits = lex.resolve_orphans_lexical_tiered(db, G, "orphan", flags=_flags())
        assert hits == []

    def test_self_hit_filtered(self):
        nodes = {
            "orphan": {
                "symbol_name": "AuthHandler",
                "symbol_type": "class",
                "rel_path": "src/auth/handlers.py",
            },
        }
        # The only "match" is the orphan itself.
        fts = {
            "AuthHandler": [_hit("orphan", rel_path="src/auth/handlers.py", symbol_type="class")],
        }
        db = _FakeDB(
            nodes=nodes, fts_results=fts,
            total_nodes=1000, match_counts={"AuthHandler": 1},
        )
        G = nx.MultiDiGraph()
        G.add_node("orphan")

        hits = lex.resolve_orphans_lexical_tiered(db, G, "orphan", flags=_flags())
        assert hits == []

    def test_min_score_norm_filter(self):
        nodes = {
            "orphan": {
                "symbol_name": "AuthHandler",
                "symbol_type": "class",
                "rel_path": "src/auth/handlers.py",
            },
        }
        fts = {
            "AuthHandler": [
                _hit("weak", rel_path="src/auth/users.py", symbol_type="class", score_norm=0.05),
            ],
        }
        db = _FakeDB(
            nodes=nodes, fts_results=fts,
            total_nodes=1000, match_counts={"AuthHandler": 1},
        )
        G = nx.MultiDiGraph()
        G.add_node("orphan")

        hits = lex.resolve_orphans_lexical_tiered(
            db, G, "orphan", flags=_flags(fts_min_score_norm=0.15),
        )
        assert hits == []

    def test_rest_disambiguation_rewrites_query(self):
        # Orphan "API" class with a router decorator — query becomes
        # "users API" so it can find the right neighbours.
        nodes = {
            "orphan": {
                "symbol_name": "API",
                "symbol_type": "class",
                "rel_path": "src/api/users.py",
                "decorators": ["@router.get('/users')"],
            },
        }
        fts = {
            "users API": [
                _hit("user_repo", rel_path="src/api/users_repo.py", symbol_type="class"),
            ],
            # Sanity: a query for the bare "API" must NOT be issued.
            "API": [_hit("noise", rel_path="elsewhere/x.py", symbol_type="class")],
        }
        db = _FakeDB(
            nodes=nodes, fts_results=fts,
            total_nodes=1000, match_counts={"users API": 2, "API": 50},
        )
        G = nx.MultiDiGraph()
        G.add_node("orphan", **nodes["orphan"])  # mirror node_data

        hits = lex.resolve_orphans_lexical_tiered(db, G, "orphan", flags=_flags())
        assert len(hits) >= 1
        assert hits[0]["node_id"] == "user_repo"
        # Graph attr is upgraded in-place.
        assert G.nodes["orphan"]["symbol_type"] == "rest_endpoint"
