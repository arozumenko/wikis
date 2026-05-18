"""Phase 4 (graph-quality roadmap) — RRF fusion + hybrid orphan Pass 2."""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import networkx as nx
import pytest

from app.core import graph_orphan_hybrid as hyb
from app.core.feature_flags import FeatureFlags


def _flags(**overrides: Any) -> FeatureFlags:
    base = dict(
        orphan_hybrid_search=True,
        orphan_rrf_k=60,
        orphan_rrf_threshold=0.0,
        orphan_hybrid_top_n=10,
        fts_min_score_norm=0.0,
    )
    base.update(overrides)
    return FeatureFlags(**base)


# ──────────────────────────────────────────────────────────────────────────
# RRF fusion
# ──────────────────────────────────────────────────────────────────────────


class TestRRFFuse:
    def test_empty_inputs_returns_empty(self):
        assert hyb.rrf_fuse() == []
        assert hyb.rrf_fuse([], []) == []

    def test_single_list_passthrough_orderpreserved(self):
        a = [{"node_id": "x"}, {"node_id": "y"}, {"node_id": "z"}]
        out = hyb.rrf_fuse(a, k=60)
        assert [h["node_id"] for h in out] == ["x", "y", "z"]
        # Strictly decreasing rrf_score.
        scores = [h["rrf_score"] for h in out]
        assert scores == sorted(scores, reverse=True)

    def test_overlap_boosts_rank(self):
        # "shared" appears in both lists; "alone" appears only in one.
        a = [{"node_id": "shared"}, {"node_id": "alone_a"}]
        b = [{"node_id": "alone_b"}, {"node_id": "shared"}]
        out = hyb.rrf_fuse(a, b, k=60)
        ids = [h["node_id"] for h in out]
        # "shared" must outrank both unique entries.
        assert ids[0] == "shared"

    def test_score_formula(self):
        # k=10, only one list, single hit at rank 1 → 1/(10+1) = 0.0909...
        out = hyb.rrf_fuse([{"node_id": "x"}], k=10)
        assert out[0]["rrf_score"] == pytest.approx(1 / 11)

    def test_score_aggregates_across_lists(self):
        # Same id at rank 1 in both → 2/(60+1).
        a = [{"node_id": "x"}]
        b = [{"node_id": "x"}]
        out = hyb.rrf_fuse(a, b, k=60)
        assert out[0]["rrf_score"] == pytest.approx(2 / 61)

    def test_payloads_merged(self):
        a = [{"node_id": "x", "score_norm": 0.7, "from": "fts"}]
        b = [{"node_id": "x", "distance": 0.1, "from": "vec"}]
        out = hyb.rrf_fuse(a, b, k=60)
        assert out[0]["score_norm"] == 0.7
        assert out[0]["distance"] == 0.1
        # Later list wins on conflict.
        assert out[0]["from"] == "vec"
        assert "rrf_score" in out[0]

    def test_invalid_k_raises(self):
        with pytest.raises(ValueError):
            hyb.rrf_fuse([{"node_id": "x"}], k=0)

    def test_skips_hits_without_id(self):
        out = hyb.rrf_fuse([{"node_id": "x"}, {"foo": "bar"}], k=60)
        assert [h["node_id"] for h in out] == ["x"]

    def test_custom_id_key(self):
        a = [{"path": "src/a.py"}, {"path": "src/b.py"}]
        b = [{"path": "src/b.py"}]
        out = hyb.rrf_fuse(a, b, k=60, id_key="path")
        assert out[0]["path"] == "src/b.py"


# ──────────────────────────────────────────────────────────────────────────
# Hybrid Pass 2 — orphan resolution
# ──────────────────────────────────────────────────────────────────────────


class _FakeDB:
    def __init__(
        self,
        nodes: Dict[str, Dict[str, Any]],
        fts_results: Optional[Dict[tuple, List[Dict[str, Any]]]] = None,
        vec_results: Optional[Dict[tuple, List[Dict[str, Any]]]] = None,
        embeddings: Optional[Dict[str, List[float]]] = None,
    ) -> None:
        self._nodes = nodes
        self._fts = fts_results or {}
        self._vec = vec_results or {}
        self._embeddings = embeddings or {}
        self.fts_calls: List[tuple] = []
        self.vec_calls: List[tuple] = []

    def get_node(self, node_id):
        return self._nodes.get(node_id)

    def search_fts5(self, *, query, limit=10):
        return list(self._fts.get((query, None), []))[:limit]

    def search_fts_with_path(self, query, *, path_prefix, limit=10):
        self.fts_calls.append((query, path_prefix))
        return list(self._fts.get((query, path_prefix), []))[:limit]

    def search_vec(self, embedding, *, k=10, path_prefix=None):
        self.vec_calls.append((tuple(embedding), path_prefix))
        return list(self._vec.get((tuple(embedding), path_prefix), []))[:k]

    def get_embedding_by_id(self, node_id):
        return self._embeddings.get(node_id)


class TestHybridPass2:
    def test_returns_empty_when_no_name(self):
        db = _FakeDB(nodes={"o": {}})
        G = nx.MultiDiGraph()
        G.add_node("o")
        assert hyb.resolve_orphans_hybrid(db, G, "o", flags=_flags()) == []

    def test_fts_only_when_no_embedding(self):
        # Hybrid Pass 2 intentionally aborts when no embedding is available so
        # the v2 cascade falls through to the tiered lexical pass (see
        # ``resolve_orphans_hybrid`` docstring). FTS hits alone do not
        # produce hybrid results.
        nodes = {
            "o": {"symbol_name": "AuthHandler", "rel_path": "src/auth/handler.py"},
        }
        fts = {
            ("AuthHandler", "src/auth"): [
                {"node_id": "h1", "score_norm": 0.7},
                {"node_id": "h2", "score_norm": 0.5},
            ],
        }
        db = _FakeDB(nodes=nodes, fts_results=fts)
        G = nx.MultiDiGraph()
        G.add_node("o", **nodes["o"])

        out = hyb.resolve_orphans_hybrid(db, G, "o", flags=_flags())
        assert out == []

    def test_hybrid_fuses_fts_and_vec(self):
        nodes = {
            "o": {"symbol_name": "AuthHandler", "rel_path": "src/auth/handler.py"},
        }
        fts = {("AuthHandler", "src/auth"): [{"node_id": "shared", "score_norm": 0.5}, {"node_id": "fts_only"}]}
        vec = {((0.1, 0.2), "src/auth"): [{"node_id": "vec_only", "distance": 0.05}, {"node_id": "shared", "distance": 0.10}]}
        db = _FakeDB(nodes=nodes, fts_results=fts, vec_results=vec, embeddings={"o": [0.1, 0.2]})
        G = nx.MultiDiGraph()
        G.add_node("o", **nodes["o"])

        out = hyb.resolve_orphans_hybrid(db, G, "o", flags=_flags())
        # "shared" appears in both lists → must rank first.
        assert out[0]["node_id"] == "shared"
        ids = {h["node_id"] for h in out}
        assert ids == {"shared", "fts_only", "vec_only"}

    def test_threshold_drops_weak_hits(self):
        nodes = {
            "o": {"symbol_name": "AuthHandler", "rel_path": "src/auth/handler.py"},
        }
        # Many FTS hits — last few will have low rrf_score.
        many = [{"node_id": f"h{i}"} for i in range(20)]
        fts = {("AuthHandler", "src/auth"): many}
        db = _FakeDB(nodes=nodes, fts_results=fts)
        G = nx.MultiDiGraph()
        G.add_node("o", **nodes["o"])

        # rrf_score for rank 20 with k=60 is 1/80 = 0.0125 < 0.013.
        out = hyb.resolve_orphans_hybrid(
            db, G, "o", flags=_flags(orphan_rrf_threshold=0.013),
        )
        assert "h19" not in {h["node_id"] for h in out}

    def test_self_hit_filtered_in_fts(self):
        nodes = {
            "o": {"symbol_name": "AuthHandler", "rel_path": "src/auth/handler.py"},
        }
        fts = {("AuthHandler", "src/auth"): [{"node_id": "o", "score_norm": 0.9}]}
        db = _FakeDB(nodes=nodes, fts_results=fts)
        G = nx.MultiDiGraph()
        G.add_node("o", **nodes["o"])

        out = hyb.resolve_orphans_hybrid(db, G, "o", flags=_flags())
        assert out == []

    def test_min_score_norm_filters_fts(self):
        nodes = {
            "o": {"symbol_name": "AuthHandler", "rel_path": "src/auth/handler.py"},
        }
        fts = {
            ("AuthHandler", "src/auth"): [
                {"node_id": "weak", "score_norm": 0.05},
                {"node_id": "strong", "score_norm": 0.9},
            ],
        }
        # An embedding is required for hybrid Pass 2 to run; supply one via
        # the orphan_embeddings cache so the FTS branch is exercised.
        vec = {((0.1, 0.2), "src/auth"): []}
        db = _FakeDB(nodes=nodes, fts_results=fts, vec_results=vec)
        G = nx.MultiDiGraph()
        G.add_node("o", **nodes["o"])

        out = hyb.resolve_orphans_hybrid(
            db, G, "o", flags=_flags(fts_min_score_norm=0.15),
            orphan_embeddings={"o": [0.1, 0.2]},
        )
        assert {h["node_id"] for h in out} == {"strong"}

    def test_uses_orphan_embeddings_cache(self):
        nodes = {
            "o": {"symbol_name": "AuthHandler", "rel_path": "src/auth/handler.py"},
        }
        vec = {((0.7, 0.1), "src/auth"): [{"node_id": "match"}]}
        db = _FakeDB(nodes=nodes, vec_results=vec)  # no embedding stored in DB
        G = nx.MultiDiGraph()
        G.add_node("o", **nodes["o"])

        out = hyb.resolve_orphans_hybrid(
            db, G, "o", flags=_flags(),
            orphan_embeddings={"o": [0.7, 0.1]},
        )
        assert {h["node_id"] for h in out} == {"match"}

    def test_embed_fn_is_not_called(self):
        # ``embed_fn`` is intentionally not invoked by Pass 2 — the cascade
        # must reuse persisted embeddings rather than re-embed orphans
        # synchronously (see ``resolve_orphans_hybrid`` docstring). When no
        # embedding is available, the function aborts and returns ``[]``.
        nodes = {
            "o": {
                "symbol_name": "AuthHandler",
                "rel_path": "src/auth/handler.py",
                "source_text": "class AuthHandler:\n    def handle(self): pass",
            },
        }
        vec = {((0.5, 0.5), "src/auth"): [{"node_id": "match"}]}
        db = _FakeDB(nodes=nodes, vec_results=vec)  # no cached/persisted embedding
        G = nx.MultiDiGraph()
        G.add_node("o", **nodes["o"])

        called: List[str] = []

        def _embed(text: str) -> List[float]:
            called.append(text)
            return [0.5, 0.5]

        out = hyb.resolve_orphans_hybrid(
            db, G, "o", flags=_flags(), embed_fn=_embed,
        )
        assert called == []  # embed_fn must not be invoked
        assert out == []
