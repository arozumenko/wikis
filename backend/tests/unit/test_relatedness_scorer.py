"""Phase 8 — Relatedness scorer."""

from __future__ import annotations

import pytest

from app.core.code_graph.relatedness_scorer import jaccard, score_repo_pair


class TestJaccard:
    def test_identical(self):
        assert jaccard(["a", "b"], ["a", "b"]) == 1.0

    def test_disjoint(self):
        assert jaccard(["a"], ["b"]) == 0.0

    def test_partial(self):
        assert jaccard(["a", "b", "c"], ["b", "c", "d"]) == pytest.approx(2 / 4)

    def test_both_empty(self):
        assert jaccard([], []) == 0.0

    def test_skips_falsy(self):
        # Empty strings dropped before set construction.
        assert jaccard(["a", "", None], ["a"]) == 1.0


class TestPreFilter:
    def test_no_lang_no_api_overlap_returns_zero(self):
        score, breakdown = score_repo_pair(
            languages_a=["python"],
            languages_b=["go"],
            api_a=["GET /users"],
            api_b=["POST /orders"],
            symbol_names_a=["UserService"] * 10,
            symbol_names_b=["UserService"] * 10,  # naming would be 1.0 but pre-filter wins
        )
        assert score == 0.0
        assert breakdown["skipped"] == "no_overlap"
        # All component scores zeroed.
        assert breakdown["lang"] == 0.0 and breakdown["naming"] == 0.0


class TestScoreRepoPair:
    def test_pure_lang_overlap(self):
        score, breakdown = score_repo_pair(
            languages_a=["python", "go"],
            languages_b=["python", "go"],
        )
        assert breakdown["lang"] == 1.0
        # 0.30 * 1.0 = 0.30 (other components zero)
        assert score == pytest.approx(0.30)

    def test_full_overlap(self):
        score, breakdown = score_repo_pair(
            languages_a=["python"],
            languages_b=["python"],
            deps_a=["fastapi"],
            deps_b=["fastapi"],
            api_a=["GET /users"],
            api_b=["GET /users"],
            symbol_names_a=["UserService"],
            symbol_names_b=["UserService"],
        )
        assert score == pytest.approx(1.0)
        assert breakdown["lang"] == 1.0
        assert breakdown["api"] == 1.0
        assert breakdown["naming"] == 1.0

    def test_api_overlap_uses_min_denominator(self):
        # Wiki A has 1 surface; wiki B has 100 — but they share that 1.
        # Overlap should be 1/min(1,100) = 1.0 (not 1/100).
        score, breakdown = score_repo_pair(
            languages_a=["python"],
            languages_b=["python"],
            api_a=["GET /users"],
            api_b=["GET /users"] + [f"GET /{i}" for i in range(99)],
        )
        assert breakdown["api"] == 1.0

    def test_score_clamped_to_unit(self):
        # Crafted to exercise rounding edge — should still stay <= 1.0
        score, _ = score_repo_pair(
            languages_a=["a", "b", "c"],
            languages_b=["a", "b", "c"],
            deps_a=["x"], deps_b=["x"],
            api_a=["y"], api_b=["y"],
            symbol_names_a=["s"], symbol_names_b=["s"],
        )
        assert 0.0 <= score <= 1.0

    def test_top_k_caps_naming(self):
        names_a = [f"sym{i}" for i in range(500)]
        names_b = names_a[:100] + [f"other{i}" for i in range(400)]
        # With top_k=200, the top-frequency comparison still includes
        # most of the shared 100 names → meaningful naming score.
        score, breakdown = score_repo_pair(
            languages_a=["python"], languages_b=["python"],
            symbol_names_a=names_a, symbol_names_b=names_b,
            top_k=200,
        )
        assert breakdown["naming"] > 0.0
