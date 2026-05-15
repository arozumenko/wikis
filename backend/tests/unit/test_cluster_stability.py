"""Unit tests for ``app.core.cluster_stability`` (#116 PR 4).

The remap algorithm exists for one purpose: keep cluster IDs (and
therefore ``compute_page_id`` hashes, and therefore wikilinks) stable
across regenerations when membership doesn't change much.

Tests cover the contract:

* Identity case (clusters unchanged) → every new ID maps to its old peer
* Below-threshold case → un-remapped (fresh ID, no spurious match)
* Above-threshold case → remap claims the old ID
* Bijective swap → two old clusters partially shuffled members, both
  still match correctly by largest-first ordering
* Brand-new cluster → fresh ID counted from ``max(old) + 1``
* Edge cases: empty old, empty new, both empty, single-node clusters
"""

from __future__ import annotations

import pytest

from app.core.cluster_stability import (
    DEFAULT_JACCARD_THRESHOLD,
    apply_remap_to_assignments,
    compute_jaccard_remap,
    jaccard,
)


# ---------------------------------------------------------------------------
# jaccard
# ---------------------------------------------------------------------------


class TestJaccard:
    def test_identical_sets_yield_one(self) -> None:
        assert jaccard({"a", "b", "c"}, {"a", "b", "c"}) == 1.0

    def test_disjoint_sets_yield_zero(self) -> None:
        assert jaccard({"a"}, {"b"}) == 0.0

    def test_half_overlap(self) -> None:
        # |∩| = 1, |∪| = 3 → 1/3
        result = jaccard({"a", "b"}, {"b", "c"})
        assert abs(result - 1 / 3) < 1e-9

    def test_both_empty_yields_zero(self) -> None:
        # Documented edge: 0/0 → 0.0 (arbitrary but consistent).
        assert jaccard([], []) == 0.0

    def test_one_empty_yields_zero(self) -> None:
        assert jaccard({"a"}, []) == 0.0
        assert jaccard([], {"a"}) == 0.0


# ---------------------------------------------------------------------------
# compute_jaccard_remap
# ---------------------------------------------------------------------------


class TestRemapIdentity:
    def test_unchanged_clusters_remap_to_themselves(self) -> None:
        # Identical membership, different Leiden IDs.
        old = {3: ["a", "b", "c"], 7: ["d", "e"]}
        new = {0: ["a", "b", "c"], 1: ["d", "e"]}
        remap = compute_jaccard_remap(old, new)
        assert remap == {0: 3, 1: 7}

    def test_returns_empty_when_new_is_empty(self) -> None:
        assert compute_jaccard_remap({1: ["a"]}, {}) == {}


class TestRemapBelowThreshold:
    def test_below_threshold_gets_fresh_id(self) -> None:
        # Old: {a, b, c, d}; new: {a, x, y, z} → intersection 1, union 7
        # → Jaccard ~ 0.14, far below 0.5.
        old = {5: ["a", "b", "c", "d"]}
        new = {0: ["a", "x", "y", "z"]}
        remap = compute_jaccard_remap(old, new)
        # New gets a fresh ID, not 5 (would have caused false-positive remap).
        assert remap[0] != 5
        assert remap[0] == 6  # max(old) + 1

    def test_custom_threshold_is_honored(self) -> None:
        # 33% overlap — clears 0.3 but not 0.5.
        old = {1: ["a", "b"]}
        new = {0: ["b", "c"]}
        # Default threshold: no match → fresh ID.
        assert compute_jaccard_remap(old, new) == {0: 2}
        # Loose threshold: matches.
        assert compute_jaccard_remap(old, new, threshold=0.3) == {0: 1}


class TestRemapAboveThreshold:
    def test_overlap_just_above_threshold(self) -> None:
        # Old: {a, b, c, d}; new: {a, b, c, e} → intersection 3, union 5
        # → Jaccard 0.6, above 0.5.
        old = {9: ["a", "b", "c", "d"]}
        new = {0: ["a", "b", "c", "e"]}
        assert compute_jaccard_remap(old, new) == {0: 9}

    def test_largest_first_claims_best_match(self) -> None:
        # Two new clusters tie for "best match" against old=1, but the
        # larger one wins. Smaller one falls back to fresh ID.
        old = {1: ["a", "b", "c", "d", "e"], 2: ["f"]}
        # Big new cluster overlaps heavily with old 1.
        # Small new cluster also overlaps with old 1, but less.
        new = {
            10: ["a", "b", "c", "d", "x"],  # 4/6 with old 1 → 0.67
            11: ["a"],                       # 1/5 with old 1 → 0.20
        }
        remap = compute_jaccard_remap(old, new)
        assert remap[10] == 1   # big claimed old 1
        # Small didn't clear threshold against old 2 either, gets fresh ID.
        assert remap[11] not in {1, 2}


class TestRemapBijectiveSwap:
    def test_two_clusters_partially_shuffled_still_match_correctly(self) -> None:
        # Old: A={a,b,c}, B={x,y,z}. New: A'={a,b,d} (close to A),
        # B'={x,y,w} (close to B). Both should map to their counterpart.
        old = {10: ["a", "b", "c"], 20: ["x", "y", "z"]}
        new = {0: ["a", "b", "d"], 1: ["x", "y", "w"]}
        remap = compute_jaccard_remap(old, new)
        assert remap == {0: 10, 1: 20}


class TestRemapFreshIds:
    def test_brand_new_cluster_gets_max_plus_one(self) -> None:
        old = {3: ["a"], 7: ["b"]}
        new = {0: ["a"], 1: ["b"], 2: ["c"]}  # 2 is brand new
        remap = compute_jaccard_remap(old, new)
        assert remap[0] == 3
        assert remap[1] == 7
        # Fresh ID skips over the already-claimed old IDs.
        assert remap[2] == 8

    def test_fresh_ids_skip_existing_old_ids_when_unclaimed(self) -> None:
        # When old IDs aren't sequential, fresh IDs still go past max(old).
        old = {0: ["a"], 5: ["b"]}
        new = {10: ["fresh1"], 11: ["fresh2"]}  # no overlap with old
        remap = compute_jaccard_remap(old, new)
        assert sorted(remap.values()) == [6, 7]  # starts at max(0,5)+1=6

    def test_no_old_clusters_starts_fresh_at_zero(self) -> None:
        new = {99: ["a"], 100: ["b"]}
        remap = compute_jaccard_remap({}, new)
        # IDs start at 0 since there's no old high-water mark.
        assert sorted(remap.values()) == [0, 1]


class TestRemapEdgeCases:
    def test_old_cluster_loses_all_members_becomes_dead(self) -> None:
        # Old cluster has nothing in common with any new cluster.
        # The old ID effectively "dies"; no new cluster claims it.
        old = {1: ["a", "b", "c"]}
        new = {0: ["x", "y", "z"]}
        remap = compute_jaccard_remap(old, new)
        # New cluster gets fresh ID 2, not the dead old ID 1.
        assert remap == {0: 2}

    def test_single_node_clusters(self) -> None:
        # Single-node clusters require exact match for Jaccard = 1.0.
        old = {0: ["a"], 1: ["b"]}
        new = {10: ["a"], 11: ["c"]}
        remap = compute_jaccard_remap(old, new)
        assert remap[10] == 0  # exact match
        # 11 has no overlap → fresh ID, skipping the unclaimed old 1.
        assert remap[11] not in {0, 1}

    def test_threshold_exactly_at_boundary(self) -> None:
        # |∩|=2, |∪|=4 → Jaccard exactly 0.5; default threshold is ≥0.5
        # so this should match (boundary is inclusive).
        old = {1: ["a", "b", "c"]}
        new = {0: ["a", "b", "d"]}
        assert abs(2 / 4 - 0.5) < 1e-9  # confirm we're at the boundary
        # Hmm actually |∩|=2 ({a,b}), |∪|=4 ({a,b,c,d}) → 0.5 ✓
        assert compute_jaccard_remap(old, new) == {0: 1}


class TestApplyRemap:
    def test_rewrites_assignments(self) -> None:
        assignments = {"n1": 0, "n2": 1, "n3": 2}
        remap = {0: 10, 1: 20}  # 2 unmapped
        result = apply_remap_to_assignments(assignments, remap)
        assert result == {"n1": 10, "n2": 20, "n3": 2}  # n3 passes through

    def test_empty_remap_is_passthrough(self) -> None:
        assignments = {"n1": 0, "n2": 1}
        assert apply_remap_to_assignments(assignments, {}) == assignments

    def test_empty_assignments_returns_empty(self) -> None:
        assert apply_remap_to_assignments({}, {0: 10}) == {}


def test_default_threshold_is_documented() -> None:
    # Belt-and-braces: keep the constant exported, in case PR 5+ tunes it.
    assert DEFAULT_JACCARD_THRESHOLD == 0.5


# ---------------------------------------------------------------------------
# stabilize_hierarchical_assignments — full two-level remap
# ---------------------------------------------------------------------------


from app.core.cluster_stability import stabilize_hierarchical_assignments


class TestHierarchicalStabilize:
    def test_no_old_clusters_passes_through_unchanged(self) -> None:
        # First-ever generation: no remap, inputs returned verbatim.
        macro = {"n1": 0, "n2": 1}
        micro = {0: {"n1": 0}, 1: {"n2": 0}}
        out_macro, out_micro = stabilize_hierarchical_assignments(
            macro, micro, old_clusters={},
        )
        assert out_macro == macro
        assert out_micro == micro

    def test_identity_run_keeps_all_ids(self) -> None:
        # Old: macro 5 has micro 9 with {a, b}; macro 7 has micro 3 with {c}.
        # New comes back with same membership but arbitrary Leiden IDs.
        old = {5: {9: ["a", "b"]}, 7: {3: ["c"]}}
        macro = {"a": 0, "b": 0, "c": 1}
        micro = {0: {"a": 10, "b": 10}, 1: {"c": 20}}
        out_macro, out_micro = stabilize_hierarchical_assignments(
            macro, micro, old_clusters=old,
        )
        # Every node ends up in its previous macro+micro.
        assert out_macro == {"a": 5, "b": 5, "c": 7}
        assert out_micro == {5: {"a": 9, "b": 9}, 7: {"c": 3}}

    def test_membership_shift_within_macro_remaps_micro(self) -> None:
        # Macro stays stable (Jaccard well above 0.5), but micro membership
        # shifts a bit — the dominant micro still gets remapped.
        old = {
            0: {  # macro 0
                1: ["a", "b", "c", "d"],  # big micro
                2: ["e"],                 # small micro
            }
        }
        # New run: same macro membership, but micro 99 picks up {a,b,c,e}
        # (3/5 from old micro 1 → Jaccard 3/5 = 0.6, above 0.5).
        macro = {"a": 0, "b": 0, "c": 0, "d": 0, "e": 0}
        micro = {0: {"a": 99, "b": 99, "c": 99, "e": 99, "d": 100}}
        out_macro, out_micro = stabilize_hierarchical_assignments(
            macro, micro, old_clusters=old,
        )
        # Macro 0 stays 0 (only one macro in old, all nodes overlap).
        assert set(out_macro.values()) == {0}
        # Micro 99 → old micro 1 (Jaccard wins); micro 100 → fresh ID.
        assert out_micro[0]["a"] == 1
        assert out_micro[0]["b"] == 1
        assert out_micro[0]["c"] == 1
        assert out_micro[0]["e"] == 1
        # d was in old micro 1 but is now in fresh micro 100 → fresh ID.
        assert out_micro[0]["d"] != 1

    def test_brand_new_macro_gets_fresh_id(self) -> None:
        old = {0: {0: ["a", "b"]}}
        macro = {"a": 10, "b": 10, "c": 11}  # node "c" is brand new in its own macro
        micro = {10: {"a": 0, "b": 0}, 11: {"c": 0}}
        out_macro, _ = stabilize_hierarchical_assignments(
            macro, micro, old_clusters=old,
        )
        assert out_macro["a"] == 0  # original macro remapped
        assert out_macro["c"] not in {0}  # fresh macro ID
