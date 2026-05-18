"""End-to-end test for #116 PR 4 cluster ID stability.

Exercises the full path through ``_apply_cluster_stability``: seed a
real UnifiedWikiDB with prior cluster assignments, run a fresh
clustering pass with renumbered IDs, and assert that the remap brings
them back to the prior IDs (so ``compute_page_id`` hashes stay stable).

Doesn't run the real Leiden algorithm (requires igraph / leidenalg
which may not be installed) — instead exercises ``_apply_cluster_stability``
directly with deterministic input, which is the exact code path
``run_phase3`` would take after fresh clustering produces its
``macro_assignments`` / ``micro_assignments`` dicts.
"""

from __future__ import annotations

import pytest

from app.core.graph_clustering import _apply_cluster_stability
from app.core.storage.incremental import compute_page_id
from app.core.unified_db import UnifiedWikiDB


@pytest.fixture()
def db(tmp_path):
    d = UnifiedWikiDB(tmp_path / "cluster_e2e.wiki.db", embedding_dim=8)
    yield d
    d.close()


def _seed_node(
    db: UnifiedWikiDB,
    node_id: str,
    *,
    macro: int,
    micro: int,
    is_arch: int = 1,
) -> None:
    """Insert a node with an existing cluster assignment — simulates a
    prior run that already wrote clusters to repo_nodes."""
    db.upsert_node(
        node_id,
        rel_path="x.py",
        file_name="x.py",
        language="python",
        symbol_name=node_id,
        symbol_type="function",
        is_architectural=is_arch,
        source_text=f"def {node_id}(): pass",
        macro_cluster=macro,
        micro_cluster=micro,
    )


class TestApplyClusterStability:
    def test_first_run_passes_through_unchanged(
        self, db: UnifiedWikiDB,
    ) -> None:
        # No prior cluster assignments in the DB → remap is a no-op.
        macro = {"a": 0, "b": 1}
        micro = {0: {"a": 0}, 1: {"b": 0}}
        out_macro, out_micro, out_hubs = _apply_cluster_stability(
            db, macro, micro, {},
        )
        assert out_macro == macro
        assert out_micro == micro

    def test_remap_restores_old_ids_for_unchanged_membership(
        self, db: UnifiedWikiDB,
    ) -> None:
        # Seed prior state: cluster 5 = {a, b, c}, cluster 7 = {d, e}.
        _seed_node(db, "a", macro=5, micro=0)
        _seed_node(db, "b", macro=5, micro=0)
        _seed_node(db, "c", macro=5, micro=0)
        _seed_node(db, "d", macro=7, micro=0)
        _seed_node(db, "e", macro=7, micro=0)

        # Fresh Leiden re-numbered the same membership as macros 0 and 1.
        fresh_macro = {"a": 0, "b": 0, "c": 0, "d": 1, "e": 1}
        fresh_micro = {0: {"a": 9, "b": 9, "c": 9}, 1: {"d": 4, "e": 4}}

        stable_macro, stable_micro, stable_hubs = _apply_cluster_stability(
            db, fresh_macro, fresh_micro, {},
        )
        # Every node lands back in its prior cluster.
        assert stable_macro == {"a": 5, "b": 5, "c": 5, "d": 7, "e": 7}
        # Micros remap too (only one micro per cluster, so all → 0).
        assert stable_micro[5] == {"a": 0, "b": 0, "c": 0}
        assert stable_micro[7] == {"d": 0, "e": 0}
        # No hubs in this fixture → empty stable hubs.
        assert stable_hubs == {}

    def test_page_id_is_stable_when_clusters_are_stable(
        self, db: UnifiedWikiDB,
    ) -> None:
        """The original promise of PR 1 + PR 4: same input → same page_id
        across regens."""
        # Prior state.
        _seed_node(db, "auth.AuthService", macro=5, micro=0)
        _seed_node(db, "auth.login", macro=5, micro=0)

        # Fresh Leiden produced different macro IDs.
        fresh_macro = {"auth.AuthService": 99, "auth.login": 99}
        fresh_micro = {99: {"auth.AuthService": 42, "auth.login": 42}}

        # Page ID hashed from the *fresh* IDs would change every run.
        page_id_unstable = compute_page_id(
            "w1", 99, 42, "auth.AuthService", "Auth Service",
        )
        page_id_prior = compute_page_id(
            "w1", 5, 0, "auth.AuthService", "Auth Service",
        )
        assert page_id_unstable != page_id_prior

        # After remap, page_id matches the prior run.
        stable_macro, stable_micro, _ = _apply_cluster_stability(
            db, fresh_macro, fresh_micro, {},
        )
        # The stable macro for "auth.AuthService" should now be 5, micro 0.
        assert stable_macro["auth.AuthService"] == 5
        assert stable_micro[5]["auth.AuthService"] == 0

        # And the recomputed page_id matches the prior page_id.
        page_id_post_remap = compute_page_id(
            "w1",
            stable_macro["auth.AuthService"],
            stable_micro[5]["auth.AuthService"],
            "auth.AuthService",
            "Auth Service",
        )
        assert page_id_post_remap == page_id_prior

    def test_hub_assignments_are_remapped_too(self, db: UnifiedWikiDB) -> None:
        """Tech-lead blocker fix: hub_assignments dict must be rebuilt
        from the post-remap macro/micro dicts. Without this, persist_clusters
        would overwrite the remapped macro/micro cells for hub nodes
        with their pre-remap IDs via the INSERT OR REPLACE batch path."""
        # Prior state: cluster 5 has a hub h_arch + member m_1.
        _seed_node(db, "h_arch", macro=5, micro=0)
        _seed_node(db, "m_1", macro=5, micro=0)

        # Fresh Leiden produced different IDs (macro 99). The caller has
        # already merged hubs into the assignment dicts.
        fresh_macro = {"h_arch": 99, "m_1": 99}
        fresh_micro = {99: {"h_arch": 42, "m_1": 42}}
        # The hub_assignments dict still carries the *fresh* (pre-remap)
        # macro/micro tuples — this is what reintegrate_hubs returns.
        fresh_hubs = {"h_arch": (99, 42)}

        stable_macro, stable_micro, stable_hubs = _apply_cluster_stability(
            db, fresh_macro, fresh_micro, fresh_hubs,
        )

        # macro_assignments + micro_assignments remap to the prior IDs.
        assert stable_macro["h_arch"] == 5
        assert stable_micro[5]["h_arch"] == 0

        # And — the critical bit — hub_assignments matches.
        # Without the fix, this would still be (99, 42).
        assert stable_hubs["h_arch"] == (5, 0)

    def test_handles_db_read_error_gracefully(self, db: UnifiedWikiDB) -> None:
        """If the DB read fails, the helper should fall back to passthrough
        — never crash the clustering pipeline."""

        # Force get_all_clusters to raise.
        def _boom() -> dict:
            raise RuntimeError("simulated DB read failure")

        db.get_all_clusters = _boom  # type: ignore[assignment]

        macro = {"a": 0}
        micro = {0: {"a": 0}}
        hubs = {"a": (0, 0)}
        out_macro, out_micro, out_hubs = _apply_cluster_stability(
            db, macro, micro, hubs,
        )
        assert out_macro == macro
        assert out_micro == micro
        # Hubs pass through unchanged when the remap is skipped.
        assert out_hubs == hubs
