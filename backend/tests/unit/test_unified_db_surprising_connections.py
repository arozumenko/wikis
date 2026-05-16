"""Storage-level tests for UnifiedWikiDB.compute_surprising_connections.

#121 Phase 2 — cross-cluster edges ranked by Jaccard distance over
cluster "contexts" (top-level folder prefixes).

Fixture topology:

    Cluster 1 (frontend/): F1, F2
    Cluster 2 (backend/):  B1, B2
    Cluster 3 (frontend/widgets/): W1, W2

Edges:
    F1 → B1  (cross-cluster, frontend ↔ backend — surprising)
    F2 → B2  (same pair, also surprising)
    F1 → W1  (cross-cluster, both frontend — NOT surprising)
    B1 → B2  (intra-cluster, ignored)
"""

import pytest

from app.core.unified_db import UnifiedWikiDB


@pytest.fixture()
def db(tmp_path):
    d = UnifiedWikiDB(tmp_path / "surprise.wiki.db", embedding_dim=8)
    try:
        _seed(d)
        yield d
    finally:
        d.close()


def _seed(db: UnifiedWikiDB) -> None:
    nodes = [
        # (node_id, rel_path, symbol_name, macro_cluster)
        ("F1", "frontend/Foo.tsx", "Foo", 1),
        ("F2", "frontend/Bar.tsx", "Bar", 1),
        ("B1", "backend/auth.py", "Auth", 2),
        ("B2", "backend/cache.py", "Cache", 2),
        ("W1", "frontend/widgets/Btn.tsx", "Btn", 3),
        ("W2", "frontend/widgets/Inp.tsx", "Inp", 3),
    ]
    for nid, rel_path, name, cluster in nodes:
        db.upsert_node(
            nid,
            rel_path=rel_path,
            symbol_name=name,
            symbol_type="class",
            language="typescript" if rel_path.endswith(".tsx") else "python",
            macro_cluster=cluster,
        )
    edges = [
        ("F1", "B1", "calls", "EXTRACTED"),
        ("F2", "B2", "calls", "EXTRACTED"),
        ("F1", "W1", "imports", "EXTRACTED"),
        ("B1", "B2", "calls", "EXTRACTED"),  # intra-cluster
    ]
    for src, tgt, rel_type, conf in edges:
        db.upsert_edge(src, tgt, rel_type, confidence=conf)
    db.conn.commit()


# ---------------------------------------------------------------------------
# Happy path
# ---------------------------------------------------------------------------


def test_returns_top_pair_with_highest_jaccard_distance(db):
    result = db.compute_surprising_connections(top_n=10)
    assert "pairs" in result
    assert len(result["pairs"]) >= 1

    top = result["pairs"][0]
    # frontend (1) ↔ backend (2) has fully disjoint contexts: {frontend} vs
    # {backend} — Jaccard distance is 1.0.
    assert top["cluster_a"] == 1
    assert top["cluster_b"] == 2
    assert top["jaccard_distance"] == pytest.approx(1.0)
    assert top["context_a"] == ["frontend"]
    assert top["context_b"] == ["backend"]
    assert top["edge_count"] == 2


def test_intra_top_level_pair_has_lower_distance(db):
    """Both cluster 1 and cluster 3 have rel_paths starting with
    ``frontend`` (depth-1 prefix), so their contexts overlap entirely
    and the Jaccard distance is 0.0."""
    result = db.compute_surprising_connections(top_n=10, context_depth=1)
    pair_13 = next(
        (p for p in result["pairs"] if p["cluster_a"] == 1 and p["cluster_b"] == 3),
        None,
    )
    assert pair_13 is not None
    assert pair_13["jaccard_distance"] == pytest.approx(0.0)


def test_orders_by_jaccard_distance_descending(db):
    result = db.compute_surprising_connections(top_n=10)
    distances = [p["jaccard_distance"] for p in result["pairs"]]
    assert distances == sorted(distances, reverse=True)


def test_sample_edges_include_hydrated_source_and_target(db):
    result = db.compute_surprising_connections(top_n=10, sample_edges_per_pair=5)
    pair_12 = next(p for p in result["pairs"] if (p["cluster_a"], p["cluster_b"]) == (1, 2))
    # Two cross-cluster edges between F1→B1 and F2→B2 should both be
    # included as sample edges.
    assert len(pair_12["sample_edges"]) == 2
    edge_pairs = {
        (e["source_id"], e["target_id"]) for e in pair_12["sample_edges"]
    }
    assert edge_pairs == {("F1", "B1"), ("F2", "B2")}
    # Hydrated metadata is present.
    sample = pair_12["sample_edges"][0]
    assert "source_name" in sample
    assert "target_path" in sample
    assert "rel_type" in sample
    assert "confidence" in sample


def test_top_n_caps_returned_pairs(db):
    result = db.compute_surprising_connections(top_n=1)
    assert len(result["pairs"]) == 1


def test_context_depth_changes_resolution(db):
    """With depth=2, cluster 1 (``frontend/``) and cluster 3
    (``frontend/widgets/``) become more distinct — depth-2 prefixes
    are ``frontend`` vs ``frontend/widgets`` so Jaccard rises."""
    deep = db.compute_surprising_connections(top_n=10, context_depth=2)
    pair_13 = next(
        (p for p in deep["pairs"] if p["cluster_a"] == 1 and p["cluster_b"] == 3),
        None,
    )
    assert pair_13 is not None
    # contexts: {"frontend"} vs {"frontend/widgets"} — fully disjoint
    # → distance 1.0
    assert pair_13["jaccard_distance"] == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


def test_empty_graph_returns_no_pairs(tmp_path):
    db = UnifiedWikiDB(tmp_path / "empty.wiki.db", embedding_dim=8)
    try:
        result = db.compute_surprising_connections(top_n=10)
        assert result == {"pairs": [], "skipped_pairs": 0}
    finally:
        db.close()


def test_intra_cluster_edges_are_ignored(tmp_path):
    db = UnifiedWikiDB(tmp_path / "intra.wiki.db", embedding_dim=8)
    try:
        db.upsert_node("X", rel_path="a/x.py", macro_cluster=1)
        db.upsert_node("Y", rel_path="a/y.py", macro_cluster=1)
        db.upsert_edge("X", "Y", "calls", confidence="EXTRACTED")
        db.conn.commit()
        result = db.compute_surprising_connections(top_n=10)
        assert result == {"pairs": [], "skipped_pairs": 0}
    finally:
        db.close()


def test_nodes_without_cluster_ignored(tmp_path):
    """Nodes with NULL macro_cluster don't contribute to pairs."""
    db = UnifiedWikiDB(tmp_path / "nullc.wiki.db", embedding_dim=8)
    try:
        db.upsert_node("X", rel_path="a/x.py")  # no macro_cluster
        db.upsert_node("Y", rel_path="b/y.py", macro_cluster=2)
        db.upsert_edge("X", "Y", "calls", confidence="EXTRACTED")
        db.conn.commit()
        result = db.compute_surprising_connections(top_n=10)
        assert result == {"pairs": [], "skipped_pairs": 0}
    finally:
        db.close()


def test_root_level_files_dont_pollute_context(tmp_path):
    """Flat files (no folder prefix) should produce empty prefix sets,
    not be treated as their own folder. Two clusters of root-level
    files connected to each other should show up as ``skipped_pairs``
    rather than producing a misleading Jaccard score."""
    db = UnifiedWikiDB(tmp_path / "root.wiki.db", embedding_dim=8)
    try:
        # Cluster 1: only root-level files
        db.upsert_node("A", rel_path="config.py", macro_cluster=1)
        # Cluster 2: only root-level files
        db.upsert_node("B", rel_path="main.py", macro_cluster=2)
        db.upsert_edge("A", "B", "calls", confidence="EXTRACTED")
        db.conn.commit()
        result = db.compute_surprising_connections(top_n=10)
        assert result["pairs"] == []
        # The pair existed (cross-cluster edge present) but was
        # skipped — operators can see the count.
        assert result["skipped_pairs"] == 1
    finally:
        db.close()


def test_path_prefix_strips_filename(db):
    """Direct unit test: the prefix helper drops the final segment.
    Documents the contract the algorithm relies on so callers can
    reason about the Jaccard scoring without reading SQL."""
    pp = db._path_prefix
    assert pp("frontend/widgets/Btn.tsx", 1) == "frontend"
    assert pp("frontend/widgets/Btn.tsx", 2) == "frontend/widgets"
    assert pp("frontend/widgets/Btn.tsx", 5) == "frontend/widgets"
    # Root-level file → empty prefix (excluded from context set).
    assert pp("Btn.tsx", 1) == ""
    assert pp("", 1) == ""


def test_directional_edges_collapse_to_unordered_pair(db):
    """A→B and B→A between the same two clusters should produce one
    pair entry, not two."""
    # Add a reverse edge to the existing graph.
    db.upsert_edge("B1", "F1", "calls", confidence="EXTRACTED")
    db.conn.commit()
    result = db.compute_surprising_connections(top_n=10)
    f_b_pairs = [
        p for p in result["pairs"]
        if {p["cluster_a"], p["cluster_b"]} == {1, 2}
    ]
    assert len(f_b_pairs) == 1
    # Edge count now reflects the third edge.
    assert f_b_pairs[0]["edge_count"] == 3


def test_deterministic_tiebreak_when_jaccard_equal(tmp_path):
    """Two pairs at the same Jaccard distance should always come back
    in a stable order so callers can rely on the ranking."""
    db = UnifiedWikiDB(tmp_path / "tie.wiki.db", embedding_dim=8)
    try:
        # 4 clusters, fully-disjoint contexts (Jaccard = 1.0 for every pair).
        for cid, prefix in [(1, "a"), (2, "b"), (3, "c"), (4, "d")]:
            db.upsert_node(
                f"n{cid}",
                rel_path=f"{prefix}/x.py",
                macro_cluster=cid,
            )
        # Cross-cluster edges
        db.upsert_edge("n1", "n2", "calls", confidence="EXTRACTED")
        db.upsert_edge("n3", "n4", "calls", confidence="EXTRACTED")
        db.conn.commit()
        r1 = db.compute_surprising_connections(top_n=10)
        r2 = db.compute_surprising_connections(top_n=10)
        order1 = [(p["cluster_a"], p["cluster_b"]) for p in r1["pairs"]]
        order2 = [(p["cluster_a"], p["cluster_b"]) for p in r2["pairs"]]
        assert order1 == order2
        # Tie-break is (cluster_a, cluster_b) ascending after the
        # primary edge_count tiebreak: both pairs have count=1, so
        # (1,2) < (3,4).
        assert order1 == [(1, 2), (3, 4)]
    finally:
        db.close()
