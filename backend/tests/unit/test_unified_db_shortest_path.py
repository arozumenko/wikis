"""Storage-level tests for UnifiedWikiDB.shortest_path (#121 Phase 1).

These exercise the SQL recursive-CTE BFS directly against a real
SQLite database, so they catch bugs in path reconstruction and cycle
detection that mocked unit tests would miss.

Graph topology used here (undirected for traversal):

    A ── B ── C ── D
         │
         E ── F

Plus an isolated pair ``G ── H`` to exercise the "no path" case.
"""

import pytest

from app.core.unified_db import UnifiedWikiDB

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def db(tmp_path):
    d = UnifiedWikiDB(tmp_path / "graph.wiki.db", embedding_dim=8)
    try:
        _seed_graph(d)
        yield d
    finally:
        d.close()


def _seed_graph(db: UnifiedWikiDB) -> None:
    nodes = [
        ("A", "a.py", "class", "Alpha"),
        ("B", "b.py", "class", "Beta"),
        ("C", "c.py", "class", "Gamma"),
        ("D", "d.py", "class", "Delta"),
        ("E", "e.py", "function", "Epsilon"),
        ("F", "f.py", "function", "Zeta"),
        ("G", "g.py", "class", "Isolated1"),
        ("H", "h.py", "class", "Isolated2"),
    ]
    for nid, rel_path, stype, name in nodes:
        db.upsert_node(
            nid,
            rel_path=rel_path,
            symbol_name=name,
            symbol_type=stype,
            language="python",
        )
    # Directed edges from the seed topology; the shortest_path query
    # treats them as undirected so the original direction doesn't
    # matter for traversal but does matter for edge metadata.
    edges = [
        ("A", "B", "calls", "EXTRACTED"),
        ("B", "C", "calls", "INFERRED"),
        ("C", "D", "calls", "EXTRACTED"),
        ("B", "E", "imports", "EXTRACTED"),
        ("E", "F", "calls", "EXTRACTED"),
        ("G", "H", "calls", "EXTRACTED"),
    ]
    for src, tgt, rel_type, conf in edges:
        db.upsert_edge(src, tgt, rel_type, confidence=conf)
    db.conn.commit()


# ---------------------------------------------------------------------------
# Happy-path traversal
# ---------------------------------------------------------------------------


def test_returns_direct_neighbor_path(db):
    result = db.shortest_path("Alpha", "Beta")
    assert result["length"] == 1
    assert [n["node_id"] for n in result["path"]] == ["A", "B"]
    assert len(result["edges"]) == 1
    assert result["edges"][0]["rel_type"] == "calls"
    assert result["edges"][0]["confidence"] == "EXTRACTED"


def test_returns_multi_hop_path(db):
    # Alpha → Beta → Gamma → Delta (3 hops, all forward).
    result = db.shortest_path("Alpha", "Delta")
    assert result["length"] == 3
    assert [n["node_id"] for n in result["path"]] == ["A", "B", "C", "D"]
    # Edge metadata should reflect each forward step.
    assert [e["rel_type"] for e in result["edges"]] == ["calls", "calls", "calls"]


def test_treats_edges_as_undirected(db):
    # Reverse direction — Delta → Gamma → Beta → Alpha should also
    # resolve (3 hops walking edges backwards).
    result = db.shortest_path("Delta", "Alpha")
    assert result["length"] == 3
    assert [n["node_id"] for n in result["path"]] == ["D", "C", "B", "A"]


def test_takes_shorter_branch_when_both_exist(db):
    # Alpha → Beta → Epsilon → Zeta (3 hops via the E-branch).
    result = db.shortest_path("Alpha", "Zeta")
    assert result["length"] == 3
    assert [n["node_id"] for n in result["path"]] == ["A", "B", "E", "F"]


def test_resolves_by_rel_path_when_symbol_name_misses(db):
    # No symbol named "a.py" but rel_path matches the A node.
    result = db.shortest_path("a.py", "Beta")
    assert result["length"] == 1
    assert result["source"]["node_id"] == "A"


def test_source_equals_target_returns_zero_length(db):
    """Same-node is a success case (length 0 is the signal — no `reason`
    is set, so callers don't have to special-case a hybrid contract)."""
    result = db.shortest_path("Alpha", "Alpha")
    assert result["length"] == 0
    assert "reason" not in result  # contract: success path has no reason key
    assert result["edges"] == []
    assert [n["node_id"] for n in result["path"]] == ["A"]


def test_ambiguous_source_label_surfaces_candidate_count(db):
    """When a label resolves to multiple nodes, the response carries
    ``source_candidates`` / ``target_candidates`` so callers can detect
    the ambiguity instead of silently using the lowest-node_id match."""
    # Add a second node with the same symbol_name as Alpha (A) in a
    # different file. _resolve_label should still pick the lowest
    # node_id but report a count of 2.
    db.upsert_node(
        "A2",
        rel_path="dup_alpha.py",
        symbol_name="Alpha",
        symbol_type="class",
        language="python",
    )
    db.conn.commit()

    result = db.shortest_path("Alpha", "Beta")
    # First-match deterministic: node_id "A" comes before "A2".
    assert result["source"]["node_id"] == "A"
    assert result["source_candidates"] == 2
    assert result["target_candidates"] == 1


# ---------------------------------------------------------------------------
# Failure modes
# ---------------------------------------------------------------------------


def test_unknown_source_returns_reason(db):
    result = db.shortest_path("DoesNotExist", "Beta")
    assert result == {"path": None, "reason": "source_not_found"}


def test_unknown_target_returns_reason(db):
    result = db.shortest_path("Alpha", "DoesNotExist")
    assert result == {"path": None, "reason": "target_not_found"}


def test_no_path_within_max_depth_when_components_disjoint(db):
    # Alpha is in the main component; Isolated1 (G) is not connected.
    result = db.shortest_path("Alpha", "Isolated1")
    assert result == {"path": None, "reason": "no_path_within_max_depth"}


def test_max_depth_cutoff_prunes_long_paths(db):
    # Alpha → Beta → Gamma → Delta is 3 hops; max_depth=2 should
    # cut it off and return no path.
    result = db.shortest_path("Alpha", "Delta", max_depth=2)
    assert result == {"path": None, "reason": "no_path_within_max_depth"}


def test_invalid_max_depth_zero_returns_reason(db):
    result = db.shortest_path("Alpha", "Beta", max_depth=0)
    assert result == {"path": None, "reason": "invalid_max_depth"}


# ---------------------------------------------------------------------------
# Cycle resilience
# ---------------------------------------------------------------------------


def test_cycle_does_not_cause_infinite_loop(db):
    # Add an extra edge B → A so the graph has a 2-cycle. The cycle
    # guard in the CTE (path-membership check) should still terminate
    # and return the correct shortest path.
    db.upsert_edge("B", "A", "imports", confidence="EXTRACTED")
    db.conn.commit()
    result = db.shortest_path("Alpha", "Delta")
    assert result["length"] == 3
    assert [n["node_id"] for n in result["path"]] == ["A", "B", "C", "D"]


def test_prefers_extracted_edge_when_multiple_exist_between_hops(db):
    # The B → C edge is INFERRED; add an EXTRACTED alternative and
    # confirm the edge metadata picks EXTRACTED.
    db.upsert_edge("B", "C", "inheritance", confidence="EXTRACTED")
    db.conn.commit()
    result = db.shortest_path("Beta", "Gamma")
    assert result["length"] == 1
    assert result["edges"][0]["confidence"] == "EXTRACTED"
    assert result["edges"][0]["rel_type"] == "inheritance"


# ---------------------------------------------------------------------------
# Perf-regression guard — the original SQL-recursive-CTE draft hung at
# this size (>8s on a 7×7 grid at default max_depth). The current
# Python-layered-BFS impl with a visited set is O(V+E); this test
# pins the runtime so a regression to the old algorithm would be
# caught immediately rather than only in production.
# ---------------------------------------------------------------------------


def test_dense_grid_bfs_returns_quickly(tmp_path):
    """7×7 grid, no path between opposite corners — worst-case search
    (must exhaust the reachable component before giving up). Asserts
    the BFS completes well under one second; the old CTE timed out
    at >8s here.
    """
    import time

    db = UnifiedWikiDB(tmp_path / "grid.wiki.db", embedding_dim=8)
    try:
        # Build a 7×7 connected grid (rows × cols).
        for r in range(7):
            for c in range(7):
                nid = f"n_{r}_{c}"
                db.upsert_node(
                    nid,
                    rel_path=f"r{r}/c{c}.py",
                    symbol_name=nid,
                    symbol_type="class",
                    language="python",
                )
        for r in range(7):
            for c in range(7):
                if c < 6:
                    db.upsert_edge(f"n_{r}_{c}", f"n_{r}_{c+1}", "calls")
                if r < 6:
                    db.upsert_edge(f"n_{r}_{c}", f"n_{r+1}_{c}", "calls")
        # Add a disconnected node so opposite-corner traversal still
        # has to exhaust the connected component when target is the
        # disconnected node.
        db.upsert_node(
            "n_oof",
            rel_path="oof.py",
            symbol_name="n_oof",
            symbol_type="class",
            language="python",
        )
        db.conn.commit()

        start = time.perf_counter()
        result = db.shortest_path("n_0_0", "n_oof", max_depth=25)
        elapsed = time.perf_counter() - start

        assert result == {"path": None, "reason": "no_path_within_max_depth"}
        assert elapsed < 1.0, f"shortest_path on 7x7 grid took {elapsed:.2f}s"
    finally:
        db.close()
