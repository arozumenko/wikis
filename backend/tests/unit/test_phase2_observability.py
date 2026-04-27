"""
Phase 1 (graph-quality roadmap) — Phase 2 observability.

Builds a tiny graph, runs ``run_phase2``, and asserts that
``phase2_stats_v2`` is captured in ``repo_meta`` with the four
``components_*`` snapshots and per-class / per-provenance edge
breakdowns.
"""

from __future__ import annotations

import networkx as nx
import pytest

from app.core.graph_topology import run_phase2
from app.core.graph_topology_diagnostics import (
    Phase2Stats,
    explain_connectivity,
)
from app.core.unified_db import UnifiedWikiDB


def _seed() -> nx.MultiDiGraph:
    G = nx.MultiDiGraph()
    G.add_node(
        "AuthService",
        symbol_name="AuthService",
        symbol_type="class",
        rel_path="src/auth.py",
        start_line=1,
        end_line=30,
        language="python",
        source_text="class AuthService: pass",
    )
    G.add_node(
        "login_user",
        symbol_name="login_user",
        symbol_type="function",
        rel_path="src/auth.py",
        start_line=32,
        end_line=45,
        language="python",
        source_text="def login_user(): return True",
    )
    G.add_node(
        "Helper",
        symbol_name="Helper",
        symbol_type="class",
        rel_path="src/utils/helper.py",
        start_line=1,
        end_line=10,
        language="python",
        source_text="class Helper: pass",
    )
    G.add_node(
        "README",
        symbol_name="README",
        symbol_type="module_doc",
        rel_path="README.md",
        start_line=0,
        end_line=10,
        language="markdown",
        source_text="# Project\nUses `AuthService` for login.",
    )
    G.add_edge(
        "AuthService", "login_user",
        key=0, relationship_type="contains", edge_class="structural",
    )
    return G


@pytest.fixture()
def db(tmp_path):
    d = UnifiedWikiDB(str(tmp_path / "phase2.db"), embedding_dim=8)
    d.from_networkx(_seed())
    d.conn.commit()
    yield d
    d.close()


def test_explain_connectivity_counts_edge_classes(db):
    G = db.to_networkx()
    stats = explain_connectivity(db, G)
    assert isinstance(stats, Phase2Stats)
    assert sum(stats.edges_by_class.values()) == G.number_of_edges()
    assert "structural" in stats.edges_by_class


def test_run_phase2_persists_stats_v2(db):
    G = db.to_networkx()
    run_phase2(db, G)

    stats_v2 = db.get_meta("phase2_stats_v2")
    assert stats_v2 is not None, "phase2_stats_v2 should be persisted"

    expected = {
        "components_before_orphan",
        "components_after_orphan",
        "components_after_doc",
        "components_after_bridge",
        "edges_by_class",
        "edges_by_provenance",
        "orphan_resolution",
        "doc_edges",
        "bridging",
        "weighting",
        "hubs",
    }
    assert expected.issubset(stats_v2.keys()), (
        f"missing keys: {expected - set(stats_v2.keys())}"
    )

    assert isinstance(stats_v2["components_before_orphan"], int)
    assert stats_v2["components_after_bridge"] <= stats_v2["components_before_orphan"]
    assert sum(stats_v2["edges_by_class"].values()) > 0
