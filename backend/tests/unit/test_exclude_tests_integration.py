"""Integration smoke test: cluster pipeline with exclude_tests=True.

Verifies the full path: graph → clustering → unified DB → enrichment
preserves the test-exclusion flag without error.
"""

import networkx as nx
import pytest

from app.core.cluster_constants import is_test_path
from app.core.graph_clustering import run_clustering
from app.core.unified_db import UnifiedWikiDB


def _build_mixed_graph():
    """Build a realistic graph with production + test nodes."""
    G = nx.MultiDiGraph()
    prod_files = [
        ("auth_mod", "src/auth/module.py", "class"),
        ("auth_login", "src/auth/login.py", "function"),
        ("user_model", "src/models/user.py", "class"),
        ("db_conn", "src/db/connection.py", "function"),
        ("api_route", "src/api/routes.py", "function"),
        ("config", "src/config.py", "constant"),
    ]
    test_files = [
        ("test_auth", "tests/test_auth.py", "function"),
        ("test_user", "tests/test_user.py", "function"),
        ("mock_db", "mocks/mock_db.py", "function"),
        ("conftest", "tests/conftest.py", "function"),
    ]
    for nid, path, stype in prod_files + test_files:
        G.add_node(
            nid,
            rel_path=path,
            file_name=path.rsplit("/", 1)[-1],
            language="python",
            symbol_name=nid,
            symbol_type=stype,
            source_text=f"def {nid}(): pass",
        )
    # Add some edges
    G.add_edge("auth_login", "auth_mod", relationship_type="calls")
    G.add_edge("api_route", "auth_login", relationship_type="calls")
    G.add_edge("api_route", "user_model", relationship_type="uses")
    G.add_edge("user_model", "db_conn", relationship_type="calls")
    G.add_edge("test_auth", "auth_login", relationship_type="calls")
    G.add_edge("test_user", "user_model", relationship_type="calls")
    G.add_edge("mock_db", "db_conn", relationship_type="calls")
    return G


class TestExcludeTestsIntegration:
    """End-to-end: clustering → unified DB, test nodes excluded."""

    def test_clustering_excludes_test_nodes(self):
        G = _build_mixed_graph()
        result = run_clustering(G, exclude_tests=True)

        # All assigned node_ids should be production paths
        all_node_ids = set()
        for _sec_id, sec_data in result.sections.items():
            for _page_id, node_ids in sec_data["pages"].items():
                all_node_ids.update(node_ids)

        for nid in all_node_ids:
            path = G.nodes[nid].get("rel_path", "")
            assert not is_test_path(path), f"Test node {nid} ({path}) leaked through"

    def test_clustering_includes_test_nodes_when_disabled(self):
        G = _build_mixed_graph()
        result = run_clustering(G, exclude_tests=False)

        all_node_ids = set()
        for _sec_id, sec_data in result.sections.items():
            for _page_id, node_ids in sec_data["pages"].items():
                all_node_ids.update(node_ids)

        # At least one test node should be present
        test_present = any(
            is_test_path(G.nodes[nid].get("rel_path", ""))
            for nid in all_node_ids
        )
        assert test_present, "Expected test nodes when exclude_tests=False"

    def test_unified_db_marks_test_nodes(self, tmp_path):
        G = _build_mixed_graph()
        db_path = tmp_path / "integration.wiki.db"
        db = UnifiedWikiDB(db_path, embedding_dim=8)
        db.from_networkx(G)

        # Verify is_test flags
        for nid in G.nodes:
            node = db.get_node(nid)
            path = G.nodes[nid].get("rel_path", "")
            expected = 1 if is_test_path(path) else 0
            assert node["is_test"] == expected, (
                f"Node {nid} ({path}): is_test={node['is_test']}, expected={expected}"
            )
        db.close()

    def test_full_pipeline_no_errors(self, tmp_path):
        """Run clustering + DB + topology enrichment without errors."""
        G = _build_mixed_graph()

        # 1. Cluster with test exclusion
        result = run_clustering(G, exclude_tests=True)
        assert len(result.sections) > 0

        # 2. Populate unified DB
        db_path = tmp_path / "pipeline.wiki.db"
        db = UnifiedWikiDB(db_path, embedding_dim=8)
        db.from_networkx(G)
        assert db.node_count() == G.number_of_nodes()

        # 3. Run topology enrichment
        from app.core.graph_topology import run_phase2

        stats = run_phase2(db=db, G=G)
        assert "orphan_resolution" in stats
        assert "hubs" in stats
        assert "persisted_edges" in stats

        db.close()
