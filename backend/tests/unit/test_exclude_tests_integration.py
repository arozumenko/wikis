"""Integration smoke test: cluster pipeline with exclude_tests=True.

Verifies the full path: graph → unified DB → Phase 3 clustering
preserves the test-exclusion flag without error.
Uses the actual ``run_phase3`` API and ``FeatureFlags``.
"""

import networkx as nx
import pytest
from unittest.mock import patch

from app.core.cluster_constants import is_test_path
from app.core.feature_flags import FeatureFlags
from app.core.graph_clustering import run_phase3
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
    G.add_edge("auth_login", "auth_mod", relationship_type="calls", weight=1.0)
    G.add_edge("api_route", "auth_login", relationship_type="calls", weight=1.0)
    G.add_edge("api_route", "user_model", relationship_type="uses", weight=1.0)
    G.add_edge("user_model", "db_conn", relationship_type="calls", weight=1.0)
    G.add_edge("test_auth", "auth_login", relationship_type="calls", weight=1.0)
    G.add_edge("test_user", "user_model", relationship_type="calls", weight=1.0)
    G.add_edge("mock_db", "db_conn", relationship_type="calls", weight=1.0)
    return G


class TestExcludeTestsIntegration:
    """End-to-end: clustering → unified DB, test nodes excluded."""

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

    @patch("app.core.feature_flags.get_feature_flags")
    def test_run_phase3_excludes_test_nodes(self, mock_flags, tmp_path):
        """run_phase3 with exclude_tests should not cluster test nodes."""
        flags = FeatureFlags(exclude_tests=True)
        mock_flags.return_value = flags

        G = _build_mixed_graph()
        db_path = tmp_path / "phase3.wiki.db"
        db = UnifiedWikiDB(db_path, embedding_dim=8)
        db.from_networkx(G)

        result = run_phase3(db, G, feature_flags=flags)

        assert "macro" in result
        assert "micro" in result
        assert result["macro"]["cluster_count"] > 0
        db.close()

    @patch("app.core.feature_flags.get_feature_flags")
    def test_run_phase3_includes_test_nodes_when_disabled(self, mock_flags, tmp_path):
        """run_phase3 without exclude_tests clusters everything."""
        flags = FeatureFlags(exclude_tests=False)
        mock_flags.return_value = flags

        G = _build_mixed_graph()
        db_path = tmp_path / "phase3_all.wiki.db"
        db = UnifiedWikiDB(db_path, embedding_dim=8)
        db.from_networkx(G)

        result = run_phase3(db, G, feature_flags=flags)

        assert result["macro"]["cluster_count"] > 0
        db.close()

    def test_full_pipeline_no_errors(self, tmp_path):
        """Run clustering + DB + topology enrichment without errors."""
        G = _build_mixed_graph()

        # 1. Populate unified DB
        db_path = tmp_path / "pipeline.wiki.db"
        db = UnifiedWikiDB(db_path, embedding_dim=8)
        db.from_networkx(G)
        assert db.node_count() == G.number_of_nodes()

        # 2. Run topology enrichment
        from app.core.graph_topology import run_phase2

        stats = run_phase2(db=db, G=G)
        assert "orphan_resolution" in stats
        assert "hubs" in stats
        assert "persisted_edges" in stats

        # 3. Cluster with test exclusion
        flags = FeatureFlags(exclude_tests=True)
        with patch("app.core.feature_flags.get_feature_flags", return_value=flags):
            result = run_phase3(db, G, feature_flags=flags)

        assert result["macro"]["cluster_count"] > 0
        assert result["micro"]["total_pages"] > 0

        db.close()
