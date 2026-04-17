"""Tests for graph_topology.py — Phase 2 Graph Topology Enrichment."""

from __future__ import annotations

import math
from typing import Any
from unittest.mock import MagicMock, patch

import networkx as nx
import pytest

from app.core.graph_topology import (
    SYNTHETIC_WEIGHT_FLOOR,
    _build_name_index,
    _build_path_index,
    _expanding_prefixes,
    _extract_hyperlink_edges,
    _extract_proximity_edges,
    _is_doc_node,
    _normalize_path,
    apply_edge_weights,
    bridge_disconnected_components,
    detect_hubs,
    find_orphans,
    flag_hubs_in_db,
    inject_doc_edges,
    persist_weights_to_db,
    resolve_orphans,
    run_phase2,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_code_node(
    rel_path: str = "src/foo.py",
    symbol_name: str = "Foo",
    kind: str = "class",
    **extra: Any,
) -> dict[str, Any]:
    return {
        "rel_path": rel_path,
        "symbol_name": symbol_name,
        "symbol": {"name": symbol_name, "kind": kind},
        "symbol_type": kind,
        "kind": kind,
        **extra,
    }


def _make_doc_node(
    rel_path: str = "docs/README.md",
    content: str = "",
    **extra: Any,
) -> dict[str, Any]:
    return {
        "rel_path": rel_path,
        "symbol_type": "markdown_document",
        "kind": "markdown_document",
        "is_doc": 1,
        "source_text": content,
        "content": content,
        "location": {"rel_path": rel_path},
        **extra,
    }


def _mock_db() -> MagicMock:
    """Create a minimal mock DB with the methods graph_topology.py calls."""
    db = MagicMock()
    db.vec_available = False
    db.search_fts5.return_value = []
    db.search_vec.return_value = []
    db.get_node.return_value = None
    db.edge_count.return_value = 0
    db.conn = MagicMock()
    return db


# ====================================================================
# TestModuleLoadable
# ====================================================================


class TestModuleLoadable:
    """Step 1 — verify import and run_phase2 on empty graph."""

    def test_import_succeeds(self):
        from app.core.graph_topology import run_phase2  # noqa: F811

        assert callable(run_phase2)

    def test_run_phase2_empty_graph(self):
        G = nx.MultiDiGraph()
        db = _mock_db()
        db.edge_count.return_value = 0
        stats = run_phase2(db, G)
        assert isinstance(stats, dict)
        assert stats["orphan_resolution"]["orphans_found"] == 0

    def test_run_phase2_none_db(self):
        """run_phase2 works when db is None (in-memory only)."""
        G = nx.MultiDiGraph()
        G.add_node("a", **_make_code_node("a.py", "A"))
        stats = run_phase2(None, G)
        assert isinstance(stats, dict)


# ====================================================================
# TestOrphanDetection
# ====================================================================


class TestOrphanDetection:
    """Step 2 — find_orphans."""

    def test_mixed_connected_and_orphan(self):
        G = nx.MultiDiGraph()
        G.add_node("a", **_make_code_node("a.py", "A"))
        G.add_node("b", **_make_code_node("b.py", "B"))
        G.add_node("c", **_make_code_node("c.py", "C"))
        G.add_node("orphan1", **_make_code_node("o1.py", "O1"))
        G.add_node("orphan2", **_make_code_node("o2.py", "O2"))
        G.add_edge("a", "b", relationship_type="calls")
        G.add_edge("b", "c", relationship_type="calls")

        orphans = find_orphans(G)
        assert set(orphans) == {"orphan1", "orphan2"}

    def test_all_connected(self):
        G = nx.MultiDiGraph()
        G.add_node("a")
        G.add_node("b")
        G.add_edge("a", "b")
        assert find_orphans(G) == []

    def test_outgoing_only_not_orphan(self):
        G = nx.MultiDiGraph()
        G.add_node("a")
        G.add_node("b")
        G.add_edge("a", "b")
        # 'a' has out-degree > 0 → NOT orphan
        orphans = find_orphans(G)
        assert "a" not in orphans

    def test_empty_graph(self):
        assert find_orphans(nx.MultiDiGraph()) == []


# ====================================================================
# TestFTS5Resolution
# ====================================================================


class TestFTS5Resolution:
    """Step 3 — FTS5 lexical orphan resolution."""

    def test_fts5_resolves_orphan(self):
        G = nx.MultiDiGraph()
        G.add_node("orphan", **_make_code_node("src/a.py", "UserValidator"))
        G.add_node("target", **_make_code_node("src/b.py", "UserService"))
        G.add_edge("target", "target")  # self-edge to make it non-orphan degree > 0

        db = _mock_db()
        db.search_fts5.return_value = [{"node_id": "target"}]

        stats = resolve_orphans(db, G)
        assert stats["lexical"] == 1
        assert G.has_edge("orphan", "target")

    def test_fts5_excludes_self_hit(self):
        G = nx.MultiDiGraph()
        G.add_node("orphan", **_make_code_node("src/a.py", "UserValidator"))

        db = _mock_db()
        db.search_fts5.return_value = [{"node_id": "orphan"}]

        stats = resolve_orphans(db, G)
        assert stats["lexical"] == 0

    def test_fts5_max_lexical_edges(self):
        G = nx.MultiDiGraph()
        G.add_node("orphan", **_make_code_node("src/a.py", "Parser"))
        for i in range(5):
            nid = f"target_{i}"
            G.add_node(nid, **_make_code_node(f"src/{nid}.py", f"T{i}"))

        db = _mock_db()
        db.search_fts5.return_value = [{"node_id": f"target_{i}"} for i in range(5)]

        stats = resolve_orphans(db, G, max_lexical_edges=2)
        # Should add at most 2 edges from orphan
        edges_from_orphan = [
            (u, v) for u, v, _ in G.edges(data=True) if u == "orphan"
        ]
        assert len(edges_from_orphan) == 2

    def test_fts5_skips_short_names(self):
        G = nx.MultiDiGraph()
        G.add_node("orphan", **_make_code_node("a.py", "X"))  # 1-char name

        db = _mock_db()
        db.search_fts5.return_value = [{"node_id": "whatever"}]

        stats = resolve_orphans(db, G)
        assert stats["lexical"] == 0


# ====================================================================
# TestSemanticResolution
# ====================================================================


class TestSemanticResolution:
    """Step 4 — semantic vector orphan resolution."""

    def test_semantic_resolves_orphan(self):
        G = nx.MultiDiGraph()
        G.add_node(
            "orphan",
            **_make_code_node("src/auth/token.py", "TokenManager"),
            source_text="Manages JWT tokens for authentication",
        )
        G.add_node("target", **_make_code_node("src/auth/jwt.py", "JWTService"))
        G.add_edge("target", "target")

        db = _mock_db()
        db.vec_available = True
        db.search_fts5.return_value = []
        db.search_vec.return_value = [{"node_id": "target", "vec_distance": 0.10}]

        embedding_fn = lambda text: [0.1, 0.2, 0.3]  # noqa: E731
        stats = resolve_orphans(db, G, embedding_fn=embedding_fn)
        assert stats["semantic"] == 1
        assert G.has_edge("orphan", "target")

    def test_semantic_skipped_above_threshold(self):
        G = nx.MultiDiGraph()
        G.add_node(
            "orphan",
            **_make_code_node("src/a.py", "Widget"),
            source_text="A UI widget component",
        )

        db = _mock_db()
        db.vec_available = True
        db.search_fts5.return_value = []
        db.search_vec.return_value = [{"node_id": "far", "vec_distance": 0.20}]

        embedding_fn = lambda text: [0.1]  # noqa: E731
        stats = resolve_orphans(
            db, G, embedding_fn=embedding_fn, vec_distance_threshold=0.15
        )
        assert stats["semantic"] == 0

    def test_semantic_expanding_prefixes(self):
        prefixes = _expanding_prefixes("src/auth/token.py")
        assert prefixes == ["src/auth", "src", ""]

    def test_semantic_root_file_prefixes(self):
        prefixes = _expanding_prefixes("README.md")
        assert prefixes == [""]

    def test_semantic_skipped_when_no_embedding_fn(self):
        G = nx.MultiDiGraph()
        G.add_node(
            "orphan",
            **_make_code_node("a.py", "Orphan"),
            source_text="Some long text here",
        )

        db = _mock_db()
        db.vec_available = True
        db.search_fts5.return_value = []

        stats = resolve_orphans(db, G, embedding_fn=None)
        assert stats["semantic"] == 0


# ====================================================================
# TestDirectoryResolution
# ====================================================================


class TestDirectoryResolution:
    """Step 5 — directory fallback orphan resolution."""

    def test_same_dir_resolution(self):
        G = nx.MultiDiGraph()
        G.add_node("orphan", **_make_code_node("src/utils/helpers.py", "Helper"))
        G.add_node("core", **_make_code_node("src/utils/core.py", "Core"))
        G.add_edge("core", "core")  # non-orphan

        db = _mock_db()
        db.search_fts5.return_value = []
        stats = resolve_orphans(db, G)
        assert stats["directory"] == 1
        assert G.has_edge("orphan", "core")

    def test_walk_up_to_parent(self):
        G = nx.MultiDiGraph()
        G.add_node("orphan", **_make_code_node("src/utils/sub/deep.py", "Deep"))
        G.add_node("root_node", **_make_code_node("src/utils/main.py", "Main"))
        G.add_edge("root_node", "root_node")

        db = _mock_db()
        db.search_fts5.return_value = []
        stats = resolve_orphans(db, G)
        assert stats["directory"] >= 1

    def test_no_match_stays_unresolved(self):
        G = nx.MultiDiGraph()
        G.add_node("orphan", **_make_code_node("isolated/alone.py", "Alone"))
        # No other nodes at all

        db = _mock_db()
        db.search_fts5.return_value = []
        stats = resolve_orphans(db, G)
        assert stats["directory"] == 0


# ====================================================================
# TestDocHyperlinkEdges
# ====================================================================


class TestDocHyperlinkEdges:
    """Step 6 — document edge injection via hyperlinks & backticks."""

    def test_hyperlink_to_code_file(self):
        G = nx.MultiDiGraph()
        G.add_node(
            "doc",
            **_make_doc_node(
                "docs/guide.md",
                content="See [Service](../src/auth/service.py) for details.",
            ),
        )
        G.add_node("code", **_make_code_node("src/auth/service.py", "Service"))

        stats = inject_doc_edges(None, G)
        assert stats["hyperlink_edges"] == 1
        assert G.has_edge("doc", "code")

    def test_backtick_mention(self):
        G = nx.MultiDiGraph()
        G.add_node(
            "doc",
            **_make_doc_node("docs/api.md", content="Uses `TokenValidator` for auth."),
        )
        G.add_node(
            "code", **_make_code_node("src/auth/validator.py", "TokenValidator")
        )

        stats = inject_doc_edges(None, G)
        assert stats["hyperlink_edges"] == 1
        assert G.has_edge("doc", "code")

    def test_backtick_cap_per_mention(self):
        G = nx.MultiDiGraph()
        G.add_node(
            "doc", **_make_doc_node("docs/api.md", content="Uses `Widget` often.")
        )
        for i in range(5):
            G.add_node(f"w{i}", **_make_code_node(f"src/w{i}.py", "Widget"))

        stats = inject_doc_edges(None, G)
        assert stats["hyperlink_edges"] == 2  # Capped at 2

    def test_external_url_no_edge(self):
        G = nx.MultiDiGraph()
        G.add_node(
            "doc",
            **_make_doc_node(
                "docs/guide.md", content="See [example](https://example.com)"
            ),
        )
        stats = inject_doc_edges(None, G)
        assert stats["hyperlink_edges"] == 0

    def test_mailto_no_edge(self):
        G = nx.MultiDiGraph()
        G.add_node(
            "doc",
            **_make_doc_node(
                "docs/guide.md", content="Email [us](mailto:user@example.com)"
            ),
        )
        stats = inject_doc_edges(None, G)
        assert stats["hyperlink_edges"] == 0

    def test_anchor_only_no_edge(self):
        G = nx.MultiDiGraph()
        G.add_node(
            "doc",
            **_make_doc_node(
                "docs/guide.md", content="See [section](#overview)"
            ),
        )
        stats = inject_doc_edges(None, G)
        assert stats["hyperlink_edges"] == 0

    def test_dedup_same_pair(self):
        G = nx.MultiDiGraph()
        G.add_node(
            "doc",
            **_make_doc_node(
                "docs/guide.md",
                content=(
                    "See `AuthService` and also `AuthService` again."
                ),
            ),
        )
        G.add_node("code", **_make_code_node("src/auth.py", "AuthService"))

        stats = inject_doc_edges(None, G)
        # Should only add 1 edge despite 2 mentions
        assert stats["hyperlink_edges"] == 1


# ====================================================================
# TestDocProximityEdges
# ====================================================================


class TestDocProximityEdges:
    """Step 7 — proximity-based doc edges."""

    def test_docs_prefix_proximity(self):
        G = nx.MultiDiGraph()
        G.add_node("doc", **_make_doc_node("docs/auth/README.md"))
        G.add_node("code", **_make_code_node("auth/service.py", "Service"))

        stats = inject_doc_edges(None, G)
        assert stats["proximity_edges"] == 1
        assert G.has_edge("doc", "code")

    def test_no_prefix_no_proximity(self):
        G = nx.MultiDiGraph()
        G.add_node("doc", **_make_doc_node("src/README.md"))
        G.add_node("code", **_make_code_node("src/service.py", "Service"))

        stats = inject_doc_edges(None, G)
        # Even without a doc prefix to strip, a doc in src/ and code in
        # src/ share the same directory — matches deepwiki which keys by
        # directory equality (topic_dir == code_dir) after prefix strip.
        assert stats["proximity_edges"] == 1

    def test_documentation_prefix(self):
        G = nx.MultiDiGraph()
        G.add_node("doc", **_make_doc_node("documentation/auth/guide.md"))
        G.add_node("code", **_make_code_node("auth/handler.py", "Handler"))

        stats = inject_doc_edges(None, G)
        assert stats["proximity_edges"] == 1

    def test_proximity_cap_per_doc(self):
        G = nx.MultiDiGraph()
        G.add_node("doc", **_make_doc_node("docs/auth/guide.md"))
        for i in range(10):
            G.add_node(
                f"code_{i}", **_make_code_node(f"auth/file_{i}.py", f"F{i}")
            )

        stats = inject_doc_edges(None, G)
        assert stats["proximity_edges"] <= 5


# ====================================================================
# TestComponentBridging
# ====================================================================


class TestComponentBridging:
    """Step 8 — connect disconnected weak components."""

    def test_three_components_bridged(self):
        G = nx.MultiDiGraph()
        # Component 1
        G.add_node("a1", rel_path="src/a.py")
        G.add_node("a2", rel_path="src/b.py")
        G.add_edge("a1", "a2")
        # Component 2
        G.add_node("b1", rel_path="lib/x.py")
        G.add_node("b2", rel_path="lib/y.py")
        G.add_edge("b1", "b2")
        # Component 3
        G.add_node("c1", rel_path="utils/z.py")

        stats = bridge_disconnected_components(None, G)
        assert stats["components_before"] == 3
        assert stats["bridges_added"] == 4  # 2 bidirectional bridges
        # Should now be 1 component
        assert len(list(nx.weakly_connected_components(G))) == 1

    def test_bridge_edges_bidirectional(self):
        G = nx.MultiDiGraph()
        G.add_node("a", rel_path="src/a.py")
        G.add_node("b", rel_path="lib/b.py")

        stats = bridge_disconnected_components(None, G)
        assert stats["bridges_added"] == 2
        assert G.has_edge("a", "b") or G.has_edge("b", "a")

    def test_already_connected_no_bridges(self):
        G = nx.MultiDiGraph()
        G.add_node("a")
        G.add_node("b")
        G.add_edge("a", "b")

        stats = bridge_disconnected_components(None, G)
        assert stats["bridges_added"] == 0

    def test_root_level_files_use_root_key(self):
        G = nx.MultiDiGraph()
        G.add_node("a", rel_path="main.py")  # No '/' → <root>
        G.add_node("b", rel_path="app.py")
        # Separate component
        G.add_node("c", rel_path="main.py")
        G.add_edge("a", "b")

        stats = bridge_disconnected_components(None, G)
        # c is a separate component
        assert stats["components_before"] == 2


# ====================================================================
# TestEdgeWeighting
# ====================================================================


class TestEdgeWeighting:
    """Step 9 — inverse-in-degree edge weighting."""

    def test_weight_formula(self):
        G = nx.MultiDiGraph()
        G.add_node("a")
        G.add_node("b")
        # Add 10 structural edges to B
        for i in range(10):
            nid = f"s{i}"
            G.add_node(nid)
            G.add_edge(nid, "b", edge_class="structural")
        G.add_edge("a", "b", edge_class="structural")

        apply_edge_weights(G)
        # structural in-degree of B = 11, weight = 1/log(13)
        expected = 1.0 / math.log(13)
        # Check edge a→b
        edge_data = list(G.get_edge_data("a", "b").values())[0]
        assert abs(edge_data["weight"] - expected) < 1e-6

    def test_synthetic_floor(self):
        G = nx.MultiDiGraph()
        G.add_node("a")
        G.add_node("b")
        G.add_edge("a", "b", edge_class="lexical")  # synthetic

        apply_edge_weights(G)
        edge_data = list(G.get_edge_data("a", "b").values())[0]
        assert edge_data["weight"] >= SYNTHETIC_WEIGHT_FLOOR

    def test_synthetic_not_counted_in_structural(self):
        G = nx.MultiDiGraph()
        G.add_node("a")
        G.add_node("b")
        # 1 structural + 5 synthetic edges to B
        G.add_edge("a", "b", edge_class="structural")
        for i in range(5):
            G.add_node(f"s{i}")
            G.add_edge(f"s{i}", "b", edge_class="lexical")

        apply_edge_weights(G)
        # Structural in-degree of B == 1 (not 6)
        edge_data = list(G.get_edge_data("a", "b").values())[0]
        expected = 1.0 / math.log(1 + 2)
        assert abs(edge_data["weight"] - expected) < 1e-6


# ====================================================================
# TestHubDetection
# ====================================================================


class TestHubDetection:
    """Step 10 — hub detection via Z-score."""

    def test_clear_hub_detected(self):
        G = nx.MultiDiGraph()
        G.add_node("hub")
        for i in range(50):
            nid = f"n{i}"
            G.add_node(nid)
            G.add_edge(nid, "hub")

        hubs = detect_hubs(G, z_threshold=3.0)
        assert "hub" in hubs

    def test_small_graph_no_hubs(self):
        G = nx.MultiDiGraph()
        G.add_node("a")
        G.add_node("b")
        assert detect_hubs(G) == set()

    def test_uniform_degree_no_hubs(self):
        G = nx.MultiDiGraph()
        # Ring: all nodes have in-degree 1
        nodes = [f"n{i}" for i in range(10)]
        for nid in nodes:
            G.add_node(nid)
        for i in range(len(nodes)):
            G.add_edge(nodes[i], nodes[(i + 1) % len(nodes)])

        hubs = detect_hubs(G, z_threshold=3.0)
        assert len(hubs) == 0

    def test_flag_hubs_in_db(self):
        db = _mock_db()
        flag_hubs_in_db(db, {"hub1", "hub2"})
        assert db.set_hub.call_count == 2
        db.commit.assert_called_once()

    def test_flag_hubs_none_db(self):
        """flag_hubs_in_db with None db doesn't crash."""
        flag_hubs_in_db(None, {"hub1"})


# ====================================================================
# TestEdgePersistence
# ====================================================================


class TestEdgePersistence:
    """Step 11 — persist edges to DB."""

    def test_persist_writes_all_edges(self):
        G = nx.MultiDiGraph()
        for i in range(10):
            G.add_node(f"n{i}")
        for i in range(9):
            G.add_edge(f"n{i}", f"n{i+1}", relationship_type="calls", weight=0.5)

        db = _mock_db()
        db.edge_count.return_value = 9

        count = persist_weights_to_db(db, G)
        assert count == 9
        db.delete_all_edges.assert_called_once()

    def test_persist_none_db(self):
        G = nx.MultiDiGraph()
        G.add_node("a")
        G.add_node("b")
        G.add_edge("a", "b")
        assert persist_weights_to_db(None, G) == 0

    def test_persist_batch_size(self):
        """Verify batching works with many edges."""
        G = nx.MultiDiGraph()
        for i in range(100):
            G.add_node(f"n{i}")
        for i in range(99):
            G.add_edge(f"n{i}", f"n{i+1}", relationship_type="calls")

        db = _mock_db()
        db.edge_count.return_value = 99

        with patch("app.core.graph_topology.PERSIST_BATCH_SIZE", 30):
            count = persist_weights_to_db(db, G)
        assert count == 99
        # Should have called upsert_edges_batch multiple times
        assert db.upsert_edges_batch.call_count >= 3


# ====================================================================
# TestPhase2Orchestrator
# ====================================================================


class TestPhase2Orchestrator:
    """Step 12 — run_phase2 integration."""

    def test_realistic_graph(self):
        G = nx.MultiDiGraph()

        # Connected cluster
        for i in range(15):
            G.add_node(
                f"code_{i}", **_make_code_node(f"src/mod{i}.py", f"Mod{i}")
            )
        for i in range(14):
            G.add_edge(f"code_{i}", f"code_{i+1}", relationship_type="calls")

        # Orphans
        for i in range(5):
            G.add_node(
                f"orphan_{i}",
                **_make_code_node(f"src/orphan{i}.py", f"Orphan{i}"),
            )

        # Doc nodes
        G.add_node(
            "doc_readme",
            **_make_doc_node(
                "docs/src/guide.md",
                content="See `Mod0` for the main entry point.",
            ),
        )

        # Disconnected component
        G.add_node("island_a", **_make_code_node("lib/a.py", "LibA"))
        G.add_node("island_b", **_make_code_node("lib/b.py", "LibB"))
        G.add_edge("island_a", "island_b", relationship_type="calls")

        db = _mock_db()
        db.search_fts5.return_value = []
        db.edge_count.return_value = G.number_of_edges() + 20

        stats = run_phase2(db, G)

        assert "orphan_resolution" in stats
        assert "doc_edges" in stats
        assert "bridging" in stats
        assert "weighting" in stats
        assert "hubs" in stats
        assert "persisted_edges" in stats

    def test_graph_connected_after_phase2(self):
        G = nx.MultiDiGraph()

        # Two disconnected components
        G.add_node("a1", **_make_code_node("src/a.py", "A"))
        G.add_node("a2", **_make_code_node("src/b.py", "B"))
        G.add_edge("a1", "a2", relationship_type="calls")

        G.add_node("b1", **_make_code_node("lib/x.py", "X"))
        G.add_node("b2", **_make_code_node("lib/y.py", "Y"))
        G.add_edge("b1", "b2", relationship_type="calls")

        db = _mock_db()
        db.search_fts5.return_value = []
        db.edge_count.return_value = G.number_of_edges() + 5

        run_phase2(db, G)

        comps = list(nx.weakly_connected_components(G))
        assert len(comps) == 1

    def test_metadata_stored(self):
        G = nx.MultiDiGraph()
        G.add_node("a", **_make_code_node("a.py", "A"))

        db = _mock_db()
        db.edge_count.return_value = 0
        run_phase2(db, G)

        db.set_meta.assert_any_call("phase2_completed", True)


# ====================================================================
# Helper function tests
# ====================================================================


class TestNormalizePath:

    def test_relative_path(self):
        # source_dir is the DIRECTORY, not the file path
        assert _normalize_path("../src/foo.py", "docs") == "src/foo.py"

    def test_external_url(self):
        assert _normalize_path("https://example.com", "") == ""

    def test_mailto(self):
        assert _normalize_path("mailto:user@example.com", "") == ""

    def test_anchor_only(self):
        assert _normalize_path("#section", "") == ""

    def test_strips_anchor(self):
        result = _normalize_path("file.py#L10", "src")
        assert "#" not in result


class TestBuildPathIndex:

    def test_basic_index(self):
        G = nx.MultiDiGraph()
        G.add_node("n1", rel_path="src/foo.py")
        G.add_node("n2", location={"rel_path": "src/bar.py"})
        idx = _build_path_index(G)
        assert idx.get("src/foo.py") == "n1"
        assert idx.get("src/bar.py") == "n2"


class TestBuildNameIndex:

    def test_excludes_doc_nodes(self):
        G = nx.MultiDiGraph()
        G.add_node("code", **_make_code_node("a.py", "Foo"))
        G.add_node("doc", **_make_doc_node("README.md"))
        idx = _build_name_index(G)
        assert "Foo" in idx
        # Doc node should not be indexed by name
        assert all("doc" not in v for v in idx.values())


class TestIsDocNode:

    def test_is_doc_flag(self):
        G = nx.MultiDiGraph()
        G.add_node("d", is_doc=1)
        assert _is_doc_node(G, "d")

    def test_kind_markdown(self):
        G = nx.MultiDiGraph()
        G.add_node("d", kind="markdown_document")
        assert _is_doc_node(G, "d")

    def test_suffix_document(self):
        G = nx.MultiDiGraph()
        G.add_node("d", symbol_type="rst_document")
        assert _is_doc_node(G, "d")

    def test_code_node_not_doc(self):
        G = nx.MultiDiGraph()
        G.add_node("c", kind="class")
        assert not _is_doc_node(G, "c")
