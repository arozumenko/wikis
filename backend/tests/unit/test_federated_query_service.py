"""Phase 7 — FederatedQueryService tests."""

from __future__ import annotations

from app.core.code_graph.federated_query_service import FederatedQueryService
from app.core.storage.project_storage import InMemoryProjectStorage


class _Stub:
    """Minimal GraphQueryService stand-in for fan-out plumbing tests."""

    def __init__(self, nodes: dict | None = None) -> None:
        self._nodes = nodes or {}

        class _GraphView:
            class _Nodes:
                def get(_self, nid):
                    return None

            nodes = _Nodes()

        self.graph = _GraphView()
        self.storage = None

    def get_node(self, node_id):
        return self._nodes.get(node_id)


class TestCrossRepoEdges:
    def test_empty_when_no_storage(self):
        svc = FederatedQueryService({"w1": _Stub()}, project_id="p1")
        assert svc.cross_repo_edges("n1") == []

    def test_returns_filtered_edges(self):
        store = InMemoryProjectStorage()
        store.upsert_project_edge("p1", "n1", "n2", "cross_repo", 0.6)
        store.upsert_project_edge("p1", "n1", "n3", "cross_language", 0.4)
        store.upsert_project_edge("p1", "n9", "n2", "cross_repo", 0.5)

        svc = FederatedQueryService(
            {"w1": _Stub()}, project_id="p1", project_storage=store
        )
        rows = svc.cross_repo_edges("n1")
        assert len(rows) == 1
        assert rows[0]["target_node_id"] == "n2"
        assert rows[0]["edge_class"] == "cross_repo"

    def test_empty_for_unknown_node(self):
        store = InMemoryProjectStorage()
        store.upsert_project_edge("p1", "n1", "n2", "cross_repo", 0.5)
        svc = FederatedQueryService(
            {"w1": _Stub()}, project_id="p1", project_storage=store
        )
        assert svc.cross_repo_edges("missing") == []


class TestRelatedness:
    def test_zero_when_no_storage(self):
        svc = FederatedQueryService({"w1": _Stub()}, project_id="p1")
        assert svc.relatedness("a", "b") == 0.0

    def test_zero_for_self_pair(self):
        svc = FederatedQueryService({"w1": _Stub()}, project_id="p1")
        assert svc.relatedness("a", "a") == 0.0

    def test_returns_score_order_insensitive(self):
        store = InMemoryProjectStorage()
        store.upsert_repo_relatedness("p1", "wiki-a", "wiki-b", 0.42)
        svc = FederatedQueryService(
            {"w1": _Stub()}, project_id="p1", project_storage=store
        )
        assert svc.relatedness("wiki-a", "wiki-b") == 0.42
        assert svc.relatedness("wiki-b", "wiki-a") == 0.42

    def test_zero_for_unscored_pair(self):
        store = InMemoryProjectStorage()
        store.upsert_repo_relatedness("p1", "wiki-a", "wiki-b", 0.42)
        svc = FederatedQueryService(
            {"w1": _Stub()}, project_id="p1", project_storage=store
        )
        assert svc.relatedness("wiki-a", "wiki-c") == 0.0
