"""Phase 7 — FederatedQueryService tests."""

from __future__ import annotations

from app.core.code_graph.federated_query_service import FederatedQueryService
from app.core.storage.project_storage import InMemoryProjectStorage


class _Stub:
    """Minimal GraphQueryService stand-in for fan-out plumbing tests."""

    def __init__(self, nodes: dict | None = None, relationships: list | None = None) -> None:
        self._nodes = nodes or {}
        self._relationships = relationships or []

        class _GraphView:
            class _Nodes:
                def get(_self, nid):
                    return None

            nodes = _Nodes()

        self.graph = _GraphView()
        self.storage = None

    def get_node(self, node_id):
        return self._nodes.get(node_id)

    def get_relationships(self, node_id, direction="both", max_depth=2, max_results=50):
        return self._relationships[:max_results]

    def resolve_symbol(self, symbol_name, file_path="", language=""):
        if symbol_name in self._nodes:
            return symbol_name
        for node_id, data in self._nodes.items():
            if data.get("symbol_name") == symbol_name:
                return node_id
        return None


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


class TestNamespacedNodeLookup:
    def test_get_node_resolves_project_namespaced_id(self):
        svc = FederatedQueryService(
            {"wiki-b": _Stub({"n2": {"source_text": "target"}})},
            project_id="p1",
            project_storage=InMemoryProjectStorage(),
        )

        node = svc.get_node("wiki-b::n2")

        assert node["node_id"] == "n2"
        assert node["wiki_id"] == "wiki-b"
        assert node["source_text"] == "target"


class TestCrossRepoRelationships:
    def test_get_relationships_includes_persisted_outbound_project_edge(self):
        store = InMemoryProjectStorage()
        store.upsert_project_edge(
            "p1",
            "api-wiki::api.handler",
            "sdk-wiki::client.call",
            "cross_repo",
            0.42,
            provenance={
                "source_relationship_type": "cross_repo_api_surface",
                "surface": "GET /api/v1/models",
                "matcher": "api_surface:rest",
                "source_wiki_id": "api-wiki",
                "target_wiki_id": "sdk-wiki",
            },
        )
        svc = FederatedQueryService(
            {
                "api-wiki": _Stub({
                    "api.handler": {
                        "symbol_name": "ConfigurationAPI.models",
                        "symbol_type": "method",
                    }
                }),
                "sdk-wiki": _Stub({
                    "client.call": {
                        "symbol_name": "EliteAClient.models",
                        "symbol_type": "method",
                    }
                }),
            },
            project_id="p1",
            project_storage=store,
        )

        rels = svc.get_relationships("api-wiki::api.handler", direction="outgoing")

        assert len(rels) == 1
        rel = rels[0]
        assert rel.relationship_type == "cross_repo_api_surface"
        assert rel.source_name == "ConfigurationAPI.models"
        assert rel.target_name == "EliteAClient.models"
        assert rel.source_node_id == "api-wiki::api.handler"
        assert rel.target_node_id == "sdk-wiki::client.call"
        assert rel.source_wiki_id == "api-wiki"
        assert rel.target_wiki_id == "sdk-wiki"
        assert rel.weight == 0.42
        assert rel.provenance["surface"] == "GET /api/v1/models"

    def test_get_relationships_includes_persisted_incoming_project_edge(self):
        store = InMemoryProjectStorage()
        store.upsert_project_edge(
            "p1",
            "api-wiki::api.handler",
            "sdk-wiki::client.call",
            "cross_repo",
            0.9,
            provenance={"source_relationship_type": "cross_repo_api_surface"},
        )
        svc = FederatedQueryService(
            {
                "api-wiki": _Stub({"api.handler": {"symbol_name": "API", "symbol_type": "function"}}),
                "sdk-wiki": _Stub({"client.call": {"symbol_name": "Client", "symbol_type": "function"}}),
            },
            project_id="p1",
            project_storage=store,
        )

        rels = svc.get_relationships("sdk-wiki::client.call", direction="incoming")

        assert len(rels) == 1
        assert rels[0].source_name == "API"
        assert rels[0].target_name == "Client"
        assert rels[0].source_node_id == "api-wiki::api.handler"
        assert rels[0].target_node_id == "sdk-wiki::client.call"


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
