"""Phase 7 — FederatedRetrieverStack tests."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from typing import Any

import pytest

from app.core.code_graph.federated_query_service import FederatedQueryService
from app.core.federated_retriever import FederatedRetrieverStack
from app.core.storage.project_storage import InMemoryProjectStorage


@dataclass
class _Doc:
    """Tiny LangChain-Document stand-in (page_content + metadata)."""

    page_content: str
    metadata: dict = field(default_factory=dict)


class _StubStack:
    """WikiRetrieverStack stand-in returning canned hits."""

    def __init__(self, docs: list[_Doc]) -> None:
        self._docs = docs

    def search_repository(
        self,
        query: str,
        k: int = 15,
        apply_expansion: bool = True,
        min_confidence: str | None = None,
    ) -> list[_Doc]:
        return list(self._docs)


class _StubFQS:
    """FederatedQueryService stand-in."""

    def __init__(self, edges: dict[str, list[dict]], nodes: dict[str, dict]) -> None:
        self._edges = edges
        self._nodes = nodes

    def cross_repo_edges(self, node_id: str) -> list[dict]:
        return self._edges.get(node_id, [])

    def get_node(self, node_id: str) -> dict | None:
        return self._nodes.get(node_id)


class _NodeService:
    def __init__(self, nodes: dict[str, dict]) -> None:
        self._nodes = nodes
        self.graph = None
        self.storage = None

    def get_node(self, node_id: str) -> dict | None:
        return self._nodes.get(node_id)


@pytest.fixture
def base_stacks():
    return [
        ("wiki-a", _StubStack([
            _Doc("hello", {"node_id": "n1", "score": 0.9}),
        ])),
    ]


def _run(coro):
    return asyncio.run(coro)


class TestExpansion:
    def test_no_fqs_falls_back_to_parent(self, base_stacks):
        stack = FederatedRetrieverStack(base_stacks, federated_query_service=None)
        out = _run(stack.aretrieve("q", k=5))
        assert len(out) == 1 and out[0].metadata["node_id"] == "n1"

    def test_expands_along_cross_repo_edge(self, base_stacks):
        fqs = _StubFQS(
            edges={"n1": [{"target_node_id": "n2", "weight": 1.0}]},
            nodes={"n2": {"node_id": "n2", "wiki_id": "wiki-b", "content": "world"}},
        )
        stack = FederatedRetrieverStack(base_stacks, federated_query_service=fqs)
        out = _run(stack.aretrieve("q", k=5))
        node_ids = [d.metadata["node_id"] for d in out]
        assert "n1" in node_ids and "n2" in node_ids

    def test_expands_along_namespaced_cross_repo_edge(self, base_stacks):
        fqs = _StubFQS(
            edges={
                "wiki-a::n1": [
                    {"target_node_id": "wiki-b::n2", "weight": 1.0},
                ]
            },
            nodes={
                "wiki-b::n2": {"node_id": "n2", "wiki_id": "wiki-b", "content": "world"},
            },
        )
        stack = FederatedRetrieverStack(base_stacks, federated_query_service=fqs)
        out = _run(stack.aretrieve("q", k=5))
        node_ids = [d.metadata["node_id"] for d in out]
        assert "n1" in node_ids and "n2" in node_ids

    def test_dampens_expanded_score(self):
        # Two base hits so per-wiki min-max produces a real range.
        stacks = [
            ("wiki-a", _StubStack([
                _Doc("top",  {"node_id": "n1", "score": 0.9}),
                _Doc("low",  {"node_id": "n0", "score": 0.1}),
            ])),
        ]
        fqs = _StubFQS(
            edges={"n1": [{"target_node_id": "n2", "weight": 1.0}]},
            nodes={"n2": {"node_id": "n2", "wiki_id": "wiki-b", "content": "x"}},
        )
        stack = FederatedRetrieverStack(
            stacks, federated_query_service=fqs, dampening=0.5
        )
        out = _run(stack.aretrieve("q", k=5))
        by_id = {d.metadata["node_id"]: d for d in out}
        # Top base hit normalised to 1.0; expanded = 1.0 * 0.5 * 1.0 = 0.5.
        assert by_id["n1"].metadata["normalized_score"] == pytest.approx(1.0)
        assert by_id["n2"].metadata["normalized_score"] == pytest.approx(0.5)

    def test_skips_unknown_target(self, base_stacks):
        fqs = _StubFQS(
            edges={"n1": [{"target_node_id": "missing", "weight": 1.0}]},
            nodes={},
        )
        stack = FederatedRetrieverStack(base_stacks, federated_query_service=fqs)
        out = _run(stack.aretrieve("q", k=5))
        assert [d.metadata["node_id"] for d in out] == ["n1"]

    def test_dedupes_existing_target(self, base_stacks):
        # Base already returns n2 directly from wiki-b.
        stacks = base_stacks + [
            ("wiki-b", _StubStack([
                _Doc("world", {"node_id": "n2", "score": 0.5}),
            ])),
        ]
        fqs = _StubFQS(
            edges={"n1": [{"target_node_id": "n2", "weight": 1.0}]},
            nodes={"n2": {"node_id": "n2", "wiki_id": "wiki-b", "content": "x"}},
        )
        stack = FederatedRetrieverStack(stacks, federated_query_service=fqs)
        out = _run(stack.aretrieve("q", k=5))
        keys = sorted(
            f"{d.metadata.get('source_wiki_id', '')}::{d.metadata['node_id']}"
            for d in out
        )
        # n2 should appear exactly once (with the wiki-b base origin retained).
        assert keys.count("wiki-b::n2") == 1

    def test_expands_with_real_project_storage_and_preserves_provenance(self, base_stacks):
        store = InMemoryProjectStorage()
        store.upsert_project_edge(
            "project-1",
            "wiki-a::n1",
            "wiki-b::n2",
            "cross_repo",
            0.8,
            provenance={
                "source_relationship_type": "cross_repo_api_surface",
                "surface": "GET /api/v1/users",
                "matcher": "api_surface:rest",
                "level": "L1",
                "source_wiki_id": "wiki-a",
                "target_wiki_id": "wiki-b",
                "target_raw_node_id": "n2",
            },
        )
        fqs = FederatedQueryService(
            {
                "wiki-a": _NodeService({"n1": {"node_id": "n1", "content": "producer"}}),
                "wiki-b": _NodeService({"n2": {"node_id": "n2", "wiki_id": "wiki-b", "content": "consumer"}}),
            },
            project_id="project-1",
            project_storage=store,
        )
        stack = FederatedRetrieverStack(base_stacks, federated_query_service=fqs, dampening=0.5)

        out = _run(stack.aretrieve("users", k=5))
        by_id = {doc.metadata["node_id"]: doc for doc in out}

        assert "n2" in by_id
        expanded = by_id["n2"]
        assert expanded.metadata["project_node_id"] == "wiki-b::n2"
        assert expanded.metadata["cross_repo_relationship_type"] == "cross_repo_api_surface"
        assert expanded.metadata["cross_repo_surface"] == "GET /api/v1/users"
        assert expanded.metadata["cross_repo_matcher"] == "api_surface:rest"
        assert expanded.metadata["cross_repo_level"] == "L1"


class TestDampeningClamp:
    def test_invalid_dampening_falls_back(self):
        stack = FederatedRetrieverStack([], dampening="not-a-number")
        assert stack._dampening == 0.7

    def test_out_of_range_dampening_falls_back(self):
        for bad in (-0.1, 0.0, 1.5):
            assert FederatedRetrieverStack([], dampening=bad)._dampening == 0.7

    def test_in_range_kept(self):
        assert FederatedRetrieverStack([], dampening=0.4)._dampening == 0.4
