"""Phase 7 — FederatedRetrieverStack tests."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from typing import Any

import pytest

from app.core.federated_retriever import FederatedRetrieverStack


@dataclass
class _Doc:
    """Tiny LangChain-Document stand-in (page_content + metadata)."""

    page_content: str
    metadata: dict = field(default_factory=dict)


class _StubStack:
    """WikiRetrieverStack stand-in returning canned hits."""

    def __init__(self, docs: list[_Doc]) -> None:
        self._docs = docs

    def search_repository(self, query: str, k: int = 15) -> list[_Doc]:
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


class TestDampeningClamp:
    def test_invalid_dampening_falls_back(self):
        stack = FederatedRetrieverStack([], dampening="not-a-number")
        assert stack._dampening == 0.7

    def test_out_of_range_dampening_falls_back(self):
        for bad in (-0.1, 0.0, 1.5):
            assert FederatedRetrieverStack([], dampening=bad)._dampening == 0.7

    def test_in_range_kept(self):
        assert FederatedRetrieverStack([], dampening=0.4)._dampening == 0.4
