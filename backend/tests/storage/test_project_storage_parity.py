"""Phase 7 — InMemoryProjectStorage parity tests."""

from __future__ import annotations

import pytest

from app.core.storage.project_storage import (
    InMemoryProjectStorage,
    ProjectStorageProtocol,
)


@pytest.fixture
def store() -> InMemoryProjectStorage:
    return InMemoryProjectStorage()


class TestProtocol:
    def test_in_memory_satisfies_protocol(self, store):
        assert isinstance(store, ProjectStorageProtocol)


class TestNodes:
    def test_upsert_and_get(self, store):
        store.upsert_project_node("p1", "n1", "wiki-a", {"language": "python"})
        store.upsert_project_node("p1", "n2", "wiki-b", {"language": "go"})
        rows = store.get_project_nodes("p1")
        assert {r["node_id"] for r in rows} == {"n1", "n2"}

    def test_upsert_overrides(self, store):
        store.upsert_project_node("p1", "n1", "wiki-a", {"weight": 0.1})
        store.upsert_project_node("p1", "n1", "wiki-a", {"weight": 0.9})
        (row,) = store.get_project_nodes("p1")
        assert row["weight"] == 0.9

    def test_unknown_project_returns_empty(self, store):
        assert store.get_project_nodes("missing") == []

    def test_silent_noop_on_empty_ids(self, store):
        store.upsert_project_node("", "n1", "w", {})
        store.upsert_project_node("p1", "", "w", {})
        assert store.get_project_nodes("p1") == []


class TestEdges:
    def test_upsert_and_filter(self, store):
        store.upsert_project_edge("p1", "n1", "n2", "cross_repo", 0.5, {"level": "L0"})
        store.upsert_project_edge("p1", "n1", "n3", "cross_language", 0.4)
        store.upsert_project_edge("p1", "n9", "n2", "cross_repo", 0.3)

        all_rows = store.get_project_edges("p1")
        assert len(all_rows) == 3

        cross_repo = store.get_project_edges("p1", edge_class="cross_repo")
        assert {r["target_node_id"] for r in cross_repo} == {"n2"}
        assert all(r["edge_class"] == "cross_repo" for r in cross_repo)

        from_n1 = store.get_project_edges("p1", source_node_id="n1")
        assert len(from_n1) == 2

        from_n1_xrepo = store.get_project_edges(
            "p1", source_node_id="n1", edge_class="cross_repo"
        )
        assert len(from_n1_xrepo) == 1
        assert from_n1_xrepo[0]["weight"] == 0.5
        assert from_n1_xrepo[0]["provenance"]["level"] == "L0"

    def test_upsert_dedups_on_class(self, store):
        store.upsert_project_edge("p1", "n1", "n2", "cross_repo", 0.5)
        store.upsert_project_edge("p1", "n1", "n2", "cross_repo", 0.7)
        rows = store.get_project_edges("p1")
        assert len(rows) == 1 and rows[0]["weight"] == 0.7


class TestRelatedness:
    def test_order_insensitive(self, store):
        store.upsert_repo_relatedness("p1", "wiki-b", "wiki-a", 0.42, {"lang": 0.5})
        rows = store.get_repo_relatedness("p1")
        assert len(rows) == 1
        row = rows[0]
        # Stored canonicalised (alphabetical).
        assert row["wiki_a"] == "wiki-a" and row["wiki_b"] == "wiki-b"
        assert row["score"] == pytest.approx(0.42)
        assert row["breakdown"]["lang"] == 0.5

    def test_self_pair_skipped(self, store):
        store.upsert_repo_relatedness("p1", "wiki-a", "wiki-a", 1.0)
        assert store.get_repo_relatedness("p1") == []

    def test_re_upsert_overwrites(self, store):
        store.upsert_repo_relatedness("p1", "wiki-a", "wiki-b", 0.3)
        store.upsert_repo_relatedness("p1", "wiki-a", "wiki-b", 0.7)
        rows = store.get_repo_relatedness("p1")
        assert len(rows) == 1 and rows[0]["score"] == 0.7


class TestMeta:
    def test_set_and_get(self, store):
        store.set_project_meta("p1", "outline_version", "v3")
        store.set_project_meta("p1", "weights", {"alpha": 0.5})
        assert store.get_project_meta("p1", "outline_version") == "v3"
        assert store.get_project_meta("p1", "weights") == {"alpha": 0.5}

    def test_unknown_key_returns_none(self, store):
        assert store.get_project_meta("p1", "missing") is None

    def test_non_serializable_silently_skipped(self, store):
        # `set` is not JSON-serializable.
        store.set_project_meta("p1", "bad", {"set": {1, 2, 3}})
        assert store.get_project_meta("p1", "bad") is None
