"""Phase 7 — Project storage parity tests (memory + sqlite + postgres)."""

from __future__ import annotations

import os

import pytest

from app.core.storage.project_storage import (
    InMemoryProjectStorage,
    PostgresProjectStorage,
    ProjectStorageProtocol,
    SqliteProjectStorage,
    drop_project_storage,
)


def _make_postgres(tmp_path):
    dsn = os.environ.get("WIKI_TEST_PG_DSN", "")
    if not dsn:
        pytest.skip("WIKI_TEST_PG_DSN not set; skipping Postgres parity test")
    try:
        return PostgresProjectStorage(dsn, f"parity_{tmp_path.name}")
    except Exception as exc:  # pragma: no cover
        pytest.skip(f"PostgresProjectStorage unavailable: {exc}")


@pytest.fixture(params=["memory", "sqlite", "postgres"])
def store(request, tmp_path):
    backend = request.param
    if backend == "memory":
        yield InMemoryProjectStorage()
        return
    if backend == "sqlite":
        s = SqliteProjectStorage(str(tmp_path / "p.project.db"))
        try:
            yield s
        finally:
            s.close()
        return
    s = _make_postgres(tmp_path)
    try:
        yield s
    finally:
        try:
            s.close()
        except Exception:
            pass


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

    def test_clear_project_edges(self, store):
        store.upsert_project_edge("p1", "n1", "n2", "cross_repo", 0.5)
        store.upsert_project_edge("p1", "n3", "n4", "cross_language", 0.2)

        store.clear_project_edges("p1")

        assert store.get_project_edges("p1") == []


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

    def test_clear_repo_relatedness(self, store):
        store.upsert_repo_relatedness("p1", "wiki-a", "wiki-b", 0.3)
        store.upsert_repo_relatedness("p1", "wiki-a", "wiki-c", 0.7)

        store.clear_repo_relatedness("p1")

        assert store.get_repo_relatedness("p1") == []


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


class TestLifecycleCleanup:
    def test_drop_project_storage_removes_sqlite_database_and_sidecars(self, tmp_path):
        from pydantic import SecretStr

        from app.config import Settings

        settings = Settings(
            llm_api_key=SecretStr("test"),
            cache_dir=str(tmp_path),
            wiki_storage_backend="sqlite",
        )
        project_id = "Project 123"
        db_path = tmp_path / "projects" / "Project_123.project.db"
        db_path.parent.mkdir(parents=True, exist_ok=True)
        db_path.write_text("db")
        wal_path = tmp_path / "projects" / "Project_123.project.db-wal"
        shm_path = tmp_path / "projects" / "Project_123.project.db-shm"
        wal_path.write_text("wal")
        shm_path.write_text("shm")

        assert drop_project_storage(project_id, settings=settings)

        assert not db_path.exists()
        assert not wal_path.exists()
        assert not shm_path.exists()

    def test_postgres_storage_can_drop_schema_when_available(self):
        dsn = os.environ.get("WIKI_TEST_PG_DSN", "")
        if not dsn:
            pytest.skip("WIKI_TEST_PG_DSN not set; skipping Postgres drop-schema test")
        storage = PostgresProjectStorage(dsn, "cleanup_parity")
        try:
            storage.upsert_project_node("cleanup_parity", "n1", "w1", {})
            storage.drop_schema()
        finally:
            storage.close()
