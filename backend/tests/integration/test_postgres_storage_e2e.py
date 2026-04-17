"""Integration test — PostgreSQL storage backend end-to-end.

Requires a running PostgreSQL with pgvector on localhost:5433.
Start with:
  docker run -d --name wikis-pgtest -e POSTGRES_USER=wikis -e POSTGRES_PASSWORD=wikis \
    -e POSTGRES_DB=wikis_test -p 5433:5432 pgvector/pgvector:pg16
  docker exec wikis-pgtest psql -U wikis -d wikis_test -c "CREATE EXTENSION IF NOT EXISTS vector;"

Run:
  cd backend && ../.venv/bin/python -m pytest tests/integration/test_postgres_storage_e2e.py -v
"""

from __future__ import annotations

import os
import sys

import pytest

# Allow running from `backend/` directory
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

DSN = os.environ.get(
    "TEST_POSTGRES_DSN",
    "postgresql://wikis:wikis@localhost:5433/wikis_test",
)


def _pg_available() -> bool:
    """Check if PostgreSQL is reachable."""
    try:
        import sqlalchemy

        engine = sqlalchemy.create_engine(DSN)
        with engine.connect() as conn:
            conn.execute(sqlalchemy.text("SELECT 1"))
        engine.dispose()
        return True
    except Exception:
        return False


pytestmark = pytest.mark.skipif(
    not _pg_available(),
    reason="PostgreSQL not available at " + DSN,
)


@pytest.fixture
def repo_id():
    """Unique repo ID for test isolation."""
    import uuid

    return f"test_{uuid.uuid4().hex[:12]}"


@pytest.fixture
def storage(repo_id):
    """Create a PostgresWikiStorage instance and clean up after."""
    from app.core.storage.postgres import PostgresWikiStorage

    s = PostgresWikiStorage(dsn=DSN, repo_id=repo_id, embedding_dim=4)
    yield s
    # Cleanup: drop the schema
    try:
        import sqlalchemy
        from sqlalchemy import text

        engine = sqlalchemy.create_engine(DSN)
        with engine.begin() as conn:
            conn.execute(text(f"DROP SCHEMA IF EXISTS {s._schema} CASCADE"))
        engine.dispose()
    except Exception:
        pass
    s.close()


@pytest.fixture
def sample_graph():
    """Build a small NetworkX graph for testing."""
    import networkx as nx

    G = nx.MultiDiGraph()
    G.add_node(
        "mod::main",
        symbol_name="main",
        symbol_type="function",
        rel_path="src/main.py",
        language="python",
        start_line=1,
        end_line=10,
        parent_symbol="",
        docstring="Entry point function",
        source_text="def main():\n    print('hello')",
        is_architectural=1,
    )
    G.add_node(
        "mod::helper",
        symbol_name="helper",
        symbol_type="function",
        rel_path="src/helper.py",
        language="python",
        start_line=1,
        end_line=5,
        parent_symbol="",
        docstring="Helper function",
        source_text="def helper():\n    return 42",
        is_architectural=1,
    )
    G.add_node(
        "mod::Config",
        symbol_name="Config",
        symbol_type="class",
        rel_path="src/config.py",
        language="python",
        start_line=1,
        end_line=20,
        parent_symbol="",
        docstring="Configuration class for the application",
        source_text="class Config:\n    DEBUG = True",
        is_architectural=1,
    )
    G.add_edge("mod::main", "mod::helper", rel_type="calls", weight=1.0)
    G.add_edge("mod::main", "mod::Config", rel_type="uses", weight=0.5)
    return G


class TestPostgresStorageBasic:
    def test_create_and_stats(self, storage, sample_graph):
        storage.from_networkx(sample_graph)
        stats = storage.stats()
        assert stats["node_count"] == 3
        assert stats["edge_count"] == 2
        assert stats["db_size_mb"] >= 0

    def test_get_node(self, storage, sample_graph):
        storage.from_networkx(sample_graph)
        node = storage.get_node("mod::main")
        assert node is not None
        assert node["symbol_name"] == "main"
        assert node["symbol_type"] == "function"

    def test_get_node_missing(self, storage, sample_graph):
        storage.from_networkx(sample_graph)
        node = storage.get_node("nonexistent")
        assert node is None

    def test_to_networkx_roundtrip(self, storage, sample_graph):
        storage.from_networkx(sample_graph)
        G2 = storage.to_networkx()
        assert G2.number_of_nodes() == 3
        assert G2.number_of_edges() == 2
        # Check that node attributes survived
        assert G2.nodes["mod::main"]["symbol_name"] == "main"

    def test_set_and_get_meta(self, storage, sample_graph):
        storage.from_networkx(sample_graph)
        storage.set_meta("test_key", "test_value")
        assert storage.get_meta("test_key") == "test_value"
        # Overwrite
        storage.set_meta("test_key", "updated_value")
        assert storage.get_meta("test_key") == "updated_value"

    def test_clusters(self, storage, sample_graph):
        storage.from_networkx(sample_graph)
        storage.set_clusters_batch([
            ("mod::main", 0, 0),
            ("mod::helper", 0, 1),
            ("mod::Config", 1, 0),
        ])
        nodes = storage.get_nodes_by_cluster(0)
        assert len(nodes) == 2
        node_ids = {n["node_id"] for n in nodes}
        assert "mod::main" in node_ids
        assert "mod::helper" in node_ids


class TestPostgresFTS:
    def test_search_fts(self, storage, sample_graph):
        storage.from_networkx(sample_graph)
        results = storage.search_fts("configuration class", limit=5)
        assert len(results) > 0
        names = [r["symbol_name"] for r in results]
        assert "Config" in names

    def test_search_fts_empty_query(self, storage, sample_graph):
        storage.from_networkx(sample_graph)
        results = storage.search_fts("", limit=5)
        assert results == []

    def test_search_fts_by_symbol_name(self, storage, sample_graph):
        storage.from_networkx(sample_graph)
        results = storage.search_fts_by_symbol_name(
            "helper", architectural_only=False, limit=5,
        )
        assert len(results) > 0


class TestPostgresVectorSearch:
    def test_populate_and_search_embeddings(self, storage, sample_graph):
        storage.from_networkx(sample_graph)

        # Simple embedding function returning fixed-dim vectors
        def embed_fn(texts):
            import hashlib

            results = []
            for t in texts:
                h = hashlib.md5(t.encode()).hexdigest()  # noqa: S324
                results.append([float(int(h[i : i + 2], 16)) / 255.0 for i in range(0, 8, 2)])
            return results

        n = storage.populate_embeddings(embedding_fn=embed_fn, batch_size=10)
        assert n == 3

        # Vector search
        query_vec = [0.5, 0.5, 0.5, 0.5]
        results = storage.search_vec(query_vec, k=2)
        assert len(results) > 0

    def test_search_vec_with_path_prefix(self, storage, sample_graph):
        storage.from_networkx(sample_graph)

        def embed_fn(texts):
            import hashlib

            results = []
            for t in texts:
                h = hashlib.md5(t.encode()).hexdigest()  # noqa: S324
                results.append([float(int(h[i : i + 2], 16)) / 255.0 for i in range(0, 8, 2)])
            return results

        storage.populate_embeddings(embedding_fn=embed_fn, batch_size=10)

        query_vec = [0.5, 0.5, 0.5, 0.5]
        results = storage.search_vec(query_vec, k=5, path_prefix="src/")
        assert len(results) > 0
        for r in results:
            assert r["rel_path"].startswith("src/")


class TestPostgresPathPrefix:
    def test_get_nodes_by_path_prefix(self, storage, sample_graph):
        storage.from_networkx(sample_graph)
        results = storage.get_nodes_by_path_prefix("src/", limit=10)
        assert len(results) == 3

    def test_get_nodes_by_path_prefix_no_match(self, storage, sample_graph):
        storage.from_networkx(sample_graph)
        results = storage.get_nodes_by_path_prefix("nonexistent/", limit=10)
        assert len(results) == 0


class TestPostgresGraphTextIndex:
    def test_search(self, storage, sample_graph):
        from app.core.code_graph.postgres_graph_text_index import PostgresGraphTextIndex

        storage.from_networkx(sample_graph)
        idx = PostgresGraphTextIndex(storage)
        docs = idx.search("configuration class", k=5)
        assert len(docs) > 0
        assert any("Config" in d.metadata.get("symbol_name", "") for d in docs)

    def test_search_by_name(self, storage, sample_graph):
        from app.core.code_graph.postgres_graph_text_index import PostgresGraphTextIndex

        storage.from_networkx(sample_graph)
        idx = PostgresGraphTextIndex(storage)
        docs = idx.search_by_name("helper", k=5)
        assert len(docs) > 0

    def test_search_by_path_prefix(self, storage, sample_graph):
        from app.core.code_graph.postgres_graph_text_index import PostgresGraphTextIndex

        storage.from_networkx(sample_graph)
        idx = PostgresGraphTextIndex(storage)
        results = idx.search_by_path_prefix("src/", k=10)
        assert len(results) == 3

    def test_get_by_node_id(self, storage, sample_graph):
        from app.core.code_graph.postgres_graph_text_index import PostgresGraphTextIndex

        storage.from_networkx(sample_graph)
        idx = PostgresGraphTextIndex(storage)
        node = idx.get_by_node_id("mod::main")
        assert node is not None
        assert node["symbol_name"] == "main"


class TestFactoryWiring:
    def test_create_storage_postgres(self, repo_id):
        from app.core.storage import create_storage

        s = create_storage(
            backend="postgres",
            dsn=DSN,
            repo_id=repo_id,
            embedding_dim=4,
        )
        assert s is not None
        s.close()

        # Cleanup
        import sqlalchemy
        from sqlalchemy import text

        engine = sqlalchemy.create_engine(DSN)
        with engine.begin() as conn:
            conn.execute(text(f"DROP SCHEMA IF EXISTS wiki_{repo_id} CASCADE"))
        engine.dispose()

    def test_is_postgres_backend(self):
        from unittest.mock import MagicMock

        from app.core.storage import is_postgres_backend

        mock_settings = MagicMock()
        mock_settings.wiki_storage_backend = "postgres"
        assert is_postgres_backend(mock_settings) is True

        mock_settings.wiki_storage_backend = "sqlite"
        assert is_postgres_backend(mock_settings) is False

    def test_open_storage_postgres(self, repo_id):
        from unittest.mock import MagicMock

        from app.core.storage import open_storage

        mock_settings = MagicMock()
        mock_settings.wiki_storage_backend = "postgres"
        mock_settings.wiki_storage_dsn = DSN

        s = open_storage(repo_id=repo_id, embedding_dim=4, settings=mock_settings)
        assert s is not None
        s.close()

        # Cleanup
        import sqlalchemy
        from sqlalchemy import text

        engine = sqlalchemy.create_engine(DSN)
        with engine.begin() as conn:
            conn.execute(text(f"DROP SCHEMA IF EXISTS wiki_{repo_id} CASCADE"))
        engine.dispose()
