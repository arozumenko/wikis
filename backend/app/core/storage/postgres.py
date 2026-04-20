"""
PostgreSQL wiki storage backend — SQLAlchemy Core + pgvector.

Implements ``WikiStorageProtocol`` with all the same semantics as the
SQLite backend (``UnifiedWikiDB``) but stores data in PostgreSQL.

Key differences from SQLite:
- FTS uses ``tsvector``/``tsquery`` instead of FTS5
- Vector search uses ``pgvector`` extension (``vector`` type + ivfflat/hnsw)
- RRF fusion uses the same algorithm (no DB-level fusion)
- Transactions managed by SQLAlchemy Core
- Each repository gets its own schema (``wiki_{repo_id}``) for isolation

Requirements:
- PostgreSQL 15+ with ``pgvector`` extension
- ``pip install pgvector sqlalchemy psycopg2-binary``  (or ``asyncpg`` for async)
"""

from __future__ import annotations

import json
import logging
import os
import struct
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

import networkx as nx
from sqlalchemy import (
    Column,
    Float,
    Index,
    Integer,
    MetaData,
    String,
    Table,
    Text,
    create_engine,
    func,
    inspect,
    literal,
    select,
    text,
)
from sqlalchemy.engine import Engine

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Compatibility layer for raw conn.execute(SQL, (params,)) callers
# ---------------------------------------------------------------------------

class _CompatRow:
    """Row wrapper supporting both index access ``row[0]`` and dict ``row['col']``."""

    __slots__ = ("_values", "_keys")

    def __init__(self, keys: tuple, values: tuple):
        self._keys = keys
        self._values = values

    def __getitem__(self, idx):
        if isinstance(idx, str):
            try:
                return self._values[self._keys.index(idx)]
            except ValueError:
                raise KeyError(idx) from None
        return self._values[idx]

    def keys(self):
        return self._keys

    def __iter__(self):
        return iter(self._values)

    def __len__(self):
        return len(self._values)


class _CompatResultSet:
    """Thin result proxy returned by ``_ConnCompat.execute``."""

    __slots__ = ("_rows",)

    def __init__(self, rows: list[_CompatRow]):
        self._rows = rows

    def fetchall(self):
        return self._rows

    def fetchone(self):
        return self._rows[0] if self._rows else None


class _ConnCompat:
    """Makes a SQLAlchemy engine quack like a :class:`sqlite3.Connection`.

    Translates ``?``-placeholder SQL into ``:pN``-named-parameter SQL,
    sets ``search_path`` to the repository schema, and wraps result rows
    so they support both integer-index and dict-key access.

    Only used by legacy callers that still pass ``db.conn``.
    """

    def __init__(self, engine: Engine, schema: str):
        self._engine = engine
        self._schema = schema

    # noinspection PyMethodMayBeStatic
    def _adapt(self, sql: str, params: tuple | list | None):
        """Convert ``?`` placeholders → ``:p0, :p1, …`` and build param dict."""
        if params is None:
            return sql, {}
        named: dict[str, Any] = {}
        idx = 0
        parts: list[str] = []
        for ch in sql:
            if ch == "?":
                name = f"p{idx}"
                parts.append(f":{name}")
                named[name] = params[idx]
                idx += 1
            else:
                parts.append(ch)
        return "".join(parts), named

    def execute(self, sql: str, params=None):
        adapted_sql, named_params = self._adapt(sql, params)
        with self._engine.begin() as conn:
            conn.execute(text(f"SET search_path TO {self._schema}, public"))
            result = conn.execute(text(adapted_sql), named_params)
            keys = tuple(result.keys())
            rows = [_CompatRow(keys, tuple(r)) for r in result.fetchall()]
            return _CompatResultSet(rows)


# ---------------------------------------------------------------------------
# pgvector availability
# ---------------------------------------------------------------------------
_PGVECTOR_AVAILABLE = False
try:
    from pgvector.sqlalchemy import Vector  # type: ignore[import-untyped]

    _PGVECTOR_AVAILABLE = True
except ImportError:
    logger.debug("pgvector not installed — vector search disabled in PostgresWikiStorage")
    Vector = None  # type: ignore[assignment,misc]

# ---------------------------------------------------------------------------
# Architectural symbols (same as unified_db.py)
# ---------------------------------------------------------------------------
try:
    from ..constants import ARCHITECTURAL_SYMBOLS, DOC_SYMBOL_TYPES
except ImportError:
    ARCHITECTURAL_SYMBOLS = {
        "class", "interface", "struct", "enum", "trait",
        "function", "constant", "macro", "module_doc", "file_doc",
    }
    DOC_SYMBOL_TYPES = {"module_doc", "file_doc"}

try:
    from ..cluster_constants import is_test_path as _is_test_path
except ImportError:
    def _is_test_path(rel_path: str) -> bool:  # type: ignore[misc]
        return False

_LEGACY_DOC_TYPES = {"module_doc", "file_doc"}


def _is_doc_symbol(symbol_type: str) -> bool:
    return (
        symbol_type in DOC_SYMBOL_TYPES
        or symbol_type in _LEGACY_DOC_TYPES
        or symbol_type.endswith("_document")
        or symbol_type.endswith("_section")
        or symbol_type.endswith("_chunk")
    )


# ═══════════════════════════════════════════════════════════════════════════
# Schema helpers
# ═══════════════════════════════════════════════════════════════════════════

def _define_tables(
    metadata: MetaData,
    schema: str | None,
    embedding_dim: int,
    vec_available: bool,
) -> tuple[Table, Table, Table, Table | None]:
    """Define SQLAlchemy Table objects for nodes, edges, metadata, and optionally vectors."""

    nodes = Table(
        "repo_nodes",
        metadata,
        Column("node_id", Text, primary_key=True),
        Column("rel_path", Text, nullable=False, server_default=""),
        Column("file_name", Text, nullable=False, server_default=""),
        Column("language", Text, nullable=False, server_default=""),
        Column("start_line", Integer, server_default="0"),
        Column("end_line", Integer, server_default="0"),
        Column("symbol_name", Text, nullable=False, server_default=""),
        Column("symbol_type", Text, nullable=False, server_default=""),
        Column("parent_symbol", Text),
        Column("analysis_level", Text, server_default="comprehensive"),
        Column("source_text", Text, server_default=""),
        Column("docstring", Text, server_default=""),
        Column("signature", Text, server_default=""),
        Column("parameters", Text, server_default=""),
        Column("return_type", Text, server_default=""),
        Column("is_architectural", Integer, server_default="0"),
        Column("is_doc", Integer, server_default="0"),
        Column("is_test", Integer, server_default="0"),
        Column("chunk_type", Text),
        Column("macro_cluster", Integer),
        Column("micro_cluster", Integer),
        Column("is_hub", Integer, server_default="0"),
        Column("hub_assignment", Text),
        # tsvector column for FTS
        Column("fts_vector", Text),  # Will be cast to tsvector in DDL
        schema=schema,
    )

    edges = Table(
        "repo_edges",
        metadata,
        Column("id", Integer, primary_key=True, autoincrement=True),
        Column("source_id", Text, nullable=False),
        Column("target_id", Text, nullable=False),
        Column("rel_type", Text, nullable=False, server_default=""),
        Column("edge_class", Text, nullable=False, server_default="structural"),
        Column("analysis_level", Text, server_default="comprehensive"),
        Column("confidence", Text, nullable=False, server_default="EXTRACTED"),
        Column("weight", Float, server_default="1.0"),
        Column("raw_similarity", Float),
        Column("source_file", Text, server_default=""),
        Column("target_file", Text, server_default=""),
        Column("language", Text, server_default=""),
        Column("annotations", Text, server_default=""),
        Column("created_by", Text, server_default="ast"),
        schema=schema,
    )

    meta = Table(
        "wiki_meta",
        metadata,
        Column("key", Text, primary_key=True),
        Column("value", Text),
        schema=schema,
    )

    vec_table = None
    if vec_available and Vector is not None:
        vec_table = Table(
            "repo_vec",
            metadata,
            Column("node_id", Text, primary_key=True),
            Column("embedding", Vector(embedding_dim)),
            schema=schema,
        )

    return nodes, edges, meta, vec_table


# ═══════════════════════════════════════════════════════════════════════════
# PostgresWikiStorage
# ═══════════════════════════════════════════════════════════════════════════

class PostgresWikiStorage:
    """PostgreSQL storage backend conforming to ``WikiStorageProtocol``.

    Each repository is isolated in its own PostgreSQL schema
    (``wiki_{repo_id}``).  Creates tables on first use.

    Uses synchronous SQLAlchemy Core (not ORM, not async) for simplicity
    and compatibility with the existing synchronous codebase.
    """

    def __init__(
        self,
        dsn: str,
        repo_id: str,
        embedding_dim: int = 1536,
        *,
        readonly: bool = False,
        pool_size: int = 5,
        **kwargs: Any,
    ):
        self._dsn = dsn
        self._repo_id = repo_id
        self._embedding_dim = embedding_dim
        self._readonly = readonly
        # Sanitise repo_id for use as a PG schema name: only alnum + underscore.
        _safe = repo_id.replace("-", "_").replace(".", "_")
        self._schema = f"wiki_{_safe}"

        self._engine: Engine = create_engine(
            dsn,
            pool_size=pool_size,
            pool_pre_ping=True,
            echo=False,
        )

        self._vec_available = _PGVECTOR_AVAILABLE
        self._sa_meta = MetaData()

        self._nodes, self._edges, self._meta_table, self._vec_table = _define_tables(
            self._sa_meta, self._schema, embedding_dim, self._vec_available,
        )

        if not readonly:
            self._create_schema()

        logger.info(
            "PostgresWikiStorage opened: repo=%s schema=%s vec=%s dim=%d readonly=%s",
            repo_id, self._schema, self._vec_available, embedding_dim, readonly,
        )

    def _create_schema(self) -> None:
        """Create the repository schema and all tables/indexes."""
        with self._engine.begin() as conn:
            # Create schema
            conn.execute(text(f"CREATE SCHEMA IF NOT EXISTS {self._schema}"))

            # Enable pgvector extension (once per database)
            if self._vec_available:
                try:
                    conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
                except Exception as exc:
                    logger.warning("Failed to create pgvector extension: %s", exc)
                    self._vec_available = False

        # Create tables
        self._sa_meta.create_all(self._engine)

        # Create FTS components (tsvector column + GIN index + trigger)
        self._create_fts_infrastructure()

        # Create additional indexes
        self._create_indexes()

    def _create_fts_infrastructure(self) -> None:
        """Set up PostgreSQL full-text search infrastructure."""
        schema = self._schema
        with self._engine.begin() as conn:
            # Add tsvector column if not exists (ALTER TABLE is idempotent via IF NOT EXISTS trick)
            conn.execute(text(f"""
                DO $$
                BEGIN
                    IF NOT EXISTS (
                        SELECT 1 FROM information_schema.columns
                        WHERE table_schema = '{schema}' AND table_name = 'repo_nodes'
                        AND column_name = 'fts_doc'
                    ) THEN
                        ALTER TABLE {schema}.repo_nodes ADD COLUMN fts_doc tsvector;
                    END IF;
                END $$;
            """))

            # GIN index for FTS
            conn.execute(text(f"""
                CREATE INDEX IF NOT EXISTS idx_nodes_fts
                ON {schema}.repo_nodes USING gin(fts_doc)
            """))

            # Trigger to auto-update tsvector on INSERT/UPDATE
            conn.execute(text(f"""
                CREATE OR REPLACE FUNCTION {schema}.nodes_fts_update() RETURNS trigger AS $$
                BEGIN
                    NEW.fts_doc := setweight(to_tsvector('english', coalesce(NEW.symbol_name, '')), 'A') ||
                                   setweight(to_tsvector('english', coalesce(NEW.signature, '')), 'B') ||
                                   setweight(to_tsvector('english', coalesce(NEW.docstring, '')), 'C') ||
                                   setweight(to_tsvector('english', coalesce(NEW.source_text, '')), 'D');
                    RETURN NEW;
                END;
                $$ LANGUAGE plpgsql;
            """))

            conn.execute(text(f"""
                DROP TRIGGER IF EXISTS tr_nodes_fts ON {schema}.repo_nodes;
                CREATE TRIGGER tr_nodes_fts
                BEFORE INSERT OR UPDATE ON {schema}.repo_nodes
                FOR EACH ROW EXECUTE FUNCTION {schema}.nodes_fts_update();
            """))

    def _create_indexes(self) -> None:
        """Create secondary indexes."""
        schema = self._schema
        # §11.6 / B1 — backfill confidence column on legacy schemas. The
        # SQLAlchemy create_all() above is idempotent for *new* installs but
        # does not ALTER existing tables; do that here, then add the CHECK.
        with self._engine.begin() as conn:
            conn.execute(text(f"""
                DO $$
                BEGIN
                    IF NOT EXISTS (
                        SELECT 1 FROM information_schema.columns
                        WHERE table_schema = '{schema}' AND table_name = 'repo_edges'
                          AND column_name = 'confidence'
                    ) THEN
                        ALTER TABLE {schema}.repo_edges
                            ADD COLUMN confidence TEXT NOT NULL DEFAULT 'EXTRACTED';
                    END IF;
                    IF NOT EXISTS (
                        SELECT 1 FROM pg_constraint
                        WHERE conname = 'repo_edges_confidence_chk'
                    ) THEN
                        ALTER TABLE {schema}.repo_edges
                            ADD CONSTRAINT repo_edges_confidence_chk
                            CHECK (confidence IN ('EXTRACTED','INFERRED','AMBIGUOUS'));
                    END IF;
                END $$;
            """))
        indexes = [
            f"CREATE INDEX IF NOT EXISTS idx_nodes_path ON {schema}.repo_nodes(rel_path)",
            f"CREATE INDEX IF NOT EXISTS idx_nodes_name ON {schema}.repo_nodes(symbol_name)",
            f"CREATE INDEX IF NOT EXISTS idx_nodes_type ON {schema}.repo_nodes(symbol_type)",
            f"CREATE INDEX IF NOT EXISTS idx_nodes_lang ON {schema}.repo_nodes(language)",
            f"CREATE INDEX IF NOT EXISTS idx_nodes_parent ON {schema}.repo_nodes(parent_symbol)",
            f"CREATE INDEX IF NOT EXISTS idx_nodes_macro ON {schema}.repo_nodes(macro_cluster)",
            f"CREATE INDEX IF NOT EXISTS idx_nodes_micro ON {schema}.repo_nodes(macro_cluster, micro_cluster)",
            f"CREATE INDEX IF NOT EXISTS idx_nodes_arch ON {schema}.repo_nodes(is_architectural) WHERE is_architectural = 1",
            f"CREATE INDEX IF NOT EXISTS idx_nodes_hub ON {schema}.repo_nodes(is_hub) WHERE is_hub = 1",
            f"CREATE INDEX IF NOT EXISTS idx_nodes_test ON {schema}.repo_nodes(is_test) WHERE is_test = 1",
            f"CREATE INDEX IF NOT EXISTS idx_edges_source ON {schema}.repo_edges(source_id)",
            f"CREATE INDEX IF NOT EXISTS idx_edges_target ON {schema}.repo_edges(target_id)",
            f"CREATE INDEX IF NOT EXISTS idx_edges_type ON {schema}.repo_edges(rel_type)",
            f"CREATE INDEX IF NOT EXISTS idx_edges_class ON {schema}.repo_edges(edge_class)",
            f"CREATE INDEX IF NOT EXISTS idx_edges_weight ON {schema}.repo_edges(weight)",
            f"CREATE INDEX IF NOT EXISTS idx_edges_confidence ON {schema}.repo_edges(confidence)",
        ]
        with self._engine.begin() as conn:
            for ddl in indexes:
                conn.execute(text(ddl))

    # ==================================================================
    # Properties
    # ==================================================================

    @property
    def db_path(self) -> Path:
        """Sentinel path for PostgreSQL backend."""
        return Path(f"postgres:{self._schema}")

    @property
    def conn(self):
        """Return a compatibility wrapper that quacks like sqlite3.Connection.

        Used by ``shared_expansion.py`` and ``language_heuristics.py`` which
        still call ``db.conn.execute(SQL, (params,))``.
        """
        return _ConnCompat(self._engine, self._schema)

    @property
    def vec_available(self) -> bool:
        return self._vec_available

    @property
    def embedding_dim(self) -> int:
        return self._embedding_dim

    # ==================================================================
    # Lifecycle
    # ==================================================================

    def close(self) -> None:
        if self._engine:
            self._engine.dispose()
            self._engine = None  # type: ignore[assignment]

    def drop_schema(self) -> None:
        """Drop the entire ``wiki_{repo_id}`` schema and all its tables.

        Used during wiki deletion to fully remove a repository's data.
        """
        if self._engine is None:
            return
        with self._engine.begin() as conn:
            conn.execute(text(f"DROP SCHEMA IF EXISTS {self._schema} CASCADE"))
        logger.info("Dropped PostgreSQL schema: %s", self._schema)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
        return False

    # ==================================================================
    # Helpers
    # ==================================================================

    @staticmethod
    def _strip_nul(value):
        """Sanitise text for PostgreSQL: strip NUL bytes and lone surrogates.

        PostgreSQL text columns reject NUL (0x00).  Lone surrogates
        (U+D800..U+DFFF) can appear when source code is read with
        ``errors='surrogateescape'`` and cause encoding failures on
        the wire protocol.  We replace both with the empty string.
        """
        if isinstance(value, str):
            # Fast path: most strings are clean
            if "\x00" not in value:
                try:
                    value.encode("utf-8")
                    return value
                except UnicodeEncodeError:
                    pass
            # Slow path: strip NUL + re-encode to drop surrogates
            value = value.replace("\x00", "")
            value = value.encode("utf-8", errors="replace").decode("utf-8")
            return value
        return value

    def _row_to_dict(self, row) -> dict[str, Any]:
        """Convert a SQLAlchemy Row to a plain dict."""
        return dict(row._mapping) if hasattr(row, "_mapping") else dict(row)

    def _normalize_node(self, n: dict[str, Any]) -> dict[str, Any]:
        """Normalize a node dict for insertion."""
        symbol_type = n.get("symbol_type", "")
        if hasattr(symbol_type, "value"):
            symbol_type = symbol_type.value
        symbol_type = str(symbol_type).lower()

        is_arch = 1 if (
            symbol_type in ARCHITECTURAL_SYMBOLS
            or symbol_type in _LEGACY_DOC_TYPES
            or _is_doc_symbol(symbol_type)
        ) else 0
        is_doc = 1 if _is_doc_symbol(symbol_type) else 0

        rel_path = n.get("rel_path", "")
        is_test = 1 if _is_test_path(rel_path) else 0

        params = n.get("parameters", "")
        if isinstance(params, (list, tuple)):
            params = json.dumps(params)

        _s = self._strip_nul
        return {
            "node_id": _s(n["node_id"]),
            "rel_path": _s(rel_path),
            "file_name": _s(n.get("file_name", "")),
            "language": _s(n.get("language", "")),
            "start_line": n.get("start_line", 0),
            "end_line": n.get("end_line", 0),
            "symbol_name": _s(n.get("symbol_name", "")),
            "symbol_type": _s(symbol_type),
            "parent_symbol": _s(n.get("parent_symbol")),
            "analysis_level": _s(n.get("analysis_level", "comprehensive")),
            "source_text": _s(n.get("source_text", "")),
            "docstring": _s(n.get("docstring", "")),
            "signature": _s(n.get("signature", "")),
            "parameters": _s(params),
            "return_type": _s(n.get("return_type", "")),
            "is_architectural": is_arch,
            "is_doc": is_doc,
            "is_test": is_test,
            "chunk_type": _s(n.get("chunk_type")),
            "macro_cluster": n.get("macro_cluster"),
            "micro_cluster": n.get("micro_cluster"),
            "is_hub": n.get("is_hub", 0),
            "hub_assignment": _s(n.get("hub_assignment")),
        }

    # ==================================================================
    # NODE operations
    # ==================================================================

    def upsert_node(self, node_id: str, **attrs) -> None:
        self.upsert_nodes_batch([{"node_id": node_id, **attrs}])

    def upsert_nodes_batch(self, nodes: list[dict[str, Any]]) -> None:
        if not nodes:
            return

        schema = self._schema
        rows = [self._normalize_node(n) for n in nodes]

        # Use PostgreSQL UPSERT (INSERT ... ON CONFLICT ... DO UPDATE)
        cols = list(rows[0].keys())
        col_list = ", ".join(cols)
        val_placeholders = ", ".join(f":{c}" for c in cols)
        update_set = ", ".join(
            f"{c} = EXCLUDED.{c}" for c in cols if c != "node_id"
        )

        sql = text(
            f"INSERT INTO {schema}.repo_nodes ({col_list}) "
            f"VALUES ({val_placeholders}) "
            f"ON CONFLICT (node_id) DO UPDATE SET {update_set}"
        )

        with self._engine.begin() as conn:
            conn.execute(sql, rows)

    def get_node(self, node_id: str) -> dict[str, Any] | None:
        with self._engine.connect() as conn:
            row = conn.execute(
                text(f"SELECT * FROM {self._schema}.repo_nodes WHERE node_id = :nid"),
                {"nid": node_id},
            ).fetchone()
            return self._row_to_dict(row) if row else None

    def get_nodes_by_ids(self, node_ids: list[str]) -> list[dict[str, Any]]:
        if not node_ids:
            return []
        # Use ANY(array) for variable-length IN clause
        with self._engine.connect() as conn:
            rows = conn.execute(
                text(f"SELECT * FROM {self._schema}.repo_nodes WHERE node_id = ANY(:ids)"),
                {"ids": node_ids},
            ).fetchall()
            return [self._row_to_dict(r) for r in rows]

    def get_nodes_by_path_prefix(self, prefix: str, limit: int = 500) -> list[dict[str, Any]]:
        with self._engine.connect() as conn:
            rows = conn.execute(
                text(f"SELECT * FROM {self._schema}.repo_nodes WHERE rel_path LIKE :prefix LIMIT :lim"),
                {"prefix": prefix.rstrip("/") + "/%", "lim": limit},
            ).fetchall()
            return [self._row_to_dict(r) for r in rows]

    def get_nodes_by_cluster(
        self, macro: int, micro: int | None = None, limit: int = 1000,
    ) -> list[dict[str, Any]]:
        with self._engine.connect() as conn:
            if micro is not None:
                rows = conn.execute(
                    text(
                        f"SELECT * FROM {self._schema}.repo_nodes "
                        "WHERE macro_cluster = :macro AND micro_cluster = :micro LIMIT :lim"
                    ),
                    {"macro": macro, "micro": micro, "lim": limit},
                ).fetchall()
            else:
                rows = conn.execute(
                    text(f"SELECT * FROM {self._schema}.repo_nodes WHERE macro_cluster = :macro LIMIT :lim"),
                    {"macro": macro, "lim": limit},
                ).fetchall()
            return [self._row_to_dict(r) for r in rows]

    def get_architectural_nodes(self, limit: int = 5000) -> list[dict[str, Any]]:
        with self._engine.connect() as conn:
            rows = conn.execute(
                text(f"SELECT * FROM {self._schema}.repo_nodes WHERE is_architectural = 1 LIMIT :lim"),
                {"lim": limit},
            ).fetchall()
            return [self._row_to_dict(r) for r in rows]

    def get_all_nodes(self) -> list[dict[str, Any]]:
        with self._engine.connect() as conn:
            rows = conn.execute(
                text(f"SELECT * FROM {self._schema}.repo_nodes"),
            ).fetchall()
            return [self._row_to_dict(r) for r in rows]

    def node_count(self) -> int:
        with self._engine.connect() as conn:
            return conn.execute(
                text(f"SELECT count(*) FROM {self._schema}.repo_nodes"),
            ).scalar() or 0

    # ==================================================================
    # EDGE operations
    # ==================================================================

    def upsert_edge(self, source_id: str, target_id: str, rel_type: str, **attrs) -> None:
        self.upsert_edges_batch([{"source_id": source_id, "target_id": target_id, "rel_type": rel_type, **attrs}])

    def upsert_edges_batch(self, edges: list[dict[str, Any]]) -> None:
        if not edges:
            return

        # Optional in-batch deduplication of (source, target, rel_type)
        # tuples — collapses duplicates into a single row, summing
        # ``weight`` and keeping the first occurrence's other attributes.
        # Disabled by default; opt-in via WIKI_DEDUP_EDGES=1.
        if os.environ.get("WIKI_DEDUP_EDGES", "0") == "1":
            collapsed: dict[tuple[str, str, str], dict[str, Any]] = {}
            for e in edges:
                key = (
                    e.get("source_id", ""),
                    e.get("target_id", ""),
                    e.get("rel_type", ""),
                )
                existing = collapsed.get(key)
                if existing is None:
                    collapsed[key] = dict(e)
                else:
                    try:
                        existing["weight"] = float(existing.get("weight", 1.0)) + float(
                            e.get("weight", 1.0)
                        )
                    except (TypeError, ValueError):
                        pass
            edges = list(collapsed.values())

        _s = self._strip_nul
        schema = self._schema
        rows = []
        for e in edges:
            annotations = e.get("annotations", "")
            if isinstance(annotations, dict):
                annotations = json.dumps(annotations)
            rows.append({
                "source_id": _s(e["source_id"]),
                "target_id": _s(e["target_id"]),
                "rel_type": _s(e.get("rel_type", "")),
                "edge_class": _s(e.get("edge_class", "structural")),
                "analysis_level": _s(e.get("analysis_level", "comprehensive")),
                "confidence": _s(e.get("confidence", "EXTRACTED")),
                "weight": e.get("weight", 1.0),
                "raw_similarity": e.get("raw_similarity"),
                "source_file": _s(e.get("source_file", "")),
                "target_file": _s(e.get("target_file", "")),
                "language": _s(e.get("language", "")),
                "annotations": _s(annotations),
                "created_by": _s(e.get("created_by", "ast")),
            })

        cols = [c for c in rows[0].keys() if c != "id"]
        col_list = ", ".join(cols)
        val_placeholders = ", ".join(f":{c}" for c in cols)

        sql = text(f"INSERT INTO {schema}.repo_edges ({col_list}) VALUES ({val_placeholders})")

        with self._engine.begin() as conn:
            conn.execute(sql, rows)

    def get_edges_from(
        self, node_id: str, rel_types: list[str] | None = None,
    ) -> list[dict[str, Any]]:
        with self._engine.connect() as conn:
            if rel_types:
                rows = conn.execute(
                    text(
                        f"SELECT * FROM {self._schema}.repo_edges "
                        "WHERE source_id = :nid AND rel_type = ANY(:types)"
                    ),
                    {"nid": node_id, "types": rel_types},
                ).fetchall()
            else:
                rows = conn.execute(
                    text(f"SELECT * FROM {self._schema}.repo_edges WHERE source_id = :nid"),
                    {"nid": node_id},
                ).fetchall()
            return [self._row_to_dict(r) for r in rows]

    def get_edges_to(self, node_id: str) -> list[dict[str, Any]]:
        with self._engine.connect() as conn:
            rows = conn.execute(
                text(f"SELECT * FROM {self._schema}.repo_edges WHERE target_id = :nid"),
                {"nid": node_id},
            ).fetchall()
            return [self._row_to_dict(r) for r in rows]

    def get_neighbors(
        self, node_id: str, hops: int = 1, direction: str = "out",
    ) -> set[str]:
        visited: set[str] = set()
        frontier = {node_id}

        with self._engine.connect() as conn:
            for _ in range(hops):
                next_frontier: set[str] = set()
                for nid in frontier:
                    if direction in ("out", "both"):
                        for row in conn.execute(
                            text(f"SELECT target_id FROM {self._schema}.repo_edges WHERE source_id = :nid"),
                            {"nid": nid},
                        ):
                            next_frontier.add(row[0])
                    if direction in ("in", "both"):
                        for row in conn.execute(
                            text(f"SELECT source_id FROM {self._schema}.repo_edges WHERE target_id = :nid"),
                            {"nid": nid},
                        ):
                            next_frontier.add(row[0])
                visited |= frontier
                frontier = next_frontier - visited

        visited |= frontier
        visited.discard(node_id)
        return visited

    def edge_count(self) -> int:
        with self._engine.connect() as conn:
            return conn.execute(
                text(f"SELECT count(*) FROM {self._schema}.repo_edges"),
            ).scalar() or 0

    def update_edge_weights_batch(self, updates: list[dict[str, Any]]) -> int:
        if not updates:
            return 0
        with self._engine.begin() as conn:
            result = conn.execute(
                text(f"UPDATE {self._schema}.repo_edges SET weight = :weight WHERE id = :id"),
                updates,
            )
            return result.rowcount

    # ==================================================================
    # SEARCH — Full-Text Search (tsvector/tsquery)
    # ==================================================================

    def search_fts(
        self,
        query: str,
        path_prefix: str | None = None,
        cluster_id: int | None = None,
        symbol_types: list[str] | None = None,
        limit: int = 20,
    ) -> list[dict[str, Any]]:
        """BM25-like ranked text search using PostgreSQL tsvector."""
        if not query or not query.strip():
            return []

        # Convert query to tsquery (websearch format handles natural language)
        conditions = ["fts_doc @@ websearch_to_tsquery('english', :query)"]
        params: dict[str, Any] = {"query": query, "lim": limit}

        if path_prefix:
            conditions.append("rel_path LIKE :path_prefix")
            params["path_prefix"] = path_prefix.rstrip("/") + "/%"

        if cluster_id is not None:
            conditions.append("macro_cluster = :cluster_id")
            params["cluster_id"] = cluster_id

        if symbol_types:
            conditions.append("symbol_type = ANY(:symbol_types)")
            params["symbol_types"] = symbol_types

        where = " AND ".join(conditions)

        sql = text(
            f"SELECT *, ts_rank_cd(fts_doc, websearch_to_tsquery('english', :query)) AS fts_rank "
            f"FROM {self._schema}.repo_nodes "
            f"WHERE {where} "
            f"ORDER BY fts_rank DESC LIMIT :lim"
        )

        try:
            with self._engine.connect() as conn:
                rows = conn.execute(sql, params).fetchall()
                return [self._row_to_dict(r) for r in rows]
        except Exception as exc:
            logger.warning("PostgreSQL FTS search failed: %s", exc)
            return []

    # Alias for backward compat
    search_fts5 = search_fts

    # ==================================================================
    # SEARCH — Vector (pgvector)
    # ==================================================================

    def upsert_embedding(self, node_id: str, embedding: list[float]) -> None:
        if not self._vec_available or self._vec_table is None:
            return
        with self._engine.begin() as conn:
            conn.execute(
                text(
                    f"INSERT INTO {self._schema}.repo_vec (node_id, embedding) "
                    "VALUES (:nid, :emb) "
                    "ON CONFLICT (node_id) DO UPDATE SET embedding = EXCLUDED.embedding"
                ),
                {"nid": node_id, "emb": str(embedding)},
            )

    def upsert_embeddings_batch(
        self, embeddings: list[tuple[str, list[float]]],
    ) -> None:
        if not self._vec_available or not embeddings or self._vec_table is None:
            return

        rows = [{"nid": nid, "emb": str(vec)} for nid, vec in embeddings]

        with self._engine.begin() as conn:
            conn.execute(
                text(
                    f"INSERT INTO {self._schema}.repo_vec (node_id, embedding) "
                    "VALUES (:nid, :emb) "
                    "ON CONFLICT (node_id) DO UPDATE SET embedding = EXCLUDED.embedding"
                ),
                rows,
            )

    def populate_embeddings(
        self,
        embedding_fn: Callable[[list[str]], list[list[float]]],
        batch_size: int = 64,
        architectural_only: bool | None = None,
        max_workers: int | None = None,
    ) -> int:
        if not self._vec_available:
            logger.warning("populate_embeddings: pgvector not available, skipping")
            return 0

        if architectural_only is None:
            architectural_only = os.getenv("WIKI_EMBED_ARCH_ONLY", "1") != "0"
        if max_workers is None:
            try:
                max_workers = max(1, int(os.getenv("WIKI_EMBED_CONCURRENCY", "1")))
            except ValueError:
                max_workers = 1

        with self._engine.connect() as conn:
            # Fetch all nodes; use source_text with fallback to
            # docstring / symbol_name so that nodes with empty
            # source_text still get an embedding vector.
            arch_filter = " WHERE is_architectural = TRUE" if architectural_only else ""
            rows = conn.execute(
                text(
                    f"SELECT node_id, source_text, docstring, symbol_name "
                    f"FROM {self._schema}.repo_nodes" + arch_filter
                ),
            ).fetchall()

        embeddable: list[tuple[str, str]] = []
        for r in rows:
            txt = (r[1] or "").strip()
            if not txt:
                txt = (r[2] or "").strip()  # docstring
            if not txt:
                txt = (r[3] or "").strip()  # symbol_name
            if txt:
                embeddable.append((r[0], txt))

        batches: list[tuple[int, list[str], list[str]]] = []
        for i in range(0, len(embeddable), batch_size):
            batch = embeddable[i: i + batch_size]
            batches.append((i, [nid for nid, _ in batch], [t for _, t in batch]))

        total = 0

        def _embed_one(idx: int, node_ids: list[str], texts: list[str]):
            try:
                vectors = embedding_fn(texts)
            except Exception as exc:
                logger.warning(
                    "populate_embeddings: batch %d–%d failed: %s",
                    idx, idx + len(node_ids), exc,
                )
                return None
            return list(zip(node_ids, vectors, strict=True))

        if max_workers > 1 and len(batches) > 1:
            from concurrent.futures import ThreadPoolExecutor, as_completed
            with ThreadPoolExecutor(max_workers=max_workers) as pool:
                futures = [
                    pool.submit(_embed_one, idx, nids, texts)
                    for idx, nids, texts in batches
                ]
                for fut in as_completed(futures):
                    pairs = fut.result()
                    if pairs:
                        self.upsert_embeddings_batch(pairs)
                        total += len(pairs)
        else:
            for idx, nids, texts in batches:
                pairs = _embed_one(idx, nids, texts)
                if pairs:
                    self.upsert_embeddings_batch(pairs)
                    total += len(pairs)

        logger.info(
            "populate_embeddings: %d/%d nodes embedded "
            "(arch_only=%s, workers=%d)",
            total, len(rows), architectural_only, max_workers,
        )
        return total

    def search_vec(
        self,
        embedding: list[float],
        k: int = 10,
        path_prefix: str | None = None,
        cluster_id: int | None = None,
    ) -> list[dict[str, Any]]:
        if not self._vec_available or self._vec_table is None:
            return []

        conditions = []
        params: dict[str, Any] = {"emb": str(embedding), "k": k}

        if path_prefix:
            conditions.append("m.rel_path LIKE :path_prefix")
            params["path_prefix"] = path_prefix.rstrip("/") + "/%"

        if cluster_id is not None:
            conditions.append("m.macro_cluster = :cluster_id")
            params["cluster_id"] = cluster_id

        where_extra = (" AND " + " AND ".join(conditions)) if conditions else ""

        # Note: use CAST(:emb AS vector) instead of :emb::vector to avoid
        # SQLAlchemy text() misinterpreting the :: cast as a bind-param marker.
        sql = text(
            f"SELECT v.node_id, v.embedding <-> CAST(:emb AS vector) AS vec_distance, m.* "
            f"FROM {self._schema}.repo_vec v "
            f"JOIN {self._schema}.repo_nodes m ON v.node_id = m.node_id "
            f"WHERE 1=1 {where_extra} "
            f"ORDER BY v.embedding <-> CAST(:emb AS vector) "
            f"LIMIT :k"
        )

        try:
            with self._engine.connect() as conn:
                rows = conn.execute(sql, params).fetchall()
                return [self._row_to_dict(r) for r in rows]
        except Exception as exc:
            logger.warning("PostgreSQL vector search failed: %s", exc)
            return []

    # ==================================================================
    # SEARCH — Hybrid (RRF fusion) — same algorithm as SQLite backend
    # ==================================================================

    def search_hybrid(
        self,
        query: str,
        embedding: list[float] | None = None,
        path_prefix: str | None = None,
        cluster_id: int | None = None,
        fts_weight: float = 0.4,
        vec_weight: float = 0.6,
        limit: int = 20,
        fts_k: int = 30,
        vec_k: int = 30,
    ) -> list[dict[str, Any]]:
        rrf_constant = 60.0

        # FTS pass
        fts_results = self.search_fts(query, path_prefix=path_prefix, cluster_id=cluster_id, limit=fts_k)
        fts_scores: dict[str, float] = {}
        fts_nodes: dict[str, dict] = {}
        for rank, r in enumerate(fts_results):
            nid = r["node_id"]
            fts_scores[nid] = fts_weight / (rrf_constant + rank + 1)
            fts_nodes[nid] = r

        # Vector pass
        vec_scores: dict[str, float] = {}
        vec_nodes: dict[str, dict] = {}
        if embedding and self._vec_available:
            vec_results = self.search_vec(embedding, k=vec_k, path_prefix=path_prefix, cluster_id=cluster_id)
            for rank, r in enumerate(vec_results):
                nid = r["node_id"]
                vec_scores[nid] = vec_weight / (rrf_constant + rank + 1)
                vec_nodes[nid] = r

        # Fuse
        all_ids = set(fts_scores) | set(vec_scores)
        fused = []
        for nid in all_ids:
            score = fts_scores.get(nid, 0.0) + vec_scores.get(nid, 0.0)
            node_data = fts_nodes.get(nid) or vec_nodes.get(nid, {})
            entry = dict(node_data)
            entry["combined_score"] = score
            fused.append(entry)

        fused.sort(key=lambda x: x["combined_score"], reverse=True)
        return fused[:limit]

    # ==================================================================
    # CLUSTER operations
    # ==================================================================

    def set_cluster(self, node_id: str, macro: int, micro: int | None = None) -> None:
        with self._engine.begin() as conn:
            conn.execute(
                text(
                    f"UPDATE {self._schema}.repo_nodes "
                    "SET macro_cluster = :macro, micro_cluster = :micro "
                    "WHERE node_id = :nid"
                ),
                {"macro": macro, "micro": micro, "nid": node_id},
            )

    def set_clusters_batch(
        self, assignments: list[tuple[str, int, int | None]],
    ) -> None:
        if not assignments:
            return
        rows = [{"nid": nid, "macro": macro, "micro": micro} for nid, macro, micro in assignments]
        with self._engine.begin() as conn:
            conn.execute(
                text(
                    f"UPDATE {self._schema}.repo_nodes "
                    "SET macro_cluster = :macro, micro_cluster = :micro "
                    "WHERE node_id = :nid"
                ),
                rows,
            )

    def set_hub(
        self, node_id: str, is_hub: bool = True, assignment: str | None = None,
    ) -> None:
        with self._engine.begin() as conn:
            conn.execute(
                text(
                    f"UPDATE {self._schema}.repo_nodes "
                    "SET is_hub = :hub, hub_assignment = :assign "
                    "WHERE node_id = :nid"
                ),
                {"hub": 1 if is_hub else 0, "assign": assignment, "nid": node_id},
            )

    def get_cluster_nodes(self, macro: int, micro: int | None = None) -> list[str]:
        return [n["node_id"] for n in self.get_nodes_by_cluster(macro, micro)]

    def get_all_clusters(self) -> dict[int, dict[int, list[str]]]:
        with self._engine.connect() as conn:
            rows = conn.execute(
                text(
                    f"SELECT node_id, macro_cluster, micro_cluster "
                    f"FROM {self._schema}.repo_nodes "
                    "WHERE macro_cluster IS NOT NULL"
                ),
            ).fetchall()

        result: dict[int, dict[int, list[str]]] = {}
        for row in rows:
            macro = row[1]
            micro = row[2] or 0
            result.setdefault(macro, {}).setdefault(micro, []).append(row[0])
        return result

    # ==================================================================
    # GRAPH I/O — NetworkX conversion
    # ==================================================================

    def from_networkx(self, G: nx.MultiDiGraph) -> None:
        if G is None or G.number_of_nodes() == 0:
            logger.warning("from_networkx: empty graph, nothing to import")
            return

        t0 = time.time()
        logger.info(
            "Importing NetworkX graph: %d nodes, %d edges",
            G.number_of_nodes(), G.number_of_edges(),
        )

        # Nodes
        BATCH = 5000
        node_batch: list[dict[str, Any]] = []

        for node_id, data in G.nodes(data=True):
            node_dict = self._nx_node_to_dict(node_id, data)
            node_batch.append(node_dict)
            if len(node_batch) >= BATCH:
                self.upsert_nodes_batch(node_batch)
                node_batch.clear()

        if node_batch:
            self.upsert_nodes_batch(node_batch)

        t_nodes = time.time() - t0
        logger.info("Imported %d nodes in %.1fs", G.number_of_nodes(), t_nodes)

        # Edges — with per-batch recovery so a single bad edge cannot
        # leave the DB in a "nodes-only" state.
        t1 = time.time()
        edge_batch: list[dict[str, Any]] = []
        edges_ok = 0
        edges_failed = 0

        def _flush_edges(batch: list[dict[str, Any]]) -> tuple[int, int]:
            """Insert a batch; on failure, fall back to row-by-row."""
            ok = fail = 0
            try:
                self.upsert_edges_batch(batch)
                ok = len(batch)
            except Exception as batch_exc:
                logger.warning(
                    "Edge batch of %d failed (%s) — retrying row-by-row",
                    len(batch), batch_exc,
                )
                for row in batch:
                    try:
                        self.upsert_edges_batch([row])
                        ok += 1
                    except Exception as row_exc:
                        fail += 1
                        if fail <= 3:
                            logger.warning(
                                "Skipping edge %s→%s: %s",
                                row.get("source_id", "?")[:40],
                                row.get("target_id", "?")[:40],
                                row_exc,
                            )
            return ok, fail

        for u, v, key, data in G.edges(data=True, keys=True):
            edge_dict = self._nx_edge_to_dict(u, v, key, data)
            edge_batch.append(edge_dict)
            if len(edge_batch) >= BATCH:
                ok, fail = _flush_edges(edge_batch)
                edges_ok += ok
                edges_failed += fail
                edge_batch.clear()

        if edge_batch:
            ok, fail = _flush_edges(edge_batch)
            edges_ok += ok
            edges_failed += fail

        t_edges = time.time() - t1
        if edges_failed:
            logger.warning(
                "Imported %d edges in %.1fs (%d failed)",
                edges_ok, t_edges, edges_failed,
            )
        else:
            logger.info("Imported %d edges in %.1fs", edges_ok, t_edges)

        # FTS is handled by PostgreSQL trigger (auto-update on INSERT/UPDATE)
        # No manual FTS rebuild needed

        # Stale-node cleanup
        imported_ids = {str(n) for n in G.nodes()}
        existing = set()
        with self._engine.connect() as conn:
            for row in conn.execute(
                text(f"SELECT node_id FROM {self._schema}.repo_nodes"),
            ):
                existing.add(row[0])

        stale_ids = existing - imported_ids
        if stale_ids:
            stale_list = list(stale_ids)
            with self._engine.begin() as conn:
                conn.execute(
                    text(
                        f"DELETE FROM {self._schema}.repo_edges "
                        "WHERE source_id = ANY(:ids) OR target_id = ANY(:ids)"
                    ),
                    {"ids": stale_list},
                )
                conn.execute(
                    text(f"DELETE FROM {self._schema}.repo_nodes WHERE node_id = ANY(:ids)"),
                    {"ids": stale_list},
                )
            logger.info("Removed %d stale nodes and their edges", len(stale_ids))

        total = time.time() - t0
        logger.info(
            "from_networkx complete: %d nodes + %d edges in %.1fs",
            G.number_of_nodes(), G.number_of_edges(), total,
        )

    def _nx_node_to_dict(self, node_id: str, data: dict) -> dict[str, Any]:
        """Convert a NetworkX node to a dict (same logic as UnifiedWikiDB)."""
        symbol_obj = data.get("symbol")
        source_text = data.get("source_text", "")
        if not source_text and symbol_obj:
            source_text = getattr(symbol_obj, "source_text", "")

        docstring = data.get("docstring", "")
        if not docstring and symbol_obj:
            docstring = getattr(symbol_obj, "docstring", "") or ""

        signature = data.get("signature", "")
        if not signature and symbol_obj:
            signature = getattr(symbol_obj, "signature", "") or ""

        parameters = data.get("parameters", "")
        if not parameters and symbol_obj:
            parameters = getattr(symbol_obj, "parameters", []) or []

        return_type = data.get("return_type", "")
        if not return_type and symbol_obj:
            return_type = getattr(symbol_obj, "return_type", "") or ""

        return {
            "node_id": str(node_id),
            "rel_path": data.get("rel_path", ""),
            "file_name": data.get("file_name", ""),
            "language": data.get("language", ""),
            "start_line": data.get("start_line", 0),
            "end_line": data.get("end_line", 0),
            "symbol_name": data.get("symbol_name", data.get("name", "")),
            "symbol_type": data.get("symbol_type", data.get("type", "")),
            "parent_symbol": data.get("parent_symbol"),
            "analysis_level": data.get("analysis_level", "comprehensive"),
            "source_text": source_text,
            "docstring": docstring,
            "signature": signature,
            "parameters": parameters,
            "return_type": return_type,
        }

    def _nx_edge_to_dict(self, u: str, v: str, key: int, data: dict) -> dict[str, Any]:
        """Convert a NetworkX edge to a dict (same logic as UnifiedWikiDB)."""
        annotations = data.get("annotations", {})
        if isinstance(annotations, dict):
            annotations = json.dumps(annotations)

        _s = self._strip_nul
        return {
            "source_id": _s(str(u)),
            "target_id": _s(str(v)),
            "rel_type": _s(data.get("relationship_type", "")),
            "edge_class": _s(data.get("edge_class", "structural")),
            "analysis_level": _s(data.get("analysis_level", "comprehensive")),
            "weight": data.get("weight", 1.0),
            "raw_similarity": data.get("raw_similarity"),
            "source_file": _s(data.get("source_file", "")),
            "target_file": _s(data.get("target_file", "")),
            "language": _s(data.get("language", "")),
            "annotations": _s(annotations),
            "created_by": _s(data.get("created_by", "ast")),
        }

    def to_networkx(self) -> nx.MultiDiGraph:
        G = nx.MultiDiGraph()
        G.graph["analysis_type"] = "multi_tier"
        G.graph["source"] = "postgres"

        with self._engine.connect() as conn:
            # Nodes
            rows = conn.execute(text(f"SELECT * FROM {self._schema}.repo_nodes")).fetchall()
            for row in rows:
                d = self._row_to_dict(row)
                nid = d.pop("node_id")
                # Remove postgres-specific columns
                d.pop("fts_doc", None)
                d.pop("fts_vector", None)
                # Deserialize parameters
                params = d.get("parameters", "")
                if params and isinstance(params, str):
                    try:
                        d["parameters"] = json.loads(params)
                    except (json.JSONDecodeError, TypeError):
                        pass
                G.add_node(nid, **d)

            # Edges
            rows = conn.execute(text(f"SELECT * FROM {self._schema}.repo_edges")).fetchall()
            for row in rows:
                d = self._row_to_dict(row)
                src = d.pop("source_id")
                tgt = d.pop("target_id")
                d.pop("id", None)
                ann = d.get("annotations", "")
                if ann and isinstance(ann, str):
                    try:
                        d["annotations"] = json.loads(ann)
                    except (json.JSONDecodeError, TypeError):
                        pass
                d["relationship_type"] = d.pop("rel_type", "")
                G.add_edge(src, tgt, **d)

        # Attach lookup indexes so GraphQueryService / content expander
        # can do O(1) symbol resolution on the rehydrated graph instead of
        # falling back to FTS-only / full graph scans.
        try:
            from app.core.code_graph.graph_builder import attach_graph_indexes
            attach_graph_indexes(G)
        except Exception as exc:  # pragma: no cover - defensive
            logger.warning("to_networkx: failed to attach graph indexes: %s", exc)

        logger.info(
            "to_networkx: reconstructed %d nodes, %d edges",
            G.number_of_nodes(), G.number_of_edges(),
        )
        return G

    # ==================================================================
    # METADATA
    # ==================================================================

    def set_meta(self, key: str, value: Any) -> None:
        with self._engine.begin() as conn:
            conn.execute(
                text(
                    f"INSERT INTO {self._schema}.wiki_meta (key, value) "
                    "VALUES (:key, :val) "
                    "ON CONFLICT (key) DO UPDATE SET value = EXCLUDED.value"
                ),
                {"key": key, "val": json.dumps(value)},
            )

    def get_meta(self, key: str, default: Any = None) -> Any:
        with self._engine.connect() as conn:
            row = conn.execute(
                text(f"SELECT value FROM {self._schema}.wiki_meta WHERE key = :key"),
                {"key": key},
            ).fetchone()
            if row is None:
                return default
            try:
                return json.loads(row[0])
            except (json.JSONDecodeError, TypeError):
                return row[0]

    # ==================================================================
    # STATISTICS
    # ==================================================================

    def stats(self) -> dict[str, Any]:
        with self._engine.connect() as conn:
            n_nodes = conn.execute(text(f"SELECT count(*) FROM {self._schema}.repo_nodes")).scalar() or 0
            n_edges = conn.execute(text(f"SELECT count(*) FROM {self._schema}.repo_edges")).scalar() or 0

            langs: dict[str, int] = {}
            for row in conn.execute(
                text(f"SELECT language, count(*) FROM {self._schema}.repo_nodes GROUP BY language"),
            ):
                langs[row[0]] = row[1]

            types: dict[str, int] = {}
            for row in conn.execute(
                text(f"SELECT symbol_type, count(*) FROM {self._schema}.repo_nodes GROUP BY symbol_type"),
            ):
                types[row[0]] = row[1]

            edge_types: dict[str, int] = {}
            for row in conn.execute(
                text(f"SELECT rel_type, count(*) FROM {self._schema}.repo_edges GROUP BY rel_type"),
            ):
                edge_types[row[0]] = row[1]

            macro_count = conn.execute(
                text(
                    f"SELECT count(DISTINCT macro_cluster) FROM {self._schema}.repo_nodes "
                    "WHERE macro_cluster IS NOT NULL"
                ),
            ).scalar() or 0

            hub_count = conn.execute(
                text(f"SELECT count(*) FROM {self._schema}.repo_nodes WHERE is_hub = 1"),
            ).scalar() or 0

        return {
            "node_count": n_nodes,
            "edge_count": n_edges,
            "languages": langs,
            "symbol_types": types,
            "edge_types": edge_types,
            "macro_clusters": macro_count,
            "hub_count": hub_count,
            "vec_available": self._vec_available,
            "embedding_dim": self._embedding_dim,
            "db_size_mb": 0,  # N/A for PostgreSQL
        }

    # ==================================================================
    # Cluster-expansion query methods
    # ==================================================================

    def find_nodes_by_name(
        self,
        name: str,
        macro_cluster: int | None = None,
        architectural_only: bool = True,
        limit: int = 5,
    ) -> list[dict[str, Any]]:
        # P6 (§11.9): case-insensitive lookup so callers don't need to know
        # the exact casing (`EliteAClient` vs `EliteaClient` etc.).
        conditions = ["LOWER(symbol_name) = LOWER(:name)"]
        params: dict[str, Any] = {"name": name, "lim": limit}

        if macro_cluster is not None:
            conditions.append("macro_cluster = :macro")
            params["macro"] = macro_cluster

        if architectural_only:
            conditions.append("is_architectural = 1")

        where = " AND ".join(conditions)

        with self._engine.connect() as conn:
            rows = conn.execute(
                text(
                    f"SELECT * FROM {self._schema}.repo_nodes "
                    f"WHERE {where} "
                    "ORDER BY end_line - start_line DESC LIMIT :lim"
                ),
                params,
            ).fetchall()
            return [self._row_to_dict(r) for r in rows]

    def search_fts_by_symbol_name(
        self,
        name: str,
        macro_cluster: int | None = None,
        architectural_only: bool = True,
        limit: int = 3,
    ) -> list[dict[str, Any]]:
        conditions = []
        params: dict[str, Any] = {"name": name, "lim": limit}

        if architectural_only:
            conditions.append("is_architectural = 1")

        if macro_cluster is not None:
            conditions.append("macro_cluster = :macro")
            params["macro"] = macro_cluster

        where_extra = (" AND " + " AND ".join(conditions)) if conditions else ""

        try:
            with self._engine.connect() as conn:
                rows = conn.execute(
                    text(
                        f"SELECT *, ts_rank_cd(fts_doc, to_tsquery('english', :name)) AS rank "
                        f"FROM {self._schema}.repo_nodes "
                        f"WHERE fts_doc @@ to_tsquery('english', :name) {where_extra} "
                        "ORDER BY rank DESC LIMIT :lim"
                    ),
                    params,
                ).fetchall()
                return [self._row_to_dict(r) for r in rows]
        except Exception as exc:
            logger.debug("search_fts_by_symbol_name failed for '%s': %s", name, exc)
            return []

    def get_edge_targets(self, source_id: str) -> list[dict[str, Any]]:
        with self._engine.connect() as conn:
            rows = conn.execute(
                text(
                    f"SELECT target_id, rel_type, weight "
                    f"FROM {self._schema}.repo_edges WHERE source_id = :sid"
                ),
                {"sid": source_id},
            ).fetchall()
            return [self._row_to_dict(r) for r in rows]

    def get_edge_sources(self, target_id: str) -> list[dict[str, Any]]:
        with self._engine.connect() as conn:
            rows = conn.execute(
                text(
                    f"SELECT source_id, rel_type, weight "
                    f"FROM {self._schema}.repo_edges WHERE target_id = :tid"
                ),
                {"tid": target_id},
            ).fetchall()
            return [self._row_to_dict(r) for r in rows]

    def find_related_nodes(
        self,
        node_id: str,
        rel_types: list[str],
        direction: str = "out",
        target_symbol_types: list[str] | None = None,
        exclude_path: str | None = None,
        limit: int = 15,
    ) -> list[dict[str, Any]]:
        if not rel_types:
            return []

        schema = self._schema
        params: dict[str, Any] = {"nid": node_id, "rel_types": rel_types, "lim": limit}

        if direction == "out":
            base = (
                f"SELECT n.* FROM {schema}.repo_edges e "
                f"JOIN {schema}.repo_nodes n ON e.target_id = n.node_id "
                "WHERE e.source_id = :nid AND e.rel_type = ANY(:rel_types)"
            )
        else:
            base = (
                f"SELECT n.* FROM {schema}.repo_edges e "
                f"JOIN {schema}.repo_nodes n ON e.source_id = n.node_id "
                "WHERE e.target_id = :nid AND e.rel_type = ANY(:rel_types)"
            )

        if target_symbol_types:
            base += " AND n.symbol_type = ANY(:sym_types)"
            params["sym_types"] = target_symbol_types

        if exclude_path:
            base += " AND n.rel_path != :excl_path"
            params["excl_path"] = exclude_path

        base += " ORDER BY n.rel_path, n.start_line LIMIT :lim"

        with self._engine.connect() as conn:
            rows = conn.execute(text(base), params).fetchall()
            return [self._row_to_dict(r) for r in rows]

    def get_doc_nodes_by_cluster(
        self,
        macro: int,
        micro: int | None = None,
    ) -> list[dict[str, Any]]:
        params: dict[str, Any] = {"macro": macro}

        if micro is not None:
            sql = (
                f"SELECT * FROM {self._schema}.repo_nodes "
                "WHERE macro_cluster = :macro AND micro_cluster = :micro AND is_doc = 1 "
                "ORDER BY rel_path"
            )
            params["micro"] = micro
        else:
            sql = (
                f"SELECT * FROM {self._schema}.repo_nodes "
                "WHERE macro_cluster = :macro AND is_doc = 1 "
                "ORDER BY rel_path"
            )

        with self._engine.connect() as conn:
            rows = conn.execute(text(sql), params).fetchall()
            return [self._row_to_dict(r) for r in rows]

    def commit(self) -> None:
        """No-op — PostgreSQL uses auto-commit via SQLAlchemy engine.begin()."""
        pass

    # ── Bulk operations ──────────────────────────────────────────────

    def delete_all_edges(self) -> None:
        """Delete every row in the edges table."""
        with self._engine.begin() as conn:
            conn.execute(text(f"DELETE FROM {self._edges}"))

    # ── Language detection ───────────────────────────────────────────

    def detect_dominant_language(self, node_ids: list[str]) -> str | None:
        """Return the most common language among *node_ids*."""
        if not node_ids:
            return None
        placeholders = ", ".join(f":p{i}" for i in range(len(node_ids)))
        params = {f"p{i}": nid for i, nid in enumerate(node_ids)}
        with self._engine.connect() as conn:
            row = conn.execute(
                text(
                    f"SELECT language FROM {self._nodes} "
                    f"WHERE node_id IN ({placeholders}) AND language IS NOT NULL "
                    f"GROUP BY language ORDER BY COUNT(*) DESC LIMIT 1"
                ),
                params,
            ).fetchone()
        return row[0] if row else None

    # ══════════════════════════════════════════════════════════════════
    # POST-PASS HYGIENE / OBSERVABILITY (§11.7, §11.8, §11.12)
    # ══════════════════════════════════════════════════════════════════

    def demote_local_constants(self) -> int:
        """Demote constants whose every reference is in their defining file.

        See §11.8.  Constants with **zero** incoming reference edges are kept
        architectural (could be unused public API).  Returns rows demoted.
        """
        with self._engine.begin() as conn:
            result = conn.execute(
                text(
                    f"""
                    UPDATE {self._nodes} AS up
                       SET is_architectural = 0
                     WHERE up.node_id IN (
                         SELECT n.node_id
                           FROM {self._nodes} n
                           JOIN {self._edges} e   ON e.target_id = n.node_id
                           JOIN {self._nodes} src ON e.source_id = src.node_id
                          WHERE n.symbol_type = 'constant'
                            AND n.is_architectural = TRUE
                          GROUP BY n.node_id, n.rel_path
                         HAVING COUNT(DISTINCT src.rel_path) = 1
                            AND MAX(src.rel_path) = n.rel_path
                     )
                    """
                )
            )
        return int(result.rowcount or 0)

    def demote_vendored_code(self) -> int:
        """Demote nodes living under vendored / third-party paths (§11.12, P7).

        Path-based and conservative: matches obvious vendor locations and
        bundled test frameworks (gtest / gmock / googletest / googlemock).
        """
        with self._engine.begin() as conn:
            result = conn.execute(
                text(
                    f"""
                    UPDATE {self._nodes}
                       SET is_architectural = FALSE
                     WHERE is_architectural = TRUE
                       AND (
                            rel_path LIKE 'third_party/%%'
                         OR rel_path LIKE 'third-party/%%'
                         OR rel_path LIKE 'vendor/%%'
                         OR rel_path LIKE 'vendored/%%'
                         OR rel_path LIKE 'external/%%'
                         OR rel_path LIKE 'extern/%%'
                         OR rel_path LIKE 'deps/%%'
                         OR rel_path LIKE '%%/third_party/%%'
                         OR rel_path LIKE '%%/third-party/%%'
                         OR rel_path LIKE '%%/vendor/%%'
                         OR rel_path LIKE '%%/vendored/%%'
                         OR rel_path LIKE '%%/external/%%'
                         OR rel_path LIKE '%%/gtest/%%'
                         OR rel_path LIKE '%%/gmock/%%'
                         OR rel_path LIKE '%%/googletest/%%'
                         OR rel_path LIKE '%%/googlemock/%%'
                       )
                    """
                )
            )
        return int(result.rowcount or 0)

    def promote_class_members(self) -> int:
        """Synthesize ``member_uses`` edges (§11.4, P1).

        See protocol docstring for the rule.
        """
        nodes = self._nodes
        edges = self._edges
        with self._engine.begin() as conn:
            result = conn.execute(
                text(
                    f"""
                    INSERT INTO {edges} (
                        source_id, target_id, rel_type, edge_class,
                        analysis_level, confidence, weight,
                        source_file, target_file, language, created_by
                    )
                    SELECT
                        C.node_id,
                        T.node_id,
                        'member_uses',
                        'structural',
                        'comprehensive',
                        'INFERRED',
                        COUNT(*)::float,
                        COALESCE(C.rel_path, ''),
                        COALESCE(T.rel_path, ''),
                        COALESCE(C.language, ''),
                        'augmentation:11.4'
                      FROM {nodes} C
                      JOIN {edges} d
                        ON d.source_id = C.node_id
                       AND d.rel_type  = 'defines'
                      JOIN {nodes} M
                        ON M.node_id      = d.target_id
                       AND M.symbol_type IN ('method','field','property','constructor')
                      JOIN {edges} r
                        ON r.source_id = M.node_id
                       AND r.rel_type IN ('references','calls','creates','instantiates')
                      JOIN {nodes} T
                        ON T.node_id  = r.target_id
                       AND T.rel_path != C.rel_path
                     WHERE C.symbol_type IN ('class','struct','interface','trait')
                       AND C.is_architectural = TRUE
                       AND T.node_id != C.node_id
                       AND NOT EXISTS (
                           SELECT 1
                             FROM {edges} existing
                            WHERE existing.source_id = C.node_id
                              AND existing.target_id = T.node_id
                              AND existing.rel_type != 'defines'
                       )
                     GROUP BY C.node_id, T.node_id,
                              C.rel_path, T.rel_path, C.language
                    """
                )
            )
        return int(result.rowcount or 0)

    def compute_god_nodes(self, top_n: int = 20) -> dict[str, Any]:
        """Return top-N over-connected symbols and files (§11.7 / B2)."""
        with self._engine.connect() as conn:
            sym_rows = conn.execute(
                text(
                    f"""
                    SELECT n.node_id      AS symbol_id,
                           n.symbol_name  AS name,
                           n.symbol_type  AS symbol_type,
                           n.rel_path     AS rel_path,
                           ( (SELECT COUNT(*) FROM {self._edges} e1 WHERE e1.source_id = n.node_id)
                           + (SELECT COUNT(*) FROM {self._edges} e2 WHERE e2.target_id = n.node_id)
                           ) AS degree
                      FROM {self._nodes} n
                     WHERE n.is_architectural = TRUE
                     ORDER BY degree DESC
                     LIMIT :lim
                    """
                ),
                {"lim": top_n},
            ).mappings().fetchall()

            file_rows = conn.execute(
                text(
                    f"""
                    SELECT n.rel_path                                              AS rel_path,
                           SUM(CASE WHEN n.is_architectural THEN 1 ELSE 0 END)     AS internal_arch,
                           COUNT(e.id)                                             AS external_edges
                      FROM {self._nodes} n
                      LEFT JOIN {self._edges} e   ON e.source_id = n.node_id
                      LEFT JOIN {self._nodes} tgt ON e.target_id = tgt.node_id
                                                  AND tgt.rel_path != n.rel_path
                     WHERE tgt.rel_path IS NOT NULL
                     GROUP BY n.rel_path
                     ORDER BY external_edges DESC
                     LIMIT :lim
                    """
                ),
                {"lim": top_n},
            ).mappings().fetchall()

        return {
            "by_symbol_type": [dict(r) for r in sym_rows],
            "by_file": [dict(r) for r in file_rows],
        }
