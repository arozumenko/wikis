"""
Unified Wiki Database — Single SQLite DB for Graph + FTS5 + Vectors + Metadata

Consolidates 6–8 per-repo artifact files into a single .wiki.db:
- repo_nodes: all node metadata + source text (replaces docstore, pickle)
- repo_edges: all edges with weights/annotations (replaces pickle graph edges)
- repo_fts:   FTS5 virtual table for BM25-ranked text search (replaces .fts5.db + .bm25.sqlite)
- repo_vec:   sqlite-vec virtual table for KNN embeddings (replaces .faiss + .docstore.bin)
- clusters:   pre-computed macro/micro cluster assignments (Phase 3)
- wiki_meta:  build metadata, stats, feature flags

Always enabled — no feature flag.

Phase 1 scope:
- Schema creation with all tables/indexes
- Node/edge CRUD with batch operations
- from_networkx() bulk import from existing graph
- to_networkx() reconstruction for algorithm phase
- FTS5 search with BM25 ranking + path filtering
- sqlite-vec KNN search (gracefully degrades if extension unavailable)
- Hybrid RRF search combining FTS5 + vec
- Metadata storage
"""

import json
import logging
import os
import struct
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

import networkx as nx

# ---------------------------------------------------------------------------
# SQLite driver selection
# ---------------------------------------------------------------------------
# On macOS / official-site Python builds the stdlib ``sqlite3`` module is
# compiled *without* ``ENABLE_LOAD_EXTENSION``, which means sqlite-vec
# cannot be loaded.  ``pysqlite3`` (``pip install pysqlite3``) ships its
# own libsqlite3 built from source with extensions enabled — we prefer it
# when available so that vector search works locally on every platform.
# The two packages expose an identical DB-API 2.0 interface so the rest of
# the module works unchanged.
# ---------------------------------------------------------------------------
try:
    import pysqlite3 as sqlite3  # type: ignore[import-untyped]
except ImportError:
    import sqlite3  # type: ignore[no-redef]

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Always enabled (no feature flag)
# ---------------------------------------------------------------------------
UNIFIED_DB_ENABLED = True  # Kept for backward compat; always True

# ---------------------------------------------------------------------------
# sqlite-vec availability (graceful degradation)
# ---------------------------------------------------------------------------
_SQLITE_VEC_AVAILABLE = False
try:
    import sqlite_vec  # type: ignore

    _SQLITE_VEC_AVAILABLE = True
except ImportError:
    logger.debug("sqlite-vec not installed — vector search disabled in UnifiedWikiDB")


def _serialize_float32_vec(vec: list[float]) -> bytes:
    """Serialize a list of floats to bytes for sqlite-vec insertion."""
    return struct.pack(f"{len(vec)}f", *vec)


def _can_load_extensions() -> bool:
    """Check if the active sqlite3 driver supports loading extensions."""
    try:
        conn = sqlite3.connect(":memory:")
        conn.enable_load_extension(True)
        conn.close()
        return True
    except (AttributeError, sqlite3.OperationalError):
        return False


_EXTENSIONS_LOADABLE = _can_load_extensions()


# ═══════════════════════════════════════════════════════════════════════════
# Schema DDL
# ═══════════════════════════════════════════════════════════════════════════

_SCHEMA_NODES = """
CREATE TABLE IF NOT EXISTS repo_nodes (
    -- Identity
    node_id         TEXT PRIMARY KEY,

    -- Location
    rel_path        TEXT NOT NULL DEFAULT '',
    file_name       TEXT NOT NULL DEFAULT '',
    language        TEXT NOT NULL DEFAULT '',
    start_line      INTEGER DEFAULT 0,
    end_line        INTEGER DEFAULT 0,

    -- Symbol metadata
    symbol_name     TEXT NOT NULL DEFAULT '',
    symbol_type     TEXT NOT NULL DEFAULT '',
    parent_symbol   TEXT DEFAULT NULL,
    analysis_level  TEXT DEFAULT 'comprehensive',

    -- Content (replaces docstore.bin + docs.pkl)
    source_text     TEXT DEFAULT '',
    docstring       TEXT DEFAULT '',
    signature       TEXT DEFAULT '',
    parameters      TEXT DEFAULT '',
    return_type     TEXT DEFAULT '',

    -- Classification
    is_architectural INTEGER DEFAULT 0,
    is_doc           INTEGER DEFAULT 0,
    is_test          INTEGER DEFAULT 0,
    chunk_type       TEXT DEFAULT NULL,

    -- Clustering (Phase 3 — pre-populated as NULL)
    macro_cluster   INTEGER DEFAULT NULL,
    micro_cluster   INTEGER DEFAULT NULL,
    is_hub          INTEGER DEFAULT 0,
    hub_assignment  TEXT DEFAULT NULL,

    -- Timestamps
    indexed_at      TEXT DEFAULT (datetime('now'))
);
"""

_SCHEMA_NODES_INDEXES = """
CREATE INDEX IF NOT EXISTS idx_nodes_path       ON repo_nodes(rel_path);
CREATE INDEX IF NOT EXISTS idx_nodes_name       ON repo_nodes(symbol_name);
CREATE INDEX IF NOT EXISTS idx_nodes_type       ON repo_nodes(symbol_type);
CREATE INDEX IF NOT EXISTS idx_nodes_lang       ON repo_nodes(language);
CREATE INDEX IF NOT EXISTS idx_nodes_parent     ON repo_nodes(parent_symbol);
CREATE INDEX IF NOT EXISTS idx_nodes_macro      ON repo_nodes(macro_cluster);
CREATE INDEX IF NOT EXISTS idx_nodes_micro      ON repo_nodes(macro_cluster, micro_cluster);
CREATE INDEX IF NOT EXISTS idx_nodes_arch       ON repo_nodes(is_architectural) WHERE is_architectural = 1;
CREATE INDEX IF NOT EXISTS idx_nodes_hub        ON repo_nodes(is_hub) WHERE is_hub = 1;
CREATE INDEX IF NOT EXISTS idx_nodes_test       ON repo_nodes(is_test) WHERE is_test = 1;
"""

_SCHEMA_EDGES = """
CREATE TABLE IF NOT EXISTS repo_edges (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    source_id       TEXT NOT NULL,
    target_id       TEXT NOT NULL,

    -- Classification
    rel_type        TEXT NOT NULL DEFAULT '',
    edge_class      TEXT NOT NULL DEFAULT 'structural',
    analysis_level  TEXT DEFAULT 'comprehensive',

    -- Weight (Phase 2 — default 1.0)
    weight          REAL DEFAULT 1.0,
    raw_similarity  REAL DEFAULT NULL,

    -- Context
    source_file     TEXT DEFAULT '',
    target_file     TEXT DEFAULT '',
    language        TEXT DEFAULT '',
    annotations     TEXT DEFAULT '',

    -- Provenance
    created_by      TEXT DEFAULT 'ast',

    FOREIGN KEY (source_id) REFERENCES repo_nodes(node_id),
    FOREIGN KEY (target_id) REFERENCES repo_nodes(node_id)
);
"""

# Note: We deliberately do NOT add a UNIQUE constraint on (source_id, target_id, rel_type)
# because MultiDiGraph supports parallel edges (e.g. Rust parser emits duplicate DEFINES
# edges with different annotations — cross_file=True vs False).

_SCHEMA_EDGES_INDEXES = """
CREATE INDEX IF NOT EXISTS idx_edges_source  ON repo_edges(source_id);
CREATE INDEX IF NOT EXISTS idx_edges_target  ON repo_edges(target_id);
CREATE INDEX IF NOT EXISTS idx_edges_type    ON repo_edges(rel_type);
CREATE INDEX IF NOT EXISTS idx_edges_class   ON repo_edges(edge_class);
CREATE INDEX IF NOT EXISTS idx_edges_weight  ON repo_edges(weight);
"""

_SCHEMA_FTS5 = """
CREATE VIRTUAL TABLE IF NOT EXISTS repo_fts USING fts5(
    node_id UNINDEXED,
    symbol_name,
    signature,
    docstring,
    source_text,
    tokenize='porter unicode61'
);
"""

_SCHEMA_META = """
CREATE TABLE IF NOT EXISTS wiki_meta (
    key    TEXT PRIMARY KEY,
    value  TEXT
);
"""

# sqlite-vec table — created only when the extension is available
_SCHEMA_VEC_TEMPLATE = """
CREATE VIRTUAL TABLE IF NOT EXISTS repo_vec USING vec0(
    node_id TEXT PRIMARY KEY,
    embedding float32[{dim}]
);
"""


# ═══════════════════════════════════════════════════════════════════════════
# Architectural symbols — shared with graph_builder
# ═══════════════════════════════════════════════════════════════════════════
try:
    from .constants import ARCHITECTURAL_SYMBOLS, DOC_SYMBOL_TYPES
except ImportError:
    ARCHITECTURAL_SYMBOLS = {
        "class",
        "interface",
        "struct",
        "enum",
        "trait",
        "function",
        "constant",
        "macro",
        "module_doc",
        "file_doc",
    }
    DOC_SYMBOL_TYPES = {"module_doc", "file_doc"}

try:
    from .cluster_constants import is_test_path as _is_test_path
except ImportError:
    def _is_test_path(rel_path: str) -> bool:  # type: ignore[misc]
        return False

# Legacy symbol types used by graph_builder (pre-constants.py centralization)
# These coexist with the *_document types in DOC_SYMBOL_TYPES from constants.py
_LEGACY_DOC_TYPES = {"module_doc", "file_doc"}


def _is_doc_symbol(symbol_type: str) -> bool:
    """Check if symbol_type is a documentation type (handles legacy + new types)."""
    return (
        symbol_type in DOC_SYMBOL_TYPES
        or symbol_type in _LEGACY_DOC_TYPES
        or symbol_type.endswith("_document")
        or symbol_type.endswith("_section")
        or symbol_type.endswith("_chunk")
    )


class UnifiedWikiDB:
    """Single SQLite database combining graph, FTS5, vectors, and metadata.

    Wraps one ``sqlite3.Connection`` with optional ``sqlite-vec``.
    Provides typed methods for CRUD, search, graph I/O, and metadata.

    Usage::

        db = UnifiedWikiDB("/data/wiki_builder/cache/abc123.wiki.db")
        db.from_networkx(graph)          # bulk import nodes + edges
        db.upsert_embedding(node_id, embedding_vector)
        results = db.search_hybrid("authentication", embedding, k=20)
        db.close()
    """

    # ------------------------------------------------------------------
    # Construction / connection
    # ------------------------------------------------------------------

    def __init__(
        self,
        db_path: str | Path,
        embedding_dim: int = 1536,
        *,
        readonly: bool = False,
    ):
        """
        Args:
            db_path: Path to the ``.wiki.db`` file. Created if missing.
            embedding_dim: Dimension for the ``repo_vec`` table (1536 for
                platform embeddings, 384 for local MiniLM fallback).
            readonly: Open in read-only mode (``file:{path}?mode=ro``).
        """
        self.db_path = Path(db_path)
        self.embedding_dim = embedding_dim
        self._vec_available = False

        if readonly:
            uri = f"file:{self.db_path}?mode=ro"
            self.conn = sqlite3.connect(
                uri,
                uri=True,
                check_same_thread=False,
            )
        else:
            self.db_path.parent.mkdir(parents=True, exist_ok=True)
            self.conn = sqlite3.connect(
                str(self.db_path),
                check_same_thread=False,
            )

        self.conn.row_factory = sqlite3.Row  # dict-like rows
        self.conn.execute("PRAGMA journal_mode=WAL")
        self.conn.execute("PRAGMA synchronous=NORMAL")
        self.conn.execute("PRAGMA cache_size=-65536")  # 64 MB
        self.conn.execute("PRAGMA mmap_size=268435456")  # 256 MB
        self.conn.execute("PRAGMA foreign_keys=ON")

        self._load_extensions()
        if not readonly:
            self._migrate_schema()
            self._create_schema()

        logger.info(
            "UnifiedWikiDB opened: %s (vec=%s, dim=%d, readonly=%s)",
            self.db_path,
            self._vec_available,
            self.embedding_dim,
            readonly,
        )

    def _load_extensions(self) -> None:
        """Try to load sqlite-vec. Non-fatal if unavailable."""
        if not _SQLITE_VEC_AVAILABLE or not _EXTENSIONS_LOADABLE:
            logger.debug(
                "sqlite-vec not loadable (installed=%s, extensions=%s)",
                _SQLITE_VEC_AVAILABLE,
                _EXTENSIONS_LOADABLE,
            )
            return
        try:
            self.conn.enable_load_extension(True)
            sqlite_vec.load(self.conn)
            self.conn.enable_load_extension(False)
            self._vec_available = True
            logger.debug("sqlite-vec loaded successfully")
        except Exception as exc:
            logger.warning("Failed to load sqlite-vec: %s", exc)

    def _create_schema(self) -> None:
        """Create all tables and indexes (idempotent)."""
        cur = self.conn.cursor()
        cur.executescript(_SCHEMA_NODES)
        cur.executescript(_SCHEMA_NODES_INDEXES)
        cur.executescript(_SCHEMA_EDGES)
        cur.executescript(_SCHEMA_EDGES_INDEXES)
        cur.executescript(_SCHEMA_FTS5)
        cur.executescript(_SCHEMA_META)

        if self._vec_available:
            try:
                cur.executescript(_SCHEMA_VEC_TEMPLATE.format(dim=self.embedding_dim))
            except Exception as exc:
                logger.warning("Failed to create repo_vec table: %s", exc)
                self._vec_available = False

        self.conn.commit()

    def _migrate_schema(self) -> None:
        """Add columns/indexes that may be missing in pre-existing DBs.

        Called before ``_create_schema`` so that new indexes referencing
        new columns don't fail on legacy databases.
        """
        cur = self.conn.cursor()
        # Check whether repo_nodes table exists at all (fresh DB → skip)
        tables = {
            row[0]
            for row in cur.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            )
        }
        if "repo_nodes" not in tables:
            return  # fresh DB — _create_schema will create everything

        cols = {row[1] for row in cur.execute("PRAGMA table_info(repo_nodes)")}
        if "is_test" not in cols:
            logger.info("Migrating schema: adding is_test column to repo_nodes")
            cur.execute("ALTER TABLE repo_nodes ADD COLUMN is_test INTEGER DEFAULT 0")
            self.conn.commit()
            self._backfill_is_test()

    def _backfill_is_test(self) -> None:
        """Set is_test=1 for existing rows whose rel_path matches test patterns."""
        rows = self.conn.execute(
            "SELECT node_id, rel_path FROM repo_nodes WHERE is_test = 0 AND rel_path != ''"
        ).fetchall()
        updates = [(nid,) for nid, path in rows if _is_test_path(path)]
        if updates:
            self.conn.executemany(
                "UPDATE repo_nodes SET is_test = 1 WHERE node_id = ?", updates
            )
            self.conn.commit()
            logger.info("Backfilled is_test=1 for %d nodes", len(updates))

    # ------------------------------------------------------------------
    # Context manager / cleanup
    # ------------------------------------------------------------------

    def close(self) -> None:
        if self.conn:
            try:
                self.conn.execute("PRAGMA wal_checkpoint(TRUNCATE)")
            except Exception:  # noqa: S110
                pass
            self.conn.close()
            self.conn = None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
        return False

    # ══════════════════════════════════════════════════════════════════
    # NODE operations
    # ══════════════════════════════════════════════════════════════════

    def upsert_node(self, node_id: str, **attrs) -> None:
        """Insert or replace a single node."""
        self._upsert_nodes_batch([{"node_id": node_id, **attrs}])

    def upsert_nodes_batch(self, nodes: list[dict[str, Any]]) -> None:
        """Bulk upsert nodes. Each dict must contain ``node_id``."""
        self._upsert_nodes_batch(nodes)

    def _upsert_nodes_batch(self, nodes: list[dict[str, Any]]) -> None:
        """Internal bulk upsert using INSERT OR REPLACE."""
        if not nodes:
            return

        sql = """
        INSERT OR REPLACE INTO repo_nodes (
            node_id, rel_path, file_name, language,
            start_line, end_line,
            symbol_name, symbol_type, parent_symbol, analysis_level,
            source_text, docstring, signature, parameters, return_type,
            is_architectural, is_doc, is_test, chunk_type,
            macro_cluster, micro_cluster, is_hub, hub_assignment
        ) VALUES (
            :node_id, :rel_path, :file_name, :language,
            :start_line, :end_line,
            :symbol_name, :symbol_type, :parent_symbol, :analysis_level,
            :source_text, :docstring, :signature, :parameters, :return_type,
            :is_architectural, :is_doc, :is_test, :chunk_type,
            :macro_cluster, :micro_cluster, :is_hub, :hub_assignment
        )
        """

        rows = []
        for n in nodes:
            symbol_type = n.get("symbol_type", "")
            # Normalize SymbolType enum
            if hasattr(symbol_type, "value"):
                symbol_type = symbol_type.value
            symbol_type = str(symbol_type).lower()

            is_arch = (
                1
                if (
                    symbol_type in ARCHITECTURAL_SYMBOLS
                    or symbol_type in _LEGACY_DOC_TYPES
                    or _is_doc_symbol(symbol_type)
                )
                else 0
            )
            is_doc = 1 if _is_doc_symbol(symbol_type) else 0

            # Detect test nodes by path
            rel_path = n.get("rel_path", "")
            is_test = 1 if _is_test_path(rel_path) else 0

            # parameters can be a list — serialize to JSON
            params = n.get("parameters", "")
            if isinstance(params, (list, tuple)):
                params = json.dumps(params)

            rows.append(
                {
                    "node_id": n["node_id"],
                    "rel_path": rel_path,
                    "file_name": n.get("file_name", ""),
                    "language": n.get("language", ""),
                    "start_line": n.get("start_line", 0),
                    "end_line": n.get("end_line", 0),
                    "symbol_name": n.get("symbol_name", ""),
                    "symbol_type": symbol_type,
                    "parent_symbol": n.get("parent_symbol"),
                    "analysis_level": n.get("analysis_level", "comprehensive"),
                    "source_text": n.get("source_text", ""),
                    "docstring": n.get("docstring", ""),
                    "signature": n.get("signature", ""),
                    "parameters": params,
                    "return_type": n.get("return_type", ""),
                    "is_architectural": is_arch,
                    "is_doc": is_doc,
                    "is_test": is_test,
                    "chunk_type": n.get("chunk_type"),
                    "macro_cluster": n.get("macro_cluster"),
                    "micro_cluster": n.get("micro_cluster"),
                    "is_hub": n.get("is_hub", 0),
                    "hub_assignment": n.get("hub_assignment"),
                }
            )

        self.conn.executemany(sql, rows)

    def get_node(self, node_id: str) -> dict[str, Any] | None:
        """Fetch one node by ID. Returns dict or None."""
        row = self.conn.execute("SELECT * FROM repo_nodes WHERE node_id = ?", (node_id,)).fetchone()
        return dict(row) if row else None

    def get_nodes_by_path_prefix(self, prefix: str, limit: int = 500) -> list[dict[str, Any]]:
        """Get nodes whose ``rel_path`` starts with *prefix* (B-tree GLOB)."""
        rows = self.conn.execute(
            "SELECT * FROM repo_nodes WHERE rel_path GLOB ? LIMIT ?",
            (prefix + "*", limit),
        ).fetchall()
        return [dict(r) for r in rows]

    def get_nodes_by_cluster(self, macro: int, micro: int | None = None, limit: int = 1000) -> list[dict[str, Any]]:
        """Get all nodes in a macro (optionally micro) cluster."""
        if micro is not None:
            rows = self.conn.execute(
                "SELECT * FROM repo_nodes WHERE macro_cluster = ? AND micro_cluster = ? LIMIT ?",
                (macro, micro, limit),
            ).fetchall()
        else:
            rows = self.conn.execute(
                "SELECT * FROM repo_nodes WHERE macro_cluster = ? LIMIT ?",
                (macro, limit),
            ).fetchall()
        return [dict(r) for r in rows]

    def get_architectural_nodes(self, limit: int = 5000) -> list[dict[str, Any]]:
        """Get all architectural-level nodes (classes, functions, etc.)."""
        rows = self.conn.execute(
            "SELECT * FROM repo_nodes WHERE is_architectural = 1 LIMIT ?",
            (limit,),
        ).fetchall()
        return [dict(r) for r in rows]

    def node_count(self) -> int:
        return self.conn.execute("SELECT count(*) FROM repo_nodes").fetchone()[0]

    # ══════════════════════════════════════════════════════════════════
    # EDGE operations
    # ══════════════════════════════════════════════════════════════════

    def upsert_edge(self, source_id: str, target_id: str, rel_type: str, **attrs) -> None:
        """Insert a single edge."""
        self._upsert_edges_batch(
            [
                {
                    "source_id": source_id,
                    "target_id": target_id,
                    "rel_type": rel_type,
                    **attrs,
                }
            ]
        )

    def upsert_edges_batch(self, edges: list[dict[str, Any]]) -> None:
        """Bulk insert edges."""
        self._upsert_edges_batch(edges)

    def _upsert_edges_batch(self, edges: list[dict[str, Any]]) -> None:
        if not edges:
            return

        sql = """
        INSERT INTO repo_edges (
            source_id, target_id, rel_type, edge_class, analysis_level,
            weight, raw_similarity,
            source_file, target_file, language, annotations, created_by
        ) VALUES (
            :source_id, :target_id, :rel_type, :edge_class, :analysis_level,
            :weight, :raw_similarity,
            :source_file, :target_file, :language, :annotations, :created_by
        )
        """

        rows = []
        for e in edges:
            annotations = e.get("annotations", "")
            if isinstance(annotations, dict):
                annotations = json.dumps(annotations)

            rows.append(
                {
                    "source_id": e["source_id"],
                    "target_id": e["target_id"],
                    "rel_type": e.get("rel_type", ""),
                    "edge_class": e.get("edge_class", "structural"),
                    "analysis_level": e.get("analysis_level", "comprehensive"),
                    "weight": e.get("weight", 1.0),
                    "raw_similarity": e.get("raw_similarity"),
                    "source_file": e.get("source_file", ""),
                    "target_file": e.get("target_file", ""),
                    "language": e.get("language", ""),
                    "annotations": annotations,
                    "created_by": e.get("created_by", "ast"),
                }
            )

        self.conn.executemany(sql, rows)

    def get_edges_from(self, node_id: str, rel_types: list[str] | None = None) -> list[dict[str, Any]]:
        """Get outgoing edges from a node, optionally filtered by type."""
        if rel_types:
            placeholders = ",".join("?" * len(rel_types))
            rows = self.conn.execute(
                f"SELECT * FROM repo_edges WHERE source_id = ? AND rel_type IN ({placeholders})",  # noqa: S608 — placeholders are parameterized via ?, no user input in SQL structure
                [node_id] + rel_types,
            ).fetchall()
        else:
            rows = self.conn.execute("SELECT * FROM repo_edges WHERE source_id = ?", (node_id,)).fetchall()
        return [dict(r) for r in rows]

    def get_edges_to(self, node_id: str) -> list[dict[str, Any]]:
        """Get incoming edges to a node."""
        rows = self.conn.execute("SELECT * FROM repo_edges WHERE target_id = ?", (node_id,)).fetchall()
        return [dict(r) for r in rows]

    def get_neighbors(self, node_id: str, hops: int = 1, direction: str = "out") -> set[str]:
        """Get neighbor node IDs within *hops* distance."""
        visited: set[str] = set()
        frontier = {node_id}

        for _ in range(hops):
            next_frontier: set[str] = set()
            for nid in frontier:
                if direction in ("out", "both"):
                    for row in self.conn.execute(
                        "SELECT target_id FROM repo_edges WHERE source_id = ?",
                        (nid,),
                    ):
                        next_frontier.add(row[0])
                if direction in ("in", "both"):
                    for row in self.conn.execute(
                        "SELECT source_id FROM repo_edges WHERE target_id = ?",
                        (nid,),
                    ):
                        next_frontier.add(row[0])
            visited |= frontier
            frontier = next_frontier - visited

        visited |= frontier
        visited.discard(node_id)
        return visited

    def edge_count(self) -> int:
        return self.conn.execute("SELECT count(*) FROM repo_edges").fetchone()[0]

    def update_edge_weights_batch(self, updates: list[dict[str, Any]]) -> int:
        """Batch-update edge weights. Each dict needs ``id`` and ``weight``.

        Returns number of rows actually updated.
        """
        if not updates:
            return 0

        sql = "UPDATE repo_edges SET weight = :weight WHERE id = :id"
        cur = self.conn.executemany(sql, updates)
        self.conn.commit()
        return cur.rowcount

    # ══════════════════════════════════════════════════════════════════
    # FTS5 search
    # ══════════════════════════════════════════════════════════════════

    def _populate_fts5(self) -> None:
        """Populate FTS5 table from repo_nodes (full rebuild)."""
        self.conn.execute("DELETE FROM repo_fts")
        self.conn.execute("""
            INSERT INTO repo_fts (node_id, symbol_name, signature, docstring, source_text)
            SELECT node_id, symbol_name, signature, docstring, source_text
            FROM repo_nodes
        """)
        self.conn.commit()
        logger.debug("FTS5 index rebuilt (%d rows)", self.node_count())

    def search_fts5(
        self,
        query: str,
        path_prefix: str | None = None,
        cluster_id: int | None = None,
        symbol_types: list[str] | None = None,
        limit: int = 20,
    ) -> list[dict[str, Any]]:
        """BM25-ranked text search with optional path/cluster/type filtering.

        Returns list of dicts with node metadata + ``fts_rank`` score.
        """
        if not query or not query.strip():
            return []

        # Sanitize query for FTS5 (escape double quotes, wrap in quotes)
        safe_query = query.replace('"', '""')

        conditions = ["repo_fts MATCH ?"]
        params: list = [safe_query]

        if path_prefix:
            conditions.append("m.rel_path GLOB ?")
            params.append(path_prefix.rstrip("/") + "/*" if not path_prefix.endswith("*") else path_prefix)

        if cluster_id is not None:
            conditions.append("m.macro_cluster = ?")
            params.append(cluster_id)

        if symbol_types:
            placeholders = ",".join("?" * len(symbol_types))
            conditions.append(f"m.symbol_type IN ({placeholders})")
            params.extend(symbol_types)

        params.append(limit)
        where = " AND ".join(conditions)

        sql = (
            "SELECT m.*, f.rank AS fts_rank "  # noqa: S608 — FTS search query, conditions built from validated field names
            "FROM repo_fts f "
            "JOIN repo_nodes m ON f.node_id = m.node_id "
            f"WHERE {where} "
            "ORDER BY f.rank "
            "LIMIT ?"
        )

        try:
            rows = self.conn.execute(sql, params).fetchall()
            return [dict(r) for r in rows]
        except sqlite3.OperationalError as exc:
            # FTS5 query syntax errors — fall back to prefix search
            logger.debug("FTS5 query failed (%s), trying prefix match", exc)
            safe_query = safe_query.strip() + "*"
            params[0] = safe_query
            try:
                rows = self.conn.execute(sql, params).fetchall()
                return [dict(r) for r in rows]
            except sqlite3.OperationalError:
                return []

    # ══════════════════════════════════════════════════════════════════
    # VECTOR search (sqlite-vec)
    # ══════════════════════════════════════════════════════════════════

    def upsert_embedding(self, node_id: str, embedding: list[float]) -> None:
        """Insert or replace a single embedding."""
        if not self._vec_available:
            return
        blob = _serialize_float32_vec(embedding)
        self.conn.execute(
            "INSERT OR REPLACE INTO repo_vec (node_id, embedding) VALUES (?, ?)",
            (node_id, blob),
        )

    def upsert_embeddings_batch(self, embeddings: list[tuple[str, list[float]]]) -> None:
        """Bulk insert embeddings as (node_id, vector) pairs."""
        if not self._vec_available or not embeddings:
            return
        sql = "INSERT OR REPLACE INTO repo_vec (node_id, embedding) VALUES (?, ?)"
        rows = [(nid, _serialize_float32_vec(vec)) for nid, vec in embeddings]
        self.conn.executemany(sql, rows)

    def populate_embeddings(
        self,
        embedding_fn: Callable[[list[str]], list[list[float]]],
        batch_size: int = 64,
    ) -> int:
        """Embed all nodes' text and populate the ``repo_vec`` table.

        Uses ``embed_documents`` (batch API) for throughput.  Nodes whose
        ``source_text`` is empty or whitespace-only are skipped.

        Parameters
        ----------
        embedding_fn : callable
            ``(List[str]) -> List[List[float]]``  — batch text embedder.
            Typically ``embeddings.embed_documents``.
        batch_size : int
            Number of texts per embedding API call (default 64).

        Returns
        -------
        int
            Number of embeddings stored.
        """
        if not self._vec_available:
            logger.warning("populate_embeddings: sqlite-vec not available, skipping")
            return 0

        # SQLite trim() only strips spaces; use explicit whitespace chars
        _ws = " ' ' || char(9) || char(10) || char(13) "
        rows = self.conn.execute(
            "SELECT node_id, source_text FROM repo_nodes "  # noqa: S608 — internal whitespace-trim query, _ws is a hardcoded SQLite expression
            f"WHERE source_text IS NOT NULL AND length(trim(source_text, {_ws})) > 0"
        ).fetchall()

        total = 0
        for i in range(0, len(rows), batch_size):
            batch = rows[i : i + batch_size]
            texts = [r["source_text"] for r in batch]
            node_ids = [r["node_id"] for r in batch]

            try:
                vectors = embedding_fn(texts)
            except Exception as exc:
                logger.warning(
                    "populate_embeddings: batch %d–%d failed: %s",
                    i,
                    i + len(batch),
                    exc,
                )
                continue

            pairs = list(zip(node_ids, vectors, strict=True))
            self.upsert_embeddings_batch(pairs)
            total += len(pairs)

        self.conn.commit()
        logger.info("populate_embeddings: %d/%d nodes embedded", total, len(rows))
        return total

    def search_vec(
        self,
        embedding: list[float],
        k: int = 10,
        path_prefix: str | None = None,
        cluster_id: int | None = None,
    ) -> list[dict[str, Any]]:
        """KNN vector search with optional metadata filtering.

        Returns list of dicts with node metadata + ``vec_distance``.
        """
        if not self._vec_available:
            return []

        blob = _serialize_float32_vec(embedding)

        # sqlite-vec KNN base query
        conditions = []
        params: list = []

        if path_prefix:
            conditions.append("m.rel_path GLOB ?")
            params.append(path_prefix.rstrip("/") + "/*" if not path_prefix.endswith("*") else path_prefix)

        if cluster_id is not None:
            conditions.append("m.macro_cluster = ?")
            params.append(cluster_id)

        # Build query — vec search is the core, metadata filters via JOIN
        where_extra = (" AND " + " AND ".join(conditions)) if conditions else ""

        sql = (
            "SELECT v.node_id, v.distance AS vec_distance, m.* "  # noqa: S608 — vec similarity query, conditions built from validated field names
            "FROM repo_vec v "
            "JOIN repo_nodes m ON v.node_id = m.node_id "
            f"WHERE v.embedding MATCH ? AND k = ? {where_extra}"
        )

        try:
            all_params = [blob, k] + params
            rows = self.conn.execute(sql, all_params).fetchall()
            return [dict(r) for r in rows]
        except Exception as exc:
            logger.warning("Vector search failed: %s", exc)
            return []

    @property
    def vec_available(self) -> bool:
        """Whether sqlite-vec is loaded and repo_vec table exists."""
        return self._vec_available

    # ══════════════════════════════════════════════════════════════════
    # HYBRID search (RRF fusion)
    # ══════════════════════════════════════════════════════════════════

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
        """Hybrid BM25 + Vector search with Reciprocal Rank Fusion.

        Falls back to FTS5-only when vectors are unavailable.

        Args:
            query: Text query for FTS5
            embedding: Vector for KNN (if None, FTS5-only)
            path_prefix: Restrict to paths matching this prefix
            cluster_id: Restrict to this macro-cluster ID
            fts_weight: RRF weight for FTS5 results
            vec_weight: RRF weight for vector results
            limit: Max results to return
            fts_k: Internal FTS5 candidate pool size
            vec_k: Internal vector candidate pool size

        Returns:
            List of dicts with node metadata + ``combined_score``.
        """
        rrf_constant = 60.0  # Standard RRF constant

        # --- FTS5 pass ---
        fts_results = self.search_fts5(query, path_prefix=path_prefix, cluster_id=cluster_id, limit=fts_k)
        fts_scores: dict[str, float] = {}
        fts_nodes: dict[str, dict] = {}
        for rank, r in enumerate(fts_results):
            nid = r["node_id"]
            fts_scores[nid] = fts_weight / (rrf_constant + rank + 1)
            fts_nodes[nid] = r

        # --- Vector pass ---
        vec_scores: dict[str, float] = {}
        vec_nodes: dict[str, dict] = {}
        if embedding and self._vec_available:
            vec_results = self.search_vec(embedding, k=vec_k, path_prefix=path_prefix, cluster_id=cluster_id)
            for rank, r in enumerate(vec_results):
                nid = r["node_id"]
                vec_scores[nid] = vec_weight / (rrf_constant + rank + 1)
                vec_nodes[nid] = r

        # --- Fuse ---
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

    # ══════════════════════════════════════════════════════════════════
    # CLUSTER operations (Phase 3 — stubs for now)
    # ══════════════════════════════════════════════════════════════════

    def set_cluster(self, node_id: str, macro: int, micro: int | None = None) -> None:
        """Assign cluster IDs to a node."""
        self.conn.execute(
            "UPDATE repo_nodes SET macro_cluster = ?, micro_cluster = ? WHERE node_id = ?",
            (macro, micro, node_id),
        )

    def set_clusters_batch(self, assignments: list[tuple[str, int, int | None]]) -> None:
        """Batch-assign clusters: list of (node_id, macro, micro) tuples."""
        if not assignments:
            return
        self.conn.executemany(
            "UPDATE repo_nodes SET macro_cluster = ?, micro_cluster = ? WHERE node_id = ?",
            [(macro, micro, nid) for nid, macro, micro in assignments],
        )

    def set_hub(self, node_id: str, is_hub: bool = True, assignment: str | None = None) -> None:
        """Flag a node as a hub with optional assignment."""
        self.conn.execute(
            "UPDATE repo_nodes SET is_hub = ?, hub_assignment = ? WHERE node_id = ?",
            (1 if is_hub else 0, assignment, node_id),
        )

    def get_cluster_nodes(self, macro: int, micro: int | None = None) -> list[str]:
        """Return node IDs in a cluster."""
        return [n["node_id"] for n in self.get_nodes_by_cluster(macro, micro)]

    def get_all_clusters(self) -> dict[int, dict[int, list[str]]]:
        """Return nested dict: macro_id → micro_id → [node_ids].

        Useful for iterating over all clusters in Phase 4 (naming).
        """
        rows = self.conn.execute(
            "SELECT node_id, macro_cluster, micro_cluster FROM repo_nodes WHERE macro_cluster IS NOT NULL"
        ).fetchall()

        result: dict[int, dict[int, list[str]]] = {}
        for row in rows:
            macro = row["macro_cluster"]
            micro = row["micro_cluster"] or 0
            result.setdefault(macro, {}).setdefault(micro, []).append(row["node_id"])

        return result

    # ══════════════════════════════════════════════════════════════════
    # GRAPH I/O — NetworkX conversion
    # ══════════════════════════════════════════════════════════════════

    def from_networkx(self, G: nx.MultiDiGraph) -> None:
        """Bulk-import a NetworkX MultiDiGraph into the unified DB.

        This is the primary Phase 1 ingest path: the graph builder produces
        an ``nx.MultiDiGraph``, and we mirror it into SQLite for unified
        search and persistence.

        Performance: Uses ``executemany`` with batching (5000 rows) and
        deferred FTS5 population for speed.
        """
        if G is None or G.number_of_nodes() == 0:
            logger.warning("from_networkx: empty graph, nothing to import")
            return

        t0 = time.time()
        logger.info(
            "Importing NetworkX graph: %d nodes, %d edges",
            G.number_of_nodes(),
            G.number_of_edges(),
        )

        # --- Nodes ---
        BATCH = 5000
        node_batch: list[dict[str, Any]] = []

        for node_id, data in G.nodes(data=True):
            node_dict = self._nx_node_to_dict(node_id, data)
            node_batch.append(node_dict)

            if len(node_batch) >= BATCH:
                self._upsert_nodes_batch(node_batch)
                node_batch.clear()

        if node_batch:
            self._upsert_nodes_batch(node_batch)

        self.conn.commit()
        t_nodes = time.time() - t0
        logger.info("Imported %d nodes in %.1fs", G.number_of_nodes(), t_nodes)

        # --- Edges ---
        t1 = time.time()
        edge_batch: list[dict[str, Any]] = []

        for u, v, key, data in G.edges(data=True, keys=True):
            edge_dict = self._nx_edge_to_dict(u, v, key, data)
            edge_batch.append(edge_dict)

            if len(edge_batch) >= BATCH:
                self._upsert_edges_batch(edge_batch)
                edge_batch.clear()

        if edge_batch:
            self._upsert_edges_batch(edge_batch)

        self.conn.commit()
        t_edges = time.time() - t1
        logger.info("Imported %d edges in %.1fs", G.number_of_edges(), t_edges)

        # --- FTS5 rebuild (after all nodes inserted) ---
        t2 = time.time()
        self._populate_fts5()
        t_fts = time.time() - t2
        logger.info("FTS5 index rebuilt in %.1fs", t_fts)

        # --- Stale-node cleanup ---
        imported_ids = {str(n) for n in G.nodes()}
        existing_ids = {
            row[0]
            for row in self.conn.execute("SELECT node_id FROM repo_nodes").fetchall()
        }
        stale_ids = existing_ids - imported_ids
        if stale_ids:
            placeholders = ",".join("?" * len(stale_ids))
            stale_list = list(stale_ids)
            self.conn.execute(
                f"DELETE FROM repo_edges WHERE source_id IN ({placeholders}) OR target_id IN ({placeholders})",  # noqa: S608
                stale_list + stale_list,
            )
            self.conn.execute(
                f"DELETE FROM repo_nodes WHERE node_id IN ({placeholders})",  # noqa: S608
                stale_list,
            )
            self.conn.commit()
            logger.info("Removed %d stale nodes and their edges", len(stale_ids))

        total = time.time() - t0
        logger.info(
            "from_networkx complete: %d nodes + %d edges in %.1fs",
            G.number_of_nodes(),
            G.number_of_edges(),
            total,
        )

    def _nx_node_to_dict(self, node_id: str, data: dict) -> dict[str, Any]:
        """Convert a NetworkX node to a dict suitable for ``_upsert_nodes_batch``."""
        # Rich parser nodes store source_text on the Symbol object
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
        """Convert a NetworkX edge to a dict suitable for ``_upsert_edges_batch``."""
        annotations = data.get("annotations", {})
        if isinstance(annotations, dict):
            annotations = json.dumps(annotations)

        return {
            "source_id": str(u),
            "target_id": str(v),
            "rel_type": data.get("relationship_type", ""),
            "edge_class": data.get("edge_class", "structural"),
            "analysis_level": data.get("analysis_level", "comprehensive"),
            "weight": data.get("weight", 1.0),
            "raw_similarity": data.get("raw_similarity"),
            "source_file": data.get("source_file", ""),
            "target_file": data.get("target_file", ""),
            "language": data.get("language", ""),
            "annotations": annotations,
            "created_by": data.get("created_by", "ast"),
        }

    def to_networkx(self) -> nx.MultiDiGraph:
        """Reconstruct a NetworkX MultiDiGraph from the DB.

        Used when graph algorithms (Louvain, expansion engine) need an
        in-memory graph. Nodes carry all metadata; edges carry all attributes.
        """
        G = nx.MultiDiGraph()
        G.graph["analysis_type"] = "multi_tier"
        G.graph["source"] = "unified_db"

        # --- Load nodes ---
        rows = self.conn.execute("SELECT * FROM repo_nodes").fetchall()
        for row in rows:
            d = dict(row)
            nid = d.pop("node_id")
            # Deserialize parameters back to list
            params = d.get("parameters", "")
            if params and isinstance(params, str):
                try:
                    d["parameters"] = json.loads(params)
                except (json.JSONDecodeError, TypeError):
                    pass
            G.add_node(nid, **d)

        # --- Load edges ---
        rows = self.conn.execute("SELECT * FROM repo_edges").fetchall()
        for row in rows:
            d = dict(row)
            src = d.pop("source_id")
            tgt = d.pop("target_id")
            d.pop("id", None)
            # Deserialize annotations
            ann = d.get("annotations", "")
            if ann and isinstance(ann, str):
                try:
                    d["annotations"] = json.loads(ann)
                except (json.JSONDecodeError, TypeError):
                    pass
            # Map rel_type back to relationship_type for compatibility
            d["relationship_type"] = d.pop("rel_type", "")
            G.add_edge(src, tgt, **d)

        logger.info(
            "to_networkx: reconstructed %d nodes, %d edges",
            G.number_of_nodes(),
            G.number_of_edges(),
        )
        return G

    # ══════════════════════════════════════════════════════════════════
    # METADATA
    # ══════════════════════════════════════════════════════════════════

    def set_meta(self, key: str, value: Any) -> None:
        """Store a metadata key-value pair (JSON-encoded)."""
        self.conn.execute(
            "INSERT OR REPLACE INTO wiki_meta (key, value) VALUES (?, ?)",
            (key, json.dumps(value)),
        )
        self.conn.commit()

    def get_meta(self, key: str, default: Any = None) -> Any:
        """Retrieve a metadata value. Returns *default* if not found."""
        row = self.conn.execute("SELECT value FROM wiki_meta WHERE key = ?", (key,)).fetchone()
        if row is None:
            return default
        try:
            return json.loads(row[0])
        except (json.JSONDecodeError, TypeError):
            return row[0]

    # ══════════════════════════════════════════════════════════════════
    # STATISTICS / debugging
    # ══════════════════════════════════════════════════════════════════

    def stats(self) -> dict[str, Any]:
        """Return summary statistics about the DB contents."""
        n_nodes = self.node_count()
        n_edges = self.edge_count()

        # Language distribution
        langs = {}
        for row in self.conn.execute("SELECT language, count(*) FROM repo_nodes GROUP BY language"):
            langs[row[0]] = row[1]

        # Symbol type distribution
        types = {}
        for row in self.conn.execute("SELECT symbol_type, count(*) FROM repo_nodes GROUP BY symbol_type"):
            types[row[0]] = row[1]

        # Edge type distribution
        edge_types = {}
        for row in self.conn.execute("SELECT rel_type, count(*) FROM repo_edges GROUP BY rel_type"):
            edge_types[row[0]] = row[1]

        # Cluster counts
        macro_count = self.conn.execute(
            "SELECT count(DISTINCT macro_cluster) FROM repo_nodes WHERE macro_cluster IS NOT NULL"
        ).fetchone()[0]

        hub_count = self.conn.execute("SELECT count(*) FROM repo_nodes WHERE is_hub = 1").fetchone()[0]

        # DB file size
        db_size_mb = 0
        if self.db_path.exists():
            db_size_mb = self.db_path.stat().st_size / (1024 * 1024)

        return {
            "node_count": n_nodes,
            "edge_count": n_edges,
            "languages": langs,
            "symbol_types": types,
            "edge_types": edge_types,
            "macro_clusters": macro_count,
            "hub_count": hub_count,
            "vec_available": self._vec_available,
            "embedding_dim": self.embedding_dim,
            "db_size_mb": round(db_size_mb, 2),
        }
