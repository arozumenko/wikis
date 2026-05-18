"""Phase 7 — Project graph storage.

Per-project storage of cross-repo nodes/edges, repo↔repo relatedness,
and project metadata. Backend-agnostic via :class:`ProjectStorageProtocol`;
this module ships in-memory, SQLite, and PostgreSQL implementations that
satisfy the same contract.

All write paths are no-ops when ``flags.project_graph`` is ``False``,
but the **read** paths still return empty lists/None so callers can
treat the protocol as always-available.
"""

from __future__ import annotations

import json
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Iterable, Protocol, runtime_checkable

__all__ = [
    "ProjectStorageProtocol",
    "InMemoryProjectStorage",
    "SqliteProjectStorage",
    "PostgresProjectStorage",
    "open_project_storage",
    "drop_project_storage",
]


@runtime_checkable
class ProjectStorageProtocol(Protocol):
    """Cross-repo project storage surface.

    Implementations MUST be safe to call with ``WIKI_PROJECT_GRAPH=0``;
    in that case ``upsert_*`` should be cheap no-ops and ``get_*`` should
    return empty containers.
    """

    def upsert_project_node(
        self,
        project_id: str,
        node_id: str,
        wiki_id: str,
        attrs: dict[str, Any],
    ) -> None:
        """Upsert a project-scoped node row."""

    def upsert_project_edge(
        self,
        project_id: str,
        source_node_id: str,
        target_node_id: str,
        edge_class: str,
        weight: float,
        provenance: dict[str, Any] | None = None,
    ) -> None:
        """Upsert a project-scoped edge row."""

    def get_project_nodes(self, project_id: str) -> list[dict[str, Any]]:
        """Return all project nodes."""

    def clear_project_nodes(self, project_id: str) -> None:
        """Delete all project-scoped nodes."""

    def get_project_edges(
        self,
        project_id: str,
        *,
        source_node_id: str | None = None,
        target_node_id: str | None = None,
        edge_class: str | None = None,
    ) -> list[dict[str, Any]]:
        """Return project edges, optionally filtered by source, target, and class."""

    def clear_project_edges(self, project_id: str) -> None:
        """Delete all project-scoped edges."""

    def upsert_repo_relatedness(
        self,
        project_id: str,
        wiki_a: str,
        wiki_b: str,
        score: float,
        breakdown: dict[str, Any] | None = None,
    ) -> None:
        """Upsert a (wiki_a, wiki_b) relatedness row (order-insensitive)."""

    def get_repo_relatedness(self, project_id: str) -> list[dict[str, Any]]:
        """Return all relatedness rows for a project."""

    def clear_repo_relatedness(self, project_id: str) -> None:
        """Delete all repo relatedness rows for a project."""

    def set_project_meta(self, project_id: str, key: str, value: Any) -> None:
        """Persist a small JSON-serializable metadata blob."""

    def get_project_meta(self, project_id: str, key: str) -> Any:
        """Return previously-set metadata, or ``None`` if missing."""


# ----------------------------------------------------------------------
# In-memory implementation
# ----------------------------------------------------------------------


def _pair_key(a: str, b: str) -> tuple[str, str]:
    """Order-insensitive key for ``(wiki_a, wiki_b)``."""
    return (a, b) if a <= b else (b, a)


@dataclass
class _ProjectStore:
    nodes: dict[str, dict[str, Any]] = field(default_factory=dict)
    # Edges keyed by (src, tgt, edge_class) so duplicates collapse.
    edges: dict[tuple[str, str, str], dict[str, Any]] = field(default_factory=dict)
    relatedness: dict[tuple[str, str], dict[str, Any]] = field(default_factory=dict)
    meta: dict[str, Any] = field(default_factory=dict)


class InMemoryProjectStorage:
    """Reference :class:`ProjectStorageProtocol` implementation.

    Backed by a per-project in-memory store. Used by tests and as the
    default until SQLite/Postgres backends ship.
    """

    def __init__(self) -> None:
        self._stores: dict[str, _ProjectStore] = defaultdict(_ProjectStore)

    # ------------------------------------------------------------------
    # Nodes
    # ------------------------------------------------------------------

    def upsert_project_node(
        self,
        project_id: str,
        node_id: str,
        wiki_id: str,
        attrs: dict[str, Any],
    ) -> None:
        if not project_id or not node_id:
            return
        row = {"node_id": node_id, "wiki_id": wiki_id, **(attrs or {})}
        self._stores[project_id].nodes[node_id] = row

    def get_project_nodes(self, project_id: str) -> list[dict[str, Any]]:
        store = self._stores.get(project_id)
        if store is None:
            return []
        return list(store.nodes.values())

    def clear_project_nodes(self, project_id: str) -> None:
        store = self._stores.get(project_id)
        if store is not None:
            store.nodes.clear()

    # ------------------------------------------------------------------
    # Edges
    # ------------------------------------------------------------------

    def upsert_project_edge(
        self,
        project_id: str,
        source_node_id: str,
        target_node_id: str,
        edge_class: str,
        weight: float,
        provenance: dict[str, Any] | None = None,
    ) -> None:
        if not project_id or not source_node_id or not target_node_id:
            return
        key = (source_node_id, target_node_id, edge_class)
        self._stores[project_id].edges[key] = {
            "source_node_id": source_node_id,
            "target_node_id": target_node_id,
            "edge_class": edge_class,
            "weight": float(weight),
            "provenance": dict(provenance or {}),
        }

    def get_project_edges(
        self,
        project_id: str,
        *,
        source_node_id: str | None = None,
        target_node_id: str | None = None,
        edge_class: str | None = None,
    ) -> list[dict[str, Any]]:
        store = self._stores.get(project_id)
        if store is None:
            return []
        rows: list[dict[str, Any]] = []
        for (src, tgt, cls), row in store.edges.items():
            if source_node_id is not None and src != source_node_id:
                continue
            if target_node_id is not None and tgt != target_node_id:
                continue
            if edge_class is not None and cls != edge_class:
                continue
            rows.append(dict(row))
        return rows

    def clear_project_edges(self, project_id: str) -> None:
        store = self._stores.get(project_id)
        if store is not None:
            store.edges.clear()

    # ------------------------------------------------------------------
    # Relatedness
    # ------------------------------------------------------------------

    def upsert_repo_relatedness(
        self,
        project_id: str,
        wiki_a: str,
        wiki_b: str,
        score: float,
        breakdown: dict[str, Any] | None = None,
    ) -> None:
        if not project_id or not wiki_a or not wiki_b or wiki_a == wiki_b:
            return
        key = _pair_key(wiki_a, wiki_b)
        self._stores[project_id].relatedness[key] = {
            "wiki_a": key[0],
            "wiki_b": key[1],
            "score": float(score),
            "breakdown": dict(breakdown or {}),
        }

    def get_repo_relatedness(self, project_id: str) -> list[dict[str, Any]]:
        store = self._stores.get(project_id)
        if store is None:
            return []
        return [dict(r) for r in store.relatedness.values()]

    def clear_repo_relatedness(self, project_id: str) -> None:
        store = self._stores.get(project_id)
        if store is not None:
            store.relatedness.clear()

    # ------------------------------------------------------------------
    # Meta
    # ------------------------------------------------------------------

    def set_project_meta(self, project_id: str, key: str, value: Any) -> None:
        if not project_id or not key:
            return
        # Round-trip through json so callers don't accidentally hand us
        # objects we cannot persist later in a real backend.
        try:
            json.dumps(value)
        except (TypeError, ValueError):
            return
        self._stores[project_id].meta[key] = value

    def get_project_meta(self, project_id: str, key: str) -> Any:
        store = self._stores.get(project_id)
        if store is None:
            return None
        return store.meta.get(key)


# ----------------------------------------------------------------------
# SQLite implementation
# ----------------------------------------------------------------------


import logging  # noqa: E402
import os  # noqa: E402
import sqlite3  # noqa: E402
import re  # noqa: E402

logger = logging.getLogger(__name__)


_SQLITE_DDL = """
CREATE TABLE IF NOT EXISTS project_nodes (
    node_id    TEXT PRIMARY KEY,
    wiki_id    TEXT NOT NULL DEFAULT '',
    attrs      TEXT DEFAULT NULL
);

CREATE TABLE IF NOT EXISTS project_edges (
    source_id   TEXT NOT NULL,
    target_id   TEXT NOT NULL,
    edge_class  TEXT NOT NULL,
    weight      REAL DEFAULT 1.0,
    provenance  TEXT DEFAULT NULL,
    PRIMARY KEY (source_id, target_id, edge_class)
);
CREATE INDEX IF NOT EXISTS idx_project_edges_src
    ON project_edges(source_id);
CREATE INDEX IF NOT EXISTS idx_project_edges_tgt
    ON project_edges(target_id);
CREATE INDEX IF NOT EXISTS idx_project_edges_class
    ON project_edges(edge_class);

CREATE TABLE IF NOT EXISTS project_relatedness (
    wiki_a    TEXT NOT NULL,
    wiki_b    TEXT NOT NULL,
    score     REAL DEFAULT 0.0,
    breakdown TEXT DEFAULT NULL,
    PRIMARY KEY (wiki_a, wiki_b)
);

CREATE TABLE IF NOT EXISTS project_meta (
    key   TEXT PRIMARY KEY,
    value TEXT
);
"""


class SqliteProjectStorage:
    """SQLite-backed :class:`ProjectStorageProtocol` implementation.

    Each project has its own database file (no project_id column —
    isolation comes from the file path). Designed to be cheap to open
    multiple times against the same path; the connection is private to
    the instance.
    """

    def __init__(self, db_path: str | os.PathLike[str]) -> None:
        self._db_path = str(db_path)
        os.makedirs(os.path.dirname(self._db_path) or ".", exist_ok=True)
        self._conn = sqlite3.connect(self._db_path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._conn.executescript(_SQLITE_DDL)
        self._conn.commit()

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------
    def close(self) -> None:
        try:
            self._conn.close()
        except Exception:  # pragma: no cover
            pass

    # ------------------------------------------------------------------
    # Nodes
    # ------------------------------------------------------------------
    def upsert_project_node(
        self,
        project_id: str,
        node_id: str,
        wiki_id: str,
        attrs: dict[str, Any],
    ) -> None:
        if not project_id or not node_id:
            return
        self._conn.execute(
            "INSERT OR REPLACE INTO project_nodes "
            "(node_id, wiki_id, attrs) VALUES (?, ?, ?)",
            (node_id, wiki_id or "", json.dumps(attrs or {})),
        )
        self._conn.commit()

    def get_project_nodes(self, project_id: str) -> list[dict[str, Any]]:
        rows = self._conn.execute(
            "SELECT node_id, wiki_id, attrs FROM project_nodes"
        ).fetchall()
        out: list[dict[str, Any]] = []
        for r in rows:
            try:
                attrs = json.loads(r["attrs"]) if r["attrs"] else {}
            except (TypeError, ValueError):
                attrs = {}
            out.append({"node_id": r["node_id"], "wiki_id": r["wiki_id"], **attrs})
        return out

    def clear_project_nodes(self, project_id: str) -> None:
        if not project_id:
            return
        self._conn.execute("DELETE FROM project_nodes")
        self._conn.commit()

    # ------------------------------------------------------------------
    # Edges
    # ------------------------------------------------------------------
    def upsert_project_edge(
        self,
        project_id: str,
        source_node_id: str,
        target_node_id: str,
        edge_class: str,
        weight: float,
        provenance: dict[str, Any] | None = None,
    ) -> None:
        if not project_id or not source_node_id or not target_node_id:
            return
        self._conn.execute(
            "INSERT OR REPLACE INTO project_edges "
            "(source_id, target_id, edge_class, weight, provenance) "
            "VALUES (?, ?, ?, ?, ?)",
            (
                source_node_id,
                target_node_id,
                edge_class or "",
                float(weight),
                json.dumps(provenance) if provenance else None,
            ),
        )
        self._conn.commit()

    def get_project_edges(
        self,
        project_id: str,
        *,
        source_node_id: str | None = None,
        target_node_id: str | None = None,
        edge_class: str | None = None,
    ) -> list[dict[str, Any]]:
        sql = (
            "SELECT source_id, target_id, edge_class, weight, provenance "
            "FROM project_edges WHERE 1=1"
        )
        params: list[Any] = []
        if source_node_id is not None:
            sql += " AND source_id = ?"
            params.append(source_node_id)
        if target_node_id is not None:
            sql += " AND target_id = ?"
            params.append(target_node_id)
        if edge_class is not None:
            sql += " AND edge_class = ?"
            params.append(edge_class)
        rows = self._conn.execute(sql, params).fetchall()
        out: list[dict[str, Any]] = []
        for r in rows:
            try:
                prov = json.loads(r["provenance"]) if r["provenance"] else {}
            except (TypeError, ValueError):
                prov = {}
            out.append(
                {
                    "source_node_id": r["source_id"],
                    "target_node_id": r["target_id"],
                    "edge_class": r["edge_class"],
                    "weight": float(r["weight"]),
                    "provenance": prov,
                }
            )
        return out

    def clear_project_edges(self, project_id: str) -> None:
        if not project_id:
            return
        self._conn.execute("DELETE FROM project_edges")
        self._conn.commit()

    # ------------------------------------------------------------------
    # Relatedness
    # ------------------------------------------------------------------
    def upsert_repo_relatedness(
        self,
        project_id: str,
        wiki_a: str,
        wiki_b: str,
        score: float,
        breakdown: dict[str, Any] | None = None,
    ) -> None:
        if not project_id or not wiki_a or not wiki_b or wiki_a == wiki_b:
            return
        a, b = _pair_key(wiki_a, wiki_b)
        self._conn.execute(
            "INSERT OR REPLACE INTO project_relatedness "
            "(wiki_a, wiki_b, score, breakdown) VALUES (?, ?, ?, ?)",
            (a, b, float(score), json.dumps(breakdown) if breakdown else None),
        )
        self._conn.commit()

    def get_repo_relatedness(self, project_id: str) -> list[dict[str, Any]]:
        rows = self._conn.execute(
            "SELECT wiki_a, wiki_b, score, breakdown FROM project_relatedness"
        ).fetchall()
        out: list[dict[str, Any]] = []
        for r in rows:
            try:
                bd = json.loads(r["breakdown"]) if r["breakdown"] else {}
            except (TypeError, ValueError):
                bd = {}
            out.append(
                {
                    "wiki_a": r["wiki_a"],
                    "wiki_b": r["wiki_b"],
                    "score": float(r["score"]),
                    "breakdown": bd,
                }
            )
        return out

    def clear_repo_relatedness(self, project_id: str) -> None:
        if not project_id:
            return
        self._conn.execute("DELETE FROM project_relatedness")
        self._conn.commit()

    # ------------------------------------------------------------------
    # Meta
    # ------------------------------------------------------------------
    def set_project_meta(self, project_id: str, key: str, value: Any) -> None:
        if not project_id or not key:
            return
        try:
            payload = json.dumps(value)
        except (TypeError, ValueError):
            return
        self._conn.execute(
            "INSERT OR REPLACE INTO project_meta (key, value) VALUES (?, ?)",
            (key, payload),
        )
        self._conn.commit()

    def get_project_meta(self, project_id: str, key: str) -> Any:
        row = self._conn.execute(
            "SELECT value FROM project_meta WHERE key = ?", (key,)
        ).fetchone()
        if row is None:
            return None
        try:
            return json.loads(row["value"])
        except (TypeError, ValueError):
            return None


# ----------------------------------------------------------------------
# Postgres implementation
# ----------------------------------------------------------------------


_PROJECT_SCHEMA_RE = re.compile(r"[^a-zA-Z0-9_]+")


def _safe_project_schema(project_id: str) -> str:
    """Return a postgres-safe schema name derived from ``project_id``."""
    cleaned = _PROJECT_SCHEMA_RE.sub("_", str(project_id)).strip("_").lower()
    if not cleaned:
        cleaned = "default"
    if not cleaned[0].isalpha() and cleaned[0] != "_":
        cleaned = f"p_{cleaned}"
    return f"wiki_project_{cleaned}"


class PostgresProjectStorage:
    """PostgreSQL-backed project storage (schema-per-project).

    Mirrors :class:`SqliteProjectStorage` semantics on top of a dedicated
    schema. Tables are created lazily on first use and are idempotent.
    """

    def __init__(self, dsn: str, project_id: str) -> None:
        from sqlalchemy import create_engine, text  # type: ignore

        self._project_id = project_id
        self._schema = _safe_project_schema(project_id)
        self._engine = create_engine(dsn, future=True)
        self._text = text
        self._init_schema()

    def _init_schema(self) -> None:
        text = self._text
        schema = self._schema
        with self._engine.begin() as conn:
            conn.execute(text(f'CREATE SCHEMA IF NOT EXISTS "{schema}"'))
            conn.execute(text(
                f'CREATE TABLE IF NOT EXISTS "{schema}".project_nodes ('
                'node_id TEXT PRIMARY KEY, '
                "wiki_id TEXT NOT NULL DEFAULT '', "
                'attrs JSONB DEFAULT NULL)'
            ))
            conn.execute(text(
                f'CREATE TABLE IF NOT EXISTS "{schema}".project_edges ('
                'source_id TEXT NOT NULL, '
                'target_id TEXT NOT NULL, '
                'edge_class TEXT NOT NULL, '
                'weight DOUBLE PRECISION DEFAULT 1.0, '
                'provenance JSONB DEFAULT NULL, '
                'PRIMARY KEY (source_id, target_id, edge_class))'
            ))
            conn.execute(text(
                f'CREATE INDEX IF NOT EXISTS idx_project_edges_src '
                f'ON "{schema}".project_edges(source_id)'
            ))
            conn.execute(text(
                f'CREATE INDEX IF NOT EXISTS idx_project_edges_tgt '
                f'ON "{schema}".project_edges(target_id)'
            ))
            conn.execute(text(
                f'CREATE INDEX IF NOT EXISTS idx_project_edges_class '
                f'ON "{schema}".project_edges(edge_class)'
            ))
            conn.execute(text(
                f'CREATE TABLE IF NOT EXISTS "{schema}".project_relatedness ('
                'wiki_a TEXT NOT NULL, '
                'wiki_b TEXT NOT NULL, '
                'score DOUBLE PRECISION DEFAULT 0.0, '
                'breakdown JSONB DEFAULT NULL, '
                'PRIMARY KEY (wiki_a, wiki_b))'
            ))
            conn.execute(text(
                f'CREATE TABLE IF NOT EXISTS "{schema}".project_meta ('
                'key TEXT PRIMARY KEY, value JSONB)'
            ))

    def close(self) -> None:
        try:
            self._engine.dispose()
        except Exception:  # pragma: no cover
            pass

    def drop_schema(self) -> None:
        """Drop the entire per-project schema and all project graph tables."""
        text = self._text
        schema = self._schema
        with self._engine.begin() as conn:
            conn.execute(text(f'DROP SCHEMA IF EXISTS "{schema}" CASCADE'))
        logger.info("Dropped PostgreSQL project schema: %s", schema)

    # ------------------------------------------------------------------
    # Nodes
    # ------------------------------------------------------------------
    def upsert_project_node(
        self,
        project_id: str,
        node_id: str,
        wiki_id: str,
        attrs: dict[str, Any],
    ) -> None:
        if not project_id or not node_id:
            return
        text = self._text
        schema = self._schema
        with self._engine.begin() as conn:
            conn.execute(
                text(
                    f'INSERT INTO "{schema}".project_nodes (node_id, wiki_id, attrs) '
                    "VALUES (:nid, :wid, CAST(:attrs AS JSONB)) "
                    "ON CONFLICT (node_id) DO UPDATE SET "
                    "wiki_id = EXCLUDED.wiki_id, attrs = EXCLUDED.attrs"
                ),
                {
                    "nid": node_id,
                    "wid": wiki_id or "",
                    "attrs": json.dumps(attrs or {}),
                },
            )

    def get_project_nodes(self, project_id: str) -> list[dict[str, Any]]:
        text = self._text
        with self._engine.connect() as conn:
            rows = conn.execute(
                text(f'SELECT node_id, wiki_id, attrs FROM "{self._schema}".project_nodes')
            ).fetchall()
        out: list[dict[str, Any]] = []
        for r in rows:
            attrs = r._mapping["attrs"] or {}
            out.append({"node_id": r._mapping["node_id"], "wiki_id": r._mapping["wiki_id"], **attrs})
        return out

    def clear_project_nodes(self, project_id: str) -> None:
        if not project_id:
            return
        text = self._text
        with self._engine.begin() as conn:
            conn.execute(text(f'DELETE FROM "{self._schema}".project_nodes'))

    # ------------------------------------------------------------------
    # Edges
    # ------------------------------------------------------------------
    def upsert_project_edge(
        self,
        project_id: str,
        source_node_id: str,
        target_node_id: str,
        edge_class: str,
        weight: float,
        provenance: dict[str, Any] | None = None,
    ) -> None:
        if not project_id or not source_node_id or not target_node_id:
            return
        text = self._text
        schema = self._schema
        with self._engine.begin() as conn:
            conn.execute(
                text(
                    f'INSERT INTO "{schema}".project_edges '
                    "(source_id, target_id, edge_class, weight, provenance) "
                    "VALUES (:s, :t, :c, :w, CAST(:p AS JSONB)) "
                    "ON CONFLICT (source_id, target_id, edge_class) DO UPDATE SET "
                    "weight = EXCLUDED.weight, provenance = EXCLUDED.provenance"
                ),
                {
                    "s": source_node_id,
                    "t": target_node_id,
                    "c": edge_class or "",
                    "w": float(weight),
                    "p": json.dumps(provenance) if provenance else None,
                },
            )

    def get_project_edges(
        self,
        project_id: str,
        *,
        source_node_id: str | None = None,
        target_node_id: str | None = None,
        edge_class: str | None = None,
    ) -> list[dict[str, Any]]:
        text = self._text
        sql = (
            f'SELECT source_id, target_id, edge_class, weight, provenance '
            f'FROM "{self._schema}".project_edges WHERE 1=1'
        )
        params: dict[str, Any] = {}
        if source_node_id is not None:
            sql += " AND source_id = :s"
            params["s"] = source_node_id
        if target_node_id is not None:
            sql += " AND target_id = :t"
            params["t"] = target_node_id
        if edge_class is not None:
            sql += " AND edge_class = :c"
            params["c"] = edge_class
        with self._engine.connect() as conn:
            rows = conn.execute(text(sql), params).fetchall()
        return [
            {
                "source_node_id": r._mapping["source_id"],
                "target_node_id": r._mapping["target_id"],
                "edge_class": r._mapping["edge_class"],
                "weight": float(r._mapping["weight"]),
                "provenance": r._mapping["provenance"] or {},
            }
            for r in rows
        ]

    def clear_project_edges(self, project_id: str) -> None:
        if not project_id:
            return
        text = self._text
        with self._engine.begin() as conn:
            conn.execute(text(f'DELETE FROM "{self._schema}".project_edges'))

    # ------------------------------------------------------------------
    # Relatedness
    # ------------------------------------------------------------------
    def upsert_repo_relatedness(
        self,
        project_id: str,
        wiki_a: str,
        wiki_b: str,
        score: float,
        breakdown: dict[str, Any] | None = None,
    ) -> None:
        if not project_id or not wiki_a or not wiki_b or wiki_a == wiki_b:
            return
        a, b = _pair_key(wiki_a, wiki_b)
        text = self._text
        schema = self._schema
        with self._engine.begin() as conn:
            conn.execute(
                text(
                    f'INSERT INTO "{schema}".project_relatedness '
                    "(wiki_a, wiki_b, score, breakdown) "
                    "VALUES (:a, :b, :s, CAST(:bd AS JSONB)) "
                    "ON CONFLICT (wiki_a, wiki_b) DO UPDATE SET "
                    "score = EXCLUDED.score, breakdown = EXCLUDED.breakdown"
                ),
                {
                    "a": a,
                    "b": b,
                    "s": float(score),
                    "bd": json.dumps(breakdown) if breakdown else None,
                },
            )

    def get_repo_relatedness(self, project_id: str) -> list[dict[str, Any]]:
        text = self._text
        with self._engine.connect() as conn:
            rows = conn.execute(
                text(
                    f'SELECT wiki_a, wiki_b, score, breakdown '
                    f'FROM "{self._schema}".project_relatedness'
                )
            ).fetchall()
        return [
            {
                "wiki_a": r._mapping["wiki_a"],
                "wiki_b": r._mapping["wiki_b"],
                "score": float(r._mapping["score"]),
                "breakdown": r._mapping["breakdown"] or {},
            }
            for r in rows
        ]

    def clear_repo_relatedness(self, project_id: str) -> None:
        if not project_id:
            return
        text = self._text
        with self._engine.begin() as conn:
            conn.execute(text(f'DELETE FROM "{self._schema}".project_relatedness'))

    # ------------------------------------------------------------------
    # Meta
    # ------------------------------------------------------------------
    def set_project_meta(self, project_id: str, key: str, value: Any) -> None:
        if not project_id or not key:
            return
        try:
            payload = json.dumps(value)
        except (TypeError, ValueError):
            return
        text = self._text
        schema = self._schema
        with self._engine.begin() as conn:
            conn.execute(
                text(
                    f'INSERT INTO "{schema}".project_meta (key, value) '
                    "VALUES (:k, CAST(:v AS JSONB)) "
                    "ON CONFLICT (key) DO UPDATE SET value = EXCLUDED.value"
                ),
                {"k": key, "v": payload},
            )

    def get_project_meta(self, project_id: str, key: str) -> Any:
        text = self._text
        with self._engine.connect() as conn:
            row = conn.execute(
                text(f'SELECT value FROM "{self._schema}".project_meta WHERE key = :k'),
                {"k": key},
            ).fetchone()
        if row is None:
            return None
        return row._mapping["value"]


# ----------------------------------------------------------------------
# Factory
# ----------------------------------------------------------------------


def open_project_storage(
    project_id: str,
    *,
    cache_dir: str | None = None,
    settings: Any | None = None,
) -> ProjectStorageProtocol:
    """Return the appropriate concrete project storage for the runtime.

    Selection mirrors :func:`open_storage`: PostgreSQL when
    ``WIKI_STORAGE_BACKEND=postgres`` (and ``WIKI_STORAGE_DSN`` is set),
    SQLite otherwise. Falls back to :class:`InMemoryProjectStorage` when
    we cannot open a real backend (so callers never have to special-case
    a missing store).
    """
    try:
        from .factory import is_postgres_backend
    except Exception:
        is_postgres_backend = lambda *_a, **_kw: False  # type: ignore  # noqa: E731

    if settings is None:
        try:
            from ...config import get_settings

            settings = get_settings()
        except Exception:
            settings = None

    try:
        if settings is not None and is_postgres_backend(settings):
            dsn = getattr(settings, "wiki_storage_dsn", "") or ""
            if dsn:
                return PostgresProjectStorage(dsn, project_id)
            logger.warning(
                "Postgres backend requested but WIKI_STORAGE_DSN is empty; "
                "falling back to in-memory project storage"
            )
            return InMemoryProjectStorage()
    except Exception:  # pragma: no cover — defensive
        logger.warning(
            "Failed to open Postgres project storage; falling back to SQLite",
            exc_info=True,
        )

    # SQLite path
    base_dir = cache_dir
    if not base_dir and settings is not None:
        base_dir = getattr(settings, "cache_dir", None)
    base_dir = base_dir or "./data/cache"
    try:
        target_dir = os.path.join(base_dir, "projects")
        os.makedirs(target_dir, exist_ok=True)
        safe_id = _PROJECT_SCHEMA_RE.sub("_", str(project_id)).strip("_") or "default"
        db_path = os.path.join(target_dir, f"{safe_id}.project.db")
        return SqliteProjectStorage(db_path)
    except Exception:  # pragma: no cover
        logger.warning(
            "Failed to open SQLite project storage at %s; using in-memory",
            base_dir,
            exc_info=True,
        )
        return InMemoryProjectStorage()


def _project_sqlite_path(
    project_id: str,
    *,
    cache_dir: str | None = None,
    settings: Any | None = None,
) -> str:
    """Return the SQLite project DB path without opening/creating it."""
    base_dir = cache_dir
    if not base_dir and settings is not None:
        base_dir = getattr(settings, "cache_dir", None)
    base_dir = base_dir or "./data/cache"
    safe_id = _PROJECT_SCHEMA_RE.sub("_", str(project_id)).strip("_") or "default"
    return os.path.join(base_dir, "projects", f"{safe_id}.project.db")


def drop_project_storage(
    project_id: str,
    *,
    cache_dir: str | None = None,
    settings: Any | None = None,
) -> bool:
    """Delete persisted project graph storage for SQLite or PostgreSQL.

    SQLite stores each project graph in ``<cache_dir>/projects/*.project.db``.
    PostgreSQL stores each project graph in a schema derived from the project id.
    This mirrors wiki deletion: metadata rows are handled by ``ProjectService``;
    this helper removes the backend-specific graph payload.

    Returns ``True`` when a backend cleanup operation was attempted and did not
    raise. Missing SQLite files are treated as a successful no-op.
    """
    if not project_id:
        return False

    if settings is None:
        try:
            from ...config import get_settings

            settings = get_settings()
        except Exception:
            settings = None

    try:
        from .factory import is_postgres_backend
    except Exception:
        is_postgres_backend = lambda *_a, **_kw: False  # type: ignore  # noqa: E731

    try:
        if settings is not None and is_postgres_backend(settings):
            dsn = getattr(settings, "wiki_storage_dsn", "") or ""
            if not dsn:
                logger.warning(
                    "Postgres project storage cleanup requested for %s but WIKI_STORAGE_DSN is empty",
                    project_id,
                )
                return False
            from sqlalchemy import create_engine, text  # type: ignore

            schema = _safe_project_schema(project_id)
            engine = create_engine(dsn, future=True)
            try:
                with engine.begin() as conn:
                    conn.execute(text(f'DROP SCHEMA IF EXISTS "{schema}" CASCADE'))
                logger.info("Dropped PostgreSQL project schema: %s", schema)
                return True
            finally:
                engine.dispose()

        db_path = _project_sqlite_path(
            project_id,
            cache_dir=cache_dir,
            settings=settings,
        )
        deleted_any = False
        for path in (db_path, f"{db_path}-wal", f"{db_path}-shm"):
            if not os.path.exists(path):
                continue
            os.remove(path)
            deleted_any = True
            logger.info("Deleted project SQLite storage file: %s", path)
        if not deleted_any:
            logger.debug("No project SQLite storage files found for %s at %s", project_id, db_path)
        return True
    except Exception:
        logger.warning("Failed to delete project storage for %s", project_id, exc_info=True)
        return False
