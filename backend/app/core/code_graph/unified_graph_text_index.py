"""GraphTextIndex-compatible search adapter backed by UnifiedWikiDB.

This keeps the existing GraphTextIndex API surface for planners and research
tools while routing all lookups to the unified ``.wiki.db`` file.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from langchain_core.documents import Document

from ..constants import classify_symbol_layer
from .graph_text_index import GraphTextIndex

try:
    import pysqlite3 as sqlite3  # type: ignore[import-untyped]
except ImportError:
    import sqlite3  # type: ignore[no-redef]

logger = logging.getLogger(__name__)


class UnifiedGraphTextIndex(GraphTextIndex):
    """Compatibility layer over ``repo_fts`` and ``repo_nodes``.

    The legacy ``GraphTextIndex`` stored symbol metadata in a standalone
    ``.fts5.db``. This adapter preserves the same search methods while reading
    directly from the unified wiki DB.
    """

    def __init__(self, db_path: str | Path):
        self._unified_db_path = Path(db_path)
        super().__init__(cache_dir=str(self._unified_db_path.parent))
        self._cache_key = self._unified_db_path.stem

    @property
    def is_open(self) -> bool:
        return self._unified_db_path.exists()

    @property
    def _resolved_db_path(self) -> Path | None:
        return self._unified_db_path if self._unified_db_path.exists() else None

    def _open_read_conn(self) -> sqlite3.Connection:
        if not self._unified_db_path.exists():
            raise FileNotFoundError(f"Unified DB not found: {self._unified_db_path}")
        conn = sqlite3.connect(
            f"file:{self._unified_db_path}?mode=ro",
            uri=True,
            check_same_thread=False,
        )
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA query_only = ON")
        return conn

    @staticmethod
    def _repo_row_to_document(row: sqlite3.Row, score: float | None = None) -> Document:
        rel_path = row["rel_path"] or ""
        symbol_type = row["symbol_type"] or ""
        symbol_name = row["symbol_name"] or ""
        parent_symbol = row["parent_symbol"] or ""
        docstring = row["docstring"] or ""
        content = row["source_text"] or docstring or ""
        layer = classify_symbol_layer(
            symbol_type=symbol_type,
            symbol_name=symbol_name,
            parent_symbol=parent_symbol,
            file_path=rel_path,
            docstring=docstring,
        )
        metadata = {
            "source": rel_path or "unknown",
            "rel_path": rel_path,
            "symbol_name": symbol_name,
            "symbol_type": symbol_type,
            "layer": layer,
            "file_path": rel_path,
            "language": row["language"] or "",
            "start_line": row["start_line"] or "",
            "end_line": row["end_line"] or "",
            "node_id": row["node_id"],
            "full_name": "",
            "parent_symbol": parent_symbol,
            "search_source": "graph_fts",
        }
        if score is not None:
            metadata["search_score"] = score
        elif "score" in row.keys():
            metadata["search_score"] = row["score"]
        return Document(page_content=content, metadata=metadata)

    @staticmethod
    def _repo_row_to_dict(row: sqlite3.Row) -> dict[str, Any]:
        rel_path = row["rel_path"] or ""
        symbol_type = row["symbol_type"] or ""
        symbol_name = row["symbol_name"] or ""
        parent_symbol = row["parent_symbol"] or ""
        return {
            "node_id": row["node_id"],
            "symbol_name": symbol_name,
            "symbol_type": symbol_type,
            "layer": classify_symbol_layer(
                symbol_type=symbol_type,
                symbol_name=symbol_name,
                parent_symbol=parent_symbol,
                file_path=rel_path,
                docstring=row["docstring"] or "",
            ),
            "file_path": rel_path,
            "rel_path": rel_path,
            "language": row["language"] or "",
            "start_line": row["start_line"] or "",
            "end_line": row["end_line"] or "",
            "parent_symbol": parent_symbol,
            "docstring": row["docstring"] or "",
            "content": row["source_text"] or "",
        }

    def search(
        self,
        query: str,
        k: int = 10,
        symbol_types: frozenset[str] | None = None,
        exclude_types: frozenset[str] | None = None,
        language: str | None = None,
    ) -> list[Document]:
        if self._cache_key is None:
            logger.warning("UnifiedGraphTextIndex.search() called but index is not open")
            return []

        fts_query = self._prepare_fts_query(query)
        if not fts_query:
            return []

        return self._execute_fts_search(
            fts_query,
            query,
            k,
            symbol_types=symbol_types,
            exclude_types=exclude_types,
            language=language,
        )

    def _execute_fts_search(
        self,
        fts_query: str,
        original_query: str,
        k: int,
        *,
        symbol_types: frozenset[str] | None = None,
        exclude_types: frozenset[str] | None = None,
        language: str | None = None,
    ) -> list[Document]:
        conditions: list[str] = []
        params: list[Any] = [fts_query]

        if symbol_types:
            placeholders = ",".join("?" for _ in symbol_types)
            conditions.append(f"m.symbol_type IN ({placeholders})")
            params.extend(sorted(symbol_types))
        if exclude_types:
            placeholders = ",".join("?" for _ in exclude_types)
            conditions.append(f"m.symbol_type NOT IN ({placeholders})")
            params.extend(sorted(exclude_types))
        if language:
            conditions.append("m.language = ?")
            params.append(language.lower())

        where_clause = (" AND " + " AND ".join(conditions)) if conditions else ""
        params.append(k)

        weights = ", ".join(["0.0", *(str(w) for w in self.BM25_WEIGHTS)])
        sql = (
            f"SELECT m.*, bm25(repo_fts, {weights}) AS score "
            "FROM repo_fts "
            "JOIN repo_nodes m ON repo_fts.node_id = m.node_id "
            f"WHERE repo_fts MATCH ?{where_clause} "
            "ORDER BY score "
            "LIMIT ?"
        )

        conn: sqlite3.Connection | None = None
        try:
            conn = self._open_read_conn()
            try:
                rows = conn.execute(sql, params).fetchall()
            except sqlite3.OperationalError as exc:
                logger.debug("Unified FTS query error for %r: %s", fts_query, exc)
                simple_q = self._prepare_simple_query(original_query)
                if not simple_q or simple_q == fts_query:
                    return []
                params[0] = simple_q
                try:
                    rows = conn.execute(sql, params).fetchall()
                except sqlite3.OperationalError:
                    return []

            return [self._repo_row_to_document(row) for row in rows]
        except (sqlite3.InterfaceError, sqlite3.OperationalError, FileNotFoundError) as exc:
            logger.warning("UnifiedGraphTextIndex._execute_fts_search() failed: %s", exc)
            return []
        finally:
            if conn is not None:
                try:
                    conn.close()
                except Exception:
                    pass

    def search_by_name(
        self,
        name: str,
        exact: bool = False,
        k: int = 10,
        symbol_types: frozenset[str] | None = None,
        language: str | None = None,
    ) -> list[Document]:
        if self._cache_key is None:
            return []

        where_parts: list[str] = []
        params: list[Any] = []

        if exact:
            where_parts.append("symbol_name = ?")
            params.append(name)
        else:
            where_parts.append("symbol_name LIKE ?")
            params.append(f"%{name}%")

        if symbol_types:
            placeholders = ",".join("?" for _ in symbol_types)
            where_parts.append(f"symbol_type IN ({placeholders})")
            params.extend(sorted(symbol_types))
        if language:
            where_parts.append("language = ?")
            params.append(language.lower())

        sql = f"SELECT * FROM repo_nodes WHERE {' AND '.join(where_parts)} LIMIT ?"
        params.append(k)

        conn: sqlite3.Connection | None = None
        try:
            conn = self._open_read_conn()
            rows = conn.execute(sql, params).fetchall()
            return [self._repo_row_to_document(row, score=0.0) for row in rows]
        except (sqlite3.InterfaceError, sqlite3.OperationalError, FileNotFoundError) as exc:
            logger.warning("UnifiedGraphTextIndex.search_by_name() failed: %s", exc)
            return []
        finally:
            if conn is not None:
                try:
                    conn.close()
                except Exception:
                    pass

    def search_by_type(self, symbol_type: str, k: int = 100) -> list[Document]:
        if self._cache_key is None:
            return []

        conn: sqlite3.Connection | None = None
        try:
            conn = self._open_read_conn()
            rows = conn.execute(
                "SELECT * FROM repo_nodes WHERE symbol_type = ? LIMIT ?",
                [symbol_type, k],
            ).fetchall()
            return [self._repo_row_to_document(row, score=0.0) for row in rows]
        except (sqlite3.InterfaceError, sqlite3.OperationalError, FileNotFoundError) as exc:
            logger.warning("UnifiedGraphTextIndex.search_by_type() failed: %s", exc)
            return []
        finally:
            if conn is not None:
                try:
                    conn.close()
                except Exception:
                    pass

    def search_by_layer(
        self,
        layer: str,
        symbol_types: frozenset[str] | None = None,
        k: int = 100,
    ) -> list[Document]:
        if self._cache_key is None:
            return []

        conn: sqlite3.Connection | None = None
        try:
            conn = self._open_read_conn()
            params: list[Any] = []
            sql = "SELECT * FROM repo_nodes"
            if symbol_types:
                placeholders = ",".join("?" for _ in symbol_types)
                sql += f" WHERE symbol_type IN ({placeholders})"
                params.extend(sorted(symbol_types))
            rows = conn.execute(sql, params).fetchall()

            docs: list[Document] = []
            for row in rows:
                computed_layer = classify_symbol_layer(
                    symbol_type=row["symbol_type"] or "",
                    symbol_name=row["symbol_name"] or "",
                    parent_symbol=row["parent_symbol"] or "",
                    file_path=row["rel_path"] or "",
                    docstring=row["docstring"] or "",
                )
                if computed_layer != layer:
                    continue
                docs.append(self._repo_row_to_document(row, score=0.0))
                if len(docs) >= k:
                    break
            return docs
        except (sqlite3.InterfaceError, sqlite3.OperationalError, FileNotFoundError) as exc:
            logger.warning("UnifiedGraphTextIndex.search_by_layer() failed: %s", exc)
            return []
        finally:
            if conn is not None:
                try:
                    conn.close()
                except Exception:
                    pass

    def search_by_path_prefix(
        self,
        path_prefix: str,
        symbol_types: frozenset[str] | None = None,
        exclude_types: frozenset[str] | None = None,
        k: int = 500,
    ) -> list[dict[str, Any]]:
        if self._cache_key is None:
            return []

        prefix = path_prefix.strip("/")
        where_parts = ["(rel_path LIKE ? OR rel_path LIKE ? OR rel_path = ?)"]
        params: list[Any] = [f"{prefix}/%", f"{prefix}%", prefix]

        if symbol_types:
            placeholders = ",".join("?" for _ in symbol_types)
            where_parts.append(f"symbol_type IN ({placeholders})")
            params.extend(sorted(symbol_types))
        if exclude_types:
            placeholders = ",".join("?" for _ in exclude_types)
            where_parts.append(f"symbol_type NOT IN ({placeholders})")
            params.extend(sorted(exclude_types))

        sql = f"SELECT * FROM repo_nodes WHERE {' AND '.join(where_parts)} LIMIT ?"
        params.append(k)

        conn: sqlite3.Connection | None = None
        try:
            conn = self._open_read_conn()
            rows = conn.execute(sql, params).fetchall()
            return [self._repo_row_to_dict(row) for row in rows]
        except (sqlite3.InterfaceError, sqlite3.OperationalError, FileNotFoundError) as exc:
            logger.warning("UnifiedGraphTextIndex.search_by_path_prefix() failed: %s", exc)
            return []
        finally:
            if conn is not None:
                try:
                    conn.close()
                except Exception:
                    pass

    def get_by_node_id(self, node_id: str) -> dict[str, Any] | None:
        if self._cache_key is None:
            return None

        conn: sqlite3.Connection | None = None
        try:
            conn = self._open_read_conn()
            row = conn.execute("SELECT * FROM repo_nodes WHERE node_id = ?", [node_id]).fetchone()
            return self._repo_row_to_dict(row) if row else None
        except (sqlite3.InterfaceError, sqlite3.OperationalError, FileNotFoundError) as exc:
            logger.warning("UnifiedGraphTextIndex.get_by_node_id() failed: %s", exc)
            return None
        finally:
            if conn is not None:
                try:
                    conn.close()
                except Exception:
                    pass