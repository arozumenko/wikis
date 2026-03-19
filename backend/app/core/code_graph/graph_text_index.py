"""
SQLite FTS5-backed full-text search index for the code graph.

Provides O(log N) BM25-ranked search over symbol names, docstrings, and source
code, replacing the brute-force O(N) keyword scan in ``_search_graph_by_text``.

The index is built once from a NetworkX graph and persisted to disk alongside
the ``.code_graph.gz`` file using the same cache-key pattern from GraphManager.

Thread/greenlet safety
---------------------
All read methods (``search``, ``search_by_name``, ``search_by_type``,
``search_by_path_prefix``, ``get_by_node_id``) open a **per-call read-only
SQLite connection** with ``check_same_thread=False``.  This prevents the
``ProgrammingError: SQLite objects created in a thread can only be used in
that same thread`` crash when multiple gevent greenlets (e.g. parallel
structure-planner tool calls) hit the index concurrently.  The pattern is
identical to ``BM25SqliteIndex._open_conn()`` in ``bm25_disk.py``.

Usage::

    index = GraphTextIndex(cache_dir="./data/cache")

    # Build from an existing NetworkX graph
    index.build_from_graph(graph, cache_key="abc123")

    # Search with BM25 ranking
    results = index.search("AuthService login", k=10)

    # Filtered search (exclude doc nodes)
    results = index.search("database", exclude_types=DOC_SYMBOL_TYPES)

    # Load from disk on next run
    index.load("abc123")
    results = index.search("AuthService")
"""

import logging
import os
import re
import sqlite3
from pathlib import Path
from typing import Any

import networkx as nx
from langchain_core.documents import Document

from ..constants import classify_symbol_layer
from .graph_query_builder import GraphQueryBuilder

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_CAMEL_SPLIT_RE = re.compile(r"(?<=[a-z])(?=[A-Z])|(?<=[A-Z])(?=[A-Z][a-z])")


def _tokenize_name(name: str) -> str:
    """Split camelCase / PascalCase / snake_case into lowercase tokens.

    >>> _tokenize_name("AuthServiceManager")
    'auth service manager'
    >>> _tokenize_name("get_user_by_id")
    'get user by id'
    >>> _tokenize_name("XMLParser")
    'xml parser'
    """
    # 1. Split camelCase / PascalCase boundaries
    parts = _CAMEL_SPLIT_RE.sub(" ", name)
    # 2. Replace any non-alphanumeric with space
    parts = re.sub(r"[^a-zA-Z0-9]", " ", parts)
    # 3. Collapse whitespace and lowercase
    return " ".join(parts.lower().split())


# ---------------------------------------------------------------------------
# Schema
# ---------------------------------------------------------------------------

_SCHEMA_SQL = """\
CREATE TABLE IF NOT EXISTS symbols (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    node_id       TEXT    UNIQUE NOT NULL,
    symbol_name   TEXT    NOT NULL DEFAULT '',
    name_tokens   TEXT    NOT NULL DEFAULT '',
    symbol_type   TEXT    NOT NULL DEFAULT '',
    layer         TEXT    NOT NULL DEFAULT '',
    file_path     TEXT    DEFAULT '',
    rel_path      TEXT    DEFAULT '',
    language      TEXT    DEFAULT '',
    start_line    INTEGER DEFAULT 0,
    end_line      INTEGER DEFAULT 0,
    full_name     TEXT    DEFAULT '',
    parent_symbol TEXT    DEFAULT '',
    docstring     TEXT    DEFAULT '',
    content       TEXT    DEFAULT ''
);

CREATE INDEX IF NOT EXISTS idx_sym_type ON symbols(symbol_type);
CREATE INDEX IF NOT EXISTS idx_sym_name ON symbols(symbol_name);
CREATE INDEX IF NOT EXISTS idx_sym_file ON symbols(file_path);
CREATE INDEX IF NOT EXISTS idx_sym_lang ON symbols(language);
CREATE INDEX IF NOT EXISTS idx_sym_relpath ON symbols(rel_path);
CREATE INDEX IF NOT EXISTS idx_sym_layer ON symbols(layer);
"""

_FTS_TABLE_SQL = """\
CREATE VIRTUAL TABLE IF NOT EXISTS symbols_fts USING fts5(
    symbol_name,
    name_tokens,
    docstring,
    content,
    content='symbols',
    content_rowid='id',
    tokenize='unicode61 remove_diacritics 2'
);
"""


# ---------------------------------------------------------------------------
# Main class
# ---------------------------------------------------------------------------


class GraphTextIndex:
    """SQLite FTS5 full-text index over code-graph nodes.

    The index stores every graph node in a regular ``symbols`` table (for
    structured queries) *and* in an FTS5 virtual table backed by the same
    content rows (for ranked text search via BM25).

    Thread safety
    ~~~~~~~~~~~~~
    Write operations (``build_from_graph``) use a shared ``_conn``.
    All read operations open a **per-call read-only connection** via
    ``_open_read_conn()`` so that concurrent gevent greenlets or OS
    threads never share a cursor.  Read-only SQLite connections are
    extremely cheap (~0.05 ms on SSD).

    Persistence
    -----------
    Each index is a single ``.fts5.db`` file living in *cache_dir* (default
    ``./data/cache``).  The file name is ``{cache_key}.fts5.db``
    where *cache_key* is the same MD5 hash that ``GraphManager`` uses for the
    corresponding ``.code_graph.gz`` file.
    """

    # BM25 column weights: symbol_name, name_tokens, docstring, content
    # Higher weight = matches in that column score higher.
    BM25_WEIGHTS = (10.0, 8.0, 3.0, 1.0)

    def __init__(self, cache_dir: str | None = None):
        if cache_dir is None:
            cache_dir = os.path.expanduser("./data/cache")
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        self._conn: sqlite3.Connection | None = None
        self._cache_key: str | None = None

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def _db_path(self, cache_key: str) -> Path:
        return self.cache_dir / f"{cache_key}.fts5.db"

    def _open(self, cache_key: str, *, create: bool = False) -> sqlite3.Connection:
        """Open (or create) the SQLite database for *cache_key*."""
        db_path = self._db_path(cache_key)
        if not create and not db_path.exists():
            raise FileNotFoundError(f"FTS5 index not found: {db_path}")

        conn = sqlite3.connect(str(db_path))
        conn.row_factory = sqlite3.Row
        # Enable WAL for better concurrent read performance
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA synchronous=NORMAL")
        return conn

    def close(self) -> None:
        """Close any held connection and reset state.

        After ``close()``, all read methods will return empty results
        until ``load()`` or ``build_from_graph()`` is called again.
        """
        if self._conn is not None:
            try:
                self._conn.close()
            except Exception:  # noqa: S110
                pass
            self._conn = None
        self._cache_key = None

    @property
    def is_open(self) -> bool:
        return self._conn is not None or self._cache_key is not None

    @property
    def _resolved_db_path(self) -> Path | None:
        """Resolved path to the current database file (or None)."""
        if self._cache_key is None:
            return None
        return self._db_path(self._cache_key)

    def _open_read_conn(self) -> sqlite3.Connection:
        """Open a fresh **read-only** SQLite connection for a single query.

        Each search call gets its own connection so concurrent greenlets /
        threads never share cursors.  Uses ``check_same_thread=False`` and
        ``mode=ro`` to match the pattern proven in ``BM25SqliteIndex``.
        """
        db_path = self._resolved_db_path
        if db_path is None or not db_path.exists():
            raise FileNotFoundError("FTS5 index is not open")
        conn = sqlite3.connect(
            f"file:{db_path}?mode=ro",
            uri=True,
            check_same_thread=False,
        )
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA query_only = ON")
        return conn

    # ------------------------------------------------------------------
    # Build
    # ------------------------------------------------------------------

    @staticmethod
    def _detect_repo_root_from_graph(graph: "nx.DiGraph") -> str:
        """Infer the repository root from nodes that have *both* ``rel_path``
        and ``file_path``.  For cached graphs whose basic-tier nodes only
        have an absolute ``file_path`` and no ``rel_path``, this lets us
        convert the absolute path to a relative one at index time.

        Returns the most-common root or ``''`` if detection fails.
        """
        from collections import Counter

        roots: Counter = Counter()
        sampled = 0
        for _, data in graph.nodes(data=True):
            if sampled >= 500:
                break
            rp = data.get("rel_path", "")
            fp = data.get("file_path", "")
            if rp and fp and not os.path.isabs(rp) and os.path.isabs(fp) and fp.endswith(rp):
                root = fp[: -len(rp)].rstrip("/")
                if root:
                    roots[root] += 1
                    sampled += 1
        if roots:
            best_root = roots.most_common(1)[0][0]
            logger.debug(f"[FTS5] Detected repo root from graph: {best_root}")
            return best_root
        return ""

    def build_from_graph(self, graph: nx.DiGraph, cache_key: str) -> int:
        """Build the FTS5 index from an in-memory NetworkX graph.

        Drops any existing index for the same *cache_key* and rebuilds
        from scratch.

        Returns:
            Number of nodes indexed.
        """
        self.close()

        db_path = self._db_path(cache_key)
        # Remove stale file so we start clean
        if db_path.exists():
            db_path.unlink()

        conn = self._open(cache_key, create=True)

        # Detect repo root so _iter_node_rows can convert absolute paths
        repo_root = self._detect_repo_root_from_graph(graph)

        try:
            # Create schema
            conn.executescript(_SCHEMA_SQL)
            conn.executescript(_FTS_TABLE_SQL)

            # Bulk-insert nodes
            rows = list(self._iter_node_rows(graph, repo_root=repo_root))
            if rows:
                conn.executemany(
                    """INSERT INTO symbols
                       (node_id, symbol_name, name_tokens, symbol_type,
                        layer, file_path, rel_path, language,
                        start_line, end_line, full_name, parent_symbol,
                        docstring, content)
                       VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                    rows,
                )
                # Populate FTS from content table
                conn.execute("INSERT INTO symbols_fts(symbols_fts) VALUES('rebuild')")
            conn.commit()
            conn.close()  # Close write connection — reads use per-call connections

            self._conn = None
            self._cache_key = cache_key
            logger.info(f"Built FTS5 index: {len(rows)} nodes indexed → {db_path.name}")
            return len(rows)

        except Exception:
            conn.close()
            if db_path.exists():
                db_path.unlink()
            raise

    @staticmethod
    def _iter_node_rows(graph: nx.DiGraph, repo_root: str = ""):
        """Yield tuples ready for INSERT from graph nodes.

        Args:
            graph: NetworkX graph with symbol nodes.
            repo_root: Optional repo root used to convert absolute
                ``file_path`` to a relative ``rel_path`` for nodes
                from cached / old graphs that lack ``rel_path``.
        """
        for node_id, data in graph.nodes(data=True):
            symbol_name = data.get("symbol_name", "") or data.get("name", "") or ""
            symbol_type = (data.get("symbol_type") or "").lower()

            # Skip nodes without a name
            if not symbol_name:
                continue

            # Get source text — different nodes store it in different attrs
            symbol = data.get("symbol")
            if symbol and hasattr(symbol, "source_text") and symbol.source_text:
                content = symbol.source_text
            else:
                content = data.get("content") or data.get("source_text") or ""

            # Extract docstring — comprehensive parser nodes store it inside
            # the Symbol object (data['symbol'].docstring), while basic parser
            # nodes store it as a top-level attribute (data['docstring']).
            docstring = data.get("docstring", "") or ""
            if not docstring and symbol and hasattr(symbol, "docstring") and symbol.docstring:
                docstring = symbol.docstring
            parent_symbol = data.get("parent_symbol", "") or ""
            rel_path = data.get("rel_path", "") or data.get("file_path", "") or ""

            # Convert absolute paths to relative when repo_root is known
            if rel_path and os.path.isabs(rel_path) and repo_root:
                rel_path = os.path.relpath(rel_path, repo_root)

            # Deterministic layer classification (SPEC-4)
            layer = classify_symbol_layer(
                symbol_type=symbol_type,
                symbol_name=symbol_name,
                parent_symbol=parent_symbol,
                file_path=rel_path,
                docstring=docstring,
            )

            yield (
                node_id,
                symbol_name,
                _tokenize_name(symbol_name),
                symbol_type,
                layer,
                data.get("file_path", "") or "",
                rel_path,
                data.get("language", "") or "",
                data.get("start_line", 0) or 0,
                data.get("end_line", 0) or 0,
                data.get("full_name", "") or "",
                parent_symbol,
                docstring,
                content,
            )

    # ------------------------------------------------------------------
    # Load / existence check
    # ------------------------------------------------------------------

    def load(self, cache_key: str) -> bool:
        """Load an existing FTS5 index from disk.

        Sets ``_cache_key`` so that subsequent read calls can open their
        own per-call connections.  The validation connection is closed
        immediately after the sanity check.

        Returns ``True`` if loaded successfully, ``False`` if no index
        exists for this *cache_key*.
        """
        self.close()
        db_path = self._db_path(cache_key)
        if not db_path.exists():
            return False
        try:
            conn = self._open(cache_key)
            try:
                # Quick sanity check — table must exist
                row = conn.execute("SELECT count(*) FROM symbols").fetchone()
                count = row[0] if row else 0
            finally:
                conn.close()
            # Only store cache_key — reads open their own connections
            self._cache_key = cache_key
            logger.info(f"Loaded FTS5 index: {count} nodes from {db_path.name}")
            return True
        except Exception as exc:
            logger.warning(f"Failed to load FTS5 index {db_path}: {exc}")
            return False

    def exists(self, cache_key: str) -> bool:
        """Check whether an FTS5 index exists on disk for *cache_key*."""
        return self._db_path(cache_key).exists()

    @property
    def node_count(self) -> int:
        """Number of indexed nodes (0 if not open).

        Thread-safe: opens a per-call read-only connection.
        """
        if self._cache_key is None:
            return 0
        conn: sqlite3.Connection | None = None
        try:
            conn = self._open_read_conn()
            row = conn.execute("SELECT count(*) FROM symbols").fetchone()
            return row[0] if row else 0
        except (sqlite3.InterfaceError, sqlite3.OperationalError, FileNotFoundError):
            return 0
        finally:
            if conn is not None:
                try:
                    conn.close()
                except Exception:  # noqa: S110
                    pass

    # ------------------------------------------------------------------
    # Search
    # ------------------------------------------------------------------

    def search(
        self,
        query: str,
        k: int = 10,
        symbol_types: frozenset[str] | None = None,
        exclude_types: frozenset[str] | None = None,
        language: str | None = None,
    ) -> list[Document]:
        """Full-text BM25 search across symbol names, docstrings, and code.

        Thread-safe: opens a per-call read-only connection.

        Args:
            query: Free-form search string.
            k: Maximum results to return.
            symbol_types: If given, only return these types.
            exclude_types: If given, exclude these types.
            language: If given, restrict results to this language
                (e.g. ``'java'``, ``'python'``).  Prevents cross-language
                contamination in multilingual repositories.

        Returns:
            List of ``Document`` objects sorted by BM25 relevance (best first).
        """
        if self._cache_key is None:
            logger.warning("GraphTextIndex.search() called but index is not open")
            return []

        fts_query = self._prepare_fts_query(query)
        if not fts_query:
            return []

        # Build WHERE clause for type filters
        where_parts: list[str] = []
        params: list[Any] = [fts_query]

        if symbol_types:
            placeholders = ",".join("?" for _ in symbol_types)
            where_parts.append(f"s.symbol_type IN ({placeholders})")
            params.extend(sorted(symbol_types))
        if exclude_types:
            placeholders = ",".join("?" for _ in exclude_types)
            where_parts.append(f"s.symbol_type NOT IN ({placeholders})")
            params.extend(sorted(exclude_types))
        if language:
            where_parts.append("s.language = ?")
            params.append(language.lower())

        where_clause = (" AND " + " AND ".join(where_parts)) if where_parts else ""
        params.append(k)

        weights = ", ".join(str(w) for w in self.BM25_WEIGHTS)
        sql = (
            f"SELECT s.*, bm25(symbols_fts, {weights}) AS score "  # noqa: S608 — FTS5 BM25 query, weights are numeric constants, no user input
            "FROM symbols_fts f "
            "JOIN symbols s ON s.id = f.rowid "
            f"WHERE symbols_fts MATCH ?{where_clause} "
            "ORDER BY score "
            "LIMIT ?"
        )

        conn: sqlite3.Connection | None = None
        try:
            conn = self._open_read_conn()
            try:
                rows = conn.execute(sql, params).fetchall()
            except sqlite3.OperationalError as exc:
                # FTS5 query syntax error — fall back to simpler query
                logger.debug(f"FTS5 query error for '{fts_query}': {exc}")
                simple_q = self._prepare_simple_query(query)
                if simple_q and simple_q != fts_query:
                    params[0] = simple_q
                    try:
                        rows = conn.execute(sql, params).fetchall()
                    except sqlite3.OperationalError:
                        return []
                else:
                    return []

            docs = [self._row_to_document(row) for row in rows]
            if docs:
                top_names = [d.metadata.get("symbol_name", "?") for d in docs[:5]]
                logger.debug(f"[FTS5] search({query!r:.50}, k={k}) → {len(docs)} results, top={top_names}")
            return docs
        except (sqlite3.InterfaceError, sqlite3.OperationalError, FileNotFoundError) as exc:
            logger.warning(f"GraphTextIndex.search() failed: {exc}")
            return []
        finally:
            if conn is not None:
                try:
                    conn.close()
                except Exception:  # noqa: S110
                    pass

    def search_smart(
        self,
        query: str,
        k: int = 10,
        *,
        intent: str = "general",
        symbol_types: frozenset[str] | None = None,
        exclude_types: frozenset[str] | None = None,
        language: str | None = None,
    ) -> list[Document]:
        """Smart FTS5 search using ``GraphQueryBuilder`` for query construction.

        This is an enhanced version of ``search()`` that uses keyword extraction
        and intent-aware query construction for better BM25 results.

        Args:
            query: Free-form natural-language question or keyword string.
            k: Maximum results to return.
            intent: Query intent — ``'general'``, ``'symbol'``, ``'concept'``,
                    or ``'doc'``.  Passed to ``GraphQueryBuilder.from_natural_language``.
            symbol_types: If given, only return these symbol types.
            exclude_types: If given, exclude these symbol types.
            language: If given, restrict results to this language.

        Returns:
            List of ``Document`` objects sorted by BM25 relevance (best first).
        """
        if self._cache_key is None:
            logger.warning("GraphTextIndex.search_smart() called but index is not open")
            return []

        fts_query = GraphQueryBuilder.from_natural_language(query, intent=intent)
        if not fts_query:
            # Fallback to legacy query preparation
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

    def search_symbols(
        self,
        query: str,
        k: int = 20,
        *,
        symbol_types: frozenset[str] | None = None,
        exclude_types: frozenset[str] | None = None,
        path_prefix: str | None = None,
        language: str | None = None,
    ) -> list[Document]:
        """Combined FTS5 + structured search for symbol discovery.

        This is the primary method for Phase 3's ``search_symbols`` tool.
        It combines:
        - FTS5 concept search (keyword extraction + BM25)
        - Optional symbol_type filtering
        - Optional path prefix filtering (post-filter on rel_path)

        Args:
            query: Natural-language or keyword query.
            k: Maximum results.
            symbol_types: Restrict to these types (e.g. ``frozenset({'class'})``).
            exclude_types: Exclude these types.
            path_prefix: Only return symbols under this directory prefix.
            language: If given, restrict results to this language.

        Returns:
            List of ``Document`` objects sorted by BM25 relevance.
        """
        if self._cache_key is None:
            return []

        fts_query = GraphQueryBuilder.from_natural_language(query, intent="concept")
        if not fts_query:
            fts_query = self._prepare_fts_query(query)
        if not fts_query:
            return []

        # Request more results if we need to post-filter by path
        fetch_k = k * 3 if path_prefix else k

        docs = self._execute_fts_search(
            fts_query,
            query,
            fetch_k,
            symbol_types=symbol_types,
            exclude_types=exclude_types,
            language=language,
        )

        # Post-filter by path prefix
        if path_prefix and docs:
            prefix = path_prefix.strip("/")
            docs = [d for d in docs if (d.metadata.get("rel_path") or "").startswith(prefix)]

        return docs[:k]

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
        """Execute an FTS5 MATCH query with type filtering.

        This is the shared execution logic used by ``search()``,
        ``search_smart()``, and ``search_symbols()``.
        """
        # Build WHERE clause for type filters
        where_parts: list[str] = []
        params: list[Any] = [fts_query]

        if symbol_types:
            placeholders = ",".join("?" for _ in symbol_types)
            where_parts.append(f"s.symbol_type IN ({placeholders})")
            params.extend(sorted(symbol_types))
        if exclude_types:
            placeholders = ",".join("?" for _ in exclude_types)
            where_parts.append(f"s.symbol_type NOT IN ({placeholders})")
            params.extend(sorted(exclude_types))
        if language:
            where_parts.append("s.language = ?")
            params.append(language.lower())

        where_clause = (" AND " + " AND ".join(where_parts)) if where_parts else ""
        params.append(k)

        weights = ", ".join(str(w) for w in self.BM25_WEIGHTS)
        sql = (
            f"SELECT s.*, bm25(symbols_fts, {weights}) AS score "  # noqa: S608 — FTS5 BM25 query, weights are numeric constants, no user input
            "FROM symbols_fts f "
            "JOIN symbols s ON s.id = f.rowid "
            f"WHERE symbols_fts MATCH ?{where_clause} "
            "ORDER BY score "
            "LIMIT ?"
        )

        conn: sqlite3.Connection | None = None
        try:
            conn = self._open_read_conn()
            try:
                rows = conn.execute(sql, params).fetchall()
            except sqlite3.OperationalError as exc:
                # FTS5 query syntax error — fall back to simpler query
                logger.debug(f"FTS5 query error for '{fts_query}': {exc}, falling back to simple query")
                simple_q = self._prepare_simple_query(original_query)
                if simple_q and simple_q != fts_query:
                    params[0] = simple_q
                    try:
                        rows = conn.execute(sql, params).fetchall()
                    except sqlite3.OperationalError:
                        return []
                else:
                    return []

            docs = [self._row_to_document(row) for row in rows]
            if docs:
                top_names = [d.metadata.get("symbol_name", "?") for d in docs[:5]]
                logger.debug(f"[FTS5] search({original_query!r:.50}, k={k}) → {len(docs)} results, top={top_names}")
            return docs
        except (sqlite3.InterfaceError, sqlite3.OperationalError, FileNotFoundError) as exc:
            logger.warning(f"GraphTextIndex._execute_fts_search() failed: {exc}")
            return []
        finally:
            if conn is not None:
                try:
                    conn.close()
                except Exception:  # noqa: S110
                    pass

    def search_by_name(
        self,
        name: str,
        exact: bool = False,
        k: int = 10,
        symbol_types: frozenset[str] | None = None,
        language: str | None = None,
    ) -> list[Document]:
        """Search by symbol name (structured query, not FTS).

        Thread-safe: opens a per-call read-only connection.

        Args:
            name: Symbol name to search for.
            exact: If True, match exactly; otherwise LIKE prefix match.
            k: Maximum results.
            symbol_types: If given, only return these symbol types
                (e.g. architectural symbols only).
            language: If given, restrict results to this language
                (e.g. ``'java'``, ``'python'``).
        """
        if self._cache_key is None:
            return []

        where_parts: list[str] = []
        params: list = []

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

        sql = f"SELECT * FROM symbols WHERE {' AND '.join(where_parts)} LIMIT ?"  # noqa: S608 — internal symbol query, where_parts built from validated field names
        params.append(k)

        conn: sqlite3.Connection | None = None
        try:
            conn = self._open_read_conn()
            rows = conn.execute(sql, params).fetchall()
            return [self._row_to_document(row, score=0.0) for row in rows]
        except (sqlite3.InterfaceError, sqlite3.OperationalError, FileNotFoundError) as exc:
            logger.warning(f"GraphTextIndex.search_by_name() failed: {exc}")
            return []
        finally:
            if conn is not None:
                try:
                    conn.close()
                except Exception:  # noqa: S110
                    pass

    def search_by_type(
        self,
        symbol_type: str,
        k: int = 100,
    ) -> list[Document]:
        """Return all symbols of a given type.

        Thread-safe: opens a per-call read-only connection.
        """
        if self._cache_key is None:
            return []
        conn: sqlite3.Connection | None = None
        try:
            conn = self._open_read_conn()
            sql = "SELECT * FROM symbols WHERE symbol_type = ? LIMIT ?"
            rows = conn.execute(sql, [symbol_type, k]).fetchall()
            return [self._row_to_document(row, score=0.0) for row in rows]
        except (sqlite3.InterfaceError, sqlite3.OperationalError, FileNotFoundError) as exc:
            logger.warning(f"GraphTextIndex.search_by_type() failed: {exc}")
            return []
        finally:
            if conn is not None:
                try:
                    conn.close()
                except Exception:  # noqa: S110
                    pass

    def search_by_layer(
        self,
        layer: str,
        symbol_types: frozenset[str] | None = None,
        k: int = 100,
    ) -> list[Document]:
        """Return symbols belonging to a specific architectural layer.

        Layers are deterministically assigned at build time by
        ``classify_symbol_layer()`` — no LLM cost.

        Valid layers: entry_point, public_api, core_type, internal,
        infrastructure, constant.

        Thread-safe: opens a per-call read-only connection.
        """
        if self._cache_key is None:
            return []
        conn: sqlite3.Connection | None = None
        try:
            conn = self._open_read_conn()
            params: list[Any] = [layer]
            sql = "SELECT * FROM symbols WHERE layer = ?"
            if symbol_types:
                placeholders = ",".join("?" for _ in symbol_types)
                sql += f" AND symbol_type IN ({placeholders})"
                params.extend(sorted(symbol_types))
            sql += " LIMIT ?"
            params.append(k)
            rows = conn.execute(sql, params).fetchall()
            return [self._row_to_document(row, score=0.0) for row in rows]
        except (sqlite3.InterfaceError, sqlite3.OperationalError, FileNotFoundError) as exc:
            logger.warning(f"GraphTextIndex.search_by_layer() failed: {exc}")
            return []
        finally:
            if conn is not None:
                try:
                    conn.close()
                except Exception:  # noqa: S110
                    pass

    def search_by_path_prefix(
        self,
        path_prefix: str,
        symbol_types: frozenset[str] | None = None,
        exclude_types: frozenset[str] | None = None,
        k: int = 500,
    ) -> list[dict[str, Any]]:
        """Return symbols whose ``rel_path`` starts with *path_prefix*.

        This replaces the O(N) graph scan in ``structure_tools.query_graph``
        with an O(log N) SQL index lookup.

        Args:
            path_prefix: Path prefix to filter on (e.g. ``"src/services"``).
            symbol_types: Only return these types (optional).
            exclude_types: Exclude these types (optional).
            k: Maximum results.

        Returns:
            List of dicts (one per row) with all symbol columns.
        """
        if self._cache_key is None:
            return []

        prefix = path_prefix.strip("/")
        where_parts = ["(rel_path LIKE ? OR rel_path LIKE ? OR rel_path = ?)"]
        # Match prefix/... (subdirectory), prefix% (file starting with prefix), or exact
        params: list[Any] = [f"{prefix}/%", f"{prefix}%", prefix]

        if symbol_types:
            placeholders = ",".join("?" for _ in symbol_types)
            where_parts.append(f"symbol_type IN ({placeholders})")
            params.extend(sorted(symbol_types))
        if exclude_types:
            placeholders = ",".join("?" for _ in exclude_types)
            where_parts.append(f"symbol_type NOT IN ({placeholders})")
            params.extend(sorted(exclude_types))

        where_clause = " AND ".join(where_parts)
        params.append(k)
        sql = f"SELECT * FROM symbols WHERE {where_clause} LIMIT ?"  # noqa: S608 — internal symbol query, where_clause built from validated field names

        conn: sqlite3.Connection | None = None
        try:
            conn = self._open_read_conn()
            rows = conn.execute(sql, params).fetchall()
            logger.debug(f"[FTS5] search_by_path_prefix({prefix!r}) → {len(rows)} rows")
            return [dict(row) for row in rows]
        except (sqlite3.InterfaceError, sqlite3.OperationalError, FileNotFoundError) as exc:
            logger.warning(f"GraphTextIndex.search_by_path_prefix() failed: {exc}")
            return []
        finally:
            if conn is not None:
                try:
                    conn.close()
                except Exception:  # noqa: S110
                    pass

    def get_by_node_id(self, node_id: str) -> dict[str, Any] | None:
        """Direct lookup by graph node ID.  Returns a dict or ``None``.

        Thread-safe: opens a per-call read-only connection.
        """
        if self._cache_key is None:
            return None
        conn: sqlite3.Connection | None = None
        try:
            conn = self._open_read_conn()
            row = conn.execute("SELECT * FROM symbols WHERE node_id = ?", [node_id]).fetchone()
            if row is None:
                return None
            return dict(row)
        except (sqlite3.InterfaceError, sqlite3.OperationalError, FileNotFoundError) as exc:
            logger.warning(f"GraphTextIndex.get_by_node_id() failed: {exc}")
            return None
        finally:
            if conn is not None:
                try:
                    conn.close()
                except Exception:  # noqa: S110
                    pass

    # ------------------------------------------------------------------
    # Query helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _prepare_fts_query(query: str) -> str:
        """Convert a free-form query string into an FTS5 query.

        Strategy:
        - Tokenise the input (split on non-alphanumerics, drop short words).
        - Add camelCase-split variants.
        - Combine with OR so any keyword contributes to BM25.
        - Wrap each token as a prefix query (``token*``) for partial matching.
        """
        raw_tokens = re.split(r"\W+", query.strip())
        tokens: list[str] = []
        for tok in raw_tokens:
            if len(tok) < 2:
                continue
            lower = tok.lower()
            tokens.append(lower)
            # Also add camelCase-split variants
            split = _tokenize_name(tok).split()
            for part in split:
                if part not in tokens and len(part) >= 2:
                    tokens.append(part)

        if not tokens:
            return ""

        # Use prefix queries joined with OR
        return " OR ".join(f'"{t}"*' for t in tokens)

    @staticmethod
    def _prepare_simple_query(query: str) -> str:
        """Fallback: simple space-separated prefix query."""
        tokens = [t.lower() for t in re.split(r"\W+", query.strip()) if len(t) >= 2]
        if not tokens:
            return ""
        return " OR ".join(f"{t}*" for t in tokens)

    @staticmethod
    def _row_to_document(row: sqlite3.Row, score: float | None = None) -> Document:
        """Convert a database row to a LangChain Document."""
        content = row["content"] or row["docstring"] or ""
        meta = {
            "source": row["rel_path"] or row["file_path"] or "unknown",
            "rel_path": row["rel_path"] or "",
            "symbol_name": row["symbol_name"],
            "symbol_type": row["symbol_type"],
            "layer": row["layer"] if "layer" in row.keys() else "",
            "file_path": row["file_path"] or "",
            "language": row["language"] or "",
            "start_line": row["start_line"] or "",
            "end_line": row["end_line"] or "",
            "node_id": row["node_id"],
            "full_name": row["full_name"] or "",
            "parent_symbol": row["parent_symbol"] or "",
            "search_source": "graph_fts",
        }
        if score is not None:
            meta["search_score"] = score
        elif "score" in row.keys():
            meta["search_score"] = row["score"]
        return Document(page_content=content, metadata=meta)

    # ------------------------------------------------------------------
    # Cache-index registration (follows GraphManager / VectorStoreManager)
    # ------------------------------------------------------------------

    def register_in_cache_index(self, repo_identifier: str, cache_key: str) -> None:
        """Register FTS5 index in the shared ``cache_index.json``.

        Stored under a ``'fts5'`` sub-key to avoid collision with vectorstore
        and graph entries.
        """
        import json
        import uuid

        index_path = self.cache_dir / "cache_index.json"
        index: dict[str, Any] = {}
        if index_path.exists():
            try:
                with open(index_path, encoding="utf-8") as f:
                    index = json.load(f)
            except Exception:  # noqa: S110
                pass

        if "fts5" not in index:
            index["fts5"] = {}
        index["fts5"][repo_identifier] = cache_key

        try:
            tmp = index_path.with_name(f"{index_path.name}.{uuid.uuid4().hex}.tmp")
            with open(tmp, "w", encoding="utf-8") as f:
                json.dump(index, f, indent=2)
            os.replace(tmp, index_path)
        except Exception as exc:
            logger.warning(f"Failed to save FTS5 cache index entry: {exc}")

        logger.info(f"Registered FTS5 cache: {repo_identifier} → {cache_key}")

    def load_by_repo_name(self, repo_identifier: str) -> bool:
        """Load FTS5 index by repository identifier (convenience for Ask tool).

        Looks up the cache key from ``cache_index.json['fts5']``.
        """
        import json

        index_path = self.cache_dir / "cache_index.json"
        if not index_path.exists():
            return False
        try:
            with open(index_path, encoding="utf-8") as f:
                index = json.load(f)
        except Exception:
            return False

        cache_key = (index.get("fts5") or {}).get(repo_identifier)
        if not cache_key:
            # Try canonical resolution
            try:
                from ..repo_resolution import resolve_canonical_repo_identifier

                canonical = resolve_canonical_repo_identifier(
                    repo_identifier=repo_identifier,
                    cache_dir=self.cache_dir,
                    repositories_dir=self.cache_dir / "repositories",
                )
                cache_key = (index.get("fts5") or {}).get(canonical)
            except Exception:  # noqa: S110
                pass

        if not cache_key:
            return False

        return self.load(cache_key)

    # ------------------------------------------------------------------
    # Cleanup
    # ------------------------------------------------------------------

    def delete(self, cache_key: str) -> None:
        """Delete the FTS5 index file for *cache_key*."""
        self.close()
        db_path = self._db_path(cache_key)
        for suffix in ("", "-wal", "-shm"):
            p = Path(str(db_path) + suffix)
            if p.exists():
                p.unlink()

    def __del__(self):
        self.close()

    def __repr__(self) -> str:
        state = f"key={self._cache_key}" if self._cache_key else "closed"
        return f"<GraphTextIndex({state})>"
