"""GraphTextIndex-compatible search adapter backed by WikiStorageProtocol.

Provides the full ``GraphTextIndex`` API surface (search, search_smart,
search_by_name, search_by_type, search_by_layer, search_by_path_prefix,
search_symbols, get_by_node_id) while delegating all queries to the
underlying storage protocol.  Works identically with both
``UnifiedWikiDB`` (SQLite) and ``PostgresWikiStorage`` (PostgreSQL).

This replaces the former ``UnifiedGraphTextIndex`` (raw-sqlite shim) and
``PostgresGraphTextIndex`` (Postgres-only shim) with a single adapter
that is backend-agnostic.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

from langchain_core.documents import Document

from ..constants import classify_symbol_layer

if TYPE_CHECKING:
    from .protocol import WikiStorageProtocol

logger = logging.getLogger(__name__)


class StorageTextIndex:
    """Backend-agnostic FTS adapter over any ``WikiStorageProtocol``.

    Parameters
    ----------
    storage : WikiStorageProtocol
        An opened storage instance (``UnifiedWikiDB`` or
        ``PostgresWikiStorage``).
    """

    def __init__(self, storage: WikiStorageProtocol) -> None:
        self._storage = storage

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def is_open(self) -> bool:
        return self._storage is not None

    # ------------------------------------------------------------------
    # Row conversion helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _row_to_document(row: dict[str, Any], score: float | None = None) -> Document:
        rel_path = row.get("rel_path", "")
        symbol_type = row.get("symbol_type", "")
        symbol_name = row.get("symbol_name", "")
        parent_symbol = row.get("parent_symbol", "")
        docstring = row.get("docstring", "")
        content = row.get("source_text", "") or docstring or ""
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
            "language": row.get("language", ""),
            "start_line": row.get("start_line", ""),
            "end_line": row.get("end_line", ""),
            "node_id": row.get("node_id", ""),
            "full_name": "",
            "parent_symbol": parent_symbol,
            "search_source": "graph_fts",
        }
        if score is not None:
            metadata["search_score"] = score
        elif "fts_rank" in row:
            metadata["search_score"] = row["fts_rank"]
        return Document(page_content=content, metadata=metadata)

    @staticmethod
    def _row_to_dict(row: dict[str, Any]) -> dict[str, Any]:
        rel_path = row.get("rel_path", "")
        symbol_type = row.get("symbol_type", "")
        symbol_name = row.get("symbol_name", "")
        parent_symbol = row.get("parent_symbol", "")
        return {
            "node_id": row.get("node_id", ""),
            "symbol_name": symbol_name,
            "symbol_type": symbol_type,
            "layer": classify_symbol_layer(
                symbol_type=symbol_type,
                symbol_name=symbol_name,
                parent_symbol=parent_symbol,
                file_path=rel_path,
                docstring=row.get("docstring", ""),
            ),
            "file_path": rel_path,
            "rel_path": rel_path,
            "language": row.get("language", ""),
            "start_line": row.get("start_line", ""),
            "end_line": row.get("end_line", ""),
            "parent_symbol": parent_symbol,
            "docstring": row.get("docstring", ""),
            "content": row.get("source_text", ""),
        }

    # ------------------------------------------------------------------
    # Core search methods
    # ------------------------------------------------------------------

    def search(
        self,
        query: str,
        k: int = 10,
        symbol_types: frozenset[str] | None = None,
        exclude_types: frozenset[str] | None = None,
        language: str | None = None,
    ) -> list[Document]:
        st_list = sorted(symbol_types) if symbol_types else None
        rows = self._storage.search_fts(
            query=query,
            symbol_types=st_list,
            limit=k,
        )
        if exclude_types:
            rows = [r for r in rows if r.get("symbol_type") not in exclude_types]
        if language:
            rows = [r for r in rows if (r.get("language") or "").lower() == language.lower()]
        return [self._row_to_document(r) for r in rows]

    def search_by_name(
        self,
        name: str,
        exact: bool = False,
        k: int = 10,
        symbol_types: frozenset[str] | None = None,
        language: str | None = None,
    ) -> list[Document]:
        rows = self._storage.search_fts_by_symbol_name(
            name=name,
            architectural_only=False,
            limit=k,
        )
        if exact:
            rows = [r for r in rows if r.get("symbol_name") == name]
        if symbol_types:
            rows = [r for r in rows if r.get("symbol_type") in symbol_types]
        if language:
            rows = [r for r in rows if (r.get("language") or "").lower() == language.lower()]
        return [self._row_to_document(r, score=0.0) for r in rows[:k]]

    def _get_all_nodes(self, limit: int = 5000) -> list[dict[str, Any]]:
        """Retrieve all nodes via path-prefix with a universal glob."""
        # get_nodes_by_path_prefix("") produces GLOB "/*" which misses
        # relative paths.  Use a known top-level prefix "*" trick or
        # the dedicated get_all_nodes helper when available.
        if hasattr(self._storage, "get_all_nodes"):
            return self._storage.get_all_nodes()[:limit]
        # Fallback: broad prefix search
        return self._storage.get_nodes_by_path_prefix("", limit=limit)

    def search_by_type(self, symbol_type: str, k: int = 100) -> list[Document]:
        rows = self._get_all_nodes(limit=k * 5)
        rows = [r for r in rows if r.get("symbol_type") == symbol_type][:k]
        return [self._row_to_document(r, score=0.0) for r in rows]

    def search_by_layer(
        self,
        layer: str,
        symbol_types: frozenset[str] | None = None,
        k: int = 100,
    ) -> list[Document]:
        st_list = set(symbol_types) if symbol_types else None
        rows = self._get_all_nodes(limit=k * 5)
        if st_list:
            rows = [r for r in rows if r.get("symbol_type") in st_list]

        docs = []
        for row in rows:
            computed_layer = classify_symbol_layer(
                symbol_type=row.get("symbol_type", ""),
                symbol_name=row.get("symbol_name", ""),
                parent_symbol=row.get("parent_symbol", ""),
                file_path=row.get("rel_path", ""),
                docstring=row.get("docstring", ""),
            )
            if computed_layer != layer:
                continue
            docs.append(self._row_to_document(row, score=0.0))
            if len(docs) >= k:
                break
        return docs

    def search_by_path_prefix(
        self,
        path_prefix: str,
        symbol_types: frozenset[str] | None = None,
        exclude_types: frozenset[str] | None = None,
        k: int = 500,
    ) -> list[dict[str, Any]]:
        rows = self._storage.get_nodes_by_path_prefix(path_prefix, limit=k)
        if symbol_types:
            rows = [r for r in rows if r.get("symbol_type") in symbol_types]
        if exclude_types:
            rows = [r for r in rows if r.get("symbol_type") not in exclude_types]
        return [self._row_to_dict(r) for r in rows[:k]]

    # ------------------------------------------------------------------
    # Intent-aware / compound search
    # ------------------------------------------------------------------

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
        try:
            from ..code_graph.graph_query_builder import GraphQueryBuilder

            fts_keywords = GraphQueryBuilder.from_natural_language(query, intent=intent)
        except Exception:
            fts_keywords = None

        search_query = fts_keywords if fts_keywords else query
        return self.search(
            search_query,
            k=k,
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
        fetch_k = k * 3 if path_prefix else k

        docs = self.search_smart(
            query,
            k=fetch_k,
            intent="concept",
            symbol_types=symbol_types,
            exclude_types=exclude_types,
            language=language,
        )

        if path_prefix and docs:
            prefix = path_prefix.strip("/")
            docs = [d for d in docs if (d.metadata.get("rel_path") or "").startswith(prefix)]

        return docs[:k]

    # ------------------------------------------------------------------
    # Direct lookups
    # ------------------------------------------------------------------

    def get_by_node_id(self, node_id: str) -> dict[str, Any] | None:
        row = self._storage.get_node(node_id)
        return self._row_to_dict(row) if row else None
