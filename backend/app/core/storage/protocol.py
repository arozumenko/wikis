"""
WikiStorageProtocol — backend-agnostic interface for the unified wiki DB.

This Protocol captures every public method that consumers (UnifiedRetriever,
cluster_expansion, filesystem_indexer, wiki_graph_optimized, toolkit_bridge,
ask_engine, research_engine, etc.) need from the storage layer.

Both ``SqliteWikiStorage`` (wrapping the existing UnifiedWikiDB) and
``PostgresWikiStorage`` conform to this protocol.

Design principles:
- No raw SQL exposed — every query pattern used in the codebase has a
  dedicated method so backends can implement using native syntax
  (sqlite3 vs SQLAlchemy Core + pgvector).
- All methods accept and return plain dicts — no ORM objects, no Row types.
- Search methods (FTS, vector, hybrid) return dicts with the same key names
  regardless of backend.
- ``db_path`` property preserved for consumers that need a cache key or
  file path (returns ``Path("postgres")`` for PostgreSQL backend).
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

import networkx as nx


@runtime_checkable
class WikiStorageProtocol(Protocol):
    """Backend-agnostic interface for the unified wiki database.

    All storage backends (SQLite, PostgreSQL, etc.) must implement
    every method listed here.  Consumers should type-hint parameters
    as ``WikiStorageProtocol`` rather than importing a concrete class.
    """

    # ==================================================================
    # Properties
    # ==================================================================

    @property
    def db_path(self) -> Path:
        """Path to the database file (SQLite) or sentinel ``Path("postgres")``."""
        ...

    @property
    def vec_available(self) -> bool:
        """Whether vector search (sqlite-vec / pgvector) is available."""
        ...

    @property
    def embedding_dim(self) -> int:
        """Configured embedding dimension."""
        ...

    # ==================================================================
    # Lifecycle
    # ==================================================================

    def close(self) -> None:
        """Release resources (connections, files)."""
        ...

    def __enter__(self) -> WikiStorageProtocol:
        ...

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> bool:
        ...

    # ==================================================================
    # NODE operations
    # ==================================================================

    def upsert_node(self, node_id: str, **attrs: Any) -> None:
        """Insert or replace a single node."""
        ...

    def upsert_nodes_batch(self, nodes: list[dict[str, Any]]) -> None:
        """Bulk upsert nodes. Each dict must contain ``node_id``."""
        ...

    def get_node(self, node_id: str) -> dict[str, Any] | None:
        """Fetch one node by ID. Returns dict or None."""
        ...

    def get_nodes_by_ids(self, node_ids: list[str]) -> list[dict[str, Any]]:
        """Batch fetch nodes by IDs. Returns list of dicts (order not guaranteed)."""
        ...

    def get_nodes_by_path_prefix(self, prefix: str, limit: int = 500) -> list[dict[str, Any]]:
        """Get nodes whose ``rel_path`` starts with *prefix*."""
        ...

    def get_nodes_by_cluster(
        self, macro: int, micro: int | None = None, limit: int = 1000,
    ) -> list[dict[str, Any]]:
        """Get all nodes in a macro (optionally micro) cluster."""
        ...

    def get_architectural_nodes(self, limit: int = 5000) -> list[dict[str, Any]]:
        """Get all nodes flagged ``is_architectural = 1``."""
        ...

    def get_all_nodes(self) -> list[dict[str, Any]]:
        """Get every node in the database (used for cache validation)."""
        ...

    def node_count(self) -> int:
        """Total number of nodes."""
        ...

    # ==================================================================
    # EDGE operations
    # ==================================================================

    def upsert_edge(self, source_id: str, target_id: str, rel_type: str, **attrs: Any) -> None:
        """Insert a single edge."""
        ...

    def upsert_edges_batch(self, edges: list[dict[str, Any]]) -> None:
        """Bulk insert edges."""
        ...

    def get_edges_from(
        self, node_id: str, rel_types: list[str] | None = None,
    ) -> list[dict[str, Any]]:
        """Get outgoing edges from a node, optionally filtered by type."""
        ...

    def get_edges_to(self, node_id: str) -> list[dict[str, Any]]:
        """Get incoming edges to a node."""
        ...

    def get_neighbors(
        self, node_id: str, hops: int = 1, direction: str = "out",
    ) -> set[str]:
        """Get neighbor node IDs within *hops* distance."""
        ...

    def edge_count(self) -> int:
        """Total number of edges."""
        ...

    def update_edge_weights_batch(self, updates: list[dict[str, Any]]) -> int:
        """Batch-update edge weights. Each dict needs ``id`` and ``weight``."""
        ...

    # ==================================================================
    # SEARCH — Full-Text Search
    # ==================================================================

    def search_fts(
        self,
        query: str,
        path_prefix: str | None = None,
        cluster_id: int | None = None,
        symbol_types: list[str] | None = None,
        limit: int = 20,
    ) -> list[dict[str, Any]]:
        """BM25-ranked text search with optional filtering.

        Returns list of dicts with node metadata + ``fts_rank`` score.
        Implementations use FTS5 (SQLite) or tsvector/tsquery (PostgreSQL).
        """
        ...

    # ==================================================================
    # SEARCH — Vector (KNN)
    # ==================================================================

    def upsert_embedding(self, node_id: str, embedding: list[float]) -> None:
        """Insert or replace a single embedding."""
        ...

    def upsert_embeddings_batch(
        self, embeddings: list[tuple[str, list[float]]],
    ) -> None:
        """Bulk insert embeddings as (node_id, vector) pairs."""
        ...

    def populate_embeddings(
        self,
        embedding_fn: Callable[[list[str]], list[list[float]]],
        batch_size: int = 64,
    ) -> int:
        """Embed all nodes' text and populate vectors. Returns count stored."""
        ...

    def search_vec(
        self,
        embedding: list[float],
        k: int = 10,
        path_prefix: str | None = None,
        cluster_id: int | None = None,
    ) -> list[dict[str, Any]]:
        """KNN vector search. Returns dicts with ``vec_distance``."""
        ...

    # ==================================================================
    # SEARCH — Hybrid (RRF fusion)
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
        """Hybrid FTS + Vector RRF search. Falls back to FTS-only when no vectors."""
        ...

    # ==================================================================
    # CLUSTER operations
    # ==================================================================

    def set_cluster(self, node_id: str, macro: int, micro: int | None = None) -> None:
        """Assign cluster IDs to a node."""
        ...

    def set_clusters_batch(
        self, assignments: list[tuple[str, int, int | None]],
    ) -> None:
        """Batch assign clusters: list of (node_id, macro, micro) tuples."""
        ...

    def set_hub(
        self, node_id: str, is_hub: bool = True, assignment: str | None = None,
    ) -> None:
        """Flag a node as a hub with optional assignment text."""
        ...

    def get_cluster_nodes(self, macro: int, micro: int | None = None) -> list[str]:
        """Return node IDs in a cluster."""
        ...

    def get_all_clusters(self) -> dict[int, dict[int, list[str]]]:
        """Return nested dict: macro_id → micro_id → [node_ids]."""
        ...

    # ==================================================================
    # GRAPH I/O — NetworkX conversion
    # ==================================================================

    def from_networkx(self, G: nx.MultiDiGraph) -> None:
        """Bulk-import a NetworkX MultiDiGraph."""
        ...

    def to_networkx(self) -> nx.MultiDiGraph:
        """Reconstruct a NetworkX MultiDiGraph from the DB."""
        ...

    # ==================================================================
    # METADATA
    # ==================================================================

    def set_meta(self, key: str, value: Any) -> None:
        """Store a metadata key-value pair (JSON-encoded)."""
        ...

    def get_meta(self, key: str, default: Any = None) -> Any:
        """Retrieve a metadata value."""
        ...

    # ==================================================================
    # STATISTICS
    # ==================================================================

    def stats(self) -> dict[str, Any]:
        """Return summary statistics about the DB contents."""
        ...

    # ==================================================================
    # Cluster-expansion queries (replace raw conn.execute usage)
    # ==================================================================

    def find_nodes_by_name(
        self,
        name: str,
        macro_cluster: int | None = None,
        architectural_only: bool = True,
        limit: int = 5,
    ) -> list[dict[str, Any]]:
        """Find nodes by symbol_name, optionally bounded to a cluster.

        Used by cluster_expansion._resolve_symbols() to replace raw SQL.
        Results are ordered by code span (end_line - start_line) descending.
        """
        ...

    def search_fts_by_symbol_name(
        self,
        name: str,
        macro_cluster: int | None = None,
        architectural_only: bool = True,
        limit: int = 3,
    ) -> list[dict[str, Any]]:
        """FTS fallback for symbol name resolution.

        Used by cluster_expansion._fts_fallback() to replace raw SQL.
        """
        ...

    def get_edge_targets(self, source_id: str) -> list[dict[str, Any]]:
        """Get lightweight outgoing edge tuples: [{target_id, rel_type, weight}].

        Used by cluster_expansion for neighbor collection without joining
        full node metadata.
        """
        ...

    def get_edge_sources(self, target_id: str) -> list[dict[str, Any]]:
        """Get lightweight incoming edge tuples: [{source_id, rel_type, weight}].

        Used by cluster_expansion for neighbor collection without joining
        full node metadata.
        """
        ...

    def find_related_nodes(
        self,
        node_id: str,
        rel_types: list[str],
        direction: str = "out",
        target_symbol_types: list[str] | None = None,
        exclude_path: str | None = None,
        limit: int = 15,
    ) -> list[dict[str, Any]]:
        """Find nodes related via specific edge types (JOIN edges + nodes).

        Used by cluster_expansion language-specific augmenters (_augment_cpp,
        _augment_go_rust) to replace raw JOIN queries.

        Parameters
        ----------
        node_id : str
            Anchor node.
        rel_types : list[str]
            Edge types to follow (e.g. ['defines_body', 'implementation']).
        direction : str
            'out' = follow outgoing edges; 'in' = follow incoming edges.
        target_symbol_types : list[str], optional
            Only return targets matching these symbol types.
        exclude_path : str, optional
            Exclude nodes with this rel_path.
        limit : int
            Maximum results.

        Returns
        -------
        list[dict]
            Node dicts with full metadata, ordered by (rel_path, start_line).
        """
        ...

    def get_doc_nodes_by_cluster(
        self,
        macro: int,
        micro: int | None = None,
    ) -> list[dict[str, Any]]:
        """Get documentation nodes (is_doc=1) from a cluster.

        Used by cluster_expansion._get_cluster_docs().
        """
        ...

    def commit(self) -> None:
        """Explicitly commit the current transaction (no-op for auto-commit backends)."""
        ...

    # ── Bulk operations ──────────────────────────────────────────────

    def delete_all_edges(self) -> None:
        """Delete every row in the edges table (full replace before re-persist)."""
        ...

    # ── Language detection ───────────────────────────────────────────

    def detect_dominant_language(self, node_ids: list[str]) -> str | None:
        """Return the most common ``language`` value among *node_ids*, or None."""
        ...
