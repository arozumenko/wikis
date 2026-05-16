"""
WikiStorageProtocol — backend-agnostic interface for the unified wiki DB.

This Protocol captures every public method that consumers (UnifiedRetriever,
cluster_expansion, filesystem_indexer, wiki_graph_optimized, toolkit_bridge,
ask_engine, research_engine, etc.) need from the storage layer.

Both ``SqliteWikiStorage`` (the SQLite implementation, with
``UnifiedWikiDB`` kept as a compatibility alias) and
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
        self, macro: int, micro: int | None = None, limit: int | None = 1000,
    ) -> list[dict[str, Any]]:
        """Get all nodes in a macro (optionally micro) cluster.

        Pass ``limit=None`` to disable the row cap (matches pre-protocol
        raw SQL behaviour where the caller expected every row).
        """
        ...

    def get_architectural_nodes(
        self, limit: int | None = 5000,
    ) -> list[dict[str, Any]]:
        """Get all nodes flagged ``is_architectural = 1``.

        Pass ``limit=None`` to disable the row cap.
        """
        ...

    def get_all_nodes(self) -> list[dict[str, Any]]:
        """Get every node in the database (used for cache validation)."""
        ...

    def node_count(self) -> int:
        """Total number of nodes."""
        ...

    def refresh_fts_index(self) -> None:
        """Rebuild the FTS index from ``repo_nodes`` (full refresh).

        SQLite stores FTS5 as a standalone virtual table without triggers
        on ``repo_nodes``, so the index goes stale after partial upserts.
        Postgres uses a BEFORE INSERT OR UPDATE trigger so its FTS column
        stays in sync automatically — this method is a no-op on Postgres.

        Prefer :meth:`apply_incremental_node_writes` when you have an
        explicit batch — it bundles the upsert + refresh so callers
        cannot leave FTS stale.
        """
        ...

    def apply_incremental_node_writes(self, nodes: list[dict[str, Any]]) -> None:
        """Upsert a batch of nodes and refresh the FTS index in one call.

        Closes the "did the author remember to call refresh_fts_index?"
        footgun flagged by issue #131. Callers in [#116] PR 3+ should use
        this for any partial node upsert path (the full ``from_networkx``
        flow already rebuilds FTS as its last step, so it doesn't need
        this).
        """
        ...

    def fetch_indexed_node_meta(self) -> dict[str, dict[str, str | None]]:
        """Return ``{node_id: {"content_hash": ..., "rel_path": ...}}``
        for every indexed node.

        Used by [#116] PR 2 change detection as the "previous state" the
        detector diffs against. Combined into a single full-table scan so
        the detector doesn't need a second batch lookup of paths — which
        would hit SQLite's ``SQLITE_MAX_VARIABLE_NUMBER`` limit on large
        wikis.

        ``content_hash`` may be ``None`` for nodes indexed before PR 1
        landed — callers treat those as "hash unknown", which forces a
        re-parse comparison rather than a hash equality check.

        Returns an empty dict for an unindexed wiki — callers must handle
        that as "first generation, everything is new".
        """
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
        architectural_only: bool | None = None,
        max_workers: int | None = None,
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

    def find_node_exact(
        self,
        symbol_name: str,
        rel_path: str,
        language: str,
    ) -> dict[str, Any] | None:
        """Resolve a node by the exact ``(symbol_name, rel_path, language)`` triple.

        Storage-native equivalent of the NX ``_node_index`` lookup used by
        ``GraphQueryService.resolve_symbol`` strategy 1. When multiple rows
        match, the candidate with the highest architectural type priority
        wins (class > interface > ... > variable).
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

    def reset_clusters(self) -> None:
        """Clear ``macro_cluster``, ``micro_cluster``, ``is_hub``, and
        ``hub_assignment`` on every node row.

        Called before writing a fresh cluster-assignment pass so stale
        values from a previous cached run don't survive.  On SQLite the
        UPDATE runs in the current transaction — the caller is responsible
        for invoking :meth:`commit` once subsequent cluster writes are
        finished.  On Postgres the write is auto-committed.
        """
        ...

    def get_hub_node_ids(self) -> list[str]:
        """Return ``node_id`` values of all nodes flagged ``is_hub = 1``."""
        ...

    def get_clustered_architectural_nodes(
        self,
        exclude_tests: bool = False,
    ) -> list[dict[str, Any]]:
        """Return rows ``{node_id, macro_cluster, micro_cluster}`` for every
        architectural node assigned to a macro cluster.

        When *exclude_tests* is True, rows with ``is_test = 1`` are skipped.
        Used by the cluster planner to build its
        ``{macro → {micro → [node_ids]}}`` index from raw rows.
        """
        ...

    def get_architectural_node_ids(
        self,
        exclude_tests: bool = False,
        limit: int | None = 10_000,
    ) -> list[str]:
        """Return ``node_id`` values for all architectural nodes.

        Lightweight projection used when only IDs are needed (e.g. populating
        a NetworkX graph for PageRank).  When *exclude_tests* is True, rows
        with ``is_test = 1`` are skipped.  Pass ``limit=None`` to disable
        the row cap.
        """
        ...

    def get_all_edges(
        self,
        limit: int | None = 500_000,
    ) -> list[dict[str, Any]]:
        """Return all edge rows with at minimum ``source_id``, ``target_id``,
        ``rel_type``, ``weight``.

        Used for bulk graph reconstruction (PageRank, topology analysis).
        The *limit* parameter guards against OOM on very large repos; pass
        ``None`` to disable.
        """
        ...

    # ── Language detection ───────────────────────────────────────────

    def detect_dominant_language(self, node_ids: list[str]) -> str | None:
        """Return the most common ``language`` value among *node_ids*, or None."""
        ...

    # ==================================================================
    # POST-PASS HYGIENE / OBSERVABILITY (§11.7, §11.8)
    # ==================================================================

    def demote_local_constants(self) -> int:
        """Tag `constant` symbols whose every reference is in their defining file
        as ``is_architectural = 0``.

        Implements §11.8 (P4 local-leakage detection).  Constants with **zero**
        external references are kept architectural — they may be unused public
        API.  Returns the number of rows demoted.
        """
        ...

    def demote_vendored_code(self) -> int:
        """Tag nodes living under vendored / third-party paths as
        ``is_architectural = 0`` (§11.12, P7).

        The detection is purely path-based and conservative; the patterns
        target obvious vendored locations (``third_party/``, ``vendor/``,
        ``external/``, bundled ``gtest`` / ``gmock`` / ``googletest`` /
        ``googlemock`` segments).  Demoted nodes remain fully retrievable
        — only page eligibility and god-node ranking are affected.
        Returns the number of rows demoted.
        """
        ...

    def promote_class_members(self) -> int:
        """Synthesize ``member_uses`` edges from class members to cross-file
        targets they reference (§11.4, P1).

        For each architectural class / struct / interface / trait C, walks
        its ``defines`` children whose ``symbol_type`` is in
        ``{'method','field','property','constructor'}`` and aggregates
        their *cross-file* outgoing references / calls / instantiations.
        Emits one synthetic edge per (C, T) pair with
        ``rel_type='member_uses'``, ``weight=<evidence count>`` and
        ``confidence='INFERRED'``.

        Skip rule (per A2): no edge is emitted if ``(C, *, T)`` already
        exists with a ``rel_type != 'defines'`` — don't double-count
        relationships the parser already attributed to C itself.

        Self-loops (T == C) are excluded.  Returns the number of edges
        actually inserted.
        """
        ...

    def compute_god_nodes(self, top_n: int = 20) -> dict[str, Any]:
        """Return the top-N over-connected symbols and files (§11.7 / B2).

        Pure analytics; no behaviour change.  Output shape::

            {
                "by_symbol_type": [
                    {"symbol_id", "name", "symbol_type", "rel_path", "degree"},
                    ...
                ],
                "by_file": [
                    {"rel_path", "internal_arch", "external_edges"},
                    ...
                ],
            }
        """
        ...

    def shortest_path(
        self,
        source_label: str,
        target_label: str,
        max_depth: int = 25,
    ) -> dict[str, Any]:
        """Undirected shortest path between two symbols (#121).

        **Canonical entry point for undirected shortest-path queries
        from MCP / IDE clients.** Implementations layer a Python BFS
        with a visited set on top of a single SQL frontier-expansion
        query per depth — frontier growth stays O(V+E) so the search
        terminates quickly even on dense graphs at large
        ``max_depth``.

        For *directed* per-symbol traversal used by the agent loop
        (e.g. "who calls X?"), use
        :meth:`GraphQueryService.get_relationships` — that surface is
        in-memory NetworkX, depth-bounded, and respects edge direction
        and confidence filters. The two are intentionally separate so
        each can evolve independently.

        Label resolution: ``source_label`` and ``target_label`` are
        matched against ``symbol_name`` first, then ``rel_path``;
        within each table the lowest ``node_id`` wins. Implementations
        also report the number of candidates so callers can detect
        ambiguous matches.

        ``max_depth`` defaults to 25 — graph diameter on production
        codebases is typically well under 20. Callers can pass a
        smaller value for faster responses on dense graphs.

        Returns:
            On success::

                {
                    "source": {"node_id", "symbol_name", "rel_path", "symbol_type"},
                    "target": {<same>},
                    "source_candidates": int,   # total labels matching source_label
                    "target_candidates": int,   # ditto for target_label
                    "path": [<node row>, ...],  # source first, target last
                    "edges": [{"source_id", "target_id", "rel_type", "confidence"}, ...],
                    "length": int,
                }

            ``source_candidates > 1`` (or ``target_candidates > 1``)
            indicates the label was ambiguous and a different
            resolution might yield a different path.

            When either label cannot be resolved or no path is found
            within ``max_depth``::

                {"path": None, "reason": "..."}

            ``reason`` is a short tag for programmatic handling:
            ``source_not_found``, ``target_not_found``,
            ``no_path_within_max_depth``, ``invalid_max_depth``.
            Same-source-and-target is a success case (``length == 0``,
            no ``reason``) so callers don't have to special-case it.
        """
        ...

    def compute_surprising_connections(
        self,
        top_n: int = 10,
        context_depth: int = 1,
        sample_edges_per_pair: int = 3,
    ) -> dict[str, Any]:
        """Cross-cluster edge pairs ranked by surprise (#121 Phase 2).

        A "surprising connection" is an edge between two macro
        clusters whose **contexts** (the set of top-level folder
        prefixes containing their nodes) are highly disjoint.
        Quantified as Jaccard distance: ``1 - |A ∩ B| / |A ∪ B|``
        over the two clusters' context sets.

        Implementations should:
          1. Aggregate cross-cluster edges per unordered cluster pair
             ``(min, max)`` to count edges and gather a canonical
             representative pair list (no double-counting).
          2. For each cluster appearing in those pairs, compute the
             context — the set of ``rel_path`` prefixes truncated to
             ``context_depth`` segments. ``context_depth=1`` means
             top-level folder names (e.g. ``frontend``, ``backend``).
          3. Compute Jaccard distance per pair using cached contexts.
          4. Return the top-``top_n`` pairs sorted by Jaccard distance
             descending, each with up to
             ``sample_edges_per_pair`` example edges + hydrated source
             / target metadata so MCP clients have something concrete
             to surface.

        Args:
            top_n: Maximum pairs to return.
            context_depth: How many leading ``rel_path`` segments
                count as the cluster's context (1 = top-level folder).
            sample_edges_per_pair: How many example edges to include
                per pair.

        Returns:
            Dict with ``pairs`` (list, sorted by ``jaccard_distance``
            descending). Each pair has ``cluster_a``, ``cluster_b``
            (always ``cluster_a < cluster_b``), ``jaccard_distance``
            (0.0 - 1.0), ``context_a``, ``context_b`` (sorted prefix
            lists), ``edge_count``, and ``sample_edges`` (list of
            ``{source_id, target_id, rel_type, confidence,
            source_name, source_path, target_name, target_path}``).
            Empty ``pairs`` list when no cross-cluster edges exist.
        """
        ...

    # ==================================================================
    # WIKI PAGES + source→page reverse index (#116 incremental regen)
    # ==================================================================

    def upsert_wiki_page(self, page: dict[str, Any]) -> None:
        """Insert or replace a row in ``wiki_pages``.

        ``page`` must include ``page_id`` and ``wiki_id``. All other columns
        (``id_scheme``, ``title``, ``anchor_slug``, ``content_hash``,
        ``macro_cluster``, ``micro_cluster``, ``primary_symbol_id``,
        ``section_index``, ``page_index``, ``generated_at``) are optional and
        fall back to schema defaults when omitted.
        """
        ...

    def get_wiki_page(self, page_id: str) -> dict[str, Any] | None:
        """Fetch one page row by its ``page_id``. Returns None if missing."""
        ...

    def get_wiki_pages(self, wiki_id: str) -> list[dict[str, Any]]:
        """Return all page rows for a wiki, in (section_index, page_index) order."""
        ...

    def get_wiki_pages_by_cluster(
        self,
        wiki_id: str,
        macro_cluster: int,
        micro_cluster: int | None = None,
    ) -> list[dict[str, Any]]:
        """Return page rows for a (macro, micro) cluster within a wiki.

        Used by [#116] PR 2/3 to find pages affected by a cluster-level
        change. When ``micro_cluster`` is None, every page in the macro
        cluster is returned regardless of micro.
        """
        ...

    def delete_wiki_pages(self, wiki_id: str) -> int:
        """Remove all rows for a wiki from ``wiki_pages`` (cascades to
        ``page_symbols``). Returns the number of rows deleted.
        """
        ...

    def delete_wiki_page(self, page_id: str) -> bool:
        """Remove a single ``wiki_pages`` row by primary key.

        FK CASCADE on ``page_symbols.page_id`` removes its rows in the
        same transaction. Returns True when a row was deleted, False
        when no row existed for ``page_id`` (idempotent).

        Used by #141's DELETED regime to drop pages whose entire cluster
        vanished — the orchestrator needs a per-page primitive because
        ``delete_wiki_pages(wiki_id)`` would also remove pages that are
        still alive.
        """
        ...

    def record_page_symbols(
        self,
        page_id: str,
        symbols: list[tuple[str, str]],
        *,
        replace: bool = True,
    ) -> None:
        """Persist the source→page reverse index for one page.

        Args:
            page_id: the wiki page these symbols belong to.
            symbols: list of ``(node_id, citation_kind)`` tuples where
                citation_kind is ``'primary'``, ``'referenced'``, or
                ``'related'``.
            replace: when True (default), all existing rows for ``page_id``
                are deleted before inserting the new set — this is the
                desired behaviour after a regen. When False, rows are
                appended via INSERT OR IGNORE / ON CONFLICT DO NOTHING.
        """
        ...

    def upsert_wiki_page_with_symbols(
        self,
        page: dict[str, Any],
        symbols: list[tuple[str, str]],
        *,
        replace: bool = True,
    ) -> None:
        """Atomically upsert a ``wiki_pages`` row and its ``page_symbols`` rows.

        Equivalent to calling ``upsert_wiki_page(page)`` followed by
        ``record_page_symbols(page['page_id'], symbols, replace=replace)``
        inside a single transaction. If any step raises, both writes roll
        back together so callers never observe a page with stale (or
        missing) citations.

        Used by the wiki generation agent's per-page persistence hook,
        where partial writes would leave an orphaned page row whose
        absence of citation entries is indistinguishable from a page that
        legitimately cites nothing.
        """
        ...

    def get_pages_citing_node(self, node_id: str) -> list[str]:
        """Return the ``page_id``s of every page that cites ``node_id``.

        Used by [#116] change detection to find which pages need regen
        when an underlying symbol changes.
        """
        ...

    def get_page_symbols(
        self, page_id: str, citation_kind: str | None = None,
    ) -> list[dict[str, Any]]:
        """Return rows for one page: ``[{node_id, citation_kind}, ...]``.

        When ``citation_kind`` is supplied, only that kind is returned.
        """
        ...
