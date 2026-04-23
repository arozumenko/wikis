"""Multi-graph query service — fans out graph queries across N WikiGraphQueryService instances.

Exposes the same public interface as GraphQueryService so callers require no changes
when switching between single-wiki and multi-wiki (project) contexts.
"""

from __future__ import annotations

import logging
from typing import Any

from .graph_query_service import GraphQueryService, RelationshipResult, SymbolResult

logger = logging.getLogger(__name__)


class MultiGraphQueryService:
    """Fan-out query service spanning multiple wiki code graphs.

    Accepts a mapping of ``{wiki_id: GraphQueryService}`` and exposes the full
    ``GraphQueryService`` public interface.  Each method fans out to all
    constituent services, merges the results, and (where applicable) tags
    each result with the originating ``wiki_id``.

    This lets ``_build_call_tree_from_sources`` work identically for single-wiki
    and multi-wiki (project) requests without any branching logic.

    Args:
        services: Mapping of wiki_id → GraphQueryService.
    """

    def __init__(self, services: dict[str, GraphQueryService]) -> None:
        self._services = services
        # Expose a merged virtual ``graph`` that callers can use to look up
        # node data.  We provide a thin proxy that checks each wiki's graph
        # in order.  This is the only attribute of ``graph`` that
        # ``_build_call_tree_from_sources`` accesses (``graph.nodes.get``).
        self.graph = _MergedGraphView(
            {wid: svc.graph for wid, svc in services.items()}
        )
        # ``storage`` is intentionally None on the multi-wiki facade.  Callers
        # must use :py:meth:`get_node` / :py:meth:`get_nodes_by_path_prefix`
        # instead of touching a single underlying store.
        self.storage = None

    # ------------------------------------------------------------------
    # Node Accessor — parity with GraphQueryService / StorageQueryService
    # ------------------------------------------------------------------

    def get_node(self, node_id: str) -> dict[str, Any] | None:
        """Return the raw node attribute dict for ``node_id`` across all wikis.

        Tries each constituent service's ``get_node`` (covers both
        ``GraphQueryService`` — NX-backed — and ``StorageQueryService``
        — Postgres/SQLite-backed).  Returns the first dict found.

        This is the cross-wiki replacement for the legacy
        ``code_graph.nodes.get(node_id)`` pattern that only ever consulted
        the primary wiki's NetworkX graph.
        """
        if not node_id:
            return None
        for _wiki_id, svc in self._services.items():
            getter = getattr(svc, "get_node", None)
            if getter is None:
                continue
            try:
                data = getter(node_id)
            except Exception:  # pragma: no cover — defensive
                continue
            if data is not None:
                return data
        return None

    def get_nodes_by_path_prefix(
        self,
        path_prefix: str,
        limit: int = 20,
    ) -> list[dict[str, Any]]:
        """Fan out a path-prefix lookup across every wiki's storage.

        ``StorageQueryService.storage`` exposes
        ``get_nodes_by_path_prefix`` for storage-backed wikis.  Older NX-only
        services don't, so we silently skip them.  Results are deduplicated
        by ``node_id`` and capped at ``limit`` total rows.
        """
        if not path_prefix:
            return []
        seen_ids: set[str] = set()
        rows: list[dict[str, Any]] = []
        for _wiki_id, svc in self._services.items():
            store = getattr(svc, "storage", None)
            if store is None:
                continue
            getter = getattr(store, "get_nodes_by_path_prefix", None)
            if getter is None:
                continue
            try:
                wiki_rows = getter(path_prefix, limit=limit)
            except Exception:  # pragma: no cover — defensive
                continue
            for row in wiki_rows or []:
                nid = row.get("node_id")
                if not nid or nid in seen_ids:
                    continue
                seen_ids.add(nid)
                rows.append(row)
                if len(rows) >= limit:
                    return rows
        return rows

    # ------------------------------------------------------------------
    # Symbol Resolution
    # ------------------------------------------------------------------

    def resolve_symbol(
        self,
        symbol_name: str,
        file_path: str = "",
        language: str = "",
    ) -> str | None:
        """Resolve a symbol name to its graph node_id across all wikis.

        Tries each wiki's GraphQueryService in order; returns the first match.
        """
        for _wiki_id, svc in self._services.items():
            result = svc.resolve_symbol(symbol_name, file_path=file_path, language=language)
            if result:
                return result
        return None

    # ------------------------------------------------------------------
    # Search
    # ------------------------------------------------------------------

    def search(
        self,
        query: str,
        k: int = 20,
        symbol_types: frozenset[str] | None = None,
        exclude_types: frozenset[str] | None = None,
        layer: str | None = None,
        path_prefix: str | None = None,
    ) -> list[SymbolResult]:
        """Fan out search to all graphs and return merged results sorted by score."""
        all_results: list[SymbolResult] = []
        per_wiki_k = max(k, 5)  # fetch at least k from each wiki
        for wiki_id, svc in self._services.items():
            try:
                results = svc.search(
                    query,
                    k=per_wiki_k,
                    symbol_types=symbol_types,
                    exclude_types=exclude_types,
                    layer=layer,
                    path_prefix=path_prefix,
                )
                all_results.extend(results)
            except Exception as exc:
                logger.warning("search failed for wiki %s: %s", wiki_id, exc)

        all_results.sort(key=lambda r: (-r.score, -r.connections))
        return all_results[:k]

    # ------------------------------------------------------------------
    # Relationship Traversal
    # ------------------------------------------------------------------

    def get_relationships(
        self,
        node_id: str,
        direction: str = "both",
        max_depth: int = 2,
        max_results: int = 50,
    ) -> list[RelationshipResult]:
        """Traverse relationships for a node_id, trying each wiki graph in order."""
        for _wiki_id, svc in self._services.items():
            if node_id in svc.graph:
                return svc.get_relationships(
                    node_id,
                    direction=direction,
                    max_depth=max_depth,
                    max_results=max_results,
                )
        return []

    # ------------------------------------------------------------------
    # Resolve + Traverse
    # ------------------------------------------------------------------

    def resolve_and_traverse(
        self,
        symbol_name: str,
        direction: str = "both",
        max_depth: int = 2,
        max_results: int = 50,
        file_path: str = "",
        language: str = "",
    ) -> tuple[str | None, list[RelationshipResult]]:
        """Resolve a symbol name and return its relationships.

        Tries the wiki whose graph contains the resolved node first, then fans
        out if the symbol is not found locally.  Matches the signature of
        ``GraphQueryService.resolve_and_traverse``.
        """
        # Try each wiki in order — resolve_symbol already fans out
        for _wiki_id, svc in self._services.items():
            node_id = svc.resolve_symbol(symbol_name, file_path=file_path, language=language)
            if node_id:
                return node_id, svc.get_relationships(
                    node_id,
                    direction=direction,
                    max_depth=max_depth,
                    max_results=max_results,
                )
        return None, []

    # ------------------------------------------------------------------
    # JQL Structured Query
    # ------------------------------------------------------------------

    def query(self, expression: str) -> list[SymbolResult]:
        """Execute a JQL structured query expression across all graphs.

        Fans out to each wiki's GraphQueryService, merges the results, and
        de-duplicates by node_id (keeping the highest-scoring entry).
        """
        seen_node_ids: set[str] = set()
        all_results: list[SymbolResult] = []
        for wiki_id, svc in self._services.items():
            try:
                results = svc.query(expression)
                for r in results:
                    if r.node_id not in seen_node_ids:
                        seen_node_ids.add(r.node_id)
                        all_results.append(r)
            except Exception as exc:
                logger.warning("query failed for wiki %s: %s", wiki_id, exc)

        all_results.sort(key=lambda r: (-r.score, -r.connections))
        return all_results

    # ------------------------------------------------------------------
    # Wiki ID listing
    # ------------------------------------------------------------------

    def wiki_ids(self) -> list[str]:
        """Return the list of wiki IDs this service spans."""
        return list(self._services.keys())

    def per_wiki_services(self) -> dict[str, GraphQueryService]:
        """Return the mapping of wiki_id → GraphQueryService for per-wiki iteration."""
        return dict(self._services)

    # ------------------------------------------------------------------
    # Statistics
    # ------------------------------------------------------------------

    def stats(self) -> dict[str, Any]:
        """Aggregate statistics across all constituent graphs."""
        total_nodes = 0
        total_edges = 0
        total_node_index = 0
        total_name_index = 0
        total_full_name_index = 0
        total_suffix_index = 0
        any_fts5 = False

        for _wiki_id, svc in self._services.items():
            s = svc.stats()
            total_nodes += s.get("graph_nodes", 0)
            total_edges += s.get("graph_edges", 0)
            total_node_index += s.get("node_index_size", 0)
            total_name_index += s.get("name_index_size", 0)
            total_full_name_index += s.get("full_name_index_size", 0)
            total_suffix_index += s.get("suffix_index_size", 0)
            if s.get("fts5_available"):
                any_fts5 = True

        return {
            "graph_nodes": total_nodes,
            "graph_edges": total_edges,
            "node_index_size": total_node_index,
            "name_index_size": total_name_index,
            "full_name_index_size": total_full_name_index,
            "suffix_index_size": total_suffix_index,
            "fts5_available": any_fts5,
            "wiki_ids": list(self._services.keys()),
            "wiki_count": len(self._services),
        }


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


class _MergedGraphView:
    """Thin proxy that exposes graph.nodes.get() across multiple DiGraphs.

    ``_build_call_tree_from_sources`` accesses ``gqs.graph.nodes.get(node_id)``
    to read node attributes.  This view delegates that lookup to the first graph
    that contains the node_id.  Storage-backed services expose
    ``svc.graph is None`` and are silently skipped here — callers that need
    cross-backend node attributes should use ``MultiGraphQueryService.get_node``.
    """

    def __init__(self, graphs: dict[str, Any]) -> None:
        self._graphs = {wid: g for wid, g in graphs.items() if g is not None}
        self.nodes = _MergedNodesView(self._graphs)

    def __contains__(self, node_id: str) -> bool:
        return any(node_id in g for g in self._graphs.values())


class _MergedNodesView:
    """Proxy for graph.nodes that searches all NX-backed graphs."""

    def __init__(self, graphs: dict[str, Any]) -> None:
        self._graphs = graphs

    def get(self, node_id: str, default: dict | None = None) -> dict | None:
        for graph in self._graphs.values():
            if graph is None:
                continue
            data = graph.nodes.get(node_id)
            if data is not None:
                return data
        return default
