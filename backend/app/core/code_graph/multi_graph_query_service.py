"""MultiGraphQueryService — fans out graph queries across N per-wiki graphs.

Exposes the same public interface as ``GraphQueryService`` so callers (codemap
pipeline, research tools) require minimal changes.
"""

from __future__ import annotations

import logging
from typing import Any

from .graph_query_service import GraphQueryService, RelationshipResult, SymbolResult

logger = logging.getLogger(__name__)


class MultiGraphQueryService:
    """Wraps a dict of per-wiki ``GraphQueryService`` instances.

    All public methods mirror ``GraphQueryService``.  When a ``wiki_id`` hint
    is provided the call is dispatched to only that graph; otherwise the query
    fans out to all graphs and results are merged.

    Args:
        graphs: Mapping of wiki_id → GraphQueryService.
    """

    def __init__(self, graphs: dict[str, GraphQueryService]) -> None:
        self._graphs = graphs  # wiki_id → GraphQueryService

    # ------------------------------------------------------------------
    # Symbol resolution
    # ------------------------------------------------------------------

    def resolve_symbol(
        self,
        symbol_name: str,
        file_path: str = "",
        language: str = "",
        wiki_id: str | None = None,
    ) -> str | None:
        """Resolve a symbol name to a node_id.

        If *wiki_id* is provided, only that graph is queried.
        Otherwise, all graphs are tried in insertion order; the first match wins.
        """
        if wiki_id and wiki_id in self._graphs:
            return self._graphs[wiki_id].resolve_symbol(symbol_name, file_path, language)

        for _wid, gqs in self._graphs.items():
            result = gqs.resolve_symbol(symbol_name, file_path, language)
            if result is not None:
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
        wiki_id: str | None = None,
    ) -> list[SymbolResult]:
        """Search symbols across all graphs (or a single one when wiki_id is given).

        Results from all wikis are merged and sorted by score descending,
        then trimmed to *k* entries.
        """
        if wiki_id and wiki_id in self._graphs:
            return self._graphs[wiki_id].search(
                query, k=k,
                symbol_types=symbol_types,
                exclude_types=exclude_types,
                layer=layer,
                path_prefix=path_prefix,
            )

        all_results: list[SymbolResult] = []
        per_wiki_k = k  # ask each graph for k results; merge and re-sort
        for wid, gqs in self._graphs.items():
            try:
                results = gqs.search(
                    query, k=per_wiki_k,
                    symbol_types=symbol_types,
                    exclude_types=exclude_types,
                    layer=layer,
                    path_prefix=path_prefix,
                )
                all_results.extend(results)
            except Exception:
                logger.warning("Search failed for wiki %s", wid, exc_info=True)

        # Sort by score descending, deduplicate by node_id
        seen: set[str] = set()
        merged: list[SymbolResult] = []
        for r in sorted(all_results, key=lambda x: x.score, reverse=True):
            if r.node_id not in seen:
                seen.add(r.node_id)
                merged.append(r)
            if len(merged) >= k:
                break
        return merged

    # ------------------------------------------------------------------
    # Relationship traversal
    # ------------------------------------------------------------------

    def get_relationships(
        self,
        node_id: str,
        direction: str = "both",
        max_depth: int = 1,
        max_results: int = 20,
        wiki_id: str | None = None,
    ) -> list[RelationshipResult]:
        """Get relationships for a node.

        If *wiki_id* is given, only that graph is used.
        Otherwise, the first graph that contains *node_id* is used.
        """
        if wiki_id and wiki_id in self._graphs:
            return self._graphs[wiki_id].get_relationships(
                node_id, direction=direction, max_depth=max_depth, max_results=max_results,
            )

        for _wid, gqs in self._graphs.items():
            if node_id in gqs.graph:
                return gqs.get_relationships(
                    node_id, direction=direction, max_depth=max_depth, max_results=max_results,
                )
        return []

    # ------------------------------------------------------------------
    # Graph property access (primary graph delegate)
    # ------------------------------------------------------------------

    @property
    def graph(self) -> Any:
        """Return the primary (first) wiki's graph, or an empty DiGraph."""
        if self._graphs:
            return next(iter(self._graphs.values())).graph
        import networkx as nx
        return nx.DiGraph()

    def _find_graph_for_node(self, node_id: str) -> GraphQueryService | None:
        """Return the GraphQueryService whose graph contains *node_id*."""
        for gqs in self._graphs.values():
            if node_id in gqs.graph:
                return gqs
        return None

    def wiki_ids(self) -> list[str]:
        """Return the list of wiki_ids in this multi-graph."""
        return list(self._graphs.keys())
