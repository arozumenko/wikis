"""
Unified symbol resolution and search service (SPEC-1 + SPEC-2).

Consolidates 3 separate resolution strategies (FTS5, in-memory dicts, graph
traversal) behind a single facade.  Replaces the scattered O(N) scans in
``research_tools.py`` and ``structure_tools.py`` with O(1) index lookups
falling back to O(log N) FTS5 search.

SPEC-2 adds ``query()`` — a JQL structured query executor that parses
``field:value`` expressions, dispatches index-friendly clauses to FTS5/SQL,
and applies graph-level post-filters (``related``, ``has_rel``, ``connections``).

Usage::

    service = GraphQueryService(graph, fts_index)
    node_id = service.resolve_symbol("AuthService")
    results = service.search("authentication handler", k=10)
    neighbors = service.get_relationships(node_id, direction="both", max_depth=2)

    # SPEC-2: structured queries
    results = service.query('type:class file:src/auth/* connections:>5')
    results = service.query('related:"AuthService" dir:outgoing has_rel:calls')
"""

import logging
from collections import defaultdict
from dataclasses import dataclass
from typing import Any

import networkx as nx

from .graph_query_builder import EDGE_TYPE_ALIASES

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Result types
# ---------------------------------------------------------------------------


@dataclass
class SymbolResult:
    """A resolved or searched symbol."""

    node_id: str
    symbol_name: str
    symbol_type: str
    layer: str = ""
    file_path: str = ""
    rel_path: str = ""
    connections: int = 0
    score: float = 0.0
    match_source: str = ""  # 'index', 'fts5', 'scan'
    docstring: str = ""


@dataclass
class RelationshipResult:
    """A graph edge connecting two symbols."""

    source_name: str
    target_name: str
    relationship_type: str
    source_type: str = ""
    target_type: str = ""
    hop_distance: int = 1


# ---------------------------------------------------------------------------
# GraphQueryService
# ---------------------------------------------------------------------------


class GraphQueryService:
    """Unified interface over graph dicts, FTS5, and NetworkX traversal.

    Designed as the **single entry point** for symbol resolution and search
    across the entire Wikis pipeline.  Consumers (structure planner,
    content expander, deep research tools) should use this instead of
    reaching into graph internals or iterating all nodes.

    Args:
        graph: NetworkX DiGraph with in-memory indexes built by
               ``graph_builder._build_graph_indexes()``.
        fts_index: Optional ``GraphTextIndex`` for FTS5-backed search.
    """

    def __init__(self, graph: nx.DiGraph, fts_index=None):
        self.graph = graph
        self.fts = fts_index

        # Import in-memory indexes from the graph (built by graph_builder)
        self._node_index: dict[tuple, str] = getattr(graph, "_node_index", {})
        self._simple_name_index: dict[tuple, str] = getattr(graph, "_simple_name_index", {})
        self._full_name_index: dict[str, str] = getattr(graph, "_full_name_index", {})
        self._name_index: dict[str, list[str]] = getattr(graph, "_name_index", defaultdict(list))
        self._suffix_index: dict[str, list[str]] = getattr(graph, "_suffix_index", defaultdict(list))

    # ------------------------------------------------------------------
    # Symbol Resolution — O(1) with FTS5 fallback
    # ------------------------------------------------------------------

    def resolve_symbol(
        self,
        symbol_name: str,
        file_path: str = "",
        language: str = "",
    ) -> str | None:
        """Resolve a symbol name to its graph ``node_id``.

        Uses the 6-strategy cascade from ``content_expander._find_graph_node``,
        adapted as a standalone method:

        1. Exact key ``(name, file, lang)`` → ``_node_index``
        2. Simple name ``(simple, file, lang)`` → ``_simple_name_index``
        3. Full name → ``_full_name_index``
        4. Suffix match → ``_suffix_index``
        5. Name index → ``_name_index``
        6. FTS5 fuzzy search (safety net)

        Returns ``node_id`` or ``None`` if not found.
        """
        name = symbol_name.strip()
        if not name:
            return None

        # Strategy 1: exact (name, file, lang) key
        if file_path and language:
            node_id = self._node_index.get((name, file_path, language))
            if node_id:
                return node_id

        # Strategy 2: simple name (last segment)
        simple = name.rsplit(".", 1)[-1].rsplit("::", 1)[-1]
        if file_path and language and simple != name:
            node_id = self._simple_name_index.get((simple, file_path, language))
            if node_id:
                return node_id

        # Strategy 3: full name
        node_id = self._full_name_index.get(name)
        if node_id:
            return node_id

        # Strategy 4: suffix index
        suffix_candidates = self._suffix_index.get(name, [])
        if suffix_candidates:
            return self._select_best(suffix_candidates, file_path, language)

        # Strategy 5: name index (may match multiple — pick best)
        name_candidates = self._name_index.get(name, [])
        if name_candidates:
            return self._select_best(name_candidates, file_path, language)

        # Strategy 6: FTS5 fuzzy search
        if self.fts and self.fts.is_open:
            try:
                docs = self.fts.search_smart(name, k=3, intent="symbol")
                for doc in docs:
                    nid = doc.metadata.get("node_id", "")
                    if nid and nid in self.graph:
                        return nid
            except Exception as exc:
                logger.debug(f"FTS5 fallback failed for {name!r}: {exc}")

        return None

    def _select_best(
        self,
        candidates: list[str],
        file_path: str = "",
        language: str = "",
    ) -> str | None:
        """Pick the best node_id from a list of candidates.

        Prefers candidates whose file_path/language match the hint,
        otherwise returns the first (highest-priority by type).
        """
        if not candidates:
            return None
        if len(candidates) == 1:
            return candidates[0]

        # Score candidates by match quality
        if file_path or language:
            for nid in candidates:
                data = self.graph.nodes.get(nid, {})
                nid_file = data.get("rel_path", "") or data.get("file_path", "")
                nid_lang = data.get("language", "")
                if file_path and file_path in nid_file:
                    return nid
                if language and language == nid_lang:
                    return nid

        # Default: first candidate (already sorted by type priority)
        return candidates[0]

    # ------------------------------------------------------------------
    # Search — FTS5 with O(N) fallback
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
        """Search for symbols by text query.

        Uses FTS5 BM25 when available, falls back to brute-force O(N) scan
        over graph nodes.

        Args:
            query: Natural-language or keyword search query.
            k: Maximum results.
            symbol_types: Only return these types.
            exclude_types: Exclude these types (e.g. DOC_SYMBOL_TYPES).
            layer: Filter by architectural layer (SPEC-4).
            path_prefix: Filter to symbols under this path.

        Returns:
            List of ``SymbolResult`` sorted by relevance.
        """
        # FTS5 path (fast — O(log N))
        if self.fts and self.fts.is_open:
            return self._search_fts5(
                query,
                k=k,
                symbol_types=symbol_types,
                exclude_types=exclude_types,
                layer=layer,
                path_prefix=path_prefix,
            )

        # Fallback: brute-force (O(N))
        return self._search_brute_force(
            query,
            k=k,
            symbol_types=symbol_types,
            exclude_types=exclude_types,
            layer=layer,
            path_prefix=path_prefix,
        )

    def _search_fts5(
        self,
        query: str,
        k: int = 20,
        symbol_types: frozenset[str] | None = None,
        exclude_types: frozenset[str] | None = None,
        layer: str | None = None,
        path_prefix: str | None = None,
    ) -> list[SymbolResult]:
        """FTS5 BM25-ranked search with post-filtering."""
        try:
            # Use search_smart for intent-aware FTS5 queries
            docs = self.fts.search_smart(query, k=k * 3, intent="general")
        except Exception as exc:
            logger.warning(f"FTS5 search failed: {exc}")
            return self._search_brute_force(
                query,
                k=k,
                symbol_types=symbol_types,
                exclude_types=exclude_types,
                layer=layer,
                path_prefix=path_prefix,
            )

        results = []
        for doc in docs:
            meta = doc.metadata
            stype = meta.get("symbol_type", "").lower()
            sym_layer = meta.get("layer", "")

            if symbol_types and stype not in symbol_types:
                continue
            if exclude_types and stype in exclude_types:
                continue
            if layer and sym_layer != layer:
                continue
            if path_prefix:
                rp = meta.get("rel_path", "")
                if not rp.startswith(path_prefix.rstrip("/") + "/") and rp != path_prefix:
                    continue

            node_id = meta.get("node_id", "")
            connections = 0
            if node_id and node_id in self.graph:
                connections = self.graph.in_degree(node_id) + self.graph.out_degree(node_id)

            results.append(
                SymbolResult(
                    node_id=node_id,
                    symbol_name=meta.get("symbol_name", ""),
                    symbol_type=stype,
                    layer=sym_layer,
                    file_path=meta.get("file_path", ""),
                    rel_path=meta.get("rel_path", ""),
                    connections=connections,
                    score=meta.get("search_score", 0.0),
                    match_source="fts5",
                )
            )

            if len(results) >= k:
                break

        return results

    def _search_brute_force(
        self,
        query: str,
        k: int = 20,
        symbol_types: frozenset[str] | None = None,
        exclude_types: frozenset[str] | None = None,
        layer: str | None = None,
        path_prefix: str | None = None,
    ) -> list[SymbolResult]:
        """O(N) fallback search when FTS5 is unavailable."""
        query_lower = query.lower()
        keywords = [w for w in query_lower.split() if len(w) >= 2]
        results = []

        for node_id, data in self.graph.nodes(data=True):
            name = data.get("symbol_name", "") or data.get("name", "")
            if not name:
                continue

            stype = (data.get("symbol_type") or "").lower()
            if symbol_types and stype not in symbol_types:
                continue
            if exclude_types and stype in exclude_types:
                continue
            if layer:
                # Layer not available in graph nodes, only in FTS5
                pass
            if path_prefix:
                rp = data.get("rel_path", "") or data.get("file_path", "")
                if not rp.startswith(path_prefix.rstrip("/") + "/") and rp != path_prefix:
                    continue

            # Score: exact name match + keyword hits
            name_lower = name.lower()
            score = 0.0
            if query_lower == name_lower:
                score = 100.0
            elif query_lower in name_lower:
                score = 50.0
            else:
                kw_hits = sum(1 for kw in keywords if kw in name_lower)
                if kw_hits == 0:
                    # Also check docstring (handle both parser formats)
                    docstring = (data.get("docstring", "") or "").lower()
                    if not docstring:
                        _sym = data.get("symbol")
                        if _sym and hasattr(_sym, "docstring") and _sym.docstring:
                            docstring = _sym.docstring.lower()
                    kw_hits = sum(1 for kw in keywords if kw in docstring) * 0.3
                if kw_hits == 0:
                    continue
                score = kw_hits * 10.0

            connections = self.graph.in_degree(node_id) + self.graph.out_degree(node_id)

            results.append(
                SymbolResult(
                    node_id=node_id,
                    symbol_name=name,
                    symbol_type=stype,
                    file_path=data.get("file_path", "") or "",
                    rel_path=data.get("rel_path", "") or "",
                    connections=connections,
                    score=score,
                    match_source="scan",
                )
            )

        results.sort(key=lambda r: (-r.score, -r.connections))
        return results[:k]

    # ------------------------------------------------------------------
    # Relationship Traversal — bounded BFS
    # ------------------------------------------------------------------

    def get_relationships(
        self,
        node_id: str,
        direction: str = "both",
        max_depth: int = 2,
        max_results: int = 50,
    ) -> list[RelationshipResult]:
        """Traverse graph relationships from a resolved node_id.

        Args:
            node_id: Starting node (use ``resolve_symbol()`` first).
            direction: 'outgoing', 'incoming', or 'both'.
            max_depth: Maximum traversal hops.
            max_results: Cap on returned edges.

        Returns:
            List of ``RelationshipResult`` sorted by hop distance.
        """
        if node_id not in self.graph:
            return []

        results: list[RelationshipResult] = []
        visited: set[str] = {node_id}
        seen_edges: set[tuple[str, str, str]] = set()  # (src, tgt, rel_type)
        frontier: list[tuple[str, int]] = [(node_id, 0)]

        while frontier and len(results) < max_results:
            current, depth = frontier.pop(0)
            if depth >= max_depth:
                continue

            edges = []  # (src, tgt, edata, hop, other_node)
            if direction in ("outgoing", "both"):
                for succ in self.graph.successors(current):
                    edge_data = self.graph.get_edge_data(current, succ)
                    if edge_data:
                        for e in edge_data.values():
                            edges.append((current, succ, e, depth + 1, succ))
            if direction in ("incoming", "both"):
                for pred in self.graph.predecessors(current):
                    edge_data = self.graph.get_edge_data(pred, current)
                    if edge_data:
                        for e in edge_data.values():
                            edges.append((pred, current, e, depth + 1, pred))

            for src, tgt, edata, hop, other_node in edges:
                if len(results) >= max_results:
                    break

                rel_type = str(edata.get("relationship_type", "") or edata.get("type", ""))
                edge_key = (src, tgt, rel_type)
                if edge_key in seen_edges:
                    # Still advance the frontier through this node
                    if other_node not in visited:
                        visited.add(other_node)
                        frontier.append((other_node, depth + 1))
                    continue
                seen_edges.add(edge_key)

                src_data = self.graph.nodes.get(src, {})
                tgt_data = self.graph.nodes.get(tgt, {})

                results.append(
                    RelationshipResult(
                        source_name=src_data.get("symbol_name", "") or src_data.get("name", src),
                        target_name=tgt_data.get("symbol_name", "") or tgt_data.get("name", tgt),
                        relationship_type=rel_type,
                        source_type=(src_data.get("symbol_type") or "").lower(),
                        target_type=(tgt_data.get("symbol_type") or "").lower(),
                        hop_distance=hop,
                    )
                )

                if other_node not in visited:
                    visited.add(other_node)
                    frontier.append((other_node, depth + 1))

        results.sort(key=lambda r: r.hop_distance)
        return results

    # ------------------------------------------------------------------
    # Convenience: resolve + traverse in one call
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

        Combines ``resolve_symbol`` + ``get_relationships`` in one call.
        Returns ``(node_id, relationships)`` where ``node_id`` may be None.
        """
        node_id = self.resolve_symbol(symbol_name, file_path=file_path, language=language)
        if not node_id:
            return None, []
        return node_id, self.get_relationships(
            node_id,
            direction=direction,
            max_depth=max_depth,
            max_results=max_results,
        )

    # ------------------------------------------------------------------
    # SPEC-2: JQL Structured Query Execution
    # ------------------------------------------------------------------

    def query(self, expression: str) -> list[SymbolResult]:
        """Execute a JQL structured query expression.

        Parses the expression into index-friendly and post-filter clauses,
        runs FTS5/SQL for the index portion, then applies graph-level
        post-filters (``related``, ``has_rel``, ``connections``).

        Args:
            expression: JQL query, e.g. ``'type:class file:src/auth/*'``.
                See ``jql_parser`` module for full syntax.

        Returns:
            List of ``SymbolResult`` sorted by relevance/connections.

        Examples::

            service.query('type:class file:src/auth/*')
            service.query('related:"AuthService" dir:outgoing')
            service.query('type:class connections:>10 limit:5')
            service.query('layer:core_type text:authentication')
        """
        from .jql_parser import parse_jql

        jql = parse_jql(expression)
        if jql.is_empty:
            return []

        # Phase 1: Index-backed retrieval (FTS5 + SQL filters)
        results = self._execute_index_clauses(jql)

        # Phase 2: Graph-level post-filters
        results = self._apply_post_filters(results, jql)

        # Phase 3: Apply limit
        return results[: jql.limit]

    def _execute_index_clauses(self, jql) -> list[SymbolResult]:
        """Execute index-friendly clauses (type, layer, file, name, text)."""

        # Extract filter parameters from index clauses
        symbol_types = frozenset(jql.type_values) if jql.type_values else None
        layer = jql.layer_value
        path_prefix = jql.file_value
        name_value = jql.name_value
        text_value = jql.text_value

        # Determine fetch size — generous to allow post-filtering headroom
        fetch_k = jql.limit * 5

        # --- Route 1: Name-based resolution (most specific) ---
        if name_value and not text_value:
            return self._jql_name_search(
                name_value,
                fetch_k,
                symbol_types=symbol_types,
                layer=layer,
                path_prefix=path_prefix,
            )

        # --- Route 2: FTS5 text search with index filters ---
        if text_value:
            results = self._search_fts5(
                text_value,
                k=fetch_k,
                symbol_types=symbol_types,
                layer=layer,
                path_prefix=path_prefix,
            )
            # Also apply name filter as post-filter on FTS results
            if name_value:
                nv = name_value.lower()
                results = [r for r in results if nv in r.symbol_name.lower()]
            return results

        # --- Route 3: Layer/type/file only (no text) ---
        if layer and self.fts and self.fts.is_open:
            return self._jql_layer_search(
                layer,
                fetch_k,
                symbol_types=symbol_types,
                path_prefix=path_prefix,
            )

        if path_prefix and self.fts and self.fts.is_open:
            return self._jql_path_search(
                path_prefix,
                fetch_k,
                symbol_types=symbol_types,
            )

        if symbol_types and self.fts and self.fts.is_open:
            return self._jql_type_search(
                symbol_types,
                fetch_k,
            )

        # Non-FTS fallback paths for layer/path/type filters
        if path_prefix or layer or symbol_types:
            return self._jql_full_scan(
                fetch_k,
                symbol_types=symbol_types,
                layer=layer,
                path_prefix=path_prefix,
            )

        # --- Route 4: Fallback — full scan with all available filters ---
        return self._jql_full_scan(
            fetch_k,
            symbol_types=symbol_types,
            layer=layer,
            path_prefix=path_prefix,
        )

    def _jql_name_search(
        self,
        name: str,
        k: int,
        symbol_types=None,
        layer=None,
        path_prefix=None,
    ) -> list[SymbolResult]:
        """Search by symbol name (exact, prefix, or glob)."""
        has_glob = "*" in name or "?" in name

        # Try FTS5 name search first
        if self.fts and self.fts.is_open:
            try:
                exact = not has_glob
                type_list = list(symbol_types) if symbol_types else None
                rows = self.fts.search_by_name(
                    name.replace("*", "").replace("?", ""),
                    exact=exact,
                    k=k * 2,
                    symbol_types=type_list,
                )
                results = []
                for doc in rows:
                    meta = doc.metadata if hasattr(doc, "metadata") else doc
                    stype = (meta.get("symbol_type", "") or "").lower()
                    sym_layer = meta.get("layer", "")
                    rp = meta.get("rel_path", "")

                    if layer and sym_layer != layer:
                        continue
                    if path_prefix and not self._matches_path(rp, path_prefix):
                        continue
                    if has_glob:
                        from fnmatch import fnmatch

                        sname = meta.get("symbol_name", "")
                        if not fnmatch(sname.lower(), name.lower()):
                            continue

                    node_id = meta.get("node_id", "")
                    connections = 0
                    if node_id and node_id in self.graph:
                        connections = self.graph.in_degree(node_id) + self.graph.out_degree(node_id)

                    results.append(
                        SymbolResult(
                            node_id=node_id,
                            symbol_name=meta.get("symbol_name", ""),
                            symbol_type=stype,
                            layer=sym_layer,
                            file_path=meta.get("file_path", ""),
                            rel_path=rp,
                            connections=connections,
                            score=meta.get("search_score", 0.0),
                            match_source="fts5",
                        )
                    )
                    if len(results) >= k:
                        break
                return results
            except Exception as exc:
                logger.debug(f"FTS5 name search failed: {exc}")

        # Fallback: in-memory scan
        return self._jql_full_scan(
            k,
            symbol_types=symbol_types,
            name_filter=name,
            layer=layer,
            path_prefix=path_prefix,
        )

    def _jql_layer_search(
        self,
        layer: str,
        k: int,
        symbol_types=None,
        path_prefix=None,
    ) -> list[SymbolResult]:
        """Search by architectural layer via FTS5."""
        try:
            type_list = list(symbol_types) if symbol_types else None
            rows = self.fts.search_by_layer(layer, symbol_types=type_list, k=k * 2)
            results = []
            for row in rows:
                meta = row if isinstance(row, dict) else getattr(row, "metadata", row)
                rp = meta.get("rel_path", "")

                if path_prefix and not self._matches_path(rp, path_prefix):
                    continue

                node_id = meta.get("node_id", "")
                connections = 0
                if node_id and node_id in self.graph:
                    connections = self.graph.in_degree(node_id) + self.graph.out_degree(node_id)

                results.append(
                    SymbolResult(
                        node_id=node_id,
                        symbol_name=meta.get("symbol_name", ""),
                        symbol_type=(meta.get("symbol_type", "") or "").lower(),
                        layer=meta.get("layer", ""),
                        file_path=meta.get("file_path", ""),
                        rel_path=rp,
                        connections=connections,
                        score=0.0,
                        match_source="fts5",
                    )
                )
                if len(results) >= k:
                    break
            return results
        except Exception as exc:
            logger.debug(f"FTS5 layer search failed: {exc}")
            return self._jql_full_scan(
                k,
                symbol_types=symbol_types,
                layer=layer,
                path_prefix=path_prefix,
            )

    def _jql_path_search(
        self,
        path_prefix: str,
        k: int,
        symbol_types=None,
    ) -> list[SymbolResult]:
        """Search by path prefix via FTS5."""
        try:
            type_list = list(symbol_types) if symbol_types else None
            excluded = None
            rows = self.fts.search_by_path_prefix(
                path_prefix,
                symbol_types=type_list,
                exclude_types=excluded,
                k=k * 2,
            )
            results = []
            for row in rows:
                meta = row if isinstance(row, dict) else getattr(row, "metadata", row)
                node_id = meta.get("node_id", "")
                connections = 0
                if node_id and node_id in self.graph:
                    connections = self.graph.in_degree(node_id) + self.graph.out_degree(node_id)

                results.append(
                    SymbolResult(
                        node_id=node_id,
                        symbol_name=meta.get("symbol_name", ""),
                        symbol_type=(meta.get("symbol_type", "") or "").lower(),
                        layer=meta.get("layer", ""),
                        file_path=meta.get("file_path", ""),
                        rel_path=meta.get("rel_path", ""),
                        connections=connections,
                        score=0.0,
                        match_source="fts5",
                    )
                )
                if len(results) >= k:
                    break
            return results
        except Exception as exc:
            logger.debug(f"FTS5 path search failed: {exc}")
            return self._jql_full_scan(k, symbol_types=symbol_types, path_prefix=path_prefix)

    def _jql_type_search(
        self,
        symbol_types: frozenset[str],
        k: int,
    ) -> list[SymbolResult]:
        """Search by symbol type(s) via FTS5."""
        results = []
        try:
            for stype in symbol_types:
                rows = self.fts.search_by_type(stype, k=k)
                for row in rows:
                    meta = row if isinstance(row, dict) else getattr(row, "metadata", row)
                    node_id = meta.get("node_id", "")
                    connections = 0
                    if node_id and node_id in self.graph:
                        connections = self.graph.in_degree(node_id) + self.graph.out_degree(node_id)

                    results.append(
                        SymbolResult(
                            node_id=node_id,
                            symbol_name=meta.get("symbol_name", ""),
                            symbol_type=(meta.get("symbol_type", "") or "").lower(),
                            layer=meta.get("layer", ""),
                            file_path=meta.get("file_path", ""),
                            rel_path=meta.get("rel_path", ""),
                            connections=connections,
                            score=0.0,
                            match_source="fts5",
                        )
                    )
                    if len(results) >= k:
                        break
                if len(results) >= k:
                    break
            return results
        except Exception as exc:
            logger.debug(f"FTS5 type search failed: {exc}")
            return self._jql_full_scan(k, symbol_types=symbol_types)

    def _jql_full_scan(
        self,
        k: int,
        symbol_types=None,
        layer=None,
        path_prefix=None,
        name_filter=None,
    ) -> list[SymbolResult]:
        """Full graph node scan fallback for JQL queries."""
        results = []
        for node_id, data in self.graph.nodes(data=True):
            name = data.get("symbol_name", "") or data.get("name", "")
            if not name:
                continue

            stype = (data.get("symbol_type") or "").lower()
            if symbol_types and stype not in symbol_types:
                continue

            rp = data.get("rel_path", "") or data.get("file_path", "")
            if path_prefix and not self._matches_path(rp, path_prefix):
                continue

            # Layer filter (approximate — layer may not be in graph nodes)
            if layer:
                node_layer = (data.get("layer", "") or "").lower()
                if node_layer and node_layer != layer:
                    continue

            if name_filter:
                from fnmatch import fnmatch

                if "*" in name_filter or "?" in name_filter:
                    if not fnmatch(name.lower(), name_filter.lower()):
                        continue
                elif name_filter.lower() not in name.lower():
                    continue

            connections = self.graph.in_degree(node_id) + self.graph.out_degree(node_id)

            results.append(
                SymbolResult(
                    node_id=node_id,
                    symbol_name=name,
                    symbol_type=stype,
                    file_path=data.get("file_path", "") or "",
                    rel_path=rp,
                    connections=connections,
                    score=float(connections),  # Use connectivity as proxy score
                    match_source="scan",
                )
            )

        results.sort(key=lambda r: (-r.score, -r.connections))
        return results[:k]

    def _apply_post_filters(
        self,
        results: list[SymbolResult],
        jql,
    ) -> list[SymbolResult]:
        """Apply graph-level post-filters (related, has_rel, connections)."""
        # --- Filter: related (BFS from named symbol) ---
        related_name = jql.related_value
        if related_name:
            results = self._filter_related(
                results,
                related_name,
                direction=jql.direction_value,
                edge_types=jql.has_rel_values,
            )

        # --- Filter: has_rel (edge type existence) ---
        elif jql.has_rel_values:
            results = self._filter_has_relationship(results, jql.has_rel_values)

        # --- Filter: connections (degree comparison) ---
        conn_clause = jql.connections_clause
        if conn_clause:
            results = [r for r in results if conn_clause.matches_numeric(r.connections)]

        return results

    def _filter_related(
        self,
        results: list[SymbolResult],
        related_name: str,
        direction: str = "both",
        edge_types: list[str] | None = None,
        max_depth: int = 3,
    ) -> list[SymbolResult]:
        """Filter results to only symbols reachable from ``related_name``."""
        # Resolve the anchor symbol
        anchor_id = self.resolve_symbol(related_name)
        if not anchor_id:
            logger.debug(f"Could not resolve related symbol: {related_name!r}")
            return []

        # BFS to collect reachable node IDs
        reachable: set[str] = set()
        frontier: list[tuple[str, int]] = [(anchor_id, 0)]
        visited: set[str] = {anchor_id}

        while frontier:
            current, depth = frontier.pop(0)
            if depth >= max_depth:
                continue

            edges = []
            if direction in ("outgoing", "both"):
                for succ in self.graph.successors(current):
                    for edict in self._iter_edge_dicts(current, succ):
                        edges.append((succ, edict))
            if direction in ("incoming", "both"):
                for pred in self.graph.predecessors(current):
                    for edict in self._iter_edge_dicts(pred, current):
                        edges.append((pred, edict))

            for next_node, edict in edges:
                if edge_types:
                    rel_type = self._edge_rel_type(edict)
                    if rel_type not in edge_types:
                        continue
                reachable.add(next_node)
                if next_node not in visited:
                    visited.add(next_node)
                    frontier.append((next_node, depth + 1))

        # Filter results to reachable set
        if not results:
            # If no index clauses produced results, materialize from reachable
            filtered = []
            for nid in reachable:
                if nid not in self.graph:
                    continue
                data = self.graph.nodes[nid]
                name = data.get("symbol_name", "") or data.get("name", "")
                if not name:
                    continue
                connections = self.graph.in_degree(nid) + self.graph.out_degree(nid)
                filtered.append(
                    SymbolResult(
                        node_id=nid,
                        symbol_name=name,
                        symbol_type=(data.get("symbol_type") or "").lower(),
                        layer=data.get("layer", ""),
                        file_path=data.get("file_path", "") or "",
                        rel_path=data.get("rel_path", "") or "",
                        connections=connections,
                        score=float(connections),
                        match_source="graph",
                    )
                )
            filtered.sort(key=lambda r: (-r.score, -r.connections))
            return filtered
        else:
            return [r for r in results if r.node_id in reachable]

    @staticmethod
    def _normalise_edge_types(edge_types: list[str]) -> frozenset:
        """Expand user-supplied edge type strings into a set of all matching forms.

        Adds both the canonical ``RelationshipType.value`` form and any known
        legacy aliases so that ``has_rel:inherits`` matches edges stored as
        either ``"inherits"`` or ``"inheritance"``.
        """
        expanded: set = set()
        for et in edge_types:
            et_lower = et.lower()
            expanded.add(et_lower)
            # If user supplied an alias, add the canonical form
            if et_lower in EDGE_TYPE_ALIASES:
                expanded.add(EDGE_TYPE_ALIASES[et_lower])
            # Also add any alias strings that map TO this canonical form
            for alias, canonical in EDGE_TYPE_ALIASES.items():
                if canonical == et_lower:
                    expanded.add(alias)
        return frozenset(expanded)

    def _filter_has_relationship(
        self,
        results: list[SymbolResult],
        edge_types: list[str],
    ) -> list[SymbolResult]:
        """Filter to symbols with at least one edge of the specified type(s).

        Edge types are normalised via ``EDGE_TYPE_ALIASES`` so that both NL
        forms (``inherits``) and canonical enum values (``inheritance``)
        match the same graph edges.
        """
        type_set = self._normalise_edge_types(edge_types)
        filtered = []
        for r in results:
            if not r.node_id or r.node_id not in self.graph:
                continue
            found = False
            # Check outgoing edges
            for succ in self.graph.successors(r.node_id):
                for edict in self._iter_edge_dicts(r.node_id, succ):
                    if self._edge_rel_type(edict) in type_set:
                        found = True
                        break
                if found:
                    break
            # Check incoming edges
            if not found:
                for pred in self.graph.predecessors(r.node_id):
                    for edict in self._iter_edge_dicts(pred, r.node_id):
                        if self._edge_rel_type(edict) in type_set:
                            found = True
                            break
                    if found:
                        break
            if found:
                filtered.append(r)
        return filtered

    def _iter_edge_dicts(self, src: str, tgt: str) -> list[dict]:
        """Yield edge attribute dicts, handling both DiGraph and MultiDiGraph.

        For DiGraph: ``get_edge_data(u, v)`` returns a single attribute dict.
        For MultiDiGraph: it returns ``{key: attr_dict, ...}`` — one per parallel edge.
        """
        edata = self.graph.get_edge_data(src, tgt)
        if edata is None:
            return []
        if isinstance(self.graph, nx.MultiDiGraph):
            return list(edata.values())
        # DiGraph: edata IS the attribute dict
        return [edata]

    @staticmethod
    def _edge_rel_type(edict: dict) -> str:
        """Extract the relationship type string from an edge attribute dict."""
        return str(edict.get("relationship_type", "") or edict.get("type", "")).lower()

    @staticmethod
    def _matches_path(path: str, pattern: str) -> bool:
        """Check if path matches a glob or prefix pattern."""
        if not path:
            return False
        if "*" in pattern or "?" in pattern:
            from fnmatch import fnmatch

            return fnmatch(path, pattern)
        # Prefix match
        prefix = pattern.rstrip("/")
        return path.startswith(prefix + "/") or path == prefix

    # ------------------------------------------------------------------
    # Statistics
    # ------------------------------------------------------------------

    def stats(self) -> dict[str, Any]:
        """Return index statistics for debugging."""
        return {
            "graph_nodes": self.graph.number_of_nodes(),
            "graph_edges": self.graph.number_of_edges(),
            "node_index_size": len(self._node_index),
            "name_index_size": len(self._name_index),
            "full_name_index_size": len(self._full_name_index),
            "suffix_index_size": len(self._suffix_index),
            "fts5_available": bool(self.fts and self.fts.is_open),
        }
