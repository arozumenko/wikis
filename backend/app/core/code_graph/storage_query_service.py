"""
Storage-native symbol resolution / search / traversal service.

Drop-in replacement for :class:`GraphQueryService` that operates entirely
over a :class:`WikiStorageProtocol` + :class:`StorageTextIndex` instead of
an in-memory NetworkX graph.  Used by the Ask and Deep-Research engines
when a cached ``.wiki.db`` is loaded (no graph rehydration).

The public surface — ``resolve_symbol``, ``search``, ``get_relationships``,
``resolve_and_traverse``, ``query`` — matches ``GraphQueryService`` exactly
so ``research_tools.create_codebase_tools`` can accept either implementation
through duck typing.  Result dataclasses (``SymbolResult``,
``RelationshipResult``) are shared, imported from :mod:`graph_query_service`.

Resolution cascade (subset of the NX 6-strategy cascade):

    1. Exact ``(name, file, lang)`` -> ``storage.find_node_exact``
    2. Simple name + file + lang  -> ``storage.find_node_exact``
    3. Name lookup + type-priority -> ``storage.find_nodes_by_name`` + _select_best
    4. FTS5 fuzzy search           -> ``text_index.search_smart(intent='symbol')``

Strategies 3 (full_name) and 4 (suffix) of the NX cascade are skipped —
``full_name`` is not persisted in ``repo_nodes`` and suffix matching is
dominated by (2)+(3) for every real-world caller.
"""

from __future__ import annotations

import logging
from collections import deque
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Iterable

from .graph_query_builder import EDGE_TYPE_ALIASES
from .graph_query_service import RelationshipResult, SymbolResult
from .symbol_priority import symbol_priority

if TYPE_CHECKING:
    from ..storage.protocol import WikiStorageProtocol
    from ..storage.text_index import StorageTextIndex

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Small local cache for per-call node lookups
# ---------------------------------------------------------------------------


@dataclass
class _NodeCache:
    """Tiny in-session cache to batch node lookups inside a single traversal."""

    storage: "WikiStorageProtocol"
    _cache: dict[str, dict[str, Any] | None] = None  # type: ignore[assignment]

    def __post_init__(self) -> None:
        self._cache = {}

    def get(self, node_id: str) -> dict[str, Any] | None:
        if node_id in self._cache:
            return self._cache[node_id]
        row = self.storage.get_node(node_id)
        self._cache[node_id] = row
        return row

    def prime(self, node_ids: Iterable[str]) -> None:
        missing = [nid for nid in node_ids if nid and nid not in self._cache]
        if not missing:
            return
        for row in self.storage.get_nodes_by_ids(missing):
            self._cache[row.get("node_id", "")] = row
        # Mark explicit misses so repeated get() doesn't re-query.
        for nid in missing:
            self._cache.setdefault(nid, None)


# ---------------------------------------------------------------------------
# StorageQueryService
# ---------------------------------------------------------------------------


class StorageQueryService:
    """Backend-agnostic query service over WikiStorageProtocol + FTS.

    Parameters
    ----------
    storage : WikiStorageProtocol
        Opened storage (SQLite or Postgres).
    text_index : StorageTextIndex
        FTS adapter backing the same storage instance.
    """

    def __init__(
        self,
        storage: "WikiStorageProtocol",
        text_index: "StorageTextIndex | None" = None,
    ) -> None:
        self.storage = storage
        self.fts = text_index
        # Kept for attribute parity with ``GraphQueryService`` — some callers
        # read ``service.graph`` defensively.  We expose ``None`` here.
        self.graph = None

    # ------------------------------------------------------------------
    # Node accessor — parity with ``GraphQueryService.graph.nodes.get``
    # ------------------------------------------------------------------

    def get_node(self, node_id: str) -> dict[str, Any] | None:
        """Return the raw node attribute dict for ``node_id``.

        Mirrors ``GraphQueryService``'s NetworkX-backed
        ``service.graph.nodes.get(node_id)`` so callers (e.g. the Code Map
        ``_build_call_tree_from_sources`` helper) can fetch ``rel_path``,
        ``symbol_name``, ``symbol_type``, ``line_start`` etc. from a
        storage-backed wiki without relying on a rehydrated NX graph.
        """
        if not node_id:
            return None
        try:
            return self.storage.get_node(node_id)
        except Exception:  # pragma: no cover — defensive against backend errors
            return None

    # ------------------------------------------------------------------
    # Symbol resolution
    # ------------------------------------------------------------------

    def resolve_symbol(
        self,
        symbol_name: str,
        file_path: str = "",
        language: str = "",
    ) -> str | None:
        name = (symbol_name or "").strip()
        if not name:
            return None

        # Strategy 1: exact (name, file, lang) triple
        if file_path and language:
            row = self.storage.find_node_exact(name, file_path, language)
            if row:
                return row.get("node_id") or None

        # Strategy 2: simple name + file + lang
        simple = name.rsplit(".", 1)[-1].rsplit("::", 1)[-1]
        if simple != name and file_path and language:
            row = self.storage.find_node_exact(simple, file_path, language)
            if row:
                return row.get("node_id") or None

        # Strategy 3: name lookup with type-priority + file/lang scoring
        rows = self.storage.find_nodes_by_name(
            name,
            architectural_only=False,
            limit=15,
        )
        best = self._select_best(rows, file_path=file_path, language=language)
        if best:
            return best

        # Simple-name fallback (no file/lang hint) — matches NX strategy 5.
        if simple != name:
            rows = self.storage.find_nodes_by_name(
                simple,
                architectural_only=False,
                limit=15,
            )
            best = self._select_best(rows, file_path=file_path, language=language)
            if best:
                return best

        # Strategy 4: FTS fuzzy search
        if self.fts and self.fts.is_open:
            try:
                docs = self.fts.search_smart(name, k=3, intent="symbol")
                for doc in docs:
                    nid = doc.metadata.get("node_id", "")
                    if nid:
                        return nid
            except Exception as exc:
                logger.debug("FTS fallback failed for %r: %s", name, exc)

        return None

    @staticmethod
    def _select_best(
        rows: list[dict[str, Any]],
        file_path: str = "",
        language: str = "",
    ) -> str | None:
        if not rows:
            return None
        if len(rows) == 1:
            return rows[0].get("node_id") or None

        # Strong preference: exact file + language match.
        if file_path or language:
            for row in rows:
                rp = row.get("rel_path") or ""
                lang = (row.get("language") or "").lower()
                if file_path and file_path in rp:
                    return row.get("node_id") or None
                if language and language.lower() == lang:
                    return row.get("node_id") or None

        # Fall back to highest architectural priority, then longest span.
        def _key(row: dict[str, Any]) -> tuple[int, int]:
            end = row.get("end_line") or 0
            start = row.get("start_line") or 0
            try:
                span = int(end) - int(start)
            except (TypeError, ValueError):
                span = 0
            return (symbol_priority(row.get("symbol_type")), span)

        return max(rows, key=_key).get("node_id") or None

    # ------------------------------------------------------------------
    # Search (natural-language / keyword)
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
        if not query or not query.strip():
            return []

        if self.fts and self.fts.is_open:
            try:
                docs = self.fts.search_smart(
                    query,
                    k=k * 3,
                    intent="general",
                    symbol_types=symbol_types,
                    exclude_types=exclude_types,
                )
            except Exception as exc:
                logger.warning("FTS search failed: %s", exc)
                docs = []
        else:
            docs = []

        results: list[SymbolResult] = []
        for doc in docs:
            meta = doc.metadata
            stype = (meta.get("symbol_type", "") or "").lower()
            sym_layer = meta.get("layer", "")

            if layer and sym_layer != layer:
                continue
            if path_prefix and not self._matches_path(meta.get("rel_path", ""), path_prefix):
                continue

            results.append(
                SymbolResult(
                    node_id=meta.get("node_id", ""),
                    symbol_name=meta.get("symbol_name", ""),
                    symbol_type=stype,
                    layer=sym_layer,
                    file_path=meta.get("file_path", ""),
                    rel_path=meta.get("rel_path", ""),
                    connections=0,  # computed lazily only when needed
                    score=meta.get("search_score", 0.0) or 0.0,
                    match_source="fts5",
                    docstring=meta.get("docstring", "") or "",
                )
            )
            if len(results) >= k:
                break
        return results

    # ------------------------------------------------------------------
    # Relationship traversal (BFS via edge tables)
    # ------------------------------------------------------------------

    def get_relationships(
        self,
        node_id: str,
        direction: str = "both",
        max_depth: int = 2,
        max_results: int = 50,
    ) -> list[RelationshipResult]:
        if not node_id:
            return []
        cache = _NodeCache(self.storage)
        if cache.get(node_id) is None:
            return []

        results: list[RelationshipResult] = []
        visited: set[str] = {node_id}
        seen_edges: set[tuple[str, str, str]] = set()
        frontier: deque[tuple[str, int]] = deque([(node_id, 0)])

        while frontier and len(results) < max_results:
            current, depth = frontier.popleft()
            if depth >= max_depth:
                continue

            edges: list[tuple[str, str, str, str]] = []  # (src, tgt, rel_type, other)
            if direction in ("outgoing", "both"):
                for e in self.storage.get_edge_targets(current):
                    tgt = e.get("target_id") or ""
                    rel = str(e.get("rel_type", "") or "").lower()
                    if tgt:
                        edges.append((current, tgt, rel, tgt))
            if direction in ("incoming", "both"):
                for e in self.storage.get_edge_sources(current):
                    src = e.get("source_id") or ""
                    rel = str(e.get("rel_type", "") or "").lower()
                    if src:
                        edges.append((src, current, rel, src))

            # Batch-prime node metadata for every edge endpoint this hop.
            cache.prime({other for *_r, other in edges} | {current})

            for src, tgt, rel_type, other in edges:
                if len(results) >= max_results:
                    break

                edge_key = (src, tgt, rel_type)
                if edge_key in seen_edges:
                    if other not in visited:
                        visited.add(other)
                        frontier.append((other, depth + 1))
                    continue
                seen_edges.add(edge_key)

                src_row = cache.get(src) or {}
                tgt_row = cache.get(tgt) or {}

                results.append(
                    RelationshipResult(
                        source_name=src_row.get("symbol_name") or src,
                        target_name=tgt_row.get("symbol_name") or tgt,
                        relationship_type=rel_type,
                        source_type=(src_row.get("symbol_type") or "").lower(),
                        target_type=(tgt_row.get("symbol_type") or "").lower(),
                        hop_distance=depth + 1,
                    )
                )

                if other not in visited:
                    visited.add(other)
                    frontier.append((other, depth + 1))

        results.sort(key=lambda r: r.hop_distance)
        return results

    def resolve_and_traverse(
        self,
        symbol_name: str,
        direction: str = "both",
        max_depth: int = 2,
        max_results: int = 50,
        file_path: str = "",
        language: str = "",
    ) -> tuple[str | None, list[RelationshipResult]]:
        nid = self.resolve_symbol(symbol_name, file_path=file_path, language=language)
        if not nid:
            return None, []
        return nid, self.get_relationships(
            nid,
            direction=direction,
            max_depth=max_depth,
            max_results=max_results,
        )

    # ------------------------------------------------------------------
    # JQL structured queries
    # ------------------------------------------------------------------

    def query(self, expression: str) -> list[SymbolResult]:
        from .jql_parser import parse_jql

        jql = parse_jql(expression)
        if jql.is_empty:
            return []

        results = self._execute_index_clauses(jql)
        results = self._apply_post_filters(results, jql)
        return results[: jql.limit]

    def _execute_index_clauses(self, jql) -> list[SymbolResult]:
        symbol_types = frozenset(jql.type_values) if jql.type_values else None
        layer = jql.layer_value
        path_prefix = jql.file_value
        name_value = jql.name_value
        text_value = jql.text_value
        fetch_k = max(jql.limit * 5, 20)

        if name_value and not text_value:
            return self._jql_name_search(
                name_value,
                fetch_k,
                symbol_types=symbol_types,
                layer=layer,
                path_prefix=path_prefix,
            )

        if text_value:
            results = self.search(
                text_value,
                k=fetch_k,
                symbol_types=symbol_types,
                layer=layer,
                path_prefix=path_prefix,
            )
            if name_value:
                nv = name_value.lower()
                results = [r for r in results if nv in r.symbol_name.lower()]
            return results

        if layer and self.fts and self.fts.is_open:
            return self._jql_layer_search(layer, fetch_k, symbol_types=symbol_types, path_prefix=path_prefix)

        if path_prefix:
            return self._jql_path_search(path_prefix, fetch_k, symbol_types=symbol_types)

        if symbol_types and self.fts and self.fts.is_open:
            return self._jql_type_search(symbol_types, fetch_k)

        return self._jql_full_scan(
            fetch_k,
            symbol_types=symbol_types,
            layer=layer,
            path_prefix=path_prefix,
        )

    # ---- JQL route helpers -----------------------------------------------

    def _jql_name_search(
        self,
        name: str,
        k: int,
        symbol_types: frozenset[str] | None = None,
        layer: str | None = None,
        path_prefix: str | None = None,
    ) -> list[SymbolResult]:
        has_glob = "*" in name or "?" in name

        if self.fts and self.fts.is_open:
            try:
                type_list = frozenset(symbol_types) if symbol_types else None
                docs = self.fts.search_by_name(
                    name.replace("*", "").replace("?", ""),
                    exact=not has_glob,
                    k=k * 2,
                    symbol_types=type_list,
                )
                return [
                    r for r in (self._doc_to_result(d) for d in docs)
                    if self._jql_post_match(
                        r,
                        layer=layer,
                        path_prefix=path_prefix,
                        name_filter=name if has_glob else None,
                    )
                ][:k]
            except Exception as exc:
                logger.debug("FTS name search failed: %s", exc)

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
        symbol_types: frozenset[str] | None = None,
        path_prefix: str | None = None,
    ) -> list[SymbolResult]:
        try:
            docs = self.fts.search_by_layer(layer, symbol_types=symbol_types, k=k * 2)
        except Exception as exc:
            logger.debug("FTS layer search failed: %s", exc)
            return self._jql_full_scan(
                k, symbol_types=symbol_types, layer=layer, path_prefix=path_prefix
            )
        return [
            r for r in (self._doc_to_result(d) for d in docs)
            if self._jql_post_match(r, path_prefix=path_prefix)
        ][:k]

    def _jql_path_search(
        self,
        path_prefix: str,
        k: int,
        symbol_types: frozenset[str] | None = None,
    ) -> list[SymbolResult]:
        # ``StorageTextIndex.search_by_path_prefix`` returns dicts (not Documents).
        try:
            rows = self.fts.search_by_path_prefix(
                path_prefix,
                symbol_types=symbol_types,
                k=k * 2,
            ) if self.fts else []
        except Exception as exc:
            logger.debug("FTS path search failed: %s", exc)
            rows = []
        if not rows:
            rows = [
                self._row_to_dict_passthrough(r)
                for r in self.storage.get_nodes_by_path_prefix(path_prefix, limit=k * 2)
            ]
            if symbol_types:
                rows = [r for r in rows if r.get("symbol_type") in symbol_types]
        return [self._dict_to_result(r) for r in rows[:k]]

    def _jql_type_search(
        self,
        symbol_types: frozenset[str],
        k: int,
    ) -> list[SymbolResult]:
        results: list[SymbolResult] = []
        for stype in symbol_types:
            try:
                docs = self.fts.search_by_type(stype, k=k)
            except Exception as exc:
                logger.debug("FTS type search failed: %s", exc)
                continue
            for d in docs:
                results.append(self._doc_to_result(d))
                if len(results) >= k:
                    return results
        return results[:k]

    def _jql_full_scan(
        self,
        k: int,
        symbol_types: frozenset[str] | None = None,
        layer: str | None = None,
        path_prefix: str | None = None,
        name_filter: str | None = None,
    ) -> list[SymbolResult]:
        # Scoped fallback: pull a reasonable slice then filter in Python.
        if path_prefix:
            rows = self.storage.get_nodes_by_path_prefix(path_prefix, limit=k * 5)
        elif hasattr(self.storage, "get_all_nodes"):
            rows = self.storage.get_all_nodes()
        else:
            rows = self.storage.get_nodes_by_path_prefix("", limit=k * 10)

        out: list[SymbolResult] = []
        for row in rows:
            stype = (row.get("symbol_type") or "").lower()
            if symbol_types and stype not in symbol_types:
                continue
            rp = row.get("rel_path") or ""
            if path_prefix and not self._matches_path(rp, path_prefix):
                continue
            if layer:
                from ..constants import classify_symbol_layer

                computed = classify_symbol_layer(
                    symbol_type=row.get("symbol_type") or "",
                    symbol_name=row.get("symbol_name") or "",
                    parent_symbol=row.get("parent_symbol") or "",
                    file_path=rp,
                    docstring=row.get("docstring") or "",
                )
                if computed != layer:
                    continue
            if name_filter:
                sname = row.get("symbol_name") or ""
                from fnmatch import fnmatch

                if "*" in name_filter or "?" in name_filter:
                    if not fnmatch(sname.lower(), name_filter.lower()):
                        continue
                elif name_filter.lower() not in sname.lower():
                    continue

            out.append(self._dict_to_result(row))
            if len(out) >= k:
                break

        # Scan results have no FTS score — order by connectivity desc.
        self._hydrate_connections(out)
        out.sort(key=lambda r: (-r.connections, r.symbol_name))
        return out[:k]

    # ------------------------------------------------------------------
    # Post-filters (related / has_rel / connections)
    # ------------------------------------------------------------------

    def _apply_post_filters(
        self,
        results: list[SymbolResult],
        jql,
    ) -> list[SymbolResult]:
        if jql.related_value:
            results = self._filter_related(
                results,
                jql.related_value,
                direction=jql.direction_value,
                edge_types=jql.has_rel_values,
            )
        elif jql.has_rel_values:
            results = self._filter_has_relationship(results, jql.has_rel_values)

        if jql.connections_clause:
            self._hydrate_connections(results)
            results = [r for r in results if jql.connections_clause.matches_numeric(r.connections)]

        return results

    def _filter_related(
        self,
        results: list[SymbolResult],
        related_name: str,
        direction: str = "both",
        edge_types: list[str] | None = None,
        max_depth: int = 3,
    ) -> list[SymbolResult]:
        anchor = self.resolve_symbol(related_name)
        if not anchor:
            return []

        type_set = self._normalise_edge_types(edge_types) if edge_types else None
        reachable: set[str] = set()
        visited: set[str] = {anchor}
        frontier: deque[tuple[str, int]] = deque([(anchor, 0)])

        while frontier:
            current, depth = frontier.popleft()
            if depth >= max_depth:
                continue

            edges: list[tuple[str, str]] = []  # (rel_type, other)
            if direction in ("outgoing", "both"):
                for e in self.storage.get_edge_targets(current):
                    tgt = e.get("target_id") or ""
                    if tgt:
                        edges.append((str(e.get("rel_type", "") or "").lower(), tgt))
            if direction in ("incoming", "both"):
                for e in self.storage.get_edge_sources(current):
                    src = e.get("source_id") or ""
                    if src:
                        edges.append((str(e.get("rel_type", "") or "").lower(), src))

            for rel_type, other in edges:
                if type_set is not None and rel_type not in type_set:
                    continue
                reachable.add(other)
                if other not in visited:
                    visited.add(other)
                    frontier.append((other, depth + 1))

        if not results:
            # Materialize reachable set directly when no index clauses ran.
            rows = self.storage.get_nodes_by_ids(list(reachable))
            out = [self._dict_to_result(r) for r in rows]
            self._hydrate_connections(out)
            out.sort(key=lambda r: (-r.connections, r.symbol_name))
            return out

        return [r for r in results if r.node_id in reachable]

    def _filter_has_relationship(
        self,
        results: list[SymbolResult],
        edge_types: list[str],
    ) -> list[SymbolResult]:
        type_set = self._normalise_edge_types(edge_types)
        kept: list[SymbolResult] = []
        for r in results:
            if not r.node_id:
                continue
            found = False
            for e in self.storage.get_edge_targets(r.node_id):
                if str(e.get("rel_type", "") or "").lower() in type_set:
                    found = True
                    break
            if not found:
                for e in self.storage.get_edge_sources(r.node_id):
                    if str(e.get("rel_type", "") or "").lower() in type_set:
                        found = True
                        break
            if found:
                kept.append(r)
        return kept

    # ------------------------------------------------------------------
    # Small helpers
    # ------------------------------------------------------------------

    def _hydrate_connections(self, results: list[SymbolResult]) -> None:
        for r in results:
            if r.connections or not r.node_id:
                continue
            out = len(self.storage.get_edge_targets(r.node_id))
            inc = len(self.storage.get_edge_sources(r.node_id))
            r.connections = out + inc

    @staticmethod
    def _normalise_edge_types(edge_types: list[str]) -> frozenset[str]:
        expanded: set[str] = set()
        for et in edge_types or []:
            et_lower = et.lower()
            expanded.add(et_lower)
            if et_lower in EDGE_TYPE_ALIASES:
                expanded.add(EDGE_TYPE_ALIASES[et_lower])
            for alias, canonical in EDGE_TYPE_ALIASES.items():
                if canonical == et_lower:
                    expanded.add(alias)
        return frozenset(expanded)

    @staticmethod
    def _matches_path(path: str, pattern: str) -> bool:
        if not path or not pattern:
            return False
        if "*" in pattern or "?" in pattern:
            from fnmatch import fnmatch

            return fnmatch(path, pattern)
        prefix = pattern.rstrip("/")
        return path.startswith(prefix + "/") or path == prefix

    def _jql_post_match(
        self,
        r: SymbolResult,
        layer: str | None = None,
        path_prefix: str | None = None,
        name_filter: str | None = None,
    ) -> bool:
        if layer and r.layer != layer:
            return False
        if path_prefix and not self._matches_path(r.rel_path, path_prefix):
            return False
        if name_filter:
            from fnmatch import fnmatch

            if "*" in name_filter or "?" in name_filter:
                return fnmatch(r.symbol_name.lower(), name_filter.lower())
            return name_filter.lower() in r.symbol_name.lower()
        return True

    @staticmethod
    def _doc_to_result(doc) -> SymbolResult:
        meta = getattr(doc, "metadata", doc)
        return SymbolResult(
            node_id=meta.get("node_id", ""),
            symbol_name=meta.get("symbol_name", ""),
            symbol_type=(meta.get("symbol_type", "") or "").lower(),
            layer=meta.get("layer", ""),
            file_path=meta.get("file_path", ""),
            rel_path=meta.get("rel_path", ""),
            connections=0,
            score=meta.get("search_score", 0.0) or 0.0,
            match_source="fts5",
            docstring=meta.get("docstring", "") or "",
        )

    @staticmethod
    def _dict_to_result(row: dict[str, Any]) -> SymbolResult:
        rp = row.get("rel_path") or ""
        return SymbolResult(
            node_id=row.get("node_id", ""),
            symbol_name=row.get("symbol_name", ""),
            symbol_type=(row.get("symbol_type") or "").lower(),
            layer=row.get("layer", "") or "",
            file_path=rp,
            rel_path=rp,
            connections=0,
            score=0.0,
            match_source="storage",
            docstring=row.get("docstring", "") or "",
        )

    @staticmethod
    def _row_to_dict_passthrough(row: Any) -> dict[str, Any]:
        if isinstance(row, dict):
            return row
        try:
            return dict(row)
        except Exception:
            return {}

    # ------------------------------------------------------------------
    # Stats (parity with GraphQueryService)
    # ------------------------------------------------------------------

    def stats(self) -> dict[str, Any]:
        try:
            stats = self.storage.stats()
        except Exception:
            stats = {}
        return {
            "graph_nodes": stats.get("node_count", 0),
            "graph_edges": stats.get("edge_count", 0),
            "node_index_size": 0,
            "name_index_size": 0,
            "full_name_index_size": 0,
            "suffix_index_size": 0,
            "fts5_available": bool(self.fts and self.fts.is_open),
            "backend": "storage",
        }
