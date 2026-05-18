"""Phase 7 — Federated query service.

Drop-in replacement for :class:`MultiGraphQueryService` that adds
project-level cross-repo awareness:

* ``cross_repo_edges(node_id)`` — read project edges with
  ``edge_class='cross_repo'`` from :class:`ProjectStorageProtocol`.
* ``relatedness(wiki_a, wiki_b)`` — read precomputed pair score.

Side-effect free: never writes to the project store. The companion
linker (Phase 8) is responsible for populating ``project_edges`` /
``repo_relatedness``.

Wiring is gated by ``flags.federated_query``; when off, callers should
keep using :class:`MultiGraphQueryService` directly.
"""

from __future__ import annotations

import logging
from dataclasses import replace
from typing import Any

from .cross_repo_linker import make_project_node_id, split_project_node_id
from .graph_query_service import RelationshipResult
from .multi_graph_query_service import MultiGraphQueryService

logger = logging.getLogger(__name__)

__all__ = ["FederatedQueryService"]


class FederatedQueryService(MultiGraphQueryService):
    """Project-aware extension of :class:`MultiGraphQueryService`.

    Args:
        services: ``{wiki_id: GraphQueryService}`` mapping (passed up).
        project_id: Project this federation belongs to.
        project_storage: Implementation of
            :class:`ProjectStorageProtocol`. Optional — when ``None``
            the project methods return empty containers and a relatedness
            of ``0.0``.
    """

    def __init__(
        self,
        services: dict[str, Any],
        *,
        project_id: str,
        project_storage: Any = None,
    ) -> None:
        super().__init__(services)
        self._project_id = project_id
        self._project_storage = project_storage

    # ------------------------------------------------------------------
    # Cross-repo accessors
    # ------------------------------------------------------------------

    def _split_known_project_node_id(self, node_id: str) -> tuple[str | None, str]:
        wiki_id, raw_node_id = split_project_node_id(node_id)
        if wiki_id:
            return wiki_id, raw_node_id
        for candidate_wiki in self._services:
            prefix = f"{candidate_wiki}::"
            if str(node_id or "").startswith(prefix):
                return candidate_wiki, str(node_id)[len(prefix):]
        return wiki_id, raw_node_id

    def get_node(self, node_id: str) -> dict[str, Any] | None:
        """Resolve project-namespaced or raw node ids across federated wikis."""
        wiki_id, raw_node_id = self._split_known_project_node_id(node_id)
        if wiki_id and wiki_id in self._services:
            getter = getattr(self._services[wiki_id], "get_node", None)
            if getter is not None:
                try:
                    data = getter(raw_node_id)
                except Exception:  # pragma: no cover — defensive
                    data = None
                if data is not None:
                    out = dict(data)
                    out.setdefault("node_id", raw_node_id)
                    out.setdefault("wiki_id", wiki_id)
                    out.setdefault("raw_node_id", raw_node_id)
                    out.setdefault("project_node_id", node_id)
                    return out
        return super().get_node(node_id)

    def resolve_symbol(
        self,
        symbol_name: str,
        file_path: str = "",
        language: str = "",
    ) -> str | None:
        """Resolve a symbol and return a project-namespaced node id when possible."""
        if "::" in (symbol_name or "") and self.get_node(symbol_name) is not None:
            return symbol_name
        for wiki_id, svc in self._services.items():
            result = svc.resolve_symbol(
                symbol_name,
                file_path=file_path,
                language=language,
            )
            if result:
                return make_project_node_id(wiki_id, result)
        return None

    def search(
        self,
        query: str,
        k: int = 20,
        symbol_types: frozenset[str] | None = None,
        exclude_types: frozenset[str] | None = None,
        layer: str | None = None,
        path_prefix: str | None = None,
    ):
        """Fan out search and namespace result node ids by wiki."""
        all_results = []
        per_wiki_k = max(k, 5)
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
            except Exception as exc:
                logger.warning("federated search failed for wiki %s: %s", wiki_id, exc)
                continue
            for result in results:
                raw_node_id = result.node_id
                tagged = replace(
                    result,
                    node_id=make_project_node_id(wiki_id, raw_node_id),
                )
                setattr(tagged, "wiki_id", wiki_id)
                setattr(tagged, "source_wiki_id", wiki_id)
                setattr(tagged, "raw_node_id", raw_node_id)
                all_results.append(tagged)
        all_results.sort(key=lambda r: (-r.score, -r.connections))
        return all_results[:k]

    def get_relationships(
        self,
        node_id: str,
        direction: str = "both",
        max_depth: int = 2,
        max_results: int = 50,
    ):
        """Traverse local relationships plus persisted direct project edges."""
        wiki_id, raw_node_id = self._split_known_project_node_id(node_id)
        project_node_id = make_project_node_id(wiki_id, raw_node_id) if wiki_id else node_id
        project_rels = self._cross_repo_relationships(
            project_node_id,
            direction=direction,
            max_results=max_results,
        )

        if wiki_id and wiki_id in self._services:
            local_rels = self._services[wiki_id].get_relationships(
                raw_node_id,
                direction=direction,
                max_depth=max_depth,
                max_results=max(max_results - len(project_rels), 0),
            )
            return (project_rels + local_rels)[:max_results]

        local_rels = super().get_relationships(
            node_id,
            direction=direction,
            max_depth=max_depth,
            max_results=max(max_results - len(project_rels), 0),
        )
        return (project_rels + local_rels)[:max_results]

    def resolve_and_traverse(
        self,
        symbol_name: str,
        direction: str = "both",
        max_depth: int = 2,
        max_results: int = 50,
        file_path: str = "",
        language: str = "",
    ):
        """Resolve a symbol and traverse within the owning wiki."""
        node_id = self.resolve_symbol(
            symbol_name,
            file_path=file_path,
            language=language,
        )
        if not node_id:
            return None, []
        return node_id, self.get_relationships(
            node_id,
            direction=direction,
            max_depth=max_depth,
            max_results=max_results,
        )

    def query(self, expression: str):
        """Execute structured queries across wikis and namespace result node ids."""
        seen_node_ids: set[str] = set()
        all_results = []
        for wiki_id, svc in self._services.items():
            try:
                results = svc.query(expression)
            except Exception as exc:
                logger.warning("federated query failed for wiki %s: %s", wiki_id, exc)
                continue
            for result in results:
                raw_node_id = result.node_id
                project_node_id = make_project_node_id(wiki_id, raw_node_id)
                if project_node_id in seen_node_ids:
                    continue
                seen_node_ids.add(project_node_id)
                tagged = replace(result, node_id=project_node_id)
                setattr(tagged, "wiki_id", wiki_id)
                setattr(tagged, "source_wiki_id", wiki_id)
                setattr(tagged, "raw_node_id", raw_node_id)
                all_results.append(tagged)
        all_results.sort(key=lambda r: (-r.score, -r.connections))
        return all_results

    def cross_repo_edges(self, node_id: str, direction: str = "outgoing") -> list[dict[str, Any]]:
        """Return direct ``cross_repo`` edges for ``node_id``.

        Empty list when the project store is missing or the node has no
        cross-repo relationships persisted.
        """
        if not node_id or self._project_storage is None:
            return []
        getter = getattr(self._project_storage, "get_project_edges", None)
        if getter is None:
            return []
        try:
            rows: list[dict[str, Any]] = []
            seen: set[tuple[str, str, str]] = set()

            def _add(edge_rows: list[dict[str, Any]]) -> None:
                for row in edge_rows or []:
                    key = (
                        row.get("source_node_id", ""),
                        row.get("target_node_id", ""),
                        row.get("edge_class", ""),
                    )
                    if key in seen:
                        continue
                    seen.add(key)
                    rows.append(row)

            if direction in ("outgoing", "both"):
                _add(getter(
                    self._project_id,
                    source_node_id=node_id,
                    edge_class="cross_repo",
                ))
            if direction in ("incoming", "both"):
                _add(getter(
                    self._project_id,
                    target_node_id=node_id,
                    edge_class="cross_repo",
                ))
            return rows
        except Exception:  # pragma: no cover — defensive
            logger.debug("cross_repo_edges read failed", exc_info=True)
            return []

    def _cross_repo_relationships(
        self,
        node_id: str,
        *,
        direction: str,
        max_results: int,
    ) -> list[RelationshipResult]:
        """Convert persisted project edge rows into RelationshipResult entries."""
        rows = self.cross_repo_edges(node_id, direction=direction)
        rels: list[RelationshipResult] = []
        for row in rows:
            if len(rels) >= max_results:
                break
            source_node_id = row.get("source_node_id", "") or ""
            target_node_id = row.get("target_node_id", "") or ""
            if not source_node_id or not target_node_id:
                continue
            provenance = dict(row.get("provenance") or {})
            source_wiki_id, _source_raw = self._split_known_project_node_id(source_node_id)
            target_wiki_id, _target_raw = self._split_known_project_node_id(target_node_id)
            source_data = self.get_node(source_node_id) or {}
            target_data = self.get_node(target_node_id) or {}
            rel_type = (
                provenance.get("source_relationship_type")
                or row.get("relationship_type")
                or row.get("edge_class")
                or "cross_repo"
            )
            rels.append(
                RelationshipResult(
                    source_name=source_data.get("symbol_name", "")
                    or source_data.get("name", "")
                    or source_node_id,
                    target_name=target_data.get("symbol_name", "")
                    or target_data.get("name", "")
                    or target_node_id,
                    relationship_type=str(rel_type),
                    source_type=(source_data.get("symbol_type") or "").lower(),
                    target_type=(target_data.get("symbol_type") or "").lower(),
                    hop_distance=1,
                    source_node_id=source_node_id,
                    target_node_id=target_node_id,
                    source_wiki_id=provenance.get("source_wiki_id") or source_wiki_id,
                    target_wiki_id=provenance.get("target_wiki_id") or target_wiki_id,
                    weight=float(row.get("weight", 1.0) or 1.0),
                    provenance=provenance,
                )
            )
        return rels

    def relatedness(self, wiki_a: str, wiki_b: str) -> float:
        """Return the precomputed (wiki_a, wiki_b) relatedness in ``[0, 1]``.

        ``0.0`` when the project store is missing or the pair has not
        been scored. Order-insensitive.
        """
        if not wiki_a or not wiki_b or wiki_a == wiki_b:
            return 0.0
        if self._project_storage is None:
            return 0.0
        getter = getattr(self._project_storage, "get_repo_relatedness", None)
        if getter is None:
            return 0.0
        try:
            rows = getter(self._project_id) or []
        except Exception:  # pragma: no cover — defensive
            return 0.0
        target = tuple(sorted((wiki_a, wiki_b)))
        for row in rows:
            pair = tuple(sorted((row.get("wiki_a", ""), row.get("wiki_b", ""))))
            if pair == target:
                try:
                    return float(row.get("score", 0.0))
                except (TypeError, ValueError):
                    return 0.0
        return 0.0
