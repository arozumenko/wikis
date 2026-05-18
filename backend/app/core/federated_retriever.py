"""Phase 7 — Federated retriever.

Drop-in replacement for :class:`MultiWikiRetrieverStack` that adds:

1. **1-hop cross-repo expansion** — for every retrieved document whose
   ``metadata["node_id"]`` participates in a project edge with
   ``edge_class='cross_repo'``, also pull the target node's content
   from the federated query service.
2. **Cross-repo dampening** — expanded hits get their score multiplied
   by ``flags.cross_repo_dampening`` (default ``0.7``) so they cannot
   outrank in-repo results from the originating wiki.

Behaviour collapses to the parent class when no project storage is
attached or no cross-repo edges have been persisted, so the wrapper is
safe under default flag settings.
"""

from __future__ import annotations

import logging
from typing import Any

from .code_graph.cross_repo_linker import make_project_node_id
from .multi_retriever import MultiWikiRetrieverStack

logger = logging.getLogger(__name__)

__all__ = ["FederatedRetrieverStack"]


_DEFAULT_DAMPENING = 0.7


class FederatedRetrieverStack(MultiWikiRetrieverStack):
    """Project-aware retriever that expands hits along ``cross_repo`` edges.

    Args:
        wiki_stacks: ``[(wiki_id, WikiRetrieverStack), ...]``.
        federated_query_service: Optional :class:`FederatedQueryService`
            used to resolve cross-repo edge targets to node payloads.
            When ``None`` the expansion step is skipped (parent
            behaviour preserved).
        dampening: Score multiplier applied to expanded hits. Falls
            back to ``0.7`` when ``None`` or out of ``(0, 1]``.
    """

    def __init__(
        self,
        wiki_stacks: list[tuple[str, Any]],
        *,
        federated_query_service: Any = None,
        dampening: float | None = None,
    ) -> None:
        super().__init__(wiki_stacks)
        self._fqs = federated_query_service
        d = dampening if dampening is not None else _DEFAULT_DAMPENING
        try:
            d = float(d)
        except (TypeError, ValueError):
            d = _DEFAULT_DAMPENING
        if not (0.0 < d <= 1.0):
            d = _DEFAULT_DAMPENING
        self._dampening = d

    # ------------------------------------------------------------------
    # Public — override aretrieve to insert the expansion stage
    # ------------------------------------------------------------------

    async def aretrieve(self, query: str, k: int = 15) -> list:
        base_hits = await super().aretrieve(query, k=k)
        if not base_hits or self._fqs is None:
            return base_hits

        expanded = self._expand_with_cross_repo(base_hits, k=k)
        if not expanded:
            return base_hits

        # Stable merge: keep base order, append expanded hits not already
        # present, then re-sort by normalized_score desc.
        seen = {self._hit_key(h) for h in base_hits}
        merged = list(base_hits)
        for hit in expanded:
            key = self._hit_key(hit)
            if key in seen:
                continue
            seen.add(key)
            merged.append(hit)

        merged.sort(
            key=lambda d: d.metadata.get("normalized_score", 0.0),
            reverse=True,
        )
        return merged[: k * 2]

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _expand_with_cross_repo(self, base_hits: list, *, k: int) -> list:
        """For every base hit, fetch 1-hop cross-repo neighbours."""
        getter = getattr(self._fqs, "cross_repo_edges", None)
        node_lookup = getattr(self._fqs, "get_node", None)
        if getter is None or node_lookup is None:
            return []

        # Cap expansion to 2× of what the caller asked for to avoid
        # blowing up the candidate pool.
        budget = max(k, 1)
        out: list = []
        for hit in base_hits:
            if len(out) >= budget:
                break
            node_id = (hit.metadata or {}).get("node_id")
            if not node_id:
                continue
            try:
                source_wiki_id = (hit.metadata or {}).get("source_wiki_id") or (hit.metadata or {}).get("wiki_id")
                lookup_id = make_project_node_id(source_wiki_id, node_id) if source_wiki_id else node_id
                edges = getter(lookup_id) or []
                if not edges and lookup_id != node_id:
                    edges = getter(node_id) or []
            except Exception:  # pragma: no cover — defensive
                continue
            for edge in edges:
                if len(out) >= budget:
                    break
                tgt = edge.get("target_node_id")
                if not tgt:
                    continue
                try:
                    node = node_lookup(tgt)
                except Exception:  # pragma: no cover — defensive
                    node = None
                if not node:
                    continue
                doc = self._materialize_doc(hit, node, edge)
                if doc is not None:
                    out.append(doc)
        return out

    def _materialize_doc(self, source_hit, target_node: dict, edge: dict):
        """Build a Document-shaped object for an expanded hit.

        Reuses the source hit's class so we stay LangChain-agnostic;
        propagates ``normalized_score`` × dampening × edge weight.
        """
        try:
            cls = source_hit.__class__
        except AttributeError:
            return None
        try:
            edge_weight = float(edge.get("weight", 1.0) or 1.0)
        except (TypeError, ValueError):
            edge_weight = 1.0
        base_norm = float(source_hit.metadata.get("normalized_score", 0.0) or 0.0)
        new_norm = max(0.0, base_norm * self._dampening * edge_weight)
        provenance = dict(edge.get("provenance") or {})
        meta = {
            **(target_node.get("attrs") or {}),
            "node_id": target_node.get("node_id") or provenance.get("target_raw_node_id") or edge.get("target_node_id"),
            "project_node_id": target_node.get("project_node_id") or edge.get("target_node_id"),
            "source_wiki_id": target_node.get("wiki_id", ""),
            "cross_repo_origin": (source_hit.metadata or {}).get("node_id"),
            "cross_repo_edge_weight": edge_weight,
            "cross_repo_edge_class": edge.get("edge_class", "cross_repo"),
            "cross_repo_relationship_type": provenance.get("source_relationship_type", "cross_repo"),
            "cross_repo_surface": provenance.get("surface", ""),
            "cross_repo_matcher": provenance.get("matcher", ""),
            "cross_repo_level": provenance.get("level", ""),
            "cross_repo_provenance": provenance,
            "normalized_score": new_norm,
            "score": new_norm,  # legacy consumers
        }
        content = (
            target_node.get("content")
            or target_node.get("source_text")
            or ""
        )
        try:
            return cls(page_content=content, metadata=meta)
        except TypeError:
            return None

    @staticmethod
    def _hit_key(hit) -> str:
        meta = getattr(hit, "metadata", {}) or {}
        return f"{meta.get('source_wiki_id', '')}::{meta.get('node_id', '')}"
