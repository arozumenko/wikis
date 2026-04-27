"""Phase 7 — Project graph storage.

Per-project storage of cross-repo nodes/edges, repo↔repo relatedness,
and project metadata. Backend-agnostic via :class:`ProjectStorageProtocol`;
this module ships an in-memory implementation that satisfies the
contract for tests and the off-by-default ``WIKI_PROJECT_GRAPH`` flag.

Concrete SQLite/Postgres implementations live in a follow-up ticket
(Phase 7.1 schema migration). The in-memory backend is sufficient for
unit tests of :mod:`federated_query_service` and
:mod:`federated_retriever`.

All write paths are no-ops when ``flags.project_graph`` is ``False``,
but the **read** paths still return empty lists/None so callers can
treat the protocol as always-available.
"""

from __future__ import annotations

import json
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Iterable, Protocol, runtime_checkable

__all__ = [
    "ProjectStorageProtocol",
    "InMemoryProjectStorage",
]


@runtime_checkable
class ProjectStorageProtocol(Protocol):
    """Cross-repo project storage surface.

    Implementations MUST be safe to call with ``WIKI_PROJECT_GRAPH=0``;
    in that case ``upsert_*`` should be cheap no-ops and ``get_*`` should
    return empty containers.
    """

    def upsert_project_node(
        self,
        project_id: str,
        node_id: str,
        wiki_id: str,
        attrs: dict[str, Any],
    ) -> None:
        """Upsert a project-scoped node row."""

    def upsert_project_edge(
        self,
        project_id: str,
        source_node_id: str,
        target_node_id: str,
        edge_class: str,
        weight: float,
        provenance: dict[str, Any] | None = None,
    ) -> None:
        """Upsert a project-scoped edge row."""

    def get_project_nodes(self, project_id: str) -> list[dict[str, Any]]:
        """Return all project nodes."""

    def get_project_edges(
        self,
        project_id: str,
        *,
        source_node_id: str | None = None,
        edge_class: str | None = None,
    ) -> list[dict[str, Any]]:
        """Return project edges, optionally filtered by source and class."""

    def upsert_repo_relatedness(
        self,
        project_id: str,
        wiki_a: str,
        wiki_b: str,
        score: float,
        breakdown: dict[str, Any] | None = None,
    ) -> None:
        """Upsert a (wiki_a, wiki_b) relatedness row (order-insensitive)."""

    def get_repo_relatedness(self, project_id: str) -> list[dict[str, Any]]:
        """Return all relatedness rows for a project."""

    def set_project_meta(self, project_id: str, key: str, value: Any) -> None:
        """Persist a small JSON-serializable metadata blob."""

    def get_project_meta(self, project_id: str, key: str) -> Any:
        """Return previously-set metadata, or ``None`` if missing."""


# ----------------------------------------------------------------------
# In-memory implementation
# ----------------------------------------------------------------------


def _pair_key(a: str, b: str) -> tuple[str, str]:
    """Order-insensitive key for ``(wiki_a, wiki_b)``."""
    return (a, b) if a <= b else (b, a)


@dataclass
class _ProjectStore:
    nodes: dict[str, dict[str, Any]] = field(default_factory=dict)
    # Edges keyed by (src, tgt, edge_class) so duplicates collapse.
    edges: dict[tuple[str, str, str], dict[str, Any]] = field(default_factory=dict)
    relatedness: dict[tuple[str, str], dict[str, Any]] = field(default_factory=dict)
    meta: dict[str, Any] = field(default_factory=dict)


class InMemoryProjectStorage:
    """Reference :class:`ProjectStorageProtocol` implementation.

    Backed by a per-project in-memory store. Used by tests and as the
    default until SQLite/Postgres backends ship.
    """

    def __init__(self) -> None:
        self._stores: dict[str, _ProjectStore] = defaultdict(_ProjectStore)

    # ------------------------------------------------------------------
    # Nodes
    # ------------------------------------------------------------------

    def upsert_project_node(
        self,
        project_id: str,
        node_id: str,
        wiki_id: str,
        attrs: dict[str, Any],
    ) -> None:
        if not project_id or not node_id:
            return
        row = {"node_id": node_id, "wiki_id": wiki_id, **(attrs or {})}
        self._stores[project_id].nodes[node_id] = row

    def get_project_nodes(self, project_id: str) -> list[dict[str, Any]]:
        store = self._stores.get(project_id)
        if store is None:
            return []
        return list(store.nodes.values())

    # ------------------------------------------------------------------
    # Edges
    # ------------------------------------------------------------------

    def upsert_project_edge(
        self,
        project_id: str,
        source_node_id: str,
        target_node_id: str,
        edge_class: str,
        weight: float,
        provenance: dict[str, Any] | None = None,
    ) -> None:
        if not project_id or not source_node_id or not target_node_id:
            return
        key = (source_node_id, target_node_id, edge_class)
        self._stores[project_id].edges[key] = {
            "source_node_id": source_node_id,
            "target_node_id": target_node_id,
            "edge_class": edge_class,
            "weight": float(weight),
            "provenance": dict(provenance or {}),
        }

    def get_project_edges(
        self,
        project_id: str,
        *,
        source_node_id: str | None = None,
        edge_class: str | None = None,
    ) -> list[dict[str, Any]]:
        store = self._stores.get(project_id)
        if store is None:
            return []
        rows: list[dict[str, Any]] = []
        for (src, _tgt, cls), row in store.edges.items():
            if source_node_id is not None and src != source_node_id:
                continue
            if edge_class is not None and cls != edge_class:
                continue
            rows.append(dict(row))
        return rows

    # ------------------------------------------------------------------
    # Relatedness
    # ------------------------------------------------------------------

    def upsert_repo_relatedness(
        self,
        project_id: str,
        wiki_a: str,
        wiki_b: str,
        score: float,
        breakdown: dict[str, Any] | None = None,
    ) -> None:
        if not project_id or not wiki_a or not wiki_b or wiki_a == wiki_b:
            return
        key = _pair_key(wiki_a, wiki_b)
        self._stores[project_id].relatedness[key] = {
            "wiki_a": key[0],
            "wiki_b": key[1],
            "score": float(score),
            "breakdown": dict(breakdown or {}),
        }

    def get_repo_relatedness(self, project_id: str) -> list[dict[str, Any]]:
        store = self._stores.get(project_id)
        if store is None:
            return []
        return [dict(r) for r in store.relatedness.values()]

    # ------------------------------------------------------------------
    # Meta
    # ------------------------------------------------------------------

    def set_project_meta(self, project_id: str, key: str, value: Any) -> None:
        if not project_id or not key:
            return
        # Round-trip through json so callers don't accidentally hand us
        # objects we cannot persist later in a real backend.
        try:
            json.dumps(value)
        except (TypeError, ValueError):
            return
        self._stores[project_id].meta[key] = value

    def get_project_meta(self, project_id: str, key: str) -> Any:
        store = self._stores.get(project_id)
        if store is None:
            return None
        return store.meta.get(key)
