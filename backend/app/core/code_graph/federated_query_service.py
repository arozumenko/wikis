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
from typing import Any

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

    def cross_repo_edges(self, node_id: str) -> list[dict[str, Any]]:
        """Return outbound ``cross_repo`` edges for ``node_id``.

        Empty list when the project store is missing or the node has no
        cross-repo relationships persisted.
        """
        if not node_id or self._project_storage is None:
            return []
        getter = getattr(self._project_storage, "get_project_edges", None)
        if getter is None:
            return []
        try:
            return getter(
                self._project_id,
                source_node_id=node_id,
                edge_class="cross_repo",
            )
        except Exception:  # pragma: no cover — defensive
            logger.debug("cross_repo_edges read failed", exc_info=True)
            return []

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
