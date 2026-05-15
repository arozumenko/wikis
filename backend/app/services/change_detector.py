"""Change detection for #116 incremental wiki regeneration (PR 2).

Read-only service that compares a freshly-parsed graph against the indexed
state in storage and answers two questions:

1. Which nodes changed (added / modified / deleted / moved)?
2. Which wiki pages cite those nodes and therefore need attention?

The output is purely diagnostic in PR 2 — no LLM calls, no writes. PR 3
consumes the same dataclasses to route pages into the trivial / edit /
structural regimes.

Design notes:

* "Moved" means same node_id with a different ``rel_path`` and an unchanged
  ``content_hash``. We don't currently emit move detection for renamed
  symbols (different node_id) — that needs name-based matching and lands in
  PR 3's structural regime.

* ``content_hash`` may be ``None`` for nodes indexed before #116 PR 1
  landed. We treat ``None`` as "hash unknown" and assume the node is
  modified — pessimistic but correct.

* The affected-page lookup uses ``get_pages_citing_node`` which is the
  reverse index PR 1 populates. Pages whose only changes were ``moved``
  nodes (no semantic change) still show up here — PR 3 will route those
  to the no-LLM "trivial" regime.
"""

from __future__ import annotations

import logging
from collections import defaultdict
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from app.core.storage.protocol import WikiStorageProtocol

logger = logging.getLogger(__name__)


class ChangeKind(str, Enum):
    """Why a node appears in the change set.

    Ordering note: when a page is touched by multiple change kinds, the
    *highest-severity* one wins for downstream regime classification.
    Severity order (low → high): MOVED < MODIFIED < ADDED < DELETED.
    """

    ADDED = "added"
    MODIFIED = "modified"
    DELETED = "deleted"
    MOVED = "moved"


@dataclass(frozen=True)
class NodeChange:
    """One node-level change.

    ``kind`` determines which of the optional fields are populated:

    * ``ADDED``    — only ``node_id`` + ``new_hash`` + ``new_path``
    * ``MODIFIED`` — ``node_id`` + ``old_hash`` + ``new_hash``
    * ``DELETED`` — ``node_id`` + ``old_hash`` + ``old_path``
    * ``MOVED``   — ``node_id`` + ``old_path`` + ``new_path`` (hash same)
    """

    kind: ChangeKind
    node_id: str
    old_hash: str | None = None
    new_hash: str | None = None
    old_path: str | None = None
    new_path: str | None = None


@dataclass
class ChangeSet:
    """Aggregate result of a single ``detect_changes`` call.

    Each list contains :class:`NodeChange` instances of the matching kind.
    The split is for convenience — callers can also iterate ``.all()``.
    """

    added: list[NodeChange] = field(default_factory=list)
    modified: list[NodeChange] = field(default_factory=list)
    deleted: list[NodeChange] = field(default_factory=list)
    moved: list[NodeChange] = field(default_factory=list)

    def all(self) -> list[NodeChange]:
        """Iterate every change in detection order: added → modified → moved → deleted."""
        return [*self.added, *self.modified, *self.moved, *self.deleted]

    @property
    def is_empty(self) -> bool:
        return not (self.added or self.modified or self.deleted or self.moved)

    @property
    def total(self) -> int:
        return len(self.added) + len(self.modified) + len(self.deleted) + len(self.moved)


# Severity ordering used by ``AffectedPage.max_kind`` — exposed at module
# scope so PR 3's regime classifier can reuse the same ranks without
# duplicating the table. Higher number = more severe → more regen work.
_CHANGE_KIND_SEVERITY: dict[ChangeKind, int] = {
    ChangeKind.MOVED: 0,
    ChangeKind.MODIFIED: 1,
    ChangeKind.ADDED: 2,
    ChangeKind.DELETED: 3,
}


@dataclass
class AffectedPage:
    """One wiki page touched by the change set.

    ``changes`` carries every :class:`NodeChange` whose ``node_id`` is in
    this page's ``page_symbols`` rows. Pages with no citing-node changes
    are not included in the affected-pages output.
    """

    page_id: str
    changes: list[NodeChange] = field(default_factory=list)

    @property
    def max_kind(self) -> ChangeKind | None:
        """The highest-severity change kind touching this page.

        PR 3's three-regime classifier uses this to route: ``MOVED`` →
        trivial (no LLM); ``MODIFIED`` → edit; ``ADDED``/``DELETED`` →
        structural. Returns ``None`` if the page has no changes (an
        invariant violation — pages with empty ``changes`` shouldn't be
        in the affected-pages list).
        """
        if not self.changes:
            return None
        return max(
            self.changes,
            key=lambda c: _CHANGE_KIND_SEVERITY[c.kind],
        ).kind


# ---------------------------------------------------------------------------
# Service
# ---------------------------------------------------------------------------


class ChangeDetector:
    """Diff a parsed-now node set against the indexed-then state in storage.

    Construct once per wiki. Stateless beyond the ``storage`` handle, so a
    detector can be reused across multiple ``detect_changes`` calls if the
    parsed input is rebuilt between them.
    """

    def __init__(self, storage: WikiStorageProtocol) -> None:
        self._storage = storage

    # ------------------------------------------------------------------
    # detect_changes
    # ------------------------------------------------------------------

    def detect_changes(
        self,
        parsed_nodes: Iterable[Mapping[str, Any]],
    ) -> ChangeSet:
        """Compute the change set against the currently-indexed nodes.

        Args:
            parsed_nodes: freshly-parsed node dicts with at least
                ``node_id``, ``content_hash``, and ``rel_path`` keys. Other
                keys are ignored — keeps callers free to pass through the
                same dicts they would hand to ``upsert_nodes_batch``.

        Returns:
            A :class:`ChangeSet` populated with one :class:`NodeChange` per
            differing node. Unchanged nodes are not represented.
        """
        # Single full-table scan returns hash + path together — avoids a
        # second ``WHERE node_id IN (...)`` query that would hit SQLite's
        # ``SQLITE_MAX_VARIABLE_NUMBER`` limit on wikis with thousands of
        # indexed nodes.
        old_meta = self._storage.fetch_indexed_node_meta()

        # Index the parsed input by node_id for O(1) lookup. We deliberately
        # don't materialise content_text here — only hash + path are needed
        # to classify the change kind.
        parsed_index: dict[str, tuple[str | None, str]] = {}
        for node in parsed_nodes:
            node_id = node.get("node_id")
            if not node_id:
                continue
            parsed_index[node_id] = (
                node.get("content_hash"),
                node.get("rel_path", ""),
            )

        result = ChangeSet()

        # Pass 1: traverse parsed_index, classify each node against the old state.
        for node_id, (new_hash, new_path) in parsed_index.items():
            if node_id not in old_meta:
                result.added.append(
                    NodeChange(
                        kind=ChangeKind.ADDED,
                        node_id=node_id,
                        new_hash=new_hash,
                        new_path=new_path,
                    )
                )
                continue
            old = old_meta[node_id]
            old_hash = old["content_hash"]
            old_path = old["rel_path"] or ""
            if old_hash is None:
                # Pre-#116 row with no recorded hash. We can't tell whether
                # source changed — assume modified to keep correctness.
                result.modified.append(
                    NodeChange(
                        kind=ChangeKind.MODIFIED,
                        node_id=node_id,
                        old_hash=None,
                        new_hash=new_hash,
                        old_path=old_path,
                        new_path=new_path,
                    )
                )
                continue
            if old_hash == new_hash:
                if old_path != new_path:
                    result.moved.append(
                        NodeChange(
                            kind=ChangeKind.MOVED,
                            node_id=node_id,
                            old_hash=old_hash,
                            new_hash=new_hash,
                            old_path=old_path,
                            new_path=new_path,
                        )
                    )
                # else: truly unchanged — omit from the change set.
                continue
            result.modified.append(
                NodeChange(
                    kind=ChangeKind.MODIFIED,
                    node_id=node_id,
                    old_hash=old_hash,
                    new_hash=new_hash,
                    old_path=old_path,
                    new_path=new_path,
                )
            )

        # Pass 2: anything in old_meta but not in parsed_index is deleted.
        for node_id, old in old_meta.items():
            if node_id in parsed_index:
                continue
            result.deleted.append(
                NodeChange(
                    kind=ChangeKind.DELETED,
                    node_id=node_id,
                    old_hash=old["content_hash"],
                    old_path=old["rel_path"] or "",
                )
            )

        logger.info(
            "[change_detector] %d added, %d modified, %d moved, %d deleted "
            "(%d unchanged)",
            len(result.added),
            len(result.modified),
            len(result.moved),
            len(result.deleted),
            len(parsed_index) - len(result.added) - len(result.modified) - len(result.moved),
        )
        return result

    # ------------------------------------------------------------------
    # affected_pages
    # ------------------------------------------------------------------

    def affected_pages(self, change_set: ChangeSet) -> list[AffectedPage]:
        """Group node-level changes by the wiki pages that cite them.

        Pages that don't cite any changed node are not returned. Pages that
        cite multiple changed nodes get one :class:`AffectedPage` with all
        the matching :class:`NodeChange`s in ``.changes``.
        """
        if change_set.is_empty:
            return []

        # Map page_id → list of NodeChange via the reverse index.
        page_to_changes: dict[str, list[NodeChange]] = defaultdict(list)
        for change in change_set.all():
            for page_id in self._storage.get_pages_citing_node(change.node_id):
                page_to_changes[page_id].append(change)

        return [
            AffectedPage(page_id=page_id, changes=changes)
            for page_id, changes in sorted(page_to_changes.items())
        ]

