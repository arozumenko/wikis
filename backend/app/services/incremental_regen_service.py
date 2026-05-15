"""Incremental-regen orchestrator for #116 PR 3.

This service ties together the four PR 1–4 building blocks:

* PR 1 — ``wiki_pages`` + ``page_symbols`` (the reverse index).
* PR 2 — :class:`ChangeDetector` (what changed, which pages it touches).
* PR 3 — regime classifier + trivial patcher + surgical :class:`PagePatcher`.
* PR 4 — cluster ID stability + embedding reuse + ``apply_incremental_node_writes``.

Dispatch flow:

1. Snapshot old node source for "edit" candidates *before* the upsert.
2. Apply the parsed-node batch via ``apply_incremental_node_writes``
   (single call → upsert + FTS refresh bundled per issue #131).
3. Run :class:`ChangeDetector` against the now-current snapshot to
   produce the :class:`ChangeSet` and :class:`AffectedPage` list.
4. Classify with :func:`classify_pages` into a :class:`RegimePlan`.
5. Per regime:
   - ``unchanged`` → no-op.
   - ``trivial`` → :func:`patch_trivial_page`, persist body, update
     ``wiki_pages.content_hash``.
   - ``edit`` → build per-symbol diffs, run :class:`PagePatcher`. If
     the quality gate rejects, demote to ``structural``.
   - ``structural`` → delegate to the injected ``structural_handler``
     callback (production wires this to the existing wiki-generation
     agent; tests can stub it).
6. Return :class:`IncrementalRegenStats` for telemetry (PR 5 surfaces
   this as SSE events + UI banner).

The orchestrator owns no LLM and no artifact-storage handle directly —
callers inject :class:`PagePatcher` (which owns the LLM) and the
page-content read/write callbacks. Keeps PR 3 unit-testable without
spinning up the full :class:`OptimizedWikiGenerationAgent`.

Wait — step 1 of the flow above is subtle: we need to snapshot OLD
node source text BEFORE the upsert so the surgical editor has both
sides of the diff. We can't read it from storage after the upsert
(storage will hold the new source). The orchestrator captures it via
``storage.get_nodes_by_ids`` over the modified-node set, keyed by
``node_id``, before calling ``apply_incremental_node_writes``.
"""

from __future__ import annotations

import logging
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass, field
from typing import Any

from app.core.agents.page_patcher import PagePatcher, PatchResult, SymbolDiff
from app.core.storage.protocol import WikiStorageProtocol
from app.services.change_detector import (
    AffectedPage,
    ChangeDetector,
    ChangeKind,
    ChangeSet,
    NodeChange,
)
from app.services.incremental_regen import (
    PageRegime,
    Regime,
    RegimePlan,
    classify_pages,
)
from app.services.trivial_patcher import patch_trivial_page

logger = logging.getLogger(__name__)


# Callbacks the orchestrator depends on. Kept as type aliases at module
# scope so tests can pass stub functions inline.

#: Returns the current markdown body for a page_id, or ``None`` if missing.
PageBodyReader = Callable[[str], str | None]

#: Writes the new markdown body for a page_id. Production = artifact storage.
PageBodyWriter = Callable[[str, str], None]

#: Handles structural-regime regen for one page. Returns True if the page
#: was successfully regenerated; False to mark as failed in the stats.
StructuralHandler = Callable[[PageRegime], bool]


@dataclass
class IncrementalRegenStats:
    """Per-run summary surfaced to operators + PR 5 telemetry."""

    unchanged: int = 0
    trivial_patched: int = 0
    edit_applied: int = 0
    edit_demoted_to_structural: int = 0
    structural_regenerated: int = 0
    structural_failed: int = 0
    diff_ratios: list[float] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        """Telemetry-friendly snapshot."""
        return {
            "unchanged": self.unchanged,
            "trivial_patched": self.trivial_patched,
            "edit_applied": self.edit_applied,
            "edit_demoted_to_structural": self.edit_demoted_to_structural,
            "structural_regenerated": self.structural_regenerated,
            "structural_failed": self.structural_failed,
            "avg_diff_ratio": (
                sum(self.diff_ratios) / len(self.diff_ratios)
                if self.diff_ratios
                else 0.0
            ),
        }


class IncrementalRegenService:
    """One-shot orchestrator. Construct per refresh; throw away after."""

    def __init__(
        self,
        storage: WikiStorageProtocol,
        page_patcher: PagePatcher,
        *,
        read_page_body: PageBodyReader,
        write_page_body: PageBodyWriter,
        structural_handler: StructuralHandler,
    ) -> None:
        self._storage = storage
        self._patcher = page_patcher
        self._read_page = read_page_body
        self._write_page = write_page_body
        self._structural_handler = structural_handler

    def run(
        self,
        parsed_nodes: list[dict[str, Any]],
    ) -> IncrementalRegenStats:
        """Execute one incremental regen pass.

        Args:
            parsed_nodes: freshly parsed nodes (same shape as ``upsert_nodes_batch``
                input). Must include ``node_id``, ``content_hash``,
                ``rel_path``, ``source_text`` for everything the
                surgical patcher might need to diff.

        Returns:
            :class:`IncrementalRegenStats` with per-regime counts and
            the avg diff_ratio (for tuning the quality gate over time).
        """
        # Step 1: snapshot old node source/sig BEFORE the upsert. The
        # surgical patcher's SymbolDiff needs both sides; once we upsert
        # parsed_nodes the storage holds only the new state.
        old_node_snapshot = self._snapshot_old_state(parsed_nodes)

        # Step 2: detect changes (still against pre-upsert state).
        detector = ChangeDetector(self._storage)
        change_set = detector.detect_changes(parsed_nodes)
        affected = detector.affected_pages(change_set)
        plan = classify_pages(change_set, affected, self._storage)

        # Step 3: persist the new node state — single bundled call that
        # also refreshes FTS (#131). After this the storage handle
        # reflects the new repository state.
        self._storage.apply_incremental_node_writes(parsed_nodes)

        # Build a lookup of new parsed-node state for the dispatcher.
        new_by_id: dict[str, dict[str, Any]] = {
            n["node_id"]: n for n in parsed_nodes if n.get("node_id")
        }

        stats = IncrementalRegenStats()
        stats.unchanged = len(plan.unchanged)

        # Step 4: per-regime dispatch.
        for page in plan.trivial:
            self._dispatch_trivial(page, stats)

        for page in plan.edit:
            self._dispatch_edit(
                page,
                old_node_snapshot=old_node_snapshot,
                new_by_id=new_by_id,
                stats=stats,
            )

        for page in plan.structural:
            self._dispatch_structural(page, stats)

        logger.info(
            "[incremental_regen_service] run complete: %s",
            stats.as_dict(),
        )
        return stats

    # ------------------------------------------------------------------
    # dispatch helpers
    # ------------------------------------------------------------------

    def _dispatch_trivial(
        self, page: PageRegime, stats: IncrementalRegenStats,
    ) -> None:
        body = self._read_page(page.page_id)
        if body is None:
            logger.warning(
                "[incremental_regen_service] trivial page %s body missing — "
                "demoting to structural",
                page.page_id,
            )
            self._dispatch_structural(page, stats)
            return

        patch = patch_trivial_page(page.page_id, body, page.changes)
        if patch.paths_rewritten:
            self._write_page(page.page_id, patch.new_content)
        # Update the page row's content_hash so future change-detection
        # comparisons see the new body. Title / cluster / etc unchanged.
        self._update_page_content_hash(page.page_id, patch.new_content_hash)
        stats.trivial_patched += 1

    def _dispatch_edit(
        self,
        page: PageRegime,
        *,
        old_node_snapshot: Mapping[str, dict[str, Any]],
        new_by_id: Mapping[str, dict[str, Any]],
        stats: IncrementalRegenStats,
    ) -> None:
        body = self._read_page(page.page_id)
        if body is None:
            logger.warning(
                "[incremental_regen_service] edit page %s body missing — "
                "demoting to structural",
                page.page_id,
            )
            self._dispatch_structural(page, stats)
            return

        symbol_diffs, moved_paths = _build_diff_inputs(
            page.changes, old_node_snapshot, new_by_id,
        )

        page_row = self._storage.get_wiki_page(page.page_id) or {}
        result = self._patcher.patch_page(
            page_id=page.page_id,
            page_title=page_row.get("title", ""),
            primary_symbol_id=page_row.get("primary_symbol_id"),
            current_content=body,
            symbol_diffs=symbol_diffs,
            moved_paths=moved_paths,
        )
        stats.diff_ratios.append(result.diff_ratio)

        if result.accepted:
            self._write_page(page.page_id, result.new_content)
            self._update_page_content_hash(page.page_id, result.new_content_hash)
            stats.edit_applied += 1
            return

        # Quality gate rejected — fall back to structural for this page.
        logger.info(
            "[incremental_regen_service] page %s demoted to structural: %s",
            page.page_id, result.reason,
        )
        stats.edit_demoted_to_structural += 1
        self._dispatch_structural(page, stats)

    def _dispatch_structural(
        self, page: PageRegime, stats: IncrementalRegenStats,
    ) -> None:
        try:
            ok = self._structural_handler(page)
        except Exception as exc:  # noqa: BLE001 — never crash the whole run
            logger.warning(
                "[incremental_regen_service] structural handler raised on "
                "page %s: %s",
                page.page_id, exc,
            )
            stats.structural_failed += 1
            return

        if ok:
            stats.structural_regenerated += 1
        else:
            stats.structural_failed += 1

    # ------------------------------------------------------------------
    # internals
    # ------------------------------------------------------------------

    def _snapshot_old_state(
        self, parsed_nodes: Iterable[Mapping[str, Any]],
    ) -> dict[str, dict[str, Any]]:
        """Read pre-upsert ``source_text`` for nodes that might end up in
        the edit regime. We snapshot every node in the parsed batch
        because the regime isn't decided yet — it's cheap and avoids
        a second storage hit later.
        """
        ids = [n["node_id"] for n in parsed_nodes if n.get("node_id")]
        if not ids:
            return {}
        try:
            rows = self._storage.get_nodes_by_ids(ids)
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "[incremental_regen_service] could not snapshot old nodes; "
                "surgical edits will see empty 'before' content: %s",
                exc,
            )
            return {}
        return {row["node_id"]: dict(row) for row in rows if row.get("node_id")}

    def _update_page_content_hash(self, page_id: str, new_hash: str) -> None:
        """Update only ``content_hash`` on an existing ``wiki_pages`` row.

        Reuses ``upsert_wiki_page`` for simplicity — pulls the full row,
        overlays the new hash, writes back.
        """
        row = self._storage.get_wiki_page(page_id)
        if row is None:
            return
        row["content_hash"] = new_hash
        self._storage.upsert_wiki_page(row)


# ---------------------------------------------------------------------------
# Diff input builder — module-scope for testability
# ---------------------------------------------------------------------------


def _build_diff_inputs(
    changes: Iterable[NodeChange],
    old_node_snapshot: Mapping[str, dict[str, Any]],
    new_by_id: Mapping[str, dict[str, Any]],
) -> tuple[list[SymbolDiff], dict[str, str]]:
    """Turn the change set into the patcher's ``symbol_diffs`` +
    ``moved_paths`` inputs.

    Empty old/new source is tolerated — the patcher's prompt handles
    "(none)" gracefully. Move-only changes contribute paths but no
    SymbolDiff entries (their source is unchanged).
    """
    symbol_diffs: list[SymbolDiff] = []
    moved_paths: dict[str, str] = {}

    for change in changes:
        if change.kind is ChangeKind.MOVED:
            if change.old_path and change.new_path:
                moved_paths[change.old_path] = change.new_path
            continue
        if change.kind is not ChangeKind.MODIFIED:
            # ADDED / DELETED shouldn't reach here — the classifier
            # routes those to structural — but defensively skip.
            continue
        old_row = old_node_snapshot.get(change.node_id, {})
        new_row = new_by_id.get(change.node_id, {})
        symbol_diffs.append(
            SymbolDiff(
                symbol_name=old_row.get("symbol_name")
                or new_row.get("symbol_name")
                or change.node_id,
                node_id=change.node_id,
                old_source=old_row.get("source_text", ""),
                new_source=new_row.get("source_text", ""),
                signature_change=_signature_diff(old_row, new_row),
            )
        )

    return symbol_diffs, moved_paths


def _signature_diff(old: Mapping[str, Any], new: Mapping[str, Any]) -> str | None:
    """Short human-readable note about signature shape changes, or None."""
    fragments = []
    for field_name in ("signature", "parameters", "return_type"):
        o = (old.get(field_name) or "").strip()
        n = (new.get(field_name) or "").strip()
        if o != n:
            fragments.append(f"{field_name}: {o!r} → {n!r}")
    if not fragments:
        return None
    return "; ".join(fragments)
