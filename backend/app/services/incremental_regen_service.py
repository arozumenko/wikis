"""Incremental-regen orchestrator for #116 PR 3.

This service ties together the four PR 1–4 building blocks:

* PR 1 — ``wiki_pages`` + ``page_symbols`` (the reverse index).
* PR 2 — :class:`ChangeDetector` (what changed, which pages it touches).
* PR 3 — regime classifier + trivial patcher + surgical :class:`PagePatcher`.
* PR 4 — cluster ID stability + embedding reuse + ``apply_incremental_node_writes``.

Dispatch flow:

1. Run :class:`ChangeDetector` against the pre-upsert storage state.
2. Classify with :func:`classify_pages` into a :class:`RegimePlan`.
3. Snapshot pre-upsert ``source_text`` for the nodes referenced by
   edit-regime pages only — keeps the lookup small enough that it
   stays under SQLite's ``IN`` clause bind-variable limit on large
   repos.
4. Apply the parsed-node batch via ``apply_incremental_node_writes``
   (single call → upsert + FTS refresh bundled per issue #131). After
   this the storage handle reflects the new repository state.
5. Per regime:
   - ``unchanged`` → no-op.
   - ``trivial`` → :func:`patch_trivial_page`, persist body, update
     ``wiki_pages.content_hash``.
   - ``edit`` → pre-rewrite moved paths via :func:`patch_trivial_page`
     so the LLM doesn't have to (regex job, not prose-edit), then run
     :class:`PagePatcher` on the resulting body. Demote to
     ``structural`` if the quality gate rejects.
   - ``structural`` → delegate to the injected ``structural_handler``
     callback (production wires this to the existing wiki-generation
     agent; tests can stub it).
6. Return :class:`IncrementalRegenStats` for telemetry (PR 5 surfaces
   this as SSE events + UI banner).

The orchestrator owns no LLM and no artifact-storage handle directly —
callers inject :class:`PagePatcher` (which owns the LLM) and the
page-content read/write callbacks. Keeps PR 3 unit-testable without
spinning up the full :class:`OptimizedWikiGenerationAgent`.

The snapshot step (3) lives between classify and upsert because:
- After upsert, storage holds new state, so the surgical patcher's
  "before" content would be lost.
- Before classify, we don't know which edit-regime pages need
  snapshotting, so scoping to that set requires the plan first.
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
from app.services.trivial_patcher import patch_trivial_page  # noqa: F401 — used in _dispatch_edit

logger = logging.getLogger(__name__)


# Callbacks the orchestrator depends on. Kept as type aliases at module
# scope so tests can pass stub functions inline.

#: Returns the current markdown body for a page_id, or ``None`` if missing.
PageBodyReader = Callable[[str], str | None]

#: Writes the new markdown body for a page_id. Production = artifact storage.
PageBodyWriter = Callable[[str, str], None]

#: Handles structural-regime regen for one page. Returns ``None`` on
#: success; a non-empty string is the failure reason that flows into
#: ``IncrementalRegenStats.structural_failure_reasons`` for telemetry.
#: PR 5 callers had this as ``bool``; the richer return type was added
#: by #134 so PR 5+ SPA banners can show *why* a structural run failed,
#: not just *that* one did.
StructuralHandler = Callable[[PageRegime], "str | None"]

#: Per-event progress callback. Signature: ``(event_name, payload)`` →
#: ``None``. Receives one call per dispatched page (``"page_unchanged"`` /
#: ``"page_patched"`` / ``"page_edited"`` / ``"page_regenerated"`` /
#: ``"page_deleted"``) plus one call with ``"incremental_summary"`` at
#: the end. ``payload`` is a small dict the SSE layer can plumb directly
#: into its event builders. PR 5 wires this to ``app.events`` builders
#: + an ``Invocation.emit``. Optional — orchestrator runs fine without
#: one (tests don't need it).
ProgressCallback = Callable[[str, dict[str, Any]], None]


__all__ = [
    "IncrementalRegenService",
    "IncrementalRegenStats",
    "PageBodyReader",
    "PageBodyWriter",
    "ProgressCallback",
    "StructuralHandler",
]


@dataclass
class IncrementalRegenStats:
    """Per-run summary surfaced to operators + PR 5 telemetry.

    PR 5's SSE banner reads ``total_pages`` to render "N pages
    processed" and the per-regime counts for the breakdown. The
    ``avg_diff_ratio`` is for tuning :data:`DEFAULT_DIFF_THRESHOLD`
    empirically once telemetry is in production.

    #134 added ``structural_failure_reasons`` — populated by the
    :data:`StructuralHandler` callback's optional error string. Lets
    PR 5+ operators see *why* a structural run failed (LLM error,
    storage hiccup, missing PageSpec, etc.) instead of just a count.
    """

    total_pages: int = 0
    unchanged: int = 0
    trivial_patched: int = 0
    edit_applied: int = 0
    edit_demoted_to_structural: int = 0
    structural_regenerated: int = 0
    structural_failed: int = 0
    deleted: int = 0  # #141 — pages whose cluster vanished entirely
    deleted_failed: int = 0  # #141 — DELETE primitive raised; row may persist
    diff_ratios: list[float] = field(default_factory=list)
    structural_failure_reasons: list[str] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        """Telemetry-friendly snapshot."""
        return {
            "total_pages": self.total_pages,
            "unchanged": self.unchanged,
            "trivial_patched": self.trivial_patched,
            "edit_applied": self.edit_applied,
            "edit_demoted_to_structural": self.edit_demoted_to_structural,
            "structural_regenerated": self.structural_regenerated,
            "structural_failed": self.structural_failed,
            "deleted": self.deleted,
            "deleted_failed": self.deleted_failed,
            "structural_failure_reasons": list(self.structural_failure_reasons),
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
        progress_callback: ProgressCallback | None = None,
    ) -> None:
        self._storage = storage
        self._patcher = page_patcher
        self._read_page = read_page_body
        self._write_page = write_page_body
        self._structural_handler = structural_handler
        # PR 5: optional event sink. None when called from unit tests
        # or callers that don't need SSE; production wires it to the
        # invocation's emit() via `app.events` builders.
        self._progress_callback = progress_callback
        # Title cache for SSE events. Lazy-populated on first lookup per
        # page_id so we don't pay an N+1 storage round-trip when emitting
        # one event per page. ``None`` value means "tried and missing".
        self._title_cache: dict[str, str | None] = {}

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
        # Step 1: detect changes (storage still holds the pre-upsert state).
        detector = ChangeDetector(self._storage)
        change_set = detector.detect_changes(parsed_nodes)
        affected = detector.affected_pages(change_set)

        # Step 2: classify into regime buckets. classify_pages reads
        # wiki_pages.primary_symbol_id for the deleted-primary override;
        # still the pre-upsert state at this point.
        plan = classify_pages(change_set, affected, self._storage)

        # Step 3: snapshot pre-upsert source_text *only* for nodes
        # referenced by edit-regime pages. Scoping to that set keeps
        # the lookup well below SQLite's IN-clause bind-variable
        # limit (default 999) even on large repos.
        edit_node_ids = self._collect_edit_node_ids(plan)
        old_node_snapshot = self._snapshot_old_state(edit_node_ids)

        # Step 4: persist the new node state — single bundled call that
        # also refreshes FTS (#131). After this the storage handle
        # reflects the new repository state.
        self._storage.apply_incremental_node_writes(parsed_nodes)

        # Build a lookup of new parsed-node state for the dispatcher.
        new_by_id: dict[str, dict[str, Any]] = {
            n["node_id"]: n for n in parsed_nodes if n.get("node_id")
        }

        stats = IncrementalRegenStats(total_pages=plan.total)
        stats.unchanged = len(plan.unchanged)

        # Step 5: per-regime dispatch. Emit a per-page event for every
        # page touched — including the "unchanged" bucket so the SPA
        # can render a complete picture without computing deltas.
        for page in plan.unchanged:
            self._emit_page_event("page_unchanged", page)

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
            self._dispatch_structural(
                page, stats, demoted_from_edit=False,
            )

        # #141: drop pages whose entire cluster vanished.
        for page in plan.deleted:
            self._dispatch_deleted(page, stats)

        # Step 6: summary event for the SPA banner.
        self._emit("incremental_summary", {"stats": stats.as_dict()})

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
            self._dispatch_structural(page, stats, demoted_from_edit=False)
            return

        patch = patch_trivial_page(page.page_id, body, page.changes)
        if patch.paths_rewritten:
            self._write_page(page.page_id, patch.new_content)
        # Update the page row's content_hash so future change-detection
        # comparisons see the new body. Title / cluster / etc unchanged.
        self._update_page_content_hash(page.page_id, patch.new_content_hash)
        stats.trivial_patched += 1
        self._emit_page_event(
            "page_patched", page,
            citation_count=patch.paths_rewritten,
        )

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

        # Pre-rewrite moved paths deterministically before invoking the
        # LLM. Path swaps are a mechanical regex job — outsourcing them
        # to the LLM wastes context, risks errors, and leaks
        # implementation details into the prompt. After this, the LLM
        # sees a clean body and focuses on prose for MODIFIED symbols.
        trivial = patch_trivial_page(page.page_id, body, page.changes)
        body = trivial.new_content

        symbol_diffs, _moved_paths = _build_diff_inputs(
            page.changes, old_node_snapshot, new_by_id,
        )

        page_row = self._storage.get_wiki_page(page.page_id) or {}
        result = self._patcher.patch_page(
            page_id=page.page_id,
            page_title=page_row.get("title", ""),
            primary_symbol_id=page_row.get("primary_symbol_id"),
            current_content=body,
            symbol_diffs=symbol_diffs,
            # Paths already rewritten by trivial_patcher above —
            # don't ask the LLM to do it again.
            moved_paths={},
        )
        stats.diff_ratios.append(result.diff_ratio)

        if result.accepted:
            self._write_page(page.page_id, result.new_content)
            self._update_page_content_hash(page.page_id, result.new_content_hash)
            stats.edit_applied += 1
            self._emit_page_event(
                "page_edited", page, diff_ratio=result.diff_ratio,
            )
            return

        # Quality gate rejected — fall back to structural for this page.
        logger.info(
            "[incremental_regen_service] page %s demoted to structural: %s",
            page.page_id, result.reason,
        )
        stats.edit_demoted_to_structural += 1
        self._dispatch_structural(page, stats, demoted_from_edit=True)

    def _dispatch_deleted(
        self,
        page: PageRegime,
        stats: IncrementalRegenStats,
    ) -> None:
        """#141: drop a page whose entire cluster vanished.

        Calls ``storage.delete_wiki_page(page_id)`` which removes the
        ``wiki_pages`` row (FK CASCADE removes ``page_symbols`` in the
        same transaction), then blanks the artifact body via the
        page-body writer so the SPA stops rendering stale prose if it
        cached the markdown.

        Both steps are best-effort: a writer that raises records the
        failure in ``stats.deleted_failed`` + ``structural_failure_reasons``
        and does **not** increment ``stats.deleted`` (the contract that
        ``stats.deleted == N`` means N rows are gone matters for the
        SPA's accounting). The page row deletion is the load-bearing
        step; the body wipe is cosmetic backup.
        """
        try:
            self._storage.delete_wiki_page(page.page_id)
        except Exception as exc:  # noqa: BLE001
            stats.deleted_failed += 1
            stats.structural_failure_reasons.append(
                f"{page.page_id}: delete failed: {exc}",
            )
            logger.warning(
                "[incremental_regen_service] delete_wiki_page failed "
                "for %s: %s",
                page.page_id, exc,
            )
            return

        try:
            self._write_page(page.page_id, "")
        except Exception as exc:  # noqa: BLE001
            # Row is gone but artifact wipe failed. Treat as success for
            # the page_deleted contract (the canonical "this page no
            # longer exists" signal is the missing wiki_pages row), but
            # log loud — operators may see an orphan artifact in storage.
            logger.warning(
                "[incremental_regen_service] deleted-page body wipe "
                "failed for %s (row already deleted): %s",
                page.page_id, exc,
            )

        stats.deleted += 1
        self._emit_page_event("page_deleted", page)

    def _dispatch_structural(
        self,
        page: PageRegime,
        stats: IncrementalRegenStats,
        *,
        demoted_from_edit: bool,
    ) -> None:
        try:
            result = self._structural_handler(page)
        except Exception as exc:  # noqa: BLE001 — never crash the whole run
            logger.warning(
                "[incremental_regen_service] structural handler raised on "
                "page %s: %s",
                page.page_id, exc,
            )
            stats.structural_failed += 1
            stats.structural_failure_reasons.append(
                f"{page.page_id}: {exc}",
            )
            return

        # #134: contract is Optional[str] — None=success, str=failure reason.
        # Backward-compat: older bool-returning callbacks still work
        # (False → "" reason → counted as failed; True → None ish).
        if result is None or result is True:
            stats.structural_regenerated += 1
            self._emit_page_event(
                "page_regenerated", page,
                demoted_from_edit=demoted_from_edit,
            )
            return

        stats.structural_failed += 1
        reason = result if isinstance(result, str) and result else "unspecified"
        stats.structural_failure_reasons.append(
            f"{page.page_id}: {reason}",
        )

    # ------------------------------------------------------------------
    # internals
    # ------------------------------------------------------------------

    def _collect_edit_node_ids(self, plan: RegimePlan) -> list[str]:
        """Gather every MODIFIED node_id referenced by edit-regime pages.

        Bounded by the change set rather than the full parsed batch.
        Even on large repos a single regen typically touches a handful
        of pages with a handful of modified symbols each, so this stays
        well within SQLite's bind-variable limit.
        """
        seen: set[str] = set()
        for page in plan.edit:
            for change in page.changes:
                if change.kind is ChangeKind.MODIFIED and change.node_id:
                    seen.add(change.node_id)
        return sorted(seen)

    def _snapshot_old_state(
        self, node_ids: list[str],
    ) -> dict[str, dict[str, Any]]:
        """Read pre-upsert ``source_text`` for the given node_ids.

        Caller is responsible for scoping the input to the set that
        actually needs snapshotting (edit-regime nodes only) so the
        underlying ``get_nodes_by_ids`` IN-clause stays under the
        SQLite bind-variable limit.

        Fails soft: on storage error returns an empty dict and warns,
        so the surgical patcher proceeds with empty "before" content
        rather than crashing the entire regen.
        """
        if not node_ids:
            return {}
        try:
            rows = self._storage.get_nodes_by_ids(node_ids)
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "[incremental_regen_service] could not snapshot old nodes; "
                "surgical edits will see empty 'before' content: %s",
                exc,
            )
            return {}
        return {row["node_id"]: dict(row) for row in rows if row.get("node_id")}

    def _title_for(self, page_id: str) -> str:
        """Look up + cache a page title for SSE telemetry. Returns the
        page_id itself when no title is known (legacy row, deleted page).

        First lookup hits the storage; subsequent lookups for the same
        page_id come from ``self._title_cache``. PR 5 reviewers flagged
        this as an N+1 hot path when many pages are dispatched in one
        run — the cache turns it into O(N) titles, one per unique page.
        """
        if page_id in self._title_cache:
            return self._title_cache[page_id] or page_id
        try:
            row = self._storage.get_wiki_page(page_id)
            title = (row or {}).get("title") or None
        except Exception as exc:  # noqa: BLE001
            logger.debug(
                "[incremental_regen_service] title lookup failed for %s: %s",
                page_id, exc,
            )
            title = None
        self._title_cache[page_id] = title
        return title or page_id

    def _emit_page_event(
        self,
        event_name: str,
        page: PageRegime,
        **extra: Any,
    ) -> None:
        """Emit a per-page event via the configured callback.

        Fail-soft: a raising callback gets logged at DEBUG and the
        dispatch path continues. Telemetry is observability, not
        load-bearing for correctness.
        """
        if self._progress_callback is None:
            return
        payload: dict[str, Any] = {
            "page_id": page.page_id,
            "page_title": self._title_for(page.page_id),
        }
        payload.update(extra)
        try:
            self._progress_callback(event_name, payload)
        except Exception as exc:  # noqa: BLE001
            logger.debug(
                "[incremental_regen_service] progress_callback raised on %s: %s",
                event_name, exc,
            )

    def _emit(self, event_name: str, payload: dict[str, Any]) -> None:
        """Forward a non-page event (summary) to the configured callback."""
        if self._progress_callback is None:
            return
        try:
            self._progress_callback(event_name, payload)
        except Exception as exc:  # noqa: BLE001
            logger.debug(
                "[incremental_regen_service] progress_callback raised on %s: %s",
                event_name, exc,
            )

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


_SIGNATURE_FIELD_CAP = 128


def _signature_diff(old: Mapping[str, Any], new: Mapping[str, Any]) -> str | None:
    """Short human-readable note about signature shape changes, or None.

    Each captured field is truncated to ``_SIGNATURE_FIELD_CAP`` chars
    before ``repr()`` so a function with a weirdly large default value
    (e.g. a giant string literal) can't produce a malformed prompt
    block that confuses the LLM.
    """
    fragments = []
    for field_name in ("signature", "parameters", "return_type"):
        o = (old.get(field_name) or "").strip()[:_SIGNATURE_FIELD_CAP]
        n = (new.get(field_name) or "").strip()[:_SIGNATURE_FIELD_CAP]
        if o != n:
            fragments.append(f"{field_name}: {o!r} → {n!r}")
    if not fragments:
        return None
    return "; ".join(fragments)
