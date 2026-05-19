"""Wiki generation service — orchestrates background wiki generation."""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
from datetime import datetime
from typing import TYPE_CHECKING, Any
from uuid import uuid4

import app.events as events
from app.config import Settings
from app.models.invocation import Invocation
from app.storage.base import ArtifactStorage

if TYPE_CHECKING:
    from app.models.api import GenerateWikiRequest
    from app.services.wiki_management import WikiManagementService

logger = logging.getLogger(__name__)


# Statuses that mean "an operation is actively writing to this wiki's
# .wiki.db right now" — both ``generate`` and ``incremental_refresh``
# treat encountering one of these for the same ``wiki_id`` as a
# conflict and reject the new request with a 409. ``"running"`` is the
# incremental refresh's in-flight status; ``"generating"`` is the full
# regen's.
_IN_FLIGHT_STATUSES: frozenset[str] = frozenset({"running", "generating"})


class IncrementalRefreshInProgressError(Exception):
    """Raised by :meth:`WikiService.incremental_refresh` when another
    incremental refresh OR full generate is already running for the
    same wiki_id.

    #140 idempotency guard: two concurrent runs against the same
    ``.wiki.db`` race each other's content-hash updates and may produce
    inconsistent state. The route translates this into a 409.
    """

    def __init__(self, wiki_id: str, in_progress_invocation_id: str) -> None:
        self.wiki_id = wiki_id
        self.in_progress_invocation_id = in_progress_invocation_id
        super().__init__(
            f"Incremental refresh already in progress for wiki {wiki_id} "
            f"(invocation {in_progress_invocation_id})"
        )


class GenerateInProgressError(Exception):
    """Raised by :meth:`WikiService.generate` when another generate or
    incremental refresh is already running for the same wiki_id.

    #145 symmetric guard to #140's incremental-side rejection. Both
    code paths write to the same ``.wiki.db``; concurrent runs race
    each other's writes. The route translates this into a 409.
    """

    def __init__(self, wiki_id: str, in_progress_invocation_id: str) -> None:
        self.wiki_id = wiki_id
        self.in_progress_invocation_id = in_progress_invocation_id
        super().__init__(
            f"Another operation is already running for wiki {wiki_id} "
            f"(invocation {in_progress_invocation_id})"
        )


class WikiService:
    """Orchestrates wiki generation from repository analysis."""

    # #242: deprecation warning sentinel — flipped True after the first
    # request that still supplies ``planner_type``, suppressing the warning
    # for the remainder of the process to avoid log spam from older clients.
    _planner_type_deprecation_logged: bool = False

    def __init__(self, settings: Settings, storage: ArtifactStorage, wiki_management: Any = None) -> None:
        self.settings = settings
        self.storage = storage
        self.wiki_management = wiki_management
        self._invocations: dict[str, Invocation] = {}
        self._tasks: dict[str, asyncio.Task] = {}
        # #165 follow-up: set by ``shutdown()`` so concurrent
        # ``cancel_invocation`` calls during the drain window can't
        # cancel a task we're trying to await. Cancellation marks the
        # asyncio.Task done while the worker thread keeps running —
        # the exact race that caused the CI segfault.
        self._shutting_down: bool = False

    @property
    def invocations(self) -> dict[str, Invocation]:
        """Public read-only access to the invocations dict."""
        return self._invocations

    def remove_invocation(self, inv_id: str) -> None:
        """Remove an invocation and its task from tracking."""
        self._invocations.pop(inv_id, None)
        self._tasks.pop(inv_id, None)

    def _find_in_flight_for_wiki(
        self,
        wiki_id: str,
        blocked_statuses: frozenset[str] = _IN_FLIGHT_STATUSES,
    ) -> "Invocation | None":
        """Return the first in-memory invocation for ``wiki_id`` whose
        status is in ``blocked_statuses``, or ``None`` if no conflict.

        Shared between :meth:`generate` and :meth:`incremental_refresh`
        so both endpoints enforce the same mutual-exclusion contract
        (#140 + #145). The check is synchronous — callers MUST invoke
        it before any ``await`` that could yield to a racing caller.
        """
        for inv in self._invocations.values():
            # ``inv.id != ""`` guards against a defensive edge case:
            # ``Invocation`` is a Pydantic model with ``id: str`` (no
            # ``min_length`` constraint), so a malformed persisted
            # payload could in theory deserialize with ``id=""``. We
            # refuse to consider those as in-flight blockers — they'd
            # always 409-lock the wiki without a usable id for the
            # caller's response header.
            if (
                inv.wiki_id == wiki_id
                and inv.status in blocked_statuses
                and inv.id != ""
            ):
                return inv
        return None

    async def persist_invocations(self) -> None:
        """Public wrapper — save all invocations to storage."""
        await self._persist_invocations()

    async def shutdown(self, timeout: float = 30.0) -> None:
        """Drain in-flight invocation tasks before the app shuts down.

        Each entry in ``self._tasks`` is an ``asyncio.Task`` wrapping a
        coroutine that internally calls ``asyncio.to_thread(...)`` to
        offload CPU-heavy work (SQLite writes, FTS rebuilds) onto a
        worker thread. If the event loop closes while one of those
        tasks is still awaiting its thread, the asyncio side surfaces
        a ``CancelledError`` **but the underlying OS thread keeps
        running** — Python can't interrupt thread execution. That
        thread continues touching the SQLite connection while the
        fixture / lifespan tears the storage down underneath it,
        which the SQLite C library handles by segfaulting.

        We use ``asyncio.wait`` (not ``asyncio.wait_for(gather(...))``)
        because the latter cancels its children on timeout — and that
        cancellation is the exact failure mode we're guarding against
        (the asyncio side returns but the worker thread keeps running
        on shared state). ``asyncio.wait`` returns ``(done, pending)``
        without cancelling so we can log + bail without re-introducing
        the bug.

        If the timeout fires we explicitly do **not** cancel the still-
        pending tasks. Their threads need the storage / engine alive
        for as long as they're still running; cancelling here would
        only mask the leak, not fix it. The orphan threads will at
        least see live state for the rest of the shutdown sequence.
        """
        # Mark the service as shutting down before snapshotting the
        # pending set so a concurrent ``cancel_invocation`` call can't
        # cancel a task we're about to await. Cancellation races
        # shutdown — see the docstring above.
        self._shutting_down = True
        pending = [t for t in self._tasks.values() if not t.done()]
        if not pending:
            return
        _, still_pending = await asyncio.wait(pending, timeout=timeout)
        if still_pending:
            logger.error(
                "WikiService.shutdown timeout: %d task(s) still in-flight "
                "after %.0fs — proceeding with shutdown anyway (worker "
                "threads will continue against live state)",
                len(still_pending),
                timeout,
            )

    INVOCATIONS_BUCKET = "wiki_registry"
    INVOCATIONS_KEY = "invocations.json"

    async def load_persisted_invocations(self) -> None:
        """Load invocations from storage on startup."""
        try:
            import json as _json

            data = await self.storage.download(self.INVOCATIONS_BUCKET, self.INVOCATIONS_KEY)
            raw = _json.loads(data)
            orphaned_wiki_ids: list[tuple[str, str]] = []
            for inv_id, inv_data in raw.items():
                try:
                    inv = Invocation(**inv_data)
                    # #146: any non-terminal status from a previous
                    # process lifetime is by definition orphaned —
                    # the task that was driving it is gone. Without
                    # flipping these to "failed" the idempotency
                    # guards (PR #144) see phantom in-flight entries
                    # and 409-lock the wiki on the next request.
                    # ``"generating"`` = full regen, ``"running"`` =
                    # incremental refresh.
                    if inv.status in ("generating", "running"):
                        prior_status = inv.status
                        inv.status = "failed"
                        inv.error = (
                            f"Server restarted during {prior_status}"
                        )
                        inv.completed_at = datetime.now()
                        # #177 (inverts #191 C5): now that incremental_refresh
                        # writes status="running" to the DB at start, an
                        # orphaned ``running`` row means the refresh crashed
                        # mid-flight — it MUST be transitioned to ``failed``
                        # so the user sees "refresh crashed, please retry"
                        # rather than the stale ``complete`` content silently
                        # standing. Previously we skipped the WikiRecord
                        # update for ``running`` to avoid invalidating
                        # previously-good content; with the DB write in place
                        # that reasoning inverts: the record already shows
                        # ``running``, so leaving it there is worse than
                        # flipping it to ``failed``.
                        if inv.wiki_id:
                            orphaned_wiki_ids.append((inv.wiki_id, inv.error))
                    self._invocations[inv_id] = inv
                except Exception:  # noqa: S110
                    pass
            # #191: propagate orphaned-invocation failures to the persistent
            # WikiRecord so the dashboard card matches reality. Without this
            # the record stays at status="generating", progress=1.0 and the
            # SPA shows "100% Generating" indefinitely.
            if orphaned_wiki_ids and self.wiki_management:
                for wiki_id, err in orphaned_wiki_ids:
                    try:
                        await self.wiki_management.mark_status(
                            wiki_id=wiki_id, status="failed", error=err
                        )
                    except Exception as _e:  # noqa: BLE001
                        logger.warning(
                            "Failed to reconcile WikiRecord for orphaned %s: %s",
                            wiki_id,
                            _e,
                        )
            logger.info(f"Loaded {len(self._invocations)} persisted invocations")
        except FileNotFoundError:
            pass
        except Exception as e:
            logger.warning(f"Failed to load persisted invocations: {e}")

    async def _persist_invocations(self) -> None:
        """Save all invocations to storage."""
        try:
            import json as _json

            self._cleanup_old_invocations(ttl_seconds=86400)
            data = {}
            for inv_id, inv in self._invocations.items():
                d = inv.model_dump()
                for key in ("created_at", "completed_at"):
                    if d.get(key) and hasattr(d[key], "isoformat"):
                        d[key] = d[key].isoformat()
                data[inv_id] = d
            raw = _json.dumps(data, default=str).encode("utf-8")
            await self.storage.upload(self.INVOCATIONS_BUCKET, self.INVOCATIONS_KEY, raw)
        except Exception as e:
            logger.warning(f"Failed to persist invocations: {e}")

    @staticmethod
    def _canonicalize(obj: Any) -> Any:
        """Recursively sort dict keys and homogeneous list values for stable hashing.

        Ensures that ``space_keys=["A","B"]`` and ``space_keys=["B","A"]`` produce
        identical JSON so the hash-based wiki ID is order-independent.
        """
        if isinstance(obj, dict):
            return {k: WikiService._canonicalize(v) for k, v in sorted(obj.items())}
        if isinstance(obj, list):
            # Sort only homogeneous lists of primitives; leave mixed lists order-preserved.
            if all(isinstance(x, (str, int, float, bool)) for x in obj):
                return sorted(obj, key=str)
            return [WikiService._canonicalize(x) for x in obj]
        return obj

    @staticmethod
    def _make_wiki_id(source_type: str, scope: dict[str, Any]) -> str:  # type: ignore[override]
        """Deterministic wiki ID from source_type + scope.

        Stable across runs: scope keys are sorted before hashing so insertion
        order doesn't affect the result.  List values (e.g. ``space_keys``) are
        also sorted so ``["A","B"]`` and ``["B","A"]`` yield the same ID.

        Backwards-compat legacy ID
        --------------------------
        Prior to #189, git wikis were identified by
        ``sha256(f"{repo_url}:{branch}")[:16]``.  If the new hash misses the
        DB but the old-style hash hits, the caller should fall back to the
        old hash.  See :meth:`generate` for the one-cycle migration helper.
        """
        canonical = json.dumps(
            WikiService._canonicalize({"source_type": source_type, "scope": scope}),
            sort_keys=True,
        )
        return hashlib.sha256(canonical.encode()).hexdigest()[:16]

    @staticmethod
    def _make_legacy_git_wiki_id(repo_url: str, branch: str) -> str:
        """Pre-#189 git wiki ID — used as a fallback during DB migration.

        One-cycle migration helper: if the new-style ID isn't in the DB but
        the old-style ID is, we reuse the old ID so the existing wiki record
        and artifacts are preserved.  Remove this in a future cleanup PR once
        all rows have been migrated.
        """
        key = f"{repo_url}:{branch}"
        return hashlib.sha256(key.encode()).hexdigest()[:16]

    async def generate(self, request: GenerateWikiRequest, owner_id: str = "", force: bool = False) -> Invocation:
        """Start wiki generation in background, return invocation."""
        from pathlib import Path

        from app.core.local_repo_provider import extract_git_metadata, is_local_path, make_local_wiki_id

        # --- Determine wiki_id from source_type + scope --------------------
        if request.source_type == "git" and request.repo_url and is_local_path(request.repo_url):
            path = request.repo_url.removeprefix("file://")
            info = extract_git_metadata(Path(path).resolve())
            wiki_id = make_local_wiki_id(path, info.branch if info.is_git else None)
        elif request.source_type == "git" and request.scope:
            # New-style hash
            wiki_id = self._make_wiki_id(request.source_type, request.scope)
            # One-cycle migration: fall back to legacy ID if DB already has it.
            if self.wiki_management:
                legacy_id = self._make_legacy_git_wiki_id(
                    request.scope.get("repo_url", request.repo_url or ""),
                    request.scope.get("branch", request.branch or "main"),
                )
                if legacy_id != wiki_id:
                    existing_legacy = await self.wiki_management.get_wiki_record(legacy_id)
                    if existing_legacy:
                        logger.info(
                            "Reusing legacy wiki_id %s (new-style id would be %s)",
                            legacy_id, wiki_id,
                        )
                        wiki_id = legacy_id
        else:
            # Non-git sources or legacy callers that didn't set scope.
            repo_url_for_id = request.repo_url or ""
            branch_for_id = request.branch or "main"
            if request.scope:
                wiki_id = self._make_wiki_id(request.source_type, request.scope)
            else:
                wiki_id = self._make_wiki_id(
                    request.source_type,
                    {"repo_url": repo_url_for_id, "branch": branch_for_id},
                )

        # #145: symmetric in-flight check. ``incremental_refresh`` already
        # rejects when a generate is mid-flight (PR #144); this is the
        # reverse direction. Both endpoints write to the same ``.wiki.db``
        # — without this check, ``incremental_refresh running + generate
        # called`` could race content_hash updates and corrupt FTS5/
        # tsvector indices.
        #
        # Synchronous check before any ``await`` so two callers racing
        # through this function can't both pass.
        in_flight = self._find_in_flight_for_wiki(wiki_id)
        if in_flight is not None:
            raise GenerateInProgressError(wiki_id, in_flight.id)

        # Block duplicate generation — reject if a wiki is already complete or generating
        # Use raw DB lookup (no access control) to detect ANY user's wiki for this repo+branch
        # Skipped when force=True (called from refresh)
        if not force and self.wiki_management:
            existing = await self.wiki_management.get_wiki_record(wiki_id)
            if existing and existing.status in ("complete", "generating"):
                from app.services.wiki_service_errors import WikiAlreadyExistsError

                raise WikiAlreadyExistsError(wiki_id)

        invocation_id = str(uuid4())
        invocation = Invocation(
            id=invocation_id,
            wiki_id=wiki_id,
            repo_url=request.repo_url,
            branch=request.branch,
            owner_id=owner_id,
            status="generating",
            current_phase="initializing",
            message=f"Starting wiki generation for {request.repo_url}",
        )
        self._invocations[invocation_id] = invocation
        await self._persist_invocations()

        # Register wiki in DB immediately so failed/retried generations are trackable
        if self.wiki_management:
            try:
                _title = request.wiki_title or (
                    f"Wiki for {request.repo_url}"
                    if request.source_type == "git"
                    else f"Wiki from {request.source_type.capitalize()} ({request.scope.get('base_url', '')})"
                )
                await self.wiki_management.register_wiki(
                    wiki_id=wiki_id,
                    repo_url=request.repo_url or "",
                    branch=request.branch or "main",
                    title=_title,
                    page_count=0,
                    owner_id=owner_id,
                    visibility=getattr(request, "visibility", "personal"),
                    status="generating",
                    requires_token=bool(request.access_token),
                    source_type=request.source_type,
                    source_scope=request.scope or None,
                )
            except Exception as e:
                logger.warning(f"Failed to pre-register wiki {wiki_id}: {e}")

        task = asyncio.create_task(self._run_generation(invocation, request))
        self._tasks[invocation_id] = task
        return invocation

    async def _run_generation(self, invocation: Invocation, request: GenerateWikiRequest) -> None:
        """Background task: clone → index → generate → store artifacts.

        Heavy work (clone / parse / embed / LangGraph page generation) runs in
        an isolated Python subprocess via :meth:`_run_wiki_subprocess` so the
        API's event loop stays responsive on large repositories.  This
        function only orchestrates: spawn, stream progress events, then
        upload the resulting pages and artifacts.
        """
        try:
            from app.core.repo_providers.factory import RepoProviderFactory

            await self._emit_progress(invocation, "configuring", 0.05, "Preparing wiki generation")

            # The unified pipeline is the only path (#242).
            # planner_type from the request is a no-op; log a deprecation
            # warning ONCE per process (via a class-level sentinel) so older
            # clients in production don't spam the log.
            planner_type = request.planner_type if request.planner_type is not None else self.settings.planner_type
            if request.planner_type is not None and not WikiService._planner_type_deprecation_logged:
                logger.warning(
                    "DEPRECATED: planner_type=%r was supplied but is ignored — "
                    "the unified pipeline is now the only path (#242). This warning "
                    "is suppressed for the rest of this process.",
                    request.planner_type,
                )
                WikiService._planner_type_deprecation_logged = True
            exclude_tests = False

            await self._emit_progress(invocation, "indexing", 0.1, "Cloning repository and building index")

            result = await self._run_wiki_subprocess(
                invocation=invocation,
                request=request,
                planner_type=planner_type,
                exclude_tests=exclude_tests,
            )

            if result is None or not isinstance(result, dict) or not result.get("success"):
                if isinstance(result, dict):
                    error_msg = result.get("error", "Unknown generation error")
                else:
                    error_msg = "Generation returned no result" if result is None else "Generation returned non-dict"
                raise RuntimeError(error_msg)

            # Save pages incrementally as they're generated
            await self._emit_progress(invocation, "storing", 0.9, "Storing wiki pages")

            generated_pages = result.get("generated_pages", {})
            invocation.pages_total = len(generated_pages)
            invocation.pages_completed = 0

            for page_id, content in generated_pages.items():
                try:
                    data = content.encode("utf-8") if isinstance(content, str) else content
                    await self.storage.upload("wiki_artifacts", f"{invocation.wiki_id}/{page_id}.md", data)
                    invocation.pages_completed += 1
                    await invocation.emit(events.page_complete(invocation.id, page_id, page_id))
                except Exception as e:
                    logger.warning(f"Failed to save page {page_id}: {e}")

            # Index pages into PostgreSQL for full-text search
            if self.wiki_management and generated_pages:
                try:
                    await self.wiki_management.index_wiki_pages(
                        invocation.wiki_id, generated_pages,
                    )
                except Exception as e:
                    logger.warning("Failed to index pages for FTS: %s", e)

            # Store export artifacts (index, summary, etc.)
            artifacts = result.get("artifacts", [])
            for artifact in artifacts:
                name = artifact.get("name", "unknown")
                data = artifact.get("data", b"")
                if isinstance(data, str):
                    data = data.encode("utf-8")
                await self.storage.upload("wiki_artifacts", f"{invocation.wiki_id}/{name}", data)

            page_count = invocation.pages_completed
            if self.wiki_management:
                from app.core.local_repo_provider import is_local_path as _is_local

                # For local repos: store remote URL (if any) for source links, else local path
                registry_url = request.repo_url
                registry_branch = request.branch
                if _is_local(request.repo_url):
                    clone_cfg = RepoProviderFactory.from_url(url=request.repo_url, branch=request.branch)
                    from app.core.repo_providers import LocalPathConfig as _LPC

                    if isinstance(clone_cfg, _LPC) and clone_cfg.remote_url:
                        registry_url = clone_cfg.remote_url
                    registry_branch = clone_cfg.branch or request.branch
                _reg_title = request.wiki_title or (
                    f"Wiki for {request.repo_url}"
                    if request.source_type == "git"
                    else f"Wiki from {request.source_type.capitalize()} ({request.scope.get('base_url', '')})"
                )
                await self.wiki_management.register_wiki(
                    wiki_id=invocation.wiki_id,
                    repo_url=registry_url,
                    branch=registry_branch,
                    title=_reg_title,
                    page_count=page_count,
                    owner_id=invocation.owner_id,
                    visibility=getattr(request, "visibility", "personal"),
                    commit_hash=result.get("commit_hash"),
                    indexed_at=datetime.now(),
                    status="complete",
                    requires_token=bool(request.access_token),
                    source_type=request.source_type,
                    source_scope=request.scope or None,
                )

            # Mark every project containing this wiki as stale so PR-15 can
            # decide to enqueue a project recompute. Best-effort, never raises.
            try:
                await self._mark_owning_projects_stale(invocation.wiki_id)
            except Exception:  # noqa: BLE001
                logger.debug(
                    "mark_owning_projects_stale failed for %s",
                    invocation.wiki_id,
                    exc_info=True,
                )

            invocation.status = "complete"
            invocation.progress = 1.0
            invocation.current_phase = "done"
            invocation.message = "Wiki generation complete"
            invocation.completed_at = datetime.now()

            elapsed = (invocation.completed_at - invocation.created_at).total_seconds()
            await invocation.emit(
                events.task_status(
                    invocation.id,
                    "completed",
                    "Wiki generation complete",
                    wiki_id=invocation.wiki_id,
                    page_count=page_count,
                    execution_time=elapsed,
                )
            )
            logger.info(f"Wiki generation complete: {invocation.wiki_id}")

        except asyncio.CancelledError:
            invocation.status = "cancelled"
            invocation.message = "Generation cancelled"
            invocation.completed_at = datetime.now()
            await invocation.emit(events.task_status(invocation.id, "cancelled", "Generation cancelled"))
            logger.info(f"Wiki generation cancelled: {invocation.wiki_id}")
        except Exception as e:
            from app.services.context_overflow import ContextOverflowError, classify_and_wrap

            wrapped = classify_and_wrap(e)
            if isinstance(wrapped, ContextOverflowError):
                recoverable = True
                error_type = "context_overflow"
                limit_hint = f" (model limit: {wrapped.model_limit:,} tokens)" if wrapped.model_limit else ""
                error_msg = f"Context window exceeded{limit_hint}. The repository may be too large for the selected model. Try a model with a larger context window or reduce the repository scope."
                invocation.status = "partial" if invocation.pages_completed > 0 else "failed"
                invocation.error = error_msg
                invocation.message = error_msg
                invocation.completed_at = datetime.now()
                await self._emit_error(
                    invocation,
                    error_msg,
                    recoverable=recoverable,
                    error_type=error_type,
                    model_limit=wrapped.model_limit,
                    suggested_actions=wrapped.suggestions,
                )
                logger.warning(f"Wiki generation context overflow: {invocation.wiki_id} — {wrapped}")
            else:
                invocation.status = "partial" if invocation.pages_completed > 0 else "failed"
                invocation.error = str(e)
                invocation.message = f"Generation failed: {e}"
                invocation.completed_at = datetime.now()
                await self._emit_error(invocation, str(e), recoverable=False)
                logger.exception(f"Wiki generation failed: {invocation.wiki_id}")
        finally:
            self._tasks.pop(invocation.id, None)
            await self._persist_invocations()
            # #191: reconcile the persistent WikiRecord with the final
            # invocation status on *every* terminal outcome. The success
            # path above already calls register_wiki(status="complete"),
            # but if that call partially failed — or if an orphaned record
            # from a previous run is still pinned to "generating"/1.0 —
            # this defensive update guarantees the dashboard/detail view
            # reflects backend truth instead of stale "100% Generating".
            #
            # Copilot C1: mark_status is update-only. If pre-registration
            # silently failed (the catch at line ~397 swallows DB errors),
            # no WikiRecord exists and mark_status returns False. Without
            # the fallback below the failure state would never reach the
            # dashboard / retry flow.
            if self.wiki_management and invocation.wiki_id:
                try:
                    updated = await self.wiki_management.mark_status(
                        wiki_id=invocation.wiki_id,
                        status=invocation.status,
                        error=invocation.error,
                    )
                    if (
                        not updated
                        and invocation.status != "complete"
                        and invocation.repo_url
                    ):
                        await self.wiki_management.register_wiki(
                            wiki_id=invocation.wiki_id,
                            repo_url=invocation.repo_url,
                            branch=invocation.branch,
                            title=f"Wiki for {invocation.repo_url}",
                            page_count=invocation.pages_completed,
                            owner_id=invocation.owner_id,
                            status=invocation.status,
                            error=invocation.error,
                        )
                except Exception as _db_err:
                    logger.warning(f"Failed to reconcile wiki status in DB: {_db_err}")

    async def _mark_owning_projects_stale(self, wiki_id: str) -> None:
        """Find projects containing *wiki_id* and stamp them stale (PR-14).

        Best-effort: requires ``flags.project_graph``. PR-15 reads
        ``project_meta['stale']`` to decide whether to enqueue a recompute.
        """
        from sqlalchemy import select

        from app.core.feature_flags import get_feature_flags
        from app.db import get_session_factory
        from app.models.db_models import ProjectWikiRecord
        from app.services.project_recompute import mark_projects_for_wiki_stale

        if not get_feature_flags().project_graph:
            return

        session_factory = get_session_factory()
        async with session_factory() as session:
            result = await session.execute(
                select(ProjectWikiRecord.project_id).where(
                    ProjectWikiRecord.wiki_id == wiki_id
                )
            )
            project_ids = [row[0] for row in result.all() if row[0]]
        if not project_ids:
            return
        mark_projects_for_wiki_stale(
            wiki_id=wiki_id, project_ids=project_ids, settings=self.settings
        )

    async def get_invocation(self, invocation_id: str) -> Invocation | None:
        self._cleanup_old_invocations()
        return self._invocations.get(invocation_id)

    def _cleanup_old_invocations(self, ttl_seconds: int = 3600) -> None:
        """Remove completed invocations older than TTL."""
        now = datetime.now()
        expired = [
            inv_id
            for inv_id, inv in self._invocations.items()
            if inv.completed_at and (now - inv.completed_at).total_seconds() > ttl_seconds
        ]
        for inv_id in expired:
            del self._invocations[inv_id]
            self._tasks.pop(inv_id, None)

    async def _run_wiki_subprocess(
        self,
        invocation: Invocation,
        request: GenerateWikiRequest,
        planner_type: str,
        exclude_tests: bool,
    ) -> dict:
        """Run wiki generation in an isolated Python subprocess.

        Offloads the entire heavy pipeline (clone → parse → embed → LangGraph
        page generation) out of the uvicorn process so the event loop stays
        responsive regardless of repo size.  Progress events are streamed
        back line-by-line via the subprocess's stdout.
        """
        import base64
        import json
        import os
        import sys
        import tempfile

        payload = {
            "invocation_id": invocation.id,
            # --- multi-source fields (new) ---
            "source_type": request.source_type,
            "scope": request.scope,
            # auth dict intentionally excluded from logging (TokenRedactionFilter
            # is the safety net; primary defence is not serialising tokens).
            "auth": request.auth,
            # --- legacy git fields (still used by wiki_runner for git source) ---
            "repo_url": request.repo_url,
            "branch": request.branch,
            "access_token": request.access_token,
            # --- common fields ---
            "wiki_title": request.wiki_title,
            "include_research": request.include_research,
            "include_diagrams": request.include_diagrams,
            "force_rebuild_index": request.force_rebuild_index,
            "llm_model": request.llm_model,
            "embedding_model": request.embedding_model,
            "planner_type": planner_type,
            "exclude_tests": exclude_tests,
        }

        with tempfile.TemporaryDirectory(prefix="wiki-run-") as tmp:
            input_path = os.path.join(tmp, "input.json")
            output_path = os.path.join(tmp, "output.json")
            with open(input_path, "w", encoding="utf-8") as f:
                json.dump(payload, f)

            proc = await asyncio.create_subprocess_exec(
                sys.executable,
                "-u",  # unbuffered stdout — lines reach us promptly
                "-m",
                "app.core.wiki_runner",
                "--input",
                input_path,
                "--output",
                output_path,
                stdout=asyncio.subprocess.PIPE,
                # stderr inherits the parent's fd so logging lands in docker logs.
            )

            async def _pump_stdout() -> None:
                assert proc.stdout is not None
                while True:
                    line = await proc.stdout.readline()
                    if not line:
                        return
                    try:
                        ev = json.loads(line.decode("utf-8", errors="replace"))
                    except json.JSONDecodeError:
                        # Stray non-JSON line — ignore, it'll show up in docker logs via stderr.
                        continue
                    kind = ev.get("t")
                    if kind == "progress":
                        await self._emit_progress(
                            invocation,
                            ev.get("phase") or "",
                            float(ev.get("progress") or 0.0),
                            ev.get("message") or "",
                        )
                    elif kind == "token_estimate":
                        invocation.estimated_tokens = ev.get("estimated_tokens")
                        invocation.model_context_limit = ev.get("model_context_limit")
                        if invocation.estimated_tokens and invocation.model_context_limit:
                            ratio = invocation.estimated_tokens / invocation.model_context_limit
                            if ratio > 0.8:
                                await invocation.emit(
                                    events.progress(
                                        invocation.id,
                                        int(invocation.progress * 100) or 20,
                                        100,
                                        (
                                            f"Large repo: ~{invocation.estimated_tokens:,} estimated tokens "
                                            f"vs {invocation.model_context_limit:,} context limit "
                                            f"({ratio:.1f}x). Generation will use ranked truncation."
                                        ),
                                        phase="warning",
                                    )
                                )
                                logger.warning(
                                    "Context pressure ratio %.2fx for %s", ratio, request.repo_url,
                                )

            try:
                await asyncio.gather(_pump_stdout(), proc.wait())
            except asyncio.CancelledError:
                proc.terminate()
                try:
                    await asyncio.wait_for(proc.wait(), timeout=5)
                except asyncio.TimeoutError:
                    proc.kill()
                    await proc.wait()
                raise

            try:
                with open(output_path, encoding="utf-8") as f:
                    result = json.load(f)
            except (OSError, json.JSONDecodeError) as exc:
                raise RuntimeError(
                    f"Wiki runner produced no result file (exit={proc.returncode}): {exc}"
                ) from exc

            # Decode base64-wrapped artifact bytes back to ``bytes``.
            for art in result.get("artifacts") or []:
                if art.pop("_b64", False) and isinstance(art.get("data"), str):
                    art["data"] = base64.b64decode(art["data"])

            return result

    @staticmethod
    async def _emit_progress(invocation: Invocation, phase: str, progress: float, message: str) -> None:
        """Update invocation state and emit a progress event."""
        invocation.current_phase = phase
        invocation.progress = progress
        invocation.message = message
        await invocation.emit(events.progress(invocation.id, int(progress * 100), 100, message, phase=phase))

    @staticmethod
    async def _emit_error(
        invocation: Invocation,
        error: str,
        recoverable: bool = False,
        error_type: str | None = None,
        model_limit: int | None = None,
        suggested_actions: list[str] | None = None,
    ) -> None:
        """Emit a task_status failed event."""
        await invocation.emit(
            events.task_status(
                invocation.id,
                "failed",
                error,
                error=error,
                recoverable=recoverable,
                error_type=error_type,
                model_limit=model_limit,
                suggested_actions=suggested_actions,
            )
        )

    async def refresh(
        self,
        wiki_id: str,
        management: WikiManagementService,
        owner_id: str = "",
        access_token: str | None = None,
    ) -> Invocation | None:
        """Re-generate a wiki from its stored metadata. Old wiki stays until new one completes."""
        from app.models.api import GenerateWikiRequest

        # Prefer DB record — it exists from generation start now
        wiki_record = await management.get_wiki(wiki_id, user_id=owner_id or None)
        if wiki_record:
            from app.services.wiki_management import WikiManagementService as _WMS

            wiki_meta = _WMS._record_to_summary(wiki_record, owner_id or None)
        else:
            wiki_meta = None

        # Fall back to invocation data for wikis that were never registered (pre-migration)
        if not wiki_meta:
            inv = next((i for i in self._invocations.values() if i.wiki_id == wiki_id and i.repo_url), None)
            if not inv:
                return None
            request = GenerateWikiRequest(
                repo_url=inv.repo_url,
                branch=inv.branch,
                force_rebuild_index=True,
                access_token=access_token,
            )
            return await self.generate(request, owner_id=owner_id, force=True)

        request = GenerateWikiRequest(
            repo_url=wiki_meta.repo_url,
            branch=wiki_meta.branch,
            force_rebuild_index=True,
            visibility=wiki_meta.visibility,
            access_token=access_token,
        )
        return await self.generate(request, owner_id=owner_id, force=True)

    async def incremental_refresh(
        self,
        wiki_id: str,
        parsed_nodes: list[dict[str, Any]],
        management: "WikiManagementService",
        owner_id: str = "",
    ) -> tuple[Invocation, dict[str, Any]] | None:
        """Run an incremental refresh on a wiki with caller-supplied parsed nodes.

        Closes the #116 incremental-regen feature: change detection +
        three-regime dispatch + SSE telemetry, all driven from a
        pre-parsed node payload (same shape as ``/diff``).

        Production note: the structural-regen handler is intentionally a
        no-op fallback here — pages that would need a full single-page
        regen are counted in ``stats.structural_failed`` rather than
        actually regenerated. PR 5+ will wire ``make_agent_structural_handler``
        once the agent-construction prerequisites (indexer + retriever +
        LLM) are plumbed for the incremental path.

        Returns ``(invocation, stats_dict)`` on success; ``None`` when
        the wiki doesn't exist or its unified DB is missing.
        """
        # Ownership check is enforced at the route layer; we just need
        # the record to look up repo_url + cache_key.
        wiki_record = await management.get_wiki(wiki_id, user_id=owner_id or None)
        if wiki_record is None:
            return None

        # #140: idempotency guard. Two concurrent runs on the same wiki
        # race each other's content_hash updates + the trivial-patcher's
        # in-memory page_bodies dicts diverge. Reject the second caller
        # with the in-flight invocation_id so they can join the existing
        # SSE stream instead of spawning a parallel run.
        #
        # Critical: check + register BEFORE any awaits so two callers
        # racing through this function can't both pass. ``running``
        # covers incremental refreshes; ``generating`` covers a full
        # ``generate()`` on the same wiki — both write to the same
        # ``.wiki.db``. Shared helper with :meth:`generate` (#145).
        in_flight = self._find_in_flight_for_wiki(wiki_id)
        if in_flight is not None:
            raise IncrementalRefreshInProgressError(wiki_id, in_flight.id)

        # Reserve the invocation slot atomically (no awaits between the
        # guard check and this assignment) so a second caller racing
        # through the check now hits a registered "running" entry and
        # gets a 409. If the early-out branches below (missing cache key
        # / db file) fire, we pop the reservation back out before
        # returning None.
        invocation = Invocation(
            id=str(uuid4()),
            wiki_id=wiki_id,
            repo_url=wiki_record.repo_url,
            branch=wiki_record.branch,
            status="running",
            owner_id=owner_id,
        )
        self._invocations[invocation.id] = invocation

        from pathlib import Path as _Path

        from app.core.agents.page_patcher import PagePatcher
        from app.core.storage import open_storage
        from app.services.incremental_regen import PageRegime  # noqa: F401
        from app.services.incremental_regen_service import (
            IncrementalRegenService,
        )
        from app.services.wiki_management import _derive_cache_key

        cache_key = _derive_cache_key(
            self.settings.cache_dir, wiki_record.repo_url, wiki_record.branch,
        )
        if not cache_key:
            self._invocations.pop(invocation.id, None)
            return None
        db_path = _Path(self.settings.cache_dir) / f"{cache_key}.wiki.db"
        if not db_path.exists():
            self._invocations.pop(invocation.id, None)
            return None

        # Preload every page body into an in-memory dict so the
        # orchestrator's sync callbacks don't need to bridge into
        # async artifact I/O. The dict is the source of truth during
        # the run; modified entries get flushed back at the end.
        page_bodies: dict[str, str] = {}
        artifact_keys: dict[str, str] = {}
        existing = await self.storage.list_artifacts(
            "wiki_artifacts", prefix=wiki_id,
        )
        for artifact_key in existing:
            if not artifact_key.endswith(".md"):
                continue
            page_id = (
                artifact_key.removeprefix(f"{wiki_id}/")
                .removeprefix("wiki_pages/")
                .removesuffix(".md")
            )
            try:
                raw = await self.storage.download("wiki_artifacts", artifact_key)
                page_bodies[page_id] = (
                    raw.decode("utf-8") if isinstance(raw, bytes) else str(raw)
                )
                artifact_keys[page_id] = artifact_key
            except FileNotFoundError:
                continue
        # Persist immediately so a process restart between this 202 and
        # task completion doesn't strand the SSE stream — late-connecting
        # clients can still reconnect via Last-Event-ID + replay.
        await self._persist_invocations()
        await invocation.emit(events.task_status(
            invocation.id, "running", "Incremental refresh started",
        ))

        modified_bodies: dict[str, str] = {}

        def _read(pid: str) -> str | None:
            return page_bodies.get(pid)

        def _write(pid: str, body: str) -> None:
            page_bodies[pid] = body
            modified_bodies[pid] = body

        def _stub_structural(page) -> str:
            # Fallback when agent construction fails. Logged at WARNING
            # so partial-feature regressions show up in production logs.
            # Returning a reason string (per #134's new contract) lets
            # the orchestrator's structural_failure_reasons telemetry
            # capture *why* the fallback fired.
            logger.warning(
                "[incremental_refresh] structural regen unavailable for "
                "page %s (agent construction failed earlier in the run)",
                page.page_id,
            )
            return "agent construction unavailable for this run"

        # Page-event builders keyed by event_name. Each takes
        # (invocation_id, page_id, page_title, **extras). The summary
        # event has a different shape (no per-page IDs) so it's handled
        # explicitly below.
        _page_event_builders: dict[str, Any] = {
            "page_unchanged": events.page_unchanged,
            "page_patched": events.page_patched,
            "page_edited": events.page_edited,
            "page_regenerated": events.page_regenerated,
            # #141: page_deleted now wired into the dispatcher.
            "page_deleted": events.page_deleted,
        }

        def _emit(event_name: str, payload: dict[str, Any]) -> None:
            if event_name == "incremental_summary":
                invocation.emit_sync(events.incremental_summary(
                    invocation.id, payload.get("stats", {}),
                ))
                return
            builder = _page_event_builders.get(event_name)
            if builder is None:
                return
            kwargs = {
                k: v for k, v in payload.items()
                if k not in {"page_id", "page_title"}
            }
            invocation.emit_sync(builder(
                invocation.id,
                payload["page_id"],
                payload["page_title"],
                **kwargs,
            ))

        # LLM for surgical edits. Use the project's configured one.
        # Failing to construct an LLM is fatal for the edit regime but
        # the orchestrator still handles trivial + structural regimes
        # without it (the patcher's quality gate rejects on LLM error).
        try:
            from app.services.llm_factory import create_llm

            llm = create_llm(self.settings)
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "[incremental_refresh] LLM init failed; edit regime will "
                "fall back to structural: %s",
                exc,
            )
            llm = None

        async def _run() -> None:
            try:
                storage = open_storage(
                    repo_id=cache_key, db_path=str(db_path), readonly=False,
                )
                try:
                    patcher = PagePatcher(llm) if llm is not None else None
                    if patcher is None:
                        # The orchestrator unconditionally constructs a
                        # PagePatcher path; without an LLM we can't run
                        # the edit regime. Fail the run loudly here.
                        invocation.status = "failed"
                        invocation.completed_at = datetime.now()
                        await invocation.emit(events.task_status(
                            invocation.id, "failed",
                            "LLM unavailable — incremental refresh requires LLM",
                        ))
                        return

                    # #142: build the production structural handler.
                    # On any failure (agent construction error), fall
                    # back to the stub so the trivial + edit regimes
                    # still work for this run. The structural pages
                    # will be counted as failed; the SSE summary lets
                    # callers see what was missed.
                    from app.services.agent_builder import (
                        build_agent_for_incremental_refresh,
                    )
                    from app.services.structural_handler_factory import (
                        make_agent_structural_handler,
                    )

                    agent = build_agent_for_incremental_refresh(
                        wiki_record, storage, llm, self.settings,
                    )
                    if agent is not None:
                        structural_handler = make_agent_structural_handler(
                            agent,
                            storage=storage,
                            # TODO(#142-followup): thread the repo's
                            # repository_analysis from storage.get_meta()
                            # so structural prompts see the README-level
                            # context full regen would inject.
                            repository_context="",
                            write_page_body=_write,
                        )
                    else:
                        # Agent construction failed (embeddings, retriever,
                        # or agent __init__). Emit a status_message so the
                        # SPA can show "structural regime unavailable" —
                        # without this signal, callers see a partial-
                        # success summary that conflates "no structural
                        # pages in plan" with "every structural page
                        # silently failed because the deployment is mis-
                        # configured".
                        logger.warning(
                            "[incremental_refresh] agent construction "
                            "failed; structural regime will fail every page",
                        )
                        await invocation.emit(events.message(
                            "warning",
                            "Structural regen unavailable for this run "
                            "(agent construction failed). Trivial + edit "
                            "regimes will still run; any structural page "
                            "will count as failed.",
                        ))
                        structural_handler = _stub_structural

                    svc = IncrementalRegenService(
                        storage=storage,
                        page_patcher=patcher,
                        read_page_body=_read,
                        write_page_body=_write,
                        structural_handler=structural_handler,
                        progress_callback=_emit,
                    )
                    stats = await asyncio.to_thread(svc.run, parsed_nodes)
                finally:
                    try:
                        storage.close()
                    except Exception:  # noqa: S110 — best-effort close
                        pass

                # Flush modified bodies back to artifact storage.
                for pid, body in modified_bodies.items():
                    key = artifact_keys.get(
                        pid,
                        f"{wiki_id}/wiki_pages/{pid}.md",
                    )
                    try:
                        await self.storage.upload(
                            "wiki_artifacts", key, body.encode("utf-8"),
                        )
                    except Exception as exc:  # noqa: BLE001
                        logger.warning(
                            "[incremental_refresh] body flush failed for %s: %s",
                            pid, exc,
                        )

                # Critical: flip the in-memory status off "running" BEFORE
                # the persist in the finally block. Without this, the
                # idempotency guard at the top of incremental_refresh
                # would 409-lock the wiki forever — every subsequent
                # refresh (and full generate, since the guard widened to
                # include "generating") sees this phantom "running"
                # entry. Mirrors _run_generation's terminal-state writes.
                # #146: also set ``completed_at`` so the periodic
                # ``_cleanup_old_invocations`` purge can reclaim memory
                # for completed runs (without it the dict grows
                # monotonically until process restart).
                invocation.status = "complete"
                invocation.completed_at = datetime.now()
                await invocation.emit(events.task_status(
                    invocation.id, "completed",
                    "Incremental refresh complete",
                ))
            except Exception as exc:  # noqa: BLE001
                # Mirror _run_generation: capture the exception string on the
                # invocation so the finally block's mark_status call writes a
                # non-None ``error`` to WikiRecord and the dashboard surfaces
                # the failure reason. Previously this path silently flipped
                # the wiki to "failed" with error=None (#226).
                invocation.status = "failed"
                invocation.error = str(exc)
                invocation.completed_at = datetime.now()
                logger.exception(
                    "[incremental_refresh] run failed for wiki %s", wiki_id,
                )
                await invocation.emit(events.task_status(
                    invocation.id, "failed", str(exc),
                ))
            finally:
                # Re-persist so the terminal status survives a restart.
                await self._persist_invocations()
                # #177: transition the DB WikiRecord to the terminal status.
                # Mirrors _run_generation's reconciliation finally block.
                # ``mark_status`` is update-only — if the pre-register call
                # above silently failed, the record may not exist, but for a
                # refresh (as opposed to a full generate) the wiki had a
                # valid record before we started, so that case is unlikely.
                if self.wiki_management and wiki_id:
                    try:
                        await self.wiki_management.mark_status(
                            wiki_id=wiki_id,
                            status=invocation.status,
                            error=invocation.error,
                        )
                    except Exception as _db_err:
                        logger.warning(
                            "[incremental_refresh] Failed to reconcile "
                            "WikiRecord status for %s: %s",
                            wiki_id,
                            _db_err,
                        )

        # #177: write status="running" to the DB now — AFTER the early-return
        # guards — so we only record a start when we're actually committed to
        # launching the background task. Using mark_status (update-only) rather
        # than register_wiki preserves commit_hash, title, page_count, etc.
        # that the original generate() wrote. A crashed process leaves this
        # row at "running"; load_persisted_invocations flips it to "failed"
        # on restart.
        if self.wiki_management:
            try:
                await self.wiki_management.mark_status(
                    wiki_id=wiki_id,
                    status="running",
                    error=None,
                )
            except Exception as e:
                logger.warning(
                    "[incremental_refresh] Failed to pre-register refresh: %s", e
                )

        task = asyncio.create_task(_run())
        self._tasks[invocation.id] = task

        # Page-count denominator for the SPA progress bar. Use the
        # actual wiki_pages row count, not the .md artifact count —
        # the bucket can contain auxiliary files (README copies,
        # changelogs) that aren't part of the plan.
        try:
            tmp_storage = open_storage(
                repo_id=cache_key, db_path=str(db_path), readonly=True,
            )
            try:
                wiki_page_count = len(tmp_storage.get_wiki_pages(wiki_id))
            finally:
                try:
                    tmp_storage.close()
                except Exception:  # noqa: S110 — best-effort close
                    pass
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "[incremental_refresh] wiki_page_count lookup failed; "
                "falling back to body count: %s",
                exc,
            )
            wiki_page_count = len(page_bodies)

        # Pre-compute stats summary the route can include in its response
        # *before* the background run completes. Final stats arrive via SSE.
        return invocation, {"status": "running", "page_count": wiki_page_count}

    async def resume(self, wiki_id: str, management: WikiManagementService, owner_id: str = "") -> Invocation | None:
        """Resume a partial wiki generation, skipping already-completed pages."""
        from app.models.api import GenerateWikiRequest

        wiki_list = await management.list_wikis(user_id=owner_id or None)
        wiki_meta = next((w for w in wiki_list.wikis if w.wiki_id == wiki_id), None)
        if not wiki_meta:
            return None

        # Check for in-progress generation
        for inv in self._invocations.values():
            if inv.wiki_id == wiki_id and inv.status == "generating":
                return None  # 409 — already generating

        # Get existing pages from storage
        existing_pages = await self.storage.list_artifacts("wiki_artifacts", prefix=wiki_id)
        existing_page_ids = {
            p.replace(f"{wiki_id}/", "").replace(".md", "") for p in existing_pages if p.endswith(".md")
        }

        request = GenerateWikiRequest(
            repo_url=wiki_meta.repo_url,
            branch=wiki_meta.branch,
            force_rebuild_index=False,  # reuse existing index
            visibility=wiki_meta.visibility,
        )

        # Start generation — completed_page_ids passed via invocation metadata
        wiki_id_computed = self._make_wiki_id(request.repo_url, request.branch)
        invocation_id = str(uuid4())
        invocation = Invocation(
            id=invocation_id,
            wiki_id=wiki_id_computed,
            repo_url=request.repo_url,
            branch=request.branch,
            owner_id=owner_id,
            status="generating",
            current_phase="resuming",
            message=f"Resuming generation — {len(existing_page_ids)} pages already complete",
            pages_completed=len(existing_page_ids),
        )
        self._invocations[invocation_id] = invocation
        await self._persist_invocations()

        task = asyncio.create_task(self._run_generation(invocation, request))
        self._tasks[invocation_id] = task
        return invocation

    async def cancel_invocation(self, invocation_id: str) -> bool:
        if self._shutting_down:
            # #165 follow-up: refuse to cancel during shutdown drain.
            # ``shutdown()`` is awaiting these tasks so their inner
            # ``asyncio.to_thread()`` worker threads can finish
            # against live storage; cancelling here would mark the
            # asyncio.Task done while leaving the thread mid-write —
            # the exact race we hardened against.
            logger.info(
                "cancel_invocation(%s) refused: service is shutting down",
                invocation_id,
            )
            return False
        invocation = self._invocations.get(invocation_id)
        if not invocation or invocation.status != "generating":
            return False
        task = self._tasks.get(invocation_id)
        if task and not task.done():
            task.cancel()
            return True
        return False
