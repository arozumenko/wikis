"""Wiki generation service — orchestrates background wiki generation."""

from __future__ import annotations

import asyncio
import hashlib
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


class WikiService:
    """Orchestrates wiki generation from repository analysis."""

    def __init__(self, settings: Settings, storage: ArtifactStorage, wiki_management: Any = None) -> None:
        self.settings = settings
        self.storage = storage
        self.wiki_management = wiki_management
        self._invocations: dict[str, Invocation] = {}
        self._tasks: dict[str, asyncio.Task] = {}

    @property
    def invocations(self) -> dict[str, Invocation]:
        """Public read-only access to the invocations dict."""
        return self._invocations

    def remove_invocation(self, inv_id: str) -> None:
        """Remove an invocation and its task from tracking."""
        self._invocations.pop(inv_id, None)
        self._tasks.pop(inv_id, None)

    async def persist_invocations(self) -> None:
        """Public wrapper — save all invocations to storage."""
        await self._persist_invocations()

    INVOCATIONS_BUCKET = "wiki_registry"
    INVOCATIONS_KEY = "invocations.json"

    async def load_persisted_invocations(self) -> None:
        """Load invocations from storage on startup."""
        try:
            import json as _json

            data = await self.storage.download(self.INVOCATIONS_BUCKET, self.INVOCATIONS_KEY)
            raw = _json.loads(data)
            for inv_id, inv_data in raw.items():
                try:
                    inv = Invocation(**inv_data)
                    if inv.status == "generating":
                        inv.status = "failed"
                        inv.error = "Server restarted during generation"
                        inv.completed_at = datetime.now()
                    self._invocations[inv_id] = inv
                except Exception:  # noqa: S110
                    pass
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
    def _make_wiki_id(repo_url: str, branch: str) -> str:
        """Deterministic wiki ID from repo URL + branch."""
        key = f"{repo_url}:{branch}"
        return hashlib.sha256(key.encode()).hexdigest()[:16]

    async def generate(self, request: GenerateWikiRequest, owner_id: str = "", force: bool = False) -> Invocation:
        """Start wiki generation in background, return invocation."""
        from pathlib import Path

        from app.core.local_repo_provider import extract_git_metadata, is_local_path, make_local_wiki_id

        if is_local_path(request.repo_url):
            path = request.repo_url.removeprefix("file://")
            info = extract_git_metadata(Path(path).resolve())
            wiki_id = make_local_wiki_id(path, info.branch if info.is_git else None)
        else:
            wiki_id = self._make_wiki_id(request.repo_url, request.branch)

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
                await self.wiki_management.register_wiki(
                    wiki_id=wiki_id,
                    repo_url=request.repo_url,
                    branch=request.branch,
                    title=request.wiki_title or f"Wiki for {request.repo_url}",
                    page_count=0,
                    owner_id=owner_id,
                    visibility=getattr(request, "visibility", "personal"),
                    status="generating",
                    requires_token=bool(request.access_token),
                )
            except Exception as e:
                logger.warning(f"Failed to pre-register wiki {wiki_id}: {e}")

        task = asyncio.create_task(self._run_generation(invocation, request))
        self._tasks[invocation_id] = task
        return invocation

    async def _run_generation(self, invocation: Invocation, request: GenerateWikiRequest) -> None:
        """Background task: clone → index → generate → store artifacts."""
        try:
            from app.core.hybrid_wiki_toolkit_wrapper import HybridWikiToolkitWrapper
            from app.core.repo_providers.factory import RepoProviderFactory
            from app.services.llm_factory import create_embeddings, create_llm

            await self._emit_progress(invocation, "configuring", 0.05, "Configuring LLM and embeddings")

            # Build LLM and embeddings
            llm_overrides = {}
            if request.llm_model:
                llm_overrides["model"] = request.llm_model
            llm = create_llm(self.settings, tier="high", **llm_overrides)
            llm_low = create_llm(self.settings, tier="low")

            emb_overrides = {}
            if request.embedding_model:
                emb_overrides["model"] = request.embedding_model
            embeddings = create_embeddings(self.settings, **emb_overrides)

            # Build clone config from URL
            clone_config = RepoProviderFactory.from_url(
                url=request.repo_url,
                token=request.access_token,
                branch=request.branch,
            )

            await self._emit_progress(invocation, "indexing", 0.1, "Cloning repository and building index")

            # Progress callback for indexing phases
            def progress_callback(phase: str, progress: float, message: str) -> None:
                """Sync callback — uses thread-safe emit_sync."""
                try:
                    invocation.emit_sync(
                        events.progress(invocation.id, int(progress * 100), 100, message, phase=phase),
                    )
                    invocation.current_phase = phase
                    invocation.progress = progress
                    invocation.message = message
                except Exception:  # noqa: S110
                    pass  # never break generation for progress reporting

            # Create toolkit and generate
            toolkit = HybridWikiToolkitWrapper(
                clone_config=clone_config,
                llm=llm,
                embeddings=embeddings,
                cache_dir=self.settings.cache_dir,
                force_rebuild_index=request.force_rebuild_index,
                llm_low=llm_low,
                progress_callback=progress_callback,
                # Keep the cloned repo on disk so Deep Research's
                # FilesystemBackend can read source files post-generation.
                # Explicit cleanup happens via delete_wiki().
                cleanup_repos_on_exit=False,
                max_concurrent_pages=self.settings.llm_max_concurrency,
            )

            # Token pre-estimation after toolkit init
            try:
                from app.services.context_limits import get_context_limit

                _model_name = request.llm_model or self.settings.llm_model or ""
                estimated = 0
                # Use toolkit's token counter if available (tiktoken-based)
                if hasattr(toolkit, "estimate_index_tokens"):
                    estimated = toolkit.estimate_index_tokens()
                # Fallback: estimate from index summary char count
                if not estimated and hasattr(toolkit, "indexer") and toolkit.indexer:
                    summary = toolkit.get_index_summary() if hasattr(toolkit, "get_index_summary") else {}
                    total_chars = summary.get("total_chars", 0) if isinstance(summary, dict) else 0
                    estimated = total_chars // 4
                if estimated > 0:
                    invocation.estimated_tokens = estimated
                    invocation.model_context_limit = get_context_limit(_model_name)
                    ratio = (
                        invocation.estimated_tokens / invocation.model_context_limit
                        if invocation.model_context_limit
                        else 0
                    )
                    if ratio > 0.8:
                        await invocation.emit(
                            events.progress(
                                invocation.id,
                                20,
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
                            f"Context pressure ratio {ratio:.2f}x for {request.repo_url} (model={_model_name})"
                        )
            except Exception as e:
                logger.debug(f"Token estimation skipped: {e}")

            await self._emit_progress(invocation, "generating", 0.3, "Generating wiki content")

            # Resolve generation options (request overrides → config defaults)
            planner_type = request.planner_type if request.planner_type is not None else self.settings.planner_type
            # exclude_tests only applies to the cluster planner; agent planner ignores it.
            if planner_type == "cluster":
                exclude_tests = (
                    request.exclude_tests if request.exclude_tests is not None
                    else self.settings.cluster_exclude_tests
                )
            else:
                exclude_tests = False

            # generate_wiki is sync/CPU-bound — run in thread
            result = await asyncio.to_thread(
                toolkit.generate_wiki,
                query=request.wiki_title or "Generate comprehensive wiki",
                include_research=request.include_research,
                include_diagrams=request.include_diagrams,
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
                await self.wiki_management.register_wiki(
                    wiki_id=invocation.wiki_id,
                    repo_url=registry_url,
                    branch=registry_branch,
                    title=request.wiki_title or f"Wiki for {request.repo_url}",
                    page_count=page_count,
                    owner_id=invocation.owner_id,
                    visibility=getattr(request, "visibility", "personal"),
                    commit_hash=result.get("commit_hash"),
                    indexed_at=datetime.now(),
                    status="complete",
                    requires_token=bool(request.access_token),
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
            # Update DB record for non-complete outcomes so retry/404 logic sees the final status
            if invocation.status != "complete" and self.wiki_management and invocation.repo_url:
                try:
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
                    logger.warning(f"Failed to update wiki status in DB: {_db_err}")

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
        invocation = self._invocations.get(invocation_id)
        if not invocation or invocation.status != "generating":
            return False
        task = self._tasks.get(invocation_id)
        if task and not task.done():
            task.cancel()
            return True
        return False
