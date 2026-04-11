"""API routes — wiki generation is live, other endpoints are Phase 2 stubs."""

from __future__ import annotations

import json as _json
import re
from dataclasses import asdict

from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException, Query, Request
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel
from sse_starlette import EventSourceResponse, ServerSentEvent

from app.auth import CurrentUser, get_current_user
from app.dependencies import (
    get_ask_service,
    get_project_service,
    get_qa_service,
    get_research_service,
    get_wiki_management,
    get_wiki_service,
)
from app.models import (
    AskRequest,
    AskResponse,
    DeleteWikiResponse,
    ErrorResponse,
    GenerateWikiRequest,
    GenerateWikiResponse,
    HealthResponse,
    RefreshWikiRequest,
    ResearchRequest,
    ResearchResponse,
    UpdateWikiVisibilityRequest,
    WikiListResponse,
    WikiSummary,
)
from app.models.api import (
    ProjectAddWikiRequest,
    ProjectCreateRequest,
    ProjectListResponse,
    ProjectResponse,
    ProjectUpdateRequest,
)
from app.services.project_service import ProjectService
from app.models.invocation import Invocation
from app.models.qa_api import QAListResponse, QARecordResponse, QAStatsResponse, QAStatus
from app.services.ask_service import AskService
from app.services.qa_service import QAService
from app.services.research_service import ResearchService
from app.services.wiki_management import WikiManagementService
from app.services.wiki_service import WikiService

router = APIRouter(prefix="/api/v1")

NOT_IMPLEMENTED = "Not implemented — Phase 2"


# ---------------------------------------------------------------------------
# Wiki Generation (LIVE)
# ---------------------------------------------------------------------------


@router.post(
    "/generate",
    response_model=GenerateWikiResponse,
    status_code=202,
    responses={409: {"model": ErrorResponse}, 422: {"model": ErrorResponse}},
)
async def generate_wiki(
    request: GenerateWikiRequest,
    user: CurrentUser = Depends(get_current_user),
    service: WikiService = Depends(get_wiki_service),
) -> GenerateWikiResponse:
    from app.services.wiki_service_errors import WikiAlreadyExistsError

    try:
        invocation = await service.generate(request, owner_id=user.id if user else "")
    except WikiAlreadyExistsError as e:
        raise HTTPException(
            status_code=409,
            detail={"error": str(e), "wiki_id": e.wiki_id},
        )
    return GenerateWikiResponse(
        wiki_id=invocation.wiki_id,
        invocation_id=invocation.id,
        status=invocation.status,
        message=f"Generation started. Track via GET /api/v1/invocations/{invocation.id}",
    )


@router.get("/invocations/{invocation_id}", response_model=Invocation)
async def get_invocation(
    invocation_id: str,
    user: CurrentUser = Depends(get_current_user),
    service: WikiService = Depends(get_wiki_service),
) -> Invocation:
    invocation = await service.get_invocation(invocation_id)
    if not invocation:
        raise HTTPException(404, "Invocation not found")
    return invocation


@router.delete("/invocations/{invocation_id}")
async def cancel_invocation(
    invocation_id: str,
    user: CurrentUser = Depends(get_current_user),
    service: WikiService = Depends(get_wiki_service),
) -> JSONResponse:
    cancelled = await service.cancel_invocation(invocation_id)
    if not cancelled:
        raise HTTPException(404, "Invocation not found or not cancellable")
    return JSONResponse({"cancelled": True, "invocation_id": invocation_id})


@router.get("/invocations/{invocation_id}/stream")
async def stream_invocation(
    invocation_id: str,
    request: Request,
    user: CurrentUser = Depends(get_current_user),
    service: WikiService = Depends(get_wiki_service),
) -> EventSourceResponse:
    invocation = await service.get_invocation(invocation_id)
    if not invocation:
        raise HTTPException(404, "Invocation not found")

    # Support Last-Event-ID reconnection (MCP EventStore pattern)
    last_event_id_raw = request.headers.get("last-event-id")
    after_event_id = int(last_event_id_raw) if last_event_id_raw and last_event_id_raw.isdigit() else None

    async def event_generator():
        async for idx, event in invocation.events(after_event_id=after_event_id):
            yield ServerSentEvent(
                id=str(idx),
                event=getattr(event, "event", "message"),
                data=event.model_dump_json(),
            )

    return EventSourceResponse(event_generator())


# ---------------------------------------------------------------------------
# Ask / Research (stubs)
# ---------------------------------------------------------------------------


@router.post("/ask", response_model=AskResponse)
async def ask(
    request: AskRequest,
    background_tasks: BackgroundTasks,
    user: CurrentUser = Depends(get_current_user),
    service: AskService = Depends(get_ask_service),
    qa_service: QAService = Depends(get_qa_service),
    accept: str = "application/json",
) -> AskResponse | StreamingResponse:
    user_id = user.id if user else None
    try:
        if "text/event-stream" in accept:
            # SSE: generator handles its own recording via try/finally
            async def stream():
                async for event in service.ask_stream(request, user_id=user_id):
                    event_type = event.get("event_type", "message")
                    yield f"event: {event_type}\ndata: {_json.dumps(event.get('data', {}))}\n\n"

            return StreamingResponse(stream(), media_type="text/event-stream")

        # Sync: BackgroundTask for recording
        result = await service.ask_sync(request, user_id=user_id)
        if result.recording:
            background_tasks.add_task(qa_service.record_interaction, **asdict(result.recording))
        return result.response
    except FileNotFoundError as e:
        raise HTTPException(404, f"Wiki not found: {request.wiki_id or request.project_id}") from e
    except ValueError as e:
        raise HTTPException(400, str(e)) from e
    except RuntimeError as e:
        raise HTTPException(501, str(e)) from e


@router.post("/research", response_model=ResearchResponse)
async def research(
    request: ResearchRequest,
    user: CurrentUser = Depends(get_current_user),
    service: ResearchService = Depends(get_research_service),
    accept: str = "application/json",
) -> ResearchResponse | StreamingResponse:
    user_id = user.id if user else None
    try:
        if request.research_type == "codemap":
            if "text/event-stream" in accept:
                async def codemap_sse():
                    async for event in service.codemap_stream(request, user_id=user_id):
                        event_type = event.get("event_type", "message")
                        yield f"event: {event_type}\ndata: {_json.dumps(event.get('data', {}))}\n\n"
                return StreamingResponse(codemap_sse(), media_type="text/event-stream")
            return await service.codemap_sync(request, user_id=user_id)
        if "text/event-stream" in accept:

            async def stream():
                async for event in service.research_stream(request, user_id=user_id):
                    event_type = event.get("event_type", "message")
                    yield f"event: {event_type}\ndata: {_json.dumps(event.get('data', {}))}\n\n"

            return StreamingResponse(stream(), media_type="text/event-stream")
        return await service.research_sync(request, user_id=user_id)
    except FileNotFoundError as e:
        raise HTTPException(404, f"Wiki not found: {request.wiki_id or request.project_id}") from e
    except ValueError as e:
        raise HTTPException(400, str(e)) from e
    except RuntimeError as e:
        raise HTTPException(501, str(e)) from e


# ---------------------------------------------------------------------------
# Wiki Management (stubs)
# ---------------------------------------------------------------------------


@router.get("/wikis", response_model=WikiListResponse)
async def list_wikis(
    user: CurrentUser = Depends(get_current_user),
    management: WikiManagementService = Depends(get_wiki_management),
    service: WikiService = Depends(get_wiki_service),
) -> WikiListResponse:
    user_id = user.id if user else None
    result = await management.list_wikis(user_id=user_id)
    completed_wiki_ids = {w.wiki_id for w in result.wikis}

    # Enrich completed wikis with actual page count + invocation metadata
    for wiki in result.wikis:
        # Count actual stored pages (registry page_count may be stale)
        try:
            artifacts = await management.storage.list_artifacts("wiki_artifacts", prefix=wiki.wiki_id)
            actual_pages = sum(1 for a in artifacts if a.endswith(".md") and "wiki_pages" in a)
            if actual_pages > 0:
                wiki.page_count = actual_pages
        except Exception:  # noqa: S110
            pass

        # Find the most relevant invocation — prefer "generating" over terminal
        best_inv = None
        for inv in service.invocations.values():
            if inv.wiki_id == wiki.wiki_id:
                if inv.status == "generating":
                    best_inv = inv
                    break
                if best_inv is None:
                    best_inv = inv
        if best_inv:
            wiki.invocation_id = best_inv.id
            # Only override DB status if the invocation is still generating.
            # DB is source of truth for terminal states.
            if best_inv.status == "generating":
                wiki.status = best_inv.status
                wiki.progress = best_inv.progress

    # Add active/failed invocations not yet in completed list
    for inv in service.invocations.values():
        if inv.wiki_id not in completed_wiki_ids:
            # Auto-register completed invocations into DB so get_wiki works
            if inv.status == "complete" and inv.repo_url:
                try:
                    await management.register_wiki(
                        wiki_id=inv.wiki_id,
                        repo_url=inv.repo_url,
                        branch=inv.branch,
                        title=f"Wiki for {inv.repo_url}",
                        page_count=inv.pages_completed,
                        owner_id=inv.owner_id,
                    )
                    completed_wiki_ids.add(inv.wiki_id)
                    result.wikis.append(
                        WikiSummary(
                            wiki_id=inv.wiki_id,
                            repo_url=inv.repo_url,
                            branch=inv.branch,
                            title=f"Wiki for {inv.repo_url}",
                            created_at=inv.created_at,
                            page_count=inv.pages_completed,
                            status="complete",
                            invocation_id=inv.id,
                            owner_id=inv.owner_id,
                            is_owner=(user_id is not None and (not inv.owner_id or inv.owner_id == user_id)),
                        )
                    )
                    continue
                except Exception:
                    pass  # fall through to append as invocation-only
            result.wikis.append(
                WikiSummary(
                    wiki_id=inv.wiki_id,
                    repo_url=inv.repo_url,
                    branch=inv.branch,
                    title=f"Generating: {inv.wiki_id[:8]}...",
                    created_at=inv.created_at,
                    page_count=inv.pages_completed,
                    status=inv.status,
                    progress=inv.progress,
                    invocation_id=inv.id,
                    error=inv.error,
                    owner_id=inv.owner_id,
                    # Empty owner_id means unowned — let any authenticated user manage it
                    is_owner=(user_id is not None and (not inv.owner_id or inv.owner_id == user_id)),
                )
            )
    return result


def _extract_frontmatter_title_section(content: str) -> tuple[str, str]:
    """Parse YAML frontmatter and return (title, section) from it, or ('', '') if absent."""
    if not content.startswith("---"):
        return "", ""
    end = content.find("\n---", 3)
    if end == -1:
        return "", ""
    yaml_block = content[4:end]
    title = ""
    section = ""
    for line in yaml_block.splitlines():
        if not title:
            m = re.match(r'^title\s*:\s*["\']?(.+?)["\']?\s*$', line)
            if m:
                title = m.group(1).strip()
        if not section:
            m = re.match(r'^section\s*:\s*["\']?(.+?)["\']?\s*$', line)
            if m:
                section = m.group(1).strip()
        if title and section:
            break
    return title, section


@router.get("/wikis/{wiki_id}")
async def get_wiki(
    wiki_id: str,
    user: CurrentUser = Depends(get_current_user),
    management: WikiManagementService = Depends(get_wiki_management),
    service: WikiService = Depends(get_wiki_service),
) -> dict:
    """Get wiki detail with pages and their content."""
    user_id = user.id if user else None
    wiki_record = await management.get_wiki(wiki_id, user_id=user_id)
    wiki_meta = WikiManagementService._record_to_summary(wiki_record, user_id) if wiki_record else None

    # Check for active invocation — prefer "generating" over terminal states
    active_invocation = None
    for inv in service.invocations.values():
        if inv.wiki_id == wiki_id:
            if inv.status == "generating":
                active_invocation = inv
                break  # generating is always the most relevant
            if active_invocation is None:
                active_invocation = inv  # keep first match as fallback

    # B2: Don't leak in-flight invocation metadata to non-owners
    if active_invocation and user_id and active_invocation.owner_id and active_invocation.owner_id != user_id:
        active_invocation = None

    # If DB has a record in a non-complete state (new: registered at generation start),
    # serve it immediately so the UI gets repo details + status without needing an active invocation.
    if wiki_meta and wiki_meta.status in ("generating", "failed", "partial", "cancelled") and not active_invocation:
        return {
            "wiki_id": wiki_id,
            "repo_url": wiki_meta.repo_url,
            "branch": wiki_meta.branch,
            "title": wiki_meta.title,
            "page_count": wiki_meta.page_count,
            "created_at": wiki_meta.created_at.isoformat(),
            "sections": [],
            "pages": [],
            "status": wiki_meta.status,
            "invocation_id": None,
            "error": wiki_meta.error,
            "requires_token": wiki_meta.requires_token,
        }

    if not wiki_meta:
        if active_invocation:
            # Completed invocation not in DB — auto-register and serve (backward compat)
            if active_invocation.status == "complete" and active_invocation.repo_url:
                try:
                    await management.register_wiki(
                        wiki_id=wiki_id,
                        repo_url=active_invocation.repo_url,
                        branch=active_invocation.branch,
                        title=f"Wiki for {active_invocation.repo_url}",
                        page_count=active_invocation.pages_completed,
                        owner_id=active_invocation.owner_id,
                        status="complete",
                    )
                    wiki_record = await management.get_wiki(wiki_id, user_id=user_id)
                    wiki_meta = WikiManagementService._record_to_summary(wiki_record, user_id) if wiki_record else None
                except Exception:
                    pass
            # Still generating or failed — return minimal info so the UI
            # can show repo details and offer a retry button (backward compat for pre-migration wikis).
            if not wiki_meta and active_invocation.status in ("generating", "failed", "partial", "cancelled"):
                return {
                    "wiki_id": wiki_id,
                    "repo_url": active_invocation.repo_url,
                    "branch": active_invocation.branch,
                    "title": f"Wiki for {active_invocation.repo_url}",
                    "page_count": active_invocation.pages_completed,
                    "created_at": active_invocation.created_at.isoformat(),
                    "sections": [],
                    "pages": [],
                    "status": active_invocation.status,
                    "invocation_id": active_invocation.id,
                    "error": active_invocation.error,
                    "requires_token": False,
                }
        if not wiki_meta:
            raise HTTPException(404, f"Wiki not found: {wiki_id}")

    all_artifacts = await management.storage.list_artifacts("wiki_artifacts", prefix=wiki_id)

    # Load wiki structure JSON for section/page ordering
    wiki_title = wiki_meta.title
    sections = []
    structure_files = [a for a in all_artifacts if "wiki_structure" in a and a.endswith(".json")]
    if structure_files:
        try:
            struct_data = await management.storage.download("wiki_artifacts", structure_files[-1])
            structure = _json.loads(struct_data)
            wiki_title = structure.get("wiki_title", wiki_title)
            sections = structure.get("sections", [])
        except Exception:  # noqa: S110
            pass

    # Find all .md page files (nested under wiki_pages/)
    md_files = sorted(a for a in all_artifacts if a.endswith(".md") and "wiki_pages" in a)

    pages = []
    for i, artifact_path in enumerate(md_files):
        # Extract section/page from path like {wiki_id}/{repo}/wiki_pages/{section}/{page}.md
        parts = artifact_path.split("/wiki_pages/")
        if len(parts) == 2:
            rel_path = parts[1]  # section/page.md
            page_id = rel_path.replace(".md", "")
            page_name = page_id.split("/")[-1]
            section_name = page_id.split("/")[0] if "/" in page_id else ""
        else:
            page_id = artifact_path.rsplit("/", 1)[-1].replace(".md", "")
            page_name = page_id
            section_name = ""

        try:
            content = await management.storage.download("wiki_artifacts", artifact_path)
            content_str = content.decode("utf-8") if isinstance(content, bytes) else str(content)
        except Exception:
            content_str = ""

        # Prefer frontmatter title/section over filename-derived values
        fm_title, fm_section = _extract_frontmatter_title_section(content_str)
        display_title = fm_title or page_name.replace("-", " ").replace("_", " ").title().strip()
        display_section = fm_section or section_name.replace("-", " ").replace("_", " ").title().strip()

        pages.append(
            {
                "id": page_id,
                "title": display_title,
                "section": display_section,
                "order": i,
                "content": content_str,
            }
        )

    response = {
        "wiki_id": wiki_meta.wiki_id,
        "repo_url": wiki_meta.repo_url,
        "branch": wiki_meta.branch,
        "title": wiki_title,
        "page_count": len(pages),
        "created_at": wiki_meta.created_at.isoformat(),
        "indexed_at": wiki_meta.indexed_at.isoformat() if wiki_meta.indexed_at else None,
        "commit_hash": wiki_meta.commit_hash,
        "sections": [
            {"name": s.get("section_name", ""), "pages": [p.get("page_name", "") for p in s.get("pages", [])]}
            for s in sections
        ],
        "pages": pages,
        "status": wiki_meta.status or "complete",
        "requires_token": wiki_meta.requires_token,
        "error": wiki_meta.error,
    }
    if active_invocation:
        response["invocation_id"] = active_invocation.id
        # Only let the in-memory invocation override DB status if it is still generating.
        # DB is the source of truth for terminal states (complete/failed/partial).
        # Without this guard, a stale failed invocation from a prior attempt (within 1hr TTL)
        # would shadow a successful retry and display "failed" after page refresh.
        if active_invocation.status == "generating":
            response["status"] = active_invocation.status
    return response


@router.get("/wikis/{wiki_id}/pages/{page_id:path}")
async def get_wiki_page(
    wiki_id: str,
    page_id: str,
    user: CurrentUser = Depends(get_current_user),
    management: WikiManagementService = Depends(get_wiki_management),
) -> dict:
    """Get a single wiki page content. page_id can be section/page format."""
    # Access control — verify caller can view this wiki
    user_id = user.id if user else None
    wiki_record = await management.get_wiki(wiki_id, user_id=user_id)
    if wiki_record is None:
        raise HTTPException(404, f"Wiki not found: {wiki_id}")
    # Search all artifacts for matching page
    all_artifacts = await management.storage.list_artifacts("wiki_artifacts", prefix=wiki_id)
    name = page_id if page_id.endswith(".md") else f"{page_id}.md"
    # Try exact match, then search in wiki_pages/
    candidates = [a for a in all_artifacts if a.endswith(name)]
    if not candidates:
        raise HTTPException(404, f"Page not found: {wiki_id}/{page_id}")
    try:
        content = await management.storage.download("wiki_artifacts", candidates[0])
        page_name = page_id.split("/")[-1] if "/" in page_id else page_id
        return {
            "wiki_id": wiki_id,
            "page_id": page_id,
            "title": page_name.replace("-", " ").replace("_", " ").replace(".md", "").title(),
            "content": content.decode("utf-8") if isinstance(content, bytes) else str(content),
        }
    except FileNotFoundError as e:
        raise HTTPException(404, f"Page not found: {wiki_id}/{page_id}") from e


@router.post("/wikis/{wiki_id}/resume", response_model=GenerateWikiResponse, status_code=202)
async def resume_wiki(
    wiki_id: str,
    user: CurrentUser = Depends(get_current_user),
    service: WikiService = Depends(get_wiki_service),
    management: WikiManagementService = Depends(get_wiki_management),
) -> GenerateWikiResponse:
    user_id = user.id if user else ""
    # Ownership check — only the owner can resume
    wiki_record = await management.get_wiki(wiki_id, user_id=user_id or None)
    if wiki_record and wiki_record.owner_id and wiki_record.owner_id != user_id:
        raise HTTPException(403, "Only the wiki owner can perform this action")
    invocation = await service.resume(wiki_id, management, owner_id=user_id)
    if not invocation:
        raise HTTPException(404, f"Wiki not found or generation already in progress: {wiki_id}")
    return GenerateWikiResponse(
        wiki_id=invocation.wiki_id,
        invocation_id=invocation.id,
        status=invocation.status,
        message=f"Resume started. Track via GET /api/v1/invocations/{invocation.id}",
    )


# ---------------------------------------------------------------------------
# Q&A Knowledge Flywheel endpoints
# ---------------------------------------------------------------------------


@router.get("/wikis/{wiki_id}/qa", response_model=QAListResponse)
async def list_qa(
    wiki_id: str,
    user: CurrentUser = Depends(get_current_user),
    qa_service: QAService = Depends(get_qa_service),
    management: WikiManagementService = Depends(get_wiki_management),
    limit: int = Query(default=20, le=100),
    offset: int = Query(default=0, ge=0),
    status: QAStatus | None = Query(default=None),
) -> QAListResponse:
    """Paginated Q&A history for a wiki."""
    user_id = user.id if user else None
    wiki = await management.get_wiki(wiki_id, user_id=user_id)
    if not wiki:
        raise HTTPException(404, f"Wiki not found: {wiki_id}")
    records, total = await qa_service.list_qa(wiki_id, status=status, limit=limit, offset=offset)
    items = [QARecordResponse.model_validate(r) for r in records]
    return QAListResponse(items=items, total=total, limit=limit, offset=offset)


@router.get("/wikis/{wiki_id}/qa/stats", response_model=QAStatsResponse)
async def qa_stats(
    wiki_id: str,
    user: CurrentUser = Depends(get_current_user),
    qa_service: QAService = Depends(get_qa_service),
    management: WikiManagementService = Depends(get_wiki_management),
) -> QAStatsResponse:
    """QA statistics for a wiki."""
    user_id = user.id if user else None
    wiki = await management.get_wiki(wiki_id, user_id=user_id)
    if not wiki:
        raise HTTPException(404, f"Wiki not found: {wiki_id}")
    stats = await qa_service.get_stats(wiki_id)
    return QAStatsResponse(**stats)


# ---------------------------------------------------------------------------
# Wiki lifecycle
# ---------------------------------------------------------------------------


@router.post("/wikis/{wiki_id}/refresh", response_model=GenerateWikiResponse, status_code=202)
async def refresh_wiki(
    wiki_id: str,
    request: Request,
    body: RefreshWikiRequest = RefreshWikiRequest(),
    user: CurrentUser = Depends(get_current_user),
    service: WikiService = Depends(get_wiki_service),
    management: WikiManagementService = Depends(get_wiki_management),
) -> GenerateWikiResponse:
    user_id = user.id if user else ""
    # Ownership check — only the owner can refresh
    wiki_record = await management.get_wiki(wiki_id, user_id=user_id or None)
    if wiki_record and wiki_record.owner_id and wiki_record.owner_id != user_id:
        raise HTTPException(403, "Only the wiki owner can perform this action")

    invocation = await service.refresh(wiki_id, management, owner_id=user_id, access_token=body.access_token)
    if not invocation:
        raise HTTPException(404, f"Wiki not found: {wiki_id}")

    # Evict Ask/Research component caches so stale indexes aren't served post-refresh
    ask_service: AskService = request.app.state.ask_service
    research_service: ResearchService = request.app.state.research_service
    ask_service.evict_cache(wiki_id)
    research_service.evict_cache(wiki_id)

    # Invalidate QA cache only after refresh is confirmed to have started
    qa_service: QAService = request.app.state.qa_service
    await qa_service.invalidate_wiki(wiki_id, delete=False)

    return GenerateWikiResponse(
        wiki_id=invocation.wiki_id,
        invocation_id=invocation.id,
        status=invocation.status,
        message=f"Refresh started. Track via GET /api/v1/invocations/{invocation.id}",
    )


@router.delete("/wikis/{wiki_id}", response_model=DeleteWikiResponse)
async def delete_wiki(
    wiki_id: str,
    request: Request,
    user: CurrentUser = Depends(get_current_user),
    management: WikiManagementService = Depends(get_wiki_management),
    service: WikiService = Depends(get_wiki_service),
) -> DeleteWikiResponse:
    user_id = user.id if user else None
    settings = request.app.state.settings
    result = await management.delete_wiki(wiki_id, user_id=user_id, cache_dir=settings.cache_dir)

    if not result.deleted and result.message and "owner" in result.message:
        raise HTTPException(403, result.message)

    # Also purge any invocations for this wiki (covers failed/partial wikis
    # that are not yet registered but still appear in the dashboard list)
    inv_ids = [inv_id for inv_id, inv in service.invocations.items() if inv.wiki_id == wiki_id]
    for inv_id in inv_ids:
        await service.cancel_invocation(inv_id)
        service.remove_invocation(inv_id)

    # Persist invocation removals to disk
    if inv_ids:
        await service.persist_invocations()

    # Evict Ask/Research component caches so stale indexes aren't served
    ask_service: AskService = request.app.state.ask_service
    research_service: ResearchService = request.app.state.research_service
    ask_service.evict_cache(wiki_id)
    research_service.evict_cache(wiki_id)

    # Invalidate QA data: remove all QARecords and FAISS file
    qa_service: QAService = request.app.state.qa_service
    await qa_service.invalidate_wiki(wiki_id, delete=True)

    if not result.deleted and not inv_ids:
        raise HTTPException(404, f"Wiki not found: {wiki_id}")
    return DeleteWikiResponse(deleted=True, wiki_id=wiki_id)


@router.patch("/wikis/{wiki_id}/visibility", response_model=WikiSummary)
async def update_wiki_visibility(
    wiki_id: str,
    body: UpdateWikiVisibilityRequest,
    user: CurrentUser = Depends(get_current_user),
    management: WikiManagementService = Depends(get_wiki_management),
) -> WikiSummary:
    """Change a wiki's visibility between 'personal' and 'shared'."""
    if body.visibility not in ("personal", "shared"):
        raise HTTPException(422, "visibility must be 'personal' or 'shared'")
    user_id = user.id if user else ""
    updated = await management.update_visibility(wiki_id, user_id=user_id, visibility=body.visibility)
    if updated is None:
        # Either wiki not found or caller is not the owner
        wiki_record = await management.get_wiki(wiki_id, user_id=user_id or None)
        if wiki_record is None:
            raise HTTPException(404, f"Wiki not found: {wiki_id}")
        raise HTTPException(403, "Only the wiki owner can change visibility")
    return updated


# ---------------------------------------------------------------------------
# Projects
# ---------------------------------------------------------------------------


def _project_response(project, wiki_count: int = 0) -> ProjectResponse:
    return ProjectResponse(
        id=project.id,
        name=project.name,
        description=project.description,
        visibility=project.visibility,
        owner_id=project.owner_id,
        created_at=project.created_at,
        wiki_count=wiki_count,
    )


@router.post("/projects", response_model=ProjectResponse, status_code=201)
async def create_project(
    body: ProjectCreateRequest,
    user: CurrentUser = Depends(get_current_user),
    svc: ProjectService = Depends(get_project_service),
) -> ProjectResponse:
    project = await svc.create_project(
        owner_id=user.id,
        name=body.name,
        description=body.description,
        visibility=body.visibility,
    )
    return _project_response(project)


@router.get("/projects", response_model=ProjectListResponse)
async def list_projects(
    user: CurrentUser = Depends(get_current_user),
    svc: ProjectService = Depends(get_project_service),
) -> ProjectListResponse:
    projects = await svc.list_projects(user_id=user.id)
    counts = await svc.batch_get_wiki_counts([p.id for p in projects])
    items = [_project_response(p, counts.get(p.id, 0)) for p in projects]
    return ProjectListResponse(projects=items)


@router.get("/projects/{project_id}", response_model=ProjectResponse)
async def get_project(
    project_id: str,
    user: CurrentUser = Depends(get_current_user),
    svc: ProjectService = Depends(get_project_service),
) -> ProjectResponse:
    project = await svc.get_project(project_id, user_id=user.id)
    if project is None:
        raise HTTPException(404, f"Project not found: {project_id}")
    count = await svc.get_wiki_count(project_id)
    return _project_response(project, count)


@router.patch("/projects/{project_id}", response_model=ProjectResponse)
async def update_project(
    project_id: str,
    body: ProjectUpdateRequest,
    user: CurrentUser = Depends(get_current_user),
    svc: ProjectService = Depends(get_project_service),
) -> ProjectResponse:
    # First verify project exists and is accessible (to distinguish 404 vs 403)
    existing = await svc.get_project(project_id, user_id=user.id)
    if existing is None:
        raise HTTPException(404, f"Project not found: {project_id}")

    fields = {k: v for k, v in body.model_dump(exclude_none=True).items()}
    updated = await svc.update_project(project_id, owner_id=user.id, **fields)
    if updated is None:
        raise HTTPException(403, "Only the project owner can modify this project")
    count = await svc.get_wiki_count(project_id)
    return _project_response(updated, count)


@router.delete("/projects/{project_id}", status_code=200)
async def delete_project(
    project_id: str,
    user: CurrentUser = Depends(get_current_user),
    svc: ProjectService = Depends(get_project_service),
) -> JSONResponse:
    # Check existence first
    existing = await svc.get_project(project_id, user_id=user.id)
    if existing is None:
        raise HTTPException(404, f"Project not found: {project_id}")
    deleted = await svc.delete_project(project_id, owner_id=user.id)
    if not deleted:
        raise HTTPException(403, "Only the project owner can delete this project")
    return JSONResponse({"deleted": True, "project_id": project_id})


@router.post("/projects/{project_id}/wikis", response_model=ProjectResponse, status_code=201)
async def add_wiki_to_project(
    project_id: str,
    body: ProjectAddWikiRequest,
    user: CurrentUser = Depends(get_current_user),
    svc: ProjectService = Depends(get_project_service),
) -> ProjectResponse:
    existing = await svc.get_project(project_id, user_id=user.id)
    if existing is None:
        raise HTTPException(404, f"Project not found: {project_id}")
    if existing.owner_id != user.id:
        raise HTTPException(403, "Only the project owner can add wikis")
    result = await svc.add_wiki(
        project_id=project_id,
        wiki_id=body.wiki_id,
        owner_id=user.id,
        added_by=user.id,
    )
    if result is None and existing.owner_id == user.id:
        # Wiki already in project — idempotent, return current state
        pass
    count = await svc.get_wiki_count(project_id)
    return _project_response(existing, count)


@router.delete("/projects/{project_id}/wikis/{wiki_id}", status_code=200)
async def remove_wiki_from_project(
    project_id: str,
    wiki_id: str,
    user: CurrentUser = Depends(get_current_user),
    svc: ProjectService = Depends(get_project_service),
) -> JSONResponse:
    existing = await svc.get_project(project_id, user_id=user.id)
    if existing is None:
        raise HTTPException(404, f"Project not found: {project_id}")
    removed = await svc.remove_wiki(project_id, wiki_id=wiki_id, owner_id=user.id)
    if not removed:
        raise HTTPException(403, "Only the project owner can remove wikis")
    return JSONResponse({"removed": True, "project_id": project_id, "wiki_id": wiki_id})


@router.get("/projects/{project_id}/wikis", response_model=WikiListResponse)
async def list_project_wikis(
    project_id: str,
    user: CurrentUser = Depends(get_current_user),
    svc: ProjectService = Depends(get_project_service),
    management: WikiManagementService = Depends(get_wiki_management),
) -> WikiListResponse:
    wikis = await svc.list_project_wikis(project_id, user_id=user.id)
    if wikis is None:
        raise HTTPException(404, f"Project not found: {project_id}")
    summaries = [
        WikiManagementService._record_to_summary(w, user.id)
        for w in wikis
    ]
    return WikiListResponse(wikis=summaries)


class _ProjectCodeMapRequest(BaseModel):
    """Request body for the project code-map endpoint."""

    question: str


@router.post("/projects/{project_id}/map", response_model=ResearchResponse)
async def project_codemap(
    project_id: str,
    body: _ProjectCodeMapRequest,
    user: CurrentUser = Depends(get_current_user),
    svc: ProjectService = Depends(get_project_service),
    service: ResearchService = Depends(get_research_service),
    accept: str = "application/json",
) -> ResearchResponse | StreamingResponse:
    """Build a code-map for all wikis in a project.

    Proxies to the codemap pipeline with ``project_id`` set and
    ``research_type=codemap``.
    """
    user_id = user.id if user else None

    # Verify the project exists and is accessible
    project = await svc.get_project(project_id, user_id=user_id or "")
    if project is None:
        raise HTTPException(404, f"Project not found: {project_id}")

    request = ResearchRequest(
        project_id=project_id,
        question=body.question,
        research_type="codemap",
    )

    try:
        if "text/event-stream" in accept:
            async def codemap_sse():
                async for event in service.codemap_stream(request, user_id=user_id):
                    event_type = event.get("event_type", "message")
                    yield f"event: {event_type}\ndata: {_json.dumps(event.get('data', {}))}\n\n"
            return StreamingResponse(codemap_sse(), media_type="text/event-stream")
        return await service.codemap_sync(request, user_id=user_id)
    except FileNotFoundError as e:
        raise HTTPException(404, str(e)) from e
    except ValueError as e:
        raise HTTPException(400, str(e)) from e
    except RuntimeError as e:
        raise HTTPException(501, str(e)) from e


# ---------------------------------------------------------------------------
# Health
# ---------------------------------------------------------------------------


@router.get("/health", response_model=HealthResponse)
async def health(request: Request) -> HealthResponse:
    ph = getattr(request.app.state, "provider_health", None)
    services = {}
    if ph:
        services["llm"] = ph.llm
        services["embeddings"] = ph.embeddings
    overall = "ok" if all(v == "ok" for v in services.values()) else "degraded" if services else "ok"
    return HealthResponse(status=overall, version="0.1.0", services=services)
