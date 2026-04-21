from __future__ import annotations

import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    from app.config import get_settings
    from app.db import create_tables, dispose_engine, get_engine, get_session_factory, init_db
    from app.services.ask_service import AskService
    from app.services.export_service import ExportService
    from app.services.import_service import ImportService
    from app.services.research_service import ResearchService
    from app.services.wiki_management import WikiManagementService
    from app.services.wiki_service import WikiService
    from app.storage import get_storage

    settings = get_settings()
    storage = get_storage(settings)

    # Initialise async database
    init_db(settings.database_url)
    await create_tables(get_engine())

    app.state.settings = settings
    app.state.storage = storage
    app.state.session_factory = get_session_factory()
    wiki_management = WikiManagementService(storage, get_session_factory())
    app.state.wiki_management = wiki_management
    app.state.wiki_service = WikiService(settings, storage, wiki_management=wiki_management)
    # QA Knowledge Flywheel: init QACacheManager + QAService (degraded-mode)
    from app.services.qa_cache_manager import QACacheManager
    from app.services.qa_service import QAService

    qa_cache: QACacheManager | None = None
    if settings.qa_cache_enabled:
        try:
            from app.services.llm_factory import create_embeddings

            embeddings = create_embeddings(settings)
            qa_cache = QACacheManager(settings.cache_dir, embeddings, max_wikis=settings.qa_cache_max_wikis)
            logger.info("QA cache initialized")
        except Exception as e:
            logger.error("QA cache disabled — embedding init failed: %s", e)

    qa_service = QAService(get_session_factory(), qa_cache, settings)
    app.state.qa_service = qa_service

    from app.core.wiki_page_index import WikiPageIndexCache

    app.state.wiki_index_cache = WikiPageIndexCache(storage, max_wikis=settings.wiki_index_cache_max_wikis)

    app.state.ask_service = AskService(settings, storage, qa_service=qa_service)
    app.state.research_service = ResearchService(settings, storage)
    app.state.export_service = ExportService(storage, wiki_management, settings)
    app.state.import_service = ImportService(storage, wiki_management, settings)

    # Load persisted invocations from storage
    await app.state.wiki_service.load_persisted_invocations()

    logger.info("Services initialized: wiki, ask, research, wiki_management, qa, export, import")

    # Wire MCP tools to services (direct calls, no HTTP)
    from mcp_server.server import mcp as mcp_server
    from mcp_server.server import set_services

    set_services(
        wiki_management=wiki_management,
        ask_service=app.state.ask_service,
        research_service=app.state.research_service,
        storage=storage,
        settings=settings,
        qa_service=qa_service,
        session_factory=get_session_factory(),
    )
    logger.info("MCP tools wired to services")

    # Provider health check (non-blocking)
    from app.services.health_check import check_providers

    app.state.provider_health = await check_providers(settings)

    # Run FastMCP session manager for the duration of the app (required for streamable HTTP)
    async with mcp_server.session_manager.run():
        yield

    logger.info("Shutting down")
    await dispose_engine()


def create_app() -> FastAPI:
    # Reset MCP session manager so each app instance gets a fresh one
    # (the MCP library's StreamableHTTPSessionManager forbids re-entry)
    from mcp_server.server import mcp as _mcp

    _mcp._session_manager = None

    app = FastAPI(
        title="Wikis API",
        version="0.1.0",
        description="AI-powered repository documentation generator",
        lifespan=lifespan,
    )

    # CORS: allow the consolidated web app and localhost dev
    from app.config import Settings

    _settings = Settings()
    allowed_origins = [
        _settings.auth_jwks_url.rsplit("/api/", 1)[0],  # e.g. http://app:3000
        "http://localhost:3000",
    ]
    app.add_middleware(
        CORSMiddleware,
        allow_origins=allowed_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.get("/health")
    async def health(request: Request) -> dict[str, object]:
        from fastapi.responses import JSONResponse

        from app.services.health_check import check_providers

        provider_health = getattr(request.app.state, "provider_health", None)
        if provider_health is not None and provider_health.stale:
            provider_health = await check_providers(request.app.state.settings)
            request.app.state.provider_health = provider_health

        if provider_health is not None and not provider_health.healthy:
            return JSONResponse(
                status_code=503,
                content={
                    "status": "unhealthy",
                    "llm": provider_health.llm,
                    "embeddings": provider_health.embeddings,
                },
            )
        return {"status": "ok"}

    from app.api.routes import router

    app.include_router(router)

    # Mount MCP server at /mcp (with auth middleware)
    from mcp_server.server import get_mcp_app

    app.mount("/mcp", get_mcp_app())

    return app


app = create_app()
