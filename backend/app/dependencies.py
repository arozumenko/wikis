"""FastAPI dependency injection — service accessors via app.state."""

from __future__ import annotations

from fastapi import Depends, Request
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.wiki_page_index import WikiPageIndexCache
from app.services.ask_service import AskService
from app.services.export_service import ExportService
from app.services.import_service import ImportService
from app.services.project_service import ProjectService
from app.services.qa_service import QAService
from app.services.research_service import ResearchService
from app.services.wiki_management import WikiManagementService
from app.services.wiki_service import WikiService


def get_wiki_service(request: Request) -> WikiService:
    return request.app.state.wiki_service


def get_ask_service(request: Request) -> AskService:
    return request.app.state.ask_service


def get_qa_service(request: Request) -> QAService:
    return request.app.state.qa_service


def get_research_service(request: Request) -> ResearchService:
    return request.app.state.research_service


def get_wiki_management(request: Request) -> WikiManagementService:
    return request.app.state.wiki_management


async def get_db_session(request: Request) -> AsyncSession:
    """Yield an AsyncSession from the app-level session factory."""
    from app.db import get_session_factory

    session_factory = request.app.state.session_factory
    async with session_factory() as session:
        async with session.begin():
            yield session


async def get_project_service(session: AsyncSession = Depends(get_db_session)) -> ProjectService:
    """Provide a ProjectService backed by the current request's DB session."""
    return ProjectService(session)


def get_export_service(request: Request) -> ExportService:
    return request.app.state.export_service


def get_import_service(request: Request) -> ImportService:
    return request.app.state.import_service


def get_wiki_index_cache(request: Request) -> WikiPageIndexCache:
    return request.app.state.wiki_index_cache

