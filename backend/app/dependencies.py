"""FastAPI dependency injection — service accessors via app.state."""

from __future__ import annotations

from fastapi import Request

from app.services.ask_service import AskService
from app.services.research_service import ResearchService
from app.services.wiki_management import WikiManagementService
from app.services.wiki_service import WikiService


def get_wiki_service(request: Request) -> WikiService:
    return request.app.state.wiki_service


def get_ask_service(request: Request) -> AskService:
    return request.app.state.ask_service


def get_research_service(request: Request) -> ResearchService:
    return request.app.state.research_service


def get_wiki_management(request: Request) -> WikiManagementService:
    return request.app.state.wiki_management
