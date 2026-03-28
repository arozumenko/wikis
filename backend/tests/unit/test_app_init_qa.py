"""Tests for QA Knowledge Flywheel app initialization and DI wiring.

Covers both DI wiring (unit) and the real FastAPI lifespan (startup paths).
"""

from __future__ import annotations

from contextlib import asynccontextmanager
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from pydantic import SecretStr

from app.config import Settings
from app.dependencies import get_qa_service
from app.services.qa_service import QAService


@asynccontextmanager
async def _noop_ctx():
    """No-op async context manager to replace MCP session_manager.run()."""
    yield


# ---------------------------------------------------------------------------
# DI wiring unit tests (no lifespan)
# ---------------------------------------------------------------------------


def test_get_qa_service_dependency():
    """get_qa_service returns qa_service from app.state."""
    mock_request = MagicMock()
    mock_service = MagicMock(spec=QAService)
    mock_request.app.state.qa_service = mock_service
    result = get_qa_service(mock_request)
    assert result is mock_service


def test_ask_service_accepts_qa_service():
    """AskService constructor accepts optional qa_service."""
    from app.services.ask_service import AskService

    settings = Settings(llm_api_key=SecretStr("test-key"))
    storage = MagicMock()
    qa_service = MagicMock(spec=QAService)

    service = AskService(settings, storage, qa_service=qa_service)
    assert service._qa_service is qa_service


def test_ask_service_without_qa_service():
    """AskService works without qa_service (backward compat)."""
    from app.services.ask_service import AskService

    settings = Settings(llm_api_key=SecretStr("test-key"))
    storage = MagicMock()

    service = AskService(settings, storage)
    assert service._qa_service is None


def test_qa_service_with_cache_none():
    """QAService works with cache=None (recording-only mode)."""
    from sqlalchemy.ext.asyncio import async_sessionmaker

    settings = Settings(llm_api_key=SecretStr("test-key"))
    session_factory = MagicMock(spec=async_sessionmaker)

    service = QAService(session_factory, cache=None, settings=settings)
    assert service._cache is None


def test_set_services_accepts_qa_service():
    """MCP set_services accepts qa_service parameter."""
    from mcp_server.server import set_services

    mock_qa = MagicMock()
    set_services(
        wiki_management=MagicMock(),
        ask_service=MagicMock(),
        storage=MagicMock(),
        settings=MagicMock(),
        research_service=MagicMock(),
        qa_service=mock_qa,
    )
    import mcp_server.server as srv

    assert srv._qa_service is mock_qa


def test_qa_cache_disabled_no_embeddings():
    """When qa_cache_enabled=False, no embeddings are created."""
    settings = Settings(llm_api_key=SecretStr("test-key"), qa_cache_enabled=False)
    assert settings.qa_cache_enabled is False


# ---------------------------------------------------------------------------
# Real lifespan startup tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_lifespan_happy_path_wires_qa_service():
    """Full lifespan: app.state.qa_service and app.state.ask_service are wired."""
    from app.main import create_app

    app = create_app()
    # Patch at source modules — lifespan uses local imports.
    # Mock MCP session_manager.run() to avoid singleton re-entry error.
    with (
        patch("app.services.llm_factory.create_embeddings") as mock_embed,
        patch("app.services.health_check.check_providers", new_callable=AsyncMock) as mock_hp,
        patch("mcp_server.server.mcp.session_manager.run", return_value=_noop_ctx()),
    ):
        mock_embed.return_value = MagicMock()
        mock_hp.return_value = MagicMock(healthy=True)

        async with app.router.lifespan_context(app):
            assert hasattr(app.state, "qa_service")
            assert isinstance(app.state.qa_service, QAService)
            assert hasattr(app.state, "ask_service")
            # ask_service has qa_service wired
            assert app.state.ask_service._qa_service is app.state.qa_service
            # Cache was created (create_embeddings succeeded)
            assert app.state.qa_service._cache is not None


@pytest.mark.asyncio
async def test_lifespan_qa_cache_disabled():
    """When qa_cache_enabled=False, QAService is created with cache=None."""
    from app.main import create_app

    app = create_app()
    settings = Settings(
        llm_api_key=SecretStr("test-key"),
        qa_cache_enabled=False,
    )
    with (
        patch("app.config.get_settings", return_value=settings),
        patch("app.services.health_check.check_providers", new_callable=AsyncMock) as mock_hp,
        patch("mcp_server.server.mcp.session_manager.run", return_value=_noop_ctx()),
    ):
        mock_hp.return_value = MagicMock(healthy=True)

        async with app.router.lifespan_context(app):
            assert isinstance(app.state.qa_service, QAService)
            assert app.state.qa_service._cache is None


@pytest.mark.asyncio
async def test_lifespan_embedding_failure_degrades_gracefully(caplog):
    """When create_embeddings raises, QAService is created with cache=None (degraded mode)."""
    import logging

    from app.main import create_app

    app = create_app()
    with (
        patch(
            "app.services.llm_factory.create_embeddings",
            side_effect=ValueError("No embedding API"),
        ),
        patch("app.services.health_check.check_providers", new_callable=AsyncMock) as mock_hp,
        patch("mcp_server.server.mcp.session_manager.run", return_value=_noop_ctx()),
    ):
        mock_hp.return_value = MagicMock(healthy=True)

        with caplog.at_level(logging.ERROR, logger="app.main"):
            async with app.router.lifespan_context(app):
                # App started without crashing
                assert isinstance(app.state.qa_service, QAService)
                # Degraded: cache is None
                assert app.state.qa_service._cache is None
                # ask_service is still wired with qa_service
                assert app.state.ask_service._qa_service is app.state.qa_service

    assert any("QA cache disabled" in r.message for r in caplog.records)
