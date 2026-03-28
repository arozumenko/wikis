"""Shared fixtures for backend tests."""

from __future__ import annotations

import contextlib
import os
from unittest.mock import patch

import pytest
from httpx import ASGITransport, AsyncClient
from pydantic import SecretStr

from app.config import Settings
from app.main import create_app
from app.storage.local import LocalArtifactStorage

# Disable JWT auth for all tests by default
os.environ.setdefault("AUTH_ENABLED", "false")

# Override DATABASE_URL so tests never try to reach an external PostgreSQL server.
# An empty string makes db._convert_database_url() fall back to a local SQLite file.
os.environ["DATABASE_URL"] = ""


@pytest.fixture
def mock_settings() -> Settings:
    """Settings with safe defaults for testing."""
    return Settings(
        llm_api_key=SecretStr("test-key"),
        llm_provider="openai",
        llm_model="gpt-4o-mini",
        storage_backend="local",
        auth_enabled=False,
    )


@pytest.fixture
def mock_storage(tmp_path) -> LocalArtifactStorage:
    """Local artifact storage backed by a temporary directory."""
    return LocalArtifactStorage(str(tmp_path))


@contextlib.asynccontextmanager
async def _noop_run(self):
    """No-op replacement for StreamableHTTPSessionManager.run().

    pytest-asyncio runs fixture setup/teardown in separate tasks, which
    conflicts with anyio's cancel-scope requirement that entry and exit
    happen in the same task.  REST API tests don't need the MCP session
    manager, so we bypass it here.
    """
    yield


@pytest.fixture(autouse=True)
def _patch_mcp_session_manager():
    """Globally disable MCP session manager in all tests."""
    from mcp.server.streamable_http_manager import StreamableHTTPSessionManager

    with patch.object(StreamableHTTPSessionManager, "run", _noop_run):
        yield


@pytest.fixture
async def test_app():
    """Create the FastAPI app and run its lifespan."""
    app = create_app()
    async with app.router.lifespan_context(app):
        yield app


@pytest.fixture
async def client(test_app):
    """Async HTTP client wired to the test app."""
    async with AsyncClient(
        transport=ASGITransport(app=test_app),
        base_url="http://test",
    ) as c:
        yield c
