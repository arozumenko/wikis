"""Shared fixtures for backend tests."""

from __future__ import annotations

import os

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
