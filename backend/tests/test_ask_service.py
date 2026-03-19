"""Tests for Ask/Q&A service and endpoint."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from httpx import ASGITransport, AsyncClient
from pydantic import SecretStr

from app.config import Settings
from app.main import create_app
from app.models.api import AskRequest, AskResponse, ChatMessage
from app.services.ask_service import AskService
from app.storage.local import LocalArtifactStorage


def _settings() -> Settings:
    return Settings(llm_api_key=SecretStr("test-key"))


@pytest.fixture
def storage(tmp_path):
    return LocalArtifactStorage(str(tmp_path))


@pytest.fixture
def service(storage):
    return AskService(_settings(), storage)


@pytest.fixture
def ask_request():
    return AskRequest(wiki_id="w1", question="How does auth work?")


async def _mock_ask_events(*args, **kwargs):
    """Simulate AskEngine.ask() yielding events."""
    yield {"event_type": "ask_start", "data": {"session_id": "s1"}}
    yield {"event_type": "thinking_step", "data": {"tool": "search", "result": "found symbols"}}
    yield {
        "event_type": "ask_complete",
        "data": {
            "answer": "Auth uses JWT tokens.",
            "sources": [{"file_path": "src/auth.py", "line_start": 10, "snippet": "def verify():"}],
        },
    }


class TestAskServiceUnit:
    @pytest.mark.asyncio
    async def test_ask_sync_returns_response(self, service, ask_request):
        mock_engine = MagicMock()
        mock_engine.ask = _mock_ask_events

        with (
            patch.object(
                service,
                "_get_components",
                new_callable=AsyncMock,
                return_value=MagicMock(
                    retriever_stack=None, graph_manager=None, code_graph=None, repo_analysis=None, llm=MagicMock()
                ),
            ),
            patch("app.core.ask_engine.AskEngine", return_value=mock_engine),
        ):
            result = await service.ask_sync(ask_request)

        assert isinstance(result, AskResponse)
        assert result.answer == "Auth uses JWT tokens."
        assert len(result.sources) == 1
        assert result.sources[0].file_path == "src/auth.py"

    @pytest.mark.asyncio
    async def test_ask_stream_yields_events(self, service, ask_request):
        mock_engine = MagicMock()
        mock_engine.ask = _mock_ask_events

        with (
            patch.object(
                service,
                "_get_components",
                new_callable=AsyncMock,
                return_value=MagicMock(
                    retriever_stack=None, graph_manager=None, code_graph=None, repo_analysis=None, llm=MagicMock()
                ),
            ),
            patch("app.core.ask_engine.AskEngine", return_value=mock_engine),
        ):
            events = []
            async for event in service.ask_stream(ask_request):
                events.append(event)

        assert len(events) == 3
        assert events[0]["event_type"] == "ask_start"
        assert events[2]["event_type"] == "ask_complete"

    @pytest.mark.asyncio
    async def test_ask_with_chat_history(self, service):
        req = AskRequest(
            wiki_id="w1",
            question="Follow up",
            chat_history=[ChatMessage(role="user", content="First question")],
        )
        mock_engine = MagicMock()
        mock_engine.ask = _mock_ask_events

        with (
            patch.object(
                service,
                "_get_components",
                new_callable=AsyncMock,
                return_value=MagicMock(
                    retriever_stack=None, graph_manager=None, code_graph=None, repo_analysis=None, llm=MagicMock()
                ),
            ),
            patch("app.core.ask_engine.AskEngine", return_value=mock_engine),
        ):
            result = await service.ask_sync(req)

        assert result.answer == "Auth uses JWT tokens."


class TestAskEndpoint:
    @pytest.fixture
    async def client(self):
        app = create_app()
        async with app.router.lifespan_context(app):
            async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as c:
                yield c, app

    @pytest.mark.asyncio
    async def test_ask_json_mode(self, client):
        c, app = client
        mock_response = AskResponse(answer="JWT tokens.", sources=[])
        with patch.object(
            app.state.ask_service,
            "ask_sync",
            new_callable=AsyncMock,
            return_value=mock_response,
        ):
            resp = await c.post("/api/v1/ask", json={"wiki_id": "w1", "question": "How?"})
            assert resp.status_code == 200
            assert resp.json()["answer"] == "JWT tokens."

    @pytest.mark.asyncio
    async def test_ask_validation_error(self, client):
        c, _ = client
        resp = await c.post("/api/v1/ask", json={})
        assert resp.status_code == 422
