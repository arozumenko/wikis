"""Tests for AskService QA integration."""

from __future__ import annotations

import asyncio
import json
import uuid
from unittest.mock import AsyncMock, MagicMock, patch

import numpy as np
import pytest
from pydantic import SecretStr

from app.config import Settings
from app.models.api import AskRequest, AskResponse, ChatMessage, SourceReference
from app.models.qa_api import AskResult, QARecordingPayload
from app.services.ask_service import AskService


@pytest.fixture
def settings():
    return Settings(llm_api_key=SecretStr("test-key"))


@pytest.fixture
def mock_storage():
    return MagicMock()


@pytest.fixture
def mock_qa_service():
    service = AsyncMock()
    service.lookup_cache = AsyncMock(return_value=(None, None))
    service.record_interaction = AsyncMock()
    return service


@pytest.fixture
def ask_service(settings, mock_storage, mock_qa_service):
    return AskService(settings, mock_storage, qa_service=mock_qa_service)


@pytest.fixture
def ask_service_no_qa(settings, mock_storage):
    return AskService(settings, mock_storage)


def _make_request(wiki_id="wiki-1", question="What is auth?", chat_history=None):
    return AskRequest(wiki_id=wiki_id, question=question, chat_history=chat_history or [])


# --- ask_sync tests ---


@pytest.mark.asyncio
async def test_ask_sync_cache_hit(ask_service, mock_qa_service):
    """Cache hit returns AskResult with cached response."""
    cached = MagicMock()
    cached.id = "original-qa-id"
    cached.answer = "Cached answer"
    cached.sources_json = json.dumps([{"file_path": "auth.py", "snippet": "def login()"}])
    cached.tool_steps = 2
    cached.mode = "fast"
    cached.source_commit_hash = "abc123"
    mock_qa_service.lookup_cache.return_value = (cached, None)

    result = await ask_service.ask_sync(_make_request())
    assert isinstance(result, AskResult)
    assert result.response.answer == "Cached answer"
    assert result.recording is not None
    assert result.recording.is_cache_hit is True
    assert result.recording.source_qa_id == "original-qa-id"


@pytest.mark.asyncio
async def test_ask_sync_cache_miss(ask_service, mock_qa_service):
    """Cache miss runs agent and returns AskResult with recording including commit hash."""
    mock_qa_service.lookup_cache.return_value = (None, np.ones(8))
    mock_qa_service.get_wiki_commit_hash = AsyncMock(return_value="abc123")

    async def fake_stream(request, **kwargs):
        yield {
            "event_type": "task_complete",
            "data": {"answer": "Agent answer", "sources": [], "steps": 3},
        }

    with patch.object(ask_service, "_stream_from_agent", side_effect=fake_stream):
        result = await ask_service.ask_sync(_make_request())

    assert isinstance(result, AskResult)
    assert result.response.answer == "Agent answer"
    assert result.recording is not None
    assert result.recording.is_cache_hit is False
    assert result.recording.embedding is not None
    assert result.recording.source_commit_hash == "abc123"


@pytest.mark.asyncio
async def test_ask_sync_no_qa_service(ask_service_no_qa):
    """Without QA service, returns AskResult with recording=None."""

    async def fake_stream(request, **kwargs):
        yield {
            "event_type": "task_complete",
            "data": {"answer": "answer", "sources": [], "steps": 0},
        }

    with patch.object(ask_service_no_qa, "_stream_from_agent", side_effect=fake_stream):
        result = await ask_service_no_qa.ask_sync(_make_request())

    assert isinstance(result, AskResult)
    assert result.response.answer == "answer"
    assert result.recording is None


@pytest.mark.asyncio
async def test_ask_sync_context_bypass(ask_service, mock_qa_service):
    """Non-empty chat_history passes history to lookup_cache."""
    mock_qa_service.lookup_cache.return_value = (None, None)

    request = AskRequest(
        wiki_id="wiki-1",
        question="follow up",
        chat_history=[ChatMessage(role="user", content="hi")],
    )

    async def fake_stream(request, **kwargs):
        yield {
            "event_type": "task_complete",
            "data": {"answer": "ans", "sources": [], "steps": 0},
        }

    with patch.object(ask_service, "_stream_from_agent", side_effect=fake_stream):
        result = await ask_service.ask_sync(request)

    # Verify lookup_cache was called with chat_history
    call_args = mock_qa_service.lookup_cache.call_args
    assert call_args[1].get("chat_history") or call_args[0][2]  # chat_history arg is non-None


@pytest.mark.asyncio
async def test_ask_sync_cache_hit_sources_parsing(ask_service, mock_qa_service):
    """Cache hit correctly parses sources_json into SourceReference list."""
    cached = MagicMock()
    cached.id = "src-qa-id"
    cached.answer = "With sources"
    cached.sources_json = json.dumps(
        [
            {
                "file_path": "a.py",
                "snippet": "code1",
                "symbol": "foo",
                "symbol_type": "function",
                "relevance_score": 0.9,
            },
            {"file_path": "b.py", "snippet": "code2"},
        ]
    )
    cached.tool_steps = 1
    cached.mode = "fast"
    cached.source_commit_hash = None
    mock_qa_service.lookup_cache.return_value = (cached, None)

    result = await ask_service.ask_sync(_make_request())
    assert len(result.response.sources) == 2
    assert result.response.sources[0].file_path == "a.py"
    assert result.response.sources[0].symbol == "foo"
    assert result.response.sources[1].file_path == "b.py"


@pytest.mark.asyncio
async def test_ask_sync_cache_miss_sources(ask_service, mock_qa_service):
    """Cache miss correctly parses sources from agent events."""
    mock_qa_service.lookup_cache.return_value = (None, None)

    async def fake_stream(request, **kwargs):
        yield {
            "event_type": "task_complete",
            "data": {
                "answer": "Agent result",
                "sources": [
                    {"file_path": "main.py", "snippet": "entry", "symbol": "main"},
                ],
                "steps": 1,
            },
        }

    with patch.object(ask_service, "_stream_from_agent", side_effect=fake_stream):
        result = await ask_service.ask_sync(_make_request())

    assert len(result.response.sources) == 1
    assert result.response.sources[0].file_path == "main.py"
    assert result.recording is not None
    assert result.recording.is_cache_hit is False


@pytest.mark.asyncio
async def test_ask_sync_agent_error(ask_service, mock_qa_service):
    """Agent error raises RuntimeError."""
    mock_qa_service.lookup_cache.return_value = (None, None)

    async def fake_stream(request, **kwargs):
        yield {
            "event_type": "task_failed",
            "data": {"error": "Something broke"},
        }

    with patch.object(ask_service, "_stream_from_agent", side_effect=fake_stream):
        with pytest.raises(RuntimeError, match="Something broke"):
            await ask_service.ask_sync(_make_request())


@pytest.mark.asyncio
async def test_ask_sync_qa_id_present(ask_service, mock_qa_service):
    """AskResult response always contains a qa_id."""
    mock_qa_service.lookup_cache.return_value = (None, None)

    async def fake_stream(request, **kwargs):
        yield {
            "event_type": "task_complete",
            "data": {"answer": "ok", "sources": [], "steps": 0},
        }

    with patch.object(ask_service, "_stream_from_agent", side_effect=fake_stream):
        result = await ask_service.ask_sync(_make_request())

    assert result.response.qa_id != ""
    assert len(result.response.qa_id) == 36  # UUID format


# --- ask_stream tests ---


@pytest.mark.asyncio
async def test_ask_stream_cache_hit(ask_service, mock_qa_service):
    """Cache hit in ask_stream yields answer_chunk + ask_complete events."""
    cached = MagicMock()
    cached.id = "cached-id"
    cached.answer = "Cached stream answer"
    cached.sources_json = json.dumps([{"file_path": "x.py"}])
    cached.tool_steps = 1
    cached.mode = "fast"
    cached.source_commit_hash = None
    mock_qa_service.lookup_cache.return_value = (cached, None)

    events = []
    async for event in ask_service.ask_stream(_make_request()):
        events.append(event)

    assert len(events) == 2
    assert events[0]["event_type"] == "answer_chunk"
    assert events[0]["data"]["chunk"] == "Cached stream answer"
    assert events[1]["event_type"] == "ask_complete"
    assert events[1]["data"]["answer"] == "Cached stream answer"


@pytest.mark.asyncio
async def test_ask_stream_cache_miss_delegates(ask_service, mock_qa_service):
    """Cache miss in ask_stream delegates to _stream_from_agent with commit hash."""
    mock_qa_service.lookup_cache.return_value = (None, None)
    mock_qa_service.get_wiki_commit_hash = AsyncMock(return_value="stream-commit")

    async def fake_agent_stream(request, **kwargs):
        yield {"event_type": "answer_chunk", "data": {"chunk": "hello"}}
        yield {
            "event_type": "task_complete",
            "data": {"answer": "hello", "sources": [], "steps": 0},
        }

    with patch.object(ask_service, "_stream_from_agent", side_effect=fake_agent_stream):
        with patch.object(ask_service, "_record_safely", new_callable=AsyncMock) as mock_record:
            events = []
            async for event in ask_service.ask_stream(_make_request()):
                events.append(event)
            # Allow fire-and-forget task to complete
            await asyncio.sleep(0)

    assert len(events) == 2
    assert events[0]["event_type"] == "answer_chunk"
    assert events[1]["event_type"] == "task_complete"
    assert events[1]["data"].get("qa_id")
    # Verify source_commit_hash was passed to recording payload
    mock_record.assert_called_once()
    payload = mock_record.call_args[0][0]  # First positional arg: QARecordingPayload
    assert payload.source_commit_hash == "stream-commit"


@pytest.mark.asyncio
async def test_ask_stream_no_qa_service(ask_service_no_qa):
    """Without QA service, ask_stream delegates directly to _stream_from_agent."""

    async def fake_agent_stream(request, **kwargs):
        yield {"event_type": "answer_chunk", "data": {"chunk": "hi"}}
        yield {
            "event_type": "task_complete",
            "data": {"answer": "hi", "sources": [], "steps": 0},
        }

    with patch.object(ask_service_no_qa, "_stream_from_agent", side_effect=fake_agent_stream):
        events = []
        async for event in ask_service_no_qa.ask_stream(_make_request()):
            events.append(event)

    assert len(events) == 2


# --- _record_safely tests ---


def _make_payload(**overrides):
    """Build a QARecordingPayload with sensible defaults."""
    defaults = dict(
        qa_id="test-id", wiki_id="w1", question="q", answer="a",
        sources_json="[]", tool_steps=0, mode="fast", user_id=None,
        is_cache_hit=False, source_qa_id=None, embedding=None,
        has_context=False,
    )
    defaults.update(overrides)
    return QARecordingPayload(**defaults)


@pytest.mark.asyncio
async def test_record_safely_suppresses_errors(ask_service, mock_qa_service):
    """_record_safely swallows exceptions from qa_service."""
    mock_qa_service.record_interaction.side_effect = RuntimeError("DB down")

    # Should not raise
    await ask_service._record_safely(_make_payload())


@pytest.mark.asyncio
async def test_record_safely_no_qa_service(ask_service_no_qa):
    """_record_safely is a no-op without qa_service."""
    # Should not raise
    await ask_service_no_qa._record_safely(_make_payload())
