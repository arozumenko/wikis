"""Regression tests for project-ask guards and sync wrapper deadlock fix.

Bug 1: wiki_id=None written to non-nullable QARecord column on project asks.
Bug 2: get_wiki_commit_hash(None) called on project ask path.
Bug 3: sync wrapper deadlock when called from a running event loop.
"""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from pydantic import SecretStr

from app.config import Settings
from app.models.api import AskRequest
from app.services.ask_service import AskService

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def settings():
    return Settings(llm_api_key=SecretStr("test-key"))


@pytest.fixture()
def mock_qa_service():
    service = AsyncMock()
    service.lookup_cache = AsyncMock(return_value=(None, None))
    service.get_wiki_commit_hash = AsyncMock(return_value="hash-abc")
    service.record_interaction = AsyncMock()
    return service


@pytest.fixture()
def ask_service(settings, mock_qa_service):
    storage = MagicMock()
    return AskService(settings, storage, qa_service=mock_qa_service)


def _project_request():
    """AskRequest with project_id set and wiki_id=None."""
    return AskRequest(project_id="proj-1", question="What does auth do?")


def _wiki_request():
    """Normal single-wiki AskRequest."""
    return AskRequest(wiki_id="wiki-1", question="What does auth do?")


def _fake_agent_stream():
    async def _gen(request):
        yield {
            "event_type": "task_complete",
            "data": {"answer": "answer text", "sources": [], "steps": 1},
        }

    return _gen


# ---------------------------------------------------------------------------
# Bug 1 & 2 — ask_sync project path skips recording and commit hash lookup
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_ask_sync_project_skips_recording(ask_service, mock_qa_service):
    """Bug 1: project ask must NOT build QARecordingPayload (wiki_id is None)."""
    with patch.object(ask_service, "_stream_from_agent", side_effect=_fake_agent_stream()):
        result = await ask_service.ask_sync(_project_request())

    # recording must be None — wiki_id is None; writing it would violate DB constraint
    assert result.recording is None


@pytest.mark.asyncio
async def test_ask_sync_project_skips_commit_hash_lookup(ask_service, mock_qa_service):
    """Bug 2: get_wiki_commit_hash must NOT be called when wiki_id is None."""
    with patch.object(ask_service, "_stream_from_agent", side_effect=_fake_agent_stream()):
        await ask_service.ask_sync(_project_request())

    mock_qa_service.get_wiki_commit_hash.assert_not_called()


@pytest.mark.asyncio
async def test_ask_sync_project_skips_cache_lookup(ask_service, mock_qa_service):
    """Bug 1: project ask must NOT attempt a cache lookup (wiki_id is None)."""
    with patch.object(ask_service, "_stream_from_agent", side_effect=_fake_agent_stream()):
        await ask_service.ask_sync(_project_request())

    mock_qa_service.lookup_cache.assert_not_called()


@pytest.mark.asyncio
async def test_ask_sync_wiki_still_records(ask_service, mock_qa_service):
    """Verify wiki ask still produces a recording (regression guard)."""
    mock_qa_service.get_wiki_commit_hash = AsyncMock(return_value="hash-123")

    with patch.object(ask_service, "_stream_from_agent", side_effect=_fake_agent_stream()):
        result = await ask_service.ask_sync(_wiki_request())

    assert result.recording is not None
    assert result.recording.wiki_id == "wiki-1"
    mock_qa_service.get_wiki_commit_hash.assert_called_once_with("wiki-1")


# ---------------------------------------------------------------------------
# Bug 1 & 2 — ask_stream project path skips recording and commit hash lookup
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_ask_stream_project_skips_recording(ask_service, mock_qa_service):
    """Bug 1: project ask stream must NOT fire a QA recording task."""
    with patch.object(ask_service, "_stream_from_agent", side_effect=_fake_agent_stream()):
        with patch.object(ask_service, "_record_safely", new_callable=AsyncMock) as mock_rec:
            async for _ in ask_service.ask_stream(_project_request()):
                pass
            await asyncio.sleep(0)

    mock_rec.assert_not_called()


@pytest.mark.asyncio
async def test_ask_stream_project_skips_commit_hash(ask_service, mock_qa_service):
    """Bug 2: ask_stream must NOT call get_wiki_commit_hash for project asks."""
    with patch.object(ask_service, "_stream_from_agent", side_effect=_fake_agent_stream()):
        async for _ in ask_service.ask_stream(_project_request()):
            pass

    mock_qa_service.get_wiki_commit_hash.assert_not_called()


@pytest.mark.asyncio
async def test_ask_stream_project_skips_cache_lookup(ask_service, mock_qa_service):
    """Bug 1: ask_stream must NOT attempt a cache lookup for project asks."""
    with patch.object(ask_service, "_stream_from_agent", side_effect=_fake_agent_stream()):
        async for _ in ask_service.ask_stream(_project_request()):
            pass

    mock_qa_service.lookup_cache.assert_not_called()


@pytest.mark.asyncio
async def test_ask_stream_wiki_still_records(ask_service, mock_qa_service):
    """Verify wiki ask_stream still fires recording (regression guard)."""
    mock_qa_service.get_wiki_commit_hash = AsyncMock(return_value="hash-456")

    with patch.object(ask_service, "_stream_from_agent", side_effect=_fake_agent_stream()):
        with patch.object(ask_service, "_record_safely", new_callable=AsyncMock) as mock_rec:
            async for _ in ask_service.ask_stream(_wiki_request()):
                pass
            await asyncio.sleep(0)

    mock_rec.assert_called_once()
    payload = mock_rec.call_args[0][0]
    assert payload.wiki_id == "wiki-1"


# ---------------------------------------------------------------------------
# Bug 3 — MultiWikiRetrieverStack sync wrapper raises from async context
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_multi_retriever_sync_raises_in_async_context():
    """Bug 3: retrieve() must raise RuntimeError when called inside a running loop."""
    from app.core.multi_retriever import MultiWikiRetrieverStack

    stack = MultiWikiRetrieverStack(wiki_stacks=[])

    with pytest.raises(RuntimeError, match="cannot be called from an async context"):
        stack.retrieve("some query")


@pytest.mark.asyncio
async def test_multi_retriever_search_repository_raises_in_async_context():
    """Bug 3: search_repository() must raise RuntimeError when called inside a running loop."""
    from app.core.multi_retriever import MultiWikiRetrieverStack

    stack = MultiWikiRetrieverStack(wiki_stacks=[])

    with pytest.raises(RuntimeError, match="cannot be called from an async context"):
        stack.search_repository("some query")


def test_multi_retriever_retrieve_works_outside_event_loop():
    """Bug 3: retrieve() must work fine when there is no running event loop."""
    from app.core.multi_retriever import MultiWikiRetrieverStack

    stack = MultiWikiRetrieverStack(wiki_stacks=[])
    result = stack.retrieve("some query")
    assert result == []  # empty stacks → empty results


@pytest.mark.asyncio
async def test_multi_retriever_aretrieve_works_in_async_context():
    """aretrieve() is the correct async interface and must work fine."""
    from app.core.multi_retriever import MultiWikiRetrieverStack

    stack = MultiWikiRetrieverStack(wiki_stacks=[])
    result = await stack.aretrieve("some query")
    assert result == []
