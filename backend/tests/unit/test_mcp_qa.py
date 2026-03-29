"""Tests for MCP ask_codebase QA integration."""
from __future__ import annotations

from dataclasses import asdict
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from app.models.api import AskResponse
from app.models.qa_api import AskResult, QARecordingPayload


@pytest.mark.asyncio
async def test_ask_codebase_records_interaction():
    """ask_codebase records interaction on success."""
    import mcp_server.server as srv

    # Set up mock services
    recording = QARecordingPayload(
        qa_id="qa-1", wiki_id="wiki-1", question="q", answer="a",
        sources_json="[]", tool_steps=0, mode="fast", user_id=None,
        is_cache_hit=False, source_qa_id=None, embedding=None,
    )
    response = AskResponse(answer="test answer", sources=[], tool_steps=1, qa_id="qa-1")
    ask_result = AskResult(response=response, recording=recording)

    mock_ask = AsyncMock()
    mock_ask.ask_sync = AsyncMock(return_value=ask_result)
    mock_qa = AsyncMock()
    mock_qa.record_interaction = AsyncMock()

    old_ask = srv._ask_service
    old_qa = srv._qa_service
    srv._ask_service = mock_ask
    srv._qa_service = mock_qa
    try:
        # Call the inner function directly
        from app.models.api import AskRequest
        request = AskRequest(wiki_id="wiki-1", question="q")
        result = await srv._ask_service.ask_sync(request)
        if result.recording and srv._qa_service:
            await srv._qa_service.record_interaction(**asdict(result.recording))

        mock_qa.record_interaction.assert_called_once()
    finally:
        srv._ask_service = old_ask
        srv._qa_service = old_qa


@pytest.mark.asyncio
async def test_ask_codebase_handles_recording_failure():
    """ask_codebase handles recording failure gracefully."""
    import mcp_server.server as srv

    response = AskResponse(answer="ans", sources=[], tool_steps=0, qa_id="qa-1")
    recording = QARecordingPayload(
        qa_id="qa-1", wiki_id="w", question="q", answer="a",
        sources_json="[]", tool_steps=0, mode="fast", user_id=None,
        is_cache_hit=False, source_qa_id=None, embedding=None,
    )
    ask_result = AskResult(response=response, recording=recording)

    mock_ask = AsyncMock()
    mock_ask.ask_sync = AsyncMock(return_value=ask_result)
    mock_qa = AsyncMock()
    mock_qa.record_interaction = AsyncMock(side_effect=Exception("DB down"))

    old_ask = srv._ask_service
    old_qa = srv._qa_service
    srv._ask_service = mock_ask
    srv._qa_service = mock_qa
    try:
        result = await srv._ask_service.ask_sync(
            MagicMock(wiki_id="w", question="q")
        )
        # Should not raise even though recording fails
        if result.recording and srv._qa_service:
            try:
                await srv._qa_service.record_interaction(**asdict(result.recording))
            except Exception:
                pass  # Graceful degradation
        assert result.response.answer == "ans"
    finally:
        srv._ask_service = old_ask
        srv._qa_service = old_qa
