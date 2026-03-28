"""Tests for QA API models and dataclasses."""

import numpy as np
import pytest
from datetime import datetime

from app.models.api import AskResponse, SourceReference
from app.models.qa_api import (
    QARecordingPayload,
    AskResult,
    QARecordResponse,
    QAListResponse,
    QAStatsResponse,
)


def test_qa_recording_payload_creation():
    """QARecordingPayload can be created with all fields."""
    embedding = np.array([0.1, 0.2, 0.3])
    payload = QARecordingPayload(
        qa_id="qa-123",
        wiki_id="wiki-1",
        question="What is this?",
        answer="It is a test.",
        sources_json="[]",
        tool_steps=2,
        mode="fast",
        user_id="user-1",
        is_cache_hit=False,
        source_qa_id=None,
        embedding=embedding,
        has_context=False,
        source_commit_hash="abc123",
    )
    assert payload.qa_id == "qa-123"
    assert payload.is_cache_hit is False
    assert payload.embedding is not None
    assert np.array_equal(payload.embedding, embedding)
    assert payload.has_context is False
    assert payload.source_commit_hash == "abc123"


def test_qa_recording_payload_defaults():
    """QARecordingPayload defaults for optional fields."""
    payload = QARecordingPayload(
        qa_id="qa-1",
        wiki_id="w",
        question="q",
        answer="a",
        sources_json="[]",
        tool_steps=0,
        mode="fast",
        user_id=None,
        is_cache_hit=False,
        source_qa_id=None,
        embedding=None,
    )
    assert payload.has_context is False
    assert payload.source_commit_hash is None


def test_ask_result_creation():
    """AskResult wraps response and optional recording payload."""
    response = AskResponse(
        answer="test answer",
        sources=[],
        tool_steps=1,
    )
    payload = QARecordingPayload(
        qa_id="qa-1",
        wiki_id="w",
        question="q",
        answer="a",
        sources_json="[]",
        tool_steps=0,
        mode="fast",
        user_id=None,
        is_cache_hit=True,
        source_qa_id="qa-0",
        embedding=None,
    )
    result = AskResult(response=response, recording=payload)
    assert result.response.answer == "test answer"
    assert result.recording is not None
    assert result.recording.is_cache_hit is True


def test_ask_result_no_recording():
    """AskResult works with recording=None (no QA service)."""
    response = AskResponse(answer="x", sources=[], tool_steps=0)
    result = AskResult(response=response, recording=None)
    assert result.recording is None


def test_qa_list_response_serialization():
    """QAListResponse serializes correctly."""
    item = QARecordResponse(
        id="qa-1",
        wiki_id="w-1",
        question="q",
        answer="a",
        status="pending",
    )
    resp = QAListResponse(items=[item], total=1, limit=20, offset=0)
    data = resp.model_dump()
    assert data["total"] == 1
    assert len(data["items"]) == 1
    assert data["items"][0]["id"] == "qa-1"


def test_qa_stats_response_hit_rate():
    """QAStatsResponse serializes with hit_rate."""
    stats = QAStatsResponse(
        total_count=100,
        cache_hit_count=25,
        validated_count=10,
        rejected_count=5,
        hit_rate=0.25,
    )
    data = stats.model_dump()
    assert data["hit_rate"] == 0.25
    assert data["cache_hit_count"] == 25


def test_qa_stats_response_empty():
    """QAStatsResponse with zero counts."""
    stats = QAStatsResponse(
        total_count=0,
        cache_hit_count=0,
        validated_count=0,
        rejected_count=0,
        hit_rate=0.0,
    )
    assert stats.hit_rate == 0.0
