"""Test that AskResponse includes qa_id field."""
from app.models.api import AskResponse


def test_ask_response_has_qa_id():
    """AskResponse accepts and serializes qa_id."""
    resp = AskResponse(
        answer="test",
        sources=[],
        tool_steps=1,
        qa_id="qa-123",
    )
    data = resp.model_dump()
    assert data["qa_id"] == "qa-123"


def test_ask_response_qa_id_default():
    """AskResponse qa_id defaults to empty string for backward compat."""
    resp = AskResponse(answer="test", sources=[], tool_steps=0)
    assert resp.qa_id == ""
