"""Tests for API request/response models and SSE events."""

from __future__ import annotations

from datetime import datetime

import pytest
from pydantic import ValidationError

from app.models import (
    AskRequest,
    AskResponse,
    ChatMessage,
    DeleteWikiResponse,
    ErrorEvent,
    ErrorResponse,
    GenerateWikiRequest,
    GenerateWikiResponse,
    HealthResponse,
    PageCompleteEvent,
    ProgressEvent,
    ResearchRequest,
    ResearchResponse,
    SourceReference,
    SSEEvent,
    WikiCompleteEvent,
    WikiListResponse,
    WikiSummary,
)


class TestGenerateWiki:
    def test_request_minimal(self):
        req = GenerateWikiRequest(repo_url="https://github.com/org/repo")
        assert req.branch == "main"
        assert req.provider == "github"

    def test_request_full(self):
        req = GenerateWikiRequest(
            repo_url="https://gitlab.com/org/repo",
            branch="develop",
            provider="gitlab",
            access_token="glpat-xxx",
            wiki_title="My Wiki",
            llm_model="gpt-4o",
        )
        assert req.provider == "gitlab"
        assert req.llm_model == "gpt-4o"

    def test_request_missing_url_raises(self):
        with pytest.raises(ValidationError):
            GenerateWikiRequest()

    def test_response_schema(self):
        schema = GenerateWikiResponse.model_json_schema()
        assert "wiki_id" in schema["properties"]
        assert "status" in schema["properties"]

    def test_response_serializes(self):
        resp = GenerateWikiResponse(wiki_id="w1", invocation_id="inv-1", status="generating", message="Started")
        data = resp.model_dump()
        assert data["wiki_id"] == "w1"
        assert data["invocation_id"] == "inv-1"


class TestAsk:
    def test_request_defaults(self):
        req = AskRequest(wiki_id="w1", question="How does auth work?")
        assert req.k == 15
        assert req.chat_history == []

    def test_request_with_history(self):
        req = AskRequest(
            wiki_id="w1",
            question="Follow up",
            chat_history=[ChatMessage(role="user", content="First question")],
        )
        assert len(req.chat_history) == 1

    def test_response_with_sources(self):
        resp = AskResponse(
            answer="Auth uses JWT tokens.",
            sources=[SourceReference(file_path="src/auth.py", line_start=10, snippet="def auth():")],
        )
        assert len(resp.sources) == 1
        assert resp.sources[0].file_path == "src/auth.py"


class TestResearch:
    def test_request_defaults(self):
        req = ResearchRequest(wiki_id="w1", question="Architecture overview")
        assert req.research_type == "general"

    def test_response_with_steps(self):
        resp = ResearchResponse(
            answer="The system uses...",
            research_steps=["Analyzed code graph", "Queried vector store"],
        )
        assert len(resp.research_steps) == 2


class TestWikiManagement:
    def test_wiki_summary(self):
        ws = WikiSummary(
            wiki_id="w1",
            repo_url="https://github.com/org/repo",
            branch="main",
            title="Repo Wiki",
            created_at=datetime(2026, 1, 1),
            page_count=12,
        )
        assert ws.page_count == 12

    def test_wiki_list_response(self):
        resp = WikiListResponse()
        assert resp.wikis == []

    def test_delete_response(self):
        resp = DeleteWikiResponse(deleted=True, wiki_id="w1")
        data = resp.model_dump()
        assert data["deleted"] is True


class TestHealthAndErrors:
    def test_health_response(self):
        resp = HealthResponse(status="ok", version="0.1.0", services={"llm": "ok", "storage": "ok"})
        assert resp.services["llm"] == "ok"

    def test_error_response(self):
        resp = ErrorResponse(error="Not found", code="WIKI_NOT_FOUND")
        assert resp.detail is None
        schema = ErrorResponse.model_json_schema()
        assert "code" in schema["properties"]


class TestSSEEvents:
    def test_progress_event_discriminator(self):
        evt = ProgressEvent(phase="generating", progress=0.5, message="Page 3/10")
        assert evt.event == "progress"

    def test_page_complete_event(self):
        evt = PageCompleteEvent(page_id="p1", page_title="Getting Started")
        assert evt.event == "page_complete"

    def test_wiki_complete_event(self):
        evt = WikiCompleteEvent(wiki_id="w1", page_count=10, execution_time=120.5)
        assert evt.event == "wiki_complete"

    def test_error_event(self):
        evt = ErrorEvent(error="LLM timeout", recoverable=True)
        assert evt.event == "error"
        assert evt.recoverable is True

    def test_sse_event_generic(self):
        evt = SSEEvent(event="custom", data={"key": "value"})
        assert evt.data["key"] == "value"


class TestAllModelsProduceSchema:
    @pytest.mark.parametrize(
        "model_cls",
        [
            GenerateWikiRequest,
            GenerateWikiResponse,
            AskRequest,
            AskResponse,
            ResearchRequest,
            ResearchResponse,
            WikiSummary,
            WikiListResponse,
            DeleteWikiResponse,
            HealthResponse,
            ErrorResponse,
            SSEEvent,
            ProgressEvent,
            PageCompleteEvent,
            WikiCompleteEvent,
            ErrorEvent,
        ],
    )
    def test_json_schema(self, model_cls):
        schema = model_cls.model_json_schema()
        assert "properties" in schema or "anyOf" in schema


# ---------------------------------------------------------------------------
# AtlassianAuth — OAuth + API-token (basic auth) shapes (#28)
# ---------------------------------------------------------------------------


class TestAtlassianAuthShapes:
    """The model accepts OAuth (access_token) OR basic-auth (email + api_token)
    but NOT both, NOT neither, and rejects half-filled basic-auth pairs."""

    def test_oauth_only_accepted(self):
        from app.models.api import AtlassianAuth

        auth = AtlassianAuth(access_token="oauth-abc")
        assert auth.uses_basic_auth is False
        assert auth.access_token == "oauth-abc"

    def test_basic_auth_only_accepted(self):
        from app.models.api import AtlassianAuth

        auth = AtlassianAuth(email="alice@example.com", api_token="atk-123")
        assert auth.uses_basic_auth is True
        assert auth.access_token is None

    def test_both_oauth_and_basic_rejected(self):
        from app.models.api import AtlassianAuth

        with pytest.raises(ValidationError, match="not both"):
            AtlassianAuth(
                access_token="oauth",
                email="a@b.com",
                api_token="atk",
            )

    def test_neither_rejected(self):
        from app.models.api import AtlassianAuth

        with pytest.raises(ValidationError, match="provide either"):
            AtlassianAuth()

    def test_email_without_api_token_rejected(self):
        from app.models.api import AtlassianAuth

        with pytest.raises(ValidationError, match="api_token is required"):
            AtlassianAuth(email="a@b.com")

    def test_api_token_without_email_rejected(self):
        from app.models.api import AtlassianAuth

        with pytest.raises(ValidationError, match="email is required"):
            AtlassianAuth(api_token="atk")
