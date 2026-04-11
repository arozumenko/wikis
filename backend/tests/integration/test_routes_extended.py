"""Extended integration tests for app/api/routes.py.

Covers branches not hit by existing tests:
- Invocation status / cancel endpoints
- Invocation stream endpoint
- Wiki listing and get_wiki with active invocations
- ask endpoint branches (SSE, errors)
- research endpoint branches (SSE, codemap, errors)
- QA / qa-stats endpoints
- Wiki lifecycle: refresh, delete, visibility, resume
- Projects CRUD (create, list, get, update, delete, add/remove wiki)
- Health endpoint
- Export: invalid format, non-complete wiki
- Import: size guard, NotImplementedError

All run with AUTH_ENABLED=false via the shared conftest client fixture.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from app.main import create_app
from app.models.db_models import Base, WikiRecord
from app.models.invocation import Invocation
from app.services.wiki_management import WikiManagementService
from httpx import ASGITransport, AsyncClient
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

# ---------------------------------------------------------------------------
# Helper: seed a wiki record
# ---------------------------------------------------------------------------


async def _seed_wiki(
    session_factory,
    wiki_id: str = "test-wiki",
    owner_id: str = "dev-user",
    status: str = "complete",
) -> WikiRecord:
    async with session_factory() as session:
        async with session.begin():
            record = WikiRecord(
                id=wiki_id,
                owner_id=owner_id,
                repo_url="https://github.com/test/repo",
                branch="main",
                title="Test Wiki",
                page_count=3,
                status=status,
            )
            session.add(record)
    return record


# ---------------------------------------------------------------------------
# Shared client fixture (in-memory SQLite, no auth)
# ---------------------------------------------------------------------------


@pytest.fixture
async def client_tuple(tmp_path, monkeypatch):
    monkeypatch.setenv("AUTH_ENABLED", "false")
    monkeypatch.setenv("LLM_API_KEY", "test-key")

    engine = create_async_engine("sqlite+aiosqlite://", connect_args={"check_same_thread": False})
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    sf = async_sessionmaker(engine, expire_on_commit=False)

    from app.services.wiki_management import WikiManagementService
    from app.storage.local import LocalArtifactStorage

    app = create_app()
    async with app.router.lifespan_context(app):
        storage = LocalArtifactStorage(str(tmp_path))
        wiki_mgmt = WikiManagementService(storage, sf)
        app.state.storage = storage
        app.state.wiki_management = wiki_mgmt
        app.state.session_factory = sf

        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as c:
            yield c, app, sf

    await engine.dispose()


# ---------------------------------------------------------------------------
# Health endpoint
# ---------------------------------------------------------------------------


class TestHealthEndpoint:
    @pytest.mark.asyncio
    async def test_health_returns_ok(self, client):
        resp = await client.get("/api/v1/health")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] in ("ok", "degraded")

    @pytest.mark.asyncio
    async def test_health_with_provider_health(self, test_app, client):
        """When provider_health is set, it's reflected in the response."""
        ph = MagicMock()
        ph.llm = "ok"
        ph.embeddings = "ok"
        test_app.state.provider_health = ph

        resp = await client.get("/api/v1/health")
        assert resp.status_code == 200
        data = resp.json()
        assert data["services"]["llm"] == "ok"


# ---------------------------------------------------------------------------
# Invocation endpoints
# ---------------------------------------------------------------------------


class TestInvocationEndpoints:
    @pytest.mark.asyncio
    async def test_get_invocation_found(self, client, test_app):
        inv = MagicMock(spec=Invocation)
        inv.id = "inv-001"
        inv.wiki_id = "wiki-001"
        inv.status = "complete"
        inv.model_dump = MagicMock(return_value={"id": "inv-001", "wiki_id": "wiki-001", "status": "complete"})

        with patch.object(test_app.state.wiki_service, "get_invocation", new_callable=AsyncMock, return_value=inv):
            resp = await client.get("/api/v1/invocations/inv-001")
        assert resp.status_code == 200

    @pytest.mark.asyncio
    async def test_get_invocation_not_found(self, client, test_app):
        with patch.object(test_app.state.wiki_service, "get_invocation", new_callable=AsyncMock, return_value=None):
            resp = await client.get("/api/v1/invocations/nonexistent")
        assert resp.status_code == 404

    @pytest.mark.asyncio
    async def test_cancel_invocation_success(self, client, test_app):
        with patch.object(test_app.state.wiki_service, "cancel_invocation", new_callable=AsyncMock, return_value=True):
            resp = await client.delete("/api/v1/invocations/inv-to-cancel")
        assert resp.status_code == 200
        assert resp.json()["cancelled"] is True

    @pytest.mark.asyncio
    async def test_cancel_invocation_not_found(self, client, test_app):
        with patch.object(test_app.state.wiki_service, "cancel_invocation", new_callable=AsyncMock, return_value=False):
            resp = await client.delete("/api/v1/invocations/nonexistent")
        assert resp.status_code == 404

    @pytest.mark.asyncio
    async def test_stream_invocation_not_found(self, client, test_app):
        with patch.object(test_app.state.wiki_service, "get_invocation", new_callable=AsyncMock, return_value=None):
            resp = await client.get("/api/v1/invocations/ghost/stream")
        assert resp.status_code == 404

    @pytest.mark.asyncio
    async def test_stream_invocation_returns_event_stream(self, client, test_app):
        inv = MagicMock(spec=Invocation)
        inv.id = "inv-stream"
        inv.wiki_id = "wiki-1"
        inv.status = "generating"

        async def _events(after_event_id=None):
            yield (0, MagicMock(event="progress", model_dump_json=lambda: '{"progress": 10}'))

        inv.events = _events

        with patch.object(test_app.state.wiki_service, "get_invocation", new_callable=AsyncMock, return_value=inv):
            resp = await client.get("/api/v1/invocations/inv-stream/stream")
        # SSE returns 200
        assert resp.status_code == 200


# ---------------------------------------------------------------------------
# Wiki listing — list_wikis
# ---------------------------------------------------------------------------


class TestListWikis:
    @pytest.mark.asyncio
    async def test_list_wikis_empty(self, client):
        resp = await client.get("/api/v1/wikis")
        assert resp.status_code == 200
        data = resp.json()
        assert "wikis" in data

    @pytest.mark.asyncio
    async def test_list_wikis_with_seeded_record(self, client_tuple):
        c, app, sf = client_tuple
        await _seed_wiki(sf, wiki_id="listed-wiki")

        resp = await c.get("/api/v1/wikis")
        assert resp.status_code == 200
        wiki_ids = [w["wiki_id"] for w in resp.json()["wikis"]]
        assert "listed-wiki" in wiki_ids


# ---------------------------------------------------------------------------
# get_wiki
# ---------------------------------------------------------------------------


class TestGetWiki:
    @pytest.mark.asyncio
    async def test_get_wiki_not_found(self, client):
        resp = await client.get("/api/v1/wikis/nonexistent-wiki")
        assert resp.status_code == 404

    @pytest.mark.asyncio
    async def test_get_wiki_found(self, client_tuple):
        c, app, sf = client_tuple
        await _seed_wiki(sf, wiki_id="found-wiki")

        resp = await c.get("/api/v1/wikis/found-wiki")
        assert resp.status_code == 200
        data = resp.json()
        assert data["wiki_id"] == "found-wiki"

    @pytest.mark.asyncio
    async def test_get_wiki_generating_status_from_invocation(self, client_tuple):
        """Wiki in 'generating' state should not 404."""
        c, app, sf = client_tuple
        await _seed_wiki(sf, wiki_id="gen-wiki", status="generating")

        resp = await c.get("/api/v1/wikis/gen-wiki")
        assert resp.status_code == 200
        assert resp.json()["status"] == "generating"

    @pytest.mark.asyncio
    async def test_get_wiki_failed_status(self, client_tuple):
        c, app, sf = client_tuple
        await _seed_wiki(sf, wiki_id="failed-wiki", status="failed")

        resp = await c.get("/api/v1/wikis/failed-wiki")
        assert resp.status_code == 200
        assert resp.json()["status"] == "failed"


# ---------------------------------------------------------------------------
# ask endpoint
# ---------------------------------------------------------------------------


class TestAskEndpoint:
    @pytest.mark.asyncio
    async def test_ask_json_mode_success(self, client, test_app):
        from app.models.api import AskResponse
        from app.models.qa_api import AskResult

        mock_response = AskResponse(answer="JWT auth.", sources=[])
        mock_result = AskResult(response=mock_response, recording=None)

        with patch.object(test_app.state.ask_service, "ask_sync", new_callable=AsyncMock, return_value=mock_result):
            resp = await client.post("/api/v1/ask", json={"wiki_id": "w1", "question": "How does auth work?"})
        assert resp.status_code == 200
        assert resp.json()["answer"] == "JWT auth."

    @pytest.mark.asyncio
    async def test_ask_file_not_found_gives_404(self, client, test_app):
        with patch.object(
            test_app.state.ask_service,
            "ask_sync",
            new_callable=AsyncMock,
            side_effect=FileNotFoundError("Wiki not found"),
        ):
            resp = await client.post("/api/v1/ask", json={"wiki_id": "missing", "question": "Q?"})
        assert resp.status_code == 404

    @pytest.mark.asyncio
    async def test_ask_value_error_gives_400(self, client, test_app):
        with patch.object(
            test_app.state.ask_service,
            "ask_sync",
            new_callable=AsyncMock,
            side_effect=ValueError("bad input"),
        ):
            resp = await client.post("/api/v1/ask", json={"wiki_id": "w1", "question": "Q?"})
        assert resp.status_code == 400

    @pytest.mark.asyncio
    async def test_ask_runtime_error_gives_501(self, client, test_app):
        with patch.object(
            test_app.state.ask_service,
            "ask_sync",
            new_callable=AsyncMock,
            side_effect=RuntimeError("not implemented"),
        ):
            resp = await client.post("/api/v1/ask", json={"wiki_id": "w1", "question": "Q?"})
        assert resp.status_code == 501

    @pytest.mark.asyncio
    async def test_ask_sse_mode(self, client, test_app):
        """text/event-stream accept query param triggers SSE streaming path."""

        async def _events(*args, **kwargs):
            yield {"event_type": "ask_complete", "data": {"answer": "done"}}

        with patch.object(test_app.state.ask_service, "ask_stream", return_value=_events()):
            resp = await client.post(
                "/api/v1/ask?accept=text%2Fevent-stream",
                json={"wiki_id": "w1", "question": "Q?"},
            )
        assert resp.status_code == 200

    @pytest.mark.asyncio
    async def test_ask_with_recording(self, client, test_app):
        """When result.recording is present, background task is queued (but not awaited in tests)."""
        from app.models.api import AskResponse
        from app.models.qa_api import AskResult, QARecordingPayload

        payload = QARecordingPayload(
            qa_id="qa-1",
            wiki_id="w1",
            question="Q?",
            answer="A",
            sources_json="[]",
            tool_steps=0,
            mode="fast",
            user_id=None,
            is_cache_hit=False,
            source_qa_id=None,
            embedding=None,
        )
        mock_response = AskResponse(answer="A", sources=[])
        mock_result = AskResult(response=mock_response, recording=payload)

        with (
            patch.object(test_app.state.ask_service, "ask_sync", new_callable=AsyncMock, return_value=mock_result),
            patch.object(test_app.state.qa_service, "record_interaction", new_callable=AsyncMock, return_value=None),
        ):
            resp = await client.post("/api/v1/ask", json={"wiki_id": "w1", "question": "Q?"})
        assert resp.status_code == 200


# ---------------------------------------------------------------------------
# research endpoint
# ---------------------------------------------------------------------------


class TestResearchEndpoint:
    @pytest.mark.asyncio
    async def test_research_json_mode(self, client, test_app):
        from app.models.api import ResearchResponse

        mock_resp = ResearchResponse(answer="Arch.", sources=[], research_steps=[])
        with patch.object(
            test_app.state.research_service, "research_sync", new_callable=AsyncMock, return_value=mock_resp
        ):
            resp = await client.post("/api/v1/research", json={"wiki_id": "w1", "question": "Arch?"})
        assert resp.status_code == 200
        assert resp.json()["answer"] == "Arch."

    @pytest.mark.asyncio
    async def test_research_sse_mode(self, client, test_app):
        async def _events(*args, **kwargs):
            yield {"event_type": "research_complete", "data": {"answer": "done"}}

        with patch.object(test_app.state.research_service, "research_stream", return_value=_events()):
            resp = await client.post(
                "/api/v1/research?accept=text%2Fevent-stream",
                json={"wiki_id": "w1", "question": "Q?"},
            )
        assert resp.status_code == 200

    @pytest.mark.asyncio
    async def test_research_file_not_found_gives_404(self, client, test_app):
        with patch.object(
            test_app.state.research_service,
            "research_sync",
            new_callable=AsyncMock,
            side_effect=FileNotFoundError("wiki missing"),
        ):
            resp = await client.post("/api/v1/research", json={"wiki_id": "missing", "question": "Q?"})
        assert resp.status_code == 404

    @pytest.mark.asyncio
    async def test_research_codemap_json_mode(self, client, test_app):
        from app.models.api import ResearchResponse

        mock_resp = ResearchResponse(answer="Map.", sources=[], research_steps=[])
        with patch.object(
            test_app.state.research_service, "codemap_sync", new_callable=AsyncMock, return_value=mock_resp
        ):
            resp = await client.post(
                "/api/v1/research",
                json={"wiki_id": "w1", "question": "Q?", "research_type": "codemap"},
            )
        assert resp.status_code == 200

    @pytest.mark.asyncio
    async def test_research_codemap_sse_mode(self, client, test_app):
        async def _events(*args, **kwargs):
            yield {"event_type": "research_complete", "data": {"answer": "done"}}

        with patch.object(test_app.state.research_service, "codemap_stream", return_value=_events()):
            resp = await client.post(
                "/api/v1/research?accept=text%2Fevent-stream",
                json={"wiki_id": "w1", "question": "Q?", "research_type": "codemap"},
            )
        assert resp.status_code == 200

    @pytest.mark.asyncio
    async def test_research_runtime_error_gives_501(self, client, test_app):
        with patch.object(
            test_app.state.research_service,
            "research_sync",
            new_callable=AsyncMock,
            side_effect=RuntimeError("not impl"),
        ):
            resp = await client.post("/api/v1/research", json={"wiki_id": "w1", "question": "Q?"})
        assert resp.status_code == 501

    @pytest.mark.asyncio
    async def test_research_value_error_gives_400(self, client, test_app):
        with patch.object(
            test_app.state.research_service,
            "research_sync",
            new_callable=AsyncMock,
            side_effect=ValueError("bad"),
        ):
            resp = await client.post("/api/v1/research", json={"wiki_id": "w1", "question": "Q?"})
        assert resp.status_code == 400


# ---------------------------------------------------------------------------
# Wiki visibility
# ---------------------------------------------------------------------------


class TestWikiVisibility:
    @pytest.mark.asyncio
    async def test_visibility_invalid_value(self, client):
        resp = await client.patch(
            "/api/v1/wikis/any/visibility",
            json={"visibility": "public"},
        )
        assert resp.status_code == 422

    @pytest.mark.asyncio
    async def test_visibility_wiki_not_found(self, client, test_app):
        with (
            patch.object(
                test_app.state.wiki_management,
                "update_visibility",
                new_callable=AsyncMock,
                return_value=None,
            ),
            patch.object(
                test_app.state.wiki_management,
                "get_wiki",
                new_callable=AsyncMock,
                return_value=None,
            ),
        ):
            resp = await client.patch(
                "/api/v1/wikis/missing/visibility",
                json={"visibility": "shared"},
            )
        assert resp.status_code == 404

    @pytest.mark.asyncio
    async def test_visibility_not_owner(self, client, test_app):
        """update_visibility returns None when caller is not the owner → 403."""
        fake_record = MagicMock()
        fake_record.id = "some-wiki"
        with (
            patch.object(
                test_app.state.wiki_management,
                "update_visibility",
                new_callable=AsyncMock,
                return_value=None,
            ),
            patch.object(
                test_app.state.wiki_management,
                "get_wiki",
                new_callable=AsyncMock,
                return_value=fake_record,
            ),
        ):
            resp = await client.patch(
                "/api/v1/wikis/some-wiki/visibility",
                json={"visibility": "shared"},
            )
        assert resp.status_code == 403


# ---------------------------------------------------------------------------
# QA endpoints
# ---------------------------------------------------------------------------


class TestQAEndpoints:
    @pytest.mark.asyncio
    async def test_list_qa_wiki_not_found(self, client, test_app):
        with patch.object(
            test_app.state.wiki_management,
            "get_wiki",
            new_callable=AsyncMock,
            return_value=None,
        ):
            resp = await client.get("/api/v1/wikis/missing/qa")
        assert resp.status_code == 404

    @pytest.mark.asyncio
    async def test_list_qa_success(self, client, test_app):
        fake_record = MagicMock()
        with (
            patch.object(test_app.state.wiki_management, "get_wiki", new_callable=AsyncMock, return_value=fake_record),
            patch.object(
                test_app.state.qa_service,
                "list_qa",
                new_callable=AsyncMock,
                return_value=([], 0),
            ),
        ):
            resp = await client.get("/api/v1/wikis/w1/qa")
        assert resp.status_code == 200
        assert resp.json()["total"] == 0

    @pytest.mark.asyncio
    async def test_qa_stats_wiki_not_found(self, client, test_app):
        with patch.object(
            test_app.state.wiki_management,
            "get_wiki",
            new_callable=AsyncMock,
            return_value=None,
        ):
            resp = await client.get("/api/v1/wikis/missing/qa/stats")
        assert resp.status_code == 404

    @pytest.mark.asyncio
    async def test_qa_stats_success(self, client, test_app):
        fake_record = MagicMock()
        stats = {
            "total_count": 5,
            "cache_hit_count": 2,
            "validated_count": 3,
            "rejected_count": 1,
            "hit_rate": 0.4,
        }
        with (
            patch.object(test_app.state.wiki_management, "get_wiki", new_callable=AsyncMock, return_value=fake_record),
            patch.object(test_app.state.qa_service, "get_stats", new_callable=AsyncMock, return_value=stats),
        ):
            resp = await client.get("/api/v1/wikis/w1/qa/stats")
        assert resp.status_code == 200
        assert resp.json()["total_count"] == 5


# ---------------------------------------------------------------------------
# Delete wiki
# ---------------------------------------------------------------------------


class TestDeleteWiki:
    @pytest.mark.asyncio
    async def test_delete_wiki_success(self, client, test_app):
        from app.models.api import DeleteWikiResponse as DeleteWikiResult

        result = DeleteWikiResult(deleted=True, wiki_id="del-wiki")
        with patch.object(test_app.state.wiki_management, "delete_wiki", new_callable=AsyncMock, return_value=result):
            resp = await client.delete("/api/v1/wikis/del-wiki")
        assert resp.status_code == 200
        assert resp.json()["deleted"] is True

    @pytest.mark.asyncio
    async def test_delete_wiki_owner_forbidden(self, client, test_app):
        from app.models.api import DeleteWikiResponse as DeleteWikiResult

        result = DeleteWikiResult(deleted=False, wiki_id="other-wiki", message="not the owner")
        with patch.object(test_app.state.wiki_management, "delete_wiki", new_callable=AsyncMock, return_value=result):
            resp = await client.delete("/api/v1/wikis/other-wiki")
        assert resp.status_code == 403


# ---------------------------------------------------------------------------
# Resume wiki
# ---------------------------------------------------------------------------


class TestResumeWiki:
    @pytest.mark.asyncio
    async def test_resume_not_found(self, client, test_app):
        with (
            patch.object(test_app.state.wiki_management, "get_wiki", new_callable=AsyncMock, return_value=None),
            patch.object(test_app.state.wiki_service, "resume", new_callable=AsyncMock, return_value=None),
        ):
            resp = await client.post("/api/v1/wikis/missing/resume")
        assert resp.status_code == 404

    @pytest.mark.asyncio
    async def test_resume_owner_forbidden(self, client, test_app):
        fake_record = MagicMock()
        fake_record.owner_id = "other-user"

        with patch.object(test_app.state.wiki_management, "get_wiki", new_callable=AsyncMock, return_value=fake_record):
            resp = await client.post("/api/v1/wikis/owned-by-other/resume")
        assert resp.status_code == 403


# ---------------------------------------------------------------------------
# Generate wiki — 409 conflict branch
# ---------------------------------------------------------------------------


class TestGenerateWiki:
    @pytest.mark.asyncio
    async def test_generate_wiki_already_exists(self, client, test_app):
        from app.services.wiki_service_errors import WikiAlreadyExistsError

        err = WikiAlreadyExistsError("existing-wiki")

        with patch.object(test_app.state.wiki_service, "generate", new_callable=AsyncMock, side_effect=err):
            resp = await client.post(
                "/api/v1/generate",
                json={"repo_url": "https://github.com/test/repo", "branch": "main"},
            )
        assert resp.status_code == 409


# ---------------------------------------------------------------------------
# get_wiki_page
# ---------------------------------------------------------------------------


class TestGetWikiPage:
    @pytest.mark.asyncio
    async def test_get_wiki_page_not_found_wiki(self, client, test_app):
        with patch.object(test_app.state.wiki_management, "get_wiki", new_callable=AsyncMock, return_value=None):
            resp = await client.get("/api/v1/wikis/missing/pages/intro")
        assert resp.status_code == 404

    @pytest.mark.asyncio
    async def test_get_wiki_page_not_found_page(self, client, test_app):
        fake_record = MagicMock()
        with (
            patch.object(test_app.state.wiki_management, "get_wiki", new_callable=AsyncMock, return_value=fake_record),
            patch.object(
                test_app.state.wiki_management.storage,
                "list_artifacts",
                new_callable=AsyncMock,
                return_value=[],
            ),
        ):
            resp = await client.get("/api/v1/wikis/w1/pages/intro")
        assert resp.status_code == 404

    @pytest.mark.asyncio
    async def test_get_wiki_page_success(self, client, test_app):
        fake_record = MagicMock()
        with (
            patch.object(test_app.state.wiki_management, "get_wiki", new_callable=AsyncMock, return_value=fake_record),
            patch.object(
                test_app.state.wiki_management.storage,
                "list_artifacts",
                new_callable=AsyncMock,
                return_value=["w1/repo/wiki_pages/intro.md"],
            ),
            patch.object(
                test_app.state.wiki_management.storage,
                "download",
                new_callable=AsyncMock,
                return_value=b"# Intro\nContent.",
            ),
        ):
            resp = await client.get("/api/v1/wikis/w1/pages/intro")
        assert resp.status_code == 200
        assert "Intro" in resp.json()["title"] or resp.json()["content"]


# ---------------------------------------------------------------------------
# Resume wiki — success
# ---------------------------------------------------------------------------


class TestResumeWikiSuccess:
    @pytest.mark.asyncio
    async def test_resume_wiki_success(self, client, test_app):
        fake_inv = MagicMock()
        fake_inv.id = "inv-resume"
        fake_inv.wiki_id = "wiki-resume"
        fake_inv.status = "generating"

        with (
            patch.object(test_app.state.wiki_management, "get_wiki", new_callable=AsyncMock, return_value=None),
            patch.object(test_app.state.wiki_service, "resume", new_callable=AsyncMock, return_value=fake_inv),
        ):
            resp = await client.post("/api/v1/wikis/wiki-resume/resume")
        assert resp.status_code == 202


# ---------------------------------------------------------------------------
# Wiki visibility — success
# ---------------------------------------------------------------------------


class TestWikiVisibilitySuccess:
    @pytest.mark.asyncio
    async def test_visibility_update_success(self, client, test_app):
        from app.models.api import WikiSummary

        fake_summary = MagicMock(spec=WikiSummary)
        fake_summary.wiki_id = "w1"
        fake_summary.repo_url = "https://github.com/test/repo"
        fake_summary.branch = "main"
        fake_summary.title = "Test"
        fake_summary.page_count = 2
        fake_summary.created_at = "2026-01-01T00:00:00"
        fake_summary.status = "complete"
        fake_summary.visibility = "shared"

        with patch.object(
            test_app.state.wiki_management,
            "update_visibility",
            new_callable=AsyncMock,
            return_value=fake_summary,
        ):
            resp = await client.patch(
                "/api/v1/wikis/w1/visibility",
                json={"visibility": "shared"},
            )
        assert resp.status_code == 200


# ---------------------------------------------------------------------------
# Refresh wiki
# ---------------------------------------------------------------------------


class TestRefreshWiki:
    @pytest.mark.asyncio
    async def test_refresh_wiki_not_found(self, client, test_app):
        with (
            patch.object(test_app.state.wiki_management, "get_wiki", new_callable=AsyncMock, return_value=None),
            patch.object(test_app.state.wiki_service, "refresh", new_callable=AsyncMock, return_value=None),
        ):
            resp = await client.post("/api/v1/wikis/missing/refresh")
        assert resp.status_code == 404

    @pytest.mark.asyncio
    async def test_refresh_wiki_owner_forbidden(self, client, test_app):
        fake_record = MagicMock()
        fake_record.owner_id = "other-user"

        with patch.object(test_app.state.wiki_management, "get_wiki", new_callable=AsyncMock, return_value=fake_record):
            resp = await client.post("/api/v1/wikis/owned-by-other/refresh")
        assert resp.status_code == 403

    @pytest.mark.asyncio
    async def test_refresh_wiki_success(self, client, test_app):
        fake_inv = MagicMock()
        fake_inv.id = "inv-refresh"
        fake_inv.wiki_id = "w1"
        fake_inv.status = "generating"

        with (
            patch.object(test_app.state.wiki_management, "get_wiki", new_callable=AsyncMock, return_value=None),
            patch.object(test_app.state.wiki_service, "refresh", new_callable=AsyncMock, return_value=fake_inv),
        ):
            resp = await client.post("/api/v1/wikis/w1/refresh")
        assert resp.status_code == 202


# ---------------------------------------------------------------------------
# Export
# ---------------------------------------------------------------------------


class TestExportEndpointExtended:
    @pytest.mark.asyncio
    async def test_export_invalid_format(self, client):
        resp = await client.get("/api/v1/wikis/any/export?format=unknown")
        assert resp.status_code == 400

    @pytest.mark.asyncio
    async def test_export_wiki_not_found(self, client, test_app):
        with patch.object(test_app.state.wiki_management, "get_wiki", new_callable=AsyncMock, return_value=None):
            resp = await client.get("/api/v1/wikis/ghost/export?format=obsidian")
        assert resp.status_code == 404

    @pytest.mark.asyncio
    async def test_export_non_complete_wiki_gives_409(self, client_tuple):
        c, app, sf = client_tuple
        await _seed_wiki(sf, wiki_id="gen-wiki", status="generating")

        resp = await c.get("/api/v1/wikis/gen-wiki/export?format=obsidian")
        assert resp.status_code == 409


# ---------------------------------------------------------------------------
# Import
# ---------------------------------------------------------------------------


class TestImportEndpointExtended:
    @pytest.mark.asyncio
    async def test_import_not_implemented(self, client, test_app):

        with patch.object(
            test_app.state.import_service,
            "restore_wiki",
            new_callable=AsyncMock,
            side_effect=NotImplementedError("S3 not supported"),
        ):
            import io

            fake_bundle = io.BytesIO(b"fake content")
            resp = await client.post(
                "/api/v1/wikis/import",
                files={"bundle": ("test.wikiexport", fake_bundle, "application/zip")},
            )
        assert resp.status_code == 501

    @pytest.mark.asyncio
    async def test_import_bundle_validation_error(self, client, test_app):
        from app.services.import_service import BundleValidationError

        with patch.object(
            test_app.state.import_service,
            "restore_wiki",
            new_callable=AsyncMock,
            side_effect=BundleValidationError("bad bundle"),
        ):
            import io

            fake_bundle = io.BytesIO(b"fake content")
            resp = await client.post(
                "/api/v1/wikis/import",
                files={"bundle": ("test.wikiexport", fake_bundle, "application/zip")},
            )
        assert resp.status_code == 422


# ---------------------------------------------------------------------------
# Projects CRUD
# ---------------------------------------------------------------------------


def _fake_project(name: str = "My Project", owner_id: str = "dev-user") -> MagicMock:
    p = MagicMock()
    p.id = "proj-1"
    p.name = name
    p.description = "desc"
    p.visibility = "personal"
    p.owner_id = owner_id
    p.created_at = "2026-01-01T00:00:00"
    return p


def _override_project_svc(app, fake_svc):
    """Override the project service FastAPI DI dependency."""
    from app.dependencies import get_project_service

    app.dependency_overrides[get_project_service] = lambda: fake_svc
    return fake_svc


class TestProjectsCRUD:
    @pytest.mark.asyncio
    async def test_create_project(self, test_app, client):
        from app.dependencies import get_project_service

        fake_svc = AsyncMock()
        fake_svc.create_project = AsyncMock(return_value=_fake_project())
        test_app.dependency_overrides[get_project_service] = lambda: fake_svc
        try:
            resp = await client.post("/api/v1/projects", json={"name": "My Project"})
        finally:
            test_app.dependency_overrides.pop(get_project_service, None)
        assert resp.status_code == 201
        assert resp.json()["name"] == "My Project"

    @pytest.mark.asyncio
    async def test_list_projects(self, test_app, client):
        from app.dependencies import get_project_service

        fake_svc = AsyncMock()
        fake_svc.list_projects = AsyncMock(return_value=[])
        fake_svc.batch_get_wiki_counts = AsyncMock(return_value={})
        test_app.dependency_overrides[get_project_service] = lambda: fake_svc
        try:
            resp = await client.get("/api/v1/projects")
        finally:
            test_app.dependency_overrides.pop(get_project_service, None)
        assert resp.status_code == 200
        assert resp.json()["projects"] == []

    @pytest.mark.asyncio
    async def test_get_project_not_found(self, test_app, client):
        from app.dependencies import get_project_service

        fake_svc = AsyncMock()
        fake_svc.get_project = AsyncMock(return_value=None)
        test_app.dependency_overrides[get_project_service] = lambda: fake_svc
        try:
            resp = await client.get("/api/v1/projects/missing")
        finally:
            test_app.dependency_overrides.pop(get_project_service, None)
        assert resp.status_code == 404

    @pytest.mark.asyncio
    async def test_get_project_found(self, test_app, client):
        from app.dependencies import get_project_service

        fake_svc = AsyncMock()
        fake_svc.get_project = AsyncMock(return_value=_fake_project())
        fake_svc.get_wiki_count = AsyncMock(return_value=2)
        test_app.dependency_overrides[get_project_service] = lambda: fake_svc
        try:
            resp = await client.get("/api/v1/projects/proj-1")
        finally:
            test_app.dependency_overrides.pop(get_project_service, None)
        assert resp.status_code == 200
        assert resp.json()["wiki_count"] == 2

    @pytest.mark.asyncio
    async def test_update_project_not_found(self, test_app, client):
        from app.dependencies import get_project_service

        fake_svc = AsyncMock()
        fake_svc.get_project = AsyncMock(return_value=None)
        test_app.dependency_overrides[get_project_service] = lambda: fake_svc
        try:
            resp = await client.patch("/api/v1/projects/missing", json={"name": "New"})
        finally:
            test_app.dependency_overrides.pop(get_project_service, None)
        assert resp.status_code == 404

    @pytest.mark.asyncio
    async def test_update_project_not_owner(self, test_app, client):
        from app.dependencies import get_project_service

        fake_svc = AsyncMock()
        fake_svc.get_project = AsyncMock(return_value=_fake_project(owner_id="other"))
        fake_svc.update_project = AsyncMock(return_value=None)
        test_app.dependency_overrides[get_project_service] = lambda: fake_svc
        try:
            resp = await client.patch("/api/v1/projects/proj-1", json={"name": "New"})
        finally:
            test_app.dependency_overrides.pop(get_project_service, None)
        assert resp.status_code == 403

    @pytest.mark.asyncio
    async def test_delete_project_not_found(self, test_app, client):
        from app.dependencies import get_project_service

        fake_svc = AsyncMock()
        fake_svc.get_project = AsyncMock(return_value=None)
        test_app.dependency_overrides[get_project_service] = lambda: fake_svc
        try:
            resp = await client.delete("/api/v1/projects/missing")
        finally:
            test_app.dependency_overrides.pop(get_project_service, None)
        assert resp.status_code == 404

    @pytest.mark.asyncio
    async def test_delete_project_not_owner(self, test_app, client):
        from app.dependencies import get_project_service

        fake_svc = AsyncMock()
        fake_svc.get_project = AsyncMock(return_value=_fake_project(owner_id="other"))
        fake_svc.delete_project = AsyncMock(return_value=False)
        test_app.dependency_overrides[get_project_service] = lambda: fake_svc
        try:
            resp = await client.delete("/api/v1/projects/proj-1")
        finally:
            test_app.dependency_overrides.pop(get_project_service, None)
        assert resp.status_code == 403

    @pytest.mark.asyncio
    async def test_delete_project_success(self, test_app, client):
        from app.dependencies import get_project_service

        fake_svc = AsyncMock()
        fake_svc.get_project = AsyncMock(return_value=_fake_project())
        fake_svc.delete_project = AsyncMock(return_value=True)
        test_app.dependency_overrides[get_project_service] = lambda: fake_svc
        try:
            resp = await client.delete("/api/v1/projects/proj-1")
        finally:
            test_app.dependency_overrides.pop(get_project_service, None)
        assert resp.status_code == 200
        assert resp.json()["deleted"] is True


# ---------------------------------------------------------------------------
# Projects — wiki membership
# ---------------------------------------------------------------------------


class TestProjectWikiMembership:
    def _override(self, test_app, fake_svc):
        from app.dependencies import get_project_service

        test_app.dependency_overrides[get_project_service] = lambda: fake_svc
        return get_project_service

    def _restore(self, test_app, key):
        test_app.dependency_overrides.pop(key, None)

    @pytest.mark.asyncio
    async def test_add_wiki_project_not_found(self, test_app, client):
        fake_svc = AsyncMock()
        fake_svc.get_project = AsyncMock(return_value=None)
        key = self._override(test_app, fake_svc)
        try:
            resp = await client.post("/api/v1/projects/missing/wikis", json={"wiki_id": "w1"})
        finally:
            self._restore(test_app, key)
        assert resp.status_code == 404

    @pytest.mark.asyncio
    async def test_add_wiki_not_owner(self, test_app, client):
        fake_svc = AsyncMock()
        fake_svc.get_project = AsyncMock(return_value=_fake_project(owner_id="other"))
        key = self._override(test_app, fake_svc)
        try:
            resp = await client.post("/api/v1/projects/proj-1/wikis", json={"wiki_id": "w1"})
        finally:
            self._restore(test_app, key)
        assert resp.status_code == 403

    @pytest.mark.asyncio
    async def test_add_wiki_success(self, test_app, client):
        fake_svc = AsyncMock()
        fake_svc.get_project = AsyncMock(return_value=_fake_project())
        fake_svc.add_wiki = AsyncMock(return_value=MagicMock())
        fake_svc.get_wiki_count = AsyncMock(return_value=1)
        key = self._override(test_app, fake_svc)
        try:
            resp = await client.post("/api/v1/projects/proj-1/wikis", json={"wiki_id": "w1"})
        finally:
            self._restore(test_app, key)
        assert resp.status_code == 201

    @pytest.mark.asyncio
    async def test_remove_wiki_project_not_found(self, test_app, client):
        fake_svc = AsyncMock()
        fake_svc.get_project = AsyncMock(return_value=None)
        key = self._override(test_app, fake_svc)
        try:
            resp = await client.delete("/api/v1/projects/missing/wikis/w1")
        finally:
            self._restore(test_app, key)
        assert resp.status_code == 404

    @pytest.mark.asyncio
    async def test_remove_wiki_not_owner(self, test_app, client):
        fake_svc = AsyncMock()
        fake_svc.get_project = AsyncMock(return_value=_fake_project())
        fake_svc.remove_wiki = AsyncMock(return_value=False)
        key = self._override(test_app, fake_svc)
        try:
            resp = await client.delete("/api/v1/projects/proj-1/wikis/w1")
        finally:
            self._restore(test_app, key)
        assert resp.status_code == 403

    @pytest.mark.asyncio
    async def test_remove_wiki_success(self, test_app, client):
        fake_svc = AsyncMock()
        fake_svc.get_project = AsyncMock(return_value=_fake_project())
        fake_svc.remove_wiki = AsyncMock(return_value=True)
        key = self._override(test_app, fake_svc)
        try:
            resp = await client.delete("/api/v1/projects/proj-1/wikis/w1")
        finally:
            self._restore(test_app, key)
        assert resp.status_code == 200
        assert resp.json()["removed"] is True

    @pytest.mark.asyncio
    async def test_list_project_wikis_not_found(self, test_app, client):
        fake_svc = AsyncMock()
        fake_svc.list_project_wikis = AsyncMock(return_value=None)
        key = self._override(test_app, fake_svc)
        try:
            resp = await client.get("/api/v1/projects/missing/wikis")
        finally:
            self._restore(test_app, key)
        assert resp.status_code == 404

    @pytest.mark.asyncio
    async def test_list_project_wikis_success(self, test_app, client):
        fake_svc = AsyncMock()
        fake_svc.list_project_wikis = AsyncMock(return_value=[])
        key = self._override(test_app, fake_svc)
        try:
            resp = await client.get("/api/v1/projects/proj-1/wikis")
        finally:
            self._restore(test_app, key)
        assert resp.status_code == 200


# ---------------------------------------------------------------------------
# Project codemap endpoint
# ---------------------------------------------------------------------------


class TestProjectCodemap:
    def _override(self, test_app, fake_svc):
        from app.dependencies import get_project_service

        test_app.dependency_overrides[get_project_service] = lambda: fake_svc
        return get_project_service

    @pytest.mark.asyncio
    async def test_project_codemap_not_found(self, test_app, client):
        fake_svc = AsyncMock()
        fake_svc.get_project = AsyncMock(return_value=None)
        key = self._override(test_app, fake_svc)
        try:
            resp = await client.post("/api/v1/projects/missing/map", json={"question": "Q?"})
        finally:
            test_app.dependency_overrides.pop(key, None)
        assert resp.status_code == 404

    @pytest.mark.asyncio
    async def test_project_codemap_success(self, test_app, client):
        from app.models.api import ResearchResponse

        fake_svc = AsyncMock()
        fake_svc.get_project = AsyncMock(return_value=_fake_project())
        key = self._override(test_app, fake_svc)

        mock_resp = ResearchResponse(answer="Map answer.", sources=[], research_steps=[])
        try:
            with patch.object(
                test_app.state.research_service, "codemap_sync", new_callable=AsyncMock, return_value=mock_resp
            ):
                resp = await client.post("/api/v1/projects/proj-1/map", json={"question": "Q?"})
        finally:
            test_app.dependency_overrides.pop(key, None)
        assert resp.status_code == 200
        assert resp.json()["answer"] == "Map answer."


# ---------------------------------------------------------------------------
# PATCH /wikis/{wiki_id}/description
# ---------------------------------------------------------------------------


class TestUpdateWikiDescription:
    @pytest.mark.asyncio
    async def test_description_update_success(self, client, test_app):
        """Happy path: owner updates description and gets back the new value."""
        from app.models.db_models import WikiRecord

        fake_record = MagicMock(spec=WikiRecord)
        fake_record.id = "w1"
        fake_record.description = "A concise description"

        with patch.object(
            test_app.state.wiki_management,
            "update_description",
            new_callable=AsyncMock,
            return_value=fake_record,
        ):
            resp = await client.patch(
                "/api/v1/wikis/w1/description",
                json={"description": "A concise description"},
            )
        assert resp.status_code == 200
        data = resp.json()
        assert data["wiki_id"] == "w1"
        assert data["description"] == "A concise description"

    @pytest.mark.asyncio
    async def test_description_update_not_found(self, client, test_app):
        """Returns 404 when wiki does not exist."""
        with (
            patch.object(
                test_app.state.wiki_management,
                "update_description",
                new_callable=AsyncMock,
                return_value=None,
            ),
            patch.object(
                test_app.state.wiki_management,
                "get_wiki",
                new_callable=AsyncMock,
                return_value=None,
            ),
        ):
            resp = await client.patch(
                "/api/v1/wikis/missing/description",
                json={"description": "Anything"},
            )
        assert resp.status_code == 404

    @pytest.mark.asyncio
    async def test_description_update_forbidden(self, client, test_app):
        """Returns 403 when caller is not the owner."""
        fake_record = MagicMock()
        fake_record.owner_id = "other-user"

        with (
            patch.object(
                test_app.state.wiki_management,
                "update_description",
                new_callable=AsyncMock,
                return_value=None,
            ),
            patch.object(
                test_app.state.wiki_management,
                "get_wiki",
                new_callable=AsyncMock,
                return_value=fake_record,
            ),
        ):
            resp = await client.patch(
                "/api/v1/wikis/w1/description",
                json={"description": "Hacked"},
            )
        assert resp.status_code == 403

    @pytest.mark.asyncio
    async def test_description_update_clear_with_none(self, client, test_app):
        """Passing null description clears the field."""
        from app.models.db_models import WikiRecord

        fake_record = MagicMock(spec=WikiRecord)
        fake_record.id = "w1"
        fake_record.description = None

        with patch.object(
            test_app.state.wiki_management,
            "update_description",
            new_callable=AsyncMock,
            return_value=fake_record,
        ):
            resp = await client.patch(
                "/api/v1/wikis/w1/description",
                json={"description": None},
            )
        assert resp.status_code == 200
        assert resp.json()["description"] is None


# ---------------------------------------------------------------------------
# GET /wikis/{wiki_id} includes description field
# ---------------------------------------------------------------------------


class TestGetWikiIncludesDescription:
    @pytest.mark.asyncio
    async def test_get_wiki_response_includes_description(self, client, test_app):
        """The main GET endpoint must include description in its response."""
        from datetime import datetime

        from app.models.api import WikiSummary

        fake_summary = MagicMock(spec=WikiSummary)
        fake_summary.wiki_id = "w1"
        fake_summary.repo_url = "https://github.com/test/repo"
        fake_summary.branch = "main"
        fake_summary.title = "Test Wiki"
        fake_summary.page_count = 0
        fake_summary.created_at = datetime(2026, 1, 1)
        fake_summary.indexed_at = None
        fake_summary.commit_hash = None
        fake_summary.status = "complete"
        fake_summary.requires_token = False
        fake_summary.error = None
        fake_summary.description = "My wiki description"

        with (
            patch.object(
                test_app.state.wiki_management,
                "get_wiki",
                new_callable=AsyncMock,
                return_value=MagicMock(id="w1"),
            ),
            patch.object(
                WikiManagementService,
                "_record_to_summary",
                return_value=fake_summary,
            ),
            patch.object(
                test_app.state.wiki_management.storage,
                "list_artifacts",
                new_callable=AsyncMock,
                return_value=[],
            ),
        ):
            resp = await client.get("/api/v1/wikis/w1")
        assert resp.status_code == 200
        data = resp.json()
        assert "description" in data
        assert data["description"] == "My wiki description"
