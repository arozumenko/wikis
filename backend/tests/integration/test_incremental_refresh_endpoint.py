"""End-to-end test for ``POST /api/v1/wikis/{id}/incremental-refresh`` (#116 PR 5).

Validates the full path: route → WikiService.incremental_refresh →
IncrementalRegenService → SSE events on the invocation.

The structural-handler is a stub in PR 5; this test exercises the
trivial path (which doesn't need an LLM) and asserts the response
shape + that the invocation captures the expected event sequence.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from app.main import create_app
from app.models.db_models import Base, WikiRecord
from app.services.wiki_management import WikiManagementService
from app.storage.local import LocalArtifactStorage
from httpx import ASGITransport, AsyncClient
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

from app.core.storage.incremental import compute_content_hash
from app.core.unified_db import UnifiedWikiDB


def _seed_unified_db(
    cache_dir: Path,
    *,
    cache_key: str,
    repo_url: str,
    branch: str,
) -> Path:
    """Seed a .wiki.db with one wiki page that cites sym-moved. The
    incremental run will rewrite the page's path attribute (trivial)."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    db_path = cache_dir / f"{cache_key}.wiki.db"

    db = UnifiedWikiDB(db_path, embedding_dim=8)
    try:
        db.upsert_node(
            "sym-moved",
            rel_path="old_path.py",
            file_name="old_path.py",
            language="python",
            symbol_name="moved",
            symbol_type="function",
            is_architectural=1,
            source_text="def moved(): pass",
        )
        db.upsert_wiki_page_with_symbols(
            {
                "page_id": "page-1",
                "wiki_id": "test-wiki",
                "title": "Movers",
                "anchor_slug": "movers",
                "primary_symbol_id": "sym-moved",
            },
            [("sym-moved", "primary")],
        )
    finally:
        db.close()

    # cache_index.json so _derive_cache_key resolves the wiki.
    url = repo_url.rstrip("/").removesuffix(".git")
    parts = url.split("/")
    owner_repo = f"{parts[-2]}/{parts[-1]}"
    repo_identifier = f"{owner_repo}:{branch}"
    cache_index = {
        "refs": {repo_identifier: repo_identifier},
        repo_identifier: cache_key,
    }
    (cache_dir / "cache_index.json").write_text(json.dumps(cache_index))
    return db_path


@pytest.fixture
async def client_and_setup(tmp_path, monkeypatch):
    monkeypatch.setenv("AUTH_ENABLED", "false")
    monkeypatch.setenv("LLM_API_KEY", "test-key")
    monkeypatch.setenv("CACHE_DIR", str(tmp_path / "cache"))

    _seed_unified_db(
        tmp_path / "cache",
        cache_key="testkey",
        repo_url="https://github.com/example/widgets",
        branch="main",
    )

    # Pre-seed the wiki page body in artifact storage so the orchestrator
    # has something to read + patch.
    artifacts_root = tmp_path / "artifacts"
    artifacts_root.mkdir()
    page_dir = artifacts_root / "wiki_artifacts" / "test-wiki" / "wiki_pages"
    page_dir.mkdir(parents=True)
    (page_dir / "page-1.md").write_text(
        '# Movers\n\n<code_context path="old_path.py">def moved(): pass</code_context>'
    )

    engine = create_async_engine(
        "sqlite+aiosqlite://", connect_args={"check_same_thread": False}
    )
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    sf = async_sessionmaker(engine, expire_on_commit=False)

    async with sf() as session, session.begin():
        session.add(
            WikiRecord(
                id="test-wiki",
                owner_id="dev-user",
                repo_url="https://github.com/example/widgets",
                branch="main",
                title="Test Wiki",
                page_count=1,
                status="complete",
            )
        )

    app = create_app()
    async with app.router.lifespan_context(app):
        app.state.settings.cache_dir = str(tmp_path / "cache")
        storage = LocalArtifactStorage(str(artifacts_root))
        app.state.storage = storage
        app.state.wiki_management = WikiManagementService(storage, sf)
        # WikiService picks up app.state.storage via dependency injection;
        # also rebuild the wiki_service with the storage we created.
        from app.services.wiki_service import WikiService
        app.state.wiki_service = WikiService(
            app.state.settings, storage, app.state.wiki_management,
        )

        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as c:
            yield c, app

    await engine.dispose()


class TestIncrementalRefreshEndpoint:
    @pytest.mark.asyncio
    async def test_returns_202_with_invocation_id(self, client_and_setup) -> None:
        client, _ = client_and_setup
        resp = await client.post(
            "/api/v1/wikis/test-wiki/incremental-refresh",
            json={
                "parsed_nodes": [
                    {
                        "node_id": "sym-moved",
                        "content_hash": compute_content_hash("def moved(): pass"),
                        "rel_path": "new_path.py",  # path changed → MOVED
                    },
                ],
            },
        )
        assert resp.status_code == 202, resp.text
        body = resp.json()
        assert body["wiki_id"] == "test-wiki"
        assert body["status"] == "running"
        assert body["invocation_id"]
        assert body["page_count"] == 1

    @pytest.mark.asyncio
    async def test_404_when_wiki_unknown(self, client_and_setup) -> None:
        client, _ = client_and_setup
        resp = await client.post(
            "/api/v1/wikis/missing/incremental-refresh",
            json={"parsed_nodes": []},
        )
        assert resp.status_code == 404

    @pytest.mark.asyncio
    async def test_409_when_refresh_already_in_progress(
        self, client_and_setup,
    ) -> None:
        """#140: a second concurrent /incremental-refresh on the same
        wiki must return 409 with the in-flight invocation_id so the
        caller can join its SSE stream instead of spawning a racing run."""
        client, app = client_and_setup
        body = {
            "parsed_nodes": [
                {
                    "node_id": "sym-moved",
                    "content_hash": compute_content_hash("def moved(): pass"),
                    "rel_path": "new_path.py",
                },
            ],
        }
        # First request: register an in-flight invocation in the service.
        # We inject one directly to avoid racing the actual orchestrator.
        from app.models.invocation import Invocation

        service = app.state.wiki_service
        service._invocations["in-flight-fake"] = Invocation(
            id="in-flight-fake",
            wiki_id="test-wiki",
            repo_url="https://github.com/example/widgets",
            branch="main",
            status="running",
            owner_id="dev-user",
        )

        resp = await client.post(
            "/api/v1/wikis/test-wiki/incremental-refresh",
            json=body,
        )
        assert resp.status_code == 409, resp.text
        # The in-flight invocation_id is surfaced so the caller can
        # subscribe to its SSE stream.
        assert resp.headers["X-Incremental-Invocation-Id"] == "in-flight-fake"

    @pytest.mark.asyncio
    async def test_invocation_status_flips_terminal_after_run(
        self, client_and_setup,
    ) -> None:
        """C1 regression test: after the background ``_run()`` completes,
        ``invocation.status`` must transition off ``"running"``. Without
        this, the widened idempotency guard would 409-lock the wiki
        forever — every subsequent refresh (or full generate) would see
        the phantom ``"running"`` entry from a prior completed run.
        """
        client, app = client_and_setup
        body = {
            "parsed_nodes": [
                {
                    "node_id": "sym-moved",
                    "content_hash": compute_content_hash("def moved(): pass"),
                    "rel_path": "new_path.py",
                },
            ],
        }
        resp = await client.post(
            "/api/v1/wikis/test-wiki/incremental-refresh", json=body,
        )
        assert resp.status_code == 202
        invocation_id = resp.json()["invocation_id"]

        # Wait for the background task to complete. The fixture's
        # WikiService stashes the task in self._tasks; awaiting it
        # gives us the deterministic "run finished" signal.
        service = app.state.wiki_service
        task = service._tasks.get(invocation_id)
        assert task is not None
        await task

        # Status must be terminal — not "running".
        invocation = service._invocations[invocation_id]
        assert invocation.status in {"complete", "failed"}, (
            f"expected terminal status, got {invocation.status!r}"
        )

        # #146 Gap 2: ``completed_at`` must be set so
        # ``_cleanup_old_invocations`` can eventually reclaim memory
        # for this entry. Before the fix, terminal branches in the
        # incremental ``_run()`` set only ``status`` and the dict
        # grew monotonically until process restart.
        assert invocation.completed_at is not None, (
            "completed_at must be set on terminal-state transitions"
        )

        # And a subsequent refresh must NOT be 409'd by the completed entry.
        resp2 = await client.post(
            "/api/v1/wikis/test-wiki/incremental-refresh", json=body,
        )
        assert resp2.status_code == 202, resp2.text

    @pytest.mark.asyncio
    async def test_409_when_full_generate_in_progress(
        self, client_and_setup,
    ) -> None:
        """#140 follow-up: incremental refresh must also reject when a
        full ``generate()`` is mid-flight on the same wiki — both write
        to the same .wiki.db and an incremental run during a full regen
        would race the content_hash updates."""
        client, app = client_and_setup
        body = {
            "parsed_nodes": [
                {
                    "node_id": "sym-moved",
                    "content_hash": compute_content_hash("def moved(): pass"),
                    "rel_path": "new_path.py",
                },
            ],
        }
        from app.models.invocation import Invocation

        service = app.state.wiki_service
        service._invocations["full-regen-fake"] = Invocation(
            id="full-regen-fake",
            wiki_id="test-wiki",
            repo_url="https://github.com/example/widgets",
            branch="main",
            status="generating",  # full generate() uses this status
            owner_id="dev-user",
        )

        resp = await client.post(
            "/api/v1/wikis/test-wiki/incremental-refresh",
            json=body,
        )
        assert resp.status_code == 409, resp.text
        assert resp.headers["X-Incremental-Invocation-Id"] == "full-regen-fake"

    @pytest.mark.asyncio
    async def test_422_when_payload_exceeds_cap(self, client_and_setup) -> None:
        client, _ = client_and_setup
        oversized = [
            {"node_id": f"n{i}", "content_hash": "x", "rel_path": "x.py"}
            for i in range(50_001)
        ]
        resp = await client.post(
            "/api/v1/wikis/test-wiki/incremental-refresh",
            json={"parsed_nodes": oversized},
        )
        assert resp.status_code == 422
