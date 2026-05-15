"""Integration tests for ``POST /api/v1/wikis/{wiki_id}/diff`` (#116 PR 2).

End-to-end exercise of the diff endpoint: seeds a WikiRecord, builds a
real ``.wiki.db`` with PR 1's tables populated, points cache_index at it,
and asserts the response shape matches the change detection contract.

These tests use the same conftest-driven client fixture as the other
integration tests (in-memory SQLite for the web DB; on-disk .wiki.db for
the unified store).
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


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _seed_unified_db(
    cache_dir: Path,
    *,
    cache_key: str,
    repo_url: str,
    branch: str,
) -> Path:
    """Build a .wiki.db on disk with PR 1's tables populated and register it
    in cache_index.json so ``_derive_cache_key`` finds it."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    db_path = cache_dir / f"{cache_key}.wiki.db"

    db = UnifiedWikiDB(db_path, embedding_dim=8)
    try:
        # Three nodes — one will modify, one will stay, one will delete.
        for nid, src in [
            ("auth.AuthService", "class AuthService: pass"),
            ("auth.login", "def login(): pass"),
            ("cache.CacheManager", "class CacheManager: pass"),
        ]:
            db.upsert_node(
                nid,
                rel_path=f"{nid.split('.')[0]}/file.py",
                file_name="file.py",
                language="python",
                symbol_name=nid.split(".")[-1],
                symbol_type="class" if nid.endswith(("Service", "Manager")) else "function",
                source_text=src,
            )
        # One wiki page citing AuthService and login.
        db.upsert_wiki_page_with_symbols(
            {
                "page_id": "page-auth",
                "wiki_id": "test-wiki",
                "title": "Authentication",
                "anchor_slug": "authentication",
            },
            [("auth.AuthService", "primary"), ("auth.login", "related")],
        )
    finally:
        db.close()

    # Build cache_index.json so _derive_cache_key resolves repo/branch → cache_key.
    # Match the route's parsing rule: take the last two path components only.
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
async def client_and_db(tmp_path, monkeypatch):
    """Spin up the FastAPI app pointed at an on-disk cache dir with a
    seeded .wiki.db."""
    monkeypatch.setenv("AUTH_ENABLED", "false")
    monkeypatch.setenv("LLM_API_KEY", "test-key")
    monkeypatch.setenv("CACHE_DIR", str(tmp_path / "cache"))

    db_path = _seed_unified_db(
        tmp_path / "cache",
        cache_key="testkey",
        repo_url="https://github.com/example/widgets",
        branch="main",
    )

    engine = create_async_engine(
        "sqlite+aiosqlite://", connect_args={"check_same_thread": False}
    )
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    sf = async_sessionmaker(engine, expire_on_commit=False)

    # Seed the wiki record so the auth + lookup steps in the route succeed.
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
        # Force settings.cache_dir to the test cache so _derive_cache_key
        # finds our seeded cache_index.json. monkeypatch.setenv may load
        # too late if .env is present in the working directory.
        app.state.settings.cache_dir = str(tmp_path / "cache")

        storage = LocalArtifactStorage(str(tmp_path / "artifacts"))
        wiki_mgmt = WikiManagementService(storage, sf)
        app.state.storage = storage
        app.state.wiki_management = wiki_mgmt
        app.state.session_factory = sf

        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as c:
            yield c, db_path

    await engine.dispose()


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestDiffEndpoint:
    @pytest.mark.asyncio
    async def test_returns_empty_change_set_when_nothing_changed(
        self, client_and_db,
    ) -> None:
        client, _ = client_and_db
        # Pass the exact same nodes that were seeded → no changes.
        resp = await client.post(
            "/api/v1/wikis/test-wiki/diff",
            json={
                "parsed_nodes": [
                    {
                        "node_id": "auth.AuthService",
                        "content_hash": compute_content_hash("class AuthService: pass"),
                        "rel_path": "auth/file.py",
                    },
                    {
                        "node_id": "auth.login",
                        "content_hash": compute_content_hash("def login(): pass"),
                        "rel_path": "auth/file.py",
                    },
                    {
                        "node_id": "cache.CacheManager",
                        "content_hash": compute_content_hash("class CacheManager: pass"),
                        "rel_path": "cache/file.py",
                    },
                ],
            },
        )
        assert resp.status_code == 200, resp.text
        body = resp.json()
        assert body["wiki_id"] == "test-wiki"
        assert body["change_set"]["total"] == 0
        assert body["change_set"]["added"] == []
        assert body["change_set"]["modified"] == []
        assert body["affected_pages"] == []

    @pytest.mark.asyncio
    async def test_reports_modified_node_and_affected_page(
        self, client_and_db,
    ) -> None:
        client, _ = client_and_db
        # Change AuthService's body → modified; reverse index finds page-auth.
        resp = await client.post(
            "/api/v1/wikis/test-wiki/diff",
            json={
                "parsed_nodes": [
                    {
                        "node_id": "auth.AuthService",
                        "content_hash": compute_content_hash(
                            "class AuthService:\n    def authenticate(self): pass"
                        ),
                        "rel_path": "auth/file.py",
                    },
                    {
                        "node_id": "auth.login",
                        "content_hash": compute_content_hash("def login(): pass"),
                        "rel_path": "auth/file.py",
                    },
                    {
                        "node_id": "cache.CacheManager",
                        "content_hash": compute_content_hash("class CacheManager: pass"),
                        "rel_path": "cache/file.py",
                    },
                ],
            },
        )
        assert resp.status_code == 200, resp.text
        body = resp.json()
        assert body["change_set"]["total"] == 1
        mods = body["change_set"]["modified"]
        assert len(mods) == 1
        assert mods[0]["node_id"] == "auth.AuthService"
        assert mods[0]["kind"] == "modified"

        assert len(body["affected_pages"]) == 1
        page = body["affected_pages"][0]
        assert page["page_id"] == "page-auth"
        # wiki_pages row was populated → title + slug come back.
        assert page["title"] == "Authentication"
        assert page["anchor_slug"] == "authentication"

    @pytest.mark.asyncio
    async def test_403_when_caller_is_not_owner(
        self, tmp_path, monkeypatch,
    ) -> None:
        """The diff response leaks content_hash + node_id internals — only
        the wiki owner may call it, even if the wiki is shared-visibility.
        """
        monkeypatch.setenv("AUTH_ENABLED", "true")  # turn on real auth
        monkeypatch.setenv("LLM_API_KEY", "test-key")
        monkeypatch.setenv("CACHE_DIR", str(tmp_path / "cache"))
        _seed_unified_db(
            tmp_path / "cache",
            cache_key="testkey",
            repo_url="https://github.com/example/widgets",
            branch="main",
        )

        engine = create_async_engine(
            "sqlite+aiosqlite://", connect_args={"check_same_thread": False}
        )
        async with engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)
        sf = async_sessionmaker(engine, expire_on_commit=False)
        # Wiki owned by *another* user; visibility shared so get_wiki returns it.
        async with sf() as session, session.begin():
            session.add(
                WikiRecord(
                    id="test-wiki",
                    owner_id="other-user",
                    repo_url="https://github.com/example/widgets",
                    branch="main",
                    title="Test Wiki",
                    page_count=1,
                    status="complete",
                    visibility="shared",
                )
            )

        app = create_app()
        async with app.router.lifespan_context(app):
            app.state.settings.cache_dir = str(tmp_path / "cache")
            storage = LocalArtifactStorage(str(tmp_path / "artifacts"))
            app.state.storage = storage
            app.state.wiki_management = WikiManagementService(storage, sf)
            app.state.session_factory = sf

            # Stub get_current_user → returns a different user than owner.
            from app.auth import get_current_user
            from app.auth import CurrentUser

            async def _fake_user():
                return CurrentUser(id="dev-user", email="dev@example.com")
            app.dependency_overrides[get_current_user] = _fake_user

            async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as c:
                resp = await c.post(
                    "/api/v1/wikis/test-wiki/diff",
                    json={"parsed_nodes": []},
                )
                assert resp.status_code == 403, resp.text
                assert "owner" in resp.json().get("detail", "").lower()
        await engine.dispose()

    @pytest.mark.asyncio
    async def test_404_when_wiki_db_file_deleted(self, client_and_db) -> None:
        """cache_index entry can survive the .wiki.db file being deleted
        out-of-band — endpoint must return 404, not crash with 500."""
        client, db_path = client_and_db
        db_path.unlink()  # remove the .wiki.db file but leave cache_index.json
        resp = await client.post(
            "/api/v1/wikis/test-wiki/diff",
            json={"parsed_nodes": []},
        )
        assert resp.status_code == 404
        assert "missing" in resp.json().get("detail", "").lower()

    @pytest.mark.asyncio
    async def test_422_when_payload_exceeds_size_cap(self, client_and_db) -> None:
        """parsed_nodes is capped at 50_000 items to bound the reverse-index
        N+1 cost. Over-cap requests should fail Pydantic validation cleanly."""
        client, _ = client_and_db
        oversized = [
            {"node_id": f"n{i}", "content_hash": "x", "rel_path": "x.py"}
            for i in range(50_001)
        ]
        resp = await client.post(
            "/api/v1/wikis/test-wiki/diff",
            json={"parsed_nodes": oversized},
        )
        assert resp.status_code == 422

    @pytest.mark.asyncio
    async def test_404_when_wiki_unknown(self, client_and_db) -> None:
        client, _ = client_and_db
        resp = await client.post(
            "/api/v1/wikis/nonexistent/diff",
            json={"parsed_nodes": []},
        )
        assert resp.status_code == 404

    @pytest.mark.asyncio
    async def test_404_when_cache_missing(
        self, tmp_path, monkeypatch,
    ) -> None:
        """Endpoint must return 404 (not crash) when the unified DB is missing."""
        monkeypatch.setenv("AUTH_ENABLED", "false")
        monkeypatch.setenv("LLM_API_KEY", "test-key")
        # Point CACHE_DIR at an empty directory — no cache_index.json.
        monkeypatch.setenv("CACHE_DIR", str(tmp_path / "empty_cache"))
        (tmp_path / "empty_cache").mkdir()

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
            storage = LocalArtifactStorage(str(tmp_path / "artifacts"))
            app.state.storage = storage
            app.state.wiki_management = WikiManagementService(storage, sf)
            app.state.session_factory = sf
            async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as c:
                resp = await c.post(
                    "/api/v1/wikis/test-wiki/diff",
                    json={"parsed_nodes": []},
                )
                assert resp.status_code == 404
                assert "Unified DB" in resp.json().get("detail", "")
        await engine.dispose()
