"""Integration tests for the Projects API.

Tests the full HTTP surface of /api/v1/projects/* via FastAPI TestClient.

Run with:
    AUTH_ENABLED=false pytest tests/integration/test_project_routes.py -v
"""

from __future__ import annotations

import pytest
from httpx import ASGITransport, AsyncClient
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

from app.main import create_app
from app.models.db_models import Base, WikiRecord
from app.services.wiki_management import WikiManagementService
from app.storage.local import LocalArtifactStorage


async def _make_session_factory():
    engine = create_async_engine("sqlite+aiosqlite://", connect_args={"check_same_thread": False})
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    return engine, async_sessionmaker(engine, expire_on_commit=False)


@pytest.fixture
async def client(tmp_path, monkeypatch):
    """FastAPI test client with in-memory SQLite and AUTH_ENABLED=false."""
    monkeypatch.setenv("AUTH_ENABLED", "false")
    monkeypatch.setenv("LLM_API_KEY", "test-key")

    engine, session_factory = await _make_session_factory()
    app = create_app()

    async with app.router.lifespan_context(app):
        storage = LocalArtifactStorage(str(tmp_path))
        wiki_mgmt = WikiManagementService(storage, session_factory)
        app.state.storage = storage
        app.state.wiki_management = wiki_mgmt
        app.state.session_factory = session_factory

        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as c:
            yield c, app, session_factory

    await engine.dispose()


async def _seed_wiki(session_factory, wiki_id: str = "test-wiki-1", owner_id: str = "dev-user") -> None:
    """Insert a minimal WikiRecord so project wiki listing works."""
    async with session_factory() as session:
        async with session.begin():
            record = WikiRecord(
                id=wiki_id,
                owner_id=owner_id,
                repo_url="https://github.com/test/repo",
                branch="main",
                title="Test Wiki",
                page_count=1,
                status="complete",
            )
            session.add(record)


# ---------------------------------------------------------------------------
# Create
# ---------------------------------------------------------------------------


class TestCreateProject:
    @pytest.mark.asyncio
    async def test_create_project_returns_201(self, client):
        c, _app, _sf = client
        resp = await c.post(
            "/api/v1/projects",
            json={"name": "My Project", "description": "A test project", "visibility": "personal"},
        )
        assert resp.status_code == 201, resp.text
        data = resp.json()
        assert data["name"] == "My Project"
        assert data["description"] == "A test project"
        assert data["visibility"] == "personal"
        assert data["owner_id"] == "dev-user"
        assert "id" in data
        assert "created_at" in data

    @pytest.mark.asyncio
    async def test_create_project_appears_in_list(self, client):
        c, _app, _sf = client
        await c.post("/api/v1/projects", json={"name": "Listed Project"})
        resp = await c.get("/api/v1/projects")
        assert resp.status_code == 200
        names = [p["name"] for p in resp.json()["projects"]]
        assert "Listed Project" in names

    @pytest.mark.asyncio
    async def test_create_project_missing_name_returns_422(self, client):
        c, _app, _sf = client
        resp = await c.post("/api/v1/projects", json={})
        assert resp.status_code == 422

    @pytest.mark.asyncio
    async def test_create_shared_project(self, client):
        c, _app, _sf = client
        resp = await c.post(
            "/api/v1/projects",
            json={"name": "Shared Project", "visibility": "shared"},
        )
        assert resp.status_code == 201
        assert resp.json()["visibility"] == "shared"


# ---------------------------------------------------------------------------
# Get
# ---------------------------------------------------------------------------


class TestGetProject:
    @pytest.mark.asyncio
    async def test_get_project_owner_gets_200(self, client):
        c, _app, _sf = client
        create_resp = await c.post("/api/v1/projects", json={"name": "Fetch Me"})
        project_id = create_resp.json()["id"]

        resp = await c.get(f"/api/v1/projects/{project_id}")
        assert resp.status_code == 200
        assert resp.json()["id"] == project_id

    @pytest.mark.asyncio
    async def test_get_nonexistent_project_returns_404(self, client):
        c, _app, _sf = client
        resp = await c.get("/api/v1/projects/does-not-exist")
        assert resp.status_code == 404

    @pytest.mark.asyncio
    async def test_get_personal_project_other_user_returns_404(self, client):
        """A personal project is not accessible to other users — appears as 404."""
        c, app, _sf = client
        # Create project as dev-user (the default auth mock)
        create_resp = await c.post(
            "/api/v1/projects",
            json={"name": "Private Project", "visibility": "personal"},
        )
        project_id = create_resp.json()["id"]

        # Override FastAPI's get_current_user dependency to simulate a different user
        from app.auth import CurrentUser, get_current_user

        async def other_user():
            return CurrentUser(id="other-user", email="other@test.com", name="Other")

        app.dependency_overrides[get_current_user] = other_user
        try:
            resp = await c.get(f"/api/v1/projects/{project_id}")
            # Should be 404 because personal project is invisible to other user
            assert resp.status_code == 404
        finally:
            app.dependency_overrides.pop(get_current_user, None)

    @pytest.mark.asyncio
    async def test_get_shared_project_accessible_to_all(self, client, monkeypatch):
        """A shared project is readable by any authenticated user, not just the owner."""
        c, app, _sf = client
        # Create the project as dev-user (the default mock identity)
        create_resp = await c.post(
            "/api/v1/projects",
            json={"name": "Shared", "visibility": "shared"},
        )
        assert create_resp.status_code == 201
        project_id = create_resp.json()["id"]

        # Switch to a completely different user and verify they can still GET the project
        from app.auth import CurrentUser, get_current_user

        async def other_user():
            return CurrentUser(id="other-user", email="other@test.com", name="Other")

        app.dependency_overrides[get_current_user] = other_user
        try:
            resp = await c.get(f"/api/v1/projects/{project_id}")
            assert resp.status_code == 200, (
                f"other-user should be able to access a shared project, got {resp.status_code}: {resp.text}"
            )
            assert resp.json()["id"] == project_id
        finally:
            app.dependency_overrides.pop(get_current_user, None)


# ---------------------------------------------------------------------------
# Update
# ---------------------------------------------------------------------------


class TestUpdateProject:
    @pytest.mark.asyncio
    async def test_update_project_name(self, client):
        c, _app, _sf = client
        create_resp = await c.post("/api/v1/projects", json={"name": "Old Name"})
        project_id = create_resp.json()["id"]

        resp = await c.patch(f"/api/v1/projects/{project_id}", json={"name": "New Name"})
        assert resp.status_code == 200
        assert resp.json()["name"] == "New Name"

    @pytest.mark.asyncio
    async def test_update_nonexistent_project_returns_404(self, client):
        c, _app, _sf = client
        resp = await c.patch("/api/v1/projects/ghost-id", json={"name": "X"})
        assert resp.status_code == 404


# ---------------------------------------------------------------------------
# Delete
# ---------------------------------------------------------------------------


class TestDeleteProject:
    @pytest.mark.asyncio
    async def test_delete_project_returns_200(self, client):
        c, _app, _sf = client
        create_resp = await c.post("/api/v1/projects", json={"name": "To Delete"})
        project_id = create_resp.json()["id"]

        resp = await c.delete(f"/api/v1/projects/{project_id}")
        assert resp.status_code == 200
        assert resp.json()["deleted"] is True

    @pytest.mark.asyncio
    async def test_delete_project_removes_from_list(self, client):
        c, _app, _sf = client
        create_resp = await c.post("/api/v1/projects", json={"name": "Ephemeral"})
        project_id = create_resp.json()["id"]

        await c.delete(f"/api/v1/projects/{project_id}")

        resp = await c.get("/api/v1/projects")
        ids = [p["id"] for p in resp.json()["projects"]]
        assert project_id not in ids

    @pytest.mark.asyncio
    async def test_delete_nonexistent_project_returns_404(self, client):
        c, _app, _sf = client
        resp = await c.delete("/api/v1/projects/ghost-id")
        assert resp.status_code == 404

    @pytest.mark.asyncio
    async def test_delete_project_does_not_delete_wikis(self, client):
        """Deleting a project must not delete the underlying WikiRecord rows."""
        c, _app, session_factory = client
        wiki_id = "wiki-to-preserve"
        await _seed_wiki(session_factory, wiki_id=wiki_id)

        create_resp = await c.post("/api/v1/projects", json={"name": "With Wiki"})
        project_id = create_resp.json()["id"]

        await c.post(f"/api/v1/projects/{project_id}/wikis", json={"wiki_id": wiki_id})
        await c.delete(f"/api/v1/projects/{project_id}")

        # The wiki itself must still exist in the DB
        async with session_factory() as session:
            from sqlalchemy import select
            result = await session.execute(
                select(WikiRecord).where(WikiRecord.id == wiki_id)
            )
            wiki = result.scalar_one_or_none()
        assert wiki is not None, "Wiki should not be deleted when project is deleted"


# ---------------------------------------------------------------------------
# Wiki membership
# ---------------------------------------------------------------------------


class TestProjectWikis:
    @pytest.mark.asyncio
    async def test_add_wiki_to_project_returns_201(self, client):
        c, _app, session_factory = client
        wiki_id = "wiki-add-1"
        await _seed_wiki(session_factory, wiki_id=wiki_id)

        create_resp = await c.post("/api/v1/projects", json={"name": "Wiki Project"})
        project_id = create_resp.json()["id"]

        resp = await c.post(
            f"/api/v1/projects/{project_id}/wikis",
            json={"wiki_id": wiki_id},
        )
        assert resp.status_code == 201
        assert resp.json()["wiki_count"] == 1

    @pytest.mark.asyncio
    async def test_add_same_wiki_twice_is_idempotent(self, client):
        """Adding the same wiki twice must not raise an error or create duplicates."""
        c, _app, session_factory = client
        wiki_id = "wiki-idempotent"
        await _seed_wiki(session_factory, wiki_id=wiki_id)

        create_resp = await c.post("/api/v1/projects", json={"name": "Idempotent Project"})
        project_id = create_resp.json()["id"]

        await c.post(f"/api/v1/projects/{project_id}/wikis", json={"wiki_id": wiki_id})
        resp = await c.post(f"/api/v1/projects/{project_id}/wikis", json={"wiki_id": wiki_id})
        assert resp.status_code == 201
        assert resp.json()["wiki_count"] == 1  # still only one

    @pytest.mark.asyncio
    async def test_remove_wiki_from_project_returns_200(self, client):
        c, _app, session_factory = client
        wiki_id = "wiki-remove-1"
        await _seed_wiki(session_factory, wiki_id=wiki_id)

        create_resp = await c.post("/api/v1/projects", json={"name": "Remove Wiki"})
        project_id = create_resp.json()["id"]

        await c.post(f"/api/v1/projects/{project_id}/wikis", json={"wiki_id": wiki_id})

        resp = await c.delete(f"/api/v1/projects/{project_id}/wikis/{wiki_id}")
        assert resp.status_code == 200
        assert resp.json()["removed"] is True

    @pytest.mark.asyncio
    async def test_list_project_wikis(self, client):
        c, _app, session_factory = client
        wiki_id = "wiki-list-1"
        await _seed_wiki(session_factory, wiki_id=wiki_id)

        create_resp = await c.post("/api/v1/projects", json={"name": "List Wikis"})
        project_id = create_resp.json()["id"]

        await c.post(f"/api/v1/projects/{project_id}/wikis", json={"wiki_id": wiki_id})

        resp = await c.get(f"/api/v1/projects/{project_id}/wikis")
        assert resp.status_code == 200
        wiki_ids = [w["wiki_id"] for w in resp.json()["wikis"]]
        assert wiki_id in wiki_ids

    @pytest.mark.asyncio
    async def test_list_wikis_of_nonexistent_project_returns_404(self, client):
        c, _app, _sf = client
        resp = await c.get("/api/v1/projects/does-not-exist/wikis")
        assert resp.status_code == 404

    @pytest.mark.asyncio
    async def test_add_wiki_to_nonexistent_project_returns_404(self, client):
        c, _app, _sf = client
        resp = await c.post(
            "/api/v1/projects/ghost-project/wikis",
            json={"wiki_id": "any-wiki"},
        )
        assert resp.status_code == 404


# ---------------------------------------------------------------------------
# 403 non-owner
# ---------------------------------------------------------------------------


class TestProjectOwnershipEnforcement:
    """Verify that non-owners receive 403 when trying to modify projects."""

    async def _create_project_as_other_user(self, session_factory, project_name: str = "Other's Project") -> str:
        """Directly insert a project owned by 'other-user' into the DB."""
        from app.models.db_models import ProjectRecord
        import uuid as _uuid

        project_id = str(_uuid.uuid4())
        async with session_factory() as session:
            async with session.begin():
                record = ProjectRecord(
                    id=project_id,
                    owner_id="other-user",
                    name=project_name,
                    visibility="shared",  # shared so dev-user can see it
                )
                session.add(record)
        return project_id

    @pytest.mark.asyncio
    async def test_non_owner_cannot_update_project(self, client):
        c, _app, session_factory = client
        project_id = await self._create_project_as_other_user(session_factory)

        resp = await c.patch(f"/api/v1/projects/{project_id}", json={"name": "Hijacked"})
        assert resp.status_code == 403

    @pytest.mark.asyncio
    async def test_non_owner_cannot_delete_project(self, client):
        c, _app, session_factory = client
        project_id = await self._create_project_as_other_user(session_factory)

        resp = await c.delete(f"/api/v1/projects/{project_id}")
        assert resp.status_code == 403

    @pytest.mark.asyncio
    async def test_non_owner_cannot_add_wiki(self, client):
        c, _app, session_factory = client
        project_id = await self._create_project_as_other_user(session_factory)

        resp = await c.post(
            f"/api/v1/projects/{project_id}/wikis",
            json={"wiki_id": "any-wiki"},
        )
        assert resp.status_code == 403

    @pytest.mark.asyncio
    async def test_non_owner_cannot_remove_wiki(self, client):
        c, _app, session_factory = client
        project_id = await self._create_project_as_other_user(session_factory)

        resp = await c.delete(f"/api/v1/projects/{project_id}/wikis/any-wiki")
        assert resp.status_code == 403
