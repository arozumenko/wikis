"""Tests for wiki management service — database-backed metadata storage."""

from __future__ import annotations

import pytest
import pytest_asyncio
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

from app.models.db_models import Base
from app.services.wiki_management import WikiManagementService
from app.storage.local import LocalArtifactStorage

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest_asyncio.fixture
async def session_factory():
    """In-memory SQLite session factory — isolated per test."""
    engine = create_async_engine("sqlite+aiosqlite://", echo=False, connect_args={"check_same_thread": False})
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    factory = async_sessionmaker(engine, expire_on_commit=False)
    yield factory
    await engine.dispose()


@pytest.fixture
def storage(tmp_path):
    return LocalArtifactStorage(str(tmp_path))


@pytest_asyncio.fixture
async def service(storage, session_factory):
    return WikiManagementService(storage, session_factory)


# ---------------------------------------------------------------------------
# register + list
# ---------------------------------------------------------------------------


class TestRegisterAndList:
    async def test_list_empty(self, service):
        result = await service.list_wikis(user_id="user-1")
        assert result.wikis == []

    async def test_register_and_list_as_owner(self, service):
        await service.register_wiki("w1", "https://github.com/a/b", "main", "Test Wiki", 5, owner_id="user-1")
        result = await service.list_wikis(user_id="user-1")
        assert len(result.wikis) == 1
        assert result.wikis[0].wiki_id == "w1"
        assert result.wikis[0].page_count == 5
        assert result.wikis[0].is_owner is True

    async def test_register_multiple(self, service):
        await service.register_wiki("w1", "https://github.com/a/b", "main", "Wiki 1", 3, owner_id="user-1")
        await service.register_wiki("w2", "https://github.com/c/d", "dev", "Wiki 2", 7, owner_id="user-1")
        result = await service.list_wikis(user_id="user-1")
        assert len(result.wikis) == 2

    async def test_owner_sees_own_personal_wikis(self, service):
        await service.register_wiki(
            "w-personal", "https://github.com/a/b", "main", "Private", 1, owner_id="user-1", visibility="personal"
        )
        result = await service.list_wikis(user_id="user-1")
        assert any(w.wiki_id == "w-personal" for w in result.wikis)

    async def test_other_user_cannot_see_personal_wikis(self, service):
        await service.register_wiki(
            "w-personal", "https://github.com/a/b", "main", "Private", 1, owner_id="user-1", visibility="personal"
        )
        result = await service.list_wikis(user_id="user-2")
        assert not any(w.wiki_id == "w-personal" for w in result.wikis)

    async def test_shared_wikis_visible_to_others(self, service):
        await service.register_wiki(
            "w-shared", "https://github.com/a/b", "main", "Shared", 1, owner_id="user-1", visibility="shared"
        )
        result = await service.list_wikis(user_id="user-2")
        assert any(w.wiki_id == "w-shared" for w in result.wikis)
        # user-2 should NOT be marked as owner
        wiki = next(w for w in result.wikis if w.wiki_id == "w-shared")
        assert wiki.is_owner is False

    async def test_none_user_sees_only_shared(self, service):
        await service.register_wiki(
            "w1", "https://github.com/a/b", "main", "Shared", 1, owner_id="user-1", visibility="shared"
        )
        await service.register_wiki(
            "w2", "https://github.com/c/d", "main", "Personal", 1, owner_id="user-1", visibility="personal"
        )
        result = await service.list_wikis(user_id=None)
        wiki_ids = {w.wiki_id for w in result.wikis}
        assert "w1" in wiki_ids
        assert "w2" not in wiki_ids

    async def test_register_with_commit_hash(self, service):
        await service.register_wiki(
            "w1", "https://github.com/a/b", "main", "Wiki", 5, owner_id="user-1", commit_hash="abc123"
        )
        result = await service.list_wikis(user_id="user-1")
        assert result.wikis[0].commit_hash == "abc123"
        assert result.wikis[0].indexed_at is not None

    async def test_upsert_updates_existing_record(self, service):
        await service.register_wiki("w1", "https://github.com/a/b", "main", "Old Title", 2, owner_id="user-1")
        await service.register_wiki("w1", "https://github.com/a/b", "main", "New Title", 10, owner_id="user-1")
        result = await service.list_wikis(user_id="user-1")
        assert len(result.wikis) == 1
        assert result.wikis[0].title == "New Title"
        assert result.wikis[0].page_count == 10


# ---------------------------------------------------------------------------
# delete
# ---------------------------------------------------------------------------


class TestDelete:
    async def test_delete_by_owner_succeeds(self, service):
        await service.register_wiki("w1", "https://github.com/a/b", "main", "Wiki", 1, owner_id="user-1")
        result = await service.delete_wiki("w1", user_id="user-1")
        assert result.deleted is True
        listed = await service.list_wikis(user_id="user-1")
        assert listed.wikis == []

    async def test_delete_by_non_owner_fails(self, service):
        await service.register_wiki("w1", "https://github.com/a/b", "main", "Wiki", 1, owner_id="user-1")
        result = await service.delete_wiki("w1", user_id="user-2")
        assert result.deleted is False
        assert result.message is not None
        assert "owner" in result.message.lower()

    async def test_delete_nonexistent_returns_false(self, service):
        result = await service.delete_wiki("nope", user_id="user-1")
        assert result.deleted is False

    async def test_delete_removes_artifacts(self, service, storage):
        await storage.upload("wiki_artifacts", "w1/page1.md", b"# Page 1")
        await storage.upload("wiki_artifacts", "w1/page2.md", b"# Page 2")
        await service.register_wiki("w1", "https://github.com/a/b", "main", "Wiki", 2, owner_id="user-1")

        result = await service.delete_wiki("w1", user_id="user-1")
        assert result.deleted is True

        remaining = await storage.list_artifacts("wiki_artifacts", prefix="w1")
        assert remaining == []

    async def test_delete_without_user_id_succeeds(self, service):
        """System/admin deletion with no user_id — bypasses ownership check."""
        await service.register_wiki("w1", "https://github.com/a/b", "main", "Wiki", 1, owner_id="user-1")
        result = await service.delete_wiki("w1", user_id=None)
        assert result.deleted is True


# ---------------------------------------------------------------------------
# update visibility
# ---------------------------------------------------------------------------


class TestUpdateVisibility:
    async def test_owner_can_change_to_shared(self, service):
        await service.register_wiki(
            "w1", "https://github.com/a/b", "main", "Wiki", 1, owner_id="user-1", visibility="personal"
        )
        updated = await service.update_visibility("w1", user_id="user-1", visibility="shared")
        assert updated is not None
        assert updated.visibility == "shared"

    async def test_non_owner_cannot_change_visibility(self, service):
        await service.register_wiki(
            "w1", "https://github.com/a/b", "main", "Wiki", 1, owner_id="user-1", visibility="personal"
        )
        result = await service.update_visibility("w1", user_id="user-2", visibility="shared")
        assert result is None

    async def test_update_nonexistent_wiki_returns_none(self, service):
        result = await service.update_visibility("does-not-exist", user_id="user-1", visibility="shared")
        assert result is None

    async def test_visibility_change_persists(self, service):
        await service.register_wiki(
            "w1", "https://github.com/a/b", "main", "Wiki", 1, owner_id="user-1", visibility="personal"
        )
        await service.update_visibility("w1", user_id="user-1", visibility="shared")
        # user-2 should now see it
        result = await service.list_wikis(user_id="user-2")
        assert any(w.wiki_id == "w1" for w in result.wikis)


# ---------------------------------------------------------------------------
# get_wiki
# ---------------------------------------------------------------------------


class TestGetWiki:
    async def test_owner_can_get_personal_wiki(self, service):
        await service.register_wiki("w1", "https://github.com/a/b", "main", "Wiki", 1, owner_id="user-1")
        record = await service.get_wiki("w1", user_id="user-1")
        assert record is not None
        assert record.id == "w1"

    async def test_non_owner_cannot_get_personal_wiki(self, service):
        await service.register_wiki(
            "w1", "https://github.com/a/b", "main", "Wiki", 1, owner_id="user-1", visibility="personal"
        )
        record = await service.get_wiki("w1", user_id="user-2")
        assert record is None

    async def test_anyone_can_get_shared_wiki(self, service):
        await service.register_wiki(
            "w1", "https://github.com/a/b", "main", "Wiki", 1, owner_id="user-1", visibility="shared"
        )
        record = await service.get_wiki("w1", user_id="user-2")
        assert record is not None

    async def test_get_nonexistent_returns_none(self, service):
        record = await service.get_wiki("nope", user_id="user-1")
        assert record is None
