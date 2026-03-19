"""Tests for artifact storage."""

from __future__ import annotations

import pytest
from pydantic import SecretStr

from app.config import Settings
from app.storage import get_storage
from app.storage.local import LocalArtifactStorage


@pytest.fixture
def storage(tmp_path):
    return LocalArtifactStorage(str(tmp_path))


class TestLocalArtifactStorage:
    @pytest.mark.asyncio
    async def test_upload_and_download(self, storage):
        key = await storage.upload("wiki", "readme.md", b"# Hello")
        assert key == "wiki/readme.md"
        data = await storage.download("wiki", "readme.md")
        assert data == b"# Hello"

    @pytest.mark.asyncio
    async def test_download_missing_raises(self, storage):
        with pytest.raises(FileNotFoundError):
            await storage.download("wiki", "nope.txt")

    @pytest.mark.asyncio
    async def test_delete(self, storage):
        await storage.upload("wiki", "tmp.txt", b"data")
        await storage.delete("wiki", "tmp.txt")
        with pytest.raises(FileNotFoundError):
            await storage.download("wiki", "tmp.txt")

    @pytest.mark.asyncio
    async def test_delete_missing_no_error(self, storage):
        await storage.delete("wiki", "nonexistent.txt")

    @pytest.mark.asyncio
    async def test_list_artifacts(self, storage):
        await storage.upload("wiki", "a.txt", b"a")
        await storage.upload("wiki", "b.txt", b"b")
        await storage.upload("wiki", "c.md", b"c")
        result = await storage.list_artifacts("wiki")
        assert result == ["a.txt", "b.txt", "c.md"]

    @pytest.mark.asyncio
    async def test_list_artifacts_with_prefix(self, storage):
        await storage.upload("wiki", "page_1.md", b"1")
        await storage.upload("wiki", "page_2.md", b"2")
        await storage.upload("wiki", "index.md", b"i")
        result = await storage.list_artifacts("wiki", prefix="page_")
        assert result == ["page_1.md", "page_2.md"]

    @pytest.mark.asyncio
    async def test_list_empty_bucket(self, storage):
        result = await storage.list_artifacts("empty")
        assert result == []

    @pytest.mark.asyncio
    async def test_path_traversal_bucket_rejected(self, storage):
        with pytest.raises(ValueError, match="path traversal"):
            await storage.upload("../etc", "passwd", b"bad")

    @pytest.mark.asyncio
    async def test_path_traversal_name_rejected(self, storage):
        with pytest.raises(ValueError, match="path traversal"):
            await storage.upload("wiki", "../../etc/passwd", b"bad")

    @pytest.mark.asyncio
    async def test_absolute_path_rejected(self, storage):
        with pytest.raises(ValueError, match="path traversal"):
            await storage.upload("/etc", "passwd", b"bad")


class TestGetStorage:
    def test_local_backend(self):
        settings = Settings(
            llm_api_key=SecretStr("k"),
            storage_backend="local",
            storage_path="/tmp/test-storage",
        )
        s = get_storage(settings)
        assert isinstance(s, LocalArtifactStorage)

    def test_unknown_backend_raises(self):
        settings = Settings(
            llm_api_key=SecretStr("k"),
            storage_backend="gcs",
        )
        with pytest.raises(ValueError, match="Unknown storage backend"):
            get_storage(settings)
