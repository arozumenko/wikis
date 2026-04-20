"""Unit tests for ImportService — restore wiki from .wikiexport bundle."""

from __future__ import annotations

import hashlib
import io
import json
import zipfile
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from app.models.db_models import WikiRecord
from app.services.import_service import (
    BundleValidationError,
    BundleVersionError,
    ImportService,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_REPO_URL = "https://github.com/org/repo"
_BRANCH = "main"
_TITLE = "Test Wiki"
_OWNER_ID = "user-abc"
_EXPECTED_WIKI_ID = hashlib.sha256(f"{_REPO_URL}:{_BRANCH}".encode()).hexdigest()[:16]


def _make_valid_manifest(**overrides) -> dict:
    base = {
        "bundle_version": 1,
        "title": _TITLE,
        "repo_url": _REPO_URL,
        "branch": _BRANCH,
        "commit_hash": "deadbeef",
        "exported_at": "2026-01-01T00:00:00Z",
        "page_count": 2,
    }
    base.update(overrides)
    return base


def _make_bundle(
    manifest: dict | None = None,
    pages: dict[str, bytes] | None = None,
    cache_files: dict[str, bytes] | None = None,
    cache_key: str = "testcachekey",
    cache_index_fragment: dict | None = None,
    include_manifest: bool = True,
    include_pkl: bool = False,
) -> bytes:
    """Build a minimal .wikiexport ZIP in memory."""
    if manifest is None:
        manifest = _make_valid_manifest()
    if pages is None:
        pages = {
            f"pages/{_EXPECTED_WIKI_ID}/wiki_pages/overview.md": b"# Overview\n",
            f"pages/{_EXPECTED_WIKI_ID}/wiki_pages/arch.md": b"# Arch\n",
        }
    if cache_files is None:
        cache_files = {
            f"cache/{cache_key}.faiss": b"\x00fakefaiss",
            f"cache/{cache_key}.docstore.bin": b"\x00fakebm25",
        }
    if cache_index_fragment is None:
        cache_index_fragment = {
            "refs": {"org/repo:main": cache_key},
            cache_key: cache_key,
        }

    buf = io.BytesIO()
    with zipfile.ZipFile(buf, mode="w", compression=zipfile.ZIP_STORED) as zf:
        if include_manifest:
            zf.writestr("manifest.json", json.dumps(manifest))
        for name, data in pages.items():
            zf.writestr(name, data)
        for name, data in cache_files.items():
            zf.writestr(name, data)
        zf.writestr("cache/cache_key.txt", cache_key)
        zf.writestr(
            "cache/cache_index_fragment.json",
            json.dumps(cache_index_fragment),
        )
        if include_pkl:
            zf.writestr(f"cache/{cache_key}.docs.pkl", b"picklecontent")

    buf.seek(0)
    return buf.read()


def _make_wiki_record(wiki_id: str = _EXPECTED_WIKI_ID) -> WikiRecord:
    record = WikiRecord()
    record.id = wiki_id
    record.owner_id = _OWNER_ID
    record.repo_url = _REPO_URL
    record.branch = _BRANCH
    record.title = _TITLE
    record.page_count = 2
    record.commit_hash = "deadbeef"
    record.status = "complete"
    return record


def _make_settings(storage_backend: str = "local", cache_dir: str = "/tmp/test_cache") -> MagicMock:
    settings = MagicMock()
    settings.storage_backend = storage_backend
    settings.cache_dir = cache_dir
    return settings


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def mock_storage():
    storage = AsyncMock()
    storage.upload = AsyncMock(return_value="key")
    storage.list_artifacts = AsyncMock(return_value=[])
    storage.delete = AsyncMock()
    return storage


@pytest.fixture
def mock_wiki_management():
    mgmt = AsyncMock()
    mgmt.get_wiki_record = AsyncMock(return_value=None)  # no collision by default
    mgmt.register_wiki = AsyncMock()
    mgmt.delete_wiki = AsyncMock()
    return mgmt


@pytest.fixture
def mock_settings(tmp_path):
    return _make_settings(cache_dir=str(tmp_path / "cache"))


@pytest.fixture
def service(mock_storage, mock_wiki_management, mock_settings):
    return ImportService(
        storage=mock_storage,
        wiki_management=mock_wiki_management,
        settings=mock_settings,
    )


@pytest.fixture
def service_with_record(mock_storage, mock_wiki_management, mock_settings):
    """Service where get_wiki_record returns a record first call (collision), then returns post-upsert record."""
    record = _make_wiki_record()
    # First call returns existing record (collision check), second returns None (post-delete check is not called),
    # final call returns the record after register_wiki
    mock_wiki_management.get_wiki_record = AsyncMock(
        side_effect=[record, record]
    )
    return ImportService(
        storage=mock_storage,
        wiki_management=mock_wiki_management,
        settings=mock_settings,
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestRestoreWiki:
    @pytest.mark.asyncio
    async def test_valid_bundle_returns_wiki_record(self, service, mock_wiki_management):
        """A valid bundle should return a WikiRecord with correct fields."""
        record = _make_wiki_record()
        mock_wiki_management.get_wiki_record = AsyncMock(
            side_effect=[None, record]  # first=no collision, second=post-upsert fetch
        )

        bundle = _make_bundle()
        result = await service.restore_wiki(bundle, _OWNER_ID)

        assert isinstance(result, WikiRecord)
        assert result.id == _EXPECTED_WIKI_ID
        assert result.repo_url == _REPO_URL
        assert result.branch == _BRANCH
        assert result.title == _TITLE

    @pytest.mark.asyncio
    async def test_wiki_id_derived_from_repo_url_and_branch(self, service, mock_wiki_management):
        """wiki_id must be derived from repo_url+branch, never read from bundle."""
        record = _make_wiki_record()
        mock_wiki_management.get_wiki_record = AsyncMock(side_effect=[None, record])

        bundle = _make_bundle()
        await service.restore_wiki(bundle, _OWNER_ID)

        # register_wiki must be called with the derived wiki_id
        call_kwargs = mock_wiki_management.register_wiki.call_args
        assert call_kwargs.kwargs["wiki_id"] == _EXPECTED_WIKI_ID

    @pytest.mark.asyncio
    async def test_owner_id_always_from_parameter(self, service, mock_wiki_management):
        """owner_id must come from the function parameter, never from the bundle."""
        bundle_owner = "bundle-user-should-be-ignored"
        manifest = _make_valid_manifest()
        # Inject an owner into the manifest (should be ignored)
        manifest["owner_id"] = bundle_owner

        record = _make_wiki_record()
        mock_wiki_management.get_wiki_record = AsyncMock(side_effect=[None, record])

        bundle = _make_bundle(manifest=manifest)
        await service.restore_wiki(bundle, _OWNER_ID)

        call_kwargs = mock_wiki_management.register_wiki.call_args
        assert call_kwargs.kwargs["owner_id"] == _OWNER_ID

    @pytest.mark.asyncio
    async def test_docs_pkl_raises_validation_error(self, service):
        """.docs.pkl files in the bundle must raise BundleValidationError."""
        bundle = _make_bundle(include_pkl=True)
        with pytest.raises(BundleValidationError, match=r"\.docs\.pkl"):
            await service.restore_wiki(bundle, _OWNER_ID)

    @pytest.mark.asyncio
    async def test_missing_manifest_raises_validation_error(self, service):
        """Bundle without manifest.json must raise BundleValidationError."""
        bundle = _make_bundle(include_manifest=False)
        with pytest.raises(BundleValidationError, match="manifest.json"):
            await service.restore_wiki(bundle, _OWNER_ID)

    @pytest.mark.asyncio
    async def test_wrong_bundle_version_raises_version_error(self, service):
        """bundle_version != 1 must raise BundleVersionError."""
        manifest = _make_valid_manifest(bundle_version=99)
        bundle = _make_bundle(manifest=manifest)
        with pytest.raises(BundleVersionError, match="99"):
            await service.restore_wiki(bundle, _OWNER_ID)

    @pytest.mark.asyncio
    async def test_missing_bundle_version_raises_version_error(self, service):
        """Missing bundle_version must raise BundleVersionError."""
        manifest = _make_valid_manifest()
        del manifest["bundle_version"]
        bundle = _make_bundle(manifest=manifest)
        with pytest.raises(BundleVersionError):
            await service.restore_wiki(bundle, _OWNER_ID)

    @pytest.mark.asyncio
    async def test_non_zip_raises_validation_error(self, service):
        """Non-zip bytes must raise BundleValidationError."""
        with pytest.raises(BundleValidationError, match="not a zip"):
            await service.restore_wiki(b"this is not a zip file at all", _OWNER_ID)

    @pytest.mark.asyncio
    async def test_s3_backend_raises_not_implemented(self, mock_storage, mock_wiki_management):
        """S3 storage backend must raise NotImplementedError."""
        settings = _make_settings(storage_backend="s3")
        svc = ImportService(
            storage=mock_storage,
            wiki_management=mock_wiki_management,
            settings=settings,
        )
        bundle = _make_bundle()
        with pytest.raises(NotImplementedError, match="S3"):
            await svc.restore_wiki(bundle, _OWNER_ID)

    @pytest.mark.asyncio
    async def test_collision_triggers_delete_before_write(
        self, mock_storage, mock_wiki_management, mock_settings
    ):
        """When a wiki with the same derived ID exists, delete is called before writing."""
        existing_record = _make_wiki_record()
        post_upsert_record = _make_wiki_record()

        # get_wiki_record: first call = collision check, second = post-register fetch
        mock_wiki_management.get_wiki_record = AsyncMock(
            side_effect=[existing_record, post_upsert_record]
        )

        svc = ImportService(
            storage=mock_storage,
            wiki_management=mock_wiki_management,
            settings=mock_settings,
        )

        bundle = _make_bundle()
        await svc.restore_wiki(bundle, _OWNER_ID)

        # delete_wiki must have been called with the correct wiki_id
        mock_wiki_management.delete_wiki.assert_awaited_once()
        delete_call = mock_wiki_management.delete_wiki.call_args
        # wiki_id may be passed as positional or keyword arg
        called_wiki_id = (
            delete_call.kwargs.get("wiki_id")
            or (delete_call.args[0] if delete_call.args else None)
        )
        assert called_wiki_id == _EXPECTED_WIKI_ID

    @pytest.mark.asyncio
    async def test_no_collision_skips_delete(self, service, mock_wiki_management):
        """When no existing wiki, delete_wiki must NOT be called."""
        record = _make_wiki_record()
        mock_wiki_management.get_wiki_record = AsyncMock(side_effect=[None, record])

        bundle = _make_bundle()
        await service.restore_wiki(bundle, _OWNER_ID)

        mock_wiki_management.delete_wiki.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_pages_uploaded_to_artifact_storage(self, service, mock_wiki_management, mock_storage):
        """All pages/ entries must be uploaded to 'wiki_artifacts' bucket."""
        record = _make_wiki_record()
        mock_wiki_management.get_wiki_record = AsyncMock(side_effect=[None, record])

        pages = {
            f"pages/{_EXPECTED_WIKI_ID}/wiki_pages/overview.md": b"# Overview\n",
            f"pages/{_EXPECTED_WIKI_ID}/wiki_pages/deep/nested.md": b"# Nested\n",
        }
        bundle = _make_bundle(pages=pages)
        await service.restore_wiki(bundle, _OWNER_ID)

        # Check both pages were uploaded
        uploaded_keys = [
            call.args[1] for call in mock_storage.upload.call_args_list
        ]
        assert f"{_EXPECTED_WIKI_ID}/wiki_pages/overview.md" in uploaded_keys
        assert f"{_EXPECTED_WIKI_ID}/wiki_pages/deep/nested.md" in uploaded_keys

    @pytest.mark.asyncio
    async def test_cache_index_atomic_patch_preserves_other_keys(
        self, mock_storage, mock_wiki_management, tmp_path
    ):
        """Merging fragment into cache_index.json must preserve unrelated keys."""
        cache_dir = tmp_path / "cache"
        cache_dir.mkdir()
        settings = _make_settings(cache_dir=str(cache_dir))

        # Pre-populate cache_index.json with an unrelated entry
        existing_index = {
            "refs": {"other/repo:develop": "other_cache_key"},
            "other_cache_key": "other_cache_key",
        }
        (cache_dir / "cache_index.json").write_text(json.dumps(existing_index))

        cache_key = "new_cache_key_abc"
        fragment = {
            "refs": {"org/repo:main": cache_key},
            cache_key: cache_key,
        }

        record = _make_wiki_record()
        mock_wiki_management.get_wiki_record = AsyncMock(side_effect=[None, record])

        svc = ImportService(
            storage=mock_storage,
            wiki_management=mock_wiki_management,
            settings=settings,
        )
        bundle = _make_bundle(cache_key=cache_key, cache_index_fragment=fragment)
        await svc.restore_wiki(bundle, _OWNER_ID)

        # Read the updated index
        updated = json.loads((cache_dir / "cache_index.json").read_text())

        # New wiki's entries added
        assert updated["refs"].get("org/repo:main") == cache_key
        assert updated.get(cache_key) == cache_key

        # Unrelated entry preserved
        assert updated["refs"].get("other/repo:develop") == "other_cache_key"
        assert updated.get("other_cache_key") == "other_cache_key"

    @pytest.mark.asyncio
    async def test_cache_index_created_when_missing(
        self, mock_storage, mock_wiki_management, tmp_path
    ):
        """cache_index.json must be created from scratch when it doesn't exist."""
        cache_dir = tmp_path / "cache"
        cache_dir.mkdir()
        settings = _make_settings(cache_dir=str(cache_dir))

        cache_key = "fresh_cache_key"
        fragment = {
            "refs": {"org/repo:main": cache_key},
            cache_key: cache_key,
        }

        record = _make_wiki_record()
        mock_wiki_management.get_wiki_record = AsyncMock(side_effect=[None, record])

        svc = ImportService(
            storage=mock_storage,
            wiki_management=mock_wiki_management,
            settings=settings,
        )
        bundle = _make_bundle(cache_key=cache_key, cache_index_fragment=fragment)
        await svc.restore_wiki(bundle, _OWNER_ID)

        assert (cache_dir / "cache_index.json").exists()
        updated = json.loads((cache_dir / "cache_index.json").read_text())
        assert updated["refs"].get("org/repo:main") == cache_key

    @pytest.mark.asyncio
    async def test_missing_required_manifest_field_raises_validation_error(self, service):
        """manifest.json missing 'repo_url' must raise BundleValidationError."""
        manifest = _make_valid_manifest()
        del manifest["repo_url"]
        bundle = _make_bundle(manifest=manifest)
        with pytest.raises(BundleValidationError, match="repo_url"):
            await service.restore_wiki(bundle, _OWNER_ID)

    @pytest.mark.asyncio
    async def test_register_wiki_called_with_complete_status(self, service, mock_wiki_management):
        """register_wiki must be called with status='complete'."""
        record = _make_wiki_record()
        mock_wiki_management.get_wiki_record = AsyncMock(side_effect=[None, record])

        bundle = _make_bundle()
        await service.restore_wiki(bundle, _OWNER_ID)

        call_kwargs = mock_wiki_management.register_wiki.call_args.kwargs
        assert call_kwargs["status"] == "complete"

    @pytest.mark.asyncio
    async def test_cache_files_written_to_cache_dir(
        self, mock_storage, mock_wiki_management, tmp_path
    ):
        """Cache files from the bundle must be written to settings.cache_dir."""
        cache_dir = tmp_path / "cache"
        cache_dir.mkdir()
        settings = _make_settings(cache_dir=str(cache_dir))

        cache_key = "abc123key"
        bundle = _make_bundle(
            cache_key=cache_key,
            cache_files={
                f"cache/{cache_key}.faiss": b"faissdata",
            },
        )

        record = _make_wiki_record()
        mock_wiki_management.get_wiki_record = AsyncMock(side_effect=[None, record])

        svc = ImportService(
            storage=mock_storage,
            wiki_management=mock_wiki_management,
            settings=settings,
        )
        await svc.restore_wiki(bundle, _OWNER_ID)

        faiss_file = cache_dir / f"{cache_key}.faiss"
        assert faiss_file.exists()
        assert faiss_file.read_bytes() == b"faissdata"
