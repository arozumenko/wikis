"""Unit tests for ExportService.build_wikis_bundle — Wikis-to-Wikis bundle builder."""

from __future__ import annotations

import io
import json
import os
import tempfile
import zipfile
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest

from app.models.db_models import WikiRecord
from app.services.export_service import ExportService, WikiNotFoundError


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_wiki_record(
    wiki_id: str = "abc123",
    repo_url: str = "https://github.com/org/repo",
    branch: str = "main",
    title: str = "My Wiki",
    commit_hash: str = "deadbeef",
    page_count: int = 3,
) -> WikiRecord:
    record = WikiRecord()
    record.id = wiki_id
    record.repo_url = repo_url
    record.branch = branch
    record.title = title
    record.commit_hash = commit_hash
    record.page_count = page_count
    record.status = "complete"
    record.owner_id = "user1"
    return record


def _make_settings(storage_backend: str = "local", cache_dir: str = "/tmp/cache") -> MagicMock:
    settings = MagicMock()
    settings.storage_backend = storage_backend
    settings.cache_dir = cache_dir
    return settings


async def _collect_bytes(gen) -> bytes:
    """Drain an async generator into bytes."""
    chunks: list[bytes] = []
    async for chunk in gen:
        chunks.append(chunk)
    return b"".join(chunks)


def _open_zip(data: bytes) -> zipfile.ZipFile:
    return zipfile.ZipFile(io.BytesIO(data))


ARTIFACT_KEYS = [
    "abc123/wiki_pages/overview.md",
    "abc123/wiki_pages/architecture.md",
    "abc123/wiki_pages/subdir/details.md",
]

ARTIFACT_CONTENTS = {
    "abc123/wiki_pages/overview.md": b"# Overview\n\nContent.",
    "abc123/wiki_pages/architecture.md": b"# Architecture\n\nContent.",
    "abc123/wiki_pages/subdir/details.md": b"# Details\n\nContent.",
}


async def _fake_download(bucket: str, name: str) -> bytes:
    return ARTIFACT_CONTENTS.get(name, b"")


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def mock_storage():
    storage = AsyncMock()
    storage.list_artifacts = AsyncMock(return_value=ARTIFACT_KEYS)
    storage.download = AsyncMock(side_effect=_fake_download)
    return storage


@pytest.fixture
def mock_wiki_management():
    mgmt = AsyncMock()
    mgmt.get_wiki_record = AsyncMock(return_value=_make_wiki_record())
    return mgmt


@pytest.fixture
def local_settings(tmp_path):
    """Settings pointing at a temp cache dir with a populated cache_index.json."""
    cache_dir = tmp_path / "cache"
    cache_dir.mkdir()

    # Write cache_index.json
    cache_index = {
        "refs": {"org/repo:main": "org/repo:main"},
        "org/repo:main": "abc_key_42",
    }
    (cache_dir / "cache_index.json").write_text(json.dumps(cache_index))

    # Write dummy cache files
    (cache_dir / "abc_key_42.faiss").write_bytes(b"faiss-data")
    (cache_dir / "abc_key_42.docstore.bin").write_bytes(b"docstore-data")
    (cache_dir / "abc_key_42.doc_index.json").write_text('{"docs": []}')
    (cache_dir / "abc_key_42.bm25.sqlite").write_bytes(b"bm25-data")
    (cache_dir / "abc_key_42.fts5.db").write_bytes(b"fts5-data")
    (cache_dir / "abc_key_42.code_graph.gz").write_bytes(b"graph-data")
    # This .docs.pkl MUST be excluded
    (cache_dir / "abc_key_42.docs.pkl").write_bytes(b"SECRET-PICKLE")

    settings = MagicMock()
    settings.storage_backend = "local"
    settings.cache_dir = str(cache_dir)
    return settings


@pytest.fixture
def service(mock_storage, mock_wiki_management, local_settings):
    return ExportService(
        storage=mock_storage,
        wiki_management=mock_wiki_management,
        settings=local_settings,
    )


# ---------------------------------------------------------------------------
# Tests: basic structure
# ---------------------------------------------------------------------------


class TestBuildWikisBundle:
    @pytest.mark.asyncio
    async def test_produces_valid_zip(self, service):
        """Output bytes form a valid ZIP archive."""
        data = await _collect_bytes(service.build_wikis_bundle("abc123"))
        assert zipfile.is_zipfile(io.BytesIO(data))

    @pytest.mark.asyncio
    async def test_manifest_json_present(self, service):
        """manifest.json must be present at the root of the archive."""
        data = await _collect_bytes(service.build_wikis_bundle("abc123"))
        with _open_zip(data) as zf:
            assert "manifest.json" in zf.namelist()

    @pytest.mark.asyncio
    async def test_manifest_has_required_fields(self, service):
        """manifest.json must contain all required fields."""
        data = await _collect_bytes(service.build_wikis_bundle("abc123"))
        with _open_zip(data) as zf:
            manifest = json.loads(zf.read("manifest.json"))

        assert manifest["bundle_version"] == 1
        assert manifest["title"] == "My Wiki"
        assert manifest["repo_url"] == "https://github.com/org/repo"
        assert manifest["branch"] == "main"
        assert manifest["commit_hash"] == "deadbeef"
        assert "exported_at" in manifest
        assert manifest["page_count"] == 3

    @pytest.mark.asyncio
    async def test_manifest_has_no_wiki_id(self, service):
        """wiki_id must NOT appear in manifest.json — it is derived on import."""
        data = await _collect_bytes(service.build_wikis_bundle("abc123"))
        with _open_zip(data) as zf:
            manifest = json.loads(zf.read("manifest.json"))

        assert "wiki_id" not in manifest, (
            f"manifest.json must not contain wiki_id; got: {list(manifest.keys())}"
        )

    @pytest.mark.asyncio
    async def test_pages_directory_present(self, service):
        """Archive must contain entries under pages/."""
        data = await _collect_bytes(service.build_wikis_bundle("abc123"))
        with _open_zip(data) as zf:
            names = zf.namelist()
        page_entries = [n for n in names if n.startswith("pages/")]
        assert page_entries, f"No pages/ entries found; got: {names}"

    @pytest.mark.asyncio
    async def test_pages_preserve_sub_paths(self, service):
        """pages/ entries must preserve the full sub-path from storage."""
        data = await _collect_bytes(service.build_wikis_bundle("abc123"))
        with _open_zip(data) as zf:
            names = zf.namelist()

        for artifact_key in ARTIFACT_KEYS:
            expected = f"pages/{artifact_key}"
            assert expected in names, f"Missing {expected!r} in archive; got: {names}"

    @pytest.mark.asyncio
    async def test_pages_content_matches_artifacts(self, service):
        """Page content in the archive must match what storage returns."""
        data = await _collect_bytes(service.build_wikis_bundle("abc123"))
        with _open_zip(data) as zf:
            for artifact_key, expected_bytes in ARTIFACT_CONTENTS.items():
                zip_path = f"pages/{artifact_key}"
                actual = zf.read(zip_path)
                assert actual == expected_bytes, (
                    f"{zip_path}: content mismatch; got {actual!r}"
                )

    # ---------------------------------------------------------------------------
    # Tests: cache/ directory
    # ---------------------------------------------------------------------------

    @pytest.mark.asyncio
    async def test_cache_key_txt_present(self, service):
        """cache/cache_key.txt must be present."""
        data = await _collect_bytes(service.build_wikis_bundle("abc123"))
        with _open_zip(data) as zf:
            assert "cache/cache_key.txt" in zf.namelist()

    @pytest.mark.asyncio
    async def test_cache_key_txt_contains_key(self, service):
        """cache/cache_key.txt must contain the resolved cache key."""
        data = await _collect_bytes(service.build_wikis_bundle("abc123"))
        with _open_zip(data) as zf:
            key_text = zf.read("cache/cache_key.txt").decode()
        assert key_text == "abc_key_42"

    @pytest.mark.asyncio
    async def test_cache_index_fragment_present(self, service):
        """cache/cache_index_fragment.json must be present."""
        data = await _collect_bytes(service.build_wikis_bundle("abc123"))
        with _open_zip(data) as zf:
            assert "cache/cache_index_fragment.json" in zf.namelist()

    @pytest.mark.asyncio
    async def test_cache_index_fragment_contains_only_this_wiki(self, service, local_settings):
        """cache_index_fragment.json must contain only this wiki's entry."""
        # Add another wiki to the index so we can verify isolation
        cache_dir = Path(local_settings.cache_dir)
        full_index = {
            "refs": {
                "org/repo:main": "org/repo:main",
                "other/repo:develop": "other/repo:develop",
            },
            "org/repo:main": "abc_key_42",
            "other/repo:develop": "xyz_key_99",
        }
        (cache_dir / "cache_index.json").write_text(json.dumps(full_index))

        data = await _collect_bytes(service.build_wikis_bundle("abc123"))
        with _open_zip(data) as zf:
            fragment = json.loads(zf.read("cache/cache_index_fragment.json"))

        # Fragment must not contain other wiki's key mapping
        assert "xyz_key_99" not in json.dumps(fragment)
        assert "other/repo:develop" not in json.dumps(fragment)

    @pytest.mark.asyncio
    async def test_cache_files_included(self, service):
        """Cache files (.faiss, .docstore.bin, etc.) must be present under cache/."""
        data = await _collect_bytes(service.build_wikis_bundle("abc123"))
        with _open_zip(data) as zf:
            names = zf.namelist()

        expected_cache_files = [
            "cache/abc_key_42.faiss",
            "cache/abc_key_42.docstore.bin",
            "cache/abc_key_42.doc_index.json",
            "cache/abc_key_42.bm25.sqlite",
            "cache/abc_key_42.fts5.db",
            "cache/abc_key_42.code_graph.gz",
        ]
        for expected in expected_cache_files:
            assert expected in names, f"Missing {expected!r} in archive; got: {names}"

    @pytest.mark.asyncio
    async def test_docs_pkl_never_included(self, service, local_settings):
        """*.docs.pkl files must NEVER be included — security decision."""
        data = await _collect_bytes(service.build_wikis_bundle("abc123"))
        with _open_zip(data) as zf:
            names = zf.namelist()

        pkl_entries = [n for n in names if n.endswith(".pkl") or n.endswith(".docs.pkl")]
        assert not pkl_entries, (
            f".pkl files found in archive (must be excluded): {pkl_entries}"
        )

    # ---------------------------------------------------------------------------
    # Tests: error cases
    # ---------------------------------------------------------------------------

    @pytest.mark.asyncio
    async def test_raises_wiki_not_found_error(self, mock_storage, local_settings):
        """WikiNotFoundError raised when wiki_id does not exist."""
        mgmt = AsyncMock()
        mgmt.get_wiki_record = AsyncMock(return_value=None)
        svc = ExportService(storage=mock_storage, wiki_management=mgmt, settings=local_settings)

        with pytest.raises(WikiNotFoundError):
            async for _ in svc.build_wikis_bundle("nonexistent"):
                pass

    @pytest.mark.asyncio
    async def test_raises_not_implemented_for_s3_backend(
        self, mock_storage, mock_wiki_management
    ):
        """NotImplementedError raised when storage_backend is not 'local'."""
        s3_settings = _make_settings(storage_backend="s3")
        svc = ExportService(
            storage=mock_storage,
            wiki_management=mock_wiki_management,
            settings=s3_settings,
        )

        with pytest.raises(NotImplementedError, match="S3 storage backend"):
            async for _ in svc.build_wikis_bundle("abc123"):
                pass

    # ---------------------------------------------------------------------------
    # Tests: no settings (graceful degradation)
    # ---------------------------------------------------------------------------

    @pytest.mark.asyncio
    async def test_no_settings_still_produces_valid_zip(
        self, mock_storage, mock_wiki_management
    ):
        """When settings=None the service still produces a valid ZIP (no cache files)."""
        svc = ExportService(
            storage=mock_storage,
            wiki_management=mock_wiki_management,
            settings=None,
        )
        data = await _collect_bytes(svc.build_wikis_bundle("abc123"))
        assert zipfile.is_zipfile(io.BytesIO(data))
        with _open_zip(data) as zf:
            assert "manifest.json" in zf.namelist()

    @pytest.mark.asyncio
    async def test_no_settings_no_cache_files_in_bundle(
        self, mock_storage, mock_wiki_management
    ):
        """When settings=None, cache/ contains no binary files (only placeholders)."""
        svc = ExportService(
            storage=mock_storage,
            wiki_management=mock_wiki_management,
            settings=None,
        )
        data = await _collect_bytes(svc.build_wikis_bundle("abc123"))
        with _open_zip(data) as zf:
            names = zf.namelist()

        # Only placeholder files should be in cache/
        cache_files = [n for n in names if n.startswith("cache/")]
        non_placeholder = [
            n for n in cache_files
            if n not in ("cache/cache_key.txt", "cache/cache_index_fragment.json")
        ]
        assert not non_placeholder, (
            f"Unexpected cache files when settings=None: {non_placeholder}"
        )

    # ---------------------------------------------------------------------------
    # Tests: backward compat — existing ExportService(storage, mgmt) still works
    # ---------------------------------------------------------------------------

    @pytest.mark.asyncio
    async def test_obsidian_zip_unaffected_by_settings_param(
        self, mock_storage, mock_wiki_management
    ):
        """ExportService with two-arg constructor still produces Obsidian ZIP."""
        svc = ExportService(storage=mock_storage, wiki_management=mock_wiki_management)
        data = await _collect_bytes(svc.build_obsidian_zip("abc123"))
        assert zipfile.is_zipfile(io.BytesIO(data))
