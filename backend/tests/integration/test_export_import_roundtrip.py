"""End-to-end round-trip integration test for Wiki Export / Import.

This test wires up a real FastAPI TestClient with:
  - in-memory SQLite database
  - LocalArtifactStorage backed by a tmp_path directory
  - real ExportService and ImportService (no mocks of business logic)

Flow:
  1. Seed a complete WikiRecord in the database
  2. Upload stub wiki-page artifacts to storage
  3. Write stub cache files into a tmp cache dir and populate cache_index.json
  4. Export via GET /api/v1/wikis/{id}/export?format=wikis → capture bytes
  5. Delete the wiki (simulate a fresh instance)
  6. Import via POST /api/v1/wikis/import with the captured bundle
  7. Assert: wiki exists in DB, pages restored in storage, cache files on disk,
     cache_index.json contains the correct entry, no other entries corrupted

Run with:
    AUTH_ENABLED=false pytest tests/integration/test_export_import_roundtrip.py -v
"""

from __future__ import annotations

import hashlib
import io
import json
import zipfile
from pathlib import Path
from typing import AsyncGenerator

import pytest
from httpx import ASGITransport, AsyncClient
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

from app.main import create_app
from app.models.db_models import Base, WikiRecord
from app.services.export_service import ExportService
from app.services.import_service import ImportService
from app.services.wiki_management import WikiManagementService
from app.storage.local import LocalArtifactStorage


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

_REPO_URL = "https://github.com/roundtrip/repo"
_BRANCH = "main"
_WIKI_ID = hashlib.sha256(f"{_REPO_URL}:{_BRANCH}".encode()).hexdigest()[:16]
_OWNER_ID = "dev-user"
_CACHE_KEY = "rt_cache_key_abc"

#: storage key prefix for the two stub pages
_PAGE_KEYS = [
    f"{_WIKI_ID}/wiki_pages/getting-started.md",
    f"{_WIKI_ID}/wiki_pages/architecture.md",
]
_PAGE_CONTENTS: dict[str, bytes] = {
    _PAGE_KEYS[0]: b"# Getting Started\n\nWelcome.\n",
    _PAGE_KEYS[1]: b"# Architecture\n\nThe system uses microservices.\n",
}


async def _make_session_factory():
    engine = create_async_engine(
        "sqlite+aiosqlite://", connect_args={"check_same_thread": False}
    )
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    return engine, async_sessionmaker(engine, expire_on_commit=False)


def _build_cache_dir(tmp_path: Path, cache_key: str = _CACHE_KEY) -> Path:
    """Create a populated cache dir with a cache_index.json and stub files."""
    cache_dir = tmp_path / "cache"
    cache_dir.mkdir(parents=True, exist_ok=True)

    # cache_index.json
    cache_index = {
        "refs": {"roundtrip/repo:main": cache_key},
        cache_key: cache_key,
    }
    (cache_dir / "cache_index.json").write_text(json.dumps(cache_index))

    # Stub cache files — all except .docs.pkl (which must never be exported)
    (cache_dir / f"{cache_key}.faiss").write_bytes(b"faiss-stub")
    (cache_dir / f"{cache_key}.docstore.bin").write_bytes(b"docstore-stub")
    (cache_dir / f"{cache_key}.doc_index.json").write_text('{"docs":[]}')
    (cache_dir / f"{cache_key}.bm25.sqlite").write_bytes(b"bm25-stub")
    (cache_dir / f"{cache_key}.fts5.db").write_bytes(b"fts5-stub")
    (cache_dir / f"{cache_key}.code_graph.gz").write_bytes(b"graph-stub")
    # This MUST be excluded from the bundle
    (cache_dir / f"{cache_key}.docs.pkl").write_bytes(b"SECRET-PICKLE")

    return cache_dir


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
async def rt_client(tmp_path: Path, monkeypatch):
    """Full-stack test client for round-trip tests.

    Returns (async_client, app, session_factory, storage, cache_dir).
    """
    monkeypatch.setenv("AUTH_ENABLED", "false")
    monkeypatch.setenv("LLM_API_KEY", "test-key")

    engine, session_factory = await _make_session_factory()
    app = create_app()

    async with app.router.lifespan_context(app):
        storage = LocalArtifactStorage(str(tmp_path / "artifacts"))
        wiki_mgmt = WikiManagementService(storage, session_factory)
        cache_dir = _build_cache_dir(tmp_path)

        app.state.storage = storage
        app.state.wiki_management = wiki_mgmt
        app.state.session_factory = session_factory

        # Wire export service with the real settings-like object
        from unittest.mock import MagicMock

        settings_mock = MagicMock()
        settings_mock.storage_backend = "local"
        settings_mock.cache_dir = str(cache_dir)

        app.state.export_service = ExportService(
            storage=storage,
            wiki_management=wiki_mgmt,
            settings=settings_mock,
        )
        app.state.import_service = ImportService(
            storage=storage,
            wiki_management=wiki_mgmt,
            settings=settings_mock,
        )

        async with AsyncClient(
            transport=ASGITransport(app=app), base_url="http://test"
        ) as c:
            yield c, app, session_factory, storage, cache_dir

    await engine.dispose()


async def _seed_complete_wiki(session_factory, storage: LocalArtifactStorage) -> WikiRecord:
    """Insert WikiRecord and upload stub page artifacts."""
    async with session_factory() as session:
        async with session.begin():
            record = WikiRecord(
                id=_WIKI_ID,
                owner_id=_OWNER_ID,
                repo_url=_REPO_URL,
                branch=_BRANCH,
                title="Round-trip Wiki",
                page_count=2,
                status="complete",
            )
            session.add(record)

    for key, content in _PAGE_CONTENTS.items():
        await storage.upload("wiki_artifacts", key, content)

    return record


# ---------------------------------------------------------------------------
# Round-trip test
# ---------------------------------------------------------------------------


class TestExportImportRoundtrip:
    @pytest.mark.asyncio
    async def test_full_roundtrip(self, rt_client):
        """Export then import restores the wiki identically."""
        c, app, sf, storage, cache_dir = rt_client
        await _seed_complete_wiki(sf, storage)

        # ── Step 1: Export ────────────────────────────────────────────────
        export_resp = await c.get(
            f"/api/v1/wikis/{_WIKI_ID}/export?format=wikis"
        )
        assert export_resp.status_code == 200, (
            f"Export failed: {export_resp.status_code} {export_resp.text}"
        )
        bundle_bytes = export_resp.content
        assert len(bundle_bytes) > 0, "Export returned empty body"
        assert zipfile.is_zipfile(io.BytesIO(bundle_bytes)), (
            "Exported bundle is not a valid ZIP"
        )

        # Verify bundle structure before proceeding
        with zipfile.ZipFile(io.BytesIO(bundle_bytes)) as zf:
            names = zf.namelist()
            assert "manifest.json" in names, f"manifest.json missing from bundle; got {names}"
            manifest = json.loads(zf.read("manifest.json"))
            assert manifest["bundle_version"] == 1
            assert manifest["repo_url"] == _REPO_URL
            assert manifest["branch"] == _BRANCH
            assert "wiki_id" not in manifest, "wiki_id must not appear in manifest"

            # Pages present
            page_entries = [n for n in names if n.startswith("pages/")]
            assert len(page_entries) == 2, (
                f"Expected 2 page entries, got {len(page_entries)}: {page_entries}"
            )

            # .docs.pkl must be absent
            pkl_entries = [n for n in names if n.endswith(".pkl") or n.endswith(".docs.pkl")]
            assert not pkl_entries, f".pkl files leaked into bundle: {pkl_entries}"

            # Cache files present
            assert "cache/cache_key.txt" in names
            assert "cache/cache_index_fragment.json" in names
            cache_key_in_bundle = zf.read("cache/cache_key.txt").decode().strip()
            assert cache_key_in_bundle == _CACHE_KEY

        # ── Step 2: Delete the wiki (simulate fresh instance) ─────────────
        del_resp = await c.delete(f"/api/v1/wikis/{_WIKI_ID}")
        assert del_resp.status_code == 200, f"Delete failed: {del_resp.text}"
        assert del_resp.json()["deleted"] is True

        # Confirm deleted from DB
        from sqlalchemy import select as sa_select

        async with sf() as session:
            row = await session.execute(
                sa_select(WikiRecord).where(WikiRecord.id == _WIKI_ID)
            )
            assert row.scalar_one_or_none() is None, "Wiki should be gone after delete"

        # Confirm artifacts deleted
        remaining = await storage.list_artifacts("wiki_artifacts", prefix=_WIKI_ID)
        assert remaining == [], f"Artifacts should be gone after delete, got: {remaining}"

        # Remove cache files from disk to simulate fresh instance
        for f_path in cache_dir.glob(f"{_CACHE_KEY}.*"):
            f_path.unlink()
        (cache_dir / "cache_index.json").write_text(json.dumps({"refs": {}}))

        # ── Step 3: Import ────────────────────────────────────────────────
        import_resp = await c.post(
            "/api/v1/wikis/import",
            files={"bundle": ("export.wikiexport", bundle_bytes, "application/zip")},
        )
        assert import_resp.status_code == 201, (
            f"Import failed: {import_resp.status_code} {import_resp.text}"
        )
        imported_data = import_resp.json()
        assert imported_data["wiki_id"] == _WIKI_ID, (
            f"wiki_id mismatch: expected {_WIKI_ID}, got {imported_data['wiki_id']}"
        )
        assert imported_data["title"] == "Round-trip Wiki"

        # ── Step 4: Verify wiki exists in DB ──────────────────────────────
        async with sf() as session:
            row = await session.execute(
                sa_select(WikiRecord).where(WikiRecord.id == _WIKI_ID)
            )
            restored = row.scalar_one_or_none()

        assert restored is not None, "WikiRecord missing after import"
        assert restored.status == "complete"
        assert restored.repo_url == _REPO_URL
        assert restored.branch == _BRANCH
        assert restored.title == "Round-trip Wiki"
        assert restored.page_count == 2

        # ── Step 5: Verify pages restored in artifact storage ─────────────
        for key in _PAGE_KEYS:
            content = await storage.download("wiki_artifacts", key)
            assert content == _PAGE_CONTENTS[key], (
                f"Page content mismatch for {key}"
            )

        # ── Step 6: Verify ALL expected cache files are present on disk ─────
        expected_cache_files = [
            (f"{_CACHE_KEY}.faiss", b"faiss-stub"),
            (f"{_CACHE_KEY}.docstore.bin", b"docstore-stub"),
            (f"{_CACHE_KEY}.doc_index.json", b'{"docs":[]}'),
            (f"{_CACHE_KEY}.bm25.sqlite", b"bm25-stub"),
            (f"{_CACHE_KEY}.fts5.db", b"fts5-stub"),
            (f"{_CACHE_KEY}.code_graph.gz", b"graph-stub"),
        ]
        for filename, expected_bytes in expected_cache_files:
            cache_file = cache_dir / filename
            assert cache_file.exists(), (
                f"{filename} not found after import — cache restore incomplete"
            )
            assert cache_file.read_bytes() == expected_bytes, (
                f"Content mismatch for {filename} after import"
            )

        # .docs.pkl must never appear on disk after import.
        # Note: the pre-import cleanup (above) deleted the original .docs.pkl from disk.
        # ExportService excludes .docs.pkl from the bundle, so no importer write can
        # recreate it.  This assert confirms the workspace is clean post-import and that
        # the exclusion in ExportService has not regressed (if it had, the file would
        # reappear here via the import's cache-file restore path).
        pkl_file = cache_dir / f"{_CACHE_KEY}.docs.pkl"
        assert not pkl_file.exists(), (
            f".docs.pkl found on disk after import — exclusion regressed: {pkl_file}"
        )

        # ── Step 7: Verify cache_index.json updated correctly ─────────────
        cache_index_path = cache_dir / "cache_index.json"
        assert cache_index_path.exists(), "cache_index.json missing after import"
        updated_index = json.loads(cache_index_path.read_text())
        assert updated_index.get("refs", {}).get("roundtrip/repo:main") == _CACHE_KEY, (
            f"cache_index.json refs missing expected entry; got: {updated_index}"
        )

    @pytest.mark.asyncio
    async def test_roundtrip_no_other_cache_entries_corrupted(self, rt_client):
        """Import must not overwrite unrelated cache_index.json entries."""
        c, app, sf, storage, cache_dir = rt_client
        await _seed_complete_wiki(sf, storage)

        # Add an unrelated wiki entry to the cache index before export
        existing_key = "unrelated_key_xyz"
        index = json.loads((cache_dir / "cache_index.json").read_text())
        index["refs"]["other/wiki:develop"] = existing_key
        index[existing_key] = existing_key
        (cache_dir / "cache_index.json").write_text(json.dumps(index))
        (cache_dir / f"{existing_key}.faiss").write_bytes(b"other-faiss")

        # Export
        export_resp = await c.get(f"/api/v1/wikis/{_WIKI_ID}/export?format=wikis")
        assert export_resp.status_code == 200
        bundle_bytes = export_resp.content

        # Delete our wiki
        await c.delete(f"/api/v1/wikis/{_WIKI_ID}")

        # Clear our wiki's cache files but leave the unrelated one
        for f_path in cache_dir.glob(f"{_CACHE_KEY}.*"):
            f_path.unlink()
        # Re-init index with only the unrelated entry intact
        (cache_dir / "cache_index.json").write_text(
            json.dumps({"refs": {"other/wiki:develop": existing_key}, existing_key: existing_key})
        )

        # Import
        import_resp = await c.post(
            "/api/v1/wikis/import",
            files={"bundle": ("export.wikiexport", bundle_bytes, "application/zip")},
        )
        assert import_resp.status_code == 201

        # The unrelated entry must still be present
        updated = json.loads((cache_dir / "cache_index.json").read_text())
        assert updated["refs"].get("other/wiki:develop") == existing_key, (
            f"Unrelated entry was corrupted: {updated}"
        )
        assert updated.get(existing_key) == existing_key, (
            f"Unrelated key mapping was removed: {updated}"
        )

        # The imported wiki's own cache entry must also be present (proves
        # _patch_cache_index actually wrote the fragment, not just preserved others)
        assert updated["refs"].get("roundtrip/repo:main") == _CACHE_KEY, (
            f"Imported wiki refs entry missing from cache_index.json: {updated}"
        )
        assert updated.get(_CACHE_KEY) == _CACHE_KEY, (
            f"Imported wiki key mapping missing from cache_index.json: {updated}"
        )

    @pytest.mark.asyncio
    async def test_export_page_content_in_bundle(self, rt_client):
        """Every exported page byte-matches what was in storage pre-export.

        This is an export-only test: it verifies the bundle ZIP contains the
        correct bytes for each page, but does not run the import leg.  Full
        round-trip page integrity is covered by test_full_roundtrip (Step 5).
        """
        c, app, sf, storage, cache_dir = rt_client
        await _seed_complete_wiki(sf, storage)

        export_resp = await c.get(f"/api/v1/wikis/{_WIKI_ID}/export?format=wikis")
        assert export_resp.status_code == 200
        bundle_bytes = export_resp.content

        # Verify page content directly from the bundle ZIP
        with zipfile.ZipFile(io.BytesIO(bundle_bytes)) as zf:
            for key, expected in _PAGE_CONTENTS.items():
                zip_path = f"pages/{key}"
                assert zip_path in zf.namelist(), (
                    f"{zip_path!r} missing from bundle; got: {zf.namelist()}"
                )
                actual = zf.read(zip_path)
                assert actual == expected, (
                    f"Content mismatch in bundle for {zip_path}"
                )

    @pytest.mark.asyncio
    async def test_roundtrip_wiki_accessible_after_import(self, rt_client):
        """After import, the wiki is accessible via GET /api/v1/wikis."""
        c, app, sf, storage, cache_dir = rt_client
        await _seed_complete_wiki(sf, storage)

        export_resp = await c.get(f"/api/v1/wikis/{_WIKI_ID}/export?format=wikis")
        bundle_bytes = export_resp.content

        await c.delete(f"/api/v1/wikis/{_WIKI_ID}")

        # Clear cache between delete and import
        for f_path in cache_dir.glob(f"{_CACHE_KEY}.*"):
            f_path.unlink()
        (cache_dir / "cache_index.json").write_text(json.dumps({"refs": {}}))

        await c.post(
            "/api/v1/wikis/import",
            files={"bundle": ("bundle.wikiexport", bundle_bytes, "application/zip")},
        )

        list_resp = await c.get("/api/v1/wikis")
        assert list_resp.status_code == 200
        wikis = list_resp.json()["wikis"]
        ids = [w["wiki_id"] for w in wikis]
        assert _WIKI_ID in ids, (
            f"Imported wiki {_WIKI_ID!r} not in wiki list: {ids}"
        )

        # Status must be complete
        our_wiki = next(w for w in wikis if w["wiki_id"] == _WIKI_ID)
        assert our_wiki["status"] == "complete"
