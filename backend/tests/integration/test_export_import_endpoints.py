"""Integration tests for the wiki export and import endpoints.

Tests the full HTTP surface of:
  GET  /api/v1/wikis/{wiki_id}/export?format=obsidian|wikis
  POST /api/v1/wikis/import

Run with:
    AUTH_ENABLED=false pytest tests/integration/test_export_import_endpoints.py -v
"""

from __future__ import annotations

import io
import json
import zipfile
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from httpx import ASGITransport, AsyncClient
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

from app.main import create_app
from app.models.db_models import Base, WikiRecord
from app.services.export_service import ExportService
from app.services.import_service import BundleValidationError, BundleVersionError, ImportService
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


async def _seed_wiki(
    session_factory,
    wiki_id: str = "test-wiki-abc123",
    owner_id: str = "dev-user",
    status: str = "complete",
) -> WikiRecord:
    """Insert a WikiRecord so export/import tests can find it."""
    async with session_factory() as session:
        async with session.begin():
            record = WikiRecord(
                id=wiki_id,
                owner_id=owner_id,
                repo_url="https://github.com/test/repo",
                branch="main",
                title="Test Export Wiki",
                page_count=2,
                status=status,
            )
            session.add(record)
    return record


def _make_obsidian_zip_bytes() -> bytes:
    """Build a minimal in-memory ZIP that looks like an Obsidian vault."""
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, mode="w") as zf:
        zf.writestr("vault/.obsidian/app.json", '{"legacyEditor": false}')
        zf.writestr("vault/README.md", "# Test")
        zf.writestr("vault/page1.md", "# Page 1\nContent here.")
    buf.seek(0)
    return buf.read()


def _make_wikis_bundle_bytes(wiki_id: str = "test-wiki-abc123") -> bytes:
    """Build a minimal .wikiexport bundle that passes validation."""
    manifest = {
        "bundle_version": 1,
        "title": "Test Export Wiki",
        "repo_url": "https://github.com/test/repo",
        "branch": "main",
        "commit_hash": None,
        "page_count": 2,
        "exported_at": "2026-04-11T00:00:00Z",
    }
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, mode="w") as zf:
        zf.writestr("manifest.json", json.dumps(manifest))
        zf.writestr(f"pages/{wiki_id}/wiki_pages/intro.md", "# Intro\nHello.")
        zf.writestr("cache/cache_key.txt", "")
        zf.writestr("cache/cache_index_fragment.json", "{}")
    buf.seek(0)
    return buf.read()


def _async_zip_generator(zip_bytes: bytes):
    """Wrap bytes as an async generator, matching ExportService's yield pattern."""
    async def _gen():
        chunk_size = 65536
        view = memoryview(zip_bytes)
        pos = 0
        while pos < len(view):
            yield bytes(view[pos : pos + chunk_size])
            pos += chunk_size

    return _gen()


# ---------------------------------------------------------------------------
# Export endpoint tests
# ---------------------------------------------------------------------------


class TestExportEndpoint:
    @pytest.mark.asyncio
    async def test_export_obsidian_returns_200_zip(self, client):
        """GET /api/v1/wikis/{id}/export?format=obsidian → 200, Content-Type: application/zip."""
        c, app, sf = client
        wiki_id = "export-obs-001"
        await _seed_wiki(sf, wiki_id=wiki_id)

        zip_bytes = _make_obsidian_zip_bytes()
        app.state.export_service.build_obsidian_zip = MagicMock(
            return_value=_async_zip_generator(zip_bytes)
        )

        resp = await c.get(f"/api/v1/wikis/{wiki_id}/export?format=obsidian")
        assert resp.status_code == 200, resp.text
        assert resp.headers["content-type"] == "application/zip"
        assert "attachment" in resp.headers.get("content-disposition", "")
        assert len(resp.content) > 0

    @pytest.mark.asyncio
    async def test_export_wikis_format_calls_correct_builder(self, client):
        """GET /api/v1/wikis/{id}/export?format=wikis → 200 or 501 (not implemented for S3)."""
        c, app, sf = client
        wiki_id = "export-wikis-001"
        await _seed_wiki(sf, wiki_id=wiki_id)

        zip_bytes = _make_wikis_bundle_bytes(wiki_id)
        app.state.export_service.build_wikis_bundle = MagicMock(
            return_value=_async_zip_generator(zip_bytes)
        )

        resp = await c.get(f"/api/v1/wikis/{wiki_id}/export?format=wikis")
        assert resp.status_code == 200, resp.text
        assert resp.headers["content-type"] == "application/zip"

    @pytest.mark.asyncio
    async def test_export_nonexistent_wiki_returns_404(self, client):
        """GET /api/v1/wikis/nonexistent/export?format=obsidian → 404."""
        c, app, sf = client
        resp = await c.get("/api/v1/wikis/does-not-exist/export?format=obsidian")
        assert resp.status_code == 404, resp.text

    @pytest.mark.asyncio
    async def test_export_incomplete_wiki_returns_409(self, client):
        """GET /api/v1/wikis/{id}/export on a generating wiki → 409."""
        c, app, sf = client
        wiki_id = "export-incomplete-001"
        await _seed_wiki(sf, wiki_id=wiki_id, status="generating")

        resp = await c.get(f"/api/v1/wikis/{wiki_id}/export?format=obsidian")
        assert resp.status_code == 409, resp.text
        assert "fully generated" in resp.json().get("detail", "")

    @pytest.mark.asyncio
    async def test_export_invalid_format_returns_400(self, client):
        """GET /api/v1/wikis/{id}/export?format=invalid → 400."""
        c, app, sf = client
        wiki_id = "export-bad-fmt-001"
        await _seed_wiki(sf, wiki_id=wiki_id)

        resp = await c.get(f"/api/v1/wikis/{wiki_id}/export?format=invalid")
        assert resp.status_code == 400, resp.text

    @pytest.mark.asyncio
    async def test_export_not_implemented_returns_501(self, client):
        """If the export builder raises NotImplementedError before first yield → 501.

        The real ExportService raises NotImplementedError at the top of
        build_wikis_bundle (before any yield) when the storage backend is S3.
        The route handler primes the generator before returning StreamingResponse
        so this error can be caught and mapped to a 501.
        """
        c, app, sf = client
        wiki_id = "export-ni-001"
        await _seed_wiki(sf, wiki_id=wiki_id)

        async def _raise_ni():
            # Raise before yielding — mirrors the real service guard at the top
            raise NotImplementedError("S3 not supported")
            yield  # pragma: no cover — makes this an async generator

        app.state.export_service.build_obsidian_zip = MagicMock(return_value=_raise_ni())

        resp = await c.get(f"/api/v1/wikis/{wiki_id}/export?format=obsidian")
        assert resp.status_code == 501, resp.text

    @pytest.mark.asyncio
    async def test_export_failed_wiki_returns_409(self, client):
        """GET /api/v1/wikis/{id}/export on a failed wiki → 409."""
        c, app, sf = client
        wiki_id = "export-failed-001"
        await _seed_wiki(sf, wiki_id=wiki_id, status="failed")

        resp = await c.get(f"/api/v1/wikis/{wiki_id}/export?format=obsidian")
        assert resp.status_code == 409, resp.text


# ---------------------------------------------------------------------------
# Import endpoint tests
# ---------------------------------------------------------------------------


class TestImportEndpoint:
    @pytest.mark.asyncio
    async def test_import_valid_bundle_returns_201(self, client):
        """POST /api/v1/wikis/import with valid bundle → 201 with wiki JSON."""
        c, app, sf = client
        wiki_id = "import-ok-001"

        # Seed an existing record that restore_wiki will return
        async with sf() as session:
            async with session.begin():
                session.add(WikiRecord(
                    id=wiki_id,
                    owner_id="dev-user",
                    repo_url="https://github.com/test/repo",
                    branch="main",
                    title="Imported Wiki",
                    page_count=2,
                    status="complete",
                ))

        from app.models.db_models import WikiRecord as WR
        from sqlalchemy import select

        async with sf() as session:
            result = await session.execute(select(WR).where(WR.id == wiki_id))
            record = result.scalar_one_or_none()

        app.state.import_service.restore_wiki = AsyncMock(return_value=record)

        bundle_bytes = _make_wikis_bundle_bytes(wiki_id)
        resp = await c.post(
            "/api/v1/wikis/import",
            files={"bundle": ("test.wikiexport", bundle_bytes, "application/zip")},
        )
        assert resp.status_code == 201, resp.text
        data = resp.json()
        assert data["wiki_id"] == wiki_id
        assert data["title"] == "Imported Wiki"

    @pytest.mark.asyncio
    async def test_import_corrupt_file_returns_422(self, client):
        """POST /api/v1/wikis/import with corrupt bytes → 422."""
        c, app, sf = client

        app.state.import_service.restore_wiki = AsyncMock(
            side_effect=BundleValidationError("Invalid bundle: not a zip file")
        )

        resp = await c.post(
            "/api/v1/wikis/import",
            files={"bundle": ("bad.wikiexport", b"not a zip", "application/zip")},
        )
        assert resp.status_code == 422, resp.text
        assert "Invalid bundle" in resp.json().get("detail", "")

    @pytest.mark.asyncio
    async def test_import_wrong_version_returns_422(self, client):
        """POST /api/v1/wikis/import with unsupported bundle version → 422."""
        c, app, sf = client

        app.state.import_service.restore_wiki = AsyncMock(
            side_effect=BundleVersionError("Unsupported bundle version: 99")
        )

        bundle_bytes = _make_wikis_bundle_bytes()
        resp = await c.post(
            "/api/v1/wikis/import",
            files={"bundle": ("old.wikiexport", bundle_bytes, "application/zip")},
        )
        assert resp.status_code == 422, resp.text
        assert "version" in resp.json().get("detail", "").lower()

    @pytest.mark.asyncio
    async def test_import_s3_backend_returns_501(self, client):
        """POST /api/v1/wikis/import on S3 backend → 501."""
        c, app, sf = client

        app.state.import_service.restore_wiki = AsyncMock(
            side_effect=NotImplementedError("Wiki import not supported for S3 storage backend")
        )

        bundle_bytes = _make_wikis_bundle_bytes()
        resp = await c.post(
            "/api/v1/wikis/import",
            files={"bundle": ("bundle.wikiexport", bundle_bytes, "application/zip")},
        )
        assert resp.status_code == 501, resp.text

    @pytest.mark.asyncio
    async def test_import_bundle_under_size_limit_passes_to_service(self, client):
        """POST /api/v1/wikis/import with bundle under _MAX_IMPORT_BYTES → reaches service."""
        c, app, sf = client
        wiki_id = "import-size-ok-001"

        async with sf() as session:
            async with session.begin():
                session.add(WikiRecord(
                    id=wiki_id,
                    owner_id="dev-user",
                    repo_url="https://github.com/test/repo",
                    branch="main",
                    title="Size Test Wiki",
                    page_count=1,
                    status="complete",
                ))

        from app.models.db_models import WikiRecord as WR
        from sqlalchemy import select

        async with sf() as session:
            result = await session.execute(select(WR).where(WR.id == wiki_id))
            record = result.scalar_one_or_none()

        app.state.import_service.restore_wiki = AsyncMock(return_value=record)

        bundle_bytes = _make_wikis_bundle_bytes(wiki_id)
        # Bundle is well under the 500 MB limit — should reach the service
        resp = await c.post(
            "/api/v1/wikis/import",
            files={"bundle": ("small.wikiexport", bundle_bytes, "application/zip")},
        )
        assert resp.status_code == 201, resp.text
        app.state.import_service.restore_wiki.assert_called_once()

    @pytest.mark.asyncio
    async def test_import_bundle_over_size_limit_returns_413(self, client):
        """POST /api/v1/wikis/import with bundle over _MAX_IMPORT_BYTES → 413, service not called."""
        from unittest.mock import patch as _patch
        c, app, sf = client

        app.state.import_service.restore_wiki = AsyncMock()

        bundle_bytes = _make_wikis_bundle_bytes()
        # Patch _MAX_IMPORT_BYTES to a tiny value so our real bundle exceeds it
        with _patch("app.api.routes._MAX_IMPORT_BYTES", 10):
            resp = await c.post(
                "/api/v1/wikis/import",
                files={"bundle": ("large.wikiexport", bundle_bytes, "application/zip")},
            )
        assert resp.status_code == 413, resp.text
        app.state.import_service.restore_wiki.assert_not_called()

    @pytest.mark.asyncio
    async def test_import_missing_manifest_returns_422(self, client):
        """POST /api/v1/wikis/import with a zip that has no manifest.json → 422."""
        c, app, sf = client

        # Build a valid zip but omit manifest.json entirely
        buf = io.BytesIO()
        with zipfile.ZipFile(buf, mode="w") as zf:
            zf.writestr("pages/dummy/wiki_pages/intro.md", "# Intro\n")
            zf.writestr("cache/cache_key.txt", "")
            zf.writestr("cache/cache_index_fragment.json", "{}")
        buf.seek(0)
        no_manifest_bytes = buf.read()

        app.state.import_service.restore_wiki = AsyncMock(
            side_effect=BundleValidationError("Invalid bundle: missing manifest.json")
        )

        resp = await c.post(
            "/api/v1/wikis/import",
            files={"bundle": ("nomanifest.wikiexport", no_manifest_bytes, "application/zip")},
        )
        assert resp.status_code == 422, resp.text
        assert "manifest" in resp.json().get("detail", "").lower()

    @pytest.mark.asyncio
    async def test_import_docs_pkl_in_bundle_returns_422(self, client):
        """POST /api/v1/wikis/import with a .docs.pkl entry in the bundle → 422."""
        c, app, sf = client

        # Build a bundle that contains a .docs.pkl file
        manifest = {
            "bundle_version": 1,
            "title": "Pkl Test Wiki",
            "repo_url": "https://github.com/test/repo",
            "branch": "main",
            "commit_hash": None,
            "page_count": 1,
            "exported_at": "2026-04-11T00:00:00Z",
        }
        buf = io.BytesIO()
        with zipfile.ZipFile(buf, mode="w") as zf:
            zf.writestr("manifest.json", json.dumps(manifest))
            zf.writestr("pages/abc/wiki_pages/page.md", "# Page\n")
            zf.writestr("cache/cache_key.txt", "somekey")
            zf.writestr("cache/cache_index_fragment.json", "{}")
            zf.writestr("cache/somekey.docs.pkl", b"PICKLE_CONTENT")
        buf.seek(0)
        pkl_bundle = buf.read()

        app.state.import_service.restore_wiki = AsyncMock(
            side_effect=BundleValidationError("Invalid bundle: .docs.pkl files are not permitted")
        )

        resp = await c.post(
            "/api/v1/wikis/import",
            files={"bundle": ("pkl.wikiexport", pkl_bundle, "application/zip")},
        )
        assert resp.status_code == 422, resp.text
        detail = resp.json().get("detail", "")
        assert "pkl" in detail.lower() or "pickle" in detail.lower() or "invalid bundle" in detail.lower()

    @pytest.mark.asyncio
    async def test_import_same_bundle_twice_second_succeeds(self, client):
        """POST /api/v1/wikis/import the same bundle twice → second import succeeds (overwrite)."""
        c, app, sf = client
        wiki_id = "import-overwrite-001"

        # Seed the wiki record that restore_wiki will return
        async with sf() as session:
            async with session.begin():
                session.add(WikiRecord(
                    id=wiki_id,
                    owner_id="dev-user",
                    repo_url="https://github.com/test/repo",
                    branch="main",
                    title="Overwrite Wiki",
                    page_count=1,
                    status="complete",
                ))

        from sqlalchemy import select

        async with sf() as session:
            result = await session.execute(select(WikiRecord).where(WikiRecord.id == wiki_id))
            record = result.scalar_one_or_none()

        # Both imports return the same record (overwrite scenario handled by ImportService)
        app.state.import_service.restore_wiki = AsyncMock(return_value=record)

        bundle_bytes = _make_wikis_bundle_bytes(wiki_id)

        # First import
        resp1 = await c.post(
            "/api/v1/wikis/import",
            files={"bundle": ("bundle.wikiexport", bundle_bytes, "application/zip")},
        )
        assert resp1.status_code == 201, f"First import failed: {resp1.text}"
        assert resp1.json()["wiki_id"] == wiki_id

        # Second import — should also succeed (not 409 or 500)
        resp2 = await c.post(
            "/api/v1/wikis/import",
            files={"bundle": ("bundle.wikiexport", bundle_bytes, "application/zip")},
        )
        assert resp2.status_code == 201, f"Second import failed: {resp2.text}"
        data2 = resp2.json()
        assert data2["wiki_id"] == wiki_id
        assert data2["status"] == "complete"

    @pytest.mark.asyncio
    async def test_import_plain_text_file_returns_422(self, client):
        """POST /api/v1/wikis/import with a plain text file → 422 (not a zip)."""
        c, app, sf = client

        app.state.import_service.restore_wiki = AsyncMock(
            side_effect=BundleValidationError("Invalid bundle: not a zip file")
        )

        resp = await c.post(
            "/api/v1/wikis/import",
            files={"bundle": ("plain.wikiexport", b"this is just text, not a zip", "text/plain")},
        )
        assert resp.status_code == 422, resp.text
        assert "Invalid bundle" in resp.json().get("detail", "")

    @pytest.mark.asyncio
    async def test_export_wikis_format_generating_wiki_returns_409(self, client):
        """GET /api/v1/wikis/{id}/export?format=wikis on a generating wiki → 409."""
        c, app, sf = client
        wiki_id = "export-wikis-incomplete-001"
        await _seed_wiki(sf, wiki_id=wiki_id, status="generating")

        resp = await c.get(f"/api/v1/wikis/{wiki_id}/export?format=wikis")
        assert resp.status_code == 409, resp.text
        assert "fully generated" in resp.json().get("detail", "")
