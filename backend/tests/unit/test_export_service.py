"""Unit tests for ExportService — Obsidian vault zip builder."""

from __future__ import annotations

import io
import zipfile
from unittest.mock import AsyncMock, MagicMock

import pytest

from app.models.db_models import WikiRecord
from app.services.export_service import ExportService


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_wiki_record(
    wiki_id: str = "abc123",
    repo_url: str = "https://github.com/org/repo",
    branch: str = "main",
    title: str = "My Wiki",
    status: str = "complete",
) -> WikiRecord:
    record = WikiRecord()
    record.id = wiki_id
    record.repo_url = repo_url
    record.branch = branch
    record.title = title
    record.status = status
    record.owner_id = "user1"
    record.page_count = 2
    return record


async def _collect_zip(gen) -> bytes:
    """Drain an async generator into bytes."""
    chunks: list[bytes] = []
    async for chunk in gen:
        chunks.append(chunk)
    return b"".join(chunks)


def _open_zip(data: bytes) -> zipfile.ZipFile:
    return zipfile.ZipFile(io.BytesIO(data))


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def mock_storage():
    storage = AsyncMock()
    storage.list_artifacts = AsyncMock(
        return_value=[
            "abc123/wiki_pages/overview.md",
            "abc123/wiki_pages/architecture.md",
            # non-md artifact — should be skipped
            "abc123/index.faiss",
        ]
    )
    storage.download = AsyncMock(side_effect=_fake_download)
    return storage


async def _fake_download(bucket: str, name: str) -> bytes:
    pages = {
        "abc123/wiki_pages/overview.md": b"# Overview\n\nThis is the overview page.",
        "abc123/wiki_pages/architecture.md": b"# Architecture\n\nThis is the architecture page.",
    }
    return pages.get(name, b"")


async def _fake_download_with_readme(bucket: str, name: str) -> bytes:
    pages = {
        "abc123/wiki_pages/overview.md": b"# Overview\n\nThis is the overview page.",
        "abc123/wiki_pages/README.md": b"# README\n\nThis should be filtered out.",
    }
    return pages.get(name, b"")


@pytest.fixture
def mock_wiki_management():
    mgmt = AsyncMock()
    mgmt.get_wiki_record = AsyncMock(return_value=_make_wiki_record())
    return mgmt


@pytest.fixture
def service(mock_storage, mock_wiki_management):
    return ExportService(storage=mock_storage, wiki_management=mock_wiki_management)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestBuildObsidianZip:
    @pytest.mark.asyncio
    async def test_produces_valid_zip(self, service):
        """The generator yields bytes that form a valid ZIP archive."""
        data = await _collect_zip(service.build_obsidian_zip("abc123"))
        assert zipfile.is_zipfile(io.BytesIO(data)), "Output is not a valid ZIP file"

    @pytest.mark.asyncio
    async def test_readme_present_at_vault_root(self, service):
        """A README.md must exist at the vault root."""
        data = await _collect_zip(service.build_obsidian_zip("abc123"))
        with _open_zip(data) as zf:
            names = zf.namelist()
        readme_entries = [n for n in names if n.endswith("README.md")]
        assert readme_entries, f"No README.md found; got: {names}"

    @pytest.mark.asyncio
    async def test_readme_contains_wiki_title_and_repo_url(self, service):
        """README.md must contain the wiki title and repo URL."""
        data = await _collect_zip(service.build_obsidian_zip("abc123"))
        with _open_zip(data) as zf:
            readme_entries = [n for n in zf.namelist() if n.endswith("README.md")]
            content = zf.read(readme_entries[0]).decode("utf-8")
        assert "My Wiki" in content
        assert "https://github.com/org/repo" in content

    @pytest.mark.asyncio
    async def test_obsidian_app_json_present(self, service):
        """.obsidian/app.json must exist inside the vault."""
        data = await _collect_zip(service.build_obsidian_zip("abc123"))
        with _open_zip(data) as zf:
            names = zf.namelist()
        obsidian_entries = [n for n in names if n.endswith(".obsidian/app.json")]
        assert obsidian_entries, f"No .obsidian/app.json found; got: {names}"

    @pytest.mark.asyncio
    async def test_obsidian_app_json_content(self, service):
        """.obsidian/app.json must contain legacyEditor: false."""
        import json

        data = await _collect_zip(service.build_obsidian_zip("abc123"))
        with _open_zip(data) as zf:
            app_json_entries = [n for n in zf.namelist() if n.endswith(".obsidian/app.json")]
            raw = zf.read(app_json_entries[0])
        parsed = json.loads(raw)
        assert parsed.get("legacyEditor") is False

    @pytest.mark.asyncio
    async def test_pages_have_yaml_frontmatter(self, service):
        """Each wiki page must have YAML frontmatter injected."""
        data = await _collect_zip(service.build_obsidian_zip("abc123"))
        with _open_zip(data) as zf:
            md_pages = [
                n
                for n in zf.namelist()
                if n.endswith(".md") and not n.endswith("README.md")
            ]
            assert md_pages, "No wiki pages found in archive"
            for name in md_pages:
                content = zf.read(name).decode("utf-8")
                assert content.startswith("---\n"), (
                    f"Page {name} does not start with YAML frontmatter; got: {content[:80]!r}"
                )

    @pytest.mark.asyncio
    async def test_frontmatter_contains_required_fields(self, service):
        """Frontmatter must include title, repo, generated_at, and tags."""
        data = await _collect_zip(service.build_obsidian_zip("abc123"))
        with _open_zip(data) as zf:
            md_pages = [
                n
                for n in zf.namelist()
                if n.endswith(".md") and not n.endswith("README.md")
            ]
            for name in md_pages:
                content = zf.read(name).decode("utf-8")
                assert "title:" in content, f"No 'title' field in frontmatter of {name}"
                assert "repo:" in content, f"No 'repo' field in frontmatter of {name}"
                assert "generated_at:" in content, f"No 'generated_at' field in frontmatter of {name}"
                assert "tags:" in content, f"No 'tags' field in frontmatter of {name}"

    @pytest.mark.asyncio
    async def test_pages_organized_under_vault_title(self, service):
        """Wiki pages must be inside a folder named after the wiki title."""
        data = await _collect_zip(service.build_obsidian_zip("abc123"))
        with _open_zip(data) as zf:
            names = zf.namelist()
        # Vault folder is derived from wiki title "My Wiki"
        vault_pages = [n for n in names if "My Wiki" in n and n.endswith(".md") and "README" not in n]
        assert vault_pages, f"No pages found under vault folder; names: {names}"

    @pytest.mark.asyncio
    async def test_only_md_artifacts_included_as_pages(self, service):
        """Non-markdown artifacts (e.g. .faiss) must not appear as pages."""
        data = await _collect_zip(service.build_obsidian_zip("abc123"))
        with _open_zip(data) as zf:
            names = zf.namelist()
        assert not any(n.endswith(".faiss") for n in names), (
            f"Non-md artifact found in zip: {names}"
        )

    @pytest.mark.asyncio
    async def test_raises_for_unknown_wiki(self, mock_storage):
        """WikiNotFoundError is raised when wiki_id does not exist."""
        mgmt = AsyncMock()
        mgmt.get_wiki_record = AsyncMock(return_value=None)
        svc = ExportService(storage=mock_storage, wiki_management=mgmt)

        from app.services.export_service import WikiNotFoundError

        with pytest.raises(WikiNotFoundError):
            async for _ in svc.build_obsidian_zip("nonexistent"):
                pass

    @pytest.mark.asyncio
    async def test_readme_contains_generated_timestamp(self, service):
        """README.md must mention a generated_at timestamp."""
        data = await _collect_zip(service.build_obsidian_zip("abc123"))
        with _open_zip(data) as zf:
            readme_entries = [n for n in zf.namelist() if n.endswith("README.md")]
            content = zf.read(readme_entries[0]).decode("utf-8")
        assert "generated" in content.lower(), "README.md missing generated timestamp info"

    @pytest.mark.asyncio
    async def test_path_traversal_title_is_sanitized(self, mock_storage, mock_wiki_management):
        """A vault_title with path-traversal chars must be stripped before use in ZIP paths."""
        mock_wiki_management.get_wiki_record = AsyncMock(
            return_value=_make_wiki_record(title="../../evil")
        )
        svc = ExportService(storage=mock_storage, wiki_management=mock_wiki_management)
        data = await _collect_zip(svc.build_obsidian_zip("abc123"))
        with _open_zip(data) as zf:
            names = zf.namelist()
        # No ZIP entry must start with '.' or contain '..'
        for name in names:
            assert not name.startswith("."), f"ZIP entry starts with '.': {name}"
            assert ".." not in name, f"ZIP entry contains '..': {name}"

    @pytest.mark.asyncio
    async def test_no_duplicate_readme_when_artifact_storage_has_readme(self, mock_wiki_management):
        """When artifact storage contains a README.md, the ZIP must not have two README entries."""
        storage = AsyncMock()
        storage.list_artifacts = AsyncMock(
            return_value=[
                "abc123/wiki_pages/overview.md",
                "abc123/wiki_pages/README.md",   # collision candidate
                "abc123/index.faiss",
            ]
        )
        storage.download = AsyncMock(side_effect=_fake_download_with_readme)
        svc = ExportService(storage=storage, wiki_management=mock_wiki_management)
        data = await _collect_zip(svc.build_obsidian_zip("abc123"))
        with _open_zip(data) as zf:
            names = zf.namelist()
        readme_entries = [n for n in names if n.endswith("README.md")]
        assert len(readme_entries) == 1, (
            f"Expected exactly one README.md, got {len(readme_entries)}: {readme_entries}"
        )
