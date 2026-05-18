"""Unit tests for SourceMaterializer (#189)."""

from __future__ import annotations

import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import AsyncIterator

import pytest

from app.core.sources.base import FileContent, FileInfo, OriginPointer, SourceToolkit
from app.services.source_materializer import SourceMaterializer


# ---------------------------------------------------------------------------
# Helpers / fixtures
# ---------------------------------------------------------------------------

_NOW = datetime(2025, 1, 1, tzinfo=timezone.utc)


def _make_pointer(path: str, source_type: str = "fake") -> OriginPointer:
    return OriginPointer(
        source_type=source_type,
        ref=path,
        url=f"fake://{path}",
        ingested_at=_NOW,
    )


def _make_info(path: str, language: str | None = None, source_type: str = "fake") -> FileInfo:
    return FileInfo(
        path=path,
        language=language,
        size_bytes=len(path),
        origin=_make_pointer(path, source_type),
    )


class FakeToolkit(SourceToolkit):
    """Minimal toolkit yielding a fixed list of (path, language, content) tuples."""

    source_type = "fake"

    def __init__(self, files: list[tuple[str, str | None, str]]) -> None:
        # files: list of (path, language, content)
        self._files = files

    @classmethod
    def from_config(cls, config: dict) -> "FakeToolkit":
        return cls(config.get("files", []))

    async def list_files(
        self,
        include: list[str] | None = None,
        exclude: list[str] | None = None,
    ) -> AsyncIterator[FileInfo]:  # type: ignore[override]
        for path, language, _content in self._files:
            yield _make_info(path, language)

    async def fetch_content(self, pointer: OriginPointer) -> FileContent:
        for path, language, content in self._files:
            if path == pointer.ref:
                info = _make_info(path, language)
                return FileContent(info=info, content=content)
        raise FileNotFoundError(f"No fake file at {pointer.ref!r}")

    async def test_connection(self) -> str:
        return "ok"

    def build_origin_pointer(
        self,
        path: str,
        revision: str | None = None,
        line_start: int | None = None,
        line_end: int | None = None,
    ) -> OriginPointer:
        return _make_pointer(path)


# ---------------------------------------------------------------------------
# 1. Basic materialisation — 3 files written at expected paths
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_materialize_three_files() -> None:
    files = [
        ("src/main.py", "python", "print('hello')"),
        ("README.md", "markdown", "# Hello"),
        ("config/settings.json", "json", '{"key": "val"}'),
    ]
    toolkit = FakeToolkit(files)

    with tempfile.TemporaryDirectory() as tmpdir:
        mat = SourceMaterializer(toolkit, Path(tmpdir))
        workdir = await mat.materialize()

        assert workdir == Path(tmpdir)
        assert (workdir / "src" / "main.py").exists()
        assert (workdir / "README.md").exists()
        assert (workdir / "config" / "settings.json").exists()

        assert (workdir / "src" / "main.py").read_text() == "print('hello')"
        assert (workdir / "README.md").read_text() == "# Hello"
        assert (workdir / "config" / "settings.json").read_text() == '{"key": "val"}'


# ---------------------------------------------------------------------------
# 2. Confluence-shaped FileInfo: language="markdown" → .md extension appended
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_materialize_confluence_markdown_extension() -> None:
    """Confluence paths like "SPACE/Page Title" → "SPACE/Page Title.md"."""
    files = [
        ("ENGINEERING/Architecture Overview", "markdown", "# Architecture\n…"),
        ("HR/Onboarding", "markdown", "# Onboarding\n…"),
    ]
    toolkit = FakeToolkit(files)

    with tempfile.TemporaryDirectory() as tmpdir:
        mat = SourceMaterializer(toolkit, Path(tmpdir))
        workdir = await mat.materialize()

        assert (workdir / "ENGINEERING" / "Architecture Overview.md").exists()
        assert (workdir / "HR" / "Onboarding.md").exists()
        content = (workdir / "ENGINEERING" / "Architecture Overview.md").read_text()
        assert "# Architecture" in content


# ---------------------------------------------------------------------------
# 3. File that already ends in .md with language=markdown — no double extension
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_no_double_md_extension() -> None:
    files = [
        ("docs/guide.md", "markdown", "# Guide"),
    ]
    toolkit = FakeToolkit(files)

    with tempfile.TemporaryDirectory() as tmpdir:
        mat = SourceMaterializer(toolkit, Path(tmpdir))
        await mat.materialize()
        # Must exist as-is, not as guide.md.md
        assert (Path(tmpdir) / "docs" / "guide.md").exists()
        assert not (Path(tmpdir) / "docs" / "guide.md.md").exists()


# ---------------------------------------------------------------------------
# 4. Path-escape attempt → ValueError
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_path_escape_raises_value_error() -> None:
    """FileInfo.path that escapes workdir must raise ValueError."""
    files = [
        ("../escape.txt", None, "evil"),
    ]
    toolkit = FakeToolkit(files)

    with tempfile.TemporaryDirectory() as tmpdir:
        mat = SourceMaterializer(toolkit, Path(tmpdir))
        with pytest.raises(ValueError, match="escapes workdir"):
            await mat.materialize()


# ---------------------------------------------------------------------------
# 5. Context manager — cleanup removes tmpdir
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_context_manager_cleanup() -> None:
    """__aexit__ removes the workdir created by __aenter__."""
    files = [("hello.txt", None, "world")]
    toolkit = FakeToolkit(files)

    async with SourceMaterializer(toolkit) as workdir:
        assert workdir.exists()
        assert (workdir / "hello.txt").exists()

    # After exit the directory must be gone.
    assert not workdir.exists()


# ---------------------------------------------------------------------------
# 6. Explicit workdir is NOT deleted on __aexit__ when caller owns it
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_explicit_workdir_not_deleted_on_exit() -> None:
    """When caller provides a workdir, SourceMaterializer must not delete it."""
    files = [("a.py", "python", "x = 1")]
    toolkit = FakeToolkit(files)

    with tempfile.TemporaryDirectory() as tmpdir:
        provided = Path(tmpdir)
        async with SourceMaterializer(toolkit, provided) as workdir:
            assert workdir == provided

        # Caller's dir must survive.
        assert provided.exists()


# ---------------------------------------------------------------------------
# 7. Fetch failure on one file is logged but does not abort the rest
# ---------------------------------------------------------------------------


class PartialFailToolkit(FakeToolkit):
    """Fails fetch_content for the first file."""

    async def fetch_content(self, pointer: OriginPointer) -> FileContent:
        if pointer.ref == "fail.py":
            raise RuntimeError("simulated fetch error")
        return await super().fetch_content(pointer)


@pytest.mark.asyncio
async def test_partial_failure_skips_errored_file() -> None:
    files = [
        ("fail.py", "python", "oops"),
        ("ok.py", "python", "x = 1"),
    ]
    toolkit = PartialFailToolkit(files)

    with tempfile.TemporaryDirectory() as tmpdir:
        mat = SourceMaterializer(toolkit, Path(tmpdir))
        workdir = await mat.materialize()

        # The failing file must be absent; the ok file must be present.
        assert not (workdir / "fail.py").exists()
        assert (workdir / "ok.py").exists()
