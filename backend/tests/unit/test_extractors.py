"""Unit tests for the document-extractor protocol + plain-text + image extractors (#118).

PDF extractor tests live in ``test_extractors_pdf.py`` because they need
the optional ``[pdf]`` extra installed; skipped in environments without
pypdfium2.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from app.core.extractors import (
    DocumentExtractor,
    ExtractedDocument,
    ExtractorRegistry,
    build_default_registry,
)
from app.core.extractors.image import ImageExtractor
from app.core.extractors.plain_text import PlainTextExtractor


# ---------------------------------------------------------------------------
# Protocol + registry
# ---------------------------------------------------------------------------


def test_plain_text_extractor_satisfies_protocol() -> None:
    assert isinstance(PlainTextExtractor(), DocumentExtractor)


def test_registry_dispatches_by_extension(tmp_path: Path) -> None:
    registry = ExtractorRegistry()
    registry.register(PlainTextExtractor())

    assert registry.get(".rst") is not None
    assert registry.get(".mdx") is not None
    # Case-insensitive: callers pass whatever Path.suffix.lower() returns.
    assert registry.get(".RST") is not None
    # Missing leading dot is tolerated for the convenience callsites that
    # forget to include it.
    assert registry.get("adoc") is not None
    # Unknown extension yields None so the caller falls back to the legacy
    # text-read path.
    assert registry.get(".unknown") is None


def test_registry_replaces_handler_on_reregister() -> None:
    registry = ExtractorRegistry()

    class _A:
        supported_extensions = (".rst",)

        def extract(self, file_path: Path, *, llm=None) -> None:
            return None

    class _B:
        supported_extensions = (".rst",)

        def extract(self, file_path: Path, *, llm=None) -> None:
            return None

    a, b = _A(), _B()
    registry.register(a)
    assert registry.get(".rst") is a
    registry.register(b)
    assert registry.get(".rst") is b


def test_supported_extensions_returns_sorted_dot_prefixed() -> None:
    registry = ExtractorRegistry()
    registry.register(PlainTextExtractor())
    extensions = registry.supported_extensions()
    assert all(e.startswith(".") for e in extensions)
    assert extensions == tuple(sorted(extensions))


def test_build_default_registry_without_llm_skips_vision() -> None:
    """No LLM → only the plain-text extractor is registered. PDFs and
    images fall back to the legacy text-read path (which produces
    garbage for binary files but doesn't crash the index pass)."""
    registry = build_default_registry(llm=None)
    assert registry.get(".rst") is not None
    assert registry.get(".png") is None
    assert registry.get(".pdf") is None


# ---------------------------------------------------------------------------
# Plain-text extractor
# ---------------------------------------------------------------------------


class TestPlainTextExtractor:
    def test_reads_rst_file(self, tmp_path: Path) -> None:
        p = tmp_path / "doc.rst"
        p.write_text("Reading a reStructuredText file.\n=================================\n")

        result = PlainTextExtractor().extract(p)

        assert result is not None
        assert "reStructuredText" in result.text
        assert result.page_count == 1
        assert result.extraction_method == "text"
        assert result.input_tokens == 0

    def test_reads_mdx_qmd_adoc(self, tmp_path: Path) -> None:
        for ext, body in [
            (".mdx", "# Title\n\n<Component />"),
            (".qmd", "---\ntitle: Demo\n---\n```{python}\n1+1\n```"),
            (".adoc", "= Title\nAsciiDoc body."),
        ]:
            p = tmp_path / f"doc{ext}"
            p.write_text(body)
            result = PlainTextExtractor().extract(p)
            assert result is not None
            assert body in result.text

    def test_empty_file_returns_none(self, tmp_path: Path) -> None:
        p = tmp_path / "empty.rst"
        p.write_text("")
        assert PlainTextExtractor().extract(p) is None

    def test_whitespace_only_returns_none(self, tmp_path: Path) -> None:
        p = tmp_path / "blanks.rst"
        p.write_text("   \n\t\n  \n")
        assert PlainTextExtractor().extract(p) is None

    def test_md_is_not_handled_here(self, tmp_path: Path) -> None:
        # Markdown stays on the legacy path because of section-aware chunking.
        extractor = PlainTextExtractor()
        assert ".md" not in extractor.supported_extensions

    def test_non_utf8_bytes_do_not_crash(self, tmp_path: Path) -> None:
        # errors='ignore' matches legacy graph_builder behavior.
        p = tmp_path / "messy.rst"
        p.write_bytes(b"hello \xff\xfe world")
        result = PlainTextExtractor().extract(p)
        assert result is not None
        assert "hello" in result.text
        assert "world" in result.text


# ---------------------------------------------------------------------------
# Image extractor
# ---------------------------------------------------------------------------


class _StubLLM:
    """Minimal LangChain-compatible LLM stub.

    Records the messages it received so tests can assert on prompt
    structure, and returns a configurable response. Mimics LangChain's
    AIMessage shape including ``usage_metadata`` so the token-logging
    path exercises real-world fields.
    """

    def __init__(self, response_text: str = "A diagram showing X.") -> None:
        self.response_text = response_text
        self.received_messages: list = []
        self.invoke_count = 0
        self.raise_on_invoke: Exception | None = None

    def invoke(self, messages):
        self.invoke_count += 1
        self.received_messages.append(messages)
        if self.raise_on_invoke is not None:
            raise self.raise_on_invoke

        class _AIMessage:
            def __init__(self, content: str) -> None:
                self.content = content
                self.usage_metadata = {
                    "input_tokens": 850,
                    "output_tokens": 42,
                }

        return _AIMessage(self.response_text)


_TINY_PNG = bytes.fromhex(
    # 1×1 PNG, transparent — smallest valid PNG bytes for a smoke test.
    "89504E470D0A1A0A0000000D49484452000000010000000108060000001F15C4"
    "890000000A49444154789C6300010000000500010D0A2DB40000000049454E44"
    "AE426082"
)


class TestImageExtractor:
    def test_supported_extensions(self) -> None:
        ext = ImageExtractor(llm=_StubLLM()).supported_extensions
        assert ".png" in ext
        assert ".jpg" in ext
        assert ".webp" in ext
        # SVG is text-XML, not raster → intentionally not supported.
        assert ".svg" not in ext

    def test_extract_builds_multimodal_message(self, tmp_path: Path) -> None:
        p = tmp_path / "diag.png"
        p.write_bytes(_TINY_PNG)
        stub = _StubLLM("Architecture diagram with three boxes.")

        result = ImageExtractor(llm=stub).extract(p)

        assert result is not None
        assert result.text == "Architecture diagram with three boxes."
        assert result.extraction_method == "llm-vision"
        assert result.input_tokens == 850
        assert result.output_tokens == 42
        # Verify the LLM saw exactly one message with two content blocks
        # (image first, text prompt second). This is the multimodal
        # contract the extractor relies on.
        assert stub.invoke_count == 1
        sent = stub.received_messages[0][0]
        assert len(sent.content) == 2
        assert sent.content[0]["type"] == "image_url"
        assert sent.content[0]["image_url"]["url"].startswith("data:image/png;base64,")
        assert sent.content[1]["type"] == "text"
        assert "documentation" in sent.content[1]["text"].lower()

    def test_empty_response_returns_none(self, tmp_path: Path) -> None:
        p = tmp_path / "blank.png"
        p.write_bytes(_TINY_PNG)

        result = ImageExtractor(llm=_StubLLM(response_text="")).extract(p)
        assert result is None

    def test_empty_image_sentinel_returns_none(self, tmp_path: Path) -> None:
        # The prompt instructs the LLM to respond exactly "EMPTY_IMAGE"
        # when the image is blank; the extractor must drop these so blank
        # files don't pollute the wiki with garbage descriptions.
        p = tmp_path / "blank.png"
        p.write_bytes(_TINY_PNG)
        result = ImageExtractor(llm=_StubLLM(response_text="EMPTY_IMAGE")).extract(p)
        assert result is None

    def test_llm_error_returns_none_not_raises(self, tmp_path: Path) -> None:
        p = tmp_path / "fails.png"
        p.write_bytes(_TINY_PNG)
        stub = _StubLLM()
        stub.raise_on_invoke = RuntimeError("rate limited")

        result = ImageExtractor(llm=stub).extract(p)
        # Provider errors must not propagate — the index pass continues.
        assert result is None

    def test_jpeg_uses_correct_mime(self, tmp_path: Path) -> None:
        p = tmp_path / "photo.jpg"
        p.write_bytes(_TINY_PNG)  # raw bytes don't matter for the mime check
        stub = _StubLLM()
        ImageExtractor(llm=stub).extract(p)

        sent = stub.received_messages[0][0]
        url = sent.content[0]["image_url"]["url"]
        assert url.startswith("data:image/jpeg;base64,")

    def test_unsupported_extension_returns_none(self, tmp_path: Path) -> None:
        p = tmp_path / "doc.tiff"
        p.write_bytes(_TINY_PNG)
        assert ImageExtractor(llm=_StubLLM()).extract(p) is None

    def test_empty_file_returns_none(self, tmp_path: Path) -> None:
        p = tmp_path / "zero.png"
        p.write_bytes(b"")
        assert ImageExtractor(llm=_StubLLM()).extract(p) is None

    def test_oversized_image_is_skipped_with_warning(
        self, tmp_path: Path, caplog,
    ) -> None:
        """Pre-flight cap (3 MiB raw) prevents silent provider 400s from
        Anthropic (5 MiB base64 limit). The skip must surface as WARNING
        so operators can see why an image disappeared."""
        import logging

        p = tmp_path / "huge.png"
        # 4 MiB of zero bytes — over the cap, won't decode as a real
        # PNG but the size check fires before any image-decode path.
        p.write_bytes(b"\x00" * (4 * 1024 * 1024))
        stub = _StubLLM()

        with caplog.at_level(logging.WARNING, logger="app.core.extractors.image"):
            result = ImageExtractor(llm=stub).extract(p)

        assert result is None
        # LLM was NOT called — the size check short-circuited.
        assert stub.invoke_count == 0
        # The warning surfaces the actual cause, not a generic provider error.
        assert any("exceeds" in rec.message for rec in caplog.records)


# ---------------------------------------------------------------------------
# Token-usage helpers (cover the response-shape fallbacks)
# ---------------------------------------------------------------------------


def test_token_usage_falls_back_to_response_metadata() -> None:
    from app.core.extractors.image import _token_usage

    class _LegacyResponse:
        content = "x"
        response_metadata = {
            "token_usage": {"prompt_tokens": 100, "completion_tokens": 20},
        }

    assert _token_usage(_LegacyResponse()) == (100, 20)


def test_token_usage_returns_zero_when_no_metadata() -> None:
    from app.core.extractors.image import _token_usage

    class _NoMetadata:
        content = "x"

    assert _token_usage(_NoMetadata()) == (0, 0)


def test_extract_text_handles_list_content_blocks() -> None:
    from app.core.extractors.image import _extract_text

    class _BlockResponse:
        content = [
            {"type": "text", "text": "Hello "},
            {"type": "text", "text": "world."},
        ]

    assert _extract_text(_BlockResponse()) == "Hello world."
