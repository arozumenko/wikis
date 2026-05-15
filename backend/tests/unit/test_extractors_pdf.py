"""PDF-extractor tests (#118).

Requires the ``[pdf]`` extra (pypdfium2 + Pillow). When the deps aren't
installed the whole module is skipped so the rest of the test suite keeps
running.
"""

from __future__ import annotations

from pathlib import Path

import pytest

pypdfium2 = pytest.importorskip("pypdfium2")
PIL = pytest.importorskip("PIL")


from app.core.extractors.pdf import PDFExtractor  # noqa: E402


class _StubLLM:
    """Mimic LangChain AIMessage shape; reply with a per-page label so
    tests can verify the per-page concatenation contract."""

    def __init__(self) -> None:
        self.invoke_count = 0
        self.received: list = []
        self.responses: list[str] = []
        self.raise_on_invoke: Exception | None = None

    def queue_response(self, text: str) -> None:
        self.responses.append(text)

    def invoke(self, messages):
        self.invoke_count += 1
        self.received.append(messages)
        if self.raise_on_invoke is not None:
            raise self.raise_on_invoke

        text = self.responses.pop(0) if self.responses else "Default page text."

        class _AIMessage:
            def __init__(self, content: str) -> None:
                self.content = content
                self.usage_metadata = {"input_tokens": 1100, "output_tokens": 60}

        return _AIMessage(text)


def _make_pdf(path: Path, page_count: int) -> Path:
    """Create a tiny multi-page PDF with the bare minimum content
    pypdfium2 can render. Each page is blank — what matters is that
    the page count is real so the extractor's per-page loop runs the
    expected number of times.
    """
    pdf = pypdfium2.PdfDocument.new()
    try:
        for _ in range(page_count):
            pdf.new_page(595, 842)  # A4 in points
        pdf.save(str(path))
    finally:
        pdf.close()
    return path


class TestPDFExtractor:
    def test_supported_extensions(self) -> None:
        assert PDFExtractor(llm=_StubLLM()).supported_extensions == (".pdf",)

    def test_two_page_pdf_yields_two_descriptions(self, tmp_path: Path) -> None:
        pdf_path = _make_pdf(tmp_path / "doc.pdf", page_count=2)
        stub = _StubLLM()
        stub.queue_response("First page describes the architecture.")
        stub.queue_response("Second page lists API endpoints.")

        result = PDFExtractor(llm=stub).extract(pdf_path)

        assert result is not None
        assert result.page_count == 2
        assert result.extraction_method == "llm-vision"
        assert "## Page 1" in result.text
        assert "## Page 2" in result.text
        assert "architecture" in result.text
        assert "API endpoints" in result.text
        # Token usage is the sum across all pages.
        assert result.input_tokens == 2200
        assert result.output_tokens == 120
        assert stub.invoke_count == 2

    def test_blank_page_sentinel_drops_page(self, tmp_path: Path) -> None:
        pdf_path = _make_pdf(tmp_path / "doc.pdf", page_count=3)
        stub = _StubLLM()
        stub.queue_response("Page 1 content.")
        stub.queue_response("BLANK_PAGE")
        stub.queue_response("Page 3 content.")

        result = PDFExtractor(llm=stub).extract(pdf_path)

        assert result is not None
        assert "Page 1 content" in result.text
        assert "Page 3 content" in result.text
        assert "BLANK_PAGE" not in result.text
        assert "## Page 2" not in result.text
        # All three pages still cost tokens — the BLANK_PAGE call ran;
        # it just produced no usable text.
        assert stub.invoke_count == 3

    def test_all_pages_blank_returns_none(self, tmp_path: Path) -> None:
        pdf_path = _make_pdf(tmp_path / "doc.pdf", page_count=2)
        stub = _StubLLM()
        stub.queue_response("BLANK_PAGE")
        stub.queue_response("BLANK_PAGE")

        result = PDFExtractor(llm=stub).extract(pdf_path)
        # Zero usable pages → no document worth indexing.
        assert result is None

    def test_llm_error_on_one_page_records_warning(self, tmp_path: Path) -> None:
        pdf_path = _make_pdf(tmp_path / "doc.pdf", page_count=2)
        stub = _StubLLM()

        # Make page 1 fail by raising; page 2 succeeds via subclass.
        class _OneFailingStub(_StubLLM):
            def invoke(self, messages):
                self.invoke_count += 1
                if self.invoke_count == 1:
                    raise RuntimeError("rate limited")
                return super().invoke(messages)

        stub = _OneFailingStub()
        stub.queue_response("Successful page.")

        result = PDFExtractor(llm=stub).extract(pdf_path)

        assert result is not None
        assert "Successful page" in result.text
        assert any("LLM call failed" in w for w in result.warnings)
        # One page failed but the run continued — partial success is the
        # right behaviour for a flaky API.

    def test_corrupt_file_returns_none(self, tmp_path: Path) -> None:
        bad = tmp_path / "corrupt.pdf"
        bad.write_bytes(b"not a pdf at all")

        result = PDFExtractor(llm=_StubLLM()).extract(bad)
        assert result is None

    def test_construction_fails_loud_without_pypdfium2(self, monkeypatch) -> None:
        """When ``pypdfium2`` isn't importable, construction must raise
        ImportError so ``build_default_registry`` catches and logs the
        hint at registry-build time — not silently at the first PDF in
        the repo (where every PDF would be skipped without an obvious
        cause).
        """
        import sys

        # Hide pypdfium2 from the import machinery for the duration of
        # this test; ``setitem`` to None makes Python treat it as
        # "module not found" rather than a cached value.
        monkeypatch.setitem(sys.modules, "pypdfium2", None)

        with pytest.raises(ImportError):
            PDFExtractor(llm=_StubLLM())

    def test_construction_fails_loud_without_pillow(self, monkeypatch) -> None:
        """Same shape as the pypdfium2 test but for PIL — both deps must
        be present at construction time so missing-dep failures land at
        registry build, not deep in the index pass.
        """
        import sys

        monkeypatch.setitem(sys.modules, "PIL", None)

        with pytest.raises(ImportError):
            PDFExtractor(llm=_StubLLM())

    def test_render_scale_is_constructor_configurable(
        self, tmp_path: Path,
    ) -> None:
        """A non-default render_scale must flow through to ``page.render``.
        Verifies the constructor knob is wired, so Phase 2 can add env-
        var wiring without touching the contract.
        """
        pdf_path = _make_pdf(tmp_path / "doc.pdf", page_count=1)
        captured_scales: list[float] = []

        # Monkey-patch the render path indirectly: a custom subclass
        # captures the scale used during the single render call.
        class _SpyExtractor(PDFExtractor):
            def _describe_page(self, pdf, page_index, file_path):
                captured_scales.append(self._render_scale)
                return ("Scale=%.1f" % self._render_scale, 0, 0, None)

        result = _SpyExtractor(llm=_StubLLM(), render_scale=3.5).extract(pdf_path)
        assert result is not None
        assert captured_scales == [3.5]
