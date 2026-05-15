"""Office extractor tests (#118 phase 2).

The conversion step shells out to LibreOffice — we mock ``subprocess.run``
in the unit tests so they pass without the binary installed. The real
LibreOffice path is exercised by integration tests that skip when
``soffice`` isn't on ``$PATH``.
"""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

pypdfium2 = pytest.importorskip("pypdfium2")  # OfficeExtractor depends on PDFExtractor

from app.core.extractors.office import OfficeExtractor  # noqa: E402
from app.core.extractors.protocol import ExtractedDocument  # noqa: E402


class _StubLLM:
    """Minimal LangChain stub — never invoked by the office tests
    (the PDFExtractor inside is mocked out)."""

    def invoke(self, _messages):
        raise AssertionError("LLM should not be invoked in these tests")


def _make_real_pdf(path: Path, page_count: int = 1) -> Path:
    """Create a tiny multi-page PDF that pypdfium2 can open. Used by
    tests that fake the soffice output but need a real PDF to flow
    through the PDFExtractor delegation path."""
    pdf = pypdfium2.PdfDocument.new()
    try:
        for _ in range(page_count):
            pdf.new_page(595, 842)
        pdf.save(str(path))
    finally:
        pdf.close()
    return path


# ---------------------------------------------------------------------------
# Construction-time soffice availability check
# ---------------------------------------------------------------------------


class TestConstruction:
    def test_missing_soffice_raises_importerror(self, monkeypatch) -> None:
        """Construction fails fast at registry-build time with a hint
        pointing to the install instructions — not silently when the
        first office file hits the extractor."""
        monkeypatch.setattr(shutil, "which", lambda name: None)
        with pytest.raises(ImportError) as exc_info:
            OfficeExtractor(llm=_StubLLM())
        assert "LibreOffice" in str(exc_info.value)
        # The error message names the actionable install commands so
        # operators don't have to grep docs.
        assert "apt" in str(exc_info.value) or "brew" in str(exc_info.value)

    def test_present_soffice_stores_resolved_path(self, monkeypatch) -> None:
        """``which`` returns an absolute path; the extractor should
        store that resolved path, not the original (potentially
        relative) name. Lets future processes inherit a stable
        absolute reference even if PATH changes mid-run."""
        monkeypatch.setattr(shutil, "which", lambda name: "/usr/bin/soffice")
        extractor = OfficeExtractor(llm=_StubLLM())
        assert extractor._soffice == "/usr/bin/soffice"

    def test_supported_extensions(self, monkeypatch) -> None:
        monkeypatch.setattr(shutil, "which", lambda name: "/usr/bin/soffice")
        extractor = OfficeExtractor(llm=_StubLLM())
        assert extractor.supported_extensions == (".docx", ".xlsx", ".pptx")


# ---------------------------------------------------------------------------
# extract() — happy path + failure modes
# ---------------------------------------------------------------------------


class TestExtract:
    @pytest.fixture
    def extractor(self, monkeypatch, tmp_path):
        """Build an OfficeExtractor with a fake-but-valid soffice path
        and a pre-canned PDFExtractor mock. The real PDFExtractor is
        constructed inside __init__ — we replace it after construction
        so the office-specific behaviour is isolated."""
        monkeypatch.setattr(shutil, "which", lambda name: "/usr/bin/soffice")
        ext = OfficeExtractor(llm=_StubLLM())

        # Replace the inner PDFExtractor with a mock that returns a
        # known result; office-level tests only care about the
        # conversion + delegation contract.
        ext._pdf_extractor = MagicMock()
        ext._pdf_extractor.extract.return_value = ExtractedDocument(
            text="## Page 1\n\nTable: revenue 1.2M",
            page_count=1,
            extraction_method="llm-vision",
            input_tokens=1100,
            output_tokens=80,
        )
        return ext

    def test_happy_path_delegates_to_pdf_extractor(
        self, extractor, tmp_path,
    ) -> None:
        """soffice exit 0 + a PDF in outdir → call PDFExtractor → return
        the tagged result."""
        docx = tmp_path / "spec.docx"
        docx.write_bytes(b"PK\x03\x04 fake docx")

        def _fake_soffice(cmd, **kwargs):
            # Mimic soffice by writing a real PDF into the outdir
            # specified on the command line.
            outdir_idx = cmd.index("--outdir")
            outdir = Path(cmd[outdir_idx + 1])
            _make_real_pdf(outdir / "spec.pdf")
            return subprocess.CompletedProcess(
                args=cmd, returncode=0, stdout=b"", stderr=b"",
            )

        with patch("subprocess.run", side_effect=_fake_soffice) as mock_run:
            result = extractor.extract(docx)

        assert result is not None
        assert "Table: revenue 1.2M" in result.text
        # extraction_method is tagged so telemetry shows the office path.
        assert result.extraction_method == "libreoffice→llm-vision"
        # Token usage flows through unchanged from the inner PDFExtractor.
        assert result.input_tokens == 1100
        assert result.output_tokens == 80
        # The soffice invocation actually happened, with the right args.
        assert mock_run.call_count == 1
        cmd = mock_run.call_args.args[0]
        assert cmd[0] == "/usr/bin/soffice"
        assert "--headless" in cmd
        assert "--convert-to" in cmd
        assert "pdf" in cmd
        # Per-invocation user profile to avoid lock contention.
        assert any(c.startswith("-env:UserInstallation=file://") for c in cmd)

    def test_nonzero_exit_returns_none_with_warning(
        self, extractor, tmp_path, caplog,
    ) -> None:
        """soffice failed to convert → log WARNING with stderr → return
        None so the file is skipped without crashing the index pass."""
        import logging

        docx = tmp_path / "broken.docx"
        docx.write_bytes(b"corrupt")

        fake_result = subprocess.CompletedProcess(
            args=[], returncode=1,
            stdout=b"", stderr=b"unrecognized file format",
        )

        with patch("subprocess.run", return_value=fake_result), \
             caplog.at_level(logging.WARNING, logger="app.core.extractors.office"):
            result = extractor.extract(docx)

        assert result is None
        # The PDFExtractor was NEVER called — conversion failure
        # short-circuits before the LLM.
        extractor._pdf_extractor.extract.assert_not_called()
        assert any(
            "exited 1" in r.getMessage() and "unrecognized file format" in r.getMessage()
            for r in caplog.records
        )

    def test_timeout_returns_none_with_warning(
        self, extractor, tmp_path, caplog,
    ) -> None:
        """A stuck soffice (corrupt file, password-prompt-in-headless,
        etc.) must not stall the whole index pass."""
        import logging

        docx = tmp_path / "stuck.docx"
        docx.write_bytes(b"data")

        with patch(
            "subprocess.run",
            side_effect=subprocess.TimeoutExpired(cmd=[], timeout=120),
        ), caplog.at_level(logging.WARNING, logger="app.core.extractors.office"):
            result = extractor.extract(docx)

        assert result is None
        extractor._pdf_extractor.extract.assert_not_called()
        assert any("timed out" in r.getMessage() for r in caplog.records)

    def test_no_output_file_returns_none_with_warning(
        self, extractor, tmp_path, caplog,
    ) -> None:
        """soffice exited 0 but produced no PDF (weird LibreOffice version
        / disk full / glob mismatch) — graceful skip with WARNING."""
        import logging

        docx = tmp_path / "spec.docx"
        docx.write_bytes(b"data")

        # Return success but never write a PDF.
        fake_result = subprocess.CompletedProcess(
            args=[], returncode=0, stdout=b"", stderr=b"",
        )
        with patch("subprocess.run", return_value=fake_result), \
             caplog.at_level(logging.WARNING, logger="app.core.extractors.office"):
            result = extractor.extract(docx)

        assert result is None
        assert any("no PDF was produced" in r.getMessage() for r in caplog.records)

    def test_pdf_extractor_none_propagates(self, extractor, tmp_path) -> None:
        """If the inner PDFExtractor returns None (e.g. all pages blank
        per BLANK_PAGE sentinel), the office extractor must also return
        None — not an empty ExtractedDocument."""
        docx = tmp_path / "spec.docx"
        docx.write_bytes(b"data")

        def _fake_soffice(cmd, **kwargs):
            outdir = Path(cmd[cmd.index("--outdir") + 1])
            _make_real_pdf(outdir / "spec.pdf")
            return subprocess.CompletedProcess(
                args=cmd, returncode=0, stdout=b"", stderr=b"",
            )

        extractor._pdf_extractor.extract.return_value = None

        with patch("subprocess.run", side_effect=_fake_soffice):
            result = extractor.extract(docx)
        assert result is None

    def test_oserror_during_launch_returns_none(
        self, extractor, tmp_path, caplog,
    ) -> None:
        """If the soffice binary disappears between construction and
        invocation (rare but happens in containers under image upgrade),
        OSError must be caught — not propagated."""
        import logging

        docx = tmp_path / "spec.docx"
        docx.write_bytes(b"data")

        with patch(
            "subprocess.run",
            side_effect=OSError("Text file busy"),
        ), caplog.at_level(logging.WARNING, logger="app.core.extractors.office"):
            result = extractor.extract(docx)

        assert result is None
        assert any("failed to launch" in r.getMessage() for r in caplog.records)


# ---------------------------------------------------------------------------
# Integration — real LibreOffice, skipped if not installed
# ---------------------------------------------------------------------------


_SOFFICE = shutil.which("soffice") or shutil.which("libreoffice")


@pytest.mark.skipif(
    _SOFFICE is None,
    reason="LibreOffice not installed — install via apt or brew to run",
)
class TestIntegrationWithRealLibreOffice:
    """Real-binary tests. Skipped when soffice isn't on PATH so CI without
    LibreOffice (and developers on minimal envs) still get green on the
    rest of the suite."""

    def test_real_docx_round_trip(self, tmp_path: Path) -> None:
        # Construct a minimal .docx via python-docx? That's an extra
        # dep we don't want. Instead, write a real but trivial .docx
        # using zipfile (docx is just a zip of XML).
        import zipfile
        from io import BytesIO

        docx = tmp_path / "tiny.docx"
        # The smallest viable .docx structure LibreOffice will read.
        with zipfile.ZipFile(docx, "w") as zf:
            zf.writestr(
                "[Content_Types].xml",
                '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
                '<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">'
                '<Default Extension="xml" ContentType="application/xml"/>'
                '<Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>'
                '<Override PartName="/word/document.xml" '
                'ContentType="application/vnd.openxmlformats-officedocument.wordprocessingml.document.main+xml"/>'
                "</Types>",
            )
            zf.writestr(
                "_rels/.rels",
                '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
                '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
                '<Relationship Id="rId1" '
                'Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" '
                'Target="word/document.xml"/></Relationships>',
            )
            zf.writestr(
                "word/document.xml",
                '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
                '<w:document xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main">'
                "<w:body><w:p><w:r><w:t>Hello from integration test</w:t></w:r></w:p></w:body>"
                "</w:document>",
            )

        # The extract will hit a real LLM call — that's not in scope for
        # this test, so we patch the PDFExtractor inside to capture
        # what conversion produced rather than spending tokens.
        extractor = OfficeExtractor(llm=_StubLLM())
        extractor._pdf_extractor = MagicMock()
        extractor._pdf_extractor.extract.return_value = ExtractedDocument(
            text="## Page 1\n\nHello from integration test",
            page_count=1,
            extraction_method="llm-vision",
        )

        result = extractor.extract(docx)
        # Real LibreOffice converted it to PDF (the inner mock proves
        # the PDF reached the extractor); office tag is applied.
        assert result is not None
        assert result.extraction_method == "libreoffice→llm-vision"
        # The PDFExtractor was handed a real PDF — verify the
        # produced PDF actually exists by inspecting the call.
        produced_pdf = extractor._pdf_extractor.extract.call_args.args[0]
        assert isinstance(produced_pdf, Path)
        # The temp directory is cleaned up by the time we get here, so
        # we can't assert .exists() — but the path shape is the contract.
