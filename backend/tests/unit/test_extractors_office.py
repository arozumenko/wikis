"""Office extractor tests (#118 phase 2).

The conversion step shells out to LibreOffice — we mock ``subprocess.Popen``
in the unit tests so they pass without the binary installed. The real
LibreOffice path is exercised by integration tests that skip when
``soffice`` isn't on ``$PATH``.

The Popen pattern is used (rather than ``subprocess.run``) so the
extractor can ``os.killpg`` the entire process group on timeout —
``subprocess.run(timeout=...)`` only SIGKILLs the direct child, which
leaks LibreOffice helpers as zombies reparented to PID 1 (CR1 from
the code-review pass).
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


class _FakePopen:
    """Minimal stand-in for the ``subprocess.Popen`` object the extractor
    uses. Tests configure ``returncode``, ``stderr_bytes``, and a
    ``side_effect`` that writes a PDF (or doesn't) into the outdir on
    construction. Mirrors the surface the extractor actually touches:
    ``.communicate(timeout=...)``, ``.returncode``, ``.pid``.
    """

    def __init__(
        self,
        cmd: list,
        *,
        returncode: int = 0,
        stderr_bytes: bytes = b"",
        on_construct=None,
        raise_on_communicate: Exception | None = None,
    ) -> None:
        self.cmd = cmd
        self.pid = 12345  # Stable pid for killpg assertions.
        self.returncode = returncode
        self._stderr = stderr_bytes
        self._raise_on_communicate = raise_on_communicate
        if on_construct is not None:
            on_construct(cmd)

    def communicate(self, timeout=None):  # noqa: ARG002 — timeout consumed by mock
        if self._raise_on_communicate is not None:
            raise self._raise_on_communicate
        return (b"", self._stderr)


def _popen_factory(*, on_construct=None, returncode=0, stderr=b"",
                   raise_on_communicate=None):
    """Build a callable suitable for ``side_effect`` of a Popen patch."""

    def _factory(cmd, **kwargs):
        return _FakePopen(
            cmd,
            returncode=returncode,
            stderr_bytes=stderr,
            on_construct=on_construct,
            raise_on_communicate=raise_on_communicate,
        )

    return _factory


def _write_pdf_side_effect(stem: str = "spec"):
    """Side effect that writes a real PDF into the soffice ``--outdir``."""

    def _side_effect(cmd: list) -> None:
        outdir = Path(cmd[cmd.index("--outdir") + 1])
        _make_real_pdf(outdir / f"{stem}.pdf")

    return _side_effect


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

    def test_user_installation_uri_uses_path_as_uri(
        self, extractor, tmp_path,
    ) -> None:
        """Rio R1: the ``-env:UserInstallation=…`` URI must use
        ``Path.as_uri()`` (stdlib-correct) instead of a hand-rolled
        ``f"file://{path}"`` so paths with spaces / special chars
        (macOS ``/var/folders/…``, Windows drive letters, homedirs with
        spaces) work correctly. Without this, soffice silently falls
        back to the shared system profile and defeats per-invocation
        isolation."""
        docx = tmp_path / "spec.docx"
        docx.write_bytes(b"data")

        factory = _popen_factory(on_construct=_write_pdf_side_effect("spec"))
        with patch("subprocess.Popen", side_effect=factory) as mock_popen:
            extractor.extract(docx)

        cmd = mock_popen.call_args.args[0]
        user_install_arg = next(
            c for c in cmd if c.startswith("-env:UserInstallation=")
        )
        # Strip the prefix to get the URI itself.
        uri = user_install_arg.split("=", 1)[1]
        # ``Path.as_uri()`` always produces three slashes for an
        # absolute POSIX path (``file:///…``) — that's how we know
        # the stdlib encoder was used, not a hand-rolled f-string.
        assert uri.startswith("file:///"), (
            f"Expected stdlib Path.as_uri() encoding (file:///…), got {uri!r}"
        )

    def test_popen_uses_new_session_for_process_group_cleanup(
        self, extractor, tmp_path,
    ) -> None:
        """CR1: the extractor must spawn soffice with
        ``start_new_session=True`` so the whole LibreOffice helper tree
        (oosplash, soffice.bin, JVM children) can be SIGKILLed as one
        process group on timeout. Without this, helpers reparent to
        PID 1 and leak RAM + file descriptors over the index pass."""
        docx = tmp_path / "spec.docx"
        docx.write_bytes(b"data")

        factory = _popen_factory(on_construct=_write_pdf_side_effect("spec"))
        with patch("subprocess.Popen", side_effect=factory) as mock_popen:
            extractor.extract(docx)

        # ``start_new_session=True`` is the load-bearing kwarg.
        kwargs = mock_popen.call_args.kwargs
        assert kwargs.get("start_new_session") is True, (
            "Popen must be called with start_new_session=True so the "
            "whole LibreOffice process tree can be killpg'd on timeout"
        )

    def test_happy_path_delegates_to_pdf_extractor(
        self, extractor, tmp_path,
    ) -> None:
        """CR2: soffice exit 0 + a PDF in outdir → call PDFExtractor
        with the produced PDF → return the tagged result. Assertions
        verify the inner extractor was invoked AND was handed the
        actual converted PDF — not just that the mock returned its
        canned value."""
        docx = tmp_path / "spec.docx"
        docx.write_bytes(b"PK\x03\x04 fake docx")

        factory = _popen_factory(on_construct=_write_pdf_side_effect("spec"))
        with patch("subprocess.Popen", side_effect=factory) as mock_popen:
            result = extractor.extract(docx)

        assert result is not None
        assert "Table: revenue 1.2M" in result.text
        # extraction_method is tagged so telemetry shows the office path.
        assert result.extraction_method == "libreoffice→llm-vision"
        # Token usage flows through unchanged from the inner PDFExtractor.
        assert result.input_tokens == 1100
        assert result.output_tokens == 80
        # The soffice invocation actually happened, with the right args.
        assert mock_popen.call_count == 1
        cmd = mock_popen.call_args.args[0]
        assert cmd[0] == "/usr/bin/soffice"
        assert "--headless" in cmd
        assert "--convert-to" in cmd
        assert "pdf" in cmd
        # Per-invocation user profile to avoid lock contention.
        # ``file:///`` (three slashes) is the signature of
        # ``Path.as_uri()`` — the hand-rolled f-string this replaced
        # would only emit two slashes, so checking for three keeps
        # this assertion meaningful even alongside the dedicated
        # ``test_user_installation_uri_uses_path_as_uri`` regression.
        assert any(c.startswith("-env:UserInstallation=file:///") for c in cmd)

        # CR2: inner PDFExtractor was actually called with the produced
        # PDF — a refactor that returned a mock value without invoking
        # the inner extractor would fail this assertion.
        extractor._pdf_extractor.extract.assert_called_once()
        produced_pdf = extractor._pdf_extractor.extract.call_args.args[0]
        assert isinstance(produced_pdf, Path)
        assert produced_pdf.suffix == ".pdf"
        assert produced_pdf.name == "spec.pdf"
        # The produced PDF actually existed at the moment of the call
        # (we can't check post-hoc because the tempdir is cleaned up).
        # The assertion in `_write_pdf_side_effect` would have created
        # it; if it didn't, the inner extractor wouldn't have been
        # passed an existing file.

    def test_nonzero_exit_returns_none_with_warning(
        self, extractor, tmp_path, caplog,
    ) -> None:
        """soffice failed to convert → log WARNING with stderr → return
        None so the file is skipped without crashing the index pass."""
        import logging

        docx = tmp_path / "broken.docx"
        docx.write_bytes(b"corrupt")

        factory = _popen_factory(
            returncode=1, stderr=b"unrecognized file format",
        )
        with patch("subprocess.Popen", side_effect=factory), \
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

    def test_timeout_killpg_reaps_process_group(
        self, extractor, tmp_path, caplog,
    ) -> None:
        """CR1: when ``communicate`` raises TimeoutExpired, the extractor
        must ``os.killpg`` the entire process group (not just SIGKILL
        the direct child). Verifies the kill primitive is invoked with
        the process group id derived from the Popen pid."""
        import logging

        docx = tmp_path / "stuck.docx"
        docx.write_bytes(b"data")

        factory = _popen_factory(
            on_construct=_write_pdf_side_effect("spec"),
            raise_on_communicate=subprocess.TimeoutExpired(
                cmd=[], timeout=120,
            ),
        )
        with patch("subprocess.Popen", side_effect=factory), \
             patch("os.getpgid", return_value=99999) as mock_getpgid, \
             patch("os.killpg") as mock_killpg, \
             caplog.at_level(logging.WARNING, logger="app.core.extractors.office"):
            result = extractor.extract(docx)

        assert result is None
        extractor._pdf_extractor.extract.assert_not_called()
        # The whole process group was reaped — not just the direct child.
        mock_getpgid.assert_called_once_with(12345)  # _FakePopen.pid
        mock_killpg.assert_called_once()
        kill_args = mock_killpg.call_args.args
        assert kill_args[0] == 99999  # the pgid we mocked
        # SIGKILL = 9 on POSIX.
        import signal as _sig
        assert kill_args[1] == _sig.SIGKILL
        assert any("timed out" in r.getMessage() for r in caplog.records)

    def test_timeout_killpg_handles_already_exited_process(
        self, extractor, tmp_path,
    ) -> None:
        """Race window between timeout firing and killpg call —
        process exited on its own → ``getpgid`` raises
        ProcessLookupError. The extractor must swallow it and still
        return None."""
        docx = tmp_path / "stuck.docx"
        docx.write_bytes(b"data")

        factory = _popen_factory(
            raise_on_communicate=subprocess.TimeoutExpired(cmd=[], timeout=120),
        )
        with patch("subprocess.Popen", side_effect=factory), \
             patch("os.getpgid", side_effect=ProcessLookupError):
            # Must not raise; must return None.
            result = extractor.extract(docx)
        assert result is None

    def test_no_output_file_returns_none_with_warning(
        self, extractor, tmp_path, caplog,
    ) -> None:
        """soffice exited 0 but produced no PDF (weird LibreOffice version
        / disk full / glob mismatch) — graceful skip with WARNING."""
        import logging

        docx = tmp_path / "spec.docx"
        docx.write_bytes(b"data")

        # Returncode 0 but no PDF written into outdir.
        factory = _popen_factory(returncode=0)
        with patch("subprocess.Popen", side_effect=factory), \
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

        extractor._pdf_extractor.extract.return_value = None

        factory = _popen_factory(on_construct=_write_pdf_side_effect("spec"))
        with patch("subprocess.Popen", side_effect=factory):
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
            "subprocess.Popen",
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
