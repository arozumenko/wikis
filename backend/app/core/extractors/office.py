"""Office extractor — Office docs → PDF (via LibreOffice) → reuse PDF vision (#118).

For ``.docx``, ``.xlsx``, ``.pptx`` (and the legacy binary variants), the
right strategy is the same as for PDFs: render the visual representation
and let the multimodal LLM describe what it sees. Tables in spreadsheets,
slide layouts in presentations, embedded charts in documents — all of
these carry meaning that text-only parsers (python-docx, openpyxl,
python-pptx) silently lose. ``openpyxl`` for instance returns the raw
cell values but throws away the formulas as-displayed, which can flip
the numerical content of a spreadsheet from "Revenue = $1,234,567" to
just "1234567" with no thousands-separators, currency symbol, or
percentage formatting. That's data corruption disguised as extraction.

Instead this extractor invokes LibreOffice headless to convert the
office file to PDF, then delegates to the project's :class:`PDFExtractor`
to render each page and ask the LLM for a description. One pipeline,
one vision contract, one set of cost-logging lines.

System dependency: LibreOffice (the ``soffice`` binary). The backend
Docker image installs it via apt; local-dev users install via brew
(macOS) or their package manager. Construction-time check fails fast
with ``ImportError`` if ``soffice`` isn't on ``$PATH``, so
:func:`build_default_registry` catches and logs a hint pointing to
the install instructions.
"""

from __future__ import annotations

import logging
import os
import shutil
import signal
import subprocess
import tempfile
from dataclasses import replace
from pathlib import Path
from typing import TYPE_CHECKING

from app.core.extractors.pdf import PDFExtractor
from app.core.extractors.protocol import ExtractedDocument

if TYPE_CHECKING:
    from langchain_core.language_models import BaseChatModel

logger = logging.getLogger(__name__)


# Per-document timeout for the LibreOffice conversion. Most office files
# convert in under 10s; the 120s ceiling catches genuinely-stuck soffice
# processes (corrupt files, password-protected docs that prompt for a
# password even in headless mode) without sitting on the index pass.
_DEFAULT_CONVERSION_TIMEOUT_SECONDS = 120


class OfficeExtractor:
    """Convert Office docs to PDF via LibreOffice, then extract via vision.

    Holds a configured :class:`PDFExtractor` (constructed in our ``__init__``
    so vision deps fail-fast at registry-build time, not on the first
    office file in the repo). The conversion runs per file in a fresh
    temp directory; LibreOffice gets a unique ``UserInstallation`` so
    parallel invocations don't race the shared user-profile lock.
    """

    def __init__(
        self,
        *,
        llm: "BaseChatModel",
        soffice_path: str = "soffice",
        conversion_timeout: int = _DEFAULT_CONVERSION_TIMEOUT_SECONDS,
        pdf_extractor: PDFExtractor | None = None,
    ) -> None:
        # Construction-time availability check. Without this, every
        # office file in the repo would fail with the same "soffice not
        # found" error and the cause would only be visible after the
        # first conversion attempt — too late.
        resolved = shutil.which(soffice_path)
        if resolved is None:
            raise ImportError(
                f"LibreOffice ({soffice_path!r}) not found in PATH. "
                f"Install via apt (libreoffice-core, libreoffice-writer, "
                f"libreoffice-calc, libreoffice-impress) or via "
                f"`brew install --cask libreoffice` on macOS. Without "
                f"it, .docx/.xlsx/.pptx files fall back to the legacy "
                f"text-read path and produce binary-garbage content."
            )
        self._soffice = resolved
        self._timeout = conversion_timeout
        # Reuse the project's PDFExtractor verbatim so cost logging,
        # render scale, and the BLANK_PAGE sentinel all behave identically
        # across native PDFs and Office-converted PDFs.
        self._pdf_extractor = pdf_extractor or PDFExtractor(llm=llm)

    @property
    def supported_extensions(self) -> tuple[str, ...]:
        return (".docx", ".xlsx", ".pptx")

    def extract(self, file_path: Path) -> ExtractedDocument | None:
        with tempfile.TemporaryDirectory(prefix="wikis-office-") as tmpdir:
            tmpdir_path = Path(tmpdir)

            # Per-invocation LibreOffice profile dir. soffice serializes
            # access to its user profile by default; running parallel
            # conversions against one shared profile blocks or corrupts.
            # The graph builder's documentation pass is sequential today
            # but defense-in-depth costs us nothing and lets a future
            # parallel pass land safely.
            profile_dir = tmpdir_path / "lo-profile"
            profile_dir.mkdir()

            converted = self._convert_to_pdf(file_path, tmpdir_path, profile_dir)
            if converted is None:
                return None

            extracted = self._pdf_extractor.extract(converted)
            if extracted is None:
                return None

            # Tag the extraction method so telemetry can distinguish a
            # native-PDF run from an office-converted run. Useful when
            # debugging "why did this file's vision call take 30s?" —
            # office conversions add 5-15s of soffice startup on top.
            return replace(
                extracted,
                extraction_method=(
                    f"libreoffice→{extracted.extraction_method}"
                ),
            )

    def _convert_to_pdf(
        self,
        source: Path,
        outdir: Path,
        profile_dir: Path,
    ) -> Path | None:
        """Run ``soffice --headless --convert-to pdf`` for one file.

        Returns the path to the produced PDF, or ``None`` if conversion
        failed for any reason (timeout, non-zero exit, missing output
        file). All failure modes log WARNING with the underlying cause —
        operators can grep ``[extractors.office]`` to see which files
        the indexer couldn't read.
        """
        # The argument list intentionally avoids `shell=True` and passes
        # the source path as a positional arg, not interpolated into a
        # string — paths with spaces or shell metachars are safe.
        #
        # ``profile_dir.as_uri()`` is the stdlib-correct way to encode a
        # path as a file:// URI (handles spaces, %, unicode, and the
        # Windows drive-letter case). Hand-rolling ``f"file://{path}"``
        # silently breaks on macOS tmpdirs under ``/var/folders/…`` and
        # any homedir with a space — LibreOffice then falls back to the
        # shared system profile, defeating the per-invocation isolation.
        cmd = [
            self._soffice,
            f"-env:UserInstallation={profile_dir.as_uri()}",
            "--headless",
            "--convert-to",
            "pdf",
            "--outdir",
            str(outdir),
            str(source),
        ]

        # ``subprocess.run(timeout=...)`` only SIGKILLs the direct child
        # when the timeout fires. LibreOffice forks several helpers
        # (``oosplash``, ``soffice.bin``, JVM child processes) that
        # would survive as zombies reparented to PID 1 — over a long
        # index pass on a docs-heavy repo that leaks RAM + file
        # descriptors. ``start_new_session=True`` puts the whole tree
        # in a fresh process group; on TimeoutExpired we
        # ``os.killpg(SIGKILL)`` it as one unit. POSIX-only; Windows is
        # not a deploy target for this extractor (LibreOffice headless
        # behaves differently there anyway).
        try:
            proc = subprocess.Popen(  # noqa: S603 — fixed binary, no shell
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                start_new_session=True,
            )
        except OSError as exc:
            logger.warning(
                "[extractors.office] %s conversion failed to launch "
                "soffice: %s", source.name, exc,
            )
            return None

        try:
            _, stderr_bytes = proc.communicate(timeout=self._timeout)
            returncode = proc.returncode
        except subprocess.TimeoutExpired:
            # Reap the whole process group so the LibreOffice helpers
            # don't get reparented to init.
            try:
                os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
            except (ProcessLookupError, PermissionError):
                # Race: process already exited between the timeout
                # firing and the killpg call. Nothing to clean up.
                pass
            # Drain pipes + reap the direct child so we don't leave a
            # zombie. ``communicate`` here is a no-timeout wait because
            # we already SIGKILLed; it returns immediately.
            try:
                proc.communicate()
            except Exception:  # noqa: S110 — best-effort drain
                pass
            logger.warning(
                "[extractors.office] %s conversion timed out after %ds — "
                "skipping (file may be corrupt, encrypted, or unusually "
                "large)",
                source, self._timeout,
            )
            return None

        if returncode != 0:
            stderr = stderr_bytes.decode("utf-8", errors="replace").strip()
            logger.warning(
                "[extractors.office] %s conversion exited %d: %s",
                source.name, returncode,
                stderr or "(no stderr — check soffice logs)",
            )
            return None

        # soffice writes ``<stem>.pdf`` into ``outdir`` regardless of the
        # source's path. If a future LibreOffice version changes that
        # convention (or someone passes a weird filename), we glob for
        # any produced PDF rather than hard-coding the expected name.
        produced = list(outdir.glob("*.pdf"))
        if not produced:
            logger.warning(
                "[extractors.office] %s: soffice exited 0 but no PDF "
                "was produced in %s — skipping",
                source.name, outdir,
            )
            return None

        return produced[0]
