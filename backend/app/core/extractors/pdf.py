"""PDF extractor — render pages, describe each with LLM vision (#118).

Rationale (per the project's call): ``pypdf`` and similar text-extraction
libraries handle prose-only PDFs fine, but mangle tables, multi-column
layouts, mathematical typesetting, and anything else where the visual
arrangement carries meaning. For a wiki where engineers will query
"the table from the migration RFC", losing the table layout makes the
content effectively unsearchable. Rendering each page to an image and
asking the multimodal LLM to describe it preserves all that context at
the cost of one vision call per page.

The trade-off: rendering + vision is ~100× slower and ~100× more expensive
than text extraction. Per-page logging surfaces the spend; operators
monitor and intervene if they discover a 500-page scanned manual in
their docs/ folder.

Uses ``pypdfium2`` (binding to the Chromium PDFium engine) rather than
``pdf2image`` because pdfium2 is a self-contained Python wheel — no
``poppler-utils`` system dependency to install in dev or production
containers.
"""

from __future__ import annotations

import io
import logging
from pathlib import Path
from typing import TYPE_CHECKING

from app.core.extractors.image import _extract_text, _token_usage
from app.core.extractors.protocol import ExtractedDocument

if TYPE_CHECKING:
    from langchain_core.language_models import BaseChatModel

logger = logging.getLogger(__name__)


_PER_PAGE_PROMPT = (
    "You are indexing this page of a PDF for a technical documentation "
    "wiki. Describe everything on the page in detail — body text "
    "verbatim where readable, tables row by row, diagrams, charts, "
    "figures, code blocks, footnotes. Preserve the document's logical "
    "structure (headings, lists, table rows) using markdown. Do not "
    "add commentary about the page's quality or style. If the page is "
    "blank, respond with exactly: BLANK_PAGE."
)


# pypdfium2 returns pixels per inch via a scale factor; 2.0 gives ~144 DPI
# which is sharp enough for vision OCR on most text-bearing PDFs while
# keeping the image bytes small enough to stay under per-message size caps.
_RENDER_SCALE = 2.0


class PDFExtractor:
    """Render PDF pages and describe each via LLM vision.

    Lazy-imports pypdfium2 + Pillow so the rest of the codebase keeps
    working without the ``[pdf]`` extra installed. Raises ``ImportError``
    at construction time when deps are missing — :func:`build_default_registry`
    catches and logs.
    """

    def __init__(self, *, llm: "BaseChatModel") -> None:
        # Import-time check: fail fast at registry-build time, not at the
        # first PDF in the repo (where the failure would be one file
        # silently skipped per missed dependency).
        import pypdfium2 as _  # noqa: F401
        from PIL import Image as _Image  # noqa: F401

        self._llm = llm

    @property
    def supported_extensions(self) -> tuple[str, ...]:
        return (".pdf",)

    def extract(
        self,
        file_path: Path,
        *,
        llm: "BaseChatModel | None" = None,
    ) -> ExtractedDocument | None:
        import pypdfium2 as pdfium

        active_llm = llm if llm is not None else self._llm
        if active_llm is None:
            return None

        try:
            pdf = pdfium.PdfDocument(str(file_path))
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "[extractors.pdf] failed to open %s: %s", file_path, exc,
            )
            return None

        try:
            page_count = len(pdf)
            if page_count == 0:
                return None

            page_descriptions: list[str] = []
            warnings: list[str] = []
            total_in = 0
            total_out = 0

            for page_index in range(page_count):
                page_text, in_tokens, out_tokens, warning = (
                    self._describe_page(pdf, page_index, active_llm, file_path)
                )
                total_in += in_tokens
                total_out += out_tokens
                if warning:
                    warnings.append(warning)
                if page_text:
                    page_descriptions.append(
                        f"## Page {page_index + 1}\n\n{page_text}"
                    )
        finally:
            pdf.close()

        if not page_descriptions:
            return None

        # One total-spend line per file so the log is greppable without
        # walking every per-page line.
        logger.info(
            "[extractors.pdf] file=%s pages=%d input_tokens=%d output_tokens=%d",
            file_path.name, page_count, total_in, total_out,
        )

        return ExtractedDocument(
            text="\n\n".join(page_descriptions),
            page_count=page_count,
            extraction_method="llm-vision",
            input_tokens=total_in,
            output_tokens=total_out,
            warnings=tuple(warnings),
        )

    def _describe_page(
        self,
        pdf,
        page_index: int,
        llm: "BaseChatModel",
        file_path: Path,
    ) -> tuple[str, int, int, str | None]:
        """Render a single page and ask the LLM to describe it.

        Returns ``(text, input_tokens, output_tokens, warning)``.
        ``text`` is the empty string for blank or unrenderable pages;
        ``warning`` is set on those cases so the caller can surface them.
        """
        import base64

        from langchain_core.messages import HumanMessage
        from PIL import Image  # noqa: F401 — kept for type-check parity

        try:
            page = pdf[page_index]
            pil_image = page.render(scale=_RENDER_SCALE).to_pil()
            buf = io.BytesIO()
            pil_image.save(buf, format="PNG")
            png_bytes = buf.getvalue()
            page.close()
        except Exception as exc:  # noqa: BLE001
            warning = (
                f"page {page_index + 1} of {file_path.name}: "
                f"render failed: {exc}"
            )
            logger.warning("[extractors.pdf] %s", warning)
            return ("", 0, 0, warning)

        b64 = base64.b64encode(png_bytes).decode("ascii")
        message = HumanMessage(content=[
            {
                "type": "image_url",
                "image_url": {"url": f"data:image/png;base64,{b64}"},
            },
            {"type": "text", "text": _PER_PAGE_PROMPT},
        ])

        try:
            response = llm.invoke([message])
        except Exception as exc:  # noqa: BLE001
            warning = (
                f"page {page_index + 1} of {file_path.name}: "
                f"LLM call failed: {exc}"
            )
            logger.warning("[extractors.pdf] %s", warning)
            return ("", 0, 0, warning)

        text = _extract_text(response).strip()
        if not text or text == "BLANK_PAGE":
            return ("", *_token_usage(response), None)

        in_tokens, out_tokens = _token_usage(response)
        logger.debug(
            "[extractors.pdf] file=%s page=%d input_tokens=%d output_tokens=%d",
            file_path.name, page_index + 1, in_tokens, out_tokens,
        )
        return (text, in_tokens, out_tokens, None)
