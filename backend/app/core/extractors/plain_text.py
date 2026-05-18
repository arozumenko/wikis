"""Plain-text variants extractor (#118).

Handles formats that are already text but aren't named ``.md``:

* ``.mdx`` — Markdown with embedded JSX (used by Docusaurus, Astro, Next docs).
* ``.qmd`` — Quarto markdown (R/Python notebooks, scientific docs).
* ``.rst`` — reStructuredText (Sphinx).
* ``.adoc`` — AsciiDoc.

These are all UTF-8 (or close enough) text files; there's no visual content
the LLM would need to describe. Read, decode, return — same shape as if the
graph builder had handled them via the legacy path, just routed through the
extractor protocol for consistency.

``.md`` is intentionally **not** handled here. The graph builder's existing
markdown chunker has section-aware splitting that this extractor doesn't
replicate; until that's ported, leaving ``.md`` on the legacy path is
correct.
"""

from __future__ import annotations

import logging
from pathlib import Path

from app.core.extractors.protocol import ExtractedDocument

logger = logging.getLogger(__name__)


class PlainTextExtractor:
    """Read text variants as UTF-8 with errors='ignore'.

    The ``errors='ignore'`` matches the legacy graph_builder behaviour
    so a file with a stray latin-1 byte doesn't break the whole index
    pass. Operators relying on byte-perfect text round-trip should
    convert their docs to UTF-8 first; this is wiki ingestion, not
    archival storage.
    """

    @property
    def supported_extensions(self) -> tuple[str, ...]:
        return (".mdx", ".qmd", ".rst", ".adoc")

    def extract(self, file_path: Path) -> ExtractedDocument | None:
        try:
            with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                text = f.read()
        except OSError as exc:
            logger.warning(
                "[extractors.plain_text] failed to read %s: %s",
                file_path, exc,
            )
            return None

        if not text.strip():
            return None

        return ExtractedDocument(
            text=text,
            page_count=1,
            extraction_method="text",
        )
