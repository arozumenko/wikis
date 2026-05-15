"""Protocol + registry for document extractors (#118).

The extractor protocol is intentionally thin: input a file path (+ optional
LLM for vision-based extractors), output an :class:`ExtractedDocument`
containing the natural-language representation of the document. Chunking,
node creation, embedding, and FTS5 indexing happen downstream — the
extractor's only job is the format-specific text extraction step.

Extractors are constructed once and shared across the index pass; they
must be stateless beyond construction-time configuration (LLM handle,
prompt template, etc.).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Protocol, runtime_checkable

if TYPE_CHECKING:
    from langchain_core.language_models import BaseChatModel

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ExtractedDocument:
    """The output of a :class:`DocumentExtractor` run.

    Attributes:
        text: The extracted natural-language representation. Goes straight
            into the existing chunking + embedding pipeline as if it were
            the body of a markdown file.
        page_count: Number of pages / slides / images in the source.
            ``1`` for plain-text files. Used for cost logging and to size
            the SPA progress indicator on long PDFs.
        extraction_method: A short label for telemetry — ``"text"`` for
            direct reads, ``"llm-vision"`` for multimodal extraction,
            ``"whisper"`` for audio transcription, etc.
        input_tokens: Estimated input tokens consumed by an LLM during
            extraction. ``0`` for text-only extractors. Logged per file
            so operators can see the spend.
        output_tokens: Same shape, output side.
        warnings: Non-fatal issues to surface in logs (e.g. "page 3
            returned empty description, may be blank"). Each is logged at
            WARNING level by the graph builder.
    """

    text: str
    page_count: int = 1
    extraction_method: str = "text"
    input_tokens: int = 0
    output_tokens: int = 0
    warnings: tuple[str, ...] = field(default_factory=tuple)


@runtime_checkable
class DocumentExtractor(Protocol):
    """Contract for non-code file extractors.

    Implementations live alongside this protocol in ``app.core.extractors``.
    The registry below dispatches by extension. An extractor that returns
    ``None`` (rather than raising) signals "this file looked extractable
    but produced nothing usable" — the graph builder skips it without
    counting it as a failure.
    """

    @property
    def supported_extensions(self) -> tuple[str, ...]:
        """File extensions this extractor handles, lowercase, with the
        leading dot — e.g. ``('.pdf',)`` or ``('.png', '.jpg', '.jpeg')``.
        """
        ...

    def extract(self, file_path: Path) -> ExtractedDocument | None:
        """Extract text from ``file_path``.

        Args:
            file_path: Absolute path to the file.

        Returns:
            ``ExtractedDocument`` with the text + telemetry, or ``None``
            when the file is unreadable / empty / unsupported in a way
            the extractor can detect at runtime (e.g. corrupt PDF, blank
            image, or any error the extractor can recover from without
            crashing the whole index pass).

        Implementation notes:
            Vision-based extractors hold their configured LLM as an
            instance attribute set at construction time. The graph
            builder constructs the registry once via
            :func:`build_default_registry` and reuses it across the
            index pass; tests inject a stub LLM by constructing a new
            extractor with the stub in its constructor.
        """
        ...


class ExtractorRegistry:
    """Extension → extractor lookup.

    Built once at graph-builder construction time and consulted per file.
    A missing extension yields ``None`` so the graph builder can fall
    back to the legacy text-read path for any extension we haven't
    promoted to a structured extractor yet.
    """

    def __init__(self) -> None:
        self._by_extension: dict[str, DocumentExtractor] = {}

    def register(self, extractor: DocumentExtractor) -> None:
        """Register ``extractor`` for every extension in its
        ``supported_extensions``. Re-registering an extension replaces
        the prior handler — useful for tests that need to inject a stub.
        """
        for ext in extractor.supported_extensions:
            normalized = ext.lower()
            if not normalized.startswith("."):
                normalized = "." + normalized
            self._by_extension[normalized] = extractor

    def get(self, extension: str) -> DocumentExtractor | None:
        """Return the extractor for ``extension`` (case-insensitive,
        leading dot optional), or ``None`` if no handler is registered.
        """
        normalized = extension.lower()
        if not normalized.startswith("."):
            normalized = "." + normalized
        return self._by_extension.get(normalized)

    def supported_extensions(self) -> tuple[str, ...]:
        """All registered extensions, useful for surfacing in
        diagnostics and tests."""
        return tuple(sorted(self._by_extension.keys()))


def build_default_registry(
    *, llm: "BaseChatModel | None" = None,
) -> ExtractorRegistry:
    """Construct the production registry.

    Optional dependencies (pypdfium2, Pillow) are imported lazily inside
    the extractor modules. An extractor whose deps aren't installed is
    silently skipped — the file falls through to the legacy text path.

    Args:
        llm: Configured LangChain chat model. When ``None``, vision-based
            extractors are skipped (their files fall back to the legacy
            text-read path, which produces a useless binary blob but
            doesn't crash). Most callers pass the project-configured LLM
            from ``app.services.llm_factory``.
    """
    from app.core.extractors.plain_text import PlainTextExtractor

    registry = ExtractorRegistry()
    registry.register(PlainTextExtractor())

    if llm is not None:
        try:
            from app.core.extractors.image import ImageExtractor

            registry.register(ImageExtractor(llm=llm))
        except ImportError as exc:
            logger.warning(
                "[extractors] image extractor disabled: %s — install "
                "Pillow to enable (pip install wikis-backend[vision])",
                exc,
            )

        try:
            from app.core.extractors.pdf import PDFExtractor

            registry.register(PDFExtractor(llm=llm))
        except ImportError as exc:
            logger.warning(
                "[extractors] PDF extractor disabled: %s — install "
                "pypdfium2 + Pillow to enable "
                "(pip install wikis-backend[pdf])",
                exc,
            )

    return registry
