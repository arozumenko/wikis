"""Document extractors for non-code file ingestion (#118).

Code files go through tree-sitter parsers; documentation files (PDF, Office,
images, plain-text variants) go through this package. Each extractor turns
one file into a single text blob that downstream chunking, embedding, and
FTS5 indexing consume identically to a markdown body.

Vision-first design (per the project's steer): formats where text extraction
loses critical visual context — tables in PDFs, formulas in spreadsheets,
slide layouts in pptx — are rendered to images and described by a multimodal
LLM. Plain-text variants (.mdx, .rst, .adoc, .qmd) are read directly because
there is no visual content to lose. The LLM provider is whatever the project
is already configured with (Claude, GPT-4o, Gemini, etc.); the factory in
``app.services.llm_factory`` constructs vision-capable models without any
special-casing.

To add a new format:
    1. Implement ``DocumentExtractor`` in this package.
    2. Register it in :func:`build_default_registry`.
    3. Add its extension to ``DOCUMENTATION_EXTENSIONS`` in ``constants.py``
       and to ``FilterManager``'s allowlist.
    4. Add its optional pip extras to ``pyproject.toml``.

The registry is consulted by ``EnhancedUnifiedGraphBuilder._parse_documentation_files``;
files with no registered extractor fall back to the current text-read path
so previously-supported formats (.md, .yaml, .toml, etc.) keep working.
"""

from app.core.extractors.protocol import (
    DocumentExtractor,
    ExtractedDocument,
    ExtractorRegistry,
    build_default_registry,
)

__all__ = [
    "DocumentExtractor",
    "ExtractedDocument",
    "ExtractorRegistry",
    "build_default_registry",
]
