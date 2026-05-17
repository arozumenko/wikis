"""Integration test for #118: graph_builder routes non-code files through
the extractor registry.

Drives ``EnhancedUnifiedGraphBuilder._parse_documentation_files`` with a
mixed fixture (.rst, .png, .pdf, .md) and asserts:

* The extractor registry is consulted first.
* Extractor output becomes ``source_text`` on the produced ``BasicSymbol``.
* Files with no registered extractor fall back to the legacy text-read.
* An extractor returning ``None`` skips the file without crashing the
  whole batch.
"""

from __future__ import annotations

from pathlib import Path

import pytest

pypdfium2 = pytest.importorskip("pypdfium2")

from app.core.code_graph.graph_builder import EnhancedUnifiedGraphBuilder
from app.core.extractors import ExtractorRegistry
from app.core.extractors.image import ImageExtractor
from app.core.extractors.pdf import PDFExtractor
from app.core.extractors.plain_text import PlainTextExtractor


class _StubLLM:
    """Simple stub: returns a fixed string + LangChain-style usage_metadata."""

    def __init__(self, text: str = "An image of a sequence diagram.") -> None:
        self.text = text
        self.invoke_count = 0

    def invoke(self, _messages):
        self.invoke_count += 1

        class _AIMessage:
            def __init__(self, content: str) -> None:
                self.content = content
                self.usage_metadata = {"input_tokens": 100, "output_tokens": 20}

        return _AIMessage(self.text)


_TINY_PNG = bytes.fromhex(
    "89504E470D0A1A0A0000000D49484452000000010000000108060000001F15C4"
    "890000000A49444154789C6300010000000500010D0A2DB40000000049454E44"
    "AE426082"
)


def _seed_mixed_docs(tmp_path: Path) -> Path:
    """Build a tiny repo-like tree with one file per format we care about."""
    docs = tmp_path / "docs"
    docs.mkdir()

    (docs / "guide.rst").write_text(
        "Operations Guide\n================\n\nHow to deploy the service.\n"
    )
    (docs / "diagram.png").write_bytes(_TINY_PNG)

    pdf_path = docs / "spec.pdf"
    pdf = pypdfium2.PdfDocument.new()
    try:
        pdf.new_page(595, 842)
        pdf.save(str(pdf_path))
    finally:
        pdf.close()

    # .md stays on the legacy path — assert that wiring still works.
    (docs / "README.md").write_text("# Welcome\n\nWelcome to the docs.\n")

    return docs


def test_documentation_files_route_through_extractor_registry(
    tmp_path: Path,
) -> None:
    """Verifies (a) extractor.extract is actually called per file (spy on
    the registry) and (b) extractor output lands as ``source_text``. A
    future refactor that bypasses the registry would fail the call-spy
    assertion before content-match coincidences could hide it.
    """
    docs = _seed_mixed_docs(tmp_path)
    llm = _StubLLM("A sequence diagram showing client → API → DB.")

    # Wrap each extractor's extract() so the test can assert each was
    # invoked with the expected file path.
    plain = PlainTextExtractor()
    image = ImageExtractor(llm=llm)
    pdf = PDFExtractor(llm=llm)

    extract_calls: list[tuple[str, Path]] = []

    def _wrap(extractor, label: str):
        original = extractor.extract

        def _spy(file_path: Path):
            extract_calls.append((label, file_path))
            return original(file_path)

        extractor.extract = _spy  # type: ignore[method-assign]
        return extractor

    registry = ExtractorRegistry()
    registry.register(_wrap(plain, "plain"))
    registry.register(_wrap(image, "image"))
    registry.register(_wrap(pdf, "pdf"))

    builder = EnhancedUnifiedGraphBuilder(extractor_registry=registry)

    file_paths = [
        str(docs / "guide.rst"),
        str(docs / "diagram.png"),
        str(docs / "spec.pdf"),
        str(docs / "README.md"),  # legacy path — no extractor registered
    ]
    results = builder._parse_documentation_files(file_paths, str(tmp_path))

    # All four files produced parse results.
    assert len(results) == 4

    # Registry was consulted for every non-legacy file. Catches future
    # refactors that accidentally bypass the dispatch.
    call_labels = {label for label, _ in extract_calls}
    assert call_labels == {"plain", "image", "pdf"}
    # .md was NOT routed through the registry (no extractor registered).
    assert not any(p.suffix == ".md" for _, p in extract_calls)

    # .rst was read through the plain-text extractor.
    rst_result = results[str(docs / "guide.rst")]
    rst_text = rst_result.symbols[0].source_text
    assert "Operations Guide" in rst_text
    assert "How to deploy the service" in rst_text

    # .png was described by the LLM-vision stub.
    png_result = results[str(docs / "diagram.png")]
    png_text = png_result.symbols[0].source_text
    assert "sequence diagram" in png_text
    assert "client → API → DB" in png_text

    # .pdf was rendered + described.
    pdf_result = results[str(docs / "spec.pdf")]
    pdf_text = pdf_result.symbols[0].source_text
    assert "## Page 1" in pdf_text

    # .md fell through to the legacy text-read.
    md_result = results[str(docs / "README.md")]
    md_text = md_result.symbols[0].source_text
    assert "Welcome to the docs" in md_text

    # Both LLM-vision files used the stub (png + pdf = 2 invocations).
    assert llm.invoke_count == 2


def test_extractor_returning_none_skips_file_without_crashing(
    tmp_path: Path,
) -> None:
    """An empty image file → ImageExtractor returns None → graph builder
    logs "skipping" and continues with the rest of the batch."""
    docs = tmp_path / "docs"
    docs.mkdir()
    (docs / "blank.png").write_bytes(b"")  # zero bytes
    (docs / "guide.rst").write_text("Real content.")

    registry = ExtractorRegistry()
    registry.register(PlainTextExtractor())
    registry.register(ImageExtractor(llm=_StubLLM()))

    builder = EnhancedUnifiedGraphBuilder(extractor_registry=registry)
    results = builder._parse_documentation_files(
        [str(docs / "blank.png"), str(docs / "guide.rst")],
        str(tmp_path),
    )

    # blank.png was skipped (extractor returned None); guide.rst landed.
    assert str(docs / "blank.png") not in results
    assert str(docs / "guide.rst") in results


def test_no_registry_uses_legacy_text_read_path(tmp_path: Path) -> None:
    """Constructing the builder without a registry preserves the old
    behaviour — .md / .yaml / .toml etc. flow through ``open()``."""
    docs = tmp_path / "docs"
    docs.mkdir()
    (docs / "README.md").write_text("# Hello\n")

    builder = EnhancedUnifiedGraphBuilder(extractor_registry=None)
    results = builder._parse_documentation_files(
        [str(docs / "README.md")], str(tmp_path),
    )

    assert str(docs / "README.md") in results
    assert "Hello" in results[str(docs / "README.md")].symbols[0].source_text


def test_vision_eligible_file_without_extractor_warns_once_per_extension(
    tmp_path: Path, caplog,
) -> None:
    """#148: when a .pdf or .png lands on the legacy text-read path
    (no extractor registered — missing LLM config or pip extra), WARN
    so operators see why their wiki has binary-garbage entries.

    Once per extension per index pass — three PDFs and two PNGs in the
    same run produce two WARNINGs, not five. Otherwise a 200-PDF repo
    would emit 200 useless WARNINGs.
    """
    import logging

    docs = tmp_path / "docs"
    docs.mkdir()
    (docs / "spec.pdf").write_bytes(b"%PDF-1.4 not a real pdf")
    (docs / "spec2.pdf").write_bytes(b"%PDF-1.4 not a real pdf either")
    (docs / "diag.png").write_bytes(_TINY_PNG)
    (docs / "diag2.png").write_bytes(_TINY_PNG)
    (docs / "README.md").write_text("# Hello\n")  # not vision-eligible

    # Builder constructed WITHOUT a registry → vision-eligible files
    # have no extractor.
    builder = EnhancedUnifiedGraphBuilder(extractor_registry=None)

    with caplog.at_level(logging.WARNING, logger="app.core.code_graph.graph_builder"):
        builder._parse_documentation_files(
            [
                str(docs / "spec.pdf"),
                str(docs / "spec2.pdf"),
                str(docs / "diag.png"),
                str(docs / "diag2.png"),
                str(docs / "README.md"),
            ],
            str(tmp_path),
        )

    warnings = [
        r.getMessage() for r in caplog.records
        if r.levelno == logging.WARNING
        and "will be ingested via the legacy text-read path" in r.getMessage()
    ]

    # Exactly two warnings: one for .pdf, one for .png. README.md is not
    # in KNOWN_VISION_EXTENSIONS so doesn't fire one.
    assert len(warnings) == 2
    pdf_warns = [w for w in warnings if ".pdf" in w]
    png_warns = [w for w in warnings if ".png" in w]
    assert len(pdf_warns) == 1
    assert len(png_warns) == 1
    # The WARNING names the actionable config so operators don't have
    # to grep the docs to find out what to set.
    assert all("LLM_API_KEY" in w for w in warnings)


def test_warned_extensions_reset_per_index_pass(
    tmp_path: Path, caplog,
) -> None:
    """Rio's blocking finding on the second review: the dedup set must
    reset per index pass, not persist for the builder's lifetime.
    Otherwise a webhook-triggered re-index after the operator fixed
    their LLM config would silently suppress the expected WARNING."""
    import logging

    docs = tmp_path / "docs"
    docs.mkdir()
    (docs / "spec.pdf").write_bytes(b"%PDF-1.4 not real")

    builder = EnhancedUnifiedGraphBuilder(extractor_registry=None)

    # First pass: WARN expected.
    with caplog.at_level(logging.WARNING, logger="app.core.code_graph.graph_builder"):
        builder._parse_documentation_files(
            [str(docs / "spec.pdf")], str(tmp_path),
        )
    first_pass_warnings = [
        r for r in caplog.records
        if "will be ingested via the legacy text-read path" in r.getMessage()
    ]
    assert len(first_pass_warnings) == 1

    caplog.clear()

    # Second pass on the same builder: WARN must fire AGAIN. If the
    # dedup set persisted across passes, this would be 0.
    with caplog.at_level(logging.WARNING, logger="app.core.code_graph.graph_builder"):
        builder._parse_documentation_files(
            [str(docs / "spec.pdf")], str(tmp_path),
        )
    second_pass_warnings = [
        r for r in caplog.records
        if "will be ingested via the legacy text-read path" in r.getMessage()
    ]
    assert len(second_pass_warnings) == 1, (
        "Second-pass WARNING was suppressed; "
        "_warned_legacy_vision_extensions should reset per index pass"
    )


def test_vision_extension_with_registered_extractor_does_not_warn(
    tmp_path: Path, caplog,
) -> None:
    """The legacy-fallthrough WARNING must NOT fire when an extractor
    IS registered for the extension. Otherwise the WARNING would be
    permanent operator noise in correctly-configured deployments."""
    import logging

    docs = tmp_path / "docs"
    docs.mkdir()
    (docs / "diag.png").write_bytes(_TINY_PNG)

    from app.core.extractors import ExtractorRegistry
    from app.core.extractors.image import ImageExtractor
    registry = ExtractorRegistry()
    registry.register(ImageExtractor(llm=_StubLLM()))

    builder = EnhancedUnifiedGraphBuilder(extractor_registry=registry)

    with caplog.at_level(logging.WARNING, logger="app.core.code_graph.graph_builder"):
        builder._parse_documentation_files(
            [str(docs / "diag.png")], str(tmp_path),
        )

    fallthrough_warnings = [
        r.getMessage() for r in caplog.records
        if "will be ingested via the legacy text-read path" in r.getMessage()
    ]
    assert fallthrough_warnings == []


def test_vision_eligible_file_without_extractor_is_skipped(
    tmp_path: Path,
) -> None:
    """#173: a vision-eligible file (PNG / PDF / etc.) without a
    registered extractor must NOT appear in the indexed results.

    The previous behaviour was to WARN about the missing extractor
    and then fall through to ``open(path, 'r', encoding='utf-8',
    errors='ignore')``, which turned PNG bytes into 1186 chars of
    binary garbage and indexed that as a "document." The fix in
    graph_builder.py skips the file entirely when the extension is
    in ``KNOWN_VISION_EXTENSIONS`` and no extractor is configured;
    the WARNING still fires (covered by the test above) so
    operators see the configuration gap.
    """
    docs = tmp_path / "docs"
    docs.mkdir()
    (docs / "spec.pdf").write_bytes(b"%PDF-1.4 not a real pdf")
    (docs / "diag.png").write_bytes(_TINY_PNG)
    # Non-vision text file in the same batch — must still go through.
    (docs / "README.md").write_text("# Hello\n")

    builder = EnhancedUnifiedGraphBuilder(extractor_registry=None)
    results = builder._parse_documentation_files(
        [
            str(docs / "spec.pdf"),
            str(docs / "diag.png"),
            str(docs / "README.md"),
        ],
        str(tmp_path),
    )

    # The two vision-eligible files must be absent from results —
    # asserting the indexer does not produce any "document symbol"
    # for them. The previous bug would have placed both here with
    # garbage source_text.
    assert str(docs / "spec.pdf") not in results
    assert str(docs / "diag.png") not in results
    # The plain-text doc still routes through the legacy text-read
    # path — the skip is targeted, not a blanket "no registry =
    # skip everything."
    assert str(docs / "README.md") in results
    assert "Hello" in results[str(docs / "README.md")].symbols[0].source_text


def test_vision_eligible_with_extractor_still_ingests(tmp_path: Path) -> None:
    """Counter-test to #173: with a registered extractor, the PNG
    flows through the extractor and the result IS indexed. Catches a
    regression where someone widens the skip to apply unconditionally
    (which would silently break image ingestion for users with a
    properly configured vision LLM)."""
    docs = tmp_path / "docs"
    docs.mkdir()
    (docs / "diag.png").write_bytes(_TINY_PNG)

    from app.core.extractors import ExtractorRegistry
    from app.core.extractors.image import ImageExtractor
    registry = ExtractorRegistry()
    registry.register(ImageExtractor(llm=_StubLLM()))

    builder = EnhancedUnifiedGraphBuilder(extractor_registry=registry)
    results = builder._parse_documentation_files(
        [str(docs / "diag.png")], str(tmp_path),
    )

    # With an extractor wired up, the PNG IS indexed (using LLM-
    # derived text). Asserts the skip in #173 doesn't fire when an
    # extractor is present.
    assert str(docs / "diag.png") in results
