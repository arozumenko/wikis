"""Tests for wiki_content_writer/writer_tools.py (#237).

Covers: tool function shapes, read_attachment_meta metadata-only guarantee,
and stub behaviour for graph-backed tools.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock

from app.core.wiki_content_writer.writer_tools import (
    AttachmentMeta,
    DocChunk,
    FileContent,
    GrepMatch,
    SymbolCallees,
    SymbolCallers,
    SymbolSignature,
    WriterTools,
)

# ── Fixtures ────────────────────────────────────────────────────────────────


def _make_tools(
    repo_root: str = "/repo",
    storage: Any = None,
    code_graph: Any = None,
    graph_text_index: Any = None,
) -> WriterTools:
    return WriterTools(
        repo_root=repo_root,
        storage=storage,
        code_graph=code_graph,
        graph_text_index=graph_text_index,
    )


# ── WriterTools construction ─────────────────────────────────────────────────


class TestWriterToolsConstruction:
    def test_constructs_with_required_fields_only(self, tmp_path):
        tools = WriterTools(repo_root=str(tmp_path))
        assert tools is not None

    def test_repo_root_stored(self, tmp_path):
        tools = WriterTools(repo_root=str(tmp_path))
        assert tools.repo_root == str(tmp_path)

    def test_optional_deps_default_to_none(self, tmp_path):
        tools = WriterTools(repo_root=str(tmp_path))
        assert tools.storage is None
        assert tools.code_graph is None
        assert tools.graph_text_index is None

    def test_constructs_with_all_deps(self, tmp_path):
        storage = MagicMock()
        graph = MagicMock()
        fts = MagicMock()
        tools = WriterTools(
            repo_root=str(tmp_path),
            storage=storage,
            code_graph=graph,
            graph_text_index=fts,
        )
        assert tools.storage is storage
        assert tools.code_graph is graph
        assert tools.graph_text_index is fts


# ── FileContent return type ──────────────────────────────────────────────────


class TestReadFile:
    def test_returns_file_content_type(self, tmp_path):
        (tmp_path / "main.py").write_text("def hello():\n    pass\n")
        tools = WriterTools(repo_root=str(tmp_path))
        result = tools.read_file("main.py")
        assert isinstance(result, FileContent)

    def test_has_path_and_lines_fields(self, tmp_path):
        (tmp_path / "app.py").write_text("x = 1\ny = 2\n")
        tools = WriterTools(repo_root=str(tmp_path))
        result = tools.read_file("app.py")
        assert result.path == "app.py"
        assert isinstance(result.lines, list)

    def test_reads_file_content(self, tmp_path):
        (tmp_path / "cfg.py").write_text("A = 1\nB = 2\n")
        tools = WriterTools(repo_root=str(tmp_path))
        result = tools.read_file("cfg.py")
        assert "A = 1" in "\n".join(result.lines)

    def test_line_range_filters_lines(self, tmp_path):
        content = "\n".join(f"line{i}" for i in range(1, 11))
        (tmp_path / "big.py").write_text(content)
        tools = WriterTools(repo_root=str(tmp_path))
        result = tools.read_file("big.py", line_range=(3, 5))
        # Should only include lines 3-5 (1-indexed)
        assert len(result.lines) == 3
        assert "line3" in result.lines[0]

    def test_missing_file_returns_empty_content(self, tmp_path):
        tools = WriterTools(repo_root=str(tmp_path))
        result = tools.read_file("does_not_exist.py")
        assert result.lines == []
        assert result.error is not None

    def test_has_total_lines_field(self, tmp_path):
        (tmp_path / "x.py").write_text("a\nb\nc\n")
        tools = WriterTools(repo_root=str(tmp_path))
        result = tools.read_file("x.py")
        assert isinstance(result.total_lines, int)
        assert result.total_lines == 3

    def test_parent_traversal_rejected(self, tmp_path):
        # Create a sibling file outside the simulated repo_root
        outside = tmp_path / "outside.txt"
        outside.write_text("secret")
        repo = tmp_path / "repo"
        repo.mkdir()
        tools = WriterTools(repo_root=str(repo))
        result = tools.read_file("../outside.txt")
        assert result.lines == []
        assert result.error is not None
        assert "escapes" in result.error

    def test_absolute_path_rejected(self, tmp_path):
        outside = tmp_path / "secret.txt"
        outside.write_text("nope")
        repo = tmp_path / "repo"
        repo.mkdir()
        tools = WriterTools(repo_root=str(repo))
        result = tools.read_file(str(outside))
        assert result.lines == []
        assert result.error is not None

    def test_symlink_escape_rejected(self, tmp_path):
        outside = tmp_path / "secret.txt"
        outside.write_text("nope")
        repo = tmp_path / "repo"
        repo.mkdir()
        # Create a symlink inside repo that points outside
        (repo / "link.txt").symlink_to(outside)
        tools = WriterTools(repo_root=str(repo))
        result = tools.read_file("link.txt")
        assert result.lines == []
        assert result.error is not None

    def test_inverted_line_range_returns_error(self, tmp_path):
        (tmp_path / "x.py").write_text("a\nb\nc\nd\ne\n")
        tools = WriterTools(repo_root=str(tmp_path))
        result = tools.read_file("x.py", line_range=(5, 2))
        assert result.lines == []
        assert result.error is not None
        assert "inverted" in result.error


# ── SymbolSignature return type ──────────────────────────────────────────────


class TestGetSignature:
    def test_returns_symbol_signature_type(self, tmp_path):
        tools = WriterTools(repo_root=str(tmp_path))
        result = tools.get_signature("SomeClass")
        assert isinstance(result, SymbolSignature)

    def test_has_required_fields(self, tmp_path):
        tools = WriterTools(repo_root=str(tmp_path))
        result = tools.get_signature("SomeClass")
        assert hasattr(result, "symbol")
        assert hasattr(result, "signature")
        assert hasattr(result, "file_path")
        assert hasattr(result, "layer")
        assert hasattr(result, "docstring")
        assert hasattr(result, "found")

    def test_not_found_when_no_graph(self, tmp_path):
        tools = WriterTools(repo_root=str(tmp_path))
        result = tools.get_signature("Nonexistent")
        assert result.found is False

    def test_symbol_echoed(self, tmp_path):
        tools = WriterTools(repo_root=str(tmp_path))
        result = tools.get_signature("MyClass")
        assert result.symbol == "MyClass"

    def test_stub_returns_found_false_even_with_graph(self, tmp_path):
        # Stub behaviour: until #243 wires GraphQueryService, the
        # presence of a code_graph must not flip `found` to True.
        graph = MagicMock()
        tools = WriterTools(repo_root=str(tmp_path), code_graph=graph)
        result = tools.get_signature("MyClass")
        assert result.found is False


# ── SymbolCallers / SymbolCallees return types ───────────────────────────────


class TestGetCallers:
    def test_returns_symbol_callers_type(self, tmp_path):
        tools = WriterTools(repo_root=str(tmp_path))
        result = tools.get_callers("some_func")
        assert isinstance(result, SymbolCallers)

    def test_has_symbol_and_callers_fields(self, tmp_path):
        tools = WriterTools(repo_root=str(tmp_path))
        result = tools.get_callers("fn")
        assert hasattr(result, "symbol")
        assert hasattr(result, "callers")
        assert isinstance(result.callers, list)

    def test_empty_callers_when_no_graph(self, tmp_path):
        tools = WriterTools(repo_root=str(tmp_path))
        result = tools.get_callers("fn")
        assert result.callers == []

    def test_has_found_field_defaulting_false(self, tmp_path):
        # `found` discriminates "no callers" from "backend not wired".
        tools = WriterTools(repo_root=str(tmp_path))
        result = tools.get_callers("fn")
        assert hasattr(result, "found")
        assert result.found is False


class TestGetCallees:
    def test_returns_symbol_callees_type(self, tmp_path):
        tools = WriterTools(repo_root=str(tmp_path))
        result = tools.get_callees("some_func")
        assert isinstance(result, SymbolCallees)

    def test_has_symbol_and_callees_fields(self, tmp_path):
        tools = WriterTools(repo_root=str(tmp_path))
        result = tools.get_callees("fn")
        assert hasattr(result, "symbol")
        assert hasattr(result, "callees")
        assert isinstance(result.callees, list)

    def test_empty_callees_when_no_graph(self, tmp_path):
        tools = WriterTools(repo_root=str(tmp_path))
        result = tools.get_callees("fn")
        assert result.callees == []

    def test_has_found_field_defaulting_false(self, tmp_path):
        # Symmetric with TestGetCallers — `found` discriminates "no
        # callees" from "backend not wired".
        tools = WriterTools(repo_root=str(tmp_path))
        result = tools.get_callees("fn")
        assert hasattr(result, "found")
        assert result.found is False


# ── GrepMatch return type ────────────────────────────────────────────────────


class TestGrep:
    def test_returns_list_of_grep_matches(self, tmp_path):
        tools = WriterTools(repo_root=str(tmp_path))
        result = tools.grep("pattern")
        assert isinstance(result, list)

    def test_each_result_is_grep_match(self, tmp_path):
        (tmp_path / "a.py").write_text("def pattern_func(): pass\n")
        tools = WriterTools(repo_root=str(tmp_path))
        result = tools.grep("pattern_func")
        for item in result:
            assert isinstance(item, GrepMatch)

    def test_grep_match_has_required_fields(self, tmp_path):
        # Even if empty, the type is correct — verify by constructing one
        match = GrepMatch(file_path="x.py", line_number=1, line_text="x = 1", score=1.0)
        assert match.file_path == "x.py"
        assert match.line_number == 1
        assert match.line_text == "x = 1"
        assert match.score == 1.0

    def test_grep_reads_search_score_metadata_key(self, tmp_path):
        # GraphTextIndex / StorageTextIndex populate metadata["search_score"].
        # Verify grep picks up that key (with fallback to legacy "score").
        class _FakeDoc:
            def __init__(self, content, meta):
                self.page_content = content
                self.metadata = meta

        class _FakeIndex:
            def __init__(self, docs):
                self._docs = docs

            def search(self, pattern, k=20):
                return self._docs

        docs = [
            _FakeDoc("hit one", {"rel_path": "a.py", "start_line": 5, "search_score": 0.92}),
            _FakeDoc("hit two", {"rel_path": "b.py", "start_line": 7, "score": 0.41}),
            _FakeDoc("hit three", {"rel_path": "c.py", "start_line": 9}),
        ]
        tools = WriterTools(repo_root=str(tmp_path), graph_text_index=_FakeIndex(docs))
        result = tools.grep("anything")
        assert len(result) == 3
        assert result[0].score == 0.92
        assert result[1].score == 0.41
        assert result[2].score == 0.0


# ── DocChunk return type ─────────────────────────────────────────────────────


class TestListDocChunks:
    def test_returns_list(self, tmp_path):
        tools = WriterTools(repo_root=str(tmp_path))
        result = tools.list_doc_chunks("docs/README.md")
        assert isinstance(result, list)

    def test_each_result_is_doc_chunk(self, tmp_path):
        tools = WriterTools(repo_root=str(tmp_path))
        result = tools.list_doc_chunks("docs/guide.md")
        for item in result:
            assert isinstance(item, DocChunk)

    def test_doc_chunk_has_required_fields(self):
        chunk = DocChunk(
            doc_path="docs/guide.md",
            chunk_index=0,
            heading="Overview",
            text="Some content",
        )
        assert chunk.doc_path == "docs/guide.md"
        assert chunk.chunk_index == 0
        assert chunk.heading == "Overview"
        assert chunk.text == "Some content"


# ── AttachmentMeta — never returns content ──────────────────────────────────


class TestReadAttachmentMeta:
    """read_attachment_meta must return only metadata — never file content."""

    def test_returns_attachment_meta_type(self, tmp_path):
        tools = WriterTools(repo_root=str(tmp_path))
        result = tools.read_attachment_meta("diagram.png")
        assert isinstance(result, AttachmentMeta)

    def test_has_name_mime_parent_fields(self, tmp_path):
        tools = WriterTools(repo_root=str(tmp_path))
        result = tools.read_attachment_meta("chart.png")
        assert hasattr(result, "name")
        assert hasattr(result, "mime")
        assert hasattr(result, "parent")

    def test_name_echoed(self, tmp_path):
        tools = WriterTools(repo_root=str(tmp_path))
        result = tools.read_attachment_meta("logo.svg")
        assert result.name == "logo.svg"

    def test_no_content_field(self, tmp_path):
        tools = WriterTools(repo_root=str(tmp_path))
        result = tools.read_attachment_meta("file.pdf")
        # The result type must NOT have a 'content' attribute
        assert not hasattr(result, "content")
        assert not hasattr(result, "data")
        assert not hasattr(result, "bytes")
        assert not hasattr(result, "raw")

    def test_returns_none_mime_when_unknown(self, tmp_path):
        tools = WriterTools(repo_root=str(tmp_path))
        result = tools.read_attachment_meta("file.unknownext12345")
        # mime should be None or a string, never content
        assert result.mime is None or isinstance(result.mime, str)

    def test_png_mime_detected(self, tmp_path):
        tools = WriterTools(repo_root=str(tmp_path))
        result = tools.read_attachment_meta("image.png")
        assert result.mime == "image/png"

    def test_pdf_mime_detected(self, tmp_path):
        tools = WriterTools(repo_root=str(tmp_path))
        result = tools.read_attachment_meta("document.pdf")
        assert result.mime == "application/pdf"

    def test_svg_mime_detected(self, tmp_path):
        tools = WriterTools(repo_root=str(tmp_path))
        result = tools.read_attachment_meta("diagram.svg")
        assert result.mime == "image/svg+xml"
