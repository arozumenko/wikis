"""Writer tool surface for the wiki content writer deepagent (#237).

Each tool is a method on ``WriterTools`` and returns a typed dataclass so
callers (tests, agent, future wiring) can rely on a stable contract.

Tools provided
--------------
read_file(path, line_range?)      -- file content, optionally sliced
get_signature(symbol)             -- symbol signature + layer from code graph
get_callers(symbol)               -- who calls *symbol* (callers graph)
get_callees(symbol)               -- what *symbol* calls (callees graph)
grep(pattern)                     -- FTS5-backed search over the code graph
list_doc_chunks(doc_path)         -- doc chunks for a documentation file
read_attachment_meta(name)        -- {name, mime, parent} — NEVER content

Stubs and TODOs
---------------
``get_callers``, ``get_callees``, ``get_signature``, ``list_doc_chunks``, and
``grep`` delegate to the code graph / FTS index and storage respectively.
All signatures are final; the backends are wired in #243 (epic #227 wiring &
integration tests).
"""

from __future__ import annotations

import logging
import mimetypes
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any


def _safe_join(root: str, rel_path: str) -> str | None:
    """Join *rel_path* under *root*; return None if it would escape the root.

    Resolves symlinks and ``..`` so absolute paths and traversal attempts are
    rejected before any file IO happens. Mirrors the pattern in
    ``deep_research/research_tools.py``.
    """
    try:
        full = os.path.realpath(os.path.join(root, rel_path))
        root_real = os.path.realpath(root)
        if not full.startswith(root_real + os.sep) and full != root_real:
            return None
        return full
    except (ValueError, OSError):
        return None


logger = logging.getLogger(__name__)


# ── Return types ─────────────────────────────────────────────────────────────


@dataclass
class FileContent:
    """Result of ``read_file``."""

    path: str
    lines: list[str]
    total_lines: int
    error: str | None = None


@dataclass
class SymbolSignature:
    """Result of ``get_signature``."""

    symbol: str
    signature: str
    file_path: str
    layer: str
    docstring: str
    found: bool


@dataclass
class CallerEntry:
    """A single caller symbol."""

    symbol: str
    file_path: str
    layer: str


@dataclass
class SymbolCallers:
    """Result of ``get_callers``.

    ``found`` discriminates between "the symbol exists but has no
    callers" and "the backend isn't wired / lookup failed" — important
    signal for #238 prompt rules so the writer can distinguish "no
    callers" from "no answer."
    """

    symbol: str
    callers: list[CallerEntry]
    found: bool = False


@dataclass
class CalleeEntry:
    """A single callee symbol."""

    symbol: str
    file_path: str
    layer: str


@dataclass
class SymbolCallees:
    """Result of ``get_callees``.

    See ``SymbolCallers`` for the ``found`` rationale.
    """

    symbol: str
    callees: list[CalleeEntry]
    found: bool = False


@dataclass
class GrepMatch:
    """A single FTS5 hit from ``grep``."""

    file_path: str
    line_number: int
    line_text: str
    score: float


@dataclass
class DocChunk:
    """A chunk of a documentation file from ``list_doc_chunks``."""

    doc_path: str
    chunk_index: int
    heading: str
    text: str


@dataclass
class AttachmentMeta:
    """Metadata-only result of ``read_attachment_meta``.

    This dataclass intentionally omits all content fields (``content``,
    ``data``, ``bytes``, ``raw``) — the agent must never receive binary
    attachment payloads.
    """

    name: str
    mime: str | None
    parent: str | None


# ── MIME helpers ─────────────────────────────────────────────────────────────

# Supplement stdlib mimetypes with common types it may not detect.
_EXTRA_MIME: dict[str, str] = {
    ".svg": "image/svg+xml",
    ".md": "text/markdown",
    ".csv": "text/csv",
    ".jsonl": "application/jsonlines",
}


def _guess_mime(name: str) -> str | None:
    suffix = Path(name).suffix.lower()
    if suffix in _EXTRA_MIME:
        return _EXTRA_MIME[suffix]
    mime, _ = mimetypes.guess_type(name)
    return mime


# ── WriterTools ───────────────────────────────────────────────────────────────


class WriterTools:
    """Tool surface used by ``WikiContentWriter`` during page generation.

    Parameters
    ----------
    repo_root : str
        Absolute path to the checked-out repository root.
    storage : WikiStorageProtocol | None
        Optional opened storage instance for FTS and doc-chunk queries.
        Wired in #243.
    code_graph : nx.DiGraph | None
        Optional NetworkX graph for signature/caller/callee resolution.
        Wired in #243.
    graph_text_index : GraphTextIndex | StorageTextIndex | None
        Optional FTS index for ``grep``.
        Wired in #243.
    """

    def __init__(
        self,
        repo_root: str,
        storage: Any | None = None,
        code_graph: Any | None = None,
        graph_text_index: Any | None = None,
    ) -> None:
        self.repo_root = repo_root
        self.storage = storage
        self.code_graph = code_graph
        self.graph_text_index = graph_text_index

    # ── read_file ──────────────────────────────────────────────────────────

    def read_file(
        self,
        path: str,
        line_range: tuple[int, int] | None = None,
    ) -> FileContent:
        """Return lines from *path* inside the repository.

        Parameters
        ----------
        path : str
            Relative path from ``repo_root``.
        line_range : (start, end) | None
            1-indexed inclusive line range. ``None`` returns all lines.
        """
        safe_path = _safe_join(self.repo_root, path)
        if safe_path is None:
            return FileContent(
                path=path,
                lines=[],
                total_lines=0,
                error=f"path escapes repo_root: {path!r}",
            )
        try:
            text = Path(safe_path).read_text(encoding="utf-8", errors="replace")
        except (OSError, PermissionError) as exc:
            return FileContent(path=path, lines=[], total_lines=0, error=str(exc))

        all_lines = text.splitlines()
        total = len(all_lines)

        if line_range is not None:
            start, end = line_range
            if start > end:
                return FileContent(
                    path=path,
                    lines=[],
                    total_lines=total,
                    error=f"inverted line_range: start={start} > end={end}",
                )
            # Convert 1-indexed inclusive to 0-indexed slice
            sliced = all_lines[max(0, start - 1) : end]
        else:
            sliced = all_lines

        return FileContent(path=path, lines=sliced, total_lines=total)

    # ── get_signature ─────────────────────────────────────────────────────

    def get_signature(self, symbol: str) -> SymbolSignature:
        """Return the signature, layer, and docstring for *symbol*.

        Currently a stub: returns ``found=False`` regardless of whether
        ``code_graph`` is set. Wiring to ``GraphQueryService`` happens in
        #243 alongside the other graph-backed tools.
        """
        # TODO #243: wire to code_graph via GraphQueryService.resolve_symbol
        return SymbolSignature(
            symbol=symbol,
            signature="",
            file_path="",
            layer="",
            docstring="",
            found=False,
        )

    # ── get_callers ───────────────────────────────────────────────────────

    def get_callers(self, symbol: str) -> SymbolCallers:
        """Return symbols that call *symbol*.

        # TODO #243: wire to code_graph edges via GraphQueryService.get_relationships
        """
        if self.code_graph is None:
            return SymbolCallers(symbol=symbol, callers=[])

        # TODO #243: query incoming edges of type 'calls' / 'imports'
        return SymbolCallers(symbol=symbol, callers=[])

    # ── get_callees ───────────────────────────────────────────────────────

    def get_callees(self, symbol: str) -> SymbolCallees:
        """Return symbols that *symbol* calls.

        # TODO #243: wire to code_graph edges via GraphQueryService.get_relationships
        """
        if self.code_graph is None:
            return SymbolCallees(symbol=symbol, callees=[])

        # TODO #243: query outgoing edges of type 'calls' / 'imports'
        return SymbolCallees(symbol=symbol, callees=[])

    # ── grep ──────────────────────────────────────────────────────────────

    def grep(self, pattern: str, k: int = 20) -> list[GrepMatch]:
        """FTS5-backed search returning up to *k* matches.

        Falls back to empty list when no FTS index is available.

        # TODO #243: wire to graph_text_index.search(pattern, k=k) or
        #           storage.search_fts(pattern, k=k)
        """
        if self.graph_text_index is None and self.storage is None:
            return []

        index = self.graph_text_index
        if index is not None:
            try:
                docs = index.search(pattern, k=k)
            except Exception:
                logger.debug("grep FTS search failed for %r", pattern, exc_info=True)
                return []
            matches: list[GrepMatch] = []
            for doc in docs:
                meta = doc.metadata if hasattr(doc, "metadata") else {}
                matches.append(
                    GrepMatch(
                        file_path=meta.get("rel_path", meta.get("source", "")),
                        line_number=int(meta.get("start_line", 0) or 0),
                        line_text=(doc.page_content or "")[:200],
                        score=float(meta.get("search_score", meta.get("score", 0.0))),
                    )
                )
            return matches

        # TODO #243: storage.search_fts path
        return []

    # ── list_doc_chunks ───────────────────────────────────────────────────

    def list_doc_chunks(self, doc_path: str) -> list[DocChunk]:
        """Return doc chunks for the given documentation file path.

        # TODO #243: wire to storage.get_nodes_by_path_prefix(doc_path)
        """
        if self.storage is None:
            return []

        # TODO #243: fetch nodes for doc_path, split by heading/chunk boundary
        return []

    # ── read_attachment_meta ──────────────────────────────────────────────

    def read_attachment_meta(self, name: str) -> AttachmentMeta:
        """Return metadata for an attachment — name, MIME type, and parent doc.

        This tool deliberately returns ONLY metadata and NEVER the binary
        content of the attachment.  The agent must not receive file bytes.

        Parameters
        ----------
        name : str
            Attachment filename (e.g. ``"diagram.png"``).
        """
        mime = _guess_mime(name)

        # TODO #243: look up parent doc path from storage attachment registry
        parent: str | None = None
        if self.storage is not None:
            # TODO #243: storage.find_attachment_parent(name)
            pass

        return AttachmentMeta(name=name, mime=mime, parent=parent)
