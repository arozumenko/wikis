"""Markdown chunker — heading-based, attachment placeholders (#231).

Splits source markdown documents into addressable ``Chunk`` objects for use
by the evidence pack (#232, #233) and the writer's ``list_doc_chunks`` tool
(#237).

Splitting rules
---------------
* Primary splits on H1 and H2 boundaries.
* H3 fallback: if an H1/H2 chunk would exceed ~2000 tokens, H3 headings
  within that section are promoted to split boundaries.
* Preamble text before the first heading becomes a chunk with
  ``heading_path = []``.

Attachment handling
-------------------
Markdown image syntax ``![alt](filename)`` is rewritten to the canonical
placeholder ``[[attachment: filename]]`` so downstream consumers (link
resolver #241, evidence pack #233) see a uniform format. Pre-existing
``[[attachment: filename]]`` placeholders pass through unchanged.
Content of attachments is never read.

Input is pure in-memory text; no file I/O is performed.
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, field

# ── Token budget ─────────────────────────────────────────────────────────────

_H3_FALLBACK_TOKENS = 2000

# ── Regexes ──────────────────────────────────────────────────────────────────

# H1/H2/H3 ATX heading lines (with optional trailing #).
_HEADING_RE = re.compile(r"^(#{1,3})\s+(.*?)(?:\s+#+)?\s*$", re.MULTILINE)

# Markdown image syntax: ![alt text](path/to/file.ext)
_IMAGE_RE = re.compile(r"!\[([^\]]*)\]\(([^)\s]+)(?:\s+\"[^\"]*\")?\)")

# Pre-existing attachment placeholder (already canonical; pass through).
_ATTACHMENT_PLACEHOLDER_RE = re.compile(r"\[\[attachment:\s*([^\]]+)\]\]")

# Fenced code blocks — masked with offset-preserving blanks before heading
# scan so shell/python `# install` lines inside fences aren't taken as H1
# split points.
_FENCED_BLOCK_RE = re.compile(r"```[\s\S]*?```|~~~[\s\S]*?~~~")


def _mask_fenced_code(text: str) -> str:
    """Replace fenced-code-block characters with spaces, preserving offsets.

    Heading detection then ignores text inside code fences while every
    other regex / slice can still operate on the same offset space.
    """

    def _blank(m: re.Match[str]) -> str:
        return " " * (m.end() - m.start())

    return _FENCED_BLOCK_RE.sub(_blank, text)


# ── Public types ─────────────────────────────────────────────────────────────


@dataclass
class Chunk:
    """A single addressable segment of a markdown document.

    Attributes:
        chunk_id: Stable SHA-256 derived from ``doc_path`` + ``heading_path``
            + chunk index within those headings. Reproducible across runs for
            the same document.
        doc_path: Caller-supplied path identifying the source document.
        heading_path: Ordered list of headings leading to this chunk, e.g.
            ``["Auth", "OAuth Flow"]``. Empty for preamble chunks.
        body: The markdown body text with image refs rewritten to
            ``[[attachment: …]]`` placeholders.
        token_count: Rough estimate (whitespace split) for budget tracking.
        frontmatter: Caller-supplied metadata dict (title, labels, parents,
            original_url, space_key). The chunker neither parses nor mutates
            it — the same reference is stored on every chunk.
        attachments: Set of attachment filenames/paths referenced in this
            chunk (from both image refs and pre-existing placeholders).
    """

    chunk_id: str
    doc_path: str
    heading_path: list[str]
    body: str
    token_count: int
    frontmatter: dict | None
    attachments: set[str] = field(default_factory=set)


# ── Internal helpers ─────────────────────────────────────────────────────────


def _token_count(text: str) -> int:
    """Rough token estimate via whitespace split."""
    return len(text.split())


def _make_chunk_id(doc_path: str, heading_path: list[str], index: int) -> str:
    """Stable, reproducible chunk identifier."""
    key = f"{doc_path}|{'|'.join(heading_path)}|{index}"
    return hashlib.sha256(key.encode()).hexdigest()[:16]


def _normalise_attachments(body: str) -> tuple[str, set[str]]:
    """Rewrite ``![alt](path)`` to ``[[attachment: path]]`` and collect names.

    Pre-existing ``[[attachment: …]]`` placeholders are left in place and also
    collected. Returns the transformed body and the set of attachment paths.
    """
    attachments: list[str] = []

    # Collect pre-existing placeholders first so they are not double-processed.
    existing_spans: list[tuple[int, int]] = []
    existing_names: list[str] = []
    for m in _ATTACHMENT_PLACEHOLDER_RE.finditer(body):
        existing_spans.append((m.start(), m.end()))
        existing_names.append(m.group(1).strip())

    def _in_existing(start: int) -> bool:
        return any(s <= start < e for s, e in existing_spans)

    parts: list[tuple[int, int, str]] = []  # (start, end, replacement)
    for m in _IMAGE_RE.finditer(body):
        if _in_existing(m.start()):
            continue
        path = m.group(2)
        parts.append((m.start(), m.end(), f"[[attachment: {path}]]"))
        attachments.append(path)

    # Apply replacements in reverse order to keep offsets valid.
    result = body
    for start, end, replacement in sorted(parts, key=lambda t: t[0], reverse=True):
        result = result[:start] + replacement + result[end:]

    attachments += existing_names
    return result, set(attachments)


def _build_chunk(
    doc_path: str,
    heading_path: list[str],
    raw_body: str,
    frontmatter: dict | None,
    index: int,
) -> Chunk:
    body, attachments = _normalise_attachments(raw_body)
    return Chunk(
        chunk_id=_make_chunk_id(doc_path, heading_path, index),
        doc_path=doc_path,
        heading_path=list(heading_path),
        body=body,
        token_count=_token_count(body),
        frontmatter=frontmatter,
        attachments=attachments,
    )


def _split_with_h3_fallback(
    doc_path: str,
    heading_path: list[str],
    raw_body: str,
    frontmatter: dict | None,
    index_start: int,
) -> tuple[list[Chunk], int]:
    """Produce chunks for a single H1/H2 section.

    If the section body exceeds the token budget, H3 headings within it are
    used as sub-split points. Returns (chunks, next_index).
    """
    chunks: list[Chunk] = []
    idx = index_start

    # Fast path: no H3s or fits within budget.
    if _token_count(raw_body) <= _H3_FALLBACK_TOKENS:
        chunks.append(_build_chunk(doc_path, heading_path, raw_body, frontmatter, idx))
        return chunks, idx + 1

    # Find H3 positions in the body.
    h3_positions = [(m.start(), m.group(2).strip()) for m in _HEADING_RE.finditer(raw_body) if len(m.group(1)) == 3]

    if not h3_positions:
        # No H3s to fall back on; emit as one oversized chunk.
        chunks.append(_build_chunk(doc_path, heading_path, raw_body, frontmatter, idx))
        return chunks, idx + 1

    # Split on H3 boundaries.
    boundaries = [pos for pos, _ in h3_positions] + [len(raw_body)]

    # Preamble before the first H3.
    preamble = raw_body[: h3_positions[0][0]]
    if preamble.strip():
        chunks.append(_build_chunk(doc_path, heading_path, preamble, frontmatter, idx))
        idx += 1

    for i, (start, h3_name) in enumerate(h3_positions):
        end = boundaries[i + 1]
        # Strip the H3 line itself from the body text.
        section_text = raw_body[start:end]
        section_text = _HEADING_RE.sub("", section_text, count=1).lstrip("\n")
        child_path = list(heading_path) + [h3_name]
        chunks.append(_build_chunk(doc_path, child_path, section_text, frontmatter, idx))
        idx += 1

    return chunks, idx


# ── Public API ────────────────────────────────────────────────────────────────


def chunk_markdown(
    md_text: str,
    doc_path: str,
    frontmatter: dict | None = None,
) -> list[Chunk]:
    """Split a markdown document into heading-based chunks.

    Args:
        md_text: Raw markdown content (in-memory; no file I/O).
        doc_path: Caller-supplied identifier for the source document. Used
            in chunk IDs and stored on every chunk.
        frontmatter: Optional metadata dict forwarded unchanged to every
            chunk. Recognised keys: ``title``, ``labels``, ``parents``,
            ``original_url``, ``space_key``.

    Returns:
        List of :class:`Chunk` objects in document order. Returns a single
        empty-body chunk for whitespace-only / empty input so downstream
        consumers can still address the document. Heading-only sections
        with no body are skipped to avoid zero-token chunks.
    """
    if not md_text or not md_text.strip():
        return [
            Chunk(
                chunk_id=_make_chunk_id(doc_path, [], 0),
                doc_path=doc_path,
                heading_path=[],
                body=md_text,
                token_count=0,
                frontmatter=frontmatter,
                attachments=set(),
            )
        ]

    # Scan headings against a code-fence-masked copy so `# install` inside
    # a fenced block doesn't split. Offsets stay valid against md_text.
    masked = _mask_fenced_code(md_text)

    sections: list[tuple[int, list[str]]] = []  # (char_start, heading_path)
    h1_active: str | None = None

    for m in _HEADING_RE.finditer(masked):
        level = len(m.group(1))
        name = m.group(2).strip()
        if level == 1:
            h1_active = name
            sections.append((m.start(), [name]))
        elif level == 2:
            path = [h1_active, name] if h1_active else [name]
            sections.append((m.start(), path))
        # H3 is intentionally skipped here; handled inside fallback.

    chunks: list[Chunk] = []
    idx = 0

    if not sections:
        # No H1/H2 headings at all → single chunk.
        chunks.append(_build_chunk(doc_path, [], md_text, frontmatter, idx))
        return chunks

    # Preamble before the first heading.
    preamble = md_text[: sections[0][0]]
    if preamble.strip():
        chunks.append(_build_chunk(doc_path, [], preamble, frontmatter, idx))
        idx += 1

    # Each H1/H2 section.
    ends = [pos for pos, _ in sections[1:]] + [len(md_text)]
    for (start, heading_path), end in zip(sections, ends, strict=True):
        # Strip the heading line itself before passing body to helper.
        section_text = md_text[start:end]
        section_text = _HEADING_RE.sub("", section_text, count=1).lstrip("\n")
        # Skip whitespace-only sections so empty headings don't emit
        # zero-token chunks that downstream consumers must filter.
        if not section_text.strip():
            continue

        new_chunks, idx = _split_with_h3_fallback(doc_path, heading_path, section_text, frontmatter, idx)
        chunks.extend(new_chunks)

    return chunks
