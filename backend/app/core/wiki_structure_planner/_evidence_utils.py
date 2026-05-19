"""Shared helper utilities for evidence pack builders (#253).

Consolidates helpers that were previously duplicated across the four pack
modules (evidence.py, evidence_md.py, evidence_confluence.py, evidence_jira.py)
into a single authoritative location.

Public API
----------
safe_join(root, rel_path)                      -> str | None
mask_code_fences(text)                         -> str
extract_toc_from_text(md_text, max_entries=20) -> list[str]
extract_first_paragraph(chunks, cap=200)        -> str
collect_attachments(chunks)                     -> list[str]

Design notes
------------
* ``safe_join`` is the canonical security primitive (originally evidence.py:60).
  All path IO in the pack modules must go through this function — it resolves
  symlinks and ``..`` so traversal attempts are rejected before any file IO.
* ``extract_toc_from_text`` was previously ``_extract_toc`` (private) in
  evidence_md.py and evidence_confluence.py.  The ``max_entries`` parameter
  replaces the per-module ``_MAX_TOC_ENTRIES_PER_*`` constants so each caller
  can pass its own value (default 20 matches both originals).
* ``extract_first_paragraph`` and ``collect_attachments`` were identical in
  evidence_md.py and evidence_confluence.py; the ``cap`` parameter replaces
  the per-module ``_FIRST_PARA_CHAR_CAP`` constant (default 200 matches both).
"""

from __future__ import annotations

import os
import re

from ..parsers.markdown_chunker import Chunk

# ── TOC extraction regexes ───────────────────────────────────────────────────
# Match H1/H2 ATX headings with optional trailing hashes (``## Heading ##``).
# re.MULTILINE so ``^`` matches the start of each line.
TOC_HEADING_RE = re.compile(r"^(#{1,2})\s+(.*?)(?:\s+#+)?\s*$", re.MULTILINE)

# Match fenced code blocks using either ``` or ~~~.  Lazy ``.*?`` prevents
# one open fence consuming the rest of the document.  ``\Z`` handles an
# unclosed fence at EOF — replaces the entire tail so headings inside are
# still masked correctly.
TOC_FENCE_RE = re.compile(r"```[\s\S]*?(?:```|\Z)|~~~[\s\S]*?(?:~~~|\Z)")


# ── Path safety ───────────────────────────────────────────────────────────────


def safe_join(root: str, rel_path: str) -> str | None:
    """Join *rel_path* under *root*; return None if it would escape the root.

    Resolves symlinks and ``..`` so absolute paths and traversal attempts are
    rejected before any file IO happens.  This is the canonical security
    primitive for all evidence pack IO (originally evidence.py:60; replicated
    in evidence_md.py, evidence_confluence.py, and evidence_jira.py).

    Parameters
    ----------
    root : str
        Absolute path to the repository root directory.
    rel_path : str
        A repo-relative path supplied by an ArtifactInfo or cluster.

    Returns
    -------
    str
        Resolved absolute path inside *root*.
    None
        When *rel_path* resolves to a path outside *root* (traversal, absolute
        path, or symlink escape).
    """
    try:
        full = os.path.realpath(os.path.join(root, rel_path))
        root_real = os.path.realpath(root)
        if not full.startswith(root_real + os.sep) and full != root_real:
            return None
        return full
    except (ValueError, OSError):
        return None


# ── Code-fence masking ────────────────────────────────────────────────────────


def mask_code_fences(text: str) -> str:
    """Replace fenced-code-block contents with spaces, preserving offsets.

    Returned string has the same length as the input; the leading ``` / ~~~,
    the body, and the trailing fence are all replaced with spaces. Lets
    callers run heading or link regexes on the masked text without picking up
    matches that live inside code blocks, while leaving offsets valid for
    splicing back into the original.

    Handles unclosed fences via the ``\\Z`` alternative — important for
    LLM-truncated input.
    """

    def _blank(m: re.Match[str]) -> str:
        return " " * (m.end() - m.start())

    return TOC_FENCE_RE.sub(_blank, text)


# ── TOC extraction ────────────────────────────────────────────────────────────


def extract_toc_from_text(md_text: str, max_entries: int = 20) -> list[str]:
    """Return H1/H2 heading names from raw markdown in document order.

    Scans the raw text directly rather than relying on ``Chunk`` output so
    structural-only docs (heading-only with no body content, which the chunker
    drops per the "skip empty sections" rule from #231 / Rio's review of #248)
    still surface their TOC.

    Headings are NOT deduplicated by name — a doc with two ``## Overview``
    sections under different H1s legitimately has both in its TOC, and losing
    one would distort the structure the planner sees.

    H3+ are excluded by design.  Fenced code blocks are masked so a
    ``# install`` line inside ```bash isn't picked up.

    Parameters
    ----------
    md_text : str
        Raw markdown text.
    max_entries : int
        Maximum number of TOC entries to return.  Defaults to 20 (matches the
        original ``_MAX_TOC_ENTRIES_PER_DOC`` / ``_MAX_TOC_ENTRIES_PER_PAGE``).

    Returns
    -------
    list[str]
        Heading names in document order, H1 and H2 only, at most *max_entries*.
    """

    masked = mask_code_fences(md_text)
    toc: list[str] = []
    for m in TOC_HEADING_RE.finditer(masked):
        heading = m.group(2).strip()
        if heading:
            toc.append(heading)
    return toc[:max_entries]


# ── Chunk helpers ─────────────────────────────────────────────────────────────


def extract_first_paragraph(chunks: list[Chunk], cap: int = 200) -> str:
    """Return first non-empty body text from *chunks*, capped at *cap* chars.

    Prefers preamble chunk (``heading_path=[]``) if present and non-empty;
    falls back to the first heading section's body.

    Parameters
    ----------
    chunks : list[Chunk]
        Chunks produced by ``chunk_markdown``.
    cap : int
        Character cap on the returned string.  Defaults to 200 (matches the
        original ``_FIRST_PARA_CHAR_CAP`` in evidence_md.py and
        evidence_confluence.py).

    Returns
    -------
    str
        First non-empty body, stripped and capped, or ``""`` if all chunks are
        empty.
    """
    for chunk in chunks:
        stripped = chunk.body.strip()
        if stripped:
            return stripped[:cap]
    return ""


def collect_attachments(chunks: list[Chunk]) -> list[str]:
    """Return a sorted, deduplicated list of attachment filenames from all chunks.

    Attachment content is never read — only the filenames/paths extracted by
    ``chunk_markdown``'s ``[[attachment: …]]`` pass-through are included.

    Parameters
    ----------
    chunks : list[Chunk]
        Chunks produced by ``chunk_markdown``.

    Returns
    -------
    list[str]
        Sorted, deduplicated list of attachment filenames.
    """
    seen: set[str] = set()
    result: list[str] = []
    for chunk in chunks:
        for name in chunk.attachments:
            if name not in seen:
                seen.add(name)
                result.append(name)
    return sorted(result)
