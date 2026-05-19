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
parse_frontmatter(text, *, inline_lists, strip_quotes) -> tuple[dict, str]

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
* ``parse_frontmatter`` consolidates the three private ``_parse_frontmatter``
  implementations from evidence_md.py, evidence_confluence.py, and
  evidence_jira.py (#261). Feature flags preserve each caller's prior
  behaviour — see the function docstring for the per-caller flag table.
"""

from __future__ import annotations

import os
import re

from ..parsers.markdown_chunker import Chunk

# ── Frontmatter regex constants ───────────────────────────────────────────────

# Opening/closing ``---`` on its own line (optional trailing whitespace).
_FM_OPEN_RE = re.compile(r"^---\s*$")
# ``key: value`` pair — key starts with letter or underscore.
_FM_KEY_VALUE_RE = re.compile(r"^([A-Za-z_][A-Za-z0-9_]*)\s*:\s*(.*)$")
# Multi-line list item: any positive indentation followed by ``- text``.
_FM_LIST_ITEM_RE = re.compile(r"^\s+-\s+(.+)$")

# ── TOC extraction regexes ───────────────────────────────────────────────────
# Match H1/H2 ATX headings with optional trailing hashes (``## Heading ##``).
# re.MULTILINE so ``^`` matches the start of each line.
TOC_HEADING_RE = re.compile(r"^(#{1,2})\s+(.*?)(?:\s+#+)?\s*$", re.MULTILINE)

# Match fenced code blocks using either ``` or ~~~.  Lazy ``.*?`` prevents
# one open fence consuming the rest of the document.  ``\Z`` handles an
# unclosed fence at EOF — replaces the entire tail so headings inside are
# still masked correctly.
TOC_FENCE_RE = re.compile(r"```[\s\S]*?(?:```|\Z)|~~~[\s\S]*?(?:~~~|\Z)")


# ── Frontmatter helpers ───────────────────────────────────────────────────────


def _strip_yaml_quotes(value: str) -> str:
    """Strip optional surrounding single/double quotes from a YAML scalar.

    Used when ``strip_quotes=True`` is passed to ``parse_frontmatter``.
    Jira frontmatter generators frequently quote strings containing special
    characters; without stripping, the quotes leak into titles/labels/etc.
    """
    if len(value) >= 2 and value[0] == value[-1] and value[0] in ("'", '"'):
        return value[1:-1]
    return value


def _parse_inline_list(value: str) -> list[str] | None:
    """Parse an inline YAML list like ``[a, "b", 'c']``.

    Returns ``None`` if *value* is not wrapped in ``[…]``.
    Called only when ``inline_lists=True`` is passed to ``parse_frontmatter``.

    Uses a small state machine so that commas inside quoted strings are treated
    as literal characters rather than separators.  Both single and double quotes
    are supported; a quote is closed only by its matching opener.

    Tracks whether each item was *started* (saw a non-whitespace character or
    an opening quote) so an explicit empty-quoted item like ``["", "x"]``
    yields ``["", "x"]`` rather than ``["x"]``. Bare whitespace between
    commas — like ``[a, , b]`` — is treated as no item and dropped.

    Note: quote stripping of the extracted item strings is NOT performed here;
    callers that pass ``strip_quotes=True`` to ``parse_frontmatter`` apply
    ``_strip_yaml_quotes`` separately.
    """
    if not (value.startswith("[") and value.endswith("]")):
        return None
    inner = value[1:-1].strip()
    if not inner:
        return []
    items: list[str] = []
    current: list[str] = []
    quote: str | None = None
    started = False  # an item exists once we see a non-space char or an opening quote
    for ch in inner:
        if quote:
            if ch == quote:
                quote = None
            else:
                current.append(ch)
        elif ch in ("'", '"'):
            quote = ch
            started = True
        elif ch == ",":
            if started:
                items.append("".join(current).strip())
            current = []
            started = False
        elif ch.isspace():
            # Whitespace contributes only if an item is already in progress.
            if started:
                current.append(ch)
        else:
            current.append(ch)
            started = True
    if started:
        items.append("".join(current).strip())
    return items


def parse_frontmatter(
    text: str,
    *,
    inline_lists: bool = False,
    strip_quotes: bool = False,
) -> tuple[dict, str]:
    """Parse YAML-ish frontmatter from the top of *text*.

    Returns ``(frontmatter_dict, body_text_after_frontmatter)``.

    If no valid frontmatter block is found (no opening ``---``, or no closing
    ``---``), returns ``({}, text)`` unchanged.

    Handles:
    - ``key: scalar_value`` pairs.
    - Multi-line YAML list syntax (``  - item`` lines after an empty-value key).
    - Blank lines inside a list block terminate the list for that key (and
      clear ``current_key`` so a subsequent list item is treated as orphaned
      rather than appended to the flushed list — per Rio's #252 fix).

    Parameters
    ----------
    text : str
        Raw document text, potentially starting with a ``---`` fence.
    inline_lists : bool
        When ``True``, recognise ``key: [a, "b", 'c']`` inline list syntax
        (used by Jira scanner output).  When ``False`` (default), such values
        are stored as literal scalar strings (md/Confluence behaviour).
    strip_quotes : bool
        When ``True``, strip matching surrounding single/double quotes from
        scalar values and multi-line list items (used by Jira).  When
        ``False`` (default), raw values are preserved (md/Confluence).

    Per-caller flag table (#261)
    ----------------------------
    Caller              inline_lists  strip_quotes
    evidence_md.py      False         False
    evidence_confluence False         False
    evidence_jira.py    True          True

    Returns
    -------
    tuple[dict, str]
        ``(frontmatter_dict, body_text)``
    """
    lines = text.splitlines(keepends=True)
    if not lines:
        return {}, text

    first_line = lines[0].rstrip("\n").rstrip("\r")
    if not _FM_OPEN_RE.match(first_line):
        return {}, text

    fm: dict = {}
    current_key: str | None = None
    current_list: list[str] | None = None
    close_idx: int | None = None

    for i, raw_line in enumerate(lines[1:], start=1):
        line = raw_line.rstrip("\n").rstrip("\r")

        # Closing ``---``
        if _FM_OPEN_RE.match(line):
            if current_key is not None and current_list is not None:
                fm[current_key] = current_list
            close_idx = i
            break

        # List item: any indented ``- item`` line whose preceding key context
        # is still active. If the current value is a scalar, the prior parsers
        # upgrade it to a list (drop the scalar, start fresh). This preserves
        # the Confluence/md behaviour where:
        #     key: scalar
        #       - first
        #       - second
        # produces ``key: ["first", "second"]``.
        list_m = _FM_LIST_ITEM_RE.match(line)
        if list_m and current_key is not None:
            if current_list is None:
                current_list = []
                fm.pop(current_key, None)
                fm[current_key] = current_list
            item = list_m.group(1).strip()
            if strip_quotes:
                item = _strip_yaml_quotes(item)
            current_list.append(item)
            continue

        # Key: value pair
        kv_m = _FM_KEY_VALUE_RE.match(line)
        if kv_m:
            if current_key is not None and current_list is not None:
                fm[current_key] = current_list
                current_list = None
            current_key = kv_m.group(1)
            val = kv_m.group(2).strip()

            if val == "":
                # Empty value — next lines may be list items
                current_list = []
                # Pre-store the list reference so it is visible during
                # iteration (mirrors Jira parser's behaviour of assigning
                # ``fields[key] = current_list`` immediately).
                fm[current_key] = current_list
            elif inline_lists and (inline := _parse_inline_list(val)) is not None:
                # Inline list syntax: ``key: [a, b, c]``. Inline lists are
                # complete on the line — no subsequent ``- item`` lines may
                # extend them, so clear key context.
                if strip_quotes:
                    inline = [_strip_yaml_quotes(item) for item in inline]
                current_key = None
                current_list = None
                fm[kv_m.group(1)] = inline
            else:
                # Scalar value. Keep ``current_key`` set so a subsequent
                # ``- item`` line can upgrade this scalar to a list (matches
                # the prior Confluence/md parser behaviour). ``current_list``
                # stays None to signal "scalar, not list yet."
                if strip_quotes:
                    val = _strip_yaml_quotes(val)
                current_list = None
                fm[kv_m.group(1)] = val
            continue

        # Blank line inside a list terminates it.  Clear ``current_key`` too
        # so a subsequent list item under the same key is treated as orphaned
        # rather than re-entering the list-item branch — per Rio's #252 fix
        # (reproduced in both evidence_md.py and evidence_confluence.py).
        if current_list is not None and not line.strip():
            fm[current_key] = current_list  # type: ignore[index]
            current_list = None
            current_key = None

    if close_idx is None:
        return {}, text

    body = "".join(lines[close_idx + 1 :])
    return fm, body


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
