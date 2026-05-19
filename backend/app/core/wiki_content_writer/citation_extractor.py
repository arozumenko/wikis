"""Citation extractor for wiki content writer output (#238).

Parses inline ``[path:lo-hi]`` or ``[path:N]`` citation tokens from generated
markdown and returns a ``CitedClaim`` per paragraph so the verifier (#239) can
check paragraph-level grounding.

Design decisions
----------------
Separate file (not folded into ``prompts.py``)
    The extractor has a non-trivial regex state machine (fence masking,
    inline-code masking, citation token parsing) that would bloat the prompts
    module.  Keeping them separate makes each file single-responsibility and
    easier to test in isolation.  Both are imported by #239 and #243.

Paragraph model
    Paragraphs are blank-line-delimited blocks (same semantics as CommonMark).
    A single newline is treated as a soft wrap inside the same paragraph.
    Multiple consecutive blank lines collapse to one paragraph split.

Fenced code block masking
    Citation tokens inside triple-backtick or tilde fences are structurally
    identical to real citations but are not claims.  We reuse the
    ``mask_code_fences`` helper from ``_evidence_utils`` (same pattern already
    used by the TOC extractor and the markdown chunker).

Inline code masking
    Back-tick spans can contain citation-like strings that must be ignored.
    We apply an additional inline-code mask before extracting citations.

Citation token grammar (regex)
    A citation token occupies a ``[…]`` bracket that does NOT look like a
    standard Markdown link (which has a ``(…)`` immediately after the ``]``).
    Inside the bracket:
      ``path:N``      → single line (start_line == end_line == N)
      ``path:lo-hi``  → range (start_line == lo, end_line == hi)
      ``path:N, path:M-P, …``  → multiple citations, comma-separated

    Validation rules:
    - ``path`` must be non-empty.
    - All line numbers must be positive integers (≥ 1).
    - ``lo <= hi`` must hold for ranges.
    - Tokens with a third dash segment (``1-2-3``) are invalid.
    - Tokens whose ``path`` looks like a URL (contains ``://``) are ignored.
    - The bracket must NOT be immediately followed by ``(`` (Markdown link).

Public API
----------
``Citation``    — dataclass: path, start_line, end_line.
``CitedClaim``  — dataclass: paragraph_index, paragraph_text, citations.
``extract_citations(markdown)`` → list[CitedClaim]
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from app.core.wiki_structure_planner._evidence_utils import mask_code_fences

# ── Inline-code mask ─────────────────────────────────────────────────────────

# Matches `…` spans (no newlines inside, non-greedy).  Replace content with
# spaces so citation regex offsets stay valid.
_INLINE_CODE_RE = re.compile(r"`[^`\n]*`")

# ── Citation token pattern ────────────────────────────────────────────────────

# Matches [path:N] or [path:lo-hi] or [path:N, path2:lo-hi, …]
# NOT followed by '(' (which would make it a Markdown link).
# The bracket contents are captured as a single group for post-parsing.
_CITATION_BRACKET_RE = re.compile(
    r"\[([^\[\]\n]+)\](?!\()",
)

# Each comma-separated element inside the bracket: path:N or path:lo-hi
# path = one or more non-whitespace chars that contain no colon (except the
#        separator colon) and no brackets.
_SINGLE_CITE_RE = re.compile(r"^\s*([^:\s\[\]]+)\s*:\s*(\d+)(?:-(\d+))?\s*$")


# ── Paragraph splitter ───────────────────────────────────────────────────────

# Split on one or more blank lines (two or more newlines, possibly with
# intermediate whitespace-only lines).
_PARAGRAPH_SPLIT_RE = re.compile(r"\n[ \t]*\n+")


# ── Public dataclasses ────────────────────────────────────────────────────────


@dataclass
class Citation:
    """A single citation token referencing a file span.

    Attributes
    ----------
    path : str
        Repo-relative POSIX path to the cited file (e.g. ``"api/auth.py"``).
    start_line : int
        1-indexed first line of the cited span.
    end_line : int
        1-indexed last line of the cited span.  Equals ``start_line`` for
        single-line citations.
    """

    path: str
    start_line: int
    end_line: int


@dataclass
class CitedClaim:
    """A paragraph from the generated markdown together with its citations.

    Paragraphs with no citation tokens have an empty ``citations`` list.
    The verifier (#239) uses this to flag uncited paragraphs for stripping.

    Attributes
    ----------
    paragraph_index : int
        0-based index of the paragraph in the source markdown.
    paragraph_text : str
        Raw paragraph text (original, unmasked).
    citations : list[Citation]
        All citation tokens found in this paragraph, in document order.
        Empty for uncited paragraphs.
    """

    paragraph_index: int
    paragraph_text: str
    citations: list[Citation] = field(default_factory=list)


# ── Helpers ──────────────────────────────────────────────────────────────────


def _mask_inline_code(text: str) -> str:
    """Replace inline-code spans with spaces, preserving offsets."""

    def _blank(m: re.Match[str]) -> str:
        return " " * (m.end() - m.start())

    return _INLINE_CODE_RE.sub(_blank, text)


def _parse_single_token(raw: str) -> Citation | None:
    """Parse one ``path:N`` or ``path:lo-hi`` token.

    Returns ``None`` for malformed tokens (non-numeric lines, empty path,
    invalid range, URL-like paths).
    """
    m = _SINGLE_CITE_RE.match(raw)
    if not m:
        return None

    path = m.group(1).strip()
    if not path or "://" in path:
        return None

    try:
        lo = int(m.group(2))
    except (TypeError, ValueError):
        return None

    hi_str = m.group(3)
    if hi_str is not None:
        try:
            hi = int(hi_str)
        except (TypeError, ValueError):
            return None
    else:
        hi = lo

    # Line numbers must be positive and the range must be valid.
    if lo < 1 or hi < lo:
        return None

    return Citation(path=path, start_line=lo, end_line=hi)


def _extract_citations_from_masked(masked: str) -> list[Citation]:
    """Extract all valid Citation objects from a masked paragraph string.

    ``masked`` has fenced-code-block and inline-code spans replaced with
    spaces, so bracket patterns inside those regions produce no match (or
    match only spaces, which fail ``_parse_single_token``).
    """
    citations: list[Citation] = []

    for bracket_match in _CITATION_BRACKET_RE.finditer(masked):
        bracket_content = bracket_match.group(1)
        # Split on commas to get individual path:line tokens. Invalid
        # parts are silently skipped — the verifier downstream still gets
        # the well-formed tokens.
        for part in bracket_content.split(","):
            cite = _parse_single_token(part)
            if cite is not None:
                citations.append(cite)

    return citations


# ── Public API ────────────────────────────────────────────────────────────────


def extract_citations(markdown: str) -> list[CitedClaim]:
    """Parse inline citation tokens from *markdown* and return per-paragraph claims.

    Each non-empty paragraph in *markdown* yields one ``CitedClaim``.
    Paragraphs are blank-line-delimited (CommonMark semantics).  A paragraph
    with no citation tokens gets an empty ``citations`` list — the verifier
    (#239) uses this to flag uncited content.

    Citations inside fenced code blocks (`` ``` `` / ``~~~``) and inline-code
    spans (`` `…` ``) are ignored.

    Parameters
    ----------
    markdown : str
        Raw markdown string produced by the writer agent.

    Returns
    -------
    list[CitedClaim]
        One ``CitedClaim`` per non-empty paragraph, in document order.
        Returns an empty list for blank or whitespace-only input.
    """
    if not markdown or not markdown.strip():
        return []

    # Mask fenced code blocks BEFORE splitting on blank lines. A fenced block
    # is allowed to contain blank lines; if we split first, the orphaned half
    # has no opening fence and ``mask_code_fences`` would leave its citation
    # tokens exposed.  ``mask_code_fences`` replaces fence content with spaces
    # while preserving overall string length, so offsets stay aligned with the
    # raw text and we can use the masked string to drive paragraph splitting.
    fence_masked = mask_code_fences(markdown)

    # Collect (start, end) byte ranges for each masked-paragraph slice.
    ranges: list[tuple[int, int]] = []
    cursor = 0
    for split_match in _PARAGRAPH_SPLIT_RE.finditer(fence_masked):
        ranges.append((cursor, split_match.start()))
        cursor = split_match.end()
    ranges.append((cursor, len(markdown)))

    claims: list[CitedClaim] = []
    for start, end in ranges:
        raw_para = markdown[start:end]
        if not raw_para.strip():
            continue
        # Masked slice already has fences blanked; apply inline-code masking
        # on top before extracting citations.
        masked_para = _mask_inline_code(fence_masked[start:end])
        citations = _extract_citations_from_masked(masked_para)

        claims.append(
            CitedClaim(
                paragraph_index=len(claims),
                paragraph_text=raw_para,
                citations=citations,
            )
        )

    return claims
