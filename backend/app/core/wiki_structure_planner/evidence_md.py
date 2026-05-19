"""Evidence pack builder for markdown (doc) clusters (#233).

Deterministic, parallel-friendly prefetch of ~1000 tokens of real documentation
evidence per cluster, so the planner LLM names and describes doc clusters from
actual content (heading structure, opening prose, attachment inventory) rather
than blank summaries.

Public API
----------
build_md_pack(cluster, *, repo_root, token_budget) -> MarkdownEvidencePack

MarkdownEvidencePack
    .doc_entries : list of per-doc dicts with keys:
                   - source_path : str
                   - title       : str   (frontmatter title or first H1, else filename)
                   - toc         : list[str]  H1/H2 headings only (top 2 levels)
                   - first_paragraph : str  first non-empty body text, ≤200 chars
                   - attachments : list[str] filenames only — no content is read
    .serialize() : str — render all sections for prompt injection

Design choices (see PR body for rationale)
------------------------------------------
* Separate dataclass (not extending EvidencePack) — code-specific fields
  (signatures, sql_blocks, file_heads) don't apply to doc clusters.
* New file evidence_md.py — keeps concurrent PRs #234 (confluence) and #235
  (jira) non-conflicting.  A follow-up PR consolidates into a dispatcher after
  all three merge (#243).
* Per-doc soft cap: first_paragraph ≤200 chars always; TOC truncated when a
  single doc would blow the per-doc share of the total budget.
* _safe_join pattern reused from evidence.py (canonical security primitive).
"""

from __future__ import annotations

import logging
import os
import re
from dataclasses import dataclass, field
from pathlib import Path

from ..parsers.markdown_chunker import Chunk, chunk_markdown
from .structure_skeleton import Cluster

logger = logging.getLogger(__name__)

# ── Constants ─────────────────────────────────────────────────────────────────

# Hard cap on first_paragraph per doc (chars, not tokens).
_FIRST_PARA_CHAR_CAP = 200

# Approximate chars per token — conservative.
_CHARS_PER_TOKEN = 4

# TOC entries are short; allow up to this many before per-doc truncation.
_MAX_TOC_ENTRIES_PER_DOC = 20


# ── Path safety ───────────────────────────────────────────────────────────────


def _safe_join(root: str, rel_path: str) -> str | None:
    """Join *rel_path* under *root*; return None if it would escape the root.

    Resolves symlinks and ``..`` so absolute paths and traversal attempts are
    rejected before any file IO happens.  Mirrors the canonical implementation
    in ``evidence.py:60``.
    """
    try:
        full = os.path.realpath(os.path.join(root, rel_path))
        root_real = os.path.realpath(root)
        if not full.startswith(root_real + os.sep) and full != root_real:
            return None
        return full
    except (ValueError, OSError):
        return None


# ── MarkdownEvidencePack ──────────────────────────────────────────────────────


@dataclass
class MarkdownEvidencePack:
    """Prefetched evidence for one doc cluster, ready for prompt injection.

    Attributes
    ----------
    cluster_id : int
    kind : str
        Always ``"doc"`` for markdown clusters.
    doc_entries : list[dict]
        One entry per unique source file in the cluster.  Each entry has:
        - ``source_path`` (str) — repo-relative path to the .md file
        - ``title`` (str) — frontmatter title, first H1, or filename
        - ``toc`` (list[str]) — H1/H2 heading names in document order
        - ``first_paragraph`` (str) — first non-empty body text, ≤200 chars
        - ``attachments`` (list[str]) — bare filenames, no content, deduped
    """

    cluster_id: int
    kind: str
    doc_entries: list[dict] = field(default_factory=list)

    def serialize(self) -> str:
        """Render all sections as a prompt-ready string.

        The format is intentionally simple — the planner agent (#236) wraps
        this inside its own prompt template.
        """
        parts: list[str] = [f"[MarkdownEvidencePack cluster={self.cluster_id} kind={self.kind}]"]

        for entry in self.doc_entries:
            src = entry.get("source_path", "")
            title = entry.get("title", src)
            toc: list[str] = entry.get("toc", [])
            first_para: str = entry.get("first_paragraph", "")
            attachments: list[str] = entry.get("attachments", [])

            parts.append(f"\n### {src}")
            if title and title != src:
                parts.append(f"Title: {title}")

            if toc:
                parts.append("TOC:")
                for heading in toc:
                    parts.append(f"  - {heading}")

            if first_para:
                parts.append(f"Summary: {first_para}")

            if attachments:
                parts.append("Attachments:")
                for name in attachments:
                    parts.append(f"  - {name}")

        return "\n".join(parts)


# ── Internal helpers ──────────────────────────────────────────────────────────


def _estimate_tokens(text: str) -> int:
    """Rough token estimate from char count."""
    return max(1, len(text) // _CHARS_PER_TOKEN)


def _read_file_safe(repo_root: str, rel_path: str) -> str | None:
    """Read the full content of *rel_path* under *repo_root*, or None on error."""
    safe = _safe_join(repo_root, rel_path)
    if safe is None:
        logger.debug("evidence_md: path escape rejected: %r", rel_path)
        return None
    try:
        return Path(safe).read_text(encoding="utf-8", errors="replace")
    except (OSError, PermissionError):
        return None


_TOC_HEADING_RE = re.compile(r"^(#{1,2})\s+(.*?)(?:\s+#+)?\s*$", re.MULTILINE)
_TOC_FENCE_RE = re.compile(r"```[\s\S]*?(?:```|\Z)|~~~[\s\S]*?(?:~~~|\Z)")


def _extract_toc(md_text: str) -> list[str]:
    """Return H1/H2 heading names from raw markdown in document order.

    Scans the raw text directly rather than relying on ``Chunk`` output so
    structural-only docs (heading-only with no body content, which the
    chunker drops per the "skip empty sections" rule from #231 / Rio's
    review of #248) still surface their TOC.

    Headings are NOT deduplicated by name — a doc with two `## Setup`
    sections under different H1s legitimately has both in its TOC, and
    losing one would distort the structure the planner sees.

    H3+ are excluded by design. Fenced code blocks are masked so a
    ``# install`` line inside ```bash isn't picked up.
    """

    def _blank(m: re.Match[str]) -> str:
        return " " * (m.end() - m.start())

    masked = _TOC_FENCE_RE.sub(_blank, md_text)
    toc: list[str] = []
    for m in _TOC_HEADING_RE.finditer(masked):
        heading = m.group(2).strip()
        if heading:
            toc.append(heading)
    return toc[:_MAX_TOC_ENTRIES_PER_DOC]


def _extract_first_paragraph(chunks: list[Chunk]) -> str:
    """Return first non-empty body text from *chunks*, capped at 200 chars.

    Prefers preamble chunk (heading_path=[]) if present and non-empty;
    falls back to the first heading section's body.
    """
    for chunk in chunks:
        stripped = chunk.body.strip()
        if stripped:
            return stripped[:_FIRST_PARA_CHAR_CAP]
    return ""


def _collect_attachments(chunks: list[Chunk]) -> list[str]:
    """Return a sorted, deduplicated list of attachment filenames from all chunks.

    Attachment content is never read — only the filenames/paths extracted by
    chunk_markdown's ``[[attachment: …]]`` pass-through are included.
    """
    seen: set[str] = set()
    result: list[str] = []
    for chunk in chunks:
        for name in chunk.attachments:
            if name not in seen:
                seen.add(name)
                result.append(name)
    return sorted(result)


def _process_doc(repo_root: str, rel_path: str) -> dict | None:
    """Build one doc entry dict from *rel_path* under *repo_root*.

    Returns None when the file cannot be safely read.
    """
    text = _read_file_safe(repo_root, rel_path)
    if text is None:
        return None

    chunks = chunk_markdown(text, doc_path=rel_path)

    # Title: first H1 of the first non-preamble chunk, falling back to
    # filename. Frontmatter parsing is intentionally out of scope here —
    # plain markdown rarely carries YAML frontmatter, and Confluence /
    # Jira packs (#234 / #235) handle that themselves.
    title = ""
    for chunk in chunks:
        if chunk.heading_path:
            title = chunk.heading_path[0]
            break
    if not title:
        title = Path(rel_path).name

    return {
        "source_path": rel_path,
        "title": title,
        "toc": _extract_toc(text),
        "first_paragraph": _extract_first_paragraph(chunks),
        "attachments": _collect_attachments(chunks),
    }


def _enforce_total_budget(pack: MarkdownEvidencePack, char_budget: int) -> None:
    """Trim doc entries from the pack tail until total serialised size fits budget.

    Strategy (shed in this order, least costly per information lost):
    1. Truncate attachment lists per entry.
    2. Truncate TOC lists per entry.
    3. Drop entire doc entries from the tail.

    At least one entry is always kept (even if over budget) so the pack is
    never empty for a non-empty cluster.

    Implementation note: we serialise once up front and then track the
    delta as we pop list items, rather than re-serialising the whole pack
    on every loop iteration. Re-serialising would be O(n²) on cluster
    size; the delta is O(1) per pop.
    """
    current_len = len(pack.serialize())
    if current_len <= char_budget:
        return

    # Phase 1: trim attachment lists per entry (least informative per char).
    for entry in pack.doc_entries:
        attachments: list[str] = entry.get("attachments", [])
        while current_len > char_budget and len(attachments) > 1:
            dropped = attachments.pop()
            # Removed line is `"  - {dropped}\n"` — see serialize().
            current_len -= len(dropped) + 5
        if current_len <= char_budget:
            return

    # Phase 2: trim TOC entries per entry.
    for entry in pack.doc_entries:
        toc: list[str] = entry.get("toc", [])
        while current_len > char_budget and len(toc) > 2:
            dropped = toc.pop()
            current_len -= len(dropped) + 5

        if current_len <= char_budget:
            return

    # Phase 3: drop whole entries from the tail. Re-measure once per drop
    # because computing an exact serialised-size delta for a whole entry
    # is fiddly (header + multiple sections); whole-entry drops are rare
    # so the cost is fine.
    while current_len > char_budget and len(pack.doc_entries) > 1:
        pack.doc_entries.pop()
        current_len = len(pack.serialize())


# ── Public API ────────────────────────────────────────────────────────────────


def build_md_pack(
    cluster: Cluster,
    *,
    repo_root: str,
    token_budget: int = 1000,
) -> MarkdownEvidencePack:
    """Build a markdown evidence pack for *cluster*.

    Parameters
    ----------
    cluster : Cluster
        Doc cluster (``cluster.kind == "doc"``) whose ``artifacts`` are
        ``ArtifactInfo(kind="doc_section", …)`` entries.  Other artifact
        kinds in a mixed cluster are silently ignored.
    repo_root : str
        Absolute path to the checked-out repository root.  All file IO is
        confined to this directory tree via ``_safe_join``.
    token_budget : int
        Approximate total token budget for the serialised pack.  Defaults to
        1000 tokens (~4000 chars) for doc clusters — smaller than the 1500-
        token code pack because doc headings + paragraphs are more token-dense.

    Returns
    -------
    MarkdownEvidencePack
    """
    char_budget = token_budget * _CHARS_PER_TOKEN

    # Collect unique source paths from doc_section artifacts only.
    # Maintain insertion order (stable across repeated calls).
    seen_paths: set[str] = set()
    unique_paths: list[str] = []
    for artifact in cluster.artifacts:
        if artifact.kind != "doc_section":
            continue
        if artifact.source_path and artifact.source_path not in seen_paths:
            seen_paths.add(artifact.source_path)
            unique_paths.append(artifact.source_path)

    doc_entries: list[dict] = []
    for rel_path in unique_paths:
        entry = _process_doc(repo_root, rel_path)
        if entry is not None:
            doc_entries.append(entry)

    pack = MarkdownEvidencePack(
        cluster_id=cluster.cluster_id,
        kind=cluster.kind,
        doc_entries=doc_entries,
    )

    _enforce_total_budget(pack, char_budget)
    return pack
