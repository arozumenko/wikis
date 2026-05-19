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
import re
from dataclasses import dataclass, field
from pathlib import Path

from ..parsers.markdown_chunker import Chunk, chunk_markdown
from ._evidence_utils import (
    collect_attachments,
    extract_first_paragraph,
    extract_toc_from_text,
    safe_join,
)
from .structure_skeleton import Cluster

logger = logging.getLogger(__name__)

# ── Constants ─────────────────────────────────────────────────────────────────

# Approximate chars per token — conservative.
_CHARS_PER_TOKEN = 4


# ── Frontmatter parser (stdlib-only) ─────────────────────────────────────────
#
# Plain markdown files can carry YAML frontmatter (Jekyll/Hugo/MkDocs style).
# The parser below is intentionally minimal — it handles the flat key/value
# and multi-line list structure that static-site generators emit.  Nested
# objects and multi-line strings are not supported (not needed here).
#
# Kept private to evidence_md.py (Option A from issue #255): the Confluence
# parser (evidence_confluence._parse_frontmatter) and the Jira parser
# (evidence_jira._parse_frontmatter) both share the same *goal* but differ in
# meaningful ways — Jira's version adds inline-list handling `[a, b]` and
# quote-stripping that Confluence's does not have, because the Jira scanner
# emits different frontmatter shapes.  Consolidating into _evidence_utils
# without a thorough audit risks silent behaviour changes for existing pack
# tests; a follow-up issue should unify once the parsers have been
# characterised properly.
#
# This copy is taken from evidence_confluence._parse_frontmatter verbatim
# (which is the appropriate model for plain-markdown, not Jira).

_FM_OPEN_RE = re.compile(r"^---\s*$")
_FM_KEY_VALUE_RE = re.compile(r"^([A-Za-z_][A-Za-z0-9_]*)\s*:\s*(.*)$")
_FM_LIST_ITEM_RE = re.compile(r"^\s+-\s+(.+)$")


def _parse_frontmatter(md_text: str) -> tuple[dict, str]:
    """Parse YAML frontmatter from the top of *md_text*.

    Returns ``(frontmatter_dict, body_text_after_frontmatter)``.

    If no valid frontmatter block is found (no opening ``---``, or no closing
    ``---``), returns ``({}, md_text)`` unchanged.

    Handles:
    - ``key: scalar_value`` pairs.
    - Multi-line YAML list syntax (``  - item`` lines after an empty-value key).
    - Blank lines inside a list block terminate the list for that key.
    """
    lines = md_text.splitlines(keepends=True)
    if not lines:
        return {}, md_text

    first_line = lines[0].rstrip("\n").rstrip("\r")
    if not _FM_OPEN_RE.match(first_line):
        return {}, md_text

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

        # List item: must come after a key that started a list
        list_m = _FM_LIST_ITEM_RE.match(line)
        if list_m and current_key is not None:
            if current_list is None:
                current_list = []
                fm.pop(current_key, None)
            current_list.append(list_m.group(1).strip())
            continue

        # Key: value pair
        kv_m = _FM_KEY_VALUE_RE.match(line)
        if kv_m:
            if current_key is not None and current_list is not None:
                fm[current_key] = current_list
                current_list = None
            current_key = kv_m.group(1)
            val = kv_m.group(2).strip()
            if val:
                fm[current_key] = val
            continue

        # Blank line inside a list terminates it
        if current_list is not None and not line.strip():
            fm[current_key] = current_list
            current_list = None
            current_key = None

    if close_idx is None:
        return {}, md_text

    body = "".join(lines[close_idx + 1 :])
    return fm, body


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
    safe = safe_join(repo_root, rel_path)
    if safe is None:
        logger.debug("evidence_md: path escape rejected: %r", rel_path)
        return None
    try:
        return Path(safe).read_text(encoding="utf-8", errors="replace")
    except (OSError, PermissionError):
        return None


def _process_doc(repo_root: str, rel_path: str) -> dict | None:
    """Build one doc entry dict from *rel_path* under *repo_root*.

    Returns None when the file cannot be safely read.
    """
    text = _read_file_safe(repo_root, rel_path)
    if text is None:
        return None

    # Strip frontmatter before chunking so the ``---…---`` block is never
    # emitted as the preamble chunk body (#255).  Plain markdown rarely
    # carries YAML frontmatter but when it does (Jekyll/Hugo/MkDocs repos)
    # ``extract_first_paragraph`` was returning the raw YAML instead of the
    # actual opening prose.
    frontmatter, body = _parse_frontmatter(text)

    chunks = chunk_markdown(body, doc_path=rel_path, frontmatter=frontmatter or None)

    # Title priority: frontmatter "title" key → first H1 in body → filename.
    title = str(frontmatter.get("title", "")).strip() if frontmatter else ""
    if not title:
        for chunk in chunks:
            if chunk.heading_path:
                title = chunk.heading_path[0]
                break
    if not title:
        title = Path(rel_path).name

    return {
        "source_path": rel_path,
        "title": title,
        "toc": extract_toc_from_text(text),
        "first_paragraph": extract_first_paragraph(chunks),
        "attachments": collect_attachments(chunks),
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
