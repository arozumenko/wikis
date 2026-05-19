"""Evidence pack builder for Confluence clusters (#234).

Deterministic, parallel-friendly prefetch of ~1000 tokens of real Confluence
page evidence per cluster, so the planner LLM names and describes Confluence
clusters from actual page structure (heading TOC, opening prose, label
taxonomy) rather than blank summaries.

Public API
----------
build_confluence_pack(cluster, *, repo_root, token_budget) -> ConfluenceEvidencePack

ConfluenceEvidencePack
    .page_entries : list of per-page dicts with keys:
                   - source_path  : str
                   - title        : str   (frontmatter title, first H1, or filename)
                   - space_key    : str   (from ArtifactInfo.space_key)
                   - parent_path  : str   (from ArtifactInfo.parent_path — breadcrumb)
                   - labels       : list[str]  from YAML frontmatter
                   - toc          : list[str]  H1/H2 headings only (top 2 levels)
                   - first_paragraph : str  first non-empty body text, ≤200 chars
                   - attachments  : list[str]  filenames only — no content is read
    .serialize() : str — render all sections for prompt injection

Design choices (see PR body for rationale)
------------------------------------------
* Separate dataclass (not extending EvidencePack or MarkdownEvidencePack) —
  Confluence-specific fields (space_key, parent_path, labels) don't apply
  to code or plain doc clusters.
* New file evidence_confluence.py — keeps concurrent PRs #233 (md), #234
  (you), #235 (jira) non-conflicting. A follow-up PR (#243) consolidates
  into a dispatcher after all three merge.
* Minimal stdlib-only YAML frontmatter parser — the Confluence scanner
  (#211) produces a simple flat structure (title, labels, parents,
  original_url, space_key); no third-party YAML library needed.
* _safe_join pattern reused from evidence.py (canonical security primitive).
* Per-page soft caps: first_paragraph ≤200 chars always; TOC truncated
  before dropping whole entries when budget is tight.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from pathlib import Path

from ..parsers.markdown_chunker import chunk_markdown
from ._evidence_utils import (
    collect_attachments,
    extract_first_paragraph,
    extract_toc_from_text,
    mask_code_fences,
    safe_join,
)
from .structure_skeleton import Cluster

logger = logging.getLogger(__name__)

# ── Constants ─────────────────────────────────────────────────────────────────

# Approximate chars per token — conservative (matches evidence.py / evidence_md.py).
_CHARS_PER_TOKEN = 4


# ── ConfluenceEvidencePack ────────────────────────────────────────────────────


@dataclass
class ConfluenceEvidencePack:
    """Prefetched evidence for one Confluence cluster, ready for prompt injection.

    Attributes
    ----------
    cluster_id : int
    kind : str
        Always ``"confluence"`` for Confluence clusters.
    page_entries : list[dict]
        One entry per unique source file in the cluster.  Each entry has:
        - ``source_path`` (str) — repo-relative path to the exported .md file
        - ``title`` (str) — frontmatter title, first H1, or filename
        - ``space_key`` (str) — Confluence space key from ArtifactInfo
        - ``parent_path`` (str) — breadcrumb from ArtifactInfo.parent_path
        - ``labels`` (list[str]) — parsed from YAML frontmatter
        - ``toc`` (list[str]) — H1/H2 heading names in document order
        - ``first_paragraph`` (str) — first non-empty body text, ≤200 chars
        - ``attachments`` (list[str]) — bare filenames, no content, deduped
    """

    cluster_id: int
    kind: str
    page_entries: list[dict] = field(default_factory=list)

    def serialize(self) -> str:
        """Render all sections as a prompt-ready string.

        The format is intentionally simple — the planner agent (#236) wraps
        this inside its own prompt template.
        """
        parts: list[str] = [f"[ConfluenceEvidencePack cluster={self.cluster_id} kind={self.kind}]"]

        for entry in self.page_entries:
            src = entry.get("source_path", "")
            title = entry.get("title", src)
            space_key: str = entry.get("space_key", "")
            parent_path: str = entry.get("parent_path", "")
            labels: list[str] = entry.get("labels", [])
            toc: list[str] = entry.get("toc", [])
            first_para: str = entry.get("first_paragraph", "")
            attachments: list[str] = entry.get("attachments", [])

            parts.append(f"\n### {src}")
            if title and title != src:
                parts.append(f"Title: {title}")
            if space_key:
                parts.append(f"Space: {space_key}")
            if parent_path:
                parts.append(f"Breadcrumb: {parent_path}")
            if labels:
                parts.append(f"Labels: {', '.join(labels)}")

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


# ── Frontmatter parser (stdlib-only) ─────────────────────────────────────────

# Frontmatter shape produced by the Confluence scanner (#211):
#
#   ---
#   title: My Page
#   labels:
#     - infra
#     - security
#   parents:
#     - Engineering
#     - Security
#   original_url: https://confluence.example.com/...
#   space_key: ENG
#   ---
#
# Rules:
# * Bounded by leading ``---`` / trailing ``---`` on their own lines.
# * Simple ``key: value`` pairs (string values).
# * YAML list items: lines starting with ``  - item`` under a key.
# * Nested objects and multi-line strings: not emitted by scanner → punt.
# * On any parse anomaly: return what was collected so far (graceful).

_FM_OPEN_RE = re.compile(r"^---\s*$")
_FM_KEY_VALUE_RE = re.compile(r"^([A-Za-z_][A-Za-z0-9_]*)\s*:\s*(.*)$")
# Match list items at any positive indent — Confluence scanner emits 2-space
# indent in practice but a stricter pattern would silently drop items if the
# generator changes (Copilot review on #252).
_FM_LIST_ITEM_RE = re.compile(r"^\s+-\s+(.+)$")


def _parse_frontmatter(md_text: str) -> tuple[dict, str]:
    """Parse YAML frontmatter from the top of *md_text*.

    Returns ``(frontmatter_dict, body_text_after_frontmatter)``.

    If no valid frontmatter block is found, returns ``({}, md_text)``.
    Punts on nested YAML structures — the Confluence scanner does not emit
    them so this parser does not need to handle them.
    """
    lines = md_text.splitlines(keepends=True)
    if not lines:
        return {}, md_text

    # Must start with ``---``
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
            # Flush any in-progress list
            if current_key is not None and current_list is not None:
                fm[current_key] = current_list
            close_idx = i
            break

        # List item: must come after a key that started a list
        list_m = _FM_LIST_ITEM_RE.match(line)
        if list_m and current_key is not None:
            if current_list is None:
                # Convert previously-stored scalar to a list if needed
                current_list = []
                # If key already had a scalar value, drop it (list wins)
                fm.pop(current_key, None)
            current_list.append(list_m.group(1).strip())
            continue

        # Key: value pair
        kv_m = _FM_KEY_VALUE_RE.match(line)
        if kv_m:
            # Flush previous key if it was building a list
            if current_key is not None and current_list is not None:
                fm[current_key] = current_list
                current_list = None
            current_key = kv_m.group(1)
            val = kv_m.group(2).strip()
            if val:
                # Scalar value (list may follow on next lines)
                fm[current_key] = val
            else:
                # Empty value — next lines may be list items
                pass
            continue

        # Anything else (blank lines, unrecognised): continue
        if current_list is not None and not line.strip():
            # Blank line inside a list terminates it. Clear `current_key`
            # too — otherwise a subsequent list item under the same key
            # would re-enter the list-item branch, repopulate the list as
            # if starting fresh, and silently drop the already-flushed
            # items (Rio review on #252).
            fm[current_key] = current_list
            current_list = None
            current_key = None

    if close_idx is None:
        # No closing --- found; treat entire document as having no frontmatter
        return {}, md_text

    body = "".join(lines[close_idx + 1 :])
    return fm, body


# ── Internal helpers ──────────────────────────────────────────────────────────


def _read_file_safe(repo_root: str, rel_path: str) -> str | None:
    """Read the full content of *rel_path* under *repo_root*, or None on error."""
    safe = safe_join(repo_root, rel_path)
    if safe is None:
        logger.debug("evidence_confluence: path escape rejected: %r", rel_path)
        return None
    try:
        return Path(safe).read_text(encoding="utf-8", errors="replace")
    except (OSError, PermissionError):
        return None


def _parse_labels(frontmatter: dict) -> list[str]:
    """Extract labels from *frontmatter* dict.

    Handles:
    - ``labels: [a, b]`` YAML list → already a list after our parser
    - ``labels: single`` scalar → wraps in a list
    - Missing key → empty list
    """
    raw = frontmatter.get("labels")
    if raw is None:
        return []
    if isinstance(raw, list):
        return [str(item).strip() for item in raw if str(item).strip()]
    # Scalar: ``labels: security``
    val = str(raw).strip()
    return [val] if val else []


def _process_page(
    repo_root: str,
    rel_path: str,
    space_key: str,
    parent_path: str,
) -> dict | None:
    """Build one page entry dict from *rel_path* under *repo_root*.

    Returns None when the file cannot be safely read.
    """
    text = _read_file_safe(repo_root, rel_path)
    if text is None:
        return None

    # Parse frontmatter; body is the text after the closing ---
    frontmatter, body = _parse_frontmatter(text)

    # Title: prefer frontmatter "title", then first H1 in body, else filename.
    # The H1 scan masks fenced code blocks so a `# install` line inside a
    # ```bash example doesn't become the page title.
    title: str = ""
    if frontmatter:
        title = str(frontmatter.get("title", "")).strip()
    if not title:
        masked_body = mask_code_fences(body)
        h1_m = re.search(r"^# (.+)$", masked_body, re.MULTILINE)
        if h1_m:
            title = h1_m.group(1).strip()
    if not title:
        title = Path(rel_path).name

    labels = _parse_labels(frontmatter)

    # TOC scanned from the full raw text (same as evidence_md.py approach)
    toc = extract_toc_from_text(text)

    # first_paragraph + attachments come from chunk_markdown on the body text
    # (after stripping frontmatter so preamble text is the actual opening prose)
    chunks = chunk_markdown(body, doc_path=rel_path, frontmatter=frontmatter if frontmatter else None)
    first_paragraph = extract_first_paragraph(chunks)
    attachments = collect_attachments(chunks)

    return {
        "source_path": rel_path,
        "title": title,
        "space_key": space_key,
        "parent_path": parent_path,
        "labels": labels,
        "toc": toc,
        "first_paragraph": first_paragraph,
        "attachments": attachments,
    }


def _enforce_total_budget(pack: ConfluenceEvidencePack, char_budget: int) -> None:
    """Trim page entries until total serialised size fits budget.

    Strategy (shed in this order, least costly per information lost):
    1. Truncate attachment lists per entry.
    2. Truncate TOC lists per entry.
    3. Drop entire page entries from the tail.

    At least one entry is always kept (even if over budget) so the pack is
    never empty for a non-empty cluster.

    Implementation note: serialise once up front and track length deltas as
    list items are popped, rather than re-serialising the whole pack on
    every loop iteration. Re-serialising is O(n²) on cluster size; the delta
    is O(1) per pop.
    """
    current_len = len(pack.serialize())
    if current_len <= char_budget:
        return

    # Phase 1: trim attachment lists per entry (least informative per char)
    for entry in pack.page_entries:
        attachments: list[str] = entry.get("attachments", [])
        while current_len > char_budget and len(attachments) > 1:
            dropped = attachments.pop()
            current_len -= len(dropped) + 5  # "  - {dropped}\n" overhead
        if current_len <= char_budget:
            return

    # Phase 2: trim TOC entries per entry
    for entry in pack.page_entries:
        toc: list[str] = entry.get("toc", [])
        while current_len > char_budget and len(toc) > 2:
            dropped = toc.pop()
            current_len -= len(dropped) + 5
        if current_len <= char_budget:
            return

    # Phase 3: drop whole entries from the tail. Re-measure once per drop —
    # computing an exact delta for a whole entry is fiddly and whole-entry
    # drops are rare so the cost is acceptable.
    while current_len > char_budget and len(pack.page_entries) > 1:
        pack.page_entries.pop()
        current_len = len(pack.serialize())


# ── Public API ────────────────────────────────────────────────────────────────


def build_confluence_pack(
    cluster: Cluster,
    *,
    repo_root: str,
    token_budget: int = 1000,
) -> ConfluenceEvidencePack:
    """Build a Confluence evidence pack for *cluster*.

    Parameters
    ----------
    cluster : Cluster
        Confluence cluster (``cluster.kind == "confluence"``) whose
        ``artifacts`` are ``ArtifactInfo(kind="confluence_page", …)``
        entries.  Other artifact kinds in a mixed cluster are silently ignored.
    repo_root : str
        Absolute path to the checked-out repository root.  All file IO is
        confined to this directory tree via ``_safe_join``.
    token_budget : int
        Approximate total token budget for the serialised pack.  Defaults to
        1000 tokens (~4000 chars) for Confluence clusters.

    Returns
    -------
    ConfluenceEvidencePack
    """
    char_budget = token_budget * _CHARS_PER_TOKEN

    # Collect unique source paths from confluence_page artifacts only.
    # Maintain insertion order (stable across repeated calls).
    seen_paths: set[str] = set()
    # Map source_path → (space_key, parent_path) from the first artifact seen
    page_meta: dict[str, tuple[str, str]] = {}

    for artifact in cluster.artifacts:
        if artifact.kind != "confluence_page":
            continue
        if artifact.source_path and artifact.source_path not in seen_paths:
            seen_paths.add(artifact.source_path)
            page_meta[artifact.source_path] = (artifact.space_key, artifact.parent_path)

    page_entries: list[dict] = []
    for rel_path in page_meta:
        space_key, parent_path = page_meta[rel_path]
        entry = _process_page(repo_root, rel_path, space_key, parent_path)
        if entry is not None:
            page_entries.append(entry)

    pack = ConfluenceEvidencePack(
        cluster_id=cluster.cluster_id,
        kind=cluster.kind,
        page_entries=page_entries,
    )

    _enforce_total_budget(pack, char_budget)
    return pack
