"""Evidence pack builder for code clusters (#232).

Deterministic, parallel-friendly prefetch of ~1500 tokens of real code evidence
per cluster, so the planner LLM names clusters from actual signatures and file
heads rather than 5-line summaries.

Public API
----------
build_pack(cluster, kind="code", *, repo_root, top_k, token_budget) -> EvidencePack

EvidencePack
    .signatures   : list of (name, source_path, layer, summary)
    .file_heads   : list of (rel_path, text) — first 40 lines of entry/public files
    .readme_excerpt: str — up to 60 lines from first README in cluster dirs
    .sql_blocks   : list of str — CREATE TABLE blocks from *.sql files
    .serialize()  : str — render all sections for prompt injection (format TBD by #236)
"""

from __future__ import annotations

import logging
import os
import re
from dataclasses import dataclass, field
from pathlib import Path

from .structure_skeleton import ArtifactInfo, Cluster

logger = logging.getLogger(__name__)

# ── Layer priority — lower number = higher priority ───────────────────────────
_LAYER_PRIORITY: dict[str, int] = {
    "entry_point": 0,
    "public_api": 1,
    "core_type": 2,
    "infrastructure": 3,
    "internal": 4,
    "constant": 5,
}
_DEFAULT_PRIORITY = 6  # anything unknown goes last

# Layers whose source files we read the head of (first N lines).
_FILE_HEAD_LAYERS: frozenset[str] = frozenset({"entry_point", "public_api"})

_README_NAMES: frozenset[str] = frozenset({"readme.md", "readme.rst", "readme.txt", "readme"})

_SQL_CREATE_PATTERN = re.compile(
    r"(CREATE\s+TABLE\b[^;]*;)",
    re.IGNORECASE | re.DOTALL,
)

# Approximate chars per token — conservative (LLaMA-family tokenisers are closer to 3.5).
_CHARS_PER_TOKEN = 4


# ── Path safety ───────────────────────────────────────────────────────────────


def _safe_join(root: str, rel_path: str) -> str | None:
    """Join *rel_path* under *root*; return None if it would escape the root.

    Resolves symlinks and ``..`` so absolute paths and traversal attempts are
    rejected before any file IO happens. Mirrors the pattern in
    ``wiki_content_writer/writer_tools.py``.
    """
    try:
        full = os.path.realpath(os.path.join(root, rel_path))
        root_real = os.path.realpath(root)
        if not full.startswith(root_real + os.sep) and full != root_real:
            return None
        return full
    except (ValueError, OSError):
        return None


# ── EvidencePack ──────────────────────────────────────────────────────────────


@dataclass
class EvidencePack:
    """Prefetched evidence for one cluster, ready for prompt injection.

    Fields are deliberately plain Python types so the planner agent (#236) can
    format them however it likes — the serialization here is a sensible default
    only; callers should override it when they control the prompt template.

    Attributes
    ----------
    cluster_id : int
    kind : str
        Source kind of the originating cluster ("code", "doc", …).
    signatures : list of (name, source_path, layer, summary)
        Top-K symbol signatures sorted by layer priority then connections desc.
    file_heads : list of (rel_path, text)
        First ≤40 lines of each unique entry_point / public_api source file.
    readme_excerpt : str
        Head ≤60 lines of the first README found inside the cluster's dirs.
    sql_blocks : list of str
        All ``CREATE TABLE …;`` blocks extracted from *.sql files in cluster dirs.
    """

    cluster_id: int
    kind: str
    signatures: list[tuple[str, str, str, str]] = field(default_factory=list)
    file_heads: list[tuple[str, str]] = field(default_factory=list)
    readme_excerpt: str = ""
    sql_blocks: list[str] = field(default_factory=list)

    def serialize(self) -> str:
        """Render all sections as a prompt-ready string.

        The format is intentionally simple — the planner agent (#236) wraps
        this inside its own prompt template; we just produce readable text.
        """
        parts: list[str] = [f"[EvidencePack cluster={self.cluster_id} kind={self.kind}]"]

        if self.signatures:
            parts.append("## Symbol Signatures")
            for name, src, layer, summary in self.signatures:
                line = f"  {layer:16s} {name} ({src})"
                if summary:
                    line += f" — {summary}"
                parts.append(line)

        if self.file_heads:
            parts.append("## File Heads")
            for rel_path, text in self.file_heads:
                parts.append(f"### {rel_path}")
                parts.append(text)

        if self.readme_excerpt:
            parts.append("## README")
            parts.append(self.readme_excerpt)

        if self.sql_blocks:
            parts.append("## Schema (SQL)")
            for block in self.sql_blocks:
                parts.append(block)

        return "\n".join(parts)


# ── Internal helpers ──────────────────────────────────────────────────────────


def _layer_rank(layer: str) -> int:
    return _LAYER_PRIORITY.get(layer, _DEFAULT_PRIORITY)


def _sorted_artifacts(artifacts: list[ArtifactInfo]) -> list[ArtifactInfo]:
    """Sort by (layer priority asc, connections desc, name asc) for stable ordering."""
    return sorted(
        artifacts,
        key=lambda a: (_layer_rank(a.layer), -a.connections, a.name),
    )


def _read_file_head(repo_root: str, rel_path: str, max_lines: int = 40) -> str | None:
    """Read up to *max_lines* lines from *rel_path* under *repo_root*.

    Returns None when the path is unsafe or the file cannot be read.
    """
    safe = _safe_join(repo_root, rel_path)
    if safe is None:
        logger.debug("evidence: path escape rejected: %r", rel_path)
        return None
    try:
        text = Path(safe).read_text(encoding="utf-8", errors="replace")
    except (OSError, PermissionError):
        return None
    lines = text.splitlines()
    return "\n".join(lines[:max_lines])


def _find_readme(repo_root: str, dir_path: str) -> str | None:
    """Return content of the first README found in *dir_path* under *repo_root*, or None."""
    safe_dir = _safe_join(repo_root, dir_path)
    if safe_dir is None:
        return None
    try:
        entries = os.listdir(safe_dir)
    except (OSError, PermissionError):
        return None
    for entry in sorted(entries):
        if entry.lower() in _README_NAMES:
            readme_rel = os.path.join(dir_path, entry)
            content = _read_file_head(repo_root, readme_rel, max_lines=60)
            if content is not None:
                return content
    return None


def _extract_sql_blocks(repo_root: str, dir_path: str) -> list[str]:
    """Return all CREATE TABLE blocks from *.sql files directly under *dir_path*."""
    safe_dir = _safe_join(repo_root, dir_path)
    if safe_dir is None:
        return []
    blocks: list[str] = []
    try:
        entries = os.listdir(safe_dir)
    except (OSError, PermissionError):
        return []
    for entry in sorted(entries):
        if not entry.lower().endswith(".sql"):
            continue
        sql_rel = os.path.join(dir_path, entry)
        safe_sql = _safe_join(repo_root, sql_rel)
        if safe_sql is None:
            continue
        try:
            text = Path(safe_sql).read_text(encoding="utf-8", errors="replace")
        except (OSError, PermissionError):
            continue
        for match in _SQL_CREATE_PATTERN.finditer(text):
            blocks.append(match.group(1).strip())
    return blocks


def _estimate_tokens(text: str) -> int:
    return max(1, len(text) // _CHARS_PER_TOKEN)


def _truncate_signatures_to_budget(
    signatures: list[tuple[str, str, str, str]],
    budget_chars: int,
) -> list[tuple[str, str, str, str]]:
    """Drop low-priority signatures until the serialized block fits in *budget_chars*.

    Signatures are already sorted by priority (entry_point first), so we trim
    from the tail (lowest priority).
    """
    total = sum(len(f"  {s[2]:16s} {s[0]} ({s[1]}) — {s[3]}") + 1 for s in signatures)
    result = list(signatures)
    while total > budget_chars and len(result) > 1:
        dropped = result.pop()  # remove lowest-priority entry
        total -= len(f"  {dropped[2]:16s} {dropped[0]} ({dropped[1]}) — {dropped[3]}") + 1
    return result


# ── Public API ────────────────────────────────────────────────────────────────


def build_pack(
    cluster: Cluster,
    kind: str = "code",
    *,
    repo_root: str,
    top_k: int = 20,
    token_budget: int = 1500,
) -> EvidencePack:
    """Build an evidence pack for *cluster*.

    Parameters
    ----------
    cluster : Cluster
        Code cluster whose ``artifacts`` (``ArtifactInfo(kind="symbol", …)``)
        are used as the evidence source.  Pass ``cluster.kind`` via the *kind*
        parameter if you need to override (rare).
    kind : str
        Defaults to ``"code"``.  Passed through to ``EvidencePack.kind``.
    repo_root : str
        Absolute path to the checked-out repository root.  All file IO is
        confined to this directory tree.
    top_k : int
        Maximum number of symbol signatures to include.
    token_budget : int
        Approximate total token budget for the serialized pack.  Truncation
        by layer priority is applied when the budget is exceeded.

    Returns
    -------
    EvidencePack
    """
    char_budget = token_budget * _CHARS_PER_TOKEN

    # ── 1. Signatures (sorted by layer priority, capped at top_k) ────────
    sorted_arts = _sorted_artifacts(cluster.artifacts)
    top_artifacts = sorted_arts[:top_k]
    signatures: list[tuple[str, str, str, str]] = [
        (a.name, a.source_path, a.layer, a.summary)
        for a in top_artifacts
    ]

    # ── 2. File heads (entry_point / public_api layers only, deduplicated) ─
    seen_paths: set[str] = set()
    file_heads: list[tuple[str, str]] = []
    for art in top_artifacts:
        if art.layer not in _FILE_HEAD_LAYERS:
            continue
        if art.source_path in seen_paths:
            continue
        seen_paths.add(art.source_path)
        head = _read_file_head(repo_root, art.source_path)
        if head is not None:
            file_heads.append((art.source_path, head))

    # ── 3. README excerpt (first hit across cluster dirs) ────────────────
    readme_excerpt = ""
    for dir_path in cluster.dirs:
        content = _find_readme(repo_root, dir_path)
        if content is not None:
            readme_excerpt = content
            break

    # ── 4. SQL CREATE TABLE blocks ────────────────────────────────────────
    sql_blocks: list[str] = []
    for dir_path in cluster.dirs:
        sql_blocks.extend(_extract_sql_blocks(repo_root, dir_path))

    # ── 5. Truncate signatures if over budget ─────────────────────────────
    # Reserve budget proportionally for each section; signatures get ~40%.
    sig_budget_chars = int(char_budget * 0.40)
    signatures = _truncate_signatures_to_budget(signatures, sig_budget_chars)

    return EvidencePack(
        cluster_id=cluster.cluster_id,
        kind=kind,
        signatures=signatures,
        file_heads=file_heads,
        readme_excerpt=readme_excerpt,
        sql_blocks=sql_blocks,
    )
