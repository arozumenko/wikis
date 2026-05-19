"""Evidence pack builder for Jira-epic clusters (#235).

Deterministic, parallel-friendly prefetch of ~1000 tokens of real Jira
evidence per cluster, so the planner LLM names and describes Jira clusters
from actual content (epic title, first paragraph, child issue table, labels)
rather than blank summaries.

Public API
----------
build_jira_pack(cluster, *, repo_root, token_budget) -> JiraEvidencePack

JiraEvidencePack
    .epic_key         : str   — Jira issue key of the representative epic
    .epic_title       : str   — title from file H1 or ArtifactInfo.summary
    .epic_description : str   — first paragraph from the epic's disk file
    .child_issues     : list[dict]  — key/type/status/summary rows (no bodies)
    .labels           : list[str]   — from epic's YAML frontmatter
    .components       : list[str]   — from epic's YAML frontmatter
    .fix_versions     : list[str]   — from epic's YAML frontmatter
    .serialize()      : str — render all sections for prompt injection

Design choices
--------------
* Separate dataclass (not extending EvidencePack) — code-specific fields
  (signatures, sql_blocks, file_heads) don't apply to Jira clusters.
* New file evidence_jira.py — keeps concurrent PRs #233 (markdown) and #234
  (confluence) non-conflicting.  A follow-up dispatcher consolidates after
  all three merge (#243).
* Epic representative selection: first artifact whose issue_type (case-
  insensitive) equals "epic"; falls back to the first artifact if none.
* Frontmatter parsed with a minimal YAML-like scanner — no PyYAML dep;
  handles scalar and list values for labels/components/fix_versions.
* _safe_join pattern reused from evidence.py (canonical security primitive).
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from pathlib import Path

from ._evidence_utils import parse_frontmatter, safe_join
from .structure_skeleton import ArtifactInfo, Cluster

logger = logging.getLogger(__name__)

# ── Constants ─────────────────────────────────────────────────────────────────

# Approximate chars per token — conservative.
_CHARS_PER_TOKEN = 4

# Hard cap on epic_description (chars).
_FIRST_PARA_CHAR_CAP = 400

# Maximum child rows before we start truncating.
_MAX_CHILD_ROWS = 50


# ── JiraEvidencePack ──────────────────────────────────────────────────────────


@dataclass
class JiraEvidencePack:
    """Prefetched evidence for one Jira-epic cluster, ready for prompt injection.

    Attributes
    ----------
    cluster_id : int
    epic_key : str
        Jira issue key of the representative epic (e.g. ``"PROJ-1"``).
    epic_title : str
        Title extracted from the epic's disk file H1 or ArtifactInfo.summary.
    epic_description : str
        First non-empty paragraph from the epic's disk file body, ≤400 chars.
    child_issues : list[dict]
        One dict per non-epic artifact.  Keys: ``key``, ``type``, ``status``,
        ``summary``.  Descriptions / body text are intentionally excluded.
    labels : list[str]
        Labels from epic's YAML frontmatter, or empty list.
    components : list[str]
        Components from epic's YAML frontmatter, or empty list.
    fix_versions : list[str]
        Fix versions from epic's YAML frontmatter, or empty list.
    """

    cluster_id: int
    epic_key: str
    epic_title: str
    epic_description: str = ""
    child_issues: list[dict] = field(default_factory=list)
    labels: list[str] = field(default_factory=list)
    components: list[str] = field(default_factory=list)
    fix_versions: list[str] = field(default_factory=list)

    def serialize(self) -> str:
        """Render all sections as a prompt-ready string."""
        parts: list[str] = [f"[JiraEvidencePack cluster={self.cluster_id} kind=jira]"]

        parts.append(f"\nEpic: {self.epic_key}")
        if self.epic_title:
            parts.append(f"Title: {self.epic_title}")
        if self.epic_description:
            parts.append(f"Description: {self.epic_description}")

        if self.labels:
            parts.append(f"Labels: {', '.join(self.labels)}")
        if self.components:
            parts.append(f"Components: {', '.join(self.components)}")
        if self.fix_versions:
            parts.append(f"Fix Versions: {', '.join(self.fix_versions)}")

        if self.child_issues:
            parts.append("\nChild Issues:")
            parts.append("| Key | Type | Status | Summary |")
            parts.append("|-----|------|--------|---------|")
            for child in self.child_issues:
                key = _escape_table_cell(child.get("key", ""))
                ctype = _escape_table_cell(child.get("type", ""))
                status = _escape_table_cell(child.get("status", ""))
                summary = _escape_table_cell(child.get("summary", ""))
                parts.append(f"| {key} | {ctype} | {status} | {summary} |")

        return "\n".join(parts)


# ── Internal helpers ──────────────────────────────────────────────────────────


def _escape_table_cell(value: str) -> str:
    """Make *value* safe to embed in a Markdown table cell.

    Jira summaries / statuses can contain `|` (column separator) or newlines
    (row separator); both break the table format and bloat token count. Replace
    newlines with spaces and escape any literal pipe.
    """
    return value.replace("\\", "\\\\").replace("|", "\\|").replace("\n", " ").replace("\r", " ")


def _read_file_safe(repo_root: str, rel_path: str) -> str | None:
    """Read the full content of *rel_path* under *repo_root*, or None on error."""
    safe = safe_join(repo_root, rel_path)
    if safe is None:
        logger.debug("evidence_jira: path escape rejected: %r", rel_path)
        return None
    try:
        return Path(safe).read_text(encoding="utf-8", errors="replace")
    except (OSError, PermissionError):
        return None


_H1_RE = re.compile(r"^#\s+(.+)$", re.MULTILINE)


def _extract_first_paragraph(body: str) -> str:
    """Return the first non-empty, non-heading paragraph from *body*."""
    # Split on blank lines
    paragraphs = re.split(r"\n\s*\n", body)
    for para in paragraphs:
        stripped = para.strip()
        if not stripped:
            continue
        # Skip heading lines (start with #)
        if stripped.startswith("#"):
            continue
        return stripped[:_FIRST_PARA_CHAR_CAP]
    return ""


def _extract_title(frontmatter: dict, body: str, artifact: ArtifactInfo) -> str:
    """Extract the best title for the epic.

    Priority: frontmatter summary → first H1 in body → artifact.summary →
    artifact.name.
    """
    summary = frontmatter.get("summary", "")
    if summary:
        return str(summary).strip()

    h1 = _H1_RE.search(body)
    if h1:
        return h1.group(1).strip()

    if artifact.summary:
        return artifact.summary

    return artifact.name


def _build_child_row(artifact: ArtifactInfo, repo_root: str) -> dict:
    """Build a child issue dict from *artifact* and its optional disk file."""
    key = artifact.name
    issue_type = artifact.issue_type or "unknown"
    summary = artifact.summary or ""

    # Try to get status from frontmatter
    status = ""
    if artifact.source_path:
        text = _read_file_safe(repo_root, artifact.source_path)
        if text:
            fm, _ = parse_frontmatter(text, inline_lists=True, strip_quotes=True)
            status = str(fm.get("status", "")).strip()
            # Fall back to frontmatter summary if artifact.summary is empty
            if not summary:
                summary = str(fm.get("summary", "")).strip()

    return {
        "key": key,
        "type": issue_type,
        "status": status,
        "summary": summary,
    }


def _pick_epic(artifacts: list[ArtifactInfo]) -> ArtifactInfo | None:
    """Return the first artifact with issue_type=='epic' (case-insensitive).

    Falls back to the first artifact in the list when no epic is found.
    Returns None for an empty list.
    """
    for artifact in artifacts:
        if artifact.issue_type.lower() == "epic":
            return artifact
    return artifacts[0] if artifacts else None


# ── Public API ────────────────────────────────────────────────────────────────


def build_jira_pack(
    cluster: Cluster,
    *,
    repo_root: str,
    token_budget: int = 1000,
) -> JiraEvidencePack:
    """Build a Jira evidence pack for *cluster*.

    Parameters
    ----------
    cluster : Cluster
        Jira cluster (``cluster.kind == "jira"``) whose ``artifacts`` are
        ``ArtifactInfo(kind="jira_issue", …)`` entries.  Other artifact
        kinds in a mixed cluster are silently ignored.
    repo_root : str
        Absolute path to the checked-out repository root.  All file IO is
        confined to this directory tree via ``_safe_join``.
    token_budget : int
        Approximate total token budget for the serialised pack.  Defaults to
        1000 tokens (~4000 chars).

    Returns
    -------
    JiraEvidencePack
    """
    # Filter to jira_issue artifacts only
    jira_artifacts = [a for a in cluster.artifacts if a.kind == "jira_issue"]

    if not jira_artifacts:
        return JiraEvidencePack(
            cluster_id=cluster.cluster_id,
            epic_key="",
            epic_title="",
        )

    # Select representative epic (guaranteed non-None because jira_artifacts is non-empty)
    epic = _pick_epic(jira_artifacts)
    if epic is None:
        return JiraEvidencePack(cluster_id=cluster.cluster_id, epic_key="", epic_title="")

    # Build epic fields from disk file
    epic_key = epic.name
    epic_title = epic.summary or epic.name
    epic_description = ""
    labels: list[str] = []
    components: list[str] = []
    fix_versions: list[str] = []

    if epic.source_path:
        text = _read_file_safe(repo_root, epic.source_path)
        if text:
            fm, body = parse_frontmatter(text, inline_lists=True, strip_quotes=True)

            epic_title = _extract_title(fm, body, epic)
            epic_description = _extract_first_paragraph(body)

            raw_labels = fm.get("labels", [])
            labels = list(raw_labels) if isinstance(raw_labels, list) else []

            raw_components = fm.get("components", [])
            components = list(raw_components) if isinstance(raw_components, list) else []

            raw_fix_versions = fm.get("fix_versions", [])
            fix_versions = list(raw_fix_versions) if isinstance(raw_fix_versions, list) else []

    # Collect child issues (everything that is not the epic representative)
    child_artifacts = [a for a in jira_artifacts if a is not epic]

    # Build child rows — cap at _MAX_CHILD_ROWS to respect token budget
    child_issues: list[dict] = []
    char_budget = token_budget * _CHARS_PER_TOKEN
    for artifact in child_artifacts[:_MAX_CHILD_ROWS]:
        row = _build_child_row(artifact, repo_root)
        child_issues.append(row)

    pack = JiraEvidencePack(
        cluster_id=cluster.cluster_id,
        epic_key=epic_key,
        epic_title=epic_title,
        epic_description=epic_description,
        child_issues=child_issues,
        labels=labels,
        components=components,
        fix_versions=fix_versions,
    )

    # Enforce token budget — trim child rows from the tail if needed
    while len(pack.serialize()) > char_budget and len(pack.child_issues) > 0:
        pack.child_issues.pop()

    return pack
