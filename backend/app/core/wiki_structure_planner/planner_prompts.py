"""Prompts for the unified planner agent (#236).

Design
------
- Drops the explore-repo framing used by the old ``structure_engine.py``.
- The agent is told that each cluster already has an evidence pack and should
  call tools ONLY when the pack is insufficient or wrong.
- The total tool budget is N = 3 × n_clusters; the prompt makes this explicit.

Public API
----------
PLANNER_SYSTEM_PROMPT            -- static system prompt (str constant).
build_user_prompt(cluster, pack) -- per-cluster user message.
"""

from __future__ import annotations

from app.core.wiki_structure_planner.evidence import EvidencePack
from app.core.wiki_structure_planner.structure_skeleton import Cluster

# ── System prompt ─────────────────────────────────────────────────────────────

PLANNER_SYSTEM_PROMPT: str = """\
You are a wiki structure planner. Your job is to name wiki pages for software \
repository clusters.

## Inputs you receive

For each cluster you will receive:
- A **cluster header** describing the cluster id, kind, directories, and languages.
- An **evidence pack** with pre-fetched content: symbol signatures, file heads, \
README excerpts, and SQL schema blocks.

## How to work

1. **Read the evidence pack first.** It contains ~1 000 tokens of pre-fetched \
   evidence that is usually sufficient to name the page accurately.
2. **Use tools sparingly.** You have a hard total tool budget across ALL clusters. \
   Call `read_file`, `get_signature`, or `grep` ONLY when:
   - The evidence pack contradicts what you expect from the directory names.
   - A critical symbol is mentioned but its purpose is unclear from the pack.
   - The pack is empty (rare).
3. **Never invent symbols or paths.** If the evidence is insufficient and \
   the budget is exhausted, produce a reasonable name from the directory names.

## Output format

For each cluster, respond with a **single JSON object** on its own line:

```json
{"cluster_id": <int>, "title": "<capability title>", \
"description": "<1-2 sentences describing the capability>", \
"retrieval_query": "<keyword phrase for retrieval>"}
```

Rules:
- `title` must be a capability-focused phrase, NOT a symbol or file name.
  - Good: "Raft Consensus Protocol"  Bad: "raft_group_manager.py"
- `description` must be 1-2 sentences summarising what the cluster implements \
  or documents (not what files it contains).
- `retrieval_query` must be 3-8 keywords useful for dense/sparse retrieval.
- Produce EXACTLY one JSON object per cluster, in cluster_id order.
- Output ONLY the JSON lines — no markdown, no extra text.
"""

# ── Kind-specific framing ─────────────────────────────────────────────────────

_KIND_FRAMES: dict[str, str] = {
    "code": (
        "This cluster contains source code files that implement a specific capability."
    ),
    "doc": (
        "This cluster contains documentation files (markdown, RST, etc.) that "
        "describe or explain a feature or component."
    ),
    "confluence": (
        "This cluster contains Confluence wiki pages that document processes, "
        "architecture decisions, or runbooks."
    ),
    "jira": (
        "This cluster contains Jira issues (epics, stories, subtasks) that track "
        "planned or in-progress work."
    ),
}

_DEFAULT_KIND_FRAME: str = "This cluster groups related artifacts."


def _kind_prompt_fragment(kind: str) -> str:
    """Return a one-sentence framing for the cluster kind.

    The fragment is injected into the user prompt so the LLM knows how to
    interpret the artifacts it is being shown.

    Args:
        kind: Cluster kind — one of ``"code"``, ``"doc"``, ``"confluence"``,
            ``"jira"``.  Unknown values fall back to a generic sentence.

    Returns:
        One-sentence framing string.
    """
    return _KIND_FRAMES.get(kind, _DEFAULT_KIND_FRAME)


# ── User prompt builder ───────────────────────────────────────────────────────

_NO_PACK_NOTICE = (
    "No evidence pack is available for this cluster. "
    "Use the directory and artifact names to produce a reasonable page title."
)


def build_user_prompt(
    cluster: Cluster,
    pack: EvidencePack | None,
    *,
    tool_budget_remaining: int,
    tool_budget_total: int,
) -> str:
    """Format a single cluster + evidence pack into a user message.

    Args:
        cluster: The cluster to plan.
        pack: Pre-fetched evidence pack, or ``None`` if unavailable.
        tool_budget_remaining: How many tool calls are still available across
            the whole planner run.
        tool_budget_total: Total budget at the start of the run.

    Returns:
        Formatted user message string.
    """
    dirs_str = ", ".join(cluster.dirs) if cluster.dirs else "(none)"
    langs_str = ", ".join(cluster.primary_languages) if cluster.primary_languages else "(unknown)"
    kind_frame = _kind_prompt_fragment(cluster.kind)

    budget_note = (
        f"Tool budget: {tool_budget_remaining}/{tool_budget_total} calls remaining "
        f"across all clusters. Use them only if this evidence pack is insufficient."
    )

    header_lines = [
        f"## Cluster {cluster.cluster_id} (kind={cluster.kind})",
        "",
        kind_frame,
        "",
        f"**Directories**: {dirs_str}",
        f"**Languages**: {langs_str}",
        f"**Artifacts**: {cluster.total_artifacts}",
        "",
        f"_{budget_note}_",
        "",
    ]

    if pack is None:
        header_lines.append(_NO_PACK_NOTICE)
        evidence_text = ""
    else:
        evidence_text = pack.serialize()

    lines = header_lines
    if evidence_text:
        lines = lines + ["### Evidence Pack", "", evidence_text, ""]

    lines.append(
        f"Respond with a JSON object for cluster_id={cluster.cluster_id}."
    )

    return "\n".join(lines)
