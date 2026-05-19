"""Unified planner agent (#236).

The only planner path after #242 collapsed the legacy LLM-driven and
graph-first alternatives.  Evidence-first: each cluster arrives with a
pre-built evidence pack (~1K tokens).  The agent may call escape-hatch
tools (``read_file``, ``get_signature``, ``grep``) only when a pack is
insufficient, subject to a hard total budget of ``3 × n_clusters``.

Public API
----------
``run_planner(skeleton, evidence_packs, *, llm, repo_root) -> list[PageSpec]``
``run_planner_with_report(...)`` — same, but also returns the ``PlanReport``.

``PlanReport``  — budget / coverage summary.
``PageSpec``    — one wiki page specification (title, description, retrieval_query,
                  plus deterministic target_* fields from the skeleton).
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage

from app.core.wiki_structure_planner._evidence_utils import safe_join
from app.core.wiki_structure_planner.evidence import EvidencePack
from app.core.wiki_structure_planner.planner_prompts import (
    PLANNER_SYSTEM_PROMPT,
    build_user_prompt,
)
from app.core.wiki_structure_planner.structure_skeleton import (
    Cluster,
    StructureSkeleton,
)

logger = logging.getLogger(__name__)

# ── PageSpec ──────────────────────────────────────────────────────────────────


@dataclass
class PageSpec:
    """Specification for one wiki page produced by the unified planner.

    Attributes
    ----------
    cluster_id : int
        ID of the originating cluster.
    title : str
        Capability-focused page title (from LLM or fallback).
    description : str
        1-2 sentence description of what this page covers.
    retrieval_query : str
        Keyword phrase for dense/sparse retrieval during content generation.
    target_symbols : list[str]
        Symbol names injected deterministically from the skeleton.
    target_folders : list[str]
        Directory paths injected deterministically from the skeleton.
    target_docs : list[str]
        Documentation file paths injected deterministically from the skeleton.
    """

    cluster_id: int
    title: str
    description: str
    retrieval_query: str
    target_symbols: list[str] = field(default_factory=list)
    target_folders: list[str] = field(default_factory=list)
    target_docs: list[str] = field(default_factory=list)


# ── PlanReport ────────────────────────────────────────────────────────────────


@dataclass
class PlanReport:
    """Budget and coverage summary for one ``run_planner`` invocation.

    Attributes
    ----------
    tool_budget_total : int
        ``3 × n_clusters`` budget set at the start of the run.
    tool_budget_used : int
        Number of tool calls the LLM actually made.
    tool_budget_exceeded : bool
        True when ``tool_budget_used >= tool_budget_total``.
    pages_emitted : int
        Number of ``PageSpec`` objects returned.
    notes : list[str]
        Human-readable warnings — e.g. missing evidence packs, malformed LLM
        output, fallback page synthesised.
    """

    tool_budget_total: int
    tool_budget_used: int
    tool_budget_exceeded: bool
    pages_emitted: int
    notes: list[str] = field(default_factory=list)


# ── Tool implementations ──────────────────────────────────────────────────────

_MAX_FILE_LINES = 100
_MAX_GREP_RESULTS = 10


def _tool_read_file(repo_root: str, path: str) -> str:
    """Read up to _MAX_FILE_LINES lines of a file inside repo_root.

    Streams the file line-by-line via ``itertools.islice`` so a multi-MB
    source doesn't get fully loaded just to drop the tail.
    """
    safe = safe_join(repo_root, path)
    if safe is None:
        return f"[error] path escapes repo_root: {path!r}"
    try:
        from itertools import islice  # noqa: PLC0415

        with open(safe, encoding="utf-8", errors="replace") as fh:
            head = list(islice(fh, _MAX_FILE_LINES))
        return "".join(head).rstrip("\n")
    except (OSError, PermissionError) as exc:
        return f"[error] {exc}"


def _tool_get_signature(repo_root: str, symbol: str) -> str:
    """Search for a symbol's definition by scanning Python/JS/TS/Go files.

    This is a lightweight grep-based fallback for the planner phase; deep graph
    queries are not available at this stage.
    """
    import subprocess  # noqa: PLC0415

    _inc = ["--include=*.py", "--include=*.js", "--include=*.ts", "--include=*.go", "--include=*.java", "--include=*.rs"]
    # ``-F`` matches the symbol literally; ``--`` terminates option parsing so
    # symbols starting with ``-`` are not misread as flags.
    cmd = ["grep", "-rn", "-F", *_inc, "-m", "5", "--", symbol, repo_root]  # noqa: S607
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=10)  # noqa: S603
        output = result.stdout.strip()
        if not output:
            return f"[not found] {symbol}"
        return output[:2000]
    except Exception as exc:  # noqa: BLE001
        return f"[error] {exc}"


def _tool_grep(repo_root: str, pattern: str) -> str:
    """Grep for a pattern across the repo; returns up to _MAX_GREP_RESULTS lines."""
    import subprocess  # noqa: PLC0415

    _inc = ["--include=*.py", "--include=*.js", "--include=*.ts", "--include=*.go",
            "--include=*.java", "--include=*.rs", "--include=*.md"]
    # ``--`` terminates option parsing so a pattern beginning with ``-`` is
    # not misread as a flag.
    cmd = ["grep", "-rn", *_inc, "-m", str(_MAX_GREP_RESULTS), "--", pattern, repo_root]  # noqa: S607
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=10)  # noqa: S603
        output = result.stdout.strip()
        if not output:
            return f"[no matches] {pattern}"
        return output[:2000]
    except Exception as exc:  # noqa: BLE001
        return f"[error] {exc}"


# ── Tool dispatch table ───────────────────────────────────────────────────────

_TOOL_SCHEMAS: list[dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "read_file",
            "description": (
                "Read up to 100 lines of a file inside the repository. "
                "Use ONLY when the evidence pack is insufficient or wrong."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "Repo-relative file path.",
                    }
                },
                "required": ["path"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_signature",
            "description": (
                "Search for a symbol's definition in the repository source. "
                "Use ONLY when a symbol's purpose is unclear from the pack."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "symbol": {
                        "type": "string",
                        "description": "Symbol name to look up.",
                    }
                },
                "required": ["symbol"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "grep",
            "description": (
                "Search for a pattern in the repository source files. "
                "Use ONLY when the evidence pack is ambiguous."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "pattern": {
                        "type": "string",
                        "description": "Text pattern to search for.",
                    }
                },
                "required": ["pattern"],
            },
        },
    },
]


def _dispatch_tool(repo_root: str, tool_name: str, args: dict[str, Any]) -> str:
    """Invoke a planner tool by name and return its text result."""
    if tool_name == "read_file":
        return _tool_read_file(repo_root, args.get("path", ""))
    if tool_name == "get_signature":
        return _tool_get_signature(repo_root, args.get("symbol", ""))
    if tool_name == "grep":
        return _tool_grep(repo_root, args.get("pattern", ""))
    return f"[error] unknown tool: {tool_name!r}"


# ── JSON extraction ───────────────────────────────────────────────────────────

_JSON_OBJ_RE = re.compile(r"\{[^{}]*\}", re.DOTALL)


def _extract_json_object(text: str) -> dict[str, Any] | None:
    """Try to extract the first JSON object from *text*.

    Strips markdown fences, then tries to parse the whole string as JSON
    before falling back to a regex scan for the first ``{...}`` block.
    A second parse attempt after trailing-comma cleanup catches common LLM
    sloppiness like ``{"title": "X",}``.  Returns ``None`` on failure.
    """
    # Strip markdown fences
    stripped = re.sub(r"```(?:json)?\s*", "", text).strip()

    def _try_parse(candidate: str) -> dict[str, Any] | None:
        try:
            obj = json.loads(candidate)
        except json.JSONDecodeError:
            # Try once more after stripping trailing commas before } or ].
            cleaned = re.sub(r",\s*([}\]])", r"\1", candidate)
            if cleaned == candidate:
                return None
            try:
                obj = json.loads(cleaned)
            except json.JSONDecodeError:
                return None
        return obj if isinstance(obj, dict) else None

    direct = _try_parse(stripped)
    if direct is not None:
        return direct

    # Try to find first {...} block
    for m in _JSON_OBJ_RE.finditer(stripped):
        candidate = _try_parse(m.group(0))
        if candidate is not None:
            return candidate

    return None


# ── Deterministic field injection ─────────────────────────────────────────────


def _inject_from_skeleton(spec: PageSpec, cluster: Cluster) -> None:
    """Populate target_* fields from the cluster — never from the LLM.

    Args:
        spec: PageSpec to mutate in-place.
        cluster: Source cluster.
    """
    spec.target_symbols = [
        a.name
        for a in cluster.artifacts
        if a.layer in {"entry_point", "public_api"}
    ]
    spec.target_folders = sorted(cluster.dirs)
    spec.target_docs = [
        a.source_path
        for a in cluster.artifacts
        if a.kind in {"doc_section", "confluence_page", "jira_issue"}
    ]


# ── Fallback PageSpec ─────────────────────────────────────────────────────────


def _fallback_spec(cluster: Cluster, note: str) -> tuple[PageSpec, str]:
    """Synthesise a minimal PageSpec when the LLM fails for a cluster.

    Pulls fallback wording from the cluster's directory list, artifact names,
    and source paths so the downstream writer never receives an empty
    ``description`` or empty ``retrieval_query``.  An empty retrieval query
    would yield no retrieval context and the writer would either hallucinate
    or produce thin content.

    Returns:
        (spec, human-readable note explaining the fallback)
    """
    dirs = list(cluster.dirs)
    dir_leaf = dirs[0].rsplit("/", 1)[-1] if dirs else "misc"
    title = f"{dir_leaf.replace('_', ' ').replace('-', ' ').title()} Components"

    # Build description with graceful degradation:
    # 1. dirs → "Components in foo/bar, baz"
    # 2. else artifact names → "Components: Foo, Bar, Baz"
    # 3. else minimal → "Cluster <id> components ({kind})"
    if dirs:
        description = f"Components in {', '.join(dirs[:3])}"
    elif cluster.artifacts:
        names = [a.name for a in cluster.artifacts[:3] if a.name]
        if names:
            description = f"Components: {', '.join(names)}"
        else:
            description = f"Cluster {cluster.cluster_id} components ({cluster.kind})"
    else:
        description = f"Cluster {cluster.cluster_id} components ({cluster.kind})"

    # Build retrieval_query with graceful degradation:
    # 1. dirs (first 4)
    # 2. else artifact names (first 4)
    # 3. else artifact source_path leaves (first 4)
    # 4. else cluster kind keyword (never empty)
    query_terms: list[str] = []
    if dirs:
        query_terms = dirs[:4]
    elif cluster.artifacts:
        query_terms = [a.name for a in cluster.artifacts[:4] if a.name]
        if not query_terms:
            query_terms = [
                a.source_path.rsplit("/", 1)[-1]
                for a in cluster.artifacts[:4]
                if a.source_path
            ]
    if not query_terms:
        query_terms = [cluster.kind, "components"]
    retrieval_query = " ".join(t for t in query_terms if t)

    spec = PageSpec(
        cluster_id=cluster.cluster_id,
        title=title,
        description=description,
        retrieval_query=retrieval_query,
    )
    _inject_from_skeleton(spec, cluster)
    return spec, note


# ── Per-cluster planner call ──────────────────────────────────────────────────


def _plan_cluster(
    cluster: Cluster,
    pack: EvidencePack | None,
    *,
    llm_with_tools: Any,
    repo_root: str,
    budget_tracker: list[int],  # [used, total] — mutated in-place
    notes: list[str],
) -> PageSpec:
    """Run the LLM planner for a single cluster with tool-loop support.

    Args:
        cluster: Cluster to plan.
        pack: Pre-fetched evidence pack or ``None``.
        llm_with_tools: LLM bound with tool schemas (or the bare LLM when
            the provider does not support ``bind_tools``).
        repo_root: Repo root for tool IO.
        budget_tracker: ``[used_count, total_budget]`` list mutated in-place.
        notes: Accumulated plan notes list, mutated in-place.

    Returns:
        PageSpec for the cluster (guaranteed non-None).
    """
    tool_budget_remaining = budget_tracker[1] - budget_tracker[0]

    user_msg = build_user_prompt(
        cluster,
        pack,
        tool_budget_remaining=tool_budget_remaining,
        tool_budget_total=budget_tracker[1],
    )

    messages: list[Any] = [
        SystemMessage(content=PLANNER_SYSTEM_PROMPT),
        HumanMessage(content=user_msg),
    ]

    # ── Agentic tool loop ─────────────────────────────────────────────────
    max_tool_rounds = min(3, max(0, tool_budget_remaining))  # at most 3 per cluster

    for _round in range(max_tool_rounds + 1):  # +1 for the final answer round
        response = llm_with_tools.invoke(messages)

        # Check for tool calls
        tool_calls = getattr(response, "tool_calls", None) or []

        if not tool_calls:
            # LLM returned a direct answer — parse and return
            content = response.content if isinstance(response.content, str) else str(response.content)
            obj = _extract_json_object(content)
            if obj is not None:
                title = (obj.get("title") or "").strip()
                description = (obj.get("description") or "").strip()
                retrieval_query = (obj.get("retrieval_query") or "").strip()

                # Reject specs missing any load-bearing field — the downstream
                # writer agent needs ALL three (title to name the page,
                # description for the TOC, retrieval_query to fetch context).
                # An empty retrieval_query yields no retrieval evidence and
                # forces the writer to either hallucinate or produce thin
                # content, so we synthesise a deterministic fallback instead.
                missing: list[str] = []
                if not title:
                    missing.append("title")
                if not description:
                    missing.append("description")
                if not retrieval_query:
                    missing.append("retrieval_query")
                if missing:
                    note = (
                        f"cluster {cluster.cluster_id}: LLM response missing "
                        f"{', '.join(missing)} — using fallback spec"
                    )
                    notes.append(note)
                    logger.warning("[PLANNER] %s", note)
                    spec, _ = _fallback_spec(cluster, note)
                    return spec

                spec = PageSpec(
                    cluster_id=cluster.cluster_id,
                    title=title,
                    description=description,
                    retrieval_query=retrieval_query,
                )
                _inject_from_skeleton(spec, cluster)
                return spec

            # LLM returned unparseable text
            note = f"cluster {cluster.cluster_id}: malformed LLM JSON — using fallback spec"
            notes.append(note)
            logger.warning("[PLANNER] %s", note)
            spec, _ = _fallback_spec(cluster, note)
            return spec

        # Process tool calls
        messages.append(response)
        for tc in tool_calls:
            tool_name = tc.get("name") if isinstance(tc, dict) else getattr(tc, "name", "")
            tool_args = tc.get("args") if isinstance(tc, dict) else getattr(tc, "args", {})
            tc_id = tc.get("id") if isinstance(tc, dict) else getattr(tc, "id", "")

            if budget_tracker[0] >= budget_tracker[1]:
                # Budget exhausted — return a tool result indicating this
                result_text = "[budget exhausted — tool call not executed]"
                note = f"cluster {cluster.cluster_id}: tool budget exhausted mid-run"
                if note not in notes:
                    notes.append(note)
                    logger.warning("[PLANNER] %s", note)
            else:
                budget_tracker[0] += 1
                logger.debug(
                    "[PLANNER] tool=%s args=%s budget=%d/%d",
                    tool_name,
                    tool_args,
                    budget_tracker[0],
                    budget_tracker[1],
                )
                result_text = _dispatch_tool(repo_root, tool_name, tool_args or {})

            messages.append(
                ToolMessage(
                    content=result_text,
                    tool_call_id=tc_id or "",
                    name=tool_name or "unknown",
                )
            )

        # After every batch of tool results, if the LLM keeps spending and
        # has now drained the global budget, drop into the fallback path
        # rather than spending another expensive LLM round that can no
        # longer use tools anyway.
        if budget_tracker[0] >= budget_tracker[1]:
            note = (
                f"cluster {cluster.cluster_id}: tool budget drained "
                f"after a tool-call batch — using fallback spec"
            )
            notes.append(note)
            logger.warning("[PLANNER] %s", note)
            spec, _ = _fallback_spec(cluster, note)
            return spec

    # Max rounds reached without a clean answer — fallback
    note = f"cluster {cluster.cluster_id}: exceeded tool rounds — using fallback spec"
    notes.append(note)
    logger.warning("[PLANNER] %s", note)
    spec, _ = _fallback_spec(cluster, note)
    return spec


# ── Public entrypoint ─────────────────────────────────────────────────────────


def _run_planner_core(
    skeleton: StructureSkeleton,
    evidence_packs: dict[int, EvidencePack],
    *,
    llm: BaseChatModel,
    repo_root: str,
) -> tuple[list[PageSpec], PlanReport]:
    """Shared implementation for ``run_planner`` / ``run_planner_with_report``.

    Resolves the cluster list, plans each cluster with a single tool budget,
    fills in any missing specs with fallbacks, and assembles the report.
    """
    import os  # noqa: PLC0415

    if not repo_root or not os.path.isdir(repo_root):
        raise ValueError(f"repo_root must be a valid directory: {repo_root!r}")

    # ── Resolve cluster list ──────────────────────────────────────────────
    if skeleton.clusters:
        clusters: list[Cluster] = skeleton.clusters
    else:
        # Backwards compat: DirCluster is a Cluster subclass
        clusters = list(skeleton.code_clusters)

    if not clusters:
        logger.info("[PLANNER] no clusters — returning empty list")
        return [], PlanReport(
            tool_budget_total=0,
            tool_budget_used=0,
            tool_budget_exceeded=False,
            pages_emitted=0,
            notes=[],
        )

    n = len(clusters)
    budget_total = 3 * n
    budget_tracker: list[int] = [0, budget_total]  # [used, total]
    notes: list[str] = []

    logger.info(
        "[PLANNER] starting: %d clusters, tool budget=%d",
        n,
        budget_total,
    )

    # ── Bind tools to LLM (unwrap wrappers first) ────────────────────────
    base_llm = _unwrap_llm(llm)
    try:
        llm_with_tools = base_llm.bind_tools(_TOOL_SCHEMAS)
    except Exception:  # noqa: BLE001
        # Some models don't support bind_tools; fall through to no-tool mode
        llm_with_tools = base_llm

    # ── Per-cluster planning ──────────────────────────────────────────────
    specs_by_id: dict[int, PageSpec] = {}

    for cluster in clusters:
        pack = evidence_packs.get(cluster.cluster_id)
        if pack is None:
            note = f"cluster {cluster.cluster_id} had no evidence pack — falling back to skeleton-only"
            notes.append(note)
            logger.warning("[PLANNER] %s", note)

        try:
            spec = _plan_cluster(
                cluster=cluster,
                pack=pack,
                llm_with_tools=llm_with_tools,
                repo_root=repo_root,
                budget_tracker=budget_tracker,
                notes=notes,
            )
        except Exception as exc:  # noqa: BLE001
            note = f"cluster {cluster.cluster_id}: LLM call failed ({exc}) — using fallback spec"
            notes.append(note)
            logger.error("[PLANNER] %s", note, exc_info=True)
            spec, _ = _fallback_spec(cluster, note)

        specs_by_id[cluster.cluster_id] = spec

    # ── Validation: ensure every cluster has a PageSpec ──────────────────
    for cluster in clusters:
        if cluster.cluster_id not in specs_by_id:
            note = f"cluster {cluster.cluster_id}: missing spec — synthesising fallback"
            notes.append(note)
            logger.warning("[PLANNER] %s", note)
            spec, _ = _fallback_spec(cluster, note)
            specs_by_id[cluster.cluster_id] = spec

    # ── Assemble and log report ───────────────────────────────────────────
    tool_budget_used = budget_tracker[0]
    tool_budget_exceeded = tool_budget_used >= budget_total

    if tool_budget_exceeded:
        logger.warning(
            "[PLANNER] tool budget exceeded: used=%d total=%d",
            tool_budget_used,
            budget_total,
        )

    pages = [specs_by_id[c.cluster_id] for c in clusters]

    report = PlanReport(
        tool_budget_total=budget_total,
        tool_budget_used=tool_budget_used,
        tool_budget_exceeded=tool_budget_exceeded,
        pages_emitted=len(pages),
        notes=notes,
    )

    logger.info(
        "[PLANNER] done: pages=%d tool_calls=%d/%d exceeded=%s notes=%d",
        report.pages_emitted,
        report.tool_budget_used,
        report.tool_budget_total,
        report.tool_budget_exceeded,
        len(report.notes),
    )

    return pages, report


def run_planner(
    skeleton: StructureSkeleton,
    evidence_packs: dict[int, EvidencePack],
    *,
    llm: BaseChatModel,
    repo_root: str,
) -> list[PageSpec]:
    """Run the unified planner agent over all clusters in *skeleton*.

    The planner names each cluster, writes a 1-2 sentence description, and
    produces a ``retrieval_query``.  The ``target_symbols``, ``target_folders``,
    and ``target_docs`` fields are injected deterministically from the skeleton —
    the LLM never produces them.

    The cluster list is resolved from ``skeleton.clusters`` (preferred) or
    ``skeleton.code_clusters`` (backwards-compat fallback).  The total tool
    budget is ``3 × len(resolved_clusters)`` and is shared across the whole
    run; ``_plan_cluster`` decrements it as ``read_file`` / ``get_signature``
    / ``grep`` calls fire, and once the budget hits zero the agent must
    answer from the evidence pack alone.

    Args:
        skeleton: Completed structure skeleton.
        evidence_packs: Pre-built evidence packs keyed by ``cluster_id``.
        llm: Pre-configured LangChain chat model.
        repo_root: Absolute path to the checked-out repository root.

    Returns:
        List of ``PageSpec`` objects — one per resolved cluster.  Empty when
        the skeleton has neither ``clusters`` nor ``code_clusters``; otherwise
        one spec per cluster (fallbacks synthesised for any cluster the LLM
        could not produce a spec for).

    Raises:
        ValueError: If *repo_root* is not a valid directory.
    """
    pages, _report = _run_planner_core(
        skeleton, evidence_packs, llm=llm, repo_root=repo_root
    )
    return pages


def run_planner_with_report(
    skeleton: StructureSkeleton,
    evidence_packs: dict[int, EvidencePack],
    *,
    llm: BaseChatModel,
    repo_root: str,
) -> tuple[list[PageSpec], PlanReport]:
    """Same as :func:`run_planner` but also returns the ``PlanReport``.

    Preferred by callers that need budget / coverage metrics.
    """
    return _run_planner_core(
        skeleton, evidence_packs, llm=llm, repo_root=repo_root
    )


# ── LLM unwrapping helper ─────────────────────────────────────────────────────


def _unwrap_llm(llm: Any) -> BaseChatModel:
    """Unwrap RunnableRetry / RunnableBinding wrappers around a chat model.

    Mirrors ``WikiStructurePlannerEngine._unwrap_chat_model`` to avoid the same
    ``unhashable type: 'RunnableRetry'`` crash that hit the old engine.
    """
    current = llm
    for _ in range(8):
        if isinstance(current, BaseChatModel):
            return current
        inner = getattr(current, "bound", None)
        if inner is None or inner is current:
            break
        current = inner
    # Return as-is; deepagents or bind_tools will surface a clear error
    return current  # type: ignore[return-value]
