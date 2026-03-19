"""
LLM Refinement for Structure Skeleton

Phase 2 of the Graph-First approach.  The LLM receives only fixed-size
cluster *summaries* (≈5 lines each, regardless of symbol count).  It
outputs names, section grouping, and descriptions.  Symbols and folders
are injected programmatically from the skeleton — zero truncation, 100%
symbol coverage by construction.

For very large repos (200+ clusters) the prompt is automatically batched
into groups of ~60 clusters and the results deterministically merged.
"""

from __future__ import annotations

import json
import logging
import os
import re
from collections import Counter, defaultdict
from typing import Any

from .structure_skeleton import (
    DirCluster,
    DocCluster,
    StructureSkeleton,
    SymbolInfo,
)

logger = logging.getLogger(__name__)

# Maximum clusters per LLM call before batching kicks in
_BATCH_SIZE = 60

# =====================================================================
# Symbol prioritisation for content expansion budget
# =====================================================================

# Layer weights mirror architectural importance.
# Higher weight ⇒ symbol is expanded first during content generation.
_LAYER_WEIGHTS: dict[str, int] = {
    "entry_point": 6,
    "public_api": 5,
    "core_type": 4,
    "infrastructure": 3,
    "internal": 2,
    "constant": 1,
}

# Default cap: keeps most pages within simple/hierarchical expansion
# budget (~30 symbols × ~5 expanded docs × ~500 tokens ≈ 75K tokens).
_DEFAULT_MAX_TARGET_SYMBOLS = 30


def _score_symbol(sym: SymbolInfo) -> float:
    """Score a symbol for page-level prioritisation.

    Formula:  layer_weight × 200 + min(connections, 100)

    The 200× multiplier guarantees that layer always dominates:
    the worst entry_point (6×200=1200) still outscores the best
    constant (1×200+100=300).  Connections serve as tie-breaker
    within the same layer.
    """
    layer_w = _LAYER_WEIGHTS.get(sym.layer, 2)
    return layer_w * 200 + min(sym.connections, 100)


def prioritize_symbols(
    symbols: list[SymbolInfo],
    max_symbols: int | None = None,
) -> list[str]:
    """Return the top-N most architecturally important symbol *names*.

    If the cluster has ≤ *max_symbols* symbols, all are returned (no
    filtering).  Otherwise symbols are scored and ranked.  Remaining
    symbols are still reachable via ``target_folders`` and graph
    expansion from the top symbols.

    Args:
        symbols: All symbols in a cluster.
        max_symbols: Cap per page.  ``None`` → env-configured or default.

    Returns:
        List of symbol names, most important first.
    """
    if max_symbols is None:
        max_symbols = int(os.environ.get("WIKIS_MAX_TARGET_SYMBOLS", _DEFAULT_MAX_TARGET_SYMBOLS))

    if len(symbols) <= max_symbols:
        # No pruning needed — keep all, but still sort for determinism
        ranked = sorted(symbols, key=_score_symbol, reverse=True)
        return [s.name for s in ranked]

    ranked = sorted(symbols, key=_score_symbol, reverse=True)
    selected = ranked[:max_symbols]

    # Deduplicate names (same symbol name from different files)
    seen: set = set()
    unique_names: list[str] = []
    for s in selected:
        if s.name not in seen:
            seen.add(s.name)
            unique_names.append(s.name)

    pruned = len(symbols) - len(unique_names)
    if pruned > 0:
        top_scores = ", ".join(f"{s.name}({s.layer}:{s.connections})" for s in selected[:5])
        logger.info(
            "[REFINER][PRIORITIZE] Cluster pruned %d → %d symbols (cap=%d). Top-5: %s",
            len(symbols),
            len(unique_names),
            max_symbols,
            top_scores,
        )

    return unique_names


# =====================================================================
# 2a.  Cluster summary generation  (deterministic – no LLM)
# =====================================================================


def summarize_cluster(cluster: DirCluster) -> str:
    """Produce a fixed-size text summary of a DirCluster.

    Output is ≈3-5 lines regardless of how many symbols the cluster has.
    The LLM never sees individual symbol lists.
    """
    # 1. Type distribution (1 line)
    type_counts = Counter(sym.type for sym in cluster.symbols)
    type_summary = ", ".join(f"{count} {typ}{'s' if count > 1 else ''}" for typ, count in type_counts.most_common())

    # 2. Directory list (bounded: Louvain yields ~3-8 dirs per cluster)
    dir_list = ", ".join(cluster.dirs)

    # 3. Language breakdown (1 line)
    lang_counts = Counter(sym.rel_path.rsplit(".", 1)[-1].lower() for sym in cluster.symbols if "." in sym.rel_path)
    total = max(sum(lang_counts.values()), 1)
    lang_summary = ", ".join(f"{ext} ({count * 100 // total}%)" for ext, count in lang_counts.most_common(3))

    # 4. Representative symbols: top 3 by connection degree + docstrings
    top_syms = sorted(cluster.symbols, key=lambda s: -s.connections)[:3]
    representatives: list[str] = []
    for sym in top_syms:
        desc = sym.docstring.split("\n")[0][:120] if sym.docstring else f"{sym.type} in {sym.rel_path}"
        representatives.append(f"    {sym.name} — {desc}")

    return (
        f"Cluster {cluster.cluster_id}: "
        f"{cluster.total_symbols} symbols ({type_summary})\n"
        f"  Dirs: {dir_list}\n"
        f"  Languages: {lang_summary}\n"
        f"  Key symbols:\n" + "\n".join(representatives)
    )


# =====================================================================
# 2b.  LLM refinement  (single or batched)
# =====================================================================


def refine_with_llm(
    skeleton: StructureSkeleton,
    llm,
    repo_name: str,
    page_budget: int,
) -> dict[str, Any]:
    """Refine the skeleton with an LLM – naming, sectioning, describing.

    For ≤ _BATCH_SIZE clusters → single call.
    For > _BATCH_SIZE → automatic batched processing + merge.

    Returns a dict ready for ``WikiStructureSpec.model_validate()``.
    """
    clusters = skeleton.code_clusters
    if not clusters:
        # Doc-only repo: skip LLM refinement, build structure from docs
        return _build_doc_only_structure(skeleton, repo_name)

    if len(clusters) <= _BATCH_SIZE:
        llm_items = _single_llm_refinement(
            skeleton,
            llm,
            repo_name,
            page_budget,
        )
    else:
        llm_items = _batched_llm_refinement(
            skeleton,
            llm,
            repo_name,
            page_budget,
        )

    return assemble_wiki_structure(llm_items, skeleton, repo_name)


def _single_llm_refinement(
    skeleton: StructureSkeleton,
    llm,
    repo_name: str,
    page_budget: int,
    batch_info: str | None = None,
) -> list[dict[str, Any]]:
    """Single LLM call for naming + organising clusters."""
    summaries = [summarize_cluster(c) for c in skeleton.code_clusters]
    doc_section = _format_doc_clusters_for_prompt(skeleton.doc_clusters)
    doc_instruction = _format_doc_assignment_instruction(skeleton.doc_clusters)

    prompt = f"""# Name and Organise Wiki Pages for {repo_name}
{f"({batch_info})" if batch_info else ""}

Below are {len(skeleton.code_clusters)} pre-computed code clusters from \
graph-based community detection. Each cluster will become one wiki page. \
Your job:

1. **Name** each cluster with a CAPABILITY-focused title (NO symbol names \
in titles).
2. **Group** clusters into sections (3-8 sections for the whole wiki).
3. **Describe** each page briefly (1-2 sentences: what capability it documents).
4. **Write retrieval_query** for each page (keywords for doc search).
{doc_instruction}

Output a JSON array. Each element:
```json
{{"cluster_id": <int or "doc_N">, "page_name": "<capability title>", \
"section_name": "<section>", "description": "<1-2 sentences>", \
"retrieval_query": "<keywords>"}}
```

## Clusters

{chr(10).join(summaries)}
{doc_section}

IMPORTANT:
- Every code cluster_id MUST appear exactly once in your output.
{("- Every doc directory (doc_0, doc_1, ...) MUST have a standalone page entry." + chr(10)) if skeleton.doc_clusters else ""}\
- Page names MUST describe CAPABILITIES, not symbol names.
  Good: "Raft Consensus Protocol"   Bad: "raft_group_manager"
- Section names should group by feature area.
- Output ONLY the JSON array, no other text.
"""

    response = llm.invoke(prompt)
    return _parse_llm_json(response.content)


def _batched_llm_refinement(
    skeleton: StructureSkeleton,
    llm,
    repo_name: str,
    page_budget: int,
) -> list[dict[str, Any]]:
    """Process > _BATCH_SIZE clusters in batches and merge."""
    clusters = skeleton.code_clusters
    n_batches = (len(clusters) + _BATCH_SIZE - 1) // _BATCH_SIZE

    all_items: list[dict[str, Any]] = []
    for batch_idx in range(n_batches):
        start = batch_idx * _BATCH_SIZE
        end = min(start + _BATCH_SIZE, len(clusters))
        batch_clusters = clusters[start:end]

        batch_skeleton = StructureSkeleton(
            code_clusters=batch_clusters,
            doc_clusters=skeleton.doc_clusters if batch_idx == 0 else [],
            total_arch_symbols=sum(c.total_symbols for c in batch_clusters),
            total_dirs_covered=sum(len(c.dirs) for c in batch_clusters),
            total_dirs_in_repo=skeleton.total_dirs_in_repo,
            repo_languages=skeleton.repo_languages,
            effective_depth=skeleton.effective_depth,
            repo_name=repo_name,
        )

        items = _single_llm_refinement(
            batch_skeleton,
            llm,
            repo_name,
            page_budget,
            batch_info=f"Batch {batch_idx + 1}/{n_batches}",
        )
        all_items.extend(items)
        logger.info(
            "[REFINER] Batch %d/%d → %d items",
            batch_idx + 1,
            n_batches,
            len(items),
        )

    return all_items


# =====================================================================
# 2c.  Assemble WikiStructureSpec from LLM output + skeleton
# =====================================================================


def assemble_wiki_structure(
    llm_items: list[dict[str, Any]],
    skeleton: StructureSkeleton,
    repo_name: str,
) -> dict[str, Any]:
    """Combine LLM names with deterministic cluster data.

    The LLM provides: page_name, section_name, description, retrieval_query.
    Code injects:      target_symbols, target_folders (from skeleton).
    This guarantees 100% coverage regardless of LLM behaviour.
    """
    cluster_by_id = {c.cluster_id: c for c in skeleton.code_clusters}
    sections_map: dict[str, list[dict]] = defaultdict(list)

    covered_ids: set = set()
    for item in llm_items:
        cid = item.get("cluster_id")
        cluster = cluster_by_id.get(cid)
        if cluster is None:
            continue
        covered_ids.add(cid)

        page = {
            "page_name": item.get("page_name", f"Cluster {cid}"),
            "description": item.get("description", ""),
            "content_focus": item.get("description", ""),
            "rationale": (f"Graph cluster {cid}: {len(cluster.dirs)} dirs, {cluster.total_symbols} symbols"),
            "retrieval_query": item.get("retrieval_query", ""),
            # DETERMINISTIC – injected from skeleton, never from LLM.
            # Symbols are ranked by architectural importance (layer + degree)
            # and capped to prevent content-expansion explosion.
            "target_symbols": prioritize_symbols(cluster.symbols),
            "target_folders": list(cluster.dirs),
            "target_docs": [],
            "key_files": [],
        }
        section_name = item.get("section_name", "General")
        sections_map[section_name].append(page)

    # ── Handle orphan clusters (LLM missed some cluster_ids) ──────────
    for cluster in skeleton.code_clusters:
        if cluster.cluster_id in covered_ids:
            continue
        logger.warning(
            "[REFINER] Orphan cluster %d (%d symbols) – auto-naming",
            cluster.cluster_id,
            cluster.total_symbols,
        )
        dir_leaf = cluster.dirs[0].rsplit("/", 1)[-1] if cluster.dirs else "misc"
        page = {
            "page_name": f"{dir_leaf.replace('_', ' ').title()} Components",
            "description": f"Components in {', '.join(cluster.dirs[:3])}",
            "content_focus": "",
            "rationale": f"Auto-named orphan cluster {cluster.cluster_id}",
            "retrieval_query": " ".join(cluster.dirs),
            "target_symbols": prioritize_symbols(cluster.symbols),
            "target_folders": list(cluster.dirs),
            "target_docs": [],
            "key_files": [],
        }
        sections_map["Additional Components"].append(page)

    # ── Handle LLM-created doc pages (cluster_id = "doc_0", …) ────────
    doc_cluster_by_idx = {i: dc for i, dc in enumerate(skeleton.doc_clusters)}
    covered_doc_indices: set = set()

    for item in llm_items:
        cid = item.get("cluster_id")
        if not isinstance(cid, str) or not cid.startswith("doc_"):
            continue
        try:
            doc_idx = int(cid.split("_", 1)[1])
        except (ValueError, IndexError):
            continue
        dc = doc_cluster_by_idx.get(doc_idx)
        if dc is None:
            continue
        covered_doc_indices.add(doc_idx)

        page = {
            "page_name": item.get(
                "page_name",
                f"{dc.dir_path.rsplit('/', 1)[-1].replace('_', ' ').title()} Documentation",
            ),
            "description": item.get(
                "description",
                f"Documentation in {dc.dir_path} ({dc.file_count} files)",
            ),
            "content_focus": item.get("description", ""),
            "rationale": f"LLM doc page for doc_{doc_idx}: {dc.dir_path}",
            "retrieval_query": item.get("retrieval_query", dc.dir_path),
            "target_symbols": [],
            "target_folders": [dc.dir_path],
            "target_docs": dc.doc_files[:20],
            "key_files": dc.doc_files[:10],
        }
        section_name = item.get("section_name", "Documentation")
        sections_map[section_name].append(page)

    # Also mark doc clusters as covered if the LLM assigned their
    # files to code pages (not requested, but handle gracefully).
    for item in llm_items:
        cid = item.get("cluster_id")
        if isinstance(cid, str) and cid.startswith("doc_"):
            continue  # Already handled above
        assigned_docs = set(item.get("doc_files", []))
        if not assigned_docs:
            continue
        for i, dc in doc_cluster_by_idx.items():
            if i in covered_doc_indices:
                continue
            if assigned_docs & set(dc.doc_files):
                covered_doc_indices.add(i)

    # ── Safety-net: add pages for uncovered doc clusters only ─────────
    if skeleton.doc_clusters:
        uncovered = [dc for i, dc in enumerate(skeleton.doc_clusters) if i not in covered_doc_indices]
        if uncovered:
            logger.warning(
                "[REFINER] %d/%d doc clusters not covered by LLM — adding fallback pages",
                len(uncovered),
                len(skeleton.doc_clusters),
            )
            _add_doc_pages(sections_map, uncovered)

    # ── Assemble sections ─────────────────────────────────────────────
    sections: list[dict] = []
    for order, (section_name, pages) in enumerate(
        sorted(sections_map.items()),
        start=1,
    ):
        for page_order, page in enumerate(pages, start=1):
            page["page_order"] = page_order
        sections.append(
            {
                "section_name": section_name,
                "section_order": order,
                "description": f"Pages covering {section_name.lower()} capabilities",
                "rationale": "Grouped by graph-first LLM refinement",
                "pages": pages,
            }
        )

    total_pages = sum(len(s["pages"]) for s in sections)

    # ── Summary: total vs. prioritised symbol counts ──────────────────
    total_arch = skeleton.total_arch_symbols
    total_target = sum(len(p.get("target_symbols", [])) for s in sections for p in s["pages"])
    max_cap = int(os.environ.get("WIKIS_MAX_TARGET_SYMBOLS", _DEFAULT_MAX_TARGET_SYMBOLS))
    logger.info(
        "[REFINER][ASSEMBLE] %d pages, %d/%d arch symbols selected as "
        "target_symbols (cap=%d/page). Remaining covered by target_folders.",
        total_pages,
        total_target,
        total_arch,
        max_cap,
    )

    return {
        "wiki_title": f"{repo_name or 'Repository'} Documentation",
        "overview": f"Auto-generated documentation for {repo_name}",
        "sections": sections,
        "total_pages": total_pages,
    }


# =====================================================================
# Doc-only structure (repos with zero code symbols)
# =====================================================================


def _build_doc_only_structure(
    skeleton: StructureSkeleton,
    repo_name: str,
) -> dict[str, Any]:
    """Build WikiStructureSpec for a docs-only repository."""
    sections_map: dict[str, list[dict]] = defaultdict(list)
    _add_doc_pages(sections_map, skeleton.doc_clusters)

    if not sections_map:
        # Absolute fallback – one page covering the repo root
        sections_map["Documentation"].append(
            {
                "page_name": "Repository Documentation",
                "description": "Overview of the repository documentation",
                "content_focus": "General documentation",
                "rationale": "Fallback for doc-only repo with no clusters",
                "retrieval_query": repo_name,
                "target_symbols": [],
                "target_folders": ["."],
                "target_docs": [],
                "key_files": [],
                "page_order": 1,
            }
        )

    sections: list[dict] = []
    for order, (sec_name, pages) in enumerate(
        sorted(sections_map.items()),
        start=1,
    ):
        for po, page in enumerate(pages, start=1):
            page["page_order"] = po
        sections.append(
            {
                "section_name": sec_name,
                "section_order": order,
                "description": f"Documentation under {sec_name}",
                "rationale": "Grouped from documentation directory structure",
                "pages": pages,
            }
        )

    return {
        "wiki_title": f"{repo_name or 'Repository'} Documentation",
        "overview": f"Documentation for {repo_name}",
        "sections": sections,
        "total_pages": sum(len(s["pages"]) for s in sections),
    }


def _add_doc_pages(
    sections_map: dict[str, list[dict]],
    doc_clusters: list[DocCluster],
) -> None:
    """Insert pages from doc-only directories into sections_map."""
    for dc in doc_clusters:
        parts = dc.dir_path.split("/")
        section_name = (
            parts[0].replace("_", " ").replace("-", " ").title() if parts and parts[0] != "." else "Documentation"
        )
        leaf = parts[-1] if parts else dc.dir_path
        page_name = f"{leaf.replace('_', ' ').replace('-', ' ').title()} Documentation"

        sections_map[section_name].append(
            {
                "page_name": page_name,
                "description": (f"Documentation files in {dc.dir_path} ({dc.file_count} files)"),
                "content_focus": f"Documentation in {dc.dir_path}",
                "rationale": f"Doc-only directory: {dc.dir_path}",
                "retrieval_query": dc.dir_path,
                "target_symbols": [],
                "target_folders": [dc.dir_path],
                "target_docs": dc.doc_files[:20],  # cap list length
                "key_files": dc.doc_files[:10],
            }
        )


# =====================================================================
# Helpers
# =====================================================================


def _format_doc_clusters_for_prompt(doc_clusters: list[DocCluster]) -> str:
    """Format doc clusters into a prompt section.

    Presents each doc cluster as a nameable item the LLM must create
    a standalone page for.
    """
    if not doc_clusters:
        return ""
    lines = [
        "",
        "## Documentation-Only Directories",
        "",
        "These directories contain ONLY documentation (no code symbols).",
        "Create a STANDALONE PAGE for each directory below.",
        'Add a JSON object with `"cluster_id": "doc_<N>"` (matching the '
        "index), a capability-focused `page_name`, `section_name`, "
        "`description`, and `retrieval_query`.",
        "",
    ]
    for i, dc in enumerate(doc_clusters):
        files_preview = ", ".join(dc.doc_files[:5])
        if len(dc.doc_files) > 5:
            files_preview += f", ... (+{len(dc.doc_files) - 5} more)"
        lines.append(f"- **doc_{i}**: `{dc.dir_path}/` — {dc.file_count} files ({', '.join(dc.doc_types)})")
        lines.append(f"  Files: {files_preview}")
    return "\n".join(lines)


def _format_doc_assignment_instruction(
    doc_clusters: list[DocCluster],
) -> str:
    """Instruction added to the prompt when doc clusters exist."""
    if not doc_clusters:
        return ""
    return (
        "5. **Handle documentation directories**: For each doc directory "
        "listed below, create a standalone page entry (use "
        '`"cluster_id": "doc_<N>"`) with a capability-focused name.'
    )


def _parse_llm_json(content: str) -> list[dict[str, Any]]:
    """Best-effort extraction of a JSON array from LLM response text."""
    if not content:
        return []
    # Strip markdown code fences
    text = re.sub(r"```(?:json)?\s*", "", content)
    text = text.strip()

    # Try direct parse first
    try:
        parsed = json.loads(text)
        if isinstance(parsed, list):
            return parsed
        if isinstance(parsed, dict) and "clusters" in parsed:
            return parsed["clusters"]
        return [parsed]
    except json.JSONDecodeError:
        pass

    # Try extracting the first JSON array from the text
    bracket_start = text.find("[")
    bracket_end = text.rfind("]")
    if bracket_start != -1 and bracket_end > bracket_start:
        try:
            return json.loads(text[bracket_start : bracket_end + 1])
        except json.JSONDecodeError:
            pass

    logger.error("[REFINER] Failed to parse LLM JSON output (%d chars)", len(text))
    return []
