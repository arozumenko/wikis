"""
Phase 4 — Cluster-Based Structure Planner

Converts pre-computed math clusters (Phases 1-3) into a full
``WikiStructureSpec`` with cheap LLM naming calls.

**Two-pass naming** eliminates "centroid bleed" — where macro-cluster
dominant symbols would shadow per-page content:

- **Pass 1** (section naming): one LLM call per macro-cluster using
  macro-level dominant symbols → section name + description.
- **Pass 2** (page naming): one LLM call per micro-cluster using
  *only that page's own* enriched symbols → page name + description.

Activation::

    WIKIS_STRUCTURE_PLANNER=cluster

Usage::

    from .cluster_planner import ClusterStructurePlanner

    planner = ClusterStructurePlanner(db, llm)
    spec = planner.plan_structure()
"""

from __future__ import annotations

import json
import logging
import os
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Dict, List, Optional, Tuple

from langchain_core.language_models import BaseLanguageModel
from langchain_core.messages import HumanMessage, SystemMessage

import networkx as nx

from ..constants import PAGE_IDENTITY_SYMBOLS, SUPPORTING_CODE_SYMBOLS, DOC_CLUSTER_SYMBOLS, SYMBOL_TYPE_PRIORITY
from ..feature_flags import get_feature_flags
from ..graph_clustering import select_central_symbols
from ..state.wiki_state import PageSpec, SectionSpec, WikiStructureSpec
from ..unified_db import UnifiedWikiDB
from .candidate_builder import build_candidates
from .coverage_ledger import CoverageLedger
from .page_validator import validate_all, KEEP, MERGE_WITH, DEMOTE

logger = logging.getLogger(__name__)

# ═══════════════════════════════════════════════════════════════════════════
# Configuration
# ═══════════════════════════════════════════════════════════════════════════

#: Maximum dominant symbols to send to LLM per macro-cluster
MAX_DOMINANT_SYMBOLS = 10

#: Maximum symbols summarised per micro-cluster in the prompt
MAX_MICRO_SUMMARY_SYMBOLS = 8

#: Truncation length for signatures and docstrings in prompts
SIGNATURE_TRUNC = 200
DOCSTRING_TRUNC = 200



# ═══════════════════════════════════════════════════════════════════════════
# Prompts — Two-pass naming
# ═══════════════════════════════════════════════════════════════════════════

# ── Pass 1: Section naming (one call per macro-cluster) ──────────────

SECTION_NAMING_SYSTEM = """\
You are a documentation architect writing for a mixed audience: product \
managers, team leads, and engineers.  You are given the most representative \
code symbols from a group of related source files.

Your job: create a short, user-facing **capability name** and a one-sentence \
description for this section of a wiki.  The name must describe WHAT the \
software does for its users — not how it is implemented.

Rules:
- Use plain language a non-developer stakeholder can understand.
- Name the CAPABILITY or DOMAIN, not file names or class names.
- Good: "Leader Election & Health Monitoring", "Payment Processing", \
"Real-time Data Transformation SDK"
- Bad: "leader_balancer.cc and related files", "Class AuthService", \
"src/v/resource_mgmt"
- The name must be ≤ 8 words."""

SECTION_NAMING_USER = """\
SECTION CLUSTER — {node_count} symbols across {file_count} files

TOP SYMBOLS (most connected and documented in this section):
{dominant_symbols_json}

SUB-GROUPS (page-level groupings — shown only for context; name the section, \
not individual pages):
{micro_summaries_json}

Output ONLY valid JSON (no markdown fences):
{{
  "section_name": "...",
  "section_description": "..."
}}"""

# ── Pass 2: Page naming (one call per micro-cluster) ─────────────────

PAGE_NAMING_SYSTEM = """\
You are a documentation architect writing page titles for a technical wiki.  \
The audience includes non-technical stakeholders, so page names must describe \
the CAPABILITY or FUNCTIONALITY in plain language — never use class names, \
file names, or implementation jargon as the page title.

You are given the symbols that belong to ONE specific page.  Name this page \
based ONLY on these symbols — ignore anything outside this list.

Rules:
- Name the CAPABILITY (what does this code accomplish for its users?).
- Good: "Record Batch Decoding & Output Routing", "Disk Space Reclamation"
- Bad: "disk_space_manager and storage.h", "JS VM Bridge"
- The name must be ≤ 10 words.
- Description: one sentence explaining the page's value to a reader.
- retrieval_query: 3-6 keyword phrase for documentation search."""

PAGE_NAMING_USER = """\
PAGE CLUSTER — {symbol_count} symbols in {file_count} files
Parent section: "{section_name}"

PAGE SYMBOLS (this page's own representative code elements):
{page_symbols_json}

DIRECTORIES:
{directories_json}

Output ONLY valid JSON (no markdown fences):
{{
  "page_name": "...",
  "description": "...",
  "retrieval_query": "..."
}}"""

# Legacy constant kept for backward compatibility in tests
CLUSTER_NAMING_SYSTEM = SECTION_NAMING_SYSTEM
CLUSTER_NAMING_USER = SECTION_NAMING_USER


# ── Pages-first ordering (WIKI_NAMING_ORDER=pages_first) ─────────────
# Used when section name is derived from named pages instead of macro
# centroid symbols.  Avoids the "centroid bleed" problem entirely by
# letting concrete page titles drive the section name.

SECTION_FROM_PAGES_SYSTEM = """\
You are a documentation architect.  You are given the page titles and one-\
sentence descriptions that belong to a section of a wiki.  Your job is to \
produce a single short, user-facing **capability name** and a one-sentence \
description that summarises ALL the listed pages.

Rules:
- Name the CAPABILITY or DOMAIN that ties the pages together — not the most \
prominent class or file name.
- Use plain language a non-developer stakeholder can understand.
- Good: "Authentication & Session Management", "Pipeline Execution"
- Bad: "AuthService and friends", "src/auth/*"
- The name must be ≤ 8 words."""

SECTION_FROM_PAGES_USER = """\
SECTION (derived from {page_count} named pages, {node_count} symbols total)

PAGES IN THIS SECTION:
{pages_json}

Output ONLY valid JSON (no markdown fences):
{{
  "section_name": "...",
  "section_description": "..."
}}"""


# ── Batched per-macro naming (WIKI_NAMING_BATCHED=1) ─────────────────
# Single LLM call returning the section name + every page's name in one
# structured-JSON response.  Replaces both Pass 1 (section centroid) and
# Pass 2 (per-micro page calls).  We feed only micro-cluster signatures
# — no macro centroid — so the model derives the section name from the
# pages it actually contains.

BATCHED_NAMING_SYSTEM = """\
You are a documentation architect writing a wiki for a mixed audience: \
product managers, team leads, and engineers.

You are given a SECTION composed of several PAGES.  Each page is a small \
group of related code symbols.  Your job, in ONE response, is to:
  1. Name every page based ONLY on that page's own symbols.
  2. Name the section based on the pages it contains (NOT individual \
symbols).

Rules for names:
- Use plain language a non-developer stakeholder can understand.
- Name the CAPABILITY or DOMAIN, not class names or file paths.
- Page name ≤ 10 words.  Section name ≤ 8 words.
- Good page: "Record Batch Decoding & Output Routing"
- Bad page: "disk_space_manager and storage.h"
- Good section: "Authentication & Session Management"
- Bad section: "AuthService and friends"

Output strictly valid JSON, no markdown fences, no extra prose."""

BATCHED_NAMING_USER = """\
SECTION CLUSTER — {node_count} symbols across {file_count} files

PAGES IN THIS SECTION:
{pages_json}

Output ONLY valid JSON in this exact shape (no markdown fences):
{{
  "section_name": "...",
  "section_description": "...",
  "pages": [
    {{
      "page_id": <integer matching the input page_id>,
      "page_name": "...",
      "description": "...",
      "retrieval_query": "..."
    }}
    /* one entry per input page, in the same order */
  ]
}}"""


# ═══════════════════════════════════════════════════════════════════════════
# Planner
# ═══════════════════════════════════════════════════════════════════════════

class ClusterStructurePlanner:
    """Build a WikiStructureSpec from pre-computed clusters + cheap LLM naming.

    Parameters
    ----------
    db : UnifiedWikiDB
        Unified DB with Phase 3 cluster assignments already persisted.
    llm : BaseLanguageModel
        LangChain LLM used exclusively for naming (cheap, small prompts).
    wiki_title : str, optional
        Override wiki title; if *None*, derived from repo metadata.
    """

    def __init__(
        self,
        db: UnifiedWikiDB,
        llm: BaseLanguageModel,
        wiki_title: Optional[str] = None,
        llm_low: Optional[BaseLanguageModel] = None,
    ):
        self.db = db
        self.llm = llm
        # ``llm_low`` is used for naming (small, low-cognition prompts).
        # Falls back to ``llm`` when not provided so existing callers keep
        # working unchanged.
        self.llm_low = llm_low or llm
        self.wiki_title = wiki_title or self._derive_wiki_title()
        self._cluster_graph: Optional[nx.MultiDiGraph] = None
        self._central_k: Optional[int] = None

        # Pre-compute test-exclusion SQL fragment once
        flags = get_feature_flags()
        self._exclude_tests = flags.exclude_tests
        self._test_sql = " AND is_test = 0" if self._exclude_tests else ""

    # ── adaptive central-k ────────────────────────────────────────────

    def _adaptive_central_k(self) -> int:
        """Scale central symbol count with repo size.

        Smaller repos need fewer representative symbols per page;
        larger repos need more to capture the breadth of each cluster.

        | Nodes      | k (central symbols) |
        |------------|---------------------|
        | < 200      | 5                   |
        | 200–999    | 8                   |
        | 1000–4999  | 12                  |
        | 5000+      | 15                  |
        """
        if self._central_k is not None:
            return self._central_k

        G = self._get_cluster_graph()
        n = G.number_of_nodes()
        if n >= 5000:
            k = 15
        elif n >= 1000:
            k = 12
        elif n >= 200:
            k = 8
        else:
            k = 5
        self._central_k = k
        return k

    # ── public API ────────────────────────────────────────────────────

    def plan_structure(self) -> WikiStructureSpec:
        """Plan the full wiki structure from clusters.

        Returns a ``WikiStructureSpec`` ready for the page-generation graph.
        """
        t0 = time.time()

        # 1. Load cluster map from DB: macro → micro → [node_ids]
        #    Filter to architectural nodes only — methods/fields were propagated
        #    into clusters by Phase 3 but they inflate the cluster count and
        #    prompt size.  The expansion module already filters to architectural
        #    nodes (is_architectural = 1), so the planner should do the same.
        cluster_map = self._load_architectural_cluster_map()
        if not cluster_map:
            logger.warning("ClusterStructurePlanner: no clusters in DB — returning fallback")
            return self._fallback_spec()

        logger.info(
            "ClusterStructurePlanner: %d macro-clusters to name",
            len(cluster_map),
        )

        # 1b. Optional: candidate validation (Phase 5)
        flags = get_feature_flags()
        candidates = None  # may be reused for the coverage ledger if unchanged
        cluster_map_changed = False
        if flags.capability_validation:
            candidates = build_candidates(self.db, cluster_map)
            validations = validate_all(candidates)

            # Pass 1 — Remove DEMOTE candidates
            for vr in validations:
                if vr.shape_decision == DEMOTE:
                    mid = vr.candidate.macro_id
                    pid = vr.candidate.micro_id
                    if mid in cluster_map and pid in cluster_map[mid]:
                        del cluster_map[mid][pid]
                        cluster_map_changed = True
                        logger.debug(
                            "ClusterStructurePlanner: demoted micro %d.%d",
                            mid, pid,
                        )

            # Pass 2 — Merge MERGE_WITH candidates into their largest
            # surviving sibling (same macro-cluster).
            merge_count = 0
            for vr in validations:
                if vr.shape_decision != MERGE_WITH:
                    continue
                mid = vr.candidate.macro_id
                pid = vr.candidate.micro_id
                if mid not in cluster_map or pid not in cluster_map[mid]:
                    continue
                # Find largest surviving sibling in this macro
                best_sibling = None
                best_size = 0
                for other_pid, other_nids in cluster_map[mid].items():
                    if other_pid == pid:
                        continue
                    if len(other_nids) > best_size:
                        best_size = len(other_nids)
                        best_sibling = other_pid
                if best_sibling is not None:
                    cluster_map[mid][best_sibling].extend(
                        cluster_map[mid].pop(pid),
                    )
                    merge_count += 1
                    cluster_map_changed = True

            if merge_count:
                logger.info(
                    "ClusterStructurePlanner: merged %d MERGE_WITH candidates",
                    merge_count,
                )

            # Clean up empty macros
            new_cluster_map = {
                m: micros for m, micros in cluster_map.items() if micros
            }
            if len(new_cluster_map) != len(cluster_map):
                cluster_map_changed = True
            cluster_map = new_cluster_map

            total_surviving_pages = sum(
                len(micros) for micros in cluster_map.values()
            )
            logger.info(
                "ClusterStructurePlanner: after validation — %d sections, "
                "%d pages surviving",
                len(cluster_map), total_surviving_pages,
            )

        # 1c. Optional: coverage ledger (Phase 6)
        coverage_report = None
        if flags.coverage_ledger:
            # Rebuild candidates after any demotion so ledger sees current
            # state — but only if validation actually changed cluster_map.
            if cluster_map_changed or candidates is None:
                ledger_candidates = build_candidates(self.db, cluster_map)
            else:
                ledger_candidates = candidates
            ledger = CoverageLedger(self.db, ledger_candidates)
            coverage_report = ledger.report()
            logger.info(
                "CoverageLedger: symbols %d/%d (%.0f%%), docs %d/%d, dirs %d/%d, "
                "uncovered HV: %d, overlaps: %d",
                coverage_report.covered_symbols,
                coverage_report.total_symbols,
                coverage_report.symbol_coverage * 100,
                coverage_report.covered_doc_domains,
                coverage_report.total_doc_domains,
                coverage_report.covered_directories,
                coverage_report.total_directories,
                len(coverage_report.uncovered_high_value),
                len(coverage_report.page_overlap_pairs),
            )

        # 2. Name each macro-cluster via one LLM call
        sections: List[SectionSpec] = []
        section_order = 1

        for macro_id in sorted(cluster_map.keys()):
            micro_map = cluster_map[macro_id]
            try:
                section = self._name_macro_cluster(
                    macro_id=macro_id,
                    micro_map=micro_map,
                    section_order=section_order,
                )
                sections.append(section)
                section_order += 1
            except Exception as exc:
                logger.warning(
                    "LLM naming failed for macro %d, using fallback: %s",
                    macro_id, exc,
                )
                sections.append(self._fallback_section(
                    macro_id, micro_map, section_order,
                ))
                section_order += 1

        total_pages = sum(len(s.pages) for s in sections)
        elapsed = time.time() - t0

        logger.info(
            "ClusterStructurePlanner: %d sections, %d pages in %.1fs",
            len(sections), total_pages, elapsed,
        )

        return WikiStructureSpec(
            wiki_title=self.wiki_title,
            overview=self._build_overview(sections),
            sections=sections,
            total_pages=total_pages,
        )

    # ── Architectural cluster map ──────────────────────────────────

    def _load_architectural_cluster_map(
        self,
    ) -> Dict[int, Dict[int, List[str]]]:
        """Load cluster map filtered to architectural nodes only.

        Phase 3 propagates assignments to child nodes (methods, fields, etc.)
        so that the unified DB has a complete mapping.  However, the structure
        planner should only consider top-level *architectural* symbols — the
        same set used by the content expansion module — to avoid inflating
        section and page counts.

        When ``exclude_tests`` is enabled, test nodes (``is_test = 1``) are
        also excluded — they should not form wiki pages.

        Returns ``{macro_id: {micro_id: [node_ids]}}`` where every node_id
        belongs to a non-test architectural symbol.
        """
        flags = get_feature_flags()
        test_filter = self._test_sql

        rows = self.db.conn.execute(
            "SELECT node_id, macro_cluster, micro_cluster "
            "FROM repo_nodes "
            "WHERE macro_cluster IS NOT NULL AND is_architectural = 1"
            + test_filter
        ).fetchall()

        result: Dict[int, Dict[int, List[str]]] = {}
        for row in rows:
            macro = row["macro_cluster"]
            micro = row["micro_cluster"] or 0
            result.setdefault(macro, {}).setdefault(micro, []).append(row["node_id"])

        # Remove empty micro-clusters (all their arch nodes were in a
        # different micro).  This shouldn't happen often, but guard anyway.
        for macro in list(result.keys()):
            for micro in list(result[macro].keys()):
                if not result[macro][micro]:
                    del result[macro][micro]
            if not result[macro]:
                del result[macro]

        total_nodes = sum(
            len(nids) for mm in result.values() for nids in mm.values()
        )
        logger.info(
            "Loaded architectural cluster map: %d macro-clusters, %d nodes "
            "(filtered from full cluster map)",
            len(result), total_nodes,
        )
        return result

    # ── LLM naming per macro-cluster ─────────────────────────────────

    def _name_macro_cluster(
        self,
        macro_id: int,
        micro_map: Dict[int, List[str]],
        section_order: int,
    ) -> SectionSpec:
        """Two-pass naming: section name first, then each page independently.

        Pass 1 — Section naming: one LLM call with macro-cluster dominant
        symbols to produce section_name + section_description.

        Pass 2 — Page naming: one LLM call **per micro-cluster** using
        only that micro-cluster's own enriched symbols (with signatures
        and docstrings).  This eliminates the "centroid bleed" problem
        where macro-level dominant symbols would shadow page-level content.
        """

        all_node_ids = set()
        for nids in micro_map.values():
            all_node_ids.update(nids)

        # ── Batch-fetch every node touched by this macro cluster up
        # front (replaces ~200 + per-micro O(N) ``get_node`` calls).
        node_index: Dict[str, Dict[str, Any]] = {}
        try:
            batched = self.db.get_nodes_by_ids(list(all_node_ids))
            for n in batched:
                nid = n.get("node_id")
                if nid:
                    node_index[nid] = n
        except Exception:  # pragma: no cover — defensive fallback
            node_index = {}

        def _rel_path(nid: str) -> str:
            n = node_index.get(nid)
            if n is not None:
                return n.get("rel_path", "") or ""
            n2 = self.db.get_node(nid) or {}
            return n2.get("rel_path", "") or ""

        file_count = len({
            _rel_path(nid)
            for nid in list(all_node_ids)[:200]
        } - {""})

        # ── Batched single-call naming (preferred path) ──────────────
        # When ``WIKI_NAMING_BATCHED=1`` (recommended), name the section
        # and every page in a single low-tier-LLM call.  Falls back to
        # the legacy multi-call path on parse failure.
        if (os.environ.get("WIKI_NAMING_BATCHED", "0") or "0").strip() == "1":
            batched = self._batched_macro_naming(
                macro_id=macro_id,
                micro_map=micro_map,
                node_index=node_index,
                node_count=len(all_node_ids),
                file_count=file_count,
            )
            if batched is not None:
                section_name, section_desc, page_naming_by_micro = batched
                return self._assemble_section(
                    macro_id=macro_id,
                    micro_map=micro_map,
                    section_order=section_order,
                    section_name=section_name,
                    section_desc=section_desc,
                    page_naming_by_micro=page_naming_by_micro,
                    all_node_ids=all_node_ids,
                )

        # ── Naming order strategy (legacy multi-call path) ──────────
        # ``WIKI_NAMING_ORDER`` controls whether the section is named
        # before pages (default, current behaviour) or derived from
        # the page titles after pages are named.
        naming_order = (
            os.environ.get("WIKI_NAMING_ORDER", "sections_first") or "sections_first"
        ).strip().lower()
        pages_first = naming_order == "pages_first"

        if pages_first:
            section_name = f"Section {macro_id}"
            section_desc = f"Section covering {len(all_node_ids)} symbols"
        else:
            # ── Pass 1: Section naming ──────────────────────────────
            dominant = self._get_dominant_symbols(macro_id)
            micro_summaries = self._get_micro_summaries(micro_map)

            section_prompt = SECTION_NAMING_USER.format(
                node_count=len(all_node_ids),
                file_count=file_count,
                dominant_symbols_json=json.dumps(dominant, indent=2),
                micro_summaries_json=json.dumps(micro_summaries, indent=2),
            )

            section_response = self.llm.invoke([
                SystemMessage(content=SECTION_NAMING_SYSTEM),
                HumanMessage(content=section_prompt),
            ])
            section_naming = _parse_json_response(_extract_text(section_response))
            section_name = section_naming.get("section_name") or f"Section {macro_id}"
            section_desc = (
                section_naming.get("section_description")
                or f"Section covering {len(all_node_ids)} symbols"
            )

        # ── Pass 2: Page naming (one call per micro-cluster) ────────
        # Optionally parallelised via ``WIKI_NAMING_CONCURRENCY`` (default
        # 1 = sequential, fully backwards-compatible).  We build the
        # prompt for every micro first, dispatch the LLM calls, then
        # consume the parsed responses in deterministic micro-id order
        # so page_order remains stable.
        try:
            naming_workers = max(1, int(os.environ.get("WIKI_NAMING_CONCURRENCY", "1") or "1"))
        except (TypeError, ValueError):
            naming_workers = 1

        ordered_micros = sorted(micro_map.keys())
        prompts: Dict[int, Tuple[List[str], str, int]] = {}
        for micro_id in ordered_micros:
            node_ids = micro_map[micro_id]
            page_symbols = self._get_page_symbols(node_ids)
            page_dirs = self._node_ids_to_folders(node_ids)
            page_file_count = len(set(
                _rel_path(nid) for nid in node_ids[:100]
            ) - {""})
            page_prompt = PAGE_NAMING_USER.format(
                symbol_count=len(node_ids),
                file_count=page_file_count,
                section_name=section_name,
                page_symbols_json=json.dumps(page_symbols, indent=2),
                directories_json=json.dumps(page_dirs),
            )
            prompts[micro_id] = (node_ids, page_prompt, page_file_count)

        def _invoke(micro_id: int) -> Tuple[int, Dict[str, Any]]:
            _, p_prompt, _ = prompts[micro_id]
            try:
                resp = self.llm.invoke([
                    SystemMessage(content=PAGE_NAMING_SYSTEM),
                    HumanMessage(content=p_prompt),
                ])
                parsed = _parse_json_response(_extract_text(resp))
            except Exception as exc:
                logger.warning(
                    "Page naming failed for macro %d micro %d: %s",
                    macro_id, micro_id, exc,
                )
                parsed = {}
            return micro_id, parsed

        results: Dict[int, Dict[str, Any]] = {}
        if naming_workers > 1 and len(ordered_micros) > 1:
            with ThreadPoolExecutor(max_workers=naming_workers) as ex:
                for micro_id, parsed in ex.map(_invoke, ordered_micros):
                    results[micro_id] = parsed
        else:
            for micro_id in ordered_micros:
                _, parsed = _invoke(micro_id)
                results[micro_id] = parsed

        # ── Pages-first: derive section name from named pages ───────
        if pages_first:
            pages_payload = []
            for micro_id in ordered_micros:
                p_naming = results.get(micro_id) or {}
                node_ids = prompts[micro_id][0]
                pages_payload.append({
                    "page_name": p_naming.get("page_name")
                    or f"Page {micro_id}",
                    "description": p_naming.get("description")
                    or f"Page covering {len(node_ids)} symbols",
                })
            try:
                from_pages_prompt = SECTION_FROM_PAGES_USER.format(
                    page_count=len(pages_payload),
                    node_count=len(all_node_ids),
                    pages_json=json.dumps(pages_payload, indent=2),
                )
                section_response = self.llm.invoke([
                    SystemMessage(content=SECTION_FROM_PAGES_SYSTEM),
                    HumanMessage(content=from_pages_prompt),
                ])
                section_naming = _parse_json_response(
                    _extract_text(section_response)
                )
                section_name = (
                    section_naming.get("section_name") or section_name
                )
                section_desc = (
                    section_naming.get("section_description") or section_desc
                )
            except Exception as exc:
                logger.warning(
                    "Section-from-pages naming failed for macro %d: %s",
                    macro_id, exc,
                )

        pages: List[PageSpec] = []
        page_order = 1

        for micro_id in ordered_micros:
            node_ids, _, _ = prompts[micro_id]
            page_naming = results.get(micro_id) or {}

            page_name = (
                page_naming.get("page_name")
                or f"{section_name} — Page {micro_id}"
            )
            page_desc = (
                page_naming.get("description")
                or f"Page covering {len(node_ids)} symbols"
            )
            retrieval_query = (
                page_naming.get("retrieval_query")
                or page_name
            )

            # Select central symbols via PageRank (Phase 7F)
            central_ids = self._select_central_node_ids(
                node_ids, k=self._adaptive_central_k(),
            )
            target_symbols = self._node_ids_to_symbol_names(central_ids)
            key_files = self._node_ids_to_paths(node_ids)
            target_folders = self._node_ids_to_folders(node_ids)
            target_docs = self._node_ids_to_doc_paths(node_ids)

            pages.append(PageSpec(
                page_name=page_name,
                page_order=page_order,
                description=page_desc,
                content_focus=page_desc,
                rationale=f"Grouped by graph clustering (macro={macro_id}, micro={micro_id}, {len(node_ids)} symbols)",
                target_symbols=target_symbols,
                target_docs=target_docs,
                target_folders=target_folders,
                key_files=key_files,
                retrieval_query=retrieval_query,
                metadata={
                    "planner_mode": "cluster",
                    "section_id": macro_id,
                    "page_id": micro_id,
                    "cluster_node_ids": list(node_ids),
                },
            ))
            page_order += 1

        return SectionSpec(
            section_name=section_name,
            section_order=section_order,
            description=section_desc,
            rationale=f"Mathematical clustering: macro={macro_id}, {len(micro_map)} pages, {len(all_node_ids)} symbols",
            pages=pages,
        )

    # ── Batched single-call naming ───────────────────────────────────

    def _batched_macro_naming(
        self,
        macro_id: int,
        micro_map: Dict[int, List[str]],
        node_index: Dict[str, Dict[str, Any]],
        node_count: int,
        file_count: int,
    ) -> Optional[Tuple[str, str, Dict[int, Dict[str, Any]]]]:
        """One LLM call: returns (section_name, section_desc, {micro_id -> page_naming}).

        Skips the macro centroid entirely — feeds only per-micro symbols
        so the section name is derived from the pages it actually
        contains (no centroid bleed).  Uses ``self.llm_low`` since this
        is a low-cognition naming task.

        Returns ``None`` on parse failure so callers can fall back to
        the legacy multi-call path.
        """
        ordered_micros = sorted(micro_map.keys())
        pages_payload: List[Dict[str, Any]] = []
        for micro_id in ordered_micros:
            node_ids = micro_map[micro_id]
            page_symbols = self._get_page_symbols(node_ids)
            pages_payload.append({
                "page_id": micro_id,
                "symbol_count": len(node_ids),
                "page_symbols": page_symbols,
            })

        prompt = BATCHED_NAMING_USER.format(
            node_count=node_count,
            file_count=file_count,
            pages_json=json.dumps(pages_payload, indent=2),
        )
        try:
            response = self.llm_low.invoke([
                SystemMessage(content=BATCHED_NAMING_SYSTEM),
                HumanMessage(content=prompt),
            ])
            parsed = _parse_json_response(_extract_text(response))
        except Exception as exc:
            logger.warning(
                "Batched naming LLM call failed for macro %d: %s — "
                "falling back to legacy multi-call path",
                macro_id, exc,
            )
            return None

        section_name = parsed.get("section_name")
        section_desc = parsed.get("section_description")
        raw_pages = parsed.get("pages") or []
        if not section_name or not isinstance(raw_pages, list):
            logger.warning(
                "Batched naming response malformed for macro %d "
                "(section_name=%r, pages_type=%s) — falling back",
                macro_id, section_name, type(raw_pages).__name__,
            )
            return None

        page_naming_by_micro: Dict[int, Dict[str, Any]] = {}
        for entry in raw_pages:
            if not isinstance(entry, dict):
                continue
            try:
                pid = int(entry.get("page_id"))
            except (TypeError, ValueError):
                continue
            page_naming_by_micro[pid] = {
                "page_name": entry.get("page_name"),
                "description": entry.get("description"),
                "retrieval_query": entry.get("retrieval_query"),
            }

        # Verify every micro got a name; otherwise fall back so we don't
        # silently emit "Page N" placeholders the legacy path would have
        # filled in.
        missing = [m for m in ordered_micros if m not in page_naming_by_micro]
        if missing:
            logger.warning(
                "Batched naming missing %d/%d pages for macro %d "
                "(missing micro_ids=%s) — falling back",
                len(missing), len(ordered_micros), macro_id, missing[:5],
            )
            return None

        return (
            section_name,
            section_desc or f"Section covering {node_count} symbols",
            page_naming_by_micro,
        )

    def _assemble_section(
        self,
        macro_id: int,
        micro_map: Dict[int, List[str]],
        section_order: int,
        section_name: str,
        section_desc: str,
        page_naming_by_micro: Dict[int, Dict[str, Any]],
        all_node_ids: set,
    ) -> SectionSpec:
        """Build the SectionSpec from pre-computed naming results.

        Shared between the batched single-call path and (potentially)
        any future paths that produce ``page_naming_by_micro`` upfront.
        """
        pages: List[PageSpec] = []
        page_order = 1
        for micro_id in sorted(micro_map.keys()):
            node_ids = micro_map[micro_id]
            page_naming = page_naming_by_micro.get(micro_id) or {}

            page_name = (
                page_naming.get("page_name")
                or f"{section_name} — Page {micro_id}"
            )
            page_desc = (
                page_naming.get("description")
                or f"Page covering {len(node_ids)} symbols"
            )
            retrieval_query = (
                page_naming.get("retrieval_query")
                or page_name
            )

            central_ids = self._select_central_node_ids(
                node_ids, k=self._adaptive_central_k(),
            )
            target_symbols = self._node_ids_to_symbol_names(central_ids)
            key_files = self._node_ids_to_paths(node_ids)
            target_folders = self._node_ids_to_folders(node_ids)
            target_docs = self._node_ids_to_doc_paths(node_ids)

            pages.append(PageSpec(
                page_name=page_name,
                page_order=page_order,
                description=page_desc,
                content_focus=page_desc,
                rationale=f"Grouped by graph clustering (macro={macro_id}, micro={micro_id}, {len(node_ids)} symbols)",
                target_symbols=target_symbols,
                target_docs=target_docs,
                target_folders=target_folders,
                key_files=key_files,
                retrieval_query=retrieval_query,
                metadata={
                    "planner_mode": "cluster",
                    "section_id": macro_id,
                    "page_id": micro_id,
                    "cluster_node_ids": list(node_ids),
                },
            ))
            page_order += 1

        return SectionSpec(
            section_name=section_name,
            section_order=section_order,
            description=section_desc,
            rationale=f"Mathematical clustering: macro={macro_id}, {len(micro_map)} pages, {len(all_node_ids)} symbols (batched naming)",
            pages=pages,
        )

    # ── Dominant symbols for prompt ──────────────────────────────────

    def _get_dominant_symbols(self, macro_id: int) -> List[Dict[str, Any]]:
        """Get top-N most representative *architectural* symbols from a macro-cluster.

        Scoring: min(edge_count, 10) + has_docstring(2)
        Only architectural nodes are considered — methods, fields, etc. are excluded.
        """
        nodes = self.db.get_nodes_by_cluster(macro=macro_id)
        if not nodes:
            return []

        # Filter to architectural nodes only (and exclude test nodes when flag is on)
        nodes = [n for n in nodes if n.get("is_architectural")]
        if self._exclude_tests:
            nodes = [n for n in nodes if not n.get("is_test")]

        scored = []
        for n in nodes:
            # Count outgoing edges as proxy for connectivity
            edge_count = len(self.db.get_edges_from(n["node_id"]))
            score = (
                min(edge_count, 10)
                + (2 if n.get("docstring") else 0)
            )
            scored.append((score, n))

        scored.sort(key=lambda x: -x[0])

        return [
            {
                "name": n["symbol_name"],
                "type": n["symbol_type"],
                "path": n["rel_path"],
                "signature": (n.get("signature") or "")[:SIGNATURE_TRUNC],
                "docstring": (n.get("docstring") or "")[:DOCSTRING_TRUNC],
            }
            for _, n in scored[:MAX_DOMINANT_SYMBOLS]
        ]

    def _get_micro_summaries(
        self, micro_map: Dict[int, List[str]]
    ) -> List[Dict[str, Any]]:
        """Build compact summaries of each micro-cluster for the LLM prompt."""
        summaries = []
        for micro_id in sorted(micro_map.keys()):
            node_ids = micro_map[micro_id]
            # Fetch a sample of architectural nodes for the summary
            nodes = [
                self.db.get_node(nid)
                for nid in node_ids[:MAX_MICRO_SUMMARY_SYMBOLS * 3]
            ]
            nodes = [n for n in nodes if n and n.get("is_architectural")]
            nodes = nodes[:MAX_MICRO_SUMMARY_SYMBOLS]

            symbols = [
                {
                    "name": n["symbol_name"],
                    "type": n["symbol_type"],
                    "path": n["rel_path"],
                }
                for n in nodes
            ]

            # Unique directories
            dirs = sorted({
                n["rel_path"].rsplit("/", 1)[0]
                for n in nodes
                if "/" in n.get("rel_path", "")
            })

            summaries.append({
                "micro_id": micro_id,
                "symbol_count": len(node_ids),
                "symbols": symbols,
                "directories": dirs[:5],
            })

        return summaries

    # ── Page-level symbols for Pass 2 ────────────────────────────────

    #: Maximum symbols sent to the page-naming LLM call
    MAX_PAGE_NAMING_SYMBOLS = 12

    def _get_page_symbols(self, node_ids: List[str]) -> List[Dict[str, Any]]:
        """Get enriched symbols for a single micro-cluster (page-level naming).

        Unlike ``_get_micro_summaries`` (which is lean: name/type/path only),
        this returns signatures and docstrings so the page-naming LLM has
        enough context to produce a meaningful capability-based title from
        *only* this page's own content.

        Scoring: min(edge_count, 10) + has_docstring(2)
        """
        nodes = [self.db.get_node(nid) for nid in node_ids]
        nodes = [n for n in nodes if n and n.get("is_architectural")]
        if self._exclude_tests:
            nodes = [n for n in nodes if not n.get("is_test")]

        if not nodes:
            return []

        scored = []
        for n in nodes:
            edge_count = len(self.db.get_edges_from(n["node_id"]))
            score = (
                min(edge_count, 10)
                + (2 if n.get("docstring") else 0)
            )
            scored.append((score, n))

        scored.sort(key=lambda x: -x[0])

        return [
            {
                "name": n["symbol_name"],
                "type": n["symbol_type"],
                "path": n["rel_path"],
                "signature": (n.get("signature") or "")[:SIGNATURE_TRUNC],
                "docstring": (n.get("docstring") or "")[:DOCSTRING_TRUNC],
            }
            for _, n in scored[:self.MAX_PAGE_NAMING_SYMBOLS]
        ]

    # ── Helpers ───────────────────────────────────────────────────────

    def _node_ids_to_symbol_names(self, node_ids: List[str]) -> List[str]:
        """Convert node_ids to symbol names (for target_symbols field).

        Only architectural *code* nodes are included — methods/fields are
        excluded because the expansion module resolves symbols with
        ``is_architectural = 1`` filter.  Documentation nodes (``is_doc = 1``)
        are also excluded here — they go into ``target_docs`` via
        :meth:`_node_ids_to_doc_paths` instead.
        """
        names = []
        for nid in node_ids:
            node = self.db.get_node(nid)
            if node and node.get("is_architectural") and not node.get("is_doc"):
                names.append(node["symbol_name"])
        return names

    def _node_ids_to_paths(self, node_ids: List[str]) -> List[str]:
        """Extract unique file paths from node_ids."""
        paths = set()
        for nid in node_ids:
            node = self.db.get_node(nid)
            if node and node.get("rel_path"):
                paths.add(node["rel_path"])
        return sorted(paths)[:20]  # Cap at 20

    def _node_ids_to_folders(self, node_ids: List[str]) -> List[str]:
        """Extract unique directory prefixes from node_ids."""
        folders = set()
        for nid in node_ids:
            node = self.db.get_node(nid)
            if node and "/" in (node.get("rel_path") or ""):
                folders.add(node["rel_path"].rsplit("/", 1)[0])
        return sorted(folders)[:10]

    def _node_ids_to_doc_paths(self, node_ids: List[str]) -> List[str]:
        """Extract documentation file paths from node_ids.

        Returns ``rel_path`` for every node that has ``is_doc = 1``.
        These paths are used as ``target_docs`` on :class:`PageSpec`,
        enabling the MIXED retrieval path (code + docs) during content
        expansion.
        """
        paths = set()
        for nid in node_ids:
            node = self.db.get_node(nid)
            if node and node.get("is_doc") and node.get("rel_path"):
                paths.add(node["rel_path"])
        return sorted(paths)[:20]

    # ── Central symbol selection (Phase 7F) ──────────────────────────

    def _get_cluster_graph(self) -> nx.MultiDiGraph:
        """Lazily reconstruct an NX graph from DB edges for PageRank.

        The graph is cached for the lifetime of the planner instance to
        avoid rebuilding it for every micro-cluster.
        """
        if self._cluster_graph is not None:
            return self._cluster_graph

        G = nx.MultiDiGraph()

        # Add all architectural nodes (excluding test nodes when flag is on)
        rows = self.db.conn.execute(
            "SELECT node_id FROM repo_nodes WHERE is_architectural = 1"
            + self._test_sql
        ).fetchall()
        for row in rows:
            G.add_node(row["node_id"])

        # Add all edges with weights
        edge_rows = self.db.conn.execute(
            "SELECT source_id, target_id, rel_type, weight FROM repo_edges"
        ).fetchall()
        for row in edge_rows:
            src, tgt = row["source_id"], row["target_id"]
            if src in G and tgt in G:
                G.add_edge(
                    src, tgt,
                    relationship_type=row["rel_type"],
                    weight=row["weight"] or 1.0,
                )

        self._cluster_graph = G
        logger.debug(
            "Built cluster graph from DB: %d nodes, %d edges",
            G.number_of_nodes(), G.number_of_edges(),
        )
        return G

    def _select_central_node_ids(
        self, node_ids: List[str], k: int = 10,
    ) -> List[str]:
        """Select the top-k most central node IDs from a micro-cluster.

        Code-first priority (Phase 3):
        1. PAGE_IDENTITY_SYMBOLS (class, interface, function, …) ranked by PageRank
        2. SUPPORTING_CODE_SYMBOLS (constant, type_alias, …) ranked by PageRank
        3. DOC_CLUSTER_SYMBOLS (module_doc, file_doc) only if budget remains

        Doc-dominant clusters (>70% doc nodes) yield docs-only pages.
        """
        G = self._get_cluster_graph()
        cluster_set = set(node_ids) & set(G.nodes())

        if len(cluster_set) <= k:
            return list(cluster_set)

        # Partition by symbol type tier
        identity_ids = []
        supporting_ids = []
        doc_ids = []
        other_ids = []

        for nid in cluster_set:
            node = self.db.get_node(nid)
            if not node:
                continue
            stype = (node.get("symbol_type") or "").lower()
            if stype in PAGE_IDENTITY_SYMBOLS:
                identity_ids.append(nid)
            elif stype in SUPPORTING_CODE_SYMBOLS:
                supporting_ids.append(nid)
            elif stype in DOC_CLUSTER_SYMBOLS:
                doc_ids.append(nid)
            else:
                other_ids.append(nid)

        # Detect doc-dominant cluster (≥70% doc nodes)
        total = len(identity_ids) + len(supporting_ids) + len(doc_ids) + len(other_ids)
        if total > 0 and len(doc_ids) / total >= 0.7:
            # Docs-only page: prefer doc nodes as seeds
            central = select_central_symbols(G, set(doc_ids), k=k) if doc_ids else []
            if len(central) < k:
                extras = select_central_symbols(
                    G, cluster_set - set(central), k=k - len(central)
                )
                central.extend(extras)
            return central[:k]

        # Code-first: fill from identity → supporting → doc → other
        result = []
        for tier_ids in [identity_ids, supporting_ids, other_ids, doc_ids]:
            if len(result) >= k:
                break
            remaining = k - len(result)
            tier_set = set(tier_ids) - set(result)
            if not tier_set:
                continue
            if len(tier_set) <= remaining:
                # Rank by PageRank within tier
                ranked = select_central_symbols(G, tier_set, k=len(tier_set))
                result.extend(ranked if ranked else list(tier_set))
            else:
                ranked = select_central_symbols(G, tier_set, k=remaining)
                result.extend(ranked if ranked else list(tier_set)[:remaining])

        if not result:
            # Fallback: plain PageRank
            central = select_central_symbols(G, cluster_set, k=k)
            return central if central else node_ids[:k]
        return result[:k]

    def _derive_wiki_title(self) -> str:
        """Try to derive the wiki title from DB metadata."""
        repo_id = self.db.get_meta("repo_identifier")
        if repo_id:
            name = repo_id.split("/")[-1].split(":")[0] if "/" in repo_id else repo_id
            return f"{name} — Technical Documentation"
        return "Repository Technical Documentation"

    def _build_overview(self, sections: List[SectionSpec]) -> str:
        """Generate a brief overview from section names."""
        section_list = ", ".join(s.section_name for s in sections)
        return (
            f"This wiki covers {sum(len(s.pages) for s in sections)} pages "
            f"across {len(sections)} sections: {section_list}."
        )

    def _fallback_spec(self) -> WikiStructureSpec:
        """Minimal spec when no clusters are available."""
        return WikiStructureSpec(
            wiki_title=self.wiki_title,
            overview="No clusters found — using fallback structure.",
            sections=[
                SectionSpec(
                    section_name="Repository Overview",
                    section_order=1,
                    description="General repository documentation",
                    rationale="Fallback: no clusters available",
                    pages=[
                        PageSpec(
                            page_name="Overview",
                            page_order=1,
                            description="General overview of the repository",
                            content_focus="Repository structure and key components",
                            rationale="Fallback page",
                            retrieval_query="repository overview main components",
                        )
                    ],
                )
            ],
            total_pages=1,
        )

    def _fallback_section(
        self,
        macro_id: int,
        micro_map: Dict[int, List[str]],
        section_order: int,
    ) -> SectionSpec:
        """Fallback section when LLM naming fails for a cluster."""
        all_node_ids = []
        for nids in micro_map.values():
            all_node_ids.extend(nids)

        # Try to derive a name from dominant file paths
        paths = self._node_ids_to_paths(all_node_ids[:50])
        if paths:
            common = os.path.commonpath(paths) if len(paths) > 1 else paths[0]
            section_name = f"Module: {common}" if common else f"Section {macro_id}"
        else:
            section_name = f"Section {macro_id}"

        pages = []
        page_order = 1
        for micro_id in sorted(micro_map.keys()):
            node_ids = micro_map[micro_id]
            # Central symbol selection (Phase 7F)
            central_ids = self._select_central_node_ids(
                node_ids, k=self._adaptive_central_k(),
            )
            target_symbols = self._node_ids_to_symbol_names(central_ids)
            key_files = self._node_ids_to_paths(node_ids)

            pages.append(PageSpec(
                page_name=f"{section_name} — Page {micro_id}",
                page_order=page_order,
                description=f"Page with {len(node_ids)} symbols",
                content_focus=f"Symbols in cluster {macro_id}/{micro_id}",
                rationale=f"Grouped by graph clustering (macro={macro_id}, micro={micro_id}, {len(node_ids)} symbols)",
                target_symbols=target_symbols,
                target_docs=self._node_ids_to_doc_paths(node_ids),
                target_folders=self._node_ids_to_folders(node_ids),
                key_files=key_files,
                retrieval_query=" ".join(target_symbols[:5]),
                metadata={
                    "planner_mode": "cluster",
                    "section_id": macro_id,
                    "page_id": micro_id,
                    "cluster_node_ids": list(node_ids),
                },
            ))
            page_order += 1

        return SectionSpec(
            section_name=section_name,
            section_order=section_order,
            description=f"Auto-generated section with {len(all_node_ids)} symbols",
            rationale=f"Fallback: LLM naming failed for macro={macro_id}",
            pages=pages,
        )


# ═══════════════════════════════════════════════════════════════════════════
# JSON parsing helpers
# ═══════════════════════════════════════════════════════════════════════════

def _extract_text(response: Any) -> str:
    """Pull plain text from a LangChain AI message."""
    content = getattr(response, "content", response)
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict):
                parts.append(str(item.get("text") or item.get("content") or ""))
        return "\n".join(parts)
    return str(content)


def _parse_json_response(raw: str) -> Dict[str, Any]:
    """Best-effort JSON extraction from LLM output.

    Handles markdown fences, leading text, trailing commas, etc.
    """
    import re

    text = raw.strip()

    # Strip markdown code fences
    fence_match = re.search(r"```(?:json)?\s*([\s\S]*?)```", text)
    if fence_match:
        text = fence_match.group(1).strip()

    # Try direct parse first
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # Find first { ... last }
    first_brace = text.find("{")
    last_brace = text.rfind("}")
    if first_brace != -1 and last_brace > first_brace:
        candidate = text[first_brace : last_brace + 1]
        try:
            return json.loads(candidate)
        except json.JSONDecodeError:
            pass

        # Remove trailing commas before } or ]
        cleaned = re.sub(r",\s*([}\]])", r"\1", candidate)
        try:
            return json.loads(cleaned)
        except json.JSONDecodeError:
            pass

    logger.warning("Failed to parse LLM JSON output (length %d)", len(raw))
    return {}
