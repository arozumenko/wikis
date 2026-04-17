"""
Cluster-Based Structure Planner

Converts pre-computed graph clusters into a full ``WikiStructureSpec``
with cheap LLM naming calls.

**Two-pass naming** eliminates "centroid bleed" — where macro-cluster
dominant symbols would shadow per-page content:

- **Pass 1** (section naming): one LLM call per macro-cluster using
  macro-level dominant symbols → section name + description.
- **Pass 2** (page naming): one LLM call per micro-cluster using
  *only that page's own* enriched symbols → page name + description.

Unlike the DeepWiki version (which depends on a SQL database), this
implementation works entirely with in-memory data: a ``ClusterAssignment``
from the clustering engine and the original NetworkX ``MultiDiGraph``.

Usage::

    from .cluster_planner import ClusterStructurePlanner

    planner = ClusterStructurePlanner(
        graph=G,
        cluster_assignment=assignment,
        llm=chat_model,
    )
    spec = planner.plan_structure()
"""

from __future__ import annotations

import json
import logging
import os
import re
import time
from typing import Any

from langchain_core.language_models import BaseLanguageModel
from langchain_core.messages import HumanMessage, SystemMessage

import networkx as nx

from ..cluster_constants import (
    DOC_DOMINANT_THRESHOLD,
    PAGE_IDENTITY_SYMBOLS,
    SUPPORTING_CODE_SYMBOLS,
    DOC_CLUSTER_SYMBOLS,
    SYMBOL_TYPE_PRIORITY,
    is_test_path,
)
from ..constants import ARCHITECTURAL_SYMBOLS, DOC_SYMBOL_TYPES
from ..graph_clustering import ClusterAssignment, select_central_symbols
from ..state.wiki_state import PageSpec, SectionSpec, WikiStructureSpec
from .cluster_prompts import (
    PAGE_NAMING_SYSTEM,
    PAGE_NAMING_USER,
    SECTION_NAMING_SYSTEM,
    SECTION_NAMING_USER,
)

logger = logging.getLogger(__name__)

# ═══════════════════════════════════════════════════════════════════════════
# Configuration
# ═══════════════════════════════════════════════════════════════════════════

#: Maximum dominant symbols to send to LLM per macro-cluster
MAX_DOMINANT_SYMBOLS = 10

#: Maximum symbols summarised per micro-cluster in the prompt
MAX_MICRO_SUMMARY_SYMBOLS = 8

#: Maximum symbols sent to the page-naming LLM call
MAX_PAGE_NAMING_SYMBOLS = 12

#: Truncation length for signatures and docstrings in prompts
SIGNATURE_TRUNC = 200
DOCSTRING_TRUNC = 200


# ═══════════════════════════════════════════════════════════════════════════
# Planner
# ═══════════════════════════════════════════════════════════════════════════


class ClusterStructurePlanner:
    """Build a WikiStructureSpec from pre-computed clusters + cheap LLM naming.

    Parameters
    ----------
    graph : nx.MultiDiGraph
        Full code graph with node attributes (``symbol_name``, ``symbol_type``,
        ``rel_path``, ``signature``, ``docstring``, etc.).
    cluster_assignment : ClusterAssignment
        Pre-computed cluster assignments from ``run_clustering()``.
    llm : BaseLanguageModel
        LangChain LLM used exclusively for naming (cheap, small prompts).
    wiki_title : str, optional
        Override wiki title; if *None*, derived from repo name.
    repo_name : str, optional
        Repository name for title derivation.
    exclude_tests : bool
        Whether to exclude test nodes from the planner output.
    """

    def __init__(
        self,
        graph: nx.MultiDiGraph,
        cluster_assignment: ClusterAssignment,
        llm: BaseLanguageModel,
        wiki_title: str | None = None,
        repo_name: str | None = None,
        exclude_tests: bool = True,
        db=None,
    ):
        self.graph = graph
        self.cluster_assignment = cluster_assignment
        self.llm = llm
        self.repo_name = repo_name or ""
        self.wiki_title = wiki_title or self._derive_wiki_title()
        self.exclude_tests = exclude_tests
        self.db = db
        self._central_k: int | None = None

    # ── Node access helpers ──────────────────────────────────────────

    def _get_node(self, node_id: str) -> dict[str, Any] | None:
        """Fetch node attributes from the graph.  Returns ``None`` if missing."""
        if node_id not in self.graph:
            return None
        data = dict(self.graph.nodes[node_id])
        # Ensure node_id is accessible as an attribute
        data.setdefault("node_id", node_id)
        return data

    def _is_architectural(self, node_data: dict[str, Any]) -> bool:
        """Return True if the node represents an architectural symbol."""
        stype = (node_data.get("symbol_type") or "").lower()
        return stype in ARCHITECTURAL_SYMBOLS

    def _is_doc_node(self, node_data: dict[str, Any]) -> bool:
        stype = (node_data.get("symbol_type") or "").lower()
        return stype in DOC_SYMBOL_TYPES

    def _is_test_node(self, node_data: dict[str, Any]) -> bool:
        if not self.exclude_tests:
            return False
        rel_path = node_data.get("rel_path") or node_data.get("file_path") or ""
        return is_test_path(rel_path)

    def _edge_count_for(self, node_id: str) -> int:
        """Count outgoing edges as a connectivity proxy."""
        if node_id not in self.graph:
            return 0
        return self.graph.out_degree(node_id)

    # ── adaptive central-k ────────────────────────────────────────────

    def _adaptive_central_k(self) -> int:
        """Scale central symbol count with repo size.

        | Nodes      | k |
        |------------|---|
        | < 200      | 5 |
        | 200–999    | 8 |
        | 1000–4999  | 12|
        | 5000+      | 15|
        """
        if self._central_k is not None:
            return self._central_k
        n = self.graph.number_of_nodes()
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

    # ── Cluster map construction ─────────────────────────────────────

    def _build_architectural_cluster_map(
        self,
    ) -> dict[int, dict[int, list[str]]]:
        """Build ``{macro: {micro: [node_ids]}}`` from the ClusterAssignment.

        Uses the ``sections`` dict which stores
        ``{section_id: {"pages": {page_id: [node_ids]}, ...}}``.
        Filters to architectural nodes only, optionally excluding test nodes.
        """
        result: dict[int, dict[int, list[str]]] = {}

        for section_id, section_data in self.cluster_assignment.sections.items():
            pages = section_data.get("pages", {})
            for page_id, node_ids in pages.items():
                for node_id in node_ids:
                    node = self._get_node(node_id)
                    if node is None:
                        continue
                    if not self._is_architectural(node):
                        continue
                    if self._is_test_node(node):
                        continue
                    result.setdefault(section_id, {}).setdefault(
                        page_id, []
                    ).append(node_id)

        # Remove empty entries
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
            "Built architectural cluster map: %d macro-clusters, %d nodes",
            len(result),
            total_nodes,
        )
        return result

    # ── public API ────────────────────────────────────────────────────

    def plan_structure(self) -> WikiStructureSpec:
        """Plan the full wiki structure from clusters.

        Returns a ``WikiStructureSpec`` ready for the page-generation graph.
        """
        t0 = time.time()

        cluster_map = self._build_architectural_cluster_map()
        if not cluster_map:
            logger.warning(
                "ClusterStructurePlanner: no clusters — returning fallback"
            )
            return self._fallback_spec()

        # ── Phase 5/6: Optional candidate validation & coverage ──────
        cluster_map = self._maybe_validate_candidates(cluster_map)

        logger.info(
            "ClusterStructurePlanner: %d macro-clusters to name",
            len(cluster_map),
        )

        sections: list[SectionSpec] = []
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
                    macro_id,
                    exc,
                )
                sections.append(
                    self._fallback_section(macro_id, micro_map, section_order)
                )
                section_order += 1

        total_pages = sum(len(s.pages) for s in sections)
        elapsed = time.time() - t0

        logger.info(
            "ClusterStructurePlanner: %d sections, %d pages in %.1fs",
            len(sections),
            total_pages,
            elapsed,
        )

        return WikiStructureSpec(
            wiki_title=self.wiki_title,
            overview=self._build_overview(sections),
            sections=sections,
            total_pages=total_pages,
        )

    # ── Two-pass naming ───────────────────────────────────────────────

    def _name_macro_cluster(
        self,
        macro_id: int,
        micro_map: dict[int, list[str]],
        section_order: int,
    ) -> SectionSpec:
        """Two-pass naming: section name first, then each page independently."""
        all_node_ids: set[str] = set()
        for nids in micro_map.values():
            all_node_ids.update(nids)

        file_count = len(
            {
                (self._get_node(nid) or {}).get("rel_path", "")
                for nid in list(all_node_ids)[:200]
            }
            - {""}
        )

        # ── Pass 1: Section naming ──────────────────────────────────
        dominant = self._get_dominant_symbols(list(all_node_ids))
        micro_summaries = self._get_micro_summaries(micro_map)

        section_prompt = SECTION_NAMING_USER.format(
            node_count=len(all_node_ids),
            file_count=file_count,
            dominant_symbols_json=json.dumps(dominant, indent=2),
            micro_summaries_json=json.dumps(micro_summaries, indent=2),
        )

        section_response = self.llm.invoke(
            [
                SystemMessage(content=SECTION_NAMING_SYSTEM),
                HumanMessage(content=section_prompt),
            ]
        )
        section_naming = _parse_json_response(_extract_text(section_response))
        section_name = (
            section_naming.get("section_name") or f"Section {macro_id}"
        )
        section_desc = (
            section_naming.get("section_description")
            or f"Section covering {len(all_node_ids)} symbols"
        )

        # ── Pass 2: Page naming (one call per micro-cluster) ────────
        pages: list[PageSpec] = []
        page_order = 1

        for micro_id in sorted(micro_map.keys()):
            node_ids = micro_map[micro_id]

            page_symbols = self._get_page_symbols(node_ids)
            page_dirs = self._node_ids_to_folders(node_ids)
            page_file_count = len(
                {
                    (self._get_node(nid) or {}).get("rel_path", "")
                    for nid in node_ids[:100]
                }
                - {""}
            )

            page_prompt = PAGE_NAMING_USER.format(
                symbol_count=len(node_ids),
                file_count=page_file_count,
                section_name=section_name,
                page_symbols_json=json.dumps(page_symbols, indent=2),
                directories_json=json.dumps(page_dirs),
            )

            try:
                page_response = self.llm.invoke(
                    [
                        SystemMessage(content=PAGE_NAMING_SYSTEM),
                        HumanMessage(content=page_prompt),
                    ]
                )
                page_naming = _parse_json_response(
                    _extract_text(page_response)
                )
            except Exception as exc:
                logger.warning(
                    "Page naming failed for macro %d micro %d: %s",
                    macro_id,
                    micro_id,
                    exc,
                )
                page_naming = {}

            page_name = (
                page_naming.get("page_name")
                or f"{section_name} — Page {micro_id}"
            )
            page_desc = (
                page_naming.get("description")
                or f"Page covering {len(node_ids)} symbols"
            )
            retrieval_query = page_naming.get("retrieval_query") or page_name

            central_ids = self._select_central_node_ids(
                node_ids,
                k=self._adaptive_central_k(),
            )
            target_symbols = self._node_ids_to_symbol_names(central_ids)
            key_files = self._node_ids_to_paths(node_ids)
            target_folders = page_dirs
            target_docs = self._node_ids_to_doc_paths(node_ids)

            pages.append(
                PageSpec(
                    page_name=page_name,
                    page_order=page_order,
                    description=page_desc,
                    content_focus=page_desc,
                    rationale=(
                        f"Grouped by graph clustering "
                        f"(macro={macro_id}, micro={micro_id}, "
                        f"{len(node_ids)} symbols)"
                    ),
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
                )
            )
            page_order += 1

        return SectionSpec(
            section_name=section_name,
            section_order=section_order,
            description=section_desc,
            rationale=(
                f"Mathematical clustering: macro={macro_id}, "
                f"{len(micro_map)} pages, {len(all_node_ids)} symbols"
            ),
            pages=pages,
        )

    # ── Dominant symbols for prompt ──────────────────────────────────

    def _get_dominant_symbols(
        self, node_ids: list[str]
    ) -> list[dict[str, Any]]:
        """Get top-N most representative architectural symbols.

        Scoring: ``min(edge_count, 10) + has_docstring(2)``
        """
        scored: list[tuple[int, dict[str, Any]]] = []

        for nid in node_ids:
            node = self._get_node(nid)
            if node is None or not self._is_architectural(node):
                continue
            if self._is_test_node(node):
                continue
            edge_count = self._edge_count_for(nid)
            score = min(edge_count, 10) + (2 if node.get("docstring") else 0)
            scored.append((score, node))

        scored.sort(key=lambda x: -x[0])

        return [
            {
                "name": n.get("symbol_name") or n.get("name", ""),
                "type": n.get("symbol_type") or "",
                "path": n.get("rel_path") or "",
                "signature": (n.get("signature") or "")[:SIGNATURE_TRUNC],
                "docstring": (n.get("docstring") or "")[:DOCSTRING_TRUNC],
            }
            for _, n in scored[:MAX_DOMINANT_SYMBOLS]
        ]

    def _get_micro_summaries(
        self, micro_map: dict[int, list[str]]
    ) -> list[dict[str, Any]]:
        """Build compact summaries of each micro-cluster for the LLM prompt."""
        summaries: list[dict[str, Any]] = []

        for micro_id in sorted(micro_map.keys()):
            node_ids = micro_map[micro_id]
            nodes = []
            for nid in node_ids[: MAX_MICRO_SUMMARY_SYMBOLS * 3]:
                node = self._get_node(nid)
                if node and self._is_architectural(node):
                    nodes.append(node)
            nodes = nodes[:MAX_MICRO_SUMMARY_SYMBOLS]

            symbols = [
                {
                    "name": n.get("symbol_name") or n.get("name", ""),
                    "type": n.get("symbol_type") or "",
                    "path": n.get("rel_path") or "",
                }
                for n in nodes
            ]

            dirs = sorted(
                {
                    n["rel_path"].rsplit("/", 1)[0]
                    for n in nodes
                    if "/" in (n.get("rel_path") or "")
                }
            )

            summaries.append(
                {
                    "micro_id": micro_id,
                    "symbol_count": len(node_ids),
                    "symbols": symbols,
                    "directories": dirs[:5],
                }
            )

        return summaries

    # ── Page-level symbols for Pass 2 ────────────────────────────────

    def _get_page_symbols(
        self, node_ids: list[str]
    ) -> list[dict[str, Any]]:
        """Get enriched symbols for a single micro-cluster (page-level naming).

        Returns signatures and docstrings so the page-naming LLM has
        enough context to produce a meaningful capability-based title.

        Scoring: ``min(edge_count, 10) + has_docstring(2)``
        """
        scored: list[tuple[int, dict[str, Any]]] = []

        for nid in node_ids:
            node = self._get_node(nid)
            if node is None or not self._is_architectural(node):
                continue
            if self._is_test_node(node):
                continue
            edge_count = self._edge_count_for(nid)
            score = min(edge_count, 10) + (2 if node.get("docstring") else 0)
            scored.append((score, node))

        scored.sort(key=lambda x: -x[0])

        return [
            {
                "name": n.get("symbol_name") or n.get("name", ""),
                "type": n.get("symbol_type") or "",
                "path": n.get("rel_path") or "",
                "signature": (n.get("signature") or "")[:SIGNATURE_TRUNC],
                "docstring": (n.get("docstring") or "")[:DOCSTRING_TRUNC],
            }
            for _, n in scored[:MAX_PAGE_NAMING_SYMBOLS]
        ]

    # ── Node-to-field helpers ─────────────────────────────────────────

    def _node_ids_to_symbol_names(self, node_ids: list[str]) -> list[str]:
        """Convert node IDs to symbol names (architectural code nodes only)."""
        names: list[str] = []
        for nid in node_ids:
            node = self._get_node(nid)
            if node and self._is_architectural(node) and not self._is_doc_node(node):
                name = node.get("symbol_name") or node.get("name")
                if name:
                    names.append(name)
        return names

    def _node_ids_to_paths(self, node_ids: list[str]) -> list[str]:
        """Extract unique file paths from node IDs (capped at 20)."""
        paths: set[str] = set()
        for nid in node_ids:
            node = self._get_node(nid)
            if node and node.get("rel_path"):
                paths.add(node["rel_path"])
        return sorted(paths)[:20]

    def _node_ids_to_folders(self, node_ids: list[str]) -> list[str]:
        """Extract unique directory prefixes from node IDs (capped at 10)."""
        folders: set[str] = set()
        for nid in node_ids:
            node = self._get_node(nid)
            if node and "/" in (node.get("rel_path") or ""):
                folders.add(node["rel_path"].rsplit("/", 1)[0])
        return sorted(folders)[:10]

    def _node_ids_to_doc_paths(self, node_ids: list[str]) -> list[str]:
        """Extract documentation file paths from node IDs (capped at 20)."""
        paths: set[str] = set()
        for nid in node_ids:
            node = self._get_node(nid)
            if node and self._is_doc_node(node) and node.get("rel_path"):
                paths.add(node["rel_path"])
        return sorted(paths)[:20]

    # ── Central symbol selection ──────────────────────────────────────

    def _select_central_node_ids(
        self,
        node_ids: list[str],
        k: int = 10,
    ) -> list[str]:
        """Select the top-k most central node IDs from a micro-cluster.

        Code-first priority:
        1. PAGE_IDENTITY_SYMBOLS (class, interface, function, …) ranked by PageRank
        2. SUPPORTING_CODE_SYMBOLS (constant, type_alias, …) ranked by PageRank
        3. DOC_CLUSTER_SYMBOLS (module_doc, file_doc) only if budget remains

        Doc-dominant clusters (≥70% doc nodes) yield docs-only pages.
        """
        cluster_set = set(node_ids) & set(self.graph.nodes())

        if len(cluster_set) <= k:
            return list(cluster_set)

        # Partition by symbol type tier
        identity_ids: list[str] = []
        supporting_ids: list[str] = []
        doc_ids: list[str] = []
        other_ids: list[str] = []

        for nid in cluster_set:
            node = self._get_node(nid)
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

        # Detect doc-dominant cluster
        total = (
            len(identity_ids)
            + len(supporting_ids)
            + len(doc_ids)
            + len(other_ids)
        )
        if total > 0 and len(doc_ids) / total >= DOC_DOMINANT_THRESHOLD:
            central = (
                select_central_symbols(self.graph, set(doc_ids), k=k)
                if doc_ids
                else []
            )
            if len(central) < k:
                extras = select_central_symbols(
                    self.graph,
                    cluster_set - set(central),
                    k=k - len(central),
                )
                central.extend(extras)
            return central[:k]

        # Code-first: fill from identity → supporting → other → doc
        result: list[str] = []
        for tier_ids in [identity_ids, supporting_ids, other_ids, doc_ids]:
            if len(result) >= k:
                break
            remaining = k - len(result)
            tier_set = set(tier_ids) - set(result)
            if not tier_set:
                continue
            if len(tier_set) <= remaining:
                ranked = select_central_symbols(
                    self.graph, tier_set, k=len(tier_set)
                )
                result.extend(ranked if ranked else list(tier_set))
            else:
                ranked = select_central_symbols(
                    self.graph, tier_set, k=remaining
                )
                result.extend(
                    ranked if ranked else list(tier_set)[:remaining]
                )

        if not result:
            central = select_central_symbols(self.graph, cluster_set, k=k)
            return central if central else node_ids[:k]
        return result[:k]

    # ── Title / overview / fallbacks ──────────────────────────────────

    def _derive_wiki_title(self) -> str:
        if self.repo_name:
            name = (
                self.repo_name.split("/")[-1].split(":")[0]
                if "/" in self.repo_name
                else self.repo_name
            )
            return f"{name} — Technical Documentation"
        return "Repository Technical Documentation"

    # ── Phase 5/6: Candidate validation & coverage ──────────────────

    def _maybe_validate_candidates(
        self,
        cluster_map: dict[int, dict[int, list[str]]],
    ) -> dict[int, dict[int, list[str]]]:
        """Run candidate validation + coverage ledger when enabled.

        Gated behind env vars:
        - ``WIKIS_CANDIDATE_VALIDATION=1`` — Phase 5 (validate & reshape)
        - ``WIKIS_COVERAGE_LEDGER=1``       — Phase 6 (coverage metrics)

        Requires ``self.db`` to be set (protocol storage instance).
        Returns the (possibly modified) cluster_map.
        """
        validate_enabled = os.environ.get("WIKIS_CANDIDATE_VALIDATION") == "1"
        coverage_enabled = os.environ.get("WIKIS_COVERAGE_LEDGER") == "1"

        if not (validate_enabled or coverage_enabled):
            return cluster_map

        if self.db is None:
            logger.debug(
                "ClusterStructurePlanner: validation/coverage requested "
                "but no db provided — skipping"
            )
            return cluster_map

        try:
            from .candidate_builder import build_candidates
            from .page_validator import (
                DEMOTE,
                MERGE_WITH,
                PROMOTE_DOCS,
                SPLIT_BY,
                validate_all,
            )

            candidates = build_candidates(self.db, cluster_map)

            if validate_enabled:
                results = validate_all(candidates)

                # Pass 1 — Remove DEMOTE candidates
                for vr in results:
                    if vr.shape_decision == DEMOTE:
                        mid = vr.candidate.macro_id
                        pid = vr.candidate.micro_id
                        if mid in cluster_map and pid in cluster_map[mid]:
                            del cluster_map[mid][pid]
                            logger.info(
                                "[VALIDATION] Demoting micro %d (macro %d)",
                                pid, mid,
                            )

                # Pass 2 — Merge MERGE_WITH candidates into their
                # largest surviving sibling (same macro-cluster).
                merge_count = 0
                for vr in results:
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

                if merge_count:
                    logger.info(
                        "[VALIDATION] Merged %d MERGE_WITH candidates",
                        merge_count,
                    )

                # Clean up empty macros
                cluster_map = {
                    m: micros for m, micros in cluster_map.items() if micros
                }

                total_surviving = sum(
                    len(micros) for micros in cluster_map.values()
                )
                logger.info(
                    "[VALIDATION] After validation — %d sections, "
                    "%d pages surviving",
                    len(cluster_map), total_surviving,
                )

            if coverage_enabled:
                from .coverage_ledger import CoverageLedger

                # Rebuild candidates from (possibly modified) cluster_map
                candidates = build_candidates(self.db, cluster_map)
                ledger = CoverageLedger(self.db, candidates)
                report = ledger.report()
                logger.info(
                    "[COVERAGE] symbol=%.1f%% (%d/%d) doc=%.1f%% dir=%.1f%% "
                    "high_value_uncovered=%d overlaps=%d",
                    report.symbol_coverage * 100,
                    report.covered_symbols,
                    report.total_symbols,
                    report.doc_coverage * 100,
                    report.dir_coverage * 100,
                    len(report.uncovered_high_value),
                    len(report.page_overlap_pairs),
                )

        except Exception as exc:
            logger.warning(
                "[VALIDATION] Candidate validation failed — "
                "continuing with original cluster_map: %s",
                exc,
            )

        return cluster_map

    def _build_overview(self, sections: list[SectionSpec]) -> str:
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
        micro_map: dict[int, list[str]],
        section_order: int,
    ) -> SectionSpec:
        """Fallback section when LLM naming fails for a cluster."""
        all_node_ids: list[str] = []
        for nids in micro_map.values():
            all_node_ids.extend(nids)

        # Try to derive a name from dominant file paths
        paths = self._node_ids_to_paths(all_node_ids[:50])
        if paths:
            common = os.path.commonpath(paths) if len(paths) > 1 else paths[0]
            section_name = f"Module: {common}" if common else f"Section {macro_id}"
        else:
            section_name = f"Section {macro_id}"

        pages: list[PageSpec] = []
        page_order = 1

        for micro_id in sorted(micro_map.keys()):
            node_ids = micro_map[micro_id]
            central_ids = self._select_central_node_ids(
                node_ids,
                k=self._adaptive_central_k(),
            )
            target_symbols = self._node_ids_to_symbol_names(central_ids)
            key_files = self._node_ids_to_paths(node_ids)

            pages.append(
                PageSpec(
                    page_name=f"{section_name} — Page {micro_id}",
                    page_order=page_order,
                    description=f"Page with {len(node_ids)} symbols",
                    content_focus=f"Symbols in cluster {macro_id}/{micro_id}",
                    rationale=(
                        f"Grouped by graph clustering "
                        f"(macro={macro_id}, micro={micro_id}, "
                        f"{len(node_ids)} symbols)"
                    ),
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
                )
            )
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
        parts: list[str] = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict):
                parts.append(
                    str(item.get("text") or item.get("content") or "")
                )
        return "\n".join(parts)
    return str(content)


def _parse_json_response(raw: str) -> dict[str, Any]:
    """Best-effort JSON extraction from LLM output.

    Handles markdown fences, leading text, trailing commas, etc.
    """
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
