"""
Coverage Ledger + Compact LLM Refiner — Phase 6.

1. **CoverageLedger** tracks what the current set of pages covers:
   - Architectural symbol coverage (identity + supporting)
   - Public API coverage (symbols without leading underscore)
   - Documentation domain coverage (by directory)
   - Directory coverage (all source dirs)

2. **compact_refine_section()** is a single-call LLM refiner that
   receives a section summary + quality flags and returns structured
   page actions.

Gated behind ``FeatureFlags.coverage_ledger``.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple

from ..constants import (
    DOC_CLUSTER_SYMBOLS,
    PAGE_IDENTITY_SYMBOLS,
    SUPPORTING_CODE_SYMBOLS,
    SYMBOL_TYPE_PRIORITY,
)
from .candidate_builder import CandidateRecord

logger = logging.getLogger(__name__)


# ═════════════════════════════════════════════════════════════════════
# Coverage Ledger
# ═════════════════════════════════════════════════════════════════════

@dataclass
class CoverageReport:
    """Summary of coverage gaps for a section or the full wiki."""

    total_symbols: int = 0
    covered_symbols: int = 0
    uncovered_high_value: List[Dict[str, Any]] = field(default_factory=list)
    total_doc_domains: int = 0
    covered_doc_domains: int = 0
    uncovered_doc_domains: List[str] = field(default_factory=list)
    total_directories: int = 0
    covered_directories: int = 0
    page_overlap_pairs: List[Tuple[int, int, int]] = field(default_factory=list)

    @property
    def symbol_coverage(self) -> float:
        return self.covered_symbols / max(self.total_symbols, 1)

    @property
    def doc_coverage(self) -> float:
        return self.covered_doc_domains / max(self.total_doc_domains, 1)

    @property
    def dir_coverage(self) -> float:
        return self.covered_directories / max(self.total_directories, 1)


class CoverageLedger:
    """Track and report coverage of a set of CandidateRecords.

    Parameters
    ----------
    db : UnifiedWikiDB
        Open database connection.
    candidates : list[CandidateRecord]
        All candidate pages being considered.
    """

    def __init__(self, db, candidates: List[CandidateRecord]):
        self.db = db
        self.candidates = candidates
        self._all_symbols: Dict[str, Dict[str, Any]] = {}
        self._all_doc_dirs: Set[str] = set()
        self._all_source_dirs: Set[str] = set()
        self._covered_ids: Set[str] = set()
        self._covered_dirs: Set[str] = set()
        self._covered_doc_dirs: Set[str] = set()

        self._load_universe()
        self._compute_coverage()

    def _load_universe(self):
        """Load the full universe of architectural symbols and directories."""
        conn = self.db.conn
        rows = conn.execute(
            "SELECT node_id, symbol_name, symbol_type, rel_path, is_doc "
            "FROM repo_nodes WHERE is_architectural = 1"
        ).fetchall()

        for row in rows:
            node = dict(row)
            nid = node["node_id"]
            self._all_symbols[nid] = node

            rel_path = node.get("rel_path") or ""
            if rel_path:
                dir_path = "/".join(rel_path.split("/")[:-1]) or "."
                if node.get("is_doc"):
                    self._all_doc_dirs.add(dir_path)
                else:
                    self._all_source_dirs.add(dir_path)

    def _compute_coverage(self):
        """Compute which symbols and directories are covered by candidates."""
        for candidate in self.candidates:
            for nid in candidate.node_ids:
                self._covered_ids.add(nid)
                node = self._all_symbols.get(nid, {})
                rel_path = node.get("rel_path") or ""
                if rel_path:
                    dir_path = "/".join(rel_path.split("/")[:-1]) or "."
                    self._covered_dirs.add(dir_path)
                    if node.get("is_doc"):
                        self._covered_doc_dirs.add(dir_path)

    def report(self, macro_id: Optional[int] = None) -> CoverageReport:
        """Generate a coverage report, optionally scoped to a macro-cluster.

        Parameters
        ----------
        macro_id : int, optional
            If provided, report only for symbols in this macro-cluster.

        Returns
        -------
        CoverageReport
        """
        # Filter to macro if requested
        if macro_id is not None:
            conn = self.db.conn
            rows = conn.execute(
                "SELECT node_id FROM repo_nodes "
                "WHERE macro_cluster = ? AND is_architectural = 1",
                (macro_id,),
            ).fetchall()
            scope_ids = {dict(r)["node_id"] for r in rows}
        else:
            scope_ids = set(self._all_symbols.keys())

        total = len(scope_ids)
        covered = len(scope_ids & self._covered_ids)
        uncovered = scope_ids - self._covered_ids

        # High-value uncovered symbols: those with high SYMBOL_TYPE_PRIORITY
        uncovered_hv = []
        for nid in uncovered:
            node = self._all_symbols.get(nid, {})
            stype = (node.get("symbol_type") or "").lower()
            priority = SYMBOL_TYPE_PRIORITY.get(stype, 0)
            if priority >= 7:  # function or higher
                uncovered_hv.append({
                    "node_id": nid,
                    "symbol_name": node.get("symbol_name", ""),
                    "symbol_type": stype,
                    "priority": priority,
                })

        # Sort by priority descending
        uncovered_hv.sort(key=lambda x: -x["priority"])

        # Doc domain coverage
        total_doc = len(self._all_doc_dirs)
        covered_doc = len(self._all_doc_dirs & self._covered_doc_dirs)
        uncovered_doc = sorted(self._all_doc_dirs - self._covered_doc_dirs)

        # Directory coverage
        all_dirs = self._all_source_dirs | self._all_doc_dirs
        total_dirs = len(all_dirs)
        covered_dirs = len(all_dirs & self._covered_dirs)

        # Page overlap: pairs of candidates sharing >50% node overlap
        overlap_pairs = self._find_overlap_pairs()

        return CoverageReport(
            total_symbols=total,
            covered_symbols=covered,
            uncovered_high_value=uncovered_hv[:20],
            total_doc_domains=total_doc,
            covered_doc_domains=covered_doc,
            uncovered_doc_domains=uncovered_doc[:10],
            total_directories=total_dirs,
            covered_directories=covered_dirs,
            page_overlap_pairs=overlap_pairs,
        )

    def _find_overlap_pairs(self) -> List[Tuple[int, int, int]]:
        """Find candidate pairs with significant overlap (>50% shared nodes)."""
        pairs: List[Tuple[int, int, int]] = []
        for i, a in enumerate(self.candidates):
            a_set = set(a.node_ids)
            for b in self.candidates[i + 1:]:
                b_set = set(b.node_ids)
                overlap = len(a_set & b_set)
                min_size = min(len(a_set), len(b_set))
                if min_size > 0 and overlap / min_size > 0.5:
                    pairs.append((a.micro_id, b.micro_id, overlap))
        return pairs


# ═════════════════════════════════════════════════════════════════════
# Compact LLM Refiner
# ═════════════════════════════════════════════════════════════════════

# ── Prompt template ─────────────────────────────────────────────────

REFINER_SYSTEM_PROMPT = """\
You are a wiki page naming assistant. Given a section's symbols and quality \
flags, output a JSON object with page naming and actions.

Rules:
- Use capability-based names (not file/class names)
- Each page should cover one coherent capability
- Use the quality flags to decide whether to keep, merge, or split pages
- Output ONLY valid JSON, no explanation

Output schema:
{
  "section_name": "string",
  "section_description": "string",
  "page_actions": [
    {
      "action": "keep|merge|split|promote_docs|demote",
      "micro_id": <int>,
      "name": "Page Name",
      "description": "1-2 sentence description",
      "retrieval_query": "search query for content retrieval"
    }
  ]
}
"""


def build_refiner_prompt(
    section_symbols: List[Dict[str, Any]],
    candidate_summaries: List[Dict[str, Any]],
    quality_flags: Dict[str, Any],
) -> str:
    """Build a compact user prompt for the LLM refiner.

    Parameters
    ----------
    section_symbols : list[dict]
        Top symbols in this section: [{name, type, file}].
    candidate_summaries : list[dict]
        Per-page summaries: [{micro_id, classification, identity_score, symbols}].
    quality_flags : dict
        Coverage and quality metrics for this section.

    Returns
    -------
    str
        The user prompt string.
    """
    prompt_parts = []

    # Section symbols (compact)
    sym_lines = []
    for s in section_symbols[:15]:
        sym_lines.append(f"  {s.get('type','?')}: {s.get('name','?')} ({s.get('file','')})")
    prompt_parts.append("## Section Symbols\n" + "\n".join(sym_lines))

    # Candidate summaries
    cand_lines = []
    for c in candidate_summaries:
        micro = c.get("micro_id", "?")
        cls = c.get("classification", "?")
        score = c.get("identity_score", 0)
        syms = ", ".join(c.get("symbols", [])[:5])
        shape = c.get("suggested_shape", "keep")
        cand_lines.append(
            f"  Page {micro} [{cls}] identity={score:.2f} shape={shape} — {syms}"
        )
    prompt_parts.append("## Candidates\n" + "\n".join(cand_lines))

    # Quality flags
    flag_lines = []
    for k, v in quality_flags.items():
        flag_lines.append(f"  {k}: {v}")
    prompt_parts.append("## Quality Flags\n" + "\n".join(flag_lines))

    return "\n\n".join(prompt_parts)


def parse_refiner_output(raw: str) -> Optional[Dict[str, Any]]:
    """Parse LLM refiner output, extracting JSON from the response.

    Returns
    -------
    dict or None
        Parsed JSON object, or None if parsing fails.
    """
    # Try direct parse
    raw = raw.strip()
    if raw.startswith("```"):
        # Strip markdown code fence
        lines = raw.split("\n")
        lines = [l for l in lines if not l.strip().startswith("```")]
        raw = "\n".join(lines).strip()

    try:
        result = json.loads(raw)
        if isinstance(result, dict) and "page_actions" in result:
            return result
    except json.JSONDecodeError:
        pass

    # Try to find JSON object in the response
    start = raw.find("{")
    end = raw.rfind("}")
    if start != -1 and end > start:
        try:
            result = json.loads(raw[start:end + 1])
            if isinstance(result, dict):
                return result
        except json.JSONDecodeError:
            pass

    logger.warning("[REFINER] Failed to parse LLM output: %s...", raw[:200])
    return None
