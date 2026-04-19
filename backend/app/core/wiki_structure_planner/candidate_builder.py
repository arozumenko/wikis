"""
Candidate Builder — Phase 5.

Transforms micro-clusters into ``CandidateRecord`` objects with
classification and quality metrics, ready for validation by the
page validator.

Gated behind ``FeatureFlags.capability_validation``.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set

from ..constants import (
    DOC_CLUSTER_SYMBOLS,
    PAGE_IDENTITY_SYMBOLS,
    SUPPORTING_CODE_SYMBOLS,
    SYMBOL_TYPE_PRIORITY,
)

logger = logging.getLogger(__name__)

# Classification labels
CLASS_CODE = "code"
CLASS_MIXED = "mixed"
CLASS_DOCS = "docs"
CLASS_BRIDGE = "bridge"


@dataclass
class CandidateRecord:
    """A micro-cluster candidate for becoming a wiki page.

    Attributes
    ----------
    macro_id : int
        Parent section (macro-cluster) ID.
    micro_id : int
        Page (micro-cluster) ID.
    node_ids : list[str]
        All node IDs in this micro-cluster.
    classification : str
        One of: ``code``, ``mixed``, ``docs``, ``bridge``.
    code_identity_score : float
        Fraction of nodes that are page-identity symbols (0..1).
    implementation_evidence : float
        Fraction of code nodes with non-trivial source text (0..1).
    docs_dominance : float
        Fraction of nodes that are documentation nodes (0..1).
    public_api_presence : float
        Fraction of code nodes that appear to be public API (0..1).
    utility_contamination : float
        Fraction of code nodes with "util", "helper", "misc" in name (0..1).
    file_spread : int
        Number of distinct files spanned by this cluster.
    node_details : list[dict]
        Node metadata for quick reference.
    """

    macro_id: int
    micro_id: int
    node_ids: List[str]
    classification: str = ""
    code_identity_score: float = 0.0
    implementation_evidence: float = 0.0
    docs_dominance: float = 0.0
    public_api_presence: float = 0.0
    utility_contamination: float = 0.0
    file_spread: int = 0
    node_details: List[Dict[str, Any]] = field(default_factory=list)


# ── Utility name patterns (case-insensitive substrings) ──────────

_UTILITY_PATTERNS = frozenset({
    "util", "helper", "misc", "common", "base",
    "mixin", "compat", "internal", "private",
})

# ── Minimum source text length to count as "has implementation" ──
_MIN_IMPL_LENGTH = 50


def build_candidates(
    db,
    cluster_map: Dict[int, Dict[int, List[str]]],
) -> List[CandidateRecord]:
    """Build ``CandidateRecord`` objects for all micro-clusters.

    Parameters
    ----------
    db : UnifiedWikiDB
        Open database connection.
    cluster_map : dict
        ``{macro_id: {micro_id: [node_ids]}}`` from the planner.

    Returns
    -------
    list[CandidateRecord]
        One record per micro-cluster, with metrics computed.
    """
    candidates: List[CandidateRecord] = []
    conn = db.conn

    for macro_id in sorted(cluster_map.keys()):
        for micro_id in sorted(cluster_map[macro_id].keys()):
            node_ids = cluster_map[macro_id][micro_id]
            if not node_ids:
                continue

            record = _build_one_candidate(
                conn, macro_id, micro_id, node_ids,
            )
            candidates.append(record)

    logger.info(
        "[CANDIDATE_BUILDER] Built %d candidates "
        "(code=%d, mixed=%d, docs=%d, bridge=%d)",
        len(candidates),
        sum(1 for c in candidates if c.classification == CLASS_CODE),
        sum(1 for c in candidates if c.classification == CLASS_MIXED),
        sum(1 for c in candidates if c.classification == CLASS_DOCS),
        sum(1 for c in candidates if c.classification == CLASS_BRIDGE),
    )
    return candidates


def _build_one_candidate(
    conn, macro_id: int, micro_id: int, node_ids: List[str],
) -> CandidateRecord:
    """Compute metrics for a single micro-cluster."""
    # Fetch node metadata
    placeholders = ",".join("?" * len(node_ids))
    rows = conn.execute(
        f"SELECT * FROM repo_nodes WHERE node_id IN ({placeholders})",
        node_ids,
    ).fetchall()
    nodes = [dict(r) for r in rows]

    total = len(nodes) or 1  # avoid division by zero

    # Classify nodes by type
    identity_count = 0
    supporting_count = 0
    doc_count = 0
    impl_count = 0
    public_api_count = 0
    utility_count = 0
    files: Set[str] = set()

    for node in nodes:
        stype = (node.get("symbol_type") or "").lower()
        name = (node.get("symbol_name") or "").lower()
        source = node.get("source_text") or ""
        rel_path = node.get("rel_path") or ""

        if stype in PAGE_IDENTITY_SYMBOLS:
            identity_count += 1
        elif stype in SUPPORTING_CODE_SYMBOLS:
            supporting_count += 1
        elif stype in DOC_CLUSTER_SYMBOLS:
            doc_count += 1

        # Implementation evidence: has substantial source text
        if stype not in DOC_CLUSTER_SYMBOLS and len(source) >= _MIN_IMPL_LENGTH:
            impl_count += 1

        # Public API heuristic: starts with uppercase or no leading underscore
        if stype not in DOC_CLUSTER_SYMBOLS:
            sym_name = node.get("symbol_name") or ""
            if sym_name and not sym_name.startswith("_"):
                public_api_count += 1

        # Utility contamination
        if any(pat in name for pat in _UTILITY_PATTERNS):
            utility_count += 1

        if rel_path:
            files.add(rel_path)

    code_count = identity_count + supporting_count
    code_identity_score = identity_count / total
    docs_dominance = doc_count / total
    impl_evidence = impl_count / max(code_count, 1)
    public_api = public_api_count / max(code_count, 1)
    utility_contamination = utility_count / total
    file_spread = len(files)

    # Classification
    classification = _classify(
        code_identity_score=code_identity_score,
        docs_dominance=docs_dominance,
        code_count=code_count,
        doc_count=doc_count,
        file_spread=file_spread,
        total=total,
    )

    return CandidateRecord(
        macro_id=macro_id,
        micro_id=micro_id,
        node_ids=node_ids,
        classification=classification,
        code_identity_score=code_identity_score,
        implementation_evidence=impl_evidence,
        docs_dominance=docs_dominance,
        public_api_presence=public_api,
        utility_contamination=utility_contamination,
        file_spread=file_spread,
        node_details=nodes,
    )


def _classify(
    *,
    code_identity_score: float,
    docs_dominance: float,
    code_count: int,
    doc_count: int,
    file_spread: int,
    total: int,
) -> str:
    """Classify a micro-cluster into one of four categories."""
    # Docs-dominant: ≥70% documentation nodes
    if docs_dominance >= 0.7:
        return CLASS_DOCS

    # Bridge: high file spread relative to node count (linking cluster)
    # Heuristic: if files > nodes/2 and at least 3 files, likely a bridge
    if file_spread >= 3 and file_spread > total / 2:
        return CLASS_BRIDGE

    # Pure code: no doc nodes at all
    if doc_count == 0 and code_count > 0:
        return CLASS_CODE

    # Mixed: has both code and doc nodes
    if code_count > 0 and doc_count > 0:
        return CLASS_MIXED

    # Fallback
    return CLASS_CODE
