"""
Hybrid Fusion — Reciprocal Rank Fusion (RRF) for multi-source search.

Combines ranked result lists from different retrieval backends (vector store,
FTS5, brute-force graph search) into a single ranked list using the standard
RRF formula:

    RRF_score(d) = Σ  1 / (k + rank_i(d))   for each retrieval system *i*

with the standard constant *k = 60*.

RRF is rank-based (not score-based), so it gracefully handles backends that
produce incomparable score scales (BM25 floats vs cosine similarity vs ad-hoc
keyword match scores).

Feature flag
~~~~~~~~~~~~
``WIKIS_HYBRID_FUSION=1`` enables RRF in ``search_codebase``.
When OFF, the original simple concatenation logic is used.

References
~~~~~~~~~~
Cormack, Clarke & Butt, "Reciprocal Rank Fusion outperforms Condorcet and
individual Rank Learning Methods", SIGIR 2009.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field

from langchain_core.documents import Document

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Feature flag
# ---------------------------------------------------------------------------
HYBRID_FUSION_ENABLED: bool = os.environ.get("WIKIS_HYBRID_FUSION", "0") == "1"

# ---------------------------------------------------------------------------
# RRF constant (standard value from Cormack et al. 2009)
# ---------------------------------------------------------------------------
RRF_K: int = 60


# ---------------------------------------------------------------------------
# Data class for fused results
# ---------------------------------------------------------------------------
@dataclass
class FusedResult:
    """A single document with its RRF score and contributing ranks."""

    document: Document
    rrf_score: float = 0.0
    # source → rank (1-based)
    source_ranks: dict[str, int] = field(default_factory=dict)

    @property
    def best_rank(self) -> int:
        """Lowest (best) rank across all sources, or 9999 if absent."""
        return min(self.source_ranks.values()) if self.source_ranks else 9999


# ---------------------------------------------------------------------------
# Core RRF implementation
# ---------------------------------------------------------------------------
def reciprocal_rank_fusion(
    ranked_lists: dict[str, list[Document]],
    *,
    k: int = RRF_K,
    cap: int = 0,
    dedup_key: str = "symbol_name",
) -> list[FusedResult]:
    """Merge multiple ranked Document lists using Reciprocal Rank Fusion.

    Parameters
    ----------
    ranked_lists:
        Mapping of source name → ordered list of Documents.
        Example: ``{"vectorstore": vs_docs, "fts5": fts_docs}``
    k:
        RRF constant (default 60).  Higher values compress rank differences.
    cap:
        If > 0, return at most *cap* results.  0 means no cap.
    dedup_key:
        Metadata key used for deduplication.  Documents with the same
        (lowercased) value for this key are treated as the same document,
        and their RRF contributions accumulate.  When two documents share
        the same key, the one with the **better** individual rank is kept
        (i.e. the Document object from the higher-ranked source is used).

    Returns
    -------
    List of ``FusedResult`` objects sorted by descending RRF score.
    Each carries the merged document, total RRF score, and per-source ranks.
    """
    # key_value → FusedResult
    fused: dict[str, FusedResult] = {}

    for source_name, docs in ranked_lists.items():
        for rank_0, doc in enumerate(docs):
            rank = rank_0 + 1  # 1-based rank

            raw_key = _dedup_value(doc, dedup_key)
            if not raw_key:
                # If no dedup key, use page_content hash as unique ID
                raw_key = f"__content__{hash(doc.page_content[:200])}"

            rrf_contribution = 1.0 / (k + rank)

            if raw_key in fused:
                entry = fused[raw_key]
                entry.rrf_score += rrf_contribution
                entry.source_ranks[source_name] = rank
                # Keep the document from the better-ranked source
                if rank < entry.best_rank:
                    # Preserve accumulated metadata but use better-ranked doc
                    entry.document = _merge_metadata(entry.document, doc, source_name)
            else:
                # Tag the document with its source
                doc.metadata.setdefault("search_source", source_name)
                fused[raw_key] = FusedResult(
                    document=doc,
                    rrf_score=rrf_contribution,
                    source_ranks={source_name: rank},
                )

    # Sort by RRF score descending, break ties by best rank
    results = sorted(
        fused.values(),
        key=lambda fr: (-fr.rrf_score, fr.best_rank),
    )

    if cap > 0:
        results = results[:cap]

    return results


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _dedup_value(doc: Document, key: str) -> str:
    """Extract a lowercased dedup key from document metadata."""
    val = doc.metadata.get(key, "")
    return val.strip().lower() if val else ""


def _merge_metadata(existing: Document, new_doc: Document, source: str) -> Document:
    """Merge metadata from *new_doc* into *existing*, keeping the new content.

    The result document carries the new content but accumulates all metadata
    fields from both sides (new wins on conflict).
    """
    merged_meta = {**existing.metadata, **new_doc.metadata}
    # Record all sources that contributed
    sources = set()
    for s in (existing.metadata.get("search_sources", ""), source):
        if s:
            sources.add(s)
    merged_meta["search_sources"] = ",".join(sorted(sources))
    merged_meta["search_source"] = source  # last-wins for display
    return Document(page_content=new_doc.page_content, metadata=merged_meta)


# ---------------------------------------------------------------------------
# Convenience: fuse and return plain Document list
# ---------------------------------------------------------------------------
def fuse_search_results(
    ranked_lists: dict[str, list[Document]],
    *,
    k: int = RRF_K,
    cap: int = 0,
    dedup_key: str = "symbol_name",
) -> list[Document]:
    """Like :func:`reciprocal_rank_fusion` but returns plain Documents.

    Each returned Document has extra metadata:
    - ``rrf_score``: the computed RRF score
    - ``rrf_sources``: comma-separated list of contributing sources
    - ``search_source``: the source of the best-ranked copy

    This is the function that ``search_codebase`` should call.
    """
    fused = reciprocal_rank_fusion(
        ranked_lists,
        k=k,
        cap=cap,
        dedup_key=dedup_key,
    )
    docs: list[Document] = []
    for fr in fused:
        fr.document.metadata["rrf_score"] = round(fr.rrf_score, 6)
        fr.document.metadata["rrf_sources"] = ",".join(sorted(fr.source_ranks.keys()))
        docs.append(fr.document)
    return docs
