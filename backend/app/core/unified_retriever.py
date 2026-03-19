"""
Unified Retriever — LangChain-compatible retriever backed by UnifiedWikiDB.

Phase 5 of the Unified Graph & Clustering plan: replaces the legacy
WikiRetrieverStack (EnsembleRetriever + FAISS + BM25) with direct SQL
queries against the .wiki.db.

Feature-flagged via ``WIKIS_UNIFIED_RETRIEVER=1`` (default 0).

Key advantages over the legacy stack:
- Single file, single connection (no FAISS + BM25 + docstore coordination)
- Path-prefix filtering (restrict search to a directory subtree)
- Cluster-scoped search (restrict search to a macro-cluster)
- Graph-expansion via SQL JOINs (no NetworkX in-memory graph required at retrieval time)
- FTS5 + sqlite-vec RRF fusion in one query plan

The retriever exposes the same public API as WikiRetrieverStack so the wiki
agent (``wiki_graph_optimized.py``) and deep-research tools can use it as a
drop-in replacement.
"""

import logging
import os
from collections.abc import Callable
from typing import Any

from langchain_core.documents import Document

from .constants import (
    DOC_SYMBOL_TYPES,
    EXPANSION_SYMBOL_TYPES,
)

logger = logging.getLogger(__name__)

# Feature flag
UNIFIED_RETRIEVER_ENABLED = os.getenv("WIKIS_UNIFIED_RETRIEVER", "0") == "1"

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Default ensemble weights (match legacy stack)
DEFAULT_FTS_WEIGHT = 0.4
DEFAULT_VEC_WEIGHT = 0.6

# Max results for internal candidate pools
FTS_POOL = 30
VEC_POOL = 30

# Expansion limits
MAX_EXPANSION_HOPS = 1
MAX_EXPANSION_NEIGHBORS = 15
MAX_EXPANDED_DOCS = 150

# Documentation symbol types — single source of truth from constants.py.
# Used for doc-filtering in search_docs_semantic().
_DOC_SYMBOL_TYPES = DOC_SYMBOL_TYPES

# Architectural symbol types for expansion filtering — from constants.py.
# EXPANSION_SYMBOL_TYPES is the right set: code + module_doc/file_doc,
# excluding method (methods expand via parent class).
_ARCHITECTURAL_TYPES = EXPANSION_SYMBOL_TYPES


def _node_to_document(node: dict[str, Any]) -> Document:
    """Convert a UnifiedWikiDB node dict to a LangChain Document."""
    page_content = node.get("source_text") or ""
    metadata = {
        "node_id": node.get("node_id", ""),
        "symbol_name": node.get("symbol_name", ""),
        "symbol_type": node.get("symbol_type", ""),
        "source": node.get("rel_path", ""),
        "rel_path": node.get("rel_path", ""),
        "file_name": node.get("file_name", ""),
        "language": node.get("language", ""),
        "start_line": node.get("start_line", 0),
        "end_line": node.get("end_line", 0),
        "chunk_type": node.get("chunk_type", node.get("symbol_type", "")),
        "docstring": node.get("docstring", ""),
        "signature": node.get("signature", ""),
        "is_architectural": bool(node.get("is_architectural", 0)),
        "is_doc": bool(node.get("is_doc", 0)),
        "macro_cluster": node.get("macro_cluster"),
        "micro_cluster": node.get("micro_cluster"),
    }
    # Unified score if present (from search_hybrid)
    if "combined_score" in node:
        metadata["combined_score"] = node["combined_score"]
    if "fts_rank" in node:
        metadata["fts_rank"] = node["fts_rank"]
    if "vec_distance" in node:
        metadata["vec_distance"] = node["vec_distance"]

    return Document(page_content=page_content, metadata=metadata)


class UnifiedRetriever:
    """LangChain-compatible retriever backed by ``UnifiedWikiDB``.

    Drop-in replacement for ``WikiRetrieverStack``.  Exposes the same
    public methods:

    - ``search_repository(query, k, apply_expansion)``
    - ``search_docs_semantic(query, k, similarity_threshold)``

    Plus the same attributes the wiki agent reads:

    - ``relationship_graph`` — ``None`` (not needed; expansion is SQL-based)
    - ``content_expander``   — lightweight adapter
    - ``vectorstore_manager`` — lightweight adapter for ``.embeddings``

    Parameters
    ----------
    db : UnifiedWikiDB
        Opened database (readonly or read-write).
    embedding_fn : callable, optional
        ``(str) -> List[float]``  — embed a text query.
        If ``None``, hybrid search falls back to FTS5-only.
    fts_weight : float
        RRF weight for FTS5 (default 0.4).
    vec_weight : float
        RRF weight for vector KNN (default 0.6).
    """

    def __init__(
        self,
        db,  # UnifiedWikiDB — avoid import cycle
        embedding_fn: Callable | None = None,
        fts_weight: float = DEFAULT_FTS_WEIGHT,
        vec_weight: float = DEFAULT_VEC_WEIGHT,
        embeddings: Any | None = None,
    ):
        self.db = db
        self.embedding_fn = embedding_fn
        self.fts_weight = fts_weight
        self.vec_weight = vec_weight

        # Compatibility shim so deep_research can read
        # ``retriever.vectorstore_manager.embeddings``
        self.vectorstore_manager = _EmbeddingsShim(embeddings)

        # Compat shim: wiki agent reads `retriever_stack.relationship_graph`
        self.relationship_graph = None

        # Compat shim: wiki agent reads `retriever_stack.content_expander.graph`
        self.content_expander = _ContentExpanderShim()

        logger.info(
            "UnifiedRetriever initialised (vec=%s, fts_w=%.2f, vec_w=%.2f)",
            db.vec_available if hasattr(db, "vec_available") else "?",
            fts_weight,
            vec_weight,
        )

    # ------------------------------------------------------------------
    # Public API — matches WikiRetrieverStack
    # ------------------------------------------------------------------

    def search_repository(
        self,
        query: str,
        k: int = 15,
        apply_expansion: bool = True,
        path_prefix: str | None = None,
        cluster_id: int | None = None,
    ) -> list[Document]:
        """Hybrid search + optional 1-hop graph expansion.

        When ``apply_expansion=True`` (default), results are enriched with
        1-hop neighbours from ``repo_edges``, filtered to architectural
        symbols.  This mirrors the legacy ``ContentExpander`` behaviour
        but via SQL rather than in-memory NetworkX traversal.

        Returns
        -------
        list[Document]
            LangChain Documents with metadata including ``is_initially_retrieved``.
        """
        if not query or not query.strip():
            return []

        embedding = self._embed(query)

        raw = self.db.search_hybrid(
            query=query,
            embedding=embedding,
            path_prefix=path_prefix,
            cluster_id=cluster_id,
            fts_weight=self.fts_weight,
            vec_weight=self.vec_weight,
            limit=k,
            fts_k=FTS_POOL,
            vec_k=VEC_POOL,
        )

        docs = [_node_to_document(r) for r in raw]
        for doc in docs:
            doc.metadata["is_initially_retrieved"] = True

        logger.info(
            "[UNIFIED_RETRIEVER] Hybrid search returned %d docs for '%.60s'",
            len(docs),
            query,
        )

        if not apply_expansion:
            return docs

        expanded = self._expand_documents(docs, cluster_id=cluster_id)
        logger.info(
            "[UNIFIED_RETRIEVER] After expansion: %d docs (from %d initial)",
            len(expanded),
            len(docs),
        )
        return expanded

    def search_docs_semantic(
        self,
        query: str,
        k: int = 10,
        similarity_threshold: float = 0.7,
    ) -> list[Document]:
        """Semantic search restricted to documentation symbols.

        Uses the same hybrid search but filters to ``is_doc=1`` rows.
        When vector search is available, results are ranked by
        combined RRF score; the ``similarity_threshold`` is applied as a
        minimum combined-score cutoff (normalised to 0‑1).
        """
        if not query or not query.strip():
            return []

        embedding = self._embed(query)

        # Fetch a larger pool and filter to docs client-side
        # (avoids needing a separate doc-only FTS5 table)
        pool_size = max(k * 5, 50)

        raw = self.db.search_hybrid(
            query=query,
            embedding=embedding,
            fts_weight=self.fts_weight,
            vec_weight=self.vec_weight,
            limit=pool_size,
            fts_k=pool_size,
            vec_k=pool_size,
        )

        doc_results = []
        for r in raw:
            stype = (r.get("symbol_type") or "").lower()
            is_doc = bool(r.get("is_doc", 0))
            if is_doc or stype in _DOC_SYMBOL_TYPES or stype.endswith("_document"):
                doc_results.append(r)

        # Apply threshold on combined_score (normalised: max possible ≈ 0.016 for RRF-60)
        # We keep the threshold semantic: if >0 FTS+Vec docs are present, trim tail
        if similarity_threshold > 0 and doc_results:
            max_score = doc_results[0].get("combined_score", 1.0)
            if max_score > 0:
                cutoff = max_score * (1 - similarity_threshold)
                doc_results = [r for r in doc_results if r.get("combined_score", 0) >= cutoff]

        docs = [_node_to_document(r) for r in doc_results[:k]]
        for doc in docs:
            doc.metadata["is_documentation"] = True
            doc.metadata["semantic_retrieved"] = True

        logger.info(
            "[UNIFIED_RETRIEVER] Doc-semantic search: %d docs for '%.60s'",
            len(docs),
            query,
        )
        return docs

    # ------------------------------------------------------------------
    # Graph expansion via SQL
    # ------------------------------------------------------------------

    def _expand_documents(
        self,
        docs: list[Document],
        cluster_id: int | None = None,
    ) -> list[Document]:
        """1-hop graph expansion via SQL, filtered to architectural symbols.

        .. deprecated::
            This is a compatibility shim for the legacy retrieval pipeline
            (``search_codebase`` with ``apply_expansion=True``).  In the new
            cluster-driven pipeline, page symbols are assigned by
            ``cluster_planner`` and expanded by the smart expansion engine
            (``expansion_engine.py``), making retriever-level expansion
            redundant.  Will be removed in Phase 6.

        Mirrors ``ContentExpander.expand_retrieved_documents`` but uses
        ``repo_edges`` directly, avoiding NetworkX in-memory graph.
        """
        seen_ids: set[str] = set()
        expanded: list[Document] = []

        # First: keep all initially-retrieved docs
        for doc in docs:
            nid = doc.metadata.get("node_id", "")
            if nid:
                seen_ids.add(nid)
            expanded.append(doc)

        # Then: 1-hop expand each initial doc
        for doc in docs:
            if len(expanded) >= MAX_EXPANDED_DOCS:
                break

            nid = doc.metadata.get("node_id", "")
            if not nid:
                continue

            neighbors = self._get_expansion_neighbors(nid, seen_ids, cluster_id=cluster_id)
            for neighbor_node in neighbors:
                if len(expanded) >= MAX_EXPANDED_DOCS:
                    break

                neighbor_id = neighbor_node.get("node_id", "")
                if neighbor_id in seen_ids:
                    continue
                seen_ids.add(neighbor_id)

                ndoc = _node_to_document(neighbor_node)
                ndoc.metadata["is_initially_retrieved"] = False
                ndoc.metadata["expanded_from"] = nid
                expanded.append(ndoc)

        return expanded

    def _get_expansion_neighbors(
        self,
        node_id: str,
        seen_ids: set[str],
        cluster_id: int | None = None,
    ) -> list[dict[str, Any]]:
        """Fetch 1-hop neighbours, filtered to architectural types and optionally bounded to cluster."""
        # Outgoing edges
        out_edges = self.db.get_edges_from(node_id)
        # Incoming edges
        in_edges = self.db.get_edges_to(node_id)

        candidate_ids: set[str] = set()
        for e in out_edges:
            tid = e.get("target_id", "")
            if tid and tid not in seen_ids:
                candidate_ids.add(tid)
        for e in in_edges:
            sid = e.get("source_id", "")
            if sid and sid not in seen_ids:
                candidate_ids.add(sid)

        if not candidate_ids:
            return []

        # Fetch node metadata for candidates (batch, capped)
        results = []
        for cid in list(candidate_ids)[:MAX_EXPANSION_NEIGHBORS]:
            node = self.db.get_node(cid)
            if not node:
                continue
            stype = (node.get("symbol_type") or "").lower()
            if stype not in _ARCHITECTURAL_TYPES:
                continue
            # If cluster_id is specified, only include same-cluster neighbours
            if cluster_id is not None and node.get("macro_cluster") != cluster_id:
                continue
            results.append(node)

        return results

    # ------------------------------------------------------------------
    # Embedding helper
    # ------------------------------------------------------------------

    def _embed(self, query: str) -> list[float] | None:
        """Produce an embedding vector, or None if no embedding_fn."""
        if not self.embedding_fn:
            return None
        try:
            return self.embedding_fn(query)
        except Exception as exc:
            logger.warning("[UNIFIED_RETRIEVER] Embedding failed: %s", exc)
            return None

    # ------------------------------------------------------------------
    # Invoke compatibility (LangChain Retriever interface)
    # ------------------------------------------------------------------

    def invoke(self, query: str, **kwargs) -> list[Document]:
        """LangChain runnable interface — delegates to ``search_repository``."""
        return self.search_repository(query, **kwargs)

    def get_relevant_documents(self, query: str, **kwargs) -> list[Document]:
        """Legacy LangChain retriever interface."""
        return self.search_repository(query, **kwargs)


# ═══════════════════════════════════════════════════════════════════════════
# Compatibility shims
# ═══════════════════════════════════════════════════════════════════════════


class _EmbeddingsShim:
    """Shim so ``deep_research`` can access ``retriever.vectorstore_manager.embeddings``.

    Also provides ``embed_query`` for code that calls ``embedding_fn = embeddings.embed_query``.
    """

    def __init__(self, embeddings=None):
        self.embeddings = embeddings

    def embed_query(self, text: str) -> list[float]:
        if self.embeddings and hasattr(self.embeddings, "embed_query"):
            return self.embeddings.embed_query(text)
        raise RuntimeError("No embeddings model available in UnifiedRetriever")


class _ContentExpanderShim:
    """Minimal shim so ``wiki_graph_optimized.py`` reads ``content_expander.graph`` without error."""

    def __init__(self):
        self.graph = None

    def expand_retrieved_documents(self, docs: list[Document]) -> list[Document]:
        """No-op: UnifiedRetriever handles expansion internally."""
        return docs
