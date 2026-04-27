"""
Phase 4 (graph-quality roadmap) — hybrid orphan resolution Pass 2 (FTS+Vec
fused via Reciprocal Rank Fusion).

Public surface
--------------
* :func:`rrf_fuse` — pure helper; combines any number of ranked result
  lists into a single ranked list using ``score = Σ 1 / (k + rank_i)``.
* :func:`resolve_orphans_hybrid` — orphan-level Pass 2 used by the v2
  cascade. Returns ``[hit, ...]`` ranked by ``rrf_score`` desc.

Phase 4 ships only the RRF/hybrid building blocks. Matryoshka 256d
truncation and pgvector ``halfvec`` quantization live in their own
follow-up tickets — they require schema additions (see roadmap
sections 4.2 / 4.3).
"""

from __future__ import annotations

import logging
import os
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple

import networkx as nx

from .feature_flags import FeatureFlags, get_feature_flags

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────────────────
# RRF fusion
# ──────────────────────────────────────────────────────────────────────────


def rrf_fuse(
    *ranked_lists: Sequence[Dict[str, Any]],
    k: int = 60,
    id_key: str = "node_id",
) -> List[Dict[str, Any]]:
    """Fuse N ranked result lists via Reciprocal Rank Fusion.

    For each candidate ``id``, the fused score is::

        rrf_score(id) = Σ_i 1 / (k + rank_i(id))

    where ``rank_i`` is the 1-based position of ``id`` in the *i*-th
    input list (skipping lists where the candidate is absent).

    The returned dicts merge the hit payloads from every input list
    where the candidate appears (later lists win on key conflict),
    plus an attached ``rrf_score`` field. Order is descending by
    ``rrf_score``.

    Args:
        *ranked_lists: One or more lists of result dicts. Each dict
            must carry the ``id_key`` field. Order matters — index
            inside the list is the rank.
        k: RRF constant; larger ``k`` flattens the contribution of
            the top-ranked items. Default 60 (TREC standard).
        id_key: Field used to identify candidates. Default
            ``"node_id"`` (matches our storage layer).

    Returns:
        Single fused list, ordered by ``rrf_score`` desc. Each dict
        has ``rrf_score`` plus the merged payload.
    """
    if k <= 0:
        raise ValueError("k must be > 0")

    scores: Dict[Any, float] = {}
    payloads: Dict[Any, Dict[str, Any]] = {}
    for lst in ranked_lists:
        if not lst:
            continue
        for rank, hit in enumerate(lst, start=1):
            cid = hit.get(id_key)
            if cid is None:
                continue
            scores[cid] = scores.get(cid, 0.0) + 1.0 / (k + rank)
            existing = payloads.get(cid)
            if existing is None:
                payloads[cid] = dict(hit)
            else:
                # Later list contributions overwrite — but never the
                # ``rrf_score`` field we attach below.
                merged = dict(existing)
                for key, val in hit.items():
                    if key != "rrf_score":
                        merged[key] = val
                payloads[cid] = merged

    fused: List[Dict[str, Any]] = []
    for cid, score in scores.items():
        item = dict(payloads[cid])
        item["rrf_score"] = score
        fused.append(item)
    fused.sort(key=lambda h: (h["rrf_score"], h.get(id_key, "")), reverse=True)
    return fused


# ──────────────────────────────────────────────────────────────────────────
# Orphan helpers
# ──────────────────────────────────────────────────────────────────────────


def _orphan_symbol_name(
    G: nx.MultiDiGraph,
    db_node: Optional[Dict[str, Any]],
    node_id: str,
) -> str:
    if db_node:
        name = db_node.get("symbol_name", "") or ""
        if name:
            return name
    data = G.nodes.get(node_id, {}) or {}
    sym = data.get("symbol")
    if isinstance(sym, dict):
        return sym.get("name", "") or ""
    if sym is not None:
        return getattr(sym, "name", "") or ""
    return data.get("symbol_name", "") or ""


def _orphan_rel_path(
    G: nx.MultiDiGraph,
    db_node: Optional[Dict[str, Any]],
    node_id: str,
) -> str:
    if db_node:
        rp = db_node.get("rel_path", "") or ""
        if rp:
            return rp
    data = G.nodes.get(node_id, {}) or {}
    rp = data.get("rel_path", "") or ""
    if not rp:
        loc = data.get("location") or {}
        if isinstance(loc, dict):
            rp = loc.get("rel_path", "") or ""
    return rp


def _local_dir(rel_path: str) -> str:
    if not rel_path:
        return ""
    parts = rel_path.rsplit("/", 1)
    return parts[0] if len(parts) == 2 else ""


# ──────────────────────────────────────────────────────────────────────────
# Hybrid Pass 2
# ──────────────────────────────────────────────────────────────────────────


def resolve_orphans_hybrid(
    db,
    G: nx.MultiDiGraph,
    orphan_id: str,
    *,
    k: Optional[int] = None,
    threshold: Optional[float] = None,
    top_n: Optional[int] = None,
    orphan_embeddings: Optional[Dict[str, Optional[List[float]]]] = None,
    embed_fn: Optional[Callable[[str], List[float]]] = None,
    flags: Optional[FeatureFlags] = None,
) -> List[Dict[str, Any]]:
    """Run hybrid FTS+Vec Pass 2 for *orphan_id*.

    Returns the top ``top_n`` hits, RRF-fused and filtered by
    ``rrf_score >= threshold``. The list is **side-effect free** — the
    caller writes the edges and attaches provenance (``"source":
    "hybrid_rrf"``).

    When ``k`` / ``threshold`` / ``top_n`` are ``None`` they are read
    from :class:`FeatureFlags` (``orphan_rrf_k`` /
    ``orphan_rrf_threshold`` / ``orphan_hybrid_top_n``) so callers
    only need to override for explicit experiments.

    Embedding source order:

    1. ``orphan_embeddings[orphan_id]`` (Phase-3 reuse cache).
    2. ``db.get_embedding_by_id(orphan_id)`` (Phase-0 storage hook).
    3. ``embed_fn(text)`` — last resort; only invoked when the orphan
       has neither a persisted embedding nor a cached one.

    A missing embedding does NOT abort: the hybrid degrades to
    pure-FTS with the same RRF threshold.
    """
    flags = flags or get_feature_flags()
    k = k if k is not None else flags.orphan_rrf_k
    threshold = threshold if threshold is not None else flags.orphan_rrf_threshold
    top_n = top_n if top_n is not None else flags.orphan_hybrid_top_n

    db_node = db.get_node(orphan_id) if db is not None else None
    name = _orphan_symbol_name(G, db_node, orphan_id)
    if not name or len(name) < 2:
        return []

    rel_path = _orphan_rel_path(G, db_node, orphan_id)
    local_dir = _local_dir(rel_path)

    # ── FTS branch
    fts_hits: List[Dict[str, Any]] = []
    try:
        if local_dir and hasattr(db, "search_fts_with_path"):
            fts_hits = db.search_fts_with_path(
                name, path_prefix=local_dir, limit=top_n,
            ) or []
        else:
            fts_hits = db.search_fts5(query=name, limit=top_n) or []
    except Exception as exc:  # noqa: BLE001
        logger.debug("hybrid FTS for %s failed: %s", orphan_id, exc)
        fts_hits = []

    # Drop self-hits + sub-threshold BM25.
    if fts_hits:
        min_norm = flags.fts_min_score_norm
        cleaned: List[Dict[str, Any]] = []
        for h in fts_hits:
            if h.get("node_id") == orphan_id:
                continue
            if min_norm > 0:
                sn = h.get("score_norm")
                if sn is not None and float(sn) < min_norm:
                    continue
            cleaned.append(h)
        fts_hits = cleaned

    # ── Vec branch
    embedding: Optional[List[float]] = None
    if orphan_embeddings is not None:
        embedding = orphan_embeddings.get(orphan_id)
    if embedding is None and db is not None and hasattr(db, "get_embedding_by_id"):
        try:
            embedding = db.get_embedding_by_id(orphan_id)
        except Exception:  # noqa: BLE001
            embedding = None
    if embedding is None and embed_fn is not None:
        text = ""
        if db_node:
            text = (
                db_node.get("source_text", "")
                or db_node.get("docstring", "")
                or ""
            )
        if not text:
            data = G.nodes.get(orphan_id, {}) or {}
            text = data.get("source_text", "") or data.get("docstring", "") or ""
        if text and len(text.strip()) >= 10:
            try:
                embedding = embed_fn(text)
            except Exception as exc:  # noqa: BLE001
                logger.debug("embed_fn for %s failed: %s", orphan_id, exc)
                embedding = None

    vec_hits: List[Dict[str, Any]] = []
    if embedding and db is not None and hasattr(db, "search_vec"):
        try:
            vec_hits = db.search_vec(
                embedding, k=top_n, path_prefix=local_dir or None,
            ) or []
        except TypeError:
            # Older signatures may not accept path_prefix.
            try:
                vec_hits = db.search_vec(embedding, k=top_n) or []
            except Exception as exc:  # noqa: BLE001
                logger.debug("hybrid vec for %s failed: %s", orphan_id, exc)
        except Exception as exc:  # noqa: BLE001
            logger.debug("hybrid vec for %s failed: %s", orphan_id, exc)

    if vec_hits:
        vec_hits = [h for h in vec_hits if h.get("node_id") != orphan_id]

    # ── Fuse
    fused = rrf_fuse(fts_hits, vec_hits, k=k)
    if not fused:
        return []
    return [h for h in fused if h["rrf_score"] >= threshold][:top_n]


__all__ = ["rrf_fuse", "resolve_orphans_hybrid"]
