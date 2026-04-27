"""
Phase 3 (graph-quality roadmap) — explicit-reference Pass 1 + embedding reuse.

This module implements the high-precision Pass 1 of the v2 orphan
cascade, plus a simple helper to pre-fetch persisted embeddings so
later passes do not have to re-embed.

It does **not** rewire :func:`graph_topology.resolve_orphans`; that
remains gated behind ``FeatureFlags.orphan_cascade_v2`` (default off,
flipped on once Phase 4 hybrid RRF lands).

Public surface
--------------
* :func:`resolve_orphans_explicit_refs` — markdown links + backticks
  for doc orphans, parser-emitted ``imports`` for code orphans.
* :func:`collect_orphan_embeddings` — pre-fetch persisted embeddings
  via the Phase-0 ``get_embedding_by_id`` storage hook.
"""

from __future__ import annotations

import logging
import os
import re
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple

import networkx as nx

from .feature_flags import FeatureFlags, get_feature_flags

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────────────────
# Helpers — re-implemented locally so we do not depend on import-time order
# of graph_topology (which itself imports this module from inside its
# Pass-1 dispatch).
# ──────────────────────────────────────────────────────────────────────────

_MD_LINK_RE = re.compile(r"\[(?:[^\]]*)\]\(([^)]+)\)")
_BACKTICK_REF_RE = re.compile(r"`([A-Za-z_]\w+(?:\.\w+)*)`")


def _is_doc_node(G: nx.MultiDiGraph, node_id: str) -> bool:
    data = G.nodes.get(node_id, {}) or {}
    sym_type = (data.get("symbol_type") or "").lower()
    if sym_type in {"file_doc", "module_doc", "doc"}:
        return True
    rel_path = (data.get("rel_path") or "").lower()
    if not rel_path:
        loc = data.get("location") or {}
        if isinstance(loc, dict):
            rel_path = (loc.get("rel_path") or "").lower()
    return rel_path.endswith((".md", ".rst", ".txt"))


def _normalize_path(ref_path: str, source_dir: str) -> str:
    if ref_path.startswith(("http://", "https://", "mailto:", "#")):
        return ""
    if "#" in ref_path:
        ref_path = ref_path.split("#", 1)[0]
    if not ref_path:
        return ""
    if ref_path.startswith("./"):
        ref_path = ref_path[2:]
    joined = os.path.normpath(os.path.join(source_dir, ref_path))
    return joined.replace("\\", "/")


def _build_path_index(G: nx.MultiDiGraph) -> Dict[str, str]:
    index: Dict[str, str] = {}
    for nid, data in G.nodes(data=True):
        rp = data.get("rel_path", "")
        if not rp:
            loc = data.get("location") or {}
            if isinstance(loc, dict):
                rp = loc.get("rel_path", "")
        if rp:
            index[rp] = nid
    return index


def _build_simple_name_index(G: nx.MultiDiGraph) -> Dict[str, List[str]]:
    """name → [node_id, ...] for non-doc nodes."""
    index: Dict[str, List[str]] = {}
    for nid, data in G.nodes(data=True):
        if _is_doc_node(G, nid):
            continue
        name = data.get("symbol_name", "")
        if not name:
            sym = data.get("symbol")
            if isinstance(sym, dict):
                name = sym.get("name", "") or ""
        if name:
            index.setdefault(name, []).append(nid)
    return index


# ──────────────────────────────────────────────────────────────────────────
# Pass 1 — explicit refs
# ──────────────────────────────────────────────────────────────────────────


def _resolve_doc_orphan_links(
    G: nx.MultiDiGraph,
    orphan_id: str,
    source_text: str,
    source_dir: str,
    path_index: Dict[str, str],
    name_index: Dict[str, List[str]],
) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    seen: set[str] = set()

    for match in _MD_LINK_RE.finditer(source_text):
        ref = match.group(1).split("#", 1)[0].strip()
        if not ref:
            continue
        resolved = _normalize_path(ref, source_dir)
        if not resolved:
            continue
        target_nid = path_index.get(resolved)
        if not target_nid or target_nid == orphan_id or target_nid in seen:
            continue
        seen.add(target_nid)
        out.append({
            "node_id": target_nid,
            "_matcher": "md_link",
            "_raw_score": 0.95,
        })

    for match in _BACKTICK_REF_RE.finditer(source_text):
        symbol = match.group(1)
        targets = name_index.get(symbol, [])
        if not targets and "." in symbol:
            targets = name_index.get(symbol.rsplit(".", 1)[-1], [])
        for tid in targets[:2]:
            if tid == orphan_id or tid in seen:
                continue
            seen.add(tid)
            out.append({
                "node_id": tid,
                "_matcher": "backtick",
                "_raw_score": 0.95,
            })
    return out


def _resolve_code_orphan_imports(
    G: nx.MultiDiGraph,
    orphan_id: str,
    db_node: Optional[Dict[str, Any]],
    path_index: Dict[str, str],
    name_index: Dict[str, List[str]],
) -> List[Dict[str, Any]]:
    """Use the parser-emitted ``imports`` list when available."""
    out: List[Dict[str, Any]] = []
    seen: set[str] = set()

    raw_imports: Iterable[Any] = ()
    if db_node:
        raw_imports = db_node.get("imports") or ()
    if not raw_imports:
        node_data = G.nodes.get(orphan_id, {}) or {}
        raw_imports = node_data.get("imports") or ()

    if not raw_imports:
        return out

    for imp in raw_imports:
        # ``imports`` may be a list[str] (dotted module path) or
        # list[dict] with ``module``/``name``/``rel_path``.
        candidates: List[str] = []
        if isinstance(imp, str):
            candidates.append(imp)
        elif isinstance(imp, dict):
            for key in ("rel_path", "module", "name"):
                v = imp.get(key)
                if isinstance(v, str) and v:
                    candidates.append(v)

        for candidate in candidates:
            # Try direct path lookup first.
            target_nid = path_index.get(candidate)
            if target_nid and target_nid != orphan_id and target_nid not in seen:
                seen.add(target_nid)
                out.append({
                    "node_id": target_nid,
                    "_matcher": "import_path",
                    "_raw_score": 0.95,
                })
                continue
            # Fall back to symbol-name lookup (last dotted segment).
            sym = candidate.rsplit(".", 1)[-1]
            for tid in name_index.get(sym, [])[:2]:
                if tid == orphan_id or tid in seen:
                    continue
                seen.add(tid)
                out.append({
                    "node_id": tid,
                    "_matcher": "import_symbol",
                    "_raw_score": 0.95,
                })
    return out


def resolve_orphans_explicit_refs(
    db,
    G: nx.MultiDiGraph,
    orphan_ids: Iterable[str],
    *,
    flags: Optional[FeatureFlags] = None,
) -> Dict[str, List[Dict[str, Any]]]:
    """Run Pass 1 (explicit refs) for every orphan in *orphan_ids*.

    Returns a ``{orphan_id: [hit, ...]}`` mapping. Each hit is a dict
    with ``node_id``, ``_matcher`` (md_link / backtick / import_path /
    import_symbol), and ``_raw_score`` (0.95).

    The caller is responsible for converting hits into edges and
    attaching ``provenance={"source":"explicit_ref",...}``. Keeping
    this side-effect free makes the pass trivially testable.
    """
    flags = flags or get_feature_flags()  # noqa: F841 — reserved for future toggles

    orphan_list = list(orphan_ids)
    if not orphan_list:
        return {}

    path_index = _build_path_index(G)
    name_index = _build_simple_name_index(G)

    out: Dict[str, List[Dict[str, Any]]] = {}
    for nid in orphan_list:
        node_data = G.nodes.get(nid, {}) or {}
        db_node = db.get_node(nid) if db is not None else None

        if _is_doc_node(G, nid):
            source_text: str = ""
            if db_node:
                source_text = db_node.get("source_text", "") or ""
            if not source_text:
                source_text = node_data.get("source_text", "") or ""
            if not source_text:
                continue
            rel_path = (
                (db_node or {}).get("rel_path", "")
                or node_data.get("rel_path", "")
                or (node_data.get("location") or {}).get("rel_path", "")
                or ""
            )
            source_dir = os.path.dirname(rel_path) if rel_path else ""
            hits = _resolve_doc_orphan_links(
                G, nid, source_text, source_dir, path_index, name_index,
            )
        else:
            hits = _resolve_code_orphan_imports(
                G, nid, db_node, path_index, name_index,
            )

        if hits:
            out[nid] = hits
    return out


# ──────────────────────────────────────────────────────────────────────────
# Embedding reuse
# ──────────────────────────────────────────────────────────────────────────


def collect_orphan_embeddings(
    db,
    orphan_ids: Iterable[str],
    *,
    chunk: int = 500,
) -> Dict[str, Optional[List[float]]]:
    """Pre-fetch persisted embeddings for *orphan_ids*.

    Uses :meth:`get_embedding_by_id` (added in Phase 0). Falls back to
    ``None`` per node when the storage layer cannot return a vector;
    callers should treat ``None`` as "embed on demand".

    Chunking exists to keep round-trip latency low on Postgres; the
    storage hook is per-id by design (no batch get).
    """
    ids = list(orphan_ids)
    if not ids or db is None:
        return {nid: None for nid in ids}

    out: Dict[str, Optional[List[float]]] = {}
    for i in range(0, len(ids), max(1, chunk)):
        batch = ids[i:i + chunk]
        for nid in batch:
            try:
                vec = db.get_embedding_by_id(nid)
            except Exception as exc:  # noqa: BLE001
                logger.debug("get_embedding_by_id(%s) failed: %s", nid, exc)
                vec = None
            out[nid] = vec
    return out


__all__ = [
    "resolve_orphans_explicit_refs",
    "collect_orphan_embeddings",
]
