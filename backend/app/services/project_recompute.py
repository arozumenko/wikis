"""Phase 7/8 — Project recompute service (PR-14).

Drives the cross-repo pipeline end-to-end for a single project:

1. For every wiki in the project, extract relatedness features
   (languages, dependency tokens, API surfaces, top symbol names) from
   its persisted code graph.
2. Score every wiki pair via :func:`score_repo_pair`, persist
   ``repo_relatedness`` rows.
3. Build a merged cross-repo graph, run :func:`build_cross_repo_edges`
   over pairs above ``flags.relatedness_threshold``, persist
   ``project_edges`` rows with ``edge_class='cross_repo'``.
4. Optionally run a project-level Leiden community detection on the
   cross-repo subgraph (``flags.project_clustering``) and persist the
   community map under ``project_meta['communities']``.
5. Stamp ``project_meta['recomputed_at']`` with a UTC ISO timestamp so
   :func:`is_project_stale` can short-circuit reads.

Emits SSE events through the optional ``on_event`` callback so that an
SPA can render a progress widget identical to the wiki-generation flow.

Gated by ``flags.project_graph``. When the flag is OFF, the service is a
fast no-op and returns ``{"status": "skipped"}``.
"""

from __future__ import annotations

import asyncio
import logging
from datetime import datetime, timezone
from typing import Any, Awaitable, Callable, Iterable

from app.config import Settings
from app.events import (
    cross_repo_linker_progress,
    project_clustering_progress,
    project_relatedness_progress,
)
from app.storage.base import ArtifactStorage

logger = logging.getLogger(__name__)

__all__ = ["recompute_project"]


EventEmitter = Callable[[Any], Awaitable[None] | None]


# ----------------------------------------------------------------------
# Internal helpers
# ----------------------------------------------------------------------


async def _maybe_emit(on_event: EventEmitter | None, event: Any) -> None:
    """Safely fire an ``on_event`` callback (sync or async)."""
    if on_event is None:
        return
    try:
        result = on_event(event)
        if asyncio.iscoroutine(result):
            await result
    except Exception:  # pragma: no cover — defensive
        logger.debug("project recompute on_event raised", exc_info=True)


def _features_from_graph(graph) -> dict[str, list[str]]:
    """Pull (languages, deps, api_keys, symbol_names) out of a NX graph."""
    languages: set[str] = set()
    deps: set[str] = set()
    api_keys: set[str] = set()
    symbol_names: list[str] = []

    if graph is None:
        return {"languages": [], "deps": [], "api": [], "symbol_names": []}

    for _node_id, data in graph.nodes(data=True):
        symbol = data.get("symbol") or {}
        lang = symbol.get("language") or data.get("language")
        if lang:
            languages.add(str(lang).lower())

        name = symbol.get("name") or data.get("name")
        if name:
            symbol_names.append(str(name))

        surfaces = data.get("api_surface") or []
        for surf in surfaces:
            key = None
            if isinstance(surf, dict):
                key = surf.get("surface") or surf.get("key")
            if key:
                api_keys.add(str(key))

    # deps: scan import-class edges for target module path / package token
    for _src, tgt, edge_data in graph.edges(data=True):
        rel = edge_data.get("relationship_type") or edge_data.get("type")
        if rel and "import" in str(rel).lower():
            tgt_data = graph.nodes.get(tgt) or {}
            sym = tgt_data.get("symbol") or {}
            mod = sym.get("module") or sym.get("name") or tgt
            if mod:
                deps.add(str(mod).split(".")[0])

    return {
        "languages": sorted(languages),
        "deps": sorted(deps),
        "api": sorted(api_keys),
        "symbol_names": symbol_names,
    }


def _maybe_run_project_leiden(merged_graph, edge_rows: list[dict]) -> dict[str, int]:
    """Run a best-effort Louvain/Leiden over the cross-repo edge subgraph.

    Returns a mapping ``node_id -> community_id``. Empty when networkx
    community detection is unavailable or the subgraph has no edges.
    """
    if not edge_rows:
        return {}
    try:  # networkx 3.x ships louvain_communities.
        import networkx as nx
        from networkx.algorithms.community import louvain_communities  # type: ignore
    except Exception:  # pragma: no cover — defensive
        logger.debug("louvain_communities unavailable; skipping project leiden")
        return {}

    sub = nx.Graph()
    for row in edge_rows:
        s = row["source_node_id"]
        t = row["target_node_id"]
        w = float(row.get("weight") or 0.0)
        if sub.has_edge(s, t):
            sub[s][t]["weight"] += w
        else:
            sub.add_edge(s, t, weight=w)

    try:
        communities = louvain_communities(sub, weight="weight", seed=42)
    except Exception:  # pragma: no cover
        logger.debug("louvain_communities failed", exc_info=True)
        return {}

    out: dict[str, int] = {}
    for cid, members in enumerate(communities):
        for node in members:
            out[str(node)] = cid
    return out


# ----------------------------------------------------------------------
# Public — recompute_project
# ----------------------------------------------------------------------


async def recompute_project(
    project_id: str,
    *,
    user_id: str,
    storage: ArtifactStorage,
    settings: Settings,
    progress_token: str | None = None,
    on_event: EventEmitter | None = None,
) -> dict[str, Any]:
    """Recompute relatedness, cross-repo edges, and clustering for a project.

    Returns a stats dict including ``status``, ``wiki_count``,
    ``pair_count``, ``edge_count``, and (when clustering ran)
    ``community_count``.
    """
    from app.core.code_graph.cross_repo_linker import build_cross_repo_edges
    from app.core.code_graph.relatedness_scorer import score_repo_pair
    from app.core.feature_flags import get_feature_flags
    from app.core.storage.project_storage import open_project_storage
    from app.db import get_session_factory
    from app.services.project_service import ProjectService
    from app.services.toolkit_bridge import build_engine_components

    flags = get_feature_flags()
    if not flags.project_graph:
        return {"status": "skipped", "reason": "project_graph_disabled"}

    token = progress_token or f"project-{project_id}"

    # --- Load wiki list ------------------------------------------------
    session_factory = get_session_factory()
    async with session_factory() as session:
        svc = ProjectService(session)
        wikis = await svc.list_project_wikis(project_id, user_id)
    if wikis is None:
        return {"status": "error", "reason": "project_not_found_or_inaccessible"}

    wiki_ids = [w.id for w in wikis if getattr(w, "status", "complete") == "complete"]
    cap = max(int(getattr(flags, "project_max_wikis", 50) or 0), 0)
    if cap > 0:
        wiki_ids = wiki_ids[:cap]
    if len(wiki_ids) < 2:
        return {
            "status": "skipped",
            "reason": "fewer_than_two_wikis",
            "wiki_count": len(wiki_ids),
        }

    # --- Open project storage -----------------------------------------
    project_storage = open_project_storage(
        project_id,
        cache_dir=getattr(settings, "cache_dir", None),
        settings=settings,
    )

    # --- Per-wiki feature extraction ----------------------------------
    feature_map: dict[str, dict[str, list[str]]] = {}
    merged_graph = None
    try:
        import networkx as nx

        merged_graph = nx.MultiDiGraph()
    except Exception:  # pragma: no cover
        merged_graph = None

    for wid in wiki_ids:
        try:
            comp = await build_engine_components(wid, storage, settings, tier="low")
        except Exception:
            logger.warning("recompute: failed to build components for %s", wid, exc_info=True)
            continue
        graph = getattr(comp, "code_graph", None)
        feature_map[wid] = _features_from_graph(graph)

        if merged_graph is not None and graph is not None:
            for nid, data in graph.nodes(data=True):
                tagged = dict(data)
                tagged["wiki_id"] = wid
                merged_graph.add_node(nid, **tagged)
            for s, t, edata in graph.edges(data=True):
                merged_graph.add_edge(s, t, **edata)

    # --- Pair scoring -------------------------------------------------
    pair_threshold = float(getattr(flags, "relatedness_threshold", 0.15) or 0.15)
    relatedness: dict[tuple[str, str], float] = {}
    pair_count = 0
    total_pairs = max(len(wiki_ids) * (len(wiki_ids) - 1) // 2, 1)

    for i, wa in enumerate(wiki_ids):
        for wb in wiki_ids[i + 1 :]:
            pair_count += 1
            fa = feature_map.get(wa)
            fb = feature_map.get(wb)
            if not fa or not fb:
                continue
            score, breakdown = score_repo_pair(
                languages_a=fa["languages"],
                languages_b=fb["languages"],
                deps_a=fa["deps"],
                deps_b=fb["deps"],
                api_a=fa["api"],
                api_b=fb["api"],
                symbol_names_a=fa["symbol_names"],
                symbol_names_b=fb["symbol_names"],
            )
            key = (wa, wb) if wa <= wb else (wb, wa)
            relatedness[key] = score
            try:
                project_storage.upsert_repo_relatedness(
                    project_id, key[0], key[1], score, breakdown
                )
            except Exception:
                logger.warning(
                    "recompute: failed to persist relatedness for %s/%s",
                    wa,
                    wb,
                    exc_info=True,
                )
            await _maybe_emit(
                on_event,
                project_relatedness_progress(
                    token,
                    pair_count,
                    total_pairs,
                    f"scored {wa} ↔ {wb}: {score:.3f}",
                    project_id=project_id,
                ),
            )

    # --- Cross-repo edges --------------------------------------------
    edge_rows: list[dict] = []
    if merged_graph is not None and any(s >= pair_threshold for s in relatedness.values()):
        # Emit a single linker-start event so the UI can swap stages.
        await _maybe_emit(
            on_event,
            cross_repo_linker_progress(
                token,
                0,
                1,
                "building cross-repo edges",
                project_id=project_id,
            ),
        )
        try:
            edge_rows = build_cross_repo_edges(
                wiki_ids=wiki_ids,
                merged_graph=merged_graph,
                relatedness=relatedness,
                threshold=pair_threshold,
                dampening=float(getattr(flags, "cross_repo_dampening", 0.7) or 0.7),
                max_wikis=cap if cap > 0 else len(wiki_ids),
            )
        except Exception:
            logger.warning("recompute: build_cross_repo_edges failed", exc_info=True)
            edge_rows = []

        for row in edge_rows:
            try:
                project_storage.upsert_project_edge(
                    project_id,
                    row["source_node_id"],
                    row["target_node_id"],
                    row["edge_class"],
                    float(row.get("weight") or 0.0),
                    row.get("provenance") or {},
                )
            except Exception:
                logger.debug("upsert_project_edge failed", exc_info=True)

        await _maybe_emit(
            on_event,
            cross_repo_linker_progress(
                token,
                1,
                1,
                f"persisted {len(edge_rows)} cross-repo edges",
                project_id=project_id,
            ),
        )

    # --- Optional clustering -----------------------------------------
    community_count = 0
    if flags.project_clustering and edge_rows:
        await _maybe_emit(
            on_event,
            project_clustering_progress(
                token, 0, 1, "running project leiden", project_id=project_id
            ),
        )
        community_map = _maybe_run_project_leiden(merged_graph, edge_rows)
        community_count = len(set(community_map.values())) if community_map else 0
        if community_map:
            try:
                project_storage.set_project_meta(project_id, "communities", community_map)
            except Exception:
                logger.debug("persist communities meta failed", exc_info=True)
        await _maybe_emit(
            on_event,
            project_clustering_progress(
                token,
                1,
                1,
                f"detected {community_count} communities",
                project_id=project_id,
                community_count=community_count,
            ),
        )

    # --- Stamp recompute timestamp -----------------------------------
    now_iso = datetime.now(timezone.utc).isoformat()
    try:
        project_storage.set_project_meta(project_id, "recomputed_at", now_iso)
        project_storage.set_project_meta(project_id, "stale", False)
    except Exception:
        logger.debug("persist recomputed_at meta failed", exc_info=True)

    return {
        "status": "ok",
        "wiki_count": len(wiki_ids),
        "pair_count": pair_count,
        "edge_count": len(edge_rows),
        "community_count": community_count,
        "recomputed_at": now_iso,
    }


def mark_projects_for_wiki_stale(
    *,
    wiki_id: str,
    project_ids: Iterable[str],
    settings: Settings,
) -> None:
    """Mark every project that contains ``wiki_id`` as stale.

    Cheap, synchronous, and safe to call from inside the wiki rebuild
    success path. PR-15 will read ``project_meta['stale']`` /
    ``recomputed_at`` to decide whether to enqueue a recompute.
    """
    from app.core.feature_flags import get_feature_flags
    from app.core.storage.project_storage import open_project_storage

    flags = get_feature_flags()
    if not flags.project_graph:
        return

    for pid in project_ids:
        try:
            store = open_project_storage(
                pid,
                cache_dir=getattr(settings, "cache_dir", None),
                settings=settings,
            )
            store.set_project_meta(pid, "stale", True)
            store.set_project_meta(
                pid, "stale_reason", f"wiki_rebuilt:{wiki_id}"
            )
        except Exception:
            logger.debug(
                "mark_projects_for_wiki_stale failed for %s", pid, exc_info=True
            )
