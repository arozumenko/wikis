"""
Phase 2 Graph Topology Enrichment.

Injects synthetic edges to enrich the AST-derived code graph:
- Orphan resolution (FTS5 lexical → semantic vector → directory fallback)
- Document↔code edges (hyperlinks, backtick mentions, proximity)
- Component bridging (fully connected graph)
- Inverse-in-degree edge weighting
- Hub detection and flagging

Ported from DeepWiki plugin_implementation/graph_topology.py (~1050 lines).

Dependencies:
- networkx (hard)
- numpy (soft — falls back to pure Python for mean/std)
"""

from __future__ import annotations

import json
import logging
import math
import os
import re
from collections import Counter
from typing import Any, Callable

import networkx as nx

logger = logging.getLogger(__name__)

try:
    import numpy as np

    _HAS_NUMPY = True
except ImportError:
    _HAS_NUMPY = False

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Edge weight floor for synthetic (non-AST) edges
SYNTHETIC_WEIGHT_FLOOR = 0.5

# Regex patterns for Markdown content parsing
_MD_LINK_RE = re.compile(r"\[(?:[^\]]*)\]\(([^)]+)\)")
_BACKTICK_REF_RE = re.compile(r"`([A-Za-z_]\w+(?:\.\w+)*)`")

# Document node kinds
_DOC_KINDS = frozenset(
    {
        "module_doc",
        "file_doc",
        "doc",
        "readme",
        "documentation",
        "markdown",
        "markdown_document",
        "rst",
        "text",
    }
)

# Synthetic edge classes (excluded from structural in-degree computation)
_SYNTHETIC_CLASSES = frozenset(
    {
        "directory",
        "lexical",
        "semantic",
        "doc",
        "bridge",
    }
)

# Doc directory prefixes stripped for proximity matching
_DOC_PREFIXES = ("docs/", "doc/", "documentation/")

# Batch size for edge persistence
PERSIST_BATCH_SIZE = 5000


# ---------------------------------------------------------------------------
# Orphan detection
# ---------------------------------------------------------------------------


def find_orphans(G: nx.MultiDiGraph) -> list[str]:
    """Find degree-0 nodes (no incoming AND no outgoing edges)."""
    return [n for n in G.nodes() if G.in_degree(n) == 0 and G.out_degree(n) == 0]


# ---------------------------------------------------------------------------
# Edge helper
# ---------------------------------------------------------------------------


def _add_edge(
    db: Any,
    G: nx.MultiDiGraph,
    source: str,
    target: str,
    rel_type: str,
    edge_class: str,
    created_by: str,
    raw_similarity: float | None = None,
) -> None:
    """Add a synthetic edge to both the in-memory graph AND the DB."""
    attrs: dict[str, Any] = {
        "relationship_type": rel_type,
        "edge_class": edge_class,
        "created_by": created_by,
        "weight": 1.0,
    }
    if raw_similarity is not None:
        attrs["raw_similarity"] = raw_similarity

    G.add_edge(source, target, **attrs)
    if db is not None:
        db.upsert_edge(source, target, rel_type, **attrs)


# ---------------------------------------------------------------------------
# Orphan resolution (3-pass cascade)
# ---------------------------------------------------------------------------


def _expanding_prefixes(rel_path: str) -> list[str]:
    """Generate expanding directory prefixes for a file path.

    Example: "src/auth/token.py" → ["src/auth", "src", ""]
    "" (empty string) means global (no prefix filter).
    """
    parts = rel_path.replace("\\", "/").split("/")
    dir_parts = parts[:-1] if len(parts) > 1 else []
    prefixes: list[str] = []
    while dir_parts:
        prefixes.append("/".join(dir_parts))
        dir_parts = dir_parts[:-1]
    prefixes.append("")
    return prefixes


def _resolve_orphans_by_directory(
    db: Any,
    G: nx.MultiDiGraph,
    orphan_ids: list[str],
) -> int:
    """Last-resort: connect orphan to highest-degree node in same directory."""
    dir_index: dict[str, list[str]] = {}
    for nid, data in G.nodes(data=True):
        rp = data.get("rel_path", "")
        if not rp and db is not None:
            row = db.get_node(nid)
            rp = row.get("rel_path", "") if row else ""
        if not rp:
            continue
        d = rp.rsplit("/", 1)[0] if "/" in rp else "<root>"
        dir_index.setdefault(d, []).append(nid)

    resolved = 0
    for orphan_id in orphan_ids:
        if G.in_degree(orphan_id) > 0 or G.out_degree(orphan_id) > 0:
            continue

        rp = G.nodes.get(orphan_id, {}).get("rel_path", "")
        if not rp and db is not None:
            node = db.get_node(orphan_id)
            rp = node.get("rel_path", "") if node else ""
        if not rp:
            continue

        prefixes = _expanding_prefixes(rp)
        for prefix in prefixes:
            d = prefix if prefix else "<root>"
            candidates = dir_index.get(d, [])
            valid = [
                (n, G.degree(n))
                for n in candidates
                if n != orphan_id and (G.in_degree(n) > 0 or G.out_degree(n) > 0)
            ]
            if valid:
                best = max(valid, key=lambda x: x[1])
                _add_edge(
                    db,
                    G,
                    orphan_id,
                    best[0],
                    "directory_link",
                    "directory",
                    "dir_proximity_fallback",
                )
                resolved += 1
                break

    return resolved


def resolve_orphans(
    db: Any,
    G: nx.MultiDiGraph,
    embedding_fn: Callable | None = None,
    fts_limit: int = 3,
    vec_k: int = 3,
    vec_distance_threshold: float = 0.15,
    max_lexical_edges: int = 2,
) -> dict[str, Any]:
    """3-pass orphan resolution cascade.

    Pass 1: FTS5 lexical search by symbol_name
    Pass 2: Semantic vector search with expanding path prefixes
    Pass 3: Directory proximity fallback
    """
    orphan_ids = find_orphans(G)
    if not orphan_ids:
        return {"orphans_found": 0, "resolved": 0, "lexical": 0, "semantic": 0, "directory": 0}

    stats: dict[str, Any] = {
        "orphans_found": len(orphan_ids),
        "lexical": 0,
        "semantic": 0,
        "directory": 0,
    }
    resolved: set[str] = set()

    # ── Pass 1: FTS5 Lexical ──
    if db is not None:
        for nid in orphan_ids:
            node_data = G.nodes.get(nid, {})
            symbol_name = node_data.get("symbol", {}).get("name", "") or node_data.get(
                "symbol_name", ""
            )
            if not symbol_name or len(symbol_name) < 2:
                continue

            hits = db.search_fts5(query=symbol_name, limit=fts_limit)
            added = 0
            for hit in hits:
                hit_id = hit.get("node_id", "")
                if hit_id == nid or hit_id not in G:
                    continue
                _add_edge(db, G, nid, hit_id, "lexical_link", "lexical", "fts5_lexical")
                added += 1
                if added >= max_lexical_edges:
                    break

            if added > 0:
                resolved.add(nid)
                stats["lexical"] += 1

    # ── Pass 2: Semantic Vector ──
    if embedding_fn and db is not None and db.vec_available:
        for nid in orphan_ids:
            if nid in resolved:
                continue

            node_data = G.nodes.get(nid, {})
            text = node_data.get("source_text", "") or node_data.get("docstring", "")
            if not text or len(text.strip()) < 10:
                continue

            embedding = embedding_fn(text)
            rel_path = node_data.get("location", {}).get("rel_path", "") or node_data.get(
                "rel_path", ""
            )

            prefixes = _expanding_prefixes(rel_path)
            for prefix in prefixes:
                hits = db.search_vec(embedding=embedding, k=vec_k, path_prefix=prefix or None)
                valid_hits = [
                    h
                    for h in hits
                    if h.get("node_id") != nid
                    and h.get("node_id") in G
                    and h.get("distance", 1.0) < vec_distance_threshold
                ]
                if valid_hits:
                    best = valid_hits[0]
                    _add_edge(
                        db,
                        G,
                        nid,
                        best["node_id"],
                        "semantic_link",
                        "semantic",
                        "vec_semantic",
                        raw_similarity=1.0 - best.get("distance", 0),
                    )
                    resolved.add(nid)
                    stats["semantic"] += 1
                    break

    # ── Pass 3: Directory Fallback ──
    remaining_orphans = find_orphans(G)
    dir_resolved = _resolve_orphans_by_directory(db, G, remaining_orphans)
    stats["directory"] = dir_resolved
    stats["resolved"] = stats["lexical"] + stats["semantic"] + dir_resolved

    return stats


# ---------------------------------------------------------------------------
# Document edge injection
# ---------------------------------------------------------------------------


def _is_doc_node(G: nx.MultiDiGraph, node_id: str) -> bool:
    """Check if a node is a documentation node."""
    data = G.nodes.get(node_id, {})
    if data.get("is_doc") == 1:
        return True
    sym_type = data.get("symbol_type", "") or data.get("kind", "")
    if sym_type in _DOC_KINDS:
        return True
    if sym_type.endswith("_document") or sym_type.endswith("_section"):
        return True
    return False


def _normalize_path(ref_path: str, source_dir: str) -> str:
    """Resolve a relative path reference against the source document's directory.

    Returns "" for external URLs, mailto:, and #-only anchors.
    """
    if ref_path.startswith(("http://", "https://", "mailto:", "#")):
        return ""
    ref_path = ref_path.split("#")[0].strip()
    if not ref_path:
        return ""
    combined = os.path.normpath(os.path.join(source_dir, ref_path))
    combined = combined.replace("\\", "/")
    return combined.lstrip("./").lstrip("/")


def _build_path_index(G: nx.MultiDiGraph) -> dict[str, str]:
    """Map rel_path → node_id for all nodes."""
    index: dict[str, str] = {}
    for nid, data in G.nodes(data=True):
        rp = data.get("location", {}).get("rel_path", "") or data.get("rel_path", "")
        if rp:
            index[rp] = nid
    return index


def _build_name_index(G: nx.MultiDiGraph) -> dict[str, list[str]]:
    """Map symbol_name → [node_ids] for all code nodes."""
    index: dict[str, list[str]] = {}
    for nid, data in G.nodes(data=True):
        if _is_doc_node(G, nid):
            continue
        name = data.get("symbol", {}).get("name", "") or data.get("symbol_name", "")
        if name:
            index.setdefault(name, []).append(nid)
    return index


def _extract_hyperlink_edges(
    G: nx.MultiDiGraph,
    path_index: dict[str, str],
    name_index: dict[str, list[str]],
) -> list[tuple[str, str]]:
    """Extract edges from Markdown hyperlinks and backtick mentions."""
    edges: list[tuple[str, str]] = []
    for nid, data in G.nodes(data=True):
        if not _is_doc_node(G, nid):
            continue

        content = data.get("source_text", "") or data.get("content", "")
        if not content:
            continue

        source_dir = data.get("location", {}).get("rel_path", "") or data.get("rel_path", "")
        source_dir = "/".join(source_dir.replace("\\", "/").split("/")[:-1])

        # Tier 1: Markdown [text](path) links
        for match in _MD_LINK_RE.finditer(content):
            ref_path = match.group(1)
            resolved = _normalize_path(ref_path, source_dir)
            if not resolved:
                continue
            target = path_index.get(resolved)
            if target and target != nid:
                edges.append((nid, target))

        # Tier 2: Backtick `SymbolName` mentions (capped at 2 per mention)
        for match in _BACKTICK_REF_RE.finditer(content):
            symbol_ref = match.group(1)
            targets = name_index.get(symbol_ref, [])
            for target in targets[:2]:
                if target != nid:
                    edges.append((nid, target))

    return edges


def _extract_proximity_edges(
    G: nx.MultiDiGraph,
    path_index: dict[str, str],
) -> list[tuple[str, str]]:
    """Match doc directories to code directories.

    Strips docs/, doc/, documentation/ prefixes from doc paths and
    matches to code paths sharing the same directory structure.
    """
    edges: list[tuple[str, str]] = []
    for nid, data in G.nodes(data=True):
        if not _is_doc_node(G, nid):
            continue

        rp = data.get("location", {}).get("rel_path", "") or data.get("rel_path", "")
        if not rp:
            continue

        doc_dir = "/".join(rp.replace("\\", "/").split("/")[:-1]) + "/"

        code_dir = doc_dir
        for prefix in _DOC_PREFIXES:
            if doc_dir.startswith(prefix):
                code_dir = doc_dir[len(prefix) :]
                break

        if code_dir == doc_dir:
            continue  # No prefix was stripped — skip

        count = 0
        for path, target in path_index.items():
            if target != nid and path.startswith(code_dir):
                edges.append((nid, target))
                count += 1
                if count >= 5:
                    break

    return edges


def inject_doc_edges(db: Any, G: nx.MultiDiGraph) -> dict[str, Any]:
    """Inject documentation → code edges.

    Tier 1: Hyperlinks [text](path) + backtick `Symbol` mentions
    Tier 2: Directory proximity (docs/ → src/)
    """
    path_index = _build_path_index(G)
    name_index = _build_name_index(G)
    seen: set[tuple[str, str]] = set()

    hyperlink_edges = _extract_hyperlink_edges(G, path_index, name_index)
    hyperlink_count = 0
    for src, tgt in hyperlink_edges:
        if (src, tgt) not in seen and not G.has_edge(src, tgt):
            _add_edge(db, G, src, tgt, "hyperlink", "doc", "md_hyperlink")
            seen.add((src, tgt))
            hyperlink_count += 1

    proximity_edges = _extract_proximity_edges(G, path_index)
    proximity_count = 0
    for src, tgt in proximity_edges:
        if (src, tgt) not in seen and not G.has_edge(src, tgt):
            _add_edge(db, G, src, tgt, "proximity", "doc", "dir_proximity")
            seen.add((src, tgt))
            proximity_count += 1

    return {
        "hyperlink_edges": hyperlink_count,
        "proximity_edges": proximity_count,
    }


# ---------------------------------------------------------------------------
# Component bridging
# ---------------------------------------------------------------------------


def bridge_disconnected_components(db: Any, G: nx.MultiDiGraph) -> dict[str, Any]:
    """Connect disconnected components via bridge edges.

    Uses weakly_connected_components on the DiGraph (not to_undirected).
    Linear pass — each smaller component bridges to best-matching larger
    component (by directory histogram similarity).
    """
    components = list(nx.weakly_connected_components(G))

    if len(components) <= 1:
        return {"components_before": len(components), "bridges_added": 0}

    def _dir_hist(comp: set[str]) -> Counter:
        hist: Counter = Counter()
        for nid in comp:
            data = G.nodes.get(nid, {})
            rp = data.get("rel_path") or data.get("file_name") or ""
            d = rp.rsplit("/", 1)[0] if "/" in rp else "<root>"
            hist[d] += 1
        return hist

    def _dir_sim(a: Counter, b: Counter) -> int:
        return sum((a & b).values())

    comp_list = sorted([set(c) for c in components], key=len, reverse=True)
    n_before = len(comp_list)
    comp_dirs = [_dir_hist(c) for c in comp_list]
    bridges_added = 0

    for i in range(1, n_before):
        best_target = 0
        best_sim = _dir_sim(comp_dirs[i], comp_dirs[0])

        for j in range(1, i):
            sim = _dir_sim(comp_dirs[i], comp_dirs[j])
            if sim > best_sim:
                best_sim = sim
                best_target = j

        center_i = max(comp_list[i], key=lambda n: G.degree(n))
        center_j = max(comp_list[best_target], key=lambda n: G.degree(n))

        _add_edge(db, G, center_i, center_j, "component_bridge", "bridge", "component_bridging")
        _add_edge(db, G, center_j, center_i, "component_bridge", "bridge", "component_bridging")
        bridges_added += 2

    return {"components_before": n_before, "bridges_added": bridges_added}


# ---------------------------------------------------------------------------
# Edge weighting
# ---------------------------------------------------------------------------


def apply_edge_weights(G: nx.MultiDiGraph) -> dict[str, Any]:
    """Apply inverse-in-degree edge weighting to ALL edges.

    Formula: weight = 1 / log(structural_in_degree + 2)
    Synthetic edges get a weight floor of SYNTHETIC_WEIGHT_FLOOR (0.5).
    """
    structural_in: dict[str, int] = {n: 0 for n in G.nodes()}
    for _u, v, data in G.edges(data=True):
        ec = data.get("edge_class", "structural")
        if ec not in _SYNTHETIC_CLASSES:
            structural_in[v] = structural_in.get(v, 0) + 1

    weighted = 0
    for u, v, key, data in G.edges(data=True, keys=True):
        in_deg = structural_in.get(v, 0)
        w = 1.0 / math.log(in_deg + 2)

        ec = data.get("edge_class", "structural")
        if ec in _SYNTHETIC_CLASSES:
            w = max(w, SYNTHETIC_WEIGHT_FLOOR)

        G[u][v][key]["weight"] = w
        weighted += 1

    return {"edges_weighted": weighted}


# ---------------------------------------------------------------------------
# Hub detection
# ---------------------------------------------------------------------------


def detect_hubs(G: nx.MultiDiGraph, z_threshold: float = 3.0) -> set[str]:
    """Detect hub nodes via Z-score on in-degree."""
    if G.number_of_nodes() < 3:
        return set()

    degrees = [G.in_degree(n) for n in G.nodes()]

    if _HAS_NUMPY:
        arr = np.array(degrees, dtype=float)
        mean = float(np.mean(arr))
        std = float(np.std(arr))
    else:
        mean = sum(degrees) / len(degrees)
        variance = sum((d - mean) ** 2 for d in degrees) / len(degrees)
        std = variance**0.5

    if std == 0:
        return set()

    hubs: set[str] = set()
    for nid, deg in zip(G.nodes(), degrees):
        z = (deg - mean) / std
        if z > z_threshold:
            hubs.add(nid)

    return hubs


def flag_hubs_in_db(db: Any, hubs: set[str]) -> None:
    """Persist hub flags to the unified DB."""
    if db is None:
        return
    for nid in hubs:
        db.set_hub(nid, is_hub=True)
    db.conn.commit()


# ---------------------------------------------------------------------------
# Edge persistence
# ---------------------------------------------------------------------------


def persist_weights_to_db(db: Any, G: nx.MultiDiGraph) -> int:
    """Write all weighted edges back to the DB (full replace)."""
    if db is None:
        return 0

    db.conn.execute("DELETE FROM repo_edges")

    batch: list[dict[str, Any]] = []
    for u, v, _key, data in G.edges(data=True, keys=True):
        edge_dict: dict[str, Any] = {
            "source_id": u,
            "target_id": v,
            "relationship_type": data.get("relationship_type", ""),
            "edge_class": data.get("edge_class", "structural"),
            "analysis_level": data.get("analysis_level", "comprehensive"),
            "weight": data.get("weight", 1.0),
            "raw_similarity": data.get("raw_similarity"),
            "source_file": data.get("source_file", ""),
            "target_file": data.get("target_file", ""),
            "language": data.get("language", ""),
            "annotations": (
                json.dumps(data["annotations"])
                if isinstance(data.get("annotations"), dict)
                else data.get("annotations", "")
            ),
            "created_by": data.get("created_by", "ast"),
        }
        batch.append(edge_dict)

        if len(batch) >= PERSIST_BATCH_SIZE:
            db.upsert_edges_batch(batch)
            batch = []

    if batch:
        db.upsert_edges_batch(batch)

    db.conn.commit()
    return db.edge_count()


# ---------------------------------------------------------------------------
# Main orchestrator
# ---------------------------------------------------------------------------


def run_phase2(
    db: Any,
    G: nx.MultiDiGraph,
    embedding_fn: Callable | None = None,
    z_threshold: float = 3.0,
    vec_distance_threshold: float = 0.15,
) -> dict[str, Any]:
    """Run the complete Phase 2 enrichment pipeline.

    Execution order (DeepWiki "corrected order — 7F"):
    1. resolve_orphans → inject semantic/lexical/directory edges
    2. inject_doc_edges → inject hyperlink + proximity doc edges
    3. bridge_disconnected_components → connect disconnected components
    4. apply_edge_weights → inverse in-degree weighting on ALL edges
    5. detect_hubs + flag_hubs_in_db → hub detection & DB flagging
    6. persist_weights_to_db → write all weighted edges back to DB
    7. Store metadata
    """
    results: dict[str, Any] = {}

    # Step 1: Orphan resolution
    orphan_stats = resolve_orphans(
        db,
        G,
        embedding_fn,
        vec_distance_threshold=vec_distance_threshold,
    )
    results["orphan_resolution"] = orphan_stats
    logger.info("Orphan resolution: %s", orphan_stats)

    # Step 2: Document edges
    doc_stats = inject_doc_edges(db, G)
    results["doc_edges"] = doc_stats
    logger.info("Doc edges: %s", doc_stats)

    # Step 3: Component bridging
    bridge_stats = bridge_disconnected_components(db, G)
    results["bridging"] = bridge_stats
    logger.info("Bridging: %s", bridge_stats)

    # Step 4: Edge weighting
    weight_stats = apply_edge_weights(G)
    results["weighting"] = weight_stats
    logger.info("Weighting: %s", weight_stats)

    # Step 5: Hub detection
    hubs = detect_hubs(G, z_threshold)
    flag_hubs_in_db(db, hubs)
    results["hubs"] = {"count": len(hubs), "nodes": list(hubs)[:20]}
    logger.info("Hubs detected: %d", len(hubs))

    # Step 6: Persist weighted edges
    edge_count = persist_weights_to_db(db, G)
    results["persisted_edges"] = edge_count
    logger.info("Persisted %d edges", edge_count)

    # Step 7: Metadata
    if db is not None:
        db.set_meta("phase2_completed", True)
        db.set_meta("phase2_stats", results)

    return results
