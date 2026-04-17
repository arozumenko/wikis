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


def _symbol_attr(node_data: dict, attr: str, default: str = "") -> str:
    """Safely extract a symbol attribute from a graph node.

    The ``symbol`` value on a graph node may be a dict (deepwiki / DB round-trip)
    or a dataclass (fresh graph-builder output).  This helper handles both.
    Falls back to the flat ``symbol_<attr>`` key that the graph builder also stores.
    """
    sym = node_data.get("symbol")
    if sym is not None:
        if isinstance(sym, dict):
            val = sym.get(attr, "")
        else:
            val = getattr(sym, attr, "")
        if val:
            return str(val)
    return node_data.get(f"symbol_{attr}", default)


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
    *,
    skip_db: bool = False,
) -> None:
    """Add a synthetic edge to both the in-memory graph AND the DB.

    When *skip_db* is True the edge is added only to the in-memory graph.
    Use this inside ``resolve_orphans`` / ``inject_doc_edges`` /
    ``bridge_disconnected_components`` because ``persist_weights_to_db``
    rewrites **all** edges to the DB at the end of Phase 2 anyway — the
    per-edge DB upserts would be thrown away.
    """
    attrs: dict[str, Any] = {
        "relationship_type": rel_type,
        "edge_class": edge_class,
        "created_by": created_by,
        "weight": 1.0,
    }
    if raw_similarity is not None:
        attrs["raw_similarity"] = raw_similarity

    G.add_edge(source, target, **attrs)
    if not skip_db and db is not None:
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
    """Connect remaining orphans to the best anchor in same/parent directory.

    For each orphan walk up the directory tree and attach to the
    highest-degree non-orphan node at the first level that has one.

    Copied from deepwiki ``_resolve_orphans_by_directory`` — orphans
    that have no ``rel_path`` or whose prefix chain finds no bucket
    are silently skipped (no global-anchor fabrication).
    """
    if not orphan_ids:
        return 0

    orphan_set = set(orphan_ids)

    # Build directory → [(node_id, total_degree)] for NON-orphan nodes only
    dir_index: dict[str, list[tuple[str, int]]] = {}
    for nid, data in G.nodes(data=True):
        if nid in orphan_set:
            continue
        rp = data.get("rel_path", "")
        if not rp and db is not None:
            row = db.get_node(nid)
            rp = row.get("rel_path", "") if row else ""
        if not rp:
            continue
        d = os.path.dirname(rp).replace("\\", "/") or "<root>"
        deg = G.in_degree(nid) + G.out_degree(nid)
        dir_index.setdefault(d, []).append((nid, deg))

    # Sort each bucket by degree descending (best anchors first)
    for d in dir_index:
        dir_index[d].sort(key=lambda x: x[1], reverse=True)

    edges_added = 0
    for orphan_id in orphan_ids:
        node = db.get_node(orphan_id) if db is not None else None
        rp = ""
        if node:
            rp = node.get("rel_path", "")
        if not rp:
            rp = G.nodes.get(orphan_id, {}).get("rel_path", "")
        if not rp:
            continue  # match deepwiki: silently skip orphans without rel_path

        for prefix in _expanding_prefixes(rp):
            d = prefix if prefix else "<root>"
            candidates = dir_index.get(d, [])
            if candidates:
                anchor_id = candidates[0][0]
                _add_edge(
                    db,
                    G,
                    orphan_id,
                    anchor_id,
                    "directory_link",
                    "directory",
                    "dir_proximity_fallback",
                    skip_db=True,
                )
                edges_added += 1
                break

    return edges_added


def resolve_orphans(
    db: Any,
    G: nx.MultiDiGraph,
    embedding_fn: Callable | None = None,
    embed_batch_fn: Callable | None = None,
    fts_limit: int = 3,
    vec_k: int = 3,
    vec_distance_threshold: float = 0.15,
    max_lexical_edges: int = 2,
    embed_batch_size: int = 64,
) -> dict[str, Any]:
    """3-pass orphan resolution cascade (batched for performance).

    Pass 1: FTS5 lexical search by symbol_name
    Pass 2: Semantic vector search              (batched embeddings)
    Pass 3: Directory proximity fallback        (pure in-memory)

    Edge writes are graph-only (skip_db=True) because persist_weights_to_db
    rewrites all edges at the end of Phase 2.
    """
    import time as _time

    orphan_ids = find_orphans(G)
    if not orphan_ids:
        return {"orphans_found": 0, "resolved": 0, "lexical": 0, "semantic": 0, "directory": 0}

    t0 = _time.monotonic()
    stats: dict[str, Any] = {
        "orphans_found": len(orphan_ids),
        "lexical": 0,
        "semantic": 0,
        "directory": 0,
    }
    resolved: set[str] = set()

    # ── Pass 1: FTS5 Lexical ──
    # Matches deepwiki: symbol_name sourced from DB; counts EDGES added,
    # not resolved orphans.  Hit filter is just ``hit_id != nid`` (no
    # ``in G`` gate) so FTS results for nodes not in the current subgraph
    # are still linked (they share the same DB).
    lexical_edges = 0
    if db is not None:
        orphan_names: list[tuple[str, str]] = []
        for nid in orphan_ids:
            symbol_name = ""
            node = db.get_node(nid)
            if node:
                symbol_name = node.get("symbol_name", "") or ""
            if not symbol_name:
                # Fall back to graph attribute if DB lookup failed
                symbol_name = _symbol_attr(G.nodes.get(nid, {}), "name")
            if symbol_name and len(symbol_name) >= 2:
                orphan_names.append((nid, symbol_name))

        for nid, symbol_name in orphan_names:
            hits = db.search_fts5(query=symbol_name, limit=fts_limit)
            # Filter self-hits BEFORE slicing (matches deepwiki).  FTS5 often
            # ranks the orphan's own row first on exact symbol_name matches;
            # slicing before filtering would silently drop a real edge.
            hits = [h for h in hits if h.get("node_id", "") != nid]
            added = 0
            for hit in hits[:max_lexical_edges]:
                hit_id = hit.get("node_id", "")
                _add_edge(db, G, nid, hit_id, "lexical_link", "lexical", "fts5_lexical",
                          skip_db=True)
                added += 1
                lexical_edges += 1

            if added > 0:
                resolved.add(nid)

        stats["lexical"] = lexical_edges
        t1 = _time.monotonic()
        logger.info(
            "[ORPHAN] Pass 1 (FTS lexical): %d edges added (%d orphans resolved) in %.1fs",
            lexical_edges, len(resolved), t1 - t0,
        )

    # ── Pass 2: Semantic Vector (batched embeddings) ──
    # Matches deepwiki: text sourced from DB (source_text / docstring),
    # candidate only if lexical pass didn't resolve.  Hit filter is only
    # ``hit_id != nid`` + distance threshold (no extra ``in G`` gate).
    _effective_embed = embed_batch_fn or embedding_fn
    if _effective_embed and db is not None and db.vec_available:
        vec_candidates: list[tuple[str, str, str]] = []
        for nid in orphan_ids:
            if nid in resolved:
                continue
            # DB-first (matches deepwiki); fall back to graph attributes so
            # in-memory-only callers still work.
            node = db.get_node(nid)
            text = ""
            rel_path = ""
            if node:
                text = node.get("source_text", "") or node.get("docstring", "") or ""
                rel_path = node.get("rel_path", "") or ""
            if not text:
                node_data = G.nodes.get(nid, {})
                text = node_data.get("source_text", "") or node_data.get("docstring", "") or ""
                if not rel_path:
                    rel_path = (
                        node_data.get("location", {}).get("rel_path", "")
                        or node_data.get("rel_path", "")
                    )
            if not text or len(text.strip()) < 10:
                continue
            vec_candidates.append((nid, text, rel_path))

        t2 = _time.monotonic()

        for batch_start in range(0, len(vec_candidates), embed_batch_size):
            batch = vec_candidates[batch_start : batch_start + embed_batch_size]
            texts = [t for _, t, _ in batch]

            # Batch embed — single API call for *embed_batch_size* texts
            if embed_batch_fn:
                embeddings = embed_batch_fn(texts)
            else:
                embeddings = [embedding_fn(t) for t in texts]

            for (nid, _text, rel_path), emb in zip(batch, embeddings):
                if nid in resolved:
                    continue

                prefixes = _expanding_prefixes(rel_path)
                for prefix in prefixes:
                    hits = db.search_vec(
                        embedding=emb, k=vec_k,
                        path_prefix=prefix or None,
                    )
                    valid_hits = [
                        h for h in hits
                        if h.get("node_id") != nid
                        and h.get("vec_distance", 1.0) < vec_distance_threshold
                    ]
                    if valid_hits:
                        for hit in valid_hits:
                            _add_edge(
                                db, G, nid, hit["node_id"],
                                "semantic_link", "semantic", "vec_semantic",
                                raw_similarity=1.0 - hit.get("vec_distance", 0),
                                skip_db=True,
                            )
                            stats["semantic"] += 1
                        resolved.add(nid)
                        break

        t3 = _time.monotonic()
        logger.info(
            "[ORPHAN] Pass 2 (semantic vector): %d edges added from %d candidates in %.1fs",
            stats["semantic"], len(vec_candidates), t3 - t2,
        )

    # ── Pass 3: Directory Fallback ──
    t4 = _time.monotonic()
    remaining_orphans = find_orphans(G)
    dir_resolved = _resolve_orphans_by_directory(db, G, remaining_orphans)
    stats["directory"] = dir_resolved
    t5 = _time.monotonic()
    logger.info(
        "[ORPHAN] Pass 3 (directory): %d resolved in %.1fs",
        dir_resolved, t5 - t4,
    )

    stats["orphans_remaining"] = len(find_orphans(G))
    stats["resolved"] = stats["orphans_found"] - stats["orphans_remaining"]
    logger.info(
        "[ORPHAN] Total: %d/%d orphans resolved in %.1fs "
        "(lexical=%d, semantic=%d, directory=%d), %d remaining",
        stats["resolved"], stats["orphans_found"], t5 - t0,
        stats["lexical"], stats["semantic"], stats["directory"],
        stats["orphans_remaining"],
    )

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
        name = _symbol_attr(data, "name")
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
            # Dotted-name fallback: `module.ClassName` → try `ClassName`
            if not targets and "." in symbol_ref:
                targets = name_index.get(symbol_ref.rsplit(".", 1)[-1], [])
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

    Algorithm (copied from deepwiki _extract_proximity_edges):
    1. Pre-build ``dir → [code_nodes]`` map, excluding doc nodes.
    2. For each doc node, compute its bare directory (no trailing slash)
       and strip a leading doc prefix if present.
    3. Match only when a code directory equals the stripped topic dir OR
       ends with ``/topic_dir`` (suffix match, not ``startswith``).
    4. Cap at 5 edges per matched directory.

    Previous implementation had three bugs that caused 70× more edges:
    - ``doc_dir`` always had a trailing ``/`` which made stripping produce
      empty ``code_dir`` for top-level docs (e.g. ``docs/README.md``).
    - Candidate pool was ``path_index`` (all nodes, incl. doc targets).
    - Used ``path.startswith(code_dir)`` which matches every file
      recursively under a directory; empty prefix matched everything.
    """
    edges: list[tuple[str, str]] = []

    # Pre-build dir → [code node_ids] map, excluding doc nodes.
    dir_to_code: dict[str, list[str]] = {}
    for nid, data in G.nodes(data=True):
        if _is_doc_node(G, nid):
            continue
        rp = (
            data.get("location", {}).get("rel_path", "")
            or data.get("rel_path", "")
        )
        if rp:
            d = os.path.dirname(rp).replace("\\", "/")
            if d:
                dir_to_code.setdefault(d, []).append(nid)

    for nid, data in G.nodes(data=True):
        if not _is_doc_node(G, nid):
            continue
        rp = (
            data.get("location", {}).get("rel_path", "")
            or data.get("rel_path", "")
        )
        if not rp:
            continue

        doc_dir = os.path.dirname(rp).replace("\\", "/")
        if not doc_dir:
            continue

        topic_dir = doc_dir
        for prefix in _DOC_PREFIXES:
            # _DOC_PREFIXES include trailing slash (e.g. "docs/"), so
            # startswith only matches when doc_dir has subdirectories.
            if topic_dir.startswith(prefix):
                topic_dir = topic_dir[len(prefix):]
                break

        if not topic_dir:
            continue  # guard: empty topic would match everything

        for code_dir, code_nodes in dir_to_code.items():
            if code_dir == topic_dir or code_dir.endswith("/" + topic_dir):
                for cnid in code_nodes[:5]:
                    if cnid != nid:
                        edges.append((nid, cnid))

    return edges


def _enrich_graph_nodes_from_db(db, G: nx.MultiDiGraph) -> None:
    """Populate missing graph node attributes from the DB.

    When graph nodes lack ``rel_path``, ``source_text``, or ``symbol_type``,
    look them up from the unified DB. This ensures doc edge extraction works
    even when graph nodes were created with minimal attributes.
    """
    if db is None:
        return

    _NEEDED = {"rel_path", "source_text", "symbol_type"}
    to_enrich = [
        nid for nid in G.nodes()
        if not (_NEEDED <= set(G.nodes[nid].keys()))
    ]
    if not to_enrich:
        return

    for nid in to_enrich:
        row = db.get_node(nid)
        if not row:
            continue
        node_data = G.nodes[nid]
        for key in ("rel_path", "source_text", "symbol_type", "symbol_name",
                     "is_doc", "language"):
            if key not in node_data and key in row:
                node_data[key] = row[key]


def inject_doc_edges(db: Any, G: nx.MultiDiGraph) -> dict[str, Any]:
    """Inject documentation → code edges.

    Tier 1: Hyperlinks [text](path) + backtick `Symbol` mentions
    Tier 2: Directory proximity (docs/ → src/)
    """
    _enrich_graph_nodes_from_db(db, G)

    path_index = _build_path_index(G)
    name_index = _build_name_index(G)
    seen: set[tuple[str, str]] = set()

    hyperlink_edges = _extract_hyperlink_edges(G, path_index, name_index)
    hyperlink_count = 0
    for src, tgt in hyperlink_edges:
        if (src, tgt) not in seen and not G.has_edge(src, tgt):
            _add_edge(db, G, src, tgt, "hyperlink", "doc", "md_hyperlink", skip_db=True)
            seen.add((src, tgt))
            hyperlink_count += 1

    proximity_edges = _extract_proximity_edges(G, path_index)
    proximity_count = 0
    for src, tgt in proximity_edges:
        if (src, tgt) not in seen and not G.has_edge(src, tgt):
            _add_edge(db, G, src, tgt, "proximity", "doc", "dir_proximity", skip_db=True)
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

        _add_edge(db, G, center_i, center_j, "component_bridge", "bridge", "component_bridging", skip_db=True)
        _add_edge(db, G, center_j, center_i, "component_bridge", "bridge", "component_bridging", skip_db=True)
        bridges_added += 2

    n_after = nx.number_weakly_connected_components(G)

    return {
        "components_before": n_before,
        "components_after": n_after,
        "bridges_added": bridges_added,
    }


# ---------------------------------------------------------------------------
# Edge weighting
# ---------------------------------------------------------------------------


def apply_edge_weights(G: nx.MultiDiGraph) -> dict[str, Any]:
    """Apply inverse-in-degree edge weighting to ALL edges.

    Formula: weight = 1 / log(structural_in_degree + 2)
    Synthetic edges get a weight floor of SYNTHETIC_WEIGHT_FLOOR (0.5).
    """
    if G.number_of_edges() == 0:
        return {"edges_weighted": 0, "synthetic_floored": 0, "min": 0, "max": 0, "mean": 0}

    structural_in: dict[str, int] = {n: 0 for n in G.nodes()}
    for _u, v, data in G.edges(data=True):
        ec = data.get("edge_class", "structural")
        if ec not in _SYNTHETIC_CLASSES:
            structural_in[v] = structural_in.get(v, 0) + 1

    weighted = 0
    synthetic_count = 0
    weights: list[float] = []
    for u, v, key, data in G.edges(data=True, keys=True):
        in_deg = structural_in.get(v, 0)
        w = 1.0 / math.log(in_deg + 2)

        ec = data.get("edge_class", "structural")
        if ec in _SYNTHETIC_CLASSES:
            w = max(w, SYNTHETIC_WEIGHT_FLOOR)
            synthetic_count += 1

        data["weight"] = w
        weights.append(w)
        weighted += 1

    stats = {
        "edges_weighted": weighted,
        "synthetic_floored": synthetic_count,
        "min": round(min(weights), 4),
        "max": round(max(weights), 4),
        "mean": round(sum(weights) / len(weights), 4),
    }
    logger.info(
        "[EDGE_WEIGHTS] %d edges weighted (synthetic_floored=%d, "
        "min=%.4f, max=%.4f, mean=%.4f)",
        weighted, synthetic_count, stats["min"], stats["max"], stats["mean"],
    )
    return stats


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
    db.commit()


# ---------------------------------------------------------------------------
# Edge persistence
# ---------------------------------------------------------------------------


def persist_weights_to_db(db: Any, G: nx.MultiDiGraph) -> int:
    """Write all weighted edges back to the DB (full replace)."""
    if db is None:
        return 0

    db.delete_all_edges()

    batch: list[dict[str, Any]] = []
    for u, v, _key, data in G.edges(data=True, keys=True):
        edge_dict: dict[str, Any] = {
            "source_id": u,
            "target_id": v,
            "rel_type": data.get("relationship_type", ""),
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

    db.commit()
    return db.edge_count()


# ---------------------------------------------------------------------------
# Main orchestrator
# ---------------------------------------------------------------------------


def run_phase2(
    db: Any,
    G: nx.MultiDiGraph,
    embedding_fn: Callable | None = None,
    embed_batch_fn: Callable | None = None,
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
        embed_batch_fn=embed_batch_fn,
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
    G.graph["phase2_enriched"] = True
    if db is not None:
        db.set_meta("phase2_completed", True)
        db.set_meta("phase2_stats", results)

    return results
