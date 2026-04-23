# Technical Design: Graph Topology Enrichment

**Spec**: `docs/proposals/graph-topology-enrichment/spec.md`
**Created**: 2026-04-10
**Updated**: 2026-04-10
**Scope**: Port DeepWiki's `graph_topology.py` (Phase 2 enrichment) into Wikis.

---

## 1. Module Structure

### New Files

| File | Purpose |
|------|---------|
| `backend/app/core/graph_topology.py` | Phase 2 enrichment: orphan resolution, doc edges, bridging, weighting, hub detection |

### Modified Files

| File | Change |
|------|--------|
| `backend/app/core/agents/wiki_graph_optimized.py` | Call `run_phase2(db, G, embedding_fn)` after `db.from_networkx(G)` |
| `backend/app/config.py` | Add `WIKIS_GRAPH_ENRICHMENT_ENABLED`, `WIKIS_HUB_Z_THRESHOLD`, `WIKIS_VEC_DISTANCE_THRESHOLD` |

---

## 2. Constants

```python
"""Graph topology enrichment constants — ported from DeepWiki graph_topology.py."""

import re

# Edge weight floor for synthetic (non-AST) edges
SYNTHETIC_WEIGHT_FLOOR = 0.5

# Regex patterns for Markdown content parsing
_MD_LINK_RE = re.compile(r'\[(?:[^\]]*)\]\(([^)]+)\)')
_BACKTICK_REF_RE = re.compile(r'`([A-Za-z_]\w+(?:\.\w+)*)`')

# Document node kinds
_DOC_KINDS = frozenset({
    "module_doc", "file_doc", "doc", "readme", "documentation",
    "markdown", "rst", "text",
})

# Synthetic edge classes (excluded from structural in-degree computation)
_SYNTHETIC_CLASSES = frozenset({
    "directory", "lexical", "semantic", "doc", "bridge",
})

# Doc directory prefixes stripped for proximity matching
_DOC_PREFIXES = ("docs/", "doc/", "documentation/")

# Batch size for edge persistence
PERSIST_BATCH_SIZE = 5000
```

---

## 3. Core Functions

### `graph_topology.py` — Complete Module Design

The file is self-contained. It accepts a `UnifiedWikiDB` instance (`db`) and a `NetworkX MultiDiGraph` (`G`) as inputs. No imports from other project modules.

```python
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

import logging
import math
import re
import json
from collections import Counter
from typing import Any, Callable, Dict, List, Optional, Set, Tuple

import networkx as nx

logger = logging.getLogger(__name__)

try:
    import numpy as np
    _HAS_NUMPY = True
except ImportError:
    _HAS_NUMPY = False
```

### 3.1 Orphan Detection

```python
def find_orphans(G: nx.MultiDiGraph) -> List[str]:
    """Find degree-0 nodes (no incoming AND no outgoing edges).
    
    Returns list of node IDs with in_degree == 0 and out_degree == 0.
    """
    return [n for n in G.nodes() if G.in_degree(n) == 0 and G.out_degree(n) == 0]
```

### 3.2 Edge Helper

```python
def _add_edge(
    db, G: nx.MultiDiGraph,
    source: str, target: str,
    rel_type: str, edge_class: str, created_by: str,
    raw_similarity: Optional[float] = None,
) -> None:
    """Add a synthetic edge to both the in-memory graph AND the DB.
    
    Ensures consistency: graph_topology operates on G (in-memory), but
    edges must also be persisted to the unified DB for downstream use.
    """
    attrs = {
        "relationship_type": rel_type,  # DeepWiki uses "relationship_type", not "rel_type"
        "edge_class": edge_class,
        "created_by": created_by,
        "weight": 1.0,  # Will be recomputed by apply_edge_weights
    }
    if raw_similarity is not None:
        attrs["raw_similarity"] = raw_similarity

    G.add_edge(source, target, **attrs)
    db.upsert_edge(source, target, rel_type, **attrs)
```

### 3.3 Orphan Resolution — FTS5 Lexical (Pass 1)

```python
def resolve_orphans(
    db, G: nx.MultiDiGraph,
    embedding_fn: Optional[Callable] = None,
    fts_limit: int = 3,
    vec_k: int = 3,
    vec_distance_threshold: float = 0.15,
    max_lexical_edges: int = 2,
) -> Dict[str, Any]:
    """3-pass orphan resolution cascade.
    
    Pass 1: FTS5 lexical search by symbol_name
    Pass 2: Semantic vector search with expanding path prefixes
    Pass 3: Directory proximity fallback
    
    Returns stats dict with counts per pass.
    """
    orphan_ids = find_orphans(G)
    if not orphan_ids:
        return {"orphans_found": 0, "resolved": 0}

    stats = {"orphans_found": len(orphan_ids), "lexical": 0, "semantic": 0, "directory": 0}
    resolved = set()

    # ── Pass 1: FTS5 Lexical ──
    for nid in orphan_ids:
        node_data = G.nodes.get(nid, {})
        symbol_name = (
            node_data.get("symbol", {}).get("name", "")
            or node_data.get("symbol_name", "")
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
    if embedding_fn and db.vec_available:
        for nid in orphan_ids:
            if nid in resolved:
                continue

            node_data = G.nodes.get(nid, {})
            text = node_data.get("source_text", "") or node_data.get("docstring", "")
            if not text or len(text.strip()) < 10:
                continue

            embedding = embedding_fn(text)
            rel_path = (
                node_data.get("location", {}).get("rel_path", "")
                or node_data.get("rel_path", "")
            )

            # Expanding path prefix search
            prefixes = _expanding_prefixes(rel_path)
            for prefix in prefixes:
                hits = db.search_vec(embedding=embedding, k=vec_k, path_prefix=prefix or None)
                valid_hits = [
                    h for h in hits
                    if h.get("node_id") != nid
                    and h.get("node_id") in G
                    and h.get("distance", 1.0) < vec_distance_threshold
                ]
                if valid_hits:
                    best = valid_hits[0]
                    _add_edge(
                        db, G, nid, best["node_id"],
                        "semantic_link", "semantic", "vec_semantic",
                        raw_similarity=1.0 - best.get("distance", 0),
                    )
                    resolved.add(nid)
                    stats["semantic"] += 1
                    break  # Stop at first prefix level with results

    # ── Pass 3: Directory Fallback ──
    remaining_orphans = find_orphans(G)  # Re-check after passes 1+2
    dir_resolved = _resolve_orphans_by_directory(db, G, remaining_orphans)
    stats["directory"] = dir_resolved
    stats["resolved"] = stats["lexical"] + stats["semantic"] + dir_resolved

    return stats


def _expanding_prefixes(rel_path: str) -> List[str]:
    """Generate expanding directory prefixes for a file path.
    
    Example: "src/auth/token.py" → ["src/auth", "src", ""]
    "" (empty string) means global (no prefix filter).
    No trailing slash on directory prefixes.
    Root-level files (e.g. "README.md") → [""]
    """
    parts = rel_path.replace("\\", "/").split("/")
    dir_parts = parts[:-1] if len(parts) > 1 else []
    prefixes = []
    while dir_parts:
        prefixes.append("/".join(dir_parts))  # No trailing /
        dir_parts = dir_parts[:-1]
    prefixes.append("")  # Global fallback (empty string, not None)
    return prefixes


def _resolve_orphans_by_directory(
    db, G: nx.MultiDiGraph, orphan_ids: List[str],
) -> int:
    """Last-resort: connect orphan to highest-degree node in same directory.
    
    Builds a directory index first (graph data + DB fallback for rel_path).
    Uses "<root>" for root-level files (matching _expanding_prefixes
    where "" maps to "<root>").
    """
    # Build directory index: dir_name -> [(node_id, degree)]
    dir_index: Dict[str, List[str]] = {}
    for nid, data in G.nodes(data=True):
        rp = data.get("rel_path", "")
        if not rp:
            row = db.get_node(nid)
            rp = row.get("rel_path", "") if row else ""
        if not rp:
            continue
        d = rp.rsplit("/", 1)[0] if "/" in rp else "<root>"
        dir_index.setdefault(d, []).append(nid)

    resolved = 0
    for orphan_id in orphan_ids:
        if G.in_degree(orphan_id) > 0 or G.out_degree(orphan_id) > 0:
            continue  # No longer orphan

        node = db.get_node(orphan_id)  # Read from DB, not just graph
        rp = node.get("rel_path", "") if node else ""
        if not rp:
            continue

        prefixes = _expanding_prefixes(rp)
        for prefix in prefixes:
            d = prefix if prefix else "<root>"
            candidates = dir_index.get(d, [])
            # Find highest-degree non-orphan in this directory
            valid = [
                (n, G.degree(n))
                for n in candidates
                if n != orphan_id and (G.in_degree(n) > 0 or G.out_degree(n) > 0)
            ]
            if valid:
                best = max(valid, key=lambda x: x[1])
                _add_edge(
                    db, G, orphan_id, best[0],
                    "directory_link", "directory", "dir_proximity_fallback",
                )
                resolved += 1
                break

    return resolved
```

### 3.4 Document Edge Injection

```python
def _is_doc_node(G: nx.MultiDiGraph, node_id: str) -> bool:
    """Check if a node is a documentation node.
    
    Checks: is_doc flag first (fast path), then symbol_type membership,
    then suffix heuristics (_document, _section).
    Note: No .lower() call — case-sensitive matching (DeepWiki convention).
    """
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
    Example: _normalize_path("../src/foo.py", "docs/guide") → "src/foo.py"
    """
    import os
    # Guard: skip external URLs, mailto, and anchor-only refs
    if ref_path.startswith(("http://", "https://", "mailto:", "#")):
        return ""
    # Strip anchors (#section)
    ref_path = ref_path.split("#")[0].strip()
    if not ref_path:
        return ""
    combined = os.path.normpath(os.path.join(source_dir, ref_path))
    combined = combined.replace("\\", "/")
    # Strip leading ./ or /
    return combined.lstrip("./").lstrip("/")


def _build_path_index(G: nx.MultiDiGraph) -> Dict[str, str]:
    """Map rel_path → node_id for all nodes."""
    index = {}
    for nid, data in G.nodes(data=True):
        rp = data.get("location", {}).get("rel_path", "") or data.get("rel_path", "")
        if rp:
            index[rp] = nid
    return index


def _build_name_index(G: nx.MultiDiGraph) -> Dict[str, List[str]]:
    """Map symbol_name → [node_ids] for all code nodes."""
    index: Dict[str, List[str]] = {}
    for nid, data in G.nodes(data=True):
        if _is_doc_node(G, nid):
            continue
        name = data.get("symbol", {}).get("name", "") or data.get("symbol_name", "")
        if name:
            index.setdefault(name, []).append(nid)
    return index


def _extract_hyperlink_edges(
    G: nx.MultiDiGraph,
    path_index: Dict[str, str],
    name_index: Dict[str, List[str]],
) -> List[Tuple[str, str]]:
    """Extract edges from Markdown hyperlinks and backtick mentions.
    
    Parses:
    - [text](path) → hyperlink to resolved file/symbol
    - `SymbolName` → hyperlink to symbol node
    """
    edges = []
    for nid, data in G.nodes(data=True):
        if not _is_doc_node(G, nid):
            continue

        content = data.get("source_text", "") or data.get("content", "")
        if not content:
            continue

        source_dir = (
            data.get("location", {}).get("rel_path", "") or data.get("rel_path", "")
        )
        source_dir = "/".join(source_dir.replace("\\", "/").split("/")[:-1])

        # Tier 1: Markdown [text](path) links
        for match in _MD_LINK_RE.finditer(content):
            ref_path = match.group(1)
            # _normalize_path handles external URLs, mailto:, # guards
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
            for target in targets[:2]:  # Cap at 2 matches per backtick ref
                if target != nid:
                    edges.append((nid, target))

    return edges


def _extract_proximity_edges(
    G: nx.MultiDiGraph,
    path_index: Dict[str, str],
) -> List[Tuple[str, str]]:
    """Match doc directories to code directories.
    
    Strips docs/, doc/, documentation/ prefixes from doc paths
    and matches to code paths sharing the same directory structure.
    """
    edges = []
    for nid, data in G.nodes(data=True):
        if not _is_doc_node(G, nid):
            continue

        rp = data.get("location", {}).get("rel_path", "") or data.get("rel_path", "")
        if not rp:
            continue

        doc_dir = "/".join(rp.replace("\\", "/").split("/")[:-1]) + "/"

        # Strip doc prefix
        code_dir = doc_dir
        for prefix in _DOC_PREFIXES:
            if doc_dir.startswith(prefix):
                code_dir = doc_dir[len(prefix):]
                break

        if code_dir == doc_dir:
            continue  # No prefix was stripped — skip

        # Find code nodes in matching directory (capped at 5 per dir to avoid hairballs)
        for path, target in path_index.items():
            if target != nid and path.startswith(code_dir):
                edges.append((nid, target))
                if len([e for e in edges if e[0] == nid]) >= 5:
                    break  # Proximity cap per doc node per directory

    return edges


def inject_doc_edges(db, G: nx.MultiDiGraph) -> Dict[str, Any]:
    """Inject documentation → code edges.
    
    Tier 1: Hyperlinks [text](path) + backtick `Symbol` mentions → edge_class "doc" / "hyperlink"
    Tier 2: Directory proximity (docs/ → src/) → edge_class "doc" / "proximity"
    """
    path_index = _build_path_index(G)
    name_index = _build_name_index(G)
    seen: Set[Tuple[str, str]] = set()  # Dedup across both tiers

    # Tier 1: Hyperlinks + mentions
    hyperlink_edges = _extract_hyperlink_edges(G, path_index, name_index)
    hyperlink_count = 0
    for src, tgt in hyperlink_edges:
        if (src, tgt) not in seen and not G.has_edge(src, tgt):
            _add_edge(db, G, src, tgt, "hyperlink", "doc", "md_hyperlink")
            seen.add((src, tgt))
            hyperlink_count += 1

    # Tier 2: Proximity
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
```

### 3.5 Component Bridging

```python
def bridge_disconnected_components(db, G: nx.MultiDiGraph) -> Dict[str, Any]:
    """Connect disconnected components via bridge edges.
    
    Uses weakly_connected_components on the DiGraph (not to_undirected).
    Algorithm: linear pass — sort components largest-first, then each
    smaller component bridges to its best-matching larger component
    (by directory histogram similarity). This is O(n²) in component
    count but NOT all-pairs-of-nodes.
    """
    components = list(nx.weakly_connected_components(G))  # DiGraph → weak components

    if len(components) <= 1:
        return {"components_before": len(components), "bridges_added": 0}

    def _dir_hist(comp: set) -> Counter:
        """Directory histogram for a component.
        
        Counts ONLY the immediate parent directory (rsplit on /),
        not all path ancestors.
        """
        hist = Counter()
        for nid in comp:
            data = G.nodes.get(nid, {})
            rp = data.get("rel_path") or data.get("file_name") or ""
            d = rp.rsplit("/", 1)[0] if "/" in rp else "<root>"
            hist[d] += 1
        return hist

    def _dir_sim(a: Counter, b: Counter) -> int:
        """Shared directory entries between two histograms."""
        return sum((a & b).values())

    # Sort components by size (largest first)
    comp_list = sorted([set(c) for c in components], key=len, reverse=True)
    n_before = len(comp_list)
    comp_dirs = [_dir_hist(c) for c in comp_list]
    bridges_added = 0

    # Linear pass: each smaller component (i) bridges to best larger (0..i-1)
    for i in range(1, n_before):
        best_target = 0
        best_sim = _dir_sim(comp_dirs[i], comp_dirs[0])

        for j in range(1, i):
            sim = _dir_sim(comp_dirs[i], comp_dirs[j])
            if sim > best_sim:
                best_sim = sim
                best_target = j

        # Find most central node in each component (highest degree)
        center_i = max(comp_list[i], key=lambda n: G.degree(n))
        center_j = max(comp_list[best_target], key=lambda n: G.degree(n))

        # Add bidirectional bridge
        _add_edge(db, G, center_i, center_j, "component_bridge", "bridge", "component_bridging")
        _add_edge(db, G, center_j, center_i, "component_bridge", "bridge", "component_bridging")
        bridges_added += 2

    return {"components_before": n_before, "bridges_added": bridges_added}
```

### 3.6 Edge Weighting

```python
def apply_edge_weights(G: nx.MultiDiGraph) -> Dict[str, Any]:
    """Apply inverse-in-degree edge weighting to ALL edges.
    
    Formula: weight = 1 / log(structural_in_degree + 2)
    
    Structural in-degree = in-degree counting ONLY non-synthetic edges
    (excludes lexical, semantic, directory, doc, bridge edge classes).
    
    Synthetic edges get a weight floor of SYNTHETIC_WEIGHT_FLOOR (0.5).
    """
    # Compute structural in-degree (excluding synthetic edges)
    structural_in: Dict[str, int] = {n: 0 for n in G.nodes()}
    for u, v, data in G.edges(data=True):
        ec = data.get("edge_class", "structural")
        if ec not in _SYNTHETIC_CLASSES:
            structural_in[v] = structural_in.get(v, 0) + 1

    # Apply weights
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
```

### 3.7 Hub Detection

```python
def detect_hubs(G: nx.MultiDiGraph, z_threshold: float = 3.0) -> Set[str]:
    """Detect hub nodes via Z-score on in-degree.
    
    Hubs = nodes where (in_degree - mean) / std > z_threshold.
    """
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
        std = variance ** 0.5

    if std == 0:
        return set()

    hubs = set()
    for nid, deg in zip(G.nodes(), degrees):
        z = (deg - mean) / std
        if z > z_threshold:
            hubs.add(nid)

    return hubs


def flag_hubs_in_db(db, hubs: Set[str]) -> None:
    """Persist hub flags to the unified DB."""
    for nid in hubs:
        db.set_hub(nid, is_hub=True)
    db.conn.commit()
```

### 3.8 Edge Persistence

```python
def persist_weights_to_db(db, G: nx.MultiDiGraph) -> int:
    """Write all weighted edges back to the DB (full replace).
    
    Strategy: DELETE all edges, then re-insert with computed weights.
    This ensures DB edges match the in-memory graph exactly.
    """
    db.conn.execute("DELETE FROM repo_edges")

    batch = []
    for u, v, key, data in G.edges(data=True, keys=True):
        edge_dict = {
            "source_id": u,
            "target_id": v,
            "relationship_type": data.get("relationship_type", ""),  # Not "rel_type"
            "edge_class": data.get("edge_class", "structural"),
            "analysis_level": data.get("analysis_level", "comprehensive"),
            "weight": data.get("weight", 1.0),
            "raw_similarity": data.get("raw_similarity"),
            "source_file": data.get("source_file", ""),
            "target_file": data.get("target_file", ""),
            "language": data.get("language", ""),
            "annotations": (
                json.dumps(data["annotations"]) if isinstance(data.get("annotations"), dict)
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
```

### 3.9 Main Orchestrator

```python
def run_phase2(
    db, G: nx.MultiDiGraph,
    embedding_fn: Optional[Callable] = None,
    z_threshold: float = 3.0,
    vec_distance_threshold: float = 0.15,
) -> Dict[str, Any]:
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
    results = {}

    # Step 1: Orphan resolution
    orphan_stats = resolve_orphans(
        db, G, embedding_fn,
        vec_distance_threshold=vec_distance_threshold,
    )
    results["orphan_resolution"] = orphan_stats
    logger.info(f"Orphan resolution: {orphan_stats}")

    # Step 2: Document edges
    doc_stats = inject_doc_edges(db, G)
    results["doc_edges"] = doc_stats
    logger.info(f"Doc edges: {doc_stats}")

    # Step 3: Component bridging
    bridge_stats = bridge_disconnected_components(db, G)
    results["bridging"] = bridge_stats
    logger.info(f"Bridging: {bridge_stats}")

    # Step 4: Edge weighting
    weight_stats = apply_edge_weights(G)
    results["weighting"] = weight_stats
    logger.info(f"Weighting: {weight_stats}")

    # Step 5: Hub detection
    hubs = detect_hubs(G, z_threshold)
    flag_hubs_in_db(db, hubs)
    results["hubs"] = {"count": len(hubs), "nodes": list(hubs)[:20]}
    logger.info(f"Hubs detected: {len(hubs)}")

    # Step 6: Persist weighted edges
    edge_count = persist_weights_to_db(db, G)
    results["persisted_edges"] = edge_count
    logger.info(f"Persisted {edge_count} edges")

    # Step 7: Metadata
    db.set_meta("phase2_completed", True)
    db.set_meta("phase2_stats", results)

    return results
```

---

## 4. Pipeline Integration Point

In `wiki_graph_optimized.py`, after graph building and DB persistence:

```python
from app.core.graph_topology import run_phase2

# After db.from_networkx(G) and db.populate_embeddings(embedding_fn):
if config.graph_enrichment_enabled:
    phase2_stats = run_phase2(
        db=state["unified_db"],
        G=G,
        embedding_fn=embedding_fn,
        z_threshold=config.hub_z_threshold,
        vec_distance_threshold=config.vec_distance_threshold,
    )
    logger.info(f"Phase 2 enrichment complete: {phase2_stats}")

    # Reload graph from DB to get weighted edges
    G = state["unified_db"].to_networkx()
```

---

## 5. Configuration

```python
# In config.py
graph_enrichment_enabled: bool = True    # WIKIS_GRAPH_ENRICHMENT_ENABLED
hub_z_threshold: float = 3.0            # WIKIS_HUB_Z_THRESHOLD
vec_distance_threshold: float = 0.15    # WIKIS_VEC_DISTANCE_THRESHOLD
```
