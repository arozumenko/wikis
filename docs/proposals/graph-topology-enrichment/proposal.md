# Proposal: Graph Topology Enrichment (Phase 2)

**Created**: 2026-04-10
**Status**: Draft
**Depends on**: `unified-db-activation` (requires unified DB for FTS5/vector search and edge persistence)
**Depended on by**: `cluster-planner-migration` (enriched edges improve clustering quality)

---

## Problem

After tree-sitter parsing, the code graph in Wikis contains only **structural (AST-derived) edges** — explicit `import`, `call`, `inherits`, and `contains` relationships. This produces a graph with:

- **55% degree-0 orphan nodes** — parameters, constants, standalone functions, and utility classes that have no explicit AST relationships to other nodes
- **64% same-file edges** — most structural edges connect symbols within the same file, providing little cross-file signal for clustering
- **Disconnected components** — independent modules with no cross-component edges fragment community detection into dozens of singleton clusters
- **No document↔code linkage** — Markdown docs, READMEs, and changelogs float as isolated nodes with zero edges to the code they describe

DeepWiki solves this with a **Phase 2 Graph Topology Enrichment** pipeline (`graph_topology.py`, ~1050 lines) that injects 6 types of synthetic edges, applies inverse-in-degree edge weighting, detects hubs, and bridges disconnected components. The result is a dense, fully-connected graph where:
- Orphan rate drops from 55% to <5%
- Cross-file edge density increases 3–5×
- Every connected component is reachable from every other
- Document nodes are linked to the code they reference

**Wikis has ZERO post-parse graph enrichment.** This is the single largest feature gap between the two systems and directly impacts clustering quality, retrieval accuracy, and wiki coherence.

## Solution

Port DeepWiki's `graph_topology.py` as `backend/app/core/graph_topology.py` in Wikis. The module is **self-contained** — it accepts a `UnifiedWikiDB` instance and a `NetworkX MultiDiGraph` as inputs and has no imports from other project modules.

### Phase 2 Pipeline (7 steps, in order)

1. **Orphan Resolution: Lexical (FTS5)** — For each degree-0 node, search FTS5 by `symbol_name`. Link to top matches. Edge class: `lexical` / `lexical_link`.

2. **Orphan Resolution: Semantic (sqlite-vec)** — For still-orphaned nodes, embed their source text and search sqlite-vec with expanding path prefixes (same dir → parent dir → global). Edge class: `semantic` / `semantic_link`.

3. **Orphan Resolution: Directory Fallback** — For still-orphaned nodes, walk up the directory tree and find the highest-degree non-orphan node at each level. Edge class: `directory` / `directory_link`.

4. **Document Edge Injection: Hyperlinks** — Parse Markdown `[text](path)` links and backtick `` `SymbolName` `` mentions from doc nodes (capped at 2 targets per backtick mention). Guards for `mailto:`, `#`-only anchors, and external URLs. Edge dedup via `seen` set. Edge class: `doc` / `hyperlink`.

5. **Document Edge Injection: Proximity** — Match doc directories to code directories (stripping `docs/`, `doc/`, `documentation/` prefixes). Capped at 5 code nodes per matching directory. Edge class: `doc` / `proximity`.

6. **Component Bridging** — Find disconnected components via `weakly_connected_components()`. Sort largest-first, then linear pass: each smaller component bridges to its best-matching larger component by immediate-parent directory histogram similarity. Bidirectional edge between highest-degree nodes. Edge class: `bridge` / `component_bridge`.

7. **Edge Weighting** — Apply inverse-in-degree weighting to ALL edges: `weight = 1 / log(structural_in_degree + 2)`. Synthetic edges get a weight floor of `SYNTHETIC_WEIGHT_FLOOR = 0.5`.

8. **Hub Detection** — Identify high-degree nodes via Z-score on in-degree (`z_threshold=3.0`). Flag in DB via `db.set_hub()`.

9. **Persistence** — Write all weighted edges back to DB via `persist_weights_to_db()`.

### Why This Must Be Separate from Cluster Planner

The topology enrichment pipeline is **independent of the clustering algorithm**. It improves the raw graph before any clustering runs. Even without the cluster planner, enrichment improves:
- Retrieval quality (more edges = better neighborhood expansion)
- Content expansion context (connected nodes provide richer code context)
- Wiki coherence (doc↔code links ensure documentation is relevant)

## Source Files (DeepWiki)

| File | Lines | Purpose |
|------|-------|---------|
| `plugin_implementation/graph_topology.py` | ~1050 | Complete Phase 2 pipeline |

The file is self-contained. It imports only `networkx`, `re`, `math`, `logging`, `json`, `collections.Counter`, and optionally `numpy`.

## New Dependencies

None beyond what `unified-db-activation` provides. The enrichment module uses:
- `db.search_fts5()` — already in `unified_db.py`
- `db.search_vec()` — already in `unified_db.py`
- `db.upsert_edge()` / `db.upsert_edges_batch()` — already in `unified_db.py`
- `db.set_hub()` — already in `unified_db.py`
- `db.set_meta()` — already in `unified_db.py`
- `numpy` — soft dependency for hub detection (already in requirements)

## Risk Assessment

| Risk | Impact | Mitigation |
|------|--------|------------|
| Enrichment adds too many edges | Slower clustering, noisy signal | Tune `fts_limit`, `vec_k`, `vec_distance_threshold` parameters |
| FTS5/vec queries slow for large repos | Phase 2 takes >30s | Batch queries, limit per-orphan search |
| Component bridging creates garbage edges | Poor clustering boundaries | Bridge only most similar components (directory histogram) |
| Hub detection too aggressive | Too many hubs excluded from clustering | Z-score threshold (3.0) is conservative; configurable |
