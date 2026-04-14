# Implementation Plan: Graph Topology Enrichment

**Design**: `docs/proposals/graph-topology-enrichment/design.md`
**Created**: 2026-04-10
**Updated**: 2026-04-13

---

## Implementation Status

| Step | Description | Status | Notes |
|------|-------------|--------|-------|
| 1 | Create `graph_topology.py` skeleton | ✅ DONE | Module with all constants, imports, `run_phase2()` orchestrator |
| 2 | Orphan detection | ✅ DONE | `find_orphans()` — 4 tests |
| 3 | FTS5 lexical orphan resolution | ✅ DONE | 3-pass cascade in `resolve_orphans()` — 4 tests |
| 4 | Semantic vector orphan resolution | ✅ DONE | Expanding path-prefix search — 5 tests |
| 5 | Directory fallback orphan resolution | ✅ DONE | Walk-up parent dir strategy — 3 tests |
| 6 | Doc edge injection — hyperlinks + backticks | ✅ DONE | `inject_doc_edges()` — 7 tests |
| 7 | Doc edge injection — proximity | ✅ DONE | `docs/`/`documentation/` prefix matching — 4 tests |
| 8 | Component bridging | ✅ DONE | `bridge_disconnected_components()` — 4 tests |
| 9 | Edge weighting | ✅ DONE | Inverse in-degree `1/log(d+2)` + synthetic floor — 3 tests |
| 10 | Hub detection and flagging | ✅ DONE | Z-score based `detect_hubs()` — 5 tests |
| 11 | Edge persistence | ✅ DONE | `persist_weights_to_db()` batch writes — 3 tests |
| 12 | `run_phase2()` integration | ✅ DONE | Orchestrator calls all steps, stores metadata — 3 tests |
| 13 | Pipeline wiring | ✅ DONE | Wired via `_ensure_unified_db()` → `run_phase2()` in agent |

**Config settings added** (`config.py`): `graph_enrichment_enabled: bool = True`, `hub_z_threshold: float = 3.0`, `vec_distance_threshold: float = 0.15`

**Total**: 59 tests in `test_graph_topology.py`, all passing.

---

## Step 1: Create `graph_topology.py` Skeleton

**File**: `backend/app/core/graph_topology.py`
**FR Coverage**: FR-001, FR-012

**Steps**:

1. Create file with imports, constants, and `run_phase2()` stub that calls each function in order
2. Each sub-function is initially a stub returning empty stats
3. Verify: import succeeds, `run_phase2()` runs on an empty graph with no errors

**Test File**: `tests/unit/test_graph_topology.py::TestModuleLoadable`

---

## Step 2: Orphan Detection

**File**: `backend/app/core/graph_topology.py`
**FR Coverage**: FR-002

**TDD Steps**:

1. Write test: `find_orphans()` on a graph with 3 connected + 2 orphan nodes → returns the 2 orphan IDs
2. Write test: all nodes connected → returns empty list
3. Write test: node with outgoing edge only (out-degree>0, in-degree=0) → NOT an orphan (has at least 1 edge direction)
4. Implement `find_orphans()`
5. Verify all tests pass.

**Test File**: `tests/unit/test_graph_topology.py::TestOrphanDetection`

---

## Step 3: FTS5 Lexical Orphan Resolution

**File**: `backend/app/core/graph_topology.py`
**FR Coverage**: FR-002, FR-003

**TDD Steps**:

1. Create a `UnifiedWikiDB` in memory with FTS5 index containing known text
2. Write test: orphan with `symbol_name="UserValidator"` and FTS5 containing "UserValidator in docs" → `lexical_link` edge added
3. Write test: FTS5 returns self-hit → excluded, no self-edge
4. Write test: `max_lexical_edges=2` → at most 2 edges added per orphan
5. Write test: orphan resolved by FTS5 → skipped in subsequent passes
6. Implement FTS5 pass in `resolve_orphans()`
7. Verify all tests pass.

**Test File**: `tests/unit/test_graph_topology.py::TestFTS5Resolution`

---

## Step 4: Semantic Vector Orphan Resolution

**File**: `backend/app/core/graph_topology.py`
**FR Coverage**: FR-002, FR-004

**TDD Steps**:

1. Mock `embedding_fn` that returns deterministic vectors
2. Populate `UnifiedWikiDB` sqlite-vec with known embeddings
3. Write test: orphan text semantically similar to node B → `semantic_link` edge added
4. Write test: `vec_distance_threshold=0.15` → hit at distance 0.20 is skipped
5. Write test: expanding path prefix → searches `src/auth/` first, then `src/`, then global
6. Write test: `embedding_fn=None` → semantic pass is skipped, no error
7. Implement semantic pass in `resolve_orphans()`
8. Verify all tests pass.

**Test File**: `tests/unit/test_graph_topology.py::TestSemanticResolution`

---

## Step 5: Directory Fallback Orphan Resolution

**File**: `backend/app/core/graph_topology.py`
**FR Coverage**: FR-002, FR-005

**TDD Steps**:

1. Write test: orphan in `src/utils/helpers.py`, high-degree node in `src/utils/core.py` → `directory_link` edge
2. Write test: no non-orphan in same dir → walk up to parent `src/` → connects
3. Write test: no non-orphan at any level → orphan stays unresolved (no garbage edge)
4. Implement `_resolve_orphans_by_directory()`
5. Verify all tests pass.

**Test File**: `tests/unit/test_graph_topology.py::TestDirectoryResolution`

---

## Step 6: Document Edge Injection — Hyperlinks + Backtick Mentions

**File**: `backend/app/core/graph_topology.py`
**FR Coverage**: FR-006

**TDD Steps**:

1. Create doc node with `[Service](../src/auth/service.py)` in content
2. Create code node at `src/auth/service.py`
3. Write test: `inject_doc_edges()` adds `hyperlink` edge
4. Write test: doc with `` `TokenValidator` ``, code node with name "TokenValidator" → `hyperlink` edge
5. Write test: backtick mention with 5 matching symbols → only 2 edges added (cap=2 per mention)
6. Write test: external URL `[text](https://example.com)` → no edge added
7. Write test: `mailto:` link → no edge added
8. Write test: `#anchor`-only link → no edge added
9. Write test: duplicate (src, tgt) pair → only 1 edge added (seen set dedup)
10. Implement `inject_doc_edges()` hyperlink + backtick tiers with seen set
11. Verify all tests pass.

**Test File**: `tests/unit/test_graph_topology.py::TestDocHyperlinkEdges`

---

## Step 7: Document Edge Injection — Proximity

**File**: `backend/app/core/graph_topology.py`
**FR Coverage**: FR-007

**TDD Steps**:

1. Create doc node at `docs/auth/README.md`, code node at `auth/service.py`
2. Write test: `inject_doc_edges()` adds `proximity` edge (strips `docs/` prefix)
3. Write test: doc at `src/README.md` (no `docs/` prefix) → no proximity edge
4. Write test: `documentation/auth/` prefix stripped correctly
5. Write test: proximity edges capped at 5 per matching directory
6. Implement proximity tier in `inject_doc_edges()`
7. Verify all tests pass.

**Test File**: `tests/unit/test_graph_topology.py::TestDocProximityEdges`

---

## Step 8: Component Bridging

**File**: `backend/app/core/graph_topology.py`
**FR Coverage**: FR-008

**TDD Steps**:

1. Build graph with 3 disconnected components (no edges between them)
2. Write test: after `bridge_disconnected_components()`, graph is fully connected (uses `weakly_connected_components`, not `to_undirected`)
3. Write test: bridge edges are bidirectional and have class `bridge` / `component_bridge`
4. Write test: already-connected graph → 0 bridges added
5. Write test: linear pass — smaller component bridges to best-matching larger component (by immediate-parent dir histogram similarity)
6. Write test: root-level files use `"<root>"` as directory key
7. Implement `bridge_disconnected_components()`
8. Verify all tests pass.

**Test File**: `tests/unit/test_graph_topology.py::TestComponentBridging`

---

## Step 9: Edge Weighting

**File**: `backend/app/core/graph_topology.py`
**FR Coverage**: FR-009

**TDD Steps**:

1. Build graph: node B has structural in-degree 10
2. Write test: after `apply_edge_weights()`, edge A→B has `weight ≈ 1/log(12) ≈ 0.402`
3. Write test: synthetic edge with computed weight < 0.5 → clamped to 0.5
4. Write test: synthetic edges NOT counted in structural in-degree
5. Implement `apply_edge_weights()`
6. Verify all tests pass.

**Test File**: `tests/unit/test_graph_topology.py::TestEdgeWeighting`

---

## Step 10: Hub Detection and Flagging

**File**: `backend/app/core/graph_topology.py`
**FR Coverage**: FR-010

**TDD Steps**:

1. Build graph with 20 nodes: 19 have in-degree 1, 1 has in-degree 50
2. Write test: `detect_hubs(G, z_threshold=3.0)` returns the hub node
3. Write test: graph with < 3 nodes → empty set
4. Write test: all nodes same degree (std=0) → empty set
5. Write test: `flag_hubs_in_db()` calls `db.set_hub()` for each hub
6. Implement `detect_hubs()` and `flag_hubs_in_db()`
7. Verify all tests pass.

**Test File**: `tests/unit/test_graph_topology.py::TestHubDetection`

---

## Step 11: Edge Persistence

**File**: `backend/app/core/graph_topology.py`
**FR Coverage**: FR-011

**TDD Steps**:

1. Build graph with 100 edges, create DB
2. Write test: `persist_weights_to_db()` → `db.edge_count()` == 100
3. Write test: weights in DB match in-memory weights
4. Write test: batch size respected (verify via mock)
5. Implement `persist_weights_to_db()`
6. Verify all tests pass.

**Test File**: `tests/unit/test_graph_topology.py::TestEdgePersistence`

---

## Step 12: `run_phase2()` Integration

**File**: `backend/app/core/graph_topology.py`
**FR Coverage**: FR-012

**TDD Steps**:

1. Build a realistic graph (20 connected + 10 orphans + 3 doc nodes + 2 components)
2. Write test: `run_phase2()` returns stats with all sections populated
3. Write test: after `run_phase2()`, orphan count < 3 (from 10)
4. Write test: after `run_phase2()`, graph is fully connected (1 component)
5. Write test: `db.set_meta("phase2_completed", True)` was called
6. Wire `run_phase2()` as the assembled orchestrator
7. Verify all tests pass.

**Test File**: `tests/unit/test_graph_topology.py::TestPhase2Orchestrator`

---

## Step 13: Pipeline Wiring

**File**: `backend/app/core/agents/wiki_graph_optimized.py`
**FR Coverage**: FR-001

**Steps**:

1. Add config settings for `graph_enrichment_enabled`, `hub_z_threshold`, `vec_distance_threshold`
2. After `db.from_networkx(G)` + `db.populate_embeddings(embedding_fn)`:
   ```python
   if config.graph_enrichment_enabled:
       phase2_stats = run_phase2(db, G, embedding_fn, ...)
       G = db.to_networkx()  # Reload with weighted edges
   ```
3. Integration test: generate wiki for small repo → Phase 2 runs → graph has 1 component

**Test File**: `tests/integration/test_graph_topology_pipeline.py`

---

## Dependency Graph

```
Step 1 (skeleton)
  ├─► Step 2 (orphan detection)
  │     ├─► Step 3 (FTS5 resolution)
  │     ├─► Step 4 (semantic resolution)
  │     └─► Step 5 (directory resolution)
  ├─► Step 6 (hyperlink edges)
  ├─► Step 7 (proximity edges)
  ├─► Step 8 (bridging)
  ├─► Step 9 (edge weighting)
  ├─► Step 10 (hub detection)
  └─► Step 11 (edge persistence)
         └─► Step 12 (run_phase2) ─► Step 13 (pipeline wiring)
```

**Steps 2–11 are parallelizable** in development (each is a self-contained function with independent tests). Steps 12–13 integrate them.

---

## Estimated Changes

| File | Lines Changed | Risk |
|------|--------------|------|
| `graph_topology.py` (new) | ~500 | Medium — complex but self-contained |
| `wiki_graph_optimized.py` | +10 | Low — single integration call |
| `config.py` | +3 | Low |
| **Tests** | ~600 lines new | — |
