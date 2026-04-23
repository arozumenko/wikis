# Implementation Plan: Cluster-Based Structure Planner

**Design**: `docs/proposals/cluster-planner-migration/design.md`
**Created**: 2026-04-10
**Updated**: 2026-04-14

---

## Implementation Status

| Phase | Step | Status | Notes |
|-------|------|--------|-------|
| **Phase 1** | 1.1 Cluster Constants | ✅ DONE | `cluster_constants.py` + 89 tests |
| | 1.2 Architectural Projection | ✅ DONE | `graph_clustering.py::architectural_projection()` |
| | 1.3 Hub Detection | ✅ DONE | `graph_clustering.py::detect_hubs()` |
| | 1.4 File Contraction | ✅ DONE | `graph_clustering.py::_contract_to_file_graph()` |
| | 1.5 Macro-Clustering (Leiden) | ✅ DONE | `graph_clustering.py::_hierarchical_leiden()` |
| | 1.6 Micro-Clustering + Page Consolidation | ✅ DONE | `_consolidate_sections()`, `_consolidate_pages()` |
| | 1.7 Hub Re-integration | ✅ DONE | `graph_clustering.py::reintegrate_hubs()` |
| | 1.8 End-to-End `run_clustering()` | ✅ DONE | `run_clustering()` + 54 tests |
| **Phase 2** | 2.1 Cluster Prompt Templates | ✅ DONE | `cluster_prompts.py` — section + page naming templates |
| | 2.2 ClusterStructurePlanner Core | ✅ DONE | `cluster_planner.py` + 75 tests |
| | 2.3 API Extension | ✅ DONE | `planner_type` field in GenerateWikiRequest |
| | 2.4 Configuration Extension | ✅ DONE | `planner_type`, `cluster_hub_z_threshold`, `cluster_exclude_tests` |
| | 2.5 Planner Routing | ✅ DONE | `generate_wiki_structure()` routes `planner_type="cluster"` to `_generate_wiki_structure_cluster()` |
| | 2.6 Wiki Service Extension | ✅ DONE | Already wired in wiki_service.py (previous session) |
| | 2.7 SSE Progress Events | ✅ DONE | progress_callback calls in `_generate_wiki_structure_cluster()` |
| **Phase 3** | 3.1 Sync Expansion Engine | ⏭ SKIPPED | 1-line diff between DeepWiki and Wikis — already synced |
| | 3.2 Documentation | ✅ DONE | `docs/cluster-planner.md` + `.env.example` updated |
| **Expansion Port** | language_heuristics.py | ✅ DONE | 30 tests — 8 languages, budget fractions |
| | shared_expansion.py | ✅ DONE | 56 tests — per-type DB expansion |
| | cluster_expansion.py | ✅ DONE | 57 tests — expand_for_page + augmentation |
| | Agent wiring | ✅ DONE | `_try_cluster_expansion()` in wiki_graph_optimized.py |

### Method Naming (DeepWiki → Wikis)

| DeepWiki Name | Wikis Name | Notes |
|---------------|-----------|-------|
| `run_phase2()` | `_consolidate_sections()` + `_consolidate_pages()` | Split into separate functions |
| `run_phase3()` | `run_clustering()` | Clean top-level entry point |
| `hierarchical_leiden_cluster()` | `_hierarchical_leiden()` | Internal, prefixed with `_` |
| `_name_macro_cluster()` (DB-based) | `_name_macro_cluster()` (in-memory graph) | Same name, adapted to use nx.MultiDiGraph |
| `_build_cluster_map()` (DB queries) | `_build_architectural_cluster_map()` (graph nodes) | Iterates ClusterAssignment.sections |
| `build_structure()` | `plan_structure()` | Matches StructureEngine interface |

---

## Phase 1 — Graph Clustering Engine (Foundation) ✅ COMPLETE

### Step 1.1: Port Cluster Constants ✅

**File**: `backend/app/core/cluster_constants.py`
**FR Coverage**: FR-004, FR-005, FR-006, FR-007, FR-008

**TDD Steps**:

1. Write tests for `target_section_count()` formula (file count → section count):
   - 5 files → 5, 10 files → 5, 100 files → 9, 1000 files → 12, 10000 files → 16
2. Write tests for `adaptive_max_page_size()`:
   - 100 → 25, 500 → 35, 2000 → 45, 5000 → 60
3. Write tests for `target_total_pages()` formula:
   - 50 nodes → 8, 500 nodes → 9, 5000 nodes → 27, 50000 nodes → 85
4. Write tests for `is_test_path()`:
   - `"tests/test_foo.py"` → True
   - `"src/main.py"` → False
   - `"__tests__/Component.test.jsx"` → True
5. Implement all constants and formulas. Verify all tests pass.

**Test File**: `tests/unit/test_cluster_constants.py`

---

### Step 1.2: Architectural Projection

**File**: `backend/app/core/graph_clustering.py` — `architectural_projection()`
**FR Coverage**: FR-004

**TDD Steps**:

1. Build a synthetic NetworkX graph with:
   - 3 classes (architectural)
   - 5 methods (child of class, linked by 'contains' edge)
   - 2 standalone functions (architectural)
   - 2 variables (non-architectural, no parent)
   - Edges: method→class (contains), method→method (calls), function→class (calls)
2. Write test: projected graph has 5 nodes (3 classes + 2 functions), 0 methods/variables
3. Write test: method edges remapped to parent class (method calls → class calls)
4. Write test: self-loops from sibling methods in same class are dropped
5. Write test: standalone functions retained as-is
6. Implement `architectural_projection()`. Verify all tests pass.

**Test File**: `tests/unit/test_graph_clustering.py::TestArchitecturalProjection`

---

### Step 1.3: Hub Detection

**File**: `backend/app/core/graph_clustering.py` — `detect_hubs()`
**FR Coverage**: FR-008

**TDD Steps**:

1. Build a star graph: 1 hub connected to 20 periphery nodes
2. Write test: hub node has highest PageRank, is detected as hub at 90th percentile
3. Write test: with < 10 nodes, returns empty set
4. Write test: MAX_HUB_FRACTION cap (graph with many high-degree nodes, verify cap)
5. Implement `detect_hubs()`. Verify all tests pass.

**Test File**: `tests/unit/test_graph_clustering.py::TestHubDetection`

---

### Step 1.4: File Contraction

**File**: `backend/app/core/graph_clustering.py` — `_contract_to_file_graph()`
**FR Coverage**: FR-005

**TDD Steps**:

1. Build graph with 10 symbols across 3 files, with known cross-file edges
2. Write test: contracted graph has 3 file-nodes
3. Write test: cross-file edge weights are summed
4. Write test: same-file edges are dropped (no self-loops on file nodes)
5. Write test: file_to_nodes mapping is correct
6. Implement `_contract_to_file_graph()`. Verify all tests pass.

**Test File**: `tests/unit/test_graph_clustering.py::TestFileContraction`

---

### Step 1.5: Macro-Clustering (Leiden)

**File**: `backend/app/core/graph_clustering.py` — `_nx_to_igraph()`, `_leiden_macro()`
**FR Coverage**: FR-005, FR-006

**TDD Steps**:

1. Build a graph with 3 clearly separated clusters (fully connected within, sparse between)
2. Write test: Leiden produces 3 communities (using leidenalg.RBConfigurationVertexPartition, γ=1.0)
3. Write test: `_nx_to_igraph()` correctly converts NetworkX graph to igraph, returning `(ig_graph, node_list)` tuple with weights and deterministic node ordering
4. Write test: section cap enforcement via `_consolidate_sections()` merges smallest into nearest by **directory proximity** (`_dir_similarity()`)
5. Write test: re-indexing produces contiguous 0..N-1 section IDs
6. Write test: determinism — same result on multiple runs (seed=42)
7. Implement `_nx_to_igraph()` and `_leiden_macro()`. Verify all tests pass.

**Test File**: `tests/unit/test_graph_clustering.py::TestMacroClustering`

---

### Step 1.6: Micro-Clustering + Page Consolidation

**File**: `backend/app/core/graph_clustering.py` — `_leiden_micro_all()`, `_consolidate_pages()`
**FR Coverage**: FR-007

**TDD Steps**:

1. Build a section subgraph with 30 nodes in 3 natural sub-communities
2. Write test: Leiden micro-clustering produces 3 pages (using leidenalg.RBConfigurationVertexPartition, γ=1.0)
3. Write test: merge — create a 2-node page, verify `_consolidate_pages()` merges it into nearest same-section neighbour by directory proximity
4. Write test: global page target — verify total pages ≤ `target_total_pages(n_nodes)` after consolidation
5. Write test: page IDs re-indexed contiguously per section
5. Implement `_leiden_micro_all()` and `_consolidate_pages()`. Verify all tests pass.

**Test File**: `tests/unit/test_graph_clustering.py::TestMicroClustering`

---

### Step 1.7: Hub Re-integration

**File**: `backend/app/core/graph_clustering.py` — `reintegrate_hubs()`
**FR Coverage**: FR-008

**TDD Steps**:

1. Build a graph with 3 clusters and 1 hub connected to 2 nodes in cluster A, 5 nodes in cluster B
2. Write test: hub assigned to cluster B (plurality voting)
3. Write test: within cluster B, hub assigned to the page with most edges
4. Write test: hub with no neighbors → assigned to section 0, page 0
5. Implement `reintegrate_hubs()`. Verify all tests pass.

**Test File**: `tests/unit/test_graph_clustering.py::TestHubReintegration`

---

### Step 1.8: End-to-End `run_clustering()`

**File**: `backend/app/core/graph_clustering.py` — `run_clustering()`
**FR Coverage**: FR-005 through FR-008, FR-013

**TDD Steps**:

1. Build a realistic synthetic graph (~100 nodes, 5 files, mixed symbol types)
2. Write test: `run_clustering()` returns valid `ClusterAssignment` with sections ≥ 3
3. Write test: empty graph → returns trivial assignment
4. Write test: < 3 architectural nodes → returns trivial assignment
5. Write test: determinism — same `ClusterAssignment` on two `run_clustering()` calls with same input
6. Implement `run_clustering()` (composition of all previous pieces). Verify all tests pass.

**Test File**: `tests/unit/test_graph_clustering.py::TestRunClustering`

---

## Phase 2 — Cluster Structure Planner + Integration

### Step 2.1: Cluster Prompt Templates

**File**: `backend/app/core/wiki_structure_planner/cluster_prompts.py`
**FR Coverage**: FR-009, FR-010

**Implementation Steps**:

1. Port `SECTION_NAMING_PROMPT` from DeepWiki `structure_prompts.py` — adapt for JSON output format
2. Port `PAGE_NAMING_PROMPT` — ensure page-only symbol input (anti centroid-bleed)
3. Add `{symbols_json}`, `{page_count}`, `{repo_name}`, `{section_name}`, `{directories}` template vars

**No TDD** — prompts are templates, tested indirectly via planner integration tests.

---

### Step 2.2: ClusterStructurePlanner Core

**File**: `backend/app/core/wiki_structure_planner/cluster_planner.py`
**FR Coverage**: FR-009, FR-010, FR-003

**TDD Steps**:

1. Mock LLM to return predefined JSON names
2. Write test: `plan_structure()` returns `WikiStructureSpec` with correct section/page counts matching `ClusterAssignment`
3. Write test: dominant symbol ranking — class ranked higher than function (SYMBOL_TYPE_PRIORITY)
4. Write test: page naming uses only page-local symbols (inject mock, verify prompt content)
5. Write test: LLM failure → fallback naming using dominant symbol name
6. Write test: `WikiStructureSpec.total_pages` matches sum of all pages  
7. Implement `ClusterStructurePlanner`. Verify all tests pass.

**Test File**: `tests/unit/test_cluster_planner.py`

---

### Step 2.3: API Extension

**File**: `backend/app/models/api.py`
**FR Coverage**: FR-001, FR-002

**TDD Steps**:

1. Write test: `GenerateWikiRequest(planner_type="cluster")` validates
2. Write test: `GenerateWikiRequest(planner_type="invalid")` raises ValidationError
3. Write test: `GenerateWikiRequest()` defaults to `planner_type="deepagents"`
4. Add `planner_type` field. Verify all tests pass.

**Test File**: `tests/unit/test_api_models.py`

---

### Step 2.4: Configuration Extension

**File**: `backend/app/config.py`
**FR Coverage**: FR-001, FR-005, FR-008

**Implementation Steps**:

2. Add `cluster_planner_enabled: bool = True` to Settings
2. Add `cluster_hub_percentile: int = 90`
3. Add env var mapping: `WIKIS_CLUSTER_PLANNER_ENABLED`, `WIKIS_CLUSTER_HUB_PERCENTILE`

---

### Step 2.5: Planner Routing

**File**: `backend/app/core/agents/wiki_graph_optimized.py`
**FR Coverage**: FR-001, FR-002, FR-003, FR-012

**TDD Steps**:

1. Write integration test: `planner_type="cluster"` → `ClusterStructurePlanner` is invoked
2. Write integration test: `planner_type="deepagents"` → existing `WikiStructurePlannerEngine` is invoked
3. Write integration test: `planner_type="cluster"` with empty graph → fallback to DeepAgents
4. Modify `generate_wiki_structure()` to add routing logic. Verify all tests pass.

**Test File**: `tests/integration/test_planner_routing.py`

---

### Step 2.6: Wiki Service Extension

**File**: `backend/app/services/wiki_service.py`
**FR Coverage**: FR-002

**Implementation Steps**:

1. Pass `planner_type` from `GenerateWikiRequest` to the agent config/state
2. Ensure `WikiState` carries `planner_type` through the LangGraph pipeline

---

### Step 2.7: SSE Progress Events

**FR Coverage**: FR-011

**Implementation Steps**:

1. In `ClusterStructurePlanner`, emit progress via callback:
   - "Running graph clustering..." → progress 30
   - "Detected N sections, M pages" → progress 32
   - "Naming section K/N..." → progress 34 + 2*K
   - "Structure complete" → progress 44
2. In `generate_wiki_structure()`, wire the progress callback to `invocation.emit()`
3. No new event types needed — reuse existing `progress` events

---

## Phase 3 — Expansion Sync + Polish

### Step 3.1: Sync Expansion Engine

**File**: `backend/app/core/code_graph/expansion_engine.py`
**FR Coverage**: (deferred — separate proposal if divergence is significant)

**Steps**:

1. Diff DeepWiki `expansion_engine.py` vs. Wikis `expansion_engine.py`
2. If diverged: port P0/P1/P2 priority system, language-specific augmenters
3. If already synced: verify tests pass and skip

---

### Step 3.2: Documentation

1. Update `README.md` with cluster planner usage
2. Add `--planner-type cluster` to CLI help (if applicable)
3. Document env vars: `WIKIS_CLUSTER_*`

---

## Dependency Graph

```
Step 1.1 (constants)
   ├─► Step 1.2 (projection)
   │      ├─► Step 1.5 (macro-clustering)
   │      │      ├─► Step 1.6 (micro/page sizing)
   │      │      │      └─► Step 1.8 (run_clustering)
   │      │      └─► Step 1.7 (hub reintegration)
   │      │             └─► Step 1.8
   │      └─► Step 1.3 (hub detection)
   │             └─► Step 1.7
   └─► Step 1.4 (file contraction)
          └─► Step 1.5

Step 1.8 ──► Step 2.1 (prompts) ──► Step 2.2 (planner core) ──► Step 2.5 (routing)
Step 2.3 (API) ──► Step 2.5
Step 2.4 (config) ──► Step 2.5
Step 2.5 ──► Step 2.6 (wiki service) ──► Step 2.7 (SSE)
Step 2.7 ──► Step 3.1 (expansion sync) ──► Step 3.2 (docs)
```

---

## Parallelization Opportunities

| Parallel Group | Steps | Notes |
|----------------|-------|-------|
| **P1: Constants + Fixtures** | 1.1 | Must complete first — used everywhere |
| **P2: Core Algorithms** | 1.2, 1.3, 1.4 | Independent; share only 1.1 |
| **P3: Clustering** | 1.5, 1.6, 1.7 | Depend on P2 but can start independently once base functions exist |
| **P4: Integration** | 2.1, 2.3, 2.4 | Independent of each other |
| **P5: Assembly** | 2.2, 2.5, 2.6, 2.7 | Sequential chain |

---

## FR Traceability

| Step | FR Covered |
|------|------------|
| 1.1 | FR-004, FR-005, FR-006, FR-007, FR-008 |
| 1.2 | FR-004 |
| 1.3 | FR-008 |
| 1.4 | FR-005 |
| 1.5 | FR-005, FR-006 |
| 1.6 | FR-007 |
| 1.7 | FR-008 |
| 1.8 | FR-005–FR-008, FR-013 |
| 2.1 | FR-009, FR-010 |
| 2.2 | FR-003, FR-009, FR-010 |
| 2.3 | FR-001, FR-002 |
| 2.4 | FR-001, FR-005, FR-008 |
| 2.5 | FR-001, FR-002, FR-003, FR-012 |
| 2.6 | FR-002 |
| 2.7 | FR-011 |
