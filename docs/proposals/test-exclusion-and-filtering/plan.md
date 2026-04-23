# Implementation Plan: Test Exclusion & Content Filtering

**Design**: `docs/proposals/test-exclusion-and-filtering/design.md`
**Created**: 2026-04-10
**Updated**: 2026-04-13

---

## Implementation Status

| Phase | Step | Status | Notes |
|-------|------|--------|-------|
| **Phase 1** | 1. `is_test_path()` in cluster_constants.py | ✅ DONE | 17 regexes + false-positive guards, tests in test_cluster_constants.py |
| **Phase 2** | 2. Feature flags / UI options | ✅ DONE | Replaced env-var FeatureFlags with UI-driven `GenerateWikiRequest` fields + `config.py` server defaults |
| **Phase 3** | 3. Clustering soft-skip | ✅ DONE | `run_clustering(exclude_tests=True)` builds subgraph of non-test nodes |
| | 4. Empty structure fallback | ✅ DONE | All-test-nodes graph falls back to full graph with warning log |
| **Phase 4** | 5. WikiState fields | ✅ DONE | `planner_type: str`, `exclude_tests: bool` in WikiState |
| | 6. Service wiring | ✅ DONE | Request→wiki_service→toolkit→agent→WikiState, 4-layer pass-through |
| **Phase 5** | 7. Integration verification | ✅ DONE | `test_exclude_tests_integration.py` — 4 tests (clustering, DB flags, full pipeline) |

### Implementation Notes

- **Step 2 deviation**: FeatureFlags dataclass was rejected in favor of UI-driven approach. Options flow from `GenerateWikiRequest` → `wiki_service.py` (resolves None → config default) → `generate_wiki()` → `WikiState`.
- **Step 3**: `config.py` has `cluster_exclude_tests: bool = True` as server default when UI sends `None`.
- **Step 7**: Tests verify exclude_tests through clustering → unified DB → topology enrichment pipeline.

---

## Prerequisites

This proposal depends on:
- **Cluster Planner Migration** — `cluster_constants.py` exists (Step 1.1 of that plan)
- **Unified DB Activation** — `is_test` column in `repo_nodes`, `_upsert_nodes_batch()` tagging, `_backfill_is_test()` (Steps 3–4 of that plan)

Steps below assume those prerequisites are complete.

---

## Phase 1 — Detection

### Step 1: `_TEST_PATH_PATTERNS` + `is_test_path()` in `cluster_constants.py`

**File**: `backend/app/core/cluster_constants.py`
**FR Coverage**: FR-201

Port the 17 separate compiled regexes from DeepWiki's `constants.py`. The function returns `any(p.search(rel_path) for p in _TEST_PATH_PATTERNS)`.

**TDD Steps**:

1. Write test: `is_test_path("tests/test_auth.py")` → `True`
2. Write test: `is_test_path("src/auth.py")` → `False`
3. Write test: `is_test_path("__tests__/Auth.test.tsx")` → `True`
4. Write test: `is_test_path("src/testimony/service.py")` → `False` (false positive guard)
5. Write test: `is_test_path("src/contest/results.py")` → `False` (false positive guard)
6. Write test: `is_test_path("spec/models/user_spec.rb")` → `True`
7. Write test: `is_test_path("internal/auth/handler_test.go")` → `True`
8. Write test: `is_test_path("mocks/auth_mock.py")` → `True`
9. Write test: `is_test_path("testdata/sample.json")` → `True`
10. Write test: `is_test_path("testutils/helpers.py")` → `True`
11. Write test: verify `_TEST_PATH_PATTERNS` has exactly 17 entries
12. Add the 17 compiled regexes + `is_test_path()`. Verify all tests pass.

**Test File**: `tests/unit/test_cluster_constants.py`

---

## Phase 2 — Feature Flags

### Step 2: `FeatureFlags` dataclass in `feature_flags.py`

**File**: `backend/app/core/feature_flags.py`
**FR Coverage**: FR-207

**TDD Steps**:

1. Write test: `FeatureFlags()` has `exclude_tests=False` by default
2. Write test: `get_feature_flags()` with `WIKIS_EXCLUDE_TESTS=1` → `exclude_tests=True`
3. Write test: `get_feature_flags()` with env var unset → `exclude_tests=False`
4. Implement `FeatureFlags` dataclass + `get_feature_flags()`. Verify tests pass.

**Test File**: `tests/unit/test_feature_flags.py`

---

## Phase 3 — Clustering-Level Soft Skip

### Step 3: Soft-skip in `run_full_clustering_pipeline()`

**File**: `backend/app/core/graph_clustering.py`
**FR Coverage**: FR-204, FR-205, FR-206

This is the core implementation — the **only** layer that excludes test nodes.

**TDD Steps**:

1. Build a graph with 20 prod + 5 test nodes (test paths: `tests/test_*.py`, `__tests__/*.test.jsx`)
2. Write test: with `exclude_tests=True`, `G_cluster` contains 20 nodes (no test nodes)
3. Write test: with `exclude_tests=True`, test nodes have `macro_cluster=NULL` in DB after clustering
4. Write test: with `exclude_tests=False`, all 25 nodes participate in clustering
5. Write test: with `exclude_tests=True`, test nodes still exist in the original graph `G` (not removed)
6. Write test: `rel_path` fallback to `file_name` attribute works for test detection
7. Implement the soft-skip block:
   - Iterate `G.nodes(data=True)`, collect test nodes via `is_test_path()`
   - Build `G_cluster = G.subgraph(non_test_nodes).copy()`
   - Pass `G_cluster` to Leiden instead of `G`
8. Verify all tests pass.

**Test File**: `tests/unit/test_graph_clustering.py`

---

### Step 4: Empty Structure Fallback

**File**: `backend/app/core/graph_clustering.py`
**FR Coverage**: FR-208

**TDD Steps**:

1. Create a graph where ALL nodes are test files
2. Write test: with `exclude_tests=True`, `G_cluster` is empty → warning log emitted → fallback re-runs with full graph
3. Write test: warning message includes "All nodes are test files"
4. Implement fallback: if `len(G_cluster.nodes()) == 0`, log warning, set `G_cluster = G`, clear `excluded_test_nodes`
5. Verify tests pass.

**Test File**: `tests/unit/test_graph_clustering.py`

---

## Phase 4 — State Propagation

### Step 5: `exclude_tests` in WikiState

**File**: `backend/app/core/agents/wiki_state.py`
**FR Coverage**: FR-207

**TDD Steps**:

1. Write test: `WikiState()` has `exclude_tests=False` as default
2. Write test: `WikiState(exclude_tests=True)` serializes correctly
3. Add `exclude_tests: bool = False` field to `WikiState` (Pydantic BaseModel with DictAccessMixin). Verify tests pass.

**Test File**: `tests/unit/test_wiki_state.py`

---

### Step 6: UI Toggle Pass-Through

**File**: `backend/app/services/wiki_service.py`
**FR Coverage**: FR-207

**TDD Steps**:

1. Write test: wiki service passes `exclude_tests` from request/feature_flags into `WikiState`
2. Write test: `exclude_tests` from `WikiState` reaches `run_full_clustering_pipeline()` as `feature_flags.exclude_tests`
3. Wire the pass-through. Verify tests pass.

**Test File**: `tests/unit/test_wiki_service.py`

---

## Phase 5 — Verification

### Step 7: Test Nodes Remain Queryable

**File**: `tests/integration/test_test_exclusion.py`
**FR Coverage**: FR-206

**TDD Steps**:

1. Build a wiki with `exclude_tests=True` on a repo with test files
2. Write test: test nodes exist in DB with `is_test=1` and `macro_cluster=NULL`
3. Write test: FTS5 search for test symbol names returns results (no `WHERE is_test=0` filter)
4. Write test: vector search for test-related queries returns test documents
5. Verify tests pass.

**Test File**: `tests/integration/test_test_exclusion.py`

---

## Dependency Graph

```
Step 1 (is_test_path — 17 regexes)
Step 2 (FeatureFlags)
    ↓
Step 3 (clustering soft-skip) ← depends on Step 1 + Step 2
Step 4 (empty fallback) ← depends on Step 3
Step 5 (WikiState field) ← no deps
Step 6 (UI pass-through) ← depends on Step 2 + Step 5
Step 7 (verification) ← depends on Step 3 + Step 6
```

---

## Parallelization

| Group | Steps | Can Run In Parallel |
|-------|-------|---------------------|
| A | 1, 2, 5 | Yes (no deps) |
| B | 3, 6 | Yes (depend on Group A) |
| C | 4 | After Step 3 |
| D | 7 | After B + C |

---

## FR Traceability

| Step | FR Covered |
|------|------------|
| 1 | FR-201 |
| 2 | FR-207 |
| 3 | FR-204, FR-205 |
| 4 | FR-208 |
| 5 | FR-207 |
| 6 | FR-207 |
| 7 | FR-206 |

---

## What is NOT in this plan (matching DeepWiki)

The following are **explicitly excluded** because DeepWiki does not implement them:

- ❌ FAISS post-filtering — tests remain in vector store
- ❌ FTS5 `WHERE is_test = 0` — no retrieval-level SQL filtering
- ❌ `architectural_projection()` changes — no test-awareness in projection
- ❌ DeepAgents planner filtering — no filesystem middleware test skip
- ❌ Graph builder filtering — all files included in graph
- ❌ `GenerateWikiRequest.include_tests` API field — toggle is via feature flag, not request model
- ❌ Over-retrieve compensation — not needed since no retrieval filtering
