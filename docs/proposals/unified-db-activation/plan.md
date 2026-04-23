# Implementation Plan: Unified DB Activation

**Design**: `docs/proposals/unified-db-activation/design.md`
**Created**: 2026-04-10
**Updated**: 2026-04-13

---

## Implementation Status

| Step | Description | Status | Notes |
|------|-------------|--------|-------|
| 1 | Add dependencies to pyproject.toml | ✅ DONE | `sqlite-vec>=0.1.9` added, installed & verified |
| 2 | Add `is_test` column to schema | ✅ DONE | Column + partial index in `_SCHEMA_NODES` |
| 3 | `is_test_path()` + set `is_test` in upsert | ✅ DONE | Wired into `_upsert_nodes_batch()` via `cluster_constants.is_test_path` |
| 4 | Backfill `is_test` on existing DBs | ✅ DONE | `_migrate_schema()` + `_backfill_is_test()` |
| 4b | Stale-node cleanup in `from_networkx()` | ✅ DONE | Deletes nodes + edges not in imported graph |
| 5 | Flip feature flag default | ✅ DONE | `WIKIS_UNIFIED_DB` defaults to `"1"` (enabled) |
| 6 | Add `unified_db` to WikiState | ✅ DONE | `unified_db: Any | None = None` in wiki_state.py |
| 7 | Wire `from_networkx()` into pipeline | ✅ DONE | `_ensure_unified_db()` in agent + Phase 2 enrichment |
| 8 | Add `UnifiedDBRetriever` | ✅ DONE | `unified_retriever.py` ported + 42 tests |
| 9 | Pipeline cleanup | ✅ DONE | `_close_cluster_db()` already in agent |
| 10 | Verify end-to-end | ✅ DONE | Integration test in `test_exclude_tests_integration.py` ||

---

## Step 1: Add Dependencies to pyproject.toml

**File**: `backend/pyproject.toml`
**FR Coverage**: FR-007

**Steps**:

1. Add `pysqlite3==0.6.0` to `[project].dependencies`
2. Add `sqlite-vec>=0.1.9` to `[project].dependencies`
3. Run `pip install -e ".[dev]"` in backend
4. Verify: `python -c "import sqlite_vec; print(sqlite_vec.loadable_path())"`
5. Verify: `python -c "import pysqlite3; print(pysqlite3.sqlite_version)"`

**Test**: No unit test — verified via install check.

---

## Step 2: Add `is_test` Column to Schema

**File**: `backend/app/core/unified_db.py`
**FR Coverage**: FR-002, FR-008

**TDD Steps**:

1. Write test: create a `UnifiedWikiDB`, verify `repo_nodes` table has `is_test` column
2. Write test: insert a node with `is_test=1`, retrieve it, verify flag is set
3. Write test: partial index `idx_nodes_test` exists (`WHERE is_test = 1`)
4. Add `is_test INTEGER DEFAULT 0` to `CREATE TABLE repo_nodes` in `_create_schema()`
5. Add `CREATE INDEX IF NOT EXISTS idx_nodes_test ON repo_nodes(is_test) WHERE is_test = 1` to `_create_schema()`
6. Verify all tests pass.

**Test File**: `tests/unit/test_unified_db.py::TestIsTestColumn`

---

## Step 3: Add `is_test_path()` to constants.py + set `is_test` in `_upsert_nodes_batch()`

**Files**: `backend/app/core/constants.py` (new), `backend/app/core/unified_db.py`
**FR Coverage**: FR-008

**TDD Steps**:

1. Build a NetworkX graph with nodes from `src/main.py` and `tests/test_main.py`
2. Write test: after `from_networkx(G)`, node from `src/main.py` has `is_test=0`
3. Write test: node from `tests/test_main.py` has `is_test=1`
4. Write test: node from `__tests__/Component.test.jsx` has `is_test=1`
5. Write test: node from `fixtures/data.json` has `is_test=1`
6. Create `constants.py` with `_TEST_PATH_PATTERNS` (18 separate compiled regexes) and `is_test_path()` function
7. In `_upsert_nodes_batch()` (NOT `_nx_node_to_dict()`), call `is_test_path(rel_path)` and set `row["is_test"]`
8. Verify all tests pass.

**Test File**: `tests/unit/test_unified_db.py::TestIsTestFromNetworkx`

---

## Step 4: Add `_migrate_schema()` + `_backfill_is_test()`

**File**: `backend/app/core/unified_db.py`
**FR Coverage**: FR-010

**TDD Steps**:

1. Write test: open a pre-existing DB without `is_test` column, instantiate `UnifiedWikiDB`, verify column was added
2. Write test: pre-existing DB with test-path nodes, verify `_backfill_is_test()` sets `is_test=1` on them
3. Write test: pre-existing DB with stale `is_test=0` but matching test paths, verify re-backfill
4. Implement `_migrate_schema()` called from `__init__()` after `_create_schema()`
5. Implement `_backfill_is_test()` using `is_test_path()` from constants
6. Verify all tests pass.

**Test File**: `tests/unit/test_unified_db.py::TestMigrateSchema`

---

## Step 4b: Add Stale-Node Cleanup in `from_networkx()`

**File**: `backend/app/core/unified_db.py`
**FR Coverage**: FR-009

**TDD Steps**:

1. Write test: persist graph A with nodes {a,b,c}, then persist graph B with nodes {b,c,d}, verify node `a` was deleted
2. Write test: verify stale edges referencing deleted nodes are also removed
3. At the end of `from_networkx()`, compare existing node_ids vs imported node_ids, delete stale rows
4. Verify all tests pass.

**Test File**: `tests/unit/test_unified_db.py::TestStaleNodeCleanup`

---

## Step 5: Flip Feature Flag Default

**File**: `backend/app/core/unified_db.py` (NOT `config.py`)
**FR Coverage**: FR-001

**Steps**:

1. Change `os.getenv("WIKIS_UNIFIED_DB", "0")` → `os.getenv("WIKIS_UNIFIED_DB", "1")`
2. Verify existing tests still pass (they should — most tests don't check the flag)

---

## Step 6: Add `unified_db` to WikiState

**File**: `backend/app/core/state/wiki_state.py`
**FR Coverage**: FR-006

**TDD Steps**:

1. Write test: `WikiState` can be created with `unified_db=None` (default)
2. Write test: `WikiState` can carry a `UnifiedWikiDB` instance
3. Add `unified_db: Optional[Any] = None` to `WikiState` Pydantic `BaseModel` (uses `model_config = ConfigDict(arbitrary_types_allowed=True)` for DB instance)
4. Verify all tests pass.

**Test File**: `tests/unit/test_wiki_state.py`

---

## Step 7: Wire `from_networkx()` into Pipeline

**File**: `backend/app/core/agents/wiki_graph_optimized.py`
**FR Coverage**: FR-003, FR-004, FR-006

**TDD Steps**:

1. Write integration test: after graph building node runs, `state["unified_db"]` is a `UnifiedWikiDB` instance
2. Write integration test: `db.node_count()` matches `G.number_of_nodes()`
3. Write integration test: `db.edge_count()` matches `G.number_of_edges()`
4. In the graph building node (after `G` is constructed):
   - Create `UnifiedWikiDB(db_path, embedding_dim)`
   - Call `db.from_networkx(G)`
   - If `embedding_fn` available: call `db.populate_embeddings(embedding_fn)`
   - Set `state["unified_db"] = db`
5. Verify all tests pass.

**Test File**: `tests/integration/test_unified_db_pipeline.py`

---

## Step 8: Add `UnifiedDBRetriever`

**File**: `backend/app/core/retrievers.py`
**FR Coverage**: FR-005

**TDD Steps**:

1. Write test: `UnifiedDBRetriever.retrieve(query)` returns results from `search_hybrid()`
2. Write test: `exclude_test=True` filters out `is_test=1` nodes
3. Write test: `path_prefix` and `cluster_id` are passed through
4. Implement `UnifiedDBRetriever` class (thin wrapper around `db.search_hybrid()`)
5. Wire as default retriever when `UNIFIED_DB_ENABLED` is True
6. Verify all tests pass.

**Test File**: `tests/unit/test_retrievers.py::TestUnifiedDBRetriever`

---

## Step 9: Pipeline Cleanup

**File**: `backend/app/core/agents/wiki_graph_optimized.py`
**FR Coverage**: FR-006

**Steps**:

1. In the pipeline completion handler (or `finally` block), call `db.close()`
2. Ensure the DB file is WAL-checkpointed before the wiki is served

---

## Step 10: Verify End-to-End

**Integration Test**:

1. Generate a wiki for a small test repo
2. Verify `.wiki.db` file exists in storage directory
3. Open the DB directly, verify:
   - `repo_nodes` has expected count
   - `repo_edges` has expected count
   - `repo_fts` returns results for a known symbol
   - `repo_vec` has embeddings (if embedding_fn provided)
   - `wiki_meta` has generation metadata
4. Verify wiki content quality is unchanged vs. FAISS-based pipeline

**Test File**: `tests/integration/test_unified_db_e2e.py`

---

## Dependency Graph

```
Step 1 (pyproject.toml deps)
   └─► Step 2 (is_test column)
          └─► Step 3 (is_test_path + _upsert_nodes_batch)
                 └─► Step 4 (_migrate_schema + _backfill + stale cleanup)

Step 5 (flip feature flag) ─────────────┐
Step 6 (WikiState field) ──────────────┐ │
                                       ▼ ▼
Step 7 (wire into pipeline) ─► Step 8 (retriever) ─► Step 9 (cleanup) ─► Step 10 (e2e)
```

---

## Estimated Changes

| File | Lines Changed | Risk |
|------|--------------|------|
| `pyproject.toml` | +2 | Low |
| `constants.py` (new) | ~30 (test path patterns + is_test_path) | Low |
| `unified_db.py` | +50 (schema + _upsert_nodes_batch + _migrate_schema + _backfill + stale cleanup) | Medium |
| `wiki_state.py` | +2 | Low |
| `wiki_graph_optimized.py` | +15 | Medium |
| `retrievers.py` | +30 | Medium |
| **Tests** | ~150 lines new | — |
