# Feature Specification: Test Exclusion & Content Filtering

**Feature Branch**: `feature/test-exclusion`
**Created**: 2026-04-10
**Status**: Draft
**Proposal**: `docs/proposals/test-exclusion-and-filtering/proposal.md`

---

## Key Entities

| Entity | Location | Role |
|--------|----------|------|
| `_TEST_PATH_PATTERNS` | `constants.py` (DeepWiki) → `cluster_constants.py` (wikis) | List of 17 separate compiled regexes |
| `is_test_path()` | Same file as above | Returns `any(p.search(rel_path) for p in _TEST_PATH_PATTERNS)` |
| `is_test` column | `repo_nodes` table in `unified_db.py` | `1` or `0`, set in `_upsert_nodes_batch()` |
| `_backfill_is_test()` | `unified_db.py` | Migration for older DBs without `is_test` |
| `FeatureFlags.exclude_tests` | `feature_flags.py` | `bool`, env var `DEEPWIKI_EXCLUDE_TESTS` |
| `run_full_clustering_pipeline()` | `graph_clustering.py` | Builds subgraph copy excluding test nodes when flag is on |
| `macro_cluster=NULL` | DB state for test nodes | Test nodes get no cluster → no wiki pages |

---

## User Scenarios & Testing

### User Story 1 — Soft-Skip Test Exclusion (Priority: P1)

When a user generates a wiki with the "Exclude tests" toggle ON, test files are excluded from page generation but remain in the graph and DB for ask/research queries.

**Why this priority**: This is the core feature — the soft-skip model matching DeepWiki's approach.

**Independent Test**: Generate a wiki for a Python project with `src/` and `tests/` directories with the toggle ON. Verify no wiki page references `test_*` symbols or `conftest.py`. Then run an ask query about test files — verify test symbols are returned from the DB/vector store.

**Acceptance Scenarios**:

1. **Given** a repository with `src/auth.py` and `tests/test_auth.py`, **When** a wiki is generated with `exclude_tests=True`, **Then** the clustering assigns `macro_cluster=NULL` to test nodes and no wiki pages are generated for them, but test nodes exist in the DB and graph.

2. **Given** a repository with `__tests__/Component.test.jsx` (Jest), **When** a wiki is generated with `exclude_tests=True`, **Then** test components have `macro_cluster=NULL` and no pages.

3. **Given** a repository with `conftest.py` and `fixtures/`, **When** `exclude_tests=True`, **Then** conftest and fixture nodes have `macro_cluster=NULL`, but remain queryable via ask/research.

---

### User Story 2 — UI Toggle for Test Inclusion (Priority: P2)

A UI switch controls whether test exclusion is active. When OFF (test inclusion), tests are clustered normally and may get wiki pages.

**Why this priority**: The user must be able to toggle the behavior from the UI.

**Acceptance Scenarios**:

1. **Given** a repository with tests, **When** the UI toggle is OFF (`exclude_tests=False`), **Then** test nodes are clustered via Leiden like any other code and may appear in wiki pages.

2. **Given** the UI toggle is ON, **When** wiki generation runs, **Then** `run_full_clustering_pipeline()` builds a subgraph copy excluding test nodes before Leiden.

---

### User Story 3 — Test Detection Accuracy (Priority: P1)

The 17 compiled regex patterns correctly identify test files across Python, JavaScript/TypeScript, Java, Go, Ruby, Rust, C++, and other languages.

**Why this priority**: False positives (excluding production code) are worse than false negatives (including tests). The patterns use directory boundaries.

**Acceptance Scenarios**:

1. **Given** `tests/test_auth.py`, **Then** `is_test_path()` returns `True`.
2. **Given** `src/auth.py`, **Then** `is_test_path()` returns `False`.
3. **Given** `__tests__/Auth.test.tsx`, **Then** `is_test_path()` returns `True`.
4. **Given** `src/testimony/service.py` (contains "test" in non-test context), **Then** `is_test_path()` returns `False`.
5. **Given** `src/contest/results.py` (contains "test" as substring), **Then** `is_test_path()` returns `False`.
6. **Given** `spec/models/user_spec.rb`, **Then** `is_test_path()` returns `True`.
7. **Given** `internal/auth/handler_test.go`, **Then** `is_test_path()` returns `True`.
8. **Given** `mocks/auth_mock.py`, **Then** `is_test_path()` returns `True`.
9. **Given** `testdata/sample.json`, **Then** `is_test_path()` returns `True`.
10. **Given** `testutils/helpers.py`, **Then** `is_test_path()` returns `True`.

---

### User Story 4 — Test Nodes Preserved for Ask/Research (Priority: P1)

Test nodes remain fully queryable in the DB, FTS5 index, and vector store even when excluded from page generation.

**Why this priority**: The soft-skip model requires tests to be available for research queries.

**Acceptance Scenarios**:

1. **Given** test exclusion is active and wiki pages have been generated, **When** an ask query searches for "test_auth", **Then** test nodes are returned from DB/FTS5/vector search.

2. **Given** test exclusion is active, **When** examining the DB, **Then** test nodes exist with `is_test=1` and `macro_cluster=NULL`, but are NOT deleted or hidden from queries.

---

### User Story 5 — Clustering-Level Exclusion (Priority: P1)

When `feature_flags.exclude_tests` is on, `run_full_clustering_pipeline()` removes test nodes from the graph copy used for Leiden clustering.

**Acceptance Scenarios**:

1. **Given** a graph with 20 prod + 5 test nodes, **When** `exclude_tests=True`, **Then** Leiden runs on a subgraph of 20 nodes. Test nodes are not in `G_cluster`.

2. **Given** the clustering completes, **Then** test nodes have `macro_cluster=NULL` in the DB.

3. **Given** `exclude_tests=False`, **Then** all 25 nodes participate in Leiden clustering.

---

## Edge Cases

1. **No tests in the repository** — Test exclusion is a no-op. No performance or correctness impact.
2. **Repository is 100% tests** (unusual) — All nodes get `macro_cluster=NULL` → empty structure. Fall back to including tests with a warning log.
3. **Test utility used by production code** — `test_utils/` nodes get `macro_cluster=NULL`, but `src/` code that imports from them still has edges in the graph. The wiki documents production code; cross-references to test utilities are acceptable losses.
4. **"test" in non-test context** — `testimony/`, `contest/`, `attest/` should NOT match. The 17 regexes use directory boundary anchors (`(^|/)tests?/` not `test`).
5. **`conftest.py` outside tests/** — `src/conftest.py` IS detected (has its own pattern `(^|/)conftest\.py$`). This is intentional.

---

## Requirements

### Functional Requirements

**FR-201**: `is_test_path(rel_path)` shall use 17 separate compiled regex patterns (9 directory-based + 8 file-based, all `re.IGNORECASE`) and return `any(p.search(rel_path) for p in patterns)`.

**FR-202**: `_upsert_nodes_batch()` in `unified_db.py` shall compute `is_test = 1 if is_test_path(rel_path) else 0` for each node during insert. (Covered by Unified DB proposal.)

**FR-203**: `_backfill_is_test()` shall update `is_test=1` for existing rows where `is_test_path(rel_path)` is True. Called from `_migrate_schema()`. (Covered by Unified DB proposal.)

**FR-204**: When `feature_flags.exclude_tests` is True, `run_full_clustering_pipeline()` shall build `G_cluster = G.subgraph(non_test_nodes).copy()` before running Leiden, where `non_test_nodes = set(G.nodes()) - excluded_test_nodes`.

**FR-205**: Test nodes excluded from clustering shall have `macro_cluster=NULL` in the DB.

**FR-206**: Test nodes shall remain fully queryable in the DB, FTS5 index, and vector store regardless of the `exclude_tests` flag. No `WHERE is_test = 0` clauses shall be added to any search or retrieval query.

**FR-207**: A UI toggle shall control `feature_flags.exclude_tests`. When ON, test exclusion is active. When OFF, tests are clustered normally.

**FR-208**: If test exclusion produces an empty clustering (all nodes are tests), the system shall fall back to including tests with an informational log.

### Non-Functional Requirements

**NFR-201**: `is_test_path()` shall complete in under 10μs per call (17 regex searches via `any()` with short-circuit).

**NFR-202**: The soft-skip model shall not increase total graph build time by more than 5% (only adds `is_test_path()` calls per node).

---

## Success Criteria

**SC-201**: Generated wikis for 10+ test repositories contain zero wiki pages for test symbols when `exclude_tests=True`.

**SC-202**: `is_test_path()` has zero false positives on a curated set of 50 production file paths.

**SC-203**: Ask/research queries return test symbols even when `exclude_tests=True` (soft-skip preserved).

**SC-204**: Test nodes in the DB have `is_test=1` and `macro_cluster=NULL` when excluded.

---

## Traceability

| AC | Covers FR |
|----|-----------|
| US-1 AC-1 | FR-201, FR-202, FR-204, FR-205 |
| US-1 AC-2 | FR-201, FR-204, FR-205 |
| US-1 AC-3 | FR-201, FR-205, FR-206 |
| US-2 AC-1 | FR-207 |
| US-2 AC-2 | FR-204, FR-207 |
| US-3 AC-1–10 | FR-201 |
| US-4 AC-1 | FR-206 |
| US-4 AC-2 | FR-205, FR-206 |
| US-5 AC-1 | FR-204 |
| US-5 AC-2 | FR-205 |
| US-5 AC-3 | FR-204 |
