# Proposal: Test Exclusion & Content Filtering

**Feature Branch**: `feature/test-exclusion`
**Created**: 2026-04-10
**Status**: Draft
**Dependencies**: Unified DB Activation (docs/proposals/unified-db-activation/)

---

## Problem Statement

Generated wiki pages currently include test files alongside production code. For a typical repository, 20–40% of symbols are tests. Including them in wiki content has three negative consequences:

1. **Noisy, unfocused pages**: A page about "Authentication Service" may surface `test_login_success`, `test_login_failure`, `mock_token_provider` alongside the actual `LoginService` class. This confuses readers and dilutes the page's explanatory focus.

2. **Token waste**: Test symbols are retrieved during page content generation. LLM context windows are finite — every test symbol displaces a production symbol. For a 1000-token budget, test noise can consume 200–400 tokens.

3. **Incorrect structure planning**: Without test exclusion, the structure planner may create sections for test infrastructure (`conftest`, `fixtures`, `test_utils`) rather than production architecture.

DeepWiki solves this with a **soft-skip** model:
- `is_test_path(rel_path)` in `constants.py` — 17 separate compiled regexes for test file detection
- `is_test` flag on `repo_nodes` in `unified_db.py` — set during `_upsert_nodes_batch()`
- `FeatureFlags.exclude_tests` toggle (`DEEPWIKI_EXCLUDE_TESTS` env var)
- **Clustering-level exclusion only**: when the flag is on, `run_full_clustering_pipeline()` builds a subgraph copy excluding test nodes, so they end up with `macro_cluster=NULL` and no wiki pages are generated for them
- **No retrieval-level filtering**: test nodes remain fully queryable in the DB, FTS5 index, and vector store for ask/research

Key facts about what DeepWiki does **not** do:
- **No `FilterManager`** — this class does not exist in DeepWiki's codebase
- **No SQL `WHERE is_test = 0`** in any query — tests are not filtered at query time
- **No graph-builder-level filtering** — all files (test + non-test) are included in the graph
- **No retrieval-level filtering** — tests remain available for ask/research queries

Wikis has **no test detection** at any layer. The graph builder indexes all files indiscriminately.

---

## Proposed Solution

### Core Principle: Soft Skip (Preserve for Ask/Research)

The approach is a **soft skip**, matching DeepWiki's model: test nodes stay in the graph, DB, FTS5 index, and vector store — they are only excluded from clustering (and thus page generation). This means:
- Tests do **not** get wiki pages
- Tests **do** remain searchable via ask/research queries
- A UI toggle controls the behavior

### Phase 1: Test Path Detection

Port `is_test_path()` from DeepWiki's `constants.py`. This is a pure function — 17 separate compiled regex patterns matched against file paths via `any()`.

Patterns detected (17 patterns in 2 groups):
- **Directory-based** (9): `tests/`, `test/`, `__tests__/`, `specs/`, `spec/`, `testing/`, `test_helpers/`, `testhelpers/`, `fixtures/`, `mocks/`, `mock/`, `testdata/`, `testutils/`, `testutil/`
- **File-based** (8): `test_*.ext`, `*_test.ext`, `*.test.ext`, `*.spec.ext`, `*Test.ext`, `*Tests.ext`, `conftest.py`, `setup_test*.ext`

### Phase 2: Index-Level Tagging

During `_upsert_nodes_batch()`, compute `is_test` for each node and store it as a column in `repo_nodes` (already designed in the Unified DB proposal). This is **tagging only** — it does not filter anything. A `_backfill_is_test()` migration handles older databases.

### Phase 3: Clustering-Level Exclusion

When `feature_flags.exclude_tests` is on, `run_full_clustering_pipeline()` builds a **subgraph copy** excluding test nodes before running Leiden clustering. Test nodes end up with `macro_cluster=NULL` — they are never assigned to a cluster and thus never generate wiki pages.

This is the **only** layer that filters. No filtering happens at retrieval, FTS5 search, or vector search.

### Phase 4: UI Toggle

A UI toggle (switch) controls whether test exclusion is active for a given wiki generation. This maps to the `feature_flags.exclude_tests` flag. When off, tests are clustered and get wiki pages like any other code.

---

## Current Architecture Research

### DeepWiki Test Exclusion (source)

**Detection** — `constants.py`:
```python
# 17 separate compiled regexes, NOT one combined pattern
_TEST_PATH_PATTERNS = [
    # ── Directory-based (9) ──
    _re.compile(r'(^|/)tests?/', _re.IGNORECASE),
    _re.compile(r'(^|/)__tests__/', _re.IGNORECASE),
    _re.compile(r'(^|/)specs?/', _re.IGNORECASE),
    _re.compile(r'(^|/)testing/', _re.IGNORECASE),
    _re.compile(r'(^|/)test_?helpers?/', _re.IGNORECASE),
    _re.compile(r'(^|/)fixtures/', _re.IGNORECASE),
    _re.compile(r'(^|/)mocks?/', _re.IGNORECASE),
    _re.compile(r'(^|/)testdata/', _re.IGNORECASE),
    _re.compile(r'(^|/)testutils?/', _re.IGNORECASE),
    # ── File-based (8) ──
    _re.compile(r'(^|/)test_[^/]+\.\w+$', _re.IGNORECASE),
    _re.compile(r'(^|/)[^/]+_test\.\w+$', _re.IGNORECASE),
    _re.compile(r'(^|/)[^/]+\.test\.\w+$', _re.IGNORECASE),
    _re.compile(r'(^|/)[^/]+\.spec\.\w+$', _re.IGNORECASE),
    _re.compile(r'(^|/)[^/]+Tests?\.\w+$', _re.IGNORECASE),
    _re.compile(r'(^|/)conftest\.py$', _re.IGNORECASE),
    _re.compile(r'(^|/)setup_tests?\.\w+$', _re.IGNORECASE),
]

def is_test_path(rel_path: str) -> bool:
    return any(p.search(rel_path) for p in _TEST_PATH_PATTERNS)
```

**Tagging** — `unified_db.py` in `_upsert_nodes_batch()`:
```python
rel_path = n.get("rel_path", "")
is_test = 1 if is_test_path(rel_path) else 0
# Stored in repo_nodes table; migration backfill via _backfill_is_test()
```

**Feature flag** — `feature_flags.py`:
```python
@dataclass(frozen=True)
class FeatureFlags:
    exclude_tests: bool = False   # env var: DEEPWIKI_EXCLUDE_TESTS

def get_feature_flags() -> FeatureFlags:
    return FeatureFlags(
        exclude_tests=_env_bool("DEEPWIKI_EXCLUDE_TESTS"),
    )
```

**Clustering-level soft skip** — `graph_clustering.py` in `run_full_clustering_pipeline()`:
```python
if feature_flags.exclude_tests:
    for nid, data in G.nodes(data=True):
        rel_path = data.get("rel_path") or data.get("file_name") or ""
        if is_test_path(rel_path):
            excluded_test_nodes.add(nid)

    if excluded_test_nodes:
        non_test_nodes = set(G.nodes()) - excluded_test_nodes
        G_cluster = G.subgraph(non_test_nodes).copy()
# Test nodes end up with macro_cluster=NULL → no wiki pages generated
# But they remain fully queryable in DB, FTS5, and vector store for ask/research
```

**What does NOT exist in DeepWiki**:
- ❌ `FilterManager` class (does not exist — only appears in a planning doc as a future idea)
- ❌ SQL `WHERE is_test = 0` in any query (no retrieval-level filtering)
- ❌ Graph builder filtering (all files included in graph)
- ❌ FAISS post-filtering (test docs remain in vector store)
- ❌ `config.skip_tests` key (the actual key is `FeatureFlags.exclude_tests`)

### Wikis Current State (target)

- **No `is_test_path()` function** anywhere
- **No `is_test` column** in FTS5 or any index
- **No test detection** in graph builder, clustering, or planner
- **No UI toggle** for test inclusion/exclusion

---

## Gap Analysis

| Capability | DeepWiki | Wikis | Gap |
|-----------|---------|-------|-----|
| Test path detection (17 regexes) | ✅ `is_test_path()` in `constants.py` | ❌ | **Port function** |
| `is_test` flag in DB | ✅ `_upsert_nodes_batch()` | ❌ | **Add column + tagging** (Unified DB proposal) |
| Feature flag toggle | ✅ `FeatureFlags.exclude_tests` | ❌ | **Add feature flag + UI toggle** |
| Clustering-level soft skip | ✅ subgraph copy in `run_full_clustering_pipeline()` | ❌ | **Add exclusion in clustering** |
| `_backfill_is_test()` migration | ✅ in `_migrate_schema()` | ❌ | **Add migration** (Unified DB proposal) |
| Test nodes available for ask/research | ✅ No filtering at DB/FTS5/vector level | N/A (no tests detected) | **Preserve after implementing** |

**Not applicable (DeepWiki doesn't do these either)**:
- ❌ Graph-builder-level filtering — DeepWiki includes all files in graph
- ❌ Retrieval-level filtering — DeepWiki has no `WHERE is_test = 0` in queries
- ❌ FAISS post-filtering — Tests remain in vector store
- ❌ `FilterManager` — Does not exist

---

## Design

### Test Path Detection (`cluster_constants.py`)

Port `_TEST_PATH_PATTERNS` as 17 separate compiled regexes (matching DeepWiki exactly). The function uses `any(p.search(rel_path) for p in _TEST_PATH_PATTERNS)`.

### DB-Level Tagging (Unified DB proposal)

The `is_test` column is set during `_upsert_nodes_batch()` and backfilled via `_backfill_is_test()`. This is **tagging only** — no queries filter on it.

### Clustering-Level Exclusion

In `run_full_clustering_pipeline()`, when `feature_flags.exclude_tests` is on:
1. Iterate all graph nodes, collect those where `is_test_path(rel_path)` is True into `excluded_test_nodes`
2. Build `G_cluster = G.subgraph(non_test_nodes).copy()` 
3. Run Leiden clustering on `G_cluster`
4. Test nodes keep `macro_cluster=NULL` — no pages generated for them

`architectural_projection()` does NOT filter test nodes — exclusion happens only in the clustering pipeline.

### UI Toggle

A UI switch maps to `feature_flags.exclude_tests`. When off (default=False in DeepWiki), tests are clustered normally. When on, the soft-skip mechanism activates.

### No Retrieval Filtering

Tests remain fully queryable in DB, FTS5, and vector store. This is intentional — ask/research queries can reference test code.

---

## Risks & Mitigations

| Risk | Impact | Mitigation |
|------|--------|------------|
| Over-aggressive test detection (false positives) | Production code excluded from pages | Conservative regex with directory boundaries; UI toggle off to include |
| Under-detection (false negatives) | Test code gets wiki pages | Extensible pattern list (17 patterns); add patterns as discovered |
| UI toggle confusing for users | Users don't understand soft skip | Clear UI label: "Exclude test files from documentation" |
| Large test suites inflate graph/DB | More nodes in graph and DB | Acceptable — soft skip means no page generation cost for test nodes |

---

## Cost Estimate

- **Phase 1** (detection): Port 17-regex `is_test_path()` — minimal
- **Phase 2** (tagging): Already done in unified-db proposal (`is_test` column + backfill)
- **Phase 3** (clustering exclusion): ~20 lines in `run_full_clustering_pipeline()` — minimal
- **Phase 4** (UI toggle): 1 feature flag + UI switch component

Most infrastructure is built by the sibling proposals. This proposal adds soft-skip clustering logic on top.
