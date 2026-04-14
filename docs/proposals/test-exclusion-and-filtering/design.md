# Technical Design: Test Exclusion & Content Filtering

**Spec**: `docs/proposals/test-exclusion-and-filtering/spec.md`
**Created**: 2026-04-10

---

## 1. Module Structure

### New Files

| File | Purpose |
|------|---------|
| `backend/app/core/feature_flags.py` | `FeatureFlags` dataclass with `exclude_tests: bool` |

### Modified Files

| File | Change |
|------|--------|
| `backend/app/core/cluster_constants.py` | `_TEST_PATH_PATTERNS` (17 regexes) + `is_test_path()` |
| `backend/app/core/code_graph/unified_db.py` | `is_test` column + `_upsert_nodes_batch()` tagging + `_backfill_is_test()` (covered by Unified DB proposal) |
| `backend/app/core/graph_clustering.py` | Soft-skip in `run_full_clustering_pipeline()` |

### Explicitly NOT Modified (matching DeepWiki)

| File | Reason |
|------|--------|
| `backend/app/core/code_graph/graph_builder.py` | Graph builder includes ALL files — no filtering |
| `backend/app/core/retrievers.py` | No retrieval-level filtering — tests stay queryable |
| `backend/app/core/code_graph/graph_text_index.py` | No `WHERE is_test = 0` in search queries |
| `backend/app/core/graph_clustering.py` → `architectural_projection()` | No test-awareness — exclusion happens in `run_full_clustering_pipeline()` only |

---

## 2. Test Path Detection

### `cluster_constants.py`

Port from DeepWiki's `constants.py`. The patterns are 17 **separate** compiled regexes (NOT one combined pattern):

```python
import re as _re

_TEST_PATH_PATTERNS = [
    # ── Directory-based (9) ──
    _re.compile(r'(^|/)tests?/', _re.IGNORECASE),              # tests/, test/
    _re.compile(r'(^|/)__tests__/', _re.IGNORECASE),           # JavaScript __tests__/
    _re.compile(r'(^|/)specs?/', _re.IGNORECASE),              # Ruby/JS spec/
    _re.compile(r'(^|/)testing/', _re.IGNORECASE),             # Go testing/
    _re.compile(r'(^|/)test_?helpers?/', _re.IGNORECASE),      # test_helper/, testhelpers/
    _re.compile(r'(^|/)fixtures/', _re.IGNORECASE),            # Test fixtures
    _re.compile(r'(^|/)mocks?/', _re.IGNORECASE),              # Mock directories
    _re.compile(r'(^|/)testdata/', _re.IGNORECASE),            # Go test data
    _re.compile(r'(^|/)testutils?/', _re.IGNORECASE),          # Test utilities
    # ── File-based (8) ──
    _re.compile(r'(^|/)test_[^/]+\.\w+$', _re.IGNORECASE),         # test_foo.py
    _re.compile(r'(^|/)[^/]+_test\.\w+$', _re.IGNORECASE),         # foo_test.go
    _re.compile(r'(^|/)[^/]+\.test\.\w+$', _re.IGNORECASE),        # foo.test.js
    _re.compile(r'(^|/)[^/]+\.spec\.\w+$', _re.IGNORECASE),        # foo.spec.js
    _re.compile(r'(^|/)[^/]+Tests?\.\w+$', _re.IGNORECASE),        # FooTest.java
    _re.compile(r'(^|/)conftest\.py$', _re.IGNORECASE),             # pytest conftest
    _re.compile(r'(^|/)setup_tests?\.\w+$', _re.IGNORECASE),       # Test setup files
]

def is_test_path(rel_path: str) -> bool:
    """Check if a relative file path matches test code patterns.

    Uses 17 separate compiled regexes with ``any()`` short-circuit.
    Returns ``True`` if the path matches any test pattern.
    """
    return any(p.search(rel_path) for p in _TEST_PATH_PATTERNS)
```

Key design decisions:
- **17 separate regexes** with `any()` short-circuit — matches DeepWiki exactly
- All patterns are `re.IGNORECASE`
- Directory patterns use `(^|/)` boundary to avoid substring matches (`testimony/` won't match `tests?/`)
- File patterns use `\.\w+$` to require a file extension

---

## 3. Feature Flags

### `feature_flags.py`

Port from DeepWiki's `feature_flags.py`:

```python
import os
from dataclasses import dataclass

def _env_bool(key: str) -> bool:
    return os.getenv(key, "").strip().lower() in ("1", "true", "yes")

@dataclass(frozen=True)
class FeatureFlags:
    exclude_tests: bool = False   # env: WIKIS_EXCLUDE_TESTS

def get_feature_flags() -> FeatureFlags:
    return FeatureFlags(
        exclude_tests=_env_bool("WIKIS_EXCLUDE_TESTS"),
    )
```

The env var name changes from `DEEPWIKI_EXCLUDE_TESTS` to `WIKIS_EXCLUDE_TESTS` for the wikis project. The UI toggle overrides this per-request.

---

## 4. DB-Level Tagging (Unified DB proposal)

Already designed in `unified-db-activation`. The `is_test` column is:
- Set in `_upsert_nodes_batch()`: `is_test = 1 if is_test_path(rel_path) else 0`
- Backfilled via `_backfill_is_test()` for older databases
- Indexed with `CREATE INDEX IF NOT EXISTS idx_repo_nodes_is_test ON repo_nodes(is_test) WHERE is_test = 1`

**No search queries filter on `is_test`**. The column exists for:
1. Clustering exclusion (this proposal)
2. Potential future UI display (e.g., showing test badge on nodes)

---

## 5. Clustering-Level Soft Skip

### `graph_clustering.py` — `run_full_clustering_pipeline()`

This is the **only** layer that excludes test nodes. The exclusion builds a subgraph copy:

```python
def run_full_clustering_pipeline(
    G: nx.DiGraph,
    db: UnifiedDB,
    feature_flags: FeatureFlags,
    ...
) -> ClusteringResult:
    excluded_test_nodes: set = set()

    # ── Soft-skip: remove test nodes from clustering input ──
    if feature_flags.exclude_tests:
        for nid, data in G.nodes(data=True):
            rel_path = data.get("rel_path") or data.get("file_name") or ""
            if is_test_path(rel_path):
                excluded_test_nodes.add(nid)

        if excluded_test_nodes:
            non_test_nodes = set(G.nodes()) - excluded_test_nodes
            G_cluster = G.subgraph(non_test_nodes).copy()
        else:
            G_cluster = G
    else:
        G_cluster = G

    # ── Run Leiden on G_cluster (not G) ──
    # ... existing clustering logic using G_cluster ...

    # ── Test nodes keep macro_cluster=NULL ──
    # They are never assigned to a cluster → no wiki pages generated
    # They remain in DB + FTS5 + vector store for ask/research
```

Key facts:
- `architectural_projection()` does NOT filter test nodes — it only filters by `symbol_type`
- The test exclusion happens **after** `architectural_projection()`, in the clustering pipeline itself
- `G.subgraph(non_test_nodes).copy()` creates a new graph — the original `G` is not modified
- Test nodes with `macro_cluster=NULL` are never assigned to a page
- The `rel_path` field is checked with fallback to `file_name` (matching DeepWiki exactly)

---

## 6. What is NOT Implemented (Matching DeepWiki)

The following are **explicitly NOT added** because DeepWiki doesn't do them either:

### No Graph Builder Filtering
Graph builder includes ALL files (test + non-test). This preserves:
- Cross-reference edges from production to test code
- Test nodes in the graph for ask/research queries
- Ability to toggle tests back on without rebuilding the graph

### No Retrieval-Level Filtering
No `WHERE is_test = 0` is added to any FTS5 or vector search query. Test symbols remain fully searchable. This enables:
- Ask queries about test infrastructure
- Research queries that cross-reference test and production code

### No FAISS Post-Filtering
FAISS results are not filtered for test documents. Test documents remain in the vector store and are returned alongside production documents.

### No Structure Planner Filtering
Neither `architectural_projection()` nor the DeepAgents planner explicitly filter tests. Test exclusion is solely the responsibility of `run_full_clustering_pipeline()`.

---

## 7. UI Toggle

The UI provides a switch component: **"Exclude test files from documentation"**

- **ON** → `feature_flags.exclude_tests = True` → soft-skip active
- **OFF** → `feature_flags.exclude_tests = False` → tests clustered normally

The toggle value is passed from the UI through the wiki generation request to the clustering pipeline. The exact API contract (how the toggle maps to the request payload) will be determined during implementation with the UI team.

---

## 8. Empty Structure Fallback

```python
# In run_full_clustering_pipeline() — after Leiden:
if feature_flags.exclude_tests and len(G_cluster.nodes()) == 0:
    logger.warning(
        "All nodes are test files; re-running clustering with tests included"
    )
    G_cluster = G
    excluded_test_nodes.clear()
    # Re-run Leiden on full graph
```

---

## 9. State Propagation

```python
# WikiState is a Pydantic BaseModel with DictAccessMixin, NOT TypedDict
class WikiState(BaseModel, DictAccessMixin):
    class Config:
        arbitrary_types_allowed = True
    # ... existing fields ...
    exclude_tests: bool = False   # From feature flag / UI toggle
```

The `exclude_tests` flag flows: UI toggle → request → WikiState → `run_full_clustering_pipeline()`.

---

## 10. Testing Strategy

| Test | File | What |
|------|------|------|
| `is_test_path` accuracy (17 patterns) | `tests/unit/test_cluster_constants.py` | All 10 AC scenarios from spec + false-positive guards |
| `is_test_path` false positives | `tests/unit/test_cluster_constants.py` | `testimony/`, `contest/`, `attest/` → False |
| Clustering soft-skip | `tests/unit/test_graph_clustering.py` | Test nodes excluded from `G_cluster`, `macro_cluster=NULL` |
| Clustering with flag off | `tests/unit/test_graph_clustering.py` | All nodes participate when `exclude_tests=False` |
| Test nodes remain queryable | `tests/unit/test_graph_clustering.py` | After clustering, test nodes exist in DB with `is_test=1` |
| Feature flags | `tests/unit/test_feature_flags.py` | `WIKIS_EXCLUDE_TESTS=1` → `True`, unset → `False` |
| Empty structure fallback | `tests/integration/test_test_exclusion.py` | All-test repo → fallback includes tests |
