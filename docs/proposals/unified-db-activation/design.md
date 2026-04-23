# Technical Design: Unified DB Activation

**Spec**: `docs/proposals/unified-db-activation/spec.md`
**Created**: 2026-04-10
**Updated**: 2026-04-10
**Scope**: Activate existing `unified_db.py`, add `is_test` column, wire into pipeline, add dependencies.

---

## 1. Module Changes

### Modified Files

| File | Change |
|------|--------|
| `backend/app/core/unified_db.py` | Add `is_test` column + partial index; set `is_test` in `_upsert_nodes_batch()`; add `_migrate_schema()` + `_backfill_is_test()`; add stale-node cleanup in `from_networkx()` |
| `backend/app/core/constants.py` | Add `_TEST_PATH_PATTERNS` (18 compiled regexes) and `is_test_path()` function |
| `backend/app/core/state/wiki_state.py` | Add `unified_db: Optional[Any] = None` field to `WikiState` (Pydantic `BaseModel`) |
| `backend/app/core/agents/wiki_graph_optimized.py` | Create `UnifiedWikiDB`, call `from_networkx()`, store in state |
| `backend/app/core/retrievers.py` | Add `UnifiedDBRetriever` using `db.search_hybrid()` |
| `backend/pyproject.toml` | Add `sqlite-vec >= 0.1.9`, `pysqlite3 == 0.6.0` |

### No New Files

The `unified_db.py` module already exists and is feature-complete. One new file is needed: `constants.py` for test path patterns (following DeepWiki convention).

---

## 2. Schema Change: `is_test` Column

### Current `repo_nodes` schema (in `_create_schema()`)

The existing schema has 23 data columns. Add `is_test` between `chunk_type` and `macro_cluster`:

```sql
-- ADD to CREATE TABLE repo_nodes:
is_test INTEGER DEFAULT 0,

-- ADD to indexes (PARTIAL index, not composite):
CREATE INDEX IF NOT EXISTS idx_nodes_test ON repo_nodes(is_test) WHERE is_test = 1;
```

### `_upsert_nodes_batch()` change

`is_test` is set inside `_upsert_nodes_batch()`, **not** in `_nx_node_to_dict()`. This follows the DeepWiki convention where `_nx_node_to_dict()` extracts raw fields and `_upsert_nodes_batch()` computes derived flags (`is_test`, `is_architectural`, `is_doc`):

```python
import re
from app.core.constants import is_test_path

# Inside _upsert_nodes_batch():
for node_id, data in batch:
    row = self._nx_node_to_dict(node_id, data)
    rel_path = row.get("rel_path", "") or ""
    row["is_test"] = 1 if is_test_path(rel_path) else 0
    rows.append(row)
```

### Test Path Patterns (`constants.py`)

Add `_TEST_PATH_PATTERNS` as 18 separate compiled regexes (matching DeepWiki's `constants.py`):

```python
import re as _re

_TEST_PATH_PATTERNS = [
    # Directory-based
    _re.compile(r'(^|/)tests?/', _re.IGNORECASE),
    _re.compile(r'(^|/)__tests__/', _re.IGNORECASE),
    _re.compile(r'(^|/)specs?/', _re.IGNORECASE),
    _re.compile(r'(^|/)testing/', _re.IGNORECASE),
    _re.compile(r'(^|/)test_?helpers?/', _re.IGNORECASE),
    _re.compile(r'(^|/)fixtures/', _re.IGNORECASE),
    _re.compile(r'(^|/)mocks?/', _re.IGNORECASE),
    _re.compile(r'(^|/)testdata/', _re.IGNORECASE),
    _re.compile(r'(^|/)testutils?/', _re.IGNORECASE),
    # File-based
    _re.compile(r'(^|/)test_[^/]+\.\w+$', _re.IGNORECASE),
    _re.compile(r'(^|/)[^/]+_test\.\w+$', _re.IGNORECASE),
    _re.compile(r'(^|/)[^/]+\.test\.\w+$', _re.IGNORECASE),
    _re.compile(r'(^|/)[^/]+\.spec\.\w+$', _re.IGNORECASE),
    _re.compile(r'(^|/)[^/]+Tests?\.\w+$', _re.IGNORECASE),
    _re.compile(r'(^|/)conftest\.py$', _re.IGNORECASE),
    _re.compile(r'(^|/)setup_tests?\.\w+$', _re.IGNORECASE),
]

def is_test_path(rel_path: str) -> bool:
    """Return True if rel_path matches any test heuristic."""
    return any(p.search(rel_path) for p in _TEST_PATH_PATTERNS)
```

Note: DeepWiki has 16 patterns listed. Count may vary slightly; the key design is **separate compiled patterns** iterated via `any()`, not a single combined regex.

---

## 2b. Schema Migration: `_migrate_schema()` + `_backfill_is_test()`

Pre-existing DB files (from before this proposal) won't have the `is_test` column. Add `_migrate_schema()` (called in `__init__`) and `_backfill_is_test()`:

```python
def _migrate_schema(self) -> None:
    """Apply incremental schema migrations for pre-existing DBs."""
    existing = {
        row[1]
        for row in self.conn.execute("PRAGMA table_info(repo_nodes)").fetchall()
    }
    if "is_test" not in existing:
        logger.info("Migrating repo_nodes: adding is_test column")
        self.conn.execute(
            "ALTER TABLE repo_nodes ADD COLUMN is_test INTEGER DEFAULT 0"
        )
        self._backfill_is_test()
        self.conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_nodes_test "
            "ON repo_nodes(is_test) WHERE is_test = 1"
        )
        self.conn.commit()
    else:
        # Column exists but may have stale values (added before backfill).
        stale = self.conn.execute(
            "SELECT count(*) FROM repo_nodes "
            "WHERE is_test = 0 AND ("
            "  rel_path GLOB 'test/*' OR rel_path GLOB 'tests/*' "
            "  OR rel_path GLOB '*/test/*' OR rel_path GLOB '*/tests/*'"
            ")"
        ).fetchone()[0]
        if stale > 0:
            self._backfill_is_test()
            self.conn.commit()

def _backfill_is_test(self) -> None:
    """Set is_test = 1 for all nodes whose rel_path matches test heuristics."""
    rows = self.conn.execute("SELECT node_id, rel_path FROM repo_nodes").fetchall()
    if not rows:
        return
    updates = [
        (r["node_id"],) for r in rows if is_test_path(r["rel_path"] or "")
    ]
    if updates:
        self.conn.executemany(
            "UPDATE repo_nodes SET is_test = 1 WHERE node_id = ?", updates,
        )
```

## 2c. Stale-Node Cleanup in `from_networkx()`

When re-generating a wiki, `from_networkx()` must purge nodes that existed in the DB from a previous run but are no longer in the incoming graph. This follows DeepWiki's approach:

```python
# At the end of from_networkx(), after all upserts:
existing_ids = {
    row[0] for row in self.conn.execute("SELECT node_id FROM repo_nodes").fetchall()
}
stale_ids = existing_ids - imported_ids  # imported_ids collected during upsert loop
if stale_ids:
    placeholders = ",".join("?" for _ in stale_ids)
    stale_list = list(stale_ids)
    # Delete edges first (FK-like constraint)
    self.conn.execute(
        f"DELETE FROM repo_edges WHERE source_id IN ({placeholders})"
        f" OR target_id IN ({placeholders})",
        stale_list + stale_list,
    )
    self.conn.execute(
        f"DELETE FROM repo_nodes WHERE node_id IN ({placeholders})",
        stale_list,
    )
```

---

## 3. Pipeline Integration

### Graph Building Phase (in `wiki_graph_optimized.py`)

After graph construction and before enrichment/clustering:

```python
from app.core.unified_db import UnifiedWikiDB

# In the graph building node:
db_path = storage_dir / f"{wiki_id}.wiki.db"
db = UnifiedWikiDB(db_path, embedding_dim=embedding_dim)

# Persist graph
db.from_networkx(G)

# Persist embeddings (if embedding function available)
if embedding_fn:
    db.populate_embeddings(embedding_fn, batch_size=64)

# Store in state for downstream nodes
state["unified_db"] = db
```

### Retriever Integration (in `retrievers.py`)

Add a thin wrapper that delegates to `db.search_hybrid()`:

```python
class UnifiedDBRetriever:
    """Retriever backed by unified DB (FTS5 + sqlite-vec RRF fusion)."""

    def __init__(self, db: UnifiedWikiDB, embedding_fn=None):
        self.db = db
        self.embedding_fn = embedding_fn

    def retrieve(
        self,
        query: str,
        path_prefix: str | None = None,
        cluster_id: int | None = None,
        limit: int = 20,
        exclude_test: bool = True,
    ) -> list[dict]:
        embedding = self.embedding_fn(query) if self.embedding_fn else None
        results = self.db.search_hybrid(
            query=query,
            embedding=embedding,
            path_prefix=path_prefix,
            cluster_id=cluster_id,
            limit=limit,
        )
        if exclude_test:
            results = [r for r in results if not r.get("is_test")]
        return results
```

### Cleanup (in pipeline completion handler)

```python
# After pipeline finishes:
db = state.get("unified_db")
if db:
    db.close()  # WAL checkpoint + release
```

---

## 4. Dependency Addition

### `pyproject.toml` changes

```toml
[project]
dependencies = [
    # ... existing ...
    "pysqlite3==0.6.0",       # NOT pysqlite3-binary; provides extension loading
    "sqlite-vec>=0.1.9",      # >= 0.1.9 required (0.1.6 broken on aarch64)
]
```

### Import guard in `unified_db.py`

The existing code already has a try/except for `pysqlite3`:

```python
try:
    import pysqlite3 as sqlite3
except ImportError:
    import sqlite3
```

This pattern is correct and should be preserved.

---

## 5. Feature Flag Transition

The feature flag lives in `unified_db.py` itself (not in `config.py`), matching DeepWiki's convention:

### Current state
```python
# In unified_db.py:
UNIFIED_DB_ENABLED = os.getenv("WIKIS_UNIFIED_DB", "0") == "1"
```

### Target state
```python
# In unified_db.py:
UNIFIED_DB_ENABLED = os.getenv("WIKIS_UNIFIED_DB", "1") == "1"
```

The flag remains available for emergency rollback but defaults to enabled.

---

## 6. FAISS Deprecation Path

FAISS is not removed in this phase. The transition is:

1. **Phase 1 (this proposal)**: Unified DB enabled by default. New wikis use unified DB. FAISS code remains for reading old wikis.
2. **Phase 2 (future)**: Remove FAISS code paths. Add migration utility to convert old wikis to unified DB.

---

## 7. Database File Layout

### Before (current)
```
storage/
  {wiki_id}/
    graph.pkl           # NetworkX graph (pickle)
    index.faiss         # FAISS vector index
    docstore.bin        # LangChain docstore
    fts5.db             # Separate FTS5 database
```

### After (unified DB)
```
storage/
  {wiki_id}/
    {wiki_id}.wiki.db   # Single unified DB (graph + vectors + FTS5 + metadata)
```

---

## 8. Compatibility with Downstream Proposals

| Proposal | Uses from unified DB |
|----------|---------------------|
| `graph-topology-enrichment` | `search_fts5()`, `search_vec()`, `upsert_edge()`, `upsert_edges_batch()`, `set_hub()`, `set_meta()`, `edge_count()` |
| `cluster-planner-migration` | `set_cluster()`, `set_clusters_batch()`, `get_cluster_nodes()`, `get_all_clusters()` |
| `test-exclusion-and-filtering` | `is_test` column for filtering in `search_fts5()` and `search_hybrid()` |

All methods already exist in `unified_db.py`. No schema or API changes are needed for downstream consumers.
