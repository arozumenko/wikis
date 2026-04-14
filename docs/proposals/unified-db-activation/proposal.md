# Proposal: Unified DB Activation

**Created**: 2026-04-10
**Status**: Draft
**Depends on**: None (standalone)
**Depended on by**: `cluster-planner-migration`, `graph-topology-enrichment`, `test-exclusion-and-filtering`

---

## Problem

Wikis already has a complete `unified_db.py` module (`backend/app/core/unified_db.py`) that consolidates graph nodes, edges, FTS5 full-text index, sqlite-vec vector store, and wiki metadata into a single SQLite file. However, it is **disabled by default** behind `WIKIS_UNIFIED_DB=0` and not wired into the main pipeline.

Meanwhile, the generation pipeline currently uses:
- An in-memory NetworkX graph (lost between runs)
- FAISS for vector search (separate `.faiss` + `docstore.bin` files)
- `GraphTextIndex` for FTS5 (separate `.fts5.db` file)
- No persistent edge storage (structural edges reconstructed each time)

This fragmented storage prevents three incoming features from working:
1. **Graph Topology Enrichment** — needs `search_fts5()`, `search_vec()`, `upsert_edge()` on a unified DB
2. **Cluster Planner** — needs `set_cluster()`, `get_cluster_nodes()`, cluster-scoped search
3. **Test Exclusion** — needs `is_test` column on nodes for filtering

## Solution

Activate the existing `unified_db.py` as the primary persistence layer:

1. **Enable by default** — flip `WIKIS_UNIFIED_DB` to `"1"` (or remove the feature flag entirely)
2. **Add `is_test` column** — DeepWiki has this; wikis does not
3. **Add missing dependencies** — `sqlite-vec` and `pysqlite3-binary` are not in `pyproject.toml`
4. **Wire into pipeline** — after graph building + tree-sitter parsing, call `db.from_networkx(G)` to persist the graph, then use `db` throughout the rest of the pipeline (enrichment, clustering, expansion, retrieval)
5. **Replace FAISS** — the unified DB's `repo_vec` table with sqlite-vec replaces FAISS + docstore.bin
6. **Replace GraphTextIndex** — the unified DB's `repo_fts` table replaces the separate FTS5 database

## What Already Exists in Wikis

`unified_db.py` already provides:

| Capability | Method(s) | Status |
|-----------|-----------|--------|
| Node CRUD | `upsert_node()`, `upsert_nodes_batch()`, `get_node()` | ✅ Complete |
| Edge CRUD | `upsert_edge()`, `upsert_edges_batch()`, `get_edges_from/to()` | ✅ Complete |
| FTS5 search | `search_fts5(query, path_prefix, cluster_id, symbol_types, limit)` | ✅ Complete |
| Vector search | `search_vec(embedding, k, path_prefix, cluster_id)` | ✅ Complete |
| Hybrid RRF search | `search_hybrid(query, embedding, fts_weight, vec_weight)` | ✅ Complete |
| Cluster assignment | `set_cluster()`, `set_clusters_batch()`, `get_cluster_nodes()` | ✅ Complete |
| Hub flagging | `set_hub(node_id, is_hub, assignment)` | ✅ Complete |
| Metadata KV store | `set_meta()`, `get_meta()` | ✅ Complete |
| Graph I/O | `from_networkx(G)`, `to_networkx()` | ✅ Complete |
| Edge weight update | `update_edge_weights_batch()` | ✅ Complete |
| Embedding population | `populate_embeddings(embedding_fn, batch_size)` | ✅ Complete |

### Schema comparison (DeepWiki vs Wikis)

| Feature | DeepWiki `unified_db.py` | Wikis `unified_db.py` | Gap |
|---------|-------------------------|----------------------|-----|
| `repo_nodes` columns | 24 (incl. `is_test`) | 23 (no `is_test`) | **Add column** |
| `repo_edges` columns | 13 | 13 | **Identical** |
| `repo_fts` (FTS5) | `porter unicode61`, 4 searchable cols | `porter unicode61`, 4 searchable cols | **Identical** |
| `repo_vec` (sqlite-vec) | `float32[{dim}]` | `float32[{dim}]` | **Identical** |
| `wiki_meta` (KV) | key/value TEXT | key/value TEXT | **Identical** |
| Schema migration | None | None | Both need it eventually |
| sqlite-vec in deps | Yes (runtime) | **No** (not in pyproject.toml) | **Add dependency** |

## What Needs to Change

| Change | Scope | Risk |
|--------|-------|------|
| Add `is_test INTEGER DEFAULT 0` to `repo_nodes` | 1 line in schema | Low |
| Add `idx_nodes_test` partial index on `(is_test) WHERE is_test = 1` | 1 line in schema | Low |
| Add `sqlite-vec` + `pysqlite3` to `pyproject.toml` | 2 lines | Low |
| Flip feature flag to enabled-by-default | 1 line | Medium — dual-path: FAISS fallback if flag off |
| Wire `from_networkx(G)` after graph builder | ~10 lines in wiki agent | Medium |
| Replace FAISS retriever with `search_hybrid()` | ~30 lines in retrievers.py | Medium |
| Replace `GraphTextIndex` with unified DB FTS5 | ~20 lines | Medium |
| Add `db` to `WikiState` so downstream nodes access it | ~5 lines | Low |

## New Dependencies

```
sqlite-vec >= 0.1.9   # sqlite-vec extension for vector search (>= 0.1.9 required for aarch64 support)
pysqlite3 == 0.6.0    # SQLite with extension loading support (NOT pysqlite3-binary)
```

Both are **required**. The existing `unified_db.py` already attempts to load sqlite-vec via `_load_extensions()`.

## Risk Assessment

| Risk | Impact | Mitigation |
|------|--------|------------|
| sqlite-vec binary compatibility | Build failures on some platforms | `pysqlite3-binary` provides pre-built wheels; CI tests both |
| FAISS removal breaks existing wikis | Existing wikis can't be regenerated | Keep FAISS as read-only fallback for old wikis |
| Performance regression | Slower search | sqlite-vec KNN is comparable to FAISS for <100K vectors |
| DB file size | Larger than FAISS + docstore | One file vs. three; net size similar |
