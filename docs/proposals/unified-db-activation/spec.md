# Feature Specification: Unified DB Activation

**Feature Branch**: `feature/unified-db-activation`
**Created**: 2026-04-10
**Status**: Draft
**Input**: Activate the existing `unified_db.py` as the primary persistence layer, replacing FAISS + separate FTS5 + in-memory-only graph.
**Proposal**: `docs/proposals/unified-db-activation/proposal.md`

---

## User Scenarios & Testing

### User Story 1 — Unified DB as Primary Storage (Priority: P0)

When a wiki is generated, the code graph, embeddings, and FTS5 index are persisted in a single SQLite file (`{wiki_id}.wiki.db`) via the existing `UnifiedWikiDB` class. This replaces the current FAISS + docstore.bin + separate FTS5 file approach.

**Why this priority**: Every other proposal (cluster planner, graph topology enrichment, test exclusion) depends on having the unified DB active. Without it, there is no persistent storage for synthetic edges, cluster assignments, hub flags, or `is_test` markers.

**Independent Test**: Generate a wiki for a small repository. Verify a `.wiki.db` file is created containing `repo_nodes`, `repo_edges`, `repo_fts`, `repo_vec`, and `wiki_meta` tables. Verify node count matches the graph builder output.

**Acceptance Scenarios**:

1. **Given** a repository to generate a wiki for, **When** the graph builder completes, **Then** `db.from_networkx(G)` is called to persist all nodes and edges into the unified DB.

2. **Given** a persisted unified DB, **When** the retriever performs a search, **Then** it uses `db.search_hybrid()` (FTS5 + sqlite-vec RRF fusion) instead of FAISS + separate FTS5.

3. **Given** an existing wiki generated with FAISS, **When** the user regenerates, **Then** the new wiki uses the unified DB. Old FAISS files are not deleted but are no longer consulted.

---

### User Story 2 — `is_test` Column (Priority: P0)

Every node in the code graph has an `is_test` flag indicating whether it comes from a test file. This flag is set during graph building based on file path patterns.

**Why this priority**: The `test-exclusion-and-filtering` proposal requires this column to exclude test code from wiki content.

**Independent Test**: Build a code graph with `src/main.py` and `tests/test_main.py`. Persist to unified DB. Verify `is_test=0` for `main.py` nodes and `is_test=1` for `test_main.py` nodes.

**Acceptance Scenarios**:

1. **Given** a node with `rel_path` matching test patterns (`tests/`, `__tests__/`, `.test.`, `_test.`, `.spec.`, `_spec.`), **When** persisted to the DB, **Then** `is_test=1`.

2. **Given** the `is_test` column, **When** a downstream consumer queries with `search_fts5(...)`, **Then** results can be filtered by `is_test=0` to exclude test code.

---

### User Story 3 — DB in WikiState (Priority: P0)

The `UnifiedWikiDB` instance is available throughout the LangGraph pipeline via `WikiState`, so that downstream nodes (topology enrichment, cluster planner, expansion engine, content writer) can all access it.

**Why this priority**: Without the DB in state, every downstream node would need to independently open the DB file, leading to connection conflicts and inconsistent state.

**Independent Test**: Run the wiki generation pipeline. In a downstream LangGraph node (e.g., expansion), access `state["unified_db"]`. Verify it is a live `UnifiedWikiDB` instance with the expected node count.

**Acceptance Scenarios**:

1. **Given** the wiki generation pipeline, **When** graph building completes, **Then** a `UnifiedWikiDB` instance is created and stored in `WikiState["unified_db"]`.

2. **Given** the pipeline completes, **When** `WikiState["unified_db"]` is accessed by a downstream node, **Then** it contains all nodes, edges, and embeddings from the graph builder phase.

3. **Given** the pipeline finishes, **When** cleanup runs, **Then** `db.close()` is called to WAL-checkpoint and release the file.

---

### User Story 4 — Dependencies in pyproject.toml (Priority: P0)

`sqlite-vec` and `pysqlite3` are listed as required dependencies so that `unified_db.py` can load the vector extension.

**Why this priority**: Without these packages installed, `_load_extensions()` silently fails and `vec_available` is `False`, disabling vector search entirely.

**Independent Test**: In a fresh virtual environment, `pip install -e ".[dev]"` and then `python -c "import sqlite_vec; print(sqlite_vec.loadable_path())"`. Verify no ImportError.

**Acceptance Scenarios**:

1. **Given** a fresh install of the backend, **When** `UnifiedWikiDB` is instantiated, **Then** `vec_available` is `True`.

2. **Given** `pysqlite3` is installed, **When** `unified_db.py` imports `sqlite3`, **Then** it uses `pysqlite3` (which supports `enable_load_extension`).

---

## Edge Cases

1. **Extension loading fails on Alpine/musl** — `pysqlite3-binary` may not have musl wheels. Mitigation: document the system requirement or provide a Docker base image with pre-built extensions.

2. **Concurrent generation on same wiki** — Two simultaneous generations could conflict on the same `.wiki.db` file. Mitigation: wiki generation already uses locking; the unified DB uses WAL mode for concurrent reads.

3. **Very large repos (>100K nodes)** — `from_networkx()` batches at 5000 rows. Ensure batch size is sufficient and doesn't OOM. DeepWiki uses the same batch size successfully for large repos.

4. **Read-only mode for serving** — When serving wiki content, the DB should be opened `readonly=True` to allow concurrent reads without locking. The constructor already supports this.

---

## Requirements

### Functional Requirements

**FR-001**: The unified DB (`UnifiedWikiDB`) shall be enabled by default. The `WIKIS_UNIFIED_DB` feature flag shall default to `"1"`.

**FR-002**: The `repo_nodes` schema shall include an `is_test INTEGER DEFAULT 0` column with a partial index: `CREATE INDEX idx_nodes_test ON repo_nodes(is_test) WHERE is_test = 1`.

**FR-003**: After graph building, the pipeline shall call `db.from_networkx(G)` to persist all nodes and edges.

**FR-004**: After embedding computation, the pipeline shall call `db.populate_embeddings(embedding_fn)` to persist all vectors.

**FR-005**: The retrieval system shall use `db.search_hybrid()` for semantic + lexical search, replacing FAISS + separate FTS5.

**FR-006**: The `UnifiedWikiDB` instance shall be stored in `WikiState["unified_db"]` for access by downstream pipeline nodes.

**FR-007**: `sqlite-vec >= 0.1.9` and `pysqlite3 == 0.6.0` shall be listed as required dependencies in `pyproject.toml`.

**FR-008**: The `_upsert_nodes_batch()` method (not `_nx_node_to_dict()`) shall set `is_test=1` for nodes whose `rel_path` matches test path patterns (18 separate compiled regexes via `is_test_path()`).

**FR-009**: `from_networkx()` shall purge stale nodes + edges from previous runs: any `node_id` in the DB but not in the incoming graph is deleted (edges referencing stale nodes are deleted first).

**FR-010**: A `_migrate_schema()` method shall add the `is_test` column to pre-existing DB files that lack it, and call `_backfill_is_test()` to tag all existing test nodes.

### Non-Functional Requirements

**NFR-001**: `from_networkx()` for a 10,000-node graph shall complete in under 10 seconds.

**NFR-002**: `search_hybrid()` shall return results in under 100ms for typical queries.

**NFR-003**: The unified DB file size shall not exceed 2x the combined size of the previous FAISS + docstore + FTS5 files.

---

## Key Entities

- **UnifiedWikiDB** — The existing class at `backend/app/core/unified_db.py`. No new class needed.
- **WikiState["unified_db"]** — New state field carrying the DB instance through the pipeline. Note: `WikiState` is a Pydantic `BaseModel` (with `DictAccessMixin`), **not** a `TypedDict`.
- **`is_test_path()`** — Module-level function using `_TEST_PATH_PATTERNS` (18 separate compiled regexes), following the DeepWiki convention from `constants.py`.
