# Plan: Ask / Deep-Research Tooling Overhaul

**Created**: 2026-04-20
**Owner**: backend
**Status**: Draft (work to start after current bug-fix PR lands on `main`)

This plan covers the three deferred work-items from `proposal.md`. Each
phase is independently shippable and reversible.

---

## Phase 1 — Consolidate the agent toolset

**Goal**: One toolset, one prompt vocabulary, no feature flag.

### Steps

1. **Audit call sites of legacy tools**
   - `rg "search_codebase|get_symbol_relationships|search_graph"` across
     `backend/app` and tests.
   - List every caller and which engine path it belongs to.

2. **Promote progressive tools to default**
   - In `backend/app/core/deep_research/research_tools.py`, remove the
     `WIKIS_PROGRESSIVE_TOOLS` env-var branch. The progressive bundle
     becomes the only return value of `create_codebase_tools()`.
   - Update `ResearchEngine` to drop the legacy bundle; it already calls
     the same factory.
   - Delete the legacy implementations (`search_codebase`,
     `get_symbol_relationships`, `search_graph`) and their helpers.

3. **Update prompts**
   - `backend/app/core/ask_engine.py` — already references progressive
     tools. No change.
   - `backend/app/core/deep_research/research_engine.py` — replace any
     mention of `search_codebase` / `search_graph` with the progressive
     equivalents (`search_symbols`, `query_graph`).

4. **Doc page**
   - New file: `docs/guide/agent-tools.md` enumerating
     `search_symbols`, `query_graph`, `get_relationships_tool`,
     `get_code`, `search_docs`, `think`, plus their argument shapes and
     when to use each. Linked from the main guide index.

5. **Tests**
   - Update / delete unit tests under `tests/unit/deep_research/` that
     pin the legacy tool names. Most should swap to the progressive
     names with no behavior change.
   - Add one integration test that verifies both engines see the same
     tool list.

### Done when

- `git grep WIKIS_PROGRESSIVE_TOOLS` returns no hits in `backend/`.
- `git grep "search_codebase\|get_symbol_relationships"` only matches
  removed files / changelog.
- `pytest tests/unit/ -q` ≥ current pass count.

### Risk / rollback

Low. Delete one file's worth of code; the new path is already in
production behind the flag. Rollback = revert the PR.

---

## Phase 2 — Storage-protocol-native `GraphQueryService`

**Goal**: Eliminate `to_networkx()` rehydration on every Ask / Research
session.

### Steps

1. **Carve the actual API surface used by agents**
   - Grep `GraphQueryService` usage in
     `backend/app/core/deep_research/research_tools.py` and any callers.
   - Expected subset (verify before coding):
     - `resolve_symbol(name, file=None, lang=None) -> NodeRef | None`
     - `get_neighbors(node_id, rel_types=None, limit=None) -> list[NodeRef]`
     - `get_subgraph_around(node_id, depth=1) -> SubgraphView`
     - `find_path(src, dst, max_hops=4) -> list[NodeRef] | None`

2. **Define a backend-agnostic interface**
   - New file: `backend/app/core/code_graph/graph_service_protocol.py`
     with a `Protocol` covering the methods above. Both
     `GraphQueryService` (NetworkX-backed) and
     `StorageGraphService` (DB-backed) will satisfy it.

3. **Implement `StorageGraphService(WikiStorageProtocol)`**
   - New file: `backend/app/core/code_graph/storage_graph_service.py`.
   - `resolve_symbol` → reuse `StorageTextIndex.search_by_name` +
     priority cascade (port the cascade from
     `attach_graph_indexes` so SQLite/Postgres get the same ranking).
   - `get_neighbors` → SQL `SELECT * FROM repo_edges WHERE source_id = ?`
     (or `target_id = ?` for reverse). Filter by `rel_type` server-side.
   - `get_subgraph_around` → BFS via repeated `get_neighbors` capped by
     `depth` and a hard node-limit; keep the result as a thin
     `SubgraphView` (dict of node_id → row + adjacency map).
   - `find_path` → bidirectional BFS over `repo_edges`. Cap at
     `max_hops`.

4. **Wire into `toolkit_bridge`**
   - `backend/app/services/toolkit_bridge.py::_load_cached_artifacts`:
     stop calling `db.to_networkx()`; instead set
     `components.graph_service = StorageGraphService(db)`.
   - `research_tools.create_codebase_tools(...)` accepts
     `graph_service` (the protocol) instead of `code_graph` /
     `query_service`.

5. **Keep the NetworkX path for generation**
   - The wiki-generation pipeline still legitimately needs the full
     in-memory graph (clustering, expansion, etc.). It continues to use
     `to_networkx()` + `GraphQueryService`. Nothing to change there.

6. **Tests**
   - Port the `GraphQueryService` unit-test cases against
     `StorageGraphService` using a fixture sqlite-vec DB.
   - Mark the NetworkX-backed cases as still-passing reference
     implementations.

### Done when

- `toolkit_bridge` no longer references `to_networkx`.
- Time-to-first-tool-call for Ask drops by ≥ N% on a benchmark repo
  (capture before/after with the existing performance trace flag).
- All existing graph-query unit tests pass against both backends.

### Risk / rollback

Medium. Two-backend implementations always drift; mitigate with the
shared protocol + a parametrised test suite. Rollback = restore the
`to_networkx()` call in `toolkit_bridge`.

---

## Phase 3 — Optional: share tool factory with structure planner

**Goal**: Structure planner picks up future agent-tool improvements for
free.

### Steps

1. Refactor `structure_tools.py::StructureCollector.get_tools()` so the
   "exploration" subset (`query_graph`, `search_graph`, `search_symbols`)
   is delegated to `create_codebase_tools(graph_service)`.
2. Keep the planner-specific tools (`explore_tree`, `analyze_module`,
   `define_section`, `define_page`, `batch_define_pages`,
   `set_wiki_metadata`) where they are.
3. Update structure-planner prompts only if tool names / arg shapes
   actually change.

### Done when

- `git grep "_handle_query_graph\|_handle_search_graph\|_handle_search_symbols"`
  returns zero hits — those handlers are now in the shared factory.

### Risk / rollback

Low. Drop-in tool replacement; collector keeps the same `get_tools()`
contract.

---

## Sequencing

```
PR-1 (this PR — already done)
  └─ Bug fix: attach_graph_indexes from to_networkx() + structure-planner
     RunnableRetry unwrap.

PR-2  Phase 1  Consolidate toolset                    (1–2 d)
  └─ depends on PR-1
PR-3  Phase 2  StorageGraphService + toolkit_bridge   (3–5 d)
  └─ depends on PR-2 (cleaner diff)
PR-4  Phase 3  Share factory with structure planner   (½–1 d, optional)
  └─ depends on PR-3
```

Phases 2 and 3 should be reviewed together when scoping starts; the
protocol design in Phase 2 directly affects Phase 3's diff size.
