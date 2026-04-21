# Proposal: Ask / Deep-Research Tooling Overhaul

**Created**: 2026-04-20
**Status**: Draft
**Depends on**: `unified-db-activation` (already merged)
**Depended on by**: future cluster-aware retrieval work

---

## Problem

Three independent issues surfaced while auditing the agent tool surface used by
`AskEngine` and `ResearchEngine` (Deep Research):

### 1. `query_graph` was silently degraded after a recent refactor

`WikiStorageProtocol.to_networkx()` is called on every Ask / Research session
to rebuild a `MultiDiGraph` from the persisted DB. The rebuild restored
nodes/edges, but it did **not** reattach the lookup indexes
(`_node_index`, `_simple_name_index`, `_full_name_index`, `_name_index`,
`_suffix_index`, `_decl_impl_index`, `_constant_def_index`) that
`GraphQueryService.resolve_symbol` and the content expander rely on.

Effect:
- `resolve_symbol` collapsed from a 6-stage cascade to FTS-only (strategy 6).
- `_jql_name_search` lost its O(1) path and fell back to `_jql_full_scan`
  which walks the entire graph.
- The content expander could no longer cross declaration → implementation
  edges or resolve constants by name, so `query_graph` looked like it was
  "working" but returned shallow / misleading data.

**Status**: Fixed in this PR by extracting `attach_graph_indexes()` as a
module-level helper in `code_graph/graph_builder.py` and calling it from
both `unified_db.py::to_networkx()` and `storage/postgres.py::to_networkx()`.

### 2. Two parallel toolsets in `research_tools.py`

`backend/app/core/deep_research/research_tools.py` defines two overlapping
families of tools:

| Family       | Tools                                                                                       | Used by              |
|--------------|---------------------------------------------------------------------------------------------|----------------------|
| Legacy       | `search_codebase`, `get_symbol_relationships`, `search_graph`, `think`                      | `ResearchEngine`     |
| Progressive  | `search_symbols`, `get_relationships_tool`, `get_code`, `search_docs`, `query_graph`, `think` | `AskEngine` (forced) |

`AskEngine` flips `WIKIS_PROGRESSIVE_TOOLS=1` to force the progressive
toolset; `ResearchEngine` doesn't, so the two engines operate on
**different abstractions of the same data**. This is confusing for prompt
authors, doubles the surface area we have to keep working, and means
fixes / improvements to one path silently skip the other.

### 3. `to_networkx()` rehydration cost per session

Even with the index fix, every Ask / Research call still walks the entire
DB to construct a NetworkX graph that's mostly thrown away — only a small
slice is actually queried. On large repos this is an avoidable per-session
cost.

A storage-protocol-native `GraphQueryService` could answer the same
queries directly against the SQLite/Postgres backend (via `StorageTextIndex`
+ a small `StorageGraphService`) and skip the rehydration entirely.

### 4. Structure-planner fix-up (separate but related)

The deepagents structure planner was crashing with
`unhashable type: 'RunnableRetry'` because `wiki_service.create_llm()`
wraps the LLM with `.with_retry()`, and deepagents' `resolve_model()`
hashes the model object. The other deepagents callers in this codebase
already pass `skip_retry=True`; the structure planner was the only one
left.

**Status**: Fixed in this PR by unwrapping `RunnableRetry` /
`RunnableBinding` to its inner `BaseChatModel` inside
`structure_engine._create_agent()`. Retry semantics still apply to
non-deepagents LLM calls.

## Solution Overview

Three deferred work-items, in dependency order. All scoped for **after**
the current bug-fix PR lands on `main`.

### Phase 1 — Consolidate the toolset (1–2 days)

1. Make the progressive toolset the default for both `AskEngine` and
   `ResearchEngine`.
2. Delete the legacy tools (`search_codebase`, `get_symbol_relationships`,
   `search_graph` as research-tool aliases) and the `WIKIS_PROGRESSIVE_TOOLS`
   feature flag.
3. Update prompts in both engines to reference only the progressive tool
   names.
4. Add a single doc page `docs/guide/agent-tools.md` enumerating the final
   tool surface so prompt authors have one place to look.

### Phase 2 — Storage-protocol-native graph queries (3–5 days)

1. Introduce `app/core/code_graph/storage_graph_service.py` exposing the
   subset of `GraphQueryService` that Ask / Research actually call:
   `resolve_symbol`, `get_neighbors`, `get_subgraph_around`,
   `find_path`. Each method delegates to:
   - `StorageTextIndex` for name / FTS lookups (already exists).
   - Targeted SQL on `repo_nodes` / `repo_edges` for relationship walks.
2. `toolkit_bridge._load_cached_artifacts` stops calling
   `db.to_networkx()`; instead it constructs `StorageGraphService(db)`
   and passes that into the agent context.
3. Keep the in-memory NetworkX path as the implementation used by the
   wiki-generation pipeline (where the whole graph is needed anyway) and
   for the unit tests that mock graphs.

### Phase 3 — Optional: structure-planner tool sharing

Once the progressive toolset is canonical, expose a single
`create_codebase_tools(db, repo_path)` factory in
`deep_research/research_tools.py` and have `structure_engine` reuse it
in addition to its current planner-specific tools (`explore_tree`,
`analyze_module`, `define_section`, …). The planner already has
`query_graph` / `search_graph` / `search_symbols` via
`structure_tools.py::StructureCollector.get_tools()`, but those are
hand-rolled wrappers around `GraphQueryService`. Sharing the factory
would mean the structure planner picks up future improvements (caching,
storage-native queries, etc.) automatically.

## Out of Scope

- Any change to the wiki-generation pipeline's graph usage.
- Any change to the cluster-planner / structure-skeleton path.
- Any change to FAISS / vector-store lifetime.

## Success Criteria

1. `query_graph` returns substantively richer results on a known repo
   (track top-N symbol hit count for a fixed query before/after).
2. `AskEngine` and `ResearchEngine` import the same tool factory.
3. Per-session `to_networkx()` time disappears from
   `toolkit_bridge._load_cached_artifacts` traces.
