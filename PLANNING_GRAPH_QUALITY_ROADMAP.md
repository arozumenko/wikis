# Coordinated Graph-Quality + Cross-Language + Federation Execution Roadmap

> Single source of truth that supersedes/orders the four planning documents:
>
> 1. `GRAPH_QUALITY_IMPROVEMENTS.md` — earlier proposal (lexical cascade, dedup, parity).
> 2. `PLANNING_GRAPH_QUALITY_IMPROVEMENTS.md` — IDF gating, REST disambiguation, Phase2Stats.
> 3. `PLANNING_ORPHAN_RESOLUTION_V2.md` — explicit-refs Pass 1, hybrid RRF, embedding reuse.
> 4. `PLANNING_CROSS_LANGUAGE_GRAPH_FEDERATION.md` — intra-repo cross-language + project federation.
>
> The Phase 0–6 changes also back-port to the upstream **deepwiki_plugin**
> (`pylon_deepwiki/plugins/deepwiki_plugin/plugin_implementation/`); see
> `deepwiki_plugin/PLANNING_GRAPH_QUALITY_BACKPORT.md` for the trimmed plan.

---

## 0. Current implementation audit — 2026-04-28

This replaces the original pre-implementation reconciliation table. Phases 0–9 now have substantial code in place. The first 2026-04-28 follow-up wired several gaps (cross-repo API-surface evidence, Phase 6 L0 handoff, `node_id_style` default flip, lifecycle PATCH/add/remove triggers), and the later 2026-04-28 gap-closure pass addressed the second audit's P0 implementation mismatches. The audit narrative below is retained as historical diagnosis; the status table in section 0.5 is the current source of truth for what was closed and what remains.

### 0.1 Phase status snapshot

This table preserves the second-audit snapshot; section 0.5 records the gap-closure status for items that changed during the follow-up implementation pass.

| Phase | Current implementation evidence | Visible gap or ambiguity | Actionable next step |
|---|---|---|---|
| 0 — Storage parity | `backend/app/core/storage/protocol.py`, `sqlite.py`, and `postgres.py` now expose the planned FTS/path/vector methods, `score_norm`, `get_embedding_by_id`, and batch similarity paths. Tests exist in `backend/tests/storage/test_storage_parity.py` and `backend/tests/storage/test_score_normalization.py`. | Closed. | Keep as baseline; do not regress protocol parity when adding new storage methods. |
| 1 — Observability/dedup/provenance | `backend/app/core/graph_topology_diagnostics.py` provides `Phase2Stats`; `_add_edge` stores `provenance` and uses typed-edge dedup; SQLite/Postgres persist `repo_edges.provenance`. Tests exist in `backend/tests/unit/test_phase2_observability.py` and `backend/tests/unit/test_edge_dedup.py`. | The roadmap called these integration tests, but they currently live mostly in unit coverage. This is fine for fast checks, but there is no full `FilesystemIndexer -> run_phase2 -> repo_meta` regression. | Add one integration test that indexes a tiny repo and asserts `phase2_stats_v2`, provenance, dedup, and score thresholds survive the real storage path. |
| 2 — IDF/tiered lexical/REST | `backend/app/core/graph_lexical_v2.py` implements IDF, T1–T4 lexical tiers, REST disambiguation, and tests. `WIKI_ORPHAN_LEXICAL_TIERED` / `WIKI_ORPHAN_LEXICAL_IDF_GATE` default ON. | The original rollout table said lexical tiering should default OFF until Phase 3. The implementation flipped it ON to satisfy real-repo wiring. The roadmap still contains both stories. | Update the rollout schedule to mark the ON-by-default behavior intentional, or explicitly revert defaults. Do not leave the contradiction unresolved. |
| 3 — Cascade reorder/explicit refs/embedding reuse | `backend/app/core/graph_topology.py` dispatches v2 as explicit refs -> hybrid -> tiered lexical -> directory; `backend/app/core/graph_orphan_cascade_v2.py` provides explicit markdown/backtick/import refs and embedding reuse. | Closed for ordering. | Add a single integration test that asserts pass order from emitted stats on a fixture where every pass has at least one possible hit. |
| 4 — Hybrid RRF | `backend/app/core/graph_orphan_hybrid.py` implements RRF fusion and the v2 cascade calls it before lexical. Tests exist in `backend/tests/unit/test_rrf_fusion.py`. | Matryoshka/halfvec remain unimplemented, which was optional. The planned slow perf test (`test_phase2_perf.py`) is not present. | Mark Matryoshka/halfvec explicitly out of this wiring push, and add a small perf/contract test that proves hybrid does not re-embed when `repo_vec` already has orphan embeddings. |
| 5 — Node disambiguation | `backend/app/core/code_graph/graph_builder.py` implements `WIKI_NODE_ID_STYLE=stem|rel_path`; qualified/FQN indexes are implemented and tested. **2026-04-28 follow-up:** default flipped from `stem` to `rel_path` in both `FeatureFlags` and `get_feature_flags()` env loader. `WIKI_NODE_ID_STYLE=stem` still pins the legacy collision-prone behavior for back-compat. | Partial: rich symbol node creation uses `rel_path`, but documentation nodes, basic-parser nodes, and several relationship-resolution fallbacks still construct legacy stem-shaped IDs. See 0.4. | Finish node-id propagation, especially relationship source/target resolution and project-level namespacing. |
| 6 — Cross-language linker/test linker | `backend/app/core/filesystem_indexer.py` runs Phase 1c before `from_networkx`; `api_surface_extractor.py`, `cross_language_linker.py`, and `test_linker.py` exist; `cross_language`, `test_link`, and `cross_repo` are synthetic in `graph_topology.py`. `architectural_projection()` now skips `cross_language` and `test_link` edges. `test_linker` is **disabled by default** after a real-repo run produced 1.25M edges from one TS workspace (the same-stem heuristic was symbol-level Cartesian; it is now bounded to file/module-level nodes). **2026-04-28 follow-up:** Phase 1c now passes parser-supplied `cross_language_relationships` (captured from `analysis.cross_language_relationships`) into `run_cross_language_linker`. | Partial: the handoff is wired, but `_detect_cross_language_relationships()` is currently a stub returning `[]`; in the normal non-streaming path `parse_results` is also cleared before that stub runs. L1 API extraction also depends on `source_text`, which rich parser graph nodes do not currently expose as a top-level node attr. L2 remains out of scope and test linker remains opt-in. | Treat Phase 6 as structurally wired but not functionally complete until L0 emits real parser relationships and L1 works on rich parser nodes. |
| 7 — Project storage/federation | `backend/app/core/storage/project_storage.py` provides memory/SQLite/Postgres backends; `backend/app/services/multi_wiki_components.py` chooses `FederatedQueryService`/`FederatedRetrieverStack` when flags and project storage are available. | `project_nodes` exists in storage but `upsert_project_node` is unused in the application path. Federated retriever currently resolves target content through per-wiki query services rather than materialized `project_nodes`, so the storage schema and the retriever design disagree. The module docstring still says SQLite/Postgres are follow-up work even though they are implemented. | Either materialize project nodes during `recompute_project`, or revise the Phase 7 contract to say project storage only owns edges/relatedness/meta and node lookup remains delegated to per-wiki services. Add an integration test for `build_multi_wiki_components` with persisted `project_edges`. |
| 8 — Cross-repo linker/relatedness/project clustering | `backend/app/services/project_recompute.py` scores every wiki pair, persists `repo_relatedness`, calls `build_cross_repo_edges`, persists `project_edges`, and optionally stores community metadata. **2026-04-28 follow-up:** `recompute_project()` now extracts `surfaces_by_node` from the merged graph via `extract_api_surfaces_for_graph(...)` (best-effort, defaults to `{}`) and threads it into `build_cross_repo_edges`. `build_cross_repo_edges` accepts a new `allow_same_language=True` flag (default ON) which adds the `link_api_surface_any_language` matcher to the cascade so two same-language repos sharing a REST surface still produce `cross_repo` edges; the wiki_id guard immediately downstream filters within-repo pairs. | Partial: cross-repo graph nodes/edges are not namespaced by wiki id, stale project edges/relatedness are not cleared before recompute, and API-surface extraction may still be empty if source text was not present on graph nodes. FTS/vector hit maps are still not threaded through. | Add end-to-end tests with colliding node IDs and removed wikis; then namespace project graph IDs and clear/rebuild persisted project edges atomically. |
| 9 — Lifecycle/SSE/UI | Backend SSE route `POST /projects/{project_id}/recompute`, lifecycle stale marking on wiki rebuild, read-time auto-enqueue, `subscribeRecomputeSSE`, and `RecomputeWidget` exist. **2026-04-28 follow-up:** new `mark_project_stale(project_id, reason, settings)` helper added to `services/project_recompute.py` (gated by `flags.project_graph`, swallows storage errors). `PATCH /projects/{id}`, `POST /projects/{id}/wikis`, and `DELETE /projects/{id}/wikis/{wiki_id}` now call it with reasons `project_updated`, `wiki_added:<id>`, `wiki_removed:<id>` so the next read-time `maybe_enqueue_recompute` enqueues a background recompute. | Partial: triggers are wired, but there is no in-flight recompute guard, no stale-edge cleanup, manual recompute does not enforce owner-only access in the route, generated OpenAPI/types still omit the route, and there are no frontend/backend SSE contract tests. `ProjectPage.tsx` import grouping for `RecomputeWidget` is still inconsistent. | Add recompute locking/idempotency, stale cleanup, route authorization tests, generated types, SSE contract tests, component/API tests, and import cleanup. |

### 0.2 Highest-priority implementation gaps

0. **`test_linker` is disabled by default after a real-repo edge explosion.** A single TypeScript workspace produced **1,254,462** `test_link` edges in Phase 1c. Root cause: `link_same_stem` paired every test-file *symbol* with every prod-file *symbol* sharing a stem (N×M Cartesian per file pair, multiplied by common stems like `index`, `utils`, `base`). Mitigations applied: `WIKI_TEST_LINKER` defaults to `False`; `link_same_stem` is now restricted to file/module-level nodes; `architectural_projection()` skips `test_link` and `cross_language` edges so they can never inflate Leiden communities. Re-enabling on real repos is gated on a richer matcher that handles Gherkin features, AI markdown/yaml fixtures, and CLI snapshot tests — stem-only heuristics are not safe for those.
1. **(Closed 2026-04-28) Cross-repo edges in the real recompute path.** `recompute_project()` extracts API surfaces from the merged graph, namespaces project node IDs, clears stale project rows before rebuild, persists project nodes, and threads same-language API-surface evidence into `build_cross_repo_edges`. Remaining verification: FTS/vector hit maps are still not threaded through, and real-repo UI testing is required.
2. **(Closed for parser-supplied `target_file` evidence 2026-04-28) Phase 6 L0 in Phase 1c.** `EnhancedUnifiedGraphBuilder._detect_cross_language_relationships()` now runs before parse-result cleanup and emits L0 rows for parser relationships whose `target_file` resolves to another language. Remaining work: broaden L0 producers beyond `target_file` signals.
3. **(Closed 2026-04-28) `rel_path` node IDs.** Graph-builder node creation/resolution now uses centralized rel-path-aware ID construction for rich, documentation, basic-parser, diagnostic, source, and target paths. Cross-repo project IDs are also namespaced as `wiki_id::raw_node_id`.
4. **(Partially closed 2026-04-28) Project lifecycle triggers.** `mark_project_stale` plus PATCH/add/remove hooks mark stale; recompute clears stale rows, writes a lightweight `recompute_in_progress` marker, and manual recompute is owner-gated. Remaining work: route tests and a stronger lock/lease if multi-process recompute concurrency becomes a requirement.
5. **(Closed 2026-04-28) Project storage `project_nodes` design choice.** `project_storage.py` now documents memory/SQLite/Postgres support, and `recompute_project()` materializes project nodes with `wiki_id` and `raw_node_id`. Federated content lookup still delegates to per-wiki query services, with project nodes available as auxiliary materialized metadata.
6. **Frontend/backend recompute contract is untested (open).** Add backend SSE schema tests and frontend stream/widget tests; generated API types are stale for the recompute route.

### 0.3 Test gaps to add before release

- `backend/tests/integration/test_phase1c_indexing.py`: build a tiny multi-language repo through `FilesystemIndexer`, then assert persisted `api_surface`, `cross_language`, and `test_link` rows.
- `backend/tests/integration/test_orphan_cascade_v2_order.py`: force explicit, hybrid, lexical, and directory candidates and assert the first successful pass wins.
- `backend/tests/integration/test_project_recompute_edges.py`: two related repos with shared API surface in the same language and in different languages; assert relatedness rows, non-zero cross-repo edges, clustering metadata, and SSE events.
- `backend/tests/integration/test_recompute_sse_contract.py`: validate `project_relatedness_progress`, `cross_repo_linker_progress`, `project_clustering_progress`, `recompute_complete`, and `recompute_error` payloads match the frontend union.
- `web/src/spa/api/__tests__/sse-recompute.test.ts`: parse JSON-RPC and flat recompute events, including terminal complete/error events.
- `web/src/spa/components/__tests__/RecomputeWidget.test.tsx`: start/cancel/error/success paths and per-phase progress rendering.

### 0.4 Deep holistic implementation audit — 2026-04-28

This section is the second-pass audit requested after the first closure sweep. It reviews the current roadmap implementation against the three source plans (`PLANNING_GRAPH_QUALITY_IMPROVEMENTS.md`, `PLANNING_ORPHAN_RESOLUTION_V2.md`, `PLANNING_CROSS_LANGUAGE_GRAPH_FEDERATION.md`) and against the code that now exists. The headline is: the roadmap is no longer mostly theoretical, but several high-risk areas are only structurally wired. The remaining work is less about adding files and more about making the existing files agree on identity, evidence, lifecycle, and weight semantics.

#### 0.4.1 Current verdict by roadmap area

| Area | Evidence in code | Audit verdict | Why it matters |
|---|---|---|---|
| Storage parity | `backend/app/core/storage/protocol.py`, `sqlite.py`, `postgres.py` expose the new storage hooks (`count_fts_matches`, `search_fts_by_column`, `search_fts_with_path`, `get_embedding_by_id`, `batch_similarity_search`, normalized scores). | Mostly closed. | The storage surface exists, but higher-level code does not always consume the more precise methods. Example: `graph_lexical_v2.py` still implements all lexical tiers by one broad `search_fts5(...)` call and in-memory filtering rather than the planned column/path-specific query methods. |
| Phase-2 observability and provenance | `graph_topology.py` writes provenance via `_add_edge`; `Phase2Stats` is persisted under `phase2_stats_v2`; edge dedup exists. | Closed at unit level, integration gap remains. | Unit tests cover the helpers, but there is still no full `FilesystemIndexer -> run_phase2 -> storage meta` integration test proving stats/provenance survive the real storage path. |
| Tiered lexical / IDF / REST disambiguation | `graph_lexical_v2.py` implements IDF gating, tier eligibility, generic REST-class query disambiguation, and score filtering. | Partially closed. | The logic exists, but the v2 hybrid pass can preempt it with pure FTS when embeddings are missing. That means common-name FTS matches can still resolve before the tiered IDF guard gets a chance to run. |
| Orphan cascade reorder / hybrid RRF | `graph_topology.resolve_orphans()` dispatches explicit refs -> hybrid -> lexical -> directory when `orphan_cascade_v2=True`; `graph_orphan_hybrid.py` implements RRF. | Partially closed. | The cascade order exists, but `resolve_orphans_hybrid()` explicitly degrades to pure FTS on missing embeddings and then `graph_topology` stores those edges as `edge_class="semantic"`. This mismatches the source plan, where embedding-less orphans should fall through to lexical-only Pass 3. |
| Node disambiguation | `FeatureFlags.node_id_style` and `get_feature_flags()` now default to `rel_path`; rich parser symbol nodes use rel-path slugs. | Partially closed. | Several graph-builder paths still synthesize legacy stem IDs: documentation nodes, basic parser nodes, direct/fallback relationship resolution, and some diagnostics. Cross-repo storage also does not namespace node IDs by wiki. |
| Cross-language linker | `cross_language_linker.py` exists with L0/L1/L2/L3; Phase 1c calls it before `from_networkx`; `cross_language` is excluded from architectural projection. | Structurally wired, functionally incomplete. | L0 is wired but `_detect_cross_language_relationships()` returns `[]`; L1 depends on `source_text`, but rich parser graph nodes do not expose source text as a top-level attr; L2 has no real FTS/vec maps in Phase 1c. |
| Test linker safety | `WIKI_TEST_LINKER` defaults to `False`; `link_same_stem()` is restricted to file/module-level nodes; `test_link` is excluded from architectural projection. | Mitigated, not release-ready. | The 1.25M-edge explosion class is mitigated. The richer test-fixture matcher (Gherkin, AI markdown/yaml prompts, CLI snapshots) is still absent, so default-off is correct. |
| Cross-repo linker | `project_recompute.py` computes relatedness and calls `build_cross_repo_edges`; the linker now accepts `surfaces_by_node` and `allow_same_language=True`. | Partially closed, high-risk identity gap remains. | The cascade can now produce same-language API-surface edges in unit tests, but project graphs use original node IDs rather than `wiki_id::node_id` namespacing. Colliding node IDs across repos can overwrite nodes in `merged_graph` and resolve to the wrong wiki at query time. |
| Project storage / federation | `project_storage.py`, `federated_query_service.py`, `federated_retriever.py`, and `multi_wiki_components.py` exist. | Design mismatch. | The source federation plan described materialized project nodes/edges with source wiki IDs and rich node columns. Current storage persists only `(node_id, wiki_id, attrs)` and currently recompute only writes edges/relatedness/meta, not project nodes. This can be a valid design choice, but docs and retrieval code must explicitly align to it. |
| Lifecycle / SSE / UI | `mark_project_stale()`, GET-time `maybe_enqueue_recompute()`, manual recompute SSE route, `subscribeRecomputeSSE`, and `RecomputeWidget` exist. | Partially closed. | Stale marking exists, but recompute has no in-flight guard, old project edges are not cleared, manual recompute is not owner-gated in the route, generated OpenAPI/types are stale, and SSE/widget tests are missing. |

#### 0.4.2 Ranked remaining gaps and logic mismatches

1. **Hybrid Pass 2 can bypass the tiered lexical safety gates.**
    - Code path: `graph_topology._resolve_orphans_v2()` calls `graph_orphan_hybrid.resolve_orphans_hybrid()` before `graph_lexical_v2.resolve_orphans_lexical_tiered()`.
    - Mismatch: `PLANNING_ORPHAN_RESOLUTION_V2.md` says Pass 2 processes remaining orphans with embeddings and Pass 3 handles orphans without embeddings. Current code says a missing embedding "does NOT abort" and degrades to pure FTS.
    - Impact: IDF/symbol-type guards can be skipped, and pure FTS links are persisted as `edge_class="semantic"` / `created_by="hybrid_rrf"`. This can reintroduce the common-symbol flooding the roadmap was designed to prevent.
    - Fix direction: if no vector branch exists for an orphan, return `[]` from hybrid and let Pass 3 lexical handle it, or explicitly run the same IDF/tier checks before accepting pure-FTS hybrid results.

2. **Phase 1c confidence weights are overwritten by Phase 2 edge weighting.**
    - Code path: `cross_language_linker.py` clamps weights to `[0.3, 0.8]`; `test_linker.py` emits weight `0.5`; `graph_topology.apply_edge_weights()` later rewrites every edge's `weight` based on structural in-degree and a synthetic floor.
    - Mismatch: `PLANNING_CROSS_LANGUAGE_GRAPH_FEDERATION.md` defines a distinct cross-language weight tier. Current Phase 2 does not preserve that tier.
    - Impact: `cross_language` and `test_link` are excluded from architectural projection, so clustering is protected, but persisted weights used by ask/research/federated expansion no longer represent linker confidence. Synthetic edges with target structural in-degree `0` can be rewritten above the intended cross-language ceiling.
    - Fix direction: preserve explicit weights for edge classes whose linkers already assign calibrated confidence, or add per-class caps/normalization in `apply_edge_weights()`.

3. **API surface extraction may not see rich parser code text.**
    - Code path: `api_surface_extractor.extract_api_surfaces()` reads `node_data["source_text"]`; rich symbol nodes in `graph_builder._process_file_symbols()` store the symbol object but do not expose `source_text` as a top-level node attribute.
    - Impact: Phase 1c L1 can be empty on Python/TypeScript/Java/Go rich-parser repos, even though those are the main languages where REST decorators matter. Phase 8 relatedness can also under-score API overlap if `api_surface` was never persisted.
    - Fix direction: add `source_text`, `docstring`, decorators/base classes, and any parser metadata needed by `api_surface_extractor` to rich graph node attrs, or teach the extractor to read from the stored symbol object.

4. **Phase 6 L0 is wired but has no producer.**
    - Code path: `FilesystemIndexer` now forwards `analysis.cross_language_relationships`; `EnhancedUnifiedGraphBuilder._detect_cross_language_relationships()` currently returns an empty list. In the default non-streaming path, `parse_results` is cleared before that helper and before `_generate_language_stats()`.
    - Impact: the L0 handoff is connected, but it cannot promote real parser-supplied relationships until the builder actually emits them before cleanup.
    - Fix direction: move cross-language relationship detection before parse-result cleanup or derive it from `unified_graph`; replace the stub with real parser signals (proto/FFI/decorator/import relationships).

5. **The `rel_path` node ID default is not propagated through every node creator/resolver.**
    - Code paths: rich symbol node IDs use rel-path slugs; doc nodes use `doc_lang::file_stem::symbol`; basic parser nodes use `language::file_stem::symbol`; `_resolve_source_node_sync()` and `_resolve_target_node_sync()` still construct legacy stem IDs for direct matches and fallbacks.
    - Impact: the default flip fixes one important collision class for rich symbols, but not the entire graph identity system. It can also create inferred fallback nodes with old-style IDs instead of resolving to real rel-path nodes.
    - Fix direction: centralize node-id construction in one helper and require all graph-builder paths (rich, basic, docs, relationship resolution, diagnostics) to use it. Add tests with same-stem files across rich/basic/doc inputs.

6. **Project graph node IDs are not namespaced by wiki.**
    - Code path: `project_recompute.recompute_project()` merges per-wiki graphs by original `node_id` and tags node attrs with `wiki_id`; `project_edges` store only original source/target IDs; `FederatedRetrieverStack` resolves target nodes by `target_node_id` through `MultiGraphQueryService.get_node()`.
    - Mismatch: `PLANNING_CROSS_LANGUAGE_GRAPH_FEDERATION.md` specified `{wiki_id}::{original_node_id}` project IDs.
    - Impact: two repos with identical `rel_path` and symbol names can collide in the merged graph, overwrite `wiki_id`, and make cross-repo expansion resolve the wrong node. This risk increases after the `rel_path` default because common layouts (`src/api/users.py::create_user`) become deterministic across repos.
    - Fix direction: namespace project graph nodes and edges at the project boundary. Store source/target wiki IDs as first-class edge columns or edge attrs, not only inside provenance.

7. **Project recompute does not clear stale rows.**
    - Code path: `project_recompute.recompute_project()` upserts relatedness and edges, then stamps `stale=False`; `ProjectStorageProtocol` has no clear/delete API for project edges or old pair scores.
    - Impact: removing a wiki, changing API surfaces, lowering relatedness, or deleting code can leave obsolete `project_edges` and `repo_relatedness` rows in storage. Lifecycle stale triggers can fire correctly while the recompute result remains contaminated by old rows.
    - Fix direction: add `clear_project_edges(project_id)` and `clear_repo_relatedness(project_id)` or transactional "replace recompute generation" semantics before writing fresh rows. Tests must cover wiki removal.

8. **Read-time recompute can enqueue duplicate background jobs.**
    - Code path: `routes.get_project()` calls `maybe_enqueue_recompute()` on each GET; `maybe_enqueue_recompute()` schedules `asyncio.create_task()` when stale but does not set an in-flight marker before returning.
    - Impact: repeated project reads can launch overlapping recomputes for the same project. This can amplify cost and make stale-row races worse.
    - Fix direction: persist `recompute_in_progress` / `recompute_started_at` or use an in-process keyed lock with storage-backed fallback. Clear it on success/error.

9. **Manual recompute authorization is weaker than the UI contract.**
    - Code path: `RecomputeWidget` only renders the start button for owners; `POST /projects/{project_id}/recompute` only requires an authenticated user and delegates accessibility to `ProjectService.list_project_wikis()` inside `recompute_project()`, which permits shared projects.
    - Impact: a non-owner who can view a shared project can likely trigger an expensive recompute by calling the API directly, despite the UI hiding the button.
    - Fix direction: enforce owner-only access in `recompute_project_route()` (or intentionally document shared-user recompute as allowed), then add route tests.

10. **Project storage contract and current design disagree.**
     - Code path: `project_storage.py` docstring still says SQLite/Postgres implementations are follow-up even though they exist; the planned project schema included rich node columns and source wiki IDs, but the current storage stores `attrs` JSON and recompute does not materialize nodes.
     - Impact: future implementers will not know whether `project_nodes` is a required materialized table or a vestigial protocol member. Federated retriever currently depends on per-wiki query services for node lookup, so the code acts as an edge/relatedness store rather than a full project graph store.
     - Fix direction: make a deliberate contract choice. Either materialize `project_nodes` and use them in query/retrieval, or remove/de-emphasize node materialization from the Phase 7 contract and document per-wiki lookup as the intended design.

11. **Federated retriever ignores configured dampening during construction.**
     - Code path: `multi_wiki_components.py` constructs `FederatedRetrieverStack(wiki_stacks)` without passing `flags.cross_repo_dampening`; the class falls back to `_DEFAULT_DAMPENING = 0.7`.
     - Impact: changing `WIKI_CROSS_REPO_DAMPENING` affects linker output but not retriever expansion scoring.
     - Fix direction: pass `dampening=float(flags.cross_repo_dampening)` when constructing the federated retriever and cover it with a unit test.

12. **Feature-flag rollout comments and defaults still disagree.**
     - Code path: many comments say "default off until ..." while the dataclass/env defaults are already `True` (`orphan_lexical_tiered`, `orphan_cascade_v2`, `orphan_hybrid_search`, `cross_language_linking`, `project_graph`, `federated_query`, `federated_retriever`, `cross_repo_linking`, `project_clustering`).
     - Impact: operators reading the code get a rollback story that no longer matches behavior. This matters because several features are only partially complete.
     - Fix direction: update comments to say which defaults are intentionally on for dogfood and which should be considered experimental, or temporarily flip high-risk features off until the P0 items above are fixed.

13. **The test front is still mostly unit-level.**
     - Current green command: `AUTH_ENABLED=false python -m pytest tests/unit/test_cross_language_linker.py tests/unit/test_cross_repo_linker.py tests/unit/test_project_recompute.py tests/unit/test_test_linker.py tests/unit/test_graph_clustering.py tests/unit/test_node_id_collision.py -q --tb=short` (79 passed in the last run).
     - Gap: no integration test currently proves actual indexing produces persisted `api_surface` rows, real `cross_language` edges, bounded test links, project recompute edge rows, or SSE payload compatibility with frontend consumers.
     - Fix direction: prioritize the integration tests listed in 0.3 and add frontend tests before enabling the feature set as release-ready.

#### 0.4.3 Code duplication and consolidation opportunities

These are not all correctness bugs, but they increase the chance that the roadmap diverges again.

| Duplication / drift | Where | Consolidation direction |
|---|---|---|
| API-surface pairing logic duplicated between `link_l1_api_surface()` and `link_api_surface_any_language()`. | `backend/app/core/code_graph/cross_language_linker.py` | Extract one internal `_link_api_surface_pairs(..., allow_same_language, edge_class, relationship_type, provenance_source)` helper. |
| Stale-marking storage open/write repeated. | `mark_projects_for_wiki_stale()` and `mark_project_stale()` in `project_recompute.py`, plus route hooks. | Add a private `_set_project_stale(store/project_id/reason)` helper; route hooks should call a service-layer function, not inline imports in every route. |
| SSE stream parsing duplicated for ask/research and recompute. | `web/src/spa/api/sse.ts` | Extract a generic `subscribeSSE(url, method, handlers)` parser that accepts event normalization callbacks. |
| Node ID construction scattered across rich parser, basic parser, doc nodes, source/target resolvers, and diagnostics. | `graph_builder.py` | Introduce a single `make_node_id(language, rel_path, file_name, qualified_name, style)` helper and use it everywhere. |
| Project graph design split between "materialized nodes" docs and "edge store plus per-wiki lookup" code. | `project_storage.py`, `federated_query_service.py`, `federated_retriever.py`, `multi_wiki_components.py` | Decide and document one contract. If per-wiki lookup is intended, simplify project storage protocol and make edge rows carry wiki IDs. |

#### 0.4.4 Recommended future-work front

**P0 - Correctness blockers from the second audit (closed unless section 0.5 says otherwise)**

1. Fix hybrid fallback semantics so missing embeddings do not bypass tiered lexical/IDF safeguards.
2. Preserve or cap Phase 1c linker weights in `apply_edge_weights()` instead of rewriting them as generic synthetic weights.
3. Expose rich parser `source_text` / decorator metadata to `api_surface_extractor`, then add a real Phase 1c integration test.
4. Replace `_detect_cross_language_relationships()` stub with real L0 producers or remove the L0 claim from the roadmap until producers exist.
5. Centralize node ID construction and propagate `rel_path` style to docs/basic/parser relationship resolution.
6. Namespace project-level nodes/edges with `wiki_id::node_id` or add source/target wiki IDs as first-class edge keys.
7. Clear/replace project edges and relatedness on recompute, especially after wiki removal.
8. Add an in-flight recompute guard and owner/permission tests for manual recompute.

**P1 - Verification and contract coverage**

1. Add `backend/tests/integration/test_phase1c_indexing.py` covering real `FilesystemIndexer` persistence of `api_surface`, `cross_language`, and bounded `test_link` edges.
2. Add `backend/tests/integration/test_project_recompute_edges.py` with two related repos whose node IDs intentionally collide before namespacing.
3. Add `backend/tests/integration/test_recompute_sse_contract.py` validating backend event names and payload shapes.
4. Add frontend `sse-recompute` and `RecomputeWidget` tests.
5. Add route tests for PATCH/add/remove stale marking and recompute owner policy.
6. Add storage tests for project edge cleanup/replacement semantics.

**P2 - Cleanup and rollout hygiene**

1. Update feature-flag comments/default policy so code comments match current behavior.
2. Update `project_storage.py` module docstring and Phase 7 roadmap text to match the chosen project-node strategy.
3. Pass `flags.cross_repo_dampening` into `FederatedRetrieverStack`.
4. Normalize imports in `ProjectPage.tsx` by moving `RecomputeWidget` into the import block before the `CodeMapTree` runtime declaration.
5. Refactor the duplicated API-surface matcher and SSE parser once tests lock behavior.

#### 0.4.5 Updated release-readiness conclusion

The second-audit P0 implementation mismatches were closed in the follow-up 2026-04-28 gap-closure pass and are covered by focused unit/storage regressions. The implementation is now suitable for dogfood on small real repositories, with the important caveat that the remaining release risk has shifted from core wiring to integration/UI verification and real-repo tuning.

Keep high-risk features observable and easy to roll back via feature flags until the P1 integration tests pass against actual indexed repositories.

### 0.5 Gap-closure implementation status — 2026-04-28

| Audit gap | Status | Implementation evidence | Remaining verification |
|---|---|---|---|
| Hybrid Pass 2 bypassed tiered lexical safety when embeddings were missing. | **Closed** | `graph_orphan_hybrid.resolve_orphans_hybrid()` now returns `[]` without cached/persisted embeddings and no longer calls `embed_fn`, so the cascade falls through to tiered lexical. | Add a full cascade integration test showing explicit -> hybrid -> lexical -> directory ordering on one fixture. |
| Phase 1c confidence weights were overwritten by Phase 2. | **Closed** | `graph_topology.apply_edge_weights()` preserves explicit weights for calibrated synthetic classes: `cross_language`, `test_link`, and `cross_repo`. | Keep storage-level tests for persisted weights when Phase 1c runs through `FilesystemIndexer`. |
| Rich parser nodes did not expose source text to API-surface extraction. | **Closed** | Rich graph nodes now expose top-level `source_text`, `docstring`, `parameters`, and `return_type`; `api_surface_extractor` also reuses existing `api_surface` and falls back to `symbol.source_text`. | Add real FastAPI/React indexing fixture to prove REST surfaces persist through storage. |
| Phase 6 L0 was wired but had no producer. | **Closed for parser-supplied `target_file` evidence** | `_detect_cross_language_relationships()` now runs before parse-result cleanup and emits L0 rows from parser relationships with cross-language `target_file`. | Broaden producers beyond `target_file` relationships, especially proto/gRPC, FFI, GraphQL, and generated-client conventions. |
| `rel_path` node IDs were not propagated consistently. | **Closed for graph-builder node creation/resolution paths** | Central `_make_node_id()` helper is used for rich, documentation, basic-parser, diagnostics, and source/target resolver paths; rel-path cache recovery prevents stem fallback. | Add one end-to-end same-stem repo fixture covering rich/basic/doc nodes together. |
| Project graph node IDs were not namespaced by wiki. | **Closed** | `make_project_node_id()` namespaces project graph nodes/edges as `wiki_id::raw_node_id`; recompute stores `raw_node_id` and provenance for both endpoints; federated query/retriever resolve namespaced IDs. | Run real multi-wiki project recompute on repos with intentionally colliding paths/symbols. |
| Project recompute did not clear stale rows. | **Closed** | `ProjectStorageProtocol` now has `clear_project_nodes`, `clear_project_edges`, and `clear_repo_relatedness`; memory/SQLite/Postgres implement them; recompute clears before rebuilding. | Add route/integration coverage for wiki removal followed by recompute. |
| Read-time/manual recompute lifecycle was incomplete. | **Partially closed** | Recompute writes a lightweight `recompute_in_progress` meta marker, clears it on normal completion/skip, marks stale lifecycle changes, and manual recompute is owner-gated in the route. | Replace the lightweight marker with a stronger lock/lease if concurrent server processes are expected; add route tests. |
| Project storage contract/documentation drifted from implementation. | **Closed** | `project_storage.py` docstring now reflects the in-memory/SQLite/Postgres implementations, and `recompute_project()` materializes project nodes. | Decide whether project nodes should become the primary lookup path or remain auxiliary metadata next to per-wiki query services. |
| Federated retriever ignored configured dampening. | **Closed** | `build_multi_wiki_components()` passes `flags.cross_repo_dampening` into `FederatedRetrieverStack`. | Add a component-level test for the builder path; class-level dampening tests already exist. |
| Cross-repo same-language API-surface edges had no cascade path. | **Closed** | `link_api_surface_any_language()` is wired into `build_cross_repo_edges(..., allow_same_language=True)` and filtered by `wiki_id`. | Validate on a same-language two-service project with shared REST surfaces. |
| Frontend/backend recompute contract lacks tests and generated API coverage. | **Open** | Backend and frontend wiring exist, but tests/types are still missing. | Add backend SSE contract tests, frontend `subscribeRecomputeSSE` tests, `RecomputeWidget` tests, and regenerate API types. |

The phase descriptions below remain useful for the original design intent, but sections 0.1–0.5 are the current implementation record.

---

## 1. Execution principles (apply to every phase)

1. **Dual-backend parity is mandatory.** Every protocol addition lands in `protocol.py`, `sqlite.py`, *and* `postgres.py` in the same PR. Tests run against both via `@pytest.mark.parametrize("backend", ["sqlite","postgres"])` in `backend/tests/storage/`.
2. **No schema migrations unless explicitly stated.** All new state lives in `repo_meta` (KV) until a phase explicitly introduces a column. Phases that introduce columns: Phase 1 (`provenance`), Phase 6 (`api_surface`), Phase 7 (project tables).
3. **Every behavioural change is feature-flagged** through `backend/app/core/feature_flags.py`. New flags default `False` for one release, then default `True` after the next phase's tests pass.
4. **Backwards compatibility:** old method names stay as thin shims. E.g. `search_fts5` keeps its current signature; new `search_fts_with_path` is additive.
5. **Test plan per phase** lives in `backend/tests/integration/` and `backend/tests/unit/code_graph/`. CI must stay green before merging.
6. **deepwiki back-port note** is included for every phase whose change is upstream-applicable. Phases 0–6 back-port; Phases 7–9 do not (deepwiki has no project graph).

---

## 2. Phase ordering and rationale

| Phase | Title | Closes gaps | Unblocks |
|---|---|---|---|
| 0 | Storage parity foundation | G1, G2 | every later phase |
| 1 | Phase-2 observability + safe quick wins | G3 | Phase 2/3 validation |
| 2 | IDF gating + tiered lexical T1–T4 + REST disambig | G4, G7 (partial) | Phase 4 RRF |
| 3 | Cascade reorder + explicit-ref Pass 1 + embedding reuse | G5, G6 | Phase 4 |
| 4 | Hybrid RRF + Matryoshka + perf | (perf only) | optional |
| 5 | Node disambiguation 3A/3B/3C | G7 | Phase 6, Phase 7 cross-repo |
| 6 | Intra-repo cross-language linker + new edge classes | G8 | deepwiki cross-language fix |
| 7 | Project graph storage + Federated query/retriever | G9 | Phase 8 |
| 8 | Cross-repo linker + relatedness + project Leiden | — | Phase 9 |
| 9 | Lifecycle/UX, SSE, flag rollout | — | release |

**Rationale for putting Phase 5 before Phase 6**: even though Phase 5 touches stable code, Phases 2, 3, and 6 all rely on `find_node_exact` / `_simple_name_index` to resolve a `Symbol` reference to one canonical node. If Phase 6 lands on the existing colliding node IDs, every cross-language edge is ambiguous and needs rework.

---

## Phase 0 — Storage parity foundation

**Goal:** make every backend speak the same language for FTS scoring, exact-match lookup, embedding retrieval, and batch KNN.

### 0.1 Protocol additions (`backend/app/core/storage/protocol.py`)

Add to `WikiStorageProtocol`:

```python
def count_fts_matches(self, query: str, *, exact_match: bool = False) -> int: ...

def search_fts_by_column(
    self,
    query: str,
    column: str,
    *,
    limit: int = 20,
    path_prefix: str | None = None,
    symbol_types: list[str] | None = None,
    exact_match: bool = False,
) -> list[dict[str, Any]]: ...

def search_fts_with_path(
    self,
    query: str,
    path_prefix: str,
    *,
    symbol_types: list[str] | None = None,
    exact_match: bool = False,
    limit: int = 20,
) -> list[dict[str, Any]]: ...

def get_embedding_by_id(self, node_id: str) -> list[float] | None: ...

def batch_similarity_search(
    self,
    embeddings: list[tuple[str, list[float]]],
    *,
    k: int = 5,
    path_prefix: str | None = None,
    distance_threshold: float = 0.15,
) -> dict[str, list[dict[str, Any]]]: ...
```

Each FTS-returning method returns dicts with **both** `fts_rank` (raw, backend-native sign) and `score_norm` (normalized to `[0,1]`, higher is better).

Normalization rules:

- **SQLite**: `score_norm = 1.0 / (1.0 + abs(fts_rank))` (BM25 negatives → bounded positive).
- **Postgres**: `score_norm = min(fts_rank / TS_RANK_CAP, 1.0)` with `TS_RANK_CAP = 1.0` (configurable via `WIKI_TS_RANK_CAP`).

### 0.2 SQLite implementation (`backend/app/core/storage/sqlite.py`)

- `count_fts_matches`: `SELECT count(*) FROM repo_fts WHERE repo_fts MATCH ?`. When `exact_match=True`, wrap `query` as `f'"{query.replace(chr(34), chr(34)*2)}"'` to force phrase match.
- `search_fts_by_column`: rewrite query as `f'{column}:{safe_query}'` (FTS5 column filter syntax) and reuse current `search_fts5` body.
- `search_fts_with_path`: thin wrapper enforcing non-null `path_prefix`.
- `get_embedding_by_id`: `SELECT embedding FROM repo_vec WHERE node_id = ?` then `_deserialize_float32_vec`.
- `batch_similarity_search`: loop calling existing `search_vec` per embedding (sqlite-vec has no batch KNN). Document that this is "logical batch only, physical N queries"; concurrency knob `WIKI_VEC_BATCH_CONCURRENCY` (default 4, ThreadPool).
- All FTS-returning methods add `score_norm = 1.0 / (1.0 + abs(fts_rank))` to each row dict. Existing `search_fts5` is also updated (additive — `fts_rank` stays).

### 0.3 PostgreSQL implementation (`backend/app/core/storage/postgres.py`)

- `count_fts_matches`: `SELECT count(*) FROM {schema}.repo_nodes WHERE fts_doc @@ websearch_to_tsquery('english', :q)`. `exact_match=True` → `phraseto_tsquery`.
- `search_fts_by_column`: build SELECT with on-the-fly tsvector `to_tsvector('english', coalesce(<column>,''))` (cleaner than maintaining N stored columns).
- `search_fts_with_path`: required `path_prefix` → `LIKE :path || '/%'`.
- `get_embedding_by_id`: `SELECT embedding FROM {schema}.repo_vec WHERE node_id = :nid` returning `pgvector` → list[float].
- `batch_similarity_search`: single SQL using `LATERAL` and a `VALUES` subquery — pgvector batches in one round-trip.
  ```sql
  WITH q(nid, emb) AS (VALUES (:nid_1, :emb_1::vector), ...)
  SELECT q.nid AS query_nid, r.node_id, (r.embedding <=> q.emb) AS vec_distance
  FROM q
  JOIN LATERAL (
      SELECT node_id, embedding FROM {schema}.repo_vec
      ORDER BY embedding <=> q.emb LIMIT :k
  ) r ON true
  WHERE (r.embedding <=> q.emb) < :threshold
  ```
- Add `ts_rank_cd(..., 32)` (length-normalization bit flag) gated by `WIKI_PG_TS_RANK_NORM=1` (default `1`). `0` reproduces current behaviour for rollback.
- All FTS-returning methods add `score_norm = min(fts_rank / TS_RANK_CAP, 1.0)`.

### 0.4 Tests

- `backend/tests/storage/test_storage_parity.py` — parametrize backend, assert every new method returns identical result *shape* and `score_norm` ranges.
- `backend/tests/storage/test_score_normalization.py` — synthetic doc with known phrase, both backends produce `score_norm > 0.5` for exact match, `> 0` for partial match.

### 0.5 Feature flags

```python
WIKI_TS_RANK_CAP = float                          # default 1.0
WIKI_PG_TS_RANK_NORM = bool                       # default True
WIKI_VEC_BATCH_CONCURRENCY = int                  # default 4 (sqlite path only)
```

### 0.6 deepwiki back-port

Apply `score_norm` normalization to deepwiki's `unified_db.py` (SQLite-only).

---

## Phase 1 — Phase-2 observability + safe quick wins

**Goal:** measurable Phase 2 + dedup, before any behavioural reorder.

### 1.1 `Phase2Stats` dataclass + `explain_connectivity`

New file `backend/app/core/graph_topology_diagnostics.py`:

```python
from dataclasses import dataclass, field
from typing import Any

@dataclass
class Phase2Stats:
    components_before_orphan: int = 0
    components_after_orphan: int = 0
    components_after_doc: int = 0
    components_after_bridge: int = 0
    edges_by_class: dict[str, int] = field(default_factory=dict)
    edges_by_provenance: dict[str, int] = field(default_factory=dict)
    orphan_resolution: dict[str, Any] = field(default_factory=dict)
    doc_edges: dict[str, Any] = field(default_factory=dict)
    bridging: dict[str, Any] = field(default_factory=dict)
    weighting: dict[str, Any] = field(default_factory=dict)
    hubs: dict[str, Any] = field(default_factory=dict)


def explain_connectivity(db, G) -> Phase2Stats: ...
```

Hook into `run_phase2` (`backend/app/core/graph_topology.py`) to snapshot `nx.number_weakly_connected_components(G)` after each step and write the dict to `db.set_meta("phase2_stats_v2", stats)`.

### 1.2 Per-edge `provenance`

Extend `_add_edge` to accept `provenance: dict | None` and store it as JSON on the edge attribute and (when `skip_db=False`) in a new `repo_edges.provenance` column.

**Schema migration required** (single column, NULL default — no backfill):

- SQLite: `ALTER TABLE repo_edges ADD COLUMN provenance TEXT`.
- Postgres: `ALTER TABLE {schema}.repo_edges ADD COLUMN IF NOT EXISTS provenance JSONB`.

Plumb through `upsert_edge`, `upsert_edges_batch`, `persist_weights_to_db`.

### 1.3 Edge dedup

New helper in `graph_topology.py`:

```python
def _has_typed_edge(G: nx.MultiDiGraph, u: str, v: str, rel_type: str) -> bool:
    if not G.has_edge(u, v):
        return False
    return any(
        d.get("relationship_type") == rel_type
        for d in G[u][v].values()
    )
```

Replace every `_add_edge` callsite where dedup matters. This eliminates the local `seen` set in `inject_doc_edges` and protects every other pass.

### 1.4 Stopword + low-value query gate

In `graph_topology.py`:

```python
_LEXICAL_STOPWORDS = frozenset({
    "init", "main", "test", "tests", "setup", "config", "utils", "helper",
    "helpers", "common", "base", "abstract", "interface", "impl", "data",
    "model", "service", "factory", "manager", "handler", "controller",
    "view", "router", "routes", "api", "lib", "src", "app", "core",
})
_MIN_FTS_QUERY_LEN = 4
```

Apply in Pass 1 before any FTS call.

### 1.5 BM25 / `ts_rank` threshold

Add `_FTS_MIN_SCORE_NORM = 0.15` (configurable via `WIKI_FTS_MIN_SCORE_NORM`). Drop hits below it.

### 1.6 Tests

- `backend/tests/integration/test_phase2_observability.py` — small synthetic graph, run `run_phase2`, assert `Phase2Stats` is populated and stored in `repo_meta`.
- `backend/tests/integration/test_edge_dedup.py` — re-running `inject_doc_edges` does not double edge counts.

### 1.7 Feature flags

```python
WIKI_PHASE2_OBSERVABILITY = bool       # default True (cheap)
WIKI_EDGE_DEDUP = bool                 # default True
WIKI_FTS_MIN_SCORE_NORM = float        # default 0.15
WIKI_FTS_STOPWORD_GATE = bool          # default True
```

### 1.8 deepwiki back-port

100% applicable. Phase2Stats + stopword gate ports as-is.

---

## Phase 2 — IDF gating + tiered lexical (T1–T4) + REST disambiguation

**Goal:** prevent generic names like `User`, `API`, `Service` from spraying lexical edges across the whole repo.

### 2.1 IDF computation

```python
def _compute_symbol_idf(db, symbol_name: str) -> float:
    """IDF = log(N_total / (1 + N_matches))."""
    total = db.node_count()
    matches = db.count_fts_matches(symbol_name, exact_match=True)
    return math.log(total / (1 + matches))
```

Eligibility per `symbol_type`:

| symbol_type | Tiers eligible |
|---|---|
| module, class, interface, struct, enum, trait, rest_endpoint | T1, T2, T3, T4 |
| function | T1, T2, T3 |
| method | T1, T2 |
| constructor, field, variable, parameter, constant | none (skip lexical entirely) |

IDF thresholds:

| IDF | Allowed tiers |
|---|---|
| `≥ 3.0` | all eligible |
| `1.5 – 3.0` | T1, T2 only |
| `< 1.5` | skip lexical (rely on hybrid Pass 2) |

### 2.2 Tiered lexical cascade

Rewrite Pass 1 of `resolve_orphans` as `_resolve_orphans_lexical_tiered(db, G, orphan_id) -> list[hit]`:

- **T1** — exact in `symbol_name` + same dir + architectural target types.
- **T2** — exact in `symbol_name` + parent dir.
- **T3** — full FTS local.
- **T4** — global, requires `len(name) ≥ 8` AND `IDF ≥ 3.0`.

```python
ARCH_TARGETS = {"module","class","interface","function","method",
                "constant","module_doc","file_doc","rest_endpoint"}
```

Cap `max_lexical_edges = 2`. Stop at first non-empty tier.

### 2.3 REST endpoint disambiguation

```python
GENERIC_CLASS_NAMES = frozenset({"API","Resource","View","ViewSet","Endpoint",
                                  "Handler","Controller","Service","Manager",
                                  "Mixin","Base","Abstract","Interface"})

REST_DECORATORS = {"@app.route","@router.get","@router.post","@api_view",
                   "@route","@get","@post","@put","@delete","@patch",
                   "@RestController","@RequestMapping","@GetMapping",
                   "@PostMapping","@expose"}

REST_BASE_CLASSES = {"APIView","Resource","ViewSet","HTTPEndpoint","MethodView",
                     "Controller","RouteHandler"}

def _detect_rest_endpoint(node_data: dict) -> bool: ...
def _compute_fts_symbol_name(symbol_name: str, file_stem: str) -> str: ...
```

When `symbol_name in GENERIC_CLASS_NAMES` AND `_detect_rest_endpoint(node)`:

1. Set `node["symbol_type"] = "rest_endpoint"` in graph attrs only.
2. For lexical search, replace the FTS query with `_compute_fts_symbol_name(symbol_name, file_stem)` (e.g. `"users API"` from class `API` in `users.py`).

### 2.4 Tests

- `backend/tests/unit/code_graph/test_lexical_tiers.py`.
- `backend/tests/unit/code_graph/test_rest_endpoint_detection.py`.
- `backend/tests/integration/test_orphan_resolution_v2_lexical.py`.

### 2.5 Feature flags

```python
WIKI_ORPHAN_LEXICAL_TIERED = bool      # default False initially → True after Phase 3
WIKI_ORPHAN_LEXICAL_IDF_GATE = bool    # default True
WIKI_ORPHAN_REST_DISAMBIG = bool       # default True
```

### 2.6 deepwiki back-port

100% applicable.

---

## Phase 3 — Cascade reorder + explicit-ref Pass 1 + embedding reuse

**Goal:** put high-precision signals first, eliminate redundant embedding calls.

### 3.1 New cascade order in `resolve_orphans`

```
Pass 1: _resolve_orphans_by_explicit_refs   (NEW — markdown links + backticks + imports)
Pass 2: _resolve_orphans_hybrid             (Phase 4 — RRF FTS+Vec)
Pass 3: _resolve_orphans_lexical_tiered     (was Pass 1; from Phase 2)
Pass 4: _resolve_orphans_by_directory       (existing)
```

`inject_doc_edges` Tier-1 (markdown link + backtick) logic moves into Pass 1. `inject_doc_edges` keeps only Tier-2 proximity and runs in its current slot AFTER `resolve_orphans`.

### 3.2 `_resolve_orphans_by_explicit_refs`

Sources:

1. Markdown `[text](rel/path)` — for doc orphans, parse `source_text` with `_MD_LINK_RE`, resolve via `_normalize_path` + `path_index`. Edge `relationship_type="explicit_ref"`, `edge_class="doc"`, weight 0.95.
2. Inline `` `Symbol` `` backticks — resolve via `_simple_name_index` (Phase 5 adds qualified-name index).
3. Imports — for code orphans, the parser-emitted `imports` list (verify column presence; if not, defer to Phase 5).

All hits get `provenance={"source":"explicit_ref","matcher":<sub-source>,"raw_score":0.95}`.

### 3.3 Embedding reuse

Add `_collect_orphan_embeddings(db, orphan_ids) -> dict[str, list[float] | None]`:

```python
def _collect_orphan_embeddings(db, orphan_ids):
    out = {}
    CHUNK = 500
    for i in range(0, len(orphan_ids), CHUNK):
        batch = orphan_ids[i:i+CHUNK]
        for nid in batch:
            out[nid] = db.get_embedding_by_id(nid)
    return out
```

Pass 2 (hybrid) and any future embedding-needing pass first checks this dict before falling back to `embed_fn(text)`. This eliminates Phase 1b ↔ Phase 2 double-embedding.

### 3.4 Tests

- `backend/tests/unit/code_graph/test_explicit_refs.py`.
- `backend/tests/integration/test_orphan_cascade_v2.py`.
- `backend/tests/integration/test_embedding_reuse.py`.

### 3.5 Feature flags

```python
WIKI_ORPHAN_CASCADE_V2 = bool          # default False until tests stable
WIKI_ORPHAN_REUSE_EMBEDDINGS = bool    # default True (cheap, safe)
```

### 3.6 deepwiki back-port

Pass 1 (explicit refs) + embedding reuse port directly.

---

## Phase 4 — Hybrid RRF + Matryoshka + perf

**Goal:** make Pass 2 a real hybrid (FTS + Vec via RRF) and shrink embedding cost.

### 4.1 `_resolve_orphans_hybrid`

```python
def _resolve_orphans_hybrid(db, G, orphan_id, *, k=60, threshold=0.02,
                            top_n=20, alpha=0.5, strategy="rrf",
                            orphan_embeddings=None):
    name = ...
    local_dir = ...
    fts_hits = db.search_fts_with_path(name, path_prefix=local_dir, limit=top_n)
    emb = (orphan_embeddings or {}).get(orphan_id) or db.get_embedding_by_id(orphan_id)
    vec_hits = db.search_vec(emb, k=top_n, path_prefix=local_dir) if emb else []
    fused = _rrf_fuse(fts_hits, vec_hits, k=k)
    return [h for h in fused if h["rrf_score"] >= threshold][:top_n]
```

`_rrf_fuse(lists, k) -> list[Hit]` implements `score = Σ 1 / (k + rank_i)`. `score_norm` from Phase 0 makes FTS ranks comparable across backends — required input.

### 4.2 Matryoshka 256d truncation (optional)

`WIKI_ORPHAN_MATRYOSKA_DIM = 256` (default `0` = disabled). When set, Pass 2 truncates the orphan's embedding to first N dims and queries a separate `repo_vec_short` table (built on demand).

### 4.3 Quantization (Postgres only)

When `WIKI_PG_VEC_QUANTIZE=halfvec`, use pgvector's `halfvec` type for the short table. SQLite has no equivalent — flag is no-op there.

### 4.4 Async I/O

Existing `WIKI_VEC_CONCURRENCY` plumbing. Recommend SQLite=4, Postgres=8.

### 4.5 Tests

- `backend/tests/unit/code_graph/test_rrf_fusion.py`.
- `backend/tests/integration/test_phase2_perf.py` (mark `slow`).

### 4.6 Feature flags

```python
WIKI_ORPHAN_HYBRID_SEARCH = bool        # default False initially
WIKI_ORPHAN_RRF_K = int                 # default 60
WIKI_ORPHAN_RRF_THRESHOLD = float       # default 0.02
WIKI_ORPHAN_HYBRID_TOP_N = int          # default 20
WIKI_ORPHAN_HYBRID_ALPHA = float        # default 0.5
WIKI_ORPHAN_HYBRID_STRATEGY = str       # "rrf" | "linear" — default "rrf"
WIKI_ORPHAN_MATRYOSKA_DIM = int         # default 0 (off)
WIKI_PG_VEC_QUANTIZE = str              # default "" (off)
```

### 4.7 deepwiki back-port

RRF fusion + threshold logic ports directly. Matryoshka and halfvec do not.

---

## Phase 5 — Node disambiguation 3A / 3B / 3C

**Goal:** stop silent collisions between same-stem files in different directories.

### 5.1 Action 3A — rel_path-based node IDs

Change `backend/app/core/code_graph/graph_builder.py` `_process_file_symbols`:

```python
if FeatureFlags.node_id_style() == "rel_path":
    safe_path = rel_path.replace("/", "__").replace(".", "_")
    node_id = f"{language}::{safe_path}::{local_qualified_name}"
else:
    node_id = f"{language}::{file_name}::{local_qualified_name}"
```

`WIKI_NODE_ID_STYLE = "stem" | "rel_path"`, default `"stem"` for one release, then `"rel_path"`. Not retroactive — existing wikis stay on stem until rebuild.

Hash-suffix branch is removed in `rel_path` mode.

### 5.2 Action 3B — qualified `_simple_name_index`

In `attach_graph_indexes`:

```python
graph._qualified_name_index: dict[str, list[str]]   # "Parent.symbol" -> [node_id...]
graph._fqn_index:           dict[str, str]          # "rel_path::Parent.symbol" -> node_id
```

Existing `_simple_name_index` stays. `find_node_exact` and explicit-ref Pass 1 prefer FQN → qualified → simple.

### 5.3 Action 3C — decorator-aware naming (optional)

For TS classes annotated with `@Component`, use the component selector as a name hint. Already feasible — TS parser captures decorator args. Java/C#/Go file follow-ups per language.

### 5.4 Tests

- `backend/tests/unit/code_graph/test_node_id_collision.py`.
- `backend/tests/unit/code_graph/test_qualified_name_index.py`.

### 5.5 Feature flags

```python
WIKI_NODE_ID_STYLE = str               # "stem" | "rel_path", default "stem" → "rel_path" next release
WIKI_QUALIFIED_NAME_INDEX = bool       # default True
```

### 5.6 deepwiki back-port

Critical for deepwiki — same collision bug.

---

## Phase 6 — Intra-repo cross-language linker + new edge classes

**Goal:** wire the existing `cross_language_relationships` (and richer API-surface signals) into the graph as first-class weighted edges.

### 6.1 New edge classes

In `backend/app/core/graph_topology.py`:

```python
_SYNTHETIC_CLASSES = frozenset({
    "directory", "lexical", "semantic", "doc", "bridge",
    "cross_language",   # NEW
    "test_link",        # NEW
})
```

Both classes:

- Excluded from structural in-degree.
- Floored at `SYNTHETIC_WEIGHT_FLOOR = 0.5`.
- Excluded from architectural projection in clustering: pass `exclude_edge_classes={"cross_language","test_link"}` to graph clustering's view builder.

### 6.2 New file `backend/app/core/code_graph/api_surface_extractor.py`

```python
from typing import TypedDict

class APISurface(TypedDict):
    kind: str            # "rest" | "grpc" | "graphql" | "ffi" | "wasm" | "bdd" | "cli"
    surface: str         # canonical key, e.g. "POST /api/users"
    weight_hint: float   # base confidence
    metadata: dict

def extract_api_surfaces(node_data: dict, parser_metadata: dict) -> list[APISurface]: ...
```

Per-language matchers in `backend/app/core/code_graph/api_matchers/`:

- `rest_python.py`, `rest_typescript.py`, `rest_java.py`
- `grpc.py`, `graphql.py`
- `ffi.py` (`extern "C"`, `ctypes`, JNI, P/Invoke, wasm-bindgen)
- `bdd.py` (Gherkin step → step-definition)
- `cli.py` (argparse/click/cobra)

Add new column `repo_nodes.api_surface` (JSONB Postgres / TEXT SQLite). Single-column ALTER, NULL default. Matchers run during Phase 1 graph build (new sub-phase **Phase 1c**, after parsing, before Phase 1b embeddings).

### 6.3 New file `backend/app/core/code_graph/cross_language_linker.py`

4-level cascade:

- **L0** — Exact name + signature: existing `_detect_cross_language_relationships` output, promoted to graph edges with `edge_class="cross_language"`.
- **L1** — API surface: pair nodes whose `api_surface[].surface` strings match. Weight = `0.7 × locality × specificity` (specificity = 1 / log(1 + N_matches)).
- **L2** — Hybrid FTS + Vec RRF restricted to other-language candidates. Weight = `0.5 × rrf_score`.
- **L3** — Containment traversal + member-uses promotion. Weight = `parent_weight × 0.6`.

Final formula:

```
weight = base_confidence × locality_factor × specificity_factor   (clamped to [0.3, 0.8])
```

Edges: `relationship_type="cross_language_<L0|L1|L2|L3>"`, `edge_class="cross_language"`, `provenance={"source":"cross_language_linker","level":...,"matcher":...}`.

### 6.4 Test linker

Detect test↔code pairs: same-stem heuristic, shared decorators, test-framework imports. Edge `relationship_type="test_link"`, `edge_class="test_link"`, weight 0.5.

### 6.5 Pipeline integration

Insert **Phase 1c** after `_process_file_symbols` and before `populate_embeddings`:

```python
extract_api_surfaces_for_graph(graph, db)
run_cross_language_linker(db, graph)
run_test_linker(db, graph)
```

All gated by `WIKI_CROSS_LANGUAGE_LINKING` (default `False`).

### 6.6 Tests

- `backend/tests/unit/code_graph/test_api_surface_extractor.py`.
- `backend/tests/integration/test_cross_language_linker.py`.
- `backend/tests/integration/test_test_link.py`.

### 6.7 Feature flags

```python
WIKI_CROSS_LANGUAGE_LINKING = bool     # default False
WIKI_TEST_LINKER = bool                # default False
WIKI_API_SURFACE_EXTRACTION = bool     # default True (cheap, populates column)
```

### 6.8 deepwiki back-port

This is **the** fix the user asked for upstream. deepwiki already has `cross_language_relationships` in its `RepositoryGraph` but doesn't promote them. Port L0+L1, plus test linker and `_SYNTHETIC_CLASSES` membership.

---

## Phase 7 — Project graph storage + Federated query/retriever

### 7.1 New file `backend/app/core/storage/project_storage.py`

```python
class ProjectStorageProtocol(Protocol):
    def upsert_project_node(...): ...
    def upsert_project_edge(...): ...
    def get_project_nodes(project_id: str) -> list[dict]: ...
    def get_project_edges(project_id: str) -> list[dict]: ...
    def upsert_repo_relatedness(project_id, wiki_a, wiki_b, score, breakdown): ...
    def get_repo_relatedness(project_id) -> list[dict]: ...
    def set_project_meta(project_id, key, value): ...
    def get_project_meta(project_id, key) -> Any: ...
```

Schema (per backend):

```sql
project_nodes(project_id, node_id PK, wiki_id, language, symbol_type, symbol_name, ...)
project_edges(project_id, source_node_id, target_node_id, edge_class, weight, provenance)
repo_relatedness(project_id, wiki_a, wiki_b, score, breakdown JSONB, computed_at)
project_meta(project_id, key, value JSONB)
```

Backends:

- **SQLite**: separate `project.db` per project. Cross-wiki search uses **materialized copy** (FTS5 doesn't span ATTACHed DBs; ATTACH limit 10).
- **PostgreSQL**: dedicated schema `project_<project_id>`. Cross-schema `INSERT ... SELECT`.

### 7.2 New file `backend/app/core/code_graph/federated_query_service.py`

Drop-in replacement for `MultiGraphQueryService`. Same surface plus:

- `cross_repo_edges(node_id) -> list[dict]`
- `relatedness(wiki_a, wiki_b) -> float`

Factory `build_federated_components(project_id, wikis)` replaces `build_multi_wiki_components()`.

### 7.3 New file `backend/app/core/federated_retriever.py`

Drop-in replacement for `MultiWikiRetrieverStack`:

1. Per-wiki retrieval (existing).
2. Expand candidate set by 1 hop along `project_edges` where `edge_class="cross_repo"`.
3. Cross-repo dampening: hits multiplied by `0.7`.

### 7.4 Routes

`wiki_service` consumes `MultiGraphQueryService` if `WIKI_FEDERATED_QUERY=False`, else `FederatedQueryService`. Single switch, no new endpoints.

### 7.5 Tests

- `backend/tests/storage/test_project_storage_parity.py`.
- `backend/tests/integration/test_federated_query.py`.

### 7.6 Feature flags

```python
WIKI_PROJECT_GRAPH = bool              # default False
WIKI_FEDERATED_QUERY = bool            # default False
WIKI_FEDERATED_RETRIEVER = bool        # default False
WIKI_PROJECT_GRAPH_DB_PATH = str       # SQLite-only: project.db location
```

### 7.7 deepwiki back-port

Not applicable.

---

## Phase 8 — Cross-repo linker + relatedness gating + project Leiden

### 8.1 New file `backend/app/core/code_graph/relatedness_scorer.py`

```python
def score_repo_pair(wiki_a, wiki_b, db_a, db_b) -> tuple[float, dict]:
    lang     = jaccard(languages(wiki_a), languages(wiki_b))
    dep      = jaccard(deps(wiki_a),      deps(wiki_b))
    api      = api_surface_overlap(db_a, db_b)
    naming   = top_symbol_name_jaccard(db_a, db_b, top_k=200)
    score = 0.3*lang + 0.2*dep + 0.3*api + 0.2*naming
    return score, {"lang":lang,"dep":dep,"api":api,"naming":naming}
```

Threshold `≥ 0.15`. Pre-filter: if `len(languages(a) ∩ languages(b)) == 0 AND api_surface_overlap == 0`, skip pair. Cap at 50 wikis via `WIKI_PROJECT_MAX_WIKIS`.

### 8.2 New file `backend/app/core/code_graph/cross_repo_linker.py`

For each wiki pair `(a, b)` with `relatedness ≥ 0.15`:

1. Reuse Phase 6's `cross_language_linker` cascade L0–L2 across the pair.
2. Emit edges into `project_edges` with `edge_class="cross_repo"` and `weight = base_weight × 0.7 × relatedness_score`.

### 8.3 Project-level Leiden

`WIKI_PROJECT_CLUSTERING=True` runs Leiden over `project_nodes` + `project_edges`. Communities written to `project_nodes.macro_cluster`.

### 8.4 Tests

- `backend/tests/unit/code_graph/test_relatedness_scorer.py`.
- `backend/tests/integration/test_cross_repo_linker.py`.

### 8.5 Feature flags

```python
WIKI_CROSS_REPO_LINKING = bool         # default False
WIKI_PROJECT_CLUSTERING = bool         # default False
WIKI_PROJECT_MAX_WIKIS = int           # default 50
WIKI_CROSS_REPO_DAMPENING = float      # default 0.7
WIKI_RELATEDNESS_THRESHOLD = float     # default 0.15
```

### 8.6 deepwiki back-port

Not applicable.

---

## Phase 9 — Lifecycle, UX, SSE, flag rollout

### 9.1 SSE events

New event types in `app/events.py`:

- `project_relatedness_progress`
- `cross_repo_linker_progress`
- `project_clustering_progress`

### 9.2 Triggers

- After any wiki rebuild in a project, enqueue project-level recompute gated by `WIKI_PROJECT_GRAPH=True`.
- After project metadata change, full project recompute.

### 9.3 Staleness

`project_meta.staleness = max(wiki_last_built across project)`. Stale if `> 7 days` since project recompute or any constituent wiki rebuilt after last project recompute.

### 9.4 Flag rollout schedule

| Phase | Default `False` for | Flip `True` after |
|---|---|---|
| 0 | n/a (additive only) | merge |
| 1 | n/a (observability is on by default) | merge |
| 2 | `WIKI_ORPHAN_LEXICAL_TIERED` | Phase 3 tests green |
| 3 | `WIKI_ORPHAN_CASCADE_V2` | one release of telemetry |
| 4 | `WIKI_ORPHAN_HYBRID_SEARCH` | perf tests green |
| 5 | `WIKI_NODE_ID_STYLE="rel_path"` | one release |
| 6 | `WIKI_CROSS_LANGUAGE_LINKING`, `WIKI_TEST_LINKER` | sample-repo validation |
| 7 | `WIKI_PROJECT_GRAPH`, `WIKI_FEDERATED_QUERY`, `WIKI_FEDERATED_RETRIEVER` | dogfood project |
| 8 | `WIKI_CROSS_REPO_LINKING`, `WIKI_PROJECT_CLUSTERING` | dogfood + perf |

---

## 3. Backend parity matrix

| Concern | SQLite | PostgreSQL |
|---|---|---|
| FTS engine | FTS5 BM25 (negative scores) | tsvector + ts_rank_cd (positive) |
| Score normalization | `1 / (1 + abs(rank))` | `min(rank/cap, 1.0)` |
| Exact-match FTS | Phrase quoting `"x"` | `phraseto_tsquery` |
| Per-column FTS | `column:term` syntax | on-the-fly `to_tsvector(coalesce(col,''))` |
| Path prefix | `GLOB 'prefix/*'` | `LIKE 'prefix/%'` |
| Vector engine | sqlite-vec | pgvector |
| Batch KNN | logical loop (concurrency knob) | single SQL with `LATERAL` |
| Quantization | none | `halfvec` (Phase 4) |
| Project graph storage | separate `project.db` (materialized copy) | dedicated schema `project_<id>` |
| Schema migrations | `repo_edges.provenance TEXT` (Phase 1); `repo_nodes.api_surface TEXT` (Phase 6); project schema (Phase 7) | same in JSONB |

Future backends implement `WikiStorageProtocol` + `ProjectStorageProtocol`. Nothing outside `storage/` is backend-aware after Phase 0.

---

## 4. Test plan summary

| Layer | New files |
|---|---|
| Storage parity | `test_storage_parity.py`, `test_score_normalization.py`, `test_project_storage_parity.py` |
| Phase-2 unit | `test_lexical_tiers.py`, `test_rest_endpoint_detection.py`, `test_explicit_refs.py`, `test_rrf_fusion.py`, `test_node_id_collision.py`, `test_qualified_name_index.py`, `test_relatedness_scorer.py` |
| Phase-2 integration | `test_phase2_observability.py`, `test_edge_dedup.py`, `test_orphan_resolution_v2_lexical.py`, `test_orphan_cascade_v2.py`, `test_embedding_reuse.py`, `test_phase2_perf.py` |
| Cross-language | `test_cross_language_linker.py`, `test_test_link.py`, `test_cross_repo_linker.py` |
| Federation | `test_federated_query.py` |
| Per-matcher | `test_api_surface_extractor.py` |

CI: extend existing `pytest tests/ -v`. Mark perf tests `@pytest.mark.slow`, run in nightly job.

---

## 5. Risk register

| Risk | Mitigation |
|---|---|
| Phase 5 node-ID flip breaks existing wikis | Behind `WIKI_NODE_ID_STYLE` flag; old wikis keep stem IDs until rebuilt. |
| Phase 0 score normalization changes downstream ranking thresholds | Re-expressed in `score_norm` units in Phase 1. |
| Phase 6 cross_language edges pollute Leiden communities | Excluded from architectural projection by `_SYNTHETIC_CLASSES`. |
| Phase 7 project.db materialization explodes on many wikis | Cap 50 wikis; stream INSERT in 5k-row batches; nightly VACUUM. |
| pgvector `halfvec` not on managed Postgres ≤14 | Default off; fall-through to full vector. |
| Decorator REST detection misses Java/Go | Documented; per-language follow-up tickets. |
| Schema migrations on production DBs | All new columns NULL default; additive only. |

---

## 6. PR sequencing

```
PR-1  Phase 0  storage protocol + parity (no behavioural change)
PR-2  Phase 1  observability + dedup + stopword
PR-3  Phase 2  IDF + tiered lexical + REST disambig (flag default False)
PR-4  Phase 3  cascade reorder + explicit-ref Pass 1 + embedding reuse
PR-5  Phase 4  RRF + perf
PR-6  Phase 5  node disambiguation
PR-7  Phase 6  cross-language linker + new edge classes
PR-8  Phase 7  project storage + federated services
PR-9  Phase 8  cross-repo linker + relatedness + project Leiden
PR-10 Phase 9  lifecycle/UX
```

Every PR is independently reviewable, feature-flagged off by default, individually revertable.

---

## 7. Sanity check before starting

```bash
cd /Users/romanmitusov/Documents/Work/wikis
git fetch origin && git checkout -b feat/graph-quality-roadmap origin/main
source backend/.venv/bin/activate
AUTH_ENABLED=false pytest backend/tests/ -v
```

Expected: green. Any failure here pauses the roadmap.
