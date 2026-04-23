# DeepWiki ↔ Wikis Storage & Pipeline Parity Audit

> **Branch**: `feat/unified-db-and-clustering`
> **Date**: 2025-04-15
> **Status**: ✅ Audit Complete — Action Items Pending

---

## Table of Contents

1. [Critical Bugs (Data Loss)](#1-critical-bugs-data-loss)
2. [Raw SQL Leaks](#2-raw-sql-leaks)
3. [GLOB vs LIKE Semantic Mismatch](#3-glob-vs-like-semantic-mismatch)
4. [FTS Scoring Differences](#4-fts-scoring-differences)
5. [Three Parallel FTS Classes](#5-three-parallel-fts-classes)
6. [Protocol Method Gaps](#6-protocol-method-gaps)
7. [Graph Enrichment Pipeline Parity](#7-graph-enrichment-pipeline-parity)
8. [Graph Creation & Parsing Parity](#8-graph-creation--parsing-parity)
9. [Augmentation & Declaration-Heavy Detection](#9-augmentation--declaration-heavy-detection)
10. [Header/Namespace Splitting](#10-headernamespace-splitting)
11. [wiki_graph_optimized.py Parity](#11-wiki_graph_optimizedpy-parity)
12. [Clustering Pipeline Parity](#12-clustering-pipeline-parity)
13. [Expansion Engine Parity](#13-expansion-engine-parity)
14. [Topology Enrichment Parity](#14-topology-enrichment-parity)
15. [Retriever & Search Parity](#15-retriever--search-parity)
16. [Action Items](#16-action-items)

---

## 1. Critical Bugs (Data Loss)

### BUG-1: `persist_weights_to_db` loses all edge `rel_type` values

**Both SQLite and Postgres affected.**

`graph_topology.py:617` builds edge dicts with key `"relationship_type"`, but
`unified_db.py:602` and `postgres.py:636` read `e.get("rel_type", "")`.
After Phase 2 enrichment, **all edges get `rel_type = ""`** — losing calls,
inheritance, composition, etc.

Compare: `unified_db.py:1151` (`_nx_edge_to_dict`) correctly maps
`data.get("relationship_type") → rel_type`, but `persist_weights_to_db` bypasses
this helper.

**Fix**: In `persist_weights_to_db`, use `"rel_type"` key instead of
`"relationship_type"`, matching what `upsert_edges_batch` expects.

### BUG-2: `_ConnCompat.execute` silently rolls back DML (Postgres only)

`postgres.py:129` uses `with engine.connect()` which starts a transaction but
**never calls `conn.commit()`**. The `with` block auto-rolls-back on exit
(SQLAlchemy 2.x behavior).

Impact: `persist_weights_to_db` calls `db.conn.execute("DELETE FROM repo_edges")`
— this DELETE is **silently lost** on Postgres.

**Fix**: Use `engine.begin()` instead of `engine.connect()` for DML statements,
or add explicit `conn.commit()` before exit.

### BUG-3: GLOB vs LIKE semantic mismatch

| Behavior | SQLite `GLOB prefix*` | Postgres `LIKE prefix/%` |
|----------|----------------------|--------------------------|
| `src/foo.py` | ✅ matches | ✅ matches |
| `src` (exact prefix) | ✅ matches `src*` | ❌ not matched |
| `srcOther/a.py` | ✅ matches `src*` | ❌ not matched |

Postgres is more correct (only matches path descendants). Used in 3 places:
`get_nodes_by_path_prefix`, `search_fts` path filter, `search_vec` path filter.

**Fix**: Normalize both to `prefix || '%'` (match prefix itself and descendants)
or `prefix || '/%'` (descendants only). Choose one semantic and apply consistently.

---

## 2. Raw SQL Leaks

12 locations across 6 files bypass the WikiStorageProtocol:

| File | Count | Functions | Impact |
|------|-------|-----------|--------|
| `shared_expansion.py` | 5 | `_get_outgoing`, `_get_incoming`, `_fetch_node`, `resolve_alias_chain_db` | All take raw `conn`, not protocol |
| `research_tools.py` | 5 | FTS search, relationship lookup, symbol resolve | Opens own UnifiedWikiDB, uses `db.conn` |
| `graph_topology.py` | 3 | `flag_hubs_in_db`, `persist_weights_to_db` | `db.conn.execute()`, `db.conn.commit()` |
| `language_heuristics.py` | 1 | `detect_dominant_language` | Raw SQL GROUP BY |
| `cluster_expansion.py` | 2 | Passes `db.conn` to shared_expansion | Bridge only |
| `filesystem_indexer.py` | 1 | Terminal `udb.conn.commit()` | No-op on Postgres |

All use `?` placeholders (SQLite style). `_ConnCompat` translates these for
Postgres but has the transaction bug (BUG-2).

**Fix**: Add missing protocol methods (`delete_all_edges`, `get_edge_targets`,
`get_edge_sources`, `detect_dominant_language`) and refactor all callers.

---

## 3. GLOB vs LIKE Semantic Mismatch

See BUG-3 above. Three affected protocol methods:
- `get_nodes_by_path_prefix`
- `search_fts` (path_prefix filter)
- `search_vec` (path_prefix filter)

---

## 4. FTS Scoring Differences

| Feature | SQLite (FTS5) | Postgres (tsvector) |
|---------|--------------|---------------------|
| Tokenizer | `unicode61` | `english` (stems words) |
| Query syntax | FTS5 `MATCH` (OR tokens) | `websearch_to_tsquery` |
| BM25 weights | symbol=10, tokens=8, docstring=3, content=1 | ABCD: content=A(0.1), docstring=B(0.4), symbol=C(0.8), tokens=D(1.0) |
| Ranking | `bm25(repo_fts, 0, 10, 8, 3, 1)` | `ts_rank_cd(fts_doc, query, 32)` |

**Weights are inverted**: SQLite gives `symbol_name` highest weight (10), Postgres
`ts_rank_cd` ABCD gives `symbol_name` weight D=1.0 and `content` weight A=0.1.
Search results rank differently.

**Fix**: Align Postgres tsvector weights to match SQLite BM25 priority:
symbol_name > name_tokens > docstring > content.

---

## 5. Three Parallel FTS Classes

| Class | Backend | Location | Status |
|-------|---------|----------|--------|
| `GraphTextIndex` | SQLite `.fts5.db` | `graph_text_index.py` | Legacy standalone |
| `UnifiedGraphTextIndex` | SQLite `.wiki.db` | `unified_graph_text_index.py` | SQLite shim |
| `PostgresGraphTextIndex` | Delegates to protocol | `postgres_graph_text_index.py` | Postgres shim |

All expose the same interface. The protocol already has `search_fts`,
`search_fts_by_symbol_name`.

**Fix**: Collapse all three into protocol methods. Any consumer that needs a
"text index" should receive a `WikiStorageProtocol` instance directly.

---

## 6. Protocol Method Gaps

| Needed method | Used by | Current workaround |
|---------------|---------|-------------------|
| `delete_all_edges()` | `persist_weights_to_db` | `db.conn.execute("DELETE FROM repo_edges")` |
| `get_edge_targets(node_id)` | `shared_expansion._get_outgoing` | Raw SQL |
| `get_edge_sources(node_id)` | `shared_expansion._get_incoming` | Raw SQL |
| `detect_dominant_language(node_ids)` | `language_heuristics` | Raw SQL GROUP BY |
| `get_node(node_id)` — already exists | `shared_expansion._fetch_node` | Raw SQL |
| `execute(sql, params)` | Misc raw callers | `_ConnCompat` shim |

`get_edge_targets` and `get_edge_sources` exist on `UnifiedWikiDB` (lines
1386-1397) but are NOT on the protocol.

---

## 7. Graph Enrichment Pipeline Parity ✅

Both follow identical order:

1. `from_networkx()` — load graph
2. `populate_embeddings()` — vector index
3. Phase 2 (`run_phase2`) — topology enrichment
4. Phase 3 (`run_clustering`) — Leiden clustering
5. `set_clusters_batch()` — persist cluster assignments
6. `set_meta()` — persist metadata

---

## 8. Graph Creation & Parsing Parity

### 8.1 Constants — ✅ IDENTICAL
- `ARCHITECTURAL_SYMBOLS`: same 9 code symbols + DOC_SYMBOL_TYPES union
- `DOC_SYMBOL_TYPES`: same 20 members
- All classification constants (`CLASS_LIKE_TYPES`, `FUNCTION_LIKE_TYPES`, etc.): identical

### 8.2 Node Schema — ✅ IDENTICAL (with minor platform-specific columns)
Both have the same 23 columns. Differences:
- DeepWiki has `indexed_at TEXT DEFAULT datetime('now')` — **MISSING from Wikis**
- Wikis has `fts_doc` / `fts_vector` tsvector columns — Postgres-specific (expected)

### 8.3 Edge Schema — ✅ IDENTICAL
Same 13 columns: `id, source_id, target_id, rel_type, edge_class, analysis_level,
weight, raw_similarity, source_file, target_file, language, annotations, created_by`

### 8.4 `_nx_edge_to_dict` — ✅ IDENTICAL
Both correctly map `data.get("relationship_type")` → `rel_type`.

### 8.5 `from_networkx()` — ⚠️ DIFFERENCES

| Aspect | DeepWiki | Wikis | Impact |
|--------|----------|-------|--------|
| Batch size | 5000 | 2000 | **Wikis slower on large graphs** |
| Stale cleanup ordering | Between node & edge import | After edge import | May cause brief FK violations |
| FTS rebuild | Explicit `_populate_fts5()` | Postgres trigger auto-rebuilds | Architectural |
| `to_networkx` source tag | `"unified_db"` | `"postgres"` | Cosmetic |

### 8.6 Rich Parsers — ❌ C# MISSING

| Language | DeepWiki | Wikis |
|----------|----------|-------|
| C++ | ✅ CppEnhancedParser | ✅ CppEnhancedParser |
| **C#** | ✅ **CSharpVisitorParser** | ❌ **MISSING** (basic only) |
| Go | ✅ GoVisitorParser | ✅ GoVisitorParser |
| Java | ✅ JavaVisitorParser | ✅ JavaVisitorParser |
| Python | ✅ PythonParser | ✅ PythonParser |
| JavaScript | ✅ JavaScriptVisitorParser | ✅ JavaScriptVisitorParser |
| TypeScript | ✅ TypeScriptEnhancedParser | ✅ TypeScriptEnhancedParser |
| Rust | ✅ RustVisitorParser | ✅ RustVisitorParser |

**C# gets only basic (tree-sitter level) parsing in Wikis vs comprehensive
CSharpVisitorParser in DeepWiki.**

### 8.7 `is_test_path()` Regex — ⚠️ CASE SENSITIVITY

DeepWiki PascalCase pattern uses `re.IGNORECASE` (matches `footests.java`).
Wikis does NOT — only matches `FooTests.java`. All other 16 patterns identical.

### 8.8 Missing Constants in Wikis
- `DEFAULT_EMBEDDING_DIM = 1536` — not in Wikis `constants.py`
- `EMBEDDING_BATCH_SIZE = 64` — not in Wikis `constants.py`
- `indexed_at` column timestamp — not in Wikis schema

---

## 9. Augmentation & Declaration-Heavy Detection

### 9.1 ContentExpander — ❌ MAJOR GAP

DeepWiki has a **2600-line `content_expander.py`** with 25+ methods for rich
NX-graph-based content enrichment. Wikis replaces it with a **no-op shim**:

```python
class _ContentExpanderShim:
    def expand_retrieved_documents(self, docs):
        return docs  # No-op
```

**Features MISSING from Wikis (in ContentExpander)**:

| Feature | DeepWiki Function | Impact |
|---------|-------------------|--------|
| Class comprehensive expansion | `_expand_class_comprehensively()` | Inheritance, composition, constructors, key methods |
| C++ struct expansion | `_expand_cpp_struct_comprehensively()` | Struct + method impls from .cpp |
| Go struct expansion | `_expand_go_struct_comprehensively()` | Struct + receiver methods |
| C++ constant augmentation | `_augment_cpp_constant_with_definition()` | `extern const` → actual value from .cpp |
| C++ function decl+impl merge | `_augment_cpp_function_with_implementation()` | Single function declaration + implementation merge |
| Type alias chain expansion | `_expand_type_alias_comprehensively()` | Follows `alias_of` chains multi-hop |
| Constant expansion | `_expand_constant_comprehensively()` | Constants → initialization functions |
| Multi-hop inheritance | `_find_inheritance_context()` | Class hierarchy traversal |
| Composition traversal | `_find_composition_context()` | 2-hop composition relationships |
| Method parameter types | `_expand_class_method_parameter_types()` | Type references from signatures |
| Class callers | `_find_functions_using_class()` | Functions that call/use a class |
| Constructor→class resolution | `_find_instantiated_class()` | Resolves `new Class()` calls |

### 9.2 Declaration-Heavy File Detection

Neither codebase computes a declaration-heavy ratio/threshold metric. Both use
**language-based heuristic hints** via `LanguageHints`:

| Field | DeepWiki | Wikis |
|-------|----------|-------|
| `declaration_extensions` | `.h`, `.hpp`, `.d.ts` | Same values |
| `prefer_impl_over_decl` | Per-language flag | Same |
| `impl_augmentation_priority` | C++=2.0, Rust=1.8, Go=1.5, JS=0.8 | Same |

**Difference**: DeepWiki's `should_include_in_expansion()` has a stub check for
`prefer_impl_over_decl + declaration_extensions` (no-op `pass`). Wikis stripped
even this stub — only checks `skip_symbol_types`.

### 9.3 Cross-File Augmentation (cluster_expansion.py) — ⚠️ DIFFERENCES

**C++ augmentation (`_augment_cpp`)**:

| Aspect | DeepWiki | Wikis |
|--------|----------|-------|
| SQL approach | Raw SQL on repo_edges + repo_nodes | `db.find_related_nodes()` abstraction |
| Header detection | By symbol type | By file extension (`.h`, `.hpp`, `.hxx`, `.hh`) |
| 2-hop class→method→impl | YES (`class→defines→method←defines_body←impl`) | NO — 1-hop only via `find_related_nodes()` |
| Impl→Declaration direction | NOT done | **YES** — bidirectional augmentation |
| Return style | Returns `List[str]` parts | Mutates `doc.page_content` in-place |

**Go/Rust augmentation (`_augment_go_rust`)**:

| Aspect | DeepWiki | Wikis |
|--------|----------|-------|
| Edge types | `'defines'` | `'defines_body', 'implementation'` — **DIFFERENT EDGES** |
| SQL approach | Raw SQL | `db.find_related_nodes()` |

### 9.4 Feature Flag Gating

| Feature | DeepWiki | Wikis |
|---------|----------|-------|
| Smart expansion | Behind `flags.smart_expansion` | Hardcoded `True` |
| Language hints | Behind `flags.language_hints` | Always enabled |
| Hierarchical Leiden | Behind `flags.hierarchical_leiden` | Always enabled |

---

## 10. Header/Namespace Splitting

### `_file_internal_split()` — ❌ MISSING FROM WIKIS

DeepWiki's `graph_clustering.py:1025` has `_file_internal_split()`:
1. Groups symbols by `parent_symbol` or namespace scope
2. If scopes produce ≥2 groups → split within file
3. Fallback: split by `start_line` order

This handles **header-only libraries** (e.g., `stb_image.h`, `nlohmann/json.hpp`)
and **large single-file modules**.

**Wikis has NO equivalent.** The clustering pipeline only does Leiden sections +
consolidation, without any within-file scope-based splitting.

### Related Missing DeepWiki Features
- `_file_aware_chunk()` — splits oversized clusters respecting file boundaries
- `_recursive_split()` — recursive Leiden splitting for oversized clusters
- `_merge_micro_to_cap()` — merges undersized clusters

---

## 11. wiki_graph_optimized.py Parity

### 11.1 Constructor Differences

| Parameter | DeepWiki | Wikis |
|-----------|----------|-------|
| `alita_client` | ✅ Pylon platform client | ❌ MISSING |
| `llm_low` | ❌ MISSING | ✅ Tiered LLM for cheap tasks |
| `progress_callback` | ❌ MISSING (uses Pylon emitters) | ✅ SSE streaming |
| `planner_type` | ❌ MISSING | ✅ Configurable planner |
| `exclude_tests` | ❌ MISSING | ✅ Test symbol exclusion |

### 11.2 Quality Assessment & Enhancement — ❌ MISSING FROM WIKIS

DeepWiki has a full quality assessment pipeline (commented out but code present):

| Feature | DeepWiki | Wikis |
|---------|----------|-------|
| `dispatch_quality_assessment()` | ✅ Parallel QA via LangGraph `Send` | ❌ REMOVED |
| `assess_page_quality()` | ✅ Uses `QUALITY_ASSESSMENT_PROMPT` | ❌ REMOVED |
| `dispatch_enhancement()` | ✅ Re-generation for failing pages | ❌ REMOVED |
| `enhance_page_content()` | ✅ Uses `ENHANCED_RETRY_CONTENT_PROMPT` | ❌ REMOVED |
| QA state fields | `quality_assessments`, `enhanced_pages`, `retry_counts` | ❌ NOT in state |

### 11.3 Generation Modes — ❌ 2 MODES MISSING

DeepWiki has 3 content generation modes:
1. `_generate_simple()` — V3_TONE_ADJUSTED ✅ (both have)
2. `_generate_agentic()` — AgenticDocGeneratorV2 with planning + tool calling ❌ MISSING
3. `_generate_with_continuation()` — Hierarchical multi-iteration ❌ MISSING

**Also missing from Wikis**: `_parse_need_context_requests()`,
`_strip_need_context_markers()`, `_fetch_requested_items()`,
`_format_fetched_documents()`, `AgenticDocGeneratorV2` import.

### 11.4 Prompt Variants — 9 MISSING

DeepWiki imports 12 prompt variants. Wikis only uses `V3_TONE_ADJUSTED`.
Missing: `QUALITY_ASSESSMENT_PROMPT`, `ENHANCED_RETRY_CONTENT_PROMPT`,
`COMPRESSED_V3`, `V2`, `OLD`, `COMPRESSED`, `V3`, `V3_REORDERED`, `V5`,
`V3_WITH_CONTINUATION`.

### 11.5 Error Handling — Wikis is MORE Defensive

| Scenario | DeepWiki | Wikis |
|----------|----------|-------|
| `analyze_repository` failure | Returns `errors` only | Returns errors + **fallback minimal analysis** |
| `generate_wiki_structure` failure | Silent error dict | **Raises `RuntimeError`** |
| Missing spec in dispatch | Returns `[]` silently | **Raises `RuntimeError`** with explanation |
| `export_wiki` null guard | No null check | **Raises `RuntimeError`** |
| Context overflow | Generic exception | `is_context_overflow(e)` → user-friendly message |

### 11.6 `_ranked_truncation` Return Type

DeepWiki returns `List[Any]` (just docs). Wikis returns
`tuple(included_docs, token_metrics)` with `docs_retrieved`, `docs_included`,
`docs_dropped`, `tokens_before`, `tokens_after`, `utilization_pct`.

### 11.7 `_ensure_unified_db` — Wikis-only Feature

Wikis has `_ensure_unified_db()` (L1537–1625): tries pre-built DB from indexer,
falls back to creating in-memory from NX graph, integrates with Postgres backend.
DeepWiki requires pre-built DB.

---

## 12. Clustering Pipeline Parity

### 12.1 Core Algorithm — ✅ IDENTICAL
- File-level contraction (`_contract_to_file_graph`)
- Section-level Leiden at `LEIDEN_FILE_SECTION_RESOLUTION = 1.0`
- Page-level per-section Leiden at `LEIDEN_PAGE_RESOLUTION = 1.0`
- Target formulas (`target_section_count`, `target_total_pages`) — identical
- `select_central_symbols()` — identical PageRank logic
- Consolidation passes (`_consolidate_sections`, `_consolidate_pages`) — identical
- Orphan/isolated file handling — identical

### 12.2 Component Bridging — ❌ MISSING FROM WIKIS

DeepWiki has `_bridge_components()` with `BRIDGE_WEIGHT = 0.1` — connects
disconnected components using directory-similarity before Leiden to prevent
one-community-per-component fragmentation.

### 12.3 Legacy Louvain Pipeline — ❌ MISSING FROM WIKIS (intentionally removed)

DeepWiki retains full legacy pipeline (feature-flagged): `macro_cluster()`,
`micro_cluster()`, `apply_page_sizing()`, `_recursive_split()`,
`_file_aware_chunk()`, `_file_internal_split()`, `_merge_micro_to_cap()`,
InfoMap micro-clustering, `auto_resolution()`.

### 12.4 Hub Detection Approach — ⚠️ DIFFERENT

| Aspect | DeepWiki | Wikis |
|--------|----------|-------|
| When computed | Phase 2 (pre-computed, loaded from DB) | Inline during clustering |
| Function | `_load_hubs_from_db()` reads `is_hub=1` | `detect_hubs()` computes Z-score inline |

### 12.5 Cluster Metadata Return — ⚠️ DIFFERENT

| Aspect | DeepWiki | Wikis |
|--------|----------|-------|
| Return type | Raw `Dict[str, Any]` | `ClusterAssignment` dataclass |
| DB persistence | `persist_clusters()` writes to DB | Returns in-memory (caller persists) |

### 12.6 Doc-Cluster Detection — ⚠️ DIFFERENT

| Aspect | DeepWiki | Wikis |
|--------|----------|-------|
| Node lookup | `db.get_node(nid)` checking `is_doc` flag | `G.nodes.get(nid)` checking `symbol_type in DOC_SYMBOL_TYPES` |

### 12.7 `_adaptive_max_page_size()` — ❌ MISSING FROM WIKIS
DeepWiki scales max page size with repo complexity.

---

## 13. Expansion Engine Parity

### 13.1 Per-Symbol-Type Strategies — ✅ IDENTICAL
All 6 strategy types (class, function, constant, type_alias, macro, generic)
and their per-direction budgets/limits are identical in `shared_expansion.py`.

### 13.2 Test Node Filtering — ⚠️ WIKIS IS MORE ROBUST
- DeepWiki: checks only `node.get("is_test")`
- Wikis: checks `node.get("is_test")` AND `is_test_path(rel_path)` as fallback

### 13.3 `_node_to_document()` — ⚠️ DIFFERENCES

| Aspect | DeepWiki | Wikis |
|--------|----------|-------|
| Default `is_initial` | `True` | `False` |
| Fallback content | None (uses `source_text` only) | `signature + docstring` fallback |
| Metadata fields | 17 fields (extra: file_name, chunk_type, docstring, signature, is_architectural, is_doc) | 11 fields |

### 13.4 Feature Flag Gating — ⚠️ DIFFERENT
DeepWiki gates smart_expansion and language_hints behind feature flags.
Wikis hardcodes both to enabled.

### 13.5 Budget Constants — ✅ IDENTICAL
`MAX_EXPANSION_TOTAL=200`, `MAX_NEIGHBORS_PER_SYMBOL=15`,
`DEFAULT_TOKEN_BUDGET=50000`, `_CHARS_PER_TOKEN=3.5`

### 13.6 NX-Graph Expansion Engine (`expansion_engine.py`)

Both have the same core functions: `augment_cpp_node()`, `augment_go_node()`,
`_find_defines_body_impl()`. Wikis adds several 2-hop helpers:
`find_composed_types()`, `find_creates_from_methods()`,
`find_calls_to_free_functions()`, `format_type_args()`.

---

## 14. Topology Enrichment Parity

### 14.1 `run_phase2()` Steps

| Step | DeepWiki | Wikis | Status |
|------|----------|-------|--------|
| 1. resolve_orphans | ✅ | ✅ | ≈ Identical |
| 2. inject_doc_edges | ✅ | ✅ | ⚠️ Differences |
| 3. bridge_components | ✅ | ✅ | ⚠️ Differences |
| 4. apply_edge_weights | ✅ | ✅ | ≈ Identical |
| 5. detect_hubs + flag | ✅ | ✅ | ≈ Identical |
| 6. persist_weights | ✅ | ✅ | ⚠️ BUG-1 |
| 7. set_meta (step 7) | ❌ | ✅ | Wikis adds explicit metadata step |

### 14.2 Orphan Resolution — ❌ MAJOR DIFFERENCE

| Aspect | DeepWiki | Wikis |
|--------|----------|-------|
| Vector edges per orphan | ALL hits below threshold (1-3 edges) | **Only 1 best hit** |
| Node lookup | `db.get_node()` from DB | `G.nodes.get()` from memory |

**DeepWiki resolves orphans more aggressively** by adding multiple vector
similarity edges. Wikis is conservative (single best match).

### 14.3 Doc Edge Injection — ⚠️ DIFFERENCES

| Aspect | DeepWiki | Wikis |
|--------|----------|-------|
| `_DOC_KINDS` | 8 kinds | 9 kinds (adds `"markdown_document"`) |
| `_enrich_graph_nodes_from_db()` | ✅ pre-populates missing attrs | ❌ MISSING |
| Backtick reference fallback | `rsplit(".", 1)[-1]` segment match | No segment fallback |
| Graph node attr access | Direct `data.get("rel_path")` | Nested `data.get("location", {}).get("rel_path")` OR direct |

### 14.4 Bridge Edge Detection — ⚠️ DIFFERENCES

| Aspect | DeepWiki | Wikis |
|--------|----------|-------|
| Representative selection | `in_degree + out_degree` | `G.degree()` (functionally equiv for DiGraph) |
| Stats | `components_before`, `components_after`, `bridges_added` | `components_before`, `bridges_added` (NO `components_after`) |

### 14.5 Weight Formula — ✅ IDENTICAL
`weight = 1 / log(structural_in_degree(v) + 2)` with `SYNTHETIC_WEIGHT_FLOOR=0.5`.

### 14.6 Stats Richness — ⚠️ WIKIS LESS DETAILED
`apply_edge_weights()`: DeepWiki returns `edges_weighted, synthetic_floored, min,
max, mean`. Wikis returns only `edges_weighted`.

---

## 15. Retriever & Search Parity

### 15.1 Architecture — ❌ MAJOR DIFFERENCE

| Component | DeepWiki | Wikis |
|-----------|----------|-------|
| `WikiRetrieverStack` | ✅ Full legacy stack | ❌ REMOVED |
| `VectorStoreManager` (FAISS) | ✅ Full FAISS integration | ❌ REMOVED |
| `UnifiedRetriever` | ✅ Feature-flagged (OFF by default) | ✅ Always enabled |
| `bm25_disk.py` | ✅ Disk-backed mmap BM25 | ❌ REMOVED (FTS5 replaces BM25) |

### 15.2 Legacy Retriever Stack — ❌ MISSING FROM WIKIS

DeepWiki's `WikiRetrieverStack`:
- FAISS dense retriever (`k=25`)
- Disk-backed mmap BM25 (`k=25`)
- LangChain `EnsembleRetriever` (weights: `dense=0.6, bm25=0.4`)
- Cross-encoder reranker (`ms-marco-MiniLM-L-6-v2`)
- `WebRetriever` (Tavily search integration)
- Semantic doc retrieval with `EmbeddingsFilter`
- `ContentExpander` integration for graph-based expansion

### 15.3 `UnifiedRetriever` — ✅ FUNCTIONALLY IDENTICAL
Same constants (`FTS_POOL=30`, `VEC_POOL=30`), same `_expand_documents()`,
same `search_repository()`, same `search_docs_semantic()`.

### 15.4 Hybrid Search (RRF) — ✅ IDENTICAL
`fts_weight=0.4`, `vec_weight=0.6`, same RRF formula in both
`search_hybrid()` implementations.

### 15.5 `GraphTextIndex` (Standalone FTS5) — ✅ FUNCTIONALLY IDENTICAL
Same BM25 weights `(10.0, 8.0, 3.0, 1.0)`, same tokenizer, same query builder,
same all search methods.

---

## 16. Structure Validation Phases

### 16.1 Overview — ❌ 3 FILES MISSING FROM WIKIS

DeepWiki has opt-in validation for the cluster structure planner pipeline:

| File | Lines | Purpose | Feature Flag |
|------|-------|---------|-------------|
| `candidate_builder.py` | ~190 | `CandidateRecord` dataclass + `build_candidates()` + classification | `capability_validation` |
| `page_validator.py` | ~340 | 5-check quality pipeline per candidate page | `capability_validation` |
| `coverage_ledger.py` | ~330 | Symbol/doc/directory coverage tracking + LLM refiner | `coverage_ledger` |

### 16.2 Candidate Builder

`build_candidates()` converts raw cluster_map into `CandidateRecord` objects with:
- `classification`: `code`, `mixed`, `docs`, `bridge`
- `code_identity_score`: fraction of `PAGE_IDENTITY_SYMBOLS`
- `implementation_evidence`: fraction with non-trivial source text
- `docs_dominance`: fraction of doc nodes
- `public_api_presence`: fraction without leading `_`
- `utility_contamination`: fraction matching "util", "helper", "misc" etc.

### 16.3 Page Validator — 5-Check Pipeline

| Check | Function | Validates | Shape Action on Failure |
|-------|----------|-----------|------------------------|
| 1. Identity | `_check_identity()` | Code-identity score ≥ 0.1 | `demote` / `merge_with` |
| 2. Coherence | `_check_coherence()` | Utility contamination < 0.5, node count < max | `split_by` |
| 3. Grounding | `_check_grounding()` | Implementation evidence ≥ 0.2 | `merge_with` |
| 4. Coverage | `_check_coverage()` | ≥ 2 nodes minimum | `merge_with` |
| 5. Shape | `_check_shape()` | Classification-specific (docs→promote, bridge→split) | `promote_docs` / `split_by` |

Shape decisions: `keep`, `merge_with`, `split_by`, `promote_docs`, `demote`

Applied in `cluster_planner.py`:
1. Pass 1 — Remove DEMOTE candidates
2. Pass 2 — Merge MERGE_WITH candidates into their largest sibling

### 16.4 Coverage Ledger

`CoverageReport` tracks:
- Symbol coverage: `covered_symbols / total_symbols` (priority ≥ 7)
- Doc domain coverage: by directory
- Directory coverage: all source dirs
- Page overlap pairs: candidates sharing > 50% node overlap
- LLM refiner infrastructure (not yet invoked)

### 16.5 Validation in Main Agent

| Aspect | DeepWiki | Wikis |
|--------|----------|-------|
| `_validate_structure()` (log-only) | ✅ In `structure_engine.py` | ✅ IDENTICAL |
| Candidate validation (shape actions) | ✅ In `cluster_planner.py` | ❌ MISSING |
| Coverage ledger | ✅ In `cluster_planner.py` | ❌ MISSING |
| Both gated behind feature flags | Default: OFF | N/A |

---

## 17. Expansion Engine Deep Gaps (Research Round 2)

### 17.1 `augment_rust_node()` — ❌ MISSING FROM WIKIS

DeepWiki's `expansion_engine.py` has `augment_rust_node()` (~60 lines) that
enriches Rust struct/enum/trait nodes with cross-file impl methods via `defines`
edges. **Entirely absent from Wikis** — both the function AND the call site in
`expand_smart()`.

### 17.2 Go/Rust DB Augmentation Edge Types — ❌ WRONG IN WIKIS

| Aspect | DeepWiki | Wikis |
|--------|----------|-------|
| Edge types | `'defines'` | `'defines_body', 'implementation'` |
| Symbol types | `struct, class, enum, trait` | `struct, interface, trait, enum` (no `class`) |

Go receiver methods and Rust impl methods connect via `defines` edges. Wikis
searches for `defines_body` and `implementation` — **zero results** for Go/Rust.

### 17.3 `_node_to_document()` — 7 Missing Metadata Fields

| Field | DeepWiki | Wikis |
|-------|----------|-------|
| `rel_path` | ✅ | ❌ |
| `file_name` | ✅ | ❌ |
| `chunk_type` | ✅ | ❌ |
| `docstring` | ✅ | ❌ |
| `signature` | ✅ | ❌ |
| `is_architectural` | ✅ | ❌ |
| `is_doc` / `is_documentation` | ✅ | ❌ |

### 17.4 C++ Class 2-hop in DB Path — ⚠️ POTENTIALLY BROKEN

Wikis uses `db.find_related_nodes(node_id, ['defines_body'], direction='in')`
for header→impl augmentation. This finds direct `defines_body` edges TO the class,
but misses the 2-hop path: `class →[defines]→ method ←[defines_body]← impl`.
Need to verify `find_related_nodes()` implementation.

### 17.5 `shared_expansion.py` — ✅ FUNCTIONALLY IDENTICAL

Per-symbol-type strategies, budgets, directions, edge types all identical.
Only difference is explicit `exclude_tests` param (Wikis) vs feature flags (DeepWiki).

---

## 18. Scoping Decisions

The following DeepWiki features are **intentionally excluded** from Wikis:

| Feature | Reason |
|---------|--------|
| Legacy Louvain fallback | Hierarchical Leiden is the only path |
| `ContentExpander` (2600 lines) | NX-graph expansion_engine.py handles deepagents; shared_expansion handles clusters |
| `_file_internal_split()` | Not needed with current Leiden approach |
| `WikiRetrieverStack` (FAISS+BM25+reranker) | UnifiedRetriever with FTS5/tsvector replaces it |
| Quality assessment pipeline | Not needed for current generation approach |
| Enhancement/retry pipeline | Not needed |
| `_generate_agentic()` mode | Not needed |
| `_generate_with_continuation()` mode | Not needed |
| 9 legacy prompt variants | V3_TONE_ADJUSTED is the only prompt |
| `_adaptive_max_page_size()` | Not needed |
| `_file_aware_chunk()` | Not needed |

---

## 19. Action Items (Revised)

### Priority 1 — Critical Bugs (Data Loss) ✅ DONE
- [x] BUG-1: Fix `rel_type` key in `persist_weights_to_db` (both backends)
- [x] BUG-2: Fix `_ConnCompat.execute` to use `engine.begin()` for DML (Postgres)
- [x] BUG-3: Normalize GLOB/LIKE path prefix semantics
- [x] BUG-4: Fix `relationship_type` column name in `research_tools.py`

### Priority 2 — Protocol Unification (eliminate all raw SQL) ✅ DONE
- [x] Add `delete_all_edges()` to protocol + both backends
- [x] `get_edge_targets()` / `get_edge_sources()` already on protocol
- [x] Add `detect_dominant_language()` to protocol + both backends
- [x] Refactor `shared_expansion.py`: take `db: WikiStorageProtocol` not raw `conn`
- [x] Refactor `graph_topology.py`: use protocol methods, not `db.conn`
- [x] Refactor `research_tools.py`: use protocol methods, not `db.conn`
- [x] `language_heuristics.py` standalone function kept for test compat; production uses protocol
- [x] Refactor `cluster_expansion.py`: pass protocol, not `db.conn`
- [x] Remove `filesystem_indexer.py` raw `udb.conn.commit()` → `udb.commit()`
- [ ] Collapse 3 FTS classes into protocol methods
- [ ] Remove `_ConnCompat` shim after all raw SQL eliminated

### Priority 3 — Scoring & Ranking Parity ✅ DONE
- [x] Align FTS5 BM25 weights: `bm25(repo_fts, 10.0, 4.0, 2.0, 1.0)` matching Postgres ratio (A=1.0, B=0.4, C=0.2, D=0.1)
- [x] Unify `from_networkx` batch size: Postgres changed from 2000 → 5000 (matching SQLite/DeepWiki)

### Priority 4 — Expansion Bug Fixes (CRITICAL) ✅ DONE
- [x] Fix `_augment_go_rust` edge types: changed to `['defines']` (matching DeepWiki parsers)
- [x] Fix `_augment_go_rust` symbol types: added `'class'` + `target_symbol_types` filter + grouped output by file
- [x] Rust augmentation: Go/Rust share same path; `augment_rust_node()` handled by `_augment_go_rust()`
- [x] Add 7 missing metadata fields to `_node_to_document()`: rel_path, file_name, chunk_type, docstring, signature, is_architectural, is_doc/is_documentation
- [x] C++ class 2-hop augmentation: rewrote `_augment_cpp()` with symbol-type dispatch — Class→defines→Method←defines_body←Impl

### Priority 5 — Parser Parity ✅ DONE
- [x] Copy CSharpVisitorParser from DeepWiki (2,012 lines) → registered in graph_builder.py
- [x] `is_test_path` PascalCase regex is intentional (inline comment: "PascalCase only — no IGNORECASE" to prevent false positives like "contests.py")

### Priority 6 — Structure Validation Phases ✅ DONE
- [x] Port `candidate_builder.py` (~195 lines) — adapted SQL→protocol (`db.get_nodes_by_ids()`)
- [x] Port `page_validator.py` (~250 lines) — 5-check quality pipeline (identity, coherence, grounding, coverage, shape)
- [x] Port `coverage_ledger.py` (~240 lines) — symbol/doc/dir coverage + LLM refiner (adapted `db.get_architectural_nodes()`)
- [x] Wire validation into `cluster_planner.py` behind env var flags (`WIKIS_CANDIDATE_VALIDATION=1`, `WIKIS_COVERAGE_LEDGER=1`)
- [x] Added optional `db` parameter to `ClusterStructurePlanner.__init__()` + passed from `wiki_graph_optimized.py`

### Priority 7 — Topology Enrichment Parity ✅ DONE
- [x] Orphan resolution: ALL vector hits now added (loop over `valid_hits` instead of single `valid_hits[0]`)
- [x] Doc edges: added `_enrich_graph_nodes_from_db()` pre-population (populates rel_path, source_text, symbol_type, symbol_name, is_doc, language from DB)
- [x] Doc edges: added backtick `rsplit(".", 1)[-1]` dotted-name segment fallback
- [x] Bridge stats: added `components_after` via `nx.number_weakly_connected_components(G)` after bridging
- [x] Weight stats: `apply_edge_weights()` now returns `min, max, mean, synthetic_floored` + empty-graph guard + logging
