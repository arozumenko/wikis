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

## 0. Reconciliation: plan claims vs. live code

| Claim from plans | Code state | Gap to close |
|---|---|---|
| `WikiStorageProtocol` has `count_fts_matches`, `search_fts_by_column`, `search_fts_with_path`, `get_embedding_by_id`, `batch_similarity_search` | **Absent** — only `search_fts`, `search_fts5`, `search_fts_by_symbol_name`, `search_vec`, `search_hybrid` exist in `backend/app/core/storage/protocol.py` | Phase 0 — protocol additions + dual-backend impls |
| FTS scores comparable across backends | SQLite `bm25(...)` returns **negative**, ORDER BY ASC (`backend/app/core/storage/sqlite.py` L749). Postgres `ts_rank_cd` returns **positive**, ORDER BY DESC (`backend/app/core/storage/postgres.py` L825) | Phase 0 — normalize to `[0,1]` `score_norm` field at storage boundary |
| Pass 1 FTS uses path_prefix + symbol_types + IDF gate | `backend/app/core/graph_topology.py` L376–L405 calls `db.search_fts5(query=symbol_name, limit=fts_limit)` with **no filters, no IDF** | Phase 1+2 |
| Pass 2 reuses Phase 1b embeddings | Pass 2 always re-embeds via `embed_batch_fn`. Phase 1b already stores embeddings per node_id in `repo_vec` | Phase 3 — `get_embedding_by_id` + reuse |
| Cascade order: explicit-ref → hybrid → lexical → directory | Current order: lexical → vector → directory; doc-link edges injected as a **separate later step** in `inject_doc_edges` | Phase 3 — reorder + extract markdown/backtick parsing into Pass 1 |
| Per-edge `provenance` dict + `Phase2Stats` + `explain_connectivity()` | `_add_edge` stores only `relationship_type`/`edge_class`/`created_by`/`raw_similarity`. No stats dataclass, no diagnostic API | Phase 1 |
| Edge dedup before insert | `_add_edge` does NOT check `G.has_edge(u,v,rel_type)`; `inject_doc_edges` uses a local `seen` set only for doc edges | Phase 1 |
| `cross_language` and `test_link` edge classes; excluded from in-degree | `_SYNTHETIC_CLASSES = {"directory","lexical","semantic","doc","bridge"}` — both new classes missing | Phase 6 |
| `cross_language_linker.py`, `api_surface_extractor.py`, `cross_repo_linker.py`, `relatedness_scorer.py`, `project_storage.py`, `federated_query_service.py`, `federated_retriever.py` | All **absent** | Phases 6–7 |
| Node IDs collision-safe across same-stem files in different dirs | `node_id = f"{language}::{file_name}::{local_qualified_name}"` where `file_name = Path(file_path).stem`. Hash-suffix only fires when same `node_id` is seen with a *different* `file_path`, so two distinct files with the same stem and same class collide silently the first time. | Phase 5 (3A) |
| `GENERIC_CLASS_NAMES` + REST endpoint disambiguation | Not present; TS parser captures decorators but Java/Go/C# stub `has_decorators=False` | Phase 2 (REST), Phase 6 (per-language API extractors) |
| Feature flags `orphan_cascade_v2`, `WIKI_CROSS_LANGUAGE_LINKING`, `WIKI_PROJECT_GRAPH`, etc. | Only `hierarchical_leiden`, `capability_validation`, `smart_expansion`, `coverage_ledger`, `language_hints`, `exclude_tests` exist | Every phase adds its own flag |
| `cross_language_relationships` already detected in Phase 1 | Yes — `graph_builder.py` populates `RepositoryGraph.cross_language_relationships`, but this list is **not converted to weighted graph edges with `edge_class="cross_language"`**. It is metadata only. | Phase 6 — wire it into the graph |

**Bottom-line gap inventory** (single source of truth, ordered by what blocks what):

* **G1.** Storage protocol is too narrow → blocks every phase that needs richer FTS / embedding lookup.
* **G2.** FTS score semantics differ between backends → blocks RRF (Phase 4) and any threshold tuning.
* **G3.** No edge dedup, no provenance, no Phase2Stats → blocks observability needed to validate every later change.
* **G4.** Pass 1 FTS is unfiltered + un-IDF-gated → produces noise that will corrupt RRF (Phase 4) if not fixed first.
* **G5.** Pass 2 re-embeds → 5–20× slower than necessary; blocks cascade reorder because reorder makes Pass 2 fire more often.
* **G6.** Cascade ordering wrong → must be fixed *after* Pass 1 cleanup and *before* hybrid RRF lands.
* **G7.** `file_name`-stem-based node IDs collide silently → blocks reliable explicit-ref Pass 1, REST disambiguation, and cross-language/cross-repo linking (every linker resolves by stem today).
* **G8.** No `cross_language`/`test_link` edge classes and no `_SYNTHETIC_CLASSES` membership → cross-language edges would distort weighting + hub detection if added prematurely.
* **G9.** No project storage / federation layer → cross-repo linking can't be expressed without it.

This dependency graph dictates phase ordering below.

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
