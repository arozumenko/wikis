# Graph Quality Improvements: Lexical Orphan Resolution, Connectivity & Node Disambiguation

> **Date:** 2026-04-23 (updated: PostgreSQL parity section added)  
> **Scope:** `backend/app/core/graph_topology.py`, `backend/app/core/code_graph/graph_builder.py`, `backend/app/core/cluster_expansion.py`, `backend/app/core/storage/sqlite.py`, `backend/app/core/storage/postgres.py`

---

## 1. Lexical Orphan Resolution — Preventing Edge Flooding

### Current Behaviour

The FTS5 lexical pass in `resolve_orphans()` (`graph_topology.py:434-469`) works as follows:

```python
# For each orphan node:
symbol_name = ...  # extracted from DB or graph attrs
if not symbol_name or len(symbol_name) < 2:
    continue
fts_hits = db.search_fts5(query=symbol_name, limit=fts_limit)  # fts_limit=3
# Add up to max_lexical_edges=2 edges per orphan
```

**Problems:**

1. **No symbol-type filtering.** A generic name like `shared`, `config`, `utils`, `get`, `post`, `data`, `result` matches across ALL symbol types (classes, functions, fields, constants, doc sections). The `search_fts5()` method already accepts a `symbol_types` parameter — it's just never passed.

2. **No path-locality bias.** Unlike the semantic (Pass 2) and directory (Pass 3) passes which use `_expanding_prefixes()` for locality, the lexical pass searches globally. A `shared` function in `auth/` connects to an unrelated `shared` constant in `billing/`.

3. **No name-quality gate.** The only filter is `len(symbol_name) < 2`. Common short names (`get`, `set`, `run`, `init`, `data`, `key`, `name`, `type`, `id`) pass easily and produce noisy fan-out edges.

4. **FTS5 BM25 weights don't help enough.** The BM25 weights (`symbol_name=10, signature=4, docstring=2, source_text=1`) rank symbol-name matches highest, but a query for `shared` still returns every node that mentions "shared" anywhere in its source text.

5. **No self-type awareness.** An orphan `function` named `validate` connects to a `class` named `Validate` — architecturally meaningless.

### Proposed: Tiered Lexical Matching

Replace the single FTS5 call with a **4-tier cascade** that stops at the first tier producing results. Each tier progressively relaxes constraints:

| Tier | Match Target | Path Scope | Symbol Types | Min Name Length | Rationale |
|------|-------------|------------|--------------|-----------------|-----------|
| **T1 — Exact name + local** | `symbol_name` column only | Same directory | Architectural only (`class`, `interface`, `function`, `enum`, `struct`) | 4 chars | Strongest signal: same name, same area, important types |
| **T2 — Exact name + parent dir** | `symbol_name` column only | Parent directory | Architectural only | 4 chars | Slightly wider locality |
| **T3 — Full FTS + local** | All FTS5 columns (name, signature, docstring, source) | Same + parent directory | Any type | 6 chars | Allows docstring/signature matches but keeps locality |
| **T4 — Full FTS global** | All FTS5 columns | Global (no path filter) | Architectural only | 8 chars | Last resort — only long, distinctive names go global |

**Implementation sketch:**

```python
# In resolve_orphans(), Pass 1 replacement:

_ARCH_TYPES = ["class", "interface", "function", "enum", "struct", "trait", "protocol"]

def _tiered_lexical_search(db, node_id, symbol_name, rel_path, fts_limit):
    """4-tier lexical cascade — stops at first tier with results."""
    if len(symbol_name) < 4:
        return []
    
    prefixes = _expanding_prefixes(rel_path)
    local_prefix = prefixes[0] if prefixes and prefixes[0] else None
    parent_prefix = prefixes[1] if len(prefixes) > 1 and prefixes[1] else None
    
    # T1: exact name + same directory + architectural types
    if local_prefix:
        hits = db.search_fts5(
            query=f'"{symbol_name}"',  # exact phrase
            path_prefix=local_prefix,
            symbol_types=_ARCH_TYPES,
            limit=fts_limit,
        )
        hits = [h for h in hits if h["node_id"] != node_id]
        if hits:
            return hits
    
    # T2: exact name + parent directory + architectural types
    if parent_prefix:
        hits = db.search_fts5(
            query=f'"{symbol_name}"',
            path_prefix=parent_prefix,
            symbol_types=_ARCH_TYPES,
            limit=fts_limit,
        )
        hits = [h for h in hits if h["node_id"] != node_id]
        if hits:
            return hits
    
    # T3: full FTS + local scope + any type (only for longer names)
    if len(symbol_name) >= 6 and (local_prefix or parent_prefix):
        hits = db.search_fts5(
            query=symbol_name,
            path_prefix=local_prefix or parent_prefix,
            limit=fts_limit,
        )
        hits = [h for h in hits if h["node_id"] != node_id]
        if hits:
            return hits
    
    # T4: global FTS + architectural types (only for distinctive names)
    if len(symbol_name) >= 8:
        hits = db.search_fts5(
            query=f'"{symbol_name}"',
            symbol_types=_ARCH_TYPES,
            limit=fts_limit,
        )
        hits = [h for h in hits if h["node_id"] != node_id]
        if hits:
            return hits
    
    return []
```

### Additional Safeguards

1. **Stopword list for code symbols.** Maintain a set of common names that should never trigger global lexical search:

   ```python
   _LEXICAL_STOPWORDS = frozenset({
       "get", "set", "put", "post", "delete", "patch", "head",
       "run", "start", "stop", "init", "setup", "teardown",
       "data", "result", "response", "request", "context", "config",
       "shared", "common", "utils", "helpers", "base", "core",
       "index", "main", "app", "test", "spec",
       "name", "type", "id", "key", "value", "item", "node",
   })
   ```

   If `symbol_name.lower()` is in this set, skip T3 and T4 entirely (only allow T1/T2 local matches).

2. **BM25 rank threshold.** Discard FTS5 hits below a minimum BM25 score. Currently all hits within `limit` are accepted regardless of relevance. Add:

   ```python
   hits = [h for h in hits if abs(h.get("fts_rank", 0)) > MIN_FTS_RANK]
   ```

   (Note: SQLite FTS5 BM25 returns negative scores where more negative = better match. A threshold like `-15.0` would filter weak matches.)

3. **Orphan symbol-type awareness.** When the orphan is a `method` or `field`, require the target to be a `class` or `interface` (or vice versa). Don't connect `method → method` across unrelated files.

### Impact Assessment

- **Before:** A repo with 500 orphans and common names like `get`, `shared`, `config` could generate 1000+ noisy lexical edges, pulling unrelated modules into the same Leiden community.
- **After:** Tiered matching with locality + type filtering should reduce lexical edges by ~60-70% while improving their precision. Orphans with generic names fall through to the semantic (Pass 2) or directory (Pass 3) passes, which are already locality-aware.

---

## 1.5. PostgreSQL Backend Parity — FTS Differences That Affect Lexical Edges

### The Problem

The lexical orphan resolution in `graph_topology.py` calls `db.search_fts5()` which is polymorphic — it dispatches to either `SQLiteStorage.search_fts5()` or `PostgresStorage.search_fts()` (aliased as `search_fts5`). While the **interface** is identical (same parameters: `query`, `path_prefix`, `cluster_id`, `symbol_types`, `limit`), the **scoring semantics and query parsing** differ fundamentally between backends. Any improvement to lexical edge creation must account for both.

### Backend Comparison

| Aspect | SQLite (`storage/sqlite.py:749-809`) | PostgreSQL (`storage/postgres.py:825-868`) |
|--------|--------------------------------------|---------------------------------------------|
| **FTS engine** | FTS5 virtual table (`repo_fts`) | `tsvector`/`tsquery` column (`fts_doc`) on `repo_nodes` |
| **Query parsing** | FTS5 syntax: `MATCH ?` — supports `"exact phrase"`, `prefix*`, `AND`/`OR`/`NOT` | `websearch_to_tsquery('english', :query)` — natural language; `"quotes"` = exact phrase, `-` = NOT, `or` = OR |
| **Ranking function** | `bm25(repo_fts, 10.0, 4.0, 2.0, 1.0)` — returns **negative** scores (more negative = better match) | `ts_rank_cd(fts_doc, ...)` — returns **positive** scores (higher = better match) |
| **Sort order** | `ORDER BY fts_rank` (ascending — most negative first) | `ORDER BY fts_rank DESC` (descending — highest positive first) |
| **Column weights** | BM25 positional weights: `symbol_name=10, signature=4, docstring=2, source_text=1` | tsvector weights set at index time: `A` (symbol_name), `B` (signature), `C` (docstring), `D` (source_text) — default `ts_rank_cd` weights are `{0.1, 0.2, 0.4, 1.0}` for D/C/B/A |
| **Path filtering** | `GLOB` pattern (`path/*`) | `LIKE` pattern (`path/%`) |
| **Type filtering** | `IN (?, ?, ...)` with positional params | `= ANY(:symbol_types)` with array param |
| **Fallback on error** | Retries with `prefix*` query on `OperationalError` | Returns `[]` on any exception |
| **Separate FTS table** | Yes — `repo_fts` is a separate FTS5 virtual table joined to `repo_nodes` | No — `fts_doc` tsvector column lives on `repo_nodes` directly |

### Issues for the Tiered Lexical Cascade

#### Issue 1: Inverted Score Semantics Break Universal Rank Thresholds

The proposed BM25 rank threshold (Action 1C) uses:

```python
hits = [h for h in hits if abs(h.get("fts_rank", 0)) > MIN_FTS_RANK]
```

This **only works for SQLite** where BM25 scores are negative. PostgreSQL's `ts_rank_cd` returns positive floats typically in the range `0.0 – 1.0` (rarely higher). A universal `abs()` threshold would need completely different calibration per backend.

**Recommended fix — normalize scores at the storage layer:**

```python
# Option A: Normalize in each backend's search method
# SQLite — negate BM25 so higher = better (matching Postgres convention):
#   In sqlite.py search_fts5(), post-process:
for row in rows:
    row["fts_rank"] = -row["fts_rank"]  # Now positive, higher = better

# Option B: Backend-aware threshold in graph_topology.py
def _passes_rank_threshold(rank: float, backend: str) -> bool:
    if backend == "sqlite":
        return rank < -10.0  # BM25: more negative = better
    else:  # postgres
        return rank > 0.1    # ts_rank_cd: higher = better
```

**Option A is strongly preferred** — it makes all downstream code backend-agnostic. The normalization should happen inside `search_fts5()` / `search_fts()` so that `graph_topology.py` never needs to know which backend is active.

#### Issue 2: Exact-Phrase Query Syntax Differs

The tiered cascade (T1, T2, T4) wraps the symbol name in double quotes for exact matching:

```python
query=f'"{symbol_name}"'  # FTS5 exact phrase
```

- **SQLite FTS5:** `"shared"` matches only the exact token `shared` in the `symbol_name` column (when BM25 weights prioritize it). ✅
- **PostgreSQL `websearch_to_tsquery`:** `"shared"` is also treated as an exact phrase match. ✅ This works.

However, there's a subtle difference: PostgreSQL's `websearch_to_tsquery` applies **stemming** via the `'english'` dictionary. So `"validates"` would match `validate`, `validated`, `validating`. SQLite FTS5 does **not** stem by default (it uses a simple tokenizer). This means:

- PostgreSQL may return **more** hits for the same query (stemmed variants)
- SQLite returns **fewer**, more precise hits

**Recommended fix:** For T1/T2 (exact name matching), PostgreSQL should use `phraseto_tsquery('simple', :query)` instead of `websearch_to_tsquery('english', :query)` to disable stemming. This requires the tiered cascade to pass a hint or the storage layer to expose a `exact_match: bool` parameter:

```python
# In postgres.py search_fts():
def search_fts(self, query, ..., exact_match=False):
    if exact_match:
        tsquery_fn = "phraseto_tsquery('simple', :query)"
    else:
        tsquery_fn = "websearch_to_tsquery('english', :query)"
    
    conditions = [f"fts_doc @@ {tsquery_fn}"]
    # ... rest unchanged, but use same tsquery_fn in ts_rank_cd() call
```

#### Issue 3: PostgreSQL Has No BM25 — Relevance Ranking Is Weaker

SQLite's BM25 considers **term frequency**, **inverse document frequency**, and **document length normalization** — the gold standard for text relevance. PostgreSQL's `ts_rank_cd` only considers **cover density** (how close matching terms are) and tsvector weights. It does **not** penalize long documents or account for corpus-wide term rarity.

**Practical impact:** In PostgreSQL, a query for `"config"` will rank a 500-line file mentioning `config` once the same as a 10-line file where `config` is the symbol name — because `ts_rank_cd` doesn't do document-length normalization. This makes the lexical pass **noisier on PostgreSQL** than on SQLite.

**Recommended fixes (choose one or combine):**

1. **Use `ts_rank()` instead of `ts_rank_cd()`** with explicit weights that boost symbol_name:
   ```sql
   ts_rank('{0.1, 0.2, 0.4, 1.0}', fts_doc, websearch_to_tsquery('english', :query), 1|32)
   -- Flag 1: divides rank by document length
   -- Flag 32: divides rank by 1 + log(document length)
   ```
   The normalization flags (bitmask `1|32 = 33`) approximate BM25's length normalization.

2. **Install `pg_bestmatch`** or **`pgroonga`** extension for true BM25 on PostgreSQL. This is a heavier lift but gives scoring parity. Not recommended unless PostgreSQL is the primary deployment target.

3. **Stricter filtering on PostgreSQL path.** Since ranking is weaker, compensate by being more aggressive with `path_prefix` and `symbol_types` filters — i.e., never allow T4 (global search) on PostgreSQL, or reduce `fts_limit` for PostgreSQL.

#### Issue 4: tsvector Weight Mapping Is Not Aligned with BM25 Weights

The SQLite BM25 weights are `symbol_name=10, signature=4, docstring=2, source_text=1`. The code comment in `sqlite.py:787` says these are "aligned with Postgres tsvector weights (A=1.0, B=0.4, C=0.2, D=0.1)" — but this is **misleading**.

PostgreSQL's default `ts_rank_cd` weights for A/B/C/D are `{1.0, 0.4, 0.2, 0.1}` — these are **relative** weights, not absolute. The BM25 weights `{10, 4, 2, 1}` are the same ratio, so the **relative** ranking is similar. But the **absolute** score ranges are completely different:

- SQLite BM25: typically `-30.0` to `-0.1`
- PostgreSQL ts_rank_cd: typically `0.001` to `1.0`

This matters for any threshold-based filtering.

### Summary of PostgreSQL-Specific Actions

| # | Action | File(s) | Priority | Effort | Impact |
|---|--------|---------|----------|--------|--------|
| 1.5A | Normalize FTS scores to positive-higher-is-better in both backends | `sqlite.py`, `postgres.py` | **High** | Low | Enables universal rank thresholds in `graph_topology.py` |
| 1.5B | Add `exact_match` parameter to `search_fts` for stemming control | `postgres.py`, `sqlite.py` | **High** | Low | Prevents stemmed false positives in T1/T2 tiers |
| 1.5C | Add length-normalization flags to `ts_rank_cd` or switch to `ts_rank` | `postgres.py` | Medium | Low | Reduces noise from long documents on PostgreSQL |
| 1.5D | Consider skipping T4 (global search) on PostgreSQL or reducing `fts_limit` | `graph_topology.py` | Medium | Low | Compensates for weaker ranking |
| 1.5E | Unify error handling: add prefix-fallback retry in PostgreSQL (like SQLite) | `postgres.py` | Low | Low | Better resilience for unusual symbol names |

### Recommended Implementation Order (integrated with Section 1)

Update the Phase 1 quick-wins to include:

1. **1.5A (score normalization)** — do this FIRST, before any rank threshold work (1C)
2. **1.5B (exact_match param)** — do alongside the tiered cascade (1A)
3. **1.5C (ts_rank normalization flags)** — do alongside BM25 threshold (1C)
4. **1.5D (PostgreSQL T4 restriction)** — do alongside stopword list (1B)

---

## 2. Connectivity Tracing — End-to-End Audit

### Current Pipeline

The connectivity pipeline runs in `run_phase2()` (`graph_topology.py:1127-1205`):

```
Phase 1 (graph_builder.py)
  → AST parsing → structural edges (calls, imports, inherits, defines, etc.)

Phase 2 (graph_topology.py::run_phase2)
  Step 1: resolve_orphans()
    Pass 1: FTS5 lexical match → lexical_link edges
    Pass 2: Semantic vector KNN → semantic_link edges  
    Pass 3: Directory proximity → directory_link edges
  Step 2: inject_doc_edges()
    Tier 1: Markdown hyperlinks → hyperlink edges
    Tier 2: Directory proximity → proximity edges
  Step 3: bridge_disconnected_components()
    → component_bridge edges (bidirectional)
  Step 4: apply_edge_weights()
    → inverse in-degree weights on all edges
  Step 5: detect_hubs()
    → flag high-Z-score nodes
  Step 6: persist_weights_to_db()

Phase 3 (graph_clustering.py)
  → Leiden community detection on weighted graph
```

### Identified Connectivity Gaps

#### Gap A: No connectivity metrics between phases

There's no logging of graph connectivity metrics (number of components, largest component size, average degree) between pipeline steps. This makes it impossible to diagnose which step is helping or hurting.

**Action:** Add a `_log_connectivity_snapshot()` helper called after each step:

```python
def _log_connectivity_snapshot(G: nx.MultiDiGraph, label: str) -> Dict[str, Any]:
    n_components = nx.number_weakly_connected_components(G)
    components = list(nx.weakly_connected_components(G))
    largest = max(len(c) for c in components) if components else 0
    isolates = sum(1 for n in G.nodes() if G.degree(n) == 0)
    avg_degree = sum(G.degree(n) for n in G.nodes()) / max(G.number_of_nodes(), 1)
    
    stats = {
        "label": label,
        "nodes": G.number_of_nodes(),
        "edges": G.number_of_edges(),
        "components": n_components,
        "largest_component": largest,
        "isolates": isolates,
        "avg_degree": round(avg_degree, 2),
    }
    logger.info("[CONNECTIVITY] %s: %s", label, stats)
    return stats
```

Call it: after Phase 1, after each Pass in orphan resolution, after doc edges, after bridging.

#### Gap B: Component bridging uses only directory similarity

`bridge_disconnected_components()` (`graph_topology.py:1012-1120`) connects components via directory histogram overlap. This fails when:
- Components span multiple directories (e.g., a feature touches `models/`, `services/`, `api/`)
- Two components are in the same directory but semantically unrelated

**Action:** Add a secondary bridging signal using shared import targets. If component A and component B both have structural edges to the same hub node (even if the hub is quarantined), they're likely related:

```python
def _import_sim(G, comp_a, comp_b, hubs):
    """Count shared import targets between two components."""
    targets_a = {v for u in comp_a for _, v in G.out_edges(u)}
    targets_b = {v for u in comp_b for _, v in G.out_edges(u)}
    return len(targets_a & targets_b)
```

#### Gap C: Orphan resolution doesn't consider the orphan's symbol type

An orphan `class` should preferentially connect to other classes or to functions that reference it. An orphan `function` should connect to callers or to the class it belongs to. Currently all orphans are treated identically.

**Action:** In Pass 1 (lexical), pass `symbol_types` filter based on the orphan's own type:

| Orphan Type | Preferred Target Types |
|-------------|----------------------|
| `class`, `interface` | `class`, `interface`, `function` (potential users) |
| `function`, `method` | `class` (potential owner), `function` (potential caller) |
| `constant`, `enum` | `class`, `function`, `module` (potential consumers) |
| doc nodes | `class`, `function`, `module` (documented subjects) |

#### Gap D: No edge deduplication across passes

If Pass 1 (lexical) connects orphan A → node B, and Pass 2 (semantic) also finds A → B, a duplicate edge is added. While NetworkX MultiDiGraph allows this, it inflates degree counts and distorts Leiden.

**Action:** Before adding any synthetic edge, check `G.has_edge(source, target)`:

```python
# In _add_edge(), add dedup check:
if G.has_edge(source, target):
    return  # Already connected by an earlier pass
```

---

## 3. Node Disambiguation — Flask-Style API Classes

### Current Behaviour

Node IDs are generated in `_process_file_symbols()` (`graph_builder.py:1491-1544`):

```python
local_qualified_name = self._get_local_qualified_name(symbol, file_name)
node_id = f"{language}::{file_name}::{local_qualified_name}"

# On collision (same node_id, different file):
file_hash = hash(file_path) & 0xFFFF
node_id = f"{node_id}_{file_hash:04x}"
```

**The Flask API problem:**

In Flask-RESTful / Flask-MethodView projects, you commonly have:

```
api/
├── users.py      → class API(MethodView): def get(self): ...
├── products.py   → class API(MethodView): def get(self): ...
└── orders.py     → class API(MethodView): def get(self): ...
```

Each parser produces symbols with `full_name = "users.API.get"`, `full_name = "products.API.get"`, etc.

After `_get_local_qualified_name()` strips the module prefix:
- `users.py` → `local_qualified_name = "API.get"` → `node_id = "python::users::API.get"` ✅
- `products.py` → `local_qualified_name = "API.get"` → `node_id = "python::products::API.get"` ✅

**Wait — these don't actually collide** because `file_name` differs (`users` vs `products`). The hash suffix only triggers when two different `file_path` values produce the same `file_name` (e.g., `src/api/users.py` and `lib/api/users.py`).

So the real disambiguation problem is:

1. **Same `file_name` in different directories** → hash suffix makes node IDs opaque (`python::users::API.get_a3f2`)
2. **Index lookups by simple name** → `_simple_name_index` maps `get` to ALL `API.get` nodes across all modules, causing false resolution in relationship building

### Proposed Improvements

#### Action 3A: Use `rel_path` instead of `file_name` in node IDs

Replace `file_name` (stem only) with a sanitized `rel_path` to guarantee uniqueness without hashes:

```python
# Current:
node_id = f"{language}::{file_name}::{local_qualified_name}"

# Proposed:
# Sanitize rel_path: "src/api/users.py" → "src/api/users"
module_path = rel_path.replace("/", ".").removesuffix(f".{Path(rel_path).suffix.lstrip('.')}")
node_id = f"{language}::{module_path}::{local_qualified_name}"

# Result: "python::src.api.users::API.get" — unique, readable, no hash needed
```

**Trade-offs:**
- ✅ Eliminates hash suffixes entirely
- ✅ Node IDs are human-readable and debuggable
- ✅ No collision possible (rel_path is unique per file)
- ⚠️ Longer node IDs (more memory, wider DB columns)
- ⚠️ Breaking change for existing wikis (node IDs change → stored edges/clusters invalidated)

**Migration path:** Add a config flag `NODE_ID_STYLE=rel_path|file_name` (default `file_name` for backward compat, `rel_path` for new wikis). Deprecate `file_name` style after one release cycle.

#### Action 3B: Qualify the simple-name index with parent context

The `_simple_name_index` in `attach_graph_indexes()` (`graph_builder.py:74-135`) maps bare names like `get` to node IDs. For method-level symbols, also index by `Parent.method`:

```python
# Current (graph_builder.py:112-115):
simple_name = _simple_symbol_name_for_index(symbol_name)
if simple_name:
    _maybe_set(graph._simple_name_index, simple_name, node_id)

# Proposed: also index qualified parent.method
parent = node_data.get('parent_symbol', '')
if parent and simple_name:
    qualified = f"{parent}.{simple_name}"
    _maybe_set(graph._simple_name_index, qualified, node_id)
```

This way, relationship resolution for `API.get` finds the correct node without falling back to the ambiguous bare `get`.

#### Action 3C: Decorator-aware class naming in Python parser

For Flask-RESTful and similar frameworks, the class name `API` is a convention, not the real identity. The **module path** is the identity. The Python parser (`parsers/python_parser.py`) should detect common patterns:

```python
# If a class has a generic name AND inherits from a known framework base:
_FRAMEWORK_BASES = {"MethodView", "Resource", "APIView", "ViewSet", "View"}

# During parsing, if class_name in {"API", "View", "Resource"} and base in _FRAMEWORK_BASES:
#   Rename to: f"{file_name}_{class_name}" → "users_API", "products_API"
```

This is **optional and aggressive** — it changes the symbol name visible in the wiki. A safer approach is to only use this for node ID generation, not for display names.

#### Impact on Connectivity

The hash-suffix disambiguation (`_a3f2`) does NOT break connectivity directly because:
1. Relationship resolution uses the `symbol_registry` (which maps by qualified name and full path), not by node ID parsing
2. The hash suffix is deterministic (`hash(file_path) & 0xFFFF`), so the same file always produces the same suffix

However, it **does** break:
- Human debugging (opaque IDs in logs)
- FTS5 lexical orphan resolution (searching for `API.get` won't match `python::users::API.get_a3f2` by node_id)
- Cross-wiki references if file paths change

---

## 4. Summary of Recommended Actions

| # | Action | File(s) | Priority | Effort | Impact |
|---|--------|---------|----------|--------|--------|
| 1A | Tiered lexical matching (4-tier cascade) | `graph_topology.py` | **High** | Medium | Eliminates ~60-70% of noisy lexical edges |
| 1B | Code-symbol stopword list | `graph_topology.py` | **High** | Low | Prevents common names from flooding edges |
| 1C | BM25 rank threshold | `graph_topology.py` | Medium | Low | Filters weak FTS5 matches |
| 1D | Orphan-type-aware target filtering | `graph_topology.py` | Medium | Medium | Better type-compatible connections |
| **1.5A** | **Normalize FTS scores (positive = better) in both backends** | **`sqlite.py`, `postgres.py`** | **High** | **Low** | **Prerequisite for universal rank thresholds** |
| **1.5B** | **Add `exact_match` param for stemming control** | **`postgres.py`, `sqlite.py`** | **High** | **Low** | **Prevents stemmed false positives in T1/T2** |
| 1.5C | Add length-normalization flags to PostgreSQL `ts_rank_cd` | `postgres.py` | Medium | Low | Reduces noise from long documents |
| 1.5D | Skip T4 global search on PostgreSQL or reduce `fts_limit` | `graph_topology.py` | Medium | Low | Compensates for weaker PG ranking |
| 1.5E | Unify error handling: prefix-fallback retry in PostgreSQL | `postgres.py` | Low | Low | Better resilience for unusual symbol names |
| 2A | Connectivity snapshot logging | `graph_topology.py` | **High** | Low | Essential for diagnosing pipeline issues |
| 2B | Import-similarity component bridging | `graph_topology.py` | Medium | Medium | Better cross-directory bridging |
| 2C | Edge deduplication across passes | `graph_topology.py` | Medium | Low | Prevents inflated degree counts |
| 3A | `rel_path`-based node IDs | `graph_builder.py` | **High** | High | Eliminates hash suffixes, human-readable IDs |
| 3B | Qualified simple-name index | `graph_builder.py` | Medium | Low | Better method-level resolution |
| 3C | Decorator-aware class naming | `python_parser.py` | Low | Medium | Framework-specific, narrow benefit |

### Recommended Implementation Order

1. **Phase 0 (Backend parity — do FIRST):** 1.5A (score normalization), 1.5B (exact_match param)
2. **Phase 1 (Quick wins):** 1B (stopword list), 2A (connectivity logging), 2C (edge dedup), 1C (BM25 threshold), 1.5C (PG length normalization)
3. **Phase 2 (Core improvements):** 1A (tiered lexical), 1D (type-aware filtering), 3B (qualified index), 1.5D (PG T4 restriction)
4. **Phase 3 (Breaking changes):** 3A (rel_path node IDs — requires migration strategy), 2B (import-sim bridging), 3C (decorator naming)
