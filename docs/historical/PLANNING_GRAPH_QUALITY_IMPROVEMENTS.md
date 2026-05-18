# Graph Quality Improvements: Lexical Orphan Resolution, Node Disambiguation & Connectivity

> **Date**: 2026-04-24
> **Scope**: `wikis` project — `backend/app/core/graph_topology.py`, `graph_builder.py`, `storage/sqlite.py`, `storage/postgres.py`
> **Backends**: SQLite (FTS5 + sqlite-vec) and PostgreSQL (tsvector/tsquery + pgvector)

---

## Table of Contents

1. [Executive Summary](#1-executive-summary)
2. [Problem 1 — Lexical Orphan Resolution Flooding](#2-problem-1--lexical-orphan-resolution-flooding)
   - 2.1 [Current Behavior Analysis](#21-current-behavior-analysis)
   - 2.2 [Root Cause: No Frequency Gating](#22-root-cause-no-frequency-gating)
   - 2.3 [Proposed Solution: Tiered Lexical Resolution with IDF Gating](#23-proposed-solution-tiered-lexical-resolution-with-idf-gating)
   - 2.4 [Implementation Plan](#24-implementation-plan)
   - 2.5 [Backend-Specific Considerations (SQLite vs PostgreSQL)](#25-backend-specific-considerations-sqlite-vs-postgresql)
3. [Problem 2 — Node Disambiguation for Python REST API Classes](#3-problem-2--node-disambiguation-for-python-rest-api-classes)
   - 3.1 [Current Behavior Analysis](#31-current-behavior-analysis)
   - 3.2 [The Flask/Django REST Pattern](#32-the-flaskdjango-rest-pattern)
   - 3.3 [Proposed Solution: Context-Enriched Symbol Names + symbol_type Metadata](#33-proposed-solution-context-enriched-symbol-names--symbol_type-metadata)
   - 3.4 [Implementation Plan](#34-implementation-plan)
   - 3.5 [Impact on Connectivity](#35-impact-on-connectivity)
4. [Problem 3 — Connectivity Tracing & Diagnostics](#4-problem-3--connectivity-tracing--diagnostics)
   - 4.1 [Current Observability Gaps](#41-current-observability-gaps)
   - 4.2 [Proposed Solution: Structured Edge Provenance](#42-proposed-solution-structured-edge-provenance)
   - 4.3 [Diagnostic Query Toolkit](#43-diagnostic-query-toolkit)
5. [Cross-Cutting: SQLite vs PostgreSQL Backend Parity](#5-cross-cutting-sqlite-vs-postgresql-backend-parity)
6. [Action Items Summary](#6-action-items-summary)
7. [Test Strategy](#7-test-strategy)

---

## 1. Executive Summary

Three interrelated graph quality issues degrade wiki structure for certain repository patterns:

| Problem | Impact | Severity |
|---------|--------|----------|
| Lexical orphan resolution uses raw FTS with no frequency filtering | Common words like `shared`, `get`, `config`, `utils` create hundreds of spurious edges, merging unrelated clusters | **High** |
| Python REST API classes with generic method names (`get`, `post`, `delete`) have ambiguous FTS symbol names | Orphan resolution cross-links unrelated API endpoints; clustering merges distinct REST resources | **Medium** |
| No observability into why synthetic edges exist or how components evolved through Phase 2 | Debugging graph quality requires reading code; no way to audit edge provenance at runtime | **Medium** |

All solutions must work identically across both **SQLite** (FTS5 + BM25) and **PostgreSQL** (tsvector/tsquery + ts_rank) backends.

---

## 2. Problem 1 — Lexical Orphan Resolution Flooding

### 2.1 Current Behavior Analysis

**Location**: `graph_topology.py` → `resolve_orphans()`, Pass 1 (lines ~430–490)

The current 3-pass orphan resolution cascade:

```
Pass 0: _resolve_orphans_by_directory()  → directory proximity edges
Pass 1: FTS5 lexical match              → search_fts5(symbol_name)
Pass 2: Vector similarity               → embedding-based nearest neighbor
```

**Pass 1 internals** (the problematic pass):

```python
# 1. Extract symbol_name — NO normalization, NO filtering
symbol_name = node.get("symbol_name", "") or ""
if not symbol_name or len(symbol_name) < 2:
    continue

# 2. Raw FTS query — entire symbol_name passed as-is
fts_hits = db.search_fts5(query=symbol_name, limit=fts_limit)

# 3. Filter only self-matches, add up to max_lexical_edges=2 edges
for hit in fts_hits:
    if hit["node_id"] != node_id:
        G.add_edge(node_id, hit["node_id"],
                   rel_type="lexical_link",
                   edge_class="lexical",
                   weight=SYNTHETIC_WEIGHT_FLOOR,  # 0.5
                   created_by="fts5_lexical")
```

**Critical issues**:

1. **No stopword filtering**: A symbol named `shared` matches every node that has "shared" anywhere in `symbol_name`, `signature`, `docstring`, or `source_text`.
2. **No frequency analysis**: Whether a query term matches 2 nodes or 200 nodes, it gets the same treatment. High-frequency terms should be suppressed.
3. **No tier differentiation**: A match on `symbol_name` (highly specific) gets the same weight as a match on `source_text` (very noisy). BM25 weights exist in the FTS table definition (`10.0, 4.0, 2.0, 1.0`), but the edge weight is always `SYNTHETIC_WEIGHT_FLOOR = 0.5`.
4. **No locality bias**: Pass 1 has no directory proximity preference — it can link a `shared` node in `/api/auth/` to a `shared` node in `/tests/fixtures/`.
5. **Flat edge weight**: All lexical edges get `weight=0.5` regardless of match quality, making them indistinguishable from high-confidence edges during Leiden clustering.

### 2.2 Root Cause: No Frequency Gating

The FTS5 `MATCH` operator in SQLite and `to_tsquery` in PostgreSQL return rows ranked by BM25/ts_rank, but there is **no pre-filter** on:

- **Document frequency (DF)** of the query term — how many nodes contain this term
- **Inverse document frequency (IDF)** — the discriminative power of the term
- **Architectural type** of the symbol — whether the orphan is a class, function, module, or method

A word like `shared` appearing in 15% of all nodes should never create lexical edges because it has near-zero discriminative power. Currently it creates `min(fts_limit, total_matches)` edges per orphan that mentions it.

**Quantified impact example**:
- Repository with 500 nodes, 80 orphans
- 12 orphans have symbol_name containing "shared", "config", "utils", or "base"
- Each creates 2 lexical edges → 24 spurious edges
- These 24 edges can merge 3-4 unrelated clusters in Leiden (γ=0.5), creating "mega-clusters" that produce incoherent wiki pages

### 2.3 Proposed Solution: Tiered Lexical Resolution with IDF Gating

Replace the current flat FTS query with a **4-tier cascade** that uses graduated confidence and an **IDF gate** to suppress high-frequency terms.

#### Tier Architecture

| Tier | Match Target | Weight Multiplier | Description | When to Use |
|------|-------------|-------------------|-------------|-------------|
| **T1** | Exact qualified name | `1.0 × SYNTHETIC_WEIGHT_FLOOR` | `Class.method` matches `Class.method` in FTS `symbol_name` column only | Always — highest confidence |
| **T2** | Same-directory `rel_path` + name | `0.7 × SYNTHETIC_WEIGHT_FLOOR` | FTS match restricted to nodes sharing a directory prefix | Always — locality-biased |
| **T3** | Docstring cross-reference | `0.4 × SYNTHETIC_WEIGHT_FLOOR` | Backtick references `` `SymbolName` `` or explicit `See also:` in docstrings | Only if T1+T2 found < max_lexical_edges |
| **T4** | Full-text context (current behavior) | `0.2 × SYNTHETIC_WEIGHT_FLOOR` | Unrestricted FTS across all columns | Only if T1+T2+T3 found 0 edges AND IDF gate passes |

#### IDF Gate Logic

Before any tier fires, compute the **document frequency** of the query term:

```python
def _compute_symbol_idf(db, symbol_name: str, total_nodes: int) -> float:
    """
    Returns IDF score: log(N / (df + 1)) where df = number of nodes
    whose symbol_name column contains the query term.
    
    High IDF = rare term (good for lexical edges)
    Low IDF = common term (should be suppressed)
    """
    df = db.count_fts_matches(query=symbol_name, column="symbol_name")
    if total_nodes == 0:
        return 0.0
    return math.log(total_nodes / (df + 1))
```

**Gating rules**:

| IDF Score | Interpretation | Action |
|-----------|---------------|--------|
| `≥ 3.0` | Rare symbol — appears in < 5% of nodes | All tiers eligible |
| `1.5 – 3.0` | Moderate frequency | Only T1 and T2 allowed |
| `< 1.5` | Ubiquitous term (e.g., `get`, `shared`, `config`) | **Skip lexical resolution entirely** — rely on directory (Pass 0) and vector (Pass 2) |

#### symbol_type Gating (Supplementary)

Not all symbol types benefit equally from lexical edges. Methods inside classes should inherit connectivity from their parent class, not from FTS matching their bare name.

> **Note**: The `repo_nodes` table already has `symbol_type` (TEXT) and `is_architectural` (BOOLEAN) columns populated by `graph_builder.py`. No schema changes are needed — the tiered resolver simply reads these existing columns.

| `symbol_type` value | `is_architectural` | Lexical Tier Eligibility | Rationale |
|---------------------|-------------------|-------------------------|-----------|
| `module`, `class`, `interface`, `struct` | `true` | T1, T2, T3, T4 | Top-level architectural symbols — lexical is appropriate |
| `function` (top-level) | varies | T1, T2, T3 | Functions can have distinctive names |
| `method` (nested in class) | `false` | T1, T2 only | Methods like `get()`, `post()`, `__init__()` are too generic for broad FTS |
| `constructor`, `field`, `variable` | `false` | **None** — skip lexical | These should get connectivity from their parent, not lexical matching |

### 2.4 Implementation Plan

#### Step 1: Add `count_fts_matches()` to Both Backends

**SQLite** (`storage/sqlite.py`):
```sql
-- New method: count_fts_matches
SELECT COUNT(*) FROM repo_fts WHERE symbol_name MATCH ?
```

**PostgreSQL** (`storage/postgres.py`):
```sql
-- New method: count_fts_matches
SELECT COUNT(*) FROM repo_nodes
WHERE symbol_name_tsv @@ plainto_tsquery('english', $1)
```

Add to `WikiStorageProtocol`:
```python
def count_fts_matches(self, query: str, column: str = "symbol_name") -> int:
    """Count how many nodes match the query in the specified FTS column."""
    ...
```

#### Step 2: Add `search_fts_by_column()` for Tier-Specific Queries

**SQLite**: Use column-specific FTS5 queries:
```sql
-- T1: symbol_name only
SELECT node_id, rank FROM repo_fts
WHERE symbol_name MATCH ?
ORDER BY rank LIMIT ?

-- T2: symbol_name + path_prefix filter
SELECT f.node_id, f.rank FROM repo_fts f
JOIN repo_nodes n ON f.node_id = n.node_id
WHERE f.symbol_name MATCH ? AND n.rel_path LIKE ?
ORDER BY f.rank LIMIT ?
```

**PostgreSQL**: Use column-specific ts_rank:
```sql
-- T1: symbol_name only
SELECT node_id, ts_rank(symbol_name_tsv, plainto_tsquery('english', $1)) AS rank
FROM repo_nodes
WHERE symbol_name_tsv @@ plainto_tsquery('english', $1)
ORDER BY rank DESC LIMIT $2

-- T2: symbol_name + path prefix
SELECT node_id, ts_rank(symbol_name_tsv, plainto_tsquery('english', $1)) AS rank
FROM repo_nodes
WHERE symbol_name_tsv @@ plainto_tsquery('english', $1)
  AND rel_path LIKE $2
ORDER BY rank DESC LIMIT $3
```

#### Step 3: Refactor `resolve_orphans()` Pass 1

```python
def _resolve_orphan_lexical_tiered(
    db, G, node_id, symbol_name, symbol_type, total_nodes,
    max_edges=2, fts_limit=5
):
    """
    Tiered lexical resolution with IDF gating.
    Uses the existing `symbol_type` column from repo_nodes
    (populated by graph_builder.py) to determine tier eligibility.
    Returns list of (target_node_id, weight, tier) tuples.
    """
    # IDF gate
    idf = _compute_symbol_idf(db, symbol_name, total_nodes)
    
    if idf < 1.5:
        return []  # Too common — skip lexical entirely
    
    edges_found = []
    remaining = max_edges
    
    # Determine eligible tiers based on symbol_type (existing column)
    if symbol_type in ("constructor", "field", "variable"):
        return []  # No lexical for these
    
    max_tier = {
        "module": 4, "class": 4, "interface": 4, "struct": 4,
        "function": 3,
        "method": 2,
    }.get(symbol_type, 2)
    
    if idf < 3.0:
        max_tier = min(max_tier, 2)  # Moderate frequency: T1+T2 only
    
    # --- T1: Exact qualified name match ---
    if remaining > 0 and max_tier >= 1:
        t1_hits = db.search_fts_by_column(
            query=symbol_name, column="symbol_name", limit=fts_limit
        )
        for hit in t1_hits:
            if hit["node_id"] != node_id and remaining > 0:
                edges_found.append((hit["node_id"], 1.0 * SYNTHETIC_WEIGHT_FLOOR, "T1"))
                remaining -= 1
    
    # --- T2: Same-directory locality ---
    if remaining > 0 and max_tier >= 2:
        orphan_path = G.nodes[node_id].get("rel_path", "")
        for prefix in _expanding_prefixes(orphan_path):
            t2_hits = db.search_fts_with_path(
                query=symbol_name, path_prefix=prefix, limit=fts_limit
            )
            for hit in t2_hits:
                if hit["node_id"] != node_id and hit["node_id"] not in {e[0] for e in edges_found}:
                    if remaining > 0:
                        edges_found.append((hit["node_id"], 0.7 * SYNTHETIC_WEIGHT_FLOOR, "T2"))
                        remaining -= 1
            if remaining <= 0:
                break
    
    # --- T3: Docstring cross-reference ---
    if remaining > 0 and max_tier >= 3:
        orphan_docstring = _get_node_docstring(db, G, node_id)
        backtick_refs = _BACKTICK_REF_RE.findall(orphan_docstring)
        for ref in backtick_refs:
            t3_hits = db.search_fts_by_column(
                query=ref, column="symbol_name", limit=3
            )
            for hit in t3_hits:
                if hit["node_id"] != node_id and hit["node_id"] not in {e[0] for e in edges_found}:
                    if remaining > 0:
                        edges_found.append((hit["node_id"], 0.4 * SYNTHETIC_WEIGHT_FLOOR, "T3"))
                        remaining -= 1
    
    # --- T4: Broad FTS context (current behavior, IDF-gated) ---
    if len(edges_found) == 0 and max_tier >= 4:
        t4_hits = db.search_fts5(query=symbol_name, limit=fts_limit)
        for hit in t4_hits:
            if hit["node_id"] != node_id and remaining > 0:
                edges_found.append((hit["node_id"], 0.2 * SYNTHETIC_WEIGHT_FLOOR, "T4"))
                remaining -= 1
    
    return edges_found
```

#### Step 4: Read `symbol_type` from Existing Node Attributes

The `repo_nodes` table **already has** `symbol_type` (TEXT) and `is_architectural` (BOOLEAN) columns, populated during graph building. No schema migration is needed.

In `graph_topology.py` → `resolve_orphans()`, read the existing column when calling the tiered resolver:

```python
# Read existing symbol_type from graph node attributes or DB
symbol_type = G.nodes[node_id].get("symbol_type", "unknown")

edges = _resolve_orphan_lexical_tiered(
    db, G, node_id, symbol_name,
    symbol_type=symbol_type,
    total_nodes=total_nodes,
)
```

> **Important**: `symbol_type` values are already set by `graph_builder.py` → `_add_symbol_node()` and stored in both the NetworkX graph attributes and the `repo_nodes` table. Values include: `"class"`, `"function"`, `"method"`, `"module"`, `"interface"`, `"struct"`, `"constructor"`, `"field"`, `"variable"`, etc.

### 2.5 Backend-Specific Considerations (SQLite vs PostgreSQL)

| Aspect | SQLite (FTS5) | PostgreSQL (tsvector) |
|--------|--------------|----------------------|
| Column-specific search | `{column} : {query}` syntax in FTS5 MATCH | Separate `tsvector` columns or use `setweight()` zones |
| IDF computation | `SELECT COUNT(*) FROM repo_fts WHERE symbol_name MATCH ?` | `SELECT COUNT(*) FROM repo_nodes WHERE symbol_name_tsv @@ plainto_tsquery(...)` |
| Path-filtered search | JOIN `repo_fts` with `repo_nodes` on `node_id` | Direct `WHERE rel_path LIKE ...` on same table |
| BM25 weights | Built-in: `bm25(repo_fts, 10.0, 4.0, 2.0, 1.0)` | Must use `ts_rank_cd()` with weight array `{0.1, 0.2, 0.4, 1.0}` (D,C,B,A zones) |
| Performance concern | FTS5 COUNT is a full scan of the index — cache results | `COUNT(*)` with tsquery can use GIN index efficiently |

**Recommendation**: Compute IDF scores **once** at the start of `resolve_orphans()` for all unique symbol names across orphans, then cache in a dict. This avoids N × M queries.

```python
# Pre-compute IDF cache at start of resolve_orphans()
orphan_symbol_names = set()
for oid in orphan_ids:
    sn = _get_symbol_name(db, G, oid)
    if sn:
        orphan_symbol_names.add(sn)

idf_cache = {}
total_nodes = G.number_of_nodes()
for sn in orphan_symbol_names:
    idf_cache[sn] = _compute_symbol_idf(db, sn, total_nodes)
```

---

## 3. Problem 2 — Node Disambiguation for Python REST API Classes

### 3.1 Current Behavior Analysis

**Location**: `graph_builder.py` → `_get_local_qualified_name()` (lines 1449–1489)

Node IDs follow the pattern: `{language}::{file_stem}::{local_qualified_name}`

The `local_qualified_name` is derived from `full_name` by stripping the module prefix:

```python
# full_name = "users_api.API.get" → local_name = "API.get"
# node_id = "python::users_api::API.get"
```

For hash-based disambiguation (collision handling):

```python
# If node_id already exists in graph, append hash suffix:
# "python::users_api::API.get" → "python::users_api::API.get_a1b2c3"
if node_id in G:
    hash_suffix = hashlib.sha256(
        f"{file_path}:{symbol.start_line}".encode()
    ).hexdigest()[:6]
    node_id = f"{node_id}_{hash_suffix}"
```

### 3.2 The Flask/Django REST Pattern

In Python REST frameworks (Flask-RESTful, Django REST Framework, FastAPI), a common pattern is:

```
project/
├── api/
│   ├── users.py       → class UsersAPI(Resource): def get(), post(), delete()
│   ├── orders.py      → class OrdersAPI(Resource): def get(), post(), delete()
│   ├── products.py    → class ProductsAPI(Resource): def get(), post(), delete()
│   └── auth.py        → class AuthAPI(Resource): def get(), post()
```

**Current node IDs produced**:

| File | Symbol | Node ID | FTS `symbol_name` |
|------|--------|---------|-------------------|
| `users.py` | `UsersAPI.get` | `python::users::UsersAPI.get` | `UsersAPI.get` |
| `orders.py` | `OrdersAPI.get` | `python::orders::OrdersAPI.get` | `OrdersAPI.get` |

When the class is named identically across modules (common in some frameworks):

| File | Symbol | Node ID | FTS `symbol_name` |
|------|--------|---------|-------------------|
| `users.py` | `API.get` | `python::users::API.get` | `API.get` |
| `orders.py` | `API.get` | `python::orders::API.get` | `API.get` |
| `products.py` | `API.get` | `python::products::API.get` | `API.get` |

**Node IDs are unique** (different `file_stem`), but **FTS `symbol_name` is identical** (`API.get`). This means:
1. Orphan resolution for `python::users::API.get` finds `python::orders::API.get` and `python::products::API.get` as FTS hits
2. Lexical edges cross-link all three, merging them into one cluster
3. The resulting wiki page conflates three unrelated REST endpoints

**Note**: Even though the node IDs are disambiguated by file_stem, the symbol_name stored in FTS is `API.get` for all three — and that's what Pass 1 queries.

### 3.3 Proposed Solution: Context-Enriched Symbol Names + symbol_type Metadata

#### A. Enrich `symbol_name` in FTS with Module Context

For **methods nested inside classes**, the FTS `symbol_name` should include the module (file_stem) as a qualifying prefix when the parent class name is generic:

```python
GENERIC_CLASS_NAMES = frozenset({
    "API", "Resource", "View", "ViewSet", "Endpoint",
    "Handler", "Controller", "Service", "Manager",
    "Mixin", "Base", "Abstract", "Interface",
})

def _compute_fts_symbol_name(symbol, file_stem: str) -> str:
    """
    Compute the symbol_name to store in FTS for search purposes.
    
    For methods inside generic class names, prefix with module name
    to prevent cross-pollution during orphan resolution.
    """
    local_name = symbol.local_qualified_name  # e.g., "API.get"
    parts = local_name.split(".")
    
    if len(parts) >= 2:
        parent_class = parts[0]
        if parent_class in GENERIC_CLASS_NAMES:
            # Prefix with module: "users.API.get" instead of "API.get"
            return f"{file_stem}.{local_name}"
    
    return local_name
```

**Result**:

| File | Old FTS `symbol_name` | New FTS `symbol_name` |
|------|----------------------|----------------------|
| `users.py` → `API.get` | `API.get` | `users.API.get` |
| `orders.py` → `API.get` | `API.get` | `orders.API.get` |
| `users.py` → `UsersAPI.get` | `UsersAPI.get` | `UsersAPI.get` (unchanged — not generic) |

#### B. Detect REST Framework Decorators and Set `symbol_type` / `is_architectural`

In the Python parser (`parsers/python_parser.py`), detect common REST framework patterns and use the **existing** `symbol_type` and `is_architectural` columns (no schema changes needed):

```python
REST_DECORATORS = {
    "route", "api_view", "action",
    "get", "post", "put", "patch", "delete",  # FastAPI
    "app.route", "app.get", "app.post",        # Flask
}

REST_BASE_CLASSES = {
    "Resource", "MethodView", "View",           # Flask-RESTful
    "APIView", "ViewSet", "ModelViewSet",       # DRF
    "GenericAPIView", "ListAPIView",
    "HTTPEndpoint",                             # Starlette
}

def _detect_rest_endpoint(cls_node, method_node) -> bool:
    """Detect if a class+method is a REST endpoint.
    
    Returns True if the method should have symbol_type='rest_endpoint'
    (stored in the existing symbol_type column in repo_nodes).
    """
    base_classes = {b.name for b in cls_node.bases}
    if base_classes & REST_BASE_CLASSES:
        return True
    
    decorators = {d.name for d in method_node.decorators}
    if decorators & REST_DECORATORS:
        return True
    
    return False
```

When a REST endpoint is detected, set the **existing** columns:
- `symbol_type = "rest_endpoint"` (extends the current value set)
- `is_architectural = False` (REST methods should not be treated as architectural symbols)

REST endpoints with `symbol_type="rest_endpoint"` would be restricted to **T1+T2 only** in the tiered lexical resolution (same as regular methods), preventing cross-module FTS pollution.

#### C. Hash Suffix Readability Improvement

The current hash suffix (`_a1b2c3`) is opaque. While this doesn't affect connectivity directly, it affects debuggability. Consider replacing with a more readable suffix:

```python
# Current: "python::users::API.get_a1b2c3"
# Proposed: "python::users::API.get" (no suffix needed if file_stem differs)
# Collision case: "python::users::API.get@L42" (line number suffix)
```

Since collisions within the same `{language}::{file_stem}` are rare (would require two symbols with identical qualified names in the same file), the hash suffix mainly fires for edge cases. Adding the line number is more debuggable than a hex hash.

### 3.4 Implementation Plan

| Step | File | Change | Effort |
|------|------|--------|--------|
| 1 | `graph_builder.py` | Add `_compute_fts_symbol_name()` helper | Small |
| 2 | `graph_builder.py` → `_add_symbol_node()` | Use enriched symbol_name when populating `repo_nodes` and FTS | Small |
| 3 | `parsers/python_parser.py` | Add `_detect_rest_endpoint()` for REST patterns | Medium |
| 4 | `graph_builder.py` | Set `symbol_type="rest_endpoint"` for detected REST methods (uses existing column) | Small |
| 5 | `graph_topology.py` → `resolve_orphans()` | Read existing `symbol_type` and pass to tiered resolver | Small |
| 6 | Tests | Unit tests for generic class detection and enriched FTS names | Medium |

### 3.5 Impact on Connectivity

**Before fix**: 
- 3 modules × `API.get` → 6 spurious lexical edges (each orphan finds the other 2)
- Leiden merges all 3 into one cluster → incoherent page mixing users/orders/products

**After fix**:
- FTS for `users.API.get` → matches only `users.API.get` (self) → no spurious edges
- Each REST module stays in its own cluster (connected by directory proximity or imports if they exist)
- If they genuinely share code (e.g., common base class), structural edges (inheritance) provide the correct connectivity

---

## 4. Problem 3 — Connectivity Tracing & Diagnostics

### 4.1 Current Observability Gaps

The Phase 2 pipeline (`run_phase2()`) runs 6 steps:

```
1. apply_edge_weights()          → weight structural edges by 1/in_degree
2. find_orphans()                → identify 0-degree nodes
3. resolve_orphans()             → 3-pass cascade (directory, FTS, vector)
4. inject_doc_edges()            → link documentation nodes
5. add_directory_proximity()     → same-directory edges
6. bridge_disconnected_components() → inter-component bridges
```

**What's missing**:

1. **No per-edge provenance**: Edges have `created_by` attribute (`"fts5_lexical"`, `"directory_proximity"`, `"component_bridging"`), but there's no record of:
   - Which tier created it (T1 vs T4)
   - What the original query term was
   - What the BM25/ts_rank score was
   - What the IDF of the query term was at creation time

2. **No component evolution log**: After each step, there's no snapshot of how many components exist, how many orphans remain, or what the largest component size is.

3. **No way to query "why are these two nodes connected?"**: Given two node IDs, there's no function to trace all paths and explain which edges (structural vs synthetic) connect them.

4. **No aggregate statistics**: How many lexical edges were created? How many were T1 vs T4? What percentage of orphans were resolved by each pass?

### 4.2 Proposed Solution: Structured Edge Provenance

#### A. Enrich Edge Metadata

Every synthetic edge should carry a provenance dict:

```python
G.add_edge(source, target,
    rel_type="lexical_link",
    edge_class="lexical",
    weight=computed_weight,
    created_by="tiered_lexical",
    provenance={
        "tier": "T2",
        "query_term": "UsersAPI",
        "match_score": 0.85,       # BM25 or ts_rank score
        "idf": 4.2,                # IDF of query term at creation time
        "path_prefix": "api/",     # For T2 locality matches
        "orphan_symbol_type": "class",
    }
)
```

#### B. Phase 2 Step-Level Statistics

Add a `Phase2Stats` dataclass that accumulates after each step:

```python
@dataclass
class Phase2Stats:
    step_name: str
    components_before: int
    components_after: int
    orphans_before: int
    orphans_after: int
    edges_added: int
    edges_by_tier: Dict[str, int]  # {"T1": 5, "T2": 12, ...}
    largest_component: int
    elapsed_ms: float
```

Return from `run_phase2()`:
```python
def run_phase2(db, G, ...) -> Tuple[nx.MultiDiGraph, List[Phase2Stats]]:
    stats = []
    # After each step, capture stats
    ...
    return G, stats
```

#### C. Path Explanation Function

```python
def explain_connectivity(G, node_a: str, node_b: str) -> List[Dict]:
    """
    Find all simple paths between two nodes (up to length 5)
    and explain each edge along each path.
    """
    explanations = []
    for path in nx.all_simple_paths(G, node_a, node_b, cutoff=5):
        path_edges = []
        for i in range(len(path) - 1):
            edge_data = G.get_edge_data(path[i], path[i+1])
            for key, data in edge_data.items():
                path_edges.append({
                    "from": path[i],
                    "to": path[i+1],
                    "edge_class": data.get("edge_class", "unknown"),
                    "rel_type": data.get("rel_type", "unknown"),
                    "weight": data.get("weight", 0),
                    "created_by": data.get("created_by", "unknown"),
                    "provenance": data.get("provenance", {}),
                })
        explanations.append({"path": path, "edges": path_edges})
    return explanations
```

### 4.3 Diagnostic Query Toolkit

#### SQLite Diagnostic Queries

```sql
-- 1. Find all lexical edges and their provenance
SELECT source_id, target_id, weight, json_extract(metadata, '$.provenance.tier') AS tier,
       json_extract(metadata, '$.provenance.query_term') AS query_term,
       json_extract(metadata, '$.provenance.idf') AS idf
FROM repo_edges
WHERE edge_class = 'lexical'
ORDER BY idf ASC;  -- Most suspicious (lowest IDF) first

-- 2. Find "hub" nodes created by lexical flooding
SELECT target_id, COUNT(*) AS lexical_in_degree
FROM repo_edges
WHERE edge_class = 'lexical'
GROUP BY target_id
HAVING COUNT(*) > 5
ORDER BY lexical_in_degree DESC;

-- 3. Component sizes after Phase 2
-- (requires storing component_id on nodes)
SELECT component_id, COUNT(*) AS size
FROM repo_nodes
WHERE component_id IS NOT NULL
GROUP BY component_id
ORDER BY size DESC;

-- 4. Orphan resolution effectiveness
SELECT 
    SUM(CASE WHEN in_degree = 0 AND out_degree = 0 THEN 1 ELSE 0 END) AS remaining_orphans,
    COUNT(*) AS total_nodes,
    ROUND(100.0 * SUM(CASE WHEN in_degree = 0 AND out_degree = 0 THEN 1 ELSE 0 END) / COUNT(*), 1) AS orphan_pct
FROM (
    SELECT n.node_id,
           (SELECT COUNT(*) FROM repo_edges e WHERE e.target_id = n.node_id) AS in_degree,
           (SELECT COUNT(*) FROM repo_edges e WHERE e.source_id = n.node_id) AS out_degree
    FROM repo_nodes n
);
```

#### PostgreSQL Diagnostic Queries

```sql
-- 1. Find all lexical edges and their provenance
SELECT source_id, target_id, weight, 
       (metadata->>'provenance'->>'tier') AS tier,
       (metadata->>'provenance'->>'query_term') AS query_term,
       (metadata->'provenance'->>'idf')::float AS idf
FROM repo_edges
WHERE edge_class = 'lexical'
ORDER BY (metadata->'provenance'->>'idf')::float ASC NULLS LAST;

-- 2. Find "hub" nodes created by lexical flooding  
SELECT target_id, COUNT(*) AS lexical_in_degree
FROM repo_edges
WHERE edge_class = 'lexical'
GROUP BY target_id
HAVING COUNT(*) > 5
ORDER BY lexical_in_degree DESC;

-- 3. Symbol frequency analysis (IDF pre-computation)
SELECT symbol_name, COUNT(*) AS doc_freq,
       LN(total.cnt::float / (COUNT(*) + 1)) AS idf
FROM repo_nodes,
     (SELECT COUNT(*) AS cnt FROM repo_nodes) total
GROUP BY symbol_name, total.cnt
HAVING COUNT(*) > 3
ORDER BY doc_freq DESC;

-- 4. REST endpoint disambiguation check
SELECT node_id, symbol_name, rel_path, symbol_type, is_architectural
FROM repo_nodes
WHERE symbol_name LIKE '%.get' OR symbol_name LIKE '%.post' OR symbol_name LIKE '%.delete'
ORDER BY symbol_name, rel_path;
```

---

## 5. Cross-Cutting: SQLite vs PostgreSQL Backend Parity

### Implementation Differences by Feature

| Feature | SQLite Implementation | PostgreSQL Implementation | Shared Logic |
|---------|----------------------|--------------------------|-------------|
| IDF computation | `SELECT COUNT(*) FROM repo_fts WHERE symbol_name MATCH ?` | `SELECT COUNT(*) FROM repo_nodes WHERE symbol_name_tsv @@ plainto_tsquery(...)` | `_compute_symbol_idf()` wrapper |
| Column-specific FTS | FTS5 column filter syntax: `symbol_name : "query"` | Separate `tsvector` column or `setweight()` zones | Tier selection logic in `graph_topology.py` |
| Path-filtered search | `JOIN repo_fts f ON f.node_id = n.node_id WHERE n.rel_path LIKE ?` | `WHERE rel_path LIKE $1 AND name_tsv @@ ...` | `search_fts_with_path()` protocol method |
| Edge metadata storage | `repo_edges.metadata` as JSON text column | `repo_edges.metadata` as JSONB column | Same schema, different JSON handling |
| `symbol_type` column | **Already exists** in `repo_nodes` — no migration needed | **Already exists** in `repo_nodes` — no migration needed | Both backends already populate `symbol_type` and `is_architectural` |
| Provenance storage | `json_extract(metadata, '$.provenance.tier')` | `metadata->'provenance'->>'tier'` | Different JSON path syntax |

### New Protocol Methods Required

Add to `WikiStorageProtocol` (`storage/protocol.py`):

```python
class WikiStorageProtocol(Protocol):
    # ... existing methods ...
    
    def count_fts_matches(self, query: str, column: str = "symbol_name") -> int:
        """Count nodes matching query in specified FTS column."""
        ...
    
    def search_fts_by_column(
        self, query: str, column: str, limit: int = 10
    ) -> List[Dict]:
        """Search FTS restricted to a specific column."""
        ...
    
    def search_fts_with_path(
        self, query: str, path_prefix: str, limit: int = 10
    ) -> List[Dict]:
        """Search FTS with path locality filter."""
        ...
```

### Migration Strategy

> **No schema migration is required.** The `repo_nodes` table already has `symbol_type` (TEXT) and `is_architectural` (BOOLEAN) columns in both SQLite and PostgreSQL backends. These are populated during graph building by `graph_builder.py` → `_add_symbol_node()`.
>
> The only change needed is to extend the set of `symbol_type` values to include `"rest_endpoint"` for detected REST API methods. This is a data-level change, not a schema change.

---

## 6. Action Items Summary

### Priority 1 — Lexical Edge Flooding Fix (High Impact)

| # | Action | File(s) | Effort | Risk |
|---|--------|---------|--------|------|
| 1.1 | Add `count_fts_matches()` to protocol + both backends | `protocol.py`, `sqlite.py`, `postgres.py` | Small | Low |
| 1.2 | Implement IDF cache computation at start of `resolve_orphans()` | `graph_topology.py` | Small | Low |
| 1.3 | Add IDF gating logic (skip if IDF < 1.5) | `graph_topology.py` | Small | Low — conservative threshold |
| 1.4 | Implement 4-tier cascade replacing flat FTS query | `graph_topology.py` | Medium | Medium — needs thorough testing |
| 1.5 | Add `search_fts_by_column()` + `search_fts_with_path()` to both backends | `sqlite.py`, `postgres.py` | Medium | Low |
| 1.6 | Read existing `symbol_type` column in tiered resolver (no schema change) | `graph_topology.py` | Small | Low |
| 1.7 | Implement `symbol_type`-based tier eligibility gating | `graph_topology.py` | Small | Low |

### Priority 2 — Node Disambiguation (Medium Impact)

| # | Action | File(s) | Effort | Risk |
|---|--------|---------|--------|------|
| 2.1 | Add `GENERIC_CLASS_NAMES` set and `_compute_fts_symbol_name()` | `graph_builder.py` | Small | Low |
| 2.2 | Use enriched symbol_name when populating FTS | `graph_builder.py` | Small | Low — backward compatible |
| 2.3 | Add REST framework detection in Python parser | `parsers/python_parser.py` | Medium | Low |
| 2.4 | Set `symbol_type="rest_endpoint"` for detected REST methods (existing column) | `parsers/python_parser.py`, `graph_builder.py` | Small | Low |

### Priority 3 — Connectivity Diagnostics (Medium Impact)

| # | Action | File(s) | Effort | Risk |
|---|--------|---------|--------|------|
| 3.1 | Add `provenance` dict to all synthetic edge creation | `graph_topology.py` | Medium | Low |
| 3.2 | Implement `Phase2Stats` dataclass and step-level capture | `graph_topology.py` | Small | Low |
| 3.3 | Implement `explain_connectivity()` function | `graph_topology.py` | Small | Low |
| 3.4 | Add diagnostic SQL queries to documentation / admin tools | docs / admin CLI | Small | None |
| 3.5 | Store edge metadata with provenance in `repo_edges` | `sqlite.py`, `postgres.py` | Medium | Low |

---

## 7. Test Strategy

### Unit Tests

```python
# test_tiered_lexical_resolution.py

class TestIdfGating:
    """Verify IDF gate blocks high-frequency terms."""
    
    def test_common_term_blocked(self):
        """Symbol 'get' appearing in >5% of nodes → IDF < 1.5 → no lexical edges."""
        # Setup: 100 nodes, 20 contain 'get' in symbol_name
        # Assert: resolve_orphan_lexical_tiered returns []
    
    def test_rare_term_allowed(self):
        """Symbol 'DatabaseMigrationRunner' in 1 node → high IDF → T1-T4 eligible."""
        # Assert: edges created
    
    def test_moderate_term_restricted(self):
        """Symbol 'config' in 8% of nodes → 1.5 < IDF < 3.0 → T1+T2 only."""
        # Assert: no T3/T4 edges


class TestTierCascade:
    """Verify tier precedence and weight graduation."""
    
    def test_t1_exact_match_highest_weight(self):
        """Exact qualified name match → weight = 1.0 * SYNTHETIC_WEIGHT_FLOOR."""
    
    def test_t2_locality_bias(self):
        """Same-directory match → weight = 0.7 * SYNTHETIC_WEIGHT_FLOOR."""
    
    def test_t4_only_when_no_other_matches(self):
        """Broad FTS only fires when T1+T2+T3 found zero edges."""
    
    def test_method_symbol_type_max_t2(self):
        """symbol_type='method' → max tier T2, even if IDF is high."""


class TestRestApiDisambiguation:
    """Verify Python REST API classes don't cross-pollinate in FTS."""
    
    def test_generic_class_name_enriched(self):
        """API.get in users.py → FTS symbol_name = 'users.API.get'."""
    
    def test_non_generic_class_unchanged(self):
        """UsersAPI.get → FTS symbol_name stays 'UsersAPI.get'."""
    
    def test_no_cross_module_lexical_edges(self):
        """Orphan resolution for users.API.get does NOT create edge to orders.API.get."""


class TestConnectivityDiagnostics:
    """Verify edge provenance and path explanation."""
    
    def test_provenance_stored_on_lexical_edge(self):
        """Lexical edge has provenance dict with tier, query_term, idf."""
    
    def test_explain_connectivity_finds_path(self):
        """explain_connectivity(G, A, B) returns path with edge details."""
    
    def test_phase2_stats_captured(self):
        """run_phase2() returns list of Phase2Stats, one per step."""
```

### Integration Tests

```python
class TestLexicalFloodingRegression:
    """End-to-end test with a repo that has 'shared' in many file names."""
    
    def test_shared_term_does_not_create_mega_cluster(self):
        """
        Given: 50-node repo where 10 nodes have 'shared' in symbol_name
        When: Full pipeline (parse → graph → topology → clustering)
        Then: No cluster has > 20 nodes (proves no mega-cluster from 'shared' flooding)
        """

class TestRestApiRepo:
    """End-to-end test with Flask-RESTful style project."""
    
    def test_each_api_module_in_own_cluster(self):
        """
        Given: 5 modules each with class API(Resource): get(), post(), delete()
        When: Full pipeline
        Then: At least 3 distinct clusters (not all merged into one)
        """
```

### Backend Parity Tests

```python
@pytest.mark.parametrize("backend", ["sqlite", "postgres"])
class TestBackendParity:
    """Ensure identical behavior across both backends."""
    
    def test_count_fts_matches_same_result(self, backend):
        """Same repo → same count for a given query on both backends."""
    
    def test_search_fts_by_column_same_results(self, backend):
        """Column-specific search returns same node_ids on both backends."""
    
    def test_idf_computation_equivalent(self, backend):
        """IDF values agree within 0.1 tolerance between backends."""