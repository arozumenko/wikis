# Graph Quality Improvements: Lexical Orphans, Connectivity, and Node Disambiguation

> **Status:** Draft review — 2026-04-23
> **Scope:** `backend/app/core/graph_topology.py`, `backend/app/core/code_graph/graph_builder.py`, `backend/app/core/parsers/python_parser.py`, `backend/app/core/storage/sqlite.py`

This report is an actionable audit of three related weaknesses in the wiki code graph:

1. **Lexical orphan resolution** floods the graph with edges from common words.
2. **Connectivity tracing** treats all edges as equal, even synthetic ones.
3. **Node disambiguation** for Python classes decorated as framework resources (Flask / flask-restful / FastAPI) is unstable and, in some cases, incorrect.

Each section identifies the current behaviour, quantifies the risk, and lists prioritised actions with code references so work can start without further investigation.

---

## Executive Summary

| Area | Current state | Risk | Recommended action tier |
|------|---------------|------|------------------------|
| Lexical orphan edges | FTS5 BM25 on `symbol_name` with 2‑char min, no stopword filter, no architectural gating | High — common tokens (`shared`, `manager`, `base`) flood edges; every orphan gets up to `max_lexical_edges` (default 2) blind edges | A1–A3 = quick wins; A4–A6 = medium |
| Connectivity | BFS is edge‑class‑agnostic; `bridge_disconnected_components` forces 1 component regardless of naturalness | Medium — synthetic edges inflate reachability; bridges obscure real cluster structure | B1–B3 = medium |
| Node IDs — `hash(file_path) & 0xFFFF` | Python `hash()` of strings is **salted per interpreter run** (PYTHONHASHSEED) | **Critical** — node IDs are not stable between indexing runs, breaking incremental snapshots and cross-run comparisons | C1 = immediate fix |
| Flask-style class split across modules | Each file creates its own `class API(Resource)`; methods only DEFINE‑linked within that file; cross-file methods become pseudo-orphans | Medium — routing calls resolve to wrong method or miss; duplicate `API` class nodes inflate graph | C2–C4 = strategic |

---

## 1. Lexical Orphan Resolution

### 1.1 Current implementation

Pipeline entry — `backend/app/core/graph_topology.py:354-629` — `resolve_orphans()` runs a three-pass cascade:

| Pass | File:lines | Criterion | Edge type |
|------|------------|-----------|-----------|
| 1 — FTS5 lexical | `graph_topology.py:434-475` | BM25 on `symbol_name` / `signature` / `docstring` / `source_text` | `lexical_link` (`edge_class="lexical"`) |
| 2 — Semantic vector | `graph_topology.py:477-602` | Cosine distance < `0.15` on embedding, with expanding path prefix | `semantic_link` |
| 3 — Directory proximity | `graph_topology.py:604-615`, `_resolve_orphans_by_directory()` at `graph_topology.py:271-351` | Highest-degree connected node in same/parent dir | `directory_link` |

BM25 weights — `backend/app/core/storage/sqlite.py:786-789`:

```
bm25(repo_fts, 10.0, 4.0, 2.0, 1.0)
         symbol_name  signature  docstring  source_text
```

Gating in Pass 1 — `graph_topology.py:451-452`:

```python
if not symbol_name or len(symbol_name) < 2:
    continue
```

Per-orphan cap — `graph_topology.py:458`:

```python
for hit in fts_hits[:max_lexical_edges]:   # default 2
```

A stricter gate already exists for framework reference discovery — `backend/app/core/cluster_expansion.py:117-118`:

```python
# Minimum symbol name length for FTS search (shorter names produce noise).
_MIN_FTS_NAME_LEN = 4
```

…but **it is not applied to orphan resolution**. This is the first inconsistency.

### 1.2 Why it floods

1. **No stopword / IDF filter.** A class called `Shared`, a module `shared_utils.py`, a function `make_shared()` all match any orphan whose symbol name contains "shared". BM25 sorts them but does not discard them.
2. **Architectural gate missing.** The graph has a concept of `ARCHITECTURAL_SYMBOLS` (see `backend/app/core/code_graph/constants.py:189-208`) — classes, interfaces, structs, enums, traits, standalone functions, constants, macros, type aliases, doc symbols. Variables, fields, properties are excluded from the *graph* but **still indexed in FTS**, so they can be targets of lexical matches from architectural orphans. Symmetrically, a low-value orphan can spray edges into the architectural layer.
3. **Minimum name length = 2.** This lets `id`, `db`, `io`, `fn`, `op` into the match. The `_MIN_FTS_NAME_LEN = 4` already used in `cluster_expansion.py` is the correct floor.
4. **Identifier atomicity.** `SharedCache`, `sharedCache`, `shared_cache`, `SHARED_CACHE` are treated as different tokens by FTS5's default tokenizer. So the IDF of any sub-token (`cache`, `shared`) is never computed.
5. **No tier awareness.** A match against a symbol with the exact same `rel_path` prefix is worth far more than a match in a completely unrelated package — but the current code treats them identically.
6. **No edge weight.** The edge is boolean. Downstream BFS / clustering can't suppress weak edges because they carry no confidence.

### 1.3 Proposed tiered approach (the user's ask)

Introduce **tiered lexical matching** with descending confidence, and **only fire for architectural symbol types on both sides**:

| Tier | Signal | Confidence | When |
|------|--------|------------|------|
| **L0** | Exact-name match in **same directory** | 0.90 | always for arch symbols |
| **L1** | Exact-name match with **shared rel_path prefix** ≥ 2 segments | 0.70 | always for arch symbols |
| **L2** | Exact-name match elsewhere, **rare token** (IDF above threshold) | 0.50 | only if no L0/L1 hit |
| **L3** | Docstring token co-occurrence — at least K rare tokens in common | 0.30 | only if no L0–L2 hit |
| **L4** | Source-text co-occurrence of rare tokens (last resort) | 0.15 | behind an env flag, off by default |

Rules:

- Tiers are **exclusive** — a resolved orphan does not cascade into weaker tiers.
- A tier's edge carries `weight=<confidence>` and `tier="L0"|"L1"|...` attributes so reachability / clustering can filter.
- Architectural-type gating: the source (orphan) **and** the target hit must be in `ARCHITECTURAL_SYMBOLS`. Methods remain linked through their parent `DEFINES` edges (they're already not in `ARCHITECTURAL_SYMBOLS` — `priority=3`, `backend/app/core/code_graph/symbol_priority.py`).
- Tier L3+ requires identifier splitting (camelCase + snake_case + kebab) and a pre-computed IDF table so that `shared`, `manager`, `handler`, `utils`, `helper`, `base`, `common`, `default`, `config`, `service`, `app`, `core`, `main` contribute ~zero weight.

### 1.4 Action items — Lexical

| ID | Action | File(s) | Effort | Priority |
|----|--------|---------|--------|----------|
| **A1** | Raise min symbol-name length for Pass 1 from 2 → 4 (align with `cluster_expansion._MIN_FTS_NAME_LEN`). Expose `WIKI_LEXICAL_MIN_NAME_LEN` env var. | `graph_topology.py:451` | S | P0 |
| **A2** | Gate Pass 1 to `ARCHITECTURAL_SYMBOLS` on both sides. Skip if orphan OR hit is `variable`/`field`/`property`/`parameter`. | `graph_topology.py:434-475`, use constants from `code_graph/constants.py:189` | S | P0 |
| **A3** | Build a per-repo **token frequency / IDF table** once per indexing run (tokenise symbol_name + rel_path segments via camelCase/snake_case split). Persist in `UnifiedWikiDB` with TTL = indexing session. | new helper, e.g. `code_graph/lexical_stats.py`; invoke in `wiki_service` pipeline before `resolve_orphans` | M | P0 |
| **A4** | Introduce **tier ladder L0–L4** as in §1.3. Emit edges with `tier`, `weight`, and `matched_token` attributes. | `graph_topology.py:434-475` → new `_resolve_lexical_tiered()` | M | P1 |
| **A5** | Maintain a **stopword/boilerplate list** (`shared`, `manager`, `handler`, `utils`, `helper`, `base`, `common`, `default`, `config`, `service`, `main`, `app`, `core`, `impl`, `factory`, `builder`, `client`, `server`, `model`, `type`, `data`, `info`, `context`). Configurable via env or JSON. Tokens in this list are excluded from IDF matching entirely. | `code_graph/lexical_stats.py`; sane default list in constants | S | P1 |
| **A6** | Emit lexical-flooding telemetry: per indexing run, log top-N nodes that received > K lexical edges. Raise a warning event over SSE when a symbol spawns > `WIKI_LEXICAL_FLOOD_THRESHOLD` (default 10) edges across the repo. | `graph_topology.py` after Pass 1; `services/wiki_service.py` progress events | S | P1 |
| **A7** | Skip Pass 1 entirely for symbols whose **in-file DEFINES parent is non-orphan**. If the class or module of an orphan method is already well-connected, the method's lexical edges are noise. | `graph_topology.py:432` (preflight) | S | P1 |
| **A8** | Treat `fts_limit` and `max_lexical_edges` as **per-tier knobs**: L0 can take up to 5 edges (high confidence), L3 at most 1, L4 exactly 1. | config block in `graph_topology.py` | S | P2 |
| **A9** | Consider switching FTS5 to the `trigram` tokenizer or adding a custom tokenizer that splits identifiers. Evaluate impact on index size. | `sqlite.py:749-809` + schema migration | L | P3 |

---

## 2. Connectivity Tracing

### 2.1 Current implementation

Two systems run over the graph:

**BFS reachability** — `backend/app/core/code_graph/graph_query_service.py:920-985`. Traverses predecessors/successors up to `max_depth`, optionally filtered by `edge_types` (relationship-type list). Does not consider edge class (`structural` vs `lexical` vs `semantic` vs `directory` vs `component_bridge`).

**Weakly-connected-component bridging** — `backend/app/core/graph_topology.py:1012-1120`, function `bridge_disconnected_components()`. Calls `nx.weakly_connected_components(G)`, then iteratively connects the largest node in each smaller component to the largest node of a directory-similar peer with bidirectional `component_bridge` edges.

### 2.2 Issues

1. **Reachability blind to edge class.** A structural chain `file → class → method → call → class` is not distinguishable from a two-hop lexical link. Clusterers, research engines, and outline planners consume reachability without knowing which edges are synthetic.
2. **Bridging is aggressive.** Any disconnected component is bridged, regardless of size. A one-node component is bridged to the giant component via directory similarity, which creates a spurious edge that later looks like a real connection.
3. **No confidence propagation.** Even if lexical edges gain a `weight`/`tier` attribute (A4), no downstream consumer will read it unless the BFS is taught to filter/weight.
4. **No "structural-only" view.** The graph is used both for semantic retrieval (where lexical/semantic is useful) and for architectural analysis (where it's misleading). A single view does both.

### 2.3 Action items — Connectivity

| ID | Action | File(s) | Effort | Priority |
|----|--------|---------|--------|----------|
| **B1** | Add `edge_classes` / `min_confidence` parameters to `GraphQueryService.filter_by_reachability()` and `_iter_edge_dicts()`. Default behaviour unchanged, but callers can request `{structural, hyperlink}` only. | `graph_query_service.py:920-985` | S | P0 |
| **B2** | Introduce a "**structural reachability**" helper (`structural_reachability(anchor)`) that traverses only `{defines, calls, inherits, imports, decorates}` edges. Use it for architecture diagrams, page planning, and cluster-health metrics. | `graph_query_service.py` new method | S | P0 |
| **B3** | Gate `bridge_disconnected_components` on **component size**: do not bridge components below `WIKI_BRIDGE_MIN_COMPONENT_SIZE` (default 3). Small components are more informative as orphans than as force-connected peers. | `graph_topology.py:1012-1120` | S | P1 |
| **B4** | Tag bridge edges explicitly (`edge_class="component_bridge"` is already set) and **exclude them by default** from reachability. Let the outline planner opt in if needed. | `graph_query_service.py:927-985` + `graph_topology.py:1070-1109` | S | P1 |
| **B5** | Emit **connectivity health metrics** as part of the indexing progress event: `{structural_edges, lexical_edges, semantic_edges, directory_edges, bridge_edges, components_before, components_after, orphans_remaining}`. | `wiki_service.py` progress payload | S | P1 |
| **B6** | Propagate edge weights into cluster scoring. Any edge with `tier=L3` or `tier=L4` should count as ≤ 0.3 of a structural edge in modularity computations. | downstream consumers (cluster planner) | M | P2 |

---

## 3. Node Disambiguation

### 3.1 Current node ID scheme

`backend/app/core/code_graph/graph_builder.py:1508-1525`:

```python
local_qualified_name = self._get_local_qualified_name(symbol, file_name)
node_id = f"{language}::{file_name}::{local_qualified_name}"

if graph.has_node(node_id):
    existing_file = graph.nodes[node_id].get('file_path', '')
    if existing_file == file_path:
        continue
    else:
        file_hash = hash(file_path) & 0xFFFF
        node_id = f"{node_id}_{file_hash:04x}"
```

Examples:
- `python::models::User.__init__`
- `python::api::API.get`
- `python::handlers::API.delete_3c7f` — hash suffix because `API.delete` also exists somewhere else

### 3.2 Critical bug — `hash()` is salted

Since Python 3.3, `hash(str)` is randomised per interpreter invocation unless `PYTHONHASHSEED=0` is set ([PEP 456](https://peps.python.org/pep-0456/)). Consequences:

- Node IDs with hash suffixes are **not stable between indexing runs**.
- A node stored today as `...API.delete_3c7f` can be `...API.delete_8b02` tomorrow, even with the same source.
- Any persisted snapshot (unified DB, cluster plans, downloaded wikis) silently loses identity on re-index.
- Incremental indexing, diff-based regeneration, or cross-run analytics are broken in the collision path.

The project declares unified DB and clustering goals on this very branch (`feat/unified-db-and-clustering`). This is incompatible with a randomised hash.

### 3.3 Flask / flask-restful / decorated-API case

The user's scenario: a class `API(Resource)` with methods `get`, `post`, `delete`, each defined in a **separate module file** (per-verb module convention). Current behaviour:

1. Each file is parsed independently — `python_parser.py:152-153`, `ast.parse(content, ...)`.
2. Each file's parser sees its own `class API(Resource)` and creates **its own class node**: `python::users_get::API`, `python::users_post::API`, `python::users_delete::API`.
3. First class wins its canonical ID. Second and third hit the collision path (§3.1) and get `_<hash>` suffixes.
4. `DEFINES` edges (`python_parser.py:745-811`) are intra-file only. The class node in `users_get.py` defines only `API.get` (in the same file). It does **not** define `API.post` or `API.delete`.
5. `INHERITANCE` edges to `Resource` are resolved across files by `graph_builder.py:1851-1970` (multi-strategy target search), so that part works.
6. Decorators are captured as symbol metadata (`python_parser.py:256, 322`) and as `DECORATES` relationships, but they are **not** part of the node ID.

Downstream effects:

- **Lexical resolution pulls the wrong API.** A call `api.get()` from test code can resolve to either `API.get` node, and the lookup order is essentially first-match. See resolution strategies `graph_builder.py:1861-1970`.
- **Duplicate class nodes are orphan-adjacent.** The second/third `API` class has no incoming edges apart from its own file's defines and potentially a bridge edge. Lexical resolution (§1) will then fire and create weak edges to unrelated `API`s in the project (every repo has `FooAPI`, `BarAPI`, `InternalAPI`…).
- **Methods split across modules look unrelated.** `API.get`, `API.post`, `API.delete` share a class name but belong to three separate `API` class nodes. An outline planner reading the graph sees three thin classes instead of one resource.

### 3.4 Does the hash *itself* hurt connectivity?

Directly — not much. The hash is just a suffix; outgoing/incoming edges are added to whichever node the resolver picks. Indirectly — yes:

- Non-deterministic IDs (§3.2) break incremental runs and any caching keyed on node ID.
- The hash does not carry semantic meaning, so collisions that *should* be merged (same class reopened across files) look identical to collisions that *must* stay separate (two unrelated `Logger` classes).

### 3.5 Action items — Disambiguation

| ID | Action | File(s) | Effort | Priority |
|----|--------|---------|--------|----------|
| **C1** | **Immediate:** replace `hash(file_path) & 0xFFFF` with a stable digest: `hashlib.blake2b(file_path.encode("utf-8"), digest_size=3).hexdigest()` (6 hex chars, ~16M space — collisions rare). Add unit test asserting `hash_suffix(path)` is deterministic. | `graph_builder.py:1524` | S | **P0** |
| **C2** | Embed **decorator / route info into node attributes** (not the ID) consistently: `decorator_names: list[str]`, `http_methods: list[str]`, `route_pattern: str | None`. Already partly captured in `metadata`; promote to first-class attributes. | `python_parser.py:256-322`; `graph_builder.py` when adding node (`:1532-1544`) | S | P1 |
| **C3** | Add a **framework-aware symbol pass**. When a class inherits from `flask.Views`, `flask_restful.Resource`, `fastapi.APIRouter` bound-methods, `rest_framework.views.APIView`, `django.views.View`, flag the class as `symbol_subtype="http_resource"`. Method nodes inside such a class get extra metadata `http_verb`, `route`. | new pass in `python_parser.py` after class extraction; reference base-class list configurable | M | P1 |
| **C4** | Implement a **cross-file class-reopen heuristic** for Python: if two files define a class with the same name, both inheriting from the same base class, and the qualified module paths share a package prefix (e.g. `handlers.users.get` and `handlers.users.post`), **merge** them into one logical node. Record the original file list as `defined_in: list[str]`. Gate behind `WIKI_PYTHON_MERGE_REOPENED_CLASSES=true`. | `graph_builder.py` — add post-parse merge step after `_add_symbols_to_graph()` | L | P2 |
| **C5** | If C4 is deferred, add **sibling-class edges** as a lighter workaround: when two `class API(Resource)` nodes live in sibling modules, create `rel_type="reopens"`, `edge_class="framework"`, `weight=0.8`. Outline planner and retriever can follow these to see the full resource. | new pass; leverage existing `INHERITANCE` index | M | P2 |
| **C6** | Canonicalise node IDs for framework resources to `python::<package>::API@<route>.<verb>` when C3 metadata is present. This is **visible in the ID**, solving disambiguation across files naturally and giving the LLM a readable handle (`API@/users.get`). | `graph_builder.py:1508-1512` — route to resource-aware ID when symbol_subtype == http_resource | M | P3 |
| **C7** | Expose `symbol_subtype` in the unified DB schema so MCP / ask endpoints can filter by resource/endpoint/command/handler. | `storage/sqlite.py` schema migration | S | P3 |
| **C8** | Add parser diagnostics: emit a warning when `_get_local_qualified_name()` falls back to the bare `symbol.name` fallback (`python_parser.py:1486-1489`) — this usually indicates a parse quality issue. | `python_parser.py:1449-1489` | S | P2 |

---

## 4. Prioritised Roadmap

### Sprint 1 — Immediate (P0, ~1 week)

1. **C1** — Fix the salted-hash bug. Deterministic node IDs unlock everything downstream.
2. **A1** — Raise min symbol-name length to 4.
3. **A2** — Gate Pass 1 lexical to architectural types on both sides.
4. **A3** — Per-repo IDF table (needed for A4 next sprint).
5. **B1 / B2** — Edge-class filters in reachability and a `structural_reachability()` helper.

*Expected impact:* Noisy lexical edges drop by an estimated 60–80 % based on the stopword list (A5) and the min-length bump; node-ID stability means cluster plans persist across runs.

### Sprint 2 — Tiered matching (P1, ~1–2 weeks)

6. **A4** — Implement L0–L4 tiered resolution with weights.
7. **A5** — Stopword list and env-configurable additions.
8. **A6 / A7** — Flood telemetry + skip resolved-via-parent orphans.
9. **B3 / B4** — Gate and tag component bridges.
10. **B5** — Connectivity health metrics in progress events.
11. **C2** — Promote decorator/route info to first-class node attributes.
12. **C3** — Framework-aware pass (http_resource subtype).

*Expected impact:* Graph becomes analysable by tier/confidence; downstream consumers can choose their view.

### Sprint 3 — Strategic (P2–P3, multi-week)

13. **C4 / C5** — Cross-file class-reopen handling (merge or `reopens` edges).
14. **C6 / C7** — Canonical resource IDs and unified-DB schema extension.
15. **A8 / A9** — Per-tier caps and optional trigram tokenizer.
16. **B6** — Weight-aware modularity in cluster planner.
17. **C8** — Parser diagnostics.

*Expected impact:* Flask/FastAPI-heavy repos render as coherent resources; retrieval quality for endpoint-level questions improves materially.

---

## 5. Measurement Plan

Before/after metrics to run on a known repo (e.g. a mid-size Flask app from the test fixtures):

| Metric | Source | Target |
|--------|--------|--------|
| Total lexical edges per 1000 nodes | orphan-resolution stats | ↓ by ≥ 50 % after Sprint 1 |
| Top 10 symbol names producing edges | A6 telemetry | No stopword-ish name in top 10 |
| Orphans remaining after resolution | `stats["orphans_remaining"]` | unchanged or ↓ (quality, not quantity) |
| Component count before/after bridging | `bridge_disconnected_components` | ≥ 1, monotone; bridges ↓ after B3 |
| Node-ID stability across two consecutive indexing runs | hash of `sorted(graph.nodes())` | **identical** after C1 |
| Resource-class coverage | count of `symbol_subtype="http_resource"` | > 0 on Flask/FastAPI fixtures after C3 |
| Ask/research recall for "which endpoint handles X" | manual eval set | ↑ after C3–C6 |

---

## 6. Open Questions for Owner

1. **IDF table persistence.** Should the IDF table be per-indexing-run (simplest, rebuild every time) or persisted in unified DB and incrementally updated? The latter helps incremental reindex but adds migration complexity.
2. **Stopword list ownership.** Ship a sensible default in `constants.py`, allow project-level override via `.wikis/stopwords.txt`, or expose via env only?
3. **Node-ID migration.** After C1, existing persisted snapshots will have old randomised suffixes. Do we version the DB schema (`unified_db.version`) and rebuild on bump, or try to migrate in place?
4. **Resource-class ID canonicalisation (C6).** Changing the ID scheme is visible to the LLM prompts. Worth a focused eval before rollout.
5. **Scope of framework detection (C3).** Start with Flask + FastAPI + DRF + Django, or broader (Tornado, aiohttp, Starlette direct, Sanic)? Recommend starting with the three that cover the majority of Python web repos.

---

## Appendix A — File-and-Line Reference Index

| Concern | File | Lines |
|---------|------|-------|
| Orphan identification | `backend/app/core/graph_topology.py` | 258-268 |
| Orphan resolution cascade | `backend/app/core/graph_topology.py` | 354-629 |
| FTS5 lexical pass | `backend/app/core/graph_topology.py` | 434-475 |
| Semantic vector pass | `backend/app/core/graph_topology.py` | 477-602 |
| Directory fallback | `backend/app/core/graph_topology.py` | 271-351, 604-615 |
| Component bridging | `backend/app/core/graph_topology.py` | 1012-1120 |
| FTS5 backend & BM25 weights | `backend/app/core/storage/sqlite.py` | 749-809 |
| Framework-reference discovery | `backend/app/core/cluster_expansion.py` | 155-342 |
| `_MIN_FTS_NAME_LEN = 4` (already stricter) | `backend/app/core/cluster_expansion.py` | 117-118 |
| Architectural symbol set | `backend/app/core/code_graph/constants.py` | 189-208 |
| Symbol priority table | `backend/app/core/code_graph/symbol_priority.py` | — |
| Node ID construction + hash collision | `backend/app/core/code_graph/graph_builder.py` | 1508-1525 |
| Cross-file target resolution strategies | `backend/app/core/code_graph/graph_builder.py` | 1851-1970 |
| `_get_local_qualified_name` | `backend/app/core/parsers/python_parser.py` | 1449-1489 |
| Scope stack and `parent_symbol` | `backend/app/core/parsers/python_parser.py` | 283-314 |
| Decorator extraction | `backend/app/core/parsers/python_parser.py` | 1196-1206 |
| Class DEFINES relationship | `backend/app/core/parsers/python_parser.py` | 745-811 |
| Class INHERITANCE relationship | `backend/app/core/parsers/python_parser.py` | 482-492 |
| BFS reachability | `backend/app/core/code_graph/graph_query_service.py` | 920-985 |

---

*End of report.*
