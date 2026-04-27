# Lexical Orphan Resolution, Connectivity, and Node Disambiguation Review

> Status: review draft
> Date: 2026-04-23
> Scope:
> - `backend/app/core/graph_topology.py`
> - `backend/app/core/code_graph/graph_builder.py`
> - `backend/app/core/parsers/python_parser.py`
> - `backend/app/core/graph_clustering.py`
> - `backend/app/core/code_graph/graph_query_service.py`
> - `backend/app/core/storage/sqlite.py`

This report focuses on three coupled issues:

1. Lexical orphan resolution currently over-connects the graph on common words.
2. Connectivity is traced on a graph that already contains strong synthetic edges, so synthetic rescue can distort clustering, hub detection, and graph traversal.
3. Python node disambiguation is not just cosmetic. The current identity and registry strategy can change which node wins resolution, and it can vary between runs.

The main conclusion is:

- The lexical pass should become a strict, tiered, architecture-aware rescue mechanism.
- Connectivity should be traced in at least two views: structural and enriched.
- Node identity should move away from file-stem-based ids and last-writer-wins registries.

## Executive Summary

| Area | Current behavior | Why it is a problem | Best next step |
|------|------------------|---------------------|----------------|
| Lexical orphan rescue | `resolve_orphans()` runs FTS over all indexed nodes and all indexed text fields, with only a 2-character minimum | Common names like `shared`, `base`, `config`, `manager` can create edges that are too easy to add and too strong downstream | Restrict lexical rescue to high-confidence architectural matches, then fall back to semantic and directory passes |
| Connectivity | Phase 2 adds lexical, semantic, doc, and bridge edges before weighting and hub detection | Synthetic edges can change component count, inflate hub degree, and alter traversal results | Trace and query structural connectivity separately from enriched connectivity |
| Python node ids | Node ids are keyed by `file stem + local qualified name`, and collisions get `hash(file_path) & 0xFFFF` suffixes | Same-stem files collide often; the suffix is non-deterministic across runs; many registries are simple-name maps | Switch to stable path-based identity, deterministic tie-breakers, and multi-candidate registries |
| Python API-style classes split across modules | `API.get`, `API.post`, `API.delete` in sibling files become separate class islands | Connectivity, retrieval, and planner understanding of a resource become fragmented | Add a logical resource key / reopen grouping for framework-style classes |

## Connectivity Trace

This is the current path by which connectivity is formed.

1. Python rich parsing starts in `PythonParser.parse_multiple_files()` (`python_parser.py:1553`).
2. Files are parsed in parallel and collected in completion order via `as_completed()` (`python_parser.py:1721-1730`).
3. Global Python registries are then built from those results, keyed mostly by simple names such as class name or function name (`python_parser.py:1785-1833`).
4. Graph nodes are created in `_process_file_symbols()` using `language::file_stem::local_qualified_name` (`graph_builder.py:1491-1562`).
5. When that id collides with a different file, a suffix derived from `hash(file_path)` is appended (`graph_builder.py:1516-1525`).
6. Relationships are resolved using a mix of direct string construction and registry lookups (`graph_builder.py:1771-1978`).
7. Phase 2 starts with orphan rescue on the full graph, not on the projected architectural graph (`graph_topology.py:354-620`).
8. Phase 2 then injects doc edges, bridges disconnected components, weights all edges, and finally detects hubs (`graph_topology.py:1127-1207`).
9. Architectural projection happens later and relies partly on node-id string parsing to promote child symbols to parents (`graph_clustering.py:225-329`).
10. Query-time reachability in `GraphQueryService._filter_related()` traverses whatever edge classes are present unless the caller filters only by relationship type (`graph_query_service.py:920-985`).

The important point is that lexical rescue is not an isolated heuristic. It changes the graph before hub detection, before projection-based clustering, and before later BFS-style traversal.

## Findings

### 1. Lexical orphan rescue currently runs too early, too broadly, and too strongly

`find_orphans()` defines an orphan as any node with zero in-degree and zero out-degree (`graph_topology.py:258-268`).

That means the lexical pass can target:

- classes
- functions
- methods
- fields
- variables
- doc nodes
- inferred nodes

It does not first ask whether the symbol is a child implementation detail whose parent is already connected.

The first lexical pass inside `resolve_orphans()` does this (`graph_topology.py:434-469`):

- derive `symbol_name`
- skip only names shorter than 2 chars
- call `db.search_fts5(query=symbol_name, limit=fts_limit)`
- attach up to `max_lexical_edges` targets
- mark the orphan as resolved immediately

Problems:

1. `search_fts5()` indexes `symbol_name`, `signature`, `docstring`, and `source_text` for every row in `repo_nodes` (`sqlite.py:738-799`).
2. The lexical orphan pass does not pass `symbol_types` or `path_prefix`, even though `search_fts5()` supports both (`sqlite.py:749-781`).
3. The lexical orphan pass does not distinguish exact symbol-name hits from docstring/source-text hits.
4. The lexical orphan pass does not distinguish architectural nodes from implementation-detail nodes.
5. The lexical orphan pass resolves the orphan immediately on any FTS hit, so semantic rescue is skipped even when it would be safer.

This is exactly how a token like `shared` can flood the graph:

- it is common in symbol names
- it is common in docstrings
- it is common in source text
- the pass has no rarity gate
- the pass has no architecture gate
- the pass has no path locality gate

### 2. The current weighting model makes weak synthetic edges too important

The code does not leave lexical edges unweighted. It gives all synthetic edges a floor of `0.5` in `apply_edge_weights()` (`graph_topology.py:72-134`).

That means:

- a weak lexical edge
- a weak doc proximity edge
- a component bridge edge

all get at least the same floor weight, regardless of how weak or noisy the matching signal was.

This is better than zero weight, but it is still too blunt.

The system currently knows:

- `edge_class`
- `created_by`
- optional `raw_similarity` for semantic edges

It does not know:

- lexical tier
- matched field (`symbol_name`, `docstring`, `source_text`)
- path distance
- rarity / idf strength
- why this edge should be weaker than another lexical edge

So low-quality lexical rescue can become a meaningful clustering signal even when it should be treated as a last-ditch hint.

### 3. Hub detection sees synthetic degree, not just structural degree

`apply_edge_weights()` computes structural in-degree by excluding synthetic classes from the weighting formula (`graph_topology.py:95-115`).

But `detect_hubs()` later computes hub z-scores from `G.in_degree(n)` on the enriched graph (`graph_topology.py:141-183`).

That means synthetic edges added during orphan rescue, doc injection, and component bridging can still:

- increase node degree
- push nodes into hub status
- change which nodes are quarantined before clustering

So even if synthetic edges are excluded from the inverse in-degree weight calculation, they are still part of hub detection.

This is a direct way for noisy lexical rescue to distort connectivity.

### 4. Architectural projection depends too heavily on node-id string shape

`architectural_projection()` promotes child nodes to parents partly by parsing node ids like `lang::file::Parent.child` (`graph_clustering.py:278-298`).

This is fragile for two reasons:

1. If the stored parent node got a collision suffix, the expected parent id may not exist.
2. The fallback `parent_symbol` lookup is keyed by `(file_name, symbol_name)` (`graph_clustering.py:254-264`), but Python `parent_symbol` often carries module-qualified values, so the fallback is weaker than it looks.

As a result, node-id instability is not just a naming problem. It can affect whether child symbols are correctly promoted to architectural parents during clustering.

### 5. Python node identity is too collision-prone before the hash suffix is even considered

The core id is currently:

```python
node_id = f"{language}::{file_name}::{local_qualified_name}"
```

from `graph_builder.py:1511-1512`.

`file_name` here is only `Path(file_path).stem` (`graph_builder.py:1494`).

That means these files collide structurally before any tie-breaker:

- `src/users/api.py`
- `src/admin/api.py`
- `tests/api.py`

If each defines `API.get`, they all want the same node id prefix:

- `python::api::API.get`

So the problem is not primarily the suffix algorithm. The real problem is the identity namespace being too small.

The suffix just papers over a frequent collision pattern.

### 6. The hash suffix is non-deterministic across runs

On collision, the builder appends:

```python
file_hash = hash(file_path) & 0xFFFF
node_id = f"{node_id}_{file_hash:04x}"
```

from `graph_builder.py:1523-1525`.

That suffix is not stable across Python processes because `hash(str)` is salted.

I confirmed this locally with two separate subprocesses for the same path string; they produced different hash values.

Impact:

- node ids can change between indexing runs without source changes
- persisted graph snapshots cannot rely on stable ids
- diffs and incremental repair become harder
- any fallback that relies on raw node-id suffix matching becomes unreliable

This must be fixed immediately, but a deterministic suffix alone is not enough because the underlying collision frequency stays high.

### 7. Python simple-name registries are last-writer-wins, and the writer order is non-deterministic

This is the most important disambiguation issue beyond the hash suffix.

In `PythonParser._parse_files_parallel()`, parse results are inserted into a dict in `as_completed()` order (`python_parser.py:1721-1730`).

Then the parser builds global registries such as:

- `_global_class_locations[symbol.name] = file_path`
- `_global_function_locations[symbol.name] = file_path`

in `python_parser.py:1803-1817`.

So if multiple files define `API`, `get`, `post`, `delete`, or any other common name:

- the registry keeps only one winner
- the winner depends on parse completion order
- parse completion order can vary between runs

This means cross-file enhancement in `_enhance_relationship()` (`python_parser.py:1859-1918`) can point relationships at different files on different runs for the same repository.

This is a real connectivity bug.

### 8. Builder registries also collapse ambiguity too early

Inside `_process_file_symbols()`, the builder keeps:

- `by_name` as a list
- `by_qualified_name` as a single dict entry

from `graph_builder.py:1546-1558`.

`by_qualified_name` is overwritten on collision. So a local qualified name like `API.get` or a full name chosen as a key can silently point to whichever symbol was registered last.

This creates another source of incorrect connectivity when target resolution falls through to strategy 5 in `_resolve_target_node_sync()` (`graph_builder.py:1937-1941`).

### 9. Qualified-name lookup and suffix indexes do not normalize collision suffixes

`attach_graph_indexes()` stores `_suffix_index` using the raw final node-id suffix (`graph_builder.py:129-133`).

If the actual stored node id is:

- `python::api::API.get_abcd`

then the suffix index stores:

- `API.get_abcd`

not:

- `API.get`

That means a consumer trying to resolve `API.get` through suffix-based lookup can miss the node and fall back to:

- a simpler name match
- an FTS fallback
- or a wrong candidate

So the suffix leaks into resolution, not just display.

### 10. The current `API.get_<hash>` pattern can affect connectivity

The user asked whether the `python::<module_name>::API.get_<hash>` shape is just cosmetic.

It is not purely cosmetic.

It can affect connectivity indirectly through:

1. wrong target resolution when direct or suffix-based lookup misses
2. wrong parent promotion in architectural projection
3. last-writer-wins registry selection among same-named methods
4. fragmented representation of one logical resource across multiple thin class nodes

The effect is especially bad in Python repos that split one REST resource across sibling modules, for example:

- `users_get.py`
- `users_post.py`
- `users_delete.py`

Each file defines a separate `API` class, and each method stays contained inside its own local class graph. Without an explicit logical resource layer, connectivity is fragmented even before lexical rescue starts trying to compensate.

### 11. Test coverage misses the risky cases

Current tests are useful but too optimistic for these failure modes:

- `backend/tests/unit/test_graph_topology.py:161` and `:184` validate happy-path lexical rescue and caps, but they do not test common-token flooding, architecture gating, or parent-aware orphan skipping.
- `backend/tests/unit/test_graph_clustering.py:161` and `:170` use simplified node ids like `ClassA` and `method1`, so they do not exercise the real promotion path that depends on `lang::file::Parent.child`.
- `backend/tests/unit/test_graph_builder.py:1772` and `:1816` cover direct target resolution strategies, but not same-stem file collisions, collision suffixes, or non-deterministic registry order.

The current suite therefore underestimates the real risk.

## Recommended Approach

### A. Redefine which nodes are eligible for lexical orphan rescue

Do not run lexical rescue on all zero-degree symbols.

Use one of these two options:

1. Preferred: compute orphan rescue candidates on the architectural view.
2. Acceptable: keep the full graph, but skip child symbols whose parent has structural connectivity.

Practical rule:

- class, function, constant, doc node: eligible
- method, field, variable, property: only eligible if their resolved parent is also structurally disconnected

This removes a large amount of implementation-detail noise before lexical matching even starts.

### B. Use a tiered rescue ladder with exact and path-aware lexical tiers only

The best structure is:

1. `L0`: exact symbol-name match in same directory
2. `L1`: exact symbol-name match in same package / rel_path prefix
3. `S0`: semantic vector search with local path prefixes
4. `L2`: rare-token docstring/context overlap
5. `L3`: rare-token global lexical fallback
6. directory fallback

Why this order:

- high-confidence lexical exact/path matches are cheap and useful
- semantic local search is usually safer than weak global lexical matches
- docstring/source-text lexical should be low confidence, not the first thing that runs

### C. Gate lexical rescue to architectural targets

The lexical pass should use the existing `search_fts5()` filters that are already available but unused.

Recommended rule:

- source orphan must be architectural, or the parent architectural anchor must be orphaned
- target hit must be architectural
- same-language targets get a bonus
- cross-language lexical rescue is allowed only at lower tiers and should be opt-in

### D. Add rarity and boilerplate suppression

Build a per-repo lexical statistics table over:

- symbol names
- path segments
- optional docstring tokens

Use identifier splitting:

- `SharedCache` -> `shared`, `cache`
- `shared_cache` -> `shared`, `cache`
- `shared-cache` -> `shared`, `cache`

Maintain a default boilerplate list such as:

- shared
- common
- base
- manager
- handler
- helper
- util
- utils
- config
- service
- app
- core
- main
- data
- model
- context
- factory
- builder
- client
- server

Low-idf or boilerplate-heavy candidates should not create lexical rescue edges.

### E. Make synthetic confidence explicit

Every synthetic edge should carry:

- `confidence`
- `tier`
- `matched_on`
- `path_distance`
- `matched_tokens`

Then use that information in later stages:

- low-confidence lexical edges should not get the same floor as high-confidence exact/path matches
- low-confidence synthetic edges should be excluded from default reachability
- hub detection should be able to ignore or discount low-confidence synthetic degree

### F. Trace connectivity in two views

Add two connectivity views and keep both visible:

1. structural connectivity
2. enriched connectivity

Suggested structural edge classes:

- AST-derived structural relationships
- maybe explicit doc hyperlinks

Suggested enriched-only edge classes:

- lexical
- semantic
- directory
- component bridge
- doc proximity

This is the key to making later debugging understandable.

## Recommended Actions

### P0: Immediate fixes

| ID | Action | Why |
|----|--------|-----|
| P0.1 | Replace `hash(file_path) & 0xFFFF` with a deterministic digest immediately | Stops cross-run node-id drift |
| P0.2 | Stop using `file stem` as the file scope segment in node ids; use normalized `rel_path` or Python module path instead | Removes most collisions at the source |
| P0.3 | Sort or re-materialize parsed Python results into deterministic file order before building global registries and graph nodes | Removes completion-order nondeterminism |
| P0.4 | Change Python global registries from simple-name singletons to multimaps | Prevents `API`, `get`, `post`, `delete` from becoming last-writer-wins |
| P0.5 | Restrict lexical orphan pass to architectural sources/targets and require a minimum token length of at least 4 | Eliminates the noisiest rescue cases immediately |
| P0.6 | Run semantic local search before weak global lexical fallback | Prefers safer rescue signals |

### P1: Connectivity correctness

| ID | Action | Why |
|----|--------|-----|
| P1.1 | Add `edge_classes` and `min_confidence` filters to reachability traversal in `GraphQueryService` | Lets callers ask structural questions structurally |
| P1.2 | Add `detect_hubs(structural_only=True)` or equivalent | Prevents low-quality synthetic rescue from creating false hubs |
| P1.3 | Emit per-pass connectivity metrics: components before/after each pass, edge counts by class, top lexical anchors, remaining structural orphans | Makes graph quality observable |
| P1.4 | Store a small per-orphan trace sample: candidate list, chosen edge, rejected reasons | Makes lexical tuning practical |
| P1.5 | Discount or exclude low-confidence lexical edges from bridge and clustering inputs | Prevents weak rescue from dominating communities |

### P2: Node model and Python resource grouping

| ID | Action | Why |
|----|--------|-----|
| P2.1 | Introduce explicit `parent_node_id` during graph construction instead of reconstructing parenthood from node-id strings later | Makes projection and traversal robust |
| P2.2 | Introduce `logical_entity_key` separate from physical `node_id` | Allows stable grouping of reopened or framework-style resources |
| P2.3 | For Python framework resources, derive `symbol_subtype=http_resource`, `http_verb`, `route_pattern`, and a shared resource key | Prevents REST resource fragmentation |
| P2.4 | If full merge is too risky, add `reopens` / `same_resource_as` edges between sibling `API(Resource)` classes | Gives the planner and retriever a bridge without destructive merging |
| P2.5 | Normalize lookup indexes so `API.get` can still resolve even if the stored node id is suffixed | Removes suffix leakage from query-time symbol resolution |

### P3: Test coverage

| ID | Action | Why |
|----|--------|-----|
| P3.1 | Add lexical orphan tests for common tokens like `shared`, `base`, `config`, `manager` | Verifies flood control |
| P3.2 | Add tests where the orphan is a method or field whose parent is already connected | Verifies parent-aware skipping |
| P3.3 | Add graph-builder tests with same-stem files in different directories | Verifies node-id collision strategy |
| P3.4 | Add parser tests proving deterministic Python registry output regardless of future completion order | Verifies reproducibility |
| P3.5 | Add architectural projection tests with real node ids such as `python::src/users/api.py::API.get` and suffixed variants | Verifies child promotion in realistic conditions |

## Best Concrete Design

If the goal is to improve the system with the least churn and the highest payoff, this is the recommended design:

### 1. Split identity into three layers

- `node_id`: opaque stable storage id
- `symbol_key`: normalized local qualified name
- `logical_entity_key`: optional cross-file / framework-aware grouping key

Example:

- `node_id = python::src/users/get.py::API.get`
- `symbol_key = API.get`
- `logical_entity_key = python::users::API@Resource(/users)`

This keeps storage stable while giving the planner and retriever better grouping handles.

### 2. Keep lexical rescue exact and path-biased

Lexical rescue should mostly answer:

- "same symbol name, near the same code area"

It should not try to answer:

- "the word appears somewhere in repo text"

That second job belongs to semantic retrieval or deliberately low-confidence context matching.

### 3. Make structural connectivity first-class

All of these should be computed and exposed:

- `structural_component_id`
- `enriched_component_id`
- `structural_degree`
- `synthetic_degree`
- `orphan_kind`:
  - structural orphan
  - enriched-only connected
  - child orphan with connected parent

That will make graph behavior explainable.

### 4. Treat Python framework resources as logical groups

For the `API(Resource)` family of patterns, the graph should expose:

- the physical class nodes in each file
- a logical resource grouping
- same-resource or reopen edges
- verb / route metadata

This is enough to improve connectivity even before a full merge model is implemented.

## Final Recommendation

The best path is not to keep adding stronger lexical heuristics to rescue everything.

The best path is:

1. reduce the candidate set
2. make lexical rescue exact and local
3. promote semantic rescue ahead of weak lexical fallback
4. separate structural from enriched connectivity
5. fix node identity and registry determinism first

If I had to prioritize only three actions, they would be:

1. switch node ids away from file stem + salted hash
2. convert Python registries to deterministic multimaps
3. replace the current FTS-first orphan pass with an architecture-gated, exact/path-tiered rescue ladder

Those three changes address the root causes rather than just softening the symptoms.
