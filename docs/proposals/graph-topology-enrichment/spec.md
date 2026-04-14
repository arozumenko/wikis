# Feature Specification: Graph Topology Enrichment

**Feature Branch**: `feature/graph-topology-enrichment`
**Created**: 2026-04-10
**Status**: Draft
**Input**: Port DeepWiki's Phase 2 graph topology enrichment pipeline into Wikis.
**Proposal**: `docs/proposals/graph-topology-enrichment/proposal.md`
**Depends on**: `unified-db-activation` (unified DB provides the FTS5/vector/edge persistence layer)

---

## User Scenarios & Testing

### User Story 1 — Orphan Resolution via FTS5 Lexical Match (Priority: P0)

Degree-0 orphan nodes (no incoming or outgoing edges) are connected to related nodes by searching FTS5 for their `symbol_name`. This is the first and cheapest resolution pass.

**Why this priority**: Orphans fragment community detection. FTS5 is fast and handles exact/partial name matches without embedding computation.

**Independent Test**: Build a graph with 10 connected nodes and 5 orphan nodes (unique symbol names that appear in other nodes' source text). Run orphan resolution. Verify orphans now have ≥1 edge. Verify edge class is `lexical` and `created_by` is `fts5_lexical`.

**Acceptance Scenarios**:

1. **Given** an orphan node with `symbol_name="UserValidator"`, **When** FTS5 search finds nodes whose `source_text` or `docstring` mentions "UserValidator", **Then** a `lexical_link` edge is added from the orphan to the top matches (up to `max_lexical_edges=2`).

2. **Given** FTS5 returns results including the orphan itself, **When** filtering, **Then** self-hits are excluded.

3. **Given** FTS5 finds matches for an orphan, **When** the orphan is resolved, **Then** it is skipped in subsequent resolution passes (semantic, directory).

---

### User Story 2 — Orphan Resolution via Semantic Vector Search (Priority: P0)

Orphans that remain after FTS5 resolution are resolved by embedding their source text and searching sqlite-vec with expanding path prefixes.

**Why this priority**: Semantic search catches conceptual relationships that exact name matching misses (e.g., `validate_input` ↔ `InputValidator`).

**Independent Test**: Build a graph with an orphan whose `symbol_name` doesn't appear in any other node's text, but whose `source_text` is semantically similar to another node. Run semantic resolution. Verify a `semantic_link` edge is added.

**Acceptance Scenarios**:

1. **Given** an orphan with `source_text` that is semantically similar to another node, **When** vector search finds matches within `vec_distance_threshold=0.15` (i.e., cosine similarity ≥ 0.85), **Then** a `semantic_link` edge is added with `raw_similarity = 1.0 - vec_distance`.

2. **Given** an orphan in `src/auth/token.py`, **When** expanding path prefix search runs, **Then** it searches: `src/auth` first, then `src`, then `""` (global) — stopping at the first level that yields results. Prefixes have no trailing slash; `""` (empty string) is the global fallback.

3. **Given** no `embedding_fn` is provided, **When** semantic resolution is attempted, **Then** it is skipped gracefully (FTS5 and directory fallback still run).

---

### User Story 3 — Orphan Resolution via Directory Proximity (Priority: P1)

Remaining orphans are resolved by finding the highest-degree non-orphan node in the same or parent directory.

**Why this priority**: This is the last-resort fallback. Even if a symbol has no textual or semantic relationship, co-location in the same directory implies some relationship.

**Independent Test**: Build a graph with an orphan in `src/utils/helpers.py` and a high-degree node in `src/utils/core.py`. Run directory resolution. Verify a `directory_link` edge connects them.

**Acceptance Scenarios**:

1. **Given** an orphan with `rel_path="src/utils/helpers.py"`, **When** directory resolution walks up the tree, **Then** it finds the highest-degree non-orphan node in `src/utils/`, then `src/`, then root — stopping at first match.

2. **Given** all directory levels have no non-orphan nodes, **When** resolution completes, **Then** the orphan remains unresolved (no garbage edge is added).

---

### User Story 4 — Document-to-Code Edge Injection (Priority: P0)

Markdown documents are linked to code nodes via:
- **Hyperlink parsing**: `[text](path)` links in Markdown content are resolved to code file/symbol nodes
- **Backtick mentions**: `` `ClassName` `` references in Markdown are resolved to symbol nodes
- **Directory proximity**: Doc directories are matched to code directories (stripping `docs/` prefix)

**Why this priority**: Without doc↔code edges, documentation nodes are isolated islands that never appear in code-related clusters, leading to wikis that ignore existing documentation.

**Independent Test**: Create a doc node with content `See [UserService](../src/auth/user_service.py) and \`TokenValidator\` for details.` Create code nodes for `user_service.py` and `TokenValidator`. Run doc edge injection. Verify both `hyperlink` edges are created.

**Acceptance Scenarios**:

1. **Given** a Markdown doc node with `[text](../src/auth/service.py)`, **When** hyperlink extraction runs, **Then** a `hyperlink` edge is added from the doc to the resolved code node.

2. **Given** a Markdown doc node with `` `TokenValidator` ``, **When** backtick mention extraction runs, **Then** a `hyperlink` edge is added from the doc to the `TokenValidator` symbol node (capped at 2 target matches per backtick mention).

3. **Given** a doc node in `docs/auth/README.md`, **When** proximity matching runs, **Then** it matches to code nodes in `auth/` (stripping `docs/` prefix). Edge class: `doc` / `proximity`. Capped at 5 code nodes per matching directory to avoid hairball edges.

4. **Given** a relative path `../src/foo.py` in a doc at `docs/guide.md`, **When** path normalization runs, **Then** it resolves to `src/foo.py` relative to the repo root. Guards: `mailto:`, `#`-only anchors, and external URLs (`http://`, `https://`) are skipped.

5. **Given** duplicate edges (same source→target), **When** doc injection runs, **Then** a `seen` set prevents duplicate edge insertion across both hyperlink and proximity tiers.

---

### User Story 5 — Component Bridging (Priority: P1)

Disconnected components in the graph are connected via bidirectional bridge edges. Bridges are placed between the most central nodes of each pair of components, weighted by directory histogram similarity.

**Why this priority**: Without bridging, disconnected modules produce separate wiki sections with no cross-references. The cluster planner works better on a fully-connected graph.

**Independent Test**: Build a graph with 3 disconnected components (no cross-component edges). Run component bridging. Verify the graph now has exactly 1 connected component. Verify bridge edges have class `bridge` / `component_bridge`.

**Acceptance Scenarios**:

1. **Given** 3 disconnected components, **When** bridging runs, **Then** bridge edges are added until the graph is fully connected. Algorithm: linear pass — sort components largest-first, each smaller component bridges to its best-matching larger component by directory histogram similarity (immediate parent directory only, using `<root>` for root-level files). Uses `nx.weakly_connected_components()` (not `to_undirected()`).

2. **Given** two components with overlapping directory prefixes, **When** bridge pair selection runs, **Then** the pair with highest directory similarity is bridged first.

3. **Given** a bridge edge, **When** it is added, **Then** it is bidirectional (both directions) with `created_by="component_bridging"`.

---

### User Story 6 — Edge Weighting (Priority: P0)

All edges (structural + synthetic) are weighted using inverse-in-degree: nodes that are referenced by many other nodes receive lower-weight incoming edges (they are "common" and less discriminative).

**Why this priority**: Without edge weighting, a utility class referenced by 50 callers has the same edge strength as a rare dependency. Weighting enables Leiden to find natural community boundaries.

**Independent Test**: Build a graph with a node that has in-degree 10 (structural only). Compute weight for an edge pointing to it. Verify `weight = 1 / log(10 + 2) ≈ 0.402`. For a synthetic edge, verify `weight = max(0.402, 0.5) = 0.5`.

**Acceptance Scenarios**:

1. **Given** edge `A → B` where `B` has structural in-degree `d`, **When** edge weighting runs, **Then** `weight = 1/log(d + 2)`.

2. **Given** a synthetic edge (edge_class in `{lexical, semantic, directory, doc, bridge}`), **When** computed weight < `SYNTHETIC_WEIGHT_FLOOR` (0.5), **Then** weight is clamped to 0.5.

3. **Given** "structural in-degree", **When** computing it, **Then** only edges NOT in `{lexical, semantic, directory, doc, bridge}` are counted. Synthetic edges do not inflate in-degree.

---

### User Story 7 — Hub Detection and Flagging (Priority: P1)

High-degree connector nodes are detected via Z-score on in-degree and flagged in the DB for downstream use by the cluster planner.

**Why this priority**: Hubs distort community detection. The cluster planner needs to know which nodes are hubs to exclude them during clustering and re-integrate afterward.

**Acceptance Scenarios**:

1. **Given** a graph where node B has in-degree Z-score > 3.0, **When** hub detection runs, **Then** `db.set_hub(B, is_hub=True)` is called.

2. **Given** a graph with < 3 nodes or zero variance in degrees, **When** hub detection runs, **Then** an empty set is returned (no hubs).

---

## Edge Cases

1. **No orphans** — Skip orphan resolution entirely. Proceed to doc edges + bridging.
2. **No doc nodes** — Skip doc edge injection. Proceed to bridging.
3. **Already connected graph** — Skip bridging. Proceed to weighting.
4. **embedding_fn not provided** — Skip semantic orphan resolution. FTS5 + directory fallback still run.
5. **Empty FTS5 index** — FTS5 resolution finds nothing. Fall through to semantic/directory.
6. **Very large graph (>50K nodes)** — Batch orphan resolution to avoid O(N²) FTS5 queries. DeepWiki handles this via per-orphan limits.

---

## Requirements

### Functional Requirements

**FR-001**: The system shall run Phase 2 enrichment after graph building and before clustering.

**FR-002**: Orphan resolution shall cascade: FTS5 lexical → semantic vector → directory proximity. Each pass only processes orphans not resolved by a previous pass.

**FR-003**: FTS5 lexical resolution shall add up to `max_lexical_edges` (default 2) edges per orphan, with edge class `lexical` / `lexical_link` / `fts5_lexical`.

**FR-004**: Semantic resolution shall use sqlite-vec KNN with expanding path prefixes and `vec_distance_threshold=0.15`. Edge class: `semantic` / `semantic_link` / `vec_semantic`.

**FR-005**: Directory resolution shall walk up the directory tree and connect to the highest-degree non-orphan. Edge class: `directory` / `directory_link` / `dir_proximity_fallback`.

**FR-006**: Document edge injection shall parse Markdown `[text](path)` hyperlinks and `` `SymbolName` `` backtick mentions. Edge class: `doc` / `hyperlink` / `md_hyperlink`.

**FR-007**: Document proximity edges shall match doc directories to code directories, stripping `docs/`/`doc/`/`documentation/` prefixes. Edge class: `doc` / `proximity` / `dir_proximity`.

**FR-008**: Component bridging shall connect disconnected components via bidirectional bridge edges between most-central nodes. Edge class: `bridge` / `component_bridge` / `component_bridging`.

**FR-009**: Edge weighting shall use `weight = 1/log(structural_in_degree + 2)` with `SYNTHETIC_WEIGHT_FLOOR = 0.5` for synthetic edges.

**FR-010**: Hub detection shall use Z-score on in-degree with configurable `z_threshold` (default 3.0). Hubs shall be flagged via `db.set_hub()`.

**FR-011**: All enrichment results shall be persisted to the unified DB via `db.upsert_edges_batch()` and `persist_weights_to_db()`.

**FR-012**: The `run_phase2()` orchestrator shall execute steps in order: orphan resolution → doc edges → bridging → weighting → hub detection → persistence → metadata.

### Non-Functional Requirements

**NFR-001**: Phase 2 for a 10,000-node graph shall complete in under 60 seconds.

**NFR-002**: Orphan rate after Phase 2 shall be < 5% (down from ~55%).

**NFR-003**: The graph shall have exactly 1 connected component after Phase 2 (bridging ensures full connectivity).

---

## Key Entities

- **`run_phase2(db, G, embedding_fn, z_threshold, vec_distance_threshold)`** — Main orchestrator
- **`resolve_orphans(db, G, embedding_fn)`** — 3-pass orphan resolution cascade
- **`inject_doc_edges(db, G)`** — Hyperlink + backtick + proximity doc edges
- **`bridge_disconnected_components(db, G)`** — Component bridging
- **`apply_edge_weights(G)`** — Inverse-in-degree weighting
- **`detect_hubs(G, z_threshold)`** — Z-score hub detection
- **`persist_weights_to_db(db, G)`** — Full edge persistence
