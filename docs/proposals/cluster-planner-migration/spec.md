# Feature Specification: Cluster-Based Structure Planner

**Feature Branch**: `feature/cluster-planner`
**Created**: 2026-04-10
**Status**: Draft
**Input**: Migrate the deterministic cluster planner from DeepWiki into Wikis as an alternative to the DeepAgents planner. Users choose which planner to use at wiki generation time.
**Proposal**: `docs/proposals/cluster-planner-migration/proposal.md`

---

## User Scenarios & Testing

### User Story 1 — Cluster Planner Selection (Priority: P1)

A user generating a wiki can choose between "DeepAgents" (filesystem exploration) or "Cluster" (graph-based community detection) as the structure planning strategy. The cluster planner produces a wiki structure in under 2 minutes with full code-relationship awareness.

**Why this priority**: This is the core feature — without planner selection, nothing else matters. The cluster planner's primary value is speed (10x faster) and determinism (same input → same output).

**Independent Test**: Generate a wiki for a medium repo (~200 files) with `planner_type="cluster"`. Verify a `WikiStructureSpec` is produced with 3–15 sections and 10–50 pages. Repeat — verify the same structure is produced (deterministic). Compare with `planner_type="deepagents"` — verify both produce valid structures but cluster planner completes faster.

**Acceptance Scenarios**:

1. **Given** a repository with an indexed code graph, **When** a user generates a wiki with `planner_type="cluster"`, **Then** the cluster planner runs deterministic community detection on the code graph, produces sections from macro-clusters and pages from micro-clusters, and names them via lightweight LLM calls.

2. **Given** a repository, **When** the same wiki generation is run twice with `planner_type="cluster"` on the same commit, **Then** the clustering produces identical section/page assignments (deterministic). LLM naming may vary slightly but symbol assignments are stable.

3. **Given** a repository, **When** `planner_type` is not specified in the request, **Then** the default planner (`deepagents`) is used. The cluster planner is opt-in only.

---

### User Story 2 — Architectural Projection (Priority: P1)

The cluster planner projects the full code graph down to architectural-level symbols only (classes, functions, interfaces, enums, etc.), collapsing methods and fields into their parent classes. This reduces noise and produces meaningful clusters.

**Why this priority**: Without architectural projection, the graph has 55% degree-0 orphan nodes (parameters, variables, fields) that fragment community detection into thousands of singleton clusters. Projection is the foundation of the entire clustering pipeline.

**Independent Test**: Build a code graph for a Python project with 50 classes containing 200 methods total. Run architectural projection. Verify the projected graph has ~50 nodes (classes + standalone functions), zero orphans, and all method edges have been remapped to their parent class.

**Acceptance Scenarios**:

1. **Given** a code graph with classes, methods, functions, and fields, **When** architectural projection runs, **Then** only nodes with `symbol_type` in `ARCHITECTURAL_SYMBOLS` are retained. Methods, fields, variables, and properties are collapsed into their parent class/struct node.

2. **Given** a method `UserService.authenticate` with edges to `TokenValidator` and `PasswordHash`, **When** the method is collapsed into `UserService`, **Then** the edges are remapped: `UserService → TokenValidator` and `UserService → PasswordHash`. Self-loops (method calling sibling method in same class) are dropped.

3. **Given** a standalone function `validate_config()` with no parent class, **When** architectural projection runs, **Then** the function is retained as its own node (not collapsed).

---

### User Story 3 — Macro-Clustering (Wiki Sections) (Priority: P1)

The cluster planner partitions the architecturally-projected graph into macro-clusters using the hierarchical Leiden community detection pipeline (via `leidenalg` C++ library). First, the graph is contracted to file-level nodes; then Leiden runs at γ=1.0 to produce sections. Each section’s original symbol nodes are then clustered again with Leiden at γ=1.0 to produce pages. The number of sections scales logarithmically with repository size.

**Why this priority**: Macro-clustering is the first real structural output. Without it, there are no sections and no downstream page grouping.

**Independent Test**: Build a code graph with 100 architectural nodes and known cross-file relationships. Run macro-clustering. Verify 5–10 sections are produced with internally-connected communities. Verify section count follows the logarithmic formula: `max(5, min(20, ceil(1.2 × log₂(file_count))))` where `file_count` is the number of unique files (not node count).

**Acceptance Scenarios**:

1. **Given** an architecturally-projected graph with N nodes across F files, **When** macro-clustering runs with the Leiden algorithm (`leidenalg.find_partition` with `RBConfigurationVertexPartition`, γ=1.0, seed=42), **Then** the number of sections is capped at `max(5, min(20, ceil(1.2 × log₂(F))))` where F = file count. For 10 files → 5, 100 files → 9, 1000 files → 12, 10000 files → 16.

2. **Given** a file-contracted graph (Leiden mode), **When** macro-clustering produces communities, **Then** file-to-section assignments are propagated back to individual symbol nodes within each file. All symbols in a file belong to the same section (file-level coherence).

3. **Given** macro-clustering produces more sections than the cap, **When** the section count exceeds the cap, **Then** the smallest clusters are iteratively merged into their nearest neighbor by **directory proximity** (`_dir_similarity()` — shared directory prefix entries via min-intersect of directory histograms) until within cap.

---

### User Story 4 — Micro-Clustering (Wiki Pages) (Priority: P1)

Within each macro-cluster (section), the planner runs a second round of Leiden community detection (`RBConfigurationVertexPartition`, γ=1.0) on the section’s original symbol-level nodes. This splits sections into wiki pages. Page sizes are dynamically bounded.

**Why this priority**: Without micro-clustering, each section would be a single massive page. Pages are the unit of content generation.

**Independent Test**: Given a macro-cluster with 30 nodes, run micro-clustering at resolution 1.0. Verify 2–6 pages are produced. Verify the global page target `max(8, min(200, ceil(sqrt(total_nodes/7))))` is respected across all sections.

**Acceptance Scenarios**:

1. **Given** a macro-cluster with N nodes, **When** micro-clustering runs with Leiden (`RBConfigurationVertexPartition`, γ=1.0, seed=42) at the symbol level, **Then** pages are produced where each page has ≥5 nodes (merge threshold enforced).

2. **Given** a micro-cluster with more than `adaptive_max_page_size(N)` nodes, **When** page sizing runs, **Then** the oversized cluster is recursively split at higher resolution until all pages are within bounds.

3. **Given** a micro-cluster with fewer than `merge_threshold` (4) nodes, **When** page sizing runs, **Then** the undersized cluster is merged into its nearest same-section neighbor by **directory proximity** (`_dir_similarity()`).

4. **Given** different repository sizes, **When** `adaptive_max_page_size()` is computed, **Then**: <500 nodes → max 25/page; 500–1999 → 35; 2000–4999 → 45; 5000+ → 60.

---

### User Story 5 — Hub Detection & Re-integration (Priority: P2)

High-degree connector nodes (hubs) are detected via PageRank and temporarily excluded from clustering to prevent them from absorbing smaller communities. After clustering, hubs are re-integrated via plurality voting.

**Why this priority**: Without hub handling, a single widely-referenced utility class can pull multiple unrelated modules into one cluster, producing poor section boundaries.

**Independent Test**: Build a graph with one hub node (`BaseModel`) that has edges to 20 other nodes across 5 distinct clusters. Run clustering with and without hub detection. Verify that with hub detection, the 5 clusters remain distinct.

**Acceptance Scenarios**:

1. **Given** a code graph, **When** hub detection runs, **Then** nodes with PageRank above the 90th percentile are identified as hubs. Hub detection is configured via `DEFAULT_HUB_PERCENTILE` constant (not a runtime feature flag).

2. **Given** identified hub nodes, **When** macro-clustering runs, **Then** hubs are excluded from the clustering input to prevent community collapse.

3. **Given** hub nodes and completed macro-clusters, **When** hub re-integration runs, **Then** each hub is assigned to the cluster it has the most edges to (plurality voting).

---

### User Story 6 — Section & Page Naming via LLM (Priority: P1)

After deterministic clustering, the planner names sections and pages using lightweight LLM calls with capability-based names (not file/class names).

**Why this priority**: Clustering produces numbered clusters. Without naming, the wiki structure is unusable. The naming step is the only LLM-consuming part of the cluster planner.

**Independent Test**: Given a macro-cluster containing `UserService`, `TokenValidator`, `PasswordHash`, `MFAProvider`, generate a section name. Verify the name is capability-based (e.g., "Authentication & Authorization") and not symbol-based (e.g., "UserService and Related Classes").

**Acceptance Scenarios**:

1. **Given** a macro-cluster with dominant symbols, **When** section naming runs, **Then** the LLM produces a capability-based name and description. The prompt includes the top 10 symbols ranked by PageRank + type priority.

2. **Given** a micro-cluster within a section, **When** page naming runs, **Then** the LLM uses **only that page's own central symbols** (not the section's dominant symbols). This prevents "centroid bleed" where macro-level dominant symbols shadow page-specific content.

3. **Given** the naming LLM call fails, **When** a fallback is needed, **Then** the page is named using the dominant symbol's name + primary directory (e.g., "UserService — src/auth/").

---

### User Story 7 — SSE Progress Events for Cluster Planner (Priority: P2)

The cluster planner emits progress events via the same SSE streaming system used by the DeepAgents planner, so the frontend can show real-time progress.

**Why this priority**: Without progress events, the UI shows a spinner for the full clustering + naming duration with no feedback. Users need visibility into what's happening.

**Independent Test**: Generate a wiki with `planner_type="cluster"`. Subscribe to SSE events. Verify progress events are emitted for: clustering start, section count detected, naming progress per section, structure complete.

**Acceptance Scenarios**:

1. **Given** a wiki generation with cluster planner, **When** the SSE stream is active, **Then** progress events are emitted: "Clustering code graph...", "Detected N sections, M pages", "Naming section 1/N...", "Structure planning complete".

2. **Given** the clustering phase, **When** it completes, **Then** a progress event includes the algorithm used (`leiden`), section count, page count, and time elapsed.

---

## Edge Cases

1. **Very small repos (< 10 architectural nodes)** — Leiden may produce only 1 cluster. The planner should detect this and fall back to DeepAgents or produce a minimal structure with 1 section and 1–3 pages.

2. **Mono-file repos (single large file)** — File contraction produces 1 node. Macro-clustering is degenerate. Fall back to micro-clustering directly at the node level.

3. **No code graph available** — If the graph builder fails or produces an empty graph, the cluster planner should fail gracefully with a clear error, and the generation pipeline should fall back to DeepAgents.

4. **Disconnected graph** — Some repos have multiple independent clusters with no cross-component edges. Leiden handles this natively (each connected component becomes its own community). Note: the `graph-topology-enrichment` proposal adds synthetic bridge edges that make graphs fully connected before clustering.

5. **All nodes are hubs** — If the hub percentile threshold catches too many nodes, the clustering input is empty. Mitigation: cap hubs at 15% of total nodes.

---

## Requirements

### Functional Requirements

**FR-001**: The system shall support two structure planning strategies: `deepagents` (existing) and `cluster` (new).

**FR-002**: The `GenerateWikiRequest` API model shall accept a `planner_type` field with values `"deepagents"` (default) and `"cluster"`.

**FR-003**: The cluster planner shall produce a `WikiStructureSpec` with the identical schema used by the DeepAgents planner — no downstream pipeline changes required.

**FR-004**: Architectural projection shall collapse methods, fields, variables, and properties into their parent class/struct node, remapping all edges accordingly.

**FR-005**: Macro-clustering shall use the two-pass hierarchical Leiden pipeline: (1) contract graph to file-level, (2) `leidenalg.find_partition(RBConfigurationVertexPartition, resolution_parameter=1.0, seed=42)` on file-contracted graph → sections, (3) Leiden on per-section original nodes → pages. Both passes use the `leidenalg` C++ library (REQUIRED dependency). If `leidenalg` is not installed, the cluster planner shall be unavailable and the user must use the DeepAgents planner.

**FR-006**: The macro-cluster (section) count shall be capped at `max(5, min(20, ceil(1.2 × log₂(file_count))))` where `file_count` is the number of unique files in the repository. For repos with < 10 files, the cap is 5. Sections exceeding the cap are iteratively merged by directory proximity.

**FR-007**: Micro-clustering shall run per macro-cluster using Leiden (`RBConfigurationVertexPartition`, γ=1.0, seed=42) on the original symbol-level nodes within each section. Global page count is capped at `max(8, min(200, ceil(sqrt(node_count / 7))))`. Pages are consolidated globally: the smallest page across all sections is merged into its best same-section neighbour by directory proximity, repeating until total pages ≤ target. Adaptive maximum page size per page: <500 nodes → 25, 500–1999 → 35, 2000–4999 → 45, 5000+ → 60.

**FR-008**: Hub detection shall use PageRank with configurable percentile threshold (default: 90th percentile). Hubs shall be excluded during clustering and re-integrated via plurality voting afterward.

**FR-009**: Section naming shall use 1 LLM call per macro-cluster with the top 10 dominant symbols (ranked by PageRank + type priority). Names shall be capability-based.

**FR-010**: Page naming shall use 1 LLM call per micro-cluster with **page-only** central symbols (not section-level dominant symbols) to prevent centroid bleed.

**FR-011**: The cluster planner shall emit SSE progress events via `invocation.emit()` at each phase: clustering, section detection, naming progress.

**FR-012**: If the code graph is empty or has fewer than 10 architectural nodes, the system shall automatically fall back to the DeepAgents planner with an informational log.

**FR-013**: All clustering shall operate on in-memory NetworkX graphs. No separate SQLite database is required for Phase 1.

### Non-Functional Requirements

**NFR-001**: Clustering for a 1000-node graph shall complete in under 5 seconds.

**NFR-002**: Total structure planning time (clustering + naming) for a 1000-node graph shall be under 2 minutes.

**NFR-003**: The cluster planner shall produce deterministic cluster assignments given the same graph input (Leiden is deterministic for a given resolution and seed).

**NFR-004**: `leidenalg==0.11.0` and `python-igraph==1.0.0` are **required** dependencies for the cluster planner. The system uses ONLY the Leiden algorithm via the `leidenalg` C++ library. If Leiden is not available, the cluster planner is disabled and users must use the DeepAgents planner.

---

## Key Entities

- **ClusterAssignment** — Mapping of node IDs to macro-cluster (section) and micro-cluster (page) IDs
- **HubNode** — High-degree connector node identified by PageRank
- **ArchitecturalProjection** — Reduced graph with only top-level symbols
- **FileContractedGraph** — Graph where each file is a single node (for Leiden section clustering)

---

## Success Criteria

**SC-001**: Cluster planner produces valid `WikiStructureSpec` for 95% of test repositories (10+ repos of varying sizes).

**SC-002**: Cluster planner structure planning completes in < 2 minutes for repos with < 2000 files.

**SC-003**: Cluster planner uses < 25K tokens total for structure planning (vs. 30K–80K for DeepAgents).

**SC-004**: Generated wiki quality (reader survey) is comparable or better than DeepAgents-generated wikis for the same repositories.

**SC-005**: No regression in DeepAgents planner behavior — existing planner works identically when selected.

---

## Three-Phase Delivery

| Phase | Scope | What Ships |
|-------|-------|------------|
| **Phase 1** | Graph Clustering Engine | `graph_clustering.py` with architectural projection, Leiden/Louvain macro/micro clustering, hub detection, dynamic page sizing. Unit tests. |
| **Phase 2** | Cluster Structure Planner + Integration | `cluster_planner.py` with LLM naming, planner routing in `wiki_graph_optimized.py`, API `planner_type` field, SSE events. Integration tests. |
| **Phase 3** | Smart Expansion Sync + Polish | Verify/sync `expansion_engine.py` with DeepWiki parity, performance tuning, documentation, user guide. |

---

## Traceability

| AC | Covers FR |
|----|-----------|
| US-1 AC-1 | FR-001, FR-002, FR-003 |
| US-1 AC-2 | FR-003 (determinism) |
| US-1 AC-3 | FR-001, FR-002 |
| US-2 AC-1 | FR-004 |
| US-2 AC-2 | FR-004 |
| US-3 AC-1 | FR-005, FR-006 |
| US-3 AC-2 | FR-005 |
| US-3 AC-3 | FR-005 |
| US-3 AC-4 | FR-006 |
| US-4 AC-1 | FR-007 |
| US-4 AC-2 | FR-007 |
| US-4 AC-3 | FR-007 |
| US-5 AC-1 | FR-008 |
| US-5 AC-2 | FR-008 |
| US-5 AC-3 | FR-008 |
| US-6 AC-1 | FR-009 |
| US-6 AC-2 | FR-010 |
| US-6 AC-3 | FR-009, FR-010 |
| US-7 AC-1 | FR-011 |
