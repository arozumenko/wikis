# Proposal: Cluster-Based Structure Planner (DeepWiki Migration)

> **Status**: Draft
> **Author**: Roman Mitusov (with Claude)
> **Date**: 2026-04-10
> **Target repo**: [arozumenko/wikis](https://github.com/arozumenko/wikis)
> **Source repo**: deepwiki_plugin (pylon_deepwiki/plugins/deepwiki_plugin)

---

## Problem

Wikis currently has only **one structure planning strategy**: a DeepAgents filesystem-exploration planner (`WikiStructurePlannerEngine`). This planner creates wiki outlines by giving an LLM agent filesystem tools (`ls`, `glob`, `grep`, `read_file`) and asking it to explore the repository and define sections/pages via tool calls.

**Limitations of the current approach:**

1. **Expensive** — Every structure planning session runs 50–100 LLM tool-calling iterations. For a medium repository (500 files), that's 30K–80K tokens just for structure planning, plus 10–30 minutes of wall-clock time due to serial tool calls.

2. **Non-deterministic** — The agent explores different paths on different runs. Same repository, same branch can produce radically different wiki structures depending on which directories the agent visits first and what patterns it recognizes.

3. **Blind to code relationships** — Filesystem exploration sees files and directories, but misses cross-file dependencies. Two classes in different directories that share an inheritance chain or compose each other end up in unrelated sections because the planner never sees the code graph.

4. **No relationship awareness in page grouping** — The planner groups by directory proximity, not by functional cohesion. A `PaymentProcessor` in `src/payments/` and a `PaymentValidator` in `src/validators/` should be on the same wiki page, but the current planner separates them.

The production DeepWiki plugin has solved all four problems with a **deterministic cluster planner** — a 3-phase mathematical pipeline that uses Leiden community detection (via the `leidenalg` C++ library) on the code graph to produce wiki structure. The structure is computed in seconds (zero LLM calls for clustering), with only lightweight LLM naming calls afterward.

---

## Proposed Solution

Migrate the **cluster-based structure planner** from DeepWiki into Wikis as an **alternative planner strategy** alongside the existing DeepAgents planner. Users choose which planner to use at generation time via a UI toggle.

The migration involves four phases:

1. **Graph Clustering Engine** — Port the proven hierarchical Leiden pipeline from DeepWiki: file-level contraction → Leiden macro-clustering (sections) → Leiden micro-clustering (pages) → hub detection + re-integration, dynamic page sizing. Uses `leidenalg` (C++ Leiden implementation) exclusively — no Louvain fallback.
2. **Cluster Structure Planner** — Port the LLM naming layer (Phase 4): section naming, page naming with per-page symbols, capability-based titles
3. **Smart Graph Expansion Engine** — Port the priority-based edge traversal that enriches page context with related symbols during content generation
4. **Planner Selection & Integration** — Wire up the planner switch in the generation pipeline and expose it in the API/UI

---

## Current Architecture (Research Summary)

### How Structure Planning Works Today (Wikis)

```
WikiService._run_generation()
  → OptimizedWikiGenerationAgent.generate_wiki()
    → generate_wiki_structure() node
      → WikiStructurePlannerEngine.plan_structure()
        → _create_agent() with FilesystemMiddleware
        → Agent runs 50-100 iterations:
            ls() → glob() → grep() → read_file() → define_section() → define_page()
        → Collects tool-call outputs into WikiStructureSpec
```

**Key files (Wikis):**
- `backend/app/core/wiki_structure_planner/structure_engine.py` — DeepAgents planner engine
- `backend/app/core/wiki_structure_planner/structure_prompts.py` — System/task prompts
- `backend/app/core/wiki_structure_planner/structure_tools.py` — Agent tools
- `backend/app/core/agents/wiki_graph_optimized.py` — Generation agent (calls planner)
- `backend/app/core/state/wiki_state.py` — `WikiStructureSpec`, `PageSpec`, `SectionSpec`

### How Cluster Planning Works in DeepWiki

```
Phase 1: Graph Building
  → EnhancedUnifiedGraphBuilder.analyze_repository()
  → Produces NetworkX MultiDiGraph + Document[]

Phase 2: Edge Weighting + Hub Detection
  → PageRank centrality → identify high-degree hubs
  → Apply edge weights (cross-file > same-file, hub-adjacent boosted)

Phase 3: Deterministic Clustering (ZERO LLM calls)
  → architectural_projection() — collapse methods/fields into parent classes
  → _contract_to_file_graph() — each rel_path becomes one node, cross-file edge weights summed
  → Leiden pass 1 — leidenalg.find_partition(RBConfigurationVertexPartition, γ=1.0) on file-contracted graph → sections
  → Leiden pass 2 — leidenalg.find_partition(RBConfigurationVertexPartition, γ=1.0) per section on original nodes → pages
  → _consolidate_sections() + _consolidate_pages() — merge small, enforce caps
  → reintegrate_hubs() — plurality voting for hub assignment
  → persist_clusters() → store in unified DB

Phase 4: LLM Naming (cheap, ~13K-22K tokens total)
  → ClusterStructurePlanner.plan_structure()
  → 1 LLM call per section (dominant symbols → capability name)
  → 1 LLM call per page (page-only symbols → page name)
  → Output: WikiStructureSpec (identical schema to current planner)
```

**Key files (DeepWiki):**
- `plugin_implementation/graph_clustering.py` — Clustering algorithms (2500+ lines)
- `plugin_implementation/wiki_structure_planner/cluster_planner.py` — LLM naming (900+ lines)
- `plugin_implementation/wiki_structure_planner/structure_prompts.py` — Naming prompts
- `plugin_implementation/code_graph/expansion_engine.py` — Smart context expansion (1300+ lines)
- `plugin_implementation/unified_db.py` — Persistent storage (1500+ lines)
- `plugin_implementation/constants.py` — ARCHITECTURAL_SYMBOLS, symbol types (700+ lines)

### Shared Data Structures

Both systems already use the same core output schema:

**WikiStructureSpec:**
```python
class WikiStructureSpec(BaseModel):
    wiki_title: str
    overview: str
    sections: list[SectionSpec]
    total_pages: int

class SectionSpec(BaseModel):
    section_name: str
    section_order: int
    description: str
    pages: list[PageSpec]

class PageSpec(BaseModel):
    page_name: str
    page_order: int
    description: str
    target_symbols: list[str]
    target_docs: list[str]
    target_folders: list[str]
    retrieval_query: str
    metadata: dict[str, Any]
```

This is critical — the cluster planner produces the **exact same `WikiStructureSpec`** that the downstream generation pipeline expects. The rest of the pipeline (page content generation, quality assessment, enhancement, export) remains unchanged.

### Gap Analysis

| Capability | Wikis (current) | DeepWiki | Gap |
|-----------|-----------------|----------|-----|
| Structure planning | DeepAgents filesystem exploration | Deterministic cluster + LLM naming | **Cluster planner missing** |
| Graph schema | NetworkX MultiDiGraph, `lang::file::symbol` composite IDs | NetworkX MultiDiGraph, `lang::file::symbol` composite IDs | **Already aligned** |
| Architectural filtering | `ARCHITECTURAL_SYMBOLS` exists | Same set, more comprehensive | Constants sync |
| Graph expansion | `expansion_engine.py` exists (shared code) | Priority-based P0/P1/P2 expansion | **Already present** (verify parity) |
| FTS5 index | `graph_text_index.py` (SQLite) | `unified_db.py` (SQLite FTS5) | Same technology |
| Community detection | Not present | Hierarchical Leiden via leidenalg (C++) | **New dependency** |
| Hub detection | Not present | PageRank + percentile threshold | **New capability** |
| File contraction | Not present | `_contract_to_file_graph()` | **New capability** |
| Dynamic page sizing | `_split_overloaded_pages()` exists | Full merge/split pipeline | Extend existing |
| Test exclusion | Not present | `is_test` flag + filtering | **Separate proposal** |
| Unified DB | `unified_db.py` exists (feature-flagged OFF) | SQLite consolidated DB | **Separate proposal** (activation + porting) |

---

## Design: Phase 1 — Graph Clustering Engine

### New Dependencies

```
python-igraph == 1.0.0    # Graph conversion for Leiden (NetworkX → igraph)
leidenalg == 0.11.0       # C++ Leiden community detection (REQUIRED — proven pipeline)
```

Both are **required** dependencies. The cluster planner uses ONLY the Leiden algorithm via the specialized `leidenalg` C++ library with `RBConfigurationVertexPartition`. No Louvain fallback — if `leidenalg` is not installed, the cluster planner is unavailable and the user must use the DeepAgents planner.

### New Files

| File | Purpose |
|------|---------|
| `backend/app/core/graph_clustering.py` | Core clustering algorithms (ported from DeepWiki) |
| `backend/app/core/cluster_constants.py` | Clustering constants, resolution formulas, symbol type sets |

### Modified Files

| File | Change |
|------|--------|
| `backend/app/core/code_graph/graph_builder.py` | Add `is_architectural` classification to nodes; add `rel_path` normalization |
| `backend/app/core/constants.py` | Sync `ARCHITECTURAL_SYMBOLS` with DeepWiki; add `EXPANSION_SYMBOL_TYPES`, `PAGE_IDENTITY_SYMBOLS` |

### Clustering Algorithm (Adapted for Wikis)

The DeepWiki clustering pipeline assumes a `unified_db.py` SQLite backend. Wikis uses in-memory NetworkX graphs + SQLAlchemy for persistence. The adaptation:

1. **Input**: NetworkX DiGraph (already built by `EnhancedUnifiedGraphBuilder`)
2. **Processing**: Pure function pipeline operating on NetworkX graph objects
3. **Output**: `ClusterAssignment` dataclass → fed into `ClusterStructurePlanner`
4. **No DB dependency**: All clustering operates in-memory; no `unified_db.py` port needed for Phase 1

```python
@dataclass
class ClusterAssignment:
    """Result of Phase 3 clustering."""
    macro_clusters: dict[str, int]      # node_id → section_id
    micro_clusters: dict[str, int]      # node_id → page_id (within section)
    hub_nodes: set[str]                 # High-degree connector nodes
    hub_assignments: dict[str, int]     # hub_node_id → assigned section_id
    section_count: int
    page_count: int
    stats: dict[str, Any]               # node_count, edge_count, resolution, algorithm
```

---

## Design: Phase 2 — Cluster Structure Planner

### New Files

| File | Purpose |
|------|---------|
| `backend/app/core/wiki_structure_planner/cluster_planner.py` | LLM naming of pre-computed clusters |
| `backend/app/core/wiki_structure_planner/cluster_prompts.py` | Section/page naming prompts |

### Modified Files

| File | Change |
|------|--------|
| `backend/app/core/agents/wiki_graph_optimized.py` | Route to cluster planner when `planner_type="cluster"` |
| `backend/app/core/state/wiki_state.py` | Add `planner_type` field to `WikiState`; add `metadata.cluster_node_ids` to PageSpec |

### Naming Strategy

Two-pass naming (directly ported from DeepWiki):

1. **Section naming** — 1 LLM call per macro-cluster with top 10 dominant symbols (by PageRank)
2. **Page naming** — 1 LLM call per micro-cluster with page-only central symbols (prevents "centroid bleed")

Total cost: ~13K–22K tokens for a typical repository (vs. 30K–80K for DeepAgents).

---

## Design: Phase 3 — Smart Graph Expansion

### Analysis

Wikis already has `backend/app/core/code_graph/expansion_engine.py` (ported from the same DeepWiki codebase). A diff is needed to verify feature parity. Likely gaps:

- Language-specific augmenters (C++ decl/impl, Go receivers, Rust impl blocks)
- Per-type expansion strategies (`_expand_class`, `_expand_function`, etc.)
- Type alias chain resolution
- Global + per-symbol expansion caps

### Modified Files

| File | Change |
|------|--------|
| `backend/app/core/code_graph/expansion_engine.py` | Sync with DeepWiki's P0/P1/P2 priority system; add missing augmenters |
| `backend/app/core/content_expander.py` | Update to use smart expansion when `SMART_EXPANSION_ENABLED=1` |

---

## Design: Phase 4 — Planner Selection & Integration

### API Changes

Add `planner_type` to `GenerateWikiRequest`:

```python
class GenerateWikiRequest(BaseModel):
    # ... existing fields ...
    planner_type: str = "deepagents"  # "deepagents" | "cluster"
```

### SSE Event Contract

The cluster planner emits progress events via the same `invocation.emit()` pattern:

```python
# During clustering (Phase 3 — 0-5% of structure planning)
progress(invocation_id, progress=31, message="Clustering code graph (Leiden)...")
progress(invocation_id, progress=33, message="Detected 8 sections, 24 pages")

# During naming (Phase 4 — 5-15% of structure planning)
progress(invocation_id, progress=35, message="Naming section 1/8: analyzing symbols...")
progress(invocation_id, progress=38, message="Named: 'Authentication & Authorization'")
```

### Feature Flags

| Flag | Default | Purpose |
|------|---------|---------|
| `WIKIS_CLUSTER_PLANNER_ENABLED` | `0` | Enable cluster planner option |
| `WIKIS_CLUSTER_MAX_SECTIONS` | `20` | Hard cap on section count (actual DeepWiki cap is 20) |

---

## Cost & Performance Estimates

| Metric | DeepAgents Planner | Cluster Planner |
|--------|-------------------|-----------------|
| Structure planning tokens | 30K–80K | 13K–22K |
| Structure planning time | 3–15 min (serial tool calls) | 10–30 sec (clustering) + 30–60 sec (naming) |
| Determinism | Low (different runs → different structures) | High (same graph → same clusters) |
| Code relationship awareness | None (files + directories only) | Full (inheritance, calls, composition) |
| New dependencies | None | igraph, leidenalg (optional) |

---

## Risks & Mitigations

| Risk | Impact | Mitigation |
|------|--------|------------|
| igraph/leidenalg install issues | Cluster planner unavailable | Cluster planner disabled; user must use DeepAgents planner |
| Graph schema differences (UUID vs composite IDs) | Cluster algorithms fail | Adapter layer normalizes node IDs to composite format |
| Small repos (< 20 nodes) produce trivial clusters | Poor structure for tiny repos | Fall back to DeepAgents for repos below threshold |
| Naming LLM calls fail | Structure has numbered sections | Fallback names from dominant symbol + directory path |
