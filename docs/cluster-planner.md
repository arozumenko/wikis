# Cluster-Based Structure Planner

The cluster planner is an alternative wiki structure generation strategy that uses **graph-based Leiden community detection** instead of LLM prompting to organize wiki sections and pages.

## When to Use

| Planner | Best For | How It Works |
|---------|----------|--------------|
| `agent` (default) | Small-medium repos, natural language structure | LLM reads the codebase and proposes sections/pages |
| `cluster` | Large repos (1000+ symbols), deterministic structure | Graph clustering groups related code; LLM only names sections |

## Configuration

Set via environment variables or the API request body:

| Variable | Default | Description |
|----------|---------|-------------|
| `PLANNER_TYPE` | `agent` | `agent` or `cluster` |
| `CLUSTER_HUB_Z_THRESHOLD` | `3.0` | Z-score threshold for hub detection (higher = fewer hubs) |
| `CLUSTER_EXCLUDE_TESTS` | `true` | Exclude test files from clustering |

## API Usage

Pass `planner_type` in the wiki generation request:

```json
{
  "repo_url": "https://github.com/owner/repo",
  "planner_type": "cluster",
  "exclude_tests": true
}
```

## How It Works

1. **Graph Topology Enrichment** — Adds documentation edges, bridges disconnected components, resolves orphan nodes
2. **Architectural Projection** — Filters the code graph to architectural symbols (classes, functions, interfaces, enums, etc.)
3. **Hub Detection** — Identifies highly-connected hub nodes (z-score > threshold) and temporarily removes them
4. **Leiden Clustering** — Runs hierarchical Leiden community detection on a file-contracted graph (macro sections) then on per-section node graphs (micro pages)
5. **Page Consolidation** — Merges small clusters, splits oversized ones, reintegrates hubs
6. **LLM Naming** — Sends cluster summaries to the LLM for human-readable section/page names
7. **Smart Expansion** — Each page's symbols are expanded with graph neighbours for rich context

## Key Modules

| Module | Purpose |
|--------|---------|
| `cluster_constants.py` | Symbol tiers, architectural types, sizing thresholds |
| `graph_clustering.py` | Leiden clustering, hub detection, file contraction |
| `graph_topology.py` | Phase 2 enrichment: doc edges, component bridging, orphan resolution |
| `cluster_planner.py` | Structure planner that builds WikiStructureSpec from clusters |
| `cluster_prompts.py` | LLM prompt templates for section/page naming |
| `cluster_expansion.py` | Smart context expansion per page |
| `shared_expansion.py` | Shared expansion utilities (token budgets, node scoring) |
| `language_heuristics.py` | Per-language parsing hints and budget fractions |
| `unified_db.py` | SQLite database for code graph + FTS5 + sqlite-vec |
| `unified_retriever.py` | LangChain-compatible retriever backed by UnifiedWikiDB |
