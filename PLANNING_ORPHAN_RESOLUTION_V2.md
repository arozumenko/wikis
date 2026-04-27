# Orphan Resolution V2 — Cascade Reorder, Hybrid Search & Embedding Performance

> **Scope**: Three interrelated improvements to orphan resolution in `graph_topology.py`.
> This document does **not** duplicate content from `PLANNING_GRAPH_QUALITY_IMPROVEMENTS.md`
> (IDF gating, node disambiguation, connectivity diagnostics, hub damping, etc.) — it focuses
> exclusively on cascade reordering, hybrid lexical+semantic fusion, and embedding performance
> at scale.

---

## Table of Contents

1. [Motivation](#1-motivation)
2. [Reordered Orphan Resolution Cascade](#2-reordered-orphan-resolution-cascade)
3. [Hybrid Orphan Resolution — Lexical + Semantic Fusion](#3-hybrid-orphan-resolution--lexical--semantic-fusion)
4. [Embedding Performance for Large Repositories](#4-embedding-performance-for-large-repositories)
5. [Why Merging Lexical + Semantic Improves Relevance](#5-why-merging-lexical--semantic-improves-relevance)
6. [How Performance Optimizations Survive Hybrid](#6-how-performance-optimizations-survive-hybrid)
7. [Feature Flags & Configuration](#7-feature-flags--configuration)
8. [Migration & Rollout Strategy](#8-migration--rollout-strategy)
9. [Testing Strategy](#9-testing-strategy)
10. [Files Modified](#10-files-modified)
11. [Relationship to Existing Planning](#11-relationship-to-existing-planning)

---

## 1. Motivation

### Current state

`resolve_orphans()` in `graph_topology.py` (lines 354-629) runs a 3-pass cascade:

1. **FTS lexical match** — `search_by_name(symbol_name)` via FTS5 / tsvector
2. **Semantic vector search** — embed orphan text on-the-fly → KNN against vector store
3. **Directory proximity** — same-directory / parent-directory heuristic fallback

Separately, `inject_doc_edges()` (lines 874-938) runs as a distinct Phase 2 step **after**
`resolve_orphans()` completes. It extracts `[text](path)` markdown hyperlinks and
`` `SymbolName` `` backtick references from documentation nodes and creates doc→code edges.

### Problems

| # | Problem | Impact |
|---|---------|--------|
| P1 | Explicit references (markdown links, backticks, imports) are never used to resolve orphans — they only add supplementary edges after the fact | High-confidence deterministic matches wasted |
| P2 | FTS and semantic are independent sequential passes. FTS "claims" orphans greedily — if it finds *any* match, semantic never gets a chance to find a *better* match | Sub-optimal link relevance |
| P3 | Semantic pass re-embeds each orphan on-the-fly via external API. For 5,000 orphans at `embed_batch_size=64`, that's ~78 batches of API calls | Timeout / OOM on large repos |
| P4 | No score fusion between FTS BM25 scores and vector cosine similarity | Misses cases where combined signal would produce better ranking |

---

## 2. Reordered Orphan Resolution Cascade

### Current order in `run_phase2()` orchestrator

```
resolve_orphans()        ← FTS → Semantic → Directory
inject_doc_edges()       ← Markdown links + backtick refs (separate step)
bridge_disconnected_components()
compute_edge_weights()
detect_hub_nodes()
persist()
```

### Proposed new order — 4-pass cascade

| Pass | Name | Source | Confidence | Processes |
|------|------|--------|------------|-----------|
| 1 | **Explicit Reference Resolution** | Markdown `[text](path)`, backtick `` `Symbol` ``, import/include statements | Highest — deterministic string matching | All orphans |
| 2 | **Hybrid Search (Lexical + Semantic)** | FTS BM25 + vector cosine fused via RRF | High — combines keyword precision with semantic recall | Remaining orphans **with** embeddings |
| 3 | **Lexical-Only Fallback** | FTS `search_by_name` only | Medium — safety net for embedding gap | Remaining orphans **without** embeddings |
| 4 | **Directory Proximity** | `_resolve_orphans_by_directory()` — same/parent directory | Lowest — last resort | Whatever is still left |

Each pass only processes orphans that remain unresolved from previous passes (shrinking orphan set).

### Key changes to `run_phase2()` orchestrator

- Extract hyperlink/backtick extraction logic from `inject_doc_edges()` into a reusable
  `_extract_explicit_references()` helper.
- Call it as **Pass 1** inside `resolve_orphans()` instead of as a separate post-step.
- `inject_doc_edges()` still runs after orphan resolution but only for supplementary
  doc→code cross-references between already-connected nodes (not orphan resolution).

### Implementation spec — Pass 1: `_resolve_orphans_by_explicit_refs()`

```python
def _resolve_orphans_by_explicit_refs(
    self,
    orphan_ids: Set[str],
    G: nx.DiGraph,
    storage: StorageProtocol,
) -> Dict[str, str]:
    """
    Scan orphan doc_text / source_code for explicit references:
    - Markdown hyperlinks: [text](relative/path.py)
    - Backtick refs: `SymbolName`
    - Import statements: import X, from X import Y, require('X'), #include "X"
    
    Match targets against existing node IDs or file paths in the graph.
    Returns {orphan_id: best_target_id}.
    """
```

- For each orphan, scan its `doc_text` and `source_code` for patterns:
  - `[...](path)` → resolve `path` against file nodes in graph
  - `` `SymbolName` `` → exact-match against node `symbol_name` attributes
  - `import X` / `from X import Y` → resolve module path → node ID
  - `require('X')` / `#include "X.h"` → resolve file path → node ID
- Create edges with `edge_type="explicit_ref"` and high confidence weight (0.95)
- **No external API calls** — purely deterministic string matching against graph metadata

---

## 3. Hybrid Orphan Resolution — Lexical + Semantic Fusion

### Current behavior

FTS and vector search are independent sequential passes. If FTS resolves an orphan in
Pass 1, vector search never runs for it. If FTS misses, vector search tries independently.
The two signals are never combined to improve ranking.

### Proposed: Reciprocal Rank Fusion (RRF)

For each orphan in Pass 2:

1. Run FTS `search_by_name(orphan.symbol_name)` → top-N candidates with BM25 ranks
2. Run vector KNN with orphan's embedding → top-N candidates with cosine distance ranks
3. Fuse via RRF to produce a single ranked list
4. Accept top candidate if `RRF_score > rrf_threshold`

```
RRF_score(node) = Σ  1 / (k + rank_i(node))
                  i∈{fts, vec}

where k = 60 (standard RRF constant)
```

### Alternative fusion strategies (evaluate during benchmarking)

| Strategy | Formula | Pros | Cons |
|----------|---------|------|------|
| **RRF** (recommended) | `Σ 1/(k + rank_i)` | Rank-based, no score normalization needed | Ignores score magnitude |
| **Weighted linear** | `α·norm(bm25) + (1-α)·norm(cosine)` | Uses score magnitude | Requires careful normalization |
| **Two-stage re-rank** | FTS pre-filter (top 50) → vector re-rank (top 5) | Lower latency for large candidate sets | Sequential, not truly fused |
| **Conditional hybrid** | Vector only if FTS top-1 below confidence threshold | Saves embedding lookups when FTS is confident | Misses cases where fusion helps |

### Implementation spec — Pass 2: `_resolve_orphans_hybrid()`

```python
def _resolve_orphans_hybrid(
    self,
    orphan_ids: Set[str],
    G: nx.DiGraph,
    storage: StorageProtocol,
    text_index: StorageTextIndex,
    *,
    top_n: int = 20,
    rrf_k: int = 60,
    rrf_threshold: float = 0.02,
) -> Dict[str, Tuple[str, float]]:
    """
    Hybrid orphan resolution combining FTS + semantic vector via RRF.
    
    Returns {orphan_id: (best_target_id, fusion_score)}.
    Only processes orphans that have embeddings available.
    """
```

- Accepts both `storage` (vector store) and `text_index` (FTS) handles
- Edge type: `edge_type="hybrid_orphan"` with weight = normalized fusion score
- Orphans without stored embeddings are **skipped** and fall through to Pass 3 (lexical-only)

### Pass 3: Lexical-Only Fallback

Handles orphans that could not participate in hybrid because they lack embeddings
(non-architectural symbols that weren't embedded during Phase 1b). Uses existing
FTS `search_by_name` logic. This pass exists purely as a safety net for the
embedding coverage gap.

---

## 4. Embedding Performance for Large Repositories

### Current bottleneck

During the semantic pass, each orphan's `doc_text` is embedded on-the-fly via
`storage.embed_text()` → external API call. For a repo with 5,000 orphans, this
means 5,000 embedding calls (even batched at `embed_batch_size=64`, that's ~78 batches).

**Root cause**: Phase 1b (`filesystem_indexer.py`) already embeds all ARCHITECTURAL_SYMBOLS
and stores them in the vector store. But orphan resolution re-embeds orphan text instead
of looking up existing embeddings.

### 4a. Reuse Phase 1b Embeddings ★ High Impact, Low Effort

During Phase 1b, `filesystem_indexer` stores embeddings for all architectural symbols.
In orphan resolution, instead of re-embedding, look up the orphan's existing embedding
from the vector store by node ID.

```python
# In _resolve_orphans_hybrid:
existing_embedding = storage.get_embedding_by_id(orphan_id)
if existing_embedding is not None:
    # Fast path: KNN lookup with pre-computed embedding
    candidates = storage.similarity_search_by_vector(
        existing_embedding, k=top_n, prefix_filter=...
    )
else:
    # Slow path: orphan has no embedding → skip hybrid, fall to Pass 3
    pass
```

- Requires adding `get_embedding_by_id()` to the storage protocol (both SQLite-vec and pgvector)
- **Expected result**: Eliminates ~80% of embedding API calls (most orphans are architectural symbols)

### 4b. Batch KNN Queries ★ High Impact, Medium Effort

Instead of per-orphan KNN queries, collect all orphan embeddings first, then execute
a single batch KNN query.

```python
# Collect all orphan embeddings (from 4a: mostly lookups)
orphan_vectors = batch_get_embeddings(orphan_ids)
# Single batch KNN
results = storage.batch_similarity_search(orphan_vectors, k=top_n)
```

- New storage protocol method: `batch_similarity_search(vectors: List[List[float]], k: int) -> List[List[SearchResult]]`
- For SQLite-vec: iterate `vec_distance_L2` with batch parameter binding
- For pgvector: use batch `<->` queries with `ANY()` or parallel cursors

### 4c. Matryoshka Dimensionality Reduction ★ Medium Impact, Low Effort

`text-embedding-3-large` supports Matryoshka representation — embeddings can be truncated
from 3072d to 256d or 512d with minimal quality loss for approximate matching tasks.

For orphan resolution (approximate graph linking, not precision retrieval), 256d is
likely sufficient.

```python
# Config:
orphan_embed_dimensions: int = 256  # Matryoshka truncation for orphan resolution
# Full 3072d remains for ask/research queries
```

**Tradeoff**: Slightly lower semantic precision vs. ~12× reduction in vector storage
and ~12× faster distance computation.

### 4d. Quantized Vectors ★ Medium Impact, Low Effort

Convert float32 vectors to int8 or binary quantization for orphan resolution:

- sqlite-vec supports int8 quantization
- pgvector supports `halfvec` (float16) and binary quantization

**Tradeoff**: ~2-4× memory reduction, ~2× faster search, marginal quality loss acceptable
for graph linking (not user-facing search).

### 4e. Locality-Aware Pruning ★ Medium Impact, Low Effort

Current `_expanding_prefixes()` generates path prefixes for locality bias. Optimization:

1. First attempt: restrict candidates to **same-module/same-package** nodes only
2. If no match: expand to parent directory
3. If no match: expand to full repo scope

Reduces the candidate set for both FTS and vector halves of hybrid simultaneously.

### 4f. Parallel Embedding with Async I/O ★ Low-Medium Impact, Medium Effort

Current `embed_max_workers` uses `ThreadPoolExecutor`. Improvement: `asyncio.gather()`
with async embedding client for true parallelism. Beneficial when embedding provider
supports HTTP/2 multiplexing.

### Performance targets

| Repo Size | Current Orphan Resolution | Target |
|-----------|--------------------------|--------|
| Small (<500 nodes) | <5s | <2s |
| Medium (500–5,000 nodes) | 30–120s | <10s |
| Large (5,000–20,000 nodes) | 5–15min | <60s |
| Very large (>20,000 nodes) | Often times out | <3min |

---

## 5. Why Merging Lexical + Semantic Improves Relevance

The current separate-pass design has a fundamental flaw: **FTS runs first and claims
orphans greedily**. If FTS finds *any* match above threshold, the orphan is resolved
and semantic never gets a chance to find a *better* match.

### Concrete example

- **Orphan**: `ConfigParser` (a utility class)
- **FTS match**: `ConfigParserTest` — lexically very similar, BM25 rank #1
- **Semantic match**: `ConfigurationManager` — the actual parent module, semantically
  related but different name

With **separate passes**: FTS claims `ConfigParser` → `ConfigParserTest`. Wrong link.

With **hybrid fusion**: Both candidates appear in the fused ranking. `ConfigParserTest`
gets high FTS rank but low semantic rank (test ≠ utility). `ConfigurationManager` gets
moderate FTS rank but high semantic rank. RRF fuses both signals → `ConfigurationManager`
wins. Correct link.

### When hybrid doesn't help

Hybrid provides no benefit when:

- The orphan has no embedding (non-architectural symbol) → falls to lexical-only Pass 3
- FTS and semantic agree on the same top-1 candidate → fusion produces the same result
  (no harm, just redundant computation)

The **lexical-only fallback** (Pass 3) exists as a safety net for orphans without
embeddings. It's not a competing approach — it handles a disjoint subset of orphans.

---

## 6. How Performance Optimizations Survive Hybrid

Hybrid does *more* work per orphan (both FTS + vector), so naively it could be slower
than current sequential passes. Here's why the optimizations make hybrid **faster** than
the current approach:

| Step | Current (separate passes) | Hybrid (with optimizations) |
|------|---------------------------|-----------------------------|
| Embedding | Re-embed each orphan via API (**dominant cost**) | Lookup from Phase 1b store (instant — §4a) |
| FTS search | Per-orphan FTS queries | Same, batched |
| Vector search | Per-orphan KNN (only for FTS misses) | Batch KNN for all orphans at once, 256d Matryoshka (§4b + §4c) |
| Fusion | N/A | RRF in-memory (microseconds) |
| **Total** | **Dominated by embedding API latency** | **Dominated by batch KNN (~ms)** |

The key insight: hybrid isn't "run FTS *then* run semantic" — it's "run both in parallel
over the same candidate set and fuse scores." With pre-computed embeddings (§4a), the
vector half is just a KNN probe against an existing index. With batch queries (§4b) and
Matryoshka truncation (§4c), the entire batch of orphans resolves in a single vectorized
operation.

**Net result**: Hybrid with optimizations is *faster* than current separate passes because
the dominant cost (on-the-fly re-embedding via external API) is completely eliminated.

---

## 7. Feature Flags & Configuration

### New feature flags (`feature_flags.py`)

| Flag | Type | Default | Purpose |
|------|------|---------|---------|
| `orphan_cascade_v2` | `bool` | `True` | Enable new 4-pass cascade order |
| `orphan_hybrid_search` | `bool` | `True` | Enable RRF/hybrid fusion in Pass 2 |
| `orphan_reuse_embeddings` | `bool` | `True` | Reuse Phase 1b embeddings instead of re-embedding |
| `orphan_matryoshka_dim` | `int` | `0` | If > 0, truncate embeddings to this dimension for orphan resolution |

### New config params (`config.py`)

```python
# Hybrid orphan resolution
orphan_rrf_k: int = 60                  # RRF constant
orphan_rrf_threshold: float = 0.02      # Minimum RRF score to accept match
orphan_hybrid_top_n: int = 20           # Candidates per search modality
orphan_hybrid_alpha: float = 0.4        # Weight for weighted linear fusion (if used)
orphan_hybrid_strategy: str = "rrf"     # "rrf" | "weighted" | "two_stage"
```

---

## 8. Migration & Rollout Strategy

### Phase 1: Cascade Reorder + Embedding Reuse (low risk)

1. Ship `orphan_cascade_v2=True` — reorder passes, add explicit-ref Pass 1
2. Ship `orphan_reuse_embeddings=True` — immediate performance win
3. Hybrid flag remains `False` — Pass 2 uses existing FTS-only logic

### Phase 2: Hybrid Fusion (after benchmarking)

4. Enable `orphan_hybrid_search=True` after benchmarking RRF vs. weighted vs. two-stage
   on real repositories (EliteaAI/configurations ~3k nodes, elitea-sdk ~8k nodes)
5. Tune `orphan_rrf_k` and `orphan_rrf_threshold` based on benchmark results

### Phase 3: Matryoshka + Quantization (for large repos)

6. Enable `orphan_matryoshka_dim=256` for repos >10k nodes after quality validation
7. Enable int8 quantization for orphan-resolution vector index (separate from ask/research)

### Backward compatibility

- All changes behind feature flags with `True` defaults after validation
- Old 3-pass cascade available by setting `orphan_cascade_v2=False`
- No schema changes to graph storage — only new edge types (`explicit_ref`, `hybrid_orphan`)
- Storage protocol additions (`get_embedding_by_id`, `batch_similarity_search`) are
  additive and backward-compatible

---

## 9. Testing Strategy

### Unit tests

1. **`_resolve_orphans_by_explicit_refs()`** — verify markdown link parsing, backtick
   matching, import/include resolution against fixture graph
2. **RRF fusion** — synthetic FTS + vector rank lists → verify fused ranking order and
   threshold filtering
3. **`get_embedding_by_id()`** — verify retrieval for both SQLite-vec and pgvector backends
4. **`batch_similarity_search()`** — verify batch KNN returns correct shape and distances

### Integration tests

5. **End-to-end orphan resolution** on a fixture graph — verify cascade order
   (explicit → hybrid → lexical → directory) and that each pass shrinks the orphan set
6. **Regression test** — verify that graphs produced with `orphan_cascade_v2=True` have
   equal or fewer orphans than `orphan_cascade_v2=False`

### Performance benchmarks

7. Measure orphan resolution wall-clock time on:
   - EliteaAI/configurations (~3k nodes)
   - elitea-sdk (~8k nodes)
   - fmtlib/fmt (~15k nodes, C++ heavy)
   
   Compare: old 3-pass vs. new 4-pass vs. new 4-pass + hybrid

### Quality benchmarks

8. Compare orphan resolution accuracy (% correctly resolved) using manually-labeled
   ground truth from `INVESTIGATION_GRAPH_QUALITY.md` orphan archetype findings
9. Measure false-positive rate (incorrect links) for each fusion strategy

---

## 10. Files Modified

| File | Changes |
|------|---------|
| `backend/app/core/graph_topology.py` | New `_resolve_orphans_by_explicit_refs()`, new `_resolve_orphans_hybrid()`, reordered cascade in `resolve_orphans()`, refactored `inject_doc_edges()` to extract shared helper |
| `backend/app/core/storage/__init__.py` | Add `get_embedding_by_id()`, `batch_similarity_search()` to `StorageProtocol` |
| `backend/app/core/storage/factory.py` | Implement new protocol methods for SQLite-vec and pgvector backends |
| `backend/app/core/storage/text_index.py` | No changes needed — FTS interface already sufficient |
| `backend/app/core/feature_flags.py` | Add 4 new flags: `orphan_cascade_v2`, `orphan_hybrid_search`, `orphan_reuse_embeddings`, `orphan_matryoshka_dim` |
| `backend/app/config.py` | Add hybrid config params (`rrf_k`, `rrf_threshold`, `hybrid_top_n`, `hybrid_alpha`, `hybrid_strategy`) |
| `backend/tests/unit/test_graph_topology.py` | New test cases for reordered cascade, explicit-ref resolution, and hybrid fusion |

---

## 11. Relationship to Existing Planning

| Existing Document | Relationship |
|-------------------|-------------|
| **PLANNING_GRAPH_QUALITY_IMPROVEMENTS.md §1** (IDF-gated lexical resolution with T1-T4 tiers) | Complementary — IDF gating applies *within* lexical search scoring. That tiered cascade operates inside what becomes Pass 3 (lexical-only fallback) in this plan. No duplication. |
| **PLANNING_GRAPH_QUALITY_IMPROVEMENTS.md §11.7** (Edge confidence model) | Synergistic — hybrid RRF fusion scores naturally map to edge confidence tiers. Higher RRF score → higher confidence tier. |
| **INVESTIGATION_GRAPH_QUALITY.md** (orphan archetypes) | Direct resolution — the "documented but disconnected" archetype is directly addressed by explicit-ref Pass 1. The "semantically related but lexically different" archetype is addressed by hybrid Pass 2. |
| **PLANNING_GRAPH_QUALITY_IMPROVEMENTS.md §11.1-§11.13** (C++ parser, universal doc model, hub damping, etc.) | Orthogonal — those are parser/model improvements, not orphan resolution changes. No overlap. |