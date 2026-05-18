# Graph Quality Investigation — `feat/unified-db-and-clustering`

Scope: connectivity, cluster splitting, orphan archetypes, class transitive linking, and page-coverage gaps observed on three production caches:

| repo                          | nodes  | edges   | clusters | orphan rate (arch) |
| ----------------------------- | -----: | ------: | -------: | -----------------: |
| `EliteaAI/configurations`     |    763 |   2 920 |        5 |              21.3% |
| `EliteaAI/elitea-sdk`         | 27 547 | 130 543 |       12 |              47.5% |
| `fmtlib/fmt` (C++)            | 11 380 |  45 131 |        5 |              58.6% |

Audit & drill scripts: `/tmp/connectivity_audit.py`, `/tmp/connectivity_drill.py`.

---

## 1. Headline findings

### F1. The "missing pages" are not orphans — they are drowned in a mega-cluster

| symbol           | cluster | direct edges (non-defines) | members | external edges via members |
| ---------------- | ------: | -------------------------: | ------: | -------------------------: |
| `Assistant`      |       0 |                         18 |      34 |                        580 |
| `EliteAClient`*  |       0 |                         21 |     102 |                      1 036 |
| `BaseToolApiWrapper` |   5 |                         91 |      10 |                         24 |

*Important*: the symbol is `EliteAClient` (capital "A"), not `EliteaClient`. Any planner code searching for `EliteaClient` will silently find nothing.

Both `Assistant` and `EliteAClient` live in cluster 0, which contains **7 060 nodes (1 222 architectural)**. The planner sees the class with a tiny direct-edge fan-out compared to neighbors; the page slot for them is contested by hundreds of similarly-sized class nodes inside the same mega-cluster.

The class itself looks "small" because **all the relevance is one hop down through its methods**:

- `Assistant`: 18 inbound non-defines edges, but its 34 methods participate in 580 external edges.
- `EliteAClient`: 21 inbound non-defines edges, but its 102 methods participate in 1 036 external edges.

This is exactly the pattern the user described:

> classes ... direct links are only available for inheritance/implementation but other links they are two-hop through the transitive dependencies (fields and methods mostly via their connections).

### F2. The mega-cluster (cluster 0) is real and structurally heterogeneous

130 files, 21 300 same-file structural edges, 6 661 cross-file edges (24% cross-file). It merges:

| prefix              |  nodes | arch |
| ------------------- | -----: | ---: |
| `runtime/`          | 4 229  | 512  |
| `tools/`            | 1 494  | 124  |
| `cli/`              |   518  |  73  |
| `.elitea/`          |   302  | 301  |
| `community/`        |   228  |  15  |

**Hub culprit**: `runtime/clients/artifact.py` appears in 8 of the top-10 cross-file edge pairs (figma, langraph_agent, github, postman, llm, inventory, tools/artifact, clients/client). One central file with edges to 130 others is enough to defeat modularity-based community detection — Leiden treats it as a "shortcut" and absorbs both endpoints into the same community.

Cluster 0 also pulls in 178 markdown nodes from a single agent file (`test-creator.agent.md`) and 200 markdown sections from `cli/README.md`. Heterogeneous content + structural hub = "merge everything" outcome.

### F3. Orphan stats are inflated by **non-code artifacts** that should never be tagged architectural

The 47.5% orphan rate on elitea-sdk is dominated by:

| symbol_type        | orphans | example sources                          |
| ------------------ | ------: | ---------------------------------------- |
| `markdown_section` |   1 290 | `cli/README.md` (200), agent `.md` files |
| `yaml_document`    |     401 | configs, GH workflows                    |
| `text_chunk`       |     250 | **PDF test fixtures, shell scripts**     |

**`text_chunk` is never a valid architectural symbol** — top sources are:

- 164 chunks from `test-image.pdf`
- 36 chunks from `.github/workflows/test-runner-reusable.yml`
- 27 chunks from `run_all_suites.sh`

These are byproducts of the document/embedding pipeline that escape into `is_architectural=1`. They should be filtered in the parser/indexer.

### F4. Class transitive coupling (D6) — 152 "rescuable" classes

After excluding `defines` from the direct-edge count (so the class isn't credited for owning its own methods):

| direct | transitive | classes |                                  meaning |
| -----: | ---------: | ------: | ---------------------------------------: |
|     >0 |         >0 |     484 |                          well-connected  |
|     >0 |          0 |      84 |             defined but no member usage  |
|      0 |         >0 | **152** | **rescuable via member edge propagation** |
|      0 |          0 |      41 |                          truly isolated  |

Top "rescuable" classes (would gain pages with member-edge promotion):

- `EliteAGmailToolkit` — 4 024 transitive
- `SubgraphRunnable`, `SandboxClient`, `ElasticConfig`, `SwarmResultAdapter`, `EliteAQTestApiDataLoader`, `ProcessPdfArgs`, `OpenApiConfig`, `StepBackSearchDocumentsModel`...

Assistant and EliteAClient are **above this threshold (>0 direct)** so they're not in this list — their problem is being buried in cluster 0, not edge starvation.

The 41 truly-isolated classes are mostly Pydantic v1 inner `Config` classes and exception classes — by design, no propagation can rescue them.

### F5. Local-symbol leakage detection is incomplete

Q4 (`parent_symbol IN ('function','method','lambda')`) returned **0** results in all three DBs. But the constant-orphan breakdown shows:

| region                 | constants |
| ---------------------- | --------: |
| sdk source             |       262 |
| test-pipeline scripts  |        21 |
| unit tests             |         6 |

The 21 + 6 + (likely many more in "sdk source") that are module-level constants used only inside one function are effectively local — but our parser stores them with `parent_symbol = '<module>'` or similar, so the heuristic misses them. Detection must be **path-based** (`is_test=1`, `rel_path LIKE '.elitea/%' OR LIKE 'tests/%'`) and **scope-based** (`only used by symbols defined in the same file`).

### F6. Toolkit-registry functions look like orphans because of dynamic dispatch

The function-orphan sample is dominated by `get_tools` / `get_toolkit` / `get_available_tools` per integration package:

```
get_toolkit @ elitea_sdk/tools/ado/test_plan/__init__.py
get_tools   @ elitea_sdk/tools/ado/repos/__init__.py
get_tools   @ elitea_sdk/tools/aws/__init__.py
... (dozens more)
```

These are real entry points called via `__import__` / `getattr` from `elitea_sdk/tools/__init__.py`'s aggregator. Static AST analysis cannot see those calls, **and FTS-on-name won't help either** — the aggregator iterates package metadata, never mentioning `"get_tools"` literally per integration. We need a different signal:

- **Naming convention** (`name in {get_tools, get_toolkit, get_available_tools}` AND parent file is `__init__.py`) → synthetic edge from the package node
- Or a **manifest pass** that walks `tools/__init__.py`'s registry and attaches edges for each registered toolkit

### F7. Page-coverage cannot be audited from the DB today

`wiki_meta.repository_analysis` holds **LLM-generated prose** (an executive summary), not a structured page array. Auditing "do we have a page for X" requires re-running the planner. Suggest persisting the page list separately (e.g., `wiki_meta.pages_json`) so post-hoc audits are possible.

---

## 2. Root causes ranked by impact

| #  | root cause                                                                | observable symptom                                              | effort |
| -- | ------------------------------------------------------------------------- | --------------------------------------------------------------- | ------ |
| R1 | Hub files (`artifact.py`) and broad markdown indexing collapse Leiden     | 7 060-node mega-cluster on a 27 K-node graph                    |  high  |
| R2 | Class node carries no transitive credit from member edges                 | 152 "rescuable" classes; planner ignores well-used classes      |   med  |
| R3 | Doc/data symbols (`markdown_section`, `yaml_document`, `text_chunk`) tagged architectural | 47.5% orphan rate is mostly false-positive doc noise |   low  |
| R4 | `text_chunk` produced from PDFs and shell scripts, then marked arch       | 250 nonsense "orphan" nodes                                     |   low  |
| R5 | Toolkit registry functions invisible to AST + FTS                         | dozens of `get_tools` orphans                                   |   low  |
| R6 | Local-leakage detection only checks `parent_symbol`                       | constants leaked from test scripts and helpers go uncaught      |   low  |
| R7 | No persisted page list; planner output not audit-friendly                 | Q6 returns 0 pages on every DB                                  |   low  |
| R8 | Symbol naming inconsistency (`EliteAClient` vs `EliteaClient`)            | planner "missing page" complaints can be lookup typos           |  trivial |

---

## 3. Recommended remediations (prioritized)

### P0 — Stop indexing non-symbols as architectural (low effort, immediate orphan-rate fix)

In the parser/indexer, set `is_architectural = 0` for:

- `text_chunk` (always — it's an embedding fragment)
- `markdown_section` of files matching `**/*.agent.md`, `**/README.md`, `**/test-plan*.md` unless explicitly opted-in
- `yaml_document` for `.github/workflows/**`, `mkdocs.yml`, lockfiles

Expected effect on elitea-sdk: arch count drops from ~3 884 to ~2 000; orphan rate drops below 20%; cluster 0 loses ~500 doc nodes.

### P1 — Promote class transitive edges into clustering & relevance (medium effort, fixes Assistant/EliteAClient drowning)

Add a graph-augmentation pass after symbol parsing:

```python
# pseudo: for each class, add a synthetic edge to every external symbol
# that one of its methods/fields references, with weight = count
for cls in classes:
    members = defines_of(cls)
    for m in members:
        for tgt in non_defines_targets_of(m):
            if file_of(tgt) != file_of(cls):
                add_edge(cls, tgt, rel_type='member_uses',
                         edge_class='structural', weight=1)
```

This single change:

- Boosts `Assistant`'s direct external degree from ~10 to ~580.
- Boosts `EliteAClient`'s from ~21 to ~1 036.
- Pulls `EliteAGmailToolkit`, `SwarmResultAdapter`, etc. above the planner's relevance threshold.
- In Leiden, gives high-cohesion classes a chance to anchor their own community instead of melting into cluster 0.

### P2 — Tame mega-clusters with hub damping + recursive split (medium-high effort)

Two complementary techniques:

1. **Hub damping at edge-weight time**: when contracting to file-graph, divide the contributed edge weight by `log(degree(file_a) + degree(file_b))`. This down-weights `artifact.py`'s edges so it stops monopolizing modularity.

2. **Recursive Leiden when |cluster| > N**: after the first pass, if any cluster exceeds `MAX_CLUSTER_SIZE` (e.g., 1 500 nodes or 25% of the graph), re-run Leiden on its induced subgraph at higher resolution. Stop when no cluster exceeds threshold or recursion depth ≥ 3.

Cluster 0 has 24% cross-file structural edges — there's enough modularity left to split it into ~5 sub-clusters along `runtime/`, `tools/`, `cli/`, `community/`, `.elitea/`.

### P3 — Toolkit-registry synthetic edges (low effort)

In the post-parse pass, when a `__init__.py` defines `get_tools` / `get_toolkit` / `get_available_tools` and the parent directory contains other modules:

- Add `defines` edge from package directory node → these functions
- Add `references` edge from any caller of `tools/__init__.py`'s aggregator → each integration's `get_tools`

Removes ~80 false-positive function orphans.

### P4 — Path/scope-based local-leakage detection (low effort)

Replace Q4's `parent_symbol IN (...)` check with a multi-signal heuristic:

```sql
-- "leaked local" = arch-tagged constant whose only users are in the same file
SELECT n.* FROM repo_nodes n
WHERE n.symbol_type = 'constant' AND n.is_architectural = 1
  AND NOT EXISTS (
    SELECT 1 FROM repo_edges e
    JOIN repo_nodes user ON user.node_id = e.source_id
    WHERE e.target_id = n.node_id
      AND user.rel_path != n.rel_path
  )
```

Combine with `is_test = 1` and path-prefix heuristics for the common case of `tests/`, `.elitea/tests/`, `scripts/`.

### P5 — Persist a structured page list in `wiki_meta` (low effort)

Add a `pages` key alongside `repository_analysis`:

```json
{"pages": [{"id": "...", "title": "Assistant", "primary_symbol_id": "python::assistant::Assistant", "cluster": 0}, ...]}
```

This unlocks downstream auditing without re-running the planner.

### P6 — Symbol-name lookup must be case-insensitive (trivial)

Anywhere the planner accepts symbol names from prompts/configs, normalize to a case-insensitive match against `symbol_name`. Avoids invisible misses on `EliteAClient` vs `EliteaClient`.

### P7 — Cross-resource linking (graphify v4 inspiration — exploratory)

Defer to a follow-up: review https://github.com/safishamsi/graphify/blob/v4 for LLM-assisted cross-language and cross-resource linking strategies (e.g., linking a Python class to a YAML config that configures it, or to a Markdown doc that documents it). This addresses the long tail of orphans where static analysis cannot bridge resource boundaries.

---

## 4. Validation plan

After implementing P0+P1+P3+P4 on a fresh build of `elitea-sdk`:

| metric                                            | current | target |
| ------------------------------------------------- | ------: | -----: |
| arch nodes                                        |   3 884 | ≤ 2 100 |
| orphan rate (arch)                                |   47.5% |   <15% |
| largest cluster size (% of graph)                 |   25.6% |   ≤10% |
| `Assistant` external edges (direct)               |      18 |  ≥150  |
| `EliteAClient` external edges (direct)            |      21 |  ≥300  |
| classes with direct=0 ∧ transitive>0              |     152 |    <30 |
| `text_chunk` orphans                              |     250 |      0 |
| function orphans matching `get_tools/get_toolkit` |     ~80 |      0 |

If P2 is also implemented, expect cluster 0 to split into 4–6 sub-clusters of 800–2 000 nodes each, with `runtime/clients/` and `runtime/langchain/` becoming distinct.

---

## 5. Open questions

1. Is `text_chunk` produced by the document loader pipeline ever supposed to be architectural? (Suspected no — needs confirmation in parser.)
2. Should `markdown_section` orphans get rescued via parent-doc `defines` edges (already proposed for orphan resolution in cluster_expansion) or filtered out of arch entirely?
3. Does the planner already have a notion of "hub class via member edges", or is it strictly direct-edge-driven? If the former, R2 may already be partially implemented but not weighted high enough.
4. For `fmt` (C++): the audit hasn't been run; does C++ exhibit the same hub/mega-cluster pattern, or is the C++ parser more frugal with arch tagging?

---

## 6. Cross-language picture — `fmtlib/fmt` (C++)

Audit on the C++ DB confirms the patterns are **not Python-specific**, but with different orphan archetypes and a more pronounced parser deficit.

### 6.1 Cluster topology — fmt is *more* concentrated than elitea-sdk

| cluster | nodes  | arch    | non-arch | dominant types                                        |
| ------: | -----: | ------: | -------: | ----------------------------------------------------- |
|       0 | 2 454  |    884  |    1 570 | function/method/struct/macro (the entire C++ surface) |
|       3 |   990  |    990  |        0 | **markdown_section only** — pure doc cluster          |
|       1 |   756  |    755  |        1 | **markdown_section + yaml** — pure doc/config cluster |
|       2 |   310  |     91  |      219 | header-only utility code                              |
|       4 |     2  |      2  |        0 | yaml fragments                                        |

Cluster 0 is **54% of all arch nodes** (vs 26% in elitea-sdk). All edges in clusters 0 and 2 are 100% intra-cluster — Leiden cannot split them at all. There is literally one giant code community for the whole library.

**Counter-intuitive positive**: in fmt, doc nodes form their own clusters (3 and 1) instead of being absorbed into the code mega-cluster. This is what we want — and it works in fmt because there's no `runtime/clients/artifact.py`-style hub bridging code↔markdown. Confirms the hub-damping hypothesis from §3 (P2): kill the bridges, and Leiden naturally separates content types.

### 6.2 Orphan archetypes — language-specific, not universal

| archetype                           | elitea-sdk |   fmt | comment |
| ----------------------------------- | ---------: | ----: | ------- |
| `markdown_section`                  |     1 290  | 1 712 | universal — both repos flooded |
| `text_chunk`                        |       250  |    20 | universal but smaller in C++ (no PDF fixtures) |
| `yaml_document`                     |       401  |    10 | Python repo has many config schemas |
| `macro`                             |        ~  |   772 | **C++-specific** — header guards (`FMT_ARGS_H_`, `FMT_COMPILE_H_`) |
| `type_alias`                        |        ~  |   118 | **C++-specific** — template metafunctions (`decay_t`, `nullptr_t`, `void_t_impl`) |
| `constant`                          |       262 |   115 | language-neutral; in fmt, color enum constants `black`/`red`/`green` |
| `function` (header-only/operators)  |        ~  |    93 | **C++-specific** — `operator""_cf`, `operator[]`, free functions in headers |
| `class`/`struct`                    |       ~50 |   112 | mostly gtest internals + `std::` traits |

### 6.3 Class transitive coupling — C++ parser is worse than Python

| state                            | elitea-sdk |    fmt |
| -------------------------------- | ---------: | -----: |
| direct>0 / transitive>0          |        484 |    412 |
| direct>0 / transitive=0          |         84 |    163 |
| direct=0 / transitive>0          |   **152**  |  **0** |
| direct=0 / transitive=0          |         41 |    148 |

In fmt, **0 classes are rescuable via member edges** because the C++ parser does not emit external edges for member methods. Every class is either directly connected or completely isolated. This is a much stronger version of the same gap the Python parser has — Python at least gets the within-method calls right; C++ does not propagate at all.

### 6.4 Edge-class distribution in fmt — what's actually in the graph

```
structural   34 433  (references 18594, calls 6373, creates 3715, defines_body 2625, defines 2298, ...)
lexical       6 279  lexical_link
directory     2 216  directory_link
semantic        920  semantic_link
bridge          808  component_bridge
doc             475  hyperlink + proximity
```

Note `defines_body` (2 625) and `specializes` (87) appear in fmt — these are C++-only relations the parser emits. `decorates` (2) is essentially unused outside Python. The lexical/semantic/bridge edge classes exist but are tiny vs structural; they won't move clustering needles on their own.

### 6.5 Q4 leakage detector returned 0 in fmt too

Same gap — module-level constants in C++ headers (color enums, `kSomething`-style consts) have `parent_symbol=<file>` so the heuristic misses them. P4 (path/scope-based detection) applies cross-language.

### 6.6 What this implies for the remediation plan

| remediation                | helps Python? | helps C++? | helps a Go/Rust/TS repo? (predicted) |
| -------------------------- | :-----------: | :--------: | :----------------------------------: |
| P0 demote text_chunk/markdown overflow | ✅ | ✅ | ✅ universal |
| P1 class member-promotion edges        | ✅ | ❌ (parser blocker first) | ✅ |
| P2 hub damping + recursive Leiden      | ✅ | ✅ (cluster 0 = 54%) | ✅ |
| P3 toolkit-registry synthetic edges    | ✅ | ⚠️ (different pattern: header-only template functions) | ⚠️ language-specific recipes |
| P4 path/scope leakage detection        | ✅ | ✅ | ✅ |
| P5 persist pages list                  | ✅ | ✅ | ✅ |
| **NEW P8 macro / type_alias / header-guard demotion** | n/a | ✅ | ⚠️ need per-language tables |

**New cross-language item P8**: each language parser needs a "what is genuinely architectural" allow-list. C++ should not tag `FMT_ARGS_H_`-style header guards (regex `^[A-Z_]+_H_*$` + zero call-sites = guard). C++ template metafunctions like `void_t_impl` are arguably internal — could be demoted unless directly named in a public API. Python equivalents: dunder-only classes, single-use Pydantic `Config` inner classes.

### 6.7 What we still don't know without indexing redpanda

The user's instinct to index redpanda (C++/Go/Rust/Python/TS/Java mix in one repo) is correct — three things would be measurable for the first time:

1. **Cross-language structural edges** — does our parser emit *any* edge that crosses a language boundary today? (Suspected no — every parser is language-siloed.) This is the most likely root cause for "siloed" clusters when one project has multiple languages.
2. **Language-specific orphan baselines** — Go/Rust have very different idioms (interfaces, traits, `pub use` re-exports); we need ground-truth orphan rates per language to know whether 58% in C++ is bad-parser or just-the-language.
3. **Whether the doc-cluster cleanliness in fmt holds** when the same repo also has a generated OpenAPI in YAML, a CLI in Python, and a docs site in markdown — i.e., does the bridge problem return at scale?

Recommendation: if we get a redpanda index, we should re-run `connectivity_audit.py` against it before promoting any of P1–P8 to design.

---

## 7. graphify v4 — what's worth borrowing, what isn't

I read the v4 README and ARCHITECTURE.md (no internal `cluster.py` or `analyze.py` source available without cloning). The project is mostly an LLM-orchestration shell; the algorithmic substrate is identical to ours: NetworkX + Leiden via graspologic + tree-sitter for AST. So the question is what *upstream* ideas are worth porting.

### 7.1 Worth borrowing — high signal-to-effort

**B1. Confidence labels on edges (`EXTRACTED` / `INFERRED` / `AMBIGUOUS`)**

Their schema: every edge carries one of three confidence levels with an optional 0.0–1.0 score for `INFERRED`. We already have `edge_class` (structural / lexical / semantic / etc.) which is roughly the same axis but not explicit. If we add a `confidence` column to `repo_edges`:

- The proposed P1 `member_uses` edges become `INFERRED` with `confidence=1.0` (deterministic propagation).
- The proposed P3 toolkit-registry edges become `INFERRED` with `confidence=0.9`.
- Future LLM-derived `semantic_link` edges become `INFERRED` with model-score.
- Auditing becomes trivial: "which clusters depend on guesses?".

This is a **trivial schema change** with outsized leverage on later LLM-augmentation work.

**B2. "God nodes" diagnostic = our hub problem, formalized**

graphify reports the top-N degree nodes as "god nodes" in `GRAPH_REPORT.md` and recommends investigating them. We've already spotted `runtime/clients/artifact.py` ad-hoc; let's make this a first-class report metric (top-10 degree per cluster, per file, per symbol_type) so over-connectedness is visible without writing one-off SQL.

**B3. Hyperedges for shared-protocol / shared-pattern groups**

graphify supports edges connecting 3+ nodes (e.g., "these 12 classes implement `IRunnable`", "these 7 modules form the auth flow"). We have no native support — we'd encode it as N pairwise edges or invent a `pattern` node. The toolkit-registry pattern (`get_tools` across 60+ packages) is the textbook hyperedge use case. Recommendation: introduce `repo_node_groups(group_id, group_kind, member_node_id)` table; a single row in `repo_node_groups` of kind `toolkit_registry` linking 60+ `get_tools` functions is cleaner than 60 `references` edges to a synthetic anchor.

**B4. `rationale_for` nodes from comments (`# WHY:`, `# NOTE:`, `# HACK:`, `# IMPORTANT:`)**

graphify extracts intent comments as separate nodes attached to the symbol. In our wiki-generation use case this is *gold* — page generators currently have to fish docstrings out of `source_text`; a clean `rationale` edge would let the planner answer "why was this written this way?" directly. Cost: a regex pass over `source_text`. Effort: trivial.

### 7.2 Don't borrow — wrong fit

**D1. LLM-per-file extraction.** graphify uses Claude subagents for *every* doc/paper/image. We already pay LLM cost during page generation; doubling it in the index pass is unjustified at our scale (27 K-node repos).

**D2. `semantically_similar_to` from LLM.** Clever, but we have unified DB + text_index protocol (SQLite/postgres) that can compute cosine similarity over function bodies cheaply without an LLM. Add deterministic similarity edges before reaching for an LLM-derived equivalent. (Note: FAISS in this repo is currently used only for the Q&A cache, not for indexing — so this work would extend the text_index protocol, not introduce a new vector store.)

**D3. Whisper transcripts → graph.** Out of scope.

**D4. Their cluster-merging strategy is the same Leiden as ours.** No magic — they explicitly say "Clustering is graph-topology-based — no embeddings ... Leiden finds communities by edge density." They have the same mega-cluster failure mode we do; their answer is to surface it in a report, not to fix it. P2 (hub damping + recursive Leiden) is still our best bet — graphify offers nothing here.

### 7.3 What graphify reveals about our blind spots

The fact that a popular tool ships a "god node" report suggests *every* graph-of-code project hits hub-induced mega-clusters. There is no off-the-shelf algorithmic fix; the standard remediation is post-hoc explanation, not pre-clustering surgery. Our P2 approach (alter edge weights before clustering) is **more ambitious than what graphify attempts** and worth pursuing precisely because it's not solved upstream.

---

## 8. text_chunk parent linkage — critical pushback on the proposed fix

User proposal: *"connect all chunks to the document they were produced from … so that we have at least parent for the multi-content pages (code + docs)."*

**Half right, half not enough. Here's the criticism.**

### 8.1 What's right about the proposal

- A chunk with no parent edge is unrescuable — it sits in the orphan list forever, can never be expanded into context, and can never be cited by name. **Parent linkage should be mandatory.**
- For multi-content pages, the planner does need a way to reach related markdown content from a code anchor. A chunk-to-parent edge is part of that path.
- Fixes the visible symptom: chunks are no longer "orphans" once they have a parent.

### 8.2 Why parent linkage alone doesn't fix the real problem

Three observations from the audit data:

1. **Orphan inflation is a planner problem, not a graph problem.** The 47.5% orphan rate is bad because it (a) makes the relevance signal noisy, (b) pollutes "missing pages" diagnostics, and (c) bloats the architectural-node count the planner must rank. Adding a parent edge converts an "orphan" into a "1-degree leaf" — the relevance signal is still useless. The planner doesn't care about leaves any more than orphans.

2. **"200 markdown_section nodes for `cli/README.md`" is the fundamental issue.** Whether they hang off the document via `defines` or sit alone, **they should not all be page-eligible**. The wiki generator does not want 200 candidate pages from one README. Linking them to a parent doesn't reduce the candidate count — it just gives the planner a chain to walk.

3. **`text_chunk` is genuinely a different kind of object than `markdown_section`.** A markdown_section corresponds to a heading and is a meaningful retrieval unit. A `text_chunk` is a fixed-window slice of bytes — there is no semantic claim that chunk-N of a PDF "is about" anything. Tagging chunks as `is_architectural=1` was always wrong; making them connected orphans doesn't make them architectural.

### 8.3 Refined proposal — three-tier doc model

Replace the binary `is_architectural` with a tiered taxonomy for doc/data nodes:

| tier             | nodes that qualify                             | use                                  | `is_architectural` |
| ---------------- | ---------------------------------------------- | ------------------------------------ | :----------------: |
| **anchor**       | `markdown_document`, `*.agent.md`, `README.md` per file | one per file; planner-eligible       |         1          |
| **navigable**    | `markdown_section` at depth ≤ 2 (H1/H2)        | TOC entries; can be cross-referenced |         1          |
| **retrievable**  | `markdown_section` depth ≥ 3, `text_chunk`     | FTS hits during page expansion only  |         0          |

And **always** emit:

- `markdown_document` → `markdown_section` via `defines`
- `markdown_section` → `text_chunk` via `contains`
- `text_chunk` → `markdown_section` via `chunk_of` (reverse, for retrieval)

This way:

- The user's "multi-content page" use case works: a code page about `Assistant` can hold a `cross_reference` edge to the `markdown_document` for `runtime/langchain/README.md` (anchor), and the page generator pulls in relevant H2 sections (navigable) and chunks (retrievable) via FTS scored against the page topic.
- Orphan stats become honest: 1 800+ false orphans disappear from the count overnight without losing any information.
- The planner sees `cli/README.md` as **one** page candidate, not 200.
- Chunks remain in the DB and FTS index for content retrieval — nothing is deleted, just downgraded.

### 8.4 The "code + docs" page case — addressed

A page like "Assistant agent factory" should be generated from:

1. `class Assistant` (structural anchor)
2. `assistant.py` `module_doc` (code-side rationale)
3. **Hyperedge** linking `Assistant` to relevant doc anchors: `runtime/langchain/README.md`, `.elitea/agents/test-fixer.agent.md`, etc.
4. At expansion time: pull retrievable sections + chunks via FTS scored against the symbol name + module_doc.

Steps 3 and 4 do not need 200-section-per-doc nodes promoted to architectural. They need:

- Better cross-resource hyperedges (graphify B3 — see §7.1).
- A retrieval-first content expansion phase (we already have unified-DB text_index + retriever logic; FAISS in this repo is Q&A-cache-only).

### 8.5 Verdict

Implement **P0 (demote chunks/sections by tier)** AND **always emit parent edges** simultaneously. The user is right that parent edges are needed; I'm pushing back on the implication that parent edges *replace* the demotion. They complement it. Without demotion, the planner still drowns in section-pages; without parent edges, retrieval can't walk back to the anchor.

---

## 9. Updated remediation table (incorporating §6, §7, §8)

| #  | item                                                                  | priority | repos that benefit |
| -- | --------------------------------------------------------------------- | :------: | :-----------------: |
| P0 | Three-tier doc model: anchor / navigable / retrievable + parent edges |   **P0** | all                 |
| P1 | Class member-promotion edges (`member_uses`, INFERRED, conf=1.0)      |     P1   | Python (C++ blocked on parser) |
| P2 | Hub damping + recursive Leiden when cluster > N                       |     P1   | all (fmt cluster 0 = 54%) |
| P3 | Toolkit-registry hyperedge (graphify B3 pattern)                      |     P2   | Python (`tools/*/__init__.py`) |
| P4 | Path/scope-based local-leakage detection                              |     P2   | all                 |
| P5 | Persist structured `pages` list in `wiki_meta`                        |     P2   | all                 |
| P6 | Case-insensitive symbol-name lookup                                   |     P3   | trivial             |
| **B1** | **Confidence column on `repo_edges`** (graphify-style)            |     P1   | unlocks P1/P3/P8 audits |
| **B2** | **"God nodes" report metric in `wiki_meta`**                      |     P2   | all (visibility)    |
| **B4** | **`rationale` edges from `# WHY:`/`# NOTE:`/`# HACK:` comments** |     P2   | all (page generator) |
| **P8** | **Per-language "what is architectural" allow-list** (incl. C++ macro/type_alias demotion) | P1 | C++/Go/Rust priority |
| **P9** | **Inter-language structural edges** (TS imports Python via FFI/HTTP/proto) | P3 | exploratory; needs redpanda |

---

## 10. Open questions — resolved

| # | question                                                                                                                       | answer (user)                                                                          | implication                                                                                                                         |
| - | ------------------------------------------------------------------------------------------------------------------------------ | -------------------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------- |
| 1 | Does the planner weight class transitive connectivity?                                                                         | **No** — no transitive container weights anywhere today.                               | P1 (member-promotion edges) has bigger upside than the §3 estimate. Every page-eligibility decision today ignores member relevance. |
| 2 | Is `text_chunk` ever architectural? Three-tier doc model OK?                                                                   | **Adopt three-tier**, but generalize beyond markdown. `text_chunk` always demoted.     | §8 needs a generalization for YAML / JSON / proto / large config. See §11.2.                                                        |
| 3 | Adopt graphify's `EXTRACTED` / `INFERRED` / `AMBIGUOUS` confidence labels (B1)?                                                | **Defer to author's judgment** — recommendation is yes (foundation for P1/P3/P8/LLM).  | Adopt B1 now as part of schema work in §11.6 — required by P1 to cleanly distinguish synthetic edges from parsed ones.              |
| 4 | Cheapest way to get a redpanda index?                                                                                          | **Already running** on current pre-fix code (in progress).                             | Defer P9 (inter-language edges) and §6.7 (redpanda audit) until index lands. Everything else can proceed.                           |
| 5 | C++ parser member-edge gap — real gap, or audit artifact?                                                                      | **Treat as a real gap; investigate and fix.**                                          | Started in §11.1. Code audit shows the parser DOES emit `REFERENCES`/`CALLS`, but symbol-tagging hygiene is the bigger gap.         |

## 10b. Resolved ambiguities (A1 / A2 / A3)

- **A1 — universal two-tier doc model (simplified per author).** Forget per-format navigable tiers. Universal rule:
    - **Anchor** = 1 per file. `symbol_type` is `<kind>_document`: `markdown_document`, `yaml_document`, `json_document`, `proto_document`, `text_document`, `cmake_document`, `makefile_document`, `bazel_document`, `dockerfile_document`, … always architectural.
    - **Chunks** = `text_chunk` (always non-architectural). Emitted only when the document body exceeds the chunk-size threshold; small files stay as a single chunk that is the body of the anchor.
    - **Markdown is the only special case** — additionally emit `section` nodes (H1/H2 only) under the anchor; chunks attach to the nearest section. Everything else (Makefile, Vagrantfile, CMakeLists.txt, BUILD, .txt, k8s YAML, OpenAPI JSON, proto, …) is anchor → chunks. No structural parsing required for non-markdown.
    - **Edges**: `<doc_anchor> — defines → <section>` (markdown only); `<section|doc_anchor> — contains → <text_chunk>`; `<text_chunk> — chunk_of → <section|doc_anchor>` (reverse for retrieval back-walk).
    - **Implication for §11.2**: now trivially implementable. No A1 blocker remaining.
- **A2 — sum edge weights (option a).** For class `C` with 50 methods all referencing external type `T`, emit a single `member_uses(C → T, weight=50)` edge. Reason: `repo_edges` uses `(source_id, target_id, rel_type)` as the natural key, parallel writes already collapse, and "evidence count" matches Leiden modularity intuition. Hub-damping (§11.3) handles the god-class edge-cases separately. **A2 unblocked.**
- **A3 — deterministic confidence labels (no LLM); deferred.** Recommendation when we eventually add B1: pure provenance enum on `repo_edges.confidence` — `EXTRACTED` (parser saw it directly), `INFERRED` (we synthesized it from rules — member-promotion, registry hyperedges, regex rationale), `AMBIGUOUS` (parser saw a CALL but couldn't uniquely resolve target). No LLM required; deterministic in our pipeline. **Deferred** until §11.4 (member edges) is ready to land — no value adding the column with only `EXTRACTED` rows. Removing from §11.1 critical path.

---

## 11. Implementation specifications

> Designed so each section is independently shippable. C++ parser hygiene (§11.1) does not depend on any other section and is the **first** work item per the user's direction.

### 11.1. C++ parser hygiene — `cpp_enhanced_parser.py` (REVISED, scope reduced)

**Goal**: stop emitting non-user-defined symbols and tree-sitter parse-recovery garbage. **Do NOT touch the function/declaration distinction** — header-decl + .cpp-impl pairing is a clustering concern and the existing C++/Rust/Go augmentation already handles it. **Do NOT introduce new architectural symbol types.** Only filter / re-tag.

**Concrete deliverables (in cpp_enhanced_parser.py + tests):**

1. **Drop header-guard macros.** In `visit_preproc_def` (lines 1538 / 2915), if `symbol_type='macro'`, `name` matches `^[A-Z][A-Z0-9_]*_(H|HPP|HXX|H_)$`, and the body is empty (`#define <NAME>` with no replacement) — do not emit. Reference: 772 macro orphans in fmt; samples `FMT_ARGS_H_`, `FMT_COMPILE_H_`, `FMT_OSTREAM_H_`, etc. all match.

2. **Drop stdlib trait-shim type aliases.** In `visit_alias_declaration` (line 490), skip the symbol if `symbol_type='type_alias'` AND `name` is in this curated allow-list (list-driven, reviewable, extensible):

    ```
    decay_t, nullptr_t, void_t, void_t_impl, enable_if_t, conditional_t,
    remove_cv_t, remove_reference_t, remove_cvref_t, remove_const_t, remove_volatile_t,
    remove_pointer_t, remove_extent_t, remove_all_extents_t,
    is_same_v, is_base_of_v, is_convertible_v, is_void_v, is_null_pointer_v,
    is_integral_v, is_floating_point_v, is_arithmetic_v, is_signed_v, is_unsigned_v,
    is_const_v, is_volatile_v, is_array_v, is_pointer_v, is_reference_v,
    is_lvalue_reference_v, is_rvalue_reference_v, is_function_v,
    add_const_t, add_volatile_t, add_cv_t,
    add_pointer_t, add_lvalue_reference_t, add_rvalue_reference_t,
    common_type_t, underlying_type_t, invoke_result_t,
    type_t, value_t  (only when defined inline at namespace scope as `using foo = typename ...::type`)
    ```

3. **Drop tree-sitter parse-recovery garbage.** Filter symbols whose `name`:
    - Starts with `auto ` (e.g., `auto p` — fmt sample showed this as a "class").
    - Equals `typename...`, `class...`, `template...`, `T`, `U`, `V` literally (template-parameter-pack misread).
    - Anonymous enums (`<anonymous_enum:…>`) are **kept at every scope**. Rationale: the synthetic wrapper binds the enumerators (which are emitted as `CONSTANT` children with `parent_symbol = wrapper.full_name`); dropping the wrapper would orphan the constants. Top-level anonymous enums often hold real public-API constants (`enum { K_VERSION = 1, K_MAX = 100 };`) so demoting them was incorrect — reverted.

4. **Tag *member* `operator…` overloads as `symbol_type='operator'`.** Member operators (e.g. `Vec::operator+`, `operator[]`) are documented as part of their owning class, so they get `symbol_type='operator'` (NOT in `ARCHITECTURAL_SYMBOLS`) but stay in the graph for CALLS / REFERENCES resolution. **Free-standing operators stay `FUNCTION`** (architectural) — `std::ostream& operator<<(std::ostream&, const MyClass&)` is real public API and must drive its own page. Detection: demote when the parser already classified the symbol as `METHOD` (i.e., inside a `class`/`struct` scope or qualified with a class name); leave `FUNCTION` symbols untouched. Conversion-operator constructors (`operator int()`) keep their `CONSTRUCTOR` tag.

5. **Force `is_test=1` on test-tree symbols.** Any symbol whose `rel_path` matches one of `test/`, `tests/`, `test/gtest/`, `test/gmock/` (prefix match, case-sensitive) MUST be tagged `is_test=1`. Combined with the existing `exclude_tests` option, this drops `GTEST_API_`, `ContainerPrinter`, `WithArgsAction`, etc. from generation surface. Reference: fmt is ~65% test code; this is essential to make the orphan metric meaningful.

**Explicitly out of scope for §11.1** (per author):

- Standalone `function` symbols stay architectural. No `function_decl` split.
- Header-decl ↔ .cpp-impl pairing → clustering concern, not parser. Existing micro_cluster + augmentation logic handles it.
- One-header-two-impls disambiguation → keep both, augmentation pulls both. Documented as expected behavior, not a bug.
- `_impl` suffix demotion → over-eager; many user APIs end in `_impl` (e.g., `format_to_impl`).

**Validation criteria for §11.1 (post-fix audit on fmt, with `exclude_tests=True`):**

| metric                                        | before |     after target |
| --------------------------------------------- | -----: | ---------------: |
| arch nodes (total)                            | 5 136  |          ≤ 3 500 |
| orphan rate (arch)                            |  58.6% |            ≤ 35% |
| `macro` orphans (header guards)               |    772 |              ≤ 5 |
| `type_alias` orphans (stdlib aliases)         |    118 |             ≤ 30 |
| `class` orphans matching parse-recovery (`auto p`, `typename...`) | ~5–15 | = 0 |
| pages generated for a known class (e.g. `format_to_n_args`) | unknown | ≥ 1 (smoke) |

> Note: targets relaxed slightly versus the original draft because (a) free operators stay architectural and (b) anonymous enums stay architectural. The dominant noise sources (header guards, stdlib aliases, parse-recovery classes, member operators) are still filtered.

**Tests** (committed):

- `backend/tests/unit/parsers/__init__.py`
- `backend/tests/unit/parsers/test_cpp_parser_hygiene.py` — 47 cases: pure-helper parametrizations + 5 fixture-based tests (one per filter) using inline C++ snippets parsed via `CppEnhancedParser.parse_file`. **All passing.**

**Validation experiment** (NOT a committed test — local-only, throwaway):

- Re-parse the fmt repo end-to-end via the local `docker compose up` stack and re-run `python /tmp/connectivity_audit.py fmt` against the resulting wiki DB. The audit script verifies the metrics in the table above. We do not commit fmt fixtures or integration tests against persisted DBs because the user's local DBs are recreated freely; long-lived integration tests against pinned fixtures would couple this work to fmt's release cadence without commensurate value.

- Existing `tests/unit/test_graph_clustering.py` and `test_cluster_expansion.py` MUST keep passing unchanged — these fixes touch parser only, never clustering. **Verified: 605 unit tests pass.**

### 11.2. Universal two-tier doc model — all non-code documents (UNBLOCKED per A1)

**Goal**: replace per-format structural-parsing fantasies with a trivial universal rule. `text_chunk` is **always** non-architectural and **always** carries a parent edge.

**Rule (per A1 resolution):**

| kind                           | anchor (architectural)         | sub-structure                   | retrievable                                                |
| ------------------------------ | ------------------------------ | ------------------------------- | ---------------------------------------------------------- |
| `.md` / `.markdown` / `.mdx`   | `markdown_document`            | `section` (H1/H2 only)          | `text_chunk` attached to nearest `section` (or doc anchor) |
| `.yaml` / `.yml`               | `yaml_document`                | (none)                          | `text_chunk` attached to doc anchor                        |
| `.json`                        | `json_document`                | (none)                          | `text_chunk` attached to doc anchor                        |
| `.proto`                       | `proto_document`               | (none)                          | `text_chunk` attached to doc anchor                        |
| `.cmake` / `CMakeLists.txt`    | `cmake_document`               | (none)                          | `text_chunk` attached to doc anchor                        |
| `Makefile` / `*.mk`            | `makefile_document`            | (none)                          | `text_chunk` attached to doc anchor                        |
| `BUILD` / `BUILD.bazel` / `WORKSPACE` | `bazel_document`        | (none)                          | `text_chunk` attached to doc anchor                        |
| `Dockerfile` / `*.dockerfile`  | `dockerfile_document`          | (none)                          | `text_chunk` attached to doc anchor                        |
| `Vagrantfile`                  | `vagrant_document`             | (none)                          | `text_chunk` attached to doc anchor                        |
| `.txt` / unknown text          | `text_document`                | (none)                          | `text_chunk` attached to doc anchor                        |

**Edge invariants (always emitted, regardless of kind):**

- `<doc_anchor> — defines → <section>` (markdown only)
- `<section | doc_anchor> — contains → <text_chunk>`
- `<text_chunk> — chunk_of → <section | doc_anchor>` (reverse, for retrieval back-walk)

**Architectural / page-eligibility:**

- All `*_document` types are added to `ARCHITECTURAL_SYMBOLS` (already includes `markdown_document` via `DOC_SYMBOL_TYPES` — extend the set).
- `text_chunk` is **never** in `ARCHITECTURAL_SYMBOLS`. It is **always** carried in the graph for retrieval.
- `section` is added to `DOC_SYMBOL_TYPES` (markdown-only; other doc kinds don't emit it).

**Chunking threshold** (per author): if document body fits under threshold (currently the parser-side default), the body is the single chunk and effectively *is* the anchor body. Above threshold, split into chunks; each chunk becomes a `text_chunk` node.

**Implementation surface:**

- `backend/app/core/parsers/markdown_parser.py` — already does sections; add `chunk_of` reverse edge if missing.
- `backend/app/core/parsers/text_parser.py` (NEW or extend existing generic) — single entry point that maps file extension → `<kind>_document` anchor, then chunks.
- `backend/app/core/parsers/__init__.py` — file-extension → parser dispatch updated to route everything not handled by a code parser through the universal text parser.
- `backend/app/core/constants.py` — extend `DOC_SYMBOL_TYPES` with the new anchor types listed above; ensure `text_chunk` is NOT in `ARCHITECTURAL_SYMBOLS`.

**Tests to add:**

- `backend/tests/unit/parsers/test_universal_doc_model.py` — one fixture per kind (yaml, makefile, dockerfile, txt), assert (a) anchor exists with correct `symbol_type`, (b) chunks exist with `chunk_of` back-edge, (c) `is_architectural` true for anchor, false for chunks.
- `backend/tests/unit/parsers/test_markdown_sections.py` — assert H1/H2 produce `section` nodes; H3+ collapse into chunks; chunks attach to nearest section.

**Out of scope (per A1):**

- YAML/JSON/proto structural parsing — explicitly NOT done. No `service` / `message` / `top_level_key` nodes. Keeps the model trivial.
- Schema-aware JSON handling (`$id`, `definitions`) — deferred, can add later as a refinement if a real Q&A use-case demands it.

**Retrievability invariant — `is_architectural` does NOT gate retrieval.**

This is a frequent source of confusion. `is_architectural` is consulted in exactly two places:

1. **Page eligibility** — only architectural symbols can become a wiki page (otherwise every comment chunk would attempt to grow into a page).
2. **Cluster centroid eligibility** — Leiden centroid resolution and "headline symbol" picking ignore non-architectural nodes so that doc chunks don't get nominated as the structural backbone of a section.

That means a `text_chunk` with `is_architectural = 0` is **fully retrievable** by:

- BM25 / FTS5 keyword scoring,
- dense embedding ANN search,
- graph-walk expansion (its `chunk_of` parent edge is followed normally).

So the universal two-tier model never hides content from Q&A; chunks are always reachable through their architectural anchor, and through standalone text search if the user keyword-matches the chunk body directly.

**Page-builder budget rule (mandatory).** Because `chunk_of` edges expand transitively, a wiki page that picks up a long markdown anchor as a related node could blow its context budget on raw doc content. The page builder MUST cap pulled-through doc content per page (the constant currently lives near the page-build pipeline as `MAX_DOC_CHARS_PER_PAGE`); chunks beyond the budget are summarized to "see file: …" pointers rather than inlined. This rule is the only thing that prevents one big README from poisoning every neighbouring page.

### 11.3. Hub damping + recursive Leiden — `graph_clustering.py`

(See §3 / §6.1.) Implementation outline:

- Edge-weight contribution during file-graph contraction:
    `w_contributed = w_raw / (1 + log(deg(file_a) + deg(file_b)))`.
- After first Leiden pass, for any cluster with `|nodes| > MAX_CLUSTER_SIZE` (proposed default 1500 or 25% of graph, whichever smaller), induce its subgraph and re-run Leiden at higher resolution. Cap recursion depth at 3.
- New constant `LEIDEN_HUB_DAMPING = True` and `LEIDEN_MAX_CLUSTER_SIZE = 1500` in `cluster_constants.py` (feature flag + threshold).

Tests:

- New: `tests/unit/test_hub_damping.py` — fixture graph with a deliberate hub; assert hub's own cluster size shrinks and the dependent communities split.
- Existing `test_file_contraction.py` and `test_hierarchical_leiden.py` MUST be updated to pass the new flag.

### 11.4. Class member-promotion edges (P1) — **SHIPPED**

Deterministic post-parse pass on `WikiStorageProtocol.promote_class_members()` (sqlite + postgres impls). For every architectural class / struct / interface / trait `C`:

1. Collects member ids = targets of `(C, defines, *)` where the target's `symbol_type ∈ {'method','field','property','constructor'}`.
2. For each member `m`, finds every `(m, ref_type, T)` where `ref_type ∈ {'references','calls','creates','instantiates'}` AND `file_of(T) != file_of(C)`.
3. Counts occurrences per `T` (the "evidence count"). Emits a single synthetic edge `(C, member_uses, T, weight=<count>, edge_class='structural', confidence='INFERRED', created_by='augmentation:11.4')` per A2's sum strategy.
4. Skips emission entirely if `(C, *, T)` already exists with a non-`defines` `rel_type` (don't double-count what the parser already attributed to `C` itself).
5. Self-loops (`T == C`) excluded.

Called from `filesystem_indexer._write_unified_db` after vendored demotion (so bundled gtest classes don't pollute the augmented graph) and before god-nodes computation (so the report reflects final connectivity).

Implementation:
- `backend/app/core/storage/protocol.py` — abstract method.
- `backend/app/core/unified_db.py` — sqlite impl using one SELECT + `upsert_edges_batch`.
- `backend/app/core/storage/postgres.py` — postgres impl using `INSERT ... SELECT`.
- `backend/app/core/filesystem_indexer.py` — caller wired in post-pass chain.
- Tests: `backend/tests/unit/test_unified_db_post_passes.py::TestPromoteClassMembers` (8 cases: basic, evidence aggregation, same-file ignored, self-loop excluded, skip-on-existing-non-defines, defines-doesn't-block, only-class-like-sources, field members count).

Validated on fmt DB after vendored demotion: 189 fmt-internal `member_uses` edges synthesized; top sources (`scan_handler`, `basic_fstring`) are real fmt API surfaces.

### 11.5. Toolkit-registry hyperedge (P3) — `repo_node_groups`

(Specifies storage but does not implement; depends on §11.6 schema landing first.)

- New table `repo_node_groups(group_id PK, group_kind TEXT, label TEXT)` and `repo_node_group_members(group_id, node_id, role TEXT, PRIMARY KEY (group_id, node_id))`.
- Indexer pass: scan `tools/*/__init__.py`; if a function is named `get_tools` / `get_toolkit` / `get_available_tools`, add it to a single `toolkit_registry` group with `role='entrypoint'`.
- Planner consumes group as one virtual hyperedge for cluster-relevance scoring. **No** pairwise edges synthesized — keeps the structural graph honest.

### 11.6. Edge-confidence schema (B1) — **SHIPPED**

Pure provenance enum on `repo_edges.confidence`, no LLM:

- `'EXTRACTED'` — parser-derived AST edge (default).
- `'INFERRED'` — synthetic post-pass edge (§11.4 `member_uses`, future §11.5 / §11.10).
- `'AMBIGUOUS'` — reserved for call-resolver picking one of multiple candidates.

Fresh-install DDL (sqlite + postgres) includes the column with default + CHECK constraint. Migration on legacy DBs:
- **sqlite**: `ALTER TABLE repo_edges ADD COLUMN confidence TEXT NOT NULL DEFAULT 'EXTRACTED'` (CHECK omitted — sqlite cannot add CHECK via ALTER; invariant enforced application-side via `e.get('confidence', 'EXTRACTED')` in `_upsert_edges_batch`).
- **postgres**: idempotent `DO $$ BEGIN IF NOT EXISTS ... THEN ALTER TABLE ... ADD COLUMN ...; ALTER TABLE ... ADD CONSTRAINT repo_edges_confidence_chk CHECK (...); END IF; END $$` block in `_create_indexes()`.
- Index `idx_edges_confidence` created on both backends.

Implementation:
- `backend/app/core/unified_db.py` — schema, migration, upsert.
- `backend/app/core/storage/postgres.py` — schema, migration, upsert.
- Tests: `backend/tests/unit/test_unified_db_post_passes.py::TestEdgeConfidenceColumn` (column exists, default is `'EXTRACTED'`, `confidence='INFERRED'` kwarg propagates, index exists).

### 11.7. "God nodes" report metric (B2) — `wiki_meta.god_nodes` — **SHIPPED**

Protocol method `compute_god_nodes(top_n=20)` on both `UnifiedWikiDB` and `PostgresWikiStorage`; called from `filesystem_indexer._write_unified_db` after Phase 3 clustering and stored under `wiki_meta.god_nodes`:

```json
{
  "by_symbol_type": [{"symbol_id": "...", "name": "...", "degree": 580, "rel_path": "..."}, ...],
  "by_file": [{"rel_path": "runtime/clients/artifact.py", "internal_arch": 12, "external_edges": 2541}, ...]
}
```

Top 20 per axis. Pure analytics — no behavior change. Tested in `backend/tests/unit/test_unified_db_post_passes.py::TestComputeGodNodes`.

### 11.8. Path/scope-based local-leakage detection (P4) — **SHIPPED**

Protocol method `demote_local_constants()` (sqlite + postgres). Called from `filesystem_indexer._write_unified_db` after Phase 3. Single SQL pass: any `symbol_type='constant'` whose every incoming reference originates from the same `rel_path` as the constant gets `is_architectural=0`.  Critically, constants with **zero** incoming references are kept architectural (could be unused public API). Tested in `TestDemoteLocalConstants`.

The scope-stack heuristic from the original spec proposal was rejected as too brittle once we realised the SQL post-pass operates on actually-observed reference edges, which is the ground truth.

### 11.9. Persisted page list (P5) and case-insensitive symbol lookup (P6)

- **P6 — SHIPPED.** `find_nodes_by_name` in both `UnifiedWikiDB` and `PostgresWikiStorage` now does `LOWER(symbol_name) = LOWER(?)`, fixing `EliteAClient`/`EliteaClient`-class lookups. Tested in `TestFindNodesByNameCaseInsensitive`.
- **P5 — DEFERRED.** `wiki_meta.pages` JSON: `[{"id", "title", "primary_symbol_id", "primary_symbol_name", "cluster", "kind"}]` would be written by the planner. Trivial in shape but requires careful integration into the existing planner pipeline; punted to the next pass once we have re-parse evidence on what page identity should encode.

### 11.10. Rationale edges (B4) and inter-language edges (P9)

Deferred. B4 is cheap (regex `^\s*#\s*(WHY|NOTE|HACK|IMPORTANT|TODO):` per language, attach as a `rationale_for` edge from synthetic `rationale` node to the nearest enclosing symbol). P9 explicitly waits for redpanda data per A4.

### 11.12. Vendored / third-party demotion (P7) — **SHIPPED**

**Origin:** the first post-`§11.7` god-nodes report on the fmt rebuild revealed that 1752 / 4963 (35%) architectural nodes lived under `test/gtest/` (bundled GoogleTest). Top god-node by degree was `EXPECT_EQ` (2249), top god-file was `test/gtest/gmock-gtest-all.cc` (2671 external edges). This is the exact problem `is_test` was supposed to catch but doesn't, because the bundled test framework lives inside `test/` and the existing test heuristic looks at filenames not directory provenance.

**Rule:** protocol method `demote_vendored_code()` (sqlite + postgres). Path-based and conservative — `is_architectural` is set to 0 for any node whose `rel_path` contains a vendored / bundled segment:

- top-level: `third_party/`, `third-party/`, `vendor/`, `vendored/`, `external/`, `extern/`, `deps/`
- nested: `*/third_party/*`, `*/third-party/*`, `*/vendor/*`, `*/vendored/*`, `*/external/*`
- bundled test frameworks: `*/gtest/*`, `*/gmock/*`, `*/googletest/*`, `*/googlemock/*`

**Order matters.** Vendored demotion runs FIRST in the post-pass chain (before P4 local-constants and before B2 god-nodes computation) so the local-constant pass and god-nodes report both reflect the post-demoted state. Page eligibility on the next index of fmt should drop EXPECT_EQ-shaped noise out of consideration entirely.

Demoted nodes remain fully retrievable (per §11.2 invariant) — only page eligibility and god-node ranking are affected.

Tested in `backend/tests/unit/test_unified_db_post_passes.py::TestDemoteVendoredCode`.

### 11.13. Sequencing / dependency graph

```
§11.1 (C++ parser)       — INDEPENDENT — STARTING NOW
§11.2 (universal docs)   — INDEPENDENT — unblocked by A1
§11.4 (member edges)     — **SHIPPED** — paired with §11.6 confidence column
§11.7 (B2 god-nodes)     — INDEPENDENT
§11.8 (P4 leakage)       — INDEPENDENT
§11.9 (P5/P6)            — INDEPENDENT

§11.3 (hub damping)      — INDEPENDENT but high-risk; ship after §11.1 + §11.2
§11.5 (registry groups)  — needs §11.6
§11.6 (B1 schema)        — **SHIPPED** — confidence column on both sqlite + postgres backends
§11.10 (rationale + P9)  — deferred (B4 trivial; P9 needs redpanda)
```

**Recommended landing order** (smallest blast radius first):

1. §11.1 (C++ parser hygiene) — starting now, no downstream impact
2. §11.7 (god-nodes report) — observability only
3. §11.9 (page list + case-insensitive lookup) — small, independent
4. §11.8 (P4 local-leakage) — Python-side hygiene
5. §11.2 (universal docs) — schema-additive, no breaking changes
6. §11.4 + §11.6 (member edges + confidence) — **SHIPPED** (paired); first synthetic edges live
7. §11.3 (hub damping + recursive Leiden) — high-impact clustering change, needs careful evaluation
8. §11.5 (registry groups) — depends on §11.6
9. §11.10 (rationale + P9) — deferred
