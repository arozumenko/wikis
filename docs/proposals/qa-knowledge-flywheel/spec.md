# Feature Specification: Q&A Knowledge Flywheel

**Feature Branch**: `feature/qa-knowledge-flywheel`
**Created**: 2026-03-20
**Status**: Draft
**Input**: Turn stateless Q&A into a knowledge flywheel — capture interactions, cache semantically, validate answers, enrich the wiki.
**Proposal**: `tool/docs/proposals/qa-knowledge-flywheel/proposal.md`

## User Scenarios & Testing

### User Story 1 — Repeat Question Cache Hit (Priority: P1)

A user asks a question that was already answered for this wiki. The system returns the cached answer instantly without running the full LangGraph agent pipeline.

**Why this priority**: This is the core value proposition. Every cached response saves 5–30 seconds of agent execution, multiple LLM calls, and token costs. Based on industry data (GPTCache achieves 68.8% API call reduction), even a modest cache hit rate delivers significant value.

**Independent Test**: Ask a question on a wiki, wait for the `BackgroundTask` to complete (so the DB row and QA FAISS index are updated), then ask the same question again. Verify the second response is returned in under 500ms with the same answer content.

**Acceptance Scenarios**:

1. **Given** a wiki with no prior Q&A, **When** a user asks "How does authentication work?", **Then** the system runs the full agent pipeline, returns an answer with sources, and persists a `QARecord` in the database with `status='pending'`.

2. **Given** a wiki where "How does authentication work?" was previously answered, **When** a user asks the exact same question, **Then** the system returns the cached answer in under 500ms without invoking the LangGraph agent.

3. **Given** a cached Q&A interaction, **When** the cached response is returned via the sync path, **Then** it is an `AskResponse` with the same fields as a fresh answer. **When** returned via the SSE path, **Then** the system emits `answer_chunk` + `task_complete` events (no `thinking_step` events, since no agent ran). In both cases, no cache-hit indicator is shown.

---

### User Story 2 — Q&A Persistence (Priority: P1)

Every Q&A interaction is persisted to the database with the question, answer, sources, metadata, and a unique ID. Answers survive page refreshes and browser restarts.

**Why this priority**: Persistence is the foundation for every other feature (cache, history, validation, enrichment). Without it, the flywheel has no memory.

**Independent Test**: Ask a question, wait for the `BackgroundTask` to complete, then verify a `QARecord` exists in the database via the list API. Refresh the page. Call `GET /api/v1/wikis/{wiki_id}/qa` and confirm the interaction is returned.

**Acceptance Scenarios**:

1. **Given** any wiki, **When** a user asks a question and the agent produces an answer, **Then** a `QARecord` is created with all fields populated: `wiki_id`, `question`, `question_hash`, `answer`, `sources_json`, `tool_steps`, `mode`, `user_id`, `created_at`.

2. **Given** an active SSE stream delivering an answer, **When** the response stream completes, **Then** a `BackgroundTask` persists the `QARecord` to the database without delaying the stream.

3. **Given** a completed Q&A interaction, **When** the API returns a response, **Then** the response includes a pre-generated `qa_id` (UUID): in `AskResponse` for sync clients, and in the `task_complete` event data for SSE clients. The `QARecord` row is persisted asynchronously via `BackgroundTask` using this same ID.

---

### User Story 3 — Semantic Similarity Cache (Priority: P1)

A user asks a question that is semantically similar (but not identical) to a previously answered question. The system recognizes the similarity and returns the cached answer.

**Why this priority**: Exact-match caching alone misses the majority of repeat intent. Semantic similarity captures paraphrased questions ("How does auth work?" vs "How does authentication work?"), which is where the real cache hit rate comes from. Industry consensus places the optimal threshold at 0.90–0.95 for Q&A; our 0.92 is well-calibrated.

**Independent Test**: Ask "How does authentication work?", wait for the `BackgroundTask` to complete (so the QA FAISS index is updated), then ask "How does auth work?". Verify the second question returns the cached answer. Then ask "How does auth work in the middleware?" and verify it does NOT cache-hit (too specific).

**Acceptance Scenarios**:

1. **Given** a cached answer for "How does authentication work?", **When** a user asks "How does auth work?", **Then** the system finds a semantic match (cosine >= 0.92) in the QA FAISS index and returns the cached answer.

2. **Given** a cached answer for "How does authentication work?", **When** a user asks "How does auth work in the middleware?", **Then** the semantic similarity is below 0.92 and the system runs the full agent pipeline (different scope = different answer).

3. **Given** the two-layer cache design, **When** a question arrives, **Then** the system checks SHA256 hash first (O(1) exact match), and only falls back to FAISS cosine similarity if the hash misses.

---

### User Story 4 — Answer Validation (Priority: P2)

A user can upvote or downvote an answer. The validation status is persisted and available for filtering and future enrichment.

**Why this priority**: Validation is the quality gate for the enrichment phase. Without human feedback, the system cannot distinguish good answers from bad ones. Binary thumbs with specific copy ("Did this answer your question?") achieves 5x higher feedback rate than generic ratings.

**Independent Test**: Ask a question, receive a `qa_id` in the response, wait briefly for the `BackgroundTask` to complete (sub-second), call `POST /api/v1/wikis/{wiki_id}/qa/{qa_id}/validate` with `{"action": "upvote"}`, verify the record's status changes to `validated` and `validation_type` to `user_upvote`. In practice, the user reads the answer before voting — the `BackgroundTask` completes well before any human interaction.

**Acceptance Scenarios**:

1. **Given** a Q&A interaction with `qa_id` whose `BackgroundTask` has completed, **When** a user sends a validate request with `action: "upvote"`, **Then** the record's `status` changes to `validated`, `validation_type` to `user_upvote`, and `validated_at` is set.

2. **Given** a Q&A interaction with `qa_id` whose `BackgroundTask` has completed, **When** a user sends a validate request with `action: "downvote"`, **Then** the record's `status` changes to `rejected`, `validation_type` to `user_downvote`, and `validated_at` is set.

3. **Given** a previously validated record, **When** a user sends a new validate request, **Then** the previous validation is overwritten (last vote wins).

---

### User Story 5 — Q&A History Browsing (Priority: P2)

A user can browse past Q&A interactions for a wiki, filtered by status, with pagination.

**Why this priority**: History browsing turns ephemeral Q&A into accumulated knowledge. It also provides the administrative view needed for bulk validation in the enrichment phase.

**Independent Test**: Ask 5+ questions on a wiki, waiting for each `BackgroundTask` to complete before the next assertion. Call `GET /api/v1/wikis/{wiki_id}/qa?limit=3&offset=0`. Verify 3 records returned with correct total count. Call with `?status=validated` after upvoting one, verify only 1 returned.

**Acceptance Scenarios**:

1. **Given** a wiki with 10 Q&A interactions, **When** a user calls `GET /api/v1/wikis/{wiki_id}/qa?limit=5&offset=0`, **Then** the response contains 5 records sorted by `created_at` descending, plus total count metadata.

2. **Given** Q&A interactions with mixed statuses, **When** a user calls with `?status=validated`, **Then** only validated records are returned.

3. **Given** a wiki with Q&A stats, **When** a user calls `GET /api/v1/wikis/{wiki_id}/qa/stats`, **Then** the response includes `total_count`, `cache_hit_count`, `validated_count`, `rejected_count`, and `hit_rate` (cache_hit_count / total_count).

---

### User Story 6 — Knowledge Enrichment (Priority: P3)

An admin triggers enrichment, which adds validated Q&A into the main wiki FAISS index so future questions benefit from prior answers.

**Why this priority**: This closes the flywheel loop — validated knowledge feeds back into retrieval, improving future answers. No production tool does direct Q&A re-ingestion into the vector store; this is a differentiator. Gated behind validation to prevent bad answers from polluting the index.

**Independent Test**: Validate a Q&A, trigger enrichment via API, then ask a related question. Verify the enriched content appears in the agent's retrieved context.

**Acceptance Scenarios**:

1. **Given** validated Q&A records for a wiki, **When** an admin calls the enrichment endpoint, **Then** each validated record is added to the main wiki FAISS index as a Document with `chunk_type: "qa"` metadata, and the record's status changes to `enriched`.

2. **Given** enriched Q&A in the main FAISS index, **When** a user asks a question similar to an enriched Q&A, **Then** the agent's retrieval includes the enriched Q&A document in its context.

3. **Given** the enrichment process, **When** enrichment completes, **Then** the `ComponentCache` is evicted so the next ask loads the updated index.

---

### User Story 7 — FAQ Page Generation (Priority: P3)

An admin triggers FAQ page generation from accumulated validated Q&A. The system clusters related questions and generates wiki pages that appear in the sidebar.

**Why this priority**: This is the highest-visibility output of the flywheel — generated documentation from real user questions. No production tool auto-generates FAQ from Q&A; Mintlify's Agent Suggestions is the closest analog. This is a genuine differentiator.

**Independent Test**: Accumulate 10+ validated Q&A, trigger FAQ generation via API. Verify generated pages appear under `wiki_pages/faq/` and are discoverable via the existing wiki pages API.

**Acceptance Scenarios**:

1. **Given** 10+ validated Q&A records for a wiki, **When** an admin calls `POST /api/v1/wikis/{wiki_id}/qa/generate-pages`, **Then** the system clusters Q&A by topic, generates markdown pages, and saves them as `wiki_pages/faq/{topic}.md`.

2. **Given** generated FAQ pages, **When** the frontend fetches the wiki page list, **Then** FAQ pages appear in the sidebar under a "FAQ" section (automatic — existing API discovers all `.md` files under `wiki_pages/`).

---

### Edge Cases

- **Wiki deleted while QA cache exists**: Cascade delete all `QARecord` rows for the wiki and delete the `.qa.faiss` file. Follows existing `ComponentCache.evict()` pattern.
- **Concurrent identical questions during cold start**: Both requests get a cache miss and run the agent. Both record to DB as separate rows (no uniqueness constraint on `question_hash` — the same question asked twice is two interactions). The cache will serve whichever was indexed first for subsequent lookups. This is acceptable: no data loss, no errors, minor duplication at worst.
- **QA FAISS file corrupted or missing**: Rebuild from DB records on next ask. Log a warning. If no DB records exist, start fresh.
- **Empty QA index (first question ever)**: Skip semantic search entirely — go straight to the agent. Create the QA FAISS index when the first answer is recorded.
- **Question in a different language than wiki content**: Cache works on question similarity, not content language. A French question won't match an English one (correct behavior — different answers expected).
- **Wiki regenerated from new commit**: Mark all `status='pending'` records as `stale`, delete the `.qa.faiss` file. Validated/enriched records are preserved (human-confirmed knowledge survives regen).
- **Cache TTL expiry**: Records older than `qa_cache_max_age_seconds` (default 24hr) are skipped during cache lookup even if similarity matches. Combines event-based invalidation (wiki regen) with TTL safety net.
- **Hot-swap QA FAISS index**: FAISS has no built-in hot-swap. Use build-then-rename pattern: write new index to a temp path, then atomic `os.rename()` to the final path.

## Requirements

### Functional Requirements

- **FR-001**: System MUST persist every Q&A interaction — including cache hits — as a separate `QARecord` row. Fields: `id` (UUID), `wiki_id`, `question`, `question_hash` (SHA256), `answer`, `sources_json` (serialized `list[SourceReference]`), `tool_steps`, `mode` (`fast` | `deep`), `status` (`pending` | `validated` | `rejected` | `enriched` | `stale`), `validation_type` (`null` | `user_upvote` | `user_downvote`), `is_cache_hit` (bool, default `False`), `source_qa_id` (str | None — UUID of the original QARecord that supplied the cached answer, null for fresh answers), `user_id`, `created_at`, `validated_at`. Cache-hit rows copy the `answer` and `sources_json` from the source record and set `is_cache_hit=True` + `source_qa_id` to the original's ID. This ensures: (a) every interaction appears in history, (b) `cache_hit_count` is a simple `COUNT WHERE is_cache_hit=True`, (c) each interaction has its own `qa_id` for validation. **Graceful degradation**: if the DB INSERT fails (e.g., database unavailable), the system MUST still deliver the answer to the user. The pre-generated `qa_id` already included in the response becomes a **dangling reference** — the `QARecord` row was never created, so subsequent calls to FR-009 (validate) will return HTTP 404. The system MUST log the persistence failure at ERROR level. The interaction is lost from history — this is acceptable as a degraded-mode behavior, not a normal operating condition.

- **FR-002**: System MUST implement two-layer cache lookup: (1) SHA256 hash exact match against `question_hash` column (O(1)), then (2) FAISS cosine similarity search against the per-wiki QA index (threshold configurable, default 0.92). Industry consensus places optimal Q&A cache threshold at 0.90–0.95.

- **FR-003**: System MUST maintain a separate per-wiki FAISS index for Q&A embeddings, stored as `{cache_key}.qa.faiss`. MUST use `IndexFlatIP` (no training required, correct for <10K scale). All question embeddings MUST be L2-normalized before insertion and search, because `IndexFlatIP` computes inner product — which equals cosine similarity only on unit vectors. This index is independent of the main code/docs FAISS index.

- **FR-004**: Cache lookup MUST happen before agent invocation in `ask_service.py`. QA recording uses `BackgroundTask` (per task.md constraint) to avoid delaying response delivery:
  - **Sync path** (`AskResponse`): Return `AskResponse` immediately with `qa_id` (pre-generated UUID). Persist the `QARecord` via `BackgroundTask` using that ID.
  - **SSE path** (`StreamingResponse`): Yield all events including `task_complete` (with pre-generated `qa_id`). Persist the `QARecord` via `BackgroundTask`. Note: FastAPI runs `BackgroundTask` on `StreamingResponse` after the generator is exhausted. If the client disconnects before the generator completes, the task may not fire — this is an accepted limitation documented for design-phase mitigation (e.g., try/finally in the generator as a fallback).
  - **Cache hits** (both paths): Same pattern — return/emit response with pre-generated `qa_id`, persist via `BackgroundTask`.
  - **`qa_id` validity window**: Between response delivery and `BackgroundTask` completion, `qa_id` refers to a row that does not yet exist. FR-009 (validate) MUST handle this: if the referenced row is not found, return HTTP 404. If the `BackgroundTask` fails entirely (DB unavailable), `qa_id` points to a non-existent row permanently — the graceful degradation in FR-001 applies.
  - **Cache/list visibility window**: The same `BackgroundTask` also controls when a new QA interaction becomes visible to cache lookups (FR-002/FR-003) and list queries (FR-007). Until the `BackgroundTask` completes: (a) the DB row does not exist, so hash-based exact-match lookups miss, (b) the QA FAISS index has not been updated, so semantic similarity lookups miss, (c) the list API does not return the record. In normal usage, the `BackgroundTask` completes before a user formulates their next question. Automated test suites MUST wait for `BackgroundTask` completion before asserting cache hits or list visibility. Build-phase target: `BackgroundTask` completion within 50ms (see SC-003).
  - **Coordinator note**: The SSE client-disconnect limitation is an inherent trade-off of `BackgroundTask` on `StreamingResponse`. If build-phase testing reveals unacceptable recording loss rates, a constraint amendment to allow pre-response INSERT (negligible latency, but technically delays `task_complete` by a single DB write) should be proposed to the coordinator.

- **FR-005**: Cache hits MUST be transparent to the user — no visible indicator distinguishing cached from fresh responses. For the SSE path, cache hits MUST emit the same terminal event sequence as a fresh agent run: `answer_chunk` events (for the cached answer text) followed by a `task_complete` event (with `answer`, `sources`, `steps`). No synthetic `thinking_step` events are emitted — the cache hit is fast enough that the absence of thinking steps is not jarring, and emitting fake agent steps would be misleading.

- **FR-006**: Both response paths MUST include a pre-generated `qa_id` (see FR-004):
  - **Sync path**: `AskResponse` gains a `qa_id: str` field. Set to the pre-generated UUID. If the `BackgroundTask` later fails to persist the row, this UUID becomes a dangling reference (see FR-001 graceful degradation).
  - **SSE path**: The `task_complete` event data payload gains a `qa_id: str` field (alongside existing `answer`, `sources`, `steps`). Same dangling-reference degradation applies.
  - For cache hits, `qa_id` refers to the NEW cache-hit row (not the original source row). The source row's ID is available via `source_qa_id` on the new row.

- **FR-007**: System MUST expose `GET /api/v1/wikis/{wiki_id}/qa` — paginated list of Q&A interactions, filterable by `status`, sortable by `created_at` descending.

- **FR-008**: System MUST expose `GET /api/v1/wikis/{wiki_id}/qa/stats` — returns `total_count`, `cache_hit_count` (count of records where `is_cache_hit=True`), `validated_count`, `rejected_count`, and `hit_rate` (`cache_hit_count / total_count`, 0 if no records).

- **FR-009**: System MUST expose `POST /api/v1/wikis/{wiki_id}/qa/{qa_id}/validate` — accepts `{"action": "upvote" | "downvote"}`, updates status and validation fields.

- **FR-010**: System MUST invalidate QA cache on wiki refresh or delete: delete the `.qa.faiss` file and mark all `status='pending'` records as `stale`. Validated/enriched records are preserved. When the QA FAISS index is rebuilt (e.g., after deletion or corruption), only records with `status` in (`pending`, `validated`, `enriched`) — NOT `stale` or `rejected` — are re-indexed. This means after wiki regeneration the QA cache restarts near-empty (only human-validated knowledge survives), which is the correct behavior since pending answers may reference outdated code.

- **FR-011**: System MUST provide an enrichment endpoint that adds validated Q&A records to the main wiki FAISS index as Documents with `chunk_type: "qa"` metadata, then marks records as `enriched` and evicts the `ComponentCache`.

- **FR-012**: System MUST provide an FAQ generation endpoint that clusters validated Q&A by topic using QA embeddings (clustering algorithm TBD in design phase — candidates: k-means, DBSCAN, hierarchical), generates markdown pages via the existing LLM pipeline, and saves them as `wiki_pages/faq/{topic}.md` (auto-discovered by the existing page listing API).

- **FR-013**: QA recording MUST NOT delay SSE streaming or sync response delivery. Recording runs as a `BackgroundTask` after the response is sent (see FR-004). The user receives their answer with zero added latency from recording.

- **FR-014**: System MUST support configuration via environment variables / Pydantic settings: `qa_cache_enabled` (bool, default `True`), `qa_cache_similarity_threshold` (float, default `0.92`), `qa_cache_max_age_seconds` (int, default `86400`).

### Key Entities

- **QARecord**: Database model (SQLAlchemy, same pattern as `WikiRecord`). Represents a single Q&A interaction — including cache hits. Key fields beyond the obvious: `is_cache_hit` (bool) distinguishes cached from fresh answers; `source_qa_id` (str | None) links cache-hit rows to the original answer. No FK on `wiki_id` (matches existing pattern). Indexed on `wiki_id`, `question_hash`, `status`, and `is_cache_hit`.

- **QACacheManager**: Manages per-wiki QA FAISS index lifecycle — creation, search, addition, deletion, rebuild from DB. Uses `IndexFlatIP` for inner-product similarity on normalized embeddings.

- **QAService**: Orchestrates QA recording, cache lookup (two-layer), listing with pagination/filtering, validation, and stats aggregation. Injected into `AskService` and exposed via routes.

## Success Criteria

### Measurable Outcomes

- **SC-001**: Repeat question (exact or semantic match) response time < 500ms, compared to 5–30s for a full agent run.
- **SC-002**: Cache hit rate > 30% after 50+ questions on the same wiki (conservative target; GPTCache reports 68.8% with a looser 0.8 threshold).
- **SC-003**: QA recording adds zero latency to response delivery — recording runs as a `BackgroundTask` after the response is sent. Build-phase validation target: `BackgroundTask` DB INSERT completes within 50ms (to be benchmarked against both SQLite and PostgreSQL backends).
- **SC-004**: All existing tests pass with no regressions after each phase.
- **SC-005**: Feedback rate > 10% of questions receive a vote (enabled by binary thumbs with specific copy: "Did this answer your question?").

## Three-Phase Delivery

This feature is designed for incremental delivery. Each phase is an independently valuable, testable, and shippable PR:

| Phase | Scope | Key Deliverables |
|-------|-------|-----------------|
| **Phase 1 — Capture + Cache** | FR-001 through FR-008, FR-010, FR-013, FR-014 | `QARecord` model, `QACacheManager`, `QAService`, cache integration in `ask_service.py`, list/stats endpoints, invalidation |
| **Phase 2 — Validate + Browse** | FR-009, frontend components | Validation endpoint, thumbs up/down UI, Q&A history panel, sidebar integration |
| **Phase 3 — Enrich** | FR-011, FR-012 | Vector store enrichment, FAQ page generation, enrichment stats |

## Traceability: Acceptance Criteria Coverage

| AC (from task.md) | Covered by |
|---|---|
| AC-1: QARecord creation | FR-001, Story 2 |
| AC-2: Exact cache hit | FR-002 (layer 1), Story 1 |
| AC-3: Semantic cache hit | FR-002 (layer 2), FR-003, Story 3 |
| AC-4: Paginated QA list | FR-007, Story 5 |
| AC-5: QA stats | FR-008, Story 5 |
| AC-6: Upvote/downvote | FR-009, Story 4 |
| AC-7: Enrichment | FR-011, Story 6 |
| AC-8: FAQ generation | FR-012, Story 7 |
| AC-9: Cache invalidation | FR-010, Edge Cases |
| AC-10: Unit tests | SC-004 (all tests pass), per-phase test coverage |
