# Implementation Plan: Q&A Knowledge Flywheel — Phase 1

**Spec**: `tool/docs/proposals/qa-knowledge-flywheel/spec.md`
**Design**: `tool/docs/proposals/qa-knowledge-flywheel/design.md`
**Created**: 2026-03-24
**Scope**: Phase 1 (Capture + Cache) — FR-001 through FR-008, FR-010, FR-013, FR-014

---

## Implementation Steps

### Step 1: Configuration settings
**Files**: `backend/app/config.py`
**FR coverage**: FR-014
**Dependencies**: none
**Parallelizable with**: Steps 2, 3

Add `qa_cache_enabled`, `qa_cache_similarity_threshold`, `qa_cache_max_age_seconds` to the `Settings` class.

**TDD**:
- Test: verify defaults (True, 0.92, 86400)
- Test: verify env var override (`QA_CACHE_ENABLED=false`)
- Implement: add three fields to Settings

---

### Step 2: QARecord data model
**Files**: `backend/app/models/db_models.py`
**FR coverage**: FR-001
**Dependencies**: none
**Parallelizable with**: Steps 1, 3

Add `QARecord` SQLAlchemy model with all columns per design Section 2. Three composite indexes: `ix_qa_wiki_hash`, `ix_qa_wiki_status`, `ix_qa_wiki_cache`.

**TDD**:
- Test: QARecord instantiation with all fields
- Test: table auto-creation via `Base.metadata.create_all`
- Test: composite indexes exist on created table
- Implement: QARecord class in db_models.py

---

### Step 3: QA API models
**Files**: `backend/app/models/qa_api.py` (new)
**FR coverage**: FR-001, FR-006, FR-007, FR-008
**Dependencies**: none (uses standard lib + pydantic)
**Parallelizable with**: Steps 1, 2

Create `QARecordingPayload` (dataclass), `AskResult` (dataclass), `QARecordResponse`, `QAListResponse`, `QAStatsResponse` (Pydantic).

**TDD**:
- Test: QARecordingPayload creation with all fields
- Test: AskResult creation with response + payload
- Test: QAListResponse serialization (items, total, limit, offset)
- Test: QAStatsResponse serialization with hit_rate calculation
- Implement: qa_api.py with all models

---

### Step 4: QACacheManager
**Files**: `backend/app/services/qa_cache_manager.py` (new)
**FR coverage**: FR-002 (layer 2), FR-003, FR-010 (FAISS delete)
**Dependencies**: none (uses faiss, numpy, LangChain Embeddings interface)
**Parallelizable with**: Steps 1, 2, 3

Implement per-wiki FAISS IndexFlatIP lifecycle: `search(k=5)`, `add`, `delete_index`, `rebuild_from_records`. Includes atomic write pattern, lazy index creation, per-wiki asyncio.Lock, `asyncio.to_thread` for embedding calls.

**TDD**:
- Test: add single embedding, search returns it above threshold
- Test: search returns empty list below threshold
- Test: k=5 returns multiple candidates in descending similarity order
- Test: delete_index removes files and clears in-memory state
- Test: rebuild_from_records creates index from provided records
- Test: lazy index creation — first add determines dimension
- Test: corruption detection — len(qa_ids) != index.ntotal triggers rebuild
- Test: atomic write — tmp files renamed, no partial writes on error
- Test: concurrent add operations don't corrupt index (asyncio.Lock)
- Implement: qa_cache_manager.py

---

### Step 5: QAService
**Files**: `backend/app/services/qa_service.py` (new)
**FR coverage**: FR-001, FR-002 (both layers), FR-007, FR-008, FR-010
**Dependencies**: Steps 1, 2, 3, 4
**Parallelizable with**: none at this point

Implement `lookup_cache` (two-layer: hash exact → FAISS k=5 with candidate iteration), `record_interaction`, `list_qa`, `get_stats`, `invalidate_wiki`. Context bypass, commit-awareness, TTL filtering, graceful degradation.

**TDD**:
- Test: lookup_cache exact hash hit — returns record, no embedding computed
- Test: lookup_cache semantic hit — returns record + reusable embedding
- Test: lookup_cache miss — returns (None, embedding)
- Test: lookup_cache context bypass — non-empty chat_history returns (None, None)
- Test: lookup_cache commit filtering — old commit record excluded
- Test: lookup_cache TTL filtering — expired record excluded
- Test: lookup_cache k=5 iteration — first candidate invalid, second valid, returns second
- Test: lookup_cache disabled — qa_cache_enabled=False returns (None, None)
- Test: record_interaction creates QARecord with all fields
- Test: record_interaction adds embedding to FAISS (non-cache-hit, non-contextual)
- Test: record_interaction skips FAISS for cache hits
- Test: record_interaction skips FAISS for contextual answers (has_context=True)
- Test: record_interaction graceful degradation — DB failure logs error, no exception
- Test: list_qa with pagination and status filter
- Test: get_stats returns correct counts and hit_rate
- Test: invalidate_wiki delete=True removes all rows + FAISS file
- Test: invalidate_wiki delete=False marks pending→stale + deletes FAISS file
- Implement: qa_service.py

---

### Step 6: AskService integration
**Files**: `backend/app/services/ask_service.py`
**FR coverage**: FR-004, FR-006, FR-013
**Dependencies**: Step 5
**Parallelizable with**: Step 7

Modify `AskService.__init__` to accept optional `qa_service`. Modify `ask_sync` to perform cache lookup and return `AskResult`. Modify `ask_stream` to perform cache lookup and record via `try/finally` with success gating.

**TDD**:
- Test: ask_sync cache hit — returns AskResult with cached response + recording payload (is_cache_hit=True)
- Test: ask_sync cache miss — returns AskResult with agent response + recording payload (is_cache_hit=False)
- Test: ask_sync without qa_service — backward compat, returns AskResult with recording=None
- Test: ask_stream cache hit — yields answer_chunk + task_complete events
- Test: ask_stream success gating — completed=True triggers recording
- Test: ask_stream failure — completed=False skips recording
- Test: ask_sync context bypass — non-empty chat_history skips cache
- Implement: modifications to ask_service.py

---

### Step 7: AskResponse model update
**Files**: `backend/app/models/api.py`
**FR coverage**: FR-006
**Dependencies**: none (small, independent change)
**Parallelizable with**: Step 6

Add `qa_id: str` field to `AskResponse`.

**TDD**:
- Test: AskResponse accepts and serializes qa_id field
- Implement: add field to AskResponse

---

### Step 8: App initialization and DI wiring
**Files**: `backend/app/main.py`, `backend/app/dependencies.py`
**FR coverage**: FR-004 (wiring)
**Dependencies**: Steps 4, 5, 6
**Parallelizable with**: Step 7

Initialize QACacheManager and QAService in lifespan with degraded-mode support matching the existing non-blocking `check_providers` pattern in `health_check.py`:

1. If `qa_cache_enabled=False`: skip `create_embeddings()` and `QACacheManager`. Create QAService with `qa_cache=None`.
2. If `qa_cache_enabled=True`: wrap `create_embeddings()` in try/except. On success, create QACacheManager and QAService with it. On failure (unsupported provider, missing credentials), log a warning and create QAService with `qa_cache=None`.
3. QAService is always created and wired into AskService (Step 6's `qa_service: QAService | None = None` constructor). Add `get_qa_service` dependency provider. Update MCP `set_services()` with qa_service parameter.

QAService constructor signature: `cache: QACacheManager | None = None` (per design.md Section 4). When `cache` is None, `lookup_cache` returns `(None, None)` and `record_interaction` skips FAISS indexing (DB recording still works).

**TDD**:
- Test: app startup creates qa_service in app.state (happy path with working embeddings)
- Test: get_qa_service returns the initialized service
- Test: AskService receives qa_service parameter
- Test: app startup with `qa_cache_enabled=False` — qa_service created without cache, app starts
- Test: app startup with embedding creation failure — qa_service created without cache, warning logged, app starts
- Test: existing `test_app` fixture works without seeding embedding credentials (degraded mode, no crash)
- Implement: main.py lifespan + dependencies.py

---

### Step 9: Route integration — /ask recording
**Files**: `backend/app/api/routes.py`
**FR coverage**: FR-004, FR-005, FR-013
**Dependencies**: Steps 6, 7, 8

Modify `/ask` route: sync path uses `BackgroundTask` for recording from `AskResult.recording`. SSE path delegates recording to the generator's `try/finally` (already handled in Step 6's ask_stream changes).

**TDD**:
- Test: sync /ask returns response with qa_id, BackgroundTask scheduled
- Test: SSE /ask streams events including qa_id in task_complete
- Test: sync /ask cache hit returns cached answer with qa_id
- Implement: route handler modifications

---

### Step 10: Route integration — QA endpoints
**Files**: `backend/app/api/routes.py`
**FR coverage**: FR-007, FR-008
**Dependencies**: Steps 5, 8

Add `GET /wikis/{wiki_id}/qa` (paginated list) and `GET /wikis/{wiki_id}/qa/stats` endpoints. Both require wiki authorization via `WikiManagementService.get_wiki`.

**TDD**:
- Test: GET /qa returns paginated QARecords
- Test: GET /qa with status filter returns only matching records
- Test: GET /qa/stats returns correct counts and hit_rate
- Test: GET /qa returns 404 for unauthorized wiki
- Test: GET /qa/stats returns 404 for unauthorized wiki
- Implement: two new route handlers

---

### Step 11: MCP integration
**Files**: `backend/mcp_server/server.py`
**FR coverage**: FR-004 (MCP coverage)
**Dependencies**: Steps 6, 8

Update `ask_codebase` tool to handle `AskResult` from `ask_sync`. Record interaction via direct `await` with error handling.

**TDD**:
- Test: ask_codebase records interaction on success
- Test: ask_codebase handles recording failure gracefully (logs, doesn't crash)
- Implement: server.py modifications

---

### Step 12: Wiki lifecycle integration
**Files**: `backend/app/api/routes.py`
**FR coverage**: FR-010
**Dependencies**: Steps 5, 8

Add `invalidate_wiki(delete=True)` call in `delete_wiki` route. Add `invalidate_wiki(delete=False)` call in `refresh_wiki` route.

**TDD**:
- Test: deleting a wiki removes all QARecords and FAISS file
- Test: refreshing a wiki marks pending→stale and deletes FAISS file
- Test: refreshing a wiki preserves validated/enriched records
- Implement: two route handler additions

---

## Dependency Graph

```
Step 1 (config) ──────────┐
Step 2 (QARecord) ────────┤
Step 3 (qa_api) ──────────┼──→ Step 5 (QAService) ──→ Step 6 (AskService) ──→ Step 8 (init/DI) ──→ Step 9 (/ask) ──→ Step 10 (QA endpoints) ──→ Step 12 (lifecycle)
Step 4 (QACacheManager) ──┘                                                       │
                                                                                  └──→ Step 11 (MCP, parallel with 9-12)
Step 7 (AskResponse) ────────────────────────────────────────────────────────────────→ Step 9
```

**Parallel lanes:**
- Lane A: Steps 1 + 2 + 3 (models + config) — can all start immediately
- Lane B: Step 4 (QACacheManager) — can start immediately, independent of Lane A
- Lane C: Step 7 (AskResponse) — can start immediately, no dependencies
- Critical path: Step 5 → Step 6 → Step 8 → Step 9 → Step 10 → Step 12

After Step 8, Steps 9, 10, and 12 are sequential (all modify `routes.py`). Step 11 (MCP, modifies `server.py`) can run in parallel with any of them. Step 9 also requires Step 7.

## FR Traceability

| FR | Steps |
|----|-------|
| FR-001 | 2, 3, 5 |
| FR-002 | 4, 5 |
| FR-003 | 4 |
| FR-004 | 6, 8, 9, 11 |
| FR-005 | 6, 9 |
| FR-006 | 3, 6, 7 |
| FR-007 | 3, 10 |
| FR-008 | 3, 10 |
| FR-010 | 4, 5, 12 |
| FR-013 | 6, 9 |
| FR-014 | 1 |

## AC Traceability

| AC | Steps |
|----|-------|
| AC-1 (QARecord creation) | 2, 5, 6, 9 |
| AC-2 (Exact cache hit) | 5, 6, 9 |
| AC-3 (Semantic cache hit) | 4, 5, 6, 9 |
| AC-4 (Paginated QA list) | 10 |
| AC-5 (QA stats) | 10 |
| AC-9 (Cache invalidation) | 5, 12 |
| AC-10 (Unit tests) | all steps |
