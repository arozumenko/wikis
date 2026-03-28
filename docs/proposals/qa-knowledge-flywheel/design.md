# Technical Design: Q&A Knowledge Flywheel

**Spec**: `tool/docs/proposals/qa-knowledge-flywheel/spec.md`
**Created**: 2026-03-23
**Updated**: 2026-03-24 (Round 3 — addresses judge R2 findings H-1, H-2, M-1)
**Scope**: Phase 1 (Capture + Cache). Phases 2-3 noted at boundaries only.

---

## 1. Module Structure

### New Files (Phase 1)

| File | Purpose |
|------|---------|
| `backend/app/services/qa_service.py` | `QAService` — cache lookup, recording, listing, stats, invalidation |
| `backend/app/services/qa_cache_manager.py` | `QACacheManager` — per-wiki QA FAISS index lifecycle |
| `backend/app/models/qa_api.py` | Pydantic/dataclass models: `QARecordingPayload`, `AskResult`, QA endpoint schemas |

### Modified Files (Phase 1)

| File | Change |
|------|--------|
| `backend/app/models/db_models.py` | Add `QARecord` model (same file as `WikiRecord`) |
| `backend/app/services/ask_service.py` | Add QAService dependency; cache lookup before agent; return `AskResult` from sync path |
| `backend/app/api/routes.py` | Add QA endpoints (list, stats); `/ask` sync uses BackgroundTask for recording |
| `backend/app/models/api.py` | Add `qa_id: str` to `AskResponse` |
| `backend/app/main.py` | Initialize `QACacheManager` + `QAService`; pass QAService to AskService |
| `backend/app/dependencies.py` | Add `get_qa_service` dependency |
| `backend/app/config.py` | Add `qa_cache_*` settings |
| `backend/mcp_server/server.py` | Update `ask_codebase` to handle `AskResult` + direct-await recording |

`db.py` needs no changes — `Base.metadata.create_all` auto-creates the new `qa_record` table.

### Phase 2 additions (Validate + Browse)

Frontend components (`QAHistoryPanel.tsx`, `QAValidation.tsx`, `qa.ts` API client), validate endpoint implementation.

### Phase 3 additions (Enrich)

`qa_enrichment.py` service — enrichment + FAQ generation logic.

---

## 2. Data Model

### QARecord (SQLAlchemy — in `db_models.py`)

```python
class QARecord(Base):
    __tablename__ = "qa_record"

    id = Column(String, primary_key=True)              # Pre-generated UUID
    wiki_id = Column(String, nullable=False)           # No FK (matches WikiRecord pattern)
    question = Column(String, nullable=False)
    question_hash = Column(String, nullable=False)     # SHA256 hex digest
    answer = Column(String, nullable=False)
    sources_json = Column(String, nullable=True)       # JSON list[SourceReference]
    tool_steps = Column(Integer, default=0)
    mode = Column(String, default="fast")              # "fast" | "deep"
    status = Column(String, default="pending")         # pending|validated|rejected|enriched|stale
    validation_type = Column(String, nullable=True)    # null|user_upvote|user_downvote
    is_cache_hit = Column(Integer, default=0)          # SQLite-compatible boolean (0/1)
    source_qa_id = Column(String, nullable=True)       # UUID of original record for cache hits
    has_context = Column(Integer, default=0)           # 1 if chat_history was non-empty (contextual follow-up)
    source_commit_hash = Column(String, nullable=True) # Wiki commit_hash at recording time
    user_id = Column(String, nullable=True)
    created_at = Column(DateTime, server_default=func.now())
    validated_at = Column(DateTime, nullable=True)

    __table_args__ = (
        Index("ix_qa_wiki_hash", "wiki_id", "question_hash"),
        Index("ix_qa_wiki_status", "wiki_id", "status"),
        Index("ix_qa_wiki_cache", "wiki_id", "is_cache_hit"),
    )
```

**Design decisions:**
- `is_cache_hit` and `has_context` use `Integer` (0/1) — matches `WikiRecord.requires_token` pattern for SQLite compat
- No FK on `wiki_id` — matches existing WikiRecord which uses string IDs without FKs
- `sources_json` stored as JSON string — avoids join-heavy schema for read-mostly workload
- `has_context` distinguishes standalone questions (cacheable) from conversational follow-ups (not cacheable). Contextual answers depend on `chat_history` and must not be served to different conversations.
- `source_commit_hash` records the wiki's commit at answer time. Cache lookups only match records from the current commit, preventing stale validated/enriched answers from surviving across code revisions.
- Three composite indexes target the three main query patterns: hash lookup, status filter, cache stats

### QARecordingPayload (dataclass — in `qa_api.py`)

```python
@dataclass
class QARecordingPayload:
    """Data needed to persist a QA interaction. Created by AskService, consumed by callers."""
    qa_id: str
    wiki_id: str
    question: str
    answer: str
    sources_json: str
    tool_steps: int
    mode: str
    user_id: str | None
    is_cache_hit: bool
    source_qa_id: str | None
    embedding: np.ndarray | None
    has_context: bool
    source_commit_hash: str | None
```

### AskResult (dataclass — in `qa_api.py`)

```python
@dataclass
class AskResult:
    """Returned by AskService.ask_sync. Callers handle recording per their context."""
    response: AskResponse
    recording: QARecordingPayload | None
```

---

## 3. QACacheManager

Manages per-wiki QA FAISS indexes. Uses raw FAISS (`IndexFlatIP`) instead of LangChain's `FAISS` wrapper — simpler for single-vector operations with explicit normalization control.

### Interface

```python
class QACacheManager:
    def __init__(self, cache_dir: str, embeddings: Embeddings)

    async def search(
        self, wiki_id: str, question: str, threshold: float, k: int = 5
    ) -> tuple[list[str], np.ndarray]
    # Returns (qa_ids above threshold ordered by descending similarity, L2-normalized embedding)

    async def add(self, wiki_id: str, qa_id: str, embedding: np.ndarray) -> None
    # Adds pre-computed, L2-normalized embedding to the index

    def delete_index(self, wiki_id: str) -> bool

    async def rebuild_from_records(
        self, wiki_id: str, records: list[QARecord]
    ) -> None
    # Caller MUST pre-filter records to cache-eligible only
```

**Design decision — `k=5` multi-candidate search (addresses R2-H-1):** The FAISS index may contain rows that are no longer cache-eligible (expired by TTL, from a previous commit after refresh, or recovered from corruption). With `k=1`, if the nearest neighbor is one of these invalid rows, it is rejected post-fetch and the lookup falls through — missing a valid candidate that might exist at position 2 or 3. Searching `k=5` returns up to 5 candidates above threshold, ordered by descending similarity. `QAService.lookup_cache` iterates through them until one passes all validity checks. At <10K vectors with brute-force `IndexFlatIP`, k=5 adds negligible overhead (~same 10ms).

### Storage

| File | Contents |
|------|----------|
| `{cache_dir}/{wiki_id}.qa.faiss` | FAISS IndexFlatIP binary |
| `{cache_dir}/{wiki_id}.qa.ids.json` | Parallel list of qa_id strings |

**Design decision — `wiki_id` as index key (not `cache_key`):**
The spec references `{cache_key}.qa.faiss`. However, the QA index lifecycle is tied to the wiki entity, not a specific commit hash. Wiki regeneration (FR-010) deletes the QA FAISS file regardless. Using `wiki_id` directly eliminates the `cache_index.json` dependency and is semantically correct.

### Index Type: `faiss.IndexFlatIP`

- Brute-force inner product — exact, no training step, correct for <10K vectors
- With L2-normalized vectors (via `faiss.normalize_L2()`), inner product equals cosine similarity
- All embeddings normalized before insertion **and** search

### Atomic Write Pattern

Prevents corrupted reads during concurrent access:
1. Write index to `{wiki_id}.qa.faiss.tmp`
2. Write IDs to `{wiki_id}.qa.ids.json.tmp`
3. `os.rename()` temp -> final (atomic on POSIX, same filesystem)

On load: verify `len(qa_ids) == index.ntotal`. On mismatch -> rebuild from DB.

### Lazy Index Creation

FAISS dimension is unknown until the first embedding is computed. Index is created on first `add()` call with dimension inferred from the embedding vector.

### Concurrency

- Per-wiki `asyncio.Lock` prevents concurrent FAISS mutations
- FAISS embedding calls run in `asyncio.to_thread()` to avoid blocking the event loop
- `IndexFlatIP.search` is safe for concurrent reads (immutable data during search)

---

## 4. QAService

Orchestrates all Q&A knowledge flywheel operations. Injected into AskService and routes via FastAPI DI.

### Interface

```python
class QAService:
    def __init__(
        self, session_factory: async_sessionmaker, cache: QACacheManager | None, settings: Settings
    )

    async def lookup_cache(
        self, wiki_id: str, question: str, chat_history: list | None = None
    ) -> tuple[QARecord | None, np.ndarray | None]

    async def record_interaction(
        self, qa_id: str, wiki_id: str, question: str, answer: str,
        sources_json: str, tool_steps: int, mode: str, user_id: str | None,
        is_cache_hit: bool, source_qa_id: str | None, embedding: np.ndarray | None,
        has_context: bool = False, source_commit_hash: str | None = None
    ) -> None

    async def list_qa(
        self, wiki_id: str, status: str | None, limit: int, offset: int
    ) -> tuple[list[QARecord], int]

    async def get_stats(self, wiki_id: str) -> dict

    async def validate(
        self, wiki_id: str, qa_id: str, action: str
    ) -> QARecord  # Phase 2 implementation

    async def invalidate_wiki(self, wiki_id: str, delete: bool = False) -> None
```

### Cache Lookup Flow

```
lookup_cache(wiki_id, question, chat_history=None)
  |-- If qa_cache_enabled is False -> return (None, None)
  |-- If chat_history is non-empty -> return (None, None)  [contextual: not cacheable]
  |-- current_commit = _get_wiki_commit_hash(wiki_id)
  |-- question_hash = SHA256(question)
  |-- cutoff = now - qa_cache_max_age_seconds
  |
  |-- Layer 1: Exact hash match (DB)
  |   SELECT FROM qa_record
  |     WHERE wiki_id=? AND question_hash=?
  |       AND status IN ('pending','validated','enriched')
  |       AND has_context=0
  |       AND (source_commit_hash=? OR source_commit_hash IS NULL)
  |       AND created_at > cutoff
  |     ORDER BY created_at DESC LIMIT 1
  |   '-- If found -> return (record, None)  [no embedding computed]
  |
  |-- Layer 2: FAISS semantic search (k=5)
  |   qa_ids, embedding = cache_manager.search(wiki_id, question, threshold, k=5)
  |   for qa_id in qa_ids:                        <-- iterate candidates
  |     record = fetch QARecord by qa_id
  |     if record
  |       AND status IN (pending, validated, enriched)
  |       AND has_context=0
  |       AND (source_commit_hash=current_commit OR source_commit_hash IS NULL)
  |       AND created_at > cutoff:
  |         return (record, embedding)            <-- first valid match
  |   # No valid candidate among k results -> fall through
  |
  '-- return (None, embedding)  [embedding reused in recording]
```

**Context-awareness:** When `chat_history` is non-empty, the question is a conversational follow-up whose meaning depends on prior messages. These are skipped entirely for cache lookup. The early return avoids computing an embedding for uncacheable questions.

**Commit-awareness:** `_get_wiki_commit_hash(wiki_id)` queries `WikiRecord.commit_hash` for the current wiki state. Cache results must match the current commit (or have NULL commit hash for backward compat). This prevents validated/enriched answers from commit A from being served after the wiki refreshes to commit B.

**Multi-candidate iteration (addresses R2-H-1):** The FAISS search returns up to k=5 candidates above threshold. Each is validated against DB state (status, context, commit, TTL). The first valid candidate wins. This prevents false misses when an invalid row (old commit, expired) happens to be the nearest FAISS neighbor but a valid row exists behind it.

**Embedding reuse**: `lookup_cache` computes the embedding during FAISS search. On cache miss, this embedding is passed to `record_interaction` via `cache_manager.add()`, saving one embedding API call (~100-300ms).

### Recording Flow

```
record_interaction(qa_id, wiki_id, question, answer, ..., embedding,
                   has_context, source_commit_hash)
  |-- Create QARecord with all fields (including has_context, source_commit_hash)
  |-- INSERT via async session
  |-- If NOT cache_hit AND NOT has_context AND embedding provided:
  |   cache_manager.add(wiki_id, qa_id, embedding)
  '-- On failure: log at ERROR level (graceful degradation per FR-001)
```

**Contextual answers are not indexed in FAISS:** When `has_context=True`, the answer depends on conversational context and cannot be reused. The DB record is still created (for history and stats), but the FAISS index is not updated.

### Invalidation

```python
async def invalidate_wiki(self, wiki_id: str, delete: bool = False) -> None:
    async with self.session_factory() as session:
        if delete:
            # Wiki deleted -- remove all QA rows
            await session.execute(
                sa_delete(QARecord).where(QARecord.wiki_id == wiki_id)
            )
        else:
            # Wiki refreshed -- mark pending -> stale, preserve validated/enriched
            await session.execute(
                sa_update(QARecord)
                .where(QARecord.wiki_id == wiki_id, QARecord.status == "pending")
                .values(status="stale")
            )
        await session.commit()
    self.cache.delete_index(wiki_id)
```

**Refresh + commit-awareness:** On wiki refresh, `pending` records are marked `stale` and the FAISS index is deleted (per FR-010). Validated/enriched records survive in the DB for history purposes. After refresh, the wiki's `commit_hash` changes and cache lookups check `source_commit_hash` against the current commit — old validated/enriched records are naturally excluded without deletion.

---

## 5. Ask Integration

Cache lookup lives in `AskService` (in `ask_service.py`), satisfying FR-004 and covering both HTTP and MCP callers. Recording responsibility varies by caller context to preserve the accepted `BackgroundTask` contract for HTTP responses.

### AskService Changes

`AskService.__init__` gains a `QAService` parameter (optional, for backward compat):

```python
class AskService:
    def __init__(
        self, settings: Settings, storage: ArtifactStorage,
        qa_service: QAService | None = None,
    ) -> None:
        # ... existing init ...
        self._qa_service = qa_service
```

### Sync Path (`ask_sync`) — returns `AskResult`

`ask_sync` performs cache lookup and agent invocation, but does **not** record. It returns an `AskResult` containing both the response and a `QARecordingPayload`. The caller handles recording per its context (BackgroundTask for HTTP, direct await for MCP).

```
ask_sync(request: AskRequest) -> AskResult
  |-- qa_id = uuid4()
  |-- has_context = bool(request.chat_history)
  |-- commit_hash = None
  |
  |-- Cache lookup (if QAService available):
  |   cached, embedding = qa_service.lookup_cache(wiki_id, question, chat_history)
  |   '-- If hit:
  |       response = AskResponse(answer=cached.answer, ..., qa_id=qa_id)
  |       payload = QARecordingPayload(qa_id, ..., is_cache_hit=True, source_qa_id=cached.id)
  |       return AskResult(response, payload)
  |
  |-- Cache miss -- run agent:
  |   response = _ask_sync_agent(request)  [existing agent logic, renamed]
  |   response.qa_id = qa_id
  |   payload = QARecordingPayload(qa_id, ..., is_cache_hit=False, embedding=embedding)
  |   return AskResult(response, payload)
```

**Why AskResult instead of internal recording:** The accepted spec (FR-004, FR-013) requires `BackgroundTask` for recording on HTTP responses. `BackgroundTask` is an HTTP-specific abstraction — it requires a route handler and a response lifecycle. `AskService` is a shared service used by both HTTP routes and MCP tools; it cannot assume either context. By returning the recording payload, each caller uses the appropriate mechanism.

### SSE Path (`ask_stream`) — with success gating

The SSE generator is produced by `AskService` and handles its own recording via `try/finally` with success gating. This is the design-phase mitigation explicitly approved in FR-004: "try/finally in the generator as a fallback."

```
ask_stream(request: AskRequest) -> AsyncGenerator
  |-- qa_id = uuid4()
  |-- has_context = bool(request.chat_history)
  |
  |-- Cache lookup (if QAService available):
  |   cached, embedding = qa_service.lookup_cache(wiki_id, question, chat_history)
  |   '-- If hit:
  |       yield answer_chunk (cached answer text)
  |       yield task_complete (answer, sources, steps=0, qa_id=qa_id)
  |       asyncio.create_task(_record_safely(..., is_cache_hit=True))
  |       return
  |
  |-- Cache miss -- stream from agent:
  |   completed = False
  |   answer, sources_json, tool_steps = "", "[]", 0
  |
  |   try:
  |     async for event in _ask_stream_agent(request):
  |       if event_type in ("task_complete", "ask_complete"):
  |         extract answer, sources, steps
  |         inject qa_id into event data
  |         completed = True        <-- success gate
  |       yield event
  |   finally:
  |     if completed and qa_service:
  |       asyncio.create_task(_record_safely(..., is_cache_hit=False))
  |     elif not completed:
  |       logger.warning("Ask stream did not complete -- recording skipped")
```

**Why `asyncio.create_task` in `finally` (not `await`):** The `finally` block must not keep the generator open after all events are yielded. `asyncio.create_task` schedules recording as fire-and-forget — matching `BackgroundTask`'s non-blocking semantics while using the spec-approved `try/finally` mechanism for the SSE path. On client disconnect or agent failure, `completed=False` and no recording is attempted.

**Success gating:** The `completed` flag is set to `True` only when a successful terminal event is received. This prevents three failure modes from polluting the cache:
1. **Client disconnect** — generator cancelled, `completed=False`, no recording
2. **Agent failure** — `task_failed`/`ask_error` emitted, `completed` stays `False`, no recording
3. **Exception** — uncaught error, `completed=False`, no recording

### `/ask` Route — BackgroundTask for sync recording

The route uses `BackgroundTask` for the sync path, preserving the accepted spec contract (FR-004, FR-013):

```python
@router.post("/ask")
async def ask(
    request: AskRequest,
    background_tasks: BackgroundTasks,
    service: AskService = Depends(get_ask_service),
    qa_service: QAService = Depends(get_qa_service),
):
    if "text/event-stream" in accept:
        # SSE: generator handles its own recording via try/finally
        async def stream():
            async for event in service.ask_stream(request):
                yield SSE-formatted event
        return StreamingResponse(stream(), media_type="text/event-stream")

    # Sync: BackgroundTask for recording (preserves FR-004/FR-013 contract)
    result = await service.ask_sync(request)
    if result.recording:
        background_tasks.add_task(qa_service.record_interaction, **asdict(result.recording))
    return result.response
```

### MCP Coverage

The MCP tool `ask_codebase` (`mcp_server/server.py`) calls `ask_sync` and handles recording directly — there is no HTTP response lifecycle, so `BackgroundTask` is unavailable:

```python
async def ask_codebase(wiki_id: str, question: str) -> dict:
    request = AskRequest(wiki_id=wiki_id, question=question)
    result = await _ask_service.ask_sync(request)
    if result.recording and _qa_service:
        try:
            await _qa_service.record_interaction(**asdict(result.recording))
        except Exception:
            logger.error("MCP QA recording failed", exc_info=True)
    return result.response.model_dump()
```

**Recording per caller context (summary):**

| Caller | Cache lookup | Recording mechanism | Spec basis |
|--------|-------------|-------------------|------------|
| HTTP sync | AskService (before agent) | `BackgroundTask` via route | FR-004, FR-013 |
| HTTP SSE | AskService (before agent) | `try/finally` + `create_task` in generator | FR-004 design-phase mitigation |
| MCP | AskService (before agent) | Direct `await` in tool function | No HTTP context; only option |

### qa_id in Responses

| Path | Where qa_id appears |
|------|-------------------|
| Sync | `AskResponse.qa_id: str` — always populated with pre-generated UUID |
| SSE | `qa_id` field in `task_complete` event data payload |

**`qa_id` is `str`, not optional (addresses R2-M-1).** The accepted spec (FR-006) defines `qa_id: str`. The pre-generated UUID is always set before the response is returned. If recording later fails, the `qa_id` becomes a dangling reference (FR-001 graceful degradation), but it is never absent from a successful response.

### Cache-Hit SSE Event Sequence (FR-005)

Cache hits emit the same terminal events as a fresh agent run:
1. `answer_chunk` events — cached answer text
2. `task_complete` event — `{answer, sources, steps: 0, qa_id}`

No synthetic `thinking_step` events. The absence of thinking steps is expected for fast responses.

---

## 6. API Endpoints (New — Phase 1)

### Authorization

All QA endpoints MUST verify the caller's access to the parent wiki before returning or mutating QA data. Each endpoint calls `WikiManagementService.get_wiki(wiki_id, user_id=user_id)` — if this returns `None`, the endpoint returns HTTP 404. This reuses the existing owner/shared visibility rules and prevents leaking or mutating private Q&A data.

This matches the existing pattern in `get_wiki_page` (`routes.py:446-448`), which calls `management.get_wiki(wiki_id, user_id=user_id)` before serving content.

### GET /api/v1/wikis/{wiki_id}/qa

Paginated Q&A history (FR-007).

| Param | Type | Default | Description |
|-------|------|---------|-------------|
| `limit` | int | 20 | Page size (max 100) |
| `offset` | int | 0 | Pagination offset |
| `status` | str? | null | Filter: pending, validated, rejected, enriched, stale |

Response:
```python
class QAListResponse(BaseModel):
    items: list[QARecordResponse]
    total: int
    limit: int
    offset: int
```

### GET /api/v1/wikis/{wiki_id}/qa/stats

QA statistics (FR-008).

Response:
```python
class QAStatsResponse(BaseModel):
    total_count: int
    cache_hit_count: int       # COUNT WHERE is_cache_hit=1
    validated_count: int       # COUNT WHERE status='validated'
    rejected_count: int        # COUNT WHERE status='rejected'
    hit_rate: float            # cache_hit_count / total_count (0 if empty)
```

### POST /api/v1/wikis/{wiki_id}/qa/{qa_id}/validate (Phase 2)

Designed now, implemented in Phase 2 (FR-009).

Request: `{ "action": "upvote" | "downvote" }`

Behavior:
- `upvote` -> status=validated, validation_type=user_upvote, validated_at=now
- `downvote` -> status=rejected, validation_type=user_downvote, validated_at=now
- Last vote wins (overwrite previous validation)
- qa_id not found -> HTTP 404 (dangling reference per FR-001)

---

## 7. Configuration

New `Settings` fields in `config.py`:

```python
# Q&A Cache
qa_cache_enabled: bool = True
qa_cache_similarity_threshold: float = 0.92
qa_cache_max_age_seconds: int = 86400  # 24 hours
```

Follows existing env-var-driven Pydantic settings pattern. All three are overridable via environment variables (`QA_CACHE_ENABLED`, `QA_CACHE_SIMILARITY_THRESHOLD`, `QA_CACHE_MAX_AGE_SECONDS`).

---

## 8. App Initialization

In `main.py` lifespan, after existing service initialization. Uses degraded-mode initialization matching the existing `check_providers()` non-blocking pattern — embedding failures must not prevent app startup:

```python
from app.services.qa_cache_manager import QACacheManager
from app.services.qa_service import QAService

qa_cache: QACacheManager | None = None
if settings.qa_cache_enabled:
    try:
        from app.services.llm_factory import create_embeddings
        embeddings = create_embeddings(settings)
        qa_cache = QACacheManager(settings.cache_dir, embeddings)
        logger.info("QA cache initialized")
    except Exception as e:
        logger.warning(f"QA cache disabled — embedding init failed: {e}")

qa_service = QAService(get_session_factory(), qa_cache, settings)
app.state.qa_service = qa_service

# Pass QAService to AskService for cache integration
app.state.ask_service = AskService(settings, storage, qa_service=qa_service)
```

**Degraded mode:** When `qa_cache_enabled=False` or embedding creation fails, QAService runs without a cache — `lookup_cache` returns `(None, None)` and `record_interaction` skips FAISS indexing. DB recording (QARecord persistence) still works. This ensures the existing `test_app` fixture (which doesn't seed embedding credentials) continues to work.

**Order matters:** QAService must be initialized before AskService, since AskService receives it as a dependency.

Reuses the same embedding model as the code vectorstore — ensures dimensional compatibility for Phase 3 enrichment (QA docs added to main FAISS index).

In `dependencies.py`:

```python
def get_qa_service(request: Request) -> QAService:
    return request.app.state.qa_service
```

MCP `set_services()` gains `qa_service` parameter so the MCP tool can record interactions.

---

## 9. Wiki Lifecycle Integration

### Delete Wiki

In `routes.py:delete_wiki`, after existing cleanup:

```python
qa_service: QAService = request.app.state.qa_service
await qa_service.invalidate_wiki(wiki_id, delete=True)
```

DELETE all QARecord rows + delete QA FAISS file (FR-010, edge case).

### Refresh Wiki

In `routes.py:refresh_wiki`, before starting refresh:

```python
qa_service: QAService = request.app.state.qa_service
await qa_service.invalidate_wiki(wiki_id, delete=False)
```

Mark pending -> stale + delete QA FAISS. Validated/enriched records survive in DB for history purposes, but cache lookups check `source_commit_hash` against the new wiki commit, so old validated/enriched answers are naturally excluded from cache results.

| Action | QARecords | QA FAISS | Cache effect |
|--------|-----------|----------|-------------|
| Delete | DELETE all rows | Delete file | Fully cleared |
| Refresh | pending -> stale | Delete file | Rebuilt from scratch; old validated/enriched excluded by commit mismatch |

---

## 10. Error Handling

### Graceful Degradation (FR-001)

If `record_interaction` fails (DB unavailable, FAISS error):
1. Log at ERROR level
2. Answer was already delivered to user — no impact on UX
3. `qa_id` becomes a dangling reference — `validate` returns 404
4. Interaction lost from history — acceptable degraded behavior

### Success-Gated Recording

The SSE `try/finally` recording only executes when `completed=True` (a successful terminal event was received). Partial or failed answers are never persisted:
- Client disconnect before completion -> no recording
- Agent failure (`task_failed`) -> no recording
- Exception during streaming -> no recording

In all cases, the pre-generated `qa_id` becomes a dangling reference — same behavior as FR-001 DB failures.

### FAISS Index Corruption

If QA FAISS file is corrupted or IDs/index are out of sync:
1. Detected on load: `len(qa_ids) != index.ntotal`
2. Recovery: `QAService` rebuilds from DB using **cache-eligible records only**: `status IN (pending, validated, enriched)` AND `has_context=0` AND (`source_commit_hash` = current wiki commit OR `source_commit_hash IS NULL`) AND `created_at > cutoff`
3. Log warning, then proceed normally

This ensures the rebuilt index contains only rows that would pass the lookup validity checks, preventing the false-miss scenario where an invalid nearest neighbor blocks valid candidates behind it.

### DB Concurrency

- SQLite: single-writer — acceptable at expected QA volume (<1 write/sec typical)
- PostgreSQL: concurrent writes handled natively
- All DB access via `async_sessionmaker` — matches existing codebase pattern

---

## 11. Phase Boundaries

### Phase 1 (this design) — Capture + Cache

Covers: FR-001 through FR-008, FR-010, FR-013, FR-014

Deliverables:
- `QARecord` model + auto-created table (with `has_context` and `source_commit_hash` fields)
- `QACacheManager` (FAISS index lifecycle, k=5 multi-candidate search)
- `QAService` (lookup, record, list, stats, invalidate)
- `AskService` integration (cache check before agent, `AskResult` return for caller-specific recording)
- `/ask` route with BackgroundTask recording (sync) and try/finally generator (SSE)
- MCP `ask_codebase` with direct-await recording
- `/wikis/{wiki_id}/qa` list endpoint (with wiki authorization)
- `/wikis/{wiki_id}/qa/stats` stats endpoint (with wiki authorization)
- Wiki delete/refresh invalidation hooks
- `qa_cache_*` configuration settings
- Unit tests for all new code

### Phase 2 — Validate + Browse (separate PR)

Covers: FR-009, frontend components

- `POST /wikis/{wiki_id}/qa/{qa_id}/validate` implementation (with wiki authorization)
- `QAValidation.tsx` — thumbs up/down component
- `QAHistoryPanel.tsx` — browsable Q&A history
- `ChatDrawer.tsx` integration (show qa_id, enable voting)

### Phase 3 — Enrich (separate PR)

Covers: FR-011, FR-012

- Enrichment endpoint (add validated QA to main FAISS via `VectorStoreManager.add_documents`)
- FAQ generation endpoint (cluster validated QA + LLM page generation)
- Clustering algorithm selection (candidates: DBSCAN for natural cluster count, k-means for fixed)
- `ComponentCache` eviction after enrichment
