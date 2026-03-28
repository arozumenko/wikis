# Proposal: Q&A Knowledge Flywheel

> **Status**: Draft
> **Author**: Peter Petroczy (with Claude)
> **Date**: 2026-03-20
> **Target repo**: [arozumenko/wikis](https://github.com/arozumenko/wikis)

---

## Problem

Every Q&A interaction in wikis is **stateless and ephemeral**. The system runs a full LangGraph agent pipeline (up to 8 tool-calling iterations, FAISS lookups, code graph traversal, LLM calls) for every single question — even if an identical or semantically similar question was answered five minutes ago.

Chat history lives in a React `useState` hook and vanishes the moment the user refreshes the page or closes the drawer. Good answers never improve the wiki. Bad answers are never flagged. There is no feedback loop.

This wastes LLM tokens, adds latency for repeat questions, and throws away valuable knowledge that could be feeding back into the documentation.

---

## Proposed Solution

Turn Q&A from a throwaway search into a **knowledge flywheel** with three phases:

1. **Capture + Cache** — Store every Q&A interaction, skip the full agent when a similar question was already answered
2. **Validate + Browse** — Let users review past Q&A, mark answers as good/bad, browse accumulated knowledge
3. **Enrich** — Feed validated Q&A back into the wiki's vector store and generate FAQ pages

---

## Current Architecture (Research Summary)

### How Q&A Works Today

```
User question
    |
    v
POST /api/v1/ask (AskRequest: wiki_id, question, chat_history, k)
    |
    v
AskService.ask_stream() / ask_sync()
    |
    +-- ComponentCache.get_or_build()     <-- caches retriever stack (TTL 1hr, max 5 wikis)
    |       loads: FAISS index, code graph, BM25 index, repo analysis
    |
    +-- Creates fresh AskEngine per request
    |
    v
AskEngine.ask()
    |
    +-- Builds system prompt (workflow: discover -> connect -> read -> answer)
    +-- Builds user prompt (question + repo context + last 4 chat messages)
    +-- Creates LangGraph agent with tools:
    |       search_symbols, get_relationships, get_code, search_docs, query_graph, think
    +-- Streams agent execution (max 8 iterations)
    |
    v
SSE events -> Frontend
    thinking_step (tool calls)
    answer_chunk (streaming text)
    ask_complete (final answer + sources)
```

**Key files:**
- `backend/app/api/routes.py` — HTTP endpoints (`/ask`, `/research`)
- `backend/app/services/ask_service.py` — orchestrates engine, manages component cache
- `backend/app/core/ask_engine.py` — LangGraph agent creation, streaming loop
- `backend/app/services/research_service.py` — deep research (same pattern, 15 iterations, more tools)
- `backend/app/core/deep_research/research_engine.py` — full research agent with todos + filesystem

### How Data is Stored

**Database** (`backend/app/models/db_models.py`):
- Single `WikiRecord` table — stores wiki metadata only (id, repo_url, branch, status, page_count)
- No Q&A tables exist
- SQLAlchemy async with SQLite (or PostgreSQL)
- Migrations via `_add_missing_columns()` in `backend/app/db.py` (idempotent ALTER TABLE)

**Vector Store** (`backend/app/core/vectorstore.py`):
- FAISS index per wiki, stored as `{cache_key}.faiss` + docstore files
- `VectorStoreManager.add_documents()` method exists for adding new docs post-generation
- BM25 disk-backed index alongside FAISS
- Cache key derived from `repo_path + commit_hash`

**Storage** (`backend/app/storage/`):
- Abstract `ArtifactStorage` with local filesystem and S3 backends
- Wiki artifacts at `data/artifacts/wiki_artifacts/{wiki_id}/`
- Wiki pages as markdown: `wiki_pages/{section}/{page}.md`
- The API auto-discovers all `.md` files under `wiki_pages/`

**Code Graph** (`backend/app/core/code_graph/`):
- NetworkX MultiDiGraph with code symbols + relationships
- FTS5 virtual table for symbol search
- Serialized as `.code_graph.gz`

### How the Frontend Works

**ChatDrawer** (`web/src/spa/components/ChatDrawer.tsx`):
- Floating FAB button, slides in from right (420px drawer)
- Two modes: Fast (ask) and Deep Research
- Messages in `useState` — gone on refresh
- Sends full chat history to backend per request

**WikiViewerPage** (`web/src/spa/pages/WikiViewerPage.tsx`):
- Three states: wiki content, Q&A view, generation progress
- AskBar at bottom of page
- AnswerView streams answer with SSE
- ToolCallPanel shows agent steps (deep research)

**WikiSidebar** (`web/src/spa/components/WikiSidebar.tsx`):
- Collapsible sections with pages
- Pages sorted by `order` field
- Active page highlighted

**API client** (`web/src/spa/api/wiki.ts` + `sse.ts`):
- `subscribeAskSSE()` / `subscribeResearchSSE()` for streaming
- `askQuestion()` / `deepResearch()` for sync responses

### Request/Response Models

```python
# backend/app/models/api.py

class AskRequest(BaseModel):
    wiki_id: str
    question: str
    chat_history: list[ChatMessage] = []
    k: int = 15

class AskResponse(BaseModel):
    answer: str
    sources: list[SourceReference] = []
    tool_steps: int = 0

class SourceReference(BaseModel):
    file_path: str
    line_start: int | None = None
    line_end: int | None = None
    snippet: str | None = None
    symbol: str | None = None
    symbol_type: str | None = None
    relevance_score: float | None = None
```

### Service Initialization

```python
# backend/app/main.py lifespan()

app.state.settings = settings
app.state.storage = storage
app.state.wiki_management = WikiManagementService(storage, get_session_factory())
app.state.wiki_service = WikiService(settings, storage, wiki_management=wiki_management)
app.state.ask_service = AskService(settings, storage)
app.state.research_service = ResearchService(settings, storage)
```

---

## Design: Phase 1 — Capture + Cache

### New Database Table: `qa_interaction`

```
id              UUID PK
wiki_id         VARCHAR INDEX       — references wiki.id by value (no FK, matches existing pattern)
question        TEXT NOT NULL
question_hash   VARCHAR INDEX       — SHA256 of normalized question for O(1) exact dedup
answer          TEXT NOT NULL
sources_json    TEXT                 — JSON serialized list[SourceReference]
tool_steps      INTEGER DEFAULT 0
mode            VARCHAR DEFAULT 'fast'   — 'fast' | 'deep'
status          VARCHAR DEFAULT 'pending' — 'pending' | 'validated' | 'rejected' | 'enriched'
validation_type VARCHAR              — null | 'user_upvote' | 'user_downvote' | 'llm_auto'
user_id         VARCHAR              — who asked
created_at      DATETIME DEFAULT now()
validated_at    DATETIME
```

### QA Cache: Separate Per-Wiki FAISS Index

A lightweight `QACacheManager` class manages a secondary FAISS index per wiki, stored as `{wiki_cache_key}.qa.faiss`. This is separate from the main code index because:

- **Different content**: natural language Q&A vs. code chunks
- **Different lifecycle**: grows with usage, not tied to repo commits
- **Independent rebuild**: can wipe QA cache without re-indexing the codebase

### Cache Lookup Flow

```
1. Hash question → check question_hash in DB (exact match, O(1))
2. If miss → embed question → search QA FAISS index (cosine threshold 0.92)
3. If hit → return cached AskResponse (fully transparent to user)
4. If miss → run full agent pipeline
5. After agent completes → record to DB + embed into QA FAISS (as BackgroundTask)
```

The 0.92 threshold is deliberately high — "How does auth work?" should NOT match "How does auth work in the middleware?" as they may yield different agent discoveries. Configurable via `qa_cache_similarity_threshold`.

### Integration Point: `ask_service.py`

```python
# Before agent run:
cached = await self.qa_service.find_cached(request.wiki_id, request.question)
if cached:
    return cached  # Skip agent entirely

# After agent completes (as BackgroundTask for streaming):
await self.qa_service.record_interaction(
    wiki_id=request.wiki_id,
    question=request.question,
    answer=final_answer,
    sources=sources,
    tool_steps=step_count,
    mode="fast",
    user_id=user_id,
)
```

### New API Endpoints

- `GET /api/v1/wikis/{wiki_id}/qa` — paginated list of Q&A interactions (filterable by status)
- `GET /api/v1/wikis/{wiki_id}/qa/stats` — cache hit rate, total interactions, validated count

### Cache Invalidation

When a wiki is refreshed/regenerated or deleted:
- Delete the `.qa.faiss` file for that wiki
- Mark all `status='pending'` records as `status='stale'`
- The existing `ComponentCache.evict()` call in the delete route already handles the code vector store

### Config Additions

```python
# backend/app/config.py
qa_cache_enabled: bool = True
qa_cache_similarity_threshold: float = 0.92
qa_cache_max_age_seconds: int = 86400  # 24 hours
```

### Response Model Changes

```python
# backend/app/models/api.py — AskResponse gets one new field:
class AskResponse(BaseModel):
    answer: str
    sources: list[SourceReference] = []
    tool_steps: int = 0
    qa_id: str | None = None  # ID of the recorded QA interaction
```

Cache hits are fully transparent — no `cache_hit` indicator shown to the user.

---

## Design: Phase 2 — Validate + Browse

### Validation UX

After each answer (in both ChatDrawer and AnswerView), show thumbs-up/thumbs-down buttons. This maps to:

`POST /api/v1/wikis/{wiki_id}/qa/{qa_id}/validate` with body `{"action": "upvote" | "downvote"}`

The `qa_id` from Phase 1's response tells the frontend which record to validate.

### Q&A History Panel

New component: `QAHistoryPanel.tsx` — accessible from the WikiSidebar as a "Q&A History" section.

Shows:
- Paginated list of past Q&A
- Filter by status (all / validated / rejected / pending)
- Each entry: question, truncated answer, timestamp, validation badge, mode
- Click to expand full answer + sources
- Bulk actions: validate all, reject all

Integration: follows the same state pattern as `askState` in WikiViewerPage — clicking "Q&A History" in the sidebar swaps the main content area.

### Frontend Changes

- `web/src/spa/api/wiki.ts` — add `listQAInteractions()`, `validateQA()`
- `web/src/spa/components/QAHistoryPanel.tsx` — new
- `web/src/spa/components/QAValidationButtons.tsx` — new (thumbs up/down)
- `web/src/spa/components/ChatDrawer.tsx` — add validation buttons
- `web/src/spa/components/WikiSidebar.tsx` — add "Q&A History" section
- `web/src/spa/pages/WikiViewerPage.tsx` — handle QA history panel state

---

## Design: Phase 3 — Enrich (Knowledge Flywheel)

### Vector Store Enrichment

Validated Q&A pairs get added to the main wiki FAISS index as new documents:

```python
Document(
    page_content=f"Q: {question}\n\nA: {answer}",
    metadata={
        "source": "qa_interaction",
        "qa_id": interaction.id,
        "chunk_type": "qa",
    },
)
```

This means future questions that are similar to previously validated Q&A will find those answers during retrieval, giving the agent better context before it even starts.

After enrichment: mark records as `status='enriched'`, evict `ComponentCache` so the next ask loads the updated index.

### FAQ Page Generation

Batch endpoint: `POST /api/v1/wikis/{wiki_id}/qa/generate-pages`

1. Fetch all validated Q&A for the wiki
2. Cluster by topic using QA embeddings
3. For each cluster, generate a wiki page using existing LLM pipeline
4. Save as `wiki_pages/faq/{topic}.md` via `ArtifactExporter`
5. Pages appear automatically in sidebar (existing API discovers all `.md` files)

### New Service

`backend/app/services/enrichment_service.py`:
- `enrich_vectorstore(wiki_id)` — adds validated Q&A to main FAISS
- `generate_faq_pages(wiki_id)` — clusters + generates FAQ pages
- `get_enrichment_stats(wiki_id)` — counts of validated/enriched/pending

---

## File Change Summary

### New Files
| File | Phase | Purpose |
|------|-------|---------|
| `backend/app/services/qa_service.py` | 1 | QA recording, cache, listing |
| `backend/app/core/qa_cache.py` | 1 | QA FAISS index management |
| `backend/app/services/enrichment_service.py` | 3 | Vector store enrichment, FAQ generation |
| `web/src/spa/components/QAHistoryPanel.tsx` | 2 | Q&A history browsing |
| `web/src/spa/components/QAValidationButtons.tsx` | 2 | Thumbs up/down component |

### Modified Files
| File | Phase | Change |
|------|-------|--------|
| `backend/app/models/db_models.py` | 1 | Add `QARecord` model |
| `backend/app/models/api.py` | 1+2 | Add `qa_id` to `AskResponse`, add QA list/validate models |
| `backend/app/db.py` | 1 | Add QA table migrations |
| `backend/app/config.py` | 1 | Add `qa_cache_*` settings |
| `backend/app/main.py` | 1 | Instantiate `QAService` |
| `backend/app/api/routes.py` | 1+2+3 | Add QA endpoints |
| `backend/app/services/ask_service.py` | 1 | Cache check before agent, record after |
| `backend/app/services/research_service.py` | 1 | Same cache/record integration |
| `web/src/spa/api/wiki.ts` | 2 | Add QA API client functions |
| `web/src/spa/components/ChatDrawer.tsx` | 2 | Add validation buttons |
| `web/src/spa/components/WikiSidebar.tsx` | 2 | Add Q&A History section |
| `web/src/spa/pages/WikiViewerPage.tsx` | 2 | Handle QA history panel |

---

## Key Design Decisions

| Decision | Rationale |
|----------|-----------|
| **Separate QA FAISS index** | Different content type, independent lifecycle, can rebuild without re-indexing code |
| **Hash-based exact match before semantic** | O(1) dedup for the common case (same user, exact same question) before expensive FAISS search |
| **0.92 cosine threshold** | High to avoid wrong cache hits. Configurable. |
| **No FK on wiki_id** | Follows existing `WikiRecord` pattern. Cascade deletion in app code. |
| **BackgroundTask for recording** | SSE streaming response isn't delayed by DB write |
| **Transparent cache hits** | User sees no difference. Simplest UX. |
| **Reuses existing infra** | `VectorStoreManager.add_documents()`, `ArtifactExporter`, `ComponentCache.evict()`, `_add_missing_columns()` migration pattern |

---

## Implementation Order

Phase 1 is the foundation and delivers immediate value (faster repeat queries, persistent Q&A history). Phases 2 and 3 build on top incrementally. Each phase is a separate PR.
