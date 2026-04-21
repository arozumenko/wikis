# Architecture

## System Overview

Wikis is a two-service monorepo. The **web app** handles auth (Better-Auth), serves the React SPA, and acts as an API gateway — proxying REST calls and streaming SSE responses from the backend. The **backend** runs the wiki generation engine, Q&A/research services, and an MCP server. Cross-service auth uses RS256 JWTs: the web app issues them, the backend validates via JWKS.

## Components

| Component | Purpose | Tech | Port |
|-----------|---------|------|------|
| Web App | Auth, SPA host, API proxy | Next.js 15, Better-Auth, Prisma | :3000 |
| Backend | Wiki engine, Q&A, research | FastAPI, LangGraph, FAISS | :8000 |
| MCP Server | AI IDE integration | FastMCP (embedded in backend) | :8000/mcp |
| Docs site | Static documentation | Next.js (GitHub Pages) | (deployed) |
| Ollama | Local LLM inference | Ollama (optional profile) | :11434 |
| PostgreSQL | Shared DB (optional) | Postgres 16 (optional profile) | :5432 |

## Data Flow

### Wiki Generation

```
User submits repo URL (GeneratePage)
  → POST /api/v1/generate → backend
  → WikiService.generate() → stores Invocation record
  → background WikiJobWorker:
      LocalRepositoryManager.clone()
      FilesystemIndexer.scan()
      tree-sitter parsers → CodeGraph (NetworkX + SQLite FTS5)
      sentence-transformers → FAISS vector index
      WikiStructurePlanner (LLM) → page outline
      OptimizedWikiGenerationAgent (LangGraph, parallel) → page content
      artifacts stored (local disk or S3)
  → SSE events → GET /api/v1/invocations/{id}/stream
  → SPA GenerationProgress component
```

### Q&A / Ask

```
User submits question (AskBar)
  → POST /api/v1/ask → backend (SSE)
  → AskService → loads wiki artifacts
  → EnsembleRetriever (FAISS + BM25 + FTS5)
  → LangChain agent (tool calls: search, get_page)
  → QAService: check QACacheManager for semantic hit (FAISS, threshold 0.92)
  → stream answer chunks via SSE
  → citations from retrieved chunks
```

### Deep Research

```
User submits complex query (AskBar, research mode)
  → POST /api/v1/research → backend (SSE)
  → ResearchService → multi-step LangGraph graph
  → planning agent → tool calls → synthesis agent
  → ThinkingSteps + ToolCallPanel streamed to SPA
```

### Auth Flow

```
User → POST /api/auth/sign-in → Better-Auth → issues RS256 JWT
Browser stores session cookie (Better-Auth)
API calls: middleware extracts session → fetches JWT → adds Authorization header
Backend: auth.py → GET /api/auth/jwks → validates JWT signature
Local dev: AUTH_ENABLED=false → all auth skipped
```

## API Boundaries

### Backend REST API (prefix `/api/v1`)

| Method | Path | Purpose |
|--------|------|---------|
| POST | `/generate` | Start wiki generation (202 + invocation_id) |
| GET | `/invocations/{id}` | Poll invocation status |
| GET | `/invocations/{id}/stream` | SSE: generation progress events |
| GET | `/wikis` | List wikis for current user |
| GET | `/wikis/{id}` | Get wiki summary + page list |
| GET | `/wikis/{id}/pages/{page}` | Get rendered wiki page |
| DELETE | `/wikis/{id}` | Delete wiki + artifacts |
| POST | `/ask` | SSE: Q&A against a wiki |
| POST | `/research` | SSE: deep research against a wiki |
| GET | `/health` | Health check (LLM + embeddings status) |

### SSE Event Types

Wiki generation: `progress`, `page_complete`, `complete`, `error`
Ask/Research: `chunk`, `thinking_step`, `tool_call`, `source`, `complete`, `error`

### MCP Tools (`:8000/mcp`)

| Tool | Purpose |
|------|---------|
| `list_wikis` | List available wikis |
| `get_wiki_page` | Retrieve a wiki page by ID |
| `ask` | Q&A against a wiki |
| `research` | Deep multi-step research |

### Web Auth API

| Path | Purpose |
|------|---------|
| `/api/auth/sign-in/email` | Email/password login |
| `/api/auth/sign-out` | Logout |
| `/api/auth/jwks` | RS256 public keys (consumed by backend) |
| `/api/auth/token` | JWT token endpoint |
| `/api/auth/api-key/verify` | API key verification |

## Database

### Backend (SQLAlchemy async)

| Table | Purpose |
|-------|---------|
| `wikis` | Wiki registry (repo URL, status, owner_id) |
| `invocations` | Generation job records (status, progress) |
| `qa_records` | Q&A conversation history + cache metadata |

Migration: tables created by `create_tables()` on startup (SQLAlchemy metadata). No Alembic.

Default: SQLite at `./data/artifacts/`. Set `DATABASE_URL` for PostgreSQL.

### Frontend (Prisma)

| Table | Purpose |
|-------|---------|
| Better-Auth tables | Users, sessions, accounts, verifications |
| `api_keys` | API key records (hash, config_id) |

Migrations: `npx prisma migrate dev`. Schema: `web/prisma/schema.prisma`.

## LLM Provider Architecture

`llm_factory.py` is the single provider abstraction. All services call:
- `create_llm(settings)` → `BaseLanguageModel`
- `create_embeddings(settings)` → `Embeddings`

Provider selection via `LLM_PROVIDER` env var. Optional pip extras activate each provider:
`ollama`, `gemini`, `bedrock`, `all-providers`.

LLM tiers: `LLM_MODEL` (high-tier, page generation) vs `LLM_MODEL_LOW` (low-tier, quality checks).
Fallback chain: `LLM_FALLBACK_MODELS` comma-separated with retry (max 5).

## Storage

`storage/` provides a unified interface:
- `LocalArtifactStorage` — writes to `./data/artifacts/` (default)
- `S3ArtifactStorage` — writes to S3 (`STORAGE_BACKEND=s3`)

Wiki artifacts: JSON page content, FAISS indexes, BM25 indexes, SQLite FTS5 DBs.

## Key Design Decisions

- **LangGraph for parallelism**: wiki page generation runs multiple LLM calls in parallel via LangGraph nodes
- **Ensemble retrieval**: FAISS (dense) + BM25 (sparse) + FTS5 (keyword) combined for best recall
- **SSE not WebSockets**: simpler, stateless, works through Next.js middleware proxy
- **MCP embedded**: MCP tools call services directly (no HTTP round-trip) — wired in `lifespan()`
- **QA cache**: semantic deduplication via FAISS (threshold 0.92) avoids redundant LLM calls
