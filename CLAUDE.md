# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**Wikis** is an AI-powered documentation generator that turns code repositories into browsable wikis with architecture diagrams and an AI Q&A assistant. It is a two-service monorepo: a FastAPI backend and a consolidated Next.js web app (auth + frontend SPA).

## Services and Ports

| Service | Port | Directory | Tech |
|---------|------|-----------|------|
| Web App | 3000 | `web/` | Next.js 15 + Better-Auth + React 18 SPA |
| Backend | 8000 | `backend/` | FastAPI + Python 3.11 |
| MCP Server | :8000/mcp | `backend/mcp_server/` | streamable HTTP (embedded); stdio (standalone CLI) |

## Development Commands

### Docker (recommended)
```bash
docker compose up -d          # Start all services
docker compose up -d --build  # Rebuild and start
docker compose ps             # Check health status
```

### Backend (FastAPI)
```bash
cd backend
python3.11 -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"

# Optional LLM providers
pip install -e ".[ollama]"         # Ollama (local)
pip install -e ".[all-providers]"  # OpenAI + Anthropic + Ollama + Gemini + Bedrock

# Run with hot reload (skip JWT for local dev)
AUTH_ENABLED=false uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

### Web App (Next.js + React SPA)
```bash
cd web
npm ci
npx prisma migrate dev         # Apply DB migrations
npx prisma db seed             # Create default admin user (admin@wikis.dev / changeme123)
npm run dev                    # Starts on :3000
```

## Running Tests

### Backend unit tests (fast, no external deps)
```bash
cd backend
pytest tests/unit/ -v
```

### Backend integration tests (uses FastAPI TestClient)
```bash
cd backend
pytest tests/integration/ -v
AUTH_ENABLED=false pytest tests/ -v   # Skip JWT setup
```

### Backend with coverage
```bash
cd backend
pytest tests/ -v --cov=app --cov-report=term-missing
```

### Single test file
```bash
cd backend
pytest tests/unit/test_llm_factory.py -v
```

## Git Workflow

**Required setup after clone:**
```bash
git config core.hooksPath .githooks
```

The pre-push hook blocks pushes if the branch is behind `origin/main`. Fix with:
```bash
git fetch origin && git rebase origin/main
```

Branch from fresh main: `git fetch origin && git checkout -b <branch> origin/main`. All work goes via feature branches + PRs — no direct commits to main.

## Architecture

### Request Flow
```
Browser → Next.js Web App (:3000)
  → Auth pages: /login, /logout (Next.js App Router)
  → Auth API: /api/auth/* (Better-Auth, same process)
  → SPA: /, /wiki/:id, /settings (React SPA via catch-all page)
  → Backend API: /api/v1/* (Next.js rewrites → Backend :8000)
    → JWT validation via JWKS
    → Clone repo → Parse (tree-sitter) → Index (FAISS+BM25+FTS5) → Generate (LangGraph)
    → SSE progress events → SPA
```

### Web App Structure (`web/`)
- `src/app/` — Next.js App Router: login, logout, auth API routes
- `src/app/(spa)/` — Catch-all pages mounting the React SPA (`'use client'`, `ssr: false`)
- `src/spa/` — React SPA source (migrated from former `frontend/`):
  - `api/` — API client, SSE handlers, generated types
  - `components/` — 35+ React components (MUI)
  - `pages/` — Dashboard, WikiViewer, Settings, Generate
  - `hooks/` — useAuth, useThemeMode, useCopyToClipboard
  - `App.tsx` — BrowserRouter + theme provider + AuthGuard
- `src/lib/` — Better-Auth config, JWT issuance
- `src/middleware.ts` — Session guard + API proxy (rewrites non-SSE `/api/v1/*` → backend; SSE paths handled by App Router route handlers at `src/app/api/v1/`)
- `prisma/` — SQLite schema + migrations
- `next.config.ts` — Package transpilation and MUI import optimization (no rewrite config here)

### Backend Structure (`backend/app/`)
- `main.py` — FastAPI app factory + lifespan hooks (wires services + MCP)
- `config.py` — Pydantic settings (env vars)
- `auth.py` — JWT validation via JWKS from web app
- `db.py` — SQLAlchemy async engine + session factory
- `api/routes.py` — HTTP route handlers (thin; delegate to services)
- `services/` — Business logic: `wiki_service`, `wiki_management`, `ask_service`, `research_service`, `qa_service`, `qa_cache_manager`, `project_service`, `export_service`, `import_service`, `toolkit_bridge`, `health_check`, `llm_factory`
- `core/` — The wiki engine (40+ modules):
  - `parsers/` — Tree-sitter for 14+ languages
  - `code_graph/` — Graph builder + FTS/vector index (`unified_graph_text_index`, `postgres_graph_text_index`)
  - `agents/` — LangGraph wiki generation agents (`wiki_graph_optimized`)
  - `deep_research/` — Multi-step research engine (LangGraph)
  - `wiki_structure_planner/` — LLM outline + cluster planners
  - `storage/` — Wiki storage protocol + backends (`sqlite.py` UnifiedWikiDB / `postgres.py` pgvector) selected via `WIKI_STORAGE_BACKEND`
  - `multi_retriever.py`, `unified_retriever.py` — ensemble retrieval (dense + sparse + reranking)
  - `wiki_search_engine.py`, `project_search_engine.py`, `wiki_page_index.py` — search + wikilink graph
  - `graph_clustering.py`, `cluster_expansion.py`, `graph_topology.py` — Leiden clustering + topology enrichment
  - `repo_providers/` — GitHub, GitLab, Bitbucket, Azure DevOps
  - `feature_flags.py` — `WIKIS_CLUSTER_*` toggles
- `models/` — Pydantic request/response/event/search models
- `storage/` — Artifact storage abstraction (local / S3)
- `mcp_server/` (sibling to `app/`) — FastMCP tools wired to services in `app/main.py`

### Auth Cross-Service JWT
The web app issues RS256 JWTs. The backend validates them using `JWT_PUBLIC_KEY`. For local dev without auth: `AUTH_ENABLED=false`.

### Wiki Generation Pipeline
Repository → `LocalRepositoryManager` (clone) → `FilesystemIndexer` (filter) → tree-sitter parsers → graph builder + `WikiStorageProtocol` (SQLite/FTS5 + sqlite-vec **or** PostgreSQL/tsvector + pgvector) → topology enrichment + Leiden clustering → `WikiStructurePlanner` (LLM outline) → `OptimizedWikiGenerationAgent` (LangGraph parallel page writer) → SSE events to frontend.

### Wiki Storage Backend
Graph nodes/edges, FTS, vector embeddings, and clustering metadata all sit behind `WikiStorageProtocol` (`backend/app/core/storage/`):
- **`sqlite`** (default) — `UnifiedWikiDB` with FTS5 + sqlite-vec; zero infra
- **`postgres`** — `PostgresWikiStorage` with `tsvector`/`tsquery` + pgvector

Select via `WIKI_STORAGE_BACKEND` env var. Search engines (`WikiSearchEngine`, `ProjectSearchEngine`) are backend-agnostic.

### MCP Server
Embedded at `:8000/mcp` (streamable HTTP) and available as a stdio CLI (`wikis-mcp`). 13 tools wired to services in `app/main.py`:
- Discovery: `search_wikis`, `list_wiki_pages`, `get_wiki_page`, `list_projects`
- Q&A: `ask_codebase`, `ask_project`
- Research: `research_codebase`, `research_project`
- Code mapping: `map_codebase`, `map_project`
- Search/graph: `search_wiki`, `search_project`, `get_page_neighbors`

### LLM Provider Pattern
Add providers in `backend/app/services/llm_factory.py`. LangChain interfaces (`BaseChatModel`, `Embeddings`) are used throughout — all providers are interchangeable. Supported: `openai`, `anthropic`, `custom` (OpenAI-compatible), `ollama`, `gemini`, `bedrock`, `github` (GitHub Models), `copilot`. Installed as optional pip extras.

### Frontend API Client
- `web/src/spa/api/` — Generated TypeScript API client (from `npm run generate:types`)
- `src/middleware.ts` proxies non-SSE `/api/v1/*` → backend (same-origin, no CORS)
- SSE endpoints (`/api/v1/ask`, `/api/v1/research`, `/api/v1/invocations/{id}/stream`) bypass middleware and go through `src/app/api/v1/` route handlers that add no-buffer headers

## Code Conventions

### Python
- snake_case functions/variables, PascalCase classes
- Google-style docstrings on major classes
- LangGraph state uses TypedDict (`WikiState`, `QualityAssessmentState`)
- Routes are thin — delegate to services, services delegate to core modules
- Logger per module: `logger = logging.getLogger(__name__)`

### TypeScript/React
- Functional components with hooks, `.jsx`/`.tsx` extensions
- MUI `sx` prop + theme for styling (no CSS files)
- No Redux — React `useState`/`useEffect` + custom hooks in `hooks/`

## Key Configuration

Both services read from `.env` at the project root (Docker Compose loads it). Copy `.env.example` to `.env` to start. Required at minimum:
- `LLM_API_KEY` + `LLM_PROVIDER` for wiki generation
- `JWT_PRIVATE_KEY` / `JWT_PUBLIC_KEY` — generate with `openssl genrsa` + `openssl rsa -pubout`

Default admin credentials (change immediately): `admin@wikis.dev` / `changeme123`

Swagger UI available at `http://localhost:8000/docs` when backend is running.

