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
- `main.py` — FastAPI app factory + lifespan hooks
- `config.py` — Pydantic settings (env vars)
- `auth.py` — JWT validation via JWKS from web app
- `api/routes.py` — HTTP route handlers (thin; delegate to services)
- `services/` — Business logic: `wiki_service`, `ask_service`, `research_service`, `llm_factory`
- `core/` — The wiki engine (45+ modules preserved from original plugin):
  - `parsers/` — Tree-sitter for 14+ languages
  - `code_graph/` — NetworkX graph + SQLite FTS5 indexing
  - `agents/` — LangGraph wiki generation agents
  - `deep_research/` — Multi-step research engine
  - `vectorstore.py` — FAISS management
  - `retrievers.py` — Ensemble retrieval (dense + sparse + reranking)
  - `repo_providers/` — GitHub, GitLab, Bitbucket, Azure DevOps
- `models/` — Pydantic request/response/event models
- `storage/` — S3/local storage abstraction

### Auth Cross-Service JWT
The web app issues RS256 JWTs. The backend validates them using `JWT_PUBLIC_KEY`. For local dev without auth: `AUTH_ENABLED=false`.

### Wiki Generation Pipeline
Repository → `LocalRepositoryManager` (clone) → `FilesystemIndexer` (filter) → tree-sitter parsers → `CodeGraph` (NetworkX + FTS5) + FAISS vector index → `WikiStructurePlanner` (LLM outline) → `OptimizedWikiGenerationAgent` (LangGraph parallel page writer) → SSE events to frontend.

### LLM Provider Pattern
Add providers in `backend/app/services/llm_factory.py`. LangChain interfaces (`BaseLanguageModel`, `Embeddings`) are used throughout — all providers are interchangeable. Providers are installed as optional pip extras (`ollama`, `gemini`, `bedrock`, `all-providers`).

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

