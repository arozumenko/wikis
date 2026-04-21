# Wikis — Agent Reference

AI-powered documentation generator that turns any code repository into a browsable, searchable wiki with architecture diagrams, code explanations, and an AI Q&A assistant. Two-service monorepo: FastAPI backend (Python 3.11) and Next.js 15 web app (auth + React SPA).

> **Quick start:** See `CLAUDE.md` for essential commands. This file is the full reference.

---

## Tech Stack

| Layer | Technology |
|-------|-----------|
| Backend language | Python 3.11 |
| Backend framework | FastAPI + uvicorn |
| LLM orchestration | LangChain, LangGraph, deepagents |
| LLM providers | OpenAI, Anthropic, Gemini, Ollama, AWS Bedrock (optional extras) |
| Embeddings + search | FAISS, BM25, SQLite FTS5, sentence-transformers |
| Code parsing | tree-sitter-language-pack (14+ languages) |
| Backend DB | SQLAlchemy async + aiosqlite (SQLite default) / asyncpg (PostgreSQL) |
| MCP server | FastMCP (embedded HTTP at `:8000/mcp`; standalone stdio CLI) |
| Frontend language | TypeScript 5.4 |
| Frontend framework | Next.js 15 (App Router + React 18 SPA) |
| UI library | MUI v5 (Material UI) |
| Auth | Better-Auth v1.5 (web) + RS256 JWT (cross-service) |
| Frontend DB | Prisma + SQLite (default) / PostgreSQL |
| Package manager | npm (web) — `package-lock.json` is the lockfile |
| Linting | ruff (Python), ESLint + Prettier (TS) |
| CI | GitHub Actions (Docker image build on tag; docs deploy on main push) |
| Infra | Docker Compose (single-host); GHCR images |

---

## Repository Structure

```
wikis/
├── backend/                  ← FastAPI service
│   ├── app/
│   │   ├── main.py           ← App factory + lifespan hooks
│   │   ├── config.py         ← Pydantic Settings (all env vars)
│   │   ├── auth.py           ← JWKS-based JWT validation
│   │   ├── db.py             ← SQLAlchemy engine + session factory
│   │   ├── dependencies.py   ← FastAPI Depends() injection
│   │   ├── events.py         ← SSE event models
│   │   ├── api/
│   │   │   └── routes.py     ← HTTP handlers (thin — delegate to services)
│   │   ├── core/             ← Wiki engine (45+ modules)
│   │   │   ├── agents/       ← LangGraph wiki generation agents
│   │   │   ├── code_graph/   ← NetworkX + SQLite FTS5 code index
│   │   │   ├── deep_research/← Multi-step agentic research engine
│   │   │   ├── parsers/      ← tree-sitter for 14+ languages
│   │   │   ├── repo_providers/← GitHub, GitLab, Bitbucket, Azure DevOps
│   │   │   ├── wiki_structure_planner/ ← LLM-driven outline planner
│   │   │   ├── retrievers.py ← Ensemble retrieval (FAISS + BM25 + reranking)
│   │   │   └── vectorstore.py← FAISS index management
│   │   ├── models/           ← Pydantic request/response/event models
│   │   ├── services/         ← Business logic layer
│   │   │   ├── wiki_service.py
│   │   │   ├── ask_service.py
│   │   │   ├── research_service.py
│   │   │   ├── qa_service.py ← QA Knowledge Flywheel
│   │   │   ├── qa_cache_manager.py
│   │   │   ├── llm_factory.py← LLM + embeddings provider factory
│   │   │   └── wiki_management.py
│   │   └── storage/          ← S3 / local artifact storage abstraction
│   ├── mcp_server/
│   │   └── server.py         ← FastMCP tools (wiki, ask, research)
│   ├── scripts/
│   │   └── export_openapi.py ← Generates openapi.json for type gen
│   └── tests/
│       ├── conftest.py       ← Shared fixtures (auth disabled, SQLite)
│       ├── unit/             ← No external deps
│       ├── integration/      ← FastAPI TestClient + in-memory SQLite
│       └── e2e/              ← Full-stack (requires running services)
├── web/                      ← Next.js service
│   ├── src/
│   │   ├── app/              ← Next.js App Router
│   │   │   ├── (spa)/        ← Catch-all: mounts React SPA (ssr: false)
│   │   │   ├── api/auth/     ← Better-Auth routes + JWKS endpoint
│   │   │   ├── api/v1/       ← SSE proxy route handlers (ask/research/stream)
│   │   │   ├── login/        ← Login page (App Router)
│   │   │   └── logout/       ← Logout page
│   │   ├── spa/              ← React SPA (full client-side)
│   │   │   ├── App.tsx       ← BrowserRouter + ThemeProvider + AuthGuard
│   │   │   ├── api/          ← Generated TS client + SSE handlers
│   │   │   ├── components/   ← 35+ React/MUI components
│   │   │   ├── context/      ← RepoContext
│   │   │   ├── hooks/        ← useAuth, useThemeMode, useCopyToClipboard
│   │   │   ├── pages/        ← Dashboard, WikiViewer, Settings, Generate
│   │   │   └── theme.ts      ← MUI theme factory
│   │   ├── lib/              ← Better-Auth config, JWT issuance
│   │   └── middleware.ts     ← Session guard + API proxy (non-SSE /api/v1/*)
│   ├── prisma/
│   │   ├── schema.prisma     ← SQLite schema (Better-Auth + API keys)
│   │   └── migrations/       ← Migration history
│   ├── next.config.ts        ← Package transpilation, MUI import optimization
│   └── package.json
├── docs/                     ← Next.js static docs site (GitHub Pages)
├── skills/wikis/             ← Project-specific Claude Code skills
├── .github/workflows/
│   ├── build-push.yml        ← Docker images on tag push → GHCR
│   └── docs.yml              ← Docs deploy on main push → GitHub Pages
├── docker-compose.yml        ← Production compose (GHCR images)
├── docker-compose.override.yml
├── .env.example              ← Source of truth for env vars
└── CLAUDE.md                 ← Concise quick-reference (auto-loaded)
```

---

## Services and Ports

| Service | Port | Notes |
|---------|------|-------|
| Web App | 3000 | Next.js — auth, SPA, API proxy |
| Backend | 8000 | FastAPI — wiki engine, Q&A, research |
| MCP Server | :8000/mcp | Embedded in backend; `wikis-mcp` CLI uses stdio |
| Ollama | 11434 | Optional — `docker compose --profile ollama up` |
| PostgreSQL | 5432 | Optional — `docker compose --profile postgres up` |

---

## Build & Run

### Docker (recommended)

```bash
cp .env.example .env        # Edit: set LLM_PROVIDER + LLM_API_KEY
docker compose up -d        # Start backend + web
docker compose up -d --build  # After code changes
docker compose ps           # Health status
```

### Backend (local dev)

```bash
cd backend
python3.11 -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"                 # Core + dev tools
pip install -e ".[all-providers]"       # All LLM providers (optional)

# Run — skip JWT for local dev
AUTH_ENABLED=false uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

### Web App (local dev)

```bash
cd web
npm ci
npx prisma migrate dev        # Apply DB migrations
npx prisma db seed            # Seed admin user (admin@wikis.dev / changeme123)
npm run dev                   # http://localhost:3000
```

### Generate TypeScript types

```bash
cd web
npm run generate:types   # Runs export_openapi.py + openapi-typescript
```

---

## Environment Variables

All vars live in `.env` at project root. Both services read it.

| Variable | Required | Purpose |
|----------|----------|---------|
| `LLM_PROVIDER` | Yes | `openai` \| `anthropic` \| `gemini` \| `ollama` \| `bedrock` \| `github` \| `copilot` \| `custom` |
| `LLM_API_KEY` | Yes | API key for the chosen provider |
| `LLM_MODEL` | Yes | Model name (e.g. `gpt-4o-mini`) |
| `JWT_PRIVATE_KEY` | Yes (prod) | RS256 private key for web → backend auth |
| `JWT_PUBLIC_KEY` | Yes (prod) | RS256 public key validated by backend |
| `EMBEDDING_PROVIDER` | No | Defaults to `LLM_PROVIDER`; required for Anthropic |
| `DATABASE_URL` | No | Empty = SQLite; set for PostgreSQL |
| `AUTH_ENABLED` | No | `false` disables JWT check (local dev only) |
| `STORAGE_BACKEND` | No | `local` (default) or `s3` |

Generate JWT keys:
```bash
openssl genrsa -out private.pem 2048
openssl rsa -in private.pem -pubout -out public.pem
```

---

## Testing

### Backend

```bash
cd backend

# Unit tests — fast, no external deps
pytest tests/unit/ -v

# Integration tests — FastAPI TestClient + in-memory SQLite
pytest tests/integration/ -v

# All tests (auth disabled — no JWT setup needed)
AUTH_ENABLED=false pytest tests/ -v

# With coverage
pytest tests/ -v --cov=app --cov-report=term-missing

# Single file
pytest tests/unit/test_llm_factory.py -v
```

**Test setup:**
- `AUTH_ENABLED=false` is set in `conftest.py` by default — no JWT setup needed
- `DATABASE_URL=""` redirects all DB calls to in-memory SQLite
- MCP session manager is monkey-patched to a no-op in all tests
- Fixtures: `mock_settings`, `mock_storage`, `client` (async httpx), `test_app`
- Test patterns: `async_engine` + `session_factory` fixtures for DB tests; `AsyncMock` for services

### Frontend

No automated frontend tests currently exist. Manual Playwright E2E is the current approach.

```bash
# Playwright (manual / QA sessions)
cd web
npx playwright test
```

**Note:** This is a gap — adding vitest + React Testing Library for component tests is a recommended next step.

---

## Coding Conventions

### Python

- `from __future__ import annotations` in all modules
- snake_case functions/variables, PascalCase classes, UPPER_SNAKE for constants
- Google-style docstrings on public classes and complex functions
- Logger per module: `logger = logging.getLogger(__name__)`
- **Routes are thin** — delegate to services; services delegate to core
- LangGraph state: `TypedDict` (`WikiState`, `QualityAssessmentState`)
- Pydantic v2 models for all request/response schemas
- Custom exceptions in service files (e.g. `WikiAlreadyExistsError`)
- `pathlib.Path` over `os.path`; f-strings over `.format()`
- Ruff for linting + formatting (line length 120, target Python 3.11)

```bash
cd backend
ruff check app/          # Lint
ruff format app/         # Format
```

### TypeScript / React

- Named exports — no default exports
- Functional components with hooks only (no class components)
- MUI `sx` prop + theme for all styling — no CSS files
- No Redux — React `useState` / `useContext` + custom hooks in `hooks/`
- Lazy-load all pages: `lazy(() => import('./pages/...'))`
- Discriminated unions for API response variants
- `npx tsc --noEmit` before committing

```bash
cd web
npm run lint          # ESLint
npm run format:check  # Prettier
npm run lint:fix      # Auto-fix
```

---

## Architecture

### Request Flow

```
Browser → Next.js Web App (:3000)
  ├── /login, /logout        → Next.js App Router (server-rendered)
  ├── /api/auth/*            → Better-Auth (same process, issues RS256 JWTs)
  ├── /api/v1/ask            → App Router route handler (SSE proxy, no-buffer)
  ├── /api/v1/research       → App Router route handler (SSE proxy, no-buffer)
  ├── /api/v1/invocations/*  → App Router route handler (SSE proxy, no-buffer)
  ├── /api/v1/* (REST)       → middleware.ts rewrites → Backend :8000
  └── /*                     → (spa)/ catch-all → React SPA (client only)

Backend (:8000)
  ├── JWT validation (JWKS from web :3000)
  ├── Wiki generation: clone → tree-sitter → CodeGraph+FAISS → LangGraph → SSE
  ├── Q&A / Ask: retrieval (FAISS+BM25+FTS5) → LangChain agent → SSE
  ├── Deep research: multi-step LangGraph research → SSE
  └── /mcp → FastMCP (streamable HTTP, tools wired to services)
```

### Wiki Generation Pipeline

```
repo URL
  → LocalRepositoryManager (clone / pull)
  → FilesystemIndexer (language detection, file filtering)
  → tree-sitter parsers (14+ languages) + code_graph (NetworkX + FTS5)
  → FAISS vector index (sentence-transformers embeddings)
  → WikiStructurePlanner (LLM: generates page outline)
  → OptimizedWikiGenerationAgent (LangGraph: parallel page writer)
  → SSE progress events → SPA GenerationProgress component
  → artifacts stored (local or S3)
```

### Auth Cross-Service JWT

```
User logs in → Better-Auth → issues RS256 JWT (signed with JWT_PRIVATE_KEY)
Browser sends JWT in Authorization header → Next.js middleware proxies to backend
Backend auth.py → fetches JWKS from web :3000/api/auth/jwks → validates JWT
LOCAL DEV: AUTH_ENABLED=false bypasses all JWT validation
```

### LLM Provider Pattern

All providers implement `BaseLanguageModel` (LangChain). Add a new provider in `backend/app/services/llm_factory.py`:
1. Add pip extra to `pyproject.toml`
2. Add `if settings.llm_provider == "myprovider":` branch in `create_llm()` and `create_embeddings()`
3. Install: `pip install -e ".[myprovider]"`

### MCP Server

MCP tools are wired directly to backend services (no HTTP round-trip). Available tools:
- `list_wikis` — list available wikis for the user
- `get_wiki_page` — retrieve a specific wiki page
- `ask` — Q&A against a wiki's knowledge base
- `research` — deep multi-step research

Connect from Claude Code: `http://localhost:8000/mcp`

---

## CI/CD

| Workflow | Trigger | Action |
|---------|---------|--------|
| `build-push.yml` | Push tag `v*` | Builds + pushes Docker images to GHCR (amd64 + arm64) |
| `docs.yml` | Push to `main` (docs/** path) | Builds + deploys Next.js docs to GitHub Pages |

**Gap:** No CI pipeline runs tests on PRs. Adding a test workflow is recommended.

---

## Git Workflow

```bash
git config core.hooksPath .githooks   # Required after clone

# Pre-push hook: blocks if branch is behind origin/main
git fetch origin && git rebase origin/main  # Fix if blocked

# Branch from fresh main
git fetch origin && git checkout -b feat/my-feature origin/main
```

All work via feature branches + PRs — no direct commits to `main`.

---

## Known Gaps & Notes

- **No test CI**: tests don't run on PRs — adding a GitHub Actions test workflow is recommended
- **No frontend tests**: vitest + React Testing Library would cover SPA components  
- **QA cache**: semantic Q&A caching uses FAISS; threshold tunable via `QA_CACHE_SIMILARITY_THRESHOLD`
- **tree-sitter-language-pack pinned at 0.9.1**: version-locked for parser compatibility
- **Swagger UI**: `http://localhost:8000/docs` when backend is running
- **Admin default credentials**: `admin@wikis.dev` / `changeme123` — change immediately
