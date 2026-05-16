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
| Embeddings + search | FAISS, BM25, SQLite FTS5 + sqlite-vec, PostgreSQL tsvector + pgvector, sentence-transformers |
| Code parsing | tree-sitter-language-pack (14+ languages) |
| Backend DB | SQLAlchemy async + aiosqlite (SQLite default) / asyncpg (PostgreSQL) |
| Wiki storage | `WikiStorageProtocol` with two backends: sqlite-vec (`UnifiedWikiDB`, default) or pgvector — selected by `WIKI_STORAGE_BACKEND` |
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
│   │   ├── core/             ← Wiki engine (40+ modules)
│   │   │   ├── agents/       ← LangGraph wiki generation agents (`wiki_graph_optimized`)
│   │   │   ├── code_graph/   ← Graph builder + unified FTS/vector index (SQLite or Postgres)
│   │   │   ├── deep_research/← Multi-step agentic research engine
│   │   │   ├── parsers/      ← tree-sitter for 14+ languages
│   │   │   ├── repo_providers/← GitHub, GitLab, Bitbucket, Azure DevOps
│   │   │   ├── storage/      ← `WikiStorageProtocol` + `sqlite.py` (UnifiedWikiDB) / `postgres.py`
│   │   │   ├── wiki_structure_planner/ ← LLM outline + cluster planners
│   │   │   ├── prompts/      ← LangChain prompt templates
│   │   │   ├── state/        ← LangGraph state (`WikiState`)
│   │   │   ├── multi_retriever.py / unified_retriever.py ← Ensemble retrieval (dense + sparse + reranking)
│   │   │   ├── wiki_search_engine.py / project_search_engine.py ← FTS + graph re-ranking
│   │   │   ├── wiki_page_index.py / wiki_page_search.py ← Wikilink graph + page search adapter
│   │   │   ├── graph_clustering.py / cluster_expansion.py / cluster_planner.py ← Leiden pipeline
│   │   │   ├── graph_topology.py ← Topology enrichment (hubs, density)
│   │   │   └── feature_flags.py ← `WIKIS_CLUSTER_*` toggles
│   │   ├── models/           ← Pydantic request/response/event/search models
│   │   ├── services/         ← Business logic layer
│   │   │   ├── wiki_service.py / wiki_management.py
│   │   │   ├── ask_service.py
│   │   │   ├── research_service.py
│   │   │   ├── qa_service.py / qa_cache_manager.py ← QA Knowledge Flywheel + cache
│   │   │   ├── project_service.py
│   │   │   ├── export_service.py / import_service.py ← Wiki bundle export/import
│   │   │   ├── toolkit_bridge.py ← Bridges core wiki toolkit to services
│   │   │   ├── health_check.py
│   │   │   ├── context_limits.py / context_overflow.py
│   │   │   └── llm_factory.py← LLM + embeddings provider factory
│   │   └── storage/          ← Artifact storage abstraction (local / S3) — distinct from `core/storage/`
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
| `LLM_PROVIDER` | Yes | `openai` \| `anthropic` \| `custom` \| `ollama` \| `gemini` \| `bedrock` \| `github` \| `copilot` |
| `LLM_API_KEY` | Yes | API key for the chosen provider |
| `LLM_MODEL` | Yes | Model name (e.g. `gpt-4o-mini`) |
| `JWT_PRIVATE_KEY` | Yes (prod) | RS256 private key for web → backend auth |
| `JWT_PUBLIC_KEY` | Yes (prod) | RS256 public key validated by backend |
| `EMBEDDING_PROVIDER` | No | Defaults to `LLM_PROVIDER`; required for Anthropic |
| `DATABASE_URL` | No | App DB (users, projects, invocations). Empty = SQLite; set for PostgreSQL |
| `WIKI_STORAGE_BACKEND` | No | `sqlite` (default — UnifiedWikiDB + sqlite-vec) or `postgres` (pgvector) |
| `WIKI_STORAGE_DSN` | No | PostgreSQL DSN for wiki storage when backend = `postgres` |
| `AUTH_ENABLED` | No | `false` disables JWT check (local dev only) |
| `STORAGE_BACKEND` | No | Artifact storage: `local` (default) or `s3` |
| `LLM_MAX_CONCURRENCY` | No | Cap concurrent LLM calls (per process) |
| `WIKIS_CLUSTER_*` | No | Feature flags for the Leiden clustering pipeline (default on) |

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
  → tree-sitter parsers (14+ languages) → graph builder
  → WikiStorageProtocol (UnifiedWikiDB / SQLite + sqlite-vec  OR  PostgreSQL + pgvector)
  → graph topology enrichment (hubs, density) + Leiden clustering (cluster_expansion, coverage_ledger)
  → WikiStructurePlanner (agent planner OR cluster planner — set by PLANNER_TYPE)
  → OptimizedWikiGenerationAgent (LangGraph: parallel page writer)
  → SSE progress events → SPA GenerationProgress component
  → wiki page markdown + WikiPageIndex (wikilink graph) → artifact storage (local or S3)
```

### Non-code file ingestion (#118)

Beyond source code and markdown, the indexer ingests PDFs, images, and plain-text variants via the **DocumentExtractor** registry at `backend/app/core/extractors/`. Each extractor turns one file into a single text blob that flows through the same chunking + embedding + FTS5 pipeline as native markdown.

**Vision-first design**: formats where text extraction loses critical visual context — tables, formulas, layouts, diagrams — are rendered to images and described by the project-configured multimodal LLM (Claude 3+, GPT-4o, Gemini 1.5+). Plain-text variants (`.mdx`, `.rst`, `.adoc`, `.qmd`) are read directly because there's no visual content to lose.

| Extension | Extractor | Method | Extras |
|-----------|-----------|--------|--------|
| `.mdx` `.qmd` `.rst` `.adoc` | `PlainTextExtractor` | UTF-8 read | — |
| `.png` `.jpg` `.jpeg` `.gif` `.webp` | `ImageExtractor` | LLM vision describe | `[vision]` (Pillow) |
| `.pdf` | `PDFExtractor` | pdfium2 render → LLM vision per page | `[pdf]` (pypdfium2 + Pillow) |
| `.docx` `.xlsx` `.pptx` | `OfficeExtractor` | LibreOffice headless → PDF → PDFExtractor | `[office]` + system `libreoffice` |
| `.md` `.txt` `.yaml` `.toml` `…` | _(legacy text-read path)_ | `open(file, 'r')` | — |

**Office system dep**: `OfficeExtractor` shells out to `soffice` (LibreOffice headless). The backend Docker image installs it via apt. Local-dev macOS users: `brew install --cask libreoffice`. Without it, `OfficeExtractor` fails to construct at registry-build time, logs the missing-binary WARNING, and `.docx/.xlsx/.pptx` files fall through to the legacy text-read path (binary garbage in `source_text`).

**Cost handling**: every LLM call logs `[extractors.image]` / `[extractors.pdf]` at INFO with input/output token counts. No env-var gating — operators monitor the log stream. A 500-page scanned manual will spend tokens; grep for `[extractors.pdf]` to spot.

**Adding a new extractor**:
1. Implement the `DocumentExtractor` protocol in `app/core/extractors/`.
2. Register in `build_default_registry()` (lazy import so the dep stays optional).
3. Add the extension to `DOCUMENTATION_EXTENSIONS` (`constants.py`) and the `FilterManager` allowlist (`filter_manager.py`).
4. Add the optional pip extras to `pyproject.toml`.
5. Bottom-line wiring point: `EnhancedUnifiedGraphBuilder._parse_documentation_files` calls `registry.get(extension)` and falls back to the legacy text-read path when no handler is registered.

### Auth Cross-Service JWT

```
User logs in → Better-Auth → issues RS256 JWT (signed with JWT_PRIVATE_KEY)
Browser sends JWT in Authorization header → Next.js middleware proxies to backend
Backend auth.py → fetches JWKS from web :3000/api/auth/jwks → validates JWT
LOCAL DEV: AUTH_ENABLED=false bypasses all JWT validation
```

### Language Parsers (#119 — lightweight tier)

Three tiers of parser coverage at `backend/app/core/parsers/`:

| Tier | Parsers | Path | Approach |
|------|---------|------|----------|
| **Rich (deep)** | C++, C#, Go, Java, Python, JavaScript, TypeScript, Rust | `<lang>_parser.py` / `<lang>_visitor_parser.py` (1500–4000 LOC each) | Per-language type inference, field resolution, template instantiation |
| **Basic (lightweight visitor)** | Ruby, PHP, Kotlin, Scala, Lua, Swift, Dart, PowerShell, Bash, Objective-C, Verilog, Fortran, Julia, Pascal | `basic_visitor.py` + `lang_configs/<lang>.py` (~30 LOC config each) | Generic tree-sitter visitor driven by per-language `LanguageConfig` (node-type strings for class/function/call/import + inheritance field/types + name fallbacks + name-chain drill-down) |
| **Basic (bespoke subclass)** | Elixir, R, Zig, Groovy | `basic_visitor.py` subclass in `lang_configs/_special.py` | Same `BasicVisitorParser` two-pass design, but `_visit_structural` overridden where the grammar's shape doesn't fit pure config (Elixir: `call`-text discrimination for `defmodule`/`def`; R: backward parent-lookup on `binary_operator`; Zig: var-decl-with-struct-RHS pattern; Groovy: nested `command`/`unit`/`block` token-sequence walk) |
| **Regex fallback** | (legacy / unsupported) | `code_splitter.py` | Last-resort extraction for languages without any tree-sitter coverage |

All three tiers produce the same `Symbol` / `Relationship` / `ParseResult` shape so downstream graph builders, retrievers, and the wiki agent need zero special-casing.

**Adding a new lightweight language**:
1. Inspect the tree-sitter grammar — find node-type strings for class/function/import/call/inheritance (a 20-line introspection script is enough).
2. Create `backend/app/core/parsers/lang_configs/<lang>.py` with a `LanguageConfig` instance.
3. Re-export from `lang_configs/__init__.py`'s `build_basic_parsers()` factory.
4. Add a `tests/fixtures/parsers/<lang>/hello.<ext>` fixture exercising class + methods + inheritance + calls.
5. Add a test class to `tests/unit/test_basic_visitor_parser.py` following the existing per-language pattern.

**Promotion path basic → deep**: when a language sees enough user demand to justify type inference / cross-file resolution (Ruby + PHP are the likely first candidates), write a full parser inheriting from `BaseParser` directly. The `LanguageConfig` can either retire or stay as a fallback for partial parses.

### Edge Confidence Levels (#120)

Every edge in `repo_edges` carries a `confidence` label that propagates from the parser's `Relationship.confidence` float through `graph_builder._add_relationships_bulk` (mapping at the storage write boundary: `< 0.7` → `"INFERRED"`, else `"EXTRACTED"`; AMBIGUOUS reserved for future per-target dedup work).

| Label | When it's assigned | Trust signal |
|-------|---------------------|--------------|
| `EXTRACTED` | Parser observed the relationship directly. Deep parsers (Python/Java/Go/C#/JS/TS/Rust/C++) default to `confidence=1.0`; their explicit lookups like Python's `composition (is_instantiation)` at 0.9 also stay EXTRACTED. | High — graph algorithms can rely on these as-is. |
| `INFERRED` | Name-only resolution (no type context, no cross-file disambiguation). Basic-tier visitor parsers (Ruby/PHP/Kotlin/Scala/Lua/Swift/Dart/PowerShell/Bash/Obj-C/Verilog/Fortran/Julia/Pascal/Elixir/R/Zig/Groovy) emit `confidence=0.6` for all CALLS edges. Python parser's "indirect call" tier (0.7) is the boundary case — it stays EXTRACTED because the threshold is `< 0.7`. | Medium — useful for graph topology + first-pass clustering, but UI consumers should disambiguate before quoting in user-facing answers. |
| `AMBIGUOUS` | Reserved. No callsites today; introduced via the schema CHECK constraint for future "multiple plausible targets" handling. | Treat as `INFERRED` for any current downstream code. |

**Where the label surfaces today**:
- **Storage layer** — `repo_edges.confidence` column (TEXT NOT NULL DEFAULT 'EXTRACTED', CHECK constraint on the three values). Both SQLite + Postgres backends.
- **`UnifiedWikiDB.stats()` / `PostgresWikiStorage.stats()`** — returns `confidence_breakdown: {extracted, inferred, ambiguous}`. Stable shape; keys always present even when zero.
- **MCP tool `get_graph_stats(wiki_id)`** — exposes the full `stats()` dict to AI IDE clients.
- **`SourceReference.confidence`** — optional field on citation responses. Default `None`; carries the underlying graph edge's confidence label when the retrieval path surfaced it. Propagated from cached QA records and live agent event streams in `ask_service` + `research_service`.

**Surfaced as of #120 Phase 2** (#157):
- `min_confidence` parameter on `ask_codebase` / `ask_project` MCP tools + `AskRequest` Pydantic model + `AskConfig` dataclass. Threads through to `UnifiedRetriever._get_expansion_neighbors` which drops edges below the threshold during graph expansion. `None` keeps the legacy "include all" behavior; `"EXTRACTED"` is the strictest (only direct parser observations); `"INFERRED"` allows name-only resolution edges too. Case-insensitive; missing edge labels default to EXTRACTED (legacy-row compat).
- `SourceReference.confidence` now actually populates: the strongest-rank edge confidence reaching a candidate flows through the retriever → `Document.metadata["confidence"]` → agent source dict → `SourceReference.confidence`. Seed nodes (FTS/vector hits with no incoming edge) still have `None` because they weren't reached via an edge.

**Not yet surfaced** (#120 Phase 3, tracked separately):
- `search_wiki` MCP tool's `min_confidence` param — the wikilink graph (`WikiPageIndex`) is in-memory and has no confidence column, so the filter would be a no-op there. Deferred until the wikilink graph either gains a confidence dimension or the MCP tool delegates to repo_edges-based search.
- SPA citation chips rendering with confidence indicator — `CitationChips.tsx` / `SourceCitations.tsx` exist but aren't wired into any page; needs the SSE-source-collection-to-render pipeline plumbed end-to-end first.

### LLM Provider Pattern

All providers implement `BaseLanguageModel` (LangChain). Add a new provider in `backend/app/services/llm_factory.py`:
1. Add pip extra to `pyproject.toml`
2. Add `if settings.llm_provider == "myprovider":` branch in `create_llm()` and `create_embeddings()`
3. Install: `pip install -e ".[myprovider]"`

### MCP Server

MCP tools are wired directly to backend services (no HTTP round-trip) in `app/main.py` via `set_services()`. 13 tools across discovery, Q&A, research, code mapping, and search:

| Group | Tool | Purpose |
|-------|------|---------|
| Discovery | `search_wikis` | List/filter wikis the user can access |
| Discovery | `list_wiki_pages` | Page index for a wiki |
| Discovery | `get_wiki_page` | Fetch a wiki page (with offset/limit) |
| Discovery | `list_projects` | List/filter projects |
| Q&A | `ask_codebase` | Q&A against one wiki |
| Q&A | `ask_project` | Q&A across all wikis in a project |
| Research | `research_codebase` | Deep multi-step research over one wiki |
| Research | `research_project` | Deep research across a project |
| Mapping | `map_codebase` | Code map for a wiki (entry points → flow) |
| Mapping | `map_project` | Code map across project entry points |
| Search | `search_wiki` | FTS + wikilink graph re-rank over one wiki |
| Search | `search_project` | FTS + graph search across project wikis |
| Search | `get_page_neighbors` | Wikilink graph neighbors of a page |

Connect from Claude Code: `http://localhost:8000/mcp` (streamable HTTP). Standalone stdio CLI: `wikis-mcp`.

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

---

## Incremental wiki regeneration (#116)

`POST /api/v1/wikis/{wiki_id}/incremental-refresh` takes a freshly-parsed
node payload (same shape as `/diff`) and dispatches affected pages through
three regimes:

| Regime | Trigger | Action | LLM cost |
|---|---|---|---|
| **trivial** | only `MOVED` changes | regex rewrites `<code_context path="...">` | 0 |
| **edit** | only `MODIFIED` changes | surgical LLM patch with quality gate | ~10–30% of full page |
| **structural** | `ADDED`/`DELETED` or primary-symbol-deleted | full single-page regen via agent | full page |
| **deleted** (#141) | all of a page's symbols deleted | drop the page row + artifact | 0 |

Progress streams over SSE: `page_unchanged`, `page_patched`, `page_edited`,
`page_regenerated`, `page_deleted`, `incremental_summary`. The SPA banner
(`IncrementalRefreshBanner.tsx`) renders the regime counts.

### Migration note: first run after deploy

The incremental machinery (PRs #126–#143) introduced two new columns whose
NULL state forces extra work on the first deployment:

- **`repo_nodes.content_hash`** — sha256 of normalized source_text. NULL on
  pre-PR1 rows. The change detector treats `NULL` as "hash unknown, assume
  modified", so the first incremental run after deploy marks every
  pre-existing node as modified. One-time cost; subsequent runs use the
  populated hashes.

- **`repo_nodes.embedding_content_hash`** — snapshot at last embed time.
  NULL on pre-PR4 rows. `populate_embeddings` skips nodes whose
  `embedding_content_hash` equals their `content_hash`; NULL forces re-embed.
  First post-deploy embedding pass re-embeds everything once and stamps the
  column, after which subsequent runs reuse.

### Backfill query (optional, for operators)

To shortcut the first-run re-embed cost on a large wiki, a one-off backfill
sets `content_hash` for any row whose `source_text` is non-empty. The
storage layer auto-hashes on next upsert, so any path that touches each
file (full `/refresh`, incremental refresh, or this query) lands the hash:

```sql
-- SQLite. content_hash := sha256(normalize_for_hash(source_text))
-- python -c "from app.core.storage.incremental import compute_content_hash; …"
-- (run from Python so the normalization rules stay in one place).
```

Operators typically just trigger one `/refresh` after deploy and let the
storage layer's auto-hash populate everything in one pass.
