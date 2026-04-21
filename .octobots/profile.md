---
project: wikis
issue-tracker: https://github.com/arozumenko/wikis/issues
default-branch: main
languages: [python, typescript]
---

# Wikis

AI-powered documentation generator. Points at any git repo and generates a browsable wiki with architecture diagrams, code explanations, and an AI Q&A assistant.

## Tech Stack

- **Backend**: Python 3.11, FastAPI, LangChain + LangGraph, FAISS + BM25, tree-sitter
- **Frontend**: TypeScript, Next.js 15, React 18, MUI v5, Better-Auth
- **LLM providers**: OpenAI, Anthropic, Gemini, Ollama, AWS Bedrock
- **Database**: SQLite (default) / PostgreSQL
- **Infra**: Docker Compose, GHCR

## Build & Test

- **Backend dev**: `AUTH_ENABLED=false uvicorn app.main:app --reload` (in `backend/`)
- **Frontend dev**: `npm run dev` (in `web/`)
- **Tests**: `pytest tests/ -v` (in `backend/`)
- **Lint (Python)**: `ruff check app/`
- **Lint (TS)**: `npm run lint` (in `web/`)
- **Docker**: `docker compose up -d`

## Conventions

- Python: thin routes → services → core; `from __future__ import annotations` everywhere
- TypeScript: named exports, MUI sx styling, no Redux, lazy-load all pages
- No direct commits to `main` — all work via feature branches + PRs
- `git config core.hooksPath .githooks` required after clone
- Backend tests: `AUTH_ENABLED=false` in conftest, in-memory SQLite, MCP session patched
