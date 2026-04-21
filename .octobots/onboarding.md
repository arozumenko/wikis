# Onboarding — 2026-04-11

Performed by: scout (Kit)

---

## What Was Found

| Item | Detail |
|------|--------|
| Project | Wikis — AI-powered documentation generator |
| Stack | Python 3.11 (FastAPI + LangGraph + FAISS) + TypeScript (Next.js 15 + React 18 + MUI) |
| Stage | Active development — production-quality, deployed via Docker Compose |
| Python files | 344 (backend, excl. .venv) |
| TypeScript files | ~3,400 (web/src, excl. node_modules) |
| Backend test files | 40 (unit + integration + e2e) |
| Frontend test files | 0 (gap) |
| CI | Docker image builds on tag; docs deploy on main push; **no test CI** |
| Open issues | See https://github.com/arozumenko/wikis/issues |
| CLAUDE.md | Existed — 3 drift items corrected |

---

## What Was Generated / Modified

| File | Action | Notes |
|------|--------|-------|
| `CLAUDE.md` | Updated | Fixed 3 drift items (MCP port, middleware proxy, SSE routes) |
| `AGENTS.md` | Created | Full team reference (stack, commands, architecture, conventions, testing, CI) |
| `.octobots/profile.md` | Created | Quick-reference project card |
| `.octobots/architecture.md` | Created | Request flow, data flow, API boundaries, design decisions |
| `.octobots/conventions.md` | Created | Python + TypeScript conventions detected from codebase |
| `.octobots/testing.md` | Created | Test infrastructure, fixtures, patterns, gaps |
| `.octobots/roles-manifest.yaml` | Created | Role configuration for all 7 roles |
| `.octobots/memory/project-manager.md` | Created | PM context seeded |
| `.octobots/memory/python-dev.md` | Created | Backend dev context seeded |
| `.octobots/memory/js-dev.md` | Created | Frontend dev context seeded |
| `.octobots/memory/ba.md` | Created | BA context seeded |
| `.octobots/memory/tech-lead.md` | Created | Tech lead context seeded |
| `.octobots/memory/qa-engineer.md` | Created | QA context seeded |
| `.octobots/memory/scout.md` | Created | Scout memory with exploration shortcuts |

---

## Role Configuration

All 7 default roles adopted without repurposing — the web app stack fits the default team perfectly:

| Role | Persona | Focus |
|------|---------|-------|
| project-manager | Max | Wiki pipeline features, issue triage, milestone coordination |
| python-dev | Py | FastAPI, LangGraph agents, FAISS, tree-sitter, MCP server |
| js-dev | Jay | Next.js, React SPA, MUI, Better-Auth, SSE stream handling |
| ba | Alex | Wiki feature specs, Q&A requirements, Phase 2 stubs |
| tech-lead | Rio | Architecture, cross-service JWT, LLM providers, PR review |
| qa-engineer | Sage | pytest (40 tests), Playwright E2E, coverage gaps |
| scout | Kit | Codebase exploration |

---

## Infrastructure Changes

No octobots infrastructure scripts (roles.py, supervisor.py, telegram-bridge.py) found in this project. Phases 8–9 (infrastructure check + spawn readiness) skipped.

---

## CLAUDE.md Drift Corrected

1. **MCP port**: was `8080` → corrected to `:8000/mcp` (embedded in backend) / stdio (standalone CLI)
2. **next.config.ts**: was described as "Backend API rewrites" → corrected to describe MUI/package config; middleware.ts handles the proxy
3. **SSE proxy**: Frontend API Client section now correctly documents the SSE path exception (App Router route handlers bypass middleware)

---

## Open Items / Gaps

| Priority | Item |
|----------|------|
| High | No test CI — tests don't run on PRs (`add .github/workflows/test.yml`) |
| High | Zero frontend tests — React SPA (35+ components) has no automated coverage |
| Medium | Several `NOT_IMPLEMENTED` ("Phase 2") stubs in `backend/app/api/routes.py` |
| Medium | E2E tests (`backend/tests/e2e/`) require live services — not runnable in standard CI |
| Low | Swagger UI only accessible locally — no link in README |

---

## First Recommended Task

**Add a CI test workflow** — create `.github/workflows/test.yml` that runs `pytest tests/ -v` on every PR.

This is the single highest-impact improvement: the codebase has 40 backend tests but they never run automatically. Every PR is merged blind. Adding test CI costs ~30 minutes of dev time and prevents regressions immediately.

Template:
```yaml
name: Test
on: [push, pull_request]
jobs:
  backend-tests:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with: { python-version: '3.11' }
      - run: pip install -e ".[dev]"
        working-directory: backend
      - run: AUTH_ENABLED=false pytest tests/ -v --cov=app
        working-directory: backend
```
