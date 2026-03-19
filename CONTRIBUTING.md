# Contributing to Wikis

Thanks for your interest in contributing! Here's how to get started.

## Development Setup

```bash
git clone https://github.com/arozumenko/wikis.git
cd wikis
cp .env.example .env
# Edit .env — set LLM_PROVIDER and LLM_API_KEY at minimum
```

### Backend (FastAPI)

```bash
cd backend
python3.11 -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
AUTH_ENABLED=false uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

### Web App (Next.js)

```bash
cd web
npm ci
npx prisma migrate dev
npx prisma db seed
npm run dev
```

### Running Tests

```bash
# Backend unit tests
cd backend && pytest tests/unit/ -v

# Backend integration tests
cd backend && AUTH_ENABLED=false pytest tests/ -v

# With coverage
cd backend && pytest tests/ -v --cov=app --cov-report=term-missing
```

## Git Workflow

1. Fork the repository
2. Create a feature branch from `main`: `git checkout -b feature/your-feature`
3. Make your changes
4. Run tests and ensure they pass
5. Commit with a clear message describing _why_, not just _what_
6. Push to your fork and open a Pull Request

### Branch Naming

- `feature/` — new features
- `fix/` — bug fixes
- `docs/` — documentation changes
- `refactor/` — code refactoring

### Commit Messages

Keep them concise — one line summary, optional body for context. Focus on the "why".

## Pull Requests

- Keep PRs focused — one feature or fix per PR
- Include a description of what changed and why
- Add/update tests for new functionality
- Make sure CI passes before requesting review

## Reporting Issues

Use the [GitHub issue templates](https://github.com/arozumenko/wikis/issues/new/choose) to report bugs or request features.

## Code Style

- **Python:** Follow existing patterns — snake_case, Google-style docstrings, Pydantic models
- **TypeScript/React:** Functional components, hooks, MUI styling via `sx` prop
- Run `ruff check` for Python linting

## License

By contributing, you agree that your contributions will be licensed under the [MIT License](LICENSE).
