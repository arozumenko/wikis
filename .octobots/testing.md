# Test Infrastructure

## Framework

| Layer | Tool | Scope |
|-------|------|-------|
| Backend unit | pytest + pytest-asyncio | Pure logic, no external deps |
| Backend integration | pytest + httpx AsyncClient | FastAPI TestClient + in-memory SQLite |
| Backend e2e | pytest (separate conftest) | Requires running services |
| Frontend | None (gap) | No automated frontend tests |
| Browser E2E | Playwright (manual) | Used for QA sessions |

---

## Commands

```bash
# --- Backend ---
cd backend

# Unit tests — fast, isolated
pytest tests/unit/ -v

# Integration — TestClient + in-memory SQLite
pytest tests/integration/ -v

# All backend tests (auth disabled automatically)
AUTH_ENABLED=false pytest tests/ -v

# With coverage report
pytest tests/ -v --cov=app --cov-report=term-missing

# Single test file
pytest tests/unit/test_llm_factory.py -v

# Single test function
pytest tests/unit/test_qa_service.py::test_qa_record_lifecycle -v

# --- Frontend (manual QA) ---
cd web
npx playwright test
```

---

## Structure

```
backend/tests/
├── __init__.py
├── conftest.py            ← Shared fixtures (auth off, SQLite, MCP patch)
├── test_api_models.py     ← Pydantic model validation
├── test_ask_service.py    ← AskService unit tests
├── test_auth.py           ← JWT validation tests
├── test_context_limits.py ← Context window handling
├── test_dependencies.py   ← FastAPI Depends() resolution
├── test_health_check.py   ← Health endpoint
├── test_llm_factory.py    ← LLM + embeddings provider creation
├── test_local_repo.py     ← LocalRepositoryManager
├── test_mcp_server.py     ← MCP tool registration + calls
├── test_openapi.py        ← OpenAPI schema generation
├── test_research_service.py
├── test_routes_errors.py  ← HTTP error response shapes
├── test_sse_streaming.py  ← SSE event format
├── test_storage.py        ← Local + S3 storage
├── test_toolkit_bridge.py ← Hybrid wiki toolkit
├── test_wiki_config.py    ← Wiki configuration models
├── test_wiki_management.py
├── test_wiki_refresh.py
├── test_wiki_service.py
├── test_wiki_state.py     ← LangGraph state transitions
├── unit/
│   ├── test_app_init_qa.py
│   ├── test_ask_response_qa_id.py
│   ├── test_ask_service_qa.py
│   ├── test_config_qa.py
│   ├── test_context_overflow.py
│   ├── test_mcp_qa.py
│   ├── test_qa_api_models.py
│   ├── test_qa_cache_manager.py
│   ├── test_qa_record.py
│   ├── test_qa_service.py    ← Most detailed unit test example
│   └── test_routes_qa.py
├── integration/
│   └── test_e2e_smoke.py
└── e2e/
    ├── conftest.py           ← E2E-specific fixtures (live services)
    ├── test_auth_service.py
    ├── test_error_handling.py
    └── test_happy_path.py
```

---

## Fixtures & Setup

### Global (conftest.py)

```python
# Auth always disabled in tests
os.environ.setdefault("AUTH_ENABLED", "false")
# DB always in-memory SQLite
os.environ["DATABASE_URL"] = ""
```

Key fixtures:
| Fixture | Type | Purpose |
|---------|------|---------|
| `mock_settings` | function | Safe Settings with `llm_provider="openai"`, `auth_enabled=False` |
| `mock_storage` | function | `LocalArtifactStorage` backed by `tmp_path` |
| `test_app` | async | Full FastAPI app with lifespan (services initialized) |
| `client` | async | `AsyncClient` wired to `test_app` via `ASGITransport` |
| `_patch_mcp_session_manager` | autouse | Replaces MCP session manager `run()` with no-op |

### DB Fixtures (unit tests)

```python
@pytest.fixture
async def async_engine():
    engine = create_async_engine("sqlite+aiosqlite:///:memory:")
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    yield engine
    await engine.dispose()

@pytest.fixture
def session_factory(async_engine):
    return async_sessionmaker(async_engine, expire_on_commit=False)
```

### Service Mocks

Services are mocked with `AsyncMock` + `MagicMock`:
```python
mock_cache = AsyncMock()
mock_cache.search = AsyncMock(return_value=([], np.zeros(8, dtype=np.float32)))
mock_cache.add = AsyncMock()
mock_cache.check_needs_rebuild = MagicMock(return_value=False)  # sync — MagicMock, not AsyncMock
```

---

## Patterns Detected

- **Arrange-Act-Assert** structure in all test functions
- **Descriptive names**: `test_expired_token_returns_401`, `test_wiki_already_exists_returns_409`
- **One logical concept per test** — multiple asserts per test are common for related checks
- **async tests**: `asyncio_mode = "auto"` in `pyproject.toml` — all async fixtures/tests work without `@pytest.mark.asyncio`
- **No `time.sleep()`** — async `await` throughout
- **Real in-memory SQLite** for DB tests — not mocked SQLAlchemy
- **Patch at boundary**: external LLM calls mocked, internal service logic tested real

---

## CI Integration

Tests do **not** currently run in CI. Only Docker image builds are automated (on tag push).

**Recommended fix**: add `.github/workflows/test.yml`:
```yaml
on: [push, pull_request]
jobs:
  test:
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

---

## Known Issues

- **No frontend tests**: React SPA has zero automated test coverage
- **E2E tests require live services**: `tests/e2e/` can't run in standard CI without service containers
- **MCP session manager patched globally**: the autouse fixture replaces `StreamableHTTPSessionManager.run()` — be aware if adding MCP-specific tests
