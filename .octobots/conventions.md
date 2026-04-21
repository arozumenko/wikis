# Coding Conventions

Detected from codebase analysis. Descriptive — what IS, not what should be.

---

## Python (backend/)

### Module Header

Every module starts with:
```python
from __future__ import annotations
```
This enables `X | Y` union syntax on Python 3.11.

### Imports

Stdlib → third-party → local. Ruff handles ordering automatically.
Local imports are **absolute** (e.g. `from app.services.wiki_service import WikiService`).
Heavy optional deps (LangChain providers) are lazy-imported inside functions.

### Naming

| Thing | Convention | Example |
|-------|-----------|---------|
| Variables / functions | `snake_case` | `wiki_id`, `get_current_user` |
| Classes | `PascalCase` | `WikiService`, `CodeGraph` |
| Constants | `UPPER_SNAKE` | `NOT_IMPLEMENTED`, `DEFAULT_TIMEOUT` |
| Private helpers | `_prefixed` | `_noop_run`, `_current_user_id` |
| TypedDict states | `PascalCase` + `State` suffix | `WikiState`, `QualityAssessmentState` |

### Code Organization

```
routes.py (thin) → service.py (orchestration) → core/*.py (algorithms)
```

Routes do: parse request, call service, return response. Nothing else.
Services do: business logic, coordinate core modules, raise domain exceptions.
Core modules do: algorithms, parsers, indexes. No FastAPI or HTTP concerns.

### Error Handling

Domain exceptions defined in service files:
```python
class WikiAlreadyExistsError(Exception):
    def __init__(self, wiki_id: str): ...
```
Routes catch domain exceptions and map to HTTP status codes.
No bare `except:`. Specific catches only.

### Logging

```python
logger = logging.getLogger(__name__)   # One per module, always
logger.info("Services initialized: wiki, ask, research")  # Past tense events
logger.error("QA cache disabled — embedding init failed: %s", e)  # % formatting
```

### Docstrings

Google-style on public classes and complex functions. Not on simple getters/setters.
```python
def create_llm(settings: Settings) -> BaseLanguageModel:
    """Create LLM instance from settings.

    Args:
        settings: Application settings with LLM configuration.

    Returns:
        Configured LangChain LLM instance.
    """
```

### Type Hints

On all public function signatures. Pydantic v2 models for API I/O. TypedDict for LangGraph state.

### Async

FastAPI route handlers are `async def`. Service methods that touch DB or make external calls are `async def`. Pure CPU logic (parsers, algorithms) can be sync.

### Ruff Config

Target Python 3.11, line length 120. Key ignores:
- `E501` (line too long — handled by formatter)  
- `B008` (function call in default arg — FastAPI `Depends()` pattern)
- `S` checks disabled in `tests/`
- `E402`, `E741`, `F811` ignored in `app/core/` (conditional imports, algorithm vars)

---

## TypeScript / React (web/src/)

### Component Patterns

All components are functional with hooks. No class components.

```tsx
// Named export always (not default export)
export function WikiCard({ wiki }: { wiki: WikiSummary }) {
  return <Card sx={{ p: 2 }}>...</Card>;
}
```

### Styling

MUI `sx` prop only. No `.css` files, no `styled-components`, no Tailwind.
Theme colors via `theme.palette.*`. Spacing via `theme.spacing()` or numeric `sx` values.

```tsx
<Box sx={{ display: 'flex', gap: 2, alignItems: 'center' }}>
```

### State Management

No Redux. Component state via `useState`. Cross-component state via `useContext` (see `RepoContext`).
Custom hooks in `hooks/` encapsulate reusable stateful logic.

Derived state is computed — not stored in state:
```tsx
// Good
const isLoading = status === 'pending';
// Avoid
const [isLoading, setIsLoading] = useState(false);
```

### Lazy Loading

All pages are lazy-loaded:
```tsx
const DashboardPage = lazy(() =>
  import('./pages/DashboardPage').then((m) => ({ default: m.DashboardPage })),
);
```

### TypeScript Strictness

- Named exports everywhere
- No `any` without a comment explaining why
- Discriminated unions for API response variants
- Generated types in `spa/api/types.generated.ts` — don't edit manually
- Run `npx tsc --noEmit` before committing

### File Naming

| Thing | Convention | Example |
|-------|-----------|---------|
| Components | `PascalCase.tsx` | `WikiCard.tsx` |
| Hooks | `camelCase.ts` starting with `use` | `useThemeMode.ts` |
| Pages | `PascalCase + Page.tsx` | `DashboardPage.tsx` |
| Utilities | `camelCase.ts` | `constants.ts`, `theme.ts` |

### ESLint + Prettier

```bash
npm run lint          # Check
npm run lint:fix      # Auto-fix ESLint
npm run format        # Prettier write
npm run format:check  # Prettier check only
```

---

## Git

- Feature branches from fresh main: `git fetch origin && git checkout -b feat/name origin/main`
- Pre-push hook at `.githooks/` blocks pushes if behind `origin/main`
- Conventional commit style loosely followed (feat, fix, chore, docs)
- Squash merge preferred for PRs

---

## Dependency Pinning

- `tree-sitter-language-pack==0.9.1` — exact pin for parser compatibility
- `better-auth` pinned to `^1.5.5` — API surface changes between minors
- `mcp>=1.0.0` — minimum, not pinned; FastMCP API stabilized at 1.x

---

## TODOs / Technical Debt

```bash
# Count markers in backend
grep -r "TODO\|FIXME\|HACK\|Phase 2" backend/app/ --include="*.py" | wc -l
```

`NOT_IMPLEMENTED = "Not implemented — Phase 2"` string is used as a placeholder in several
route stubs in `api/routes.py`. These indicate planned but not yet built endpoints.
