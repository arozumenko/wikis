"""E2E test configuration and shared constants.

These tests use Playwright MCP for browser automation.
Run manually via Claude Code or automate with the playwright-testing skill.

Prerequisites:
  - App service on :3000 with AUTH_SECRET set (serves both SPA and auth API)
  - Backend on :8000
  - Admin user seeded (admin / changeme123)
"""

import httpx
import pytest

# Service URLs
AUTH_URL = "http://localhost:3000"
BACKEND_URL = "http://localhost:8000"
FRONTEND_URL = "http://localhost:3000"

# Test credentials (seeded by prisma db seed)
ADMIN_USERNAME = "admin"
ADMIN_PASSWORD = "changeme123"

# Test data
TEST_REPO_URL = "https://github.com/octocat/Spoon-Knife"
TEST_REPO_BRANCH = "main"


def _services_reachable() -> bool:
    """Check whether the required services are running."""
    try:
        httpx.get(f"{AUTH_URL}/api/auth/health", timeout=2)
        return True
    except (httpx.ConnectError, httpx.TimeoutException):
        return False


@pytest.fixture(autouse=True)
def skip_if_services_unavailable():
    """Skip all E2E tests when the web app / backend are not running."""
    if not _services_reachable():
        pytest.skip("E2E services not running (need web app on :3000 and backend on :8000)")
