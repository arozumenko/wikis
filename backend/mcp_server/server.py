"""Wikis MCP Server — exposes wiki knowledge as tools for AI IDEs.

Can be mounted into FastAPI app or run standalone.
"""

from __future__ import annotations

import contextvars
import json as _json
import logging
import os
from typing import Any

from mcp.server.fastmcp import FastMCP
from starlette.types import ASGIApp, Receive, Scope, Send

logger = logging.getLogger(__name__)

mcp = FastMCP("wikis", streamable_http_path="/")

# Service references — set by mount_mcp() when embedded in FastAPI
_wiki_management = None
_ask_service = None
_research_service = None
_qa_service = None
_storage = None
_settings = None

# Per-request user context (populated by auth middleware)
_current_user_id: contextvars.ContextVar[str | None] = contextvars.ContextVar("_current_user_id", default=None)


def set_services(
    wiki_management: Any,
    ask_service: Any,
    storage: Any,
    settings: Any = None,
    research_service: Any = None,
    qa_service: Any = None,
) -> None:
    """Inject service references for direct calls (no HTTP round-trip)."""
    global _wiki_management, _ask_service, _research_service, _qa_service, _storage, _settings
    _wiki_management = wiki_management
    _ask_service = ask_service
    _research_service = research_service
    _qa_service = qa_service
    _storage = storage
    _settings = settings


# ---------------------------------------------------------------------------
# Auth middleware
# ---------------------------------------------------------------------------


class MCPAuthMiddleware:
    """Raw ASGI middleware — validates Authorization header without buffering the stream.

    BaseHTTPMiddleware breaks FastMCP's anyio task group, so we use the raw ASGI interface.
    We also proxy the inner app's `router` attribute so Starlette's Mount can propagate
    the lifespan event (which initializes FastMCP's session manager task group).
    """

    def __init__(self, app: ASGIApp) -> None:
        self.app = app

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] not in ("http", "websocket"):
            await self.app(scope, receive, send)
            return

        # Auth disabled in dev mode
        if _settings and not getattr(_settings, "auth_enabled", True):
            token = _current_user_id.set("dev-user")
            try:
                await self.app(scope, receive, send)
            finally:
                _current_user_id.reset(token)
            return

        headers = {k.lower(): v for k, v in scope.get("headers", [])}
        auth = headers.get(b"authorization", b"").decode()

        if not auth.startswith("Bearer "):
            await self._send_401(
                send,
                "No API key provided. Add 'Authorization: Bearer wikis_...' to your MCP config headers. "
                "Generate a key in Wikis Settings → API Keys. This server uses API key auth, not OAuth.",
            )
            return

        raw_token = auth.split(" ", 1)[1]
        user_id, error_reason = await _validate_token(raw_token)
        if not user_id:
            await self._send_401(send, error_reason)
            return

        ctx_token = _current_user_id.set(user_id)
        try:
            await self.app(scope, receive, send)
        finally:
            _current_user_id.reset(ctx_token)

    @staticmethod
    async def _send_401(send: Send, message: str) -> None:
        body = _json.dumps({"error": message}).encode()
        await send(
            {
                "type": "http.response.start",
                "status": 401,
                "headers": [
                    (b"content-type", b"application/json"),
                    (b"content-length", str(len(body)).encode()),
                ],
            }
        )
        await send({"type": "http.response.body", "body": body})


_AUTH_ERR_GENERIC = (
    "Invalid or revoked API key. Generate a new one in Wikis Settings → API Keys, "
    "update your MCP config headers, then run /mcp to reconnect. "
    "This server uses API key auth — the re-authenticate OAuth flow does not apply."
)


async def _validate_token(token: str) -> tuple[str | None, str]:
    """Validate a wikis_ PAT and return (user_id, error_reason).

    Returns (user_id, "") on success, or (None, reason) on failure.
    """
    if not _settings:
        return "anon", ""
    try:
        import httpx

        verify_url = _settings.auth_jwks_url.replace("/jwks", "/api-key/verify")
        async with httpx.AsyncClient(timeout=10) as client:
            resp = await client.post(verify_url, json={"key": token})
            if resp.status_code != 200:
                return None, f"API key verification failed (HTTP {resp.status_code}). {_AUTH_ERR_GENERIC}"
            data = resp.json()
            if not data.get("valid", False):
                err = data.get("error", {})
                if err.get("code") == "RATE_LIMITED":
                    retry_ms = err.get("details", {}).get("tryAgainIn", 0)
                    retry_min = max(1, retry_ms // 60000)
                    return None, (
                        f"API key rate limit exceeded ({data.get('error', {}).get('message', 'Rate limited')}). "
                        f"Try again in ~{retry_min} minutes, or increase the rate limit in Wikis Settings → API Keys."
                    )
                return None, _AUTH_ERR_GENERIC
            return data.get("userId") or data.get("user_id") or "unknown", ""
    except Exception as e:
        return None, f"API key verification error: {e}. Is the web app running?"


def get_mcp_app() -> ASGIApp:
    """Return the MCP ASGI app wrapped with auth middleware."""
    return MCPAuthMiddleware(mcp.streamable_http_app())


# ---------------------------------------------------------------------------
# Tools
# ---------------------------------------------------------------------------


@mcp.tool()
async def search_wikis(query: str = "") -> dict[str, Any]:
    """List all indexed wikis; filter by repo name or title. Start here to get wiki_id values.

    Args:
        query: Optional case-insensitive filter against repo URL or title. Empty = return all.
    """
    user_id = _current_user_id.get()
    if _wiki_management:
        result = await _wiki_management.list_wikis(user_id=user_id)
        wikis = [
            {
                "wiki_id": w.wiki_id,
                "title": w.title,
                "repo_url": w.repo_url,
                "branch": w.branch,
                "page_count": w.page_count,
                "status": w.status,
            }
            for w in result.wikis
            if w.status == "complete"
        ]
        if query:
            q = query.lower()
            wikis = [w for w in wikis if q in (w["repo_url"] or "").lower() or q in (w["title"] or "").lower()]
        return {"wikis": wikis, "count": len(wikis)}
    return await _http_search_wikis(query)


@mcp.tool()
async def list_wiki_pages(wiki_id: str) -> dict[str, Any]:
    """Return all page IDs, titles, and sections for a wiki. Use before get_wiki_page or ask_codebase.

    Args:
        wiki_id: Wiki identifier from search_wikis().
    """
    user_id = _current_user_id.get()
    if _wiki_management:
        result = await _wiki_management.list_wikis(user_id=user_id)
        wiki = next((w for w in result.wikis if w.wiki_id == wiki_id), None)
        if not wiki:
            return {"error": f"Wiki not found: {wiki_id}. Use search_wikis() to find available wiki IDs."}
        pages = []
        structure = {}
        if _storage:
            try:
                artifacts = await _storage.list_artifacts("wiki_artifacts", prefix=wiki_id)
                md_files = sorted(a for a in artifacts if a.endswith(".md") and "wiki_pages" in a)
                for path in md_files:
                    parts = path.split("/wiki_pages/")
                    if len(parts) == 2:
                        rel = parts[1]
                        page_id = rel.replace(".md", "")
                        section = page_id.split("/")[0] if "/" in page_id else ""
                        name = page_id.split("/")[-1]
                    else:
                        page_id = path.rsplit("/", 1)[-1].replace(".md", "")
                        section = ""
                        name = page_id
                    pages.append(
                        {
                            "page_id": page_id,
                            "title": name.replace("-", " ").replace("_", " ").title(),
                            "section": section,
                        }
                    )

                # Load structure JSON for ordering/descriptions
                struct_files = [a for a in artifacts if "wiki_structure" in a and a.endswith(".json")]
                if struct_files:
                    import json as _json

                    data = await _storage.download("wiki_artifacts", struct_files[-1])
                    structure = _json.loads(data)
            except Exception:
                pass
        # Strip page_content from sections — structure JSON embeds full markdown per page
        slim_sections = [
            {
                "section_name": s.get("section_name", ""),
                "pages": [p.get("page_name", "") for p in s.get("pages", [])],
            }
            for s in structure.get("sections", [])
        ]
        return {
            "wiki_id": wiki.wiki_id,
            "title": wiki.title,
            "repo_url": wiki.repo_url,
            "branch": wiki.branch,
            "page_count": len(pages),
            "pages": pages,
            "sections": slim_sections,
        }
    return await _http_list_wiki_pages(wiki_id)


@mcp.tool()
async def get_wiki_page(wiki_id: str, page_id: str, offset: int = 0, limit: int = 200) -> dict[str, Any]:
    """Read a wiki page's markdown content, with optional line-based pagination for large pages.

    Args:
        wiki_id: Wiki identifier from search_wikis().
        page_id: Page path from list_wiki_pages() (e.g. "architecture/overview").
        offset: First line to return, 0-indexed. Default 0 (start of page).
        limit: Maximum lines to return. Default 200. Increase or paginate if has_more is true.
    """
    if _storage:
        try:
            name = page_id if page_id.endswith(".md") else f"{page_id}.md"
            # Storage path has extra segments between wiki_id and wiki_pages/
            # (e.g. {wiki_id}/owner--repo--branch/wiki_pages/{page}.md)
            # Search artifacts to find the matching file.
            artifacts = await _storage.list_artifacts("wiki_artifacts", prefix=wiki_id)
            match = next((a for a in artifacts if a.endswith(f"/wiki_pages/{name}") or a.endswith(f"/{name}")), None)
            if not match:
                return {"error": f"Page not found: {page_id}. Use list_wiki_pages('{wiki_id}') to see available pages."}
            data = await _storage.download("wiki_artifacts", match)
            all_lines = data.decode("utf-8").splitlines(keepends=True)
            total = len(all_lines)
            chunk = all_lines[offset : offset + limit]
            return {
                "wiki_id": wiki_id,
                "page_id": page_id,
                "content": "".join(chunk),
                "line_from": offset,
                "line_to": offset + len(chunk),
                "total_lines": total,
                "has_more": offset + len(chunk) < total,
            }
        except FileNotFoundError:
            return {"error": f"Page not found: {page_id}. Use list_wiki_pages('{wiki_id}') to see available pages."}
    return await _http_get_wiki_page(wiki_id, page_id, offset, limit)


@mcp.tool()
async def ask_codebase(wiki_id: str, question: str) -> dict[str, Any]:
    """Ask a natural language question about a codebase; returns an AI answer grounded in the wiki.

    Args:
        wiki_id: Wiki identifier from search_wikis().
        question: Question about the codebase — architecture, implementation details, how-tos.
    """
    if _ask_service:
        try:
            from dataclasses import asdict

            from app.models.api import AskRequest

            request = AskRequest(wiki_id=wiki_id, question=question)
            result = await _ask_service.ask_sync(request)
            # Record QA interaction (direct await — no HTTP lifecycle)
            if result.recording and _qa_service:
                try:
                    await _qa_service.record_interaction(**asdict(result.recording))
                except Exception:
                    logger.error("MCP QA recording failed", exc_info=True)
            return result.response.model_dump()
        except FileNotFoundError:
            return {"error": f"Wiki not found: {wiki_id}. Use search_wikis() to find available wiki IDs."}
        except Exception as e:
            return {"error": str(e)}
    return await _http_ask_codebase(wiki_id, question)


@mcp.tool()
async def research_codebase(wiki_id: str, question: str, research_type: str = "comprehensive") -> dict[str, Any]:
    """Run a deep multi-step research agent on a codebase wiki. Slower but more thorough than ask_codebase.

    Use for complex questions requiring cross-file analysis, architectural deep-dives,
    or when ask_codebase gives an incomplete answer.

    Args:
        wiki_id: Wiki identifier from search_wikis().
        question: The research question — supports complex, multi-part questions.
        research_type: "comprehensive" (default) or "quick".
    """
    if _research_service:
        try:
            from app.models.api import ResearchRequest

            request = ResearchRequest(wiki_id=wiki_id, question=question, research_type=research_type)
            response = await _research_service.research_sync(request)
            return response.model_dump()
        except FileNotFoundError:
            return {"error": f"Wiki not found: {wiki_id}. Use search_wikis() to find available wiki IDs."}
        except Exception as e:
            return {"error": str(e)}
    return await _http_research_codebase(wiki_id, question, research_type)


# --- HTTP fallback for standalone mode ---


async def _http_search_wikis(query: str) -> dict[str, Any]:
    import httpx

    url = os.getenv("WIKIS_BACKEND_URL", "http://localhost:8000")
    async with httpx.AsyncClient(base_url=url, timeout=60) as client:
        resp = await client.get("/api/v1/wikis")
        resp.raise_for_status()
        wikis = resp.json().get("wikis", [])
        if query:
            q = query.lower()
            wikis = [w for w in wikis if q in (w.get("repo_url") or "").lower() or q in (w.get("title") or "").lower()]
        return {"wikis": wikis, "count": len(wikis)}


async def _http_list_wiki_pages(wiki_id: str) -> dict[str, Any]:
    import httpx

    url = os.getenv("WIKIS_BACKEND_URL", "http://localhost:8000")
    async with httpx.AsyncClient(base_url=url, timeout=60) as client:
        resp = await client.get(f"/api/v1/wikis/{wiki_id}")
        if resp.status_code == 404:
            return {"error": f"Wiki not found: {wiki_id}"}
        resp.raise_for_status()
        data = resp.json()
        return {
            "wiki_id": data.get("wiki_id"),
            "title": data.get("title"),
            "repo_url": data.get("repo_url"),
            "branch": data.get("branch"),
            "page_count": data.get("page_count"),
            "pages": [
                {"page_id": p["id"], "title": p["title"], "section": p.get("section", "")}
                for p in data.get("pages", [])
            ],
        }


async def _http_get_wiki_page(wiki_id: str, page_id: str, offset: int = 0, limit: int = 200) -> dict[str, Any]:
    import httpx

    url = os.getenv("WIKIS_BACKEND_URL", "http://localhost:8000")
    async with httpx.AsyncClient(base_url=url, timeout=60) as client:
        resp = await client.get(f"/api/v1/wikis/{wiki_id}/pages/{page_id}")
        if resp.status_code == 404:
            return {"error": f"Page not found: {wiki_id}/{page_id}"}
        resp.raise_for_status()
        data = resp.json()
        all_lines = (data.get("content") or "").splitlines(keepends=True)
        total = len(all_lines)
        chunk = all_lines[offset : offset + limit]
        return {
            **data,
            "content": "".join(chunk),
            "line_from": offset,
            "line_to": offset + len(chunk),
            "total_lines": total,
            "has_more": offset + len(chunk) < total,
        }


async def _http_ask_codebase(wiki_id: str, question: str) -> dict[str, Any]:
    import httpx

    url = os.getenv("WIKIS_BACKEND_URL", "http://localhost:8000")
    async with httpx.AsyncClient(base_url=url, timeout=60) as client:
        resp = await client.post("/api/v1/ask", json={"wiki_id": wiki_id, "question": question})
        if resp.status_code == 404:
            return {"error": f"Wiki not found: {wiki_id}"}
        resp.raise_for_status()
        return resp.json()


async def _http_research_codebase(wiki_id: str, question: str, research_type: str) -> dict[str, Any]:
    import httpx

    url = os.getenv("WIKIS_BACKEND_URL", "http://localhost:8000")
    async with httpx.AsyncClient(base_url=url, timeout=300) as client:
        resp = await client.post(
            "/api/v1/research", json={"wiki_id": wiki_id, "question": question, "research_type": research_type}
        )
        if resp.status_code == 404:
            return {"error": f"Wiki not found: {wiki_id}"}
        resp.raise_for_status()
        return resp.json()


def main():
    """Entry point for standalone wikis-mcp CLI."""
    import sys

    args = sys.argv[1:]
    transport = args[0] if args else "stdio"

    for i, arg in enumerate(args):
        if arg == "--host" and i + 1 < len(args):
            os.environ.setdefault("HOST", args[i + 1])
        elif arg == "--port" and i + 1 < len(args):
            os.environ.setdefault("PORT", args[i + 1])

    mcp.run(transport=transport)


if __name__ == "__main__":
    main()
