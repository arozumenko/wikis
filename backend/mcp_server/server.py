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
from mcp.server.transport_security import TransportSecuritySettings
from starlette.types import ASGIApp, Receive, Scope, Send

logger = logging.getLogger(__name__)

# Disable MCP SDK's built-in DNS rebinding protection — the endpoint is mounted
# behind FastAPI (which handles CORS) and must accept requests from any Host
# (e.g. host.docker.internal, custom domains, reverse proxies).
mcp = FastMCP(
    "wikis",
    streamable_http_path="/",
    transport_security=TransportSecuritySettings(enable_dns_rebinding_protection=False),
)

# Service references — set by mount_mcp() when embedded in FastAPI
_wiki_management = None
_ask_service = None
_research_service = None
_qa_service = None
_storage = None
_settings = None
_session_factory = None  # async_sessionmaker[AsyncSession] for project DB queries
_page_index_cache = None  # WikiPageIndexCache for search tools

# Per-request user context (populated by auth middleware)
_current_user_id: contextvars.ContextVar[str | None] = contextvars.ContextVar("_current_user_id", default=None)


def set_services(
    wiki_management: Any,
    ask_service: Any,
    storage: Any,
    settings: Any = None,
    research_service: Any = None,
    qa_service: Any = None,
    session_factory: Any = None,
    page_index_cache: Any = None,
) -> None:
    """Inject service references for direct calls (no HTTP round-trip)."""
    global _wiki_management, _ask_service, _research_service, _qa_service, _storage, _settings, _session_factory, _page_index_cache
    _wiki_management = wiki_management
    _ask_service = ask_service
    _research_service = research_service
    _qa_service = qa_service
    _storage = storage
    _settings = settings
    _session_factory = session_factory
    _page_index_cache = page_index_cache


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
        # Access control — verify the caller owns or can see this wiki.
        user_id = _current_user_id.get()
        if _wiki_management:
            result_list = await _wiki_management.list_wikis(user_id=user_id)
            wiki_obj = next((w for w in result_list.wikis if w.wiki_id == wiki_id), None)
            if wiki_obj is None:
                return {"error": f"Wiki not found: {wiki_id}. Use search_wikis() to find available wiki IDs."}

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
async def ask_codebase(
    wiki_id: str,
    question: str,
    min_confidence: str | None = None,
) -> dict[str, Any]:
    """Ask a natural language question about a codebase; returns an AI answer grounded in the wiki.

    Args:
        wiki_id: Wiki identifier from search_wikis().
        question: Question about the codebase — architecture, implementation details, how-tos.
        min_confidence: Optional edge-confidence floor for graph expansion (#120/#157).
            ``None`` (default) includes all edges. ``"EXTRACTED"`` keeps only direct
            parser observations; ``"INFERRED"`` includes name-only resolution edges
            (basic-tier languages like Ruby, PHP, Kotlin). Use this when answer
            precision matters more than recall — INFERRED edges from lightweight
            parsers can cite wrong call targets when symbol names collide.
    """
    if _ask_service:
        try:
            from dataclasses import asdict

            from app.models.api import AskRequest

            request = AskRequest(
                wiki_id=wiki_id, question=question,
                min_confidence=min_confidence,
            )
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


@mcp.tool()
async def map_codebase(wiki_id: str, question: str) -> dict[str, Any]:
    """Generate a hierarchical call-tree (code map) showing which files and functions are involved in a feature or flow.

    Returns symbol-level call stacks — each step names the function, its file, and what it does.
    Best for tracing request flows, understanding how a feature is wired end-to-end,
    or answering "what calls what?" questions.

    Use this instead of research_codebase when you need a structured call-graph view
    rather than a prose answer.

    Args:
        wiki_id: Wiki identifier from search_wikis().
        question: The flow or feature to map — e.g. "how does authentication work?" or "trace a wiki generation request".
    """
    if _research_service:
        try:
            from app.models.api import ResearchRequest

            request = ResearchRequest(wiki_id=wiki_id, question=question, research_type="codemap")
            response = await _research_service.codemap_sync(request)
            return response.model_dump()
        except FileNotFoundError:
            return {"error": f"Wiki not found: {wiki_id}. Use search_wikis() to find available wiki IDs."}
        except Exception as e:
            return {"error": str(e)}
    return await _http_research_codebase(wiki_id, question, "codemap")


@mcp.tool()
async def list_projects(query: str = "") -> dict[str, Any]:
    """List all projects the current user owns or has access to.

    Returns: list of {project_id, name, description, visibility, wiki_count}
    Filter by name with optional query string.

    Args:
        query: Optional case-insensitive prefix filter on project name. Empty = return all.
    """
    user_id = _current_user_id.get()
    if not _session_factory:
        return {"error": "Project service unavailable — session factory not configured."}
    try:
        from app.services.project_service import ProjectService

        async with _session_factory() as session:
            project_service = ProjectService(session)
            projects = await project_service.list_projects(user_id=user_id)
            if query:
                q = query.lower()
                projects = [p for p in projects if p.name.lower().startswith(q)]
            project_ids = [p.id for p in projects]
            wiki_counts = await project_service.batch_get_wiki_counts(project_ids)
            result = [
                {
                    "project_id": p.id,
                    "name": p.name,
                    "description": p.description,
                    "visibility": p.visibility,
                    "wiki_count": wiki_counts.get(p.id, 0),
                }
                for p in projects
            ]
            return {"projects": result, "total": len(result)}
    except Exception as e:
        return {"error": str(e)}


@mcp.tool()
async def ask_project(
    project_id: str,
    question: str,
    min_confidence: str | None = None,
) -> dict[str, Any]:
    """Ask a question across all wikis in a project.

    Returns a cross-repo answer with source attribution per wiki.

    Args:
        project_id: Project identifier from list_projects().
        question: Question about the codebase — architecture, implementation details, how-tos.
        min_confidence: Optional edge-confidence floor for graph expansion (#120/#157).
            See ``ask_codebase`` for the same parameter semantics. The filter
            applies independently to each wiki's retriever in the project fan-out.
    """
    user_id = _current_user_id.get()
    if _ask_service:
        try:
            from dataclasses import asdict

            from app.models.api import AskRequest

            request = AskRequest(
                project_id=project_id, question=question,
                min_confidence=min_confidence,
            )
            result = await _ask_service.ask_sync(request, user_id=user_id)
            if result.recording and _qa_service:
                try:
                    await _qa_service.record_interaction(**asdict(result.recording))
                except Exception:
                    logger.error("MCP QA recording failed for ask_project", exc_info=True)
            return result.response.model_dump()
        except FileNotFoundError:
            return {"error": f"Project not found: {project_id}. Use list_projects() to find available project IDs."}
        except Exception as e:
            return {"error": str(e)}
    return {"error": "Ask service unavailable in standalone mode. Use ask_codebase() per wiki instead."}


@mcp.tool()
async def research_project(project_id: str, question: str) -> dict[str, Any]:
    """Run deep multi-step research across all wikis in a project.

    Slower but more thorough than ask_project. Use for complex cross-repo questions
    requiring multi-file analysis or architectural deep-dives.

    Args:
        project_id: Project identifier from list_projects().
        question: The research question — supports complex, multi-part questions.
    """
    user_id = _current_user_id.get()
    if _research_service:
        try:
            from app.models.api import ResearchRequest

            request = ResearchRequest(project_id=project_id, question=question)
            response = await _research_service.research_sync(request, user_id=user_id)
            return response.model_dump()
        except FileNotFoundError:
            return {"error": f"Project not found: {project_id}. Use list_projects() to find available project IDs."}
        except Exception as e:
            return {"error": str(e)}
    return {"error": "Research service unavailable in standalone mode. Use research_codebase() per wiki instead."}


@mcp.tool()
async def map_project(project_id: str, entry_points: list[str]) -> dict[str, Any]:
    """Build a cross-repo call graph / source map for the given entry points across all wikis in a project.

    Returns symbol-level call stacks showing which files and functions are involved,
    spanning multiple repos in the project.

    Args:
        project_id: Project identifier from list_projects().
        entry_points: List of entry point descriptions or file paths to trace
                      (e.g. ["main.py", "authenticate user flow"]).
    """
    user_id = _current_user_id.get()
    if _research_service:
        try:
            from app.models.api import ResearchRequest

            question = "; ".join(entry_points) if entry_points else "map all entry points"
            request = ResearchRequest(project_id=project_id, question=question, research_type="codemap")
            response = await _research_service.codemap_sync(request, user_id=user_id)
            return response.model_dump()
        except FileNotFoundError:
            return {"error": f"Project not found: {project_id}. Use list_projects() to find available project IDs."}
        except Exception as e:
            return {"error": str(e)}
    return {"error": "Research service unavailable in standalone mode. Use map_codebase() per wiki instead."}


@mcp.tool()
async def search_wiki(
    wiki_id: str,
    query: str,
    hop_depth: int = 1,
    top_k: int = 10,
) -> dict[str, Any]:
    """Search a wiki using graph-aware FTS5 search with wikilink expansion.

    Args:
        wiki_id: Wiki identifier from search_wikis().
        query: Search query string.
        hop_depth: Graph expansion depth (1-5, default 1).
        top_k: Maximum results to return (default 10).

    Returns:
        Dict with query, results (list of page matches with snippets and neighbors),
        and wiki_summary (per-wiki statistics).
    """
    if not 1 <= hop_depth <= 5:
        raise ValueError(f"hop_depth must be between 1 and 5, got {hop_depth}")

    if _page_index_cache and _session_factory:
        from app.core.wiki_search_engine import WikiSearchEngine

        user_id = _current_user_id.get()
        wiki = None
        if _wiki_management:
            result_list = await _wiki_management.list_wikis(user_id=user_id)
            wiki = next((w for w in result_list.wikis if w.wiki_id == wiki_id), None)

        if wiki is None:
            return {"error": f"Wiki not found: {wiki_id}. Use search_wikis() to find available wiki IDs."}

        wiki_name = getattr(wiki, "title", wiki_id) or wiki_id

        page_index = await _page_index_cache.get(wiki_id)

        # Lazy backfill: ensure old wikis have pages indexed for FTS.
        if _wiki_management:
            await _wiki_management.ensure_pages_indexed(wiki_id)

        engine = WikiSearchEngine(wiki_id, wiki_name, _session_factory, page_index)
        result = await engine.search(query, hop_depth=hop_depth, top_k=top_k)

        return {
            "query": result.query,
            "results": [
                {
                    "wiki_id": item.wiki_id,
                    "wiki_name": item.wiki_name,
                    "page_title": item.page_title,
                    "snippet": item.snippet,
                    "score": item.score,
                    "neighbors": [{"title": n.title, "rel": n.rel} for n in item.neighbors],
                }
                for item in result.results
            ],
            "wiki_summary": [
                {
                    "wiki_id": s.wiki_id,
                    "wiki_name": s.wiki_name,
                    "match_count": s.match_count,
                    "relevance": s.relevance,
                }
                for s in result.wiki_summary
            ],
        }
    return await _http_search_wiki(wiki_id, query, hop_depth, top_k)


# ── Shared storage-open helper for graph-native MCP tools ────────────
#
# Resolves ``wiki_id`` → an open read-only ``WikiStorageProtocol``
# handle, applying the same auth + cache-key derivation used by every
# graph tool (``get_graph_stats``, ``god_nodes``, ``shortest_path``,
# ``get_community``). Returns an error string instead of raising so
# tools can forward it verbatim in their JSON response.


async def _open_wiki_storage(wiki_id: str) -> tuple[Any, str | None]:
    """Resolve ``wiki_id`` → open read-only storage handle.

    Returns ``(storage, None)`` on success or ``(None, error_message)``
    when the wiki is unknown / unowned / has no on-disk index.
    """
    if not _wiki_management or not _settings:
        return None, "graph tools unavailable in this deployment"

    user_id = _current_user_id.get()
    wiki_record = await _wiki_management.get_wiki(wiki_id, user_id=user_id)
    if wiki_record is None:
        return None, (
            f"Wiki not found: {wiki_id}. Use search_wikis() to find "
            f"available wiki IDs."
        )

    from pathlib import Path

    from app.core.storage import open_storage
    from app.services.wiki_management import derive_cache_key

    cache_key = derive_cache_key(
        _settings.cache_dir, wiki_record.repo_url, wiki_record.branch,
    )
    if not cache_key:
        return None, f"Wiki {wiki_id} has no cached index."
    db_path = Path(_settings.cache_dir) / f"{cache_key}.wiki.db"
    if not db_path.exists():
        return None, f"Wiki {wiki_id} has no unified DB on disk."

    storage = open_storage(repo_id=cache_key, db_path=str(db_path), readonly=True)
    return storage, None


def _close_storage_quietly(storage: Any) -> None:
    if storage is None:
        return
    try:
        storage.close()
    except Exception:  # noqa: S110 — best-effort close
        pass


@mcp.tool()
async def get_graph_stats(wiki_id: str) -> dict[str, Any]:
    """Return summary statistics for a wiki's underlying code graph (#120).

    Surfaces the ``confidence_breakdown`` over ``repo_edges`` so AI IDE
    clients can decide whether to trust the graph's relationship edges
    (``EXTRACTED`` = explicit parser observation, ``INFERRED`` =
    name-only resolution, ``AMBIGUOUS`` = multiple candidates) before
    asking follow-up questions.

    Args:
        wiki_id: Wiki identifier from search_wikis().

    Returns:
        Dict with ``node_count``, ``edge_count``, ``confidence_breakdown``
        (``{extracted, inferred, ambiguous}``), ``edge_types`` (rel_type
        → count), ``symbol_types``, ``languages``, ``macro_clusters``,
        ``hub_count``, and ``embedding_dim``. Shape is stable — keys
        always present even when counts are zero.
    """
    storage, err = await _open_wiki_storage(wiki_id)
    if err:
        return {"error": err}
    try:
        return storage.stats()
    finally:
        _close_storage_quietly(storage)


# ── Graph-native MCP surface (#121 Phase 1) ──────────────────────────
#
# Three tools that answer structural graph questions deterministically
# from the unified DB — no LLM, no retrieval. AI IDE clients use these
# for "what calls X?", "shortest dependency chain between X and Y", and
# "all nodes in cluster N" without paying retrieval cost. All three
# share ``_open_wiki_storage`` so auth + cache-key lookup is
# consistent with ``get_graph_stats``.


@mcp.tool()
async def god_nodes(wiki_id: str, top_n: int = 20) -> dict[str, Any]:
    """Return the most-connected symbols and files in the wiki's graph.

    Surfaces ``compute_god_nodes`` from the storage layer (§11.7 / B2)
    so AI IDE clients can spot architectural hot-spots — classes
    everything depends on, files with high cross-file fan-out — without
    paying for retrieval or LLM cost.

    Only architectural symbols (``is_architectural = 1``) are ranked by
    degree; files are ranked by their cross-file outgoing edges.

    Args:
        wiki_id: Wiki identifier from ``search_wikis()``.
        top_n: Number of top symbols / files to return (default 20,
            max 200).

    Returns:
        Dict with ``by_symbol_type`` (top symbols by total degree) and
        ``by_file`` (top files by external edges). Shape mirrors the
        storage protocol's ``compute_god_nodes``. Returns
        ``{"error": ...}`` when the wiki is unknown or has no index.
    """
    if not 1 <= top_n <= 200:
        return {"error": f"top_n must be between 1 and 200, got {top_n}"}

    storage, err = await _open_wiki_storage(wiki_id)
    if err:
        return {"error": err}
    try:
        return storage.compute_god_nodes(top_n=top_n)
    finally:
        _close_storage_quietly(storage)


@mcp.tool()
async def shortest_path(
    wiki_id: str,
    source_label: str,
    target_label: str,
    max_depth: int = 25,
) -> dict[str, Any]:
    """Undirected shortest path between two symbols in the graph (#121).

    Resolves each label by ``symbol_name`` first then ``rel_path``
    (deterministic, first match by ``node_id`` ASC). Treats edges as
    undirected and returns the shortest hop-count path. Edge metadata
    (``rel_type``, ``confidence``) is included for each hop. When
    multiple edges exist between two consecutive nodes, the EXTRACTED
    edge is preferred over INFERRED / AMBIGUOUS.

    ``max_depth`` defaults to 25 (graph diameter on production
    codebases is typically <20). Tighten it on dense graphs for
    faster responses; loosen it only when you know the path is long.

    Args:
        wiki_id: Wiki identifier from ``search_wikis()``.
        source_label: Symbol name or file path of the starting node.
        target_label: Symbol name or file path of the destination.
        max_depth: Maximum hops to search (1-50, default 25).

    Returns:
        On success::

            {
                "source": {"node_id", "symbol_name", "rel_path", "symbol_type"},
                "target": {<same>},
                "path": [<node>, ...],   # source first, target last
                "edges": [{"source_id", "target_id", "rel_type", "confidence"}, ...],
                "length": int,
            }

        On failure ``{"path": None, "reason": "..."}`` where reason
        is one of ``source_not_found``, ``target_not_found``,
        ``no_path_within_max_depth``, ``same_node``,
        ``invalid_max_depth``.
    """
    if not 1 <= max_depth <= 50:
        return {"path": None, "reason": "invalid_max_depth"}

    storage, err = await _open_wiki_storage(wiki_id)
    if err:
        return {"error": err}
    try:
        return storage.shortest_path(
            source_label=source_label,
            target_label=target_label,
            max_depth=max_depth,
        )
    finally:
        _close_storage_quietly(storage)


@mcp.tool()
async def get_community(
    wiki_id: str,
    macro_cluster: int,
    micro_cluster: int | None = None,
    limit: int = 500,
) -> dict[str, Any]:
    """All nodes in a Leiden macro (optionally micro) community.

    Communities are precomputed during graph build (``macro_cluster``
    and ``micro_cluster`` columns on ``repo_nodes``). Pass only
    ``macro_cluster`` to get every node in the macro community; add
    ``micro_cluster`` to drill into a sub-cluster.

    Args:
        wiki_id: Wiki identifier from ``search_wikis()``.
        macro_cluster: Macro cluster ID (integer from ``stats()``
            ``macro_clusters`` or from a node's ``macro_cluster``).
        micro_cluster: Optional sub-cluster within ``macro_cluster``.
        limit: Maximum nodes to return (1-2000, default 500).

    Returns:
        Dict with ``macro_cluster``, ``micro_cluster`` (echoed),
        ``count`` (returned), and ``nodes`` (list of node rows with
        ``node_id``, ``symbol_name``, ``symbol_type``, ``rel_path``,
        and other ``repo_nodes`` columns).
    """
    if not 1 <= limit <= 2000:
        return {"error": f"limit must be between 1 and 2000, got {limit}"}

    storage, err = await _open_wiki_storage(wiki_id)
    if err:
        return {"error": err}
    try:
        nodes = storage.get_nodes_by_cluster(
            macro=macro_cluster,
            micro=micro_cluster,
            limit=limit,
        )
        return {
            "macro_cluster": macro_cluster,
            "micro_cluster": micro_cluster,
            "count": len(nodes),
            "nodes": nodes,
        }
    finally:
        _close_storage_quietly(storage)


@mcp.tool()
async def surprising_connections(
    wiki_id: str,
    top_n: int = 10,
    context_depth: int = 1,
    sample_edges_per_pair: int = 3,
) -> dict[str, Any]:
    """Find the most surprising cross-cluster edges in the graph (#121).

    A connection between two macro communities is "surprising" when
    their **contexts** — the set of top-level folder prefixes
    containing their nodes — are highly disjoint. Quantified as
    Jaccard distance over the prefix sets. ``frontend/`` linking to
    ``backend/`` is more surprising than two ``backend/`` clusters
    linking to each other.

    Useful for AI IDE clients to surface "this edge crosses
    architectural boundaries — worth a closer look" without paying
    LLM cost or retrieval cost.

    Args:
        wiki_id: Wiki identifier from ``search_wikis()``.
        top_n: Maximum cluster-pairs to return (1-100, default 10).
        context_depth: How many leading ``rel_path`` segments form
            the cluster context (1-5, default 1 = top-level folder).
        sample_edges_per_pair: Example edges to include per pair
            (1-10, default 3).

    Returns:
        Dict with ``pairs`` (list, sorted by ``jaccard_distance``
        descending). Each pair has ``cluster_a`` / ``cluster_b``
        (always ``cluster_a < cluster_b``), ``jaccard_distance``
        (0.0 - 1.0), ``context_a`` / ``context_b`` (sorted prefix
        lists), ``edge_count``, ``sample_edges``. Empty list when
        no cross-cluster edges exist.
    """
    if not 1 <= top_n <= 100:
        return {"error": f"top_n must be between 1 and 100, got {top_n}"}
    if not 1 <= context_depth <= 5:
        return {
            "error": (
                f"context_depth must be between 1 and 5, "
                f"got {context_depth}"
            ),
        }
    if not 1 <= sample_edges_per_pair <= 10:
        return {
            "error": (
                f"sample_edges_per_pair must be between 1 and 10, "
                f"got {sample_edges_per_pair}"
            ),
        }

    storage, err = await _open_wiki_storage(wiki_id)
    if err:
        return {"error": err}
    try:
        return storage.compute_surprising_connections(
            top_n=top_n,
            context_depth=context_depth,
            sample_edges_per_pair=sample_edges_per_pair,
        )
    finally:
        _close_storage_quietly(storage)


@mcp.tool()
async def get_page_neighbors(
    wiki_id: str,
    page_title: str,
    hop_depth: int = 1,
) -> dict[str, Any]:
    """Get the wikilink graph neighbors of a wiki page.

    Args:
        wiki_id: Wiki identifier from search_wikis().
        page_title: Exact page title as returned by list_wiki_pages() or search_wiki().
        hop_depth: Expansion depth (1-5, default 1).

    Returns:
        Dict with wiki_id, page_title, links_to (pages this page links to),
        and linked_from (pages that link to this page).
    """
    if not 1 <= hop_depth <= 5:
        raise ValueError(f"hop_depth must be between 1 and 5, got {hop_depth}")

    if _page_index_cache:
        # Access control — verify the caller owns or can see this wiki.
        user_id = _current_user_id.get()
        if _wiki_management:
            result_list = await _wiki_management.list_wikis(user_id=user_id)
            wiki = next((w for w in result_list.wikis if w.wiki_id == wiki_id), None)
            if wiki is None:
                return {"error": f"Wiki not found: {wiki_id}. Use search_wikis() to find available wiki IDs."}

        page_index = await _page_index_cache.get(wiki_id)
        if page_title not in page_index.pages:
            raise ValueError(f"Page not found: {page_title}. Use search_wiki('{wiki_id}', ...) to find available pages.")

        forward_metas = page_index.neighbors(page_title, hop_depth=hop_depth)
        links_to = [m.title for m in forward_metas]
        linked_from = page_index.backlinks(page_title)

        return {
            "wiki_id": wiki_id,
            "page_title": page_title,
            "links_to": links_to,
            "linked_from": linked_from,
        }
    return await _http_get_page_neighbors(wiki_id, page_title, hop_depth)


@mcp.tool()
async def search_project(
    project_id: str,
    query: str,
    hop_depth: int = 1,
    top_k: int = 10,
) -> dict[str, Any]:
    """Search across all wikis in a project using graph-aware FTS5 with wikilink expansion.

    Args:
        project_id: Project identifier from list_projects().
        query: Search query string.
        hop_depth: Graph expansion depth (1-5, default 1).
        top_k: Maximum results to return across all wikis (default 10).

    Returns:
        Dict with query, results (merged and ranked from all wikis), and wiki_summary.
    """
    if not 1 <= hop_depth <= 5:
        raise ValueError(f"hop_depth must be between 1 and 5, got {hop_depth}")

    if _page_index_cache and _session_factory:
        from app.core.project_search_engine import ProjectSearchEngine
        from app.core.wiki_search_engine import WikiSearchEngine
        from app.services.project_service import ProjectService

        user_id = _current_user_id.get()
        async with _session_factory() as session:
            project_svc = ProjectService(session)
            wikis = await project_svc.list_project_wikis(project_id, user_id=user_id)

        if wikis is None:
            return {"error": f"Project not found: {project_id}. Use list_projects() to find available project IDs."}

        wiki_tuples = [(w.id, getattr(w, "title", w.id) or w.id) for w in wikis]

        # Pre-load page indexes and lazy-backfill FTS for old wikis.
        page_indexes: dict[str, Any] = {}
        for wid, _ in wiki_tuples:
            page_indexes[wid] = await _page_index_cache.get(wid)
            if _wiki_management:
                await _wiki_management.ensure_pages_indexed(wid)

        def _wiki_engine_factory(wid: str, wiki_name: str) -> WikiSearchEngine:
            page_index = page_indexes[wid]
            return WikiSearchEngine(wid, wiki_name, _session_factory, page_index)

        engine = ProjectSearchEngine(_wiki_engine_factory)
        result = await engine.search(query, wikis=wiki_tuples, hop_depth=hop_depth, top_k=top_k)

        return {
            "query": result.query,
            "results": [
                {
                    "wiki_id": item.wiki_id,
                    "wiki_name": item.wiki_name,
                    "page_title": item.page_title,
                    "snippet": item.snippet,
                    "score": item.score,
                    "neighbors": [{"title": n.title, "rel": n.rel} for n in item.neighbors],
                }
                for item in result.results
            ],
            "wiki_summary": [
                {
                    "wiki_id": s.wiki_id,
                    "wiki_name": s.wiki_name,
                    "match_count": s.match_count,
                    "relevance": s.relevance,
                }
                for s in result.wiki_summary
            ],
        }
    return await _http_search_project(project_id, query, hop_depth, top_k)


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


async def _http_search_wiki(wiki_id: str, query: str, hop_depth: int, top_k: int) -> dict[str, Any]:
    import httpx

    url = os.getenv("WIKIS_BACKEND_URL", "http://localhost:8000")
    async with httpx.AsyncClient(base_url=url, timeout=60) as client:
        resp = await client.get(
            f"/api/v1/wikis/{wiki_id}/search",
            params={"q": query, "hop_depth": hop_depth, "top_k": top_k},
        )
        if resp.status_code == 404:
            return {"error": f"Wiki not found: {wiki_id}"}
        resp.raise_for_status()
        return resp.json()


async def _http_get_page_neighbors(wiki_id: str, page_title: str, hop_depth: int) -> dict[str, Any]:
    import httpx

    url = os.getenv("WIKIS_BACKEND_URL", "http://localhost:8000")
    async with httpx.AsyncClient(base_url=url, timeout=60) as client:
        resp = await client.get(
            f"/api/v1/wikis/{wiki_id}/pages/{page_title}/neighbors",
            params={"hop_depth": hop_depth},
        )
        if resp.status_code == 404:
            return {"error": f"Page or wiki not found: {wiki_id}/{page_title}"}
        resp.raise_for_status()
        return resp.json()


async def _http_search_project(project_id: str, query: str, hop_depth: int, top_k: int) -> dict[str, Any]:
    import httpx

    url = os.getenv("WIKIS_BACKEND_URL", "http://localhost:8000")
    async with httpx.AsyncClient(base_url=url, timeout=120) as client:
        resp = await client.get(
            f"/api/v1/projects/{project_id}/search",
            params={"q": query, "hop_depth": hop_depth, "top_k": top_k},
        )
        if resp.status_code == 404:
            return {"error": f"Project not found: {project_id}"}
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
