"""Integration test for WikiService multi-source dispatch (#189).

Network-dependent tests are skipped automatically when GitHub is unreachable.
Confluence/Jira integration tests are intentionally omitted — no public
sandbox is available.
"""

from __future__ import annotations

import socket


def _has_network() -> bool:
    try:
        with socket.create_connection(("github.com", 443), timeout=2):
            return True
    except OSError:
        return False


_NETWORK = _has_network()

import pytest

# ---------------------------------------------------------------------------
# 1. GenerateWikiRequest validates correctly in end-to-end model round-trip
# ---------------------------------------------------------------------------


def test_generate_wiki_request_roundtrip_git() -> None:
    """Round-trip: JSON-like dict → GenerateWikiRequest → verify shape."""
    from app.models.api import GenerateWikiRequest

    raw = {
        "source_type": "git",
        "scope": {"repo_url": "https://github.com/public/repo", "branch": "main"},
        "auth": {},
        "wiki_title": "Test Wiki",
    }
    req = GenerateWikiRequest(**raw)
    assert req.source_type == "git"
    assert req.scope["repo_url"] == "https://github.com/public/repo"
    assert req.repo_url == "https://github.com/public/repo"  # back-filled


def test_generate_wiki_request_roundtrip_confluence() -> None:
    from app.models.api import GenerateWikiRequest

    raw = {
        "source_type": "confluence",
        "scope": {"base_url": "https://acme.atlassian.net", "space_keys": ["DEV"]},
        "auth": {"access_token": "tok", "refresh_token": None, "client_id": None},
    }
    req = GenerateWikiRequest(**raw)
    assert req.source_type == "confluence"
    assert req.scope["space_keys"] == ["DEV"]
    assert req.auth["access_token"] == "tok"
    # Tokens must not end up in scope.
    assert "access_token" not in req.scope


def test_generate_wiki_request_roundtrip_jira() -> None:
    from app.models.api import GenerateWikiRequest

    raw = {
        "source_type": "jira",
        "scope": {"base_url": "https://acme.atlassian.net", "jql": "project=ENG ORDER BY created"},
        "auth": {"access_token": "tok"},
    }
    req = GenerateWikiRequest(**raw)
    assert req.source_type == "jira"
    assert "project=ENG" in req.scope["jql"]


# ---------------------------------------------------------------------------
# 2. Backwards-compat: old-style requests reach the service unchanged
# ---------------------------------------------------------------------------


def test_legacy_request_still_works_with_service_generate() -> None:
    """Verify that the service's generate() accepts a legacy-style request without raising."""
    import asyncio
    from unittest.mock import AsyncMock, MagicMock, patch

    from app.models.api import GenerateWikiRequest
    from app.services.wiki_service import WikiService

    settings = MagicMock()
    settings.planner_type = "agent"
    settings.cluster_exclude_tests = False
    settings.llm_max_concurrency = 4

    storage = MagicMock()
    storage.upload = AsyncMock()
    storage.download = AsyncMock(side_effect=FileNotFoundError)

    wiki_management = MagicMock()
    wiki_management.get_wiki_record = AsyncMock(return_value=None)
    wiki_management.register_wiki = AsyncMock()

    service = WikiService(settings=settings, storage=storage, wiki_management=wiki_management)

    req = GenerateWikiRequest(
        repo_url="https://github.com/octocat/Hello-World",
        branch="main",
    )

    async def _run() -> None:
        with patch.object(service, "_run_wiki_subprocess", new_callable=AsyncMock) as mock_sub:
            mock_sub.return_value = {"success": True, "generated_pages": {}, "artifacts": []}
            with patch.object(service, "_persist_invocations", new_callable=AsyncMock):
                inv = await service.generate(req, owner_id="user1")
                # Give the background task a chance to run.
                await asyncio.sleep(0.05)
                # Verify subprocess was called.
                assert mock_sub.called
                payload_call = mock_sub.call_args
                passed_request = payload_call[1]["request"] if "request" in (payload_call[1] or {}) else payload_call[0][1]
                assert passed_request.source_type == "git"
                assert passed_request.scope.get("repo_url") == "https://github.com/octocat/Hello-World"

    asyncio.run(_run())


# ---------------------------------------------------------------------------
# 3. Full pipeline smoke test (network-gated)
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not _NETWORK, reason="GitHub unreachable — skipping network integration test")
def test_wiki_dispatch_git_full_pipeline_smoke() -> None:
    """End-to-end: generate() dispatches a Git wiki via the new source_type path.

    Uses a tiny public repo (github.com/octocat/Hello-World) and mocks
    the subprocess to avoid real LLM calls, while exercising the full
    generate() → _run_wiki_subprocess() orchestration path.
    """
    import asyncio
    from unittest.mock import AsyncMock, MagicMock, patch

    from app.models.api import GenerateWikiRequest
    from app.services.wiki_service import WikiService

    settings = MagicMock()
    settings.planner_type = "agent"
    settings.cluster_exclude_tests = False
    settings.llm_max_concurrency = 4

    storage = MagicMock()
    storage.upload = AsyncMock()
    storage.download = AsyncMock(side_effect=FileNotFoundError)
    storage.list_artifacts = AsyncMock(return_value=[])

    wiki_management = MagicMock()
    wiki_management.get_wiki_record = AsyncMock(return_value=None)
    wiki_management.register_wiki = AsyncMock()
    wiki_management.index_wiki_pages = AsyncMock()

    service = WikiService(settings=settings, storage=storage, wiki_management=wiki_management)

    req = GenerateWikiRequest(
        source_type="git",
        scope={"repo_url": "https://github.com/octocat/Hello-World", "branch": "master"},
        auth={},
        wiki_title="Hello World Wiki",
    )

    async def _run() -> None:
        with patch.object(service, "_run_wiki_subprocess", new_callable=AsyncMock) as mock_sub:
            mock_sub.return_value = {
                "success": True,
                "generated_pages": {"intro": "# Hello World\n\nWelcome."},
                "artifacts": [],
                "commit_hash": "abc1234",
            }
            with patch.object(service, "_persist_invocations", new_callable=AsyncMock):
                inv = await service.generate(req, owner_id="user1")
                await asyncio.sleep(0.1)

            # Confirm the wiki was registered with the new source fields.
            register_calls = wiki_management.register_wiki.call_args_list
            assert register_calls, "register_wiki was never called"
            final_call_kwargs = register_calls[-1][1]
            assert final_call_kwargs.get("source_type") == "git"

    asyncio.run(_run())
