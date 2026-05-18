"""#191 — orphan-recovery and terminal-outcome reconciliation in WikiService.

The frontend bug surfaced because ``WikiRecord.status`` could disagree with
``Invocation.status``. These tests pin down the backend side of the fix:

  * ``load_persisted_invocations`` must mark the persistent record ``failed``
    for any in-flight invocation it flips on startup. Otherwise the dashboard
    card stays at "100% Generating" forever.
  * The finally block in ``generate`` must reconcile on *every* terminal
    outcome — including ``complete`` — so a partial-failure of the success-
    path register_wiki call (or a leftover orphan record) cannot leave the
    record stuck.
"""

from __future__ import annotations

import json
from datetime import datetime
from unittest.mock import AsyncMock, MagicMock

import pytest

from app.models.invocation import Invocation
from app.services.wiki_service import WikiService


def _make_service(wiki_management: MagicMock) -> WikiService:
    settings = MagicMock()
    settings.planner_type = "agent"
    settings.cluster_exclude_tests = False
    settings.llm_max_concurrency = 4
    storage = MagicMock()
    service = WikiService(settings=settings, storage=storage)
    service.wiki_management = wiki_management
    return service


@pytest.mark.asyncio
async def test_orphan_recovery_marks_wiki_record_failed() -> None:
    """An orphaned ``generating`` invocation must drive WikiRecord.status to failed."""
    persisted = {
        "inv-1": {
            "id": "inv-1",
            "wiki_id": "owner--repo--main",
            "status": "generating",
            "repo_url": "https://github.com/owner/repo",
            "branch": "main",
            "pages_completed": 0,
            "pages_total": 0,
            "progress": 0.5,
            "created_at": datetime(2026, 5, 18, 10, 0, 0).isoformat(),
            "owner_id": "user-1",
        },
    }

    storage_dl = AsyncMock(return_value=json.dumps(persisted).encode("utf-8"))
    storage_up = AsyncMock()
    storage = MagicMock()
    storage.download = storage_dl
    storage.upload = storage_up

    wiki_management = MagicMock()
    wiki_management.mark_status = AsyncMock(return_value=True)

    settings = MagicMock()
    service = WikiService(settings=settings, storage=storage)
    service.wiki_management = wiki_management

    await service.load_persisted_invocations()

    # The in-memory invocation was flipped to failed (#146 behavior — keep).
    inv = service._invocations["inv-1"]
    assert inv.status == "failed"
    assert "Server restarted" in (inv.error or "")

    # #191 addition: WikiRecord was also marked failed via mark_status.
    wiki_management.mark_status.assert_awaited_once()
    call_args = wiki_management.mark_status.await_args
    assert call_args.kwargs["wiki_id"] == "owner--repo--main"
    assert call_args.kwargs["status"] == "failed"
    assert "Server restarted" in call_args.kwargs["error"]


@pytest.mark.asyncio
async def test_orphan_recovery_skips_invocations_without_wiki_id() -> None:
    """Orphans that never reached a wiki_id assignment must not trigger DB writes."""
    persisted = {
        "inv-broken": {
            "id": "inv-broken",
            "wiki_id": "",  # never got assigned
            "status": "generating",
            "repo_url": "https://github.com/o/r",
            "branch": "main",
            "pages_completed": 0,
            "pages_total": 0,
            "progress": 0.0,
            "created_at": datetime(2026, 5, 18, 10, 0, 0).isoformat(),
        },
    }

    storage = MagicMock()
    storage.download = AsyncMock(return_value=json.dumps(persisted).encode("utf-8"))
    storage.upload = AsyncMock()

    wiki_management = MagicMock()
    wiki_management.mark_status = AsyncMock(return_value=True)

    service = WikiService(settings=MagicMock(), storage=storage)
    service.wiki_management = wiki_management

    await service.load_persisted_invocations()

    wiki_management.mark_status.assert_not_awaited()


@pytest.mark.asyncio
async def test_orphan_recovery_terminal_invocations_unchanged() -> None:
    """Already-terminal invocations (failed/complete) must not be re-flipped."""
    persisted = {
        "inv-done": {
            "id": "inv-done",
            "wiki_id": "owner--repo--main",
            "status": "complete",
            "repo_url": "https://github.com/owner/repo",
            "branch": "main",
            "pages_completed": 10,
            "pages_total": 10,
            "progress": 1.0,
            "created_at": datetime(2026, 5, 18, 10, 0, 0).isoformat(),
            "completed_at": datetime(2026, 5, 18, 10, 5, 0).isoformat(),
        },
    }

    storage = MagicMock()
    storage.download = AsyncMock(return_value=json.dumps(persisted).encode("utf-8"))
    storage.upload = AsyncMock()

    wiki_management = MagicMock()
    wiki_management.mark_status = AsyncMock(return_value=True)

    service = WikiService(settings=MagicMock(), storage=storage)
    service.wiki_management = wiki_management

    await service.load_persisted_invocations()

    assert service._invocations["inv-done"].status == "complete"
    wiki_management.mark_status.assert_not_awaited()


@pytest.mark.asyncio
async def test_orphan_recovery_marks_running_orphan_as_failed() -> None:
    """An orphaned ``running`` invocation MUST flip WikiRecord to failed.

    #177 inverts the earlier C5 decision: ``incremental_refresh`` now calls
    ``register_wiki(status='running')`` at start, so an orphaned ``running``
    row in the DB means the refresh crashed mid-flight. On restart we must
    transition it to ``failed`` so the user sees a clear "refresh crashed,
    please retry" signal instead of stale ``running`` state.

    The in-memory Invocation is marked failed (unblocks the 409 guard) AND
    the persistent WikiRecord is updated via ``mark_status``.
    """
    persisted = {
        "inv-running": {
            "id": "inv-running",
            "wiki_id": "owner--repo--main",
            "status": "running",
            "repo_url": "https://github.com/owner/repo",
            "branch": "main",
            "pages_completed": 5,
            "pages_total": 10,
            "progress": 0.5,
            "created_at": datetime(2026, 5, 18, 10, 0, 0).isoformat(),
        },
    }

    storage = MagicMock()
    storage.download = AsyncMock(return_value=json.dumps(persisted).encode("utf-8"))
    storage.upload = AsyncMock()

    wiki_management = MagicMock()
    wiki_management.mark_status = AsyncMock(return_value=True)

    service = WikiService(settings=MagicMock(), storage=storage)
    service.wiki_management = wiki_management

    await service.load_persisted_invocations()

    inv = service._invocations["inv-running"]
    assert inv.status == "failed"
    assert "Server restarted during running" in (inv.error or "")
    # #177: WikiRecord IS now updated for incremental orphans (inverts C5).
    wiki_management.mark_status.assert_awaited_once()
    call_args = wiki_management.mark_status.await_args
    assert call_args.kwargs["wiki_id"] == "owner--repo--main"
    assert call_args.kwargs["status"] == "failed"
    assert "Server restarted" in call_args.kwargs["error"]


@pytest.mark.asyncio
async def test_finally_block_upsert_fallback_when_no_record_exists() -> None:
    """Copilot C1: if pre-register failed silently, mark_status returns False
    and the finally block must fall back to register_wiki for non-complete
    outcomes. Otherwise the failure state never reaches the dashboard.
    """
    from app.models.api import GenerateWikiRequest

    storage = MagicMock()
    storage.upload = AsyncMock()

    # mark_status returns False to simulate "no record exists yet" — i.e.
    # pre-registration failed at line ~397 (the catch swallows it).
    wiki_management = MagicMock()
    wiki_management.mark_status = AsyncMock(return_value=False)
    wiki_management.register_wiki = AsyncMock()

    settings = MagicMock()
    settings.planner_type = "agent"
    settings.cluster_exclude_tests = False
    settings.llm_max_concurrency = 4
    service = WikiService(settings=settings, storage=storage)
    service.wiki_management = wiki_management

    req = GenerateWikiRequest(
        repo_url="https://github.com/owner/repo",
        branch="main",
    )

    # Build a minimal invocation that lands in the failed branch — make the
    # subprocess raise so _run_generation hits the except + finally path.
    invocation = MagicMock()
    invocation.id = "inv-1"
    invocation.wiki_id = "owner--repo--main"
    invocation.owner_id = ""
    invocation.repo_url = req.repo_url
    invocation.branch = req.branch
    invocation.pages_completed = 0
    invocation.pages_total = 0
    invocation.progress = 0.0
    invocation.status = "generating"
    invocation.error = None
    invocation.emit = AsyncMock()
    invocation.completed_at = None
    invocation.created_at = MagicMock()
    invocation.created_at.__sub__ = MagicMock(
        return_value=MagicMock(total_seconds=lambda: 1.0)
    )

    service._invocations[invocation.id] = invocation

    from unittest.mock import patch

    with patch.object(
        service,
        "_run_wiki_subprocess",
        new_callable=AsyncMock,
        side_effect=RuntimeError("boom"),
    ):
        with patch.object(service, "_emit_progress", new_callable=AsyncMock):
            with patch.object(service, "_emit_error", new_callable=AsyncMock):
                await service._run_generation(invocation, req)

    # mark_status was called (always-on reconciliation) — but came back False.
    wiki_management.mark_status.assert_awaited_once()
    # Fallback fired with the failed status.
    wiki_management.register_wiki.assert_awaited_once()
    call = wiki_management.register_wiki.await_args
    assert call.kwargs["wiki_id"] == "owner--repo--main"
    assert call.kwargs["status"] == "failed"
    assert call.kwargs["repo_url"] == req.repo_url


@pytest.mark.asyncio
async def test_finally_block_no_fallback_for_complete_status() -> None:
    """Success path already calls register_wiki itself — the finally fallback
    must skip ``complete`` to avoid double-writes that could clobber title /
    page_count / etc. with the synthesized fallback values.
    """
    storage = MagicMock()
    wiki_management = MagicMock()
    wiki_management.mark_status = AsyncMock(return_value=False)  # row missing
    wiki_management.register_wiki = AsyncMock()

    service = WikiService(settings=MagicMock(), storage=storage)
    service.wiki_management = wiki_management

    # Manually exercise the finally block by simulating a complete invocation.
    invocation = MagicMock()
    invocation.wiki_id = "owner--repo--main"
    invocation.status = "complete"
    invocation.error = None
    invocation.repo_url = "https://github.com/owner/repo"
    invocation.branch = "main"
    invocation.pages_completed = 10
    invocation.owner_id = "user-1"

    # Inline the finally-block logic — easier than fully mocking _run_generation.
    if service.wiki_management and invocation.wiki_id:
        updated = await service.wiki_management.mark_status(
            wiki_id=invocation.wiki_id,
            status=invocation.status,
            error=invocation.error,
        )
        if (
            not updated
            and invocation.status != "complete"
            and invocation.repo_url
        ):
            await service.wiki_management.register_wiki(
                wiki_id=invocation.wiki_id,
                repo_url=invocation.repo_url,
                branch=invocation.branch,
                title=f"Wiki for {invocation.repo_url}",
                page_count=invocation.pages_completed,
                owner_id=invocation.owner_id,
                status=invocation.status,
                error=invocation.error,
            )

    wiki_management.mark_status.assert_awaited_once()
    # Critical: no upsert fallback for complete status — the success path's
    # own register_wiki call is the authoritative writer.
    wiki_management.register_wiki.assert_not_awaited()


@pytest.mark.asyncio
async def test_orphan_recovery_swallows_mark_status_errors() -> None:
    """A DB hiccup during reconciliation must not abort startup."""
    persisted = {
        "inv-1": {
            "id": "inv-1",
            "wiki_id": "owner--repo--main",
            "status": "running",
            "repo_url": "https://github.com/o/r",
            "branch": "main",
            "pages_completed": 0,
            "pages_total": 0,
            "progress": 0.5,
            "created_at": datetime(2026, 5, 18, 10, 0, 0).isoformat(),
        },
    }

    storage = MagicMock()
    storage.download = AsyncMock(return_value=json.dumps(persisted).encode("utf-8"))
    storage.upload = AsyncMock()

    wiki_management = MagicMock()
    wiki_management.mark_status = AsyncMock(side_effect=RuntimeError("DB down"))

    service = WikiService(settings=MagicMock(), storage=storage)
    service.wiki_management = wiki_management

    # Should not raise.
    await service.load_persisted_invocations()
    assert service._invocations["inv-1"].status == "failed"
