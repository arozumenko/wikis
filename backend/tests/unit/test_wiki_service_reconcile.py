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
