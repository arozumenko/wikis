"""#177 — register incremental_refresh start in DB + terminal-status transitions.

Tests cover:
  * register_wiki(status="running") is called at refresh start (before heavy work)
  * finally block writes the terminal status (complete / failed) via mark_status
  * orphan recovery now marks running WikiRecords failed (inverts #191 C5)
"""

from __future__ import annotations

import asyncio
import json
from datetime import datetime
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from app.models.invocation import Invocation
from app.services.wiki_service import WikiService


def _make_wiki_record(
    wiki_id: str = "owner--repo--main",
    repo_url: str = "https://github.com/owner/repo",
    branch: str = "main",
    title: str = "Test Wiki",
    page_count: int = 3,
    owner_id: str = "user-1",
) -> MagicMock:
    r = MagicMock()
    r.id = wiki_id
    r.repo_url = repo_url
    r.branch = branch
    r.title = title
    r.page_count = page_count
    r.owner_id = owner_id
    return r


def _make_service(wiki_management: MagicMock) -> WikiService:
    settings = MagicMock()
    settings.planner_type = "agent"
    settings.cluster_exclude_tests = False
    settings.llm_max_concurrency = 4
    settings.cache_dir = "/tmp/fake-cache"
    storage = MagicMock()
    storage.list_artifacts = AsyncMock(return_value=[])
    storage.upload = AsyncMock()
    service = WikiService(settings=settings, storage=storage)
    service.wiki_management = wiki_management
    return service


# ---------------------------------------------------------------------------
# test_incremental_refresh_registers_running_status
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_incremental_refresh_registers_running_status() -> None:
    """register_wiki(status='running') must be called at refresh start.

    We trigger the early-return path (cache_key=None) so the heavy machinery
    never runs — the key assertion is that register_wiki was already called
    before the early return.
    """
    wiki_management = MagicMock()
    wiki_management.get_wiki = AsyncMock(return_value=_make_wiki_record())
    wiki_management.register_wiki = AsyncMock()
    wiki_management.mark_status = AsyncMock(return_value=True)

    service = _make_service(wiki_management)

    # Patch _derive_cache_key at its source module to return None.
    # incremental_refresh imports it locally from wiki_management, so we
    # patch the attribute on that module. This causes an early return after
    # register_wiki has already been called.
    with patch(
        "app.services.wiki_management._derive_cache_key",
        return_value=None,
    ):
        result = await service.incremental_refresh(
            wiki_id="owner--repo--main",
            parsed_nodes=[],
            management=wiki_management,
            owner_id="user-1",
        )

    assert result is None  # early return confirms the mock worked

    wiki_management.register_wiki.assert_awaited_once()
    call_kwargs = wiki_management.register_wiki.await_args.kwargs
    assert call_kwargs["wiki_id"] == "owner--repo--main"
    assert call_kwargs["status"] == "running"
    assert call_kwargs["error"] is None
    assert call_kwargs["repo_url"] == "https://github.com/owner/repo"
    assert call_kwargs["branch"] == "main"


# ---------------------------------------------------------------------------
# test_incremental_refresh_terminal_status_on_success
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_incremental_refresh_terminal_status_on_success(tmp_path) -> None:
    """finally block must call mark_status('complete') on a successful refresh.

    We patch open_storage and IncrementalRegenService.run so _run() finishes
    without real I/O, then await the background task before asserting.
    """
    wiki_management = MagicMock()
    wiki_management.get_wiki = AsyncMock(return_value=_make_wiki_record())
    wiki_management.register_wiki = AsyncMock()
    wiki_management.mark_status = AsyncMock(return_value=True)

    service = _make_service(wiki_management)
    cache_key = "testcachekey"
    # Touch a fake .wiki.db so the db_path.exists() check passes.
    db_path = tmp_path / f"{cache_key}.wiki.db"
    db_path.touch()

    cache_index = {
        "refs": {"owner/repo:main": "owner/repo:main"},
        "owner/repo:main": cache_key,
    }
    (tmp_path / "cache_index.json").write_text(json.dumps(cache_index))
    service.settings.cache_dir = str(tmp_path)

    fake_storage = MagicMock()
    fake_storage.close = MagicMock()
    fake_storage.get_wiki_pages = MagicMock(return_value=[])

    fake_stats = MagicMock()
    mock_llm = MagicMock()

    mock_svc_instance = MagicMock()
    mock_svc_instance.run = MagicMock(return_value=fake_stats)

    with (
        # Patch at the source modules that incremental_refresh imports from.
        patch("app.services.llm_factory.create_llm", return_value=mock_llm),
        patch("app.core.storage.open_storage", return_value=fake_storage),
        patch("app.core.agents.page_patcher.PagePatcher", return_value=MagicMock()),
        patch(
            "app.services.agent_builder.build_agent_for_incremental_refresh",
            return_value=None,
        ),
        patch(
            "app.services.incremental_regen_service.IncrementalRegenService",
            return_value=mock_svc_instance,
        ),
    ):
        result = await service.incremental_refresh(
            wiki_id="owner--repo--main",
            parsed_nodes=[],
            management=wiki_management,
            owner_id="user-1",
        )

        assert result is not None
        inv, stats = result
        task = service._tasks.get(inv.id)
        assert task is not None
        await task  # wait for _run() to finish

    # finally block must have called mark_status with "complete"
    # (register_wiki was called at start; mark_status is the terminal update)
    final_calls = wiki_management.mark_status.await_args_list
    assert len(final_calls) >= 1
    last_call = final_calls[-1]
    assert last_call.kwargs["wiki_id"] == "owner--repo--main"
    assert last_call.kwargs["status"] == "complete"


# ---------------------------------------------------------------------------
# test_incremental_refresh_terminal_status_on_failure
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_incremental_refresh_terminal_status_on_failure(tmp_path) -> None:
    """finally block must call mark_status('failed') when the run raises."""
    wiki_management = MagicMock()
    wiki_management.get_wiki = AsyncMock(return_value=_make_wiki_record())
    wiki_management.register_wiki = AsyncMock()
    wiki_management.mark_status = AsyncMock(return_value=True)

    service = _make_service(wiki_management)
    cache_key = "testcachekey"
    db_path = tmp_path / f"{cache_key}.wiki.db"
    db_path.touch()

    cache_index = {
        "refs": {"owner/repo:main": "owner/repo:main"},
        "owner/repo:main": cache_key,
    }
    (tmp_path / "cache_index.json").write_text(json.dumps(cache_index))
    service.settings.cache_dir = str(tmp_path)

    with (
        patch(
            "app.core.storage.open_storage",
            side_effect=RuntimeError("storage exploded"),
        ),
    ):
        result = await service.incremental_refresh(
            wiki_id="owner--repo--main",
            parsed_nodes=[],
            management=wiki_management,
            owner_id="user-1",
        )

        assert result is not None
        inv, stats = result
        task = service._tasks.get(inv.id)
        assert task is not None
        await task

    # finally block must have called mark_status with "failed"
    final_calls = wiki_management.mark_status.await_args_list
    assert len(final_calls) >= 1
    last_call = final_calls[-1]
    assert last_call.kwargs["wiki_id"] == "owner--repo--main"
    assert last_call.kwargs["status"] == "failed"
    # error kwarg must be present (may be None — invocation.error is set in
    # some paths but mark_status is always called with it for symmetry)
    assert "error" in last_call.kwargs


# ---------------------------------------------------------------------------
# test_orphan_recovery_marks_running_orphan_as_failed
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_orphan_recovery_marks_running_orphan_as_failed() -> None:
    """On startup, orphaned 'running' invocations must update WikiRecord to failed.

    #177: incremental_refresh now writes status='running' to the DB before the
    heavy work. On hard-crash the DB row stays 'running'. The orphan-recovery
    loop on restart must flip it to 'failed' so the user gets a clear
    "refresh crashed, please retry" signal.
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

    # Critical: WikiRecord IS marked failed (inverts the old C5 behavior).
    wiki_management.mark_status.assert_awaited_once()
    call_args = wiki_management.mark_status.await_args
    assert call_args.kwargs["wiki_id"] == "owner--repo--main"
    assert call_args.kwargs["status"] == "failed"
    assert "Server restarted" in call_args.kwargs["error"]
