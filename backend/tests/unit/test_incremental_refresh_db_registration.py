"""#177 — register incremental_refresh start in DB + terminal-status transitions.

Tests cover:
  * No DB write happens when early-return guards fire (cache_key=None / db missing)
  * mark_status(status="running") is called before asyncio.create_task(_run())
  * finally block writes the terminal status (complete / failed) via mark_status
"""

from __future__ import annotations

import json
from datetime import datetime
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

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
# C3a — no DB write before early-return guards
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_incremental_refresh_does_not_write_to_db_when_cache_key_missing() -> None:
    """mark_status must NOT be called when _derive_cache_key returns None.

    After C1+C2, the DB write moved to AFTER the early-return guards, so
    triggering the guard (cache_key=None) must leave the DB completely
    untouched — no partially-started "running" row to confuse the dashboard.
    """
    wiki_management = MagicMock()
    wiki_management.get_wiki = AsyncMock(return_value=_make_wiki_record())
    wiki_management.register_wiki = AsyncMock()
    wiki_management.mark_status = AsyncMock(return_value=True)

    service = _make_service(wiki_management)

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

    assert result is None  # early return fired

    # Neither DB method must have been touched before the guard fired.
    wiki_management.register_wiki.assert_not_awaited()
    wiki_management.mark_status.assert_not_awaited()


# ---------------------------------------------------------------------------
# C3b — mark_status(status="running") written before the task is created
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_incremental_refresh_writes_running_status_when_task_starts(
    tmp_path,
) -> None:
    """mark_status(status='running', error=None) must be called before _run() runs.

    We drive a happy-path refresh with a real .wiki.db file on disk so the
    early-return guards pass, then mock the inner pipeline so the task body
    exits immediately without real I/O. The assertion checks that mark_status
    was called with the expected kwargs once the task is scheduled.
    """
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

    fake_storage = MagicMock()
    fake_storage.close = MagicMock()
    fake_storage.get_wiki_pages = MagicMock(return_value=[])

    mock_svc_instance = MagicMock()
    mock_svc_instance.run = MagicMock(return_value=MagicMock())

    with (
        patch("app.services.llm_factory.create_llm", return_value=MagicMock()),
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
        inv, _stats = result
        task = service._tasks.get(inv.id)
        assert task is not None
        await task  # let _run() finish before asserting

    # mark_status must have been called with status="running" before the task
    # body ran (it's called synchronously in the outer coroutine, before
    # create_task), and then again at the end with the terminal status.
    all_calls = wiki_management.mark_status.await_args_list
    assert len(all_calls) >= 1

    # First call: the pre-task "running" registration.
    first_call = all_calls[0]
    assert first_call.kwargs["wiki_id"] == "owner--repo--main"
    assert first_call.kwargs["status"] == "running"
    assert first_call.kwargs["error"] is None

    # register_wiki must NOT have been touched (C2: we use mark_status only).
    wiki_management.register_wiki.assert_not_awaited()


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
    # (mark_status is the terminal update)
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
    # #226: the exception path must capture the exception message on
    # invocation.error so mark_status forwards it to the WikiRecord and
    # the dashboard shows the failure reason. Previously the exception
    # path forgot to set invocation.error, so the DB silently stored
    # error=None on every refresh failure.
    error_value = last_call.kwargs.get("error")
    assert error_value is not None, (
        f"expected non-None error in mark_status call; got {error_value!r}"
    )
    assert "storage exploded" in error_value, (
        f"expected exception message in error; got {error_value!r}"
    )


# ---------------------------------------------------------------------------
# patcher-is-None early-return must also set invocation.error (followup #273)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_incremental_refresh_patcher_none_writes_error(tmp_path) -> None:
    """When ``create_llm`` raises, the patcher-is-None early-return path
    must still capture a non-None ``error`` so the dashboard shows why the
    refresh failed.  Without the fix, ``mark_status`` was forwarded
    ``error=None`` and the user saw "failed" with no message.
    """
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

    # Force the LLM-init except block to fire → llm = None → patcher = None
    # → early-return path executes inside _run().
    with patch(
        "app.services.llm_factory.create_llm",
        side_effect=RuntimeError("no llm creds configured"),
    ):
        result = await service.incremental_refresh(
            wiki_id="owner--repo--main",
            parsed_nodes=[],
            management=wiki_management,
            owner_id="user-1",
        )

        assert result is not None
        inv, _stats = result
        task = service._tasks.get(inv.id)
        assert task is not None
        await task

    final_calls = wiki_management.mark_status.await_args_list
    assert len(final_calls) >= 1
    last_call = final_calls[-1]
    assert last_call.kwargs["wiki_id"] == "owner--repo--main"
    assert last_call.kwargs["status"] == "failed"
    error_value = last_call.kwargs.get("error")
    assert error_value is not None, (
        f"expected non-None error from patcher-is-None path; got {error_value!r}"
    )
    assert "LLM" in error_value, (
        f"expected the LLM-unavailable message in error; got {error_value!r}"
    )
