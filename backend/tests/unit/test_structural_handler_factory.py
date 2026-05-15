"""Unit tests for :func:`make_agent_structural_handler`.

The factory is the production wiring of PR 3's structural-handler
callback (PR #133 left it as an injected callable so the orchestrator
stays unit-testable). These tests pin down the observable behaviors:

* Happy path: agent returns markdown, factory writes it, stamps the
  new content_hash, returns True.
* Agent returns None (legacy row, deserialize failure, etc.): factory
  returns False without calling the writer or stamping the hash.
* Writer raises: factory catches and returns False.
* Content_hash stamp failure: writer succeeded so we still return
  True; the stale-hash regret is bounded to one extra regen.
"""

from __future__ import annotations

from dataclasses import dataclass

import pytest

from app.core.storage.incremental import compute_content_hash
from app.services.incremental_regen import Regime
from app.services.structural_handler_factory import make_agent_structural_handler


@dataclass
class _FakePage:
    page_id: str = "p1"
    regime: Regime = Regime.STRUCTURAL
    changes: tuple = ()
    reason: str = ""


class _AgentStub:
    """Captures every regenerate_single_page call so tests can assert
    the args. Returns the canned output."""

    def __init__(self, output: str | None) -> None:
        self._output = output
        self.calls: list[dict] = []

    def regenerate_single_page(
        self, page_id: str, *, repository_context: str, config=None,
    ) -> str | None:
        self.calls.append(
            {
                "page_id": page_id,
                "repository_context": repository_context,
                "config": config,
            }
        )
        return self._output


class _StorageStub:
    """Minimal storage stub. Records every upsert so tests can assert
    the new content_hash was persisted."""

    def __init__(self, initial_rows: dict[str, dict] | None = None) -> None:
        self._rows = initial_rows or {}
        self.upsert_calls: list[dict] = []
        self.get_call_count = 0
        self.get_should_raise: Exception | None = None
        self.upsert_should_raise: Exception | None = None

    def get_wiki_page(self, page_id: str) -> dict | None:
        self.get_call_count += 1
        if self.get_should_raise is not None:
            raise self.get_should_raise
        row = self._rows.get(page_id)
        return dict(row) if row else None

    def upsert_wiki_page(self, page: dict) -> None:
        if self.upsert_should_raise is not None:
            raise self.upsert_should_raise
        self.upsert_calls.append(dict(page))
        self._rows[page["page_id"]] = dict(page)


def test_happy_path_writes_stamps_hash_and_returns_true() -> None:
    agent = _AgentStub("# Regenerated\n\nFresh content.")
    writes: list[tuple[str, str]] = []
    storage = _StorageStub(
        {"p1": {"page_id": "p1", "wiki_id": "w", "content_hash": "old"}},
    )

    handler = make_agent_structural_handler(
        agent,
        storage=storage,
        repository_context="repo context here",
        write_page_body=lambda pid, body: writes.append((pid, body)),
    )

    assert handler(_FakePage(page_id="p1")) is True
    assert writes == [("p1", "# Regenerated\n\nFresh content.")]
    # content_hash stamp ran: storage saw an upsert with the new hash.
    assert len(storage.upsert_calls) == 1
    stamped = storage.upsert_calls[0]
    assert stamped["page_id"] == "p1"
    assert stamped["content_hash"] == compute_content_hash(
        "# Regenerated\n\nFresh content."
    )
    # The pre-existing hash was replaced (not preserved).
    assert stamped["content_hash"] != "old"


def test_agent_returns_none_skips_write_and_stamp_and_returns_false() -> None:
    agent = _AgentStub(None)
    writes: list[tuple[str, str]] = []
    storage = _StorageStub({"p1": {"page_id": "p1", "wiki_id": "w"}})

    handler = make_agent_structural_handler(
        agent,
        storage=storage,
        repository_context="ctx",
        write_page_body=lambda pid, body: writes.append((pid, body)),
    )

    assert handler(_FakePage(page_id="legacy-no-spec")) is False
    assert writes == []
    # No stamp attempted either.
    assert storage.upsert_calls == []
    assert storage.get_call_count == 0


def test_writer_failure_is_caught_and_skips_stamp_and_returns_false() -> None:
    agent = _AgentStub("# Some content")
    storage = _StorageStub({"p1": {"page_id": "p1", "wiki_id": "w"}})

    def _broken_writer(pid: str, body: str) -> None:
        raise RuntimeError("artifact storage unreachable")

    handler = make_agent_structural_handler(
        agent,
        storage=storage,
        repository_context="ctx",
        write_page_body=_broken_writer,
    )

    # Must not propagate the writer's exception — orchestrator depends
    # on every structural callback returning bool, not raising.
    assert handler(_FakePage(page_id="p1")) is False
    # Hash stamp should NOT happen when the body write failed.
    assert storage.upsert_calls == []


def test_content_hash_stamp_failure_does_not_fail_the_regen() -> None:
    """A failed hash stamp means the next regen wastes one round trip
    on this page (re-classifies as MODIFIED, re-processes it). That's
    bad but recoverable. The body write already succeeded, so we
    return True; ops sees the WARNING in the log."""
    agent = _AgentStub("# Some content")
    writes: list[tuple[str, str]] = []
    storage = _StorageStub({"p1": {"page_id": "p1", "wiki_id": "w"}})
    storage.upsert_should_raise = RuntimeError("transient DB blip")

    handler = make_agent_structural_handler(
        agent,
        storage=storage,
        repository_context="ctx",
        write_page_body=lambda pid, body: writes.append((pid, body)),
    )

    # Body wrote; only the hash stamp failed → still True.
    assert handler(_FakePage(page_id="p1")) is True
    assert writes == [("p1", "# Some content")]


def test_content_hash_stamp_skipped_when_page_row_missing() -> None:
    """If get_wiki_page returns None (page was deleted between regen
    start and stamp), the factory shouldn't try to upsert a NULL row."""
    agent = _AgentStub("# Some content")
    writes: list[tuple[str, str]] = []
    storage = _StorageStub({})  # no row for p1

    handler = make_agent_structural_handler(
        agent,
        storage=storage,
        repository_context="ctx",
        write_page_body=lambda pid, body: writes.append((pid, body)),
    )

    assert handler(_FakePage(page_id="p1")) is True
    assert writes == [("p1", "# Some content")]
    # No upsert attempted on a missing row.
    assert storage.upsert_calls == []
