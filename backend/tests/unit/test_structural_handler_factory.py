"""Unit tests for :func:`make_agent_structural_handler`.

The factory is the production wiring of PR 3's structural-handler
callback (PR #133 left it as an injected callable so the orchestrator
stays unit-testable). These tests pin down the three observable
behaviors:

* Happy path: agent returns markdown, factory writes it, returns True.
* Agent returns None (legacy row, deserialize failure, etc.): factory
  returns False without calling the writer.
* Writer raises: factory catches and returns False.
"""

from __future__ import annotations

from dataclasses import dataclass

import pytest

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


def test_happy_path_writes_and_returns_true() -> None:
    agent = _AgentStub("# Regenerated\n\nFresh content.")
    writes: list[tuple[str, str]] = []

    handler = make_agent_structural_handler(
        agent,
        repository_context="repo context here",
        write_page_body=lambda pid, body: writes.append((pid, body)),
    )

    assert handler(_FakePage(page_id="p1")) is True
    assert writes == [("p1", "# Regenerated\n\nFresh content.")]
    assert agent.calls == [
        {
            "page_id": "p1",
            "repository_context": "repo context here",
            "config": None,
        },
    ]


def test_agent_returns_none_skips_write_and_returns_false() -> None:
    agent = _AgentStub(None)
    writes: list[tuple[str, str]] = []

    handler = make_agent_structural_handler(
        agent,
        repository_context="ctx",
        write_page_body=lambda pid, body: writes.append((pid, body)),
    )

    assert handler(_FakePage(page_id="legacy-no-spec")) is False
    # Writer wasn't invoked.
    assert writes == []


def test_writer_failure_is_caught_and_returns_false() -> None:
    agent = _AgentStub("# Some content")

    def _broken_writer(pid: str, body: str) -> None:
        raise RuntimeError("artifact storage unreachable")

    handler = make_agent_structural_handler(
        agent,
        repository_context="ctx",
        write_page_body=_broken_writer,
    )

    # Must not propagate the writer's exception — orchestrator depends
    # on every structural callback returning bool, not raising.
    assert handler(_FakePage(page_id="p1")) is False
