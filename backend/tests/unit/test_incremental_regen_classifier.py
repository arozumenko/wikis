"""Unit tests for the regime classifier in ``app.services.incremental_regen``.

The classifier maps each :class:`AffectedPage` into one of four
regimes (unchanged / trivial / edit / structural) using the rules
documented in the module docstring. Tests cover every rule including
the primary-symbol-deleted override that takes priority over ``max_kind``.
"""

from __future__ import annotations

from dataclasses import dataclass

import pytest

from app.services.change_detector import (
    AffectedPage,
    ChangeKind,
    ChangeSet,
    NodeChange,
)
from app.services.incremental_regen import (
    PageRegime,
    Regime,
    RegimePlan,
    classify_pages,
)


class _StubStorage:
    """Minimal storage stub — only ``get_wiki_page`` is read by the classifier."""

    def __init__(self, pages: dict[str, dict] | None = None) -> None:
        self._pages = pages or {}

    def get_wiki_page(self, page_id: str) -> dict | None:
        return self._pages.get(page_id)


def _change(kind: ChangeKind, node_id: str = "n", **kwargs) -> NodeChange:
    return NodeChange(kind=kind, node_id=node_id, **kwargs)


def _affected(page_id: str, *changes: NodeChange) -> AffectedPage:
    return AffectedPage(page_id=page_id, changes=list(changes))


# ---------------------------------------------------------------------------
# Rule 1: primary symbol deleted → structural (overrides everything else)
# ---------------------------------------------------------------------------


class TestPrimarySymbolDeletedOverride:
    def test_primary_deleted_routes_to_structural_even_if_kind_is_moved(self) -> None:
        # Page is "about" sym-1; sym-1 is deleted in the change set.
        # Without the override, a MOVED-only change would route to trivial.
        change_set = ChangeSet(
            deleted=[_change(ChangeKind.DELETED, "sym-1", old_hash="abc")]
        )
        affected = [
            _affected("p1", _change(ChangeKind.MOVED, "sym-other", old_hash="x", new_hash="x")),
        ]
        storage = _StubStorage({"p1": {"primary_symbol_id": "sym-1"}})

        plan = classify_pages(change_set, affected, storage)
        assert len(plan.structural) == 1
        assert plan.structural[0].page_id == "p1"
        assert "primary symbol" in plan.structural[0].reason
        assert not plan.trivial

    def test_primary_intact_does_not_trigger_override(self) -> None:
        # Different node deleted; page's primary still in repo.
        change_set = ChangeSet(deleted=[_change(ChangeKind.DELETED, "sym-other")])
        affected = [
            _affected("p1", _change(ChangeKind.MOVED, "sym-1", old_hash="x", new_hash="x")),
        ]
        storage = _StubStorage({"p1": {"primary_symbol_id": "sym-1"}})

        plan = classify_pages(change_set, affected, storage)
        # MOVED-only routes to trivial since primary is intact.
        assert len(plan.trivial) == 1


# ---------------------------------------------------------------------------
# Rule 2: ADDED or DELETED in changes → structural
# ---------------------------------------------------------------------------


class TestTopologyRoutes:
    def test_added_only_routes_structural(self) -> None:
        change_set = ChangeSet()
        affected = [_affected("p1", _change(ChangeKind.ADDED, "n-new"))]
        plan = classify_pages(change_set, affected, _StubStorage())
        assert len(plan.structural) == 1
        assert "topology change" in plan.structural[0].reason

    def test_deleted_only_routes_structural(self) -> None:
        change_set = ChangeSet(deleted=[_change(ChangeKind.DELETED, "n-gone")])
        affected = [_affected("p1", _change(ChangeKind.DELETED, "n-gone"))]
        plan = classify_pages(change_set, affected, _StubStorage())
        # Deleted node isn't a primary for any page → override doesn't fire.
        # Still routes structural via Rule 2.
        assert len(plan.structural) == 1

    def test_mixed_with_added_routes_structural(self) -> None:
        change_set = ChangeSet()
        affected = [
            _affected(
                "p1",
                _change(ChangeKind.MODIFIED, "n-mod"),
                _change(ChangeKind.ADDED, "n-new"),
            ),
        ]
        plan = classify_pages(change_set, affected, _StubStorage())
        # Topology change wins over MODIFIED.
        assert len(plan.structural) == 1


# ---------------------------------------------------------------------------
# Rule 3: only MODIFIED → edit
# ---------------------------------------------------------------------------


def test_modified_only_routes_edit() -> None:
    change_set = ChangeSet()
    affected = [_affected("p1", _change(ChangeKind.MODIFIED, "n1"))]
    plan = classify_pages(change_set, affected, _StubStorage())
    assert len(plan.edit) == 1
    assert plan.edit[0].reason == "modified only"


def test_multiple_modified_still_routes_edit() -> None:
    change_set = ChangeSet()
    affected = [
        _affected(
            "p1",
            _change(ChangeKind.MODIFIED, "n1"),
            _change(ChangeKind.MODIFIED, "n2"),
        ),
    ]
    plan = classify_pages(change_set, affected, _StubStorage())
    assert len(plan.edit) == 1


# ---------------------------------------------------------------------------
# Rule 4: only MOVED → trivial
# ---------------------------------------------------------------------------


def test_moved_only_routes_trivial() -> None:
    change_set = ChangeSet()
    affected = [
        _affected(
            "p1",
            _change(ChangeKind.MOVED, "n1", old_path="old.py", new_path="new.py"),
        ),
    ]
    plan = classify_pages(change_set, affected, _StubStorage())
    assert len(plan.trivial) == 1
    assert plan.trivial[0].reason == "moved only"


# ---------------------------------------------------------------------------
# Mixed MODIFIED + MOVED → edit (surgical patcher handles both)
# ---------------------------------------------------------------------------


def test_modified_plus_moved_routes_edit() -> None:
    change_set = ChangeSet()
    affected = [
        _affected(
            "p1",
            _change(ChangeKind.MODIFIED, "n1"),
            _change(ChangeKind.MOVED, "n2", old_path="a.py", new_path="b.py"),
        ),
    ]
    plan = classify_pages(change_set, affected, _StubStorage())
    assert len(plan.edit) == 1
    assert "mixed" in plan.edit[0].reason


# ---------------------------------------------------------------------------
# Empty / defensive cases
# ---------------------------------------------------------------------------


def test_empty_changes_routes_unchanged() -> None:
    affected = [AffectedPage(page_id="p1", changes=[])]
    plan = classify_pages(ChangeSet(), affected, _StubStorage())
    assert len(plan.unchanged) == 1


def test_no_affected_pages_yields_empty_plan() -> None:
    plan = classify_pages(ChangeSet(), [], _StubStorage())
    assert plan.total == 0
    assert plan.counts()["total"] == 0


# ---------------------------------------------------------------------------
# Plan accessors
# ---------------------------------------------------------------------------


def test_plan_counts_match_buckets() -> None:
    plan = RegimePlan(
        trivial=[PageRegime("p1", Regime.TRIVIAL, ())],
        edit=[
            PageRegime("p2", Regime.EDIT, ()),
            PageRegime("p3", Regime.EDIT, ()),
        ],
        structural=[PageRegime("p4", Regime.STRUCTURAL, ())],
    )
    counts = plan.counts()
    assert counts["trivial"] == 1
    assert counts["edit"] == 2
    assert counts["structural"] == 1
    assert counts["unchanged"] == 0
    assert counts["total"] == 4
