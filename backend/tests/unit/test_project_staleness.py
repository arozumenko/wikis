"""Phase 9 — Project staleness helper."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from app.core.project_lifecycle import (
    DEFAULT_STALENESS_DAYS,
    is_project_stale,
    max_wiki_built_at,
)


_NOW = datetime(2025, 1, 15, 12, 0, 0, tzinfo=timezone.utc)


class TestMaxBuiltAt:
    def test_empty_returns_none(self):
        assert max_wiki_built_at([]) is None

    def test_picks_latest(self):
        a = datetime(2025, 1, 1, tzinfo=timezone.utc)
        b = datetime(2025, 1, 10, tzinfo=timezone.utc)
        assert max_wiki_built_at([a, b]) == b

    def test_naive_treated_as_utc(self):
        naive = datetime(2025, 1, 12, 0, 0, 0)  # No tz.
        out = max_wiki_built_at([naive])
        assert out.tzinfo == timezone.utc

    def test_skips_none(self):
        a = datetime(2025, 1, 1, tzinfo=timezone.utc)
        assert max_wiki_built_at([None, a, None]) == a


class TestIsStale:
    def test_never_computed(self):
        stale, reason = is_project_stale(
            project_recomputed_at=None, wiki_built_ats=[], now=_NOW
        )
        assert stale and reason == "never_computed"

    def test_fresh_within_window(self):
        stale, reason = is_project_stale(
            project_recomputed_at=_NOW - timedelta(days=1),
            wiki_built_ats=[_NOW - timedelta(days=2)],
            now=_NOW,
        )
        assert not stale and reason == "fresh"

    def test_age_threshold(self):
        stale, reason = is_project_stale(
            project_recomputed_at=_NOW - timedelta(days=DEFAULT_STALENESS_DAYS + 1),
            wiki_built_ats=[],
            now=_NOW,
        )
        assert stale and reason == "age"

    def test_wiki_rebuilt_after_recompute(self):
        recomputed = _NOW - timedelta(days=2)
        rebuilt   = _NOW - timedelta(hours=1)
        stale, reason = is_project_stale(
            project_recomputed_at=recomputed,
            wiki_built_ats=[rebuilt],
            now=_NOW,
        )
        assert stale and reason == "wiki_rebuilt"

    def test_age_check_disabled(self):
        # max_age_days=0 disables the age check; only the rebuild check
        # remains. With no rebuild, project is considered fresh.
        stale, reason = is_project_stale(
            project_recomputed_at=_NOW - timedelta(days=365),
            wiki_built_ats=[],
            now=_NOW,
            max_age_days=0,
        )
        assert not stale and reason == "fresh"

    def test_age_wins_over_rebuild_priority(self):
        # When BOTH age and rebuild trigger, age comes first per the
        # function's check order — pin the contract.
        stale, reason = is_project_stale(
            project_recomputed_at=_NOW - timedelta(days=DEFAULT_STALENESS_DAYS + 1),
            wiki_built_ats=[_NOW - timedelta(hours=1)],
            now=_NOW,
        )
        assert stale and reason == "age"
