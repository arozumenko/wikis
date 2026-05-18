"""Phase 9 — Project lifecycle / staleness helpers.

Pure functions that compute when a project-level recompute is required
(relatedness, cross-repo linker, project clustering). Used by the
upstream rebuild trigger once the Phase 7/8 pipeline wiring lands.

Two staleness signals matter:

1. **Time-based** — project-level artifacts older than
   ``staleness_max_age_days`` (default ``7``) are stale.
2. **Constituent rebuild** — any wiki rebuilt **after** the last
   project recompute makes the project stale, regardless of age.

The functions are side-effect free and storage-agnostic; callers feed
in the relevant timestamps.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Iterable

__all__ = [
    "DEFAULT_STALENESS_DAYS",
    "max_wiki_built_at",
    "is_project_stale",
]


DEFAULT_STALENESS_DAYS = 7


def _ensure_tz(dt: datetime) -> datetime:
    """Treat naive timestamps as UTC."""
    return dt if dt.tzinfo is not None else dt.replace(tzinfo=timezone.utc)


def max_wiki_built_at(
    wiki_built_ats: Iterable[datetime],
) -> datetime | None:
    """Return the latest ``built_at`` across constituent wikis.

    Returns ``None`` when the iterable is empty.
    """
    latest: datetime | None = None
    for ts in wiki_built_ats:
        if ts is None:
            continue
        ts = _ensure_tz(ts)
        if latest is None or ts > latest:
            latest = ts
    return latest


def is_project_stale(
    *,
    project_recomputed_at: datetime | None,
    wiki_built_ats: Iterable[datetime] = (),
    now: datetime | None = None,
    max_age_days: int = DEFAULT_STALENESS_DAYS,
) -> tuple[bool, str]:
    """Return ``(stale, reason)``.

    Args:
        project_recomputed_at: Timestamp of the last successful
            project-level recompute. ``None`` means it never ran.
        wiki_built_ats: Iterable of ``built_at`` timestamps for the
            constituent wikis.
        now: Override for ``datetime.now(UTC)`` — useful in tests.
        max_age_days: Time-based staleness window. Values ``<= 0``
            disable the time-based check (only the rebuild check
            remains).

    Returns:
        ``(True, reason)`` when stale, otherwise ``(False, "fresh")``.
        ``reason`` is one of ``"never_computed"``, ``"age"``,
        ``"wiki_rebuilt"``, ``"fresh"``.
    """
    if project_recomputed_at is None:
        return True, "never_computed"

    project_recomputed_at = _ensure_tz(project_recomputed_at)
    now = _ensure_tz(now or datetime.now(timezone.utc))

    if max_age_days > 0:
        if now - project_recomputed_at > timedelta(days=max_age_days):
            return True, "age"

    latest_wiki = max_wiki_built_at(wiki_built_ats)
    if latest_wiki is not None and latest_wiki > project_recomputed_at:
        return True, "wiki_rebuilt"

    return False, "fresh"
