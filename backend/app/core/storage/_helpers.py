"""Private helpers shared between the SQLite and PostgreSQL storage backends.

Both ``SqliteWikiStorage`` and ``PostgresWikiStorage`` implement the same
:class:`WikiStorageProtocol`.  When their implementations need exactly the
same auxiliary logic (currently: row-cap truncation warnings) it should
live here so a change to the wording or threshold lands in one place.

Not part of the public API — leading underscore on the module name.
"""

from __future__ import annotations

import logging
from typing import Any

__all__ = ["warn_if_truncated"]

_default_logger = logging.getLogger(__name__)


def warn_if_truncated(
    rows: list[Any],
    limit: int | None,
    method: str,
    *,
    logger: logging.Logger | None = None,
    **ctx: Any,
) -> None:
    """Log a WARNING when *rows* count equals the active *limit*.

    The protocol's bounded fetchers (``get_architectural_nodes``,
    ``get_nodes_by_cluster``, ``get_architectural_node_ids``,
    ``get_all_edges``) silently truncate when the row count hits the cap.
    On large repos this can corrupt downstream graph reconstruction
    (PageRank, coverage analysis).  This helper makes the boundary visible
    in operator logs without changing behaviour.

    The optional *logger* keyword preserves per-backend attribution so
    operators can filter warnings by storage module name (e.g.
    ``app.core.storage.sqlite``).  When omitted, the warning is emitted
    against this module's logger.

    Callers that need *all* rows should pass ``limit=None``.
    """
    if limit is None:
        return
    if len(rows) >= limit:
        ctx_str = ", ".join(f"{k}={v}" for k, v in ctx.items() if v is not None)
        suffix = f" ({ctx_str})" if ctx_str else ""
        (logger or _default_logger).warning(
            "[STORAGE] %s hit row cap (limit=%d)%s; result is likely truncated. "
            "Pass limit=None when full coverage is required.",
            method, limit, suffix,
        )
