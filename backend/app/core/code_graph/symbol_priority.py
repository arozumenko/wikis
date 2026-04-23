"""Shared architectural type-priority table.

Extracted into its own module so that storage backends (``UnifiedWikiDB``,
``PostgresWikiStorage``) and the ``StorageQueryService`` can reuse the
same priority ordering as ``attach_graph_indexes`` / ``GraphQueryService``
without pulling in the heavy tree-sitter parser dependencies from
``graph_builder``.

Higher = more architecturally significant.  Kept in sync with
``graph_builder._GRAPH_INDEX_TYPE_PRIORITY`` (single source of truth lives
here; ``graph_builder`` re-exports it).
"""

from __future__ import annotations

GRAPH_INDEX_TYPE_PRIORITY: dict[str, int] = {
    "class": 10, "interface": 10, "trait": 10, "protocol": 10,
    "enum": 9, "struct": 9, "record": 9, "data_class": 9, "object": 9,
    "module": 8, "namespace": 8,
    "function": 7,
    "constant": 6, "type_alias": 6, "annotation": 6, "decorator": 6, "macro": 6,
    "method": 3,
    "constructor": 2, "field": 2, "property": 2,
    "parameter": 1, "variable": 1, "local_variable": 1, "argument": 1,
    "unknown": 0,
}


def symbol_priority(symbol_type: str | None) -> int:
    """Return the priority for a symbol type string (case-insensitive)."""
    if not symbol_type:
        return 0
    return GRAPH_INDEX_TYPE_PRIORITY.get(str(symbol_type).lower(), 0)
