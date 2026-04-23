"""Backward-compatibility alias — use ``StorageTextIndex`` instead.

This module re-exports ``StorageTextIndex`` under the old name for
existing callers (integration tests, etc.).  New code should import
directly from ``app.core.storage.text_index``.
"""

from ..storage.text_index import StorageTextIndex as PostgresGraphTextIndex

__all__ = ["PostgresGraphTextIndex"]
