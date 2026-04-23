"""Backward-compatible import shim for the SQLite storage backend.

The real SQLite implementation now lives in ``app.core.storage.sqlite``.
This module remains only to keep older imports working while callers are
moved to ``app.core.storage`` / ``open_storage``.
"""

from .storage.sqlite import SqliteWikiStorage, UNIFIED_DB_ENABLED, UnifiedWikiDB

__all__ = [
    "SqliteWikiStorage",
    "UnifiedWikiDB",
    "UNIFIED_DB_ENABLED",
]
