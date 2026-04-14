"""
Storage factory — create the right backend from configuration.

Usage::

    from app.core.storage import create_storage

    # SQLite (default)
    db = create_storage(db_path="/data/cache/abc123.wiki.db")

    # PostgreSQL
    db = create_storage(
        backend="postgres",
        dsn="postgresql://user:pass@localhost/wikis",
        repo_id="abc123",
    )
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from .protocol import WikiStorageProtocol

logger = logging.getLogger(__name__)


def create_storage(
    backend: str = "sqlite",
    *,
    # SQLite parameters
    db_path: str | Path | None = None,
    embedding_dim: int = 1536,
    readonly: bool = False,
    # PostgreSQL parameters
    dsn: str | None = None,
    repo_id: str | None = None,
    **kwargs: Any,
) -> WikiStorageProtocol:
    """Create a storage backend instance.

    Parameters
    ----------
    backend : str
        ``"sqlite"`` (default) or ``"postgres"``.
    db_path : Path, optional
        Path to the ``.wiki.db`` file (SQLite only).
    embedding_dim : int
        Dimensionality for vector embeddings (default 1536).
    readonly : bool
        Open in read-only mode.
    dsn : str, optional
        PostgreSQL connection string (postgres only).
    repo_id : str, optional
        Repository identifier for table scoping (postgres only).
    **kwargs
        Additional backend-specific keyword arguments.

    Returns
    -------
    WikiStorageProtocol
        Configured storage backend.
    """
    backend = backend.lower().strip()

    if backend == "sqlite":
        if db_path is None:
            raise ValueError("db_path is required for SQLite backend")

        from ..unified_db import UnifiedWikiDB

        return UnifiedWikiDB(
            db_path=db_path,
            embedding_dim=embedding_dim,
            readonly=readonly,
        )

    if backend in ("postgres", "postgresql"):
        if dsn is None:
            raise ValueError("dsn is required for PostgreSQL backend")
        if repo_id is None:
            raise ValueError("repo_id is required for PostgreSQL backend")

        from .postgres import PostgresWikiStorage

        return PostgresWikiStorage(
            dsn=dsn,
            repo_id=repo_id,
            embedding_dim=embedding_dim,
            readonly=readonly,
            **kwargs,
        )

    raise ValueError(
        f"Unknown storage backend: {backend!r}. Supported: 'sqlite', 'postgres'"
    )
