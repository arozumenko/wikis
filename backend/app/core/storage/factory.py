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


def repo_id_from_path(db_path: str | Path) -> str:
    """Extract a clean repo_id (cache key) from a ``*.wiki.db`` path.

    ``os.path.splitext`` only strips the last extension, so for
    ``abc123.wiki.db`` it returns ``abc123.wiki``.  Dots in the
    repo_id break PostgreSQL schema names, so we strip *both*
    suffixes.
    """
    name = Path(db_path).name          # e.g. "abc123.wiki.db"
    if name.endswith(".wiki.db"):
        return name[: -len(".wiki.db")]
    # fallback: strip all extensions
    stem = name.split(".")[0]
    return stem


def open_storage(
    *,
    repo_id: str,
    db_path: str | Path | None = None,
    embedding_dim: int = 1536,
    readonly: bool = False,
    settings: Any | None = None,
) -> WikiStorageProtocol:
    """High-level helper that reads ``wiki_storage_backend`` from settings.

    For SQLite: delegates to ``create_storage(backend="sqlite", db_path=…)``.
    For PostgreSQL: derives the DSN from settings and uses ``repo_id`` as the
    schema scope.

    Parameters
    ----------
    repo_id : str
        Identifier scoping the storage (cache_key for SQLite filename,
        schema prefix for PostgreSQL).
    db_path : Path, optional
        Full ``.wiki.db`` path (SQLite only). When omitted, the caller
        is expected to have already resolved it or the backend is postgres.
    embedding_dim : int
        Embedding dimension (default 1536).
    readonly : bool
        Open in read-only mode.
    settings : Settings, optional
        App settings — loaded lazily from ``get_settings()`` if not provided.
    """
    if settings is None:
        from ...config import get_settings
        settings = get_settings()

    backend = getattr(settings, "wiki_storage_backend", "sqlite").lower().strip()

    if backend in ("postgres", "postgresql"):
        dsn = getattr(settings, "wiki_storage_dsn", "")
        if not dsn:
            raise ValueError(
                "WIKI_STORAGE_DSN must be set when WIKI_STORAGE_BACKEND=postgres"
            )
        return create_storage(
            backend="postgres",
            dsn=dsn,
            repo_id=repo_id,
            embedding_dim=embedding_dim,
            readonly=readonly,
        )

    # Default: SQLite
    if db_path is None:
        raise ValueError("db_path is required for SQLite backend")
    return create_storage(
        backend="sqlite",
        db_path=db_path,
        embedding_dim=embedding_dim,
        readonly=readonly,
    )


def is_postgres_backend(settings: Any | None = None) -> bool:
    """Check if the configured wiki storage backend is PostgreSQL."""
    if settings is None:
        from ...config import get_settings
        settings = get_settings()
    return getattr(settings, "wiki_storage_backend", "sqlite").lower().strip() in (
        "postgres", "postgresql"
    )


def drop_storage(repo_id: str, settings: Any | None = None) -> None:
    """Drop all PostgreSQL data for a repository (schema CASCADE).

    No-op when the backend is SQLite (file cleanup is handled separately).
    """
    if settings is None:
        from ...config import get_settings
        settings = get_settings()

    if not is_postgres_backend(settings):
        return

    dsn = getattr(settings, "wiki_storage_dsn", "")
    if not dsn:
        return

    from .postgres import PostgresWikiStorage

    storage = PostgresWikiStorage(dsn, repo_id, readonly=True)
    try:
        storage.drop_schema()
    finally:
        storage.close()
