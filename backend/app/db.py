"""Database engine and session factory for async SQLAlchemy."""

from __future__ import annotations

import logging
import os
from pathlib import Path

from sqlalchemy.ext.asyncio import AsyncEngine, AsyncSession, async_sessionmaker, create_async_engine

logger = logging.getLogger(__name__)

# Module-level singletons — initialised by init_db()
_engine: AsyncEngine | None = None
_session_factory: async_sessionmaker[AsyncSession] | None = None


def _convert_database_url(url: str) -> str:
    """Convert a Prisma-style DATABASE_URL to a SQLAlchemy async URL.

    Handles three forms:
    - ``postgresql://...``  → ``postgresql+asyncpg://...``
    - ``file:path``         → ``sqlite+aiosqlite:///absolute_path``
    - empty / None          → ``sqlite+aiosqlite:///data/wikis.db`` (relative to backend root)
    """
    if not url:
        default_path = Path(__file__).parent.parent / "data" / "wikis.db"
        default_path.parent.mkdir(parents=True, exist_ok=True)
        return f"sqlite+aiosqlite:///{default_path}"

    if url.startswith("postgresql://"):
        return url.replace("postgresql://", "postgresql+asyncpg://", 1)

    if url.startswith("postgresql+asyncpg://"):
        return url  # already correct

    if url.startswith("file:"):
        raw_path = url[len("file:") :]
        # Resolve relative paths from the project root (two levels up from this file)
        project_root = Path(__file__).parent.parent.parent
        abs_path = (project_root / raw_path).resolve()
        abs_path.parent.mkdir(parents=True, exist_ok=True)
        return f"sqlite+aiosqlite:///{abs_path}"

    if url.startswith("sqlite"):
        # Already a valid SQLAlchemy URL — ensure async driver
        if "aiosqlite" not in url:
            return url.replace("sqlite://", "sqlite+aiosqlite://", 1)
        return url

    # Unknown scheme — return as-is and let SQLAlchemy report the error
    logger.warning("Unrecognised DATABASE_URL scheme, passing through unchanged: %s", url[:40])
    return url


def init_db(database_url: str) -> None:
    """Initialise the async engine and session factory from *database_url*.

    Safe to call multiple times; subsequent calls are no-ops unless the
    engine has been disposed.
    """
    global _engine, _session_factory

    if _engine is not None:
        return

    converted = _convert_database_url(database_url)
    logger.info("Initialising database engine: %s", converted[:60])

    connect_args: dict = {}
    if "sqlite" in converted:
        connect_args["check_same_thread"] = False

    _engine = create_async_engine(
        converted,
        echo=os.getenv("SQL_ECHO", "false").lower() == "true",
        connect_args=connect_args,
    )
    _session_factory = async_sessionmaker(_engine, expire_on_commit=False)


async def create_tables(engine: AsyncEngine) -> None:
    """Create all tables that don't yet exist and add missing columns (idempotent)."""
    from app.models.db_models import Base

    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

    # Auto-migrate: add columns that create_all can't add to existing tables.
    # Each statement is idempotent (IF NOT EXISTS / OR IGNORE).
    await _add_missing_columns(engine)
    logger.info("Database tables ensured")


async def _add_missing_columns(engine: AsyncEngine) -> None:
    """Add columns introduced after initial table creation."""
    from sqlalchemy import text

    is_pg = "postgresql" in str(engine.url)

    migrations = [
        ("wiki", "status", "VARCHAR", "'complete'"),
        ("wiki", "requires_token", "INTEGER", "0"),
        ("wiki", "error", "VARCHAR", None),
        ("wiki", "description", "VARCHAR", None),
    ]

    # All identifiers are hardcoded constants — no user input in SQL.
    async with engine.begin() as conn:
        for table, column, col_type, default in migrations:
            assert all(c.isalnum() or c == "_" for c in column), f"Invalid column name: {column}"
            default_clause = f" DEFAULT {default}" if default else ""
            if is_pg:
                await conn.execute(text(
                    f"ALTER TABLE {table} ADD COLUMN IF NOT EXISTS {column} {col_type}{default_clause}"
                ))
            else:
                # SQLite: no IF NOT EXISTS — catch only "duplicate column" errors
                try:
                    await conn.execute(text(
                        f"ALTER TABLE {table} ADD COLUMN {column} {col_type}{default_clause}"
                    ))
                except Exception as e:
                    if "duplicate column" in str(e).lower():
                        pass  # Column already exists
                    else:
                        logger.warning("Migration failed for %s.%s: %s", table, column, e)


def get_engine() -> AsyncEngine:
    """Return the module-level engine (must call init_db first)."""
    if _engine is None:
        raise RuntimeError("Database not initialised — call init_db() first")
    return _engine


def get_session_factory() -> async_sessionmaker[AsyncSession]:
    """Return the module-level session factory (must call init_db first)."""
    if _session_factory is None:
        raise RuntimeError("Database not initialised — call init_db() first")
    return _session_factory


async def dispose_engine() -> None:
    """Dispose the engine on shutdown, releasing all connections."""
    global _engine, _session_factory
    if _engine is not None:
        await _engine.dispose()
        _engine = None
        _session_factory = None
        logger.info("Database engine disposed")
