"""Migrate wikis.json flat-file registry to the SQLAlchemy database.

Usage
-----
    python scripts/migrate_wikis_json.py --default-owner <user_id>

Options
-------
--default-owner   User ID to assign as owner of all migrated wikis (required)
--storage-path    Path to local artifact storage (default: ./data/artifacts)
--database-url    Override DATABASE_URL (default: reads from environment / config)
--dry-run         Print what would be inserted without writing to the database

All migrated wikis are set to ``visibility="shared"`` because the previous
flat-file approach made every wiki visible to all users.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from datetime import datetime
from pathlib import Path

# Ensure backend package is importable when run from repo root or backend/
_backend_root = Path(__file__).parent.parent
sys.path.insert(0, str(_backend_root))


async def migrate(
    default_owner: str,
    storage_path: str,
    database_url: str,
    dry_run: bool,
) -> None:
    from sqlalchemy import select

    from app.db import create_tables, get_engine, get_session_factory, init_db
    from app.models.db_models import WikiRecord
    from app.storage.local import LocalArtifactStorage

    storage = LocalArtifactStorage(storage_path)

    # Load wikis.json from storage
    try:
        raw = await storage.download("wiki_registry", "wikis.json")
        registry: dict = json.loads(raw)
    except FileNotFoundError:
        print("No wikis.json found — nothing to migrate.")
        return

    print(f"Found {len(registry)} wiki(s) in wikis.json")

    if dry_run:
        for wiki_id, _entry in registry.items():
            print(f"  [dry-run] Would insert: {wiki_id!r} owner={default_owner!r} visibility=shared")
        print("Dry run complete — no data written.")
        return

    init_db(database_url)
    await create_tables(get_engine())
    session_factory = get_session_factory()

    inserted = 0
    skipped = 0

    async with session_factory() as session:
        async with session.begin():
            for wiki_id, entry in registry.items():
                result = await session.execute(select(WikiRecord).where(WikiRecord.id == wiki_id))
                existing = result.scalar_one_or_none()
                if existing is not None:
                    print(f"  skip (already exists): {wiki_id}")
                    skipped += 1
                    continue

                created_raw = entry.get("created_at")
                indexed_raw = entry.get("indexed_at")

                record = WikiRecord(
                    id=wiki_id,
                    owner_id=default_owner,
                    repo_url=entry.get("repo_url", ""),
                    branch=entry.get("branch", "main"),
                    title=entry.get("title", wiki_id),
                    page_count=entry.get("page_count", 0),
                    visibility="shared",
                    commit_hash=entry.get("commit_hash"),
                    created_at=datetime.fromisoformat(created_raw) if created_raw else datetime.now(),
                    updated_at=datetime.now(),
                    indexed_at=datetime.fromisoformat(indexed_raw) if indexed_raw else None,
                )
                session.add(record)
                inserted += 1
                print(f"  insert: {wiki_id!r} → owner={default_owner!r}, visibility=shared")

    print(f"\nMigration complete: {inserted} inserted, {skipped} skipped.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Migrate wikis.json to SQLAlchemy database")
    parser.add_argument("--default-owner", required=True, help="User ID to set as owner for all wikis")
    parser.add_argument("--storage-path", default="./data/artifacts", help="Local artifact storage path")
    parser.add_argument("--database-url", default="", help="Database URL (overrides DATABASE_URL env var)")
    parser.add_argument("--dry-run", action="store_true", help="Print actions without writing")
    args = parser.parse_args()

    database_url = args.database_url
    if not database_url:
        import os

        database_url = os.getenv("DATABASE_URL", "")

    asyncio.run(
        migrate(
            default_owner=args.default_owner,
            storage_path=args.storage_path,
            database_url=database_url,
            dry_run=args.dry_run,
        )
    )


if __name__ == "__main__":
    main()
