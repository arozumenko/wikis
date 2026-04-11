"""Wiki management service — list and delete generated wikis.

Metadata is stored in a relational database via SQLAlchemy async.
Wiki artifact files (markdown pages) continue to live in ArtifactStorage.
"""

from __future__ import annotations

import logging
from datetime import datetime

from sqlalchemy import delete, select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from app.models.api import DeleteWikiResponse, WikiListResponse, WikiSummary
from app.models.db_models import WikiRecord
from app.storage.base import ArtifactStorage

logger = logging.getLogger(__name__)


class WikiManagementService:
    """Handles wiki listing, deletion, and metadata via a SQLAlchemy async session."""

    def __init__(self, storage: ArtifactStorage, session_factory: async_sessionmaker[AsyncSession]) -> None:
        self.storage = storage
        self.session_factory = session_factory

    # ------------------------------------------------------------------
    # Write
    # ------------------------------------------------------------------

    async def register_wiki(
        self,
        wiki_id: str,
        repo_url: str,
        branch: str,
        title: str,
        page_count: int,
        owner_id: str = "",
        visibility: str = "personal",
        commit_hash: str | None = None,
        indexed_at: datetime | None = None,
        status: str = "complete",
        requires_token: bool = False,
        error: str | None = None,
    ) -> None:
        """Upsert a wiki record into the database.

        Called at generation START (status='generating') and again on
        completion/failure so the record always exists for retry/404 recovery.
        """
        async with self.session_factory() as session:
            async with session.begin():
                result = await session.execute(select(WikiRecord).where(WikiRecord.id == wiki_id))
                record = result.scalar_one_or_none()

                now = datetime.now()
                if record is None:
                    record = WikiRecord(
                        id=wiki_id,
                        owner_id=owner_id,
                        repo_url=repo_url,
                        branch=branch,
                        title=title,
                        page_count=page_count,
                        visibility=visibility,
                        commit_hash=commit_hash,
                        indexed_at=indexed_at if indexed_at is not None else (now if status == "complete" else None),
                        created_at=now,
                        updated_at=now,
                        status=status,
                        requires_token=1 if requires_token else 0,
                        error=error,
                    )
                    session.add(record)
                else:
                    record.repo_url = repo_url
                    record.branch = branch
                    record.title = title
                    record.page_count = page_count
                    record.commit_hash = commit_hash
                    if indexed_at is not None or status == "complete":
                        record.indexed_at = indexed_at or now
                    record.updated_at = now
                    record.status = status
                    record.error = error
                    # Only update requires_token when explicitly set (not on failure updates)
                    if requires_token:
                        record.requires_token = 1
                    # Preserve owner/visibility on re-generation — only update when provided
                    if owner_id:
                        record.owner_id = owner_id
                    if visibility != "personal" or status == "generating":
                        # Only overwrite visibility when explicitly non-default or on fresh start
                        if visibility != "personal":
                            record.visibility = visibility

        logger.info("Registered wiki: %s (owner=%s, status=%s)", wiki_id, owner_id, status)

    async def update_description(
        self,
        wiki_id: str,
        user_id: str,
        description: str | None,
    ) -> WikiRecord | None:
        """Update the user-provided description for a wiki. Only the owner may do this.

        Returns the updated WikiRecord, or None if not found / caller is not the owner.
        """
        async with self.session_factory() as session:
            async with session.begin():
                result = await session.execute(select(WikiRecord).where(WikiRecord.id == wiki_id))
                record = result.scalar_one_or_none()
                if record is None:
                    return None
                if record.owner_id and record.owner_id != user_id:
                    return None
                record.description = description
                record.updated_at = datetime.now()
                # Refresh so returned object has the new values detached from session
                await session.flush()
                await session.refresh(record)

        return record

    async def update_visibility(
        self,
        wiki_id: str,
        user_id: str,
        visibility: str,
    ) -> WikiSummary | None:
        """Change a wiki's visibility.  Only the owner may do this.

        Returns the updated WikiSummary, or None if the wiki doesn't exist or
        the caller is not the owner.
        """
        async with self.session_factory() as session:
            async with session.begin():
                result = await session.execute(select(WikiRecord).where(WikiRecord.id == wiki_id))
                record = result.scalar_one_or_none()
                if record is None:
                    return None
                if record.owner_id and record.owner_id != user_id:
                    return None  # caller checks and returns 403
                # Claim ownership of legacy wikis (empty owner_id)
                if not record.owner_id:
                    record.owner_id = user_id
                record.visibility = visibility
                record.updated_at = datetime.now()
                # Build summary while session is still open
                summary = self._record_to_summary(record, user_id)

        return summary

    # ------------------------------------------------------------------
    # Read
    # ------------------------------------------------------------------

    async def list_wikis(self, user_id: str | None = None) -> WikiListResponse:
        """List wikis visible to *user_id*.

        - Authenticated user: own wikis + shared wikis + legacy wikis (no owner set).
        - None (MCP / system): shared wikis + legacy wikis only.

        Legacy wikis (owner_id == "") are visible to everyone for backward compatibility
        with the previous flat-file registry which had no ownership concept.
        """
        async with self.session_factory() as session:
            if user_id:
                stmt = select(WikiRecord).where(
                    (WikiRecord.owner_id == user_id) | (WikiRecord.visibility == "shared") | (WikiRecord.owner_id == "")
                )
            else:
                stmt = select(WikiRecord).where((WikiRecord.visibility == "shared") | (WikiRecord.owner_id == ""))

            result = await session.execute(stmt)
            records = result.scalars().all()

        wikis = [self._record_to_summary(r, user_id) for r in records]
        return WikiListResponse(wikis=wikis)

    async def get_wiki_record(self, wiki_id: str) -> WikiRecord | None:
        """Return the raw wiki record without access control. For internal checks only."""
        async with self.session_factory() as session:
            result = await session.execute(select(WikiRecord).where(WikiRecord.id == wiki_id))
            return result.scalar_one_or_none()

    async def get_wiki(self, wiki_id: str, user_id: str | None = None) -> WikiRecord | None:
        """Return the wiki record if the caller may view it, else None."""
        async with self.session_factory() as session:
            result = await session.execute(select(WikiRecord).where(WikiRecord.id == wiki_id))
            record = result.scalar_one_or_none()

        if record is None:
            return None
        # Legacy wikis (no owner) are accessible to everyone
        if not record.owner_id:
            return record
        if user_id and (record.owner_id == user_id or record.visibility == "shared"):
            return record
        if user_id is None and record.visibility == "shared":
            return record
        return None

    # ------------------------------------------------------------------
    # Delete
    # ------------------------------------------------------------------

    async def delete_wiki(
        self,
        wiki_id: str,
        user_id: str | None = None,
        cache_dir: str | None = None,
    ) -> DeleteWikiResponse:
        """Delete a wiki and its artifacts.

        If *user_id* is provided, only the owner may delete.
        Returns ``deleted=False`` with a message when the caller lacks permission.
        """
        async with self.session_factory() as session:
            result = await session.execute(select(WikiRecord).where(WikiRecord.id == wiki_id))
            record = result.scalar_one_or_none()

        if record is None:
            return DeleteWikiResponse(deleted=False, wiki_id=wiki_id)

        if user_id and record.owner_id and record.owner_id != user_id:
            return DeleteWikiResponse(
                deleted=False,
                wiki_id=wiki_id,
                message="Only the wiki owner can delete this wiki",
            )

        # Delete artifact files from object storage
        try:
            artifacts = await self.storage.list_artifacts("wiki_artifacts", prefix=wiki_id)
            for name in artifacts:
                await self.storage.delete("wiki_artifacts", name)
        except Exception:
            logger.warning("Failed to delete some artifacts for wiki %s", wiki_id)

        # Clean up cache files (FAISS vectors, code graphs, cloned repos)
        if cache_dir and record.repo_url:
            import asyncio

            await asyncio.to_thread(_cleanup_cache_files, cache_dir, record.repo_url, record.branch or "main")

        # Remove the metadata record
        async with self.session_factory() as session:
            async with session.begin():
                await session.execute(delete(WikiRecord).where(WikiRecord.id == wiki_id))

        logger.info("Deleted wiki: %s", wiki_id)
        return DeleteWikiResponse(deleted=True, wiki_id=wiki_id)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _record_to_summary(record: WikiRecord, user_id: str | None) -> WikiSummary:
        return WikiSummary(
            wiki_id=record.id,
            repo_url=record.repo_url,
            branch=record.branch,
            title=record.title,
            page_count=record.page_count or 0,
            created_at=record.created_at or datetime.now(),
            commit_hash=record.commit_hash,
            indexed_at=record.indexed_at,
            owner_id=record.owner_id,
            visibility=record.visibility or "personal",
            is_owner=(user_id is not None and (record.owner_id == user_id or not record.owner_id)),
            status=record.status or "complete",
            requires_token=bool(record.requires_token),
            error=record.error,
            description=record.description,
        )


def _derive_cache_key(cache_dir: str, repo_url: str, branch: str) -> str | None:
    """Read cache_index.json and return the cache_key for this repo+branch, or None.

    The cache key is the opaque string stored in cache_index.json that maps a
    repo+branch identifier to its set of on-disk FAISS/BM25/graph cache files.

    Args:
        cache_dir: Directory that contains ``cache_index.json``.
        repo_url: Full URL of the repository (``https://github.com/org/repo``).
        branch: Branch name (e.g. ``"main"``).

    Returns:
        The cache key string, or ``None`` if not found.
    """
    import json
    from pathlib import Path

    url = repo_url.rstrip("/")
    if url.endswith(".git"):
        url = url[:-4]
    parts = url.split("/")

    owner_repo = f"{parts[-2]}/{parts[-1]}" if len(parts) >= 2 else url
    repo_identifier = f"{owner_repo}:{branch}"

    index_file = Path(cache_dir) / "cache_index.json"
    if not index_file.exists():
        return None

    try:
        with open(index_file) as f:
            index = json.load(f)
    except Exception:
        return None

    resolved = index.get("refs", {}).get(repo_identifier, repo_identifier)
    cache_key = index.get(resolved) or index.get(repo_identifier)
    return cache_key or None


def _cleanup_cache_files(cache_dir: str, repo_url: str, branch: str) -> None:
    """Remove FAISS vectors, code graphs, and cloned repos for a wiki."""
    import glob as glob_mod
    import json
    import os
    import shutil
    from pathlib import Path

    cache_path = Path(cache_dir)

    # Derive repo identifier (same logic as toolkit_bridge._extract_repo_identifier)
    url = repo_url.rstrip("/")
    if url.endswith(".git"):
        url = url[:-4]
    parts = url.split("/")
    if len(parts) >= 2:
        safe_name = f"{parts[-2]}_{parts[-1]}"
    else:
        safe_name = url.replace("/", "_")

    # 1. Remove cloned repo directories: {cache_dir}/{owner_repo}_{branch}_*
    for d in glob_mod.glob(os.path.join(cache_dir, f"{safe_name}_{branch}_*")):
        if os.path.isdir(d):
            try:
                shutil.rmtree(d)
                logger.info("Deleted cloned repo: %s", d)
            except Exception as e:
                logger.warning("Failed to delete repo clone %s: %s", d, e)

    # 2. Remove FAISS + graph cache files using cache_index.json
    index_file = cache_path / "cache_index.json"
    if not index_file.exists():
        return

    try:
        with open(index_file) as f:
            index = json.load(f)
    except Exception:
        return

    owner_repo = f"{parts[-2]}/{parts[-1]}" if len(parts) >= 2 else url
    repo_identifier = f"{owner_repo}:{branch}"
    resolved = index.get("refs", {}).get(repo_identifier, repo_identifier)

    # Find and delete vectorstore files
    cache_key = index.get(resolved) or index.get(repo_identifier)
    if cache_key:
        for ext in (".faiss", ".docs.pkl"):
            f = cache_path / f"{cache_key}{ext}"
            if f.exists():
                try:
                    f.unlink()
                    logger.info("Deleted cache file: %s", f)
                except Exception as e:
                    logger.warning("Failed to delete %s: %s", f, e)

    # Find and delete graph files
    dirty = False
    graphs = index.get("graphs", {})
    for k, v in list(graphs.items()):
        if k.startswith(resolved) or k.startswith(repo_identifier):
            graph_file = cache_path / f"{v}.code_graph.gz"
            if graph_file.exists():
                try:
                    graph_file.unlink()
                    logger.info("Deleted graph cache: %s", graph_file)
                except Exception as e:
                    logger.warning("Failed to delete %s: %s", graph_file, e)
            del graphs[k]
            dirty = True

    # Remove vectorstore entry from index
    refs = index.get("refs", {})
    if repo_identifier in refs:
        del refs[repo_identifier]
        dirty = True
    for key in (resolved, repo_identifier):
        if key in index:
            del index[key]
            dirty = True

    # Persist updated index so there are no ghost references
    if dirty:
        try:
            with open(index_file, "w") as f:
                json.dump(index, f, indent=2)
            logger.info("Updated cache_index.json after cleanup for %s", repo_identifier)
        except Exception as e:
            logger.warning("Failed to update cache_index.json: %s", e)
