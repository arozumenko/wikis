"""Toolkit bridge — builds engine components from cached unified wiki DB.

Shared by AskService and ResearchService. Uses cache_index.json to find
the correct .wiki.db for a specific wiki's repository.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import time
from collections import OrderedDict
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from app.config import Settings
from app.storage.base import ArtifactStorage

logger = logging.getLogger(__name__)


@dataclass
class EngineComponents:
    """Components needed by AskEngine and DeepResearchEngine."""

    retriever_stack: Any = None  # UnifiedRetriever
    graph_manager: Any = None  # GraphManager
    code_graph: Any = None  # NetworkX DiGraph (legacy; None when storage is loaded directly)
    storage: Any = None  # WikiStorageProtocol (UnifiedWikiDB / PostgresWikiStorage)
    repo_analysis: dict | None = None
    llm: Any = None  # BaseChatModel
    repo_path: str | None = None  # Path to cloned repo (if still on disk)
    unified_db_path: str | None = None  # Path to .wiki.db
    query_service: Any = None  # GraphQueryService / StorageQueryService / MultiGraphQueryService


class ComponentCache:
    """LRU + TTL cache for EngineComponents.

    Evicts least-recently-used entries when the cache exceeds ``max_size``,
    and expires entries older than ``ttl_seconds``.

    The ``get_or_build`` method prevents duplicate concurrent loads for the
    same key using per-key ``asyncio.Lock``s.
    """

    def __init__(
        self,
        max_size: int = 5,
        ttl_seconds: int = 3600,
    ) -> None:
        self._entries: OrderedDict[str, tuple[float, EngineComponents]] = OrderedDict()
        self.max_size = max_size
        self.ttl_seconds = ttl_seconds
        self._locks: dict[str, asyncio.Lock] = {}

    def get(self, key: str) -> EngineComponents | None:
        """Return cached components if present and not expired, else None."""
        entry = self._entries.get(key)
        if entry is None:
            return None
        ts, components = entry
        if time.monotonic() - ts > self.ttl_seconds:
            logger.info("Cache expired for wiki %s (%.0fs old)", key, time.monotonic() - ts)
            del self._entries[key]
            return None
        # Move to end (most recently used)
        self._entries.move_to_end(key)
        return components

    def put(self, key: str, components: EngineComponents) -> None:
        """Store components, evicting LRU entry if at capacity."""
        if key in self._entries:
            self._entries.move_to_end(key)
            self._entries[key] = (time.monotonic(), components)
            return
        if len(self._entries) >= self.max_size:
            evicted_key, _ = self._entries.popitem(last=False)
            logger.info("Cache evicted wiki %s (capacity %d)", evicted_key, self.max_size)
        self._entries[key] = (time.monotonic(), components)

    def evict(self, key: str) -> bool:
        """Remove a specific key from the cache. Returns True if it was present."""
        if key in self._entries:
            del self._entries[key]
            self._locks.pop(key, None)
            logger.info("Cache evicted wiki %s (explicit)", key)
            return True
        return False

    async def get_or_build(
        self,
        key: str,
        factory: Callable[[], Awaitable[EngineComponents]],
    ) -> EngineComponents:
        """Return cached components or build them, preventing duplicate concurrent loads."""
        cached = self.get(key)
        if cached is not None:
            return cached
        if key not in self._locks:
            self._locks[key] = asyncio.Lock()
        async with self._locks[key]:
            cached = self.get(key)
            if cached is not None:
                return cached
            components = await factory()
            self.put(key, components)
            return components

    def __len__(self) -> int:
        return len(self._entries)


async def build_engine_components(
    wiki_id: str,
    storage: ArtifactStorage,
    settings: Settings,
    tier: str = "high",
) -> EngineComponents:
    """Build engine components from cached unified wiki DB.

    Locates the ``.wiki.db`` file for this wiki's repo via cache_index.json,
    creates an ``UnifiedRetriever`` for retrieval, and reconstructs the NX
    code graph from the DB if needed.

    Raises FileNotFoundError if wiki doesn't exist.
    """
    # Get wiki metadata from database
    from app.db import get_session_factory
    from app.models.db_models import WikiRecord

    wiki_meta: dict | None = None
    try:
        session_factory = get_session_factory()
        if session_factory:
            from sqlalchemy import select

            async with session_factory() as session:
                result = await session.execute(select(WikiRecord).where(WikiRecord.id == wiki_id))
                record = result.scalar_one_or_none()
            if record:
                wiki_meta = {
                    "repo_url": record.repo_url,
                    "branch": record.branch,
                    "commit_hash": record.commit_hash,
                    "description": record.description,
                }
    except Exception:
        wiki_meta = None

    if not wiki_meta:
        # Verify wiki exists in storage at all
        try:
            artifacts = await storage.list_artifacts("wiki_artifacts", prefix=wiki_id)
            if not artifacts:
                raise FileNotFoundError(f"Wiki not found: {wiki_id}")
        except Exception as e:
            raise FileNotFoundError(f"Wiki not found: {wiki_id}") from e

    # Build LLM
    from app.services.llm_factory import create_llm

    llm = create_llm(settings, tier=tier, skip_retry=True)

    components = EngineComponents(llm=llm)

    # Find correct cache key from repo info
    repo_url = wiki_meta.get("repo_url", "") if wiki_meta else ""
    branch = wiki_meta.get("branch", "main") if wiki_meta else "main"

    # Extract owner/repo from URL
    repo_identifier = _extract_repo_identifier(repo_url, branch)

    await asyncio.to_thread(_load_cached_artifacts, components, settings.cache_dir, wiki_id, repo_identifier, settings)

    # Inject user-provided wiki description into repo_analysis
    wiki_description = wiki_meta.get("description") if wiki_meta else None
    if wiki_description:
        if components.repo_analysis is None:
            components.repo_analysis = {}
        components.repo_analysis["description"] = wiki_description

    # Ask/Research run storage-native: no NX rehydration.  A tiny empty graph
    # is kept only for legacy callers that still check ``number_of_nodes()``.
    if components.code_graph is None:
        import networkx as nx

        components.code_graph = nx.DiGraph()

    # Try to locate the cloned repo on disk for direct file access
    if wiki_meta:
        components.repo_path = _find_repo_clone(
            settings.cache_dir,
            repo_url,
            branch,
            wiki_meta.get("commit_hash"),
        )

    return components


def _find_repo_clone(
    cache_dir: str,
    repo_url: str,
    branch: str,
    commit_hash: str | None = None,
) -> str | None:
    """Find the cloned repo directory on disk.

    LocalRepositoryManager (used by FilesystemIndexer) clones into the
    ``repositories/`` subdirectory of the cache:
      {cache_dir}/repositories/{owner_repo}_{branch}_{commit_short}

    Older deployments may have placed clones directly under ``cache_dir``,
    so we check both layouts. Returns the path if it exists, else None.
    """
    import os

    url = repo_url.rstrip("/")
    if url.endswith(".git"):
        url = url[:-4]
    parts = url.split("/")
    if len(parts) >= 2:
        safe_name = f"{parts[-2]}_{parts[-1]}"
    else:
        safe_name = url.replace("/", "_")

    # Search both the canonical "repositories/" subdir and the legacy
    # cache_dir root.
    search_roots = [os.path.join(cache_dir, "repositories"), cache_dir]

    # Exact match with commit hash
    if commit_hash:
        for root in search_roots:
            candidate = os.path.join(root, f"{safe_name}_{branch}_{commit_hash[:8]}")
            if os.path.isdir(candidate):
                logger.info("Found repo clone at %s", candidate)
                return candidate

    # Fallback: glob for any clone matching owner_repo_branch_*
    import glob as glob_mod

    for root in search_roots:
        pattern = os.path.join(root, f"{safe_name}_{branch}_*")
        matches = sorted(glob_mod.glob(pattern), key=os.path.getmtime, reverse=True)
        if matches and os.path.isdir(matches[0]):
            logger.info("Found repo clone (glob fallback) at %s", matches[0])
            return matches[0]

    logger.info("No repo clone found for %s branch=%s", repo_url, branch)
    return None


def _extract_repo_identifier(repo_url: str, branch: str) -> str:
    """Extract owner/repo:branch from a repo URL."""
    # https://github.com/owner/repo → owner/repo
    url = repo_url.rstrip("/")
    if url.endswith(".git"):
        url = url[:-4]
    parts = url.split("/")
    if len(parts) >= 2:
        owner_repo = f"{parts[-2]}/{parts[-1]}"
    else:
        owner_repo = url
    return f"{owner_repo}:{branch}"


def _load_cached_artifacts(
    components: EngineComponents,
    cache_dir: str,
    wiki_id: str,
    repo_identifier: str,
    settings: Any = None,
) -> None:
    """Load unified wiki DB and create UnifiedRetriever for Ask/Research.

    Looks up cache_index.json to find the correct ``.wiki.db`` file, then
    creates an ``UnifiedRetriever``.
    Also reconstructs the NX code graph from the DB for graph-based tools.
    """
    cache_path = Path(cache_dir)
    index_file = cache_path / "cache_index.json"

    cache_key = None

    # Load cache index to find unified DB key
    if index_file.exists():
        try:
            with open(index_file) as f:
                index = json.load(f)

            # Resolve ref: owner/repo:branch → owner/repo:branch:commit
            resolved = index.get("refs", {}).get(repo_identifier, repo_identifier)

            # Find unified DB cache key
            udb_section = index.get("unified_db", {})
            cache_key = udb_section.get(resolved) or udb_section.get(repo_identifier)

            # Fallback: try all entries that match prefix
            if not cache_key:
                for k, v in udb_section.items():
                    if k.startswith(repo_identifier) or repo_identifier.startswith(k.rsplit(":", 1)[0]):
                        cache_key = v
                        break

            logger.info(
                "Cache index lookup: repo=%s resolved=%s unified_db=%s",
                repo_identifier,
                resolved,
                cache_key,
            )
        except Exception as e:
            logger.warning(f"Failed to read cache_index.json: {e}")

    # If no cache_index entry, try glob for .wiki.db files
    if not cache_key:
        import glob as glob_mod

        wiki_dbs = glob_mod.glob(str(cache_path / "*.wiki.db"))
        if wiki_dbs:
            # Use the most recent one
            wiki_dbs.sort(key=os.path.getmtime, reverse=True)
            from app.core.storage import repo_id_from_path
            cache_key = repo_id_from_path(wiki_dbs[0])
            logger.info("Found .wiki.db via glob fallback: %s", wiki_dbs[0])

    if not cache_key:
        logger.warning(f"No unified DB found for {repo_identifier} — ask/research may fail")
        return

    # Open unified DB and create retriever
    db_path = cache_path / f"{cache_key}.wiki.db"

    try:
        from app.core.storage import is_postgres_backend, open_storage
        from app.core.unified_retriever import UnifiedRetriever

        if is_postgres_backend():
            db = open_storage(repo_id=cache_key, db_path=str(db_path), readonly=True)
        else:
            if not db_path.exists():
                logger.warning(f"Unified DB file not found: {db_path}")
                return
            db = open_storage(repo_id=cache_key, db_path=str(db_path), readonly=True)

        components.unified_db_path = str(db_path)

        # Create embedding function for hybrid search
        embedding_fn = None
        embeddings = None
        if settings:
            try:
                from app.services.llm_factory import create_embeddings

                embeddings = create_embeddings(settings)
                if hasattr(embeddings, "embed_query"):
                    embedding_fn = embeddings.embed_query
            except Exception as e:
                logger.warning(f"Failed to create embeddings for Ask retriever: {e}")

        components.retriever_stack = UnifiedRetriever(
            db=db,
            embedding_fn=embedding_fn,
            embeddings=embeddings,
        )

        # Expose the storage handle directly — ask/research tools use
        # ``StorageQueryService`` + ``StorageTextIndex`` instead of a
        # rehydrated NX graph.
        components.storage = db

        # Build a storage-native query service so research/ask tools can
        # treat single- and multi-wiki components uniformly.
        try:
            from app.core.code_graph.storage_query_service import StorageQueryService

            components.query_service = StorageQueryService(db)
        except Exception as e:  # pragma: no cover — defensive
            logger.warning("Failed to build StorageQueryService for %s: %s", cache_key, e)

        # Load repository analysis from unified DB (Ask tool context)
        raw_analysis = db.get_meta("repository_analysis")
        if raw_analysis:
            try:
                components.repo_analysis = json.loads(raw_analysis)
            except (json.JSONDecodeError, TypeError):
                # Analysis may be plain text rather than JSON
                components.repo_analysis = {"summary": raw_analysis}

        stats = db.stats()
        logger.info(
            "Loaded unified DB %s (%d nodes, %d edges, %.1f MB)",
            cache_key,
            stats["node_count"],
            stats["edge_count"],
            stats["db_size_mb"],
        )
    except Exception as e:
        logger.warning(f"Failed to load unified DB {cache_key}: {e}")
