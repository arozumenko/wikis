"""Toolkit bridge — builds engine components from cached index artifacts.

Shared by AskService and ResearchService. Uses cache_index.json to find
the correct vectorstore/graph for a specific wiki's repository.
"""

from __future__ import annotations

import asyncio
import json
import logging
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

    retriever_stack: Any = None  # WikiRetrieverStack
    graph_manager: Any = None  # GraphManager
    code_graph: Any = None  # NetworkX DiGraph
    repo_analysis: dict | None = None
    llm: Any = None  # BaseChatModel
    repo_path: str | None = None  # Path to cloned repo (if still on disk)


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
    """Build engine components from cached index artifacts.

    Uses cache_index.json to find the correct FAISS/graph for this wiki's repo.

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

    # Ensure code_graph exists (empty graph for small repos without one)
    if components.code_graph is None:
        import networkx as nx

        components.code_graph = nx.DiGraph()
        logger.info(f"Created empty code graph for {wiki_id} (small repo)")

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

    Uses the same path convention as LocalRepositoryManager:
      {cache_dir}/{owner_repo}_{branch}_{commit_short}

    Returns the path if it exists, else None.
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

    # Exact match with commit hash
    if commit_hash:
        candidate = os.path.join(cache_dir, f"{safe_name}_{branch}_{commit_hash[:8]}")
        if os.path.isdir(candidate):
            logger.info("Found repo clone at %s", candidate)
            return candidate

    # Fallback: glob for any clone matching owner_repo_branch_*
    import glob as glob_mod

    pattern = os.path.join(cache_dir, f"{safe_name}_{branch}_*")
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
    """Load vectorstore and graph using cache_index.json to find correct files."""
    cache_path = Path(cache_dir)
    index_file = cache_path / "cache_index.json"

    cache_key = None
    graph_key = None

    # Load cache index to find correct keys
    if index_file.exists():
        try:
            with open(index_file) as f:
                index = json.load(f)

            # Resolve ref: owner/repo:branch → owner/repo:branch:commit
            resolved = index.get("refs", {}).get(repo_identifier, repo_identifier)

            # Find vectorstore cache key
            cache_key = index.get(resolved) or index.get(repo_identifier)

            # Find graph cache key
            graphs = index.get("graphs", {})
            graph_key = graphs.get(f"{resolved}:combined") or graphs.get(f"{repo_identifier}:combined")
            # Try without :combined suffix
            if not graph_key:
                for k, v in graphs.items():
                    if k.startswith(resolved) or k.startswith(repo_identifier):
                        graph_key = v
                        break

            logger.info(
                f"Cache index lookup: repo={repo_identifier} resolved={resolved} "
                f"vectorstore={cache_key} graph={graph_key}"
            )
        except Exception as e:
            logger.warning(f"Failed to read cache_index.json: {e}")

    # Load graph
    if graph_key:
        try:
            import gzip
            import pickle

            import networkx as nx

            from app.core.graph_manager import GraphManager

            gm = GraphManager(cache_dir=cache_dir)
            graph_file = cache_path / f"{graph_key}.code_graph.gz"
            if graph_file.exists():
                with gzip.open(graph_file, "rb") as f:
                    graph = pickle.load(f)  # noqa: S301 — pickle used for graph cache, data is self-generated
                if isinstance(graph, nx.DiGraph):
                    components.code_graph = graph
                    gm.graph = graph
                    components.graph_manager = gm
                    logger.info(f"Loaded graph {graph_key} ({graph.number_of_nodes()} nodes)")
        except Exception as e:
            logger.warning(f"Failed to load graph {graph_key}: {e}")

    # Load vectorstore using _load_from_cache with correct file paths
    # Use same embeddings as generation to match FAISS dimensions
    if cache_key:
        try:
            from app.core.vectorstore import VectorStoreManager
            from app.services.llm_factory import create_embeddings

            embeddings = create_embeddings(settings)
            vsm = VectorStoreManager(cache_dir=cache_dir, embeddings=embeddings)
            cache_file = cache_path / f"{cache_key}.faiss"
            docs_file = cache_path / f"{cache_key}.docs.pkl"
            if cache_file.exists():
                vectorstore, documents = vsm._load_from_cache(cache_file, docs_file)
                vsm.vectorstore = vectorstore
                vsm.documents = documents
                logger.info(
                    f"Loaded vectorstore {cache_key} ({vectorstore.index.ntotal} vectors, {len(documents)} docs)"
                )

                from app.core.retrievers import WikiRetrieverStack

                components.retriever_stack = WikiRetrieverStack(
                    vectorstore_manager=vsm,
                    relationship_graph=components.code_graph,
                )
            else:
                logger.warning(f"FAISS file not found: {cache_file}")
        except Exception as e:
            logger.warning(f"Failed to load vectorstore {cache_key}: {e}")

    if not cache_key and not graph_key:
        logger.warning(f"No cache entries found for {repo_identifier} — ask/research may fail")
