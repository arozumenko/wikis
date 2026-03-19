#!/usr/bin/python3

"""
Wiki Loader

Loads wiki artifacts from bucket folder structure for querying.
Handles downloading and initializing vector stores, graphs, and BM25 indexes.

NOTE: This module is intended for a SEPARATE toolkit implementation that
downloads wiki artifacts from bucket. For wikis_plugin itself, artifacts
are loaded from local disk cache and this module is NOT used.

Bucket structure (Context7-style):
    wiki_artifacts/
        ├── _registry/wikis.json           # Global registry of all wikis
        └── {wiki_id}/                     # Folder per wiki (flat inside)
            ├── wiki_manifest_{version}.json   # Versioned manifests
            ├── {hash}.faiss                   # FAISS index
            ├── {hash}.docstore.bin            # Docstore binary
            ├── {hash}.doc_index.json          # Doc index
            ├── {hash}.bm25.sqlite             # BM25 index
            ├── combined.code_graph.gz         # Code graph
            ├── {hash}_analysis.json           # Analysis data
            └── wiki_pages/*.md                # Wiki markdown pages

Manifest structure (from wiki_subprocess_worker.py):
    {
        "schema_version": 2,
        "wiki_id": "owner--repo--main",
        "wiki_version_id": "uuid",
        "created_at": "ISO timestamp",
        "canonical_repo_identifier": "owner/repo",
        "branch": "main",
        "commit_hash": "abc123",
        "analysis_key": "hash",
        "pages": ["page1.md", "page2.md"],
        "faiss_cache_key": "hash",
        "graph_cache_key": "hash",
        "docstore_cache_key": "hash",
        "docstore_files": ["hash.docstore.bin", "hash.doc_index.json"],
        "bm25_cache_key": "hash",
        "bm25_files": ["hash.bm25.sqlite"]
    }
"""

import gzip
import json
import logging
import pickle
import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import networkx as nx

logger = logging.getLogger(__name__)


@dataclass
class LoadedWiki:
    """Container for loaded wiki artifacts."""

    wiki_id: str
    manifest: dict[str, Any]  # Full manifest from bucket
    cache_dir: Path

    # Loaded artifacts (lazy loaded)
    graph: nx.DiGraph | None = None

    # Status flags
    graph_loaded: bool = False
    vectorstore_loaded: bool = False
    bm25_loaded: bool = False
    analysis_loaded: bool = False
    analysis: dict[str, Any] | None = None

    # Convenience properties from manifest
    @property
    def repo_identifier(self) -> str:
        return self.manifest.get("canonical_repo_identifier", "")

    @property
    def branch(self) -> str:
        return self.manifest.get("branch", "main")

    @property
    def wiki_version_id(self) -> str:
        return self.manifest.get("wiki_version_id", "")

    @property
    def faiss_cache_key(self) -> str | None:
        return self.manifest.get("faiss_cache_key")

    @property
    def graph_cache_key(self) -> str | None:
        return self.manifest.get("graph_cache_key")

    @property
    def docstore_cache_key(self) -> str | None:
        return self.manifest.get("docstore_cache_key")

    @property
    def bm25_cache_key(self) -> str | None:
        return self.manifest.get("bm25_cache_key")

    @property
    def analysis_key(self) -> str | None:
        return self.manifest.get("analysis_key")


class WikiLoader:
    """
    Loads wiki artifacts from bucket for querying.

    Downloads artifacts to a temporary directory and initializes
    the necessary components for ask/deep_research tools.
    """

    def __init__(
        self,
        artifacts_client,
        bucket_name: str = "wiki_artifacts",
        temp_dir: Path | None = None,
        max_cached_wikis: int = 5,
    ):
        """
        Initialize wiki loader.

        Args:
            artifacts_client: Client for bucket operations
            bucket_name: Name of the bucket containing wiki artifacts
            temp_dir: Base directory for temporary files (default: system temp)
            max_cached_wikis: Maximum number of wikis to keep in local cache
        """
        self.client = artifacts_client
        self.bucket = bucket_name
        self.max_cached_wikis = max_cached_wikis

        # Set up temp directory
        if temp_dir:
            self._temp_base = Path(temp_dir)
        else:
            self._temp_base = Path(tempfile.mkdtemp(prefix="wiki_loader_"))
        self._temp_base.mkdir(parents=True, exist_ok=True)

        # Cache of loaded wikis
        self._loaded_wikis: dict[str, LoadedWiki] = {}
        self._load_order: list[str] = []  # LRU tracking

    def _ensure_cache_size(self) -> None:
        """Evict oldest wikis if cache exceeds max size."""
        while len(self._load_order) > self.max_cached_wikis:
            oldest_id = self._load_order.pop(0)
            if oldest_id in self._loaded_wikis:
                self._loaded_wikis.pop(oldest_id)
                # Clean up temp files
                wiki_dir = self._temp_base / oldest_id
                if wiki_dir.exists():
                    shutil.rmtree(wiki_dir, ignore_errors=True)
                logger.info(f"Evicted wiki from cache: {oldest_id}")

    def _download_file(self, bucket_path: str, local_path: Path) -> bool:
        """
        Download a file from bucket to local path.

        Returns:
            True if successful, False if file not found
        """
        try:
            data = self.client.download_artifact(self.bucket, bucket_path)
            local_path.parent.mkdir(parents=True, exist_ok=True)

            if isinstance(data, str):
                local_path.write_text(data, encoding="utf-8")
            else:
                local_path.write_bytes(data)

            return True
        except Exception as e:
            logger.debug(f"Failed to download {bucket_path}: {e}")
            return False

    def _download_json(self, bucket_path: str) -> dict | None:
        """Download and parse a JSON file from bucket."""
        try:
            data = self.client.download_artifact(self.bucket, bucket_path)
            if isinstance(data, bytes):
                data = data.decode("utf-8")
            return json.loads(data)
        except Exception as e:
            logger.debug(f"Failed to download JSON {bucket_path}: {e}")
            return None

    def load_wiki_manifest(self, wiki_id: str) -> dict | None:
        """
        Load the manifest for a wiki.

        Args:
            wiki_id: Wiki folder name

        Returns:
            Manifest dict or None if not found
        """
        return self._download_json(f"{wiki_id}/manifest.json")

    def load_wiki(
        self,
        wiki_id: str,
        load_graph: bool = True,
        load_vectorstore: bool = True,
        load_bm25: bool = True,
        load_analysis: bool = True,
    ) -> LoadedWiki | None:
        """
        Load wiki artifacts from bucket.

        Args:
            wiki_id: Wiki folder name (e.g., "owner--repo--main")
            load_graph: Whether to download the graph
            load_vectorstore: Whether to download vectorstore files
            load_bm25: Whether to download BM25 index
            load_analysis: Whether to download analysis

        Returns:
            LoadedWiki with paths to downloaded artifacts, or None if wiki not found
        """
        # Check cache first
        if wiki_id in self._loaded_wikis:
            # Move to end of LRU list
            if wiki_id in self._load_order:
                self._load_order.remove(wiki_id)
            self._load_order.append(wiki_id)
            logger.info(f"Using cached wiki: {wiki_id}")
            return self._loaded_wikis[wiki_id]

        # Download manifest first
        manifest = self.load_wiki_manifest(wiki_id)
        if not manifest:
            logger.error(f"Wiki not found: {wiki_id}")
            return None

        # Set up local cache directory
        wiki_dir = self._temp_base / wiki_id
        wiki_dir.mkdir(parents=True, exist_ok=True)

        loaded = LoadedWiki(
            wiki_id=wiki_id,
            manifest=manifest,
            cache_dir=wiki_dir,
        )

        # Download graph using manifest's file list
        if load_graph:
            graph_dir = wiki_dir / "graph"
            graph_dir.mkdir(parents=True, exist_ok=True)

            # Manifest v2 stores paths relative to wiki_id folder
            graph_files = manifest.get("graph_files", [])

            # Fallback to default path for v1 manifests
            if not graph_files and manifest.get("graph_cache_key"):
                graph_files = ["graph/combined.code_graph.gz"]

            for rel_path in graph_files:
                local_file = wiki_dir / rel_path
                if self._download_file(f"{wiki_id}/{rel_path}", local_file):
                    try:
                        with gzip.open(local_file, "rb") as f:
                            loaded.graph = pickle.load(f)  # noqa: S301 — pickle used for graph cache, data is self-generated
                        loaded.graph_loaded = True
                        logger.info(f"Loaded graph with {loaded.graph.number_of_nodes()} nodes")
                    except Exception as e:
                        logger.error(f"Failed to load graph: {e}")

        # Download vectorstore files using manifest's file list
        if load_vectorstore:
            vs_dir = wiki_dir / "vectorstore"
            vs_dir.mkdir(parents=True, exist_ok=True)

            # Manifest v2 stores paths relative to wiki_id folder
            # e.g., "vectorstore/{hash}.faiss", "vectorstore/{hash}.docstore.bin"
            vs_files = manifest.get("vectorstore_files", [])
            docstore_files = manifest.get("docstore_files", [])

            all_files = vs_files + docstore_files

            # Fallback to key-based naming for v1 manifests
            if not all_files:
                faiss_key = manifest.get("faiss_cache_key", "")
                docstore_key = manifest.get("docstore_cache_key", "")
                if faiss_key:
                    all_files.append(f"vectorstore/{faiss_key}.faiss")
                if docstore_key:
                    all_files.extend(
                        [
                            f"vectorstore/{docstore_key}.docstore.bin",
                            f"vectorstore/{docstore_key}.doc_index.json",
                        ]
                    )

            all_downloaded = True
            for rel_path in all_files:
                # rel_path is like "vectorstore/{hash}.faiss"
                # Download from {wiki_id}/{rel_path} to local {wiki_dir}/{rel_path}
                local_file = wiki_dir / rel_path
                if not self._download_file(f"{wiki_id}/{rel_path}", local_file):
                    all_downloaded = False

            if all_downloaded and all_files:
                loaded.vectorstore_loaded = True
                logger.info(f"Downloaded vectorstore files for {wiki_id}")

        # Download BM25 index using manifest's file list
        if load_bm25:
            bm25_dir = wiki_dir / "bm25"
            bm25_dir.mkdir(parents=True, exist_ok=True)

            # Manifest v2 stores paths relative to wiki_id folder
            bm25_files = manifest.get("bm25_files", [])

            # Fallback to key-based naming for v1 manifests
            if not bm25_files:
                bm25_key = manifest.get("bm25_cache_key", "")
                if bm25_key:
                    bm25_files = [f"bm25/{bm25_key}.bm25.sqlite"]

            all_downloaded = True
            for rel_path in bm25_files:
                local_file = wiki_dir / rel_path
                if not self._download_file(f"{wiki_id}/{rel_path}", local_file):
                    all_downloaded = False

            if all_downloaded and bm25_files:
                loaded.bm25_loaded = True
                logger.info(f"Downloaded BM25 index for {wiki_id}")

        # Download analysis
        if load_analysis:
            analysis_dir = wiki_dir / "analysis"
            analysis_dir.mkdir(parents=True, exist_ok=True)

            # Try to find analysis files - manifest v2 might list them, or use analysis_key
            analysis_key = manifest.get("analysis_key", "")
            if analysis_key:
                analysis = self._download_json(f"{wiki_id}/analysis/{analysis_key}_analysis.json")
                if analysis:
                    loaded.analysis = analysis
                    loaded.analysis_loaded = True
                    logger.info(f"Downloaded analysis for {wiki_id}")

        # Add to cache
        self._loaded_wikis[wiki_id] = loaded
        self._load_order.append(wiki_id)
        self._ensure_cache_size()

        return loaded

    def get_vectorstore_dir(self, wiki_id: str) -> Path | None:
        """Get the local vectorstore directory for a loaded wiki."""
        if wiki_id not in self._loaded_wikis:
            return None
        return self._temp_base / wiki_id / "vectorstore"

    def get_bm25_dir(self, wiki_id: str) -> Path | None:
        """Get the local BM25 directory for a loaded wiki."""
        if wiki_id not in self._loaded_wikis:
            return None
        return self._temp_base / wiki_id / "bm25"

    def get_graph_path(self, wiki_id: str) -> Path | None:
        """Get the local graph file path for a loaded wiki."""
        if wiki_id not in self._loaded_wikis:
            return None
        return self._temp_base / wiki_id / "graph" / "combined.code_graph.gz"

    def is_wiki_loaded(self, wiki_id: str) -> bool:
        """Check if a wiki is currently loaded in cache."""
        return wiki_id in self._loaded_wikis

    def get_loaded_wiki(self, wiki_id: str) -> LoadedWiki | None:
        """Get a loaded wiki from cache without downloading."""
        return self._loaded_wikis.get(wiki_id)

    def cleanup(self) -> None:
        """Clean up all temporary files."""
        for wiki_id in list(self._loaded_wikis.keys()):
            wiki_dir = self._temp_base / wiki_id
            if wiki_dir.exists():
                shutil.rmtree(wiki_dir, ignore_errors=True)

        self._loaded_wikis.clear()
        self._load_order.clear()

        # Clean up base temp dir if it's empty
        try:
            if self._temp_base.exists() and not any(self._temp_base.iterdir()):
                self._temp_base.rmdir()
        except Exception:  # noqa: S110
            pass

        logger.info("Wiki loader cleanup complete")

    def __del__(self):
        """Cleanup on destruction."""
        try:
            self.cleanup()
        except Exception:  # noqa: S110
            pass
