"""
Filesystem Repository Indexer

Direct filesystem-based indexer that replaces GitHub API approach.
Uses local repository clones and Enhanced Unified Graph Builder for optimal performance.
Supports multi-provider repositories: GitHub, GitLab, Bitbucket, Azure DevOps.
"""

import logging
import os
from datetime import datetime
from typing import Any

import networkx as nx
from langchain_core.documents import Document

from .filter_manager import FilterManager
from .graph_manager import GraphManager
from .local_repository_manager import LocalRepositoryManager
from .repo_providers import GitCloneConfig
from .utils.resource_monitor import resource_monitor
from .vectorstore import VectorStoreManager

logger = logging.getLogger(__name__)

# Optional thinking emitter (safe no-op if task context not present)
try:  # pragma: no cover - environment dependent
    from tools import this  # type: ignore
except Exception:  # pragma: no cover
    this = None  # type: ignore

# Import Enhanced Unified Graph Builder
try:
    from .code_graph.graph_builder import EnhancedUnifiedGraphBuilder
except ImportError:
    logger.error("Enhanced Unified Graph Builder not available")
    EnhancedUnifiedGraphBuilder = None


class FilesystemRepositoryIndexer:
    """Filesystem-based repository indexer using local clones and EUGB.

    Supports multi-provider repositories: GitHub, GitLab, Bitbucket, Azure DevOps.
    """

    def __init__(
        self,
        cache_dir: str | None = None,
        repo_config_path: str | None = None,
        api_filters: dict[str, Any] | None = None,
        max_workers: int = 8,
        cleanup_repos_on_exit: bool = True,
        # New multi-provider support
        clone_config: GitCloneConfig | None = None,
        # Legacy GitHub credentials (fallback)
        github_access_token: str | None = None,
        github_username: str | None = None,
        embeddings: Any | None = None,
        progress_callback: Any | None = None,
    ):
        """
        Initialize filesystem-based indexer with multi-provider support

        Args:
            cache_dir: Directory for caching vector stores and graphs
            repo_config_path: Path to repo.json configuration
            api_filters: Additional filters from API level
            max_workers: Number of parallel workers for processing
            cleanup_repos_on_exit: Whether to cleanup repository clones on exit
            clone_config: Pre-built clone configuration (multi-provider)
            github_access_token: GitHub access token for private repositories (legacy)
            github_username: GitHub username (legacy)
            embeddings: Embeddings instance
        """
        self.max_workers = max_workers
        self.cleanup_repos_on_exit = cleanup_repos_on_exit
        self.clone_config = clone_config
        self.progress_callback = progress_callback

        # Initialize repository manager with multi-provider support
        self.repo_manager = LocalRepositoryManager(
            cache_dir=os.path.join(cache_dir or "/tmp/wiki_builder", "repositories"),  # noqa: S108 — intentional temp directory for repo cache
            cleanup_on_exit=cleanup_repos_on_exit,
            # Pass clone_config for multi-provider support
            clone_config=clone_config,
            # Legacy GitHub credentials (fallback if clone_config not available)
            github_access_token=github_access_token,
            github_username=github_username,
        )

        # Initialize filter manager
        self.filter_manager = FilterManager(repo_config_path=repo_config_path, api_filters=api_filters)

        # Initialize Enhanced Unified Graph Builder
        if EnhancedUnifiedGraphBuilder is None:
            raise ImportError("Enhanced Unified Graph Builder is required for filesystem indexing")
        self.graph_builder = EnhancedUnifiedGraphBuilder(max_workers=max_workers)

        # Initialize vector store and graph managers
        self.vectorstore_manager = VectorStoreManager(
            cache_dir=cache_dir,
            embeddings=embeddings,
        )
        self.graph_manager = GraphManager(cache_dir=cache_dir)

        # State containers
        self.text_documents = []
        self.code_documents = []
        self.all_documents = []
        self.relationship_graph = None
        self.vectorstore = None
        self.last_index_stats = None
        self._doc_type_counts = {
            "text": 0,
            "code": 0,
        }

        # Current repository info
        self.current_repo_path = None
        self.current_repo_info = None

        # Convenience cached values (filled after indexing)
        self._last_commit_hash = None
        self._repo_root = None

    def _emit_progress(self, progress: float, message: str) -> None:
        """Emit indexing progress event if callback is set."""
        if self.progress_callback:
            try:
                self.progress_callback("indexing", progress, message)
            except Exception:  # noqa: S110
                pass

    def index_repository(
        self,
        repository_url: str,
        branch: str = "main",
        force_rebuild: bool = False,
        force_reclone: bool = False,
        max_files: int | None = None,
    ) -> dict[str, Any]:
        """
        Index repository using filesystem approach

        Args:
            repository_url: Repository URL or owner/repo format
            branch: Branch to clone and index
            force_rebuild: Whether to force rebuild even if cache exists
            force_reclone: Whether to force reclone repository
            max_files: Maximum number of files to process

        Returns:
            Dictionary with indexing results and statistics
        """
        start_time = datetime.now()

        try:
            if this and getattr(this, "module", None):  # thinking step
                this.module.invocation_thinking(
                    f"I am on phase indexing\nStarting filesystem indexing for {repository_url}@{branch}\nReasoning: Need fresh / cached analysis artifacts (graph + vector store) to enable downstream wiki generation.\nNext: Attempt cache load; else perform file discovery & graph analysis."
                )
            # Step 1: Get local repository path
            self._emit_progress(0.10, f"Cloning repository {repository_url} (branch: {branch})...")
            logger.info(f"Getting local repository path for {repository_url} (branch: {branch})")
            repo_path = self.repo_manager.get_repository_local_path(repository_url, branch, force_reclone=force_reclone)

            self.current_repo_path = repo_path
            self.current_repo_info = self.repo_manager.get_repository_info(repo_path)
            self._repo_root = repo_path
            if self.current_repo_info:
                self._last_commit_hash = self.current_repo_info.get("commit_hash")

            # Use actual branch from cloned repo (handles cases like 'master' vs 'main')
            actual_branch = (self.current_repo_info.get("branch") if self.current_repo_info else None) or branch

            # Create repository identifier for caching (includes commit hash for isolation)
            repo_name = self.repo_manager._normalize_repository_name(repository_url)
            commit_short = self._last_commit_hash[:8] if self._last_commit_hash else "unknown"
            repo_identifier = f"{repo_name}:{actual_branch}:{commit_short}"

            self._emit_progress(0.12, f"Repository ready — branch: {actual_branch}, commit: {commit_short}")
            logger.info(f"Repository cloned to: {repo_path} (branch: {actual_branch}, commit: {commit_short})")

            # Step 2: Check cache if not forcing rebuild (graph + vector)
            if not force_rebuild:
                cached_result = self._try_load_from_cache(repo_identifier, repo_path)
                if cached_result:
                    if this and getattr(this, "module", None):
                        this.module.invocation_thinking(
                            "I am on phase indexing\nIndex loaded from cache (graph + vector store)\nReasoning: Reusing previously computed artifacts saves time & cost.\nNext: Return stats to caller for immediate generation phase."
                        )
                    cached_result["indexing_time"] = (datetime.now() - start_time).total_seconds()
                    cached_result["repository_path"] = repo_path
                    cached_result["actual_branch"] = actual_branch  # Propagate actual branch from clone
                    return cached_result

            # Step 3: Build index from filesystem
            logger.info("Building index from filesystem...")
            result = self._build_filesystem_index(repo_identifier, repo_path, max_files, force_rebuild=force_rebuild)
            result["indexing_time"] = (datetime.now() - start_time).total_seconds()
            result["repository_path"] = repo_path
            result["actual_branch"] = actual_branch  # Propagate actual branch from clone

            return result

        except Exception as e:
            logger.error(f"Failed to index repository {repository_url}: {e}")
            if this and getattr(this, "module", None):
                try:
                    this.module.invocation_thinking(
                        f"I am on phase indexing\nIndexing failed for {repository_url}@{branch}: {e}"
                    )
                except Exception:  # noqa: S110
                    pass
            # Re-raise the original exception - don't wrap with redundant context.
            # The original error message from local_repository_manager is already clear.
            raise

    def _try_load_from_cache(self, repo_identifier: str, repo_path: str) -> dict[str, Any] | None:
        """Try to load complete index from cache (graph + vectorstore + docs)."""
        try:
            commit_hash = self._last_commit_hash

            # 1. Graph cache (using commit hash for stable lookup)
            if not self.graph_manager.graph_exists(repo_path, commit_hash=commit_hash):
                logger.debug(
                    f"[cache-miss] graph not found for {repo_path} @ {commit_hash[:8] if commit_hash else 'unknown'}"
                )
                return None
            self.relationship_graph = self.graph_manager.load_graph(repo_path, commit_hash=commit_hash)
            if not self.relationship_graph:
                logger.debug(f"[cache-miss] failed to load graph object for {repo_path}")
                return None

            # 2. Vector store + documents cache (using commit hash)
            self.vectorstore, self.all_documents = self.vectorstore_manager.load_or_build(
                [], repo_path, force_rebuild=False, commit_hash=commit_hash
            )
            if not self.vectorstore or not self.all_documents:
                logger.debug(f"[cache-miss] vector store or documents missing for {repo_path}")
                return None

            # 3. Separate docs
            text_docs = [doc for doc in self.all_documents if doc.metadata.get("chunk_type") == "text"]
            code_docs = [doc for doc in self.all_documents if doc.metadata.get("chunk_type") in ["symbol", "code"]]
            self._set_document_splits(text_docs, code_docs)

            # 4. Register caches for Ask tool lookup with commit-aware identifier
            cache_key = self.vectorstore_manager._generate_repo_hash(repo_path, [], commit_hash)
            self.vectorstore_manager.register_cache(repo_identifier, cache_key)
            self.graph_manager.register_graph_cache(
                repo_identifier, self.graph_manager._generate_cache_key(repo_path, "combined", commit_hash)
            )

            logger.info(f"[cache-hit] Loaded graph + vectorstore for {repo_identifier} from {repo_path}")
            self._emit_progress(0.20, "Index loaded from cache")
            return self._generate_index_stats(repo_identifier, from_cache=True)
        except Exception as e:
            logger.debug(f"[cache-error] {e}")
            return None

    def _build_filesystem_index(
        self, repo_identifier: str, repo_path: str, max_files: int | None, force_rebuild: bool = False
    ) -> dict[str, Any]:
        """Build index from filesystem using Enhanced Unified Graph Builder"""
        logger.info(f"Building filesystem index for {repo_path}")
        if this and getattr(this, "module", None):
            this.module.invocation_thinking(
                "I am on phase indexing\nDiscovering & filtering repository files...\nReasoning: Apply inclusion/exclusion rules to bound analysis scope.\nNext: Run unified graph analysis on accepted files."
            )

        # Step 1: Filter files using filter manager
        self._emit_progress(0.14, "Discovering and filtering repository files...")
        logger.info("Discovering and filtering files...")
        with resource_monitor("index.discover_files", logger):
            all_files = self._discover_files(repo_path)

        if max_files and len(all_files) > max_files:
            logger.info(f"Limiting to {max_files} files (found {len(all_files)})")
            all_files = all_files[:max_files]

        if not all_files:
            logger.warning("No files discovered for indexing")
            return self._generate_empty_stats(repo_identifier)

        self._emit_progress(0.16, f"Discovered {len(all_files)} files — building code graph...")
        logger.info(f"Processing {len(all_files)} files with Enhanced Unified Graph Builder")
        if this and getattr(this, "module", None):
            this.module.invocation_thinking(
                f"I am on phase indexing\nRunning graph analysis on {len(all_files)} files...\nReasoning: Produce symbol-level documents + relationships for vectorization & retrieval.\nNext: Persist analysis outputs then build/load vector store."
            )

        # Step 2: Run Enhanced Unified Graph Builder analysis
        try:
            # Stream documents by default for memory efficiency
            # Set WIKIS_DISABLE_STREAM_DOCS=1 to materialize all documents at once
            stream_documents = os.getenv("WIKIS_DISABLE_STREAM_DOCS") != "1"
            with resource_monitor("index.graph_analysis", logger):
                analysis = self.graph_builder.analyze_repository(repo_path, stream_documents=stream_documents)

            # Extract results
            documents_iter = getattr(analysis, "documents_iter", None)
            self.all_documents = analysis.documents
            self.relationship_graph = analysis.unified_graph

            # Apply post-processing to add repository metadata
            self._add_repository_metadata()

            # Separate documents by type (only when materialized)
            if self.all_documents:
                text_docs = [doc for doc in self.all_documents if doc.metadata.get("chunk_type") == "text"]
                code_docs = [doc for doc in self.all_documents if doc.metadata.get("chunk_type") in ["symbol", "code"]]
                self._set_document_splits(text_docs, code_docs)
            else:
                self._set_document_splits([], [])

            if documents_iter and not self.all_documents:
                logger.info("EUGB analysis complete: documents streaming enabled")
            else:
                logger.info(
                    f"EUGB analysis complete: {len(self.all_documents)} documents, "
                    f"{self._get_text_count()} text, {self._get_code_count()} code"
                )
            if this and getattr(this, "module", None):
                this.module.invocation_thinking(
                    "I am on phase indexing\n"
                    + (
                        "Graph analysis produced streaming docs\n"
                        if documents_iter and not self.all_documents
                        else f"Graph analysis produced {len(self.all_documents)} docs "
                        f"(text {self._get_text_count()}, code {self._get_code_count()})\n"
                    )
                    + "Reasoning: Balanced text/code representation enables blended semantic search.\n"
                    + "Next: Vectorize & cache graph + documents."
                )

        except Exception as e:
            logger.error(f"Enhanced Unified Graph Builder analysis failed: {e}")
            import traceback

            logger.error(f"Full traceback: {traceback.format_exc()}")
            # Re-raise - graph analysis failure is fatal, cannot generate wiki without code analysis
            raise RuntimeError(f"Graph analysis failed: {e}") from e

        if not self.all_documents and not documents_iter:
            logger.warning("No documents created from repository analysis")
            # This is fatal - cannot generate wiki without any documents
            raise RuntimeError(
                "No documents found in repository. The repository may be empty, "
                "all files may be filtered out, or the branch may not exist."
            )

        # Step 3: Build or load vector store (allow cache reuse)
        self._emit_progress(0.20, "Building vector embeddings for semantic search...")
        logger.info("Building / loading vector store (cache enabled)...")
        if this and getattr(this, "module", None):
            this.module.invocation_thinking(
                "I am on phase indexing\nBuilding / loading vector store...\nReasoning: Create dense embeddings for retrieval-driven page generation.\n>Next: Cache graph; compute index stats."
            )

        try:
            with resource_monitor("index.vectorstore_build", logger):
                if documents_iter:
                    self.vectorstore, self.all_documents = self.vectorstore_manager.load_or_build_from_iterable(
                        documents_iter, repo_path, force_rebuild=force_rebuild, commit_hash=self._last_commit_hash
                    )

                    # Ensure repository metadata and source paths are set after streaming build
                    self._add_repository_metadata()

                    # Rebuild document splits from streamed metadata
                    text_docs = [doc for doc in self.all_documents if doc.metadata.get("chunk_type") == "text"]
                    code_docs = [
                        doc for doc in self.all_documents if doc.metadata.get("chunk_type") in ["symbol", "code"]
                    ]
                    self._set_document_splits(text_docs, code_docs)
                else:
                    self.vectorstore, self.all_documents = self.vectorstore_manager.load_or_build(
                        self.all_documents, repo_path, force_rebuild=force_rebuild, commit_hash=self._last_commit_hash
                    )

                    # Ensure repository metadata and source paths are set after build
                    self._add_repository_metadata()

                self._emit_progress(0.25, f"Indexed {len(self.all_documents)} documents ({self._get_text_count()} text, {self._get_code_count()} code)")
        except Exception as e:
            logger.error(f"Failed to build vector store: {e}")
            import traceback

            logger.error(f"Full traceback: {traceback.format_exc()}")

            # Don't fail entire indexing - return results with graph but no vectorstore
            logger.warning(
                "Vector store building failed. Graph and documents are available, "
                "but semantic search will not be functional. This may be due to empty "
                "documents or embedding model issues."
            )

            # Save graph even if vectorstore failed (with commit hash)
            if self.relationship_graph:
                logger.info("Saving relationship graph to cache...")
                self.graph_manager.save_graph(
                    self.relationship_graph,
                    repo_path,
                    commit_hash=self._last_commit_hash,
                    repo_identifier=repo_identifier,
                )

            # Return partial results
            index_stats = self._generate_index_stats(repo_identifier, from_cache=False)
            index_stats["vectorstore_error"] = str(e)
            index_stats["warning"] = "Vectorstore building failed - graph available but semantic search disabled"
            if self._last_commit_hash and "repository_info" in index_stats:
                index_stats["repository_info"]["commit_hash"] = self._last_commit_hash
            return index_stats

        # Vectorstore health check
        if self.vectorstore:
            try:
                doc_count = self.vectorstore.index.ntotal if hasattr(self.vectorstore, "index") else 0
                if doc_count == 0:
                    logger.error(
                        "VECTORSTORE EMPTY after build — embeddings may have failed. "
                        "Semantic search will not work. Check embedding model/credentials."
                    )
                    self._emit_progress(0.25, "WARNING: Vectorstore empty — embeddings may have failed")
                else:
                    logger.info(f"Vectorstore health check: {doc_count} vectors indexed")
            except Exception as e:
                logger.warning(f"Vectorstore health check failed: {e}")
        else:
            logger.error("Vectorstore is None after build — generation will produce generic content")
            self._emit_progress(0.25, "WARNING: No vectorstore — content quality will be degraded")

        # Step 4: Save relationship graph to cache (with commit hash for stability)
        self._emit_progress(0.27, "Saving code graph and caches...")
        if self.relationship_graph:
            logger.info("Saving relationship graph to cache...")
            with resource_monitor("index.graph_cache_save", logger):
                self.graph_manager.save_graph(
                    self.relationship_graph,
                    repo_path,
                    commit_hash=self._last_commit_hash,
                    repo_identifier=repo_identifier,
                )

        # Step 5: Register vectorstore cache for Ask tool lookup
        # Generate the same cache key that was used to store the vectorstore
        cache_key = self.vectorstore_manager._generate_repo_hash(repo_path, [], self._last_commit_hash)
        self.vectorstore_manager.register_cache(repo_identifier, cache_key)
        self.vectorstore_manager.register_docstore_cache(repo_identifier, cache_key)

        self._emit_progress(0.30, "Indexing complete")
        logger.info("Filesystem index building complete")
        if this and getattr(this, "module", None):
            this.module.invocation_thinking(
                "I am on phase indexing\nFilesystem index build complete\nReasoning: All prerequisite assets (graph, vectors, docs) ready.\nNext: Hand off to generation agent for analysis + drafting."
            )
        index_stats = self._generate_index_stats(repo_identifier, from_cache=False)
        self.last_index_stats = index_stats
        # Ensure commit hash bubbled up for upstream components
        if self._last_commit_hash and "repository_info" in index_stats:
            index_stats["repository_info"]["commit_hash"] = self._last_commit_hash
        return index_stats

    def _discover_files(self, repo_path: str) -> list[str]:
        """Discover files in repository using filter manager"""
        all_files = []

        for root, dirs, files in os.walk(repo_path):
            # Apply directory filters
            dirs_to_remove = []
            for dir_name in dirs:
                dir_path = os.path.relpath(os.path.join(root, dir_name), repo_path)
                if not self.filter_manager.should_process_directory(dir_path):
                    dirs_to_remove.append(dir_name)

            # Remove excluded directories from traversal
            for dir_name in dirs_to_remove:
                dirs.remove(dir_name)

            # Apply file filters
            for file_name in files:
                file_path = os.path.relpath(os.path.join(root, file_name), repo_path)
                if self.filter_manager.should_process_file(file_path):
                    all_files.append(file_path)

        logger.info(f"Discovered {len(all_files)} files after filtering")
        return all_files

    def _add_repository_metadata(self) -> None:
        """Add repository metadata to all documents"""
        if not self.current_repo_info:
            return

        repo_metadata = {
            "repository_path": self.current_repo_path,
            "commit_hash": self.current_repo_info.get("commit_hash"),
            "branch": self.current_repo_info.get("branch"),
            "remote_url": self.current_repo_info.get("remote_url"),
        }

        for doc in self.all_documents:
            doc.metadata.update(repo_metadata)
            # Provide 'source' key for compatibility with GitHubIndexer-dependent code paths
            # Use rel_path (relative) for source to avoid absolute paths in citations
            if "source" not in doc.metadata:
                doc.metadata["source"] = doc.metadata.get("rel_path") or doc.metadata.get("file_path")

    def _set_document_splits(self, text_docs: list[Document], code_docs: list[Document]) -> None:
        """Store document counts and optionally drop lists to save RAM.

        By default, we only store counts (not full lists) to reduce memory usage.
        Set WIKIS_KEEP_DOC_SPLITS=1 to preserve full document lists.
        """
        self._doc_type_counts = {
            "text": len(text_docs),
            "code": len(code_docs),
        }

        # Only keep full lists if explicitly requested (saves significant RAM)
        if os.getenv("WIKIS_KEEP_DOC_SPLITS") == "1":
            self.text_documents = text_docs
            self.code_documents = code_docs
        else:
            self.text_documents = None
            self.code_documents = None

    def _get_text_count(self) -> int:
        if isinstance(self.text_documents, list):
            return len(self.text_documents)
        return self._doc_type_counts.get("text", 0)

    def _get_code_count(self) -> int:
        if isinstance(self.code_documents, list):
            return len(self.code_documents)
        return self._doc_type_counts.get("code", 0)

    def _iter_text_documents(self):
        if isinstance(self.text_documents, list):
            return iter(self.text_documents)
        return (doc for doc in self.all_documents if doc.metadata.get("chunk_type") == "text")

    def _iter_code_documents(self):
        if isinstance(self.code_documents, list):
            return iter(self.code_documents)
        return (doc for doc in self.all_documents if doc.metadata.get("chunk_type") in ["symbol", "code"])

    def _generate_index_stats(self, repo_identifier: str, from_cache: bool = False) -> dict[str, Any]:
        """Generate comprehensive indexing statistics"""
        # Document analysis
        file_types = {}
        languages = set()
        symbols_by_type = {}
        files_with_symbols = set()

        for doc in self.all_documents:
            # File type analysis
            if "file_extension" in doc.metadata:
                ext = doc.metadata["file_extension"]
                file_types[ext] = file_types.get(ext, 0) + 1

            # Language analysis
            if doc.metadata.get("language"):
                languages.add(doc.metadata["language"])

            # Symbol analysis
            if doc.metadata.get("symbol_type"):
                symbol_type = doc.metadata["symbol_type"]
                symbols_by_type[symbol_type] = symbols_by_type.get(symbol_type, 0) + 1
                files_with_symbols.add(doc.metadata.get("file_path", ""))

        # Graph analysis
        graph_analysis = {}
        if self.relationship_graph:
            graph_analysis = self.graph_manager.analyze_graph(self.relationship_graph)

        stats = {
            "repository_identifier": repo_identifier,
            "loaded_from_cache": from_cache,
            "documents": {
                "total": len(self.all_documents),
                "text_documents": self._get_text_count(),
                "code_documents": self._get_code_count(),
                "file_types": file_types,
            },
            "code_analysis": {
                "languages": list(languages),
                "symbols_by_type": symbols_by_type,
                "files_with_symbols": len(files_with_symbols),
                "total_symbols": sum(symbols_by_type.values()),
            },
            "relationship_graph": {
                "nodes": self.relationship_graph.number_of_nodes() if self.relationship_graph else 0,
                "edges": self.relationship_graph.number_of_edges() if self.relationship_graph else 0,
                **graph_analysis,
            },
            "vector_store": {
                "has_vectorstore": self.vectorstore is not None,
                "embedding_dimensions": getattr(self.vectorstore, "embedding_dimension", None)
                if self.vectorstore
                else None,
            },
            "repository_info": self.current_repo_info or {},
            "filter_stats": self.filter_manager.get_filter_summary(),
        }
        # Persist convenience values
        if self._last_commit_hash:
            stats["repository_info"]["commit_hash"] = self._last_commit_hash

        return stats

    def _generate_empty_stats(self, repo_identifier: str) -> dict[str, Any]:
        """Generate stats for empty repository"""
        return {
            "repository_identifier": repo_identifier,
            "loaded_from_cache": False,
            "documents": {"total": 0, "text_documents": 0, "code_documents": 0, "file_types": {}},
            "code_analysis": {"languages": [], "symbols_by_type": {}, "files_with_symbols": 0, "total_symbols": 0},
            "relationship_graph": {"nodes": 0, "edges": 0},
            "vector_store": {"has_vectorstore": False, "embedding_dimensions": None},
            "repository_info": self.current_repo_info or {},
            "filter_stats": self.filter_manager.get_filter_summary(),
        }

    def export_index_summary(self) -> dict[str, Any]:
        """Export comprehensive index summary"""
        if not self.all_documents:
            return {"error": "No documents indexed"}

        # Document summary
        doc_summary = {
            "total_documents": len(self.all_documents),
            "text_documents": self._get_text_count(),
            "code_documents": self._get_code_count(),
        }

        # File summary
        files = set(doc.metadata.get("file_path") for doc in self.all_documents if doc.metadata.get("file_path"))
        file_summary = {"total_files": len(files), "files": list(files)}

        # Language summary
        languages = set()
        for doc in self._iter_code_documents():
            if doc.metadata.get("language"):
                languages.add(doc.metadata["language"])

        # Graph summary
        graph_summary = {}
        if self.relationship_graph:
            graph_summary = {
                "nodes": self.relationship_graph.number_of_nodes(),
                "edges": self.relationship_graph.number_of_edges(),
                "node_types": list(
                    set(data.get("type", "unknown") for _, data in self.relationship_graph.nodes(data=True))
                ),
                "edge_types": list(
                    set(data.get("type", "unknown") for _, _, data in self.relationship_graph.edges(data=True))
                ),
            }

        return {
            "documents": doc_summary,
            "files": file_summary,
            "languages": list(languages),
            "relationship_graph": graph_summary,
            "has_vectorstore": self.vectorstore is not None,
            "repository_info": self.current_repo_info,
        }

    def clear_cache(self):
        """Clear all caches"""
        self.vectorstore_manager.clear_cache()
        self.graph_manager.clear_cache()
        logger.info("Cleared all index caches")

    # --- Indexer Protocol Convenience Accessors ---
    def get_commit_hash(self) -> str | None:
        """Return last indexed commit hash if available"""
        if self._last_commit_hash:
            return self._last_commit_hash
        if self.current_repo_info:
            return self.current_repo_info.get("commit_hash")
        return None

    def get_repo_root(self) -> str | None:
        """Return local repository root path"""
        return self._repo_root or (self.current_repo_info or {}).get("local_path")

    def cleanup_repositories(self):
        """Clean up repository clones"""
        self.repo_manager.cleanup_all()
        logger.info("Cleaned up repository clones")

    def __enter__(self):
        """Context manager entry"""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit with cleanup"""
        if self.cleanup_repos_on_exit:
            self.cleanup_repositories()

    # ------------------------------------------------------------------
    # Protocol Interface Methods (parity with GitHubIndexer)
    # These provide a consistent duck-typed surface expected by
    # OptimizedWikiGenerationAgent and wrapper logic.
    # ------------------------------------------------------------------
    def get_all_documents(self) -> list[Document]:
        """Return all indexed documents (may be empty if not indexed yet)."""
        return self.all_documents

    def get_relationship_graph(self) -> nx.DiGraph | None:
        """Return the relationship graph if available."""
        return self.relationship_graph

    def get_vectorstore(self):
        """Return underlying vectorstore instance (FAISS or similar)."""
        return self.vectorstore

    def search_documents(self, query: str, k: int = 20, include_scores: bool = False):
        """Search documents using similarity (delegates to VectorStoreManager)."""
        if not self.vectorstore:
            logger.warning("search_documents called before vectorstore is ready")
            return []
        if include_scores:
            return self.vectorstore_manager.search_with_score(query, k=k)
        return self.vectorstore_manager.search_combined(query, k=k)

    def search_by_type(self, query: str, doc_type: str = "all", k: int = 20):
        """Search subset of documents by type (text, code, or all)."""
        if not self.vectorstore:
            return []
        if doc_type == "text":
            return self.vectorstore_manager.search_text(query, k=k)
        if doc_type == "code":
            return self.vectorstore_manager.search_code(query, k=k)
        return self.vectorstore_manager.search_combined(query, k=k)

    def find_related_symbols(self, symbol_name: str, hops: int = 2) -> list[str]:
        """Find symbols related to a given symbol via graph traversal (duck-typed)."""
        if not self.relationship_graph or not symbol_name:
            return []
        matching_nodes = []
        for node_id in self.relationship_graph.nodes():
            # Enhanced graph builder nodes often end with ::<symbol_name>
            if node_id.endswith(f"::{symbol_name}") or symbol_name in node_id:
                matching_nodes.append(node_id)
        neighbors = set()
        for node in matching_nodes:
            neighbors.update(self.graph_manager.get_node_neighbors(self.relationship_graph, node, hops=hops))
        return list(neighbors)

    def get_symbol_document(self, symbol_key: str) -> Document | None:
        """Return document matching a symbol key. Accept flexible key patterns."""
        if not symbol_key:
            return None
        for doc in self._iter_code_documents():
            sym = doc.metadata.get("symbol_name") or doc.metadata.get("symbol")
            if not sym:
                continue
            file_path = doc.metadata.get("file_path") or doc.metadata.get("source")
            # Accept direct match or composed file_path::symbol_name
            if (
                symbol_key == sym
                or symbol_key.endswith(f"::{sym}")
                or (file_path and symbol_key in {f"{file_path}::{sym}", f"{sym}::{file_path}"})
            ):
                return doc
        return None

    def get_file_documents(self, file_path: str) -> list[Document]:
        """Return all documents associated with a given file path."""
        if not file_path:
            return []
        return [
            doc
            for doc in self.all_documents
            if (doc.metadata.get("file_path") == file_path or doc.metadata.get("source") == file_path)
        ]

    def get_documents_by_language(self, language: str) -> list[Document]:
        """Return code documents for a specific programming language."""
        if not language:
            return []
        return [doc for doc in self._iter_code_documents() if doc.metadata.get("language") == language]
