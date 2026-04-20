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

        # Initialize graph manager (for analysis stats only; persistent storage is via unified DB)
        self.graph_manager = GraphManager(cache_dir=cache_dir)

        # Embeddings instance for unified DB population
        self.embeddings = embeddings

        # State containers
        self.text_documents = []
        self.code_documents = []
        self.all_documents = []
        self.relationship_graph = None
        self.last_index_stats = None
        self._doc_type_counts = {
            "text": 0,
            "code": 0,
        }

        # Unified DB state
        self._unified_db_path = None  # Path to {cache_key}.wiki.db
        self._unified_db_cache_key = None  # Cache key for lookup

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
        """Try to load complete index from unified DB cache."""
        try:
            commit_hash = self._last_commit_hash
            cache_dir = self.graph_manager.cache_dir or "/tmp/wiki_builder/cached_graphs"  # noqa: S108

            # Generate cache key matching what _write_unified_db uses
            import hashlib

            hash_input = f"{repo_path}:{commit_hash or ''}"
            cache_key = hashlib.md5(hash_input.encode()).hexdigest()  # noqa: S324 — cache key, not security
            db_path = os.path.join(str(cache_dir), f"{cache_key}.wiki.db")

            from .storage import is_postgres_backend, open_storage

            if is_postgres_backend():
                # PostgreSQL: check by opening storage and querying node count
                try:
                    db = open_storage(repo_id=cache_key, db_path=db_path, readonly=True)
                except Exception:
                    logger.debug("[cache-miss] PostgreSQL storage not available for %s", cache_key)
                    return None
            else:
                if not os.path.isfile(db_path):
                    logger.debug("[cache-miss] unified DB not found: %s", db_path)
                    return None
                db = open_storage(repo_id=cache_key, db_path=db_path, readonly=True)

            stats = db.stats()
            if stats["node_count"] < 1:
                logger.debug("[cache-miss] unified DB empty: %s", db_path)
                db.close()
                return None

            # Detect broken import: nodes present but zero edges almost
            # certainly means a prior from_networkx() crashed midway.
            if stats["node_count"] > 1 and stats["edge_count"] == 0:
                logger.warning(
                    "[cache-miss] unified DB has %d nodes but 0 edges — "
                    "likely a failed import, forcing re-index: %s",
                    stats["node_count"], db_path,
                )
                db.close()
                return None

            # Reconstruct NX graph from DB
            self.relationship_graph = db.to_networkx()

            # Reconstruct all_documents from DB nodes
            self.all_documents = self._reconstruct_documents_from_db(db)

            # Separate docs
            text_docs = [doc for doc in self.all_documents if doc.metadata.get("chunk_type") == "text"]
            code_docs = [doc for doc in self.all_documents if doc.metadata.get("chunk_type") in ["symbol", "code"]]
            self._set_document_splits(text_docs, code_docs)

            # Store unified DB state for downstream use
            self._unified_db_path = db_path
            self._unified_db_cache_key = cache_key

            db.close()
            logger.info(
                "[cache-hit] Loaded unified DB for %s: %d nodes, %d edges, %d docs — %s",
                repo_identifier,
                stats["node_count"],
                stats["edge_count"],
                len(self.all_documents),
                db_path,
            )
            self._emit_progress(0.20, "Index loaded from cache")
            return self._generate_index_stats(repo_identifier, from_cache=True)
        except Exception as e:
            logger.debug("[cache-error] %s", e)
            return None

    @staticmethod
    def _reconstruct_documents_from_db(db) -> list[Document]:
        """Reconstruct LangChain Documents from unified DB nodes."""
        rows = db.get_all_nodes()
        documents = []
        for d in rows:
            page_content = d.get("source_text") or ""
            metadata = {
                "node_id": d.get("node_id", ""),
                "symbol_name": d.get("symbol_name", ""),
                "symbol_type": d.get("symbol_type", ""),
                "source": d.get("rel_path", ""),
                "rel_path": d.get("rel_path", ""),
                "file_name": d.get("file_name", ""),
                "file_path": d.get("rel_path", ""),
                "language": d.get("language", ""),
                "start_line": d.get("start_line", 0),
                "end_line": d.get("end_line", 0),
                "chunk_type": d.get("chunk_type", d.get("symbol_type", "")),
                "docstring": d.get("docstring", ""),
                "signature": d.get("signature", ""),
            }
            documents.append(Document(page_content=page_content, metadata=metadata))
        return documents

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

        # Step 3: Write unified DB (replaces FAISS + graph pickle + BM25)
        self._emit_progress(0.20, "Building unified wiki database...")
        logger.info("Writing unified wiki DB (graph + embeddings + clusters)...")
        if this and getattr(this, "module", None):
            this.module.invocation_thinking(
                "I am on phase indexing\nBuilding unified wiki DB...\nReasoning: Single DB file consolidates graph, embeddings, FTS5, and clusters.\nNext: Compute index stats."
            )

        # If streaming, we need to materialise documents first
        if documents_iter and not self.all_documents:
            self.all_documents = list(documents_iter)
            self._add_repository_metadata()
            text_docs = [doc for doc in self.all_documents if doc.metadata.get("chunk_type") == "text"]
            code_docs = [doc for doc in self.all_documents if doc.metadata.get("chunk_type") in ["symbol", "code"]]
            self._set_document_splits(text_docs, code_docs)
            documents_iter = None  # consumed

        try:
            with resource_monitor("index.unified_db_write", logger):
                self._write_unified_db(repo_path, repo_identifier)

            self._emit_progress(0.25, f"Indexed {len(self.all_documents)} documents ({self._get_text_count()} text, {self._get_code_count()} code)")
        except Exception as e:
            logger.error(f"Failed to write unified DB: {e}")
            import traceback

            logger.error(f"Full traceback: {traceback.format_exc()}")
            # Non-fatal — in-memory graph + docs still available for generation
            logger.warning(
                "Unified DB write failed. In-memory graph and documents are available, "
                "but Ask path will not be functional until re-indexed."
            )

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

    def _write_unified_db(self, repo_path: str, repo_identifier: str) -> None:
        """Write unified SQLite DB from current graph + metadata.

        Produces a ``{cache_key}.wiki.db`` file in the cache directory that
        consolidates the code graph, FTS5 full-text index, sqlite-vec
        embeddings, Phase 2 topology enrichment, and Phase 3 clustering
        into a single file.  This replaces the legacy FAISS + BM25 +
        graph pickle + docstore stack.

        The same cache_key is registered in ``cache_index.json`` under the
        ``"unified_db"`` namespace so that ``toolkit_bridge.py`` (Ask path)
        can locate the correct DB for a given wiki.
        """
        import hashlib
        import json

        cache_dir = str(self.graph_manager.cache_dir or "/tmp/wiki_builder/cached_graphs")  # noqa: S108
        cache_key = hashlib.md5(  # noqa: S324 — cache key, not security
            f"{repo_path}:{self._last_commit_hash or ''}".encode()
        ).hexdigest()
        db_filename = f"{cache_key}.wiki.db"
        db_path = os.path.join(cache_dir, db_filename)

        os.makedirs(cache_dir, exist_ok=True)

        logger.info(
            "Writing unified DB: %s (%d nodes, %d edges)",
            db_path,
            self.relationship_graph.number_of_nodes(),
            self.relationship_graph.number_of_edges(),
        )

        from .storage import open_storage, is_postgres_backend

        _emb_obj = self.embeddings
        _embed_query_fn = getattr(_emb_obj, "embed_query", None) if _emb_obj else None
        _embed_batch_fn = getattr(_emb_obj, "embed_documents", None) if _emb_obj else None

        udb = open_storage(repo_id=cache_key, db_path=db_path)
        try:
            # Phase 1a — Populate graph (nodes + edges)
            udb.from_networkx(self.relationship_graph)

            # Phase 1b — Populate vector embeddings (repo_vec table)
            if _embed_batch_fn:
                try:
                    # Default 256: ``text-embedding-3-large`` accepts up
                    # to 2048 inputs / 300k tokens per request, and code
                    # symbols are typically short.  64 was conservative
                    # and made indexing wall-clock dominated by HTTP RTT.
                    n_embedded = udb.populate_embeddings(
                        embedding_fn=_embed_batch_fn,
                        batch_size=int(os.getenv("WIKI_EMBED_BATCH_SIZE", "256")),
                    )
                    logger.info("Unified DB: %d node embeddings stored", n_embedded)
                except Exception as exc:
                    logger.warning("Embedding population failed (non-fatal): %s", exc)
            else:
                logger.info("No embeddings model available — repo_vec will be empty")

            # Phase 2 — Edge weighting, hub detection, orphan resolution
            try:
                from .graph_topology import run_phase2

                phase2_stats = run_phase2(
                    db=udb,
                    G=self.relationship_graph,
                    embedding_fn=_embed_query_fn,
                    embed_batch_fn=_embed_batch_fn,
                )
                logger.info(
                    "Phase 2 complete: %d edges weighted, %d hubs",
                    phase2_stats.get("weighting", {}).get("edges_weighted", 0),
                    phase2_stats.get("hubs", {}).get("count", 0),
                )
            except Exception as exc:
                logger.warning("Phase 2 graph topology failed (non-fatal): %s", exc)
                phase2_stats = None

            # Phase 3 — Hierarchical Leiden clustering
            try:
                from .graph_clustering import run_phase3

                phase2_hubs = set()
                if phase2_stats:
                    phase2_hubs = set(phase2_stats.get("hubs", {}).get("node_ids", []))
                phase3_stats = run_phase3(
                    db=udb,
                    G=self.relationship_graph,
                    hubs=phase2_hubs or None,
                )

                logger.info(
                    "Phase 3 complete: %s",
                    phase3_stats.get("persistence", {}),
                )
            except Exception as exc:
                logger.warning("Phase 3 clustering failed (non-fatal): %s", exc)

            # ────────────────────────────────────────────────
            # Post-pass hygiene & observability (§11.7, §11.8, §11.12)
            # Run AFTER clustering so file-contraction/Leiden see the raw
            # arch graph; demotion only affects retrieval-eligibility flags
            # and the god-nodes report is purely informational.
            # Vendored demotion runs FIRST so the local-constant pass and
            # god-nodes report both reflect the post-vendored state.
            # ────────────────────────────────────────────────
            try:
                vendored = udb.demote_vendored_code()
                if vendored:
                    logger.info(
                        "Post-pass: demoted %d vendored / third-party nodes (§11.12)",
                        vendored,
                    )
            except Exception as exc:
                logger.warning("demote_vendored_code failed (non-fatal): %s", exc)

            try:
                demoted = udb.demote_local_constants()
                if demoted:
                    logger.info(
                        "Post-pass: demoted %d file-local constants (§11.8)",
                        demoted,
                    )
            except Exception as exc:
                logger.warning("demote_local_constants failed (non-fatal): %s", exc)

            # Class member-promotion (§11.4 / P1) — must run AFTER vendored
            # demotion (so we don't promote bundled gtest classes) and
            # BEFORE god-nodes computation (so the report reflects the
            # final connectivity).
            try:
                promoted = udb.promote_class_members()
                if promoted:
                    logger.info(
                        "Post-pass: synthesized %d member_uses edges (§11.4)",
                        promoted,
                    )
            except Exception as exc:
                logger.warning("promote_class_members failed (non-fatal): %s", exc)

            try:
                god = udb.compute_god_nodes(top_n=20)
                udb.set_meta("god_nodes", god)
                logger.info(
                    "Post-pass: god_nodes report stored (top symbol degree=%s)",
                    (god.get("by_symbol_type") or [{}])[0].get("degree"),
                )
            except Exception as exc:
                logger.warning("compute_god_nodes failed (non-fatal): %s", exc)

            # Store build metadata
            udb.set_meta("repo_identifier", repo_identifier)
            udb.set_meta("commit_hash", self._last_commit_hash)
            udb.set_meta("repo_path", repo_path)
            udb.set_meta("node_count", self.relationship_graph.number_of_nodes())
            udb.set_meta("edge_count", self.relationship_graph.number_of_edges())
            udb.set_meta("doc_count", len(self.all_documents) if self.all_documents else 0)

            stats = udb.stats()
            logger.info(
                "Unified DB written: %d nodes, %d edges, %.1f MB — %s",
                stats["node_count"],
                stats["edge_count"],
                stats["db_size_mb"],
                db_path,
            )
        finally:
            udb.close()

        # Store state for downstream use
        self._unified_db_path = db_path
        self._unified_db_cache_key = cache_key

        # Register in cache_index.json so toolkit_bridge can find it
        self._register_unified_db_cache(repo_identifier, cache_key, cache_dir)

    def _register_unified_db_cache(self, repo_identifier: str, cache_key: str, cache_dir: str) -> None:
        """Register the unified DB in cache_index.json for Ask-path lookup."""
        import json
        from pathlib import Path

        index_file = Path(cache_dir) / "cache_index.json"
        try:
            if index_file.exists():
                with open(index_file) as f:
                    index = json.load(f)
            else:
                index = {}

            # Store under "unified_db" namespace
            udb_section = index.setdefault("unified_db", {})
            udb_section[repo_identifier] = cache_key

            # Also store a ref for branch-only lookups (owner/repo:branch → cache_key)
            refs = index.setdefault("refs", {})
            # Extract owner/repo:branch from full identifier (may include commit)
            parts = repo_identifier.split(":")
            if len(parts) >= 2:
                short_ref = f"{parts[0]}:{parts[1]}"
                refs[short_ref] = repo_identifier

            with open(index_file, "w") as f:
                json.dump(index, f, indent=2)

            logger.info("Registered unified DB in cache_index.json: %s → %s", repo_identifier, cache_key)
        except Exception as e:
            logger.warning("Failed to register unified DB in cache_index.json: %s", e)

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
            "unified_db": {
                "has_unified_db": self._unified_db_path is not None,
                "unified_db_path": self._unified_db_path,
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
            "unified_db": {"has_unified_db": False, "unified_db_path": None},
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
            "has_unified_db": self._unified_db_path is not None,
            "repository_info": self.current_repo_info,
        }

    def clear_cache(self):
        """Clear all caches"""
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
