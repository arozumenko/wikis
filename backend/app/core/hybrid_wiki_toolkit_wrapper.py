"""
Wiki Toolkit Wrapper (filesystem indexing only).
Supports multi-provider repositories: GitHub, GitLab, Bitbucket, Azure DevOps.
"""

import logging
import os
import traceback
from typing import Any

try:  # Optional thinking emitter (safe no-op if task context not present)
    from tools import this  # type: ignore
except Exception:  # pragma: no cover
    this = None  # type: ignore

from .agents.wiki_graph_optimized import OptimizedWikiGenerationAgent
from .filesystem_indexer import FilesystemRepositoryIndexer
from .github_client import StandaloneGitHubClient
from .repo_providers import GitCloneConfig, LocalPathConfig

logger = logging.getLogger(__name__)


class HybridWikiToolkitWrapper:
    """
    Wrapper for filesystem-based indexing with multi-provider support.

    Supports: GitHub, GitLab, Bitbucket, Azure DevOps.
    """

    def __init__(
        self,
        # New multi-provider configuration
        repo_config: dict[str, Any] | None = None,
        clone_config: GitCloneConfig | None = None,
        # Legacy GitHub-specific fields (for backward compatibility)
        github_repository: str | None = None,
        github_access_token: str | None = None,
        github_username: str | None = None,
        github_base_branch: str = "main",
        active_branch: str | None = None,
        github_base_url: str = "https://api.github.com",
        # Common configuration
        max_files: int | None = None,
        max_depth: int | None = None,
        rate_limit_delay: float = 0.1,
        cache_dir: str | None = None,
        force_rebuild_index: bool = False,
        llm: Any = None,
        embeddings: Any = None,
        indexing_method: str = "filesystem",  # deprecated (filesystem only)
        cleanup_repos_on_exit: bool = True,
        **kwargs,
    ):
        """Initialize the wrapper with configurable multi-provider support"""

        # Store new multi-provider configuration
        self.repo_config = repo_config or {}
        self.clone_config = clone_config

        # Extract values from repo_config or fall back to legacy parameters
        if isinstance(self.clone_config, LocalPathConfig):
            # Local filesystem path — no clone needed
            self.repository = self.clone_config.local_path
            self.branch = self.clone_config.branch or "none"
            self.provider_type = "local"
        elif self.clone_config:
            # Use clone_config if available (preferred)
            self.repository = self.clone_config.repo_identifier
            self.branch = self.clone_config.branch
            self.provider_type = self.clone_config.provider.value
        elif self.repo_config:
            # Use repo_config dict
            self.repository = self.repo_config.get("repository") or github_repository
            self.branch = self.repo_config.get("branch") or github_base_branch
            self.provider_type = self.repo_config.get("provider_type", "github")
        else:
            # Legacy GitHub-only mode
            self.repository = github_repository
            self.branch = github_base_branch
            self.provider_type = "github"

        # Clean repository name (for GitHub compatibility)
        if self.provider_type == "github" and self.repository:
            self.repository = StandaloneGitHubClient.clean_repository_name(self.repository)

        # Legacy GitHub fields (maintained for backward compatibility)
        self.github_repository = self.repository  # Alias for existing code
        self.github_access_token = github_access_token or os.getenv("GITHUB_ACCESS_TOKEN")
        self.github_base_branch = self.branch
        self.github_username = github_username or os.getenv("GITHUB_USERNAME")
        self.active_branch = active_branch or self.branch
        self.github_base_url = github_base_url

        # Common configuration
        self.max_files = max_files
        self.max_depth = max_depth
        self.rate_limit_delay = rate_limit_delay
        self.cache_dir = cache_dir or "/tmp/wiki_builder/cache"  # noqa: S108 — intentional temp directory for wiki cache
        self.force_rebuild_index = force_rebuild_index
        self.indexing_method = "filesystem"  # Only filesystem is supported
        self.cleanup_repos_on_exit = cleanup_repos_on_exit

        # Core components
        self.llm = llm
        self.llm_low = kwargs.pop("llm_low", None)
        self.progress_callback = kwargs.pop("progress_callback", None)
        self.indexer = None  # type: ignore[assignment]
        self.retriever_stack = None
        self.wiki_agent = None
        self.research_agent = None
        self.embeddings = embeddings

        # Output configuration
        self.wiki_output_path = "wiki_output"
        self.bucket_name = "wiki_artifacts"

        if not self.repository:
            logger.warning("No repository specified")
            if this and getattr(this, "module", None):
                this.module.invocation_thinking(
                    "I am on phase initialization\nHybrid wiki toolkit initialized\n"
                    "Reasoning: Core components prepared (indexer placeholder, retriever stack deferred).\n"
                    "Next: Build or load repository index before content generation."
                )

        logger.info(
            f"Initialized HybridWikiToolkitWrapper for repository: {self.repository} "
            f"(provider: {self.provider_type}) using {self.indexing_method} indexing method"
        )

    def _initialize_components_sync(self, workspace_path: str = "."):
        """Initialize indexing components and agents (sync version)"""
        if self.indexer is None:
            self._initialize_filesystem_indexer()

            # Initialize retriever from unified DB
            db_path = getattr(self.indexer, "_unified_db_path", None)
            if db_path:
                from .unified_db import UnifiedWikiDB
                from .unified_retriever import UnifiedRetriever

                db = UnifiedWikiDB(db_path, readonly=True)
                embedding_fn = None
                if self.embeddings and hasattr(self.embeddings, "embed_query"):
                    embedding_fn = self.embeddings.embed_query
                self.retriever_stack = UnifiedRetriever(
                    db=db,
                    embedding_fn=embedding_fn,
                    embeddings=self.embeddings,
                )
            else:
                logger.warning("No unified DB path available — retriever will be limited")

    def _initialize_filesystem_indexer(self):
        """Initialize filesystem-based indexer with multi-provider support"""
        logger.info(f"Initializing filesystem-based repository indexer (provider: {self.provider_type})...")

        self.indexer = FilesystemRepositoryIndexer(
            cache_dir=self.cache_dir,
            api_filters=None,
            max_workers=8,
            cleanup_repos_on_exit=self.cleanup_repos_on_exit,
            # Pass clone_config for multi-provider support
            clone_config=self.clone_config,
            # Legacy GitHub credentials (fallback if clone_config not available)
            github_access_token=self.github_access_token,
            github_username=self.github_username,
            embeddings=self.embeddings,
            progress_callback=self.progress_callback,
        )

        # Build repository index using filesystem approach
        index_stats = self.indexer.index_repository(
            repository_url=self.repository,
            branch=self.branch,
            force_rebuild=self.force_rebuild_index,
            force_reclone=False,  # Only reclone if explicitly requested
            max_files=self.max_files,
        )

        # Update active_branch from actual cloned branch (handles master vs main, etc.)
        actual_branch = index_stats.get("actual_branch")
        if actual_branch:
            logger.info(f"Using actual branch from clone: {actual_branch} (requested: {self.branch})")
            self.active_branch = actual_branch
            self.branch = actual_branch
            self.github_base_branch = actual_branch

        logger.info(f"Filesystem indexing complete: {index_stats.get('documents', {}).get('total', 0)} documents")

    def generate_wiki(
        self,
        query: str,
        wiki_title: str | None = None,
        include_research: bool = True,
        include_diagrams: bool = True,
        output_format: str = "json",
        planner_type: str = "agent",
        exclude_tests: bool = False,
    ):
        """Generate comprehensive wiki from repository analysis with hybrid processing"""
        try:
            if not self.github_repository:
                raise ValueError("GitHub repository not specified")

            repo_path = self.github_repository

            if not wiki_title:
                wiki_title = f"{repo_path.split('/')[-1]} Wiki"

            # Initialize components
            if this and getattr(this, "module", None):
                this.module.invocation_thinking(
                    f"I am on phase initialization\nStarting wiki generation for {repo_path} (query: {query})\nReasoning: Kick off end-to-end pipeline (index -> analysis -> structure -> pages).\nNext: Initialize indexer & retrieval stack."
                )
            self._initialize_components_sync()
            if this and getattr(this, "module", None):
                this.module.invocation_thinking(
                    "I am on phase initialization\nIndexing complete – preparing generation agent\nReasoning: Repository artifacts available for higher-level semantic synthesis.\nNext: Dispatch structure + page generation workflow."
                )

            # Import optimized agent components
            from .state.wiki_state import TargetAudience, WikiStyle

            # Initialize wiki generation agent with optimized configuration
            wiki_agent = OptimizedWikiGenerationAgent(
                indexer=self.indexer,
                retriever_stack=self.retriever_stack,
                llm=self.llm,
                repository_url=self.github_repository,
                branch=self.active_branch,
                bucket_name=self.bucket_name,
                wiki_style=WikiStyle.COMPREHENSIVE,
                target_audience=TargetAudience.MIXED,
                require_diagrams=include_diagrams,
                enable_progress_tracking=False,
                llm_low=self.llm_low,
                enable_quality_enhancement=True,
                progress_callback=self.progress_callback,
                graph_text_index=getattr(
                    getattr(self.indexer, "graph_manager", None),
                    "fts_index",
                    None,
                ),
                planner_type=planner_type,
                exclude_tests=exclude_tests,
            )

            # Generate wiki with user message using standard LangGraph pattern
            if this and getattr(this, "module", None):
                this.module.invocation_thinking(
                    "I am on phase page_generation\nDispatching page generation tasks\nReasoning: Parallel drafting accelerates throughput while leveraging shared repository context.\nNext: For each page, retrieve focused context then invoke LLM."
                )
            wiki_result = wiki_agent.generate_wiki(query)
            if this and getattr(this, "module", None):
                this.module.invocation_thinking(
                    "I am on phase completion\nWiki generation workflow finished\nReasoning: All requested phases executed successfully.\nNext: Return artifacts & summaries to caller."
                )

            if wiki_result and wiki_result.get("success"):
                generation_summary = wiki_result.get("generation_summary", {})
                total_pages = generation_summary.get("total_pages", len(wiki_result.get("generated_pages", {})))
                total_sections = generation_summary.get("total_sections", 0)

                workflow_errors = wiki_result.get("errors") if isinstance(wiki_result, dict) else None
                if not isinstance(workflow_errors, list):
                    workflow_errors = []
                failed_pages = wiki_result.get("failed_pages") if isinstance(wiki_result, dict) else None
                if not isinstance(failed_pages, list):
                    failed_pages = []

                result_message = f"""
            **🚀 Wiki Generation Complete!**

            Repository: {repo_path}
            Title: {wiki_title}
            Query: {query}

            **📊 Processing Results:**
            - Status: ✅ Complete

            **🎯 Generated Content:**
            - Wiki Pages: {total_pages}
            - Sections: {total_sections}

            **📈 Processing Statistics:**
            - Execution Time: {wiki_result.get("execution_time", 0):.2f}s

            The comprehensive wiki has been successfully generated.
            """

                if workflow_errors or failed_pages:
                    # Keep this short; detailed errors are returned in structured fields below.
                    result_message += "\n\n⚠️ Note: Some steps reported warnings/errors during generation."

                # Get commit hash from indexer for cache key generation
                commit_hash = None
                if self.indexer and hasattr(self.indexer, "_last_commit_hash"):
                    commit_hash = self.indexer._last_commit_hash

                # Persist repository analysis into the unified DB so the Ask
                # path can read it without a standalone _analysis.json file.
                repo_context = wiki_result.get("repository_context")
                if repo_context:
                    db_path = getattr(self.indexer, "_unified_db_path", None)
                    if db_path:
                        try:
                            from .unified_db import UnifiedWikiDB

                            with UnifiedWikiDB(db_path) as udb:
                                udb.set_meta("repository_analysis", repo_context)
                            logger.info("Stored repository analysis in unified DB (%d chars)", len(repo_context))
                        except Exception as exc:
                            logger.warning("Failed to store analysis in unified DB: %s", exc)

                return {
                    "success": True,
                    "result": result_message,
                    "artifacts": wiki_result.get("artifacts", []),
                    "execution_time": wiki_result.get("execution_time", 0),
                    "errors": workflow_errors,
                    "failed_pages": failed_pages,
                    # Pass through repository_context for Ask tool
                    "repository_context": repo_context,
                    # Pass commit hash for cache key consistency
                    "commit_hash": commit_hash,
                    # Pass actual branch from clone (handles master vs main, etc.)
                    "branch": self.active_branch,
                    "provider_type": self.provider_type,
                }
            else:
                return {
                    "success": False,
                    "error": "Failed to generate wiki. Wiki generation agent failed to generate wiki.",
                }

        except Exception as e:
            logger.error(f"Failed to generate wiki: {e}")
            logger.error(traceback.format_exc())
            raise e

    def estimate_index_tokens(self) -> int:
        """Estimate total tokens across all indexed documents.

        Uses the project's TokenCounter for accuracy when tiktoken is
        available, falling back to the chars/4 heuristic otherwise.
        """
        if not self.indexer or not hasattr(self.indexer, "all_documents") or not self.indexer.all_documents:
            return 0
        try:
            from .token_counter import get_token_counter

            counter = get_token_counter()
            return counter.count_documents(self.indexer.all_documents)
        except Exception as exc:
            logger.debug(f"estimate_index_tokens: token_counter unavailable, using heuristic: {exc}")
            total_chars = sum(len(getattr(doc, "page_content", "") or "") for doc in self.indexer.all_documents)
            return total_chars // 4

    def get_index_summary(self) -> dict:
        """Get summary of current index"""
        if not self.indexer:
            return {"error": "No indexer initialized"}

        summary = self.indexer.export_index_summary()
        summary["indexing_method"] = self.indexing_method
        return summary

    def clear_cache(self):
        """Clear all caches"""
        if self.indexer:
            self.indexer.clear_cache()

    def cleanup_repositories(self):
        """Clean up repository clones (filesystem indexer only)"""
        if hasattr(self.indexer, "cleanup_repositories"):
            self.indexer.cleanup_repositories()

    def __enter__(self):
        """Context manager entry"""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit with cleanup"""
        if self.cleanup_repos_on_exit:
            self.cleanup_repositories()


# Convenience functions for backward compatibility
def create_wiki_toolkit_wrapper(*args, **kwargs):
    """Create wiki toolkit wrapper with default filesystem indexing"""
    kwargs.setdefault("indexing_method", "filesystem")
    return HybridWikiToolkitWrapper(*args, **kwargs)


def create_github_api_wrapper(*args, **kwargs):
    """Create wiki toolkit wrapper with GitHub API indexing (legacy)"""
    kwargs["indexing_method"] = "github_api"
    return HybridWikiToolkitWrapper(*args, **kwargs)


def create_filesystem_wrapper(*args, **kwargs):
    """Create wiki toolkit wrapper with filesystem indexing"""
    kwargs["indexing_method"] = "filesystem"
    return HybridWikiToolkitWrapper(*args, **kwargs)
