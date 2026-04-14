"""
Optimized API Wrapper for Wiki Toolkit using filesystem indexing.
"""

import logging
import os
import traceback
from typing import Any

from .agents.wiki_graph_optimized import OptimizedWikiGenerationAgent
from .filesystem_indexer import FilesystemRepositoryIndexer

logger = logging.getLogger(__name__)


class OptimizedWikiToolkitWrapper:
    """Optimized API Wrapper for building comprehensive wikis from repository analysis"""

    def __init__(
        self,
        github_repository: str,
        github_access_token: str | None = None,
        github_base_branch: str = "main",
        active_branch: str | None = None,
        github_base_url: str = "https://api.github.com",
        max_files: int | None = None,
        max_depth: int | None = None,
        rate_limit_delay: float = 0.1,
        cache_dir: str | None = None,
        force_rebuild_index: bool = False,
        llm: Any = None,
        **kwargs,
    ):
        """Initialize the optimized wrapper with simple constructor"""

        # Store configuration
        self.github_repository = github_repository
        self.github_access_token = github_access_token or os.getenv("GITHUB_ACCESS_TOKEN")
        self.github_base_branch = github_base_branch
        self.active_branch = active_branch or github_base_branch
        self.github_base_url = github_base_url
        self.max_files = max_files
        self.max_depth = max_depth
        self.rate_limit_delay = rate_limit_delay
        self.cache_dir = cache_dir or "/tmp/wiki_builder/cache"  # noqa: S108 — intentional temp directory for wiki cache
        self.force_rebuild_index = force_rebuild_index

        # Core components
        self.llm = llm
        self.indexer = None
        self.retriever_stack = None
        self.wiki_agent = None
        self.research_agent = None
        self.embeddings = kwargs.get("embeddings") if isinstance(kwargs, dict) else None

        # Output configuration
        self.wiki_output_path = "wiki_output"
        self.bucket_name = "wiki_artifacts"

        logger.info(f"Initialized WikiToolkitWrapper for repository: {self.github_repository}")

    def _initialize_components_sync(self, workspace_path: str = "."):
        """Initialize indexing components and agents (sync version)"""
        if self.indexer is None and self.github_repository:
            # dispatch_custom_event(
            #     name="thinking_step",
            #     data={
            #         "message": "Initializing repository indexer and retrieval components...",
            #         "tool_name": "_initialize_components",
            #         "toolkit": "wiki_toolkit"
            #     }
            # )

            # Initialize filesystem indexer
            self.indexer = FilesystemRepositoryIndexer(
                cache_dir=self.cache_dir,
                github_access_token=self.github_access_token,
                github_username=os.getenv("GITHUB_USERNAME"),
                embeddings=self.embeddings,
            )

            # Build repository index from filesystem clone
            self.indexer.index_repository(
                repository_url=self.github_repository,
                branch=self.active_branch,
                force_rebuild=self.force_rebuild_index,
                max_files=self.max_files,
            )

            # dispatch_custom_event(
            #     name="indexing_complete",
            #     data={
            #         "message": f"Indexed {index_stats.get('documents', {}).get('total', 0)} documents",
            #         "stats": index_stats,
            #         "tool_name": "github_indexer",
            #         "toolkit": "wiki_toolkit"
            #     }
            # )
            #
            # Initialize retriever from unified DB
            db_path = getattr(self.indexer, "_unified_db_path", None)
            if db_path:
                from .unified_db import UnifiedWikiDB
                from .unified_retriever import UnifiedRetriever

                db = UnifiedWikiDB(db_path, readonly=True)
                embedding_fn = None
                embeddings = getattr(self.indexer, "embeddings", None)
                if embeddings and hasattr(embeddings, "embed_query"):
                    embedding_fn = embeddings.embed_query
                self.retriever_stack = UnifiedRetriever(
                    db=db,
                    embedding_fn=embedding_fn,
                    embeddings=embeddings,
                )
            else:
                logger.warning("No unified DB path — retriever will be limited")

    def generate_wiki(
        self,
        query: str,
        wiki_title: str | None = None,
        include_research: bool = True,
        include_diagrams: bool = True,
        output_format: str = "json",
    ):
        """Generate comprehensive wiki from repository analysis with optimized processing"""
        try:
            if not self.github_repository:
                raise ValueError("GitHub repository not specified")

            repo_path = self.github_repository

            if not wiki_title:
                wiki_title = f"{repo_path.split('/')[-1]} Wiki"

            # dispatch_custom_event(
            #     name="thinking_step",
            #     data={
            #         "message": f"Starting wiki generation for {repo_path} with query: {query}",
            #         "tool_name": "generate_wiki",
            #         "toolkit": "wiki_toolkit"
            #     }
            # )

            # Initialize components
            self._initialize_components_sync()

            # Execute wiki generation using LangGraph agent
            # dispatch_custom_event(
            #     name="thinking_step",
            #     data={
            #         "message": "Executing wiki generation agent...",
            #         "tool_name": "generate_wiki",
            #         "toolkit": "wiki_toolkit"
            #     }
            # )

            # Create wiki configuration

            # Create agent with configuration parameters directly
            from .state.wiki_state import TargetAudience, WikiStyle

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
                enable_quality_enhancement=True,
                graph_text_index=getattr(
                    getattr(self.indexer, "graph_manager", None),
                    "fts_index",
                    None,
                ),
            )

            # Generate wiki with user message using standard LangGraph pattern
            wiki_result = wiki_agent.generate_wiki(query)

            if wiki_result and wiki_result.get("success"):
                generation_summary = wiki_result.get("generation_summary", {})
                total_pages = generation_summary.get("total_pages", len(wiki_result.get("generated_pages", {})))
                total_sections = generation_summary.get("total_sections", 0)
                total_diagrams = generation_summary.get("total_diagrams", 0)

                result_message = f"""
**🚀 Wiki Generation Complete!**

Repository: {repo_path}
Title: {wiki_title}
Query: {query}

**📊 Processing Results:**
- Include Research: {include_research}
- Include Diagrams: {include_diagrams}
- Output Format: {output_format}
- Status: ✅ Complete

**🎯 Generated Content:**
- Wiki Pages: {total_pages}
- Sections: {total_sections}
- Diagrams: {total_diagrams}

**📈 Processing Statistics:**
- Execution Time: {wiki_result.get("execution_time", 0):.2f}s

The comprehensive wiki has been successfully generated.
"""
                return {
                    "success": True,
                    "result": result_message,
                    "artifacts": wiki_result.get("artifacts", []),
                    "execution_time": wiki_result.get("execution_time", 0),
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
