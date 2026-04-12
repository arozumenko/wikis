"""Multi-wiki component builder for project-scoped ask / codemap pipelines.

Loads EngineComponents for every wiki in a project and merges them into a
single set of components that can be fed into AskEngine or the codemap
pipeline without any per-wiki branching.

Public API:
    build_multi_wiki_components(project_id, user_id, storage, settings, tier)
        → (merged_components, per_wiki_components)
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any

from app.config import Settings
from app.services.toolkit_bridge import EngineComponents, build_engine_components
from app.storage.base import ArtifactStorage

logger = logging.getLogger(__name__)


async def build_multi_wiki_components(
    project_id: str,
    user_id: str,
    storage: ArtifactStorage,
    settings: Settings,
    tier: str = "high",
) -> tuple[EngineComponents, dict[str, EngineComponents]]:
    """Load and merge engine components for all wikis in a project.

    Resolves the list of wiki IDs from the project, loads each wiki's
    EngineComponents concurrently, then merges them into a single
    EngineComponents suitable for passing to AskEngine / codemap pipeline.

    Also returns the per-wiki components dict so callers can build
    ``MultiGraphQueryService`` from individual ``GraphQueryService`` instances.

    Args:
        project_id: The project whose wikis should be loaded.
        user_id: ID of the requesting user (used for access control).
        storage: Artifact storage backend.
        settings: Application settings.
        tier: LLM tier ("high" or "low").

    Returns:
        Tuple of (merged_components, per_wiki_components) where:
        - merged_components is an EngineComponents with a MultiWikiRetrieverStack
          and a merged (empty combined) graph suitable for LLM access.
        - per_wiki_components maps wiki_id → EngineComponents for graph-level ops.

    Raises:
        ValueError: If the project is not found/accessible or has no loadable wikis.
    """
    from app.db import get_session_factory
    from app.services.project_service import ProjectService

    # Resolve wiki IDs for this project via a short-lived session
    session_factory = get_session_factory()
    async with session_factory() as session:
        project_svc = ProjectService(session)
        wikis = await project_svc.list_project_wikis(project_id, user_id=user_id)

    if wikis is None:
        raise ValueError(f"Project '{project_id}' not found or not accessible to user '{user_id}'")
    if not wikis:
        raise ValueError(f"Project '{project_id}' has no accessible wikis")

    wiki_ids = [w.id for w in wikis]
    logger.info("Building multi-wiki components for project %s: %s", project_id, wiki_ids)

    # Load each wiki's components concurrently
    async def _load_one(wiki_id: str) -> tuple[str, EngineComponents | None]:
        try:
            comp = await build_engine_components(wiki_id, storage, settings, tier=tier)
            return wiki_id, comp
        except Exception as exc:
            logger.warning("Failed to load components for wiki %s: %s", wiki_id, exc)
            return wiki_id, None

    results = await asyncio.gather(*[_load_one(wid) for wid in wiki_ids])

    # Filter out wikis that failed to load
    loaded: dict[str, EngineComponents] = {
        wid: comp for wid, comp in results if comp is not None
    }
    if not loaded:
        raise ValueError(
            f"No wikis in project '{project_id}' could be loaded (all {len(wiki_ids)} failed)"
        )

    # Build per-wiki stacks for fan-out retrieval
    wiki_stacks: list[tuple[str, Any]] = [
        (wid, comp.retriever_stack)
        for wid, comp in loaded.items()
        if comp.retriever_stack is not None
    ]
    if not wiki_stacks:
        raise ValueError(
            f"No wikis in project '{project_id}' have a usable retriever stack"
        )

    # Merge into a single EngineComponents using MultiWikiRetrieverStack
    from app.core.multi_retriever import MultiWikiRetrieverStack

    multi_stack = MultiWikiRetrieverStack(wiki_stacks)

    # Use the first wiki's llm / repo_analysis as the "primary" for generation
    primary = next(iter(loaded.values()))

    # Build a MultiGraphQueryService so all tools search across every wiki
    from app.core.code_graph.graph_query_service import GraphQueryService
    from app.core.code_graph.multi_graph_query_service import MultiGraphQueryService

    individual_services = {}
    for wid, comp in loaded.items():
        if comp.code_graph is not None:
            fts = comp.graph_manager.fts_index if comp.graph_manager else None
            individual_services[wid] = GraphQueryService(comp.code_graph, fts_index=fts)

    multi_gqs: MultiGraphQueryService | None = None
    if individual_services:
        multi_gqs = MultiGraphQueryService(individual_services)

    # Build a combined code_graph (use primary's; multi-graph ops use query_service)
    merged = EngineComponents(
        retriever_stack=multi_stack,
        graph_manager=primary.graph_manager,
        code_graph=primary.code_graph,
        repo_analysis=primary.repo_analysis,
        llm=primary.llm,
        repo_path=primary.repo_path,
        query_service=multi_gqs,
    )

    # Override repo_analysis description with project-level description (project takes precedence)
    async with session_factory() as session:
        from sqlalchemy import select

        from app.models.db_models import ProjectRecord

        try:
            proj_result = await session.execute(
                select(ProjectRecord).where(ProjectRecord.id == project_id)
            )
            proj_record = proj_result.scalar_one_or_none()
            if proj_record and proj_record.description:
                if merged.repo_analysis is None:
                    merged.repo_analysis = {}
                merged.repo_analysis["description"] = proj_record.description
        except Exception:
            logger.warning("Could not load project description for %s", project_id)

    return merged, loaded
