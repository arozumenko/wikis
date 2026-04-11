"""Shared helper for loading EngineComponents across all wikis in a project.

Used by both AskService and ResearchService so the logic isn't duplicated.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any

from app.services.toolkit_bridge import EngineComponents

logger = logging.getLogger(__name__)


async def build_multi_wiki_components(
    project_id: str,
    user_id: str | None,
    get_components_fn: Any,  # async callable(wiki_id: str) -> EngineComponents
    session_factory: Any,  # async session factory (or callable returning one)
) -> EngineComponents:
    """Load and compose EngineComponents for all wikis in a project.

    Fetches all WikiRecords accessible to *user_id* for *project_id*,
    loads each wiki's EngineComponents in parallel (skipping failures),
    and wraps them in a MultiWikiRetrieverStack.

    Args:
        project_id: The project whose wikis to load.
        user_id: The requesting user (None = anonymous / auth disabled).
        get_components_fn: Async callable ``(wiki_id) -> EngineComponents``
            — typically ``AskService._get_components`` or equivalent.
        session_factory: Async SQLAlchemy session factory.

    Returns:
        EngineComponents where ``retriever_stack`` is a ``MultiWikiRetrieverStack``
        spanning all accessible wikis, and ``llm`` comes from the first
        successfully loaded wiki.

    Raises:
        ValueError: If the project does not exist / is not accessible, or has
            no wikis with usable components.
    """
    from app.core.multi_retriever import MultiWikiRetrieverStack
    from app.services.project_service import ProjectService

    # --- Resolve wikis for project ---
    async with session_factory() as session:
        svc = ProjectService(session)
        wiki_records = await svc.list_project_wikis(project_id, user_id or "")

    if wiki_records is None:
        raise ValueError(f"Project {project_id!r} not found or not accessible")
    if not wiki_records:
        raise ValueError(f"Project {project_id!r} has no accessible wikis")

    wiki_ids = [w.id for w in wiki_records]

    # --- Load each wiki's components in parallel ---
    async def load_one(wiki_id: str) -> tuple[str, EngineComponents] | None:
        try:
            comp = await get_components_fn(wiki_id)
            return wiki_id, comp
        except Exception:
            logger.warning("Failed to load components for wiki %s — skipping", wiki_id, exc_info=True)
            return None

    results = await asyncio.gather(*[load_one(wid) for wid in wiki_ids])
    loaded: list[tuple[str, EngineComponents]] = [r for r in results if r is not None]

    if not loaded:
        raise ValueError(f"No usable components available for any wiki in project {project_id!r}")

    # --- Compose MultiWikiRetrieverStack ---
    wiki_stacks = [
        (wiki_id, comp.retriever_stack)
        for wiki_id, comp in loaded
        if comp.retriever_stack is not None
    ]
    multi_retriever = MultiWikiRetrieverStack(wiki_stacks)

    # Use first successfully loaded wiki for LLM + graph (primary wiki)
    first_wiki_id, first_comp = loaded[0]
    return EngineComponents(
        retriever_stack=multi_retriever,
        graph_manager=first_comp.graph_manager,
        code_graph=first_comp.code_graph,
        repo_analysis=first_comp.repo_analysis,
        llm=first_comp.llm,
        repo_path=first_comp.repo_path,
    )
