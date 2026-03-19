"""Deep research service — multi-step research against generated wikis."""

from __future__ import annotations

import logging
from collections.abc import AsyncGenerator

from app.config import Settings
from app.core.deep_research.research_engine import DeepResearchEngine, ResearchConfig
from app.models.api import ResearchRequest, ResearchResponse, SourceReference
from app.services.toolkit_bridge import ComponentCache, EngineComponents, build_engine_components
from app.storage.base import ArtifactStorage

logger = logging.getLogger(__name__)


class ResearchService:
    """Handles deep research queries using multi-step agents."""

    def __init__(self, settings: Settings, storage: ArtifactStorage) -> None:
        self.settings = settings
        self.storage = storage
        self._cache = ComponentCache(
            max_size=settings.ask_cache_max_wikis,
            ttl_seconds=settings.ask_cache_ttl_seconds,
        )

    def evict_cache(self, wiki_id: str) -> bool:
        """Remove cached engine components for a wiki."""
        return self._cache.evict(wiki_id)

    async def _get_components(self, wiki_id: str) -> EngineComponents:
        """Load or return cached engine components for a wiki."""
        return await self._cache.get_or_build(
            wiki_id,
            factory=lambda: build_engine_components(
                wiki_id,
                self.storage,
                self.settings,
                tier="high",
            ),
        )

    async def research_stream(self, request: ResearchRequest) -> AsyncGenerator[dict, None]:
        """Stream research events (research_start, thinking_step, research_complete)."""
        components = await self._get_components(request.wiki_id)
        engine = DeepResearchEngine(
            retriever_stack=components.retriever_stack,
            graph_manager=components.graph_manager,
            code_graph=components.code_graph,
            repo_analysis=components.repo_analysis,
            llm_client=components.llm,
            config=ResearchConfig(
                research_type=request.research_type,
                similarity_threshold=self.settings.research_similarity_threshold,
            ),
            repo_path=components.repo_path,
        )

        async for event in engine.research(question=request.question):
            yield event

    async def research_sync(self, request: ResearchRequest) -> ResearchResponse:
        """Non-streaming: collect final answer from event stream."""
        final_answer = ""
        sources: list[SourceReference] = []
        steps: list[str] = []

        async for event in self.research_stream(request):
            event_type = event.get("event_type", "")
            if event_type == "thinking_step":
                step_data = event.get("data", {})
                step_desc = step_data.get("tool", step_data.get("description", ""))
                if step_desc:
                    steps.append(step_desc)
            # Support both legacy (research_complete/research_error) and MCP (task_complete/task_failed) events
            elif event_type in ("research_complete", "task_complete"):
                data = event.get("data", {})
                final_answer = data.get("report", "") or data.get("answer", "")
                raw_sources = data.get("sources", [])
                sources = [
                    SourceReference(
                        file_path=s.get("file_path") or s.get("source", ""),
                        line_start=s.get("line_start"),
                        line_end=s.get("line_end"),
                        snippet=s.get("snippet"),
                        symbol=s.get("symbol"),
                        symbol_type=s.get("symbol_type") or s.get("type"),
                        relevance_score=s.get("relevance_score"),
                    )
                    for s in raw_sources
                    if isinstance(s, dict)
                ]
            elif event_type in ("research_error", "task_failed"):
                raise RuntimeError(event.get("data", {}).get("error", "Research failed"))

        return ResearchResponse(
            answer=final_answer,
            sources=sources,
            research_steps=steps,
        )
