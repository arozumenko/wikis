"""Ask (Q&A) service — agentic Q&A against generated wikis using AskEngine."""

from __future__ import annotations

import logging
from collections.abc import AsyncGenerator

from app.config import Settings
from app.models.api import AskRequest, AskResponse, SourceReference
from app.services.toolkit_bridge import ComponentCache, EngineComponents, build_engine_components
from app.storage.base import ArtifactStorage

logger = logging.getLogger(__name__)


class AskService:
    """Handles question-answering against generated wikis using AskEngine."""

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
                tier="low",
            ),
        )

    async def ask_stream(self, request: AskRequest) -> AsyncGenerator[dict, None]:
        """Stream answer events using AskEngine."""
        from app.core.ask_engine import AskEngine

        components = await self._get_components(request.wiki_id)

        chat_history = (
            [{"role": msg.role, "content": msg.content} for msg in request.chat_history]
            if request.chat_history
            else None
        )

        from app.core.ask_engine import AskConfig

        engine = AskEngine(
            retriever_stack=components.retriever_stack,
            graph_manager=components.graph_manager,
            code_graph=components.code_graph,
            repo_analysis=components.repo_analysis,
            llm_client=components.llm,
            config=AskConfig(similarity_threshold=self.settings.ask_similarity_threshold),
        )

        async for event in engine.ask(
            question=request.question,
            chat_history=chat_history,
        ):
            yield event

    async def ask_sync(self, request: AskRequest) -> AskResponse:
        """Non-streaming: collect final answer from event stream."""
        final_answer = ""
        sources: list[SourceReference] = []
        step_count = 0

        async for event in self.ask_stream(request):
            event_type = event.get("event_type", "")
            # Support both legacy (ask_complete/ask_error) and MCP (task_complete/task_failed) events
            if event_type in ("ask_complete", "task_complete"):
                data = event.get("data", {})
                final_answer = data.get("answer", "")
                step_count = data.get("steps", 0)
                for s in data.get("sources", []):
                    if isinstance(s, dict):
                        sources.append(
                            SourceReference(
                                file_path=s.get("file_path") or s.get("source", ""),
                                snippet=s.get("snippet"),
                                symbol=s.get("symbol"),
                                symbol_type=s.get("symbol_type") or s.get("type"),
                                relevance_score=s.get("relevance_score"),
                            )
                        )
            elif event_type in ("ask_error", "task_failed"):
                raise RuntimeError(event.get("data", {}).get("error", "Ask failed"))

        return AskResponse(
            answer=final_answer,
            sources=sources,
            tool_steps=step_count,
        )
