"""Ask (Q&A) service — agentic Q&A against generated wikis using AskEngine."""

from __future__ import annotations

import asyncio
import json
import logging
import uuid
from collections.abc import AsyncGenerator

from app.config import Settings
from app.models.api import AskRequest, AskResponse, SourceReference
from app.models.db_models import QARecord
from app.models.qa_api import AskResult, QARecordingPayload
from app.services.qa_service import QAService
from app.services.toolkit_bridge import ComponentCache, EngineComponents, build_engine_components
from app.storage.base import ArtifactStorage

logger = logging.getLogger(__name__)


def _to_chat_dicts(request: AskRequest) -> list[dict] | None:
    """Convert chat history from Pydantic models to plain dicts for the engine."""
    if not request.chat_history:
        return None
    return [{"role": m.role, "content": m.content} for m in request.chat_history]


def _parse_sources_json(raw: str | None) -> list[SourceReference]:
    """Parse a JSON-encoded sources list into SourceReference objects."""
    if not raw:
        return []
    return [
        SourceReference(
            file_path=s.get("file_path", ""),
            snippet=s.get("snippet"),
            symbol=s.get("symbol"),
            symbol_type=s.get("symbol_type"),
            relevance_score=s.get("relevance_score"),
        )
        for s in json.loads(raw)
        if isinstance(s, dict)
    ]


def _cache_hit_payload(
    qa_id: str,
    request: AskRequest,
    record: QARecord,
    has_context: bool,
) -> QARecordingPayload:
    """Build a QARecordingPayload from a cache-hit QARecord."""
    return QARecordingPayload(
        qa_id=qa_id,
        wiki_id=request.wiki_id,
        question=request.question,
        answer=record.answer,
        sources_json=record.sources_json or "[]",
        tool_steps=record.tool_steps or 0,
        mode=record.mode or "fast",
        user_id=None,
        is_cache_hit=True,
        source_qa_id=record.id,
        embedding=None,
        has_context=has_context,
        source_commit_hash=record.source_commit_hash,
    )


class AskService:
    """Handles question-answering against generated wikis using AskEngine."""

    def __init__(
        self,
        settings: Settings,
        storage: ArtifactStorage,
        qa_service: QAService | None = None,
    ) -> None:
        self.settings = settings
        self.storage = storage
        self._qa_service = qa_service
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

    async def _get_multi_wiki_components(
        self,
        project_id: str,
        user_id: str | None,
    ) -> EngineComponents:
        """Load and compose EngineComponents for all wikis in a project."""
        from app.db import get_session_factory
        from app.services.project_service import ProjectService

        session_factory = get_session_factory()
        async with session_factory() as session:
            ps = ProjectService(session)
            wikis = await ps.list_project_wikis(project_id, user_id or "")

        if wikis is None:
            raise ValueError(f"Project '{project_id}' not found or not accessible")

        if not wikis:
            raise ValueError(f"Project '{project_id}' has no accessible wikis")

        # Load per-wiki components in parallel, tolerating individual failures
        results = await asyncio.gather(
            *[self._get_components(w.id) for w in wikis],
            return_exceptions=True,
        )

        valid_pairs = [
            (wiki, comp)
            for wiki, comp in zip(wikis, results, strict=False)
            if not isinstance(comp, Exception)
        ]

        if not valid_pairs:
            raise ValueError(f"No wikis in project '{project_id}' could be loaded")

        from app.core.multi_retriever import MultiWikiRetrieverStack

        merged_retriever = MultiWikiRetrieverStack(
            [(wiki.id, comp.retriever_stack) for wiki, comp in valid_pairs]
        )

        # Base on first valid wiki's components, swap in the merged retriever
        base_comp = valid_pairs[0][1]
        from dataclasses import replace

        return replace(base_comp, retriever_stack=merged_retriever)

    async def _stream_from_agent(
        self, request: AskRequest, user_id: str | None = None
    ) -> AsyncGenerator[dict, None]:
        """Stream raw events from AskEngine (no cache logic)."""
        from app.core.ask_engine import AskConfig, AskEngine

        if request.project_id:
            components = await self._get_multi_wiki_components(request.project_id, user_id)
        else:
            components = await self._get_components(request.wiki_id)

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
            chat_history=_to_chat_dicts(request),
        ):
            yield event

    async def ask_stream(
        self, request: AskRequest, user_id: str | None = None
    ) -> AsyncGenerator[dict, None]:
        """Stream answer events, with cache lookup when QA service is available."""
        qa_id = str(uuid.uuid4())
        has_context = bool(request.chat_history)
        embedding = None

        # Cache lookup (only for single-wiki asks — project asks skip cache)
        if self._qa_service and request.wiki_id and not request.project_id:
            cached_record, embedding = await self._qa_service.lookup_cache(
                request.wiki_id, request.question, chat_history=_to_chat_dicts(request),
            )
            if cached_record:
                # Cache hit — emit events matching what the agent emits (ask_complete)
                yield {"event_type": "answer_chunk", "data": {"chunk": cached_record.answer}}
                sources_list = json.loads(cached_record.sources_json) if cached_record.sources_json else []
                yield {
                    "event_type": "ask_complete",
                    "data": {
                        "answer": cached_record.answer,
                        "sources": sources_list,
                        "steps": cached_record.tool_steps or 0,
                        "qa_id": qa_id,
                    },
                }
                payload = _cache_hit_payload(qa_id, request, cached_record, has_context)
                asyncio.create_task(self._record_safely(payload))
                return

        # Cache miss — get current commit for recording, then stream from agent
        source_commit_hash = None
        if self._qa_service:
            source_commit_hash = await self._qa_service.get_wiki_commit_hash(request.wiki_id)

        completed = False
        final_answer = ""
        sources_json = "[]"
        tool_steps = 0

        try:
            async for event in self._stream_from_agent(request, user_id=user_id):
                event_type = event.get("event_type", "")
                if event_type in ("task_complete", "ask_complete"):
                    data = event.get("data", {})
                    final_answer = data.get("answer", "")
                    tool_steps = data.get("steps", 0)
                    sources_json = json.dumps(data.get("sources", []))
                    event.setdefault("data", {})["qa_id"] = qa_id
                    completed = True
                yield event
        finally:
            if completed and self._qa_service:
                payload = QARecordingPayload(
                    qa_id=qa_id, wiki_id=request.wiki_id,
                    question=request.question, answer=final_answer,
                    sources_json=sources_json, tool_steps=tool_steps,
                    mode="fast", user_id=user_id,
                    is_cache_hit=False, source_qa_id=None,
                    embedding=embedding, has_context=has_context,
                    source_commit_hash=source_commit_hash,
                )
                asyncio.create_task(self._record_safely(payload))
            elif not completed:
                logger.warning("Ask stream did not complete — recording skipped for %s", qa_id)

    async def ask_sync(
        self, request: AskRequest, user_id: str | None = None
    ) -> AskResult:
        """Non-streaming: collect final answer from event stream."""
        qa_id = str(uuid.uuid4())
        has_context = bool(request.chat_history)
        cached_record = None
        embedding = None

        # Cache lookup (only for single-wiki asks — project asks skip cache)
        if self._qa_service and request.wiki_id and not request.project_id:
            cached_record, embedding = await self._qa_service.lookup_cache(
                request.wiki_id, request.question, chat_history=_to_chat_dicts(request),
            )

        if cached_record:
            response = AskResponse(
                answer=cached_record.answer,
                sources=_parse_sources_json(cached_record.sources_json),
                tool_steps=cached_record.tool_steps or 0,
                qa_id=qa_id,
            )
            payload = _cache_hit_payload(qa_id, request, cached_record, has_context)
            return AskResult(response=response, recording=payload)

        # Cache miss — run agent
        final_answer = ""
        sources: list[SourceReference] = []
        step_count = 0

        async for event in self._stream_from_agent(request, user_id=user_id):
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

        response = AskResponse(
            answer=final_answer, sources=sources,
            tool_steps=step_count, qa_id=qa_id,
        )

        recording = None
        if self._qa_service:
            source_commit_hash = await self._qa_service.get_wiki_commit_hash(request.wiki_id)
            sources_json = json.dumps([s.model_dump() for s in sources])
            recording = QARecordingPayload(
                qa_id=qa_id, wiki_id=request.wiki_id,
                question=request.question, answer=final_answer,
                sources_json=sources_json, tool_steps=step_count,
                mode="fast", user_id=user_id,
                is_cache_hit=False, source_qa_id=None,
                embedding=embedding, has_context=has_context,
                source_commit_hash=source_commit_hash,
            )

        return AskResult(response=response, recording=recording)

    async def _record_safely(self, payload: QARecordingPayload) -> None:
        """Fire-and-forget recording wrapper."""
        from dataclasses import asdict

        try:
            if self._qa_service:
                await self._qa_service.record_interaction(**asdict(payload))
        except Exception:
            logger.error("QA recording failed", exc_info=True)
