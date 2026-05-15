"""Minimal-agent construction for #116 incremental refresh.

PR #142 closes the gap left by PR #139: the orchestrator's structural
regime needs an :class:`OptimizedWikiGenerationAgent` instance for the
:func:`make_agent_structural_handler` factory to wrap, but PR 5 shipped
a no-op stub instead.

This module provides :func:`build_agent_for_incremental_refresh`, which
constructs the *minimum viable agent* for single-page regen against a
previously-indexed wiki. The agent has:

* A real :class:`UnifiedRetriever` opened against the wiki's
  ``.wiki.db`` (same construction path as ``toolkit_bridge``).
* A stub indexer that exposes the few attributes the agent's single-
  page code paths read via ``getattr(..., None)`` — no filesystem
  cloning, no source-file index. Retrieval runs against the DB.
* The LLM the caller already constructed.

Degraded vs full regen: without a cloned repo on disk, the
``_get_relevant_content_for_page`` raw-file-read fallback (strategy 3)
won't fire, so retrieval is purely DB-backed. For most pages this
produces good enough context; pages that lean heavily on raw file
content will see slightly weaker prompts. Acceptable trade for
incremental cost; full ``/refresh`` remains the path for "I want
maximum quality."
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:  # pragma: no cover — import-only
    from app.config import Settings
    from app.core.agents.wiki_graph_optimized import OptimizedWikiGenerationAgent
    from app.core.storage.protocol import WikiStorageProtocol
    from app.models.db_models import WikiRecord
    from langchain_core.language_models import BaseLanguageModel

logger = logging.getLogger(__name__)


class _StubIncrementalIndexer:
    """Minimal indexer surface for the agent's single-page code paths.

    The agent reads a handful of indexer attributes defensively via
    ``getattr(..., None)``: ``relationship_graph``, ``source_dir`` /
    ``repo_path``, ``get_repo_root``, ``_unified_db_path``,
    ``graph_manager``. None of these are *required* for the single-page
    regen path — they're optimisation hooks. Providing them as ``None``
    lets the agent's fallback branches kick in.
    """

    relationship_graph: Any = None
    source_dir: str | None = None
    repo_path: str | None = None
    graph_manager: Any = None

    def __init__(self, unified_db_path: str | None = None) -> None:
        self._unified_db_path = unified_db_path

    def get_repo_root(self) -> str | None:
        # Returning None forces the agent to skip raw-file-read fallback
        # (strategy 3 in _get_relevant_content_for_page).
        return None

    def get_all_documents(self) -> list:
        # Only used in full-regen analyze_repository, not the single-page
        # path. Defensive return.
        return []


def build_agent_for_incremental_refresh(
    wiki_record: "WikiRecord",
    storage: "WikiStorageProtocol",
    llm: "BaseLanguageModel",
    settings: "Settings",
) -> "OptimizedWikiGenerationAgent | None":
    """Construct an :class:`OptimizedWikiGenerationAgent` scoped to one
    wiki's incremental refresh.

    Returns ``None`` if any prerequisite fails (LLM unavailable,
    embeddings init error, retriever construction failure). Callers
    treat ``None`` as "structural regen unavailable for this run" and
    fall back to the stub.

    Args:
        wiki_record: SQLAlchemy ``WikiRecord`` for the target wiki.
        storage: opened :class:`WikiStorageProtocol` handle for the
            wiki's ``.wiki.db``. Held by the agent for the duration of
            the regen; caller owns its lifecycle.
        llm: already-constructed LLM (typically from
            ``create_llm(settings)`` in the caller).
        settings: project settings; used for embeddings init.
    """
    from app.core.agents.wiki_graph_optimized import OptimizedWikiGenerationAgent
    from app.core.state.wiki_state import TargetAudience, WikiStyle
    from app.core.unified_retriever import UnifiedRetriever

    # Embeddings are optional for retrieval — UnifiedRetriever's hybrid
    # search degrades gracefully to FTS5 when no embedding function is
    # available. If init fails we keep going with FTS-only retrieval.
    embedding_fn = None
    embeddings = None
    try:
        from app.services.llm_factory import create_embeddings

        embeddings = create_embeddings(settings)
        if hasattr(embeddings, "embed_query"):
            embedding_fn = embeddings.embed_query
    except Exception as exc:  # noqa: BLE001
        logger.warning(
            "[agent_builder] embeddings init failed; retrieval falls back "
            "to FTS-only: %s",
            exc,
        )

    try:
        retriever_stack = UnifiedRetriever(
            db=storage,
            embedding_fn=embedding_fn,
            embeddings=embeddings,
        )
    except Exception as exc:  # noqa: BLE001
        logger.warning(
            "[agent_builder] UnifiedRetriever construction failed; "
            "structural regen unavailable: %s",
            exc,
        )
        return None

    indexer = _StubIncrementalIndexer()

    try:
        agent = OptimizedWikiGenerationAgent(
            indexer=indexer,
            retriever_stack=retriever_stack,
            llm=llm,
            repository_url=wiki_record.repo_url,
            branch=wiki_record.branch,
            wiki_style=WikiStyle.COMPREHENSIVE,
            target_audience=TargetAudience.MIXED,
            # Disable diagram requirements + progress for the single-
            # page path: we're regenerating one page, not running a
            # full wiki generation flow with quality-enhancement passes.
            require_diagrams=False,
            enable_progress_tracking=False,
            enable_quality_enhancement=False,
        )
    except Exception as exc:  # noqa: BLE001
        logger.warning(
            "[agent_builder] agent construction failed; structural regen "
            "unavailable: %s",
            exc,
        )
        return None

    return agent
