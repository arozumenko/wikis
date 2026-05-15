"""Factory that builds a production :data:`StructuralHandler` callback
wrapping :meth:`OptimizedWikiGenerationAgent.regenerate_single_page`.

PR 3 (#133) left ``IncrementalRegenService.structural_handler`` as an
injected callable so the orchestrator stays unit-testable without the
full wiki-generation pipeline. This factory closes that gap by
producing the production wiring: a ``StructuralHandler`` that calls
into the agent's new ``regenerate_single_page`` method and persists
the result via the same page-body writer the rest of the orchestrator
uses.

PR 5 (`WikiService.incremental_refresh`) wires the resulting callback
into :class:`IncrementalRegenService` alongside the other callbacks.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from app.services.incremental_regen_service import (
    PageBodyWriter,
    StructuralHandler,
)

if TYPE_CHECKING:  # pragma: no cover — import-only
    from app.core.agents.wiki_graph_optimized import OptimizedWikiGenerationAgent
    from app.services.incremental_regen import PageRegime

logger = logging.getLogger(__name__)


def make_agent_structural_handler(
    agent: "OptimizedWikiGenerationAgent",
    *,
    repository_context: str,
    write_page_body: PageBodyWriter,
) -> StructuralHandler:
    """Build a :data:`StructuralHandler` that regenerates one page via
    the wiki-generation agent.

    Args:
        agent: an already-constructed
            :class:`OptimizedWikiGenerationAgent`. The orchestrator
            calls back into this for each structural page.
        repository_context: the repository-context string the agent
            uses for retrieval. Same value as full regen's
            ``WikiState.repository_context``.
        write_page_body: the same writer the orchestrator uses for the
            trivial/edit regimes. Passed in so all three regimes
            persist via the same callback (one writer, three readers).

    Returns:
        A ``StructuralHandler`` (``Callable[[PageRegime], bool]``)
        suitable for :class:`IncrementalRegenService`. Returns ``True``
        on successful regen, ``False`` when the agent couldn't
        reconstruct the page (legacy row, missing spec, LLM error).
    """

    def _handler(page: "PageRegime") -> bool:
        new_content = agent.regenerate_single_page(
            page.page_id, repository_context=repository_context,
        )
        if new_content is None:
            logger.info(
                "[structural_handler] page %s: agent returned None — "
                "structural regen failed",
                page.page_id,
            )
            return False
        try:
            write_page_body(page.page_id, new_content)
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "[structural_handler] page %s: write failed: %s",
                page.page_id, exc,
            )
            return False
        return True

    return _handler
