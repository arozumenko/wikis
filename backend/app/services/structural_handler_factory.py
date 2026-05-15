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

from app.core.storage.incremental import compute_content_hash
from app.services.incremental_regen_service import (
    PageBodyWriter,
    StructuralHandler,
)

if TYPE_CHECKING:  # pragma: no cover — import-only
    from app.core.agents.wiki_graph_optimized import OptimizedWikiGenerationAgent
    from app.core.storage.protocol import WikiStorageProtocol
    from app.services.incremental_regen import PageRegime

logger = logging.getLogger(__name__)


def make_agent_structural_handler(
    agent: "OptimizedWikiGenerationAgent",
    *,
    storage: "WikiStorageProtocol",
    repository_context: str,
    write_page_body: PageBodyWriter,
) -> StructuralHandler:
    """Build a :data:`StructuralHandler` that regenerates one page via
    the wiki-generation agent.

    Args:
        agent: an already-constructed
            :class:`OptimizedWikiGenerationAgent`. The orchestrator
            calls back into this for each structural page.
        storage: the same wiki storage handle the orchestrator uses.
            After a successful body write the handler stamps the new
            ``content_hash`` into ``wiki_pages``, matching the
            behavior of the trivial and edit regimes — without this,
            the next incremental run would see a hash mismatch and
            re-classify the page as MODIFIED, looping forever.
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

    def _handler(page: "PageRegime") -> "str | None":
        new_content = agent.regenerate_single_page(
            page.page_id, repository_context=repository_context,
        )
        if new_content is None:
            logger.info(
                "[structural_handler] page %s: agent returned None — "
                "structural regen failed",
                page.page_id,
            )
            return "agent returned None (missing PageSpec or LLM error)"
        try:
            write_page_body(page.page_id, new_content)
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "[structural_handler] page %s: write failed: %s",
                page.page_id, exc,
            )
            return f"page-body write failed: {exc}"

        # Stamp the new content_hash into wiki_pages so the next
        # incremental run sees this page as unchanged (parity with
        # trivial + edit regimes). Without this, every structurally-
        # regenerated page would loop "structural" on every run.
        try:
            row = storage.get_wiki_page(page.page_id)
            if row is not None:
                row["content_hash"] = compute_content_hash(new_content)
                storage.upsert_wiki_page(row)
        except Exception as exc:  # noqa: BLE001 — fail-soft on the stamp
            logger.warning(
                "[structural_handler] page %s: content_hash stamp failed "
                "(next regen will re-process this page): %s",
                page.page_id, exc,
            )
            # Don't fail the overall regen — body is written; only the
            # bookkeeping hash is stale. Logged loudly so ops can spot it.

        return None  # success

    return _handler
