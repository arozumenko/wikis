"""Wiki Content Writer — per-page deepagent shell (#237).

This module provides:

``compute_page_budget(target_symbols, target_folders, target_docs, base)``
    Pure function; derives the tool-call budget for a single page from its
    spec.  Used by ``WikiContentWriter.compute_budget``.

``WikiContentWriter``
    Class shell wired to a ``WriterTools`` instance.  Manages the per-page
    deepagent lifecycle: budget calculation, agent construction, and (future)
    page generation.

Budget formula
--------------
budget = base
       + 2 * count(target_symbols where layer in {entry_point, public_api})
       + 1 * count(target_symbols where layer in {core_type, infrastructure})
       + 1 * count(target_folders)
       + 1 * count(target_docs)

Soft target: 15-30 calls.  Hard cap: 80 calls.

Out of scope in this PR (#237)
------------------------------
- Prompt text and citation contract (#11)
- LLM verifier pass (#12)
- Source-kind gating (#13)
- Link resolver (#14)
- Wiring into wiki_service / wiki_management (#16)
"""

from __future__ import annotations

import logging
from typing import Any

from .writer_tools import WriterTools

logger = logging.getLogger(__name__)

# ── Budget constants ──────────────────────────────────────────────────────────

BUDGET_SOFT_MIN: int = 15
BUDGET_SOFT_MAX: int = 30
BUDGET_HARD_CAP: int = 80

# Symbol layers weighted at 2× (high-importance surface)
_WEIGHT_2_LAYERS: frozenset[str] = frozenset({"entry_point", "public_api"})

# Symbol layers weighted at 1× (supporting types)
_WEIGHT_1_LAYERS: frozenset[str] = frozenset({"core_type", "infrastructure"})


# ── Budget formula ────────────────────────────────────────────────────────────


def compute_page_budget(
    target_symbols: list[tuple[str, str]],
    target_folders: list[str],
    target_docs: list[str],
    base: int = 5,
) -> int:
    """Compute the tool-call budget for a single wiki page.

    Parameters
    ----------
    target_symbols : list of (name, layer) tuples
        Symbols the page should document.  Layer is one of
        ``entry_point | public_api | core_type | infrastructure`` (others
        contribute 0).
    target_folders : list[str]
        Repository folders the page covers.
    target_docs : list[str]
        Documentation files referenced by the page.
    base : int
        Baseline budget (default 5) before any scaling.

    Returns
    -------
    int
        Tool-call budget, clamped to ``BUDGET_HARD_CAP``.
    """
    weight_2 = sum(1 for _, layer in target_symbols if layer in _WEIGHT_2_LAYERS)
    weight_1 = sum(1 for _, layer in target_symbols if layer in _WEIGHT_1_LAYERS)

    budget = base + 2 * weight_2 + 1 * weight_1 + len(target_folders) + len(target_docs)
    return min(budget, BUDGET_HARD_CAP)


# ── WikiContentWriter ─────────────────────────────────────────────────────────


class WikiContentWriter:
    """Per-page deepagent shell for wiki content generation.

    This class owns the lifecycle of a single page-generation run:
    budget calculation → (future) agent construction → (future) page output.

    Parameters
    ----------
    tools : WriterTools
        Tool surface the agent uses to explore the repository.
    llm_client : BaseChatModel | None
        Pre-configured LangChain chat model.  If ``None``, the agent will
        need to be provided one before ``generate_page`` can be called.
        # TODO #16: make required once wired into wiki_service
    max_iterations : int
        Hard iteration ceiling passed to the deepagent.  Defaults to
        ``BUDGET_HARD_CAP`` (the formula ensures the agent never needs more).

    Notes
    -----
    Actual page generation (prompts, citation contract, verifier) is
    implemented in subsequent issues (#11, #12, #13, #14) and wired in #16.
    """

    def __init__(
        self,
        tools: WriterTools,
        llm_client: Any | None = None,
        max_iterations: int = BUDGET_HARD_CAP,
    ) -> None:
        self.tools = tools
        self.llm_client = llm_client
        self.max_iterations = max_iterations

    # ── Budget ────────────────────────────────────────────────────────────

    def compute_budget(self, page_spec: dict[str, Any]) -> int:
        """Derive the tool-call budget from a page spec dict.

        Accepts the same key structure as ``PageSpec`` (from
        ``app.core.state.wiki_state``):

        - ``target_symbols``: list of (name, layer) tuples or list of str
          (bare names; layer defaults to ``""``).
        - ``target_folders``: list of folder paths.
        - ``target_docs``: list of doc file paths.
        - ``base``: baseline budget (optional, default 5).

        Returns
        -------
        int
            Tool-call budget, capped at ``BUDGET_HARD_CAP``.
        """
        raw_symbols = page_spec.get("target_symbols", [])

        # Normalise: accept both (name, layer) tuples and bare strings
        symbols: list[tuple[str, str]] = []
        for entry in raw_symbols:
            if isinstance(entry, (list, tuple)) and len(entry) >= 2:
                symbols.append((str(entry[0]), str(entry[1])))
            else:
                # Bare symbol name — layer unknown, contributes 0
                symbols.append((str(entry), ""))

        return compute_page_budget(
            target_symbols=symbols,
            target_folders=list(page_spec.get("target_folders", [])),
            target_docs=list(page_spec.get("target_docs", [])),
            base=int(page_spec.get("base", 5)),
        )

    # ── Future: generate_page ─────────────────────────────────────────────

    # TODO #16: implement generate_page(page_spec, wiki_id) -> WikiPage
    #   - compute budget from page_spec
    #   - build deepagent with tools.read_file / get_signature / etc.
    #   - run agent with max_iterations = compute_budget(page_spec)
    #   - apply citation contract (#11) + verifier (#12)
