"""
Context Eviction Middleware for the Structure Planner.

Solves the "LLM stall at ~80% coverage" problem on large repos.

Problem:
  After ~60-70 pages and ~100+ tool calls, the conversation context grows
  so large that LLM inference degrades from ~8s/step to minutes/step,
  eventually stalling entirely. This happens because the accumulated
  query_graph responses, batch_define_pages outputs, and think records
  fill the context window even after SummarizationMiddleware compresses.

Solution:
  After each batch_define_pages call that includes a CONTINUE signal,
  evict ALL messages and replace with a compact continuation prompt.
  This works because:
    1. The StructureCollector (tool instance) holds all durable state:
       covered_dirs, pages, discovered_dirs, sections, metadata.
    2. The CONTINUE signal is recomputed from tool state on every call.
    3. The LLM only needs to know: what's uncovered + what to do next.
       Both are in the CONTINUE signal.

Hook point:
  before_model — runs before each LLM call. Checks if the last tool
  result was batch_define_pages with a CONTINUE signal. If so, replaces
  messages with a single compact continuation message.

Pattern:
  Uses langgraph.types.Overwrite to replace the entire message list,
  same pattern as PatchToolCallsMiddleware.
"""

import logging
import os
from typing import Any

from langchain.agents.middleware import AgentMiddleware, AgentState
from langchain_core.messages import HumanMessage, ToolMessage
from langgraph.runtime import Runtime
from langgraph.types import Overwrite

logger = logging.getLogger(__name__)

# Feature flag: disable eviction to compare behavior
ENABLE_CONTEXT_EVICTION = os.getenv("WIKIS_CONTEXT_EVICTION", "1") != "0"


class ContextEvictionMiddleware(AgentMiddleware):
    """
    Evicts stale conversation history after batch_define_pages with CONTINUE.

    The StructureCollector tool instance holds all durable state (covered_dirs,
    pages, discovered_dirs). The CONTINUE signal is recomputed from that state.
    So we can safely discard the entire conversation and start fresh with just
    the CONTINUE signal content.

    This keeps the context window small and focused regardless of how many
    tool calls have accumulated, preventing the LLM inference degradation
    observed at ~80% coverage on large repos.
    """

    def before_model(self, state: AgentState, runtime: Runtime[Any]) -> dict[str, Any] | None:
        """Check if eviction should trigger before the next LLM call."""
        if not ENABLE_CONTEXT_EVICTION:
            return None

        messages = state.get("messages", [])
        if not messages:
            return None

        # Log message types for debugging (always, not just on eviction)
        msg_types = {}
        for m in messages:
            t = type(m).__name__
            msg_types[t] = msg_types.get(t, 0) + 1
        has_system = "SystemMessage" in msg_types
        logger.debug(
            f"[CONTEXT_EVICTION] before_model: {len(messages)} messages in state: {msg_types}. "
            f"SystemMessage in state: {has_system} (expected: False — system prompt is injected by LangGraph from closure)"
        )

        # Find the last tool message from batch_define_pages
        continue_content = self._find_continue_signal(messages)
        if continue_content is None:
            return None

        # Count messages being evicted for logging
        evicted_count = len(messages)
        logger.info(
            f"[CONTEXT_EVICTION] Evicting {evicted_count} messages ({msg_types}). "
            f"Replacing with single HumanMessage continuation prompt. "
            f"System prompt is NOT in messages — it is injected separately by LangGraph."
        )

        # Build fresh context: single HumanMessage with continuation instructions
        continuation_message = HumanMessage(content=self._build_continuation_prompt(continue_content))

        return {"messages": Overwrite([continuation_message])}

    def _find_continue_signal(self, messages: list) -> str | None:
        """
        Check if the most recent ToolMessage is batch_define_pages with CONTINUE.

        IMPORTANT: Only triggers on the MOST RECENT ToolMessage, not any historical
        batch_define_pages. This prevents mid-cycle eviction — if query_graph results
        came in after batch_define_pages, those results must be preserved for the model
        to use in the next batch_define_pages call.

        Returns the content string if found, None otherwise.
        """
        for msg in reversed(messages):
            # Skip non-tool messages (AIMessage, HumanMessage)
            if not isinstance(msg, ToolMessage):
                continue

            # Found the most recent ToolMessage — check if it's batch_define_pages
            name = getattr(msg, "name", "") or ""
            if name != "batch_define_pages":
                # Most recent tool call is NOT batch_define_pages → don't evict.
                # The model is mid-cycle (e.g., query_graph results just came in).
                return None

            content = msg.content if isinstance(msg.content, str) else str(msg.content)

            # Check for CONTINUE signal markers
            if "🚨" in content and ("CONTINUE" in content or "COVERAGE" in content):
                return content

            # batch_define_pages without CONTINUE = agent is finishing → don't evict
            return None

        return None

    def _build_continuation_prompt(self, continue_content: str) -> str:
        """
        Build a compact continuation message from the CONTINUE signal.

        The CONTINUE signal already contains:
        - Coverage percentage and counts
        - Page budget remaining
        - Uncovered directory suggestions with query_graph instructions
        - Action instructions

        We wrap it with context so the LLM understands it's continuing
        a structure planning session.
        """
        return (
            "You are continuing a wiki structure planning session. "
            "Previous conversation was evicted to keep context fresh. "
            "All your prior work (pages, sections, coverage) is preserved in tool state — "
            "your next batch_define_pages call will show the updated status.\n\n"
            "QUALITY REMINDERS:\n"
            "- Each page MUST have 10-15+ target_symbols from query_graph results\n"
            "- Use capability-based page names (what it does, not where it lives)\n"
            "- Group related symbols from the same functional area into one page\n"
            "- Include classes, functions, constants, interfaces — not just top-level names\n\n"
            "Here is your current status from the last batch:\n\n"
            f"{continue_content}\n\n"
            "Continue by calling query_graph() for the uncovered areas listed above, "
            "then batch_define_pages() to create pages for them. "
            "Do NOT repeat pages you already created — the tool tracks this automatically."
        )
