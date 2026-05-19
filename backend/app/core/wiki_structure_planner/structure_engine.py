"""
Wiki Structure Planner Engine - Using langchain-ai/deepagents

This is the lean implementation for generating wiki structures using DeepAgents.
It follows the same pattern as deep_research:
- Minimal system prompt with clear workflow
- FilesystemMiddleware for repository exploration (ls, glob, grep, read_file)
- No pre-embedded context - agent discovers via tools
- Structured JSON output

Events are captured via LangGraph's stream for progress tracking.
"""

from __future__ import annotations

import logging
import os
import time
from dataclasses import dataclass
from datetime import datetime
from typing import Any

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage

from .structure_skeleton import StructureSkeleton, build_skeleton

logger = logging.getLogger(__name__)

# Safety limits - Scaled for big repos (5K+ files, 100K+ graph nodes)
# For big repos like Redpanda: 100 pages need ~150+ tool calls
MAX_ITERATIONS = 100  # Increased from 30 for big repo exploration
DEFAULT_PAGE_BUDGET = 20


@dataclass
class StructurePlannerConfig:
    """Configuration for structure planning session.

    The new workflow uses explore_tree and analyze_module for deep discovery,
    so filesystem tool limits are less critical. These are safety limits.

    For big repos (5K+ files), max_iterations should be dynamically scaled
    based on page_budget. Formula: page_budget * 1.5 + sqrt(dir_count) * 2
    """

    page_budget: int = DEFAULT_PAGE_BUDGET
    target_audience: str = "developers"
    wiki_type: str = "technical"
    max_iterations: int = MAX_ITERATIONS
    # Auto-detected effective depth (from detect_effective_depth())
    effective_depth: int = 5
    # Repository file count for dynamic FTS k scaling
    repository_file_count: int = 0
    # Tool call limits - generous for deep exploration of big repos
    max_ls_calls: int = 25  # Was 15 - more for deep exploration
    max_glob_calls: int = 15  # Was 8 - for pattern matching
    max_read_calls: int = 25  # Was 15 - for reading key files
    max_grep_calls: int = 15  # Was 8 - for code search
    # Documentation clusters that need explicit coverage
    doc_clusters: list = None  # List of doc cluster dicts from repo profiling


class WikiStructurePlannerEngine:
    """
    Lean engine for generating wiki structures using DeepAgents.

    This implementation:
    - Uses FilesystemMiddleware for repo exploration (ls, glob, grep, read_file)
    - Does NOT embed large context in prompts
    - Lets the agent discover structure via tools
    - Returns structured WikiStructureSpec JSON

    Usage:
        engine = WikiStructurePlannerEngine(
            repo_root="/path/to/repo",
            llm_client=chat_model,
            config=StructurePlannerConfig(page_budget=25)
        )

        result = engine.plan_structure()
        # result is a dict ready for WikiStructureSpec.model_validate()
    """

    def __init__(
        self,
        repo_root: str,
        llm_client: BaseChatModel | None = None,
        llm_settings: dict | None = None,
        config: StructurePlannerConfig | None = None,
        code_graph: Any | None = None,
        graph_text_index: Any | None = None,
    ):
        """
        Initialize the structure planner engine.

        Args:
            repo_root: Absolute path to the repository root
            llm_client: Pre-configured LLM client
            llm_settings: LLM configuration dict (used if llm_client not provided)
            config: Planning configuration
            code_graph: Optional nx.DiGraph with code symbols (from indexer)
            graph_text_index: Optional GraphTextIndex (FTS5) for O(log N) lookups
        """
        if not repo_root or not os.path.isdir(repo_root):
            raise ValueError(f"repo_root must be a valid directory: {repo_root}")

        self.repo_root = repo_root
        self.llm_client = llm_client
        self.code_graph = code_graph  # For enhanced module analysis
        self.graph_text_index = graph_text_index  # FTS5 index for O(log N) lookups
        self.llm_settings = llm_settings or {}
        self.config = config or StructurePlannerConfig()

        # Session tracking
        self.status: str = "idle"
        self.error: str | None = None
        self.tool_calls: list[dict] = []

    def _build_model(self) -> BaseChatModel:
        """Build the LLM from settings.

        Uses ChatOpenAI because we go through a LiteLLM proxy.
        """
        api_base = self.llm_settings.get("api_base") or self.llm_settings.get("openai_api_base")
        api_key = self.llm_settings.get("api_key") or self.llm_settings.get("openai_api_key")
        model_name = self.llm_settings.get("model_name", "gpt-4o-mini")

        from langchain_openai import ChatOpenAI

        return ChatOpenAI(
            model=model_name,
            api_key=api_key,
            base_url=api_base,
            temperature=0.0,
            max_tokens=4096,  # Reduced for speed - tool-based output needs less tokens
        )

    @staticmethod
    def _unwrap_chat_model(model: Any) -> BaseChatModel:
        """Unwrap RunnableRetry / RunnableBinding wrappers around a chat model.

        deepagents' ``resolve_model()`` puts the model in a dict key (i.e.
        hashes it), and ``RunnableRetry`` / ``RunnableBinding`` are not
        hashable, so the planner crashes with
        ``TypeError: unhashable type: 'RunnableRetry'``.

        We follow the ``.bound`` chain (used by both ``RunnableBinding`` and
        ``RunnableRetry``) until we reach a real ``BaseChatModel``.  Falls
        back to the original object if no inner chat model is found, so any
        future wrapper types still surface a clear deepagents error rather
        than being silently broken.
        """
        current = model
        for _ in range(8):  # arbitrary safety cap; nesting is normally 1
            if isinstance(current, BaseChatModel):
                return current
            inner = getattr(current, "bound", None)
            if inner is None or inner is current:
                break
            current = inner
        return current

    def _extract_dirs_from_ls_result(self, content: str) -> list[str]:
        """Extract directory names from ls tool result.

        The ls tool returns newline-separated entries, directories end with /
        Example: 'src/\nREADME.md\ntests/\n' -> ['src', 'tests']
        """
        dirs = []
        for line in content.split("\n"):
            line = line.strip()
            if line.endswith("/"):
                # It's a directory
                dir_name = line.rstrip("/")
                if dir_name:
                    dirs.append(dir_name)
        return dirs

    def _track_ls_results(self, event: Any, collector) -> None:
        """Track ls tool results to register discovered directories for coverage."""
        messages_to_check = []

        # Extract messages from various event formats
        if isinstance(event, tuple) and len(event) == 2:
            stream_mode, data = event
            if stream_mode == "messages" and isinstance(data, tuple) and len(data) >= 1:
                messages_to_check.append(data[0])
            elif stream_mode == "updates" and isinstance(data, dict):
                messages_to_check.extend(data.get("messages", []))
        elif isinstance(event, dict):
            messages_to_check.extend(event.get("messages", []))
            for key in event:
                val = event[key]
                if isinstance(val, dict) and "messages" in val:
                    messages_to_check.extend(val.get("messages", []))

        # Look for ToolMessage from ls
        for msg in messages_to_check:
            # Check if it's a tool result message (ToolMessage)
            if hasattr(msg, "type") and msg.type == "tool":
                tool_name = getattr(msg, "name", "")
                content = getattr(msg, "content", "")
                if tool_name == "ls" and content:
                    dirs = self._extract_dirs_from_ls_result(str(content))
                    if dirs:
                        collector.register_discovered_dirs(dirs)
                        logger.debug(f"[STRUCTURE_PLANNER][COVERAGE] Registered dirs from ls: {dirs[:10]}")

    def _validate_structure(self, parsed: dict[str, Any], collector) -> None:
        """Post-planning validation: log warnings about coverage, empty pages, and symbol health.

        Advisory only — never blocks or raises.  The wiki will still generate,
        but operators can review the warnings to improve structure quality.
        """
        warnings = []

        # 1. Coverage check
        try:
            stats = collector.stats
            coverage_pct = stats.get("coverage_pct", 0)
            uncovered = stats.get("uncovered_dirs", [])
            if coverage_pct < 50:
                warnings.append(
                    f"LOW COVERAGE: {coverage_pct:.0f}% of discovered directories covered "
                    f"({stats.get('covered_dirs', 0)}/{stats.get('discovered_dirs', 0)})"
                )
            if uncovered:
                uncovered_sample = uncovered[:10] if isinstance(uncovered, list) else []
                if uncovered_sample:
                    warnings.append(f"UNCOVERED DIRS: {', '.join(uncovered_sample)}")
        except Exception:  # noqa: S110
            pass

        # 2. Pages with no target_symbols
        sections = parsed.get("sections", [])
        total_pages = 0
        empty_symbol_pages = []
        oversized_pages = []

        for section in sections:
            for page in section.get("pages", []):
                total_pages += 1
                page_name = page.get("page_name", "?")
                symbols = page.get("target_symbols", [])
                has_folders = bool(page.get("target_folders"))
                has_docs = bool(page.get("target_docs"))
                if not symbols and not has_folders and not has_docs:
                    empty_symbol_pages.append(page_name)
                elif len(symbols) > 50:
                    oversized_pages.append(f"{page_name} ({len(symbols)} symbols)")

        if empty_symbol_pages:
            warnings.append(
                f"EMPTY SYMBOLS: {len(empty_symbol_pages)} pages have 0 target_symbols: "
                f"{', '.join(empty_symbol_pages[:5])}"
            )

        if oversized_pages:
            warnings.append(
                f"OVERSIZED PAGES: {len(oversized_pages)} pages have >50 target_symbols: "
                f"{', '.join(oversized_pages[:5])}"
            )

        # 3. Symbol existence validation (against graph name_index if available)
        graph = getattr(collector, "code_graph", None)
        if graph:
            try:
                graph_names_lower = set()
                for _node_id, data in graph.nodes(data=True):
                    name = data.get("symbol_name") or data.get("name", "")
                    if name:
                        graph_names_lower.add(name.lower())

                _test_prefixes = ("test_", "mock_", "fake_", "stub_", "fixture_", "conftest")
                _test_infixes = ("_test", "_mock", "_fixture", "_spec")

                missing_symbols = []
                for section in sections:
                    for page in section.get("pages", []):
                        for sym in page.get("target_symbols", []):
                            sym_lower = sym.lower()
                            if sym_lower not in graph_names_lower:
                                if any(sym_lower.startswith(p) for p in _test_prefixes):
                                    continue
                                if any(p in sym_lower for p in _test_infixes):
                                    continue
                                missing_symbols.append(sym)

                if missing_symbols:
                    unique_missing = sorted(set(missing_symbols))
                    warnings.append(
                        f"MISSING SYMBOLS: {len(unique_missing)} target_symbols not found in graph: "
                        f"{', '.join(unique_missing[:10])}"
                    )
            except Exception:  # noqa: S110
                pass

        if warnings:
            logger.warning("[STRUCTURE_PLANNER][VALIDATION] Post-planning issues found:")
            for w in warnings:
                logger.warning(f"  ⚠️ {w}")
        else:
            logger.info(
                f"[STRUCTURE_PLANNER][VALIDATION] Structure looks good: {len(sections)} sections, {total_pages} pages"
            )

    def _collect_repository_files(self) -> list[str]:
        """Walk ``self.repo_root`` and return relative file paths."""
        from ..constants import COVERAGE_IGNORE_DIRS

        files: list[str] = []
        root_len = len(self.repo_root.rstrip("/")) + 1
        for dirpath, dirnames, filenames in os.walk(self.repo_root):
            # Prune ignored directories in-place
            dirnames[:] = [d for d in dirnames if d not in COVERAGE_IGNORE_DIRS and not d.startswith(".")]
            for fname in filenames:
                full = os.path.join(dirpath, fname)
                rel = full[root_len:]
                files.append(rel)
        return files

    def _parse_json(self, raw: str) -> dict[str, Any]:
        """Parse JSON from raw response, handling markdown fences."""
        import json
        import re

        text = raw.strip()

        # Remove markdown code fences
        fence_pattern = r"^```(?:json)?\s*\n?(.*?)\n?```$"
        match = re.match(fence_pattern, text, re.DOTALL | re.IGNORECASE)
        if match:
            text = match.group(1).strip()

        # Try direct parse
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            pass

        # Try to find JSON object in text
        brace_match = re.search(r"\{[\s\S]*\}", text)
        if brace_match:
            try:
                return json.loads(brace_match.group())
            except json.JSONDecodeError:
                pass

        raise ValueError(f"Could not parse JSON from response: {text[:500]}...")
