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

from .context_eviction_middleware import ContextEvictionMiddleware
from .structure_refiner import refine_with_llm
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

    def _create_agent(self, collector):
        """
        Create the DeepAgents agent with FilesystemMiddleware + output tools.

        Uses FilesystemBackend pointed at repo_root with virtual_mode=True for:
        - ls: List directories (sandboxed to repo_root)
        - glob: Find files by pattern (sandboxed to repo_root)
        - grep: Search file contents (sandboxed to repo_root)
        - read_file: Read files with pagination (sandboxed to repo_root)

        Plus custom output tools from collector:
        - set_wiki_metadata: Set wiki title/overview
        - define_section: Define a section
        - define_page: Define a page

        This eliminates JSON generation - output is via tool calls.
        """
        try:
            from deepagents import create_deep_agent
            from deepagents.backends import FilesystemBackend
        except ImportError as e:
            raise RuntimeError(f"DeepAgents not available: {e}") from e

        from .structure_prompts import get_structure_planner_instructions

        # Get base model (either passed or built)
        base_model = self.llm_client or self._build_model()

        # NOTE: Do NOT call base_model.bind() here — RunnableBinding is not a
        # BaseChatModel, so deepagents' resolve_model() cannot hash it.
        # deepagents handles tool calling configuration internally.
        #
        # Same problem applies if the caller passes an LLM wrapped in
        # ``.with_retry()`` (RunnableRetry) or any other RunnableBinding —
        # deepagents' ``resolve_model()`` tries to hash the model and crashes
        # with ``unhashable type: 'RunnableRetry'``.  Unwrap to the inner
        # BaseChatModel here; retry semantics are still applied by the rest
        # of the wiki-generation pipeline that uses ``self.llm_client``
        # directly.
        model = self._unwrap_chat_model(base_model)

        # Build the FilesystemBackend directly (deepagents 0.5.0+ accepts a
        # backend instance; the factory pattern is deprecated). virtual_mode=True
        # sandboxes all file operations to repo_root.
        try:
            fs_backend = FilesystemBackend(root_dir=self.repo_root, virtual_mode=True)
        except TypeError:
            # Fallback for older deepagents without virtual_mode
            fs_backend = FilesystemBackend(root_dir=self.repo_root)

        # Compute repo-aware symbol cap for the prompt
        from ..agents.wiki_graph_optimized import OptimizedWikiGenerationAgent

        _graph_nodes = self.code_graph.number_of_nodes() if self.code_graph else 0
        _file_count = self.config.repository_file_count or 0
        max_symbols_per_page = OptimizedWikiGenerationAgent._compute_max_symbols_per_page(_graph_nodes, _file_count)

        # Generate system prompt with config
        system_prompt = get_structure_planner_instructions(
            page_budget=self.config.page_budget,
            target_audience=self.config.target_audience,
            wiki_type=self.config.wiki_type,
            max_ls_calls=self.config.max_ls_calls,
            max_glob_calls=self.config.max_glob_calls,
            max_read_calls=self.config.max_read_calls,
            max_grep_calls=self.config.max_grep_calls,
            max_symbols_per_page=max_symbols_per_page,
        )

        # Get output tools from collector
        output_tools = collector.get_tools()

        # Build custom middleware stack
        custom_middleware = self._build_middleware()

        # Create agent with FilesystemMiddleware + output tools
        agent = create_deep_agent(
            model=model,
            tools=output_tools,  # Output tools; FilesystemMiddleware adds fs tools
            system_prompt=system_prompt,
            backend=fs_backend,
            middleware=custom_middleware,
        )

        return agent

    def _build_middleware(self):
        """Build custom middleware stack for the structure planner agent.

        Currently adds ContextEvictionMiddleware which clears stale conversation
        history after batch_define_pages with a CONTINUE signal. This prevents
        LLM inference degradation at 80%+ coverage on large repos where the
        accumulated context overwhelms the model.
        """
        return [ContextEvictionMiddleware()]

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

    def plan_structure(self, repo_name: str = "", stream_callback: "callable | None" = None) -> dict[str, Any]:
        """
        Generate a wiki structure plan for the repository.

        Uses tool-based output: the agent calls define_section/define_page tools
        instead of generating JSON. This eliminates the slow JSON generation step.

        Args:
            repo_name: Optional repository name for context
            stream_callback: Optional callback for streaming events

        Returns:
            Dict ready for WikiStructureSpec.model_validate()
        """
        self.status = "running"
        self.error = None
        self.tool_calls = []

        start_time = time.time()
        # Enable logging by default, can be disabled with WIKIS_DEEPAGENTS_LOG_STREAM=0
        log_stream = os.getenv("WIKIS_DEEPAGENTS_LOG_STREAM", "1") == "1"
        # Use invoke instead of stream for faster execution (stream has overhead)
        use_stream = os.getenv("WIKIS_DEEPAGENTS_USE_STREAM", "0") == "1"

        try:
            from .structure_prompts import get_structure_task_prompt
            from .structure_tools import StructureCollector

            # Create collector for tool-based output
            # Pass repo_root so explore_tree and analyze_module tools can access the filesystem
            # Pass code_graph for enhanced symbol-aware complexity analysis
            collector = StructureCollector(
                page_budget=self.config.page_budget,
                repo_root=self.repo_root,
                code_graph=self.code_graph,
                graph_text_index=self.graph_text_index,
                effective_depth=self.config.effective_depth,
                repository_file_count=self.config.repository_file_count,
            )

            # Create agent with collector tools
            agent = self._create_agent(collector)

            # Generate task prompt
            task_prompt = get_structure_task_prompt(
                page_budget=self.config.page_budget, repo_name=repo_name, doc_clusters=self.config.doc_clusters
            )

            logger.info(
                f"[STRUCTURE_PLANNER] Starting with page_budget={self.config.page_budget}, "
                f"repo_root={self.repo_root}, mode={'stream' if use_stream else 'invoke'}, log_stream={log_stream}"
            )

            # Run agent
            result = None
            first_tool_call_time = None
            event_count = 0

            if use_stream and hasattr(agent, "stream"):
                logger.info("[STRUCTURE_PLANNER][STREAM] start")
                for event in agent.stream(
                    {"messages": [HumanMessage(content=task_prompt)]},
                    stream_mode=["messages", "updates"],
                    config={"recursion_limit": self.config.max_iterations * 15},  # 15x for safety margin
                ):
                    event_count += 1

                    # Always try to track tool calls (not just when log_stream is True)
                    self._track_tool_calls_from_event(event)

                    # Track ls results for coverage
                    self._track_ls_results(event, collector)

                    if log_stream:
                        self._log_event(event)

                    # Track first tool call timing
                    if first_tool_call_time is None and self._has_tool_calls(event):
                        first_tool_call_time = time.time() - start_time
                        logger.info(f"[STRUCTURE_PLANNER][TIMING] first_tool_call={first_tool_call_time:.2f}s")

                    # Notify callback
                    if stream_callback:
                        stream_callback(event)

                    # Keep last result
                    if isinstance(event, dict) and "messages" in event:
                        result = event

                logger.info(f"[STRUCTURE_PLANNER][STREAM] end (events={event_count})")
            else:
                # Use invoke directly (faster, no streaming overhead)
                logger.info("[STRUCTURE_PLANNER][INVOKE] start")
                result = agent.invoke(
                    {"messages": [HumanMessage(content=task_prompt)]},
                    config={"recursion_limit": self.config.max_iterations * 15},  # 15x for safety margin
                )

                # Log tool calls from result messages if logging enabled
                if isinstance(result, dict):
                    messages = result.get("messages", [])
                    for msg in messages:
                        # Track ls results for coverage (always, not just when logging)
                        if hasattr(msg, "type") and msg.type == "tool":
                            tool_name = getattr(msg, "name", "")
                            content = getattr(msg, "content", "")
                            if tool_name == "ls" and content:
                                dirs = self._extract_dirs_from_ls_result(str(content))
                                if dirs:
                                    collector.register_discovered_dirs(dirs)
                                    logger.debug(f"[STRUCTURE_PLANNER][COVERAGE] Registered dirs from ls: {dirs[:10]}")

                        # Log tool calls
                        if log_stream and hasattr(msg, "tool_calls") and msg.tool_calls:
                            for tc in msg.tool_calls:
                                tool_name, tool_args = self._extract_tool_info(tc)
                                logger.info(f"[STRUCTURE_PLANNER][TOOL_CALL] {tool_name}: {str(tool_args)[:200]}")
                                self.tool_calls.append(
                                    {"tool": tool_name, "args": tool_args, "timestamp": datetime.now().isoformat()}
                                )

                logger.info("[STRUCTURE_PLANNER][INVOKE] end")

            elapsed = time.time() - start_time
            logger.info(f"[STRUCTURE_PLANNER][TIMING] total={elapsed:.2f}s")

            # Assemble structure from collector (tool calls already captured)
            stats = collector.stats
            logger.info(
                f"[STRUCTURE_PLANNER][STATS] sections={stats['sections']}, "
                f"pages={stats['pages']}/{stats['budget']}, "
                f"coverage={stats['coverage_pct']}% ({stats['covered_dirs']}/{stats['discovered_dirs']} dirs), "
                f"tracked_tool_calls={len(self.tool_calls)}"
            )
            if stats.get("uncovered_dirs"):
                logger.info(f"[STRUCTURE_PLANNER][COVERAGE] Uncovered dirs: {stats['uncovered_dirs'][:10]}")

            # If we haven't captured tool calls from stream events but the collector has data,
            # it means the tools were called but we didn't capture them in the expected event format.
            # Add synthetic entries so the caller knows the agent was active.
            if len(self.tool_calls) == 0 and (stats["sections"] > 0 or stats["pages"] > 0):
                logger.info(
                    "[STRUCTURE_PLANNER] Deriving tool calls from collector (stream events didn't capture them)"
                )
                # Add metadata call if present
                if collector.metadata:
                    self.tool_calls.append(
                        {
                            "tool": "set_wiki_metadata",
                            "args": collector.metadata,
                            "timestamp": datetime.now().isoformat(),
                            "derived": True,
                        }
                    )
                # Add section calls
                for section_name, section_data in collector.sections.items():
                    self.tool_calls.append(
                        {
                            "tool": "define_section",
                            "args": {"section_name": section_name, "section_order": section_data.get("section_order")},
                            "timestamp": datetime.now().isoformat(),
                            "derived": True,
                        }
                    )
                # Add page calls
                for page_data in collector.pages:
                    self.tool_calls.append(
                        {
                            "tool": "define_page",
                            "args": {
                                "page_name": page_data.get("page_name"),
                                "section_name": page_data.get("section_name"),
                            },
                            "timestamp": datetime.now().isoformat(),
                            "derived": True,
                        }
                    )
                logger.info(f"[STRUCTURE_PLANNER] Derived {len(self.tool_calls)} tool calls from collector")

            parsed = collector.assemble()

            # Post-planning validation: coverage, symbol existence, page health
            self._validate_structure(parsed, collector)

            # Validate we got something
            if not parsed.get("sections"):
                # Fallback: try to parse JSON from response (legacy mode)
                logger.warning("[STRUCTURE_PLANNER] No tools output, trying JSON fallback")
                raw_json = self._extract_response(result)
                parsed = self._parse_json(raw_json)

            self.status = "completed"
            return parsed

        except Exception as e:
            self.status = "failed"
            self.error = str(e)
            logger.error(f"[STRUCTURE_PLANNER] Failed: {e}")
            raise

    def _validate_structure(self, parsed: dict[str, Any], collector) -> None:
        """Post-planning validation: log warnings about coverage, empty pages, and symbol health.

        This validation is purely advisory — it logs warnings but does not block
        the pipeline. The wiki will still generate, but operators can review the
        warnings to improve structure quality.
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
            pass  # Coverage stats may not be available in all modes

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
                    # Only flag pages that have NO retrieval hints at all.
                    # Pages covering config/CI/infra files legitimately have
                    # no code symbols but still have target_folders or target_docs.
                    empty_symbol_pages.append(page_name)
                elif len(symbols) > 50:
                    # Broad-page model: 10-30 symbols/page is normal.
                    # Only flag truly oversized pages (>50).
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
        #    Excludes test/fixture symbols — the graph may include test files but
        #    referencing them in wiki pages is harmless at worst.
        graph = getattr(collector, "code_graph", None)
        if graph:
            try:
                # Build a lowercase name index for validation
                graph_names_lower = set()
                for _node_id, data in graph.nodes(data=True):
                    name = data.get("symbol_name") or data.get("name", "")
                    if name:
                        graph_names_lower.add(name.lower())

                # Patterns that indicate test/fixture symbols (not real API concerns)
                _test_prefixes = ("test_", "mock_", "fake_", "stub_", "fixture_", "conftest")
                _test_infixes = ("_test", "_mock", "_fixture", "_spec")

                missing_symbols = []
                for section in sections:
                    for page in section.get("pages", []):
                        for sym in page.get("target_symbols", []):
                            sym_lower = sym.lower()
                            if sym_lower not in graph_names_lower:
                                # Skip likely test/fixture symbols
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
                pass  # Graph traversal may fail for non-standard graph objects

        # Log all warnings
        if warnings:
            logger.warning("[STRUCTURE_PLANNER][VALIDATION] Post-planning issues found:")
            for w in warnings:
                logger.warning(f"  ⚠️ {w}")
        else:
            logger.info(
                f"[STRUCTURE_PLANNER][VALIDATION] Structure looks good: {len(sections)} sections, {total_pages} pages"
            )

    def _has_tool_calls(self, event: Any) -> bool:
        """Check if event contains tool calls."""
        if not isinstance(event, dict):
            return False
        for msg in event.get("messages", []) or []:
            if hasattr(msg, "tool_calls") and msg.tool_calls:
                return True
        return False

    def _track_tool_calls_from_event(self, event: Any) -> None:
        """
        Track tool calls from any stream event format.

        This method is more robust than _log_event - it handles various
        event formats from LangGraph streaming.
        """
        messages_to_check = []
        event_type = type(event).__name__

        # Handle tuple format: (stream_mode, data)
        if isinstance(event, tuple) and len(event) == 2:
            stream_mode, data = event
            logger.debug(f"[STRUCTURE_PLANNER][EVENT] tuple: mode={stream_mode}, data_type={type(data).__name__}")
            if stream_mode == "messages" and isinstance(data, tuple) and len(data) >= 1:
                messages_to_check.append(data[0])
            elif stream_mode == "updates" and isinstance(data, dict):
                messages_to_check.extend(data.get("messages", []))
        # Handle dict format: {"messages": [...], ...}
        elif isinstance(event, dict):
            keys = list(event.keys())[:5]  # First 5 keys for debugging
            logger.debug(f"[STRUCTURE_PLANNER][EVENT] dict keys: {keys}")
            messages_to_check.extend(event.get("messages", []))
            # Also check for nested structures in LangGraph
            for key in event:
                val = event[key]
                if isinstance(val, dict) and "messages" in val:
                    messages_to_check.extend(val.get("messages", []))
        else:
            logger.debug(f"[STRUCTURE_PLANNER][EVENT] unknown type: {event_type}")

        # Extract tool calls from messages
        for msg in messages_to_check:
            if hasattr(msg, "tool_calls") and msg.tool_calls:
                for tc in msg.tool_calls:
                    tool_name, tool_args = self._extract_tool_info(tc)
                    if not tool_name:
                        continue
                    # Log the tool call as it's tracked
                    logger.info(f"[STRUCTURE_PLANNER][TOOL_CALL_TRACKED] {tool_name}")
                    # Avoid duplicates by checking if we already have this call
                    # (in streaming, the same call might appear in multiple events)
                    already_tracked = any(
                        t.get("tool") == tool_name and t.get("args") == tool_args
                        for t in self.tool_calls[-5:]
                        if isinstance(t, dict)  # Check last 5
                    )
                    if not already_tracked:
                        self.tool_calls.append(
                            {"tool": tool_name, "args": tool_args, "timestamp": datetime.now().isoformat()}
                        )

    def _extract_tool_info(self, tc: Any) -> tuple:
        """Extract tool name and args from a tool call (handles both dict and object)."""
        if isinstance(tc, dict):
            name = tc.get("name") or ""
            args = tc.get("args") or {}
        else:
            # ToolCall object with .name and .args attributes
            name = getattr(tc, "name", None) or ""
            args = getattr(tc, "args", None) or {}
        return name, args

    def _log_event(self, event: Any) -> None:
        """Log a stream event for debugging - logs tool calls to console.

        Note: In streaming mode, tool calls arrive in chunks. We only log
        complete tool calls (those with both name and args).
        """
        # Debug: log event type to understand what we're receiving
        event_type = type(event).__name__

        if isinstance(event, tuple) and len(event) == 2:
            stream_mode, data = event
            if stream_mode == "messages" and isinstance(data, tuple) and len(data) >= 1:
                msg = data[0]
                # Log tool calls (only complete ones with name)
                if hasattr(msg, "tool_calls") and msg.tool_calls:
                    for tc in msg.tool_calls:
                        tool_name, tool_args = self._extract_tool_info(tc)
                        # Skip partial tool calls (name is required, args can be empty for some tools)
                        if not tool_name:
                            continue
                        # Log to console
                        args_str = str(tool_args)[:200] if tool_args else "(no args)"
                        logger.info(f"[STRUCTURE_PLANNER][TOOL_CALL] {tool_name}: {args_str}")
                        self.tool_calls.append(
                            {"tool": tool_name, "args": tool_args, "timestamp": datetime.now().isoformat()}
                        )
                # Log tool results (ToolMessage)
                if hasattr(msg, "type") and msg.type == "tool":
                    tool_name = getattr(msg, "name", "unknown")
                    content = getattr(msg, "content", "")
                    if tool_name and content:
                        content_preview = str(content)[:150]
                        logger.info(f"[STRUCTURE_PLANNER][TOOL_RESULT] {tool_name}: {content_preview}...")
            elif stream_mode == "updates":
                # Handle updates stream mode - often contains tool call info
                if isinstance(data, dict):
                    for key, value in data.items():
                        if key == "messages" and isinstance(value, list):
                            for msg in value:
                                if hasattr(msg, "tool_calls") and msg.tool_calls:
                                    for tc in msg.tool_calls:
                                        tool_name, tool_args = self._extract_tool_info(tc)
                                        if not tool_name:
                                            continue
                                        args_str = str(tool_args)[:200] if tool_args else "(no args)"
                                        logger.info(f"[STRUCTURE_PLANNER][TOOL_CALL] {tool_name}: {args_str}")
                                        self.tool_calls.append(
                                            {
                                                "tool": tool_name,
                                                "args": tool_args,
                                                "timestamp": datetime.now().isoformat(),
                                            }
                                        )
        elif isinstance(event, dict):
            # Handle non-tuple events (direct dict format from some LangGraph versions)
            for key, value in event.items():
                if key == "messages" and isinstance(value, list):
                    for msg in value:
                        if hasattr(msg, "tool_calls") and msg.tool_calls:
                            for tc in msg.tool_calls:
                                tool_name, tool_args = self._extract_tool_info(tc)
                                # Skip partial tool calls
                                if not tool_name:
                                    continue
                                args_str = str(tool_args)[:200] if tool_args else "(no args)"
                                logger.info(f"[STRUCTURE_PLANNER][TOOL_CALL] {tool_name}: {args_str}")
                                self.tool_calls.append(
                                    {"tool": tool_name, "args": tool_args, "timestamp": datetime.now().isoformat()}
                                )
                        # Also check for tool messages
                        if hasattr(msg, "type") and msg.type == "tool":
                            tool_name = getattr(msg, "name", "unknown")
                            content = getattr(msg, "content", "")
                            if tool_name:
                                content_preview = str(content)[:150] if content else "(empty)"
                                logger.info(f"[STRUCTURE_PLANNER][TOOL_RESULT] {tool_name}: {content_preview}...")
        else:
            # Unknown event format - log for debugging
            logger.debug(f"[STRUCTURE_PLANNER][EVENT] Unhandled event type: {event_type}")

    def _extract_response(self, result: Any) -> str:
        """Extract the final text response from agent result."""
        messages = result.get("messages", []) if isinstance(result, dict) else []
        if not messages:
            raise ValueError("No messages in agent response")

        last_msg = messages[-1]
        content = getattr(last_msg, "content", last_msg) if last_msg else ""

        if isinstance(content, str):
            return content
        elif isinstance(content, list):
            parts = []
            for item in content:
                if item is None:
                    continue
                if isinstance(item, str):
                    parts.append(item)
                elif isinstance(item, dict):
                    parts.append(str(item.get("text") or item.get("content") or ""))
                else:
                    parts.append(str(item))
            return "\n".join(p for p in parts if p).strip()
        else:
            return str(content)

    # -----------------------------------------------------------------
    # Graph-First Structure Planning (Phase B — feature-flagged)
    # -----------------------------------------------------------------

    def plan_structure_graph_first(
        self,
        repo_name: str = "",
        repository_files: list[str] | None = None,
        stream_callback: "callable | None" = None,
    ) -> dict[str, Any]:
        """Generate a wiki structure using the graph-first deterministic approach.

        Phase 1 builds a deterministic skeleton (zero LLM calls):
          - Extracts architectural symbols from the code graph
          - Builds a weighted directory interaction graph
          - Clusters directories via Louvain community detection
          - Normalises cluster sizes (split/merge)
          - Detects doc-only directories

        Phase 2 refines with a single (or batched) LLM call:
          - LLM sees only fixed-size cluster *summaries*
          - LLM names pages, groups into sections, writes descriptions
          - Symbols & folders injected programmatically → 100 % coverage

        Phase 3 validates the output (advisory logging, never blocks).

        Args:
            repo_name: Repository name for prompts / metadata.
            repository_files: All file paths in the repo (relative).
                If ``None``, derived from ``self.repo_root``.
            stream_callback: Optional progress callback.

        Returns:
            Dict ready for ``WikiStructureSpec.model_validate()``.
        """
        self.status = "running"
        self.error = None
        self.tool_calls = []
        start_time = time.time()

        try:
            # ── Resolve repository_files if not provided ──────────────
            if repository_files is None:
                repository_files = self._collect_repository_files()

            # ── Phase 1: deterministic skeleton ───────────────────────
            logger.info(
                "[GRAPH_FIRST] Phase 1 – building skeleton (budget=%d, depth=%d, files=%d)",
                self.config.page_budget,
                self.config.effective_depth,
                len(repository_files),
            )
            skeleton = build_skeleton(
                code_graph=self.code_graph,
                repository_files=repository_files,
                page_budget=self.config.page_budget,
                effective_depth=self.config.effective_depth,
                repo_name=repo_name,
            )
            phase1_time = time.time() - start_time
            logger.info(
                "[GRAPH_FIRST] Phase 1 done in %.2fs: %d code clusters, %d doc clusters, %d/%d dirs",
                phase1_time,
                len(skeleton.code_clusters),
                len(skeleton.doc_clusters),
                skeleton.total_dirs_covered,
                skeleton.total_dirs_in_repo,
            )
            if stream_callback:
                stream_callback(
                    {
                        "phase": "skeleton",
                        "clusters": len(skeleton.code_clusters),
                        "elapsed": phase1_time,
                    }
                )

            # ── Phase 2: LLM refinement ───────────────────────────────
            llm = self.llm_client or self._build_model()
            logger.info("[GRAPH_FIRST] Phase 2 – LLM refinement")
            data = refine_with_llm(
                skeleton=skeleton,
                llm=llm,
                repo_name=repo_name,
                page_budget=self.config.page_budget,
            )
            phase2_time = time.time() - start_time - phase1_time
            logger.info(
                "[GRAPH_FIRST] Phase 2 done in %.2fs: %d sections, %d pages",
                phase2_time,
                len(data.get("sections", [])),
                data.get("total_pages", 0),
            )
            if stream_callback:
                stream_callback(
                    {
                        "phase": "refinement",
                        "sections": len(data.get("sections", [])),
                        "elapsed": phase2_time,
                    }
                )

            # ── Phase 3: validation (advisory) ────────────────────────
            self._validate_graph_first(data, skeleton)

            self.status = "completed"
            total_time = time.time() - start_time
            logger.info(
                "[GRAPH_FIRST] Complete in %.2fs (skeleton=%.2fs, refine=%.2fs)",
                total_time,
                phase1_time,
                phase2_time,
            )
            return data

        except Exception as e:
            self.status = "failed"
            self.error = str(e)
            logger.error("[GRAPH_FIRST] Failed: %s", e, exc_info=True)
            raise

    # ── Graph-first helpers ───────────────────────────────────────────

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

    def _validate_graph_first(
        self,
        data: dict[str, Any],
        skeleton: StructureSkeleton,
    ) -> None:
        """Advisory validation for graph-first output.

        Logs warnings about coverage, empty pages, budget compliance.
        Never raises — the output is usable regardless.
        """
        warnings: list[str] = []
        sections = data.get("sections", [])
        total_pages = sum(len(s.get("pages", [])) for s in sections)

        # 1. Coverage: every cluster should map to a page
        page_symbol_count = 0
        page_dir_set: set = set()
        for section in sections:
            for page in section.get("pages", []):
                page_symbol_count += len(page.get("target_symbols", []))
                for tf in page.get("target_folders", []):
                    page_dir_set.add(tf)

        if skeleton.total_arch_symbols > 0:
            sym_pct = page_symbol_count * 100 / skeleton.total_arch_symbols
            if sym_pct < 80:
                warnings.append(
                    f"LOW SYMBOL COVERAGE: {sym_pct:.0f}% ({page_symbol_count}/{skeleton.total_arch_symbols})"
                )

        if skeleton.total_dirs_covered > 0:
            dirs_in_pages = len(page_dir_set)
            dir_pct = dirs_in_pages * 100 / max(skeleton.total_dirs_covered, 1)
            if dir_pct < 90:
                warnings.append(f"LOW DIR COVERAGE: {dir_pct:.0f}% ({dirs_in_pages}/{skeleton.total_dirs_covered})")

        # 2. Budget compliance
        if total_pages > self.config.page_budget * 2:
            warnings.append(f"OVER BUDGET: {total_pages} pages vs budget {self.config.page_budget}")

        # 3. Empty pages
        empty = [
            p.get("page_name", "?")
            for s in sections
            for p in s.get("pages", [])
            if not p.get("target_symbols") and not p.get("target_folders") and not p.get("target_docs")
        ]
        if empty:
            warnings.append(f"EMPTY PAGES: {len(empty)} pages have no retrieval data: {', '.join(empty[:5])}")

        if warnings:
            logger.warning("[GRAPH_FIRST][VALIDATION] Issues found:")
            for w in warnings:
                logger.warning("  ⚠️ %s", w)
        else:
            logger.info(
                "[GRAPH_FIRST][VALIDATION] OK: %d sections, %d pages, %d symbols",
                len(sections),
                total_pages,
                page_symbol_count,
            )

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
