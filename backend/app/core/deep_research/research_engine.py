"""
Deep Research Engine - Using langchain.agents.create_agent

This is the main orchestration for deep research using LangChain's create_agent
with explicit middleware stack:
- TodoListMiddleware for planning and progress tracking (write_todos, read_todos)
- FilesystemMiddleware for context offloading (outputs >20K tokens → files)
- SummarizationMiddleware for auto-summarising at context limit
- AnthropicPromptCachingMiddleware for prompt caching on Anthropic models
- Custom tools wrapping WikiRetrieverStack and GraphManager

Events are captured via LangGraph's native astream with stream_mode=["messages", "updates"].
"""

import logging
from collections.abc import AsyncGenerator, Callable
from dataclasses import dataclass
from datetime import datetime
from typing import Any

from deepagents import FilesystemMiddleware
from langchain.agents import create_agent
from langchain.agents.middleware.summarization import SummarizationMiddleware
from langchain.agents.middleware.todo import TodoListMiddleware
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, ToolMessage

import app.events as _events
from app.core.chat_utils import format_chat_history

from .research_prompts import RESEARCH_INSTRUCTIONS, get_research_prompt
from .research_tools import create_codebase_tools

logger = logging.getLogger(__name__)

# Safety limits
MAX_ITERATIONS = 15


@dataclass
class ResearchConfig:
    """Configuration for a research session"""

    max_iterations: int = MAX_ITERATIONS
    max_search_results: int = 15
    enable_graph_analysis: bool = True
    research_type: str = "general"
    enable_subagents: bool = False  # Disabled - subagents add complexity without clear benefit
    similarity_threshold: float = 0.0  # EmbeddingsFilter threshold (0.0 = disabled)


class DeepResearchEngine:
    """
    Main engine for conducting deep research using the actual DeepAgents library.

    This creates a proper DeepAgents agent with:
    - TodoListMiddleware for planning (write_todos, read_todos)
    - FilesystemMiddleware for context offloading (read_file, write_file, etc.)
    - Custom tools for codebase search and graph analysis (search_codebase, get_symbol_relationships, think)

    Events are captured using LangGraph's native astream with dual stream mode
    (messages + updates) - NOT LangChain callbacks.

    Usage:
        engine = DeepResearchEngine(
            retriever_stack=wiki_retriever_stack,
            graph_manager=graph_manager,
            code_graph=loaded_graph,
            llm_settings=llm_config
        )

        async for event in engine.research("How does auth work?"):
            handle_event(event)
    """

    def __init__(
        self,
        retriever_stack: Any,  # WikiRetrieverStack
        graph_manager: Any,  # GraphManager
        code_graph: Any,  # NetworkX graph
        repo_analysis: dict | None = None,
        llm_client: BaseChatModel | None = None,
        backend: object | None = None,
        llm_settings: dict | None = None,
        config: ResearchConfig | None = None,
        repo_path: str | None = None,
    ):
        """
        Initialize the research engine.

        Args:
            retriever_stack: WikiRetrieverStack for codebase search
            graph_manager: GraphManager for relationship analysis
            code_graph: Loaded NetworkX code graph
            repo_analysis: Pre-loaded repository analysis
            llm_settings: LLM configuration dict
            config: Research configuration
            repo_path: Path to cloned repo for direct file access
        """
        self.retriever_stack = retriever_stack
        self.graph_manager = graph_manager
        self.code_graph = code_graph
        self.repo_analysis = repo_analysis or {}
        self.llm_client = llm_client
        self.backend = backend
        self.llm_settings = llm_settings or {}
        self.config = config or ResearchConfig()
        self.repo_path = repo_path

        # Session tracking (simplified - no redundant state)
        self.session_id: str | None = None
        self.question: str | None = None
        self.final_report: str = ""
        self.current_todos: list = []
        self.status: str = "idle"  # idle, running, completed, failed
        self.error: str | None = None

    def _build_model(self) -> BaseChatModel:
        """Build the LLM from settings.

        Always uses ChatOpenAI because we're going through a LiteLLM proxy
        that handles auth with JWT tokens and routes to the appropriate backend
        (OpenAI, Anthropic, etc.) based on the model name.
        """
        api_base = self.llm_settings.get("api_base") or self.llm_settings.get("openai_api_base")
        api_key = self.llm_settings.get("api_key") or self.llm_settings.get("openai_api_key")
        model_name = self.llm_settings.get("model_name", "gpt-4o-mini")
        max_tokens = self.llm_settings.get("max_tokens", 16384)  # Default to 16k for comprehensive reports

        # Always use ChatOpenAI for proxy-based setups
        # The proxy handles routing to Anthropic/OpenAI/etc based on model name
        from langchain_openai import ChatOpenAI

        return ChatOpenAI(
            model=model_name,
            api_key=api_key,
            base_url=api_base,
            temperature=0.0,
            max_tokens=max_tokens,
        )

    def _create_agent(self):
        """
        Create the agent with full middleware stack via langchain.agents.create_agent.

        Middleware:
        - TodoListMiddleware  — write_todos / read_todos for planning
        - FilesystemMiddleware — ls, read_file, write_file, edit_file, glob, grep
        - SummarizationMiddleware — auto-summarise when context hits limit
        - AnthropicPromptCachingMiddleware — prompt caching on Anthropic models
        """
        model = self.llm_client or self._build_model()

        fts_index = getattr(self.graph_manager, "fts_index", None) if self.graph_manager else None
        custom_tools = create_codebase_tools(
            retriever_stack=self.retriever_stack,
            graph_manager=self.graph_manager,
            code_graph=self.code_graph,
            repo_analysis=self.repo_analysis,
            event_callback=None,
            graph_text_index=fts_index,
            similarity_threshold=self.config.similarity_threshold,
            repo_path=self.repo_path,
        )

        _profile = getattr(model, "profile", None)
        if isinstance(_profile, dict) and isinstance(_profile.get("max_input_tokens"), int):
            trigger = ("fraction", 0.85)
            keep = ("fraction", 0.10)
        else:
            trigger = ("tokens", 170_000)
            keep = ("messages", 6)

        try:
            from langchain_anthropic.middleware import AnthropicPromptCachingMiddleware

            caching_mw = AnthropicPromptCachingMiddleware(unsupported_model_behavior="ignore")
        except ImportError:
            caching_mw = None

        middleware = [
            TodoListMiddleware(),
            FilesystemMiddleware(backend=self.backend),
            SummarizationMiddleware(
                model=model,
                trigger=trigger,
                keep=keep,
                trim_tokens_to_summarize=None,
            ),
        ]
        if caching_mw is not None:
            middleware.append(caching_mw)

        agent = create_agent(
            model=model,
            tools=custom_tools,
            system_prompt=RESEARCH_INSTRUCTIONS,
            middleware=middleware,
        )

        return agent

    async def research(
        self,
        question: str,
        session_id: str | None = None,
        chat_history: list[dict[str, str]] | None = None,
    ) -> AsyncGenerator[dict, None]:
        """
        Conduct deep research on a question.

        Uses LangGraph's astream with stream_mode=["messages", "updates"] to capture:
        - Tool calls and results (messages stream)
        - Todo updates from TodoListMiddleware (updates stream)
        - State changes (updates stream)

        Args:
            question: The research question
            session_id: Optional session identifier
            chat_history: Optional prior conversation as list of
                ``{"role": "user"|"assistant", "content": "..."}`` dicts.

        Yields:
            Event dictionaries for UI updates
        """
        self.session_id = session_id or datetime.now().strftime("%Y%m%d_%H%M%S")
        self.question = question
        self.status = "running"
        self.final_report = ""
        self.current_todos = []
        self.error = None

        step_count = 0

        # Yield start event
        _evt = _events.task_status(self.session_id, "working", question)
        yield {"event_type": _evt.event, "data": _evt.params}

        try:
            # Create the agent
            agent = self._create_agent()

            # Build research prompt with context (and optional conversation history)
            repo_context = self._get_repo_context()
            history_str = format_chat_history(chat_history) if chat_history else ""
            research_prompt = get_research_prompt(
                research_type=self.config.research_type,
                topic=question,
                context=repo_context,
                chat_history=history_str,
            )

            # Prepare input message
            stream_input = {"messages": [{"role": "user", "content": research_prompt}]}

            # Stream with dual mode: messages for tool calls, updates for todos/state
            async for chunk in agent.astream(
                stream_input,
                stream_mode=["messages", "updates"],
                # config={"recursion_limit": self.config.max_iterations * 10}
            ):
                # Handle different chunk formats
                if not isinstance(chunk, tuple) or len(chunk) != 2:
                    continue

                stream_mode, data = chunk

                # Handle UPDATES stream - todo changes and state updates
                if stream_mode == "updates":
                    if not isinstance(data, dict):
                        continue

                    # Extract node updates
                    for _node_name, node_data in data.items():
                        if not isinstance(node_data, dict):
                            continue

                        # Check for todo updates from TodoListMiddleware
                        if "todos" in node_data:
                            new_todos = node_data["todos"]
                            if new_todos != self.current_todos:
                                self.current_todos = new_todos
                                yield {
                                    "event_type": "todo_update",
                                    "data": {"todos": new_todos, "timestamp": datetime.now().isoformat()},
                                }

                # Handle MESSAGES stream - tool calls and responses
                elif stream_mode == "messages":
                    if not isinstance(data, tuple) or len(data) != 2:
                        continue

                    message, metadata = data

                    # AI message with tool calls
                    if isinstance(message, AIMessage):
                        # Check for tool calls
                        if hasattr(message, "tool_calls") and message.tool_calls:
                            for tool_call in message.tool_calls:
                                step_count += 1
                                tool_call_id = tool_call.get("id", f"call_{step_count}")
                                tool_name = tool_call.get("name", "unknown")
                                _evt = _events.thinking_step(
                                    self.session_id,
                                    step_count,
                                    "tool_call",
                                    tool_name,
                                    tool_call_id=tool_call_id,
                                    call_id=tool_call_id,
                                    input=str(tool_call.get("args", {}))[:500],
                                )
                                yield {"event_type": _evt.event, "data": _evt.params}

                        # Final content (no tool calls = answer tokens streaming)
                        if message.content and not message.tool_calls:
                            self.final_report += message.content
                            _evt = _events.answer_chunk(self.session_id, message.content)
                            yield {"event_type": _evt.event, "data": _evt.params}

                    # Tool response message
                    elif isinstance(message, ToolMessage):
                        step_count += 1
                        content_str = str(message.content) if message.content else ""
                        tool_name = getattr(message, "name", None) or metadata.get("tool", "tool")
                        _evt = _events.thinking_step(
                            self.session_id,
                            step_count,
                            "tool_result",
                            tool_name,
                            tool_call_id=message.tool_call_id,
                            call_id=message.tool_call_id,
                            output=content_str[:500],
                            output_preview=content_str[:300] + "..." if len(content_str) > 300 else content_str,
                            output_length=len(content_str),
                        )
                        yield {"event_type": _evt.event, "data": _evt.params}

            # Research completed
            self.status = "completed"

            _evt = _events.task_status(
                self.session_id,
                "completed",
                "Research complete",
                report=self.final_report,
            )
            yield {"event_type": _evt.event, "data": _evt.params}

        except Exception as e:
            logger.error(f"Research failed: {e}", exc_info=True)
            self.status = "failed"
            self.error = str(e)

            _evt = _events.task_status(self.session_id, "failed", str(e), error=str(e))
            yield {"event_type": _evt.event, "data": _evt.params}

    def _get_repo_context(self) -> str:
        """Get repository context string"""
        if not self.repo_analysis:
            return "No repository overview available."

        parts = []

        if self.repo_analysis.get("description"):
            parts.append(f"Project Description: {self.repo_analysis['description']}")

        if self.repo_analysis.get("summary"):
            parts.append(f"Repository Summary: {self.repo_analysis['summary']}")

        if self.repo_analysis.get("key_components"):
            parts.append(f"Key Components: {self.repo_analysis['key_components']}")

        if self.repo_analysis.get("languages"):
            parts.append(f"Languages: {self.repo_analysis['languages']}")

        return "\n".join(parts) if parts else "No repository overview available."

    def research_sync(
        self,
        question: str,
        on_event: Callable[[dict], None] | None = None,
        chat_history: list[dict[str, str]] | None = None,
    ) -> str:
        """
        Synchronous wrapper for deep research.

        Args:
            question: Research question
            on_event: Optional callback for events
            chat_history: Optional prior conversation history

        Returns:
            Final research report
        """
        import asyncio

        async def _run():
            report = ""
            async for event in self.research(question, chat_history=chat_history):
                if on_event:
                    on_event(event)
                if event.get("event_type") == "research_complete":
                    report = event["data"].get("report", "")
            return report

        return asyncio.run(_run())

    def get_report(self) -> str:
        """Get the final research report"""
        return self.final_report or "Research not completed"

    def get_status(self) -> dict:
        """Get current research status"""
        return {
            "session_id": self.session_id,
            "question": self.question,
            "status": self.status,
            "todos": self.current_todos,
            "error": self.error,
        }


def create_deep_research_engine(
    retriever_stack: Any,
    graph_manager: Any,
    code_graph: Any,
    repo_analysis: dict | None = None,
    llm_client: BaseChatModel | None = None,
    backend: object | None = None,
    llm_settings: dict | None = None,
    repo_path: str | None = None,
    **kwargs,
) -> DeepResearchEngine:
    """
    Factory function to create a DeepResearchEngine.

    This is the main entry point for creating research engines.

    Args:
        retriever_stack: WikiRetrieverStack for codebase search
        graph_manager: GraphManager for relationship analysis
        code_graph: Loaded NetworkX code graph
        repo_analysis: Pre-loaded repository analysis
        llm_settings: LLM configuration
        **kwargs: Additional config options

    Returns:
        Configured DeepResearchEngine instance
    """
    config = ResearchConfig(**{k: v for k, v in kwargs.items() if hasattr(ResearchConfig, k)})

    return DeepResearchEngine(
        retriever_stack=retriever_stack,
        graph_manager=graph_manager,
        code_graph=code_graph,
        repo_analysis=repo_analysis,
        llm_client=llm_client,
        backend=backend,
        llm_settings=llm_settings,
        config=config,
        repo_path=repo_path,
    )
