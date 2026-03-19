#!/usr/bin/python3

"""
Ask Tool - Q&A about repository functionality using RAG.

This module provides a RAG-based Q&A tool that uses:
- WikiRetrieverStack for document retrieval (Dense + BM25 + Reranking)
- ContentExpander for graph-based context expansion
- LLM for answer generation with citations
- Token-based context budgeting (same as wiki generation)

The tool requires a wiki to be generated first (which builds the vector store).
"""

import logging
import os
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from langchain_core.documents import Document
from langchain_core.messages import HumanMessage, SystemMessage

from .repository_analysis_store import (
    get_query_optimized_context,
    is_structured_analysis,
    render_structured_analysis_as_markdown,
)
from .token_counter import get_token_counter

logger = logging.getLogger(__name__)

# Token budgets for Ask tool
ASK_CONTEXT_TOKEN_BUDGET = 50_000  # Same as wiki generation CONTEXT_TOKEN_BUDGET
ASK_QUERY_OPTIMIZATION_OUTPUT_TOKENS = 1024  # Max tokens for query optimization response
ASK_ANSWER_OUTPUT_TOKENS = 16_000  # Max tokens for answer generation response

# Toggle to use optimized context extraction for structured analysis
# When ON: Uses query-matched capabilities instead of full/truncated analysis
ASK_USE_OPTIMIZED_CONTEXT = os.getenv("WIKIS_ASK_USE_OPTIMIZED_CONTEXT", "1") == "1"


@dataclass
class AskSource:
    """A source reference in the answer"""

    index: int
    source: str
    symbol: str | None = None
    chunk_type: str = "code"
    relevance_score: float | None = None


@dataclass
class AskResponse:
    """Response from the Ask tool"""

    answer: str
    sources: list[AskSource] = field(default_factory=list)
    thinking_steps: list[dict[str, Any]] = field(default_factory=list)
    query_used: str = ""
    documents_retrieved: int = 0


# System prompt for query optimization (uses repository context like wiki structure generation)
QUERY_OPTIMIZATION_SYSTEM_PROMPT = """You are a search query optimizer for code repositories.

Your task is to transform user questions into optimized retrieval queries that will find the most relevant code and documentation in a vector store.

**QUERY OPTIMIZATION STRATEGY:**
- Identify key technical terms, component names, and concepts from the repository context
- Include specific file names, folder paths, and symbol names when relevant
- Combine topic keywords with implementation-specific terms
- Add related functionality and patterns that may appear in code
- Use both high-level concepts and low-level implementation details

**OUTPUT FORMAT:**
Return ONLY the optimized search query - a space-separated list of relevant terms.
Do NOT include explanations, quotes, or any other text."""

QUERY_OPTIMIZATION_USER_PROMPT = """## Repository Context
{repository_context}

## User Question
"{question}"

**TASK:** Generate an optimized retrieval query that will find the most relevant code and documentation.

**QUERY OPTIMIZATION EXAMPLE:**
For a question about "How does authentication work?":
Generated Query: "authentication login logout session token JWT user validation security middleware auth_manager password hashing access control authorization user_service security_config"

This query combines topic keywords with file-specific terms and related functionality.

**Return ONLY the optimized query (space-separated terms), nothing else:**"""


# System prompt for answer generation with natural code citations
ANSWER_SYSTEM_PROMPT = """You are a helpful assistant answering questions about a code repository.

You have access to the repository's documentation, code, and architecture analysis.
Use the provided context to answer questions accurately and comprehensively.

## CITATION STYLE (NATURAL FILE REFERENCES)

**Reference code naturally by file path and symbol name - do NOT use [N] numeric citations.**

### In Prose:
- "The `authenticate()` function in `auth/auth_manager.py` handles token validation"
- "Session management is implemented via `SessionStore` class in `services/session.py`"
- "The configuration is loaded from `config/settings.py`"

### In Code Blocks:
Always include the file path as a comment header:
```python
# auth/auth_manager.py
def authenticate(token: str) -> bool:
    ...
```

### Multiple File References:
- "The authentication flow spans `auth/login.py` → `auth/auth_manager.py` → `services/session.py`"
- "Both `UserModel` in `models/user.py` and `UserService` in `services/user.py` are involved"

## RESPONSE GUIDELINES

1. **Be Accurate**: Only state facts supported by the provided context
2. **Be Specific**: Reference actual file paths, function names, and class names naturally in text
3. **Be Structured**: Use headers, bullet points, and code blocks for clarity
4. **Acknowledge Gaps**: If context is insufficient, say so clearly
5. **Show Code**: Include relevant code snippets with file path headers

## FORMAT

Use clear markdown:
- Headers (##) for major sections
- Code blocks with language tags and file path comments
- Bullet points for lists
- `backticks` for inline code/file references
- Natural file path mentions (not [N] citations)

## SOURCES SUMMARY

At the end of your response, include a brief summary:
---
**📁 Files Referenced:**
- `path/to/file.py` — Brief description of what was used from this file
"""


ANSWER_USER_PROMPT = """## Retrieved Code Context

{context}

## Available Sources
{sources_reference}

## User Question

{question}

---

**INSTRUCTIONS:**
1. Answer the question comprehensively based on the code context above
2. Reference files naturally by path (e.g., "in `path/to/file.py`") - do NOT use [N] citations
3. Include relevant code snippets with file path headers as comments
4. Reference specific symbol names (classes, functions, methods)
5. End with a "📁 Files Referenced" summary
6. If the context doesn't contain enough information, state this clearly

**Provide your answer:**"""


class AskTool:
    """
    RAG-based Q&A tool for repository questions.

    Uses the existing wiki infrastructure:
    - VectorStoreManager for embeddings search
    - WikiRetrieverStack for ensemble retrieval + reranking
    - ContentExpander for graph-based context expansion
    - Token-based context budgeting (50K tokens, same as wiki generation)
    """

    def __init__(
        self,
        retriever_stack,  # WikiRetrieverStack instance
        llm_client,  # LangChain LLM (ChatOpenAI, etc.)
        repository_analysis: str | None = None,
        thinking_callback: Callable[[dict[str, Any]], None] | None = None,
        max_context_tokens: int = ASK_CONTEXT_TOKEN_BUDGET,
        optimize_query: bool = True,
        query_output_tokens: int = ASK_QUERY_OPTIMIZATION_OUTPUT_TOKENS,
        answer_output_tokens: int = ASK_ANSWER_OUTPUT_TOKENS,
    ):
        """
        Initialize the Ask tool.

        Args:
            retriever_stack: WikiRetrieverStack instance for document retrieval
            llm_client: LangChain chat model for answer generation
            repository_analysis: Full repository analysis text for context (from wiki generation)
            thinking_callback: Optional callback for emitting thinking steps
            max_context_tokens: Maximum tokens for context (default 50K, same as wiki)
            optimize_query: Whether to optimize queries using repo analysis
            query_output_tokens: Max output tokens for query optimization (default 1024)
            answer_output_tokens: Max output tokens for answer generation (default 16000)
        """
        self.retriever = retriever_stack
        self.llm = llm_client
        self.repository_analysis = repository_analysis
        self.thinking_callback = thinking_callback or (lambda x: None)
        self.max_context_tokens = max_context_tokens
        self.optimize_query = optimize_query and repository_analysis
        self.query_output_tokens = query_output_tokens
        self.answer_output_tokens = answer_output_tokens

        # Initialize token counter
        self.token_counter = get_token_counter()

        self._thinking_steps: list[dict[str, Any]] = []

    def _emit_thinking(self, step_type: str, message: str, **extra) -> None:
        """Emit a thinking step for UI display"""
        step = {"type": step_type, "message": message, **extra}
        self._thinking_steps.append(step)
        self.thinking_callback(step)
        logger.info(f"[{step_type}] {message}")

    def _get_context_for_query_optimization(self, question: str) -> str:
        """
        Get optimized context for query optimization based on analysis format.

        For structured JSON analysis: Returns query-matched capabilities (~2K tokens)
        For legacy markdown: Returns truncated analysis (~3K tokens)
        """
        if not self.repository_analysis:
            return ""

        if ASK_USE_OPTIMIZED_CONTEXT and is_structured_analysis(self.repository_analysis):
            # Use smart extraction: executive summary + matching capabilities
            context = get_query_optimized_context(
                self.repository_analysis,
                question,
                max_tokens=2000,  # ~8K chars, focused on relevant capabilities
            )
            logger.info(f"Using query-optimized context ({len(context)} chars) for query optimization")
            return context
        else:
            # Legacy: truncate to ~3K tokens
            truncated = self.repository_analysis[:12000]
            logger.info(f"Using truncated analysis ({len(truncated)} chars) for query optimization")
            return truncated

    def _optimize_query(self, question: str) -> str:
        """
        Optimize the user's question for better retrieval using repository context.

        Uses repository analysis to understand context and generate an optimized
        search query that will find relevant code and documentation.
        Similar to retrieval_query generation in wiki structure planning.
        """
        if not self.optimize_query or not self.repository_analysis:
            return question

        self._emit_thinking("query_optimization", "Optimizing query using repository context...")

        # Get optimized context based on analysis format
        repo_context = self._get_context_for_query_optimization(question)

        # Build the optimization prompt
        user_prompt = QUERY_OPTIMIZATION_USER_PROMPT.format(repository_context=repo_context, question=question)

        try:
            # Use limited output tokens for fast query optimization
            # Bind max_tokens to the LLM for this specific call
            llm_for_query = self.llm.bind(max_tokens=self.query_output_tokens)
            response = llm_for_query.invoke(
                [SystemMessage(content=QUERY_OPTIMIZATION_SYSTEM_PROMPT), HumanMessage(content=user_prompt)]
            )

            optimized = response.content.strip()

            # Validate the optimized query
            # Don't use if it's too long, empty, or looks like an explanation
            if (
                optimized
                and len(optimized) < 1000
                and not optimized.startswith(("I ", "The ", "Here ", "Based "))
                and ":" not in optimized[:50]
            ):  # Avoid responses like "Optimized query: ..."
                self._emit_thinking(
                    "query_optimized",
                    f"Using optimized query ({len(optimized)} chars)",
                    original=question[:100],
                    optimized=optimized[:150],
                )
                return optimized
            else:
                logger.warning("Query optimization produced invalid result, using original")

        except Exception as e:
            logger.warning(f"Query optimization failed: {e}")

        return question

    def _format_context(self, documents: list[Document]) -> tuple[str, list[AskSource]]:
        """
        Format retrieved documents as context for the LLM using token-based budgeting.

        Uses structured format similar to wiki generation's _format_simple_context:
        - <code_source: path> tags for clear source identification
        - <symbol_name>, <symbol_type> for code elements
        - <implementation> tags for code content

        Returns:
            Tuple of (formatted_context, sources_list)
        """
        context_parts = []
        sources = []
        total_tokens = 0

        for i, doc in enumerate(documents):
            source_path = doc.metadata.get("source", "") or doc.metadata.get("file_path", "unknown")
            symbol = doc.metadata.get("symbol_name", "")
            symbol_type = doc.metadata.get("symbol_type", "")
            chunk_type = doc.metadata.get("chunk_type", "code")
            score = doc.metadata.get("relevance_score")
            content = doc.page_content

            # Build structured block (similar to wiki generation format)
            block_parts = [f'<source index="{i + 1}" path="{source_path}">']

            if symbol:
                block_parts.append(f"<symbol_name>{symbol}</symbol_name>")
            if symbol_type:
                block_parts.append(f"<symbol_type>{symbol_type}</symbol_type>")
            if chunk_type and chunk_type != "code":
                block_parts.append(f"<content_type>{chunk_type}</content_type>")

            block_parts.append("<implementation>")
            block_parts.append(content)
            block_parts.append("</implementation>")
            block_parts.append("</source>")
            block_parts.append("")  # Empty line separator

            block = "\n".join(block_parts)

            # Check token limit using tiktoken
            block_tokens = self.token_counter.count(block)

            if total_tokens + block_tokens > self.max_context_tokens:
                self._emit_thinking(
                    "context_truncated",
                    f"Token budget reached at {i} documents ({total_tokens:,}/{self.max_context_tokens:,} tokens)",
                    total_tokens=total_tokens,
                    budget=self.max_context_tokens,
                )
                break

            context_parts.append(block)
            total_tokens += block_tokens

            sources.append(
                AskSource(
                    index=i + 1,
                    source=source_path,
                    symbol=symbol if symbol else None,
                    chunk_type=chunk_type,
                    relevance_score=score,
                )
            )

        self._emit_thinking(
            "context_formatted",
            f"Formatted {len(sources)} sources ({total_tokens:,} tokens)",
            sources_count=len(sources),
            total_tokens=total_tokens,
        )

        return "\n".join(context_parts), sources

    def _build_sources_reference(self, sources: list[AskSource]) -> str:
        """Build a clear reference of available sources for the LLM"""
        lines = ["The following source files are available in the context:"]
        lines.append("")

        # Group by file path for cleaner presentation
        files_seen = {}
        for src in sources:
            path = src.source
            if path not in files_seen:
                files_seen[path] = []
            if src.symbol:
                files_seen[path].append(src.symbol)

        for path, symbols in files_seen.items():
            if symbols:
                symbols_str = ", ".join(f"`{s}`" for s in symbols[:5])  # Limit to 5 symbols
                if len(symbols) > 5:
                    symbols_str += f" (+{len(symbols) - 5} more)"
                lines.append(f"- `{path}` — {symbols_str}")
            else:
                lines.append(f"- `{path}`")

        return "\n".join(lines)

    def _get_context_for_answer(self, question: str) -> str:
        """
        Get repository overview context for answer generation.

        For structured JSON: Returns focused markdown with summary + relevant capabilities
        For legacy markdown: Returns truncated analysis

        Returns:
            Markdown string suitable for LLM system prompt (~3K tokens max)
        """
        if not self.repository_analysis:
            return ""

        if ASK_USE_OPTIMIZED_CONTEXT and is_structured_analysis(self.repository_analysis):
            # For answer: include summary + relevant capabilities + patterns
            # This provides broader context than query optimization
            context = render_structured_analysis_as_markdown(
                self.repository_analysis,
                include_sections=["summary", "capabilities", "patterns"],
                max_chars=10000,  # ~2.5K tokens
            )
            logger.info(f"Using rendered structured analysis ({len(context)} chars) for answer")
            return context
        else:
            # Legacy: truncate to ~3K tokens
            truncated = self.repository_analysis[:12000]
            logger.info(f"Using truncated analysis ({len(truncated)} chars) for answer")
            return truncated

    def _generate_answer(self, question: str, context: str, sources: list[AskSource]) -> str:
        """
        Generate answer using LLM with the retrieved context.

        Uses limited output tokens (16K) for faster response.
        """
        context_tokens = self.token_counter.count(context)
        self._emit_thinking(
            "generating", f"Generating answer from {len(sources)} sources ({context_tokens:,} tokens)..."
        )

        # Build sources reference for the prompt
        sources_reference = self._build_sources_reference(sources)

        # Build the user prompt with structured context
        user_prompt = ANSWER_USER_PROMPT.format(context=context, sources_reference=sources_reference, question=question)

        # Build system prompt with repository overview if available
        if self.repository_analysis:
            # Get optimized overview based on analysis format
            analysis_overview = self._get_context_for_answer(question)
            system_prompt = f"""{ANSWER_SYSTEM_PROMPT}

## Repository Overview
{analysis_overview}
"""
        else:
            system_prompt = ANSWER_SYSTEM_PROMPT

        try:
            # Use limited output tokens for answer generation
            # Bind max_tokens to the LLM for this specific call
            llm_for_answer = self.llm.bind(max_tokens=self.answer_output_tokens)
            response = llm_for_answer.invoke([SystemMessage(content=system_prompt), HumanMessage(content=user_prompt)])
            return response.content
        except Exception as e:
            logger.error(f"Answer generation failed: {e}")
            raise

    def ask(self, question: str, k: int = 15) -> AskResponse:
        """
        Answer a question about the repository.

        Args:
            question: User's question about the repository
            k: Number of documents to retrieve (default 15)

        Returns:
            AskResponse with answer, sources, and thinking steps
        """
        # Reset thinking steps
        self._thinking_steps = []

        self._emit_thinking("start", f"Processing question: {question[:100]}...")

        # Step 1: Optimize the query using repository context
        search_query = self._optimize_query(question)

        # Step 2: Retrieve documents
        self._emit_thinking("retrieving", "Searching repository...", query=search_query[:100])

        try:
            documents = self.retriever.search_repository(search_query, k=k)
        except Exception as e:
            logger.error(f"Retrieval failed: {e}")
            return AskResponse(
                answer=f"Sorry, I encountered an error while searching the repository: {str(e)}",
                thinking_steps=self._thinking_steps,
                query_used=search_query,
            )

        self._emit_thinking(
            "retrieved",
            f"Found {len(documents)} relevant documents",
            count=len(documents),
            top_sources=[doc.metadata.get("source", "unknown")[:50] for doc in documents[:5]],
        )

        # Handle empty results
        if not documents:
            self._emit_thinking("no_results", "No relevant documents found")
            return AskResponse(
                answer="I couldn't find relevant information in the repository to answer your question. "
                "Try rephrasing your question or asking about a different aspect of the codebase.",
                thinking_steps=self._thinking_steps,
                query_used=search_query,
                documents_retrieved=0,
            )

        # Step 3: Format context with token budgeting
        context, sources = self._format_context(documents)

        # Step 4: Generate answer with citations
        try:
            answer = self._generate_answer(question, context, sources)
        except Exception as e:
            logger.error(f"Answer generation failed: {e}")
            return AskResponse(
                answer=f"I found relevant documents but encountered an error generating the answer: {str(e)}",
                sources=sources,
                thinking_steps=self._thinking_steps,
                query_used=search_query,
                documents_retrieved=len(documents),
            )

        self._emit_thinking("complete", "Answer generated successfully")

        return AskResponse(
            answer=answer,
            sources=sources,
            thinking_steps=self._thinking_steps,
            query_used=search_query,
            documents_retrieved=len(documents),
        )

    def ask_with_history(self, question: str, chat_history: list[dict[str, str]], k: int = 15) -> AskResponse:
        """
        Answer a question with conversation history context.

        Args:
            question: Current question
            chat_history: List of {"role": "user"|"assistant", "content": "..."}
            k: Number of documents to retrieve

        Returns:
            AskResponse
        """
        # Build context from history
        if chat_history:
            history_context = "\n".join(
                [
                    f"{'User' if msg['role'] == 'user' else 'Assistant'}: {msg['content'][:200]}"
                    for msg in chat_history[-4:]  # Last 4 messages
                ]
            )

            # Enhance question with history context for better retrieval
            enhanced_question = f"""Given this conversation context:
{history_context}

Current question: {question}"""
        else:
            enhanced_question = question

        return self.ask(enhanced_question, k=k)
