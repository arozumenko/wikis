"""
Deep Research Module - Using langchain-ai/deepagents

This module provides deep research capabilities for codebase analysis
using the actual DeepAgents library with:
- TodoListMiddleware for planning and progress tracking
- FilesystemMiddleware for context offloading (large outputs → files)
- SubAgentMiddleware for delegation to specialist agents
- Custom tools wrapping WikiRetrieverStack and GraphManager

Events are streamed via LangGraph's astream (NOT LangChain callbacks).
"""

from .research_engine import DeepResearchEngine, ResearchConfig, create_deep_research_engine
from .research_tools import create_codebase_tools

__all__ = [
    "DeepResearchEngine",
    "create_deep_research_engine",
    "create_codebase_tools",
    "ResearchConfig",
]
