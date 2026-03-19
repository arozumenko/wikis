"""
Wiki Generation Agents

This package contains agents for wiki documentation generation:
- OptimizedWikiGenerationAgent: Main LangGraph-based wiki generator
- AgenticDocGenerator: Legacy section-based generator (token-based clustering)
- AgenticDocGeneratorV2: Planning + Tool Calling generator (LLM-driven sections)
"""

from .agentic_doc_generator import AgenticDocGenerator, should_use_agentic_mode
from .agentic_doc_generator_v2 import AgenticDocGeneratorV2
from .wiki_graph_optimized import OptimizedWikiGenerationAgent

__all__ = [
    "OptimizedWikiGenerationAgent",
    "AgenticDocGenerator",
    "AgenticDocGeneratorV2",
    "should_use_agentic_mode",
]
