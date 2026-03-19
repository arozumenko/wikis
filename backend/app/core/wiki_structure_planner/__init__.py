"""
Wiki Structure Planner - DeepAgents-based wiki structure generation

This module provides a lean, tool-based approach to generating wiki structures
using the DeepAgents library. It follows the same pattern as deep_research:

- Minimal prompts with clear workflow
- FilesystemMiddleware for repository exploration
- No pre-embedded context - agent discovers via tools
- TOOL-BASED OUTPUT: Uses define_section/define_page tools instead of JSON generation

The tool-based output approach eliminates the slow JSON generation step (~140s).
Instead, the agent calls tools incrementally to define sections and pages,
and we assemble the final structure from the tool calls.

Usage:
    from wiki_structure_planner import WikiStructurePlannerEngine, StructurePlannerConfig

    config = StructurePlannerConfig(page_budget=25)
    engine = WikiStructurePlannerEngine(
        repo_root="/path/to/repo",
        llm_client=chat_model,
        config=config
    )

    result = engine.plan_structure(repo_name="my-project")
    # result is a dict for WikiStructureSpec.model_validate()
"""

from .structure_engine import StructurePlannerConfig, WikiStructurePlannerEngine
from .structure_prompts import (
    get_structure_planner_instructions,
    get_structure_task_prompt,
)
from .structure_refiner import refine_with_llm
from .structure_skeleton import (
    DirCluster,
    DocCluster,
    StructureSkeleton,
    SymbolInfo,
    build_skeleton,
)
from .structure_tools import StructureCollector

__all__ = [
    "WikiStructurePlannerEngine",
    "StructurePlannerConfig",
    "get_structure_planner_instructions",
    "get_structure_task_prompt",
    "build_skeleton",
    "StructureSkeleton",
    "DirCluster",
    "DocCluster",
    "SymbolInfo",
    "refine_with_llm",
    "StructureCollector",
]
