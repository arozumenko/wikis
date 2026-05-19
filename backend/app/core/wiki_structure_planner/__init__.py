"""
Wiki Structure Planner — unified pipeline (#242).

Provides the skeleton builder, evidence-pack builders, and structure tools
consumed by the unified planner → writer → gate → verifier pipeline.

Usage::

    from app.core.wiki_structure_planner import build_skeleton, StructureSkeleton
"""

from .structure_engine import StructurePlannerConfig, WikiStructurePlannerEngine
from .structure_prompts import (
    get_structure_planner_instructions,
    get_structure_task_prompt,
)
from .structure_skeleton import (
    ArtifactInfo,
    Cluster,
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
    "ArtifactInfo",
    "Cluster",
    "DirCluster",
    "DocCluster",
    "SymbolInfo",
    "StructureCollector",
]
