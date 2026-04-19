"""
Shared utilities for the cluster-based wiki pipeline.

Light-weight module with no heavy dependencies (no LangGraph, no LLM imports)
so it can be imported from tests without pulling in the wiki agent stack.
"""

from __future__ import annotations

import re
from typing import Optional


def extract_macro_id(rationale: str) -> Optional[int]:
    """Parse ``macro=N`` from a cluster-planner rationale string.

    Args:
        rationale: A rationale string produced by the cluster-based
            structure planner, e.g.
            ``"Grouped by graph clustering (macro=3, micro=7, 12 symbols)"``

    Returns:
        The integer macro-cluster ID, or ``None`` if the pattern is not found
        or *rationale* is falsy.

    Examples:
        >>> extract_macro_id("Grouped by graph clustering (macro=3, micro=7, 12 symbols)")
        3
        >>> extract_macro_id("No cluster info here") is None
        True
        >>> extract_macro_id(None) is None
        True
    """
    if not rationale:
        return None
    m = re.search(r'macro=(\d+)', rationale)
    if m:
        return int(m.group(1))
    return None
