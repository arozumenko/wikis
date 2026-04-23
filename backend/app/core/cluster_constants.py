"""
Clustering-specific constants and utility functions.

Provides the mathematical foundation for hierarchical Leiden community
detection.  These constants control:

- Symbol classification for page identity and clustering roles
- Page sizing and adaptive target formulas
- Hub detection thresholds
- Leiden resolution parameters
- Centroid scoring
- Test path detection
- Doc-cluster identification

Ported from the DeepWiki plugin ``constants.py`` — only the subset
relevant to graph clustering is included here.
"""

import math
import re
from typing import FrozenSet, List, Pattern

# Re-export from main constants for clustering convenience
from .constants import ARCHITECTURAL_SYMBOLS, DOC_SYMBOL_TYPES

# =============================================================================
# Page Identity Classification (capability-first clustering)
# =============================================================================
# These sets partition code symbols by their page-identity role.
# Used by cluster_planner seed selection and candidate_builder classification.

# Symbols that can anchor a page (drive page identity in mixed clusters).
# Ordered by SYMBOL_TYPE_PRIORITY when selecting seeds.
PAGE_IDENTITY_SYMBOLS: FrozenSet[str] = frozenset(
    {
        "class",
        "interface",
        "trait",
        "protocol",
        "enum",
        "struct",
        "record",
        "module",
        "namespace",
        "function",
    }
)

# Symbols that support page content but do not drive page identity
# in mixed clusters; eligible as seeds only when no identity symbol exists.
SUPPORTING_CODE_SYMBOLS: FrozenSet[str] = frozenset(
    {
        "constant",
        "type_alias",
        "macro",
        "method",
        "property",
    }
)

# Documentation symbol kinds used for docs-only page detection.
DOC_CLUSTER_SYMBOLS: FrozenSet[str] = frozenset(
    {
        "module_doc",
        "file_doc",
    }
)

# Numeric priority for seed ordering — aligned with
# graph_builder._get_type_priority().
SYMBOL_TYPE_PRIORITY = {
    "class": 10,
    "interface": 10,
    "trait": 10,
    "protocol": 10,
    "enum": 9,
    "struct": 9,
    "record": 9,
    "module": 8,
    "namespace": 8,
    "function": 7,
    "constant": 6,
    "type_alias": 6,
    "macro": 6,
    "method": 3,
    "property": 2,
    "module_doc": 1,
    "file_doc": 1,
}


# =============================================================================
# Page Sizing Constants
# =============================================================================

MIN_PAGE_SIZE = 5  # Merge clusters smaller than this
MERGE_THRESHOLD = 4  # Hard minimum — below this, always merge
DEFAULT_MAX_PAGE_SIZE = 25  # Default cap; overridden by adaptive_max_page_size


def adaptive_max_page_size(node_count: int) -> int:
    """Scale max page size based on repo complexity.

    Larger repos need bigger pages to avoid 100+ page wikis.

    | Nodes      | Max page size |
    |------------|---------------|
    | < 500      | 25            |
    | 500–1999   | 35            |
    | 2000–4999  | 45            |
    | 5000+      | 60            |
    """
    if node_count >= 5000:
        return 60
    if node_count >= 2000:
        return 45
    if node_count >= 500:
        return 35
    return DEFAULT_MAX_PAGE_SIZE


def target_section_count(n_files: int) -> int:
    """Target section count — logarithmic scaling on file count.

    Sections represent high-level logical domains.  A repository has a
    bounded number of major subsystems regardless of node count, so
    the target grows as ``log₂(files)`` rather than ``√nodes``.

    | Files      | Target sections |
    |------------|-----------------|
    | < 10       | 5               |
    | 50         | 7               |
    | 116        | 9               |
    | 500        | 11              |
    | 1 000      | 12              |
    | 5 000      | 15              |
    | 50 000     | 19              |
    | cap        | 20              |
    """
    if n_files < 10:
        return 5
    return max(5, min(20, math.ceil(1.2 * math.log2(max(10, n_files)))))


def target_total_pages(n_nodes: int) -> int:
    """Target total-page count — ``√(nodes/7)`` scaling on node count.

    Pages represent content granularity proportional to code volume
    (node count), NOT file count.

    | Nodes      | Target pages |
    |------------|--------------|
    | 729        | 11           |
    | 5 000      | 27           |
    | 11 417     | 41           |
    | 50 000     | 85           |
    | 280 000    | 200 (cap)    |
    """
    return max(8, min(200, math.ceil(math.sqrt(n_nodes / 7))))


# =============================================================================
# Hub Detection Constants
# =============================================================================

DEFAULT_HUB_Z_THRESHOLD = 3.0  # Z-score threshold for in-degree hub detection


# =============================================================================
# Centroid Detection Constants
# =============================================================================

CENTROID_ARCHITECTURAL_MIN_PRIORITY = 5
MIN_CENTROIDS = 1
MAX_CENTROIDS = 10


# =============================================================================
# Leiden Resolution Constants
# =============================================================================

LEIDEN_FILE_SECTION_RESOLUTION = 1.0  # File-contracted section pass (γ)
LEIDEN_PAGE_RESOLUTION = 1.0  # Per-section page pass (γ)


# =============================================================================
# Doc-Cluster Detection
# =============================================================================

DOC_DOMINANT_THRESHOLD = 0.7  # >70% doc nodes → doc-dominant cluster


# =============================================================================
# Test Path Detection
# =============================================================================
# Path-based heuristics for identifying test files.
# Used when ``exclude_tests`` is enabled to remove test nodes
# before clustering.  Matches are case-insensitive.

_TEST_PATH_PATTERNS: List[Pattern[str]] = [
    # ── Directory-based ──
    re.compile(r"(^|/)tests?/", re.IGNORECASE),
    re.compile(r"(^|/)__tests__/", re.IGNORECASE),
    re.compile(r"(^|/)specs?/", re.IGNORECASE),
    re.compile(r"(^|/)testing/", re.IGNORECASE),
    re.compile(r"(^|/)test_?helpers?/", re.IGNORECASE),
    re.compile(r"(^|/)fixtures/", re.IGNORECASE),
    re.compile(r"(^|/)mocks?/", re.IGNORECASE),
    re.compile(r"(^|/)testdata/", re.IGNORECASE),
    re.compile(r"(^|/)testutils?/", re.IGNORECASE),
    # ── File-based ──
    re.compile(r"(^|/)test_[^/]+\.\w+$", re.IGNORECASE),
    re.compile(r"(^|/)[^/]+_test\.\w+$", re.IGNORECASE),
    re.compile(r"(^|/)[^/]+\.test\.\w+$", re.IGNORECASE),
    re.compile(r"(^|/)[^/]+\.spec\.\w+$", re.IGNORECASE),
    re.compile(r"(^|/)[^/]+Tests?\.\w+$", re.IGNORECASE),  # FooTest.java
    re.compile(r"(^|/)conftest\.py$", re.IGNORECASE),
    re.compile(r"(^|/)setup_tests?\.\w+$", re.IGNORECASE),
]


def is_test_path(rel_path: str) -> bool:
    """Check if a relative file path matches test code patterns.

    Uses path-based heuristics (directory names and file suffixes) to
    identify test files.  This is deterministic, language-agnostic,
    and covers the vast majority of real-world test layouts.

    Returns ``True`` if the path matches any test pattern.
    """
    return any(p.search(rel_path) for p in _TEST_PATH_PATTERNS)
