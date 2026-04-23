"""
Language Heuristics — Phase 7.

Per-language page shaping and expansion hints for the 8 rich parser
languages.  Each heuristic family provides:

1. ``extra_expansion_rels`` — additional relationship types to follow
   during smart expansion for this language.
2. ``skip_symbol_types`` — symbol types to suppress (e.g. blanket impls).
3. ``prefer_impl_over_decl`` — whether to prefer implementation files
   over declaration files for this language.
4. ``grouping_hint`` — how symbols should be grouped (package, module,
   namespace, file, etc.).
5. ``impl_augmentation_priority`` — relative priority of cross-file
   impl augmentation (higher = more likely to spend budget on impls).

Gated behind ``FeatureFlags.language_hints``.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Dict, FrozenSet, Optional, Set

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class LanguageHints:
    """Expansion hints for a specific language family."""

    language: str
    # Additional rel types to traverse beyond the standard P0/P1/P2
    extra_expansion_rels: FrozenSet[str] = frozenset()
    # Symbol types to suppress from expansion results
    skip_symbol_types: FrozenSet[str] = frozenset()
    # When True, prefer implementation files over declarations
    prefer_impl_over_decl: bool = False
    # Suggested grouping level for page identity
    grouping_hint: str = "module"
    # Priority multiplier for cross-file impl augmentation (1.0 = normal)
    impl_augmentation_priority: float = 1.0
    # File extensions that mark declaration-only files
    declaration_extensions: FrozenSet[str] = frozenset()
    # File extensions that mark implementation files
    implementation_extensions: FrozenSet[str] = frozenset()
    # Whether interface/trait → implementor expansion is high-priority
    interface_impl_priority: bool = False


# ─── Language-specific hint definitions ────────────────────────────

_CPP_HINTS = LanguageHints(
    language="cpp",
    extra_expansion_rels=frozenset({"defines_body", "specializes"}),
    skip_symbol_types=frozenset({"macro"}),  # suppress macros from expansion
    prefer_impl_over_decl=True,
    grouping_hint="namespace",
    impl_augmentation_priority=2.0,  # C++ heavily needs header→impl bridging
    declaration_extensions=frozenset({".h", ".hpp", ".hxx", ".hh"}),
    implementation_extensions=frozenset({".cpp", ".cc", ".cxx", ".c"}),
)

_CSHARP_HINTS = LanguageHints(
    language="c_sharp",
    extra_expansion_rels=frozenset({"implementation"}),
    grouping_hint="namespace",
    interface_impl_priority=True,
    impl_augmentation_priority=1.2,
)

_GO_HINTS = LanguageHints(
    language="go",
    extra_expansion_rels=frozenset({"defines_body"}),
    grouping_hint="package",
    impl_augmentation_priority=1.5,  # Go receiver methods often in other files
    interface_impl_priority=True,
)

_JAVA_HINTS = LanguageHints(
    language="java",
    extra_expansion_rels=frozenset({"implementation", "annotates"}),
    grouping_hint="package",
    interface_impl_priority=True,
    impl_augmentation_priority=1.2,
)

_JAVASCRIPT_HINTS = LanguageHints(
    language="javascript",
    extra_expansion_rels=frozenset({"exports"}),
    skip_symbol_types=frozenset({"constant"}),  # config constants are noisy
    grouping_hint="module",
    impl_augmentation_priority=0.8,
)

_PYTHON_HINTS = LanguageHints(
    language="python",
    extra_expansion_rels=frozenset({"decorates"}),
    grouping_hint="module",
    impl_augmentation_priority=1.0,
)

_RUST_HINTS = LanguageHints(
    language="rust",
    extra_expansion_rels=frozenset({"implementation", "defines_body"}),
    skip_symbol_types=frozenset(),  # blanket impls are filtered by expansion engine
    prefer_impl_over_decl=True,
    grouping_hint="module",
    impl_augmentation_priority=1.8,  # Rust trait impl blocks in separate files
    interface_impl_priority=True,
)

_TYPESCRIPT_HINTS = LanguageHints(
    language="typescript",
    extra_expansion_rels=frozenset({"exports"}),
    grouping_hint="module",
    declaration_extensions=frozenset({".d.ts"}),
    prefer_impl_over_decl=True,  # Prefer .ts over .d.ts
    impl_augmentation_priority=1.3,
)

# ─── Registry ───────────────────────────────────────────────────────

_HINTS_BY_LANGUAGE: Dict[str, LanguageHints] = {
    "cpp": _CPP_HINTS,
    "c": _CPP_HINTS,  # C uses same heuristics as C++
    "c_sharp": _CSHARP_HINTS,
    "go": _GO_HINTS,
    "java": _JAVA_HINTS,
    "javascript": _JAVASCRIPT_HINTS,
    "python": _PYTHON_HINTS,
    "rust": _RUST_HINTS,
    "typescript": _TYPESCRIPT_HINTS,
}

# Default hints for languages not in the registry
_DEFAULT_HINTS = LanguageHints(language="unknown")


def get_language_hints(language: str) -> LanguageHints:
    """Return expansion hints for the given language.

    Parameters
    ----------
    language : str
        Language identifier as stored in ``repo_nodes.language``
        (lowercase: ``python``, ``cpp``, ``go``, etc.).

    Returns
    -------
    LanguageHints
        Language-specific hints, or defaults if the language is unknown.
    """
    return _HINTS_BY_LANGUAGE.get(language.lower(), _DEFAULT_HINTS)


def detect_dominant_language(conn, node_ids: list) -> Optional[str]:
    """Detect the dominant language in a set of nodes.

    Parameters
    ----------
    conn : sqlite3.Connection
        DB connection.
    node_ids : list[str]
        Node IDs to check.

    Returns
    -------
    str or None
        The most common language, or None if empty.
    """
    if not node_ids:
        return None

    placeholders = ",".join("?" for _ in node_ids)
    rows = conn.execute(
        f"SELECT language, COUNT(*) as cnt FROM repo_nodes "
        f"WHERE node_id IN ({placeholders}) AND language IS NOT NULL "
        f"GROUP BY language ORDER BY cnt DESC LIMIT 1",
        node_ids,
    ).fetchall()

    if rows:
        return rows[0][0]
    return None


def should_include_in_expansion(
    hints: LanguageHints,
    symbol_type: str,
    rel_path: str,
) -> bool:
    """Check whether a symbol should be included based on language hints.

    Parameters
    ----------
    hints : LanguageHints
        Current language hints.
    symbol_type : str
        The symbol's type (class, function, etc.).
    rel_path : str
        The symbol's file path.

    Returns
    -------
    bool
        False if the symbol should be suppressed.
    """
    if symbol_type.lower() in hints.skip_symbol_types:
        return False

    # For declaration-preferring languages, suppress declaration-only files
    # when implementation exists (handled at a higher level — here we just
    # check for obvious declaration-only patterns)
    if hints.prefer_impl_over_decl and hints.declaration_extensions:
        for ext in hints.declaration_extensions:
            if rel_path.endswith(ext):
                # Mark as declaration-only; caller should check for impl
                # We still include it — the caller uses impl_augmentation_priority
                # to decide whether to also fetch the implementation
                pass

    return True


def get_extra_rel_types(hints: LanguageHints) -> FrozenSet[str]:
    """Return additional relationship types to follow for this language.

    These are added to the standard P0/P1/P2 sets during smart expansion.
    """
    return hints.extra_expansion_rels


def compute_augmentation_budget_fraction(
    hints: LanguageHints,
    base_fraction: float = 0.3,
) -> float:
    """Compute the fraction of token budget to allocate to cross-file augmentation.

    Parameters
    ----------
    hints : LanguageHints
        Current language hints.
    base_fraction : float
        Default fraction (30% of budget for augmentation).

    Returns
    -------
    float
        Adjusted fraction, clamped to [0.1, 0.6].
    """
    adjusted = base_fraction * hints.impl_augmentation_priority
    return max(0.1, min(0.6, adjusted))
