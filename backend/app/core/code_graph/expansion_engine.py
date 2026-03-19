"""
Smart Graph Expansion Engine — Priority-Based Edge Traversal.

Replaces the naive ``predecessors[:10] + successors[:10]`` expansion in
``wiki_graph_optimized._get_docs_by_target_symbols()`` with a
relationship-aware, budget-controlled algorithm.

Design:
    Each matched symbol is expanded by walking edges in priority order:
        P0 (always): inheritance, implementation, defines_body, creates, instantiates
        P1 (budget): composition, aggregation, alias_of, specializes
        P2 (fill):   calls (free functions only), references
        SKIP:        defines, imports, decorates

    Per-symbol budget caps prevent a single highly-connected node from
    consuming the full context window.  C++ declaration/implementation
    pairs are augmented in-place (no extra document — enriches the
    existing node's content).

See PLANNING_SMART_EXPANSION.md for the full rationale.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)

# Feature flag — when "0", falls back to naive expansion.
# Default ON since Phase 1 (bidirectional expansion) is complete and tested.
SMART_EXPANSION_ENABLED = os.getenv("WIKIS_SMART_EXPANSION", "1") == "1"

# ---------------------------------------------------------------------------
# Types that qualify as expansion targets (same as constants.EXPANSION_SYMBOL_TYPES
# but imported here to avoid circular imports if constants grows).
# ---------------------------------------------------------------------------
CLASS_LIKE_TYPES: frozenset[str] = frozenset(
    {
        "class",
        "interface",
        "struct",
        "enum",
        "trait",
    }
)

EXPANSION_WORTHY_TYPES: frozenset[str] = frozenset(
    {
        "class",
        "interface",
        "struct",
        "enum",
        "trait",
        "function",
        "constant",
        "type_alias",
        "macro",
        "module_doc",
        "file_doc",
    }
)

# ---------------------------------------------------------------------------
# Priority configuration — budget per relationship type per symbol.
#
# ``direction``:
#   'successors'   — follow outgoing edges  (source → target)
#   'predecessors' — follow incoming edges  (target ← source)
#   'augment'      — special: in-place content augmentation (no new node)
#
# ``budget``:
#   0 means "special handling" (e.g. augment), not "skip".
#   Use SKIP_RELATIONSHIPS for explicitly skipped types.
# ---------------------------------------------------------------------------
EXPANSION_PRIORITIES: dict[str, dict[str, Any]] = {
    # --- P0: Forward (structurally essential for understanding the symbol) ---
    "inheritance": {"budget": 3, "direction": "successors", "priority": 0},
    "implementation": {"budget": 2, "direction": "successors", "priority": 0},
    "defines_body": {"budget": 0, "direction": "predecessors", "priority": 0},  # augment
    "creates": {"budget": 3, "direction": "successors", "priority": 0},
    "instantiates": {"budget": 2, "direction": "successors", "priority": 0},
    # --- P1: Forward (architecturally important) ---
    "composition": {"budget": 2, "direction": "successors", "priority": 1},
    "aggregation": {"budget": 2, "direction": "successors", "priority": 1},
    "alias_of": {"budget": 2, "direction": "successors", "priority": 1},
    "specializes": {"budget": 2, "direction": "successors", "priority": 1},
    # --- P1: Backward (who extends / implements / composes me) ---
    # These use the same edge types but walk predecessors instead of successors.
    # Budgets in the per-type _expand_* functions, not here (edge_type reuse).
    # Documented here for design visibility:
    #   inheritance  (pred, 3)  → derived/child classes
    #   implementation (pred, 3) → implementors of interfaces/traits
    #   composition  (pred, 2)  → classes that compose this type
    #   creates      (pred, 2)  → functions/classes that create this type
    #   calls        (pred, 3)  → callers of functions
    #   references   (pred, 2)  → types that reference this symbol
    # --- P2: Forward fill ---
    "calls": {"budget": 2, "direction": "successors", "priority": 2},
    "references": {"budget": 2, "direction": "successors", "priority": 2},
}

# Relationships explicitly skipped (containment, import, decoration).
SKIP_RELATIONSHIPS: frozenset[str] = frozenset(
    {
        "defines",
        "imports",
        "decorates",
        "contains",
        "annotates",
        "exports",
        "overrides",
        "assigns",
        "returns",
        "parameter",
        "reads",
        "writes",
        "captures",
        "hides",
        "uses_type",  # Not emitted by any parser — covered by 'references'
    }
)

# Global cap: at most this many *new* expansion nodes across all matched
# symbols on a single wiki page.
GLOBAL_EXPANSION_CAP = 50

# Per-symbol cap: at most this many new nodes added per matched symbol.
PER_SYMBOL_CAP = 15


# ============================================================================
# Data classes
# ============================================================================


@dataclass
class AugmentedContent:
    """In-place augmentation result (e.g. C++ decl+impl body, Go struct+receiver methods)."""

    node_id: str
    original_content: str
    augmented_content: str
    impl_file: str  # relative path to the implementation / method file


@dataclass
class ExpansionResult:
    """Outcome of smart expansion for a set of matched nodes."""

    expanded_nodes: set[str] = field(default_factory=set)
    augmentations: dict[str, AugmentedContent] = field(default_factory=dict)
    expansion_reasons: dict[str, str] = field(default_factory=dict)  # node_id → reason


# ============================================================================
# Edge Utilities (handle both DiGraph and MultiDiGraph)
# ============================================================================


def edges_between(graph, source: str, target: str) -> list[dict]:
    """Return a list of edge attribute dicts between *source* and *target*.

    Works for both ``DiGraph`` (single edge → list of 1) and ``MultiDiGraph``
    (multiple edges → list of N).
    """
    edge_data = graph.get_edge_data(source, target)
    if edge_data is None:
        return []
    # MultiDiGraph: edge_data is {0: {…}, 1: {…}, …}
    if isinstance(edge_data, dict) and all(isinstance(k, int) for k in edge_data):
        return list(edge_data.values())
    # DiGraph: edge_data is a single dict
    return [edge_data]


def has_relationship(graph, source: str, target: str, *rel_types: str) -> bool:
    """Check if *any* edge (source → target) carries one of *rel_types*."""
    rel_set = set(r.lower() for r in rel_types)
    for edge in edges_between(graph, source, target):
        if edge.get("relationship_type", "").lower() in rel_set:
            return True
    return False


def get_edge_annotations(graph, source: str, target: str, rel_type: str) -> dict:
    """Return annotations dict from the first edge of *rel_type* between source→target.

    Useful for extracting metadata such as ``type_args`` from template
    instantiation edges.  Returns ``{}`` if no matching edge or no annotations.
    """
    for edge in edges_between(graph, source, target):
        if edge.get("relationship_type", "").lower() == rel_type.lower():
            return edge.get("annotations", {}) or {}
    return {}


def format_type_args(type_args: list[str]) -> str:
    """Format template type arguments for human-readable expansion reasons.

    E.g. ``['int', 'Point']`` → ``'<int, Point>'``.
    Returns empty string if no args.
    """
    if not type_args:
        return ""
    return f"<{', '.join(type_args)}>"


def get_neighbors_by_relationship(
    graph,
    node_id: str,
    rel_types: set[str],
    direction: str = "successors",
    limit: int = 10,
    type_filter: frozenset[str] | None = None,
) -> list[tuple[str, str]]:
    """Get neighbor node IDs filtered by edge relationship type.

    Args:
        graph: NetworkX graph (DiGraph or MultiDiGraph).
        node_id: Source node.
        rel_types: Lowercased relationship type names to accept.
        direction: ``'successors'`` or ``'predecessors'``.
        limit: Maximum number of results.
        type_filter: If given, only include neighbors whose
            ``symbol_type`` is in this set.

    Returns:
        List of ``(neighbor_id, matched_rel_type)`` pairs, up to *limit*.
    """
    results: list[tuple[str, str]] = []
    if direction == "successors":
        neighbors = graph.successors(node_id)
    elif direction == "predecessors":
        neighbors = graph.predecessors(node_id)
    else:
        return results

    for neighbor in neighbors:
        if len(results) >= limit:
            break

        # Optional symbol-type filter
        if type_filter:
            sym_type = graph.nodes.get(neighbor, {}).get("symbol_type", "").lower()
            if sym_type not in type_filter:
                continue

        # Check edges for matching relationship type
        if direction == "successors":
            src, tgt = node_id, neighbor
        else:
            src, tgt = neighbor, node_id

        for edge in edges_between(graph, src, tgt):
            rt = edge.get("relationship_type", "").lower()
            if rt in rel_types:
                results.append((neighbor, rt))
                break  # one match per neighbor pair is enough

    return results


def resolve_alias_chain(graph, node_id: str, max_hops: int = 5) -> str | None:
    """Follow ``alias_of`` edges to resolve a type alias to a concrete type.

    Returns the concrete target node_id, or ``None`` if resolution fails.
    """
    visited: set[str] = {node_id}
    current = node_id
    for _ in range(max_hops):
        found_next = False
        for succ in graph.successors(current):
            if succ in visited:
                continue
            if has_relationship(graph, current, succ, "alias_of"):
                visited.add(succ)
                sym_type = graph.nodes.get(succ, {}).get("symbol_type", "").lower()
                if sym_type == "type_alias":
                    current = succ
                    found_next = True
                    break
                else:
                    # Concrete type reached
                    return succ
        if not found_next:
            break
    return current if current != node_id else None


# ============================================================================
# C++ Augmentation
# ============================================================================


def augment_cpp_node(graph, node_id: str) -> AugmentedContent | None:
    """Augment a C++ declaration with its out-of-line implementation body.

    For a **function / method** node, looks for incoming ``defines_body``
    edges (impl → decl) from a *different file*.

    For a **class / struct** node, gathers implementations of all declared
    methods (class → defines → method ← defines_body ← impl) and appends
    them grouped by file.

    Returns an ``AugmentedContent`` if augmentation was possible, else ``None``.
    """
    node_data = graph.nodes.get(node_id, {})
    if not node_data:
        return None

    sym_type = node_data.get("symbol_type", "").lower()
    content = _get_source_text(node_data)
    if not content:
        return None

    decl_file = node_data.get("file_path", "") or node_data.get("rel_path", "")

    # --- Function / method augmentation: single defines_body predecessor ---
    if sym_type in ("function", "method", "constructor"):
        impl_text, impl_file = _find_defines_body_impl(graph, node_id, decl_file)
        if impl_text:
            augmented = content + f"\n\n/* Implementation from {impl_file} */\n" + impl_text
            return AugmentedContent(
                node_id=node_id,
                original_content=content,
                augmented_content=augmented,
                impl_file=impl_file,
            )
        return None

    # --- Class / struct augmentation: gather method implementations ---
    if sym_type in ("class", "struct"):
        impl_by_file: dict[str, list[str]] = {}
        # Find methods declared by this class
        for method_node in graph.successors(node_id):
            if not has_relationship(graph, node_id, method_node, "defines"):
                continue
            method_type = graph.nodes.get(method_node, {}).get("symbol_type", "").lower()
            if method_type not in ("method", "constructor", "function"):
                continue
            impl_text, impl_file = _find_defines_body_impl(graph, method_node, decl_file)
            if impl_text:
                impl_by_file.setdefault(impl_file, []).append(impl_text)

        if impl_by_file:
            augmented = content
            for impl_file, impls in sorted(impl_by_file.items()):
                augmented += f"\n\n/* Implementations from {impl_file} ({len(impls)} methods) */\n"
                augmented += "\n\n".join(impls)
            return AugmentedContent(
                node_id=node_id,
                original_content=content,
                augmented_content=augmented,
                impl_file=next(iter(impl_by_file)),
            )

    return None


def _find_defines_body_impl(graph, decl_node: str, decl_file: str) -> tuple[str | None, str]:
    """Find the out-of-line implementation for a declaration node.

    DEFINES_BODY direction is impl → decl, so look at *predecessors*.

    Returns ``(source_text, rel_path)`` or ``(None, "")``.
    """
    for pred in graph.predecessors(decl_node):
        if not has_relationship(graph, pred, decl_node, "defines_body"):
            continue
        pred_data = graph.nodes.get(pred, {})
        pred_file = pred_data.get("file_path", "") or pred_data.get("rel_path", "")
        # Only augment cross-file implementations (same-file = already in content)
        if pred_file and pred_file != decl_file:
            impl_text = _get_source_text(pred_data)
            impl_rel = pred_data.get("rel_path", pred_file)
            if impl_text:
                return impl_text, impl_rel
    return None, ""


# ============================================================================
# Go Augmentation — Receiver Methods
# ============================================================================


def augment_go_node(graph, node_id: str) -> AugmentedContent | None:
    """Augment a Go struct with receiver methods defined in other files.

    In Go, methods are **not** lexically nested inside the struct.  They
    are top-level declarations linked to the struct via
    ``DEFINES(struct → method)`` edges with ``cross_file: True``.

    For a **struct** node, gathers all receiver-method source texts that
    live in different files and appends them grouped by file — analogous
    to how ``augment_cpp_node`` merges out-of-line .cpp implementations
    into a .h declaration.

    Returns an ``AugmentedContent`` if cross-file methods were found,
    else ``None``.  Same-file methods are already visible in the struct's
    source text context, so they are not duplicated here.
    """
    node_data = graph.nodes.get(node_id, {})
    if not node_data:
        return None

    sym_type = node_data.get("symbol_type", "").lower()
    if sym_type not in ("struct", "class"):
        return None

    content = _get_source_text(node_data)
    if not content:
        return None

    struct_file = node_data.get("rel_path", "") or node_data.get("file_path", "")

    # Collect receiver methods via DEFINES edges
    methods_by_file: dict[str, list[str]] = {}
    for method_node in graph.successors(node_id):
        if not has_relationship(graph, node_id, method_node, "defines"):
            continue

        method_data = graph.nodes.get(method_node, {})
        method_type = method_data.get("symbol_type", "").lower()
        if method_type not in ("method", "function"):
            continue

        # Only augment cross-file methods; same-file methods are already
        # visible in the source context surrounding the struct.
        method_file = method_data.get("rel_path", "") or method_data.get("file_path", "")
        if not method_file or method_file == struct_file:
            continue

        # Check edge annotations for cross-file flag (belt-and-suspenders)
        edge_dicts = edges_between(graph, node_id, method_node)
        is_cross = any(
            ed.get("annotations", {}).get("cross_file", False)
            for ed in edge_dicts
            if ed.get("relationship_type", "").lower() == "defines"
        )
        # Even without the annotation, different file paths are sufficient
        if not is_cross and method_file == struct_file:
            continue

        method_text = _get_source_text(method_data)
        if method_text and method_text.strip():
            methods_by_file.setdefault(method_file, []).append(method_text)

    if not methods_by_file:
        return None

    augmented = content
    for mfile, mtexts in sorted(methods_by_file.items()):
        augmented += f"\n\n// Receiver methods from {mfile} ({len(mtexts)} method{'s' if len(mtexts) != 1 else ''})\n"
        augmented += "\n\n".join(mtexts)

    return AugmentedContent(
        node_id=node_id,
        original_content=content,
        augmented_content=augmented,
        impl_file=next(iter(methods_by_file)),
    )


def _get_source_text(node_data: dict) -> str:
    """Extract source text from a graph node, checking various attribute names."""
    content = node_data.get("source_text", "")
    if not content:
        symbol_obj = node_data.get("symbol")
        if symbol_obj and hasattr(symbol_obj, "source_text"):
            content = symbol_obj.source_text or ""
    if not content:
        content = node_data.get("content", "") or node_data.get("source_code", "") or node_data.get("code", "")
    return content or ""


# ============================================================================
# Transitive 2-Hop Helpers
# ============================================================================


def find_composed_types(graph, class_node: str, limit: int = 5) -> list[str]:
    """Find types composed by a class via 2-hop: Class →[defines]→ Field →[composition|aggregation]→ Type.

    Returns node IDs of the composed types (up to *limit*).
    """
    results: list[str] = []
    for field_node in graph.successors(class_node):
        if len(results) >= limit:
            break
        if not has_relationship(graph, class_node, field_node, "defines"):
            continue
        field_type = graph.nodes.get(field_node, {}).get("symbol_type", "").lower()
        if field_type not in ("field", "variable", "property", "attribute"):
            continue
        for type_node in graph.successors(field_node):
            if len(results) >= limit:
                break
            if has_relationship(graph, field_node, type_node, "composition", "aggregation"):
                sym_type = graph.nodes.get(type_node, {}).get("symbol_type", "").lower()
                if sym_type in CLASS_LIKE_TYPES or sym_type in EXPANSION_WORTHY_TYPES:
                    if type_node not in results:
                        results.append(type_node)
    return results


def find_creates_from_methods(graph, class_node: str, limit: int = 4) -> list[str]:
    """Find types created by a class's methods via 2-hop: Class →[defines]→ Method →[creates]→ Type.

    Returns node IDs of created types (up to *limit*).
    """
    results: list[str] = []
    for method_node in graph.successors(class_node):
        if len(results) >= limit:
            break
        if not has_relationship(graph, class_node, method_node, "defines"):
            continue
        method_type = graph.nodes.get(method_node, {}).get("symbol_type", "").lower()
        if method_type not in ("method", "constructor", "function"):
            continue
        for created_node in graph.successors(method_node):
            if len(results) >= limit:
                break
            if has_relationship(graph, method_node, created_node, "creates"):
                sym_type = graph.nodes.get(created_node, {}).get("symbol_type", "").lower()
                if sym_type in CLASS_LIKE_TYPES or sym_type in EXPANSION_WORTHY_TYPES:
                    if created_node not in results:
                        results.append(created_node)
    return results


def find_calls_to_free_functions(graph, class_node: str, limit: int = 3) -> list[str]:
    """Find free functions called by a class's methods via 2-hop.

    Class →[defines]→ Method →[calls]→ Function  (only standalone functions).
    """
    results: list[str] = []
    for method_node in graph.successors(class_node):
        if len(results) >= limit:
            break
        if not has_relationship(graph, class_node, method_node, "defines"):
            continue
        for callee_node in graph.successors(method_node):
            if len(results) >= limit:
                break
            if has_relationship(graph, method_node, callee_node, "calls"):
                sym_type = graph.nodes.get(callee_node, {}).get("symbol_type", "").lower()
                if sym_type == "function":
                    if callee_node not in results:
                        results.append(callee_node)
    return results


# ============================================================================
# Per-Symbol-Type Expansion Strategies
# ============================================================================


def _expand_class(graph, node_id: str, already_expanded: set[str]) -> tuple[set[str], dict[str, str]]:
    """Expand a class / interface / struct / enum / trait node.

    Bidirectional: follows both outgoing (forward) and incoming (backward)
    edges to capture the full architectural context.

    Forward (successors):  base classes, interfaces, composed types, created types
    Backward (predecessors): derived classes, implementors, composers, creators
    """
    new_nodes: set[str] = set()
    reasons: dict[str, str] = {}

    def _add(nid: str, reason: str):
        if nid != node_id:
            new_nodes.add(nid)
            # Keep first reason (strategies called in priority order P0 → P1 → P2)
            if nid not in reasons:
                reasons[nid] = reason

    # ── Forward (successors) ─────────────────────────────────────────────

    # P0: Inheritance (class → base classes)
    for nid, rt in get_neighbors_by_relationship(graph, node_id, {"inheritance"}, "successors", limit=3):
        _add(nid, f"base class via {rt}")

    # P0: Implementation (class → interfaces)
    for nid, rt in get_neighbors_by_relationship(graph, node_id, {"implementation"}, "successors", limit=2):
        _add(nid, f"implements interface via {rt}")

    # P0: Creates via methods (2-hop)
    for nid in find_creates_from_methods(graph, node_id, limit=3):
        _add(nid, "created by class method")

    # P0: Instantiates (template instantiations with concrete type args)
    for nid, rt in get_neighbors_by_relationship(
        graph,
        node_id,
        {"instantiates"},
        "successors",
        limit=2,
    ):
        annot = get_edge_annotations(graph, node_id, nid, "instantiates")
        ta = format_type_args(annot.get("type_args", []))
        _add(nid, f"instantiates template{ta} via {rt}")

    # P1: Composition / Aggregation via fields (2-hop)
    for nid in find_composed_types(graph, node_id, limit=2):
        _add(nid, "composed type via field")

    # P1: Direct Composition / Aggregation (1-hop)
    # Some languages (Go, Rust) emit direct composition/aggregation edges
    # from struct → composed type without a field intermediary (e.g. Go embedding).
    for nid, rt in get_neighbors_by_relationship(
        graph,
        node_id,
        {"composition", "aggregation"},
        "successors",
        limit=3,
        type_filter=CLASS_LIKE_TYPES | frozenset({"type_alias"}),
    ):
        _add(nid, f"composed type via direct {rt}")

    # P1: Alias resolution (direct alias_of on class is rare but possible)
    for nid, _rt in get_neighbors_by_relationship(graph, node_id, {"alias_of"}, "successors", limit=2):
        resolved = resolve_alias_chain(graph, nid)
        if resolved:
            _add(resolved, "alias target")

    # P1: Specializes (C++ template base) + template type arguments
    for nid, rt in get_neighbors_by_relationship(graph, node_id, {"specializes"}, "successors", limit=2):
        _add(nid, f"template base via {rt}")
        for arg_nid, _arg_rt in get_neighbors_by_relationship(
            graph,
            nid,
            {"references"},
            "successors",
            limit=2,
            type_filter=CLASS_LIKE_TYPES | frozenset({"type_alias"}),
        ):
            _add(arg_nid, f"template type arg via {rt}")

    # P2: References (return/param types, template args on this class)
    for nid, rt in get_neighbors_by_relationship(
        graph,
        node_id,
        {"references"},
        "successors",
        limit=2,
        type_filter=CLASS_LIKE_TYPES | frozenset({"type_alias"}),
    ):
        _add(nid, f"referenced type via {rt}")

    # P2: Calls to free functions via methods (2-hop)
    for nid in find_calls_to_free_functions(graph, node_id, limit=2):
        _add(nid, "free function called by method")

    # ── Backward (predecessors) — "who extends / uses / composes me" ────

    # P1-backward: Derived classes (who inherits FROM this class)
    for nid, rt in get_neighbors_by_relationship(
        graph,
        node_id,
        {"inheritance"},
        "predecessors",
        limit=3,
        type_filter=CLASS_LIKE_TYPES,
    ):
        _add(nid, f"derived class via {rt}")

    # P1-backward: Implementors (who implements this interface/trait)
    node_sym = graph.nodes.get(node_id, {}).get("symbol_type", "").lower()
    if node_sym in ("interface", "trait"):
        for nid, rt in get_neighbors_by_relationship(
            graph,
            node_id,
            {"implementation"},
            "predecessors",
            limit=3,
            type_filter=CLASS_LIKE_TYPES,
        ):
            _add(nid, f"implementor via {rt}")

    # P1-backward: Classes/functions/methods that compose or aggregate this type
    for nid, rt in get_neighbors_by_relationship(
        graph,
        node_id,
        {"composition", "aggregation"},
        "predecessors",
        limit=2,
        type_filter=CLASS_LIKE_TYPES | frozenset({"function", "method"}),
    ):
        _add(nid, f"composed by via {rt}")

    # P2-backward: Functions/classes/methods that create instances of this type
    for nid, rt in get_neighbors_by_relationship(
        graph,
        node_id,
        {"creates", "instantiates"},
        "predecessors",
        limit=2,
        type_filter=CLASS_LIKE_TYPES | frozenset({"function", "method"}),
    ):
        if rt == "instantiates":
            annot = get_edge_annotations(graph, nid, node_id, "instantiates")
            ta = format_type_args(annot.get("type_args", []))
            _add(nid, f"instantiated by{ta} via {rt}")
        else:
            _add(nid, f"created by via {rt}")

    # P2-backward: Types that reference this class (e.g. as param/return type)
    for nid, rt in get_neighbors_by_relationship(
        graph,
        node_id,
        {"references"},
        "predecessors",
        limit=2,
        type_filter=CLASS_LIKE_TYPES | frozenset({"function", "method"}),
    ):
        _add(nid, f"referenced by via {rt}")

    return new_nodes, reasons


def _expand_function(graph, node_id: str, already_expanded: set[str]) -> tuple[set[str], dict[str, str]]:
    """Expand a standalone function node.

    Bidirectional: forward to created types and callees,
    backward to callers and referencing types.
    """
    new_nodes: set[str] = set()
    reasons: dict[str, str] = {}

    def _add(nid: str, reason: str):
        if nid != node_id:
            if nid not in already_expanded or nid in new_nodes:
                new_nodes.add(nid)
                if nid not in reasons:
                    reasons[nid] = reason

    # ── Forward (successors) ─────────────────────────────────────────────

    # P0: Creates (factory functions → created types)
    for nid, rt in get_neighbors_by_relationship(
        graph,
        node_id,
        {"creates", "instantiates"},
        "successors",
        limit=3,
        type_filter=CLASS_LIKE_TYPES | frozenset({"function", "constant"}),
    ):
        if rt == "instantiates":
            annot = get_edge_annotations(graph, node_id, nid, "instantiates")
            ta = format_type_args(annot.get("type_args", []))
            _add(nid, f"instantiates template{ta} via {rt}")
        else:
            _add(nid, f"created type via {rt}")

    # P2: Calls to other functions/methods
    for nid, rt in get_neighbors_by_relationship(
        graph,
        node_id,
        {"calls"},
        "successors",
        limit=2,
        type_filter=frozenset({"function", "method"}),
    ):
        _add(nid, f"callee via {rt}")

    # P2: References (return types, parameter types, constants)
    for nid, rt in get_neighbors_by_relationship(
        graph,
        node_id,
        {"references"},
        "successors",
        limit=2,
        type_filter=CLASS_LIKE_TYPES | frozenset({"type_alias", "constant"}),
    ):
        _add(nid, f"referenced type via {rt}")

    # ── Backward (predecessors) — "who calls / uses me" ─────────────────

    # P1-backward: Callers of this function/method
    for nid, rt in get_neighbors_by_relationship(
        graph,
        node_id,
        {"calls"},
        "predecessors",
        limit=3,
        type_filter=frozenset({"function", "method"}) | CLASS_LIKE_TYPES,
    ):
        _add(nid, f"caller via {rt}")

    # P2-backward: Types that reference this function (e.g. function pointers, callbacks)
    for nid, rt in get_neighbors_by_relationship(
        graph,
        node_id,
        {"references"},
        "predecessors",
        limit=2,
        type_filter=CLASS_LIKE_TYPES | frozenset({"function", "method"}),
    ):
        _add(nid, f"referenced by via {rt}")

    return new_nodes, reasons


def _expand_constant(graph, node_id: str, already_expanded: set[str]) -> tuple[set[str], dict[str, str]]:
    """Expand a constant / module-level variable node.

    Bidirectional: forward to initialization functions and types,
    backward to symbols that consume/reference this constant.
    """
    new_nodes: set[str] = set()
    reasons: dict[str, str] = {}

    def _add(nid: str, reason: str):
        if nid != node_id:
            new_nodes.add(nid)
            if nid not in reasons:
                reasons[nid] = reason

    # ── Forward (successors) ─────────────────────────────────────────────

    # P2: Calls (initialization)
    for nid, rt in get_neighbors_by_relationship(
        graph,
        node_id,
        {"calls"},
        "successors",
        limit=2,
        type_filter=frozenset({"function"}),
    ):
        _add(nid, f"init function via {rt}")

    # P2: References (type)
    for nid, rt in get_neighbors_by_relationship(
        graph,
        node_id,
        {"references"},
        "successors",
        limit=1,
        type_filter=CLASS_LIKE_TYPES | frozenset({"type_alias"}),
    ):
        _add(nid, f"type ref via {rt}")

    # ── Backward (predecessors) — "who uses this constant" ──────────────

    # P2-backward: Symbols that reference this constant
    for nid, rt in get_neighbors_by_relationship(
        graph,
        node_id,
        {"references"},
        "predecessors",
        limit=2,
        type_filter=CLASS_LIKE_TYPES | frozenset({"function", "method", "constructor", "constant"}),
    ):
        _add(nid, f"constant user via {rt}")

    return new_nodes, reasons


def _expand_type_alias(graph, node_id: str, already_expanded: set[str]) -> tuple[set[str], dict[str, str]]:
    """Expand a type alias by resolving the alias chain.

    Bidirectional: forward to concrete type via alias chain,
    backward to symbols that use this alias.
    """
    new_nodes: set[str] = set()
    reasons: dict[str, str] = {}

    # ── Forward: resolve alias chain to concrete type ────────────────────
    resolved = resolve_alias_chain(graph, node_id)
    if resolved and resolved != node_id:
        new_nodes.add(resolved)
        reasons[resolved] = "alias chain target"

    # ── Backward (predecessors) — "who uses this alias" ──────────────────
    for nid, rt in get_neighbors_by_relationship(
        graph,
        node_id,
        {"references"},
        "predecessors",
        limit=2,
        type_filter=CLASS_LIKE_TYPES | frozenset({"function"}),
    ):
        new_nodes.add(nid)
        reasons[nid] = f"alias user via {rt}"

    return new_nodes, reasons


def _expand_macro(graph, node_id: str, already_expanded: set[str]) -> tuple[set[str], dict[str, str]]:
    """Expand a macro node.

    Macros are architecturally significant (e.g. FMT_COMPILE, DEFINE_*)
    and often wrap or reference classes/functions.  Follow references and
    calls edges to pull in the types/functions the macro interacts with.
    """
    new_nodes: set[str] = set()
    reasons: dict[str, str] = {}

    def _add(nid: str, reason: str):
        if nid != node_id:
            new_nodes.add(nid)
            if nid not in reasons:
                reasons[nid] = reason

    # P1: References (types the macro wraps/uses)
    for nid, rt in get_neighbors_by_relationship(
        graph,
        node_id,
        {"references"},
        "successors",
        limit=3,
        type_filter=CLASS_LIKE_TYPES | frozenset({"function", "type_alias"}),
    ):
        _add(nid, f"referenced type via {rt}")

    # P2: Calls (functions the macro invokes)
    for nid, rt in get_neighbors_by_relationship(
        graph,
        node_id,
        {"calls"},
        "successors",
        limit=2,
        type_filter=frozenset({"function"}),
    ):
        _add(nid, f"callee via {rt}")

    # P2: Who uses this macro (predecessors with 'references' or 'calls')
    # Include 'method' — class methods commonly use macros (e.g. CHECK(), ASSERT())
    for nid, rt in get_neighbors_by_relationship(
        graph,
        node_id,
        {"references", "calls"},
        "predecessors",
        limit=3,
        type_filter=CLASS_LIKE_TYPES | frozenset({"function", "method", "constant"}),
    ):
        _add(nid, f"macro user via {rt}")

    return new_nodes, reasons


# Strategy dispatch
_STRATEGY_MAP = {
    "class": _expand_class,
    "interface": _expand_class,
    "struct": _expand_class,
    "enum": _expand_class,
    "trait": _expand_class,
    "function": _expand_function,
    "method": _expand_function,  # methods use same strategy as functions
    "constructor": _expand_function,  # constructors use same strategy as functions
    "constant": _expand_constant,
    "type_alias": _expand_type_alias,
    "macro": _expand_macro,
}


# ============================================================================
# Main Entry Point
# ============================================================================


def expand_smart(
    matched_nodes: set[str],
    graph,
    *,
    per_symbol_cap: int = PER_SYMBOL_CAP,
    global_cap: int = GLOBAL_EXPANSION_CAP,
) -> ExpansionResult:
    """Perform relationship-aware smart expansion on *matched_nodes*.

    Args:
        matched_nodes: Node IDs that were matched by name lookup / FTS5.
        graph: NetworkX graph (``DiGraph`` or ``MultiDiGraph``).
        per_symbol_cap: Max new expansion nodes added per matched symbol.
        global_cap: Max total new expansion nodes across all matched symbols.

    Returns:
        ``ExpansionResult`` with ``expanded_nodes`` (including originals),
        ``augmentations`` for C++ in-place content enrichment, and
        ``expansion_reasons`` for debugging / metadata.
    """
    result = ExpansionResult(expanded_nodes=set(matched_nodes))
    total_added = 0

    for node_id in matched_nodes:
        if total_added >= global_cap:
            logger.debug("[SMART_EXPAND] Global cap %d reached, stopping", global_cap)
            break

        node_data = graph.nodes.get(node_id, {})
        sym_type = node_data.get("symbol_type", "").lower()

        # --- C++ augmentation (P0, in-place, no budget cost) ---
        lang = node_data.get("language", "").lower()
        if lang in ("cpp", "c++", "c", "cc", "cxx"):
            aug = augment_cpp_node(graph, node_id)
            if aug:
                result.augmentations[node_id] = aug
                logger.debug(
                    "[SMART_EXPAND] Augmented %s with impl from %s",
                    node_id,
                    aug.impl_file,
                )

        # --- Go augmentation (P0, in-place, no budget cost) ---
        # Go structs have receiver methods in separate files — analogous
        # to C++ header/impl split.  Augment the struct content with its
        # cross-file receiver methods so the LLM sees the full API surface.
        if lang == "go" and sym_type in ("struct", "class"):
            aug = augment_go_node(graph, node_id)
            if aug:
                result.augmentations[node_id] = aug
                logger.debug(
                    "[SMART_EXPAND] Go struct %s augmented with receiver methods from %s",
                    node_id,
                    aug.impl_file,
                )

        # --- Per-type expansion strategy ---
        strategy_fn = _STRATEGY_MAP.get(sym_type)
        if strategy_fn is None:
            # Unknown type — skip expansion (still keep the matched node)
            continue

        new_nodes, reasons = strategy_fn(graph, node_id, result.expanded_nodes)

        # Apply per-symbol cap
        symbol_added = 0
        for nid in new_nodes:
            if symbol_added >= per_symbol_cap or total_added >= global_cap:
                break
            if nid not in result.expanded_nodes:
                result.expanded_nodes.add(nid)
                result.expansion_reasons[nid] = reasons.get(nid, "expansion")
                symbol_added += 1
                total_added += 1

        if symbol_added > 0:
            logger.debug(
                "[SMART_EXPAND] %s (%s): added %d expansion nodes",
                node_id,
                sym_type,
                symbol_added,
            )

    logger.info(
        "[SMART_EXPAND] %d matched → %d total nodes (%d expanded, %d augmented)",
        len(matched_nodes),
        len(result.expanded_nodes),
        total_added,
        len(result.augmentations),
    )
    return result
