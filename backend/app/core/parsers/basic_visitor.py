"""Lightweight tree-sitter visitor parser driven by per-language config (#119).

For languages that don't justify a full deep-parser (4000-LOC files like
``cpp_enhanced_parser.py``) but where regex-based extraction loses too
much, this module provides a single :class:`BasicVisitorParser` that
walks any tree-sitter AST using a small :class:`LanguageConfig` map of
node-type strings. Adding a new language is then ~30 lines of config +
a fixture, not a new ~2000-LOC parser class.

Design notes — what we borrowed from graphify (and what we didn't):

* **Borrowed**: the ``_make_id`` slug helper (deterministic cross-
  language IDs) and the **two-pass pattern** (structural walk collects
  symbols + stashes function/method bodies; a second pass walks those
  bodies to emit ``CALLS`` edges). graphify earned both through real
  iteration; no point re-deriving them.

* **Not borrowed**: graphify's hand-rolled per-language ``extract_<lang>``
  closures (~2400 lines of copy-paste). The config-driven approach was
  exactly the refactor they never did — that's the value this module
  adds over a direct port.

The result is a parser whose depth is shallower than the deep parsers
(no type inference, no field-resolution, no template instantiation
graph) but whose output shape — :class:`Symbol`, :class:`Relationship`,
:class:`ParseResult` — is identical, so downstream graph builders,
retrievers, and wiki agents need zero special-casing.

Promotion path: when a language proves frequent enough in user repos
(Ruby + PHP are the most likely first candidates), it gets upgraded to
a deep parser inheriting from :class:`BaseParser` directly. The
:class:`LanguageConfig` for that language can then be retired or kept
as a fallback for partial parses.
"""

from __future__ import annotations

import logging
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

from app.core.parsers.base_parser import (
    BaseParser,
    LanguageCapabilities,
    ParseResult,
    Position,
    Range,
    Relationship,
    RelationshipType,
    Scope,
    Symbol,
    SymbolType,
)

if TYPE_CHECKING:
    from tree_sitter import Node, Parser

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# ID + name helpers
# ---------------------------------------------------------------------------


_ID_NORMALIZER = re.compile(r"[^a-zA-Z0-9]+")


def make_id(*parts: str) -> str:
    """Build a stable, lowercase, underscore-separated node ID from one
    or more name parts.

    Borrowed from graphify (``_make_id``). Deterministic across runs +
    cross-language, which lets ``BasicVisitorParser`` produce the same
    symbol ID for the same source location regardless of file path
    casing, separator style, or unicode normalization quirks.

    Empty parts are dropped; leading/trailing separators are stripped;
    runs of non-alphanumerics collapse to a single ``_``.
    """
    combined = "_".join(p.strip("_.") for p in parts if p)
    cleaned = _ID_NORMALIZER.sub("_", combined)
    return cleaned.strip("_").lower()


# ---------------------------------------------------------------------------
# Per-language configuration
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class LanguageConfig:
    """Per-language node-type map for :class:`BasicVisitorParser`.

    Most fields are tuples of tree-sitter node-type strings — the visitor
    pattern checks ``node.type in cfg.class_nodes`` to decide whether
    the current AST node should emit a :class:`Symbol` of kind ``CLASS``.

    Languages vary in their tree-sitter grammar conventions; the docstrings
    below show what's typical so config authors can grep their own
    grammar's ``grammar.js`` for matches.
    """

    #: Lowercase short name. Used by ``parse_file`` and the graph builder
    #: dispatch (``self.rich_parsers[name]``).
    name: str

    #: ``tree_sitter_language_pack.get_parser(name)`` argument. Usually
    #: matches :attr:`name` but can differ for languages with separate
    #: legacy / modern grammars.
    tree_sitter_name: str

    #: File extensions handled by this config. Lowercase with leading dot.
    extensions: tuple[str, ...]

    #: Top-level type definitions — typically ``("class_declaration",)``
    #: or similar. Anything in this tuple emits ``SymbolType.CLASS``.
    class_nodes: tuple[str, ...] = ()

    #: Interface / trait / protocol declarations → ``SymbolType.INTERFACE``.
    interface_nodes: tuple[str, ...] = ()

    #: Enum declarations → ``SymbolType.ENUM``.
    enum_nodes: tuple[str, ...] = ()

    #: Struct / record types → ``SymbolType.STRUCT``.
    struct_nodes: tuple[str, ...] = ()

    #: Callable declarations. The visitor emits ``METHOD`` when the node
    #: is nested inside a class context (``parent_id is not None``) and
    #: ``FUNCTION`` otherwise. Many grammars use a single node type for
    #: both forms (Kotlin / Scala / Ruby) — list it here; the scope
    #: decides the symbol kind.
    function_nodes: tuple[str, ...] = ()

    #: Optional override for languages where method declarations have a
    #: distinct node type from top-level functions (e.g. PHP's
    #: ``method_declaration`` vs ``function_definition``). When non-empty,
    #: only these node types emit ``METHOD`` (and they emit it only when
    #: nested inside a class context).
    method_nodes: tuple[str, ...] = ()

    #: Import / require / use statements → emit an ``IMPORTS`` relationship
    #: from the file to the imported module.
    import_nodes: tuple[str, ...] = ()

    #: Function/method call sites → emit a ``CALLS`` relationship from
    #: the containing function to the callee.
    call_nodes: tuple[str, ...] = ()

    #: Tree-sitter field-name on a class node that holds the superclass /
    #: parent type. Used to emit ``INHERITANCE`` relationships. ``None``
    #: means the language has no single inheritance or we haven't mapped it.
    inherit_field: str | None = None

    #: Alternative to :attr:`inherit_field` for grammars that don't use a
    #: field name. The visitor searches the class node's direct children
    #: for a node of one of these types and takes its text as the parent.
    inherit_node_types: tuple[str, ...] = ()

    #: Extra node types (beyond ``identifier`` and ``name``) that can be
    #: treated as a declaration's name. Ruby's ``class Foo`` uses
    #: ``constant`` for ``Foo``; Kotlin uses ``type_identifier``. Adding
    #: them here makes the generic name finder pick them up.
    name_node_types: tuple[str, ...] = ()

    #: Node types that represent a "name chain" (Lua's
    #: ``method_index_expression`` = ``Foo:bar``, PHP's ``qualified_name``).
    #: When the name finder encounters one, it drills down and picks the
    #: last identifier-shaped descendant — so the symbol name is ``bar``,
    #: not ``Foo``. Without this, ``function Foo:bar()`` would be
    #: extracted as a function named ``Foo``.
    name_chain_types: tuple[str, ...] = ()

    #: Tree-sitter field-name on a declaration node that holds the name.
    #: Most grammars use ``"name"`` but a few (e.g. tree-sitter-ruby)
    #: name the identifier child differently for some kinds.
    name_field: str = "name"

    #: Tree-sitter field-name for a node's body block. Used during the
    #: second pass to walk only the body for ``CALLS`` edges, not the
    #: entire declaration subtree.
    body_field: str = "body"



# ---------------------------------------------------------------------------
# Visitor
# ---------------------------------------------------------------------------


class BasicVisitorParser(BaseParser):
    """Generic tree-sitter visitor driven by a :class:`LanguageConfig`.

    Constructor takes the language config + a tree-sitter parser instance.
    ``parse_file`` runs the two-pass extraction and returns a standard
    :class:`ParseResult` — same shape as the deep parsers, just shallower
    content.
    """

    def __init__(self, config: LanguageConfig) -> None:
        # ``self.config`` must be set before ``super().__init__`` —
        # :class:`BaseParser` calls ``_define_capabilities`` during init,
        # and our override reads ``self.config`` to derive the matrix.
        self.config = config
        super().__init__(language=config.name)
        self._parser: Parser | None = None
        self._init_tree_sitter()

    def _init_tree_sitter(self) -> None:
        """Best-effort tree-sitter setup. A failed import means the
        ``[tree-sitter-language-pack]`` dependency is missing for this
        language — the parser is constructed in a degraded state and
        :meth:`parse_file` returns an empty result rather than crashing.
        """
        try:
            from tree_sitter_language_pack import get_parser

            self._parser = get_parser(self.config.tree_sitter_name)
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "[basic_visitor] tree-sitter init failed for %s (%s): %s",
                self.config.name, self.config.tree_sitter_name, exc,
            )
            self._parser = None

    def _define_capabilities(self) -> LanguageCapabilities:
        """Derive a capabilities matrix from the config's node-type sets."""
        supported_symbols: set[SymbolType] = set()
        if self.config.class_nodes:
            supported_symbols.add(SymbolType.CLASS)
        if self.config.interface_nodes:
            supported_symbols.add(SymbolType.INTERFACE)
        if self.config.enum_nodes:
            supported_symbols.add(SymbolType.ENUM)
        if self.config.struct_nodes:
            supported_symbols.add(SymbolType.STRUCT)
        if self.config.function_nodes:
            supported_symbols.add(SymbolType.FUNCTION)
        if self.config.method_nodes or self.config.class_nodes:
            supported_symbols.add(SymbolType.METHOD)

        supported_rels: set[RelationshipType] = set()
        if self.config.call_nodes:
            supported_rels.add(RelationshipType.CALLS)
        if self.config.import_nodes:
            supported_rels.add(RelationshipType.IMPORTS)
        # Either ``inherit_field`` OR ``inherit_node_types`` produces the
        # edge — check both. Rio R1: without the second branch, 4 of 5
        # shipped configs (Kotlin/PHP/Scala/Lua) silently advertise no
        # INHERITANCE capability despite actually emitting it.
        if self.config.inherit_field or self.config.inherit_node_types:
            supported_rels.add(RelationshipType.INHERITANCE)
        supported_rels.add(RelationshipType.DEFINES)

        return LanguageCapabilities(
            language=self.config.name,
            supported_symbols=supported_symbols,
            supported_relationships=supported_rels,
            supports_type_inference=False,
            supports_cross_file_analysis=False,
            has_classes=bool(self.config.class_nodes),
            has_interfaces=bool(self.config.interface_nodes),
        )

    def _get_supported_extensions(self) -> set[str]:
        return set(self.config.extensions)

    # ------------------------------------------------------------------
    # parse_file — the public entry point
    # ------------------------------------------------------------------

    def parse_file(
        self,
        file_path: str | Path,
        content: str | None = None,
    ) -> ParseResult:
        """Parse a source file into symbols + relationships.

        Two-pass design:
          1. **Structural pass**: walk the AST once. Emit Symbol records
             for class/function/method/interface/enum declarations.
             Stash ``(containing_id, body_node)`` for each function/method
             so the second pass has a scoped slice to walk.
          2. **Calls pass**: for each stashed body, walk it and emit a
             ``CALLS`` Relationship for every call-site node, where the
             source is the containing function's ID and the target is
             the callee's name slug.
        """
        start = time.time()
        file_path = str(file_path)

        if self._parser is None:
            return ParseResult(
                file_path=file_path,
                language=self.config.name,
                symbols=[],
                relationships=[],
                errors=[f"tree-sitter parser unavailable for {self.config.name}"],
                parse_time=time.time() - start,
            )

        if content is None:
            try:
                with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                    content = f.read()
            except OSError as exc:
                return ParseResult(
                    file_path=file_path,
                    language=self.config.name,
                    symbols=[],
                    relationships=[],
                    errors=[f"failed to read file: {exc}"],
                    parse_time=time.time() - start,
                )

        source_bytes = content.encode("utf-8", errors="replace")
        tree = self._parser.parse(source_bytes)

        file_stem = Path(file_path).stem
        file_id = make_id(file_stem)

        state = _VisitState(
            file_path=file_path,
            file_stem=file_stem,
            file_id=file_id,
            source_bytes=source_bytes,
            config=self.config,
        )
        self._visit_structural(tree.root_node, state, parent_id=None)

        # Second pass: per stashed body, walk for calls. Done after the
        # structural pass completes so every Symbol ID is already known
        # and call-edges can target real declarations when names collide.
        for containing_id, body_node in state.bodies:
            self._visit_calls(body_node, state, containing_id)

        # Implementation:
        return ParseResult(
            file_path=file_path,
            language=self.config.name,
            symbols=state.symbols,
            relationships=state.relationships,
            imports=state.imports,
            parse_time=time.time() - start,
        )

    # ------------------------------------------------------------------
    # BaseParser abstract requirements — most logic lives in parse_file
    # ------------------------------------------------------------------

    def extract_symbols(self, ast_node: Any, file_path: str) -> list[Symbol]:  # noqa: ARG002
        """Not used directly — :meth:`parse_file` produces symbols. Kept
        for the :class:`BaseParser` contract; some legacy callers may
        still invoke it. Returns an empty list to signal "look at parse_file".
        """
        return []

    def extract_relationships(  # noqa: D401
        self,
        ast_node: Any,  # noqa: ARG002
        symbols: list[Symbol],  # noqa: ARG002
        file_path: str,  # noqa: ARG002
    ) -> list[Relationship]:
        return []

    # ------------------------------------------------------------------
    # Structural traversal
    # ------------------------------------------------------------------

    def _visit_structural(
        self,
        node: Node,
        state: _VisitState,
        parent_id: str | None,
    ) -> None:
        cfg = self.config
        t = node.type

        # Imports — emit an edge from the file to the imported name.
        # Doesn't recurse: most languages have import statements at
        # module top-level and they don't contain nested declarations.
        if t in cfg.import_nodes:
            target = _read_node_text(node, state.source_bytes).strip()
            if target:
                state.imports.append(target)
                state.relationships.append(
                    Relationship(
                        source_symbol=state.file_id,
                        target_symbol=make_id(target),
                        relationship_type=RelationshipType.IMPORTS,
                        source_file=state.file_path,
                        source_range=_node_range(node),
                    )
                )
            # Imports may sit inside an outer container (e.g. Lua's
            # ``do … end``); still recurse for languages that nest them.
            for child in node.children:
                self._visit_structural(child, state, parent_id)
            return

        # Class-like declarations. Note: the ``return`` statements
        # below are unconditional once this branch matches — even if
        # ``_make_symbol`` returns None (anonymous class, malformed
        # source), we don't fall through to the default recurse at the
        # bottom (which would re-walk the same subtree). The class
        # branch takes ownership of the node.
        if t in cfg.class_nodes or t in cfg.struct_nodes or t in cfg.interface_nodes:
            symbol_type = SymbolType.CLASS
            if t in cfg.interface_nodes:
                symbol_type = SymbolType.INTERFACE
            elif t in cfg.struct_nodes:
                symbol_type = SymbolType.STRUCT
            class_symbol = _make_symbol(
                node, state, symbol_type, parent_id,
                scope=Scope.GLOBAL if parent_id is None else Scope.CLASS,
            )
            if class_symbol is None:
                # Anonymous / unnamed class — recurse over children
                # without changing the parent_id, then exit this branch.
                # Without this explicit return, control would fall to
                # the default-recurse at the bottom of the function and
                # walk the same subtree twice (code-review CR1).
                for child in node.children:
                    self._visit_structural(child, state, parent_id)
                return

            state.symbols.append(class_symbol)
            # Inheritance edge — config gives us either a field name
            # or a set of child node types to look for.
            parent_node = None
            if cfg.inherit_field:
                parent_node = node.child_by_field_name(cfg.inherit_field)
            if parent_node is None and cfg.inherit_node_types:
                for child in node.children:
                    if child.type in cfg.inherit_node_types:
                        parent_node = child
                        break
            if parent_node is not None:
                parent_name = _read_node_text(parent_node, state.source_bytes).strip()
                # Strip language-specific prefix tokens (``extends ``,
                # ``: ``, ``< ``) — the inherit edge target is just
                # the parent type name.
                for prefix in ("extends ", "implements ", "<", ":"):
                    if parent_name.startswith(prefix):
                        parent_name = parent_name[len(prefix):].strip()
                if parent_name:
                    state.relationships.append(
                        Relationship(
                            source_symbol=class_symbol.get_qualified_name(),
                            target_symbol=parent_name,
                            relationship_type=RelationshipType.INHERITANCE,
                            source_file=state.file_path,
                            source_range=_node_range(parent_node),
                        )
                    )
            # Recurse over all children with this class as parent_id
            # so methods get it as their parent_symbol. We don't try
            # to narrow to a body field — walking everything is
            # cheap and robust across grammars whose class bodies
            # are different child shapes.
            for child in node.children:
                self._visit_structural(child, state, class_symbol.get_qualified_name())
            return

        # Enum. Recurse with the enum's own qualified name as parent_id
        # so any callable members (PHP 8.1 backed enum methods, Kotlin
        # ``enum class`` companion methods) attribute to the enum and
        # not to the surrounding scope. Rio R2.
        if t in cfg.enum_nodes:
            enum_symbol = _make_symbol(node, state, SymbolType.ENUM, parent_id)
            if enum_symbol is not None:
                state.symbols.append(enum_symbol)
                enum_parent = enum_symbol.get_qualified_name()
            else:
                enum_parent = parent_id
            for child in node.children:
                self._visit_structural(child, state, enum_parent)
            return

        # Callable declaration. The same node type may be a function at
        # top level and a method inside a class — the scope flips on
        # ``parent_id``. Grammars that DO distinguish (PHP) list both
        # in method_nodes + function_nodes.
        all_callable_nodes = set(cfg.function_nodes) | set(cfg.method_nodes)
        if t in all_callable_nodes:
            # If method_nodes is non-empty AND this node isn't in it,
            # treat it as function-only (PHP's function_definition
            # outside a class). If method_nodes IS this type AND we have
            # no parent, also treat as function. Otherwise default to
            # METHOD-if-nested, FUNCTION-if-top.
            is_method = (parent_id is not None) and (
                not cfg.method_nodes or t in cfg.method_nodes
            )
            sym_type = SymbolType.METHOD if is_method else SymbolType.FUNCTION
            scope = Scope.CLASS if is_method else Scope.GLOBAL
            fn_symbol = _make_symbol(node, state, sym_type, parent_id, scope=scope)
            if fn_symbol is None:
                # Anonymous / lambda-shaped callable — don't fall through
                # to the default recurse (would re-walk the same body
                # as a structural pass, double-counting nested
                # declarations). The body will still be walked once the
                # outer scope's calls pass runs.
                return

            state.symbols.append(fn_symbol)
            body = node.child_by_field_name(cfg.body_field)
            if body is None:
                # Some grammars don't expose body via a field — find
                # the largest child that looks like a block.
                for child in node.children:
                    if "body" in child.type or "block" in child.type or "statements" in child.type:
                        body = child
                        break
            if body is not None:
                state.bodies.append((fn_symbol.get_qualified_name(), body))
            # Don't recurse into the body during the structural pass.
            return

        # Default: recurse over children with the same parent_id.
        for child in node.children:
            self._visit_structural(child, state, parent_id)

    # ------------------------------------------------------------------
    # Calls pass — walks ONLY stashed function bodies, not the full tree
    # ------------------------------------------------------------------

    def _visit_calls(
        self,
        body: Node,
        state: _VisitState,
        containing_id: str,
    ) -> None:
        """Walk a single function/method body and emit ``CALLS`` edges.

        Dedup keyed on ``(containing_id, target, start_line)`` so a
        chained call like ``Greeter().greet("world")`` — which produces
        two nested call nodes in most grammars — emits two distinct
        edges (one to ``Greeter`` constructor, one to ``greet``) but
        the same call site walked twice doesn't double-count. The line
        component lets a function legitimately call the same target on
        two lines without dedup collapsing them.
        """
        cfg = self.config
        seen: set[tuple[str, str, int]] = set()
        stack = [body]
        while stack:
            node = stack.pop()
            if node.type in cfg.call_nodes:
                target_name = _extract_call_target(node, state.source_bytes)
                if target_name:
                    key = (containing_id, target_name, node.start_point[0])
                    if key not in seen:
                        seen.add(key)
                        state.relationships.append(
                            Relationship(
                                source_symbol=containing_id,
                                target_symbol=target_name,
                                relationship_type=RelationshipType.CALLS,
                                source_file=state.file_path,
                                source_range=_node_range(node),
                                confidence=0.6,  # name-only resolution
                            )
                        )
            stack.extend(node.children)


# ---------------------------------------------------------------------------
# Helpers — state, node text, symbol construction
# ---------------------------------------------------------------------------


@dataclass
class _VisitState:
    """Mutable per-file state for the visit passes."""

    file_path: str
    file_stem: str
    file_id: str
    source_bytes: bytes
    config: LanguageConfig

    symbols: list[Symbol] = field(default_factory=list)
    relationships: list[Relationship] = field(default_factory=list)
    imports: list[str] = field(default_factory=list)
    # (containing_id, body_node) tuples deferred for the calls pass.
    bodies: list[tuple[str, Any]] = field(default_factory=list)


def _read_node_text(node: Node, source_bytes: bytes) -> str:
    """Slice the source bytes for ``node``'s range and decode as UTF-8.

    Used over ``node.text.decode(...)`` because some tree-sitter bindings
    expose ``.text`` only when constructed with ``include_byte_ranges=True``
    and we want to be agnostic to the binding version.
    """
    return source_bytes[node.start_byte:node.end_byte].decode(
        "utf-8", errors="replace",
    )


def _node_range(node: Node) -> Range:
    start_row, start_col = node.start_point
    end_row, end_col = node.end_point
    return Range(
        start=Position(line=start_row + 1, column=start_col),
        end=Position(line=end_row + 1, column=end_col),
    )


def _make_symbol(
    node: Node,
    state: _VisitState,
    symbol_type: SymbolType,
    parent_id: str | None,
    scope: Scope = Scope.GLOBAL,
) -> Symbol | None:
    """Build a :class:`Symbol` from a declaration node.

    Three-step name resolution:

    1. Field lookup via ``cfg.name_field`` (default ``"name"``). Most
       modern grammars expose the identifier on a named field.
    2. Type-based fallback: first child whose type contains
       ``identifier`` / ``name`` OR appears in ``cfg.name_node_types``
       (Ruby's ``constant``, Kotlin's ``type_identifier``).
    3. Name-chain drill-down: if the matched node is in
       ``cfg.name_chain_types`` (Lua's ``method_index_expression``,
       PHP's ``qualified_name``), descend to its rightmost identifier-
       shaped descendant so ``Foo:bar`` resolves to ``bar``.

    Returns ``None`` if no name is found — we don't emit anonymous-
    symbol noise into the graph.
    """
    cfg = state.config
    name_node = node.child_by_field_name(cfg.name_field)
    if name_node is None:
        for child in node.children:
            if (
                "identifier" in child.type
                or "name" in child.type
                or child.type in cfg.name_node_types
            ):
                name_node = child
                break
    if name_node is None:
        return None

    # Name-chain drill-down: walk into the rightmost identifier-like
    # descendant. Handles colon-method syntax (Lua), namespaced names
    # (PHP), and member-access declarations.
    while name_node.type in cfg.name_chain_types:
        leaves = [
            c for c in name_node.children
            if c.is_named and (
                "identifier" in c.type
                or "name" in c.type
                or c.type in cfg.name_node_types
            )
        ]
        if not leaves:
            break
        name_node = leaves[-1]

    name = _read_node_text(name_node, state.source_bytes).strip()
    if not name:
        return None

    full_name = f"{parent_id}.{name}" if parent_id else name
    return Symbol(
        name=name,
        symbol_type=symbol_type,
        scope=scope,
        range=_node_range(node),
        file_path=state.file_path,
        parent_symbol=parent_id,
        full_name=full_name,
        source_text=_read_node_text(node, state.source_bytes),
    )


def _extract_call_target(node: Node, source_bytes: bytes) -> str | None:
    """Pull the callee identifier from a call node.

    Most grammars expose the function via either a ``function`` /
    ``method`` field or as the first identifier-shaped child. When the
    callee is a member access (``a.b.c()``), we use the rightmost name
    (``c``) — full path resolution lives in the deep parsers; this
    module is shallower by design.
    """
    fn_node = (
        node.child_by_field_name("function")
        or node.child_by_field_name("method")
        or node.child_by_field_name("callee")
        or node.child_by_field_name("name")
    )
    if fn_node is None:
        if not node.children:
            return None
        fn_node = node.children[0]

    text = source_bytes[fn_node.start_byte:fn_node.end_byte].decode(
        "utf-8", errors="replace",
    ).strip()
    if not text:
        return None
    # Reduce ``a.b.c`` → ``c`` for member calls; deep parsers handle the
    # full chain. Cover the standard member-access separators across
    # the supported languages: ``.`` (most), ``::`` (C++/Ruby/PHP),
    # ``:`` (Ruby symbols / Lua method index), ``->`` (PHP / Rust deref).
    # Order matters: longer separators first so ``::`` isn't shadowed by
    # ``:``.
    for sep in ("->", "::", ".", ":"):
        if sep in text:
            text = text.rsplit(sep, 1)[-1]
    return text or None
