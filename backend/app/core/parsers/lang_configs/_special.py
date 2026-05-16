"""Bespoke BasicVisitorParser subclasses for languages that don't fit
the pure-config pattern (#119 phase 2).

Four languages need overrides beyond what ``LanguageConfig`` can express:

* **Elixir** — every declaration is a ``call`` node; the discriminator
  is the text of the call's first identifier (``defmodule`` / ``def``
  / ``defp``). The pure-config visitor would either miss declarations
  (no class_nodes match) or treat every call as one.

* **R** — function declarations are ``binary_operator`` nodes with
  ``<-`` operator: ``name <- function() { … }``. The name is on the
  LHS, the function expression on the RHS. The pure-config visitor
  has no backward parent-lookup mechanism.

* **Zig** — struct declarations are ``VarDecl`` nodes whose RHS is a
  struct expression: ``const Greeter = struct { … }``. Detecting one
  requires inspecting the RHS, which the pure-config visitor doesn't do.

* **Groovy** — top-level constructs parse as ``command`` /
  ``unit`` token sequences (the tree-sitter-groovy grammar is loose).
  Class declarations appear as a sequence of unit tokens
  ``[class, Foo, extends, Bar]`` rather than a structured
  ``class_declaration`` node.

Each subclass keeps the two-pass design but overrides
``_visit_structural`` (and sometimes ``_visit_calls``) for its
language-specific shape. They share the symbol/relationship dataclasses
+ the slug-helper + the ParseResult shape, so downstream consumers
remain agnostic.

Scope reality: these four parsers extract less than the pure-config
ones. Elixir produces full class+method coverage; R/Zig/Groovy
extract class/function declarations only — no inheritance, no
attempt at call edges. That's a deliberate trade-off: a usable
shallow signal is better than nothing for these languages, and
upgrading to deep parsers is the documented promotion path.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from app.core.parsers.base_parser import Relationship, RelationshipType, Scope, Symbol, SymbolType
from app.core.parsers.basic_visitor import (
    BasicVisitorParser,
    LanguageConfig,
    _extract_call_target,
    _node_range,
    _read_node_text,
)

if TYPE_CHECKING:
    from tree_sitter import Node

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def _first_identifier_text(node, source_bytes: bytes) -> str | None:
    """Return the text of the first ``identifier``-type child of
    ``node``, or ``None`` if none exists. Used by the Elixir and R
    parsers to discriminate declaration calls from regular calls."""
    for c in node.children:
        if c.type == "identifier":
            return source_bytes[c.start_byte:c.end_byte].decode(
                "utf-8", errors="replace",
            ).strip()
    return None


def _emit_symbol(state, name, sym_type, parent_id, node, scope=Scope.GLOBAL):
    """Append a Symbol to ``state.symbols``. Helper shared across the
    bespoke parsers so they all produce the same shape as the pure-
    config visitor."""
    if not name:
        return None
    full_name = f"{parent_id}.{name}" if parent_id else name
    sym = Symbol(
        name=name,
        symbol_type=sym_type,
        scope=scope,
        range=_node_range(node),
        file_path=state.file_path,
        parent_symbol=parent_id,
        full_name=full_name,
        source_text=_read_node_text(node, state.source_bytes),
    )
    state.symbols.append(sym)
    return sym


# ---------------------------------------------------------------------------
# Elixir — discriminator on call.identifier text
# ---------------------------------------------------------------------------


_ELIXIR_FUNCTION_KEYWORDS = frozenset({"def", "defp", "defmacro", "defmacrop"})
_ELIXIR_CLASS_KEYWORDS = frozenset({"defmodule", "defprotocol", "defimpl"})


class ElixirParser(BasicVisitorParser):
    """Elixir: ``defmodule X do … end`` is a module (class-like).
    ``def name(args) do … end`` is a function. Both parse as ``call``
    nodes; we discriminate by the first identifier child's text.
    """

    def _visit_structural(self, node: "Node", state, parent_id: str | None) -> None:
        if node.type == "call":
            ident = _first_identifier_text(node, state.source_bytes)
            if ident in _ELIXIR_CLASS_KEYWORDS:
                name = self._elixir_module_name(node, state.source_bytes)
                sym = _emit_symbol(
                    state, name, SymbolType.CLASS, parent_id, node,
                    scope=Scope.GLOBAL if parent_id is None else Scope.CLASS,
                )
                if sym is not None:
                    do_block = self._elixir_do_block(node)
                    if do_block is not None:
                        for child in do_block.children:
                            self._visit_structural(
                                child, state, sym.get_qualified_name(),
                            )
                return
            if ident in _ELIXIR_FUNCTION_KEYWORDS:
                name = self._elixir_function_name(node, state.source_bytes)
                is_method = parent_id is not None
                sym = _emit_symbol(
                    state, name,
                    SymbolType.METHOD if is_method else SymbolType.FUNCTION,
                    parent_id, node,
                    scope=Scope.CLASS if is_method else Scope.GLOBAL,
                )
                if sym is not None:
                    do_block = self._elixir_do_block(node)
                    if do_block is not None:
                        state.bodies.append(
                            (sym.get_qualified_name(), do_block),
                        )
                return
        # Default: recurse over children without changing parent_id.
        for child in node.children:
            self._visit_structural(child, state, parent_id)

    def _visit_calls(self, body, state, containing_id: str) -> None:
        """Override the calls pass to skip declaration-shaped calls
        (``def`` / ``defmodule`` / etc.) — without this, every function
        body would emit a CALLS edge to ``def``."""
        seen: set[tuple[str, str, int]] = set()
        stack = [body]
        while stack:
            node = stack.pop()
            if node.type == "call":
                ident = _first_identifier_text(node, state.source_bytes)
                if ident not in (
                    _ELIXIR_FUNCTION_KEYWORDS | _ELIXIR_CLASS_KEYWORDS
                ):
                    target = _extract_call_target(node, state.source_bytes)
                    if target:
                        key = (containing_id, target, node.start_point[0])
                        if key not in seen:
                            seen.add(key)
                            state.relationships.append(
                                Relationship(
                                    source_symbol=containing_id,
                                    target_symbol=target,
                                    relationship_type=RelationshipType.CALLS,
                                    source_file=state.file_path,
                                    source_range=_node_range(node),
                                    confidence=0.6,
                                )
                            )
            stack.extend(node.children)

    @staticmethod
    def _elixir_module_name(call_node: "Node", source_bytes: bytes) -> str | None:
        # call → arguments → alias (the module name).
        for child in call_node.children:
            if child.type == "arguments":
                for sub in child.children:
                    if sub.type == "alias":
                        return source_bytes[sub.start_byte:sub.end_byte].decode(
                            "utf-8", errors="replace",
                        ).strip()
        return None

    @staticmethod
    def _elixir_function_name(call_node: "Node", source_bytes: bytes) -> str | None:
        # call → arguments → call (the signature, e.g. ``greet(name)``)
        # → identifier (the function name).
        for child in call_node.children:
            if child.type == "arguments":
                for sub in child.children:
                    if sub.type == "call":
                        for inner in sub.children:
                            if inner.type == "identifier":
                                return source_bytes[
                                    inner.start_byte:inner.end_byte
                                ].decode("utf-8", errors="replace").strip()
                    if sub.type == "identifier":
                        return source_bytes[
                            sub.start_byte:sub.end_byte
                        ].decode("utf-8", errors="replace").strip()
        return None

    @staticmethod
    def _elixir_do_block(call_node: "Node") -> "Node | None":
        for child in call_node.children:
            if child.type == "do_block":
                return child
        return None


# ---------------------------------------------------------------------------
# R — function/class declarations are LHS of binding operator
# ---------------------------------------------------------------------------


class RParser(BasicVisitorParser):
    """R: declarations look like ``name <- function() { … }`` or
    ``name <- setRefClass("X", ...)``. The visitor walks
    ``binary_operator`` nodes and inspects the RHS to decide.

    Coverage is intentionally minimal: function/class declarations
    only, no inheritance edges, no method-within-class extraction
    (R's S4/R6 classes use a different paradigm that would require
    deeper analysis). Sufficient for graph-level discoverability of
    where functions are defined.
    """

    def _visit_structural(self, node: "Node", state, parent_id: str | None) -> None:
        if node.type == "binary_operator":
            decl = self._r_binding_decl(node, state.source_bytes)
            if decl is not None:
                name, kind, body_node = decl
                sym = _emit_symbol(state, name, kind, parent_id, node)
                if sym is not None and body_node is not None and kind == SymbolType.FUNCTION:
                    state.bodies.append((sym.get_qualified_name(), body_node))
                return
        for child in node.children:
            self._visit_structural(child, state, parent_id)

    @staticmethod
    def _r_binding_decl(
        node: "Node", source_bytes: bytes,
    ) -> "tuple[str, SymbolType, Node | None] | None":
        # binary_operator → identifier (LHS name) + operator-token +
        # function_definition / call (RHS). Rio R2 caught that R uses
        # ``binary_operator`` for every binary expression (``<``, ``==``,
        # ``+``, ``->``, etc.), not just assignment. Without gating on
        # the operator-token, comparisons like ``x < some_call()`` would
        # emit spurious declarations.
        #
        # Only these operators are R bindings that declare a new name:
        # ``<-`` (standard assign), ``<<-`` (super-assign), ``=``
        # (assignment form). ``->`` and ``->>`` (right-assign) put the
        # name on the RIGHT side; the current LHS-name extraction would
        # name the symbol after the function expression — out of scope
        # for Phase 2, deferred to a future refinement.
        _R_ASSIGN_OPS = frozenset({"<-", "<<-", "="})

        # Note: only direct ``identifier`` children become LHS names.
        # ``obj$method <- function()`` (member assignment) parses with
        # the LHS as an ``extract_operator`` node, not an ``identifier``,
        # so it's silently dropped at this tier — acceptable miss
        # documented in the docstring; deep parser handles it.
        lhs_name = None
        rhs = None
        op_text: str | None = None
        for child in node.children:
            if child.type == "identifier" and lhs_name is None:
                lhs_name = source_bytes[child.start_byte:child.end_byte].decode(
                    "utf-8", errors="replace",
                ).strip()
            elif child.type in ("function_definition", "call"):
                rhs = child
            else:
                # Anonymous-token operator child. Anything that isn't
                # an identifier or one of the expected RHS types is a
                # candidate operator token.
                text = source_bytes[child.start_byte:child.end_byte].decode(
                    "utf-8", errors="replace",
                ).strip()
                if text in _R_ASSIGN_OPS:
                    op_text = text

        if lhs_name is None or rhs is None or op_text is None:
            return None
        if rhs.type == "function_definition":
            body = None
            for sub in rhs.children:
                if sub.type in ("braced_expression", "block"):
                    body = sub
                    break
            return (lhs_name, SymbolType.FUNCTION, body)
        # call RHS — only emit as class when the call target is one of
        # R's known class-constructor functions (setRefClass, R6Class,
        # setClass). Other calls are just regular assignments.
        for sub in rhs.children:
            if sub.type == "identifier":
                fn_name = source_bytes[sub.start_byte:sub.end_byte].decode(
                    "utf-8", errors="replace",
                ).strip()
                if fn_name in ("setRefClass", "R6Class", "setClass"):
                    return (lhs_name, SymbolType.CLASS, None)
                break
        return None


# ---------------------------------------------------------------------------
# Zig — struct decls via VarDecl with struct RHS; fns via FnProto
# ---------------------------------------------------------------------------


class ZigParser(BasicVisitorParser):
    """Zig: ``const Greeter = struct { … }`` declares a struct (CLASS);
    ``fn name() { … }`` declares a function. Methods inside a struct
    aren't tracked here — Zig's struct-method scoping uses the same
    FnProto node as top-level functions, and disambiguating requires
    walking the var-decl context (deferrable to a deep parser).

    Known limitations (Rio R2):

    * Struct detection uses a **text-prefix heuristic** on the
      ``ErrorUnionExpr`` child's source bytes (``startswith
      "struct"/"enum"/"union"``). This misses:

      - ``const Foo = packed struct { … }`` (text starts with
        ``packed``)
      - ``const Foo = extern struct { … }`` (text starts with
        ``extern``)
      - ``const Foo = comptime blk: { break :blk struct { … }; }``
        (text starts with ``comptime``)
      - Error-union types wrapping a struct (``const Foo =
        error{X}!struct {…}``)

      The bare ``struct {}`` / ``enum {}`` / ``union {}`` form does
      work, which covers the common case. A future refinement could
      walk the RHS for any descendant ``struct``/``enum``/``union``
      keyword token instead of prefix-matching the surface text.
    """

    def _visit_structural(self, node: "Node", state, parent_id: str | None) -> None:
        if node.type == "VarDecl":
            decl = self._zig_var_decl(node, state.source_bytes)
            if decl is not None:
                name, is_struct = decl
                _emit_symbol(
                    state, name,
                    SymbolType.STRUCT if is_struct else SymbolType.VARIABLE,
                    parent_id, node,
                )
                # Don't recurse into the var's RHS — nested struct fields
                # aren't tracked.
                return
        if node.type == "FnProto":
            name = self._zig_fn_name(node, state.source_bytes)
            _emit_symbol(state, name, SymbolType.FUNCTION, parent_id, node)
            # Don't recurse into the body — basic shallow extraction.
            return
        for child in node.children:
            self._visit_structural(child, state, parent_id)

    @staticmethod
    def _zig_var_decl(
        node: "Node", source_bytes: bytes,
    ) -> "tuple[str, bool] | None":
        name = None
        is_struct = False
        for child in node.children:
            if child.type == "IDENTIFIER" and name is None:
                name = source_bytes[child.start_byte:child.end_byte].decode(
                    "utf-8", errors="replace",
                ).strip()
            elif child.type == "ErrorUnionExpr":
                # Look for a ``struct``/``enum``/``union`` keyword
                # token anywhere in the first ~80 chars of the
                # expression. Tokenising on whitespace handles common
                # modifier-prefixed forms (``packed struct {…}``,
                # ``extern struct {…}``, ``comptime blk: { break :blk
                # struct {…} }``) that a naive ``startswith`` misses
                # (code-review I1, Rio R2).
                #
                # Brace-attached forms like ``struct{`` are also
                # caught because we split on whitespace AND rstrip
                # the open-brace before comparing.
                text = source_bytes[child.start_byte:child.end_byte].decode(
                    "utf-8", errors="replace",
                ).strip()
                tokens = text[:80].split()
                if any(
                    tok.rstrip("{(,") in ("struct", "enum", "union")
                    for tok in tokens
                ):
                    is_struct = True
        if name is None:
            return None
        return (name, is_struct)

    @staticmethod
    def _zig_fn_name(node: "Node", source_bytes: bytes) -> str | None:
        for child in node.children:
            if child.type == "IDENTIFIER":
                return source_bytes[child.start_byte:child.end_byte].decode(
                    "utf-8", errors="replace",
                ).strip()
        return None


# ---------------------------------------------------------------------------
# Groovy — token-sequence scan over commands
# ---------------------------------------------------------------------------


class GroovyParser(BasicVisitorParser):
    """Groovy: tree-sitter-groovy parses constructs as deeply-nested
    ``command`` / ``unit`` / ``block`` sequences. A class declaration
    is ``command(unit "class", block(unit <Name>, { … methods … }))``
    where each method is itself a ``command(unit <return-type>,
    block(unit <name+params>, { … body … }))``.

    Coverage: top-level classes + their methods. No inheritance,
    no call edges (the call sites are syntactically indistinguishable
    from regular statements in this grammar — would need text-pattern
    matching that's deeper than this tier promises).
    """

    def _visit_structural(self, node: "Node", state, parent_id: str | None) -> None:
        if node.type == "command":
            cls_decl = self._groovy_class_decl(node, state.source_bytes)
            if cls_decl is not None:
                name, body_block = cls_decl
                sym = _emit_symbol(
                    state, name, SymbolType.CLASS, parent_id, node,
                )
                if sym is not None and body_block is not None:
                    for child in body_block.children:
                        self._visit_structural(
                            child, state, sym.get_qualified_name(),
                        )
                return
            # Method-shape pattern: a command inside a class body
            # whose units look like ``[<type>, <name+params>]`` — the
            # first unit is the return type, the inner block's first
            # unit is the method name with parameters.
            if parent_id is not None:
                method_name = self._groovy_method_name(node, state.source_bytes)
                if method_name is not None:
                    _emit_symbol(
                        state, method_name, SymbolType.METHOD, parent_id,
                        node, scope=Scope.CLASS,
                    )
                    return
        for child in node.children:
            self._visit_structural(child, state, parent_id)

    @staticmethod
    def _groovy_class_decl(
        node: "Node", source_bytes: bytes,
    ) -> "tuple[str, Node | None] | None":
        """Match ``command(unit "class", block(unit <Name>, …))`` and
        return (class_name, body_block)."""
        kw_unit = None
        outer_block = None
        for child in node.children:
            if child.type == "unit" and kw_unit is None:
                text = source_bytes[child.start_byte:child.end_byte].decode(
                    "utf-8", errors="replace",
                ).strip()
                if text == "class":
                    kw_unit = child
                else:
                    return None
            elif child.type == "block" and kw_unit is not None:
                outer_block = child
                break
        if kw_unit is None or outer_block is None:
            return None
        # The class name is the first ``unit`` child inside the block.
        name = None
        for child in outer_block.children:
            if child.type == "unit":
                name = source_bytes[child.start_byte:child.end_byte].decode(
                    "utf-8", errors="replace",
                ).strip()
                break
        if not name:
            return None
        return (name, outer_block)

    @staticmethod
    def _groovy_method_name(
        node: "Node", source_bytes: bytes,
    ) -> str | None:
        """Match ``command(unit <type>, block(unit <name+params>, …))``
        inside a class body and return just the method name (text
        before the first paren)."""
        type_unit = None
        body_block = None
        for child in node.children:
            if child.type == "unit" and type_unit is None:
                type_unit = child
            elif child.type == "block" and type_unit is not None:
                body_block = child
                break
        if type_unit is None or body_block is None:
            return None
        # First unit inside the body block is the name+params signature.
        for child in body_block.children:
            if child.type == "unit":
                text = source_bytes[child.start_byte:child.end_byte].decode(
                    "utf-8", errors="replace",
                ).strip()
                # Strip params: ``greet(String name)`` → ``greet``.
                if "(" in text:
                    text = text.split("(", 1)[0].strip()
                return text or None
        return None
