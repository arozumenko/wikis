"""Lua config for :class:`BasicVisitorParser` (#119).

Grammar reference: https://github.com/tree-sitter/tree-sitter-lua

Notes on Lua's quirks:
* Lua has no classes — only functions. The visitor emits all callable
  declarations as ``FUNCTION``. "Method-like" calls (``Foo:bar()``)
  still get a CALLS edge.
* Function declarations use ``method_index_expression`` for the
  ``function Foo:bar()`` form. The name chain machinery drills into
  it to extract ``bar`` as the symbol name rather than ``Foo``.
* ``require`` is a function call, not a distinct node — no imports
  edge here for the same reason as Ruby.
* The ``local function name() … end`` form parses as a regular
  ``function_declaration`` with a plain identifier — already handled.
"""

from __future__ import annotations

from app.core.parsers.basic_visitor import LanguageConfig

LUA = LanguageConfig(
    name="lua",
    tree_sitter_name="lua",
    extensions=(".lua",),
    class_nodes=(),  # Lua has no class concept
    function_nodes=("function_declaration",),
    method_nodes=(),
    call_nodes=("function_call",),
    name_chain_types=("method_index_expression", "dot_index_expression"),
)
