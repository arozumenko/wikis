"""Julia config for :class:`BasicVisitorParser` (#119 phase 2).

Grammar reference: https://github.com/tree-sitter/tree-sitter-julia

Notes:
* Julia has no classes — ``struct_definition`` is the type abstraction.
  We emit it as STRUCT.
* Functions: ``function_definition`` for ``function name() ... end``
  forms; ``short_function_definition`` for ``f(x) = x`` assignments.
* The function name lives several levels deep:
  ``function_definition > signature > call_expression > identifier``.
  Adding ``signature`` and ``call_expression`` to ``name_chain_types``
  lets the chain-drill machinery find it.
* Imports: ``using_statement`` / ``import_statement``.
"""

from __future__ import annotations

from app.core.parsers.basic_visitor import LanguageConfig

JULIA = LanguageConfig(
    name="julia",
    tree_sitter_name="julia",
    extensions=(".jl",),
    struct_nodes=("struct_definition",),
    function_nodes=("function_definition", "short_function_definition"),
    method_nodes=("function_definition", "short_function_definition"),
    call_nodes=("call_expression",),
    import_nodes=("using_statement", "import_statement"),
    # Julia nests names deep: ``function_definition > signature >
    # call_expression > identifier`` and ``struct_definition >
    # type_head > identifier``.
    name_chain_types=("signature", "call_expression", "type_head"),
)
