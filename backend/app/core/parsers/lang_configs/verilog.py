"""Verilog / SystemVerilog config for :class:`BasicVisitorParser` (#119 phase 2).

Grammar reference: https://github.com/tree-sitter/tree-sitter-verilog

Notes:
* Verilog is HDL — its abstraction units are ``module``, ``task``,
  and ``function``. We map ``module_declaration`` to CLASS (the module
  is the encapsulation boundary), and ``task_declaration`` /
  ``function_declaration`` to FUNCTION.
* SystemVerilog adds ``class_declaration`` for proper OO support;
  also captured.
* No imports in basic Verilog; SystemVerilog has ``package_import_declaration``.
"""

from __future__ import annotations

from app.core.parsers.basic_visitor import LanguageConfig

VERILOG = LanguageConfig(
    name="verilog",
    tree_sitter_name="verilog",
    extensions=(".v", ".vh", ".sv", ".svh"),
    class_nodes=("module_declaration", "class_declaration"),
    interface_nodes=("interface_declaration",),
    function_nodes=("task_declaration", "function_declaration"),
    method_nodes=("task_declaration", "function_declaration"),
    call_nodes=("task_enable_statement", "function_subroutine_call"),
    import_nodes=("package_import_declaration",),
    name_node_types=("simple_identifier",),
    # module_declaration > module_header > simple_identifier
    # task_declaration > task_body_declaration > task_identifier > task_identifier > simple_identifier
    name_chain_types=("module_header", "task_body_declaration", "task_identifier"),
)
