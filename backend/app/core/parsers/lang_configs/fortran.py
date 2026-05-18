"""Fortran config for :class:`BasicVisitorParser` (#119 phase 2).

Grammar reference: https://github.com/stadelmanma/tree-sitter-fortran

Notes:
* ``module`` is the class-like abstraction. ``subroutine`` and
  ``function`` are the callable units.
* Modules contain an ``internal_procedures`` section (``contains``
  keyword) under which subroutines / functions live. The visitor
  recurses naturally.
* ``subroutine_call`` is the call-edge source.
* Imports: ``use_statement``.
"""

from __future__ import annotations

from app.core.parsers.basic_visitor import LanguageConfig

FORTRAN = LanguageConfig(
    name="fortran",
    tree_sitter_name="fortran",
    extensions=(".f", ".for", ".f90", ".f95", ".f03", ".f08"),
    class_nodes=("module", "program"),
    function_nodes=("subroutine", "function"),
    method_nodes=("subroutine", "function"),
    call_nodes=("subroutine_call", "call_expression"),
    import_nodes=("use_statement",),
    # module > module_statement > name
    # subroutine > subroutine_statement > name
    name_chain_types=("module_statement", "subroutine_statement", "function_statement"),
)
