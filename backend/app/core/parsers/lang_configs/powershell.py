"""PowerShell config for :class:`BasicVisitorParser` (#119 phase 2).

Grammar reference: https://github.com/airbus-cert/tree-sitter-powershell

Notes:
* PowerShell 5.0+ has full classes (``class_statement``) with
  ``class_method_definition`` and ``class_property_definition`` children.
* Function declarations use ``function_statement``.
* Imports via ``Import-Module`` are regular commands — no distinct
  node type, so we don't track them as IMPORT edges (calls cover them).
* ``simple_name`` carries the class / function name.
"""

from __future__ import annotations

from app.core.parsers.basic_visitor import LanguageConfig

POWERSHELL = LanguageConfig(
    name="powershell",
    tree_sitter_name="powershell",
    extensions=(".ps1", ".psm1", ".psd1"),
    class_nodes=("class_statement",),
    function_nodes=("function_statement",),
    method_nodes=("class_method_definition",),
    call_nodes=("invokation_expression", "invocation_expression", "command"),
    name_node_types=("simple_name",),
)
