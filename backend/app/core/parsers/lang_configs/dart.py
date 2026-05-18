"""Dart config for :class:`BasicVisitorParser` (#119 phase 2).

Grammar reference: https://github.com/UserNobody14/tree-sitter-dart

Notes:
* ``class_definition`` for classes; ``mixin_declaration`` for mixins
  (treated as class for graph purposes).
* Methods are wrapped: ``method_signature`` contains ``function_signature``
  which holds the name. The visitor's children-recursion finds it.
* Inheritance via ``superclass`` child.
"""

from __future__ import annotations

from app.core.parsers.basic_visitor import LanguageConfig

DART = LanguageConfig(
    name="dart",
    tree_sitter_name="dart",
    extensions=(".dart",),
    class_nodes=("class_definition", "mixin_declaration"),
    enum_nodes=("enum_declaration",),
    function_nodes=("function_signature",),
    method_nodes=("function_signature",),
    call_nodes=("argument_part",),  # Dart wraps calls oddly; closest match
    import_nodes=("import_or_export", "library_import"),
    inherit_node_types=("superclass",),
    name_node_types=("type_identifier",),
)
