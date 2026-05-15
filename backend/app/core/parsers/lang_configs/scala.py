"""Scala config for :class:`BasicVisitorParser` (#119).

Grammar reference: https://github.com/tree-sitter/tree-sitter-scala

Notes:
* ``class_definition`` and ``object_definition`` both emit CLASS;
  Scala traits get ``trait_definition``.
* ``function_definition`` covers both top-level and method-scoped
  function-likes — scope-flips-on-parent_id classifies appropriately.
* Inheritance is on an ``extends_clause`` child.
* ``type_identifier`` is the name node for Scala (similar to Kotlin).
"""

from __future__ import annotations

from app.core.parsers.basic_visitor import LanguageConfig

SCALA = LanguageConfig(
    name="scala",
    tree_sitter_name="scala",
    extensions=(".scala", ".sc"),
    class_nodes=("class_definition", "object_definition"),
    interface_nodes=("trait_definition",),
    function_nodes=("function_definition",),
    method_nodes=("function_definition",),
    call_nodes=("call_expression",),
    import_nodes=("import_declaration",),
    inherit_node_types=("extends_clause",),
    name_node_types=("type_identifier",),
)
