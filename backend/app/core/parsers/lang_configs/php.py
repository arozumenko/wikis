"""PHP config for :class:`BasicVisitorParser` (#119).

Grammar reference: https://github.com/tree-sitter/tree-sitter-php

Notes:
* PHP has distinct nodes for class methods (``method_declaration``)
  and module-level functions (``function_definition``) — listed in
  separate fields so the visitor classifies them correctly even if
  someone defines a top-level function alongside classes.
* Inheritance is exposed as a ``base_clause`` child rather than a
  named field — using ``inherit_node_types`` instead of ``inherit_field``.
* Three call shapes — bare function calls, instance method calls
  (``$obj->m()``), and static / scoped calls (``Foo::bar()``) — all
  emit CALLS edges.
"""

from __future__ import annotations

from app.core.parsers.basic_visitor import LanguageConfig

PHP = LanguageConfig(
    name="php",
    tree_sitter_name="php",
    extensions=(".php",),
    class_nodes=("class_declaration",),
    interface_nodes=("interface_declaration",),
    enum_nodes=("enum_declaration",),
    function_nodes=("function_definition",),
    method_nodes=("method_declaration",),
    call_nodes=(
        "function_call_expression",
        "member_call_expression",
        "scoped_call_expression",
    ),
    import_nodes=("namespace_use_declaration",),
    inherit_node_types=("base_clause",),
    name_chain_types=("qualified_name",),
)
