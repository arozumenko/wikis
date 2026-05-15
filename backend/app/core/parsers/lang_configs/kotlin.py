"""Kotlin config for :class:`BasicVisitorParser` (#119).

Grammar reference: https://github.com/fwcd/tree-sitter-kotlin

Notes:
* Kotlin uses ``function_declaration`` for both top-level functions
  and methods inside a class — the visitor's scope-flips-on-parent_id
  rule does the right thing.
* Class names are exposed as ``type_identifier`` (not on a ``name``
  field) — added to ``name_node_types``.
* Inheritance is on a ``delegation_specifier`` child (no field name).
* Supports ``.kt`` and ``.kts`` (script) extensions.
"""

from __future__ import annotations

from app.core.parsers.basic_visitor import LanguageConfig

KOTLIN = LanguageConfig(
    name="kotlin",
    tree_sitter_name="kotlin",
    extensions=(".kt", ".kts"),
    class_nodes=("class_declaration", "object_declaration"),
    interface_nodes=(),
    function_nodes=("function_declaration",),
    method_nodes=("function_declaration",),
    call_nodes=("call_expression",),
    import_nodes=("import_header",),
    inherit_node_types=("delegation_specifier",),
    name_node_types=("type_identifier", "simple_identifier"),
)
