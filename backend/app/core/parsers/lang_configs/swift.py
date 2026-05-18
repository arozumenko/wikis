"""Swift config for :class:`BasicVisitorParser` (#119 phase 2).

Grammar reference: https://github.com/alex-pinkus/tree-sitter-swift

Notes:
* Class + struct + protocol declarations all exist; we map them
  uniformly to CLASS / STRUCT / INTERFACE.
* ``inheritance_specifier`` is the child node carrying ``: Base``.
* Function bodies are in ``function_body``; same shape for methods.
"""

from __future__ import annotations

from app.core.parsers.basic_visitor import LanguageConfig

SWIFT = LanguageConfig(
    name="swift",
    tree_sitter_name="swift",
    extensions=(".swift",),
    class_nodes=("class_declaration", "protocol_declaration"),
    struct_nodes=("struct_declaration",),
    interface_nodes=("protocol_declaration",),
    enum_nodes=("enum_declaration",),
    function_nodes=("function_declaration",),
    method_nodes=("function_declaration",),
    call_nodes=("call_expression",),
    import_nodes=("import_declaration",),
    inherit_node_types=("inheritance_specifier",),
    name_node_types=("type_identifier", "simple_identifier"),
)
