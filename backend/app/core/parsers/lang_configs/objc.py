"""Objective-C config for :class:`BasicVisitorParser` (#119 phase 2).

Grammar reference: https://github.com/tree-sitter/tree-sitter-objc

Notes:
* Objective-C splits each class into two AST regions: a
  ``class_interface`` (``@interface`` block with ``method_declaration``
  child stubs) and a ``class_implementation`` (``@implementation`` block
  with ``method_definition`` children). Both emit CLASS — the resulting
  graph has two nodes per class, which is acceptable for shallow
  extraction.
* Method declarations exist in both interface (``method_declaration``)
  and implementation (``method_definition``) forms.
* Imports via ``preproc_include`` (``#import``).
"""

from __future__ import annotations

from app.core.parsers.basic_visitor import LanguageConfig

OBJC = LanguageConfig(
    name="objc",
    tree_sitter_name="objc",
    extensions=(".m", ".mm", ".h"),
    class_nodes=("class_interface", "class_implementation"),
    function_nodes=("function_definition",),
    method_nodes=("method_declaration", "method_definition"),
    call_nodes=("message_expression", "call_expression"),
    import_nodes=("preproc_include",),
)
