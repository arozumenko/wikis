"""Ruby config for :class:`BasicVisitorParser` (#119).

Grammar reference: https://github.com/tree-sitter/tree-sitter-ruby

Notes on Ruby specifics worth knowing for future config edits:
* Class names are exposed as a ``constant`` child, not via a ``name``
  field — hence ``name_node_types=("constant",)``.
* Method bodies use the type ``body_statement`` (different from most
  other grammars' ``body``).
* Ruby has no module-level ``function`` distinct from ``method`` — the
  visitor uses ``function_nodes=method_nodes`` so a top-level ``def``
  emits as a ``FUNCTION`` while a ``def`` inside a class emits as a
  ``METHOD``.
* ``require`` is a regular method call, not a distinct AST node, so
  imports aren't captured as a separate edge — the call edge to
  ``require`` is enough for graph-level signals.
"""

from __future__ import annotations

from app.core.parsers.basic_visitor import LanguageConfig

RUBY = LanguageConfig(
    name="ruby",
    tree_sitter_name="ruby",
    extensions=(".rb",),
    class_nodes=("class", "module"),
    function_nodes=("method", "singleton_method"),
    method_nodes=("method", "singleton_method"),
    call_nodes=("call",),
    inherit_field="superclass",
    body_field="body_statement",
    name_node_types=("constant",),  # class Foo → Foo is a constant
)
