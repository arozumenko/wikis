"""Bash / Shell config for :class:`BasicVisitorParser` (#119 phase 2).

Grammar reference: https://github.com/tree-sitter/tree-sitter-bash

Notes:
* Shell has no classes — only functions. The visitor emits each
  ``function_definition`` as a FUNCTION.
* Function names use the ``word`` node type, not ``identifier``;
  added to ``name_node_types``.
* Calls are ``command`` nodes; the command name lives in ``command_name``.
* ``source`` / ``.`` commands could mark imports, but they're regular
  commands at the AST level — left to the CALLS pass.
"""

from __future__ import annotations

from app.core.parsers.basic_visitor import LanguageConfig

BASH = LanguageConfig(
    name="bash",
    tree_sitter_name="bash",
    extensions=(".sh", ".bash", ".zsh"),
    function_nodes=("function_definition",),
    call_nodes=("command",),
    name_node_types=("word",),
)
