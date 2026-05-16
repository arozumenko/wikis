"""Elixir config (#119 phase 2).

Elixir's grammar parses every declaration as a generic ``call`` node;
the discriminating signal is the first identifier child's text
(``defmodule`` / ``def`` / ``defp`` / etc.). The pure-config visitor
can't express this — see :class:`ElixirParser` in
``lang_configs/_special.py`` for the override.

This config object exists for parity with the other languages so the
registry registration loop in ``__init__.py`` is uniform.
"""

from __future__ import annotations

from app.core.parsers.basic_visitor import LanguageConfig

ELIXIR = LanguageConfig(
    name="elixir",
    tree_sitter_name="elixir",
    extensions=(".ex", ".exs"),
    class_nodes=("call",),  # discriminated by ElixirParser
    function_nodes=("call",),
    call_nodes=("call",),
)
