"""Zig config (#119 phase 2).

Struct declarations are var-bindings to struct expressions
(``const X = struct { … }``); functions are ``FnProto`` nodes.
Custom logic lives in :class:`ZigParser`.
"""

from __future__ import annotations

from app.core.parsers.basic_visitor import LanguageConfig

ZIG = LanguageConfig(
    name="zig",
    tree_sitter_name="zig",
    extensions=(".zig",),
    function_nodes=("FnProto",),
    struct_nodes=("VarDecl",),  # discriminated by ZigParser
)
