"""R config (#119 phase 2).

R declarations are assignment expressions (``name <- function() { … }``).
Custom logic lives in :class:`RParser`.
"""

from __future__ import annotations

from app.core.parsers.basic_visitor import LanguageConfig

R = LanguageConfig(
    name="r",
    tree_sitter_name="r",
    extensions=(".r", ".R"),
    function_nodes=("function_definition",),
    call_nodes=("call",),
)
