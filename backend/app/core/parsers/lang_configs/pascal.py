"""Pascal / Delphi config for :class:`BasicVisitorParser` (#119 phase 2).

Grammar reference: https://github.com/Isopod/tree-sitter-pascal

Notes:
* Pascal mixes module-style (``program``) and OO (``class``) constructs.
  We treat ``declType`` (type declaration containing ``declClass``) as
  CLASS; ``program`` as a top-level container (also CLASS for graph
  purposes).
* Procedure / function bodies are ``defProc`` / ``defFunc``. Their
  names use ``genericDot`` for ``TFoo.Bar`` — added to
  ``name_chain_types`` so we get ``Bar``, not ``TFoo``.
* Uses-clause imports.
"""

from __future__ import annotations

from app.core.parsers.basic_visitor import LanguageConfig

PASCAL = LanguageConfig(
    name="pascal",
    tree_sitter_name="pascal",
    extensions=(".pas", ".pp", ".dpr", ".lpr"),
    class_nodes=("declType", "program"),
    function_nodes=("defProc", "defFunc"),
    method_nodes=("defProc", "defFunc"),
    call_nodes=("call_expression",),
    import_nodes=("declUses",),
    # defProc > declProc > genericDot > identifier (for ``TFoo.Bar``)
    # declProc carries the signature; genericDot is the dotted name;
    # we drill to the rightmost identifier (``Bar``).
    name_chain_types=("declProc", "declFunc", "genericDot"),
)
