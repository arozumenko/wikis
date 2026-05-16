"""Groovy config (#119 phase 2).

Groovy's tree-sitter grammar parses top-level declarations as loose
``command`` + ``unit`` token sequences rather than structured
declarations. Custom logic lives in :class:`GroovyParser`.
"""

from __future__ import annotations

from app.core.parsers.basic_visitor import LanguageConfig

GROOVY = LanguageConfig(
    name="groovy",
    tree_sitter_name="groovy",
    extensions=(".groovy", ".gvy", ".gradle"),
    class_nodes=("command",),  # discriminated by GroovyParser
)
