"""Per-language configs for :class:`BasicVisitorParser` (#119).

Each module here defines a ``LanguageConfig`` instance keyed by node-type
strings from the corresponding tree-sitter grammar. Adding a new language
is the small, repeatable work this package exists for:

1. Add ``<lang>.py`` with the ``LanguageConfig``.
2. Re-export from this ``__init__``.
3. Register a ``BasicVisitorParser(<lang>.CONFIG)`` in
   :class:`EnhancedUnifiedGraphBuilder` via the
   :func:`build_basic_parsers` factory below.

Fixture conventions: each language gets ``tests/fixtures/parsers/<lang>/``
with a ``hello.<ext>`` file exercising a class with one method that
calls another function — enough to verify CLASS / FUNCTION / METHOD /
CALLS / INHERITANCE wiring without needing a real-world repo's worth
of source.
"""

from __future__ import annotations

from app.core.parsers.lang_configs.kotlin import KOTLIN
from app.core.parsers.lang_configs.lua import LUA
from app.core.parsers.lang_configs.php import PHP
from app.core.parsers.lang_configs.ruby import RUBY
from app.core.parsers.lang_configs.scala import SCALA

__all__ = ["KOTLIN", "LUA", "PHP", "RUBY", "SCALA", "build_basic_parsers"]


def build_basic_parsers() -> dict:
    """Construct one :class:`BasicVisitorParser` per registered language.

    Lazy import of ``BasicVisitorParser`` so this module stays usable
    in test code that wants to introspect the configs without spinning
    up tree-sitter parsers.
    """
    from app.core.parsers.basic_visitor import BasicVisitorParser

    parsers: dict = {}
    for cfg in (RUBY, PHP, KOTLIN, SCALA, LUA):
        parser = BasicVisitorParser(cfg)
        parsers[cfg.name] = parser
    return parsers
