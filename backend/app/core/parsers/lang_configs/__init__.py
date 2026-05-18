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

from app.core.parsers.lang_configs.bash import BASH
from app.core.parsers.lang_configs.dart import DART
from app.core.parsers.lang_configs.elixir import ELIXIR
from app.core.parsers.lang_configs.fortran import FORTRAN
from app.core.parsers.lang_configs.groovy import GROOVY
from app.core.parsers.lang_configs.julia import JULIA
from app.core.parsers.lang_configs.kotlin import KOTLIN
from app.core.parsers.lang_configs.lua import LUA
from app.core.parsers.lang_configs.objc import OBJC
from app.core.parsers.lang_configs.pascal import PASCAL
from app.core.parsers.lang_configs.php import PHP
from app.core.parsers.lang_configs.powershell import POWERSHELL
from app.core.parsers.lang_configs.r import R
from app.core.parsers.lang_configs.ruby import RUBY
from app.core.parsers.lang_configs.scala import SCALA
from app.core.parsers.lang_configs.swift import SWIFT
from app.core.parsers.lang_configs.verilog import VERILOG
from app.core.parsers.lang_configs.zig import ZIG

__all__ = [
    # Phase 1
    "KOTLIN", "LUA", "PHP", "RUBY", "SCALA",
    # Phase 2 — pure-config visitor fits
    "BASH", "DART", "FORTRAN", "JULIA", "OBJC", "PASCAL", "POWERSHELL",
    "SWIFT", "VERILOG",
    # Phase 2 — bespoke subclasses (see _special.py)
    "ELIXIR", "GROOVY", "R", "ZIG",
    "build_basic_parsers",
]


def build_basic_parsers() -> dict:
    """Construct one BasicVisitorParser (or bespoke subclass) per
    registered language. 18 languages across Phase 1 + Phase 2.

    Lazy import of the parser classes so this module stays usable in
    test code that wants to introspect the configs without spinning up
    tree-sitter parsers.
    """
    from app.core.parsers.basic_visitor import BasicVisitorParser
    from app.core.parsers.lang_configs._special import (
        ElixirParser,
        GroovyParser,
        RParser,
        ZigParser,
    )

    parsers: dict = {}
    # Pure-config languages — the generic BasicVisitorParser handles
    # them with only a LanguageConfig.
    pure_config_languages = (
        # Phase 1
        RUBY, PHP, KOTLIN, SCALA, LUA,
        # Phase 2 — clean fits
        SWIFT, DART, POWERSHELL, BASH, OBJC,
        VERILOG, FORTRAN, JULIA, PASCAL,
    )
    for cfg in pure_config_languages:
        parsers[cfg.name] = BasicVisitorParser(cfg)

    # Bespoke subclasses for languages whose grammar shape can't be
    # expressed via the config — see ``_special.py`` for the rationale
    # per language.
    parsers[ELIXIR.name] = ElixirParser(ELIXIR)
    parsers[R.name] = RParser(R)
    parsers[ZIG.name] = ZigParser(ZIG)
    parsers[GROOVY.name] = GroovyParser(GROOVY)
    return parsers
