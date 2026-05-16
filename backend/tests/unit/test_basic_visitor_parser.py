"""Tests for :class:`BasicVisitorParser` (#119).

Drives the parser over fixture files for all five Phase-1 languages and
asserts the per-language config produces the expected symbol kinds +
relationships. Each language fixture follows the same shape so the same
assertions catch regressions across configs:

* one class ``Greeter`` with two methods ``greet`` + ``format_name``
  (or the language's closest equivalent — Lua has no class concept and
  emits both as top-level functions)
* one top-level function ``standalone_helper`` (or ``run`` for Scala
  where it lives in an ``object Main``)
* method-to-method call (``greet`` → ``format_name``)
* inheritance edge (``Greeter`` → ``Base``) — for languages that have
  classes
* an import / require statement

The fixtures live under ``backend/tests/fixtures/parsers/<lang>/hello.<ext>``.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from app.core.parsers.base_parser import (
    RelationshipType,
    SymbolType,
)
from app.core.parsers.basic_visitor import (
    BasicVisitorParser,
    LanguageConfig,
    make_id,
)
from app.core.parsers.lang_configs import (
    BASH,
    DART,
    ELIXIR,
    FORTRAN,
    GROOVY,
    JULIA,
    KOTLIN,
    LUA,
    OBJC,
    PASCAL,
    PHP,
    POWERSHELL,
    R,
    RUBY,
    SCALA,
    SWIFT,
    VERILOG,
    ZIG,
    build_basic_parsers,
)
from app.core.parsers.lang_configs._special import (
    ElixirParser,
    GroovyParser,
    RParser,
    ZigParser,
)

FIXTURE_ROOT = Path(__file__).parent.parent / "fixtures" / "parsers"


# ---------------------------------------------------------------------------
# Helpers + shared assertions
# ---------------------------------------------------------------------------


def _names_by_type(result, symbol_type: SymbolType) -> set[str]:
    return {s.name for s in result.symbols if s.symbol_type == symbol_type}


def _symbol(result, name: str):
    """Look up a symbol by name. Returns the first match or None."""
    for s in result.symbols:
        if s.name == name:
            return s
    return None


def _call_pairs(result) -> set[tuple[str, str]]:
    """Set of (source, target) for every CALLS edge — useful for
    asserting on shape without caring about the call-site line."""
    return {
        (r.source_symbol, r.target_symbol)
        for r in result.relationships
        if r.relationship_type == RelationshipType.CALLS
    }


def _inherit_pairs(result) -> set[tuple[str, str]]:
    return {
        (r.source_symbol, r.target_symbol)
        for r in result.relationships
        if r.relationship_type == RelationshipType.INHERITANCE
    }


def _assert_positions_populated(result) -> None:
    """Every emitted symbol must have a non-trivial range. Catches
    regressions where positions all collapse to (0, 0) (code-review
    CR2: name-only assertions wouldn't notice this kind of bug)."""
    assert result.symbols, "no symbols emitted — separate failure"
    for s in result.symbols:
        assert s.range.start.line >= 1, (
            f"symbol {s.name!r} has start.line={s.range.start.line} "
            f"— expected 1-indexed line, got 0 or negative"
        )
        assert s.range.end.line >= s.range.start.line, (
            f"symbol {s.name!r} has end before start "
            f"({s.range.start.line}..{s.range.end.line})"
        )


# ---------------------------------------------------------------------------
# Foundation tests
# ---------------------------------------------------------------------------


class TestMakeId:
    def test_basic_slug(self) -> None:
        assert make_id("Foo") == "foo"

    def test_separator_collapse(self) -> None:
        assert make_id("Foo Bar", "Baz") == "foo_bar_baz"

    def test_strips_leading_trailing(self) -> None:
        assert make_id("__Foo__", ".Bar.") == "foo_bar"

    def test_drops_empty_parts(self) -> None:
        assert make_id("foo", "", "bar") == "foo_bar"

    def test_non_alphanumeric_collapsed(self) -> None:
        assert make_id("a.b.c", "x/y/z") == "a_b_c_x_y_z"


_EXPECTED_LANGS = frozenset({
    # Phase 1
    "ruby", "php", "kotlin", "scala", "lua",
    # Phase 2 — pure-config
    "swift", "dart", "powershell", "bash", "objc",
    "verilog", "fortran", "julia", "pascal",
    # Phase 2 — bespoke subclasses
    "elixir", "r", "zig", "groovy",
})


class TestRegistry:
    def test_build_basic_parsers_returns_all_18_languages(self) -> None:
        parsers = build_basic_parsers()
        assert set(parsers.keys()) == _EXPECTED_LANGS
        for name, parser in parsers.items():
            assert isinstance(parser, BasicVisitorParser)
            assert parser.config.name == name

    def test_basic_visitor_langs_sync_with_factory(self) -> None:
        """#151: the literal ``_BASIC_VISITOR_LANGS`` tuple in
        ``graph_builder.py`` is hand-maintained so it can serve as a
        lost-language diagnostic even when ``build_basic_parsers()``
        fails to import. This test catches drift: a contributor adds
        a new language to the factory but forgets the literal (or
        vice versa). Failure here turns a silent omission in the
        diagnostic into a CI break."""
        from app.core.code_graph.graph_builder import _BASIC_VISITOR_LANGS

        factory_keys = set(build_basic_parsers().keys())
        literal_set = set(_BASIC_VISITOR_LANGS)
        assert factory_keys == literal_set, (
            f"build_basic_parsers() returns {sorted(factory_keys)} but "
            f"_BASIC_VISITOR_LANGS lists {sorted(literal_set)} — keep "
            f"them in sync to avoid silent omission from the diagnostic"
        )

    def test_each_parser_reports_its_extensions(self) -> None:
        parsers = build_basic_parsers()
        assert ".rb" in parsers["ruby"]._get_supported_extensions()
        assert ".php" in parsers["php"]._get_supported_extensions()
        assert ".kt" in parsers["kotlin"]._get_supported_extensions()
        assert ".scala" in parsers["scala"]._get_supported_extensions()
        assert ".lua" in parsers["lua"]._get_supported_extensions()

    def test_each_parser_advertises_its_capabilities(self) -> None:
        parsers = build_basic_parsers()
        # Languages with classes should report so; Lua + Bash should not.
        assert parsers["ruby"].capabilities.has_classes is True
        assert parsers["php"].capabilities.has_classes is True
        assert parsers["swift"].capabilities.has_classes is True
        assert parsers["lua"].capabilities.has_classes is False
        assert parsers["bash"].capabilities.has_classes is False

    def test_bespoke_subclasses_used_for_quirky_grammars(self) -> None:
        """Elixir/R/Zig/Groovy use BasicVisitorParser subclasses.
        Verify the registry hands out the right instance type."""
        parsers = build_basic_parsers()
        assert isinstance(parsers["elixir"], ElixirParser)
        assert isinstance(parsers["r"], RParser)
        assert isinstance(parsers["zig"], ZigParser)
        assert isinstance(parsers["groovy"], GroovyParser)
        # All bespoke parsers still inherit from BasicVisitorParser so
        # they slot into the dispatch path uniformly.
        for name in ("elixir", "r", "zig", "groovy"):
            assert isinstance(parsers[name], BasicVisitorParser)


class TestUnsupportedConfig:
    def test_parser_with_bad_tree_sitter_name_returns_empty_result(
        self, tmp_path: Path,
    ) -> None:
        """When ``get_parser(...)`` fails (missing grammar), the parser
        must construct in a degraded state and return an empty
        ``ParseResult`` from ``parse_file`` rather than raising. Lets
        partial-environment deploys (e.g. someone missing a tree-sitter
        binding for one language) still index the rest of the repo."""
        bogus = LanguageConfig(
            name="bogus",
            tree_sitter_name="does-not-exist",
            extensions=(".bogus",),
            function_nodes=("function_definition",),
        )
        parser = BasicVisitorParser(bogus)
        assert parser._parser is None

        f = tmp_path / "x.bogus"
        f.write_text("anything")
        result = parser.parse_file(f)
        assert result.symbols == []
        assert result.errors  # explanatory error message present


# ---------------------------------------------------------------------------
# Per-language fixture tests
# ---------------------------------------------------------------------------


class TestRuby:
    @pytest.fixture
    def result(self):
        parser = BasicVisitorParser(RUBY)
        return parser.parse_file(FIXTURE_ROOT / "ruby" / "hello.rb")

    def test_class_emitted(self, result) -> None:
        assert "Greeter" in _names_by_type(result, SymbolType.CLASS)

    def test_methods_emitted(self, result) -> None:
        methods = _names_by_type(result, SymbolType.METHOD)
        assert "greet" in methods
        assert "format_name" in methods

    def test_top_level_function_emitted(self, result) -> None:
        # Ruby has no separate "function" — but a top-level ``def`` should
        # land as FUNCTION via the scope-flip rule.
        assert "standalone_helper" in _names_by_type(result, SymbolType.FUNCTION)

    def test_inheritance_edge(self, result) -> None:
        pairs = _inherit_pairs(result)
        assert any(src == "Greeter" and tgt == "Base" for src, tgt in pairs)

    def test_method_to_method_call(self, result) -> None:
        # greet() calls format_name() — confidence is name-based but the
        # CALLS edge must be present with the right source qualifier.
        pairs = _call_pairs(result)
        assert any(
            src.endswith("greet") and tgt == "format_name"
            for src, tgt in pairs
        )

    def test_symbol_positions_populated(self, result) -> None:
        _assert_positions_populated(result)
        # ``Greeter`` class is declared on line 3 in the fixture.
        greeter = _symbol(result, "Greeter")
        assert greeter is not None
        assert greeter.range.start.line == 3


class TestPHP:
    @pytest.fixture
    def result(self):
        parser = BasicVisitorParser(PHP)
        return parser.parse_file(FIXTURE_ROOT / "php" / "hello.php")

    def test_class_emitted(self, result) -> None:
        assert "Greeter" in _names_by_type(result, SymbolType.CLASS)

    def test_methods_emitted(self, result) -> None:
        methods = _names_by_type(result, SymbolType.METHOD)
        assert "greet" in methods
        assert "formatName" in methods

    def test_function_emitted(self, result) -> None:
        assert "standaloneHelper" in _names_by_type(result, SymbolType.FUNCTION)

    def test_inheritance_edge(self, result) -> None:
        pairs = _inherit_pairs(result)
        assert any(src == "Greeter" and "Base" in tgt for src, tgt in pairs)

    def test_imports_recorded(self, result) -> None:
        # ``use Other\Base`` should emit an IMPORTS edge from the file.
        imports = [
            r for r in result.relationships
            if r.relationship_type == RelationshipType.IMPORTS
        ]
        assert imports

    def test_method_to_method_call(self, result) -> None:
        # $this->formatName() — member call.
        pairs = _call_pairs(result)
        assert any(
            "greet" in src and tgt == "formatName"
            for src, tgt in pairs
        )

    def test_symbol_positions_populated(self, result) -> None:
        _assert_positions_populated(result)
        greeter = _symbol(result, "Greeter")
        assert greeter is not None
        # ``Greeter`` class is declared on line 7 in the fixture
        # (after ``<?php``, blank line, namespace, blank, use, blank).
        assert greeter.range.start.line == 7


class TestKotlin:
    @pytest.fixture
    def result(self):
        parser = BasicVisitorParser(KOTLIN)
        return parser.parse_file(FIXTURE_ROOT / "kotlin" / "hello.kt")

    def test_class_emitted(self, result) -> None:
        assert "Greeter" in _names_by_type(result, SymbolType.CLASS)

    def test_methods_and_top_level_function(self, result) -> None:
        methods = _names_by_type(result, SymbolType.METHOD)
        functions = _names_by_type(result, SymbolType.FUNCTION)
        assert "greet" in methods
        assert "formatName" in methods
        assert "standaloneHelper" in functions

    def test_method_to_method_call(self, result) -> None:
        pairs = _call_pairs(result)
        assert any(
            "greet" in src and tgt == "formatName"
            for src, tgt in pairs
        )

    def test_symbol_positions_populated(self, result) -> None:
        _assert_positions_populated(result)
        greeter = _symbol(result, "Greeter")
        assert greeter is not None
        # ``class Greeter`` is declared on line 5 in the fixture.
        assert greeter.range.start.line == 5


class TestScala:
    @pytest.fixture
    def result(self):
        parser = BasicVisitorParser(SCALA)
        return parser.parse_file(FIXTURE_ROOT / "scala" / "hello.scala")

    def test_class_and_object_emitted(self, result) -> None:
        classes = _names_by_type(result, SymbolType.CLASS)
        # Both class_definition AND object_definition map to CLASS.
        assert "Greeter" in classes
        assert "Main" in classes

    def test_methods_emitted(self, result) -> None:
        methods = _names_by_type(result, SymbolType.METHOD)
        assert "greet" in methods
        assert "formatName" in methods
        assert "run" in methods

    def test_method_to_method_call(self, result) -> None:
        pairs = _call_pairs(result)
        assert any(
            "greet" in src and tgt == "formatName"
            for src, tgt in pairs
        )

    def test_symbol_positions_populated(self, result) -> None:
        _assert_positions_populated(result)
        greeter = _symbol(result, "Greeter")
        assert greeter is not None
        # ``class Greeter`` is on line 5 in the fixture.
        assert greeter.range.start.line == 5


class TestLua:
    @pytest.fixture
    def result(self):
        parser = BasicVisitorParser(LUA)
        return parser.parse_file(FIXTURE_ROOT / "lua" / "hello.lua")

    def test_no_classes_emitted(self, result) -> None:
        # Lua has no class concept in the config.
        assert _names_by_type(result, SymbolType.CLASS) == set()

    def test_functions_use_chain_name(self, result) -> None:
        # The ``function Greeter:greet(...)`` form has its name on a
        # ``method_index_expression`` chain — drill-down should yield
        # ``greet``, not ``Greeter``.
        functions = _names_by_type(result, SymbolType.FUNCTION)
        assert "greet" in functions
        assert "format_name" in functions
        assert "standalone_helper" in functions

    def test_call_edge_to_chain_callee(self, result) -> None:
        # ``self:format_name(name)`` is a chain call; we drill to the
        # last name (``format_name``).
        pairs = _call_pairs(result)
        assert any(tgt == "format_name" for _, tgt in pairs)

    def test_symbol_positions_populated(self, result) -> None:
        _assert_positions_populated(result)
        greet = _symbol(result, "greet")
        assert greet is not None
        # ``function Greeter:greet`` is on line 5 in the fixture.
        assert greet.range.start.line == 5


# ---------------------------------------------------------------------------
# Phase 2 — pure-config languages (Swift, Dart, PowerShell, Bash, Obj-C,
# Verilog, Fortran, Julia, Pascal)
# ---------------------------------------------------------------------------


def _basic_result(cfg, fixture_path: Path):
    return BasicVisitorParser(cfg).parse_file(fixture_path)


class TestSwift:
    @pytest.fixture
    def result(self):
        return _basic_result(SWIFT, FIXTURE_ROOT / "swift" / "hello.swift")

    def test_class_and_methods(self, result) -> None:
        assert "Greeter" in _names_by_type(result, SymbolType.CLASS)
        methods = _names_by_type(result, SymbolType.METHOD)
        assert "greet" in methods and "formatName" in methods

    def test_top_level_function(self, result) -> None:
        assert "standaloneHelper" in _names_by_type(result, SymbolType.FUNCTION)

    def test_inheritance(self, result) -> None:
        assert any(
            src == "Greeter" and tgt == "Base"
            for src, tgt in _inherit_pairs(result)
        )

    def test_positions(self, result) -> None:
        _assert_positions_populated(result)


class TestDart:
    @pytest.fixture
    def result(self):
        return _basic_result(DART, FIXTURE_ROOT / "dart" / "hello.dart")

    def test_class_and_methods(self, result) -> None:
        assert "Greeter" in _names_by_type(result, SymbolType.CLASS)
        methods = _names_by_type(result, SymbolType.METHOD)
        assert "greet" in methods and "formatName" in methods

    def test_inheritance(self, result) -> None:
        assert any(
            src == "Greeter" and "Base" in tgt
            for src, tgt in _inherit_pairs(result)
        )

    def test_positions(self, result) -> None:
        _assert_positions_populated(result)


class TestPowerShell:
    @pytest.fixture
    def result(self):
        return _basic_result(POWERSHELL, FIXTURE_ROOT / "powershell" / "hello.ps1")

    def test_class_and_methods(self, result) -> None:
        assert "Greeter" in _names_by_type(result, SymbolType.CLASS)
        methods = _names_by_type(result, SymbolType.METHOD)
        assert "Greet" in methods and "FormatName" in methods

    def test_top_level_function(self, result) -> None:
        assert "StandaloneHelper" in _names_by_type(result, SymbolType.FUNCTION)

    def test_positions(self, result) -> None:
        _assert_positions_populated(result)


class TestBash:
    @pytest.fixture
    def result(self):
        return _basic_result(BASH, FIXTURE_ROOT / "bash" / "hello.sh")

    def test_no_classes_only_functions(self, result) -> None:
        assert _names_by_type(result, SymbolType.CLASS) == set()
        funcs = _names_by_type(result, SymbolType.FUNCTION)
        assert {"greet", "format_name", "standalone_helper"} <= funcs

    def test_call_edges(self, result) -> None:
        pairs = _call_pairs(result)
        assert any(src == "greet" and tgt == "format_name" for src, tgt in pairs)

    def test_positions(self, result) -> None:
        _assert_positions_populated(result)


class TestObjC:
    @pytest.fixture
    def result(self):
        return _basic_result(OBJC, FIXTURE_ROOT / "objc" / "hello.m")

    def test_both_interface_and_implementation_emit_class(self, result) -> None:
        # Obj-C emits Greeter twice: once for @interface, once for
        # @implementation. Documented in the config; downstream
        # dedup happens at the graph level if needed.
        classes = [s for s in result.symbols if s.symbol_type == SymbolType.CLASS]
        names = [s.name for s in classes]
        assert names.count("Greeter") == 2

    def test_methods_emitted(self, result) -> None:
        methods = _names_by_type(result, SymbolType.METHOD)
        assert "greet" in methods and "formatName" in methods

    def test_positions(self, result) -> None:
        _assert_positions_populated(result)


class TestVerilog:
    @pytest.fixture
    def result(self):
        return _basic_result(VERILOG, FIXTURE_ROOT / "verilog" / "hello.v")

    def test_module_emitted_as_class(self, result) -> None:
        # ``module greeter`` maps to CLASS (modules are Verilog's unit
        # of encapsulation).
        assert "greeter" in _names_by_type(result, SymbolType.CLASS)

    def test_task_emitted_as_function(self, result) -> None:
        assert "format_signal" in _names_by_type(result, SymbolType.FUNCTION)

    def test_positions(self, result) -> None:
        _assert_positions_populated(result)


class TestFortran:
    @pytest.fixture
    def result(self):
        return _basic_result(FORTRAN, FIXTURE_ROOT / "fortran" / "hello.f90")

    def test_module_and_subroutines(self, result) -> None:
        assert "greeter_mod" in _names_by_type(result, SymbolType.CLASS)
        methods = _names_by_type(result, SymbolType.METHOD)
        assert "greet" in methods and "format_name" in methods

    def test_positions(self, result) -> None:
        _assert_positions_populated(result)


class TestJulia:
    @pytest.fixture
    def result(self):
        return _basic_result(JULIA, FIXTURE_ROOT / "julia" / "hello.jl")

    def test_struct_emitted(self, result) -> None:
        assert "Greeter" in _names_by_type(result, SymbolType.STRUCT)

    def test_functions_emitted(self, result) -> None:
        funcs = _names_by_type(result, SymbolType.FUNCTION)
        assert {"greet", "format_name", "standalone_helper"} <= funcs

    def test_positions(self, result) -> None:
        _assert_positions_populated(result)


class TestPascal:
    @pytest.fixture
    def result(self):
        return _basic_result(PASCAL, FIXTURE_ROOT / "pascal" / "hello.pas")

    def test_class_emitted(self, result) -> None:
        assert "TGreeter" in _names_by_type(result, SymbolType.CLASS)

    def test_method_implementations_emit_with_dotted_drill_down(self, result) -> None:
        # ``procedure TGreeter.Greet`` parses with ``genericDot`` as the
        # name chain; we drill to ``Greet`` (rightmost identifier).
        funcs = _names_by_type(result, SymbolType.FUNCTION)
        assert "Greet" in funcs
        assert "FormatName" in funcs

    def test_positions(self, result) -> None:
        _assert_positions_populated(result)


# ---------------------------------------------------------------------------
# Phase 2 — bespoke subclasses (Elixir, R, Zig, Groovy)
# ---------------------------------------------------------------------------


class TestElixir:
    @pytest.fixture
    def result(self):
        return ElixirParser(ELIXIR).parse_file(FIXTURE_ROOT / "elixir" / "hello.ex")

    def test_modules_emitted_as_classes(self, result) -> None:
        classes = _names_by_type(result, SymbolType.CLASS)
        assert "Greeter" in classes and "Main" in classes

    def test_def_emits_method_inside_module(self, result) -> None:
        methods = _names_by_type(result, SymbolType.METHOD)
        assert {"greet", "format_name", "run"} <= methods

    def test_call_edge_skips_def_keyword(self, result) -> None:
        # ``def greet(name) do format_name(name) end`` should emit a
        # CALLS edge to ``format_name`` from within Greeter.greet —
        # NOT a CALLS edge to ``def`` (the discriminator-driven calls
        # pass filters out declaration calls).
        pairs = _call_pairs(result)
        assert any(
            "greet" in src and tgt == "format_name"
            for src, tgt in pairs
        )
        assert not any(tgt in ("def", "defp", "defmodule") for _, tgt in pairs)

    def test_positions(self, result) -> None:
        _assert_positions_populated(result)


class TestR:
    @pytest.fixture
    def result(self):
        return RParser(R).parse_file(FIXTURE_ROOT / "r" / "hello.R")

    def test_function_bindings_emit_as_functions(self, result) -> None:
        funcs = _names_by_type(result, SymbolType.FUNCTION)
        assert {"greet", "format_name", "standalone_helper"} <= funcs

    def test_setRefClass_emits_as_class(self, result) -> None:
        # ``Greeter <- setRefClass("Greeter", ...)`` — the binding's
        # RHS is a call to a known R class-constructor.
        assert "Greeter" in _names_by_type(result, SymbolType.CLASS)

    def test_call_edges(self, result) -> None:
        pairs = _call_pairs(result)
        assert any(src == "greet" and tgt == "format_name" for src, tgt in pairs)

    def test_positions(self, result) -> None:
        _assert_positions_populated(result)

    def test_comparison_not_treated_as_binding(self, tmp_path: Path) -> None:
        """Rio R2 regression: R uses ``binary_operator`` for every
        binary expression (``<``, ``==``, ``+``, etc.), not just
        assignment. Without operator-gating, ``x < some_call()`` would
        emit ``x`` as a spurious symbol with ``some_call`` confused as
        a function value.
        """
        src = tmp_path / "compare.R"
        src.write_text(
            "x <- 1\n"
            "if (x < length(things)) { print('ok') }\n"
            "result <- function() 42\n"
        )
        result = RParser(R).parse_file(src)
        # ``x <- 1`` doesn't bind a function — no symbol.
        # ``x < length(things)`` is a comparison — no symbol.
        # ``result <- function() 42`` IS a binding — one function.
        funcs = _names_by_type(result, SymbolType.FUNCTION)
        assert "result" in funcs
        # ``x`` should NOT appear as a function (would be the bug).
        assert "x" not in funcs


class TestZig:
    @pytest.fixture
    def result(self):
        return ZigParser(ZIG).parse_file(FIXTURE_ROOT / "zig" / "hello.zig")

    def test_struct_decl_via_var_binding(self, result) -> None:
        # ``const Greeter = struct {...}`` → STRUCT.
        assert "Greeter" in _names_by_type(result, SymbolType.STRUCT)

    def test_functions_via_fnproto(self, result) -> None:
        funcs = _names_by_type(result, SymbolType.FUNCTION)
        assert {"greet", "formatName", "standalone"} <= funcs

    def test_positions(self, result) -> None:
        _assert_positions_populated(result)

    def test_packed_and_extern_struct_recognised(self, tmp_path: Path) -> None:
        """Code-review I1 regression: modifier-prefixed structs
        (``packed struct``, ``extern struct``, comptime blocks
        wrapping ``struct``) used to fall through to VARIABLE because
        the prefix check only matched bare ``struct``."""
        src = tmp_path / "modifiers.zig"
        src.write_text(
            'const Packed = packed struct { x: u8, y: u8 };\n'
            'const Extern = extern struct { ptr: ?*anyopaque };\n'
            'const Plain  = struct { name: []const u8 };\n'
        )
        result = ZigParser(ZIG).parse_file(src)
        structs = _names_by_type(result, SymbolType.STRUCT)
        assert "Packed" in structs
        assert "Extern" in structs
        assert "Plain" in structs

    def test_error_union_wrapping_struct_recognised(self, tmp_path: Path) -> None:
        """#153 regression: ``error{E}!struct {…}`` was the last form
        the prior text-prefix heuristic missed (``!`` is a mid-token
        separator). Structural ContainerDecl walk handles it.
        """
        src = tmp_path / "errunion.zig"
        src.write_text(
            'const SingleErr = error{X}!struct { field: u8 };\n'
            'const MultiErr  = error{X, Y}!struct { a: u8, b: u8 };\n'
            'const ErrEnum   = error{E}!enum { Foo, Bar };\n'
        )
        result = ZigParser(ZIG).parse_file(src)
        structs = _names_by_type(result, SymbolType.STRUCT)
        assert "SingleErr" in structs
        assert "MultiErr" in structs
        assert "ErrEnum" in structs

    def test_nested_struct_literal_in_call_arg_not_classified(
        self, tmp_path: Path,
    ) -> None:
        """Structural detection guarantees that only direct
        ``ContainerDecl`` children of ``SuffixExpr`` count — struct
        LITERALS passed as call arguments live inside
        ``FnCallArguments`` and are skipped. Forward-guard against a
        future change that loosened the direct-child requirement (the
        prior text-prefix code happened to also reject this, but for
        less robust reasons)."""
        src = tmp_path / "callarg.zig"
        src.write_text(
            'const Real    = struct { field: u8 };\n'
            'const NotDecl = makeThing(struct { field: u8 });\n'
        )
        result = ZigParser(ZIG).parse_file(src)
        structs = _names_by_type(result, SymbolType.STRUCT)
        assert "Real" in structs
        # NotDecl is a call expression whose argument happens to be
        # a struct literal — NOT a struct declaration. Should be
        # VARIABLE, not STRUCT.
        assert "NotDecl" not in structs

    def test_double_nested_error_union_struct_recognised(
        self, tmp_path: Path,
    ) -> None:
        """Stress the descent loop: ``error{E}!error{F}!struct {…}``
        wraps the struct under two ``ErrorUnionExpr`` levels. The
        rightmost-child walk must descend through both before
        finding the leaf with the ``ContainerDecl``."""
        src = tmp_path / "double_eu.zig"
        src.write_text(
            'const Deep = error{X}!error{Y}!struct { field: u8 };\n'
        )
        result = ZigParser(ZIG).parse_file(src)
        assert "Deep" in _names_by_type(result, SymbolType.STRUCT)

    def test_comptime_block_struct_is_known_limitation(
        self, tmp_path: Path,
    ) -> None:
        """Pinning test for the documented comptime-block limitation
        (see ZigParser docstring). ``const X = comptime blk: { break
        :blk struct {…}; };`` parses as ``VarDecl > LabeledTypeExpr >
        Block > … > ContainerDecl``, and the descent here doesn't
        enter ``LabeledTypeExpr``. Classified as VARIABLE; a future
        refinement would extend the descent through labeled blocks.
        """
        src = tmp_path / "comptime.zig"
        src.write_text(
            'const ComptimeStruct = comptime blk: {\n'
            '    break :blk struct { field: u8 };\n'
            '};\n'
        )
        result = ZigParser(ZIG).parse_file(src)
        # Not in STRUCT — this is the documented limitation. If a
        # future fix lifts the comptime-block restriction, this
        # assertion fails and the test signals the limitation has
        # been resolved (update the docstring accordingly).
        assert "ComptimeStruct" not in _names_by_type(result, SymbolType.STRUCT)


class TestGroovy:
    @pytest.fixture
    def result(self):
        return GroovyParser(GROOVY).parse_file(FIXTURE_ROOT / "groovy" / "hello.groovy")

    def test_class_emitted_from_command_unit_sequence(self, result) -> None:
        # Groovy parses ``class Greeter { … }`` as a deeply-nested
        # command/unit/block tree — the bespoke parser walks it to
        # extract the class name.
        assert "Greeter" in _names_by_type(result, SymbolType.CLASS)

    def test_methods_emitted_inside_class_body(self, result) -> None:
        methods = _names_by_type(result, SymbolType.METHOD)
        assert "greet" in methods and "formatName" in methods

    def test_positions(self, result) -> None:
        _assert_positions_populated(result)
