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
    KOTLIN,
    LUA,
    PHP,
    RUBY,
    SCALA,
    build_basic_parsers,
)

FIXTURE_ROOT = Path(__file__).parent.parent / "fixtures" / "parsers"


# ---------------------------------------------------------------------------
# Helpers + shared assertions
# ---------------------------------------------------------------------------


def _names_by_type(result, symbol_type: SymbolType) -> set[str]:
    return {s.name for s in result.symbols if s.symbol_type == symbol_type}


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


class TestRegistry:
    def test_build_basic_parsers_returns_five_languages(self) -> None:
        parsers = build_basic_parsers()
        assert set(parsers.keys()) == {"ruby", "php", "kotlin", "scala", "lua"}
        for name, parser in parsers.items():
            assert isinstance(parser, BasicVisitorParser)
            assert parser.config.name == name

    def test_each_parser_reports_its_extensions(self) -> None:
        parsers = build_basic_parsers()
        assert ".rb" in parsers["ruby"]._get_supported_extensions()
        assert ".php" in parsers["php"]._get_supported_extensions()
        assert ".kt" in parsers["kotlin"]._get_supported_extensions()
        assert ".scala" in parsers["scala"]._get_supported_extensions()
        assert ".lua" in parsers["lua"]._get_supported_extensions()

    def test_each_parser_advertises_its_capabilities(self) -> None:
        parsers = build_basic_parsers()
        # Languages with classes should report so; Lua should not.
        assert parsers["ruby"].capabilities.has_classes is True
        assert parsers["php"].capabilities.has_classes is True
        assert parsers["lua"].capabilities.has_classes is False


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
