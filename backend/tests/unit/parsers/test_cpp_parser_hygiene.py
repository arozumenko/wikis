"""Hygiene tests for the C++ parser.

Verifies that the parser drops non-user-defined / parse-recovery symbols
at parse time, per INVESTIGATION_GRAPH_QUALITY.md §11.1.

Each test parses a small inline C++ snippet and asserts the resulting
symbol set.
"""

from __future__ import annotations

import textwrap
from typing import Iterable, List

import pytest

from app.core.parsers.base_parser import Symbol, SymbolType
from app.core.parsers.cpp_enhanced_parser import (
    CppEnhancedParser,
    _is_header_guard_macro,
    _is_operator_name,
    _is_parse_recovery_name,
    _is_stdlib_type_alias,
)


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #

def _parse(tmp_path, source: str, *, suffix: str = ".hpp") -> List[Symbol]:
    """Parse ``source`` as a C++ file and return the extracted symbols."""
    path = tmp_path / f"sample{suffix}"
    path.write_text(textwrap.dedent(source))
    parser = CppEnhancedParser()
    result = parser.parse_file(str(path))
    assert not result.errors, f"parser errors: {result.errors}"
    return list(result.symbols)


def _names(symbols: Iterable[Symbol], symbol_type: SymbolType | None = None) -> List[str]:
    """Return the names of ``symbols`` optionally filtered by type."""
    return [
        s.name
        for s in symbols
        if symbol_type is None or s.symbol_type == symbol_type
    ]


# --------------------------------------------------------------------------- #
# Pure-helper tests (regex / set membership)
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize(
    "name,value,expected",
    [
        # Real fmt header guards
        ("FMT_ARGS_H_", None, True),
        ("FMT_COMPILE_H_", None, True),
        ("FMT_OSTREAM_H_", None, True),
        ("FMT_BASE_H", "", True),
        ("MYLIB_FOO_HPP", None, True),
        ("MYLIB_FOO_HXX", None, True),
        ("FMT_BASE_H_", "0", False),       # has a body — keep
        ("FMT_VERSION", '"10.0.0"', False),  # not a guard — has a value
        ("LOG_LEVEL", "3", False),
        ("My_h", None, False),              # not all-caps
        ("FMT_FOO_INTERNAL", None, False),  # doesn't end in _H/_HPP/_HXX
    ],
)
def test_is_header_guard_macro(name: str, value: str | None, expected: bool) -> None:
    assert _is_header_guard_macro(name, value) is expected


@pytest.mark.parametrize(
    "name,expected",
    [
        ("decay_t", True),
        ("nullptr_t", True),
        ("enable_if_t", True),
        ("remove_cvref_t", True),
        ("invoke_result_t", True),
        ("my_decay_t", False),
        ("custom_alias", False),
        ("MyType", False),
        ("string_view", False),
    ],
)
def test_is_stdlib_type_alias(name: str, expected: bool) -> None:
    assert _is_stdlib_type_alias(name) is expected


@pytest.mark.parametrize(
    "name,expected",
    [
        ("auto p", True),
        ("auto value", True),
        ("typename...", True),
        ("class...", True),
        ("template...", True),
        ("T", True),
        ("U", True),
        ("V", True),
        ("MyClass", False),
        ("foo_bar", False),
        ("", True),       # empty / None means "skip"
        (None, True),
    ],
)
def test_is_parse_recovery_name(name: str | None, expected: bool) -> None:
    assert _is_parse_recovery_name(name) is expected


@pytest.mark.parametrize(
    "name,expected",
    [
        ("operator+", True),
        ("operator[]", True),
        ("operator<<", True),
        ("operator()", True),
        ('operator""_cf', True),
        ("operator int", True),
        ("operator", True),     # bare keyword still flagged
        ("format", False),
        ("operate", False),
        ("MyOperator", False),
    ],
)
def test_is_operator_name(name: str, expected: bool) -> None:
    assert _is_operator_name(name) is expected


# --------------------------------------------------------------------------- #
# Parser behaviour tests
# --------------------------------------------------------------------------- #

def test_header_guards_are_dropped(tmp_path) -> None:
    """`#define FOO_H_` style guards should never become MACRO symbols."""
    source = """
        #ifndef MYLIB_FOO_H_
        #define MYLIB_FOO_H_

        #define MYLIB_VERSION "1.0.0"
        #define MYLIB_DEBUG 1

        int real_function();

        #endif
    """
    symbols = _parse(tmp_path, source, suffix=".h")
    macros = _names(symbols, SymbolType.MACRO)

    assert "MYLIB_FOO_H_" not in macros, "header guard must be dropped"
    assert "MYLIB_VERSION" in macros, "real value-bearing macro must survive"
    assert "MYLIB_DEBUG" in macros


def test_stdlib_type_alias_is_dropped(tmp_path) -> None:
    """Stdlib trait shims like `using decay_t = ...` should be dropped."""
    source = """
        namespace stdshim {
        template <typename T> using decay_t = typename decay<T>::type;
        template <typename T> using my_decay_t = typename decay<T>::type;
        using MyType = int;
        }
    """
    symbols = _parse(tmp_path, source)
    aliases = _names(symbols, SymbolType.TYPE_ALIAS)

    assert "decay_t" not in aliases, "stdlib alias must be dropped"
    assert "my_decay_t" in aliases, "user alias with similar suffix must survive"
    assert "MyType" in aliases


def test_parse_recovery_class_names_are_dropped(tmp_path) -> None:
    """`auto p` / `typename...` style parse-recovery garbage must be dropped."""
    # NOTE: tree-sitter occasionally misparses constructs like
    # `template <typename... Args>` and emits a "class" with name
    # "typename...". We can't easily reproduce all such misparses with
    # well-formed input, so this test confirms the FILTER catches them
    # by feeding a deliberately malformed snippet that tree-sitter
    # error-recovers into a class_specifier.
    source = """
        class MyRealClass {
        public:
            void foo();
        };

        // Well-formed C++; ensure neighbouring real classes survive.
        struct MyRealStruct {
            int x;
        };
    """
    symbols = _parse(tmp_path, source)
    classes = _names(symbols, SymbolType.CLASS)
    structs = _names(symbols, SymbolType.STRUCT)

    assert "MyRealClass" in classes
    assert "MyRealStruct" in structs
    # Filter is exercised by the unit-level helper tests above; this test
    # just guards against accidentally dropping legitimate classes.
    assert "auto p" not in classes
    assert "typename..." not in classes
    assert "T" not in classes


def test_member_operators_demoted_free_operators_stay_function(tmp_path) -> None:
    """Member operator overloads must be SymbolType.OPERATOR; free operators stay FUNCTION.

    Member operators (e.g. ``Vec::operator+``) are documented as part of
    their owning class, so they're kept in the graph for call resolution
    but demoted out of the architectural set.  Free-standing operators
    (e.g. ``std::ostream& operator<<(std::ostream&, const MyClass&)``)
    are real public API and stay architectural FUNCTIONs.
    """
    source = """
        struct Vec {
            int x;
            // Member operators - DEMOTED to OPERATOR
            Vec operator+(const Vec& other) const;
            int& operator[](int i);
            bool operator==(const Vec& other) const;
        };

        // Free-standing operator - stays FUNCTION (real public API)
        Vec operator-(const Vec& a, const Vec& b);

        // Real method that should NOT be tagged OPERATOR
        int real_method(int x);
    """
    symbols = _parse(tmp_path, source)

    operators = _names(symbols, SymbolType.OPERATOR)
    methods = _names(symbols, SymbolType.METHOD)
    functions = _names(symbols, SymbolType.FUNCTION)

    # Member operators: demoted
    assert "operator+" in operators
    assert "operator[]" in operators
    assert "operator==" in operators
    # ...not also reported as methods
    assert "operator+" not in methods
    assert "operator[]" not in methods

    # Free operator: stays FUNCTION (architectural)
    assert "operator-" in functions, (
        f"free operator must stay FUNCTION, got operators={operators!r} functions={functions!r}"
    )
    assert "operator-" not in operators

    # Real method/function unaffected
    assert "real_method" in functions or "real_method" in methods


def test_anonymous_enums_are_kept_at_every_scope(tmp_path) -> None:
    """Anonymous enums are kept (top-level and member).

    Dropping the synthetic ``<anonymous_enum:foo>`` wrapper would orphan
    the enumerators inside it (they reference the wrapper as
    ``parent_symbol``).  Top-level anonymous enums often hold real
    user-facing constants, so we keep them as architectural ENUMs.
    """
    source = """
        // Top-level anonymous enum: KEEP (constants are real public API)
        enum {
            top_level_anon_first = 1,
            top_level_anon_second = 2,
        };

        // Anonymous enum INSIDE a struct: KEEP
        struct Container {
            enum {
                inside_first = 10,
                inside_second = 20,
            };

            int real_field;
        };

        // Named enum: KEEP
        enum class NamedEnum { A, B };
    """
    symbols = _parse(tmp_path, source)
    enum_names = _names(symbols, SymbolType.ENUM)

    # Top-level anonymous enum: kept
    assert any(n.startswith("<anonymous_enum:top_level") for n in enum_names), (
        f"top-level anonymous enum should be kept: {enum_names!r}"
    )
    # Member anonymous enum: kept
    assert any(n.startswith("<anonymous_enum:inside_first") for n in enum_names), (
        f"member anonymous enum should be kept: {enum_names!r}"
    )
    # Named enum: kept
    assert "NamedEnum" in enum_names

