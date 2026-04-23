"""Tests for language_heuristics module."""

import pytest

from app.core.wiki_structure_planner.language_heuristics import (
    LanguageHints,
    compute_augmentation_budget_fraction,
    detect_dominant_language,
    get_language_hints,
    should_include_in_expansion,
)


# ═══════════════════════════════════════════════════════════════════════════
# LanguageHints dataclass
# ═══════════════════════════════════════════════════════════════════════════


class TestLanguageHintsDataclass:
    def test_default_hints(self):
        h = LanguageHints(language="test")
        assert h.language == "test"
        assert h.extra_expansion_rels == frozenset()
        assert h.skip_symbol_types == frozenset()
        assert h.prefer_impl_over_decl is False
        assert h.grouping_hint == "module"
        assert h.impl_augmentation_priority == 1.0
        assert h.declaration_extensions == frozenset()
        assert h.implementation_extensions == frozenset()
        assert h.interface_impl_priority is False

    def test_frozen(self):
        h = LanguageHints(language="frozen")
        with pytest.raises(AttributeError):
            h.language = "mutated"


# ═══════════════════════════════════════════════════════════════════════════
# get_language_hints
# ═══════════════════════════════════════════════════════════════════════════


class TestGetLanguageHints:
    @pytest.mark.parametrize(
        "lang,expected_lang",
        [
            ("python", "python"),
            ("cpp", "cpp"),
            ("c", "cpp"),  # C shares C++ hints
            ("go", "go"),
            ("java", "java"),
            ("javascript", "javascript"),
            ("typescript", "typescript"),
            ("rust", "rust"),
            ("c_sharp", "c_sharp"),
        ],
    )
    def test_known_languages(self, lang, expected_lang):
        hints = get_language_hints(lang)
        assert hints.language == expected_lang

    def test_unknown_language_returns_default(self):
        hints = get_language_hints("brainfuck")
        assert hints.language == "unknown"

    def test_case_insensitive(self):
        hints = get_language_hints("Python")
        assert hints.language == "python"

    # ── Per-language property checks ─────────────────────────

    def test_cpp_has_defines_body(self):
        hints = get_language_hints("cpp")
        assert "defines_body" in hints.extra_expansion_rels
        assert hints.prefer_impl_over_decl is True
        assert hints.impl_augmentation_priority == 2.0
        assert ".h" in hints.declaration_extensions

    def test_go_has_defines_body(self):
        hints = get_language_hints("go")
        assert "defines_body" in hints.extra_expansion_rels
        assert hints.interface_impl_priority is True

    def test_rust_has_implementation(self):
        hints = get_language_hints("rust")
        assert "implementation" in hints.extra_expansion_rels
        assert "defines_body" in hints.extra_expansion_rels
        assert hints.interface_impl_priority is True

    def test_python_has_decorates(self):
        hints = get_language_hints("python")
        assert "decorates" in hints.extra_expansion_rels

    def test_javascript_skips_constants(self):
        hints = get_language_hints("javascript")
        assert "constant" in hints.skip_symbol_types

    def test_typescript_prefers_impl(self):
        hints = get_language_hints("typescript")
        assert hints.prefer_impl_over_decl is True
        assert ".d.ts" in hints.declaration_extensions

    def test_java_has_annotates(self):
        hints = get_language_hints("java")
        assert "annotates" in hints.extra_expansion_rels

    def test_csharp_has_implementation(self):
        hints = get_language_hints("c_sharp")
        assert "implementation" in hints.extra_expansion_rels
        assert hints.interface_impl_priority is True


# ═══════════════════════════════════════════════════════════════════════════
# should_include_in_expansion
# ═══════════════════════════════════════════════════════════════════════════


class TestShouldIncludeInExpansion:
    def test_normal_symbol_included(self):
        hints = get_language_hints("python")
        assert should_include_in_expansion(hints, "class", "src/main.py") is True

    def test_skipped_type_excluded(self):
        hints = get_language_hints("cpp")
        assert should_include_in_expansion(hints, "macro", "src/main.cpp") is False

    def test_constant_excluded_for_javascript(self):
        hints = get_language_hints("javascript")
        assert should_include_in_expansion(hints, "constant", "lib/config.js") is False

    def test_function_included_for_javascript(self):
        hints = get_language_hints("javascript")
        assert should_include_in_expansion(hints, "function", "lib/main.js") is True

    def test_unknown_lang_includes_everything(self):
        hints = get_language_hints("unknown")
        assert should_include_in_expansion(hints, "macro", "test.unk") is True


# ═══════════════════════════════════════════════════════════════════════════
# compute_augmentation_budget_fraction
# ═══════════════════════════════════════════════════════════════════════════


class TestComputeAugmentationBudgetFraction:
    def test_python_default_fraction(self):
        """Python has priority 1.0 → base 0.3 * 1.0 = 0.3"""
        hints = get_language_hints("python")
        assert compute_augmentation_budget_fraction(hints) == pytest.approx(0.3)

    def test_cpp_high_priority(self):
        """C++ has priority 2.0 → base 0.3 * 2.0 = 0.6"""
        hints = get_language_hints("cpp")
        assert compute_augmentation_budget_fraction(hints) == pytest.approx(0.6)

    def test_javascript_low_priority(self):
        """JS has priority 0.8 → base 0.3 * 0.8 = 0.24"""
        hints = get_language_hints("javascript")
        assert compute_augmentation_budget_fraction(hints) == pytest.approx(0.24)

    def test_clamped_minimum(self):
        """Even with very low base, result is clamped to 0.1."""
        hints = LanguageHints(language="test", impl_augmentation_priority=0.01)
        assert compute_augmentation_budget_fraction(hints, base_fraction=0.1) >= 0.1

    def test_clamped_maximum(self):
        """Even with very high priority, result is clamped to 0.6."""
        hints = LanguageHints(language="test", impl_augmentation_priority=10.0)
        assert compute_augmentation_budget_fraction(hints) <= 0.6


# ═══════════════════════════════════════════════════════════════════════════
# detect_dominant_language
# ═══════════════════════════════════════════════════════════════════════════


class TestDetectDominantLanguage:
    @pytest.fixture()
    def db_conn(self):
        """Create an in-memory SQLite DB with minimal schema."""
        import sqlite3

        conn = sqlite3.connect(":memory:")
        conn.row_factory = sqlite3.Row
        conn.execute(
            "CREATE TABLE repo_nodes ("
            "  node_id TEXT PRIMARY KEY,"
            "  language TEXT"
            ")"
        )
        conn.executemany(
            "INSERT INTO repo_nodes (node_id, language) VALUES (?, ?)",
            [
                ("n1", "python"),
                ("n2", "python"),
                ("n3", "go"),
                ("n4", None),
            ],
        )
        conn.commit()
        yield conn
        conn.close()

    def test_detects_dominant(self, db_conn):
        result = detect_dominant_language(db_conn, ["n1", "n2", "n3"])
        assert result == "python"

    def test_empty_ids(self, db_conn):
        assert detect_dominant_language(db_conn, []) is None

    def test_nonexistent_ids(self, db_conn):
        assert detect_dominant_language(db_conn, ["nonexistent"]) is None

    def test_all_null_language(self, db_conn):
        assert detect_dominant_language(db_conn, ["n4"]) is None
