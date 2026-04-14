"""Tests for app.core.cluster_constants.

Covers:
- Symbol set membership and non-overlap
- SYMBOL_TYPE_PRIORITY completeness and range
- Page sizing constants and adaptive_max_page_size() thresholds
- target_section_count() logarithmic scaling
- target_total_pages() sqrt scaling
- Hub / centroid / resolution constants
- Test path detection (directory patterns, file patterns, negatives)
- DOC_DOMINANT_THRESHOLD range
"""

import math
import re

import pytest

from app.core.cluster_constants import (
    CENTROID_ARCHITECTURAL_MIN_PRIORITY,
    DEFAULT_HUB_Z_THRESHOLD,
    DEFAULT_MAX_PAGE_SIZE,
    DOC_CLUSTER_SYMBOLS,
    DOC_DOMINANT_THRESHOLD,
    LEIDEN_FILE_SECTION_RESOLUTION,
    LEIDEN_PAGE_RESOLUTION,
    MAX_CENTROIDS,
    MERGE_THRESHOLD,
    MIN_CENTROIDS,
    MIN_PAGE_SIZE,
    PAGE_IDENTITY_SYMBOLS,
    SUPPORTING_CODE_SYMBOLS,
    SYMBOL_TYPE_PRIORITY,
    _TEST_PATH_PATTERNS,
    adaptive_max_page_size,
    is_test_path,
    target_section_count,
    target_total_pages,
)
from app.core.constants import ARCHITECTURAL_SYMBOLS, DOC_SYMBOL_TYPES


# ═══════════════════════════════════════════════════════════════════════════
# Symbol Set Tests
# ═══════════════════════════════════════════════════════════════════════════


class TestSymbolSets:
    """Validate symbol classification sets."""

    def test_page_identity_symbols_are_frozenset(self):
        assert isinstance(PAGE_IDENTITY_SYMBOLS, frozenset)

    def test_supporting_code_symbols_are_frozenset(self):
        assert isinstance(SUPPORTING_CODE_SYMBOLS, frozenset)

    def test_doc_cluster_symbols_are_frozenset(self):
        assert isinstance(DOC_CLUSTER_SYMBOLS, frozenset)

    def test_page_identity_and_supporting_are_disjoint(self):
        overlap = PAGE_IDENTITY_SYMBOLS & SUPPORTING_CODE_SYMBOLS
        assert overlap == set(), f"Unexpected overlap: {overlap}"

    def test_page_identity_and_doc_cluster_are_disjoint(self):
        overlap = PAGE_IDENTITY_SYMBOLS & DOC_CLUSTER_SYMBOLS
        assert overlap == set()

    def test_supporting_and_doc_cluster_are_disjoint(self):
        overlap = SUPPORTING_CODE_SYMBOLS & DOC_CLUSTER_SYMBOLS
        assert overlap == set()

    def test_known_identity_symbols_present(self):
        for sym in ("class", "interface", "function", "enum", "struct"):
            assert sym in PAGE_IDENTITY_SYMBOLS

    def test_known_supporting_symbols_present(self):
        for sym in ("constant", "type_alias", "macro"):
            assert sym in SUPPORTING_CODE_SYMBOLS

    def test_doc_cluster_symbols(self):
        assert DOC_CLUSTER_SYMBOLS == {"module_doc", "file_doc"}


class TestSymbolTypePriority:
    """Validate SYMBOL_TYPE_PRIORITY mapping."""

    def test_all_identity_symbols_have_priority(self):
        for sym in PAGE_IDENTITY_SYMBOLS:
            assert sym in SYMBOL_TYPE_PRIORITY, f"Missing priority for {sym}"

    def test_all_supporting_symbols_have_priority(self):
        for sym in SUPPORTING_CODE_SYMBOLS:
            assert sym in SYMBOL_TYPE_PRIORITY

    def test_all_doc_cluster_symbols_have_priority(self):
        for sym in DOC_CLUSTER_SYMBOLS:
            assert sym in SYMBOL_TYPE_PRIORITY

    def test_priorities_are_positive_integers(self):
        for sym, pri in SYMBOL_TYPE_PRIORITY.items():
            assert isinstance(pri, int), f"{sym} priority is not int"
            assert pri > 0, f"{sym} priority must be positive"

    def test_class_has_highest_priority(self):
        assert SYMBOL_TYPE_PRIORITY["class"] == 10

    def test_identity_symbols_outrank_supporting(self):
        min_identity = min(SYMBOL_TYPE_PRIORITY[s] for s in PAGE_IDENTITY_SYMBOLS)
        max_supporting = max(SYMBOL_TYPE_PRIORITY[s] for s in SUPPORTING_CODE_SYMBOLS)
        assert min_identity >= max_supporting

    def test_priority_range(self):
        values = SYMBOL_TYPE_PRIORITY.values()
        assert min(values) >= 1
        assert max(values) <= 10


# ═══════════════════════════════════════════════════════════════════════════
# Page Sizing Tests
# ═══════════════════════════════════════════════════════════════════════════


class TestPageSizingConstants:
    """Validate page sizing constants."""

    def test_min_page_size(self):
        assert MIN_PAGE_SIZE == 5

    def test_merge_threshold(self):
        assert MERGE_THRESHOLD == 4

    def test_default_max_page_size(self):
        assert DEFAULT_MAX_PAGE_SIZE == 25

    def test_merge_threshold_less_than_min_page(self):
        assert MERGE_THRESHOLD < MIN_PAGE_SIZE


class TestAdaptiveMaxPageSize:
    """Test adaptive_max_page_size() thresholds."""

    def test_small_repo(self):
        assert adaptive_max_page_size(100) == 25

    def test_medium_repo(self):
        assert adaptive_max_page_size(500) == 35
        assert adaptive_max_page_size(1999) == 35

    def test_large_repo(self):
        assert adaptive_max_page_size(2000) == 45
        assert adaptive_max_page_size(4999) == 45

    def test_very_large_repo(self):
        assert adaptive_max_page_size(5000) == 60
        assert adaptive_max_page_size(50000) == 60

    def test_boundary_values(self):
        assert adaptive_max_page_size(499) == 25
        assert adaptive_max_page_size(500) == 35

    def test_zero_nodes(self):
        assert adaptive_max_page_size(0) == 25

    def test_monotonically_non_decreasing(self):
        """Page size should never decrease as node count increases."""
        prev = 0
        for n in range(0, 10001, 100):
            current = adaptive_max_page_size(n)
            assert current >= prev, f"Decreased at n={n}: {prev} → {current}"
            prev = current


class TestTargetSectionCount:
    """Test target_section_count() logarithmic scaling."""

    def test_small_files(self):
        assert target_section_count(5) == 5
        assert target_section_count(9) == 5

    def test_medium_files(self):
        result = target_section_count(50)
        assert 5 <= result <= 20

    def test_large_files(self):
        result = target_section_count(5000)
        assert result >= 10

    def test_floor_is_5(self):
        for n in range(1, 15):
            assert target_section_count(n) >= 5

    def test_cap_is_20(self):
        assert target_section_count(1_000_000) <= 20

    def test_monotonically_non_decreasing(self):
        prev = 0
        for n in [1, 10, 50, 100, 500, 1000, 5000, 50000]:
            current = target_section_count(n)
            assert current >= prev, f"Decreased at n={n}"
            prev = current

    def test_known_values(self):
        """Verify table from docstring."""
        assert target_section_count(10) == 5
        # 50 files: 1.2 * log2(50) ≈ 1.2 * 5.64 = 6.77 → ceil = 7
        assert target_section_count(50) == 7
        # 500 files: 1.2 * log2(500) ≈ 1.2 * 8.97 = 10.76 → ceil = 11
        assert target_section_count(500) == 11


class TestTargetTotalPages:
    """Test target_total_pages() sqrt scaling."""

    def test_floor_is_8(self):
        assert target_total_pages(1) == 8
        assert target_total_pages(0) == 8

    def test_cap_is_200(self):
        assert target_total_pages(1_000_000) == 200

    def test_known_values(self):
        # 729 nodes: sqrt(729/7) = sqrt(104.14) ≈ 10.2 → ceil = 11
        assert target_total_pages(729) == 11
        # 5000 nodes: sqrt(5000/7) ≈ 26.7 → ceil = 27
        assert target_total_pages(5000) == 27

    def test_monotonically_non_decreasing(self):
        prev = 0
        for n in [0, 100, 500, 1000, 5000, 50000, 500000]:
            current = target_total_pages(n)
            assert current >= prev
            prev = current


# ═══════════════════════════════════════════════════════════════════════════
# Constants Value Tests
# ═══════════════════════════════════════════════════════════════════════════


class TestHubConstants:
    def test_z_threshold_positive(self):
        assert DEFAULT_HUB_Z_THRESHOLD > 0

    def test_z_threshold_value(self):
        assert DEFAULT_HUB_Z_THRESHOLD == 3.0


class TestCentroidConstants:
    def test_min_priority(self):
        assert CENTROID_ARCHITECTURAL_MIN_PRIORITY == 5

    def test_min_centroids(self):
        assert MIN_CENTROIDS == 1

    def test_max_centroids(self):
        assert MAX_CENTROIDS == 10

    def test_min_less_than_max(self):
        assert MIN_CENTROIDS < MAX_CENTROIDS


class TestLeidenConstants:
    def test_section_resolution(self):
        assert LEIDEN_FILE_SECTION_RESOLUTION == 1.0

    def test_page_resolution(self):
        assert LEIDEN_PAGE_RESOLUTION == 1.0

    def test_resolutions_positive(self):
        assert LEIDEN_FILE_SECTION_RESOLUTION > 0
        assert LEIDEN_PAGE_RESOLUTION > 0


class TestDocDominantThreshold:
    def test_threshold_between_0_and_1(self):
        assert 0 < DOC_DOMINANT_THRESHOLD < 1

    def test_threshold_value(self):
        assert DOC_DOMINANT_THRESHOLD == 0.7


# ═══════════════════════════════════════════════════════════════════════════
# Test Path Detection Tests
# ═══════════════════════════════════════════════════════════════════════════


class TestTestPathPatterns:
    """Validate _TEST_PATH_PATTERNS list."""

    def test_patterns_are_compiled(self):
        for p in _TEST_PATH_PATTERNS:
            assert isinstance(p, re.Pattern)

    def test_most_patterns_case_insensitive(self):
        """All patterns except the PascalCase Tests? pattern are case-insensitive."""
        case_sensitive = 0
        for p in _TEST_PATH_PATTERNS:
            if not (p.flags & re.IGNORECASE):
                case_sensitive += 1
        # Only the PascalCase *Test(s).ext pattern is case-sensitive
        assert case_sensitive == 1


class TestIsTestPath:
    """Test is_test_path() function."""

    # ── Directory-based matches ──

    @pytest.mark.parametrize(
        "path",
        [
            "tests/test_foo.py",
            "test/helpers.py",
            "src/tests/unit/test_bar.py",
            "Tests/MyTest.cs",
        ],
    )
    def test_test_directory(self, path):
        assert is_test_path(path) is True

    @pytest.mark.parametrize(
        "path",
        [
            "__tests__/Button.test.js",
            "src/__tests__/utils.spec.ts",
        ],
    )
    def test_dunder_tests_directory(self, path):
        assert is_test_path(path) is True

    @pytest.mark.parametrize(
        "path",
        [
            "spec/models/user_spec.rb",
            "specs/helpers.rb",
        ],
    )
    def test_spec_directory(self, path):
        assert is_test_path(path) is True

    def test_testing_directory(self):
        assert is_test_path("testing/utils.go") is True

    def test_fixtures_directory(self):
        assert is_test_path("fixtures/sample.json") is True

    def test_mocks_directory(self):
        assert is_test_path("mocks/api_mock.py") is True
        assert is_test_path("mock/data.json") is True

    def test_testdata_directory(self):
        assert is_test_path("testdata/input.txt") is True

    def test_testutils_directory(self):
        assert is_test_path("testutils/helpers.py") is True
        assert is_test_path("testutil/base.py") is True

    # ── File-based matches ──

    @pytest.mark.parametrize(
        "path",
        [
            "test_foo.py",
            "src/test_bar.py",
            "pkg/test_utils.go",
        ],
    )
    def test_test_prefix_file(self, path):
        assert is_test_path(path) is True

    @pytest.mark.parametrize(
        "path",
        [
            "foo_test.go",
            "src/bar_test.go",
        ],
    )
    def test_test_suffix_file(self, path):
        assert is_test_path(path) is True

    @pytest.mark.parametrize(
        "path",
        [
            "Button.test.js",
            "src/utils.test.ts",
        ],
    )
    def test_dot_test_file(self, path):
        assert is_test_path(path) is True

    @pytest.mark.parametrize(
        "path",
        [
            "Button.spec.js",
            "src/utils.spec.ts",
        ],
    )
    def test_dot_spec_file(self, path):
        assert is_test_path(path) is True

    @pytest.mark.parametrize(
        "path",
        [
            "FooTest.java",
            "BarTests.cs",
            "src/FooTest.java",
        ],
    )
    def test_test_class_file(self, path):
        assert is_test_path(path) is True

    def test_conftest(self):
        assert is_test_path("conftest.py") is True
        assert is_test_path("src/tests/conftest.py") is True

    def test_setup_test(self):
        assert is_test_path("setup_test.py") is True
        assert is_test_path("setup_tests.js") is True

    # ── Non-test paths (should NOT match) ──

    @pytest.mark.parametrize(
        "path",
        [
            "src/main.py",
            "app/models/user.py",
            "lib/utils.js",
            "pkg/handler.go",
            "README.md",
            "src/contest.py",  # "contest" should NOT match "conftest"
            "src/testimony.py",  # "testimony" should NOT match "test"
            "latest/data.json",  # "latest" should NOT match "test"
            "src/attest.py",  # should NOT match
        ],
    )
    def test_non_test_paths(self, path):
        assert is_test_path(path) is False

    # ── Case insensitivity ──

    def test_case_insensitive_directory(self):
        assert is_test_path("Tests/Foo.py") is True
        assert is_test_path("TESTS/bar.py") is True

    def test_case_insensitive_file(self):
        assert is_test_path("Test_Foo.py") is True
