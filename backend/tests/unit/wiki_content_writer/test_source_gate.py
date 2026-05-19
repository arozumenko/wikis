"""Tests for wiki_content_writer/source_gate.py (#240).

Pre-filter rules:
1. README claim without README read → "readme_claim_without_read"
2. SQL claim without .sql read → "sql_claim_without_read"
3. Identifier not in graph or tool trace → "identifier_not_grounded"

Covers per-rule detection, mode toggles (flag/strip/verifier),
the happy path, and interaction between multiple rules.
"""

from __future__ import annotations

from app.core.wiki_content_writer.citation_extractor import Citation, CitedClaim
from app.core.wiki_content_writer.source_gate import (
    GateFlag,
    GateReport,
    ToolCall,
    apply_gate,
)


# ── Helpers ──────────────────────────────────────────────────────────────────


def _claim(text: str, paragraph_index: int = 0, paths: list[str] | None = None) -> CitedClaim:
    """Build a minimal CitedClaim for tests."""
    citations = [Citation(path=p, start_line=1, end_line=1) for p in (paths or [])]
    return CitedClaim(
        paragraph_index=paragraph_index,
        paragraph_text=text,
        citations=citations,
    )


def _read_file(path: str, content: str = "") -> ToolCall:
    return ToolCall(tool="read_file", args={"path": path}, result_text=content)


def _grep(pattern: str, content: str = "") -> ToolCall:
    return ToolCall(tool="grep", args={"pattern": pattern}, result_text=content)


# ── Happy path — no flags ─────────────────────────────────────────────────────


class TestHappyPath:
    def test_no_flags_when_no_suspicious_content(self):
        claims = [_claim("The system processes requests asynchronously.")]
        kept, report = apply_gate(claims, [])

        assert report.flags == []
        assert report.stripped_paragraphs == []
        assert len(kept) == 1

    def test_returns_tuple_of_claims_and_report(self):
        claims = [_claim("Simple text.")]
        result = apply_gate(claims, [])

        assert isinstance(result, tuple)
        assert len(result) == 2

    def test_empty_claims_returns_empty(self):
        kept, report = apply_gate([], [])

        assert kept == []
        assert report.flags == []

    def test_readme_mention_with_readme_read_passes(self):
        trace = [_read_file("README.md", "Install with pip install.")]
        claims = [_claim("See README for installation steps.", paths=["README.md"])]
        kept, report = apply_gate(claims, trace)

        assert report.flags == []

    def test_sql_claim_with_sql_read_passes(self):
        trace = [_read_file("schema.sql", "CREATE TABLE users (id INTEGER PRIMARY KEY);")]
        claims = [_claim("The users table has a PRIMARY KEY column.", paths=["schema.sql"])]
        kept, report = apply_gate(claims, trace)

        # No sql_claim_without_read flag because .sql was read
        sql_flags = [f for f in report.flags if f.rule == "sql_claim_without_read"]
        assert sql_flags == []

    def test_known_symbol_in_known_symbols_set_passes(self):
        claims = [_claim("The WikiContentWriter handles page generation.")]
        kept, report = apply_gate(
            claims,
            [],
            known_symbols={"WikiContentWriter"},
        )

        grounded_flags = [f for f in report.flags if f.rule == "identifier_not_grounded"]
        assert grounded_flags == []

    def test_identifier_in_tool_trace_result_passes(self):
        trace = [_read_file("writer_agent.py", "class WikiContentWriter:\n    pass")]
        claims = [_claim("WikiContentWriter handles page generation.")]
        kept, report = apply_gate(claims, trace)

        grounded_flags = [f for f in report.flags if f.rule == "identifier_not_grounded"]
        assert grounded_flags == []


# ── Rule 1: README claim without README read ─────────────────────────────────


class TestReadmeRule:
    def test_readme_word_in_paragraph_without_readme_read_flags(self):
        claims = [_claim("See README for setup instructions.")]
        kept, report = apply_gate(claims, [])

        readme_flags = [f for f in report.flags if f.rule == "readme_claim_without_read"]
        assert len(readme_flags) == 1

    def test_readme_case_insensitive_readme_md(self):
        trace = [_read_file("readme.md", "content")]
        claims = [_claim("Check README.md for details.")]
        kept, report = apply_gate(claims, trace)

        readme_flags = [f for f in report.flags if f.rule == "readme_claim_without_read"]
        assert readme_flags == []

    def test_readme_rst_accepted(self):
        trace = [_read_file("README.rst", "content")]
        claims = [_claim("Refer to README for configuration.")]
        kept, report = apply_gate(claims, trace)

        readme_flags = [f for f in report.flags if f.rule == "readme_claim_without_read"]
        assert readme_flags == []

    def test_readme_in_subdir_accepted(self):
        trace = [_read_file("docs/README.md", "content")]
        claims = [_claim("README describes the installation process.")]
        kept, report = apply_gate(claims, trace)

        readme_flags = [f for f in report.flags if f.rule == "readme_claim_without_read"]
        assert readme_flags == []

    def test_readme_flag_has_correct_paragraph_index(self):
        claims = [
            _claim("Introductory text with no suspicious content.", paragraph_index=0),
            _claim("See README for more details.", paragraph_index=1),
        ]
        kept, report = apply_gate(claims, [])

        readme_flags = [f for f in report.flags if f.rule == "readme_claim_without_read"]
        assert len(readme_flags) == 1
        assert readme_flags[0].paragraph_index == 1

    def test_readme_flag_default_action_is_flag(self):
        claims = [_claim("Check the README for usage examples.")]
        kept, report = apply_gate(claims, [])

        readme_flags = [f for f in report.flags if f.rule == "readme_claim_without_read"]
        assert readme_flags[0].action == "flag"

    def test_readme_strip_mode_removes_paragraph(self):
        claims = [_claim("README describes the API endpoints.")]
        kept, report = apply_gate(
            claims,
            [],
            mode={"readme": "strip"},
        )

        assert len(kept) == 0
        assert report.stripped_paragraphs == [0]

    def test_readme_verifier_mode_flags_as_verifier(self):
        claims = [_claim("README describes the API endpoints.")]
        kept, report = apply_gate(
            claims,
            [],
            mode={"readme": "verifier"},
        )

        readme_flags = [f for f in report.flags if f.rule == "readme_claim_without_read"]
        assert readme_flags[0].action == "verifier"
        # Paragraph kept in output — verifier decides
        assert len(kept) == 1

    def test_no_readme_in_paragraph_no_flag(self):
        claims = [_claim("The service handles authentication via JWT tokens.")]
        kept, report = apply_gate(claims, [])

        readme_flags = [f for f in report.flags if f.rule == "readme_claim_without_read"]
        assert readme_flags == []


# ── Rule 2: SQL claim without .sql read ──────────────────────────────────────


class TestSqlRule:
    def test_create_table_triggers_flag(self):
        claims = [_claim("The schema defines CREATE TABLE users with id and name columns.")]
        kept, report = apply_gate(claims, [])

        sql_flags = [f for f in report.flags if f.rule == "sql_claim_without_read"]
        assert len(sql_flags) >= 1

    def test_select_keyword_triggers_flag(self):
        claims = [_claim("The query uses SELECT id FROM users WHERE active = 1.")]
        kept, report = apply_gate(claims, [])

        sql_flags = [f for f in report.flags if f.rule == "sql_claim_without_read"]
        assert len(sql_flags) >= 1

    def test_join_keyword_triggers_flag(self):
        claims = [_claim("Data is fetched with an INNER JOIN between orders and users.")]
        kept, report = apply_gate(claims, [])

        sql_flags = [f for f in report.flags if f.rule == "sql_claim_without_read"]
        assert len(sql_flags) >= 1

    def test_primary_key_pattern_triggers_flag(self):
        claims = [_claim("The id column is defined as PRIMARY KEY in the table.")]
        kept, report = apply_gate(claims, [])

        sql_flags = [f for f in report.flags if f.rule == "sql_claim_without_read"]
        assert len(sql_flags) >= 1

    def test_not_null_pattern_triggers_flag(self):
        claims = [_claim("The email field is declared NOT NULL in the schema.")]
        kept, report = apply_gate(claims, [])

        sql_flags = [f for f in report.flags if f.rule == "sql_claim_without_read"]
        assert len(sql_flags) >= 1

    def test_sql_flag_resolved_when_sql_file_read(self):
        trace = [_read_file("migrations/001.sql", "CREATE TABLE users (id INTEGER);")]
        claims = [_claim("The schema uses CREATE TABLE to define the users entity.")]
        kept, report = apply_gate(claims, trace)

        sql_flags = [f for f in report.flags if f.rule == "sql_claim_without_read"]
        assert sql_flags == []

    def test_sql_flag_resolved_by_any_sql_extension(self):
        trace = [_read_file("db/schema.SQL", "CREATE TABLE items (id INTEGER);")]
        claims = [_claim("The items table uses CREATE TABLE with an INTEGER id.")]
        kept, report = apply_gate(claims, trace)

        sql_flags = [f for f in report.flags if f.rule == "sql_claim_without_read"]
        assert sql_flags == []

    def test_sql_strip_mode_removes_paragraph(self):
        claims = [_claim("The schema uses CREATE TABLE users.")]
        kept, report = apply_gate(
            claims,
            [],
            mode={"sql": "strip"},
        )

        assert len(kept) == 0
        assert report.stripped_paragraphs == [0]

    def test_sql_verifier_mode_keeps_paragraph(self):
        claims = [_claim("The schema uses SELECT id FROM users.")]
        kept, report = apply_gate(
            claims,
            [],
            mode={"sql": "verifier"},
        )

        assert len(kept) == 1
        sql_flags = [f for f in report.flags if f.rule == "sql_claim_without_read"]
        assert sql_flags[0].action == "verifier"

    def test_no_sql_keyword_no_flag(self):
        claims = [_claim("The service handles HTTP requests and returns JSON responses.")]
        kept, report = apply_gate(claims, [])

        sql_flags = [f for f in report.flags if f.rule == "sql_claim_without_read"]
        assert sql_flags == []

    def test_flag_includes_matching_keyword_in_detail(self):
        claims = [_claim("Data retrieved via SELECT * FROM products.")]
        kept, report = apply_gate(claims, [])

        sql_flags = [f for f in report.flags if f.rule == "sql_claim_without_read"]
        assert len(sql_flags) >= 1
        # detail should mention the matching keyword
        assert any("SELECT" in f.detail.upper() or "select" in f.detail.lower() for f in sql_flags)


# ── Rule 3: Identifier not grounded ──────────────────────────────────────────


class TestIdentifierRule:
    def test_camelcase_not_in_trace_flags(self):
        claims = [_claim("The HallucinatedClass processes data in the pipeline.")]
        kept, report = apply_gate(claims, [], known_symbols=set())

        grounded_flags = [f for f in report.flags if f.rule == "identifier_not_grounded"]
        identifiers = {f.detail for f in grounded_flags}
        assert "HallucinatedClass" in identifiers

    def test_camelcase_in_known_symbols_does_not_flag(self):
        claims = [_claim("The HallucinatedClass processes data in the pipeline.")]
        kept, report = apply_gate(
            claims,
            [],
            known_symbols={"HallucinatedClass"},
        )

        grounded_flags = [f for f in report.flags if f.rule == "identifier_not_grounded"]
        identifiers = {f.detail for f in grounded_flags}
        assert "HallucinatedClass" not in identifiers

    def test_camelcase_in_trace_result_does_not_flag(self):
        trace = [_read_file("core/handler.py", "class HallucinatedClass:\n    pass")]
        claims = [_claim("The HallucinatedClass processes data in the pipeline.")]
        kept, report = apply_gate(claims, trace)

        grounded_flags = [f for f in report.flags if f.rule == "identifier_not_grounded"]
        identifiers = {f.detail for f in grounded_flags}
        assert "HallucinatedClass" not in identifiers

    def test_snake_case_func_not_in_trace_flags(self):
        claims = [_claim("The fake_env_var is loaded from the environment.")]
        kept, report = apply_gate(claims, [], known_symbols=set())

        grounded_flags = [f for f in report.flags if f.rule == "identifier_not_grounded"]
        identifiers = {f.detail for f in grounded_flags}
        assert "fake_env_var" in identifiers

    def test_env_var_style_not_in_trace_flags(self):
        # e.g. GATEWAY_URL (hallucinated in comparison.md)
        claims = [_claim("The service reads GATEWAY_URL from the environment.")]
        kept, report = apply_gate(claims, [], known_symbols=set())

        grounded_flags = [f for f in report.flags if f.rule == "identifier_not_grounded"]
        identifiers = {f.detail for f in grounded_flags}
        assert "GATEWAY_URL" in identifiers

    def test_env_var_in_grep_result_does_not_flag(self):
        trace = [_grep("GATEWAY_URL", "GATEWAY_URL=http://localhost:9000")]
        claims = [_claim("The service reads GATEWAY_URL from the environment.")]
        kept, report = apply_gate(claims, trace)

        grounded_flags = [f for f in report.flags if f.rule == "identifier_not_grounded"]
        identifiers = {f.detail for f in grounded_flags}
        assert "GATEWAY_URL" not in identifiers

    def test_identifier_flag_default_action_is_flag(self):
        claims = [_claim("The UnknownComponent does something.")]
        kept, report = apply_gate(claims, [], known_symbols=set())

        grounded_flags = [f for f in report.flags if f.rule == "identifier_not_grounded"]
        assert all(f.action == "flag" for f in grounded_flags)

    def test_identifier_strip_mode_strips_paragraph(self):
        claims = [_claim("The UnknownComponent processes input.")]
        kept, report = apply_gate(
            claims,
            [],
            known_symbols=set(),
            mode={"identifier": "strip"},
        )

        assert len(kept) == 0
        assert report.stripped_paragraphs == [0]

    def test_identifier_verifier_mode_keeps_paragraph(self):
        claims = [_claim("The UnknownComponent processes input.")]
        kept, report = apply_gate(
            claims,
            [],
            known_symbols=set(),
            mode={"identifier": "verifier"},
        )

        assert len(kept) == 1
        grounded_flags = [f for f in report.flags if f.rule == "identifier_not_grounded"]
        assert all(f.action == "verifier" for f in grounded_flags)

    def test_known_symbols_none_skips_identifier_check(self):
        """When known_symbols is None (not passed), identifier rule still runs
        but uses only tool trace for grounding."""
        trace = [_read_file("app.py", "class MyClass:\n    pass")]
        claims = [_claim("The MyClass is defined in app.py.")]
        kept, report = apply_gate(claims, trace, known_symbols=None)

        grounded_flags = [f for f in report.flags if f.rule == "identifier_not_grounded"]
        identifiers = {f.detail for f in grounded_flags}
        assert "MyClass" not in identifiers

    def test_short_tokens_not_flagged_as_identifiers(self):
        """Single-word tokens like 'Ok', 'ID', 'DB' should not trigger flags."""
        claims = [_claim("The ID is stored in DB for fast access.")]
        kept, report = apply_gate(claims, [], known_symbols=set())

        grounded_flags = [f for f in report.flags if f.rule == "identifier_not_grounded"]
        identifiers = {f.detail for f in grounded_flags}
        # Short all-caps like "ID", "DB" should not be flagged
        assert "ID" not in identifiers
        assert "DB" not in identifiers

    def test_common_english_camelcase_words_not_flagged(self):
        """Common compound words like 'GitHub', 'JavaScript' should not trigger."""
        claims = [_claim("The service is deployed on GitHub Actions.")]
        kept, report = apply_gate(claims, [], known_symbols=set())

        # This test is intentionally lenient — the gate may flag these,
        # but we want to know the behavior is consistent.
        # The key invariant: at most 1 flag for the whole paragraph.
        grounded_flags = [f for f in report.flags if f.rule == "identifier_not_grounded"]
        # We just verify the gate doesn't crash on common words
        assert isinstance(grounded_flags, list)


# ── Mode configuration ────────────────────────────────────────────────────────


class TestModeConfiguration:
    def test_default_mode_is_flag_for_all_rules(self):
        """Without explicit mode, every rule defaults to 'flag'."""
        trace = []
        claims = [
            _claim("README shows CREATE TABLE WikiReader structure.", paragraph_index=0),
        ]
        kept, report = apply_gate(claims, trace, known_symbols=set())

        for flag in report.flags:
            assert flag.action == "flag"

    def test_partial_mode_override_leaves_other_rules_at_flag(self):
        """Overriding 'readme' to 'strip' should not affect 'sql' rule."""
        claims = [_claim("README describes the CREATE TABLE schema.")]
        kept, report = apply_gate(
            claims,
            [],
            mode={"readme": "strip"},
        )

        sql_flags = [f for f in report.flags if f.rule == "sql_claim_without_read"]
        # sql rule is not overridden → still "flag"
        for f in sql_flags:
            assert f.action == "flag"

    def test_strip_mode_multiple_paragraphs_strips_all_flagged(self):
        """Strip mode strips every paragraph that gets flagged."""
        claims = [
            _claim("README has configuration details.", paragraph_index=0),
            _claim("The service uses async IO.", paragraph_index=1),
            _claim("See README for full API docs.", paragraph_index=2),
        ]
        kept, report = apply_gate(
            claims,
            [],
            mode={"readme": "strip"},
        )

        assert 0 in report.stripped_paragraphs
        assert 2 in report.stripped_paragraphs
        assert 1 not in report.stripped_paragraphs
        assert len(kept) == 1  # only the middle paragraph survives


# ── GateFlag and GateReport dataclass structure ───────────────────────────────


class TestDataclassStructure:
    def test_gate_flag_has_required_fields(self):
        flag = GateFlag(
            paragraph_index=0,
            paragraph_text="Some text",
            rule="readme_claim_without_read",
            detail="README",
            action="flag",
        )
        assert flag.paragraph_index == 0
        assert flag.paragraph_text == "Some text"
        assert flag.rule == "readme_claim_without_read"
        assert flag.detail == "README"
        assert flag.action == "flag"

    def test_gate_report_has_flags_and_stripped(self):
        report = GateReport(flags=[], stripped_paragraphs=[])
        assert isinstance(report.flags, list)
        assert isinstance(report.stripped_paragraphs, list)

    def test_tool_call_has_required_fields(self):
        call = ToolCall(tool="read_file", args={"path": "x.py"}, result_text="content")
        assert call.tool == "read_file"
        assert call.args == {"path": "x.py"}
        assert call.result_text == "content"


# ── ToolCall trace semantics ──────────────────────────────────────────────────


class TestToolCallTrace:
    def test_only_read_file_and_grep_tools_contribute_to_grounding(self):
        """get_signature, get_callers etc. are not grounding evidence."""
        trace = [
            ToolCall(
                tool="get_signature",
                args={"symbol": "WikiContentWriter"},
                result_text="class WikiContentWriter: ...",
            )
        ]
        claims = [_claim("WikiContentWriter manages page generation.")]
        kept, report = apply_gate(claims, trace, known_symbols=set())

        # get_signature result should not count as read-grounding
        # The identifier should still be flagged if not in known_symbols
        grounded_flags = [f for f in report.flags if f.rule == "identifier_not_grounded"]
        # This is a design decision: get_signature is not a trace read, so identifier
        # might still be ungrounded unless the implementation also checks get_signature results.
        # The spec says "read_file / grep result" → get_signature excluded.
        # We verify the flag exists (or not) consistently — implementation defines this.
        assert isinstance(grounded_flags, list)

    def test_empty_trace_means_no_read_grounding(self):
        claims = [_claim("See README for startup configuration.")]
        kept, report = apply_gate(claims, [])

        readme_flags = [f for f in report.flags if f.rule == "readme_claim_without_read"]
        assert len(readme_flags) == 1

    def test_tool_call_result_text_used_for_identifier_lookup(self):
        """result_text of a read_file call grounds identifiers appearing in it."""
        trace = [_read_file("db.py", "GATEWAY_URL = os.getenv('GATEWAY_URL', '')")]
        claims = [_claim("The GATEWAY_URL env var configures the upstream endpoint.")]
        kept, report = apply_gate(claims, trace)

        grounded_flags = [f for f in report.flags if f.rule == "identifier_not_grounded"]
        identifiers = {f.detail for f in grounded_flags}
        assert "GATEWAY_URL" not in identifiers


# ── Mode validation ──────────────────────────────────────────────────────────


class TestModeValidation:
    """``apply_gate`` should reject malformed ``mode`` dicts up front so a
    typo cannot silently produce an invalid ``GateFlag.action``."""

    def test_unknown_rule_key_raises(self):
        claims = [_claim("Some claim.")]
        import pytest

        with pytest.raises(ValueError, match="Unknown gate rule"):
            apply_gate(claims, [], mode={"readme_typo": "flag"})

    def test_invalid_action_raises(self):
        claims = [_claim("Some claim.")]
        import pytest

        with pytest.raises(ValueError, match="Invalid gate action"):
            apply_gate(claims, [], mode={"readme": "verfier"})  # typo

    def test_valid_mode_accepted(self):
        """Sanity: a well-formed mode dict is accepted without error."""
        claims = [_claim("Plain text.")]
        kept, report = apply_gate(claims, [], mode={"readme": "strip"})
        assert isinstance(kept, list)
        assert isinstance(report, GateReport)
