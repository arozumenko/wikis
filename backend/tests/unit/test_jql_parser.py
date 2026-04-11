"""
Unit tests for app/core/code_graph/jql_parser.py

Pure functions — no mocks required.
"""

import pytest

from app.core.code_graph.jql_parser import (
    ClauseOp,
    INDEX_FIELDS,
    JQLClause,
    JQLQuery,
    POST_FILTER_FIELDS,
    is_jql_expression,
    parse_jql,
    _parse_clause,
    _tokenize,
)


# ---------------------------------------------------------------------------
# ClauseOp values
# ---------------------------------------------------------------------------


def test_clause_op_values():
    assert ClauseOp.EQ.value == "eq"
    assert ClauseOp.GT.value == "gt"
    assert ClauseOp.GTE.value == "gte"
    assert ClauseOp.LT.value == "lt"
    assert ClauseOp.LTE.value == "lte"
    assert ClauseOp.MATCH.value == "match"


# ---------------------------------------------------------------------------
# JQLClause.matches_value
# ---------------------------------------------------------------------------


def test_clause_matches_value_exact():
    c = JQLClause(field="type", op=ClauseOp.EQ, value="class")
    assert c.matches_value("class")
    assert c.matches_value("CLASS")  # case-insensitive
    assert not c.matches_value("function")


def test_clause_matches_value_glob():
    c = JQLClause(field="file", op=ClauseOp.EQ, value="src/auth/*")
    assert c.matches_value("src/auth/handler.py")
    assert not c.matches_value("src/util/helper.py")


def test_clause_matches_value_question_mark_glob():
    c = JQLClause(field="name", op=ClauseOp.EQ, value="Auth?ervice")
    assert c.matches_value("AuthService")
    assert not c.matches_value("AuthXXervice")


def test_clause_matches_value_bracket_glob():
    c = JQLClause(field="name", op=ClauseOp.EQ, value="[Aa]uthService")
    assert c.matches_value("AuthService")
    assert c.matches_value("authService")


def test_clause_matches_value_match_op():
    c = JQLClause(field="text", op=ClauseOp.MATCH, value="authentication")
    assert c.matches_value("handles authentication logic")
    assert not c.matches_value("handles login logic")


def test_clause_matches_value_other_op_returns_false():
    c = JQLClause(field="connections", op=ClauseOp.GT, value="5")
    # GT is not EQ or MATCH — matches_value should return False
    assert not c.matches_value("10")


# ---------------------------------------------------------------------------
# JQLClause.matches_numeric
# ---------------------------------------------------------------------------


def test_clause_matches_numeric_gt():
    c = JQLClause(field="connections", op=ClauseOp.GT, value="10")
    assert c.matches_numeric(11)
    assert not c.matches_numeric(10)
    assert not c.matches_numeric(9)


def test_clause_matches_numeric_gte():
    c = JQLClause(field="connections", op=ClauseOp.GTE, value="10")
    assert c.matches_numeric(10)
    assert c.matches_numeric(11)
    assert not c.matches_numeric(9)


def test_clause_matches_numeric_lt():
    c = JQLClause(field="connections", op=ClauseOp.LT, value="10")
    assert c.matches_numeric(9)
    assert not c.matches_numeric(10)


def test_clause_matches_numeric_lte():
    c = JQLClause(field="connections", op=ClauseOp.LTE, value="10")
    assert c.matches_numeric(10)
    assert c.matches_numeric(9)
    assert not c.matches_numeric(11)


def test_clause_matches_numeric_eq():
    c = JQLClause(field="connections", op=ClauseOp.EQ, value="5")
    assert c.matches_numeric(5)
    assert not c.matches_numeric(6)


def test_clause_matches_numeric_invalid_value():
    c = JQLClause(field="connections", op=ClauseOp.GT, value="not_a_number")
    assert not c.matches_numeric(5)


def test_clause_matches_numeric_unknown_op():
    # MATCH op in matches_numeric returns False
    c = JQLClause(field="connections", op=ClauseOp.MATCH, value="5")
    assert not c.matches_numeric(5)


# ---------------------------------------------------------------------------
# JQLQuery properties
# ---------------------------------------------------------------------------


def test_jql_query_is_empty_when_no_clauses():
    q = JQLQuery()
    assert q.is_empty


def test_jql_query_has_text_clause_true():
    q = JQLQuery(index_clauses=[JQLClause("text", ClauseOp.MATCH, "auth")])
    assert q.has_text_clause
    assert q.text_value == "auth"


def test_jql_query_has_text_clause_false():
    q = JQLQuery(index_clauses=[JQLClause("type", ClauseOp.EQ, "class")])
    assert not q.has_text_clause
    assert q.text_value is None


def test_jql_query_type_values():
    q = JQLQuery(
        index_clauses=[
            JQLClause("type", ClauseOp.EQ, "Class"),
            JQLClause("type", ClauseOp.EQ, "INTERFACE"),
        ]
    )
    assert "class" in q.type_values
    assert "interface" in q.type_values


def test_jql_query_layer_value():
    q = JQLQuery(index_clauses=[JQLClause("layer", ClauseOp.EQ, "CORE_TYPE")])
    assert q.layer_value == "core_type"


def test_jql_query_layer_value_none():
    q = JQLQuery(index_clauses=[JQLClause("type", ClauseOp.EQ, "class")])
    assert q.layer_value is None


def test_jql_query_file_value():
    q = JQLQuery(index_clauses=[JQLClause("file", ClauseOp.EQ, "src/auth/*")])
    assert q.file_value == "src/auth/*"


def test_jql_query_file_value_none():
    q = JQLQuery()
    assert q.file_value is None


def test_jql_query_name_value():
    q = JQLQuery(index_clauses=[JQLClause("name", ClauseOp.EQ, "AuthService")])
    assert q.name_value == "AuthService"


def test_jql_query_name_value_none():
    q = JQLQuery()
    assert q.name_value is None


def test_jql_query_related_value():
    q = JQLQuery(post_filters=[JQLClause("related", ClauseOp.EQ, "BaseHandler")])
    assert q.related_value == "BaseHandler"


def test_jql_query_related_value_none():
    q = JQLQuery()
    assert q.related_value is None


def test_jql_query_direction_value_default():
    q = JQLQuery()
    assert q.direction_value == "both"


def test_jql_query_direction_value_set():
    q = JQLQuery(post_filters=[JQLClause("dir", ClauseOp.EQ, "INCOMING")])
    assert q.direction_value == "incoming"


def test_jql_query_has_rel_values():
    q = JQLQuery(
        post_filters=[
            JQLClause("has_rel", ClauseOp.EQ, "INHERITS"),
            JQLClause("has_rel", ClauseOp.EQ, "CALLS"),
        ]
    )
    vals = q.has_rel_values
    assert "inherits" in vals
    assert "calls" in vals


def test_jql_query_connections_clause():
    c = JQLClause("connections", ClauseOp.GT, "10")
    q = JQLQuery(post_filters=[c])
    assert q.connections_clause is c


def test_jql_query_connections_clause_none():
    q = JQLQuery()
    assert q.connections_clause is None


# ---------------------------------------------------------------------------
# _tokenize
# ---------------------------------------------------------------------------


def test_tokenize_single_clause():
    tokens = _tokenize("type:class")
    assert len(tokens) == 1
    assert tokens[0] == ("clause", "type:class")


def test_tokenize_multiple_clauses():
    tokens = _tokenize("type:class file:src/auth/*")
    types = [t for t, _ in tokens]
    assert types.count("clause") == 2


def test_tokenize_and_operator():
    tokens = _tokenize("type:class AND file:src/*")
    token_types = [t for t, _ in tokens]
    assert "AND" in token_types
    assert token_types.count("clause") == 2


def test_tokenize_or_operator():
    tokens = _tokenize("type:class OR type:interface")
    token_types = [t for t, _ in tokens]
    assert "OR" in token_types


def test_tokenize_case_insensitive_and():
    tokens = _tokenize("type:class and file:src/*")
    # The tokenizer upper-cases the remaining string before checking for AND,
    # so lowercase "and" is recognized as an AND operator token.
    token_types = [t for t, _ in tokens]
    assert "AND" in token_types


def test_tokenize_empty_string():
    tokens = _tokenize("")
    assert tokens == []


def test_tokenize_skips_unknown_tokens():
    # A clause without known field should be skipped if _CLAUSE_RE doesn't match
    tokens = _tokenize("!!!invalid")
    assert isinstance(tokens, list)


# ---------------------------------------------------------------------------
# _parse_clause
# ---------------------------------------------------------------------------


def test_parse_clause_exact():
    c = _parse_clause("type:class")
    assert c is not None
    assert c.field == "type"
    assert c.op == ClauseOp.EQ
    assert c.value == "class"


def test_parse_clause_quoted():
    c = _parse_clause('related:"Base Handler"')
    assert c is not None
    assert c.field == "related"
    assert c.value == "Base Handler"


def test_parse_clause_gt():
    c = _parse_clause("connections:>10")
    assert c is not None
    assert c.op == ClauseOp.GT
    assert c.value == "10"


def test_parse_clause_gte():
    c = _parse_clause("connections:>=10")
    assert c is not None
    assert c.op == ClauseOp.GTE


def test_parse_clause_lt():
    c = _parse_clause("connections:<5")
    assert c is not None
    assert c.op == ClauseOp.LT


def test_parse_clause_lte():
    c = _parse_clause("connections:<=5")
    assert c is not None
    assert c.op == ClauseOp.LTE


def test_parse_clause_text_gets_match_op():
    c = _parse_clause("text:authentication")
    assert c is not None
    assert c.op == ClauseOp.MATCH


def test_parse_clause_unknown_field_returns_none():
    c = _parse_clause("unknownfield:value")
    assert c is None


def test_parse_clause_no_match_returns_none():
    c = _parse_clause("not_a_clause")
    assert c is None


# ---------------------------------------------------------------------------
# parse_jql — empty and whitespace
# ---------------------------------------------------------------------------


def test_parse_jql_empty_string():
    q = parse_jql("")
    assert q.is_empty
    assert q.limit == 20


def test_parse_jql_whitespace_only():
    q = parse_jql("   ")
    assert q.is_empty


# ---------------------------------------------------------------------------
# parse_jql — single clauses
# ---------------------------------------------------------------------------


def test_parse_jql_type_class():
    q = parse_jql("type:class")
    assert "class" in q.type_values
    assert not q.is_empty


def test_parse_jql_file_glob():
    q = parse_jql("file:src/auth/*")
    assert q.file_value == "src/auth/*"


def test_parse_jql_name():
    q = parse_jql("name:AuthService")
    assert q.name_value == "AuthService"


def test_parse_jql_text_match():
    q = parse_jql("text:authentication")
    assert q.has_text_clause
    assert q.text_value == "authentication"
    assert q.index_clauses[0].op == ClauseOp.MATCH


def test_parse_jql_layer():
    q = parse_jql("layer:core_type")
    assert q.layer_value == "core_type"


def test_parse_jql_related():
    q = parse_jql('related:"BaseHandler"')
    assert q.related_value == "BaseHandler"


def test_parse_jql_dir():
    q = parse_jql("dir:incoming")
    assert q.direction_value == "incoming"


def test_parse_jql_has_rel():
    q = parse_jql("has_rel:inherits")
    assert "inherits" in q.has_rel_values


def test_parse_jql_connections_gt():
    q = parse_jql("connections:>10")
    assert q.connections_clause is not None
    assert q.connections_clause.op == ClauseOp.GT
    assert q.connections_clause.value == "10"


def test_parse_jql_limit():
    q = parse_jql("limit:5")
    assert q.limit == 5


def test_parse_jql_limit_clamped_min():
    q = parse_jql("limit:0")
    assert q.limit == 1


def test_parse_jql_limit_clamped_max():
    q = parse_jql("limit:9999")
    assert q.limit == 500


def test_parse_jql_limit_invalid_ignored():
    q = parse_jql("limit:abc")
    assert q.limit == 20  # unchanged default


# ---------------------------------------------------------------------------
# parse_jql — compound queries
# ---------------------------------------------------------------------------


def test_parse_jql_multiple_index_clauses():
    q = parse_jql("type:class file:src/auth/*")
    assert "class" in q.type_values
    assert q.file_value == "src/auth/*"


def test_parse_jql_index_and_post_filter():
    q = parse_jql('type:class related:"BaseHandler"')
    assert "class" in q.type_values
    assert q.related_value == "BaseHandler"


def test_parse_jql_full_example():
    q = parse_jql("type:class connections:>10 limit:5")
    assert "class" in q.type_values
    assert q.connections_clause is not None
    assert q.connections_clause.op == ClauseOp.GT
    assert q.limit == 5


def test_parse_jql_or_sets_is_or_flag():
    q = parse_jql("type:class OR type:interface")
    assert q.is_or
    assert "class" in q.type_values
    assert "interface" in q.type_values


def test_parse_jql_explicit_and():
    q = parse_jql("type:class AND file:src/*")
    assert not q.is_or
    assert "class" in q.type_values
    assert q.file_value == "src/*"


def test_parse_jql_related_dir_has_rel():
    q = parse_jql('related:"BaseHandler" dir:incoming has_rel:inherits')
    assert q.related_value == "BaseHandler"
    assert q.direction_value == "incoming"
    assert "inherits" in q.has_rel_values


def test_parse_jql_layer_text():
    q = parse_jql("layer:core_type text:authentication")
    assert q.layer_value == "core_type"
    assert q.has_text_clause
    assert q.text_value == "authentication"


# ---------------------------------------------------------------------------
# INDEX_FIELDS / POST_FILTER_FIELDS constants
# ---------------------------------------------------------------------------


def test_index_fields_contains_expected():
    assert "type" in INDEX_FIELDS
    assert "layer" in INDEX_FIELDS
    assert "file" in INDEX_FIELDS
    assert "name" in INDEX_FIELDS
    assert "text" in INDEX_FIELDS


def test_post_filter_fields_contains_expected():
    assert "related" in POST_FILTER_FIELDS
    assert "dir" in POST_FILTER_FIELDS
    assert "has_rel" in POST_FILTER_FIELDS
    assert "connections" in POST_FILTER_FIELDS


# ---------------------------------------------------------------------------
# is_jql_expression
# ---------------------------------------------------------------------------


def test_is_jql_expression_true_for_type():
    assert is_jql_expression("type:class")


def test_is_jql_expression_true_for_related():
    assert is_jql_expression("related:AuthService")


def test_is_jql_expression_false_for_plain_text():
    assert not is_jql_expression("how does authentication work")


def test_is_jql_expression_false_for_empty():
    assert not is_jql_expression("")


def test_is_jql_expression_false_for_none_like():
    assert not is_jql_expression("   ")


def test_is_jql_expression_true_for_unknown_field_pattern():
    # Even if field is unknown, regex still matches but we check _KNOWN_FIELDS
    assert not is_jql_expression("unknown:value")


def test_is_jql_expression_true_for_compound_query():
    assert is_jql_expression("type:class file:src/auth/* text:auth")
