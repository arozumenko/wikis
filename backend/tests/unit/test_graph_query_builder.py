"""
Unit tests for app/core/code_graph/graph_query_builder.py

Pure/near-pure functions — no external dependencies needed.
"""

import pytest

from app.core.code_graph.graph_query_builder import (
    CODE_SIGNIFICANT_WORDS,
    CODE_STOP_WORDS,
    EDGE_TYPE_ALIASES,
    GraphQueryBuilder,
    MODULE_KEYWORDS,
    ParsedNLQuery,
    RELATIONSHIP_KEYWORDS,
    TYPE_KEYWORDS,
    _escape_fts5_token,
    _split_identifier,
    extract_keywords,
    parse_natural_language,
)


# ---------------------------------------------------------------------------
# _split_identifier
# ---------------------------------------------------------------------------


def test_split_identifier_camel_case():
    parts = _split_identifier("AuthServiceManager")
    assert parts == ["auth", "service", "manager"]


def test_split_identifier_snake_case():
    parts = _split_identifier("get_user_by_id")
    assert parts == ["get", "user", "by", "id"]


def test_split_identifier_pascal_case():
    parts = _split_identifier("UserProfile")
    assert parts == ["user", "profile"]


def test_split_identifier_single_word():
    parts = _split_identifier("auth")
    assert parts == ["auth"]


def test_split_identifier_all_caps():
    parts = _split_identifier("XML")
    # Should not crash; may return one token
    assert isinstance(parts, list)
    assert len(parts) >= 1


def test_split_identifier_uppercase_then_lower():
    parts = _split_identifier("XMLParser")
    assert "xml" in parts
    assert "parser" in parts


# ---------------------------------------------------------------------------
# _escape_fts5_token
# ---------------------------------------------------------------------------


def test_escape_fts5_token_simple():
    result = _escape_fts5_token("auth")
    assert result == '"auth"'


def test_escape_fts5_token_with_quotes():
    result = _escape_fts5_token('say "hello"')
    assert '""' in result  # embedded quotes doubled


def test_escape_fts5_token_empty():
    result = _escape_fts5_token("")
    assert result == '""'


# ---------------------------------------------------------------------------
# extract_keywords
# ---------------------------------------------------------------------------


def test_extract_keywords_removes_stop_words():
    result = extract_keywords("How does authentication work?")
    assert "authentication" in result
    assert "how" not in result
    assert "does" not in result
    assert "work" not in result


def test_extract_keywords_preserves_code_significant_words():
    result = extract_keywords("What classes handle payment processing?")
    assert "classes" in result
    assert "handle" in result


def test_extract_keywords_splits_camel_case():
    result = extract_keywords("AuthServiceManager login flow")
    assert "authservicemanager" in result
    assert "auth" in result
    assert "service" in result
    assert "manager" in result


def test_extract_keywords_deduplicates():
    result = extract_keywords("auth auth auth")
    assert result.count("auth") == 1


def test_extract_keywords_min_length_filter():
    result = extract_keywords("a an the db")
    assert "a" not in result
    assert "an" not in result
    # "db" should be included (length >= 2)
    assert "db" in result


def test_extract_keywords_empty_string():
    result = extract_keywords("")
    assert result == []


def test_extract_keywords_keep_identifiers_false():
    result = extract_keywords("AuthService", keep_identifiers=False)
    # Should have "authservice" but not split parts
    assert "authservice" in result
    assert "auth" not in result


def test_extract_keywords_snake_case_split():
    result = extract_keywords("get_user_by_id")
    assert "get" in result or "user" in result  # split parts should appear


# ---------------------------------------------------------------------------
# TYPE_KEYWORDS
# ---------------------------------------------------------------------------


def test_type_keywords_map_contains_expected():
    assert TYPE_KEYWORDS["class"] == "class"
    assert TYPE_KEYWORDS["function"] == "function"
    assert TYPE_KEYWORDS["method"] == "method"
    assert TYPE_KEYWORDS["interface"] == "interface"
    assert TYPE_KEYWORDS["classes"] == "class"
    assert TYPE_KEYWORDS["functions"] == "function"


# ---------------------------------------------------------------------------
# RELATIONSHIP_KEYWORDS
# ---------------------------------------------------------------------------


def test_relationship_keywords_inherits():
    assert RELATIONSHIP_KEYWORDS["inherits"] == "inheritance"
    assert RELATIONSHIP_KEYWORDS["extends"] == "inheritance"


def test_relationship_keywords_implements():
    assert RELATIONSHIP_KEYWORDS["implements"] == "implementation"


def test_relationship_keywords_calls():
    assert RELATIONSHIP_KEYWORDS["calls"] == "calls"


# ---------------------------------------------------------------------------
# EDGE_TYPE_ALIASES
# ---------------------------------------------------------------------------


def test_edge_type_aliases_inherits():
    assert EDGE_TYPE_ALIASES["inherits"] == "inheritance"


def test_edge_type_aliases_implements():
    assert EDGE_TYPE_ALIASES["implements"] == "implementation"


# ---------------------------------------------------------------------------
# ParsedNLQuery.to_jql
# ---------------------------------------------------------------------------


def test_parsed_nl_query_to_jql_type_only():
    p = ParsedNLQuery(type_filter="class", text_keywords=[])
    jql = p.to_jql()
    assert "type:class" in jql


def test_parsed_nl_query_to_jql_with_text():
    p = ParsedNLQuery(type_filter="class", text_keywords=["auth"])
    jql = p.to_jql()
    assert "type:class" in jql
    assert "text:auth" in jql


def test_parsed_nl_query_to_jql_path_hint_no_wildcard():
    p = ParsedNLQuery(path_hint="src/auth")
    jql = p.to_jql()
    assert "file:src/auth/*" in jql


def test_parsed_nl_query_to_jql_path_hint_already_has_wildcard():
    p = ParsedNLQuery(path_hint="src/auth/*")
    jql = p.to_jql()
    assert "file:src/auth/*" in jql


def test_parsed_nl_query_to_jql_related_symbol():
    p = ParsedNLQuery(related_symbol="BaseHandler")
    jql = p.to_jql()
    assert "related:BaseHandler" in jql


def test_parsed_nl_query_to_jql_related_symbol_with_space():
    p = ParsedNLQuery(related_symbol="Base Handler")
    jql = p.to_jql()
    assert 'related:"Base Handler"' in jql


def test_parsed_nl_query_to_jql_relationship_hint():
    p = ParsedNLQuery(relationship_hint="inheritance")
    jql = p.to_jql()
    assert "has_rel:inheritance" in jql


def test_parsed_nl_query_to_jql_empty():
    p = ParsedNLQuery()
    assert p.to_jql() == ""


def test_parsed_nl_query_has_structural_filters_true():
    p = ParsedNLQuery(type_filter="class")
    assert p.has_structural_filters


def test_parsed_nl_query_has_structural_filters_false():
    p = ParsedNLQuery(text_keywords=["auth"])
    assert not p.has_structural_filters


# ---------------------------------------------------------------------------
# parse_natural_language
# ---------------------------------------------------------------------------


def test_parse_natural_language_extracts_type():
    result = parse_natural_language("What classes handle authentication?")
    assert result.type_filter == "class"


def test_parse_natural_language_extracts_function_type():
    result = parse_natural_language("List all functions in the auth module")
    assert result.type_filter == "function"


def test_parse_natural_language_extracts_path_hint_from_in():
    result = parse_natural_language("functions in src/auth")
    assert result.type_filter == "function"
    assert result.path_hint == "src/auth"


def test_parse_natural_language_extracts_relationship():
    result = parse_natural_language("classes that extend BaseHandler")
    assert result.type_filter == "class"
    assert result.relationship_hint == "inheritance"


def test_parse_natural_language_extracts_related_symbol():
    result = parse_natural_language("classes that extend BaseHandler")
    assert result.related_symbol == "BaseHandler"


def test_parse_natural_language_empty():
    result = parse_natural_language("")
    assert result.type_filter is None
    assert result.text_keywords == []


def test_parse_natural_language_whitespace_only():
    result = parse_natural_language("   ")
    assert result.type_filter is None


def test_parse_natural_language_detects_intent_architecture():
    result = parse_natural_language("architecture overview")
    assert result.intent == "architecture"


def test_parse_natural_language_detects_intent_security():
    result = parse_natural_language("auth security flow")
    assert result.intent == "security"


def test_parse_natural_language_text_keywords_non_structural():
    result = parse_natural_language("payment processing pipeline")
    assert "payment" in result.text_keywords or "processing" in result.text_keywords


def test_parse_natural_language_module_keyword_path():
    result = parse_natural_language("module payments")
    # "module" is a TYPE_KEYWORDS entry mapping to "module_doc", so type_filter is set
    # path_hint may or may not be set depending on token consumption order
    assert result.type_filter == "module_doc" or result.path_hint == "payments"


# ---------------------------------------------------------------------------
# GraphQueryBuilder.symbol_resolution
# ---------------------------------------------------------------------------


def test_symbol_resolution_simple():
    q = GraphQueryBuilder.symbol_resolution("AuthService")
    assert "authservice" in q.lower()
    assert "symbol_name" in q.lower()


def test_symbol_resolution_compound_name():
    q = GraphQueryBuilder.symbol_resolution("AuthServiceManager")
    assert "name_tokens" in q.lower()
    assert " OR " in q


def test_symbol_resolution_empty():
    q = GraphQueryBuilder.symbol_resolution("")
    assert q == ""


def test_symbol_resolution_single_word():
    q = GraphQueryBuilder.symbol_resolution("Logger")
    assert "logger" in q.lower()


# ---------------------------------------------------------------------------
# GraphQueryBuilder.concept_search
# ---------------------------------------------------------------------------


def test_concept_search_with_boost():
    q = GraphQueryBuilder.concept_search(["authentication", "token"], boost_docstring=True)
    assert "authentication" in q
    assert "docstring:" in q


def test_concept_search_no_boost():
    q = GraphQueryBuilder.concept_search(["authentication", "token"], boost_docstring=False)
    assert "authentication" in q
    assert "docstring:" not in q


def test_concept_search_empty_keywords():
    q = GraphQueryBuilder.concept_search([])
    assert q == ""


def test_concept_search_single_keyword():
    q = GraphQueryBuilder.concept_search(["auth"])
    assert '"auth"' in q or "auth" in q


# ---------------------------------------------------------------------------
# GraphQueryBuilder.phrase_search
# ---------------------------------------------------------------------------


def test_phrase_search_simple():
    q = GraphQueryBuilder.phrase_search("dependency injection")
    assert '"dependency injection"' in q


def test_phrase_search_with_column():
    q = GraphQueryBuilder.phrase_search("dependency injection", "docstring")
    assert q.startswith("docstring:")


def test_phrase_search_empty():
    q = GraphQueryBuilder.phrase_search("")
    assert q == ""


# ---------------------------------------------------------------------------
# GraphQueryBuilder.proximity_search
# ---------------------------------------------------------------------------


def test_proximity_search_basic():
    q = GraphQueryBuilder.proximity_search(["UserService", "authenticate"], 50)
    assert "NEAR(" in q
    assert "50" in q


def test_proximity_search_with_column():
    q = GraphQueryBuilder.proximity_search(["auth", "token"], 20, column="docstring")
    assert q.startswith("docstring:NEAR(")


def test_proximity_search_single_term_returns_empty():
    q = GraphQueryBuilder.proximity_search(["auth"], 10)
    assert q == ""


def test_proximity_search_empty_returns_empty():
    q = GraphQueryBuilder.proximity_search([], 10)
    assert q == ""


# ---------------------------------------------------------------------------
# GraphQueryBuilder.combined_symbol_and_concept
# ---------------------------------------------------------------------------


def test_combined_symbol_and_concept():
    q = GraphQueryBuilder.combined_symbol_and_concept(["AuthService"], ["authentication"])
    assert "authservice" in q.lower()
    assert "authentication" in q.lower()


def test_combined_symbol_and_concept_empty_both():
    q = GraphQueryBuilder.combined_symbol_and_concept([], [])
    assert q == ""


def test_combined_symbol_and_concept_symbols_only():
    q = GraphQueryBuilder.combined_symbol_and_concept(["AuthService"], [])
    assert "authservice" in q.lower()


def test_combined_symbol_and_concept_keywords_only():
    q = GraphQueryBuilder.combined_symbol_and_concept([], ["auth"])
    assert "auth" in q


# ---------------------------------------------------------------------------
# GraphQueryBuilder.from_natural_language
# ---------------------------------------------------------------------------


def test_from_natural_language_general_intent():
    q = GraphQueryBuilder.from_natural_language("authentication service", intent="general")
    assert q  # Non-empty result


def test_from_natural_language_symbol_intent():
    q = GraphQueryBuilder.from_natural_language("AuthService", intent="symbol")
    assert "symbol_name" in q.lower()


def test_from_natural_language_concept_intent():
    q = GraphQueryBuilder.from_natural_language("authentication flow", intent="concept")
    assert q


def test_from_natural_language_doc_intent():
    q = GraphQueryBuilder.from_natural_language("authentication overview", intent="doc")
    assert q
    assert "docstring" in q or "content" in q


def test_from_natural_language_empty():
    q = GraphQueryBuilder.from_natural_language("")
    assert q == ""


# ---------------------------------------------------------------------------
# GraphQueryBuilder.from_natural_language_parsed
# ---------------------------------------------------------------------------


def test_from_natural_language_parsed_returns_tuple():
    fts_q, parsed = GraphQueryBuilder.from_natural_language_parsed("auth classes")
    assert isinstance(fts_q, str)
    assert isinstance(parsed, ParsedNLQuery)


def test_from_natural_language_parsed_empty():
    fts_q, parsed = GraphQueryBuilder.from_natural_language_parsed("")
    assert fts_q == ""
    assert isinstance(parsed, ParsedNLQuery)


def test_from_natural_language_parsed_symbol_intent():
    fts_q, parsed = GraphQueryBuilder.from_natural_language_parsed("AuthService", intent="symbol")
    assert "symbol_name" in fts_q.lower()
    assert parsed.intent == "symbol"


# ---------------------------------------------------------------------------
# GraphQueryBuilder.for_vector_store
# ---------------------------------------------------------------------------


def test_for_vector_store_extracts_keywords():
    q = GraphQueryBuilder.for_vector_store("How does authentication work?")
    assert "authentication" in q


def test_for_vector_store_with_context_terms():
    q = GraphQueryBuilder.for_vector_store(
        "payment processing",
        context_terms=["billing", "stripe"],
    )
    assert "billing" in q
    assert "stripe" in q


def test_for_vector_store_no_duplicate_context():
    q = GraphQueryBuilder.for_vector_store("payment", context_terms=["payment"])
    # "payment" should appear only once
    assert q.count("payment") == 1


def test_for_vector_store_empty():
    q = GraphQueryBuilder.for_vector_store("")
    assert isinstance(q, str)


# ---------------------------------------------------------------------------
# GraphQueryBuilder.type_scoped_search
# ---------------------------------------------------------------------------


def test_type_scoped_search_returns_query_and_filter():
    fts_q, type_filter = GraphQueryBuilder.type_scoped_search(["auth"], "class")
    assert "auth" in fts_q
    assert type_filter == frozenset({"class"})


def test_type_scoped_search_lowercases_type():
    _, type_filter = GraphQueryBuilder.type_scoped_search(["test"], "Class")
    assert "class" in type_filter


# ---------------------------------------------------------------------------
# GraphQueryBuilder.path_scoped_concept
# ---------------------------------------------------------------------------


def test_path_scoped_concept_returns_query_and_prefix():
    fts_q, prefix = GraphQueryBuilder.path_scoped_concept(["auth"], "src/auth/")
    assert "auth" in fts_q
    assert prefix == "src/auth"  # strip trailing slash


def test_path_scoped_concept_strips_leading_slash():
    _, prefix = GraphQueryBuilder.path_scoped_concept(["test"], "/src/auth")
    assert not prefix.startswith("/")
