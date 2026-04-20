"""
Unit tests for app/core/code_graph/graph_query_service.py

Uses synthetic NetworkX graphs — no real FAISS/LLM/network calls.
"""

from collections import defaultdict
from unittest.mock import MagicMock, patch

import networkx as nx
import pytest

from app.core.code_graph.graph_query_service import (
    GraphQueryService,
    RelationshipResult,
    SymbolResult,
)


# ---------------------------------------------------------------------------
# Graph helpers
# ---------------------------------------------------------------------------


def _make_digraph():
    """Create a minimal DiGraph with index attributes."""
    g = nx.DiGraph()
    g._node_index = {}
    g._simple_name_index = {}
    g._full_name_index = {}
    g._name_index = defaultdict(list)
    g._suffix_index = defaultdict(list)
    return g


def _add_symbol(g, node_id, name, stype="class", file_path="src/auth.py", language="python", **extras):
    attrs = {
        "symbol_name": name,
        "symbol_type": stype,
        "file_path": file_path,
        "rel_path": file_path,
        "language": language,
    }
    attrs.update(extras)
    g.add_node(node_id, **attrs)
    # Update indexes
    g._node_index[(name, file_path, language)] = node_id
    g._name_index[name].append(node_id)
    g._full_name_index[name] = node_id
    return node_id


def _basic_graph():
    """A 3-node DiGraph with typical symbols."""
    g = _make_digraph()
    _add_symbol(g, "n::auth.py::AuthService", "AuthService", "class", "src/auth.py", "python")
    _add_symbol(g, "n::auth.py::login", "login", "function", "src/auth.py", "python")
    _add_symbol(g, "n::util.py::Logger", "Logger", "class", "src/util.py", "python")
    return g


def _graph_with_edges():
    """MultiDiGraph with inheritance and calls edges (get_relationships requires MultiDiGraph or compatible)."""
    g = nx.MultiDiGraph()
    g._node_index = {}
    g._simple_name_index = {}
    g._full_name_index = {}
    g._name_index = defaultdict(list)
    g._suffix_index = defaultdict(list)

    def add_sym(node_id, name, stype="class", file_path="src/auth.py", language="python"):
        g.add_node(node_id, symbol_name=name, symbol_type=stype, file_path=file_path,
                   rel_path=file_path, language=language)
        g._node_index[(name, file_path, language)] = node_id
        g._name_index[name].append(node_id)
        g._full_name_index[name] = node_id

    add_sym("n::auth.py::AuthService", "AuthService", "class", "src/auth.py")
    add_sym("n::auth.py::login", "login", "function", "src/auth.py")
    add_sym("n::base.py::BaseService", "BaseService", "class", "src/base.py")
    g.add_edge("n::auth.py::AuthService", "n::base.py::BaseService", relationship_type="inheritance")
    g.add_edge("n::auth.py::login", "n::auth.py::AuthService", relationship_type="calls")
    return g


# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------


def test_init_with_empty_graph():
    g = _make_digraph()
    svc = GraphQueryService(g)
    assert svc.graph is g
    assert svc.fts is None


def test_init_reads_indexes_from_graph():
    g = _basic_graph()
    svc = GraphQueryService(g)
    assert len(svc._name_index) > 0


def test_init_with_fts_none():
    g = _make_digraph()
    svc = GraphQueryService(g, fts_index=None)
    assert svc.fts is None


# ---------------------------------------------------------------------------
# resolve_symbol — strategies 1–5 (no FTS)
# ---------------------------------------------------------------------------


def test_resolve_symbol_by_full_name_index():
    g = _basic_graph()
    svc = GraphQueryService(g)
    node_id = svc.resolve_symbol("AuthService")
    assert node_id == "n::auth.py::AuthService"


def test_resolve_symbol_with_file_and_language():
    g = _basic_graph()
    svc = GraphQueryService(g)
    node_id = svc.resolve_symbol("AuthService", file_path="src/auth.py", language="python")
    assert node_id == "n::auth.py::AuthService"


def test_resolve_symbol_returns_none_for_unknown():
    g = _basic_graph()
    svc = GraphQueryService(g)
    result = svc.resolve_symbol("NoSuchSymbol")
    assert result is None


def test_resolve_symbol_empty_name():
    g = _basic_graph()
    svc = GraphQueryService(g)
    result = svc.resolve_symbol("")
    assert result is None


def test_resolve_symbol_via_suffix_index():
    g = _make_digraph()
    g._suffix_index = defaultdict(list)
    node_id = "n::auth.py::AuthService"
    g.add_node(node_id, symbol_name="AuthService", symbol_type="class")
    g._suffix_index["AuthService"].append(node_id)
    svc = GraphQueryService(g)
    result = svc.resolve_symbol("AuthService")
    assert result == node_id


def test_resolve_symbol_via_name_index():
    g = _make_digraph()
    node_id = "n::auth.py::AuthService"
    g.add_node(node_id, symbol_name="AuthService", symbol_type="class")
    g._name_index["AuthService"].append(node_id)
    svc = GraphQueryService(g)
    result = svc.resolve_symbol("AuthService")
    assert result == node_id


# ---------------------------------------------------------------------------
# resolve_symbol — FTS5 fallback (strategy 6)
# ---------------------------------------------------------------------------


def test_resolve_symbol_fts5_fallback_when_not_in_indexes():
    g = _make_digraph()
    node_id = "n::auth.py::AuthService"
    g.add_node(node_id, symbol_name="AuthService", symbol_type="class")
    # No indexes populated — force FTS5 path
    mock_fts = MagicMock()
    mock_fts.is_open = True
    mock_doc = MagicMock()
    mock_doc.metadata = {"node_id": node_id}
    mock_fts.search_smart.return_value = [mock_doc]

    svc = GraphQueryService(g, fts_index=mock_fts)
    result = svc.resolve_symbol("AuthService")
    assert result == node_id


def test_resolve_symbol_fts5_fallback_exception_handled():
    g = _make_digraph()
    mock_fts = MagicMock()
    mock_fts.is_open = True
    mock_fts.search_smart.side_effect = RuntimeError("FTS5 crashed")

    svc = GraphQueryService(g, fts_index=mock_fts)
    result = svc.resolve_symbol("AuthService")
    assert result is None


# ---------------------------------------------------------------------------
# _select_best
# ---------------------------------------------------------------------------


def test_select_best_returns_none_for_empty():
    g = _make_digraph()
    svc = GraphQueryService(g)
    assert svc._select_best([]) is None


def test_select_best_returns_single():
    g = _make_digraph()
    svc = GraphQueryService(g)
    assert svc._select_best(["n::x"]) == "n::x"


def test_select_best_prefers_file_match():
    g = _make_digraph()
    g.add_node("n1", rel_path="src/auth.py")
    g.add_node("n2", rel_path="src/util.py")
    svc = GraphQueryService(g)
    best = svc._select_best(["n1", "n2"], file_path="src/auth.py")
    assert best == "n1"


def test_select_best_falls_back_to_first():
    g = _make_digraph()
    g.add_node("n1")
    g.add_node("n2")
    svc = GraphQueryService(g)
    best = svc._select_best(["n1", "n2"])
    assert best == "n1"


# ---------------------------------------------------------------------------
# search — brute-force path (no FTS)
# ---------------------------------------------------------------------------


def test_search_brute_force_finds_exact_name():
    g = _basic_graph()
    svc = GraphQueryService(g)
    results = svc.search("AuthService", k=5)
    names = [r.symbol_name for r in results]
    assert "AuthService" in names


def test_search_brute_force_returns_symbol_results():
    g = _basic_graph()
    svc = GraphQueryService(g)
    results = svc.search("Logger", k=5)
    assert all(isinstance(r, SymbolResult) for r in results)


def test_search_brute_force_filters_by_type():
    g = _basic_graph()
    svc = GraphQueryService(g)
    results = svc.search("auth", k=10, symbol_types=frozenset({"function"}))
    for r in results:
        assert r.symbol_type == "function"


def test_search_brute_force_excludes_types():
    g = _basic_graph()
    svc = GraphQueryService(g)
    results = svc.search("auth", k=10, exclude_types=frozenset({"function"}))
    for r in results:
        assert r.symbol_type != "function"


def test_search_brute_force_no_match():
    g = _basic_graph()
    svc = GraphQueryService(g)
    results = svc.search("xyznotfound", k=5)
    assert results == []


def test_search_brute_force_path_prefix_filter():
    g = _basic_graph()
    svc = GraphQueryService(g)
    results = svc.search("auth", k=10, path_prefix="src/auth")
    for r in results:
        assert r.rel_path.startswith("src/auth")


def test_search_brute_force_keyword_hit_in_name():
    g = _basic_graph()
    svc = GraphQueryService(g)
    results = svc.search("log", k=5)
    names = [r.symbol_name for r in results]
    assert "Logger" in names


def test_search_uses_fts5_when_available():
    g = _basic_graph()
    mock_fts = MagicMock()
    mock_fts.is_open = True
    mock_doc = MagicMock()
    mock_doc.metadata = {
        "node_id": "n::auth.py::AuthService",
        "symbol_name": "AuthService",
        "symbol_type": "class",
        "layer": "",
        "file_path": "src/auth.py",
        "rel_path": "src/auth.py",
        "search_score": 0.9,
    }
    mock_fts.search_smart.return_value = [mock_doc]

    svc = GraphQueryService(g, fts_index=mock_fts)
    results = svc.search("AuthService", k=5)
    assert len(results) >= 1
    assert results[0].symbol_name == "AuthService"


# ---------------------------------------------------------------------------
# get_relationships
# ---------------------------------------------------------------------------


def test_get_relationships_returns_empty_for_unknown_node():
    g = _graph_with_edges()
    svc = GraphQueryService(g)
    result = svc.get_relationships("nonexistent_node")
    assert result == []


def test_get_relationships_outgoing():
    g = _graph_with_edges()
    svc = GraphQueryService(g)
    results = svc.get_relationships("n::auth.py::AuthService", direction="outgoing")
    rel_types = [r.relationship_type for r in results]
    assert "inheritance" in rel_types


def test_get_relationships_incoming():
    g = _graph_with_edges()
    svc = GraphQueryService(g)
    results = svc.get_relationships("n::auth.py::AuthService", direction="incoming")
    rel_types = [r.relationship_type for r in results]
    assert "calls" in rel_types


def test_get_relationships_both_directions():
    g = _graph_with_edges()
    svc = GraphQueryService(g)
    results = svc.get_relationships("n::auth.py::AuthService", direction="both")
    assert len(results) >= 2


def test_get_relationships_respects_max_depth():
    g = _graph_with_edges()
    svc = GraphQueryService(g)
    results = svc.get_relationships("n::auth.py::AuthService", max_depth=1)
    for r in results:
        assert r.hop_distance <= 1


def test_get_relationships_respects_max_results():
    g = _graph_with_edges()
    svc = GraphQueryService(g)
    results = svc.get_relationships("n::auth.py::AuthService", max_results=1)
    assert len(results) <= 1


def test_get_relationships_returns_relationship_results():
    g = _graph_with_edges()
    svc = GraphQueryService(g)
    results = svc.get_relationships("n::auth.py::AuthService")
    assert all(isinstance(r, RelationshipResult) for r in results)


def test_get_relationships_sorted_by_hop_distance():
    g = _graph_with_edges()
    svc = GraphQueryService(g)
    results = svc.get_relationships("n::auth.py::AuthService", max_depth=3)
    hops = [r.hop_distance for r in results]
    assert hops == sorted(hops)


# ---------------------------------------------------------------------------
# resolve_and_traverse
# ---------------------------------------------------------------------------


def test_resolve_and_traverse_success():
    g = _graph_with_edges()
    svc = GraphQueryService(g)
    node_id, rels = svc.resolve_and_traverse("AuthService")
    assert node_id == "n::auth.py::AuthService"
    assert isinstance(rels, list)


def test_resolve_and_traverse_not_found():
    g = _make_digraph()
    svc = GraphQueryService(g)
    node_id, rels = svc.resolve_and_traverse("NonExistent")
    assert node_id is None
    assert rels == []


# ---------------------------------------------------------------------------
# query (JQL execution)
# ---------------------------------------------------------------------------


def test_query_empty_expression_returns_empty():
    g = _basic_graph()
    svc = GraphQueryService(g)
    results = svc.query("")
    assert results == []


def test_query_type_filter_brute_force():
    g = _basic_graph()
    svc = GraphQueryService(g)
    results = svc.query("type:class")
    for r in results:
        assert r.symbol_type == "class"


def test_query_type_filter_function():
    g = _basic_graph()
    svc = GraphQueryService(g)
    results = svc.query("type:function")
    assert any(r.symbol_name == "login" for r in results)


def test_query_name_filter():
    g = _basic_graph()
    svc = GraphQueryService(g)
    results = svc.query("name:AuthService")
    assert any(r.symbol_name == "AuthService" for r in results)


def test_query_respects_limit():
    g = _basic_graph()
    svc = GraphQueryService(g)
    results = svc.query("type:class limit:1")
    assert len(results) <= 1


def test_query_connections_post_filter():
    g = _graph_with_edges()
    svc = GraphQueryService(g)
    # AuthService has 2 edges
    results = svc.query("connections:>0")
    # At least some connected nodes should pass
    assert isinstance(results, list)


def test_query_with_fts5_text_clause():
    g = _basic_graph()
    mock_fts = MagicMock()
    mock_fts.is_open = True
    mock_doc = MagicMock()
    mock_doc.metadata = {
        "node_id": "n::auth.py::AuthService",
        "symbol_name": "AuthService",
        "symbol_type": "class",
        "layer": "",
        "file_path": "src/auth.py",
        "rel_path": "src/auth.py",
        "search_score": 1.0,
    }
    mock_fts.search_smart.return_value = [mock_doc]

    svc = GraphQueryService(g, fts_index=mock_fts)
    results = svc.query("type:class text:authentication")
    assert isinstance(results, list)


# ---------------------------------------------------------------------------
# _jql_full_scan
# ---------------------------------------------------------------------------


def test_jql_full_scan_all_nodes():
    g = _basic_graph()
    svc = GraphQueryService(g)
    results = svc._jql_full_scan(k=10)
    assert len(results) == 3


def test_jql_full_scan_with_type_filter():
    g = _basic_graph()
    svc = GraphQueryService(g)
    results = svc._jql_full_scan(k=10, symbol_types=frozenset({"class"}))
    assert all(r.symbol_type == "class" for r in results)


def test_jql_full_scan_with_path_prefix():
    g = _basic_graph()
    svc = GraphQueryService(g)
    results = svc._jql_full_scan(k=10, path_prefix="src/auth")
    for r in results:
        assert r.rel_path.startswith("src/auth")


def test_jql_full_scan_with_name_filter_exact():
    g = _basic_graph()
    svc = GraphQueryService(g)
    results = svc._jql_full_scan(k=10, name_filter="AuthService")
    assert any(r.symbol_name == "AuthService" for r in results)


def test_jql_full_scan_with_name_filter_glob():
    g = _basic_graph()
    svc = GraphQueryService(g)
    results = svc._jql_full_scan(k=10, name_filter="Auth*")
    assert any(r.symbol_name == "AuthService" for r in results)


# ---------------------------------------------------------------------------
# _apply_post_filters
# ---------------------------------------------------------------------------


def test_apply_post_filters_connections_clause():
    g = _graph_with_edges()
    svc = GraphQueryService(g)

    # Build a simple query with connections:>0
    from app.core.code_graph.jql_parser import JQLClause, JQLQuery, ClauseOp
    jql = JQLQuery(
        post_filters=[JQLClause("connections", ClauseOp.GT, "0")]
    )
    all_results = svc._jql_full_scan(k=10)
    filtered = svc._apply_post_filters(all_results, jql)
    for r in filtered:
        assert r.connections > 0


def test_apply_post_filters_has_rel():
    g = _graph_with_edges()
    svc = GraphQueryService(g)

    from app.core.code_graph.jql_parser import JQLClause, JQLQuery, ClauseOp
    jql = JQLQuery(
        post_filters=[JQLClause("has_rel", ClauseOp.EQ, "inheritance")]
    )
    all_results = svc._jql_full_scan(k=10)
    filtered = svc._apply_post_filters(all_results, jql)
    assert isinstance(filtered, list)


def test_apply_post_filters_related_unresolvable():
    g = _basic_graph()
    svc = GraphQueryService(g)

    from app.core.code_graph.jql_parser import JQLClause, JQLQuery, ClauseOp
    jql = JQLQuery(
        post_filters=[JQLClause("related", ClauseOp.EQ, "NonExistentSymbol")]
    )
    all_results = svc._jql_full_scan(k=10)
    filtered = svc._apply_post_filters(all_results, jql)
    assert filtered == []


# ---------------------------------------------------------------------------
# _matches_path helper
# ---------------------------------------------------------------------------


def test_matches_path_prefix():
    g = _make_digraph()
    svc = GraphQueryService(g)
    assert svc._matches_path("src/auth/handler.py", "src/auth")


def test_matches_path_exact():
    g = _make_digraph()
    svc = GraphQueryService(g)
    assert svc._matches_path("src/auth", "src/auth")


def test_matches_path_mismatch():
    g = _make_digraph()
    svc = GraphQueryService(g)
    assert not svc._matches_path("src/util/helper.py", "src/auth")


def test_matches_path_glob():
    g = _make_digraph()
    svc = GraphQueryService(g)
    assert svc._matches_path("src/auth/handler.py", "src/auth/*")


# ---------------------------------------------------------------------------
# stats
# ---------------------------------------------------------------------------


def test_stats_returns_dict():
    g = _basic_graph()
    svc = GraphQueryService(g)
    s = svc.stats()
    assert isinstance(s, dict)
    assert "graph_nodes" in s
    assert s["graph_nodes"] == 3


def test_stats_with_fts5():
    g = _basic_graph()
    mock_fts = MagicMock()
    mock_fts.is_open = True
    mock_fts.node_count = 3
    svc = GraphQueryService(g, fts_index=mock_fts)
    s = svc.stats()
    assert s["fts5_available"] is True


def test_stats_without_fts5():
    g = _basic_graph()
    svc = GraphQueryService(g)
    s = svc.stats()
    assert s["fts5_available"] is False


# ---------------------------------------------------------------------------
# FTS5-backed JQL searches via mocked fts index
# ---------------------------------------------------------------------------


def _make_fts_doc(node_id, name, stype="class", layer="", rp="src/auth.py"):
    doc = MagicMock()
    doc.metadata = {
        "node_id": node_id,
        "symbol_name": name,
        "symbol_type": stype,
        "layer": layer,
        "file_path": rp,
        "rel_path": rp,
        "search_score": 1.0,
    }
    return doc


def _make_fts_row(node_id, name, stype="class", layer="", rp="src/auth.py"):
    """Row that behaves like a dict (for FTS path searches)."""
    return {
        "node_id": node_id,
        "symbol_name": name,
        "symbol_type": stype,
        "layer": layer,
        "file_path": rp,
        "rel_path": rp,
    }


def test_jql_layer_search_via_fts():
    g = _basic_graph()
    mock_fts = MagicMock()
    mock_fts.is_open = True
    row = _make_fts_row("n::auth.py::AuthService", "AuthService", "class", "core_type")
    mock_fts.search_by_layer.return_value = [row]

    svc = GraphQueryService(g, fts_index=mock_fts)
    results = svc._jql_layer_search("core_type", k=5)
    assert isinstance(results, list)
    assert len(results) == 1
    assert results[0].symbol_name == "AuthService"


def test_jql_layer_search_fts_exception_falls_back():
    g = _basic_graph()
    mock_fts = MagicMock()
    mock_fts.is_open = True
    mock_fts.search_by_layer.side_effect = RuntimeError("layer failed")

    svc = GraphQueryService(g, fts_index=mock_fts)
    results = svc._jql_layer_search("core_type", k=10)
    # Falls back to _jql_full_scan
    assert isinstance(results, list)


def test_jql_path_search_via_fts():
    g = _basic_graph()
    mock_fts = MagicMock()
    mock_fts.is_open = True
    mock_fts.search_by_path_prefix.return_value = [
        _make_fts_row("n::auth.py::AuthService", "AuthService")
    ]

    svc = GraphQueryService(g, fts_index=mock_fts)
    results = svc._jql_path_search("src/auth", k=5)
    assert isinstance(results, list)
    assert len(results) == 1


def test_jql_path_search_fts_exception_falls_back():
    g = _basic_graph()
    mock_fts = MagicMock()
    mock_fts.is_open = True
    mock_fts.search_by_path_prefix.side_effect = RuntimeError("path failed")

    svc = GraphQueryService(g, fts_index=mock_fts)
    results = svc._jql_path_search("src/auth", k=10)
    assert isinstance(results, list)


def test_jql_type_search_via_fts():
    g = _basic_graph()
    mock_fts = MagicMock()
    mock_fts.is_open = True
    doc = MagicMock()
    doc.metadata = {
        "node_id": "n::auth.py::AuthService",
        "symbol_name": "AuthService",
        "symbol_type": "class",
        "layer": "",
        "file_path": "src/auth.py",
        "rel_path": "src/auth.py",
        "search_score": 1.0,
    }
    mock_fts.search_by_type.return_value = [doc]

    svc = GraphQueryService(g, fts_index=mock_fts)
    results = svc._jql_type_search(frozenset({"class"}), k=5)
    assert isinstance(results, list)
    assert len(results) == 1


def test_jql_type_search_fts_exception_falls_back():
    g = _basic_graph()
    mock_fts = MagicMock()
    mock_fts.is_open = True
    mock_fts.search_by_type.side_effect = RuntimeError("type failed")

    svc = GraphQueryService(g, fts_index=mock_fts)
    results = svc._jql_type_search(frozenset({"class"}), k=10)
    assert isinstance(results, list)


def test_fts5_search_fallback_to_brute_force_on_exception():
    g = _basic_graph()
    mock_fts = MagicMock()
    mock_fts.is_open = True
    mock_fts.search_smart.side_effect = RuntimeError("FTS5 exploded")

    svc = GraphQueryService(g, fts_index=mock_fts)
    results = svc._search_fts5("auth", k=5)
    # Should fall back to brute force
    assert isinstance(results, list)


def test_search_fts5_path_prefix_filter():
    g = _basic_graph()
    mock_fts = MagicMock()
    mock_fts.is_open = True
    doc = MagicMock()
    doc.metadata = {
        "node_id": "n::auth.py::AuthService",
        "symbol_name": "AuthService",
        "symbol_type": "class",
        "layer": "",
        "file_path": "src/auth/handler.py",
        "rel_path": "src/auth/handler.py",
        "search_score": 1.0,
    }
    mock_fts.search_smart.return_value = [doc]

    svc = GraphQueryService(g, fts_index=mock_fts)
    results = svc._search_fts5("auth", k=5, path_prefix="src/auth")
    assert isinstance(results, list)


def test_search_fts5_type_filter_exclusion():
    g = _basic_graph()
    mock_fts = MagicMock()
    mock_fts.is_open = True
    doc = MagicMock()
    doc.metadata = {
        "node_id": "n::auth.py::login",
        "symbol_name": "login",
        "symbol_type": "function",
        "layer": "",
        "file_path": "src/auth.py",
        "rel_path": "src/auth.py",
        "search_score": 1.0,
    }
    mock_fts.search_smart.return_value = [doc]

    svc = GraphQueryService(g, fts_index=mock_fts)
    results = svc._search_fts5("login", k=5, exclude_types=frozenset({"function"}))
    assert all(r.symbol_type != "function" for r in results)


# ---------------------------------------------------------------------------
# _filter_related (BFS graph traversal)
# ---------------------------------------------------------------------------


def test_filter_related_finds_reachable_nodes():
    g = _graph_with_edges()
    svc = GraphQueryService(g)
    all_results = svc._jql_full_scan(k=10)
    filtered = svc._filter_related(all_results, "AuthService")
    # BFS from AuthService (both directions): reaches BaseService (outgoing inheritance)
    # and login (incoming calls). AuthService itself is also reachable via login -> AuthService.
    filtered_ids = [r.node_id for r in filtered]
    assert "n::base.py::BaseService" in filtered_ids
    assert "n::auth.py::login" in filtered_ids


def test_filter_related_with_direction_outgoing():
    g = _graph_with_edges()
    svc = GraphQueryService(g)
    all_results = svc._jql_full_scan(k=10)
    filtered = svc._filter_related(all_results, "AuthService", direction="outgoing")
    # Outgoing from AuthService: AuthService -> BaseService (inheritance)
    filtered_ids = [r.node_id for r in filtered]
    assert "n::base.py::BaseService" in filtered_ids
    assert "n::auth.py::login" not in filtered_ids


def test_filter_related_with_direction_incoming():
    g = _graph_with_edges()
    svc = GraphQueryService(g)
    all_results = svc._jql_full_scan(k=10)
    filtered = svc._filter_related(all_results, "AuthService", direction="incoming")
    # Incoming to AuthService: login -> AuthService (calls)
    filtered_ids = [r.node_id for r in filtered]
    assert "n::auth.py::login" in filtered_ids
    assert "n::base.py::BaseService" not in filtered_ids


def test_filter_related_with_edge_types():
    g = _graph_with_edges()
    svc = GraphQueryService(g)
    all_results = svc._jql_full_scan(k=10)
    filtered = svc._filter_related(all_results, "AuthService", edge_types=["inheritance"])
    # Only inheritance edges: AuthService -> BaseService; login uses "calls" so excluded
    filtered_ids = [r.node_id for r in filtered]
    assert "n::base.py::BaseService" in filtered_ids
    assert "n::auth.py::login" not in filtered_ids


# ---------------------------------------------------------------------------
# _normalise_edge_types
# ---------------------------------------------------------------------------


def test_normalise_edge_types_direct():
    g = _make_digraph()
    svc = GraphQueryService(g)
    result = svc._normalise_edge_types(["inheritance"])
    assert "inheritance" in result


def test_normalise_edge_types_alias():
    g = _make_digraph()
    svc = GraphQueryService(g)
    result = svc._normalise_edge_types(["inherits"])
    assert "inheritance" in result


def test_normalise_edge_types_canonical_reverse():
    g = _make_digraph()
    svc = GraphQueryService(g)
    result = svc._normalise_edge_types(["inheritance"])
    # Should also include the alias that maps to it
    assert "inherits" in result


def test_normalise_edge_types_multiple():
    g = _make_digraph()
    svc = GraphQueryService(g)
    result = svc._normalise_edge_types(["inherits", "calls"])
    assert "inheritance" in result
    assert "calls" in result


# ---------------------------------------------------------------------------
# _filter_has_relationship
# ---------------------------------------------------------------------------


def test_filter_has_relationship_with_matching_edges():
    g = _graph_with_edges()
    svc = GraphQueryService(g)
    all_results = svc._jql_full_scan(k=10)
    filtered = svc._filter_has_relationship(all_results, ["inheritance"])
    names = [r.symbol_name for r in filtered]
    assert "AuthService" in names


def test_filter_has_relationship_no_match():
    g = _basic_graph()
    svc = GraphQueryService(g)
    all_results = svc._jql_full_scan(k=10)
    filtered = svc._filter_has_relationship(all_results, ["nonexistent_rel_type"])
    assert filtered == []


# ---------------------------------------------------------------------------
# _iter_edge_dicts
# ---------------------------------------------------------------------------


def test_iter_edge_dicts_digraph():
    g = nx.DiGraph()
    g.add_edge("A", "B", relationship_type="calls")
    svc = GraphQueryService(g)
    edges = svc._iter_edge_dicts("A", "B")
    assert len(edges) == 1
    assert edges[0]["relationship_type"] == "calls"


def test_iter_edge_dicts_multidigraph():
    g = nx.MultiDiGraph()
    g.add_edge("A", "B", relationship_type="calls")
    g.add_edge("A", "B", relationship_type="references")
    svc = GraphQueryService(g)
    edges = svc._iter_edge_dicts("A", "B")
    assert len(edges) == 2


def test_iter_edge_dicts_missing():
    g = nx.DiGraph()
    g.add_node("A")
    g.add_node("B")
    svc = GraphQueryService(g)
    edges = svc._iter_edge_dicts("A", "B")
    assert edges == []


# ---------------------------------------------------------------------------
# _edge_rel_type
# ---------------------------------------------------------------------------


def test_edge_rel_type_relationship_type_key():
    assert GraphQueryService._edge_rel_type({"relationship_type": "inheritance"}) == "inheritance"


def test_edge_rel_type_type_key():
    assert GraphQueryService._edge_rel_type({"type": "calls"}) == "calls"


def test_edge_rel_type_empty():
    assert GraphQueryService._edge_rel_type({}) == ""


# ---------------------------------------------------------------------------
# query — FTS5-backed routes with mocked fts
# ---------------------------------------------------------------------------


def test_query_name_with_fts5():
    g = _basic_graph()
    mock_fts = MagicMock()
    mock_fts.is_open = True
    doc = MagicMock()
    doc.metadata = {
        "node_id": "n::auth.py::AuthService",
        "symbol_name": "AuthService",
        "symbol_type": "class",
        "layer": "",
        "file_path": "src/auth.py",
        "rel_path": "src/auth.py",
        "search_score": 1.0,
    }
    mock_fts.search_by_name.return_value = [doc]

    svc = GraphQueryService(g, fts_index=mock_fts)
    results = svc.query("name:AuthService")
    assert isinstance(results, list)


def test_query_layer_with_fts5():
    g = _basic_graph()
    mock_fts = MagicMock()
    mock_fts.is_open = True
    row = _make_fts_row("n::auth.py::AuthService", "AuthService", "class", "core_type")
    mock_fts.search_by_layer.return_value = [row]

    svc = GraphQueryService(g, fts_index=mock_fts)
    results = svc.query("layer:core_type")
    assert isinstance(results, list)


def test_query_path_with_fts5():
    g = _basic_graph()
    mock_fts = MagicMock()
    mock_fts.is_open = True
    mock_fts.search_by_path_prefix.return_value = [
        _make_fts_row("n::auth.py::AuthService", "AuthService")
    ]

    svc = GraphQueryService(g, fts_index=mock_fts)
    results = svc.query("file:src/auth/*")
    assert isinstance(results, list)


def test_query_type_with_fts5():
    g = _basic_graph()
    mock_fts = MagicMock()
    mock_fts.is_open = True
    doc = MagicMock()
    doc.metadata = {
        "node_id": "n::auth.py::AuthService",
        "symbol_name": "AuthService",
        "symbol_type": "class",
        "layer": "",
        "file_path": "src/auth.py",
        "rel_path": "src/auth.py",
        "search_score": 1.0,
    }
    mock_fts.search_by_type.return_value = [doc]

    svc = GraphQueryService(g, fts_index=mock_fts)
    results = svc.query("type:class")
    assert isinstance(results, list)


def test_query_related_after_resolve():
    """Test query with related: filter where symbol resolves."""
    g = _graph_with_edges()
    svc = GraphQueryService(g)
    # AuthService is in name index
    results = svc.query('related:"AuthService"')
    assert isinstance(results, list)


def test_query_jql_name_glob():
    g = _basic_graph()
    svc = GraphQueryService(g)
    results = svc.query("name:Auth*")
    assert isinstance(results, list)
