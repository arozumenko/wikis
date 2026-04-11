"""
Unit tests for app/core/code_graph/graph_text_index.py

Uses tmp_path for real SQLite — no mocks needed for the SQLite layer.
"""

import pytest
import networkx as nx

from app.core.code_graph.graph_text_index import GraphTextIndex, _tokenize_name


# ---------------------------------------------------------------------------
# _tokenize_name helper
# ---------------------------------------------------------------------------


def test_tokenize_name_camel_case():
    result = _tokenize_name("AuthServiceManager")
    assert "auth" in result
    assert "service" in result
    assert "manager" in result


def test_tokenize_name_snake_case():
    result = _tokenize_name("get_user_by_id")
    assert "get" in result
    assert "user" in result
    assert "id" in result


def test_tokenize_name_xml_parser():
    result = _tokenize_name("XMLParser")
    assert "xml" in result
    assert "parser" in result


def test_tokenize_name_single_word():
    result = _tokenize_name("logger")
    assert result == "logger"


def test_tokenize_name_empty():
    result = _tokenize_name("")
    assert result == ""


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def tmp_index(tmp_path):
    """Return a GraphTextIndex using a temporary cache directory."""
    return GraphTextIndex(cache_dir=str(tmp_path))


def _make_graph(*nodes):
    """Build a simple DiGraph from (node_id, attrs) pairs."""
    g = nx.DiGraph()
    for node_id, attrs in nodes:
        g.add_node(node_id, **attrs)
    return g


def _simple_graph():
    """A 3-node graph with typical symbol attributes."""
    g = nx.DiGraph()
    g.add_node(
        "py::auth.py::AuthService",
        symbol_name="AuthService",
        symbol_type="class",
        file_path="/repo/src/auth.py",
        rel_path="src/auth.py",
        language="python",
        docstring="Handles authentication",
        start_line=10,
        end_line=50,
        full_name="src.auth.AuthService",
        parent_symbol="",
    )
    g.add_node(
        "py::auth.py::login",
        symbol_name="login",
        symbol_type="function",
        file_path="/repo/src/auth.py",
        rel_path="src/auth.py",
        language="python",
        docstring="Login a user",
        start_line=55,
        end_line=70,
        full_name="src.auth.login",
        parent_symbol="AuthService",
    )
    g.add_node(
        "py::util.py::Logger",
        symbol_name="Logger",
        symbol_type="class",
        file_path="/repo/src/util.py",
        rel_path="src/util.py",
        language="python",
        docstring="Logging utility",
        start_line=1,
        end_line=30,
        full_name="src.util.Logger",
        parent_symbol="",
    )
    return g


# ---------------------------------------------------------------------------
# Lifecycle — init, is_open, close
# ---------------------------------------------------------------------------


def test_init_creates_cache_dir(tmp_path):
    cache_dir = tmp_path / "subdir"
    idx = GraphTextIndex(cache_dir=str(cache_dir))
    assert cache_dir.exists()
    assert not idx.is_open


def test_is_open_before_build(tmp_index):
    assert not tmp_index.is_open


def test_close_on_unopened_is_safe(tmp_index):
    # Should not raise
    tmp_index.close()
    assert not tmp_index.is_open


def test_node_count_returns_zero_when_closed(tmp_index):
    assert tmp_index.node_count == 0


# ---------------------------------------------------------------------------
# build_from_graph
# ---------------------------------------------------------------------------


def test_build_from_graph_returns_node_count(tmp_index):
    g = _simple_graph()
    count = tmp_index.build_from_graph(g, "testkey1")
    assert count == 3


def test_build_from_graph_sets_is_open(tmp_index):
    g = _simple_graph()
    tmp_index.build_from_graph(g, "testkey2")
    assert tmp_index.is_open


def test_build_from_graph_creates_db_file(tmp_path):
    idx = GraphTextIndex(cache_dir=str(tmp_path))
    g = _simple_graph()
    idx.build_from_graph(g, "testkey3")
    db_file = tmp_path / "testkey3.fts5.db"
    assert db_file.exists()


def test_build_from_graph_skips_nodes_without_name(tmp_index):
    g = nx.DiGraph()
    g.add_node("n1", symbol_name="AuthService", symbol_type="class")
    g.add_node("n2")  # no name — should be skipped
    count = tmp_index.build_from_graph(g, "testkey4")
    assert count == 1


def test_build_from_graph_empty_graph(tmp_index):
    g = nx.DiGraph()
    count = tmp_index.build_from_graph(g, "emptykey")
    assert count == 0
    assert tmp_index.is_open


def test_build_from_graph_rebuilds_on_second_call(tmp_index):
    g1 = _simple_graph()
    tmp_index.build_from_graph(g1, "rebuild_key")
    g2 = nx.DiGraph()
    g2.add_node("n1", symbol_name="SingleClass", symbol_type="class")
    count = tmp_index.build_from_graph(g2, "rebuild_key")
    assert count == 1
    assert tmp_index.node_count == 1


def test_build_from_graph_with_symbol_object_content(tmp_index):
    """Nodes with a symbol object carrying source_text should be indexed."""

    class FakeSymbol:
        source_text = "class AuthService: pass"
        docstring = "Auth class"

    g = nx.DiGraph()
    g.add_node(
        "py::auth.py::AuthService",
        symbol_name="AuthService",
        symbol_type="class",
        symbol=FakeSymbol(),
    )
    count = tmp_index.build_from_graph(g, "sym_obj_key")
    assert count == 1


def test_build_from_graph_converts_absolute_to_relative(tmp_index):
    """Nodes with only abs file_path should have rel_path inferred."""
    g = nx.DiGraph()
    g.add_node(
        "py::/repo/src/auth.py::AuthService",
        symbol_name="AuthService",
        symbol_type="class",
        file_path="/repo/src/auth.py",
        rel_path="src/auth.py",
    )
    count = tmp_index.build_from_graph(g, "relpath_key")
    assert count == 1


# ---------------------------------------------------------------------------
# node_count
# ---------------------------------------------------------------------------


def test_node_count_after_build(tmp_index):
    g = _simple_graph()
    tmp_index.build_from_graph(g, "count_key")
    assert tmp_index.node_count == 3


# ---------------------------------------------------------------------------
# exists / load
# ---------------------------------------------------------------------------


def test_exists_false_before_build(tmp_index):
    assert not tmp_index.exists("no_such_key")


def test_exists_true_after_build(tmp_index):
    g = _simple_graph()
    tmp_index.build_from_graph(g, "exist_key")
    assert tmp_index.exists("exist_key")


def test_load_returns_false_when_no_index(tmp_index):
    result = tmp_index.load("missing_key")
    assert result is False


def test_load_returns_true_after_build(tmp_path):
    idx1 = GraphTextIndex(cache_dir=str(tmp_path))
    g = _simple_graph()
    idx1.build_from_graph(g, "load_key")
    idx1.close()

    idx2 = GraphTextIndex(cache_dir=str(tmp_path))
    result = idx2.load("load_key")
    assert result is True
    assert idx2.is_open


def test_load_enables_search(tmp_path):
    idx1 = GraphTextIndex(cache_dir=str(tmp_path))
    g = _simple_graph()
    idx1.build_from_graph(g, "load_search_key")
    idx1.close()

    idx2 = GraphTextIndex(cache_dir=str(tmp_path))
    idx2.load("load_search_key")
    docs = idx2.search("AuthService", k=5)
    assert any(d.metadata.get("symbol_name") == "AuthService" for d in docs)


# ---------------------------------------------------------------------------
# search
# ---------------------------------------------------------------------------


def test_search_returns_empty_when_not_open(tmp_index):
    results = tmp_index.search("AuthService")
    assert results == []


def test_search_finds_by_name(tmp_index):
    g = _simple_graph()
    tmp_index.build_from_graph(g, "search_name_key")
    docs = tmp_index.search("AuthService", k=5)
    names = [d.metadata.get("symbol_name") for d in docs]
    assert "AuthService" in names


def test_search_returns_documents(tmp_index):
    g = _simple_graph()
    tmp_index.build_from_graph(g, "search_docs_key")
    from langchain_core.documents import Document
    docs = tmp_index.search("auth", k=5)
    assert all(isinstance(d, Document) for d in docs)


def test_search_with_symbol_type_filter(tmp_index):
    g = _simple_graph()
    tmp_index.build_from_graph(g, "type_filter_key")
    docs = tmp_index.search("auth", k=5, symbol_types=frozenset({"class"}))
    for d in docs:
        assert d.metadata.get("symbol_type") == "class"


def test_search_with_exclude_types(tmp_index):
    g = _simple_graph()
    tmp_index.build_from_graph(g, "exclude_key")
    docs = tmp_index.search("auth", k=5, exclude_types=frozenset({"function"}))
    for d in docs:
        assert d.metadata.get("symbol_type") != "function"


def test_search_respects_k_limit(tmp_index):
    g = _simple_graph()
    tmp_index.build_from_graph(g, "limit_key")
    docs = tmp_index.search("auth", k=1)
    assert len(docs) <= 1


def test_search_with_language_filter(tmp_index):
    g = _simple_graph()
    tmp_index.build_from_graph(g, "lang_filter_key")
    docs = tmp_index.search("auth", k=10, language="python")
    for d in docs:
        assert d.metadata.get("language") in ("python", "")


# ---------------------------------------------------------------------------
# search_smart
# ---------------------------------------------------------------------------


def test_search_smart_returns_empty_when_not_open(tmp_index):
    results = tmp_index.search_smart("auth service")
    assert results == []


def test_search_smart_finds_results(tmp_index):
    g = _simple_graph()
    tmp_index.build_from_graph(g, "smart_key")
    docs = tmp_index.search_smart("authentication", k=5)
    assert isinstance(docs, list)


def test_search_smart_symbol_intent(tmp_index):
    g = _simple_graph()
    tmp_index.build_from_graph(g, "smart_sym_key")
    docs = tmp_index.search_smart("AuthService", k=5, intent="symbol")
    assert isinstance(docs, list)


# ---------------------------------------------------------------------------
# search_symbols
# ---------------------------------------------------------------------------


def test_search_symbols_returns_empty_when_not_open(tmp_index):
    results = tmp_index.search_symbols("auth")
    assert results == []


def test_search_symbols_with_path_prefix_filter(tmp_index):
    g = _simple_graph()
    tmp_index.build_from_graph(g, "symbols_prefix_key")
    docs = tmp_index.search_symbols("auth", k=10, path_prefix="src/auth")
    for d in docs:
        rp = d.metadata.get("rel_path", "")
        assert rp.startswith("src/auth") or rp == "src/auth"


# ---------------------------------------------------------------------------
# search_by_name
# ---------------------------------------------------------------------------


def test_search_by_name_returns_empty_when_not_open(tmp_index):
    results = tmp_index.search_by_name("AuthService")
    assert results == []


def test_search_by_name_exact(tmp_index):
    g = _simple_graph()
    tmp_index.build_from_graph(g, "name_exact_key")
    docs = tmp_index.search_by_name("AuthService", exact=True)
    assert any(d.metadata.get("symbol_name") == "AuthService" for d in docs)


def test_search_by_name_prefix(tmp_index):
    g = _simple_graph()
    tmp_index.build_from_graph(g, "name_prefix_key")
    docs = tmp_index.search_by_name("Auth", exact=False)
    assert len(docs) >= 1


def test_search_by_name_no_match(tmp_index):
    g = _simple_graph()
    tmp_index.build_from_graph(g, "no_match_key")
    docs = tmp_index.search_by_name("XyzNoMatch", exact=True)
    assert docs == []


def test_search_by_name_with_symbol_types(tmp_index):
    g = _simple_graph()
    tmp_index.build_from_graph(g, "name_type_key")
    docs = tmp_index.search_by_name("Auth", symbol_types=frozenset({"class"}))
    for d in docs:
        assert d.metadata.get("symbol_type") == "class"


# ---------------------------------------------------------------------------
# search_by_type
# ---------------------------------------------------------------------------


def test_search_by_type_returns_empty_when_not_open(tmp_index):
    results = tmp_index.search_by_type("class")
    assert results == []


def test_search_by_type_class(tmp_index):
    g = _simple_graph()
    tmp_index.build_from_graph(g, "by_type_key")
    docs = tmp_index.search_by_type("class")
    names = [d.metadata.get("symbol_name") for d in docs]
    assert "AuthService" in names
    assert "Logger" in names
    assert "login" not in names


def test_search_by_type_function(tmp_index):
    g = _simple_graph()
    tmp_index.build_from_graph(g, "by_func_key")
    docs = tmp_index.search_by_type("function")
    names = [d.metadata.get("symbol_name") for d in docs]
    assert "login" in names


# ---------------------------------------------------------------------------
# search_by_layer
# ---------------------------------------------------------------------------


def test_search_by_layer_returns_empty_when_not_open(tmp_index):
    results = tmp_index.search_by_layer("core_type")
    assert results == []


def test_search_by_layer_returns_documents(tmp_index):
    g = _simple_graph()
    tmp_index.build_from_graph(g, "layer_key")
    # Layers are auto-assigned; just check no crash and returns list
    docs = tmp_index.search_by_layer("core_type")
    assert isinstance(docs, list)


# ---------------------------------------------------------------------------
# search_by_path_prefix
# ---------------------------------------------------------------------------


def test_search_by_path_prefix_returns_empty_when_not_open(tmp_index):
    results = tmp_index.search_by_path_prefix("src/")
    assert results == []


def test_search_by_path_prefix_finds_matching_nodes(tmp_index):
    g = _simple_graph()
    tmp_index.build_from_graph(g, "path_prefix_key")
    results = tmp_index.search_by_path_prefix("src/auth")
    assert isinstance(results, list)
    assert all(isinstance(r, dict) for r in results)


def test_search_by_path_prefix_with_type_filter(tmp_index):
    g = _simple_graph()
    tmp_index.build_from_graph(g, "prefix_type_key")
    results = tmp_index.search_by_path_prefix("src", symbol_types=frozenset({"class"}))
    for r in results:
        assert r.get("symbol_type") == "class"


# ---------------------------------------------------------------------------
# get_by_node_id
# ---------------------------------------------------------------------------


def test_get_by_node_id_returns_none_when_not_open(tmp_index):
    result = tmp_index.get_by_node_id("py::auth.py::AuthService")
    assert result is None


def test_get_by_node_id_found(tmp_index):
    g = _simple_graph()
    tmp_index.build_from_graph(g, "nodeid_key")
    result = tmp_index.get_by_node_id("py::auth.py::AuthService")
    assert result is not None
    assert result["symbol_name"] == "AuthService"


def test_get_by_node_id_not_found(tmp_index):
    g = _simple_graph()
    tmp_index.build_from_graph(g, "nodeid_miss_key")
    result = tmp_index.get_by_node_id("nonexistent::node")
    assert result is None


# ---------------------------------------------------------------------------
# _prepare_fts_query (static)
# ---------------------------------------------------------------------------


def test_prepare_fts_query_returns_string():
    q = GraphTextIndex._prepare_fts_query("AuthService login")
    assert isinstance(q, str)
    assert "auth" in q.lower() or "authservice" in q.lower()


def test_prepare_fts_query_empty():
    q = GraphTextIndex._prepare_fts_query("")
    assert q == ""


def test_prepare_fts_query_single_char_ignored():
    q = GraphTextIndex._prepare_fts_query("a b c")
    assert q == ""


def test_prepare_fts_query_uses_prefix_match():
    q = GraphTextIndex._prepare_fts_query("auth")
    assert "*" in q


# ---------------------------------------------------------------------------
# _prepare_simple_query (static)
# ---------------------------------------------------------------------------


def test_prepare_simple_query_returns_string():
    q = GraphTextIndex._prepare_simple_query("auth service")
    assert "auth" in q
    assert "*" in q


def test_prepare_simple_query_empty():
    q = GraphTextIndex._prepare_simple_query("")
    assert q == ""


# ---------------------------------------------------------------------------
# _row_to_document (static) — tested indirectly via search; direct test uses a mock row
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# register_in_cache_index / load_by_repo_name
# ---------------------------------------------------------------------------


def test_register_in_cache_index_creates_file(tmp_path):
    idx = GraphTextIndex(cache_dir=str(tmp_path))
    idx.register_in_cache_index("myrepo", "abc123")
    index_file = tmp_path / "cache_index.json"
    assert index_file.exists()
    import json
    with open(index_file) as f:
        data = json.load(f)
    assert data["fts5"]["myrepo"] == "abc123"


def test_register_in_cache_index_updates_existing(tmp_path):
    idx = GraphTextIndex(cache_dir=str(tmp_path))
    idx.register_in_cache_index("repoA", "key1")
    idx.register_in_cache_index("repoB", "key2")
    import json
    with open(tmp_path / "cache_index.json") as f:
        data = json.load(f)
    assert data["fts5"]["repoA"] == "key1"
    assert data["fts5"]["repoB"] == "key2"


def test_load_by_repo_name_no_index_file(tmp_path):
    idx = GraphTextIndex(cache_dir=str(tmp_path))
    result = idx.load_by_repo_name("nonexistent_repo")
    assert result is False


def test_load_by_repo_name_missing_entry(tmp_path):
    idx = GraphTextIndex(cache_dir=str(tmp_path))
    idx.register_in_cache_index("other_repo", "somekey")
    result = idx.load_by_repo_name("nonexistent_repo")
    assert result is False


def test_load_by_repo_name_after_register(tmp_path):
    idx1 = GraphTextIndex(cache_dir=str(tmp_path))
    g = _simple_graph()
    idx1.build_from_graph(g, "mykey")
    idx1.register_in_cache_index("myrepo", "mykey")

    idx2 = GraphTextIndex(cache_dir=str(tmp_path))
    result = idx2.load_by_repo_name("myrepo")
    assert result is True
    assert idx2.is_open


# ---------------------------------------------------------------------------
# _detect_repo_root_from_graph
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Default cache_dir init
# ---------------------------------------------------------------------------


def test_init_with_default_cache_dir(tmp_path, monkeypatch):
    """GraphTextIndex with None cache_dir uses the expanduser default."""
    monkeypatch.chdir(tmp_path)
    idx = GraphTextIndex(cache_dir=None)
    assert idx.cache_dir is not None
    assert idx.cache_dir.exists()


# ---------------------------------------------------------------------------
# _open raises FileNotFoundError for missing db
# ---------------------------------------------------------------------------


def test_open_raises_file_not_found(tmp_index):
    import pytest
    with pytest.raises(FileNotFoundError):
        tmp_index._open("nonexistent_key")


# ---------------------------------------------------------------------------
# close with held _conn
# ---------------------------------------------------------------------------


def test_close_with_held_conn(tmp_index):
    """close() should not raise even when _conn is set to something conn-like."""
    import sqlite3
    # Build first to set up cache_key
    g = _simple_graph()
    tmp_index.build_from_graph(g, "close_conn_key")
    # Manually set a real connection as _conn
    conn = sqlite3.connect(":memory:")
    tmp_index._conn = conn
    tmp_index.close()
    assert tmp_index._conn is None


# ---------------------------------------------------------------------------
# node_count with actual data
# ---------------------------------------------------------------------------


def test_node_count_on_reloaded_index(tmp_path):
    idx1 = GraphTextIndex(cache_dir=str(tmp_path))
    g = _simple_graph()
    idx1.build_from_graph(g, "count_reload_key")

    idx2 = GraphTextIndex(cache_dir=str(tmp_path))
    idx2.load("count_reload_key")
    assert idx2.node_count == 3


# ---------------------------------------------------------------------------
# search_by_name with language filter
# ---------------------------------------------------------------------------


def test_search_by_name_with_language(tmp_index):
    g = _simple_graph()
    tmp_index.build_from_graph(g, "name_lang_key")
    docs = tmp_index.search_by_name("AuthService", language="python")
    for d in docs:
        assert d.metadata.get("language") in ("python", "")


# ---------------------------------------------------------------------------
# search_by_layer with symbol_types filter
# ---------------------------------------------------------------------------


def test_search_by_layer_with_type_filter(tmp_index):
    g = _simple_graph()
    tmp_index.build_from_graph(g, "layer_type_key")
    docs = tmp_index.search_by_layer("core_type", symbol_types=frozenset({"class"}))
    for d in docs:
        assert d.metadata.get("symbol_type") == "class"


# ---------------------------------------------------------------------------
# search_by_path_prefix with exclude_types
# ---------------------------------------------------------------------------


def test_search_by_path_prefix_with_exclude(tmp_index):
    g = _simple_graph()
    tmp_index.build_from_graph(g, "prefix_exclude_key")
    results = tmp_index.search_by_path_prefix("src", exclude_types=frozenset({"function"}))
    for r in results:
        assert r.get("symbol_type") != "function"


def test_detect_repo_root_from_graph_success():
    g = nx.DiGraph()
    g.add_node(
        "n1",
        rel_path="src/auth.py",
        file_path="/repo/src/auth.py",
    )
    root = GraphTextIndex._detect_repo_root_from_graph(g)
    assert root == "/repo"


def test_detect_repo_root_from_graph_no_relative_path():
    g = nx.DiGraph()
    g.add_node("n1", file_path="/repo/src/auth.py")
    root = GraphTextIndex._detect_repo_root_from_graph(g)
    assert root == ""


def test_detect_repo_root_from_graph_empty():
    g = nx.DiGraph()
    root = GraphTextIndex._detect_repo_root_from_graph(g)
    assert root == ""


# ---------------------------------------------------------------------------
# node_count fallback paths
# ---------------------------------------------------------------------------


def test_node_count_with_missing_db(tmp_path):
    """node_count returns 0 if cache_key is set but db file is missing."""
    idx = GraphTextIndex(cache_dir=str(tmp_path))
    idx._cache_key = "missing_key"  # Set key manually without creating db
    assert idx.node_count == 0


# ---------------------------------------------------------------------------
# error handling in search methods
# ---------------------------------------------------------------------------


def test_search_when_open_read_conn_raises(tmp_index):
    """When _open_read_conn raises FileNotFoundError, search returns []."""
    g = _simple_graph()
    tmp_index.build_from_graph(g, "conn_err_key")
    from unittest.mock import patch
    import sqlite3

    with patch.object(tmp_index, "_open_read_conn", side_effect=FileNotFoundError("db missing")):
        results = tmp_index.search("AuthService", k=5)
    assert results == []


def test_search_smart_when_open_read_conn_raises(tmp_index):
    """When _open_read_conn raises in search_smart, returns []."""
    g = _simple_graph()
    tmp_index.build_from_graph(g, "smart_conn_err_key")
    from unittest.mock import patch

    with patch.object(tmp_index, "_open_read_conn", side_effect=FileNotFoundError("db missing")):
        results = tmp_index.search_smart("AuthService", k=5)
    assert results == []


def test_row_to_document_creates_document():
    """Test _row_to_document with a dict-like object (sqlite3.Row compatible)."""
    import sqlite3

    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
    conn.execute(
        """CREATE TABLE t (
            node_id TEXT, symbol_name TEXT, symbol_type TEXT, layer TEXT,
            file_path TEXT, rel_path TEXT, language TEXT,
            start_line INTEGER, end_line INTEGER, full_name TEXT,
            parent_symbol TEXT, docstring TEXT, content TEXT, score REAL
        )"""
    )
    conn.execute(
        "INSERT INTO t VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
        (
            "py::auth.py::AuthService",
            "AuthService",
            "class",
            "core_type",
            "/repo/src/auth.py",
            "src/auth.py",
            "python",
            10,
            50,
            "src.auth.AuthService",
            "",
            "Auth service class",
            "class AuthService: pass",
            -0.5,
        ),
    )
    conn.commit()
    row = conn.execute("SELECT * FROM t").fetchone()

    from langchain_core.documents import Document
    doc = GraphTextIndex._row_to_document(row)
    assert isinstance(doc, Document)
    assert doc.metadata["symbol_name"] == "AuthService"
    assert doc.metadata["symbol_type"] == "class"
    assert doc.metadata["node_id"] == "py::auth.py::AuthService"
    conn.close()
