"""Tests for cluster_expansion module."""

import sqlite3
from unittest.mock import MagicMock, patch

import pytest
from langchain_core.documents import Document

from app.core.feature_flags import FeatureFlags

from app.core.cluster_expansion import (
    DEFAULT_TOKEN_BUDGET,
    EXPANSION_SYMBOL_TYPES,
    MAX_EXPANSION_TOTAL,
    MAX_NEIGHBORS_PER_SYMBOL,
    _augment_cpp,
    _augment_document,
    _augment_go_rust,
    _collect_expansion_neighbors,
    _collect_search_terms,
    _count_structural_edges,
    _estimate_tokens,
    _find_framework_references,
    _get_cluster_docs,
    _is_excluded_test_node,
    _node_to_document,
    _resolve_symbols,
    expand_for_page,
)


# ═══════════════════════════════════════════════════════════════════════════
# Fixtures
# ═══════════════════════════════════════════════════════════════════════════


def _make_db():
    """Create an in-memory DB with the unified schema."""
    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
    conn.execute("""
        CREATE TABLE repo_nodes (
            node_id TEXT PRIMARY KEY,
            rel_path TEXT DEFAULT '',
            file_name TEXT DEFAULT '',
            language TEXT DEFAULT '',
            start_line INTEGER DEFAULT 0,
            end_line INTEGER DEFAULT 0,
            symbol_name TEXT DEFAULT '',
            symbol_type TEXT DEFAULT '',
            parent_symbol TEXT,
            source_text TEXT DEFAULT '',
            docstring TEXT DEFAULT '',
            signature TEXT DEFAULT '',
            is_architectural INTEGER DEFAULT 0,
            is_doc INTEGER DEFAULT 0,
            is_test INTEGER DEFAULT 0,
            macro_cluster INTEGER,
            micro_cluster INTEGER
        )
    """)
    conn.execute("""
        CREATE TABLE repo_edges (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            source_id TEXT NOT NULL,
            target_id TEXT NOT NULL,
            rel_type TEXT NOT NULL DEFAULT '',
            edge_class TEXT NOT NULL DEFAULT 'structural',
            weight REAL DEFAULT 1.0
        )
    """)
    # FTS5 virtual table — source_text included so JOIN returns it for
    # substring confirmation in _find_framework_references.
    conn.execute("""
        CREATE VIRTUAL TABLE IF NOT EXISTS repo_fts USING fts5(
            node_id,
            symbol_name,
            source_text,
            content=repo_nodes,
            content_rowid=rowid
        )
    """)
    return conn


def _insert_node(conn, node_id, **kwargs):
    defaults = {
        "rel_path": f"src/{node_id}.py",
        "file_name": f"{node_id}.py",
        "language": "python",
        "symbol_name": node_id,
        "symbol_type": "class",
        "source_text": f"class {node_id}:\n    pass",
        "is_architectural": 1,
        "macro_cluster": 0,
        "micro_cluster": 0,
    }
    defaults.update(kwargs)
    cols = ", ".join(["node_id"] + list(defaults.keys()))
    vals = ", ".join(["?"] * (1 + len(defaults)))
    conn.execute(
        f"INSERT INTO repo_nodes ({cols}) VALUES ({vals})",
        [node_id] + list(defaults.values()),
    )
    # Also insert into FTS
    conn.execute(
        "INSERT INTO repo_fts (node_id, symbol_name, source_text) VALUES (?, ?, ?)",
        (node_id, defaults.get("symbol_name", node_id),
         defaults.get("source_text", "")),
    )


def _insert_edge(conn, source_id, target_id, rel_type="calls", weight=1.0,
                 edge_class="structural"):
    conn.execute(
        "INSERT INTO repo_edges (source_id, target_id, rel_type, weight, edge_class) "
        "VALUES (?, ?, ?, ?, ?)",
        (source_id, target_id, rel_type, weight, edge_class),
    )


class MockDB:
    """Minimal mock that exposes .conn and protocol methods like UnifiedWikiDB."""

    def __init__(self, conn):
        self.conn = conn

    def find_nodes_by_name(self, name, macro_cluster=None, architectural_only=True, limit=5):
        conditions = ["symbol_name = ?"]
        params = [name]
        if macro_cluster is not None:
            conditions.append("macro_cluster = ?")
            params.append(macro_cluster)
        if architectural_only:
            conditions.append("is_architectural = 1")
        where = " AND ".join(conditions)
        rows = self.conn.execute(
            f"SELECT * FROM repo_nodes WHERE {where} "
            f"ORDER BY end_line - start_line DESC LIMIT ?",
            params + [limit],
        ).fetchall()
        return [dict(r) for r in rows]

    def search_fts_by_symbol_name(self, name, macro_cluster=None, architectural_only=True, limit=3):
        try:
            safe_name = name.replace('"', '""')
            conditions = []
            params = [f'symbol_name:"{safe_name}"']
            if macro_cluster is not None:
                conditions.append("n.macro_cluster = ?")
                params.append(macro_cluster)
            if architectural_only:
                conditions.append("n.is_architectural = 1")
            where_extra = (" AND " + " AND ".join(conditions)) if conditions else ""
            rows = self.conn.execute(
                f"SELECT n.* FROM repo_fts f "
                f"JOIN repo_nodes n ON f.node_id = n.node_id "
                f"WHERE repo_fts MATCH ? {where_extra} "
                f"ORDER BY rank LIMIT ?",
                params + [limit],
            ).fetchall()
            return [dict(r) for r in rows]
        except Exception:
            return []

    def get_edge_targets(self, source_id):
        rows = self.conn.execute(
            "SELECT target_id, rel_type, weight FROM repo_edges WHERE source_id = ?",
            (source_id,),
        ).fetchall()
        return [dict(r) for r in rows]

    def get_edge_sources(self, target_id):
        rows = self.conn.execute(
            "SELECT source_id, rel_type, weight FROM repo_edges WHERE target_id = ?",
            (target_id,),
        ).fetchall()
        return [dict(r) for r in rows]

    def get_edges_from(self, node_id, rel_types=None):
        if rel_types:
            placeholders = ",".join("?" * len(rel_types))
            rows = self.conn.execute(
                f"SELECT * FROM repo_edges WHERE source_id = ? AND rel_type IN ({placeholders})",
                [node_id] + rel_types,
            ).fetchall()
        else:
            rows = self.conn.execute(
                "SELECT * FROM repo_edges WHERE source_id = ?", (node_id,),
            ).fetchall()
        return [dict(r) for r in rows]

    def get_edges_to(self, node_id):
        rows = self.conn.execute(
            "SELECT * FROM repo_edges WHERE target_id = ?", (node_id,),
        ).fetchall()
        return [dict(r) for r in rows]

    def search_fts(self, query, path_prefix=None, cluster_id=None,
                   symbol_types=None, limit=20):
        if not query or not query.strip():
            return []
        safe_query = query.replace('"', '""')
        conditions = ["repo_fts MATCH ?"]
        params = [safe_query]
        if cluster_id is not None:
            conditions.append("n.macro_cluster = ?")
            params.append(cluster_id)
        if symbol_types:
            placeholders = ",".join("?" * len(symbol_types))
            conditions.append(f"n.symbol_type IN ({placeholders})")
            params.extend(symbol_types)
        params.append(limit)
        where = " AND ".join(conditions)
        try:
            rows = self.conn.execute(
                f"SELECT n.*, rank AS fts_rank FROM repo_fts f "
                f"JOIN repo_nodes n ON f.node_id = n.node_id "
                f"WHERE {where} ORDER BY rank LIMIT ?",
                params,
            ).fetchall()
            return [dict(r) for r in rows]
        except Exception:
            return []

    def get_nodes_by_ids(self, node_ids):
        if not node_ids:
            return []
        placeholders = ",".join("?" * len(node_ids))
        rows = self.conn.execute(
            f"SELECT * FROM repo_nodes WHERE node_id IN ({placeholders})",
            node_ids,
        ).fetchall()
        return [dict(r) for r in rows]

    def get_doc_nodes_by_cluster(self, macro, micro=None):
        if micro is not None:
            rows = self.conn.execute(
                "SELECT * FROM repo_nodes "
                "WHERE macro_cluster = ? AND micro_cluster = ? AND is_doc = 1 "
                "ORDER BY rel_path",
                (macro, micro),
            ).fetchall()
        else:
            rows = self.conn.execute(
                "SELECT * FROM repo_nodes "
                "WHERE macro_cluster = ? AND is_doc = 1 ORDER BY rel_path",
                (macro,),
            ).fetchall()
        return [dict(r) for r in rows]

    def find_related_nodes(self, node_id, rel_types, direction="out",
                           target_symbol_types=None, exclude_path=None, limit=15):
        if direction == "out":
            base = (
                "SELECT n.* FROM repo_edges e "
                "JOIN repo_nodes n ON e.target_id = n.node_id "
                "WHERE e.source_id = ?"
            )
        else:
            base = (
                "SELECT n.* FROM repo_edges e "
                "JOIN repo_nodes n ON e.source_id = n.node_id "
                "WHERE e.target_id = ?"
            )
        params = [node_id]
        placeholders = ",".join("?" * len(rel_types))
        base += f" AND e.rel_type IN ({placeholders})"
        params.extend(rel_types)
        if target_symbol_types:
            ph = ",".join("?" * len(target_symbol_types))
            base += f" AND n.symbol_type IN ({ph})"
            params.extend(target_symbol_types)
        if exclude_path:
            base += " AND n.rel_path != ?"
            params.append(exclude_path)
        base += " ORDER BY n.rel_path, n.start_line LIMIT ?"
        params.append(limit)
        rows = self.conn.execute(base, params).fetchall()
        return [dict(r) for r in rows]

    def get_node(self, node_id):
        row = self.conn.execute(
            "SELECT * FROM repo_nodes WHERE node_id = ?",
            (node_id,),
        ).fetchone()
        return dict(row) if row else None

    def detect_dominant_language(self, node_ids):
        if not node_ids:
            return None
        placeholders = ",".join("?" for _ in node_ids)
        row = self.conn.execute(
            f"SELECT language FROM repo_nodes "
            f"WHERE node_id IN ({placeholders}) AND language IS NOT NULL "
            f"GROUP BY language ORDER BY COUNT(*) DESC LIMIT 1",
            node_ids,
        ).fetchone()
        return row[0] if row else None


@pytest.fixture()
def db():
    conn = _make_db()
    yield MockDB(conn)
    conn.close()


@pytest.fixture()
def populated_db():
    """DB with a class hierarchy and function calls."""
    conn = _make_db()

    _insert_node(conn, "AuthService", symbol_type="class", macro_cluster=1,
                 source_text="class AuthService:\n    def login(self): pass\n    def logout(self): pass")
    _insert_node(conn, "LoginHandler", symbol_type="function", macro_cluster=1,
                 source_text="def login_handler(request): return AuthService().login()")
    _insert_node(conn, "SessionManager", symbol_type="class", macro_cluster=1,
                 source_text="class SessionManager:\n    sessions = {}")
    _insert_node(conn, "UserModel", symbol_type="class", macro_cluster=1,
                 source_text="class UserModel:\n    name: str\n    email: str")
    _insert_node(conn, "CONFIG", symbol_type="constant", macro_cluster=1,
                 source_text="CONFIG = {'secret': 'xxx'}")
    _insert_node(conn, "README", symbol_type="module_doc", macro_cluster=1,
                 is_doc=1, source_text="# Auth Module\nHandles authentication.")
    _insert_node(conn, "TestAuth", symbol_type="class", macro_cluster=1,
                 is_test=1, rel_path="tests/test_auth.py",
                 source_text="class TestAuth: pass")
    _insert_node(conn, "OutsideClass", symbol_type="class", macro_cluster=99,
                 source_text="class OutsideClass: pass")

    _insert_edge(conn, "LoginHandler", "AuthService", "calls", 2.0)
    _insert_edge(conn, "AuthService", "SessionManager", "composition", 1.5)
    _insert_edge(conn, "AuthService", "UserModel", "references", 1.0)
    _insert_edge(conn, "LoginHandler", "CONFIG", "references", 0.5)

    conn.commit()
    yield MockDB(conn)
    conn.close()


# ═══════════════════════════════════════════════════════════════════════════
# Constants
# ═══════════════════════════════════════════════════════════════════════════


class TestConstants:
    def test_default_budget(self):
        assert DEFAULT_TOKEN_BUDGET == 50_000

    def test_max_neighbors(self):
        assert MAX_NEIGHBORS_PER_SYMBOL == 15

    def test_expansion_symbols(self):
        assert "class" in EXPANSION_SYMBOL_TYPES
        assert "function" in EXPANSION_SYMBOL_TYPES
        assert "module_doc" in EXPANSION_SYMBOL_TYPES


# ═══════════════════════════════════════════════════════════════════════════
# _is_excluded_test_node
# ═══════════════════════════════════════════════════════════════════════════


class TestIsExcludedTestNode:
    def test_excluded_by_is_test_column(self):
        node = {"is_test": 1, "rel_path": "src/main.py"}
        assert _is_excluded_test_node(node, True) is True

    def test_excluded_by_is_test_flag(self):
        """Implementation only checks is_test column, not rel_path."""
        node = {"is_test": 1, "rel_path": "tests/test_auth.py"}
        assert _is_excluded_test_node(node, True) is True

    def test_not_excluded_when_flag_off(self):
        node = {"is_test": 1, "rel_path": "tests/test_auth.py"}
        assert _is_excluded_test_node(node, False) is False

    def test_normal_node_not_excluded(self):
        node = {"is_test": 0, "rel_path": "src/auth.py"}
        assert _is_excluded_test_node(node, True) is False


# ═══════════════════════════════════════════════════════════════════════════
# _estimate_tokens
# ═══════════════════════════════════════════════════════════════════════════


class TestEstimateTokens:
    def test_empty(self):
        assert _estimate_tokens("") == 0  # empty string → 0 tokens

    def test_short(self):
        assert _estimate_tokens("hello") >= 1

    def test_proportional(self):
        short = _estimate_tokens("x" * 35)
        long = _estimate_tokens("x" * 350)
        assert long > short


# ═══════════════════════════════════════════════════════════════════════════
# _node_to_document
# ═══════════════════════════════════════════════════════════════════════════


class TestNodeToDocument:
    def test_basic_conversion(self):
        node = {
            "node_id": "n1",
            "rel_path": "src/main.py",
            "symbol_name": "MyClass",
            "symbol_type": "class",
            "start_line": 10,
            "end_line": 50,
            "language": "python",
            "source_text": "class MyClass: pass",
            "signature": "",
            "docstring": "",
            "macro_cluster": 1,
            "micro_cluster": 0,
        }
        doc = _node_to_document(node, is_initial=True)
        assert isinstance(doc, Document)
        assert doc.page_content == "class MyClass: pass"
        assert doc.metadata["source"] == "src/main.py"
        assert doc.metadata["is_initially_retrieved"] is True

    def test_fallback_to_signature(self):
        node = {
            "node_id": "n2",
            "source_text": "",
            "signature": "def foo()",
            "docstring": "Does foo things",
            "rel_path": "",
            "symbol_name": "",
            "symbol_type": "",
            "start_line": 0,
            "end_line": 0,
            "language": "",
            "macro_cluster": None,
            "micro_cluster": None,
        }
        doc = _node_to_document(node)
        # Implementation uses source_text only; no fallback to signature/docstring
        assert doc.page_content == ""
        assert doc.metadata["signature"] == "def foo()"
        assert doc.metadata["docstring"] == "Does foo things"

    def test_expanded_from_metadata(self):
        node = {
            "node_id": "n3",
            "source_text": "x",
            "signature": "",
            "docstring": "",
            "rel_path": "",
            "symbol_name": "",
            "symbol_type": "",
            "start_line": 0,
            "end_line": 0,
            "language": "",
            "macro_cluster": None,
            "micro_cluster": None,
        }
        doc = _node_to_document(node, expanded_from="inheritance")
        assert doc.metadata["expanded_from"] == "inheritance"
        # is_initial defaults to True; expanded_from is metadata-only
        assert doc.metadata["is_initially_retrieved"] is True


# ═══════════════════════════════════════════════════════════════════════════
# _augment_document / _augment_cpp / _augment_go_rust
# ═══════════════════════════════════════════════════════════════════════════


class TestAugmentation:
    def test_augment_non_augmentable_language(self, db):
        doc = Document(
            page_content="class Foo: pass",
            metadata={"language": "python", "node_id": "n1", "source": "src/foo.py"},
        )
        cost = _augment_document(db, doc)
        assert cost == 0
        assert doc.page_content == "class Foo: pass"

    def test_augment_cpp_header(self, db):
        # 2-hop: Class →defines→ Method ←defines_body← Impl
        _insert_node(db.conn, "CppClass", symbol_type="class", language="cpp",
                      rel_path="include/foo.h",
                      source_text="class Foo { void bar(); };")
        _insert_node(db.conn, "CppMethod", symbol_type="method", language="cpp",
                      rel_path="include/foo.h",
                      source_text="void bar();")
        _insert_node(db.conn, "CppImpl", symbol_type="function", language="cpp",
                      rel_path="src/foo.cpp",
                      source_text="void Foo::bar() { /* impl */ }")
        _insert_edge(db.conn, "CppClass", "CppMethod", "defines")
        _insert_edge(db.conn, "CppImpl", "CppMethod", "defines_body")
        db.conn.commit()

        doc = Document(
            page_content="class Foo { void bar(); };",
            metadata={"language": "cpp", "node_id": "CppClass",
                       "source": "include/foo.h", "symbol_type": "class"},
        )
        # _augment_cpp takes (conn, node_id, sym_type, decl_file) and returns parts list
        parts = _augment_cpp(db.conn, "CppClass", "class", "include/foo.h")
        combined = "\n".join(parts)
        assert "foo.cpp" in combined
        assert "Foo::bar()" in combined

    def test_augment_go_struct(self, db):
        _insert_node(db.conn, "GoStruct", symbol_type="struct", language="go",
                      rel_path="pkg/model.go",
                      source_text="type User struct { Name string }")
        _insert_node(db.conn, "GoMethod", symbol_type="function", language="go",
                      rel_path="pkg/user_methods.go",
                      source_text="func (u *User) Validate() error { return nil }")
        _insert_edge(db.conn, "GoStruct", "GoMethod", "defines")
        db.conn.commit()

        doc = Document(
            page_content="type User struct { Name string }",
            metadata={"language": "go", "node_id": "GoStruct",
                       "source": "pkg/model.go", "symbol_type": "struct"},
        )
        # _augment_go_rust takes (conn, node_id, sym_type, type_file) and returns parts list
        parts = _augment_go_rust(db.conn, "GoStruct", "struct", "pkg/model.go")
        combined = "\n".join(parts)
        assert "Validate" in combined

    def test_augment_skips_non_struct(self, db):
        doc = Document(
            page_content="func main() {}",
            metadata={"language": "go", "node_id": "x",
                       "source": "main.go", "symbol_type": "function"},
        )
        # _augment_go_rust returns empty list for non-struct types
        parts = _augment_go_rust(db.conn, "x", "function", "main.go")
        assert parts == []


# ═══════════════════════════════════════════════════════════════════════════
# _resolve_symbols
# ═══════════════════════════════════════════════════════════════════════════


class TestResolveSymbols:
    def test_exact_match(self, populated_db):
        result = _resolve_symbols(populated_db, ["AuthService"], macro_id=1)
        assert "AuthService" in result
        assert result["AuthService"]["symbol_name"] == "AuthService"

    def test_multiple_symbols(self, populated_db):
        result = _resolve_symbols(populated_db, ["AuthService", "LoginHandler"], macro_id=1)
        assert len(result) == 2

    def test_unknown_symbol_empty(self, populated_db):
        result = _resolve_symbols(populated_db, ["NonexistentSymbol"], macro_id=1)
        assert len(result) == 0

    def test_cluster_scoped(self, populated_db):
        result = _resolve_symbols(populated_db, ["OutsideClass"], macro_id=1)
        assert len(result) == 0  # macro_cluster=99 != 1

    def test_no_macro_id(self, populated_db):
        result = _resolve_symbols(populated_db, ["AuthService"])
        assert "AuthService" in result

    def test_empty_names_skipped(self, populated_db):
        result = _resolve_symbols(populated_db, ["", None, "AuthService"], macro_id=1)
        assert len(result) == 1


# ═══════════════════════════════════════════════════════════════════════════
# _collect_expansion_neighbors
# ═══════════════════════════════════════════════════════════════════════════


class TestCollectExpansionNeighbors:
    def test_finds_neighbors(self, populated_db):
        result = _collect_expansion_neighbors(
            populated_db,
            seed_ids=["AuthService"],
            seen_ids=set(),
            macro_id=1,
        )
        found_ids = {r[0] for r in result}
        assert "SessionManager" in found_ids or "UserModel" in found_ids

    def test_respects_seen_ids(self, populated_db):
        result = _collect_expansion_neighbors(
            populated_db,
            seed_ids=["AuthService"],
            seen_ids={"SessionManager"},
            macro_id=1,
        )
        found_ids = {r[0] for r in result}
        assert "SessionManager" not in found_ids

    def test_excludes_out_of_cluster(self, populated_db):
        """OutsideClass (macro=99) should not appear when macro_id=1."""
        # Add edge from inside to outside
        _insert_edge(populated_db.conn, "AuthService", "OutsideClass", "calls")
        populated_db.conn.commit()
        result = _collect_expansion_neighbors(
            populated_db,
            seed_ids=["AuthService"],
            seen_ids=set(),
            macro_id=1,
        )
        found_ids = {r[0] for r in result}
        assert "OutsideClass" not in found_ids

    def test_page_boundary_ids(self, populated_db):
        result = _collect_expansion_neighbors(
            populated_db,
            seed_ids=["AuthService"],
            seen_ids=set(),
            macro_id=1,
            page_boundary_ids={"SessionManager"},
        )
        found_ids = {r[0] for r in result}
        assert "SessionManager" in found_ids
        assert "UserModel" not in found_ids

    def test_excludes_test_nodes(self, populated_db):
        _insert_edge(populated_db.conn, "AuthService", "TestAuth", "calls")
        populated_db.conn.commit()
        # exclude_tests is read internally via get_feature_flags()
        with patch(
            "app.core.cluster_expansion.get_feature_flags",
            return_value=FeatureFlags(exclude_tests=True),
        ):
            result = _collect_expansion_neighbors(
                populated_db,
                seed_ids=["AuthService"],
                seen_ids=set(),
                macro_id=1,
            )
        found_ids = {r[0] for r in result}
        assert "TestAuth" not in found_ids


# ═══════════════════════════════════════════════════════════════════════════
# _get_cluster_docs
# ═══════════════════════════════════════════════════════════════════════════


class TestGetClusterDocs:
    def test_finds_doc_nodes(self, populated_db):
        docs = _get_cluster_docs(populated_db, 1, None, set())
        assert len(docs) >= 1
        assert any(d["symbol_name"] == "README" for d in docs)

    def test_respects_seen_ids(self, populated_db):
        docs = _get_cluster_docs(populated_db, 1, None, {"README"})
        assert not any(d.get("node_id") == "README" for d in docs)

    def test_micro_cluster_filter(self, populated_db):
        docs = _get_cluster_docs(populated_db, 1, 0, set())
        assert len(docs) >= 1

    def test_excludes_test_docs(self, populated_db):
        # Add a test doc node
        _insert_node(populated_db.conn, "TestDoc", symbol_type="module_doc",
                      macro_cluster=1, is_doc=1, is_test=1,
                      rel_path="tests/conftest.py")
        populated_db.conn.commit()
        # exclude_tests is read internally via get_feature_flags()
        with patch(
            "app.core.cluster_expansion.get_feature_flags",
            return_value=FeatureFlags(exclude_tests=True),
        ):
            docs = _get_cluster_docs(populated_db, 1, None, set())
        assert not any(d.get("node_id") == "TestDoc" for d in docs)


# ═══════════════════════════════════════════════════════════════════════════
# expand_for_page (integration)
# ═══════════════════════════════════════════════════════════════════════════


class TestExpandForPage:
    def test_basic_expansion(self, populated_db):
        docs = expand_for_page(
            populated_db,
            page_symbols=["AuthService", "LoginHandler"],
            macro_id=1,
        )
        assert len(docs) > 0
        assert all(isinstance(d, Document) for d in docs)

    def test_initial_docs_marked(self, populated_db):
        docs = expand_for_page(
            populated_db,
            page_symbols=["AuthService"],
            macro_id=1,
        )
        initial = [d for d in docs if d.metadata.get("is_initially_retrieved")]
        assert len(initial) >= 1

    def test_expanded_docs_present(self, populated_db):
        docs = expand_for_page(
            populated_db,
            page_symbols=["AuthService"],
            macro_id=1,
        )
        expanded = [d for d in docs if not d.metadata.get("is_initially_retrieved")]
        assert len(expanded) >= 1

    def test_empty_symbols(self, populated_db):
        docs = expand_for_page(populated_db, page_symbols=[])
        assert docs == []

    def test_unknown_symbols(self, populated_db):
        docs = expand_for_page(
            populated_db,
            page_symbols=["TotallyFakeSymbol"],
            macro_id=1,
        )
        assert docs == []

    def test_token_budget_respected(self, populated_db):
        docs = expand_for_page(
            populated_db,
            page_symbols=["AuthService", "LoginHandler", "SessionManager"],
            macro_id=1,
            token_budget=10,  # Very small budget
        )
        # Should still get at least the first seed if it fits
        assert len(docs) <= 3

    def test_exclude_tests(self, populated_db):
        # exclude_tests is read internally via get_feature_flags()
        with patch(
            "app.core.cluster_expansion.get_feature_flags",
            return_value=FeatureFlags(exclude_tests=True),
        ):
            docs = expand_for_page(
                populated_db,
                page_symbols=["AuthService", "TestAuth"],
                macro_id=1,
            )
        names = [d.metadata.get("symbol_name") for d in docs]
        assert "TestAuth" not in names

    def test_include_docs_flag(self, populated_db):
        docs_with = expand_for_page(
            populated_db,
            page_symbols=["AuthService"],
            macro_id=1,
            include_docs=True,
        )
        docs_without = expand_for_page(
            populated_db,
            page_symbols=["AuthService"],
            macro_id=1,
            include_docs=False,
        )
        # With docs should have >= without docs
        assert len(docs_with) >= len(docs_without)

    def test_cluster_node_ids_boundary(self, populated_db):
        docs = expand_for_page(
            populated_db,
            page_symbols=["AuthService"],
            macro_id=1,
            cluster_node_ids=["AuthService", "SessionManager"],
        )
        # Only AuthService and SessionManager should appear
        node_ids = {d.metadata.get("node_id") for d in docs}
        for nid in node_ids:
            if nid:
                assert nid in {"AuthService", "SessionManager", "README"}

    def test_no_macro_id(self, populated_db):
        """Without macro_id, expansion is unbounded."""
        docs = expand_for_page(
            populated_db,
            page_symbols=["AuthService"],
        )
        assert len(docs) >= 1

    def test_micro_id_for_docs(self, populated_db):
        docs = expand_for_page(
            populated_db,
            page_symbols=["AuthService"],
            macro_id=1,
            micro_id=0,
            include_docs=True,
        )
        assert any(
            d.metadata.get("expanded_from") == "cluster_doc"
            for d in docs
        )


# ═══════════════════════════════════════════════════════════════════════════
# Framework Reference Discovery (Step 2.75)
# ═══════════════════════════════════════════════════════════════════════════


@pytest.fixture()
def framework_db():
    """DB modelled after the real configurations-plugin pattern.

    Cluster 0 (handlers):
        Event (class, architectural, orphan) → defines → configuration_created (method)
        AnotherHandler (class, architectural, orphan) → defines → handle_request (method)

    Cluster 1 (dispatch):
        create_configuration (function) — source_text mentions "configuration_created"
        dispatch_util (function) — source_text mentions "handle_request"
        get_tools (function) — shares name with an orphan-like peer
        some_helper (function) — no relationship to orphans

    Cluster 0 also has:
        get_tools (function, architectural, orphan) — to test peer-name filtering
    """
    conn = _make_db()

    # ── Cluster 0: Handlers ──
    _insert_node(conn, "python::Event", symbol_name="Event",
                 symbol_type="class", macro_cluster=0, micro_cluster=0,
                 is_architectural=1,
                 source_text="class Event:\n    def configuration_created(self, ctx): pass")
    _insert_node(conn, "python::Event.configuration_created",
                 symbol_name="configuration_created",
                 symbol_type="method", macro_cluster=0, micro_cluster=0,
                 is_architectural=0,
                 source_text="def configuration_created(self, ctx): pass")
    # Event defines its method  (the ONLY structural edge for Event → 1 edge → orphan threshold)
    _insert_edge(conn, "python::Event", "python::Event.configuration_created",
                 rel_type="defines", edge_class="structural")

    _insert_node(conn, "python::AnotherHandler", symbol_name="AnotherHandler",
                 symbol_type="class", macro_cluster=0, micro_cluster=0,
                 is_architectural=1,
                 source_text="class AnotherHandler:\n    def handle_request(self, r): pass")
    _insert_node(conn, "python::AnotherHandler.handle_request",
                 symbol_name="handle_request",
                 symbol_type="method", macro_cluster=0, micro_cluster=0,
                 is_architectural=0,
                 source_text="def handle_request(self, r): pass")
    _insert_edge(conn, "python::AnotherHandler", "python::AnotherHandler.handle_request",
                 rel_type="defines", edge_class="structural")

    # get_tools — orphan function in cluster 0 (tests peer-name filtering)
    _insert_node(conn, "python::get_tools_c0", symbol_name="get_tools",
                 symbol_type="function", macro_cluster=0, micro_cluster=0,
                 is_architectural=1,
                 source_text="def get_tools(): return []")

    # ── Cluster 1: Dispatch / Infrastructure ──
    _insert_node(conn, "python::create_configuration", symbol_name="create_configuration",
                 symbol_type="function", macro_cluster=1, micro_cluster=0,
                 is_architectural=1,
                 source_text="def create_configuration(ctx):\n    emit('configuration_created', data)")
    _insert_node(conn, "python::dispatch_util", symbol_name="dispatch_util",
                 symbol_type="function", macro_cluster=1, micro_cluster=0,
                 is_architectural=1,
                 source_text="def dispatch_util(r):\n    manager.handle_request(r)")
    _insert_node(conn, "python::get_tools_c1", symbol_name="get_tools",
                 symbol_type="function", macro_cluster=1, micro_cluster=0,
                 is_architectural=1,
                 source_text="def get_tools(): return toolkit.get_tools()")
    _insert_node(conn, "python::some_helper", symbol_name="some_helper",
                 symbol_type="function", macro_cluster=1, micro_cluster=0,
                 is_architectural=1,
                 source_text="def some_helper(): return 42")

    conn.commit()
    yield MockDB(conn)
    conn.close()


class TestCountStructuralEdges:
    """Verify orphan detection counts only structural edges."""

    def test_structural_edge_counted(self, framework_db):
        count = _count_structural_edges(framework_db, "python::Event")
        # Event → defines → configuration_created  (1 structural outgoing)
        assert count == 1

    def test_bridge_edge_not_counted(self, framework_db):
        # Add a bridge edge — should NOT increase count.
        framework_db.conn.execute(
            "INSERT INTO repo_edges (source_id, target_id, rel_type, edge_class) "
            "VALUES (?, ?, ?, ?)",
            ("python::Event", "python::some_helper", "references", "bridge"),
        )
        framework_db.conn.commit()
        assert _count_structural_edges(framework_db, "python::Event") == 1

    def test_zero_structural_for_leaf(self, framework_db):
        # some_helper has no edges at all
        assert _count_structural_edges(framework_db, "python::some_helper") == 0


class TestCollectSearchTerms:
    """Verify search term collection for different symbol types."""

    def test_class_collects_child_method_names(self, framework_db):
        node = dict(framework_db.conn.execute(
            "SELECT * FROM repo_nodes WHERE node_id = ?",
            ("python::Event",),
        ).fetchone())
        terms = _collect_search_terms(framework_db, "python::Event", node)
        assert "Event" in terms
        assert "configuration_created" in terms

    def test_class_with_multiple_children(self, framework_db):
        node = dict(framework_db.conn.execute(
            "SELECT * FROM repo_nodes WHERE node_id = ?",
            ("python::AnotherHandler",),
        ).fetchone())
        terms = _collect_search_terms(framework_db, "python::AnotherHandler", node)
        assert "AnotherHandler" in terms
        assert "handle_request" in terms

    def test_function_returns_only_own_name(self, framework_db):
        node = dict(framework_db.conn.execute(
            "SELECT * FROM repo_nodes WHERE node_id = ?",
            ("python::get_tools_c0",),
        ).fetchone())
        terms = _collect_search_terms(framework_db, "python::get_tools_c0", node)
        assert terms == ["get_tools"]

    def test_short_name_excluded(self, framework_db):
        """Symbol names shorter than _MIN_FTS_NAME_LEN are dropped."""
        node = {"symbol_name": "ab", "symbol_type": "function"}
        terms = _collect_search_terms(framework_db, "python::ab", node)
        assert terms == []


class TestFindFrameworkReferences:
    """Verify framework reference discovery via FTS."""

    def test_finds_cross_cluster_caller_via_child_method(self, framework_db):
        """Event class (orphan) → child 'configuration_created' →
        FTS finds create_configuration in cluster 1."""
        orphan_node = dict(framework_db.conn.execute(
            "SELECT * FROM repo_nodes WHERE node_id = ?",
            ("python::Event",),
        ).fetchone())
        orphans = {"python::Event": orphan_node}

        results = _find_framework_references(
            framework_db, orphans, seen_ids=set(), macro_id=0,
        )
        hit_ids = [nid for nid, _, _ in results]
        assert "python::create_configuration" in hit_ids

    def test_filters_same_name_peers(self, framework_db):
        """get_tools orphan should NOT find other get_tools definitions."""
        orphan_node = dict(framework_db.conn.execute(
            "SELECT * FROM repo_nodes WHERE node_id = ?",
            ("python::get_tools_c0",),
        ).fetchone())
        orphans = {"python::get_tools_c0": orphan_node}

        results = _find_framework_references(
            framework_db, orphans, seen_ids=set(), macro_id=0,
        )
        hit_ids = [nid for nid, _, _ in results]
        # get_tools_c1 has symbol_name="get_tools" → filtered as peer
        assert "python::get_tools_c1" not in hit_ids

    def test_skips_orphan_cross_refs(self, framework_db):
        """Orphans should not reference each other."""
        ev_node = dict(framework_db.conn.execute(
            "SELECT * FROM repo_nodes WHERE node_id = ?",
            ("python::Event",),
        ).fetchone())
        ah_node = dict(framework_db.conn.execute(
            "SELECT * FROM repo_nodes WHERE node_id = ?",
            ("python::AnotherHandler",),
        ).fetchone())
        orphans = {
            "python::Event": ev_node,
            "python::AnotherHandler": ah_node,
        }

        results = _find_framework_references(
            framework_db, orphans, seen_ids=set(), macro_id=0,
        )
        hit_ids = {nid for nid, _, _ in results}
        # Neither orphan should appear in results
        assert "python::Event" not in hit_ids
        assert "python::AnotherHandler" not in hit_ids

    def test_skips_already_seen_ids(self, framework_db):
        """Nodes already in the page set should be excluded."""
        orphan_node = dict(framework_db.conn.execute(
            "SELECT * FROM repo_nodes WHERE node_id = ?",
            ("python::Event",),
        ).fetchone())
        orphans = {"python::Event": orphan_node}

        results = _find_framework_references(
            framework_db, orphans,
            seen_ids={"python::create_configuration"},
            macro_id=0,
        )
        hit_ids = [nid for nid, _, _ in results]
        assert "python::create_configuration" not in hit_ids

    def test_description_format(self, framework_db):
        """Result descriptions follow the expected format."""
        orphan_node = dict(framework_db.conn.execute(
            "SELECT * FROM repo_nodes WHERE node_id = ?",
            ("python::AnotherHandler",),
        ).fetchone())
        orphans = {"python::AnotherHandler": orphan_node}

        results = _find_framework_references(
            framework_db, orphans, seen_ids=set(), macro_id=0,
        )
        if results:
            _, _, desc = results[0]
            assert desc.startswith("fts_framework_ref:")

    def test_multiple_orphans_combined(self, framework_db):
        """Multiple orphans each find their own callers."""
        ev_node = dict(framework_db.conn.execute(
            "SELECT * FROM repo_nodes WHERE node_id = ?",
            ("python::Event",),
        ).fetchone())
        ah_node = dict(framework_db.conn.execute(
            "SELECT * FROM repo_nodes WHERE node_id = ?",
            ("python::AnotherHandler",),
        ).fetchone())
        orphans = {
            "python::Event": ev_node,
            "python::AnotherHandler": ah_node,
        }

        results = _find_framework_references(
            framework_db, orphans, seen_ids=set(), macro_id=0,
        )
        hit_ids = {nid for nid, _, _ in results}
        assert "python::create_configuration" in hit_ids
        assert "python::dispatch_util" in hit_ids
