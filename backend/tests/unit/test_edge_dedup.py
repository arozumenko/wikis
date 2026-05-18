"""
Phase 1 (graph-quality roadmap) — edge dedup invariant.

Validates that ``_add_edge`` (with ``WIKI_EDGE_DEDUP=true``, default)
refuses to add a duplicate ``(source, target, rel_type)`` triple, and
that re-running ``inject_doc_edges`` does not double the doc-edge count.
"""

from __future__ import annotations

import networkx as nx
import pytest

from app.core.graph_topology import (
    _add_edge,
    _has_typed_edge,
    inject_doc_edges,
)
from app.core.unified_db import UnifiedWikiDB


def _seed() -> nx.MultiDiGraph:
    G = nx.MultiDiGraph()
    G.add_node(
        "AuthService",
        symbol_name="AuthService",
        symbol_type="class",
        rel_path="src/auth.py",
        start_line=1,
        end_line=30,
        language="python",
        source_text="class AuthService: pass",
    )
    G.add_node(
        "README",
        symbol_name="README",
        symbol_type="module_doc",
        rel_path="README.md",
        start_line=0,
        end_line=10,
        language="markdown",
        source_text="# Project uses `AuthService` extensively.",
    )
    return G


@pytest.fixture()
def db(tmp_path):
    d = UnifiedWikiDB(str(tmp_path / "dedup.db"), embedding_dim=8)
    d.from_networkx(_seed())
    d.conn.commit()
    yield d
    d.close()


def test_add_edge_returns_false_on_duplicate(db):
    G = db.to_networkx()

    added_first = _add_edge(
        db, G,
        source="README", target="AuthService",
        rel_type="references", edge_class="doc",
        created_by="test",
        skip_db=True,
    )
    added_second = _add_edge(
        db, G,
        source="README", target="AuthService",
        rel_type="references", edge_class="doc",
        created_by="test",
        skip_db=True,
    )
    assert added_first is True
    assert added_second is False
    assert _has_typed_edge(G, "README", "AuthService", "references")


def test_add_edge_allows_distinct_rel_types(db):
    G = db.to_networkx()
    assert _add_edge(
        db, G, source="README", target="AuthService",
        rel_type="references", edge_class="doc", created_by="test",
        skip_db=True,
    )
    # A different rel_type on the same (u, v) is fine.
    assert _add_edge(
        db, G, source="README", target="AuthService",
        rel_type="lexical_link", edge_class="lexical", created_by="test",
        skip_db=True,
    )


def test_inject_doc_edges_is_idempotent(db):
    G = db.to_networkx()
    first = inject_doc_edges(db, G)
    edges_after_first = G.number_of_edges()

    second = inject_doc_edges(db, G)
    edges_after_second = G.number_of_edges()

    assert edges_after_second == edges_after_first, (
        f"second inject_doc_edges added edges: {first} → {second}"
    )
