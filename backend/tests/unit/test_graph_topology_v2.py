"""Tests for the v2 orphan-resolution cascade dispatcher.

Verifies the new pass ordering wired in ``resolve_orphans`` when
``orphan_cascade_v2`` is on:

    Pass 1 — explicit references (md links, backticks, imports)
    Pass 2 — hybrid FTS+Vec via RRF
    Pass 3 — tiered lexical T1\u2013T4
    Pass 4 — directory proximity
"""

from __future__ import annotations

from dataclasses import replace
from typing import Any
from unittest.mock import MagicMock

import networkx as nx
import pytest

from app.core import feature_flags as _ff
from app.core.graph_topology import resolve_orphans


def _code_node(rel_path: str, name: str, kind: str = "class") -> dict[str, Any]:
    return {
        "rel_path": rel_path,
        "symbol_name": name,
        "symbol": {"name": name, "kind": kind},
        "symbol_type": kind,
        "kind": kind,
        "location": {"rel_path": rel_path},
    }


def _doc_node(rel_path: str, content: str) -> dict[str, Any]:
    return {
        "rel_path": rel_path,
        "symbol_type": "markdown_document",
        "kind": "markdown_document",
        "is_doc": 1,
        "source_text": content,
        "content": content,
        "location": {"rel_path": rel_path},
    }


def _mock_db() -> MagicMock:
    db = MagicMock()
    db.vec_available = False
    db.search_fts5.return_value = []
    db.search_vec.return_value = []
    db.search_fts_with_path.return_value = []
    db.get_node.return_value = None
    db.get_embedding_by_id.return_value = None
    db.edge_count.return_value = 0
    db.conn = MagicMock()
    db.count_fts_matches.return_value = 1
    return db


@pytest.fixture
def v2_on(monkeypatch):
    """Force every v2-related flag on for this test."""
    forced = replace(
        _ff.get_feature_flags(),
        orphan_cascade_v2=True,
        orphan_lexical_tiered=True,
        orphan_hybrid_search=True,
        orphan_reuse_embeddings=True,
        # Lower the RRF threshold so a single vec hit at rank 1
        # (RRF = 1/(60+1) ~= 0.0164) survives the 0.02 default.
        orphan_rrf_threshold=0.0,
    )
    monkeypatch.setattr(_ff, "get_feature_flags", lambda: forced)
    return forced


def test_v2_explicit_ref_md_link_first(v2_on):
    """Pass 1 catches markdown links before any other pass runs."""
    G = nx.MultiDiGraph()
    G.add_node(
        "doc",
        **_doc_node("docs/intro.md", "See [target](../src/foo.py) for details."),
    )
    G.add_node("target", **_code_node("src/foo.py", "Foo"))
    G.add_edge("target", "target")  # keep target out of orphans

    db = _mock_db()
    stats = resolve_orphans(db, G)

    # Md-link is doc class.
    assert stats["explicit_ref"] >= 1
    assert G.has_edge("doc", "target")
    # Other passes did not add edges for this orphan.
    assert stats.get("hybrid", 0) == 0
    assert stats.get("lexical", 0) == 0


def test_v2_falls_through_to_directory(v2_on):
    """Orphan with no signal ends up bridged via directory proximity."""
    G = nx.MultiDiGraph()
    G.add_node("orphan", **_code_node("src/widgets/odd.py", "Odd"))
    G.add_node("anchor", **_code_node("src/widgets/main.py", "Main"))
    G.add_edge("anchor", "anchor")

    db = _mock_db()
    stats = resolve_orphans(db, G)

    assert stats["directory"] == 1
    assert G.has_edge("orphan", "anchor")
    assert stats["explicit_ref"] == 0
    assert stats.get("hybrid", 0) == 0


def test_v2_hybrid_uses_persisted_embedding(v2_on):
    """Pass 2 reuses the persisted embedding via get_embedding_by_id."""
    G = nx.MultiDiGraph()
    G.add_node(
        "orphan",
        **_code_node("src/auth/token.py", "TokenManager"),
        source_text="Manages JWT tokens.",
    )
    G.add_node("target", **_code_node("src/auth/jwt.py", "JWTService"))
    G.add_edge("target", "target")

    db = _mock_db()
    db.vec_available = True
    db.get_embedding_by_id.return_value = [0.1, 0.2, 0.3]
    db.search_vec.return_value = [
        {"node_id": "target", "vec_distance": 0.1, "score_norm": 0.9},
    ]

    stats = resolve_orphans(db, G)
    assert stats.get("hybrid", 0) >= 1
    assert G.has_edge("orphan", "target")


def test_v2_disabled_falls_back_to_legacy(monkeypatch):
    """When the flag is off the legacy cascade is exercised."""
    forced = replace(
        _ff.get_feature_flags(),
        orphan_cascade_v2=False,
        orphan_lexical_tiered=False,
        orphan_hybrid_search=False,
    )
    monkeypatch.setattr(_ff, "get_feature_flags", lambda: forced)

    G = nx.MultiDiGraph()
    G.add_node("orphan", **_code_node("src/a.py", "UserValidator"))
    G.add_node("target", **_code_node("src/b.py", "UserService"))
    G.add_edge("target", "target")

    db = _mock_db()
    db.search_fts5.return_value = [{"node_id": "target", "score_norm": 0.6}]

    stats = resolve_orphans(db, G)
    assert stats["lexical"] == 1
    assert "explicit_ref" not in stats  # v2-only key
    assert G.has_edge("orphan", "target")
