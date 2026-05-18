"""Regression tests for hybrid orphan resolution."""

from __future__ import annotations

from dataclasses import replace
from unittest.mock import MagicMock

import networkx as nx

from app.core import feature_flags as _ff
from app.core.graph_orphan_hybrid import resolve_orphans_hybrid


def _mock_db():
    db = MagicMock()
    db.get_node.return_value = {
        "node_id": "orphan",
        "symbol_name": "Config",
        "rel_path": "src/config.py",
        "source_text": "class Config:\n    pass\n",
    }
    db.search_fts_with_path.return_value = [
        {"node_id": "target", "score_norm": 0.9, "fts_rank": 1},
    ]
    db.get_embedding_by_id.return_value = None
    db.search_vec.return_value = []
    return db


def _graph():
    graph = nx.MultiDiGraph()
    graph.add_node("orphan", symbol_name="Config", rel_path="src/config.py")
    graph.add_node("target", symbol_name="Config", rel_path="src/settings.py")
    return graph


def test_missing_embedding_falls_through_without_pure_fts(monkeypatch):
    flags = replace(
        _ff.get_feature_flags(),
        orphan_rrf_threshold=0.0,
        orphan_hybrid_top_n=5,
    )
    monkeypatch.setattr(_ff, "get_feature_flags", lambda: flags)
    embed_fn = MagicMock(return_value=None)

    hits = resolve_orphans_hybrid(
        _mock_db(),
        _graph(),
        "orphan",
        orphan_embeddings={"orphan": None},
        embed_fn=embed_fn,
        flags=flags,
    )

    assert hits == []
    embed_fn.assert_not_called()


def test_hybrid_requires_vector_evidence(monkeypatch):
    flags = replace(
        _ff.get_feature_flags(),
        orphan_rrf_threshold=0.0,
        orphan_hybrid_top_n=5,
    )
    monkeypatch.setattr(_ff, "get_feature_flags", lambda: flags)
    db = _mock_db()
    db.get_embedding_by_id.return_value = [0.1, 0.2, 0.3]
    db.search_vec.return_value = [
        {"node_id": "target", "score_norm": 0.8, "vec_rank": 1},
    ]

    hits = resolve_orphans_hybrid(db, _graph(), "orphan", flags=flags)

    assert [hit["node_id"] for hit in hits] == ["target"]
    assert hits[0]["rrf_score"] > 0