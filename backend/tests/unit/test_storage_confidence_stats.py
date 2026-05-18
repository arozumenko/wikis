"""Test the ``confidence_breakdown`` field of ``UnifiedWikiDB.stats()`` (#120).

The graph-native MCP surface and the upcoming SPA citation chips both
read this dict; a regression that drops the field or changes its shape
would break both consumers silently. Pin the contract.
"""

from __future__ import annotations

import pytest

from app.core.unified_db import UnifiedWikiDB


@pytest.fixture()
def db(tmp_path):
    d = UnifiedWikiDB(tmp_path / "stats.wiki.db", embedding_dim=8)
    yield d
    d.close()


def _seed_node(db: UnifiedWikiDB, node_id: str) -> None:
    """Minimal node insertion — most fields default at the storage layer."""
    db.upsert_node(node_id=node_id, rel_path=f"{node_id}.py", symbol_name=node_id)


def _seed_edge(
    db: UnifiedWikiDB,
    source: str,
    target: str,
    rel_type: str = "calls",
    confidence: str = "EXTRACTED",
) -> None:
    """Insert an edge with an explicit confidence label."""
    db.upsert_edge(source, target, rel_type, confidence=confidence)


class TestConfidenceBreakdown:
    def test_stats_always_includes_three_buckets(self, db) -> None:
        """Even with zero edges, the breakdown has the three expected
        keys initialised to 0. Stable shape lets downstream code
        index without ``.get(key, 0)`` guards."""
        stats = db.stats()
        assert "confidence_breakdown" in stats
        breakdown = stats["confidence_breakdown"]
        assert breakdown == {"extracted": 0, "inferred": 0, "ambiguous": 0}

    def test_extracted_edges_counted(self, db) -> None:
        _seed_node(db, "a")
        _seed_node(db, "b")
        _seed_node(db, "c")
        _seed_edge(db, "a", "b", confidence="EXTRACTED")
        _seed_edge(db, "a", "c", confidence="EXTRACTED")

        breakdown = db.stats()["confidence_breakdown"]
        assert breakdown["extracted"] == 2
        assert breakdown["inferred"] == 0
        assert breakdown["ambiguous"] == 0

    def test_inferred_edges_counted_separately(self, db) -> None:
        """The whole point of #120 — operators need to know how much of
        the graph is name-only resolution vs explicit observation."""
        _seed_node(db, "a")
        _seed_node(db, "b")
        _seed_edge(db, "a", "b", confidence="INFERRED")

        breakdown = db.stats()["confidence_breakdown"]
        assert breakdown["inferred"] == 1
        assert breakdown["extracted"] == 0

    def test_mixed_distribution(self, db) -> None:
        _seed_node(db, "a")
        _seed_node(db, "b")
        _seed_node(db, "c")
        _seed_node(db, "d")
        _seed_edge(db, "a", "b", confidence="EXTRACTED")
        _seed_edge(db, "a", "c", confidence="INFERRED")
        _seed_edge(db, "b", "c", confidence="AMBIGUOUS")
        _seed_edge(db, "c", "d", confidence="INFERRED")

        breakdown = db.stats()["confidence_breakdown"]
        assert breakdown == {
            "extracted": 1, "inferred": 2, "ambiguous": 1,
        }


def test_source_reference_carries_optional_confidence() -> None:
    """``SourceReference`` now exposes the confidence field; old
    callers that don't pass it continue to construct successfully
    (None by default)."""
    from app.models.api import SourceReference

    # Without confidence (backward-compat).
    ref = SourceReference(file_path="x.py")
    assert ref.confidence is None

    # With confidence — the headline addition for #120.
    ref2 = SourceReference(file_path="x.py", confidence="EXTRACTED")
    assert ref2.confidence == "EXTRACTED"

    # Serialises through model_dump unchanged (what flows to MCP
    # clients + the SPA via the cache record JSON).
    dumped = ref2.model_dump()
    assert dumped["confidence"] == "EXTRACTED"
