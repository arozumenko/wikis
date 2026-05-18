"""Phase 3 (graph-quality roadmap) — explicit-ref Pass 1 + embedding reuse."""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import networkx as nx
import pytest

from app.core import graph_orphan_cascade_v2 as cas


# ──────────────────────────────────────────────────────────────────────────
# Fakes
# ──────────────────────────────────────────────────────────────────────────


class _FakeDB:
    def __init__(
        self,
        nodes: Dict[str, Dict[str, Any]],
        embeddings: Optional[Dict[str, List[float]]] = None,
    ) -> None:
        self._nodes = nodes
        self._embeddings = embeddings or {}
        self.calls: List[str] = []

    def get_node(self, node_id: str) -> Optional[Dict[str, Any]]:
        return self._nodes.get(node_id)

    def get_embedding_by_id(self, node_id: str) -> Optional[List[float]]:
        self.calls.append(node_id)
        return self._embeddings.get(node_id)


# ──────────────────────────────────────────────────────────────────────────
# Doc orphans — markdown links + backticks
# ──────────────────────────────────────────────────────────────────────────


class TestExplicitRefsDoc:
    def _build(self, doc_text: str) -> tuple[_FakeDB, nx.MultiDiGraph]:
        nodes = {
            "doc_orphan": {
                "rel_path": "docs/guide/intro.md",
                "symbol_type": "file_doc",
                "source_text": doc_text,
            },
            "auth_module": {
                "rel_path": "src/auth.py",
                "symbol_type": "module",
                "symbol_name": "auth",
            },
            "auth_handler": {
                "rel_path": "src/auth/handler.py",
                "symbol_type": "class",
                "symbol_name": "AuthHandler",
            },
        }
        G = nx.MultiDiGraph()
        for nid, data in nodes.items():
            G.add_node(nid, **data)
        return _FakeDB(nodes=nodes), G

    def test_markdown_link_resolved(self):
        db, G = self._build("See [auth module](../../src/auth.py) for details.")
        out = cas.resolve_orphans_explicit_refs(db, G, ["doc_orphan"])
        assert "doc_orphan" in out
        assert any(h["node_id"] == "auth_module" for h in out["doc_orphan"])
        # First match is md_link with score 0.95.
        assert out["doc_orphan"][0]["_matcher"] == "md_link"
        assert out["doc_orphan"][0]["_raw_score"] == 0.95

    def test_backtick_symbol_resolved(self):
        db, G = self._build("Use `AuthHandler` to handle auth.")
        out = cas.resolve_orphans_explicit_refs(db, G, ["doc_orphan"])
        assert "doc_orphan" in out
        assert any(
            h["node_id"] == "auth_handler" and h["_matcher"] == "backtick"
            for h in out["doc_orphan"]
        )

    def test_anchor_only_link_skipped(self):
        db, G = self._build("[home](#section-1) does not resolve.")
        out = cas.resolve_orphans_explicit_refs(db, G, ["doc_orphan"])
        assert out == {}

    def test_external_url_skipped(self):
        db, G = self._build("[google](https://google.com) is external.")
        out = cas.resolve_orphans_explicit_refs(db, G, ["doc_orphan"])
        assert out == {}

    def test_self_link_filtered(self):
        # Markdown link points back at the doc itself.
        db, G = self._build("[self](intro.md) self-link.")
        out = cas.resolve_orphans_explicit_refs(db, G, ["doc_orphan"])
        assert out == {}

    def test_dotted_backtick_falls_back_to_last_segment(self):
        db, G = self._build("Use `auth.AuthHandler` directly.")
        out = cas.resolve_orphans_explicit_refs(db, G, ["doc_orphan"])
        assert any(h["node_id"] == "auth_handler" for h in out.get("doc_orphan", []))


# ──────────────────────────────────────────────────────────────────────────
# Code orphans — parser-emitted imports
# ──────────────────────────────────────────────────────────────────────────


class TestExplicitRefsCode:
    def _build(self, imports) -> tuple[_FakeDB, nx.MultiDiGraph]:
        nodes = {
            "orphan": {
                "rel_path": "src/api/views.py",
                "symbol_type": "module",
                "symbol_name": "views",
                "imports": imports,
            },
            "auth_module": {
                "rel_path": "src/auth.py",
                "symbol_type": "module",
                "symbol_name": "auth",
            },
            "auth_handler_class": {
                "rel_path": "src/auth/handler.py",
                "symbol_type": "class",
                "symbol_name": "AuthHandler",
            },
        }
        G = nx.MultiDiGraph()
        for nid, data in nodes.items():
            G.add_node(nid, **data)
        return _FakeDB(nodes=nodes), G

    def test_import_path_match(self):
        db, G = self._build(["src/auth.py"])
        out = cas.resolve_orphans_explicit_refs(db, G, ["orphan"])
        assert any(
            h["node_id"] == "auth_module" and h["_matcher"] == "import_path"
            for h in out.get("orphan", [])
        )

    def test_import_symbol_fallback(self):
        # Module path won't resolve, but final segment matches a symbol.
        db, G = self._build(["src.auth.handler.AuthHandler"])
        out = cas.resolve_orphans_explicit_refs(db, G, ["orphan"])
        assert any(
            h["node_id"] == "auth_handler_class" and h["_matcher"] == "import_symbol"
            for h in out.get("orphan", [])
        )

    def test_import_dict_form(self):
        db, G = self._build([{"module": "src.auth", "name": "AuthHandler"}])
        out = cas.resolve_orphans_explicit_refs(db, G, ["orphan"])
        assert any(h["node_id"] == "auth_handler_class" for h in out.get("orphan", []))

    def test_no_imports_returns_nothing(self):
        db, G = self._build([])
        out = cas.resolve_orphans_explicit_refs(db, G, ["orphan"])
        assert out == {}


# ──────────────────────────────────────────────────────────────────────────
# Embedding reuse helper
# ──────────────────────────────────────────────────────────────────────────


class TestEmbeddingReuse:
    def test_collect_returns_persisted(self):
        db = _FakeDB(nodes={}, embeddings={"a": [0.1, 0.2], "b": [0.3, 0.4]})
        out = cas.collect_orphan_embeddings(db, ["a", "b", "c"])
        assert out == {"a": [0.1, 0.2], "b": [0.3, 0.4], "c": None}
        # All three node_ids were queried (chunked == 1 batch).
        assert sorted(db.calls) == ["a", "b", "c"]

    def test_collect_empty_input(self):
        db = _FakeDB(nodes={})
        out = cas.collect_orphan_embeddings(db, [])
        assert out == {}
        assert db.calls == []

    def test_collect_handles_db_error(self):
        class _ErroringDB(_FakeDB):
            def get_embedding_by_id(self, node_id):
                self.calls.append(node_id)
                raise RuntimeError("boom")

        db = _ErroringDB(nodes={})
        out = cas.collect_orphan_embeddings(db, ["x", "y"])
        assert out == {"x": None, "y": None}
        assert sorted(db.calls) == ["x", "y"]

    def test_collect_chunking(self):
        db = _FakeDB(nodes={}, embeddings={f"id{i}": [float(i)] for i in range(7)})
        out = cas.collect_orphan_embeddings(db, [f"id{i}" for i in range(7)], chunk=3)
        assert len(out) == 7
        for i in range(7):
            assert out[f"id{i}"] == [float(i)]
