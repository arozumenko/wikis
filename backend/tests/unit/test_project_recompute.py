"""PR-14 — unit tests for app.services.project_recompute."""

from __future__ import annotations

import dataclasses
from types import SimpleNamespace

import networkx as nx
import pytest

from app.core import feature_flags as ff
from app.core.storage.project_storage import InMemoryProjectStorage
from app.services import project_recompute


def _settings_stub(tmp_path):
    return SimpleNamespace(cache_dir=str(tmp_path / "cache"))


def _wiki_record(wid: str):
    return SimpleNamespace(id=wid, status="complete")


def _components_with_graph(graph):
    return SimpleNamespace(code_graph=graph)


def _build_fake_graph(wiki_id: str, languages=("python",), api=("GET /v1/x",)):
    g = nx.MultiDiGraph()
    g.add_node(
        f"{wiki_id}::A",
        symbol={"language": languages[0], "name": f"{wiki_id}_A"},
        api_surface=[{"surface": api[0]}],
    )
    g.add_node(
        f"{wiki_id}::B",
        symbol={"language": languages[0], "name": f"{wiki_id}_B"},
    )
    g.add_edge(
        f"{wiki_id}::A",
        f"{wiki_id}::B",
        relationship_type="imports",
    )
    return g


@pytest.mark.asyncio
async def test_recompute_skipped_when_flag_off(monkeypatch, tmp_path):
    forced = dataclasses.replace(ff.get_feature_flags(), project_graph=False)
    monkeypatch.setattr(project_recompute, "logger", project_recompute.logger)
    monkeypatch.setattr(
        "app.core.feature_flags.get_feature_flags", lambda: forced
    )
    result = await project_recompute.recompute_project(
        "p1", user_id="u1", storage=object(), settings=_settings_stub(tmp_path)
    )
    assert result == {"status": "skipped", "reason": "project_graph_disabled"}


@pytest.mark.asyncio
async def test_recompute_skipped_with_fewer_than_two_wikis(monkeypatch, tmp_path):
    forced = dataclasses.replace(ff.get_feature_flags(), project_graph=True)
    monkeypatch.setattr(
        "app.core.feature_flags.get_feature_flags", lambda: forced
    )

    class _SvcSingle:
        def __init__(self, *_a, **_kw): ...
        async def list_project_wikis(self, *_a, **_kw):
            return [_wiki_record("w1")]

    class _SF:
        async def __aenter__(self): return SimpleNamespace()
        async def __aexit__(self, *a): return False

    monkeypatch.setattr(
        "app.db.get_session_factory", lambda: (lambda: _SF())
    )
    monkeypatch.setattr(
        "app.services.project_service.ProjectService", _SvcSingle
    )
    monkeypatch.setattr(
        "app.core.storage.project_storage.open_project_storage",
        lambda *_a, **_kw: InMemoryProjectStorage(),
    )

    result = await project_recompute.recompute_project(
        "p1", user_id="u1", storage=object(), settings=_settings_stub(tmp_path)
    )
    assert result["status"] == "skipped"
    assert result["reason"] == "fewer_than_two_wikis"


@pytest.mark.asyncio
async def test_recompute_happy_path_two_wikis(monkeypatch, tmp_path):
    forced = dataclasses.replace(
        ff.get_feature_flags(),
        project_graph=True,
        federated_query=True,
        federated_retriever=True,
        project_clustering=True,
        relatedness_threshold=0.0,  # ensure pair passes
    )
    monkeypatch.setattr(
        "app.core.feature_flags.get_feature_flags", lambda: forced
    )

    class _SvcTwo:
        def __init__(self, *_a, **_kw): ...
        async def list_project_wikis(self, *_a, **_kw):
            return [_wiki_record("w1"), _wiki_record("w2")]

    class _SF:
        async def __aenter__(self): return SimpleNamespace()
        async def __aexit__(self, *a): return False

    monkeypatch.setattr(
        "app.db.get_session_factory", lambda: (lambda: _SF())
    )
    monkeypatch.setattr(
        "app.services.project_service.ProjectService", _SvcTwo
    )

    store = InMemoryProjectStorage()
    monkeypatch.setattr(
        "app.core.storage.project_storage.open_project_storage",
        lambda *_a, **_kw: store,
    )

    async def _fake_build(wid, *_a, **_kw):
        return _components_with_graph(_build_fake_graph(wid))

    monkeypatch.setattr(
        "app.services.toolkit_bridge.build_engine_components", _fake_build
    )

    events: list = []

    async def _on_event(evt):
        events.append(evt)

    result = await project_recompute.recompute_project(
        "p1",
        user_id="u1",
        storage=object(),
        settings=_settings_stub(tmp_path),
        on_event=_on_event,
    )

    assert result["status"] == "ok"
    assert result["wiki_count"] == 2
    assert result["pair_count"] == 1
    assert "recomputed_at" in result and result["recomputed_at"]

    # Relatedness row persisted
    rows = store.get_repo_relatedness("p1")
    assert len(rows) == 1
    assert {rows[0]["wiki_a"], rows[0]["wiki_b"]} == {"w1", "w2"}

    # At least one progress event captured
    assert events, "expected at least one SSE event"


# ---------------------------------------------------------------------
# PR-15 — maybe_enqueue_recompute
# ---------------------------------------------------------------------


@pytest.mark.asyncio
async def test_maybe_enqueue_disabled_when_flag_off(monkeypatch, tmp_path):
    forced = dataclasses.replace(ff.get_feature_flags(), project_graph=False)
    monkeypatch.setattr(
        "app.core.feature_flags.get_feature_flags", lambda: forced
    )
    stale, reason = await project_recompute.maybe_enqueue_recompute(
        "p1", user_id="u1", storage=object(), settings=_settings_stub(tmp_path)
    )
    assert stale is False
    assert reason == "project_graph_disabled"


@pytest.mark.asyncio
async def test_maybe_enqueue_fires_when_never_computed(monkeypatch, tmp_path):
    forced = dataclasses.replace(ff.get_feature_flags(), project_graph=True)
    monkeypatch.setattr(
        "app.core.feature_flags.get_feature_flags", lambda: forced
    )
    store = InMemoryProjectStorage()
    monkeypatch.setattr(
        "app.core.storage.project_storage.open_project_storage",
        lambda *_a, **_kw: store,
    )

    called: list = []

    async def _fake_recompute(*a, **kw):
        called.append((a, kw))
        return {"status": "ok"}

    monkeypatch.setattr(project_recompute, "recompute_project", _fake_recompute)

    stale, reason = await project_recompute.maybe_enqueue_recompute(
        "p1", user_id="u1", storage=object(), settings=_settings_stub(tmp_path)
    )
    # Yield to let the background task run.
    import asyncio as _aio
    await _aio.sleep(0)

    assert stale is True
    assert reason == "never_computed"
    assert called, "background recompute_project should have been scheduled"


@pytest.mark.asyncio
async def test_maybe_enqueue_skips_when_fresh(monkeypatch, tmp_path):
    from datetime import datetime, timezone

    forced = dataclasses.replace(ff.get_feature_flags(), project_graph=True)
    monkeypatch.setattr(
        "app.core.feature_flags.get_feature_flags", lambda: forced
    )
    store = InMemoryProjectStorage()
    store.set_project_meta(
        "p1", "recomputed_at", datetime.now(timezone.utc).isoformat()
    )
    monkeypatch.setattr(
        "app.core.storage.project_storage.open_project_storage",
        lambda *_a, **_kw: store,
    )

    called: list = []

    async def _fake_recompute(*a, **kw):
        called.append((a, kw))
        return {"status": "ok"}

    monkeypatch.setattr(project_recompute, "recompute_project", _fake_recompute)

    stale, reason = await project_recompute.maybe_enqueue_recompute(
        "p1", user_id="u1", storage=object(), settings=_settings_stub(tmp_path)
    )
    assert stale is False
    assert reason == "fresh"
    assert not called
