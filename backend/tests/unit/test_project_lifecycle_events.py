"""Phase 9 — Project lifecycle SSE events."""

from __future__ import annotations

import json

from app.events import (
    PHASE_CROSS_REPO_LINKER,
    PHASE_PROJECT_CLUSTERING,
    PHASE_PROJECT_RELATEDNESS,
    cross_repo_linker_progress,
    project_clustering_progress,
    project_relatedness_progress,
)


def _payload(evt) -> dict:
    return json.loads(evt.model_dump_json())


class TestRelatednessEvent:
    def test_basic(self):
        evt = project_relatedness_progress(
            "tok-1", 3, 10, "scoring", project_id="p-42"
        )
        assert evt.event == "project_relatedness_progress"
        body = _payload(evt)
        assert body["jsonrpc"] == "2.0"
        assert body["method"] == "notifications/progress"
        params = body["params"]
        assert params["progress"] == 3 and params["total"] == 10
        assert params["_phase"] == PHASE_PROJECT_RELATEDNESS
        assert params["_projectId"] == "p-42"

    def test_optional_project_id_omitted(self):
        evt = project_relatedness_progress("tok-1", 1, 1, "done")
        params = _payload(evt)["params"]
        assert "_projectId" not in params


class TestCrossRepoLinkerEvent:
    def test_pair_serialised(self):
        evt = cross_repo_linker_progress(
            "tok-1", 1, 3, "linking",
            project_id="p-1", pair=("wiki-a", "wiki-b"),
        )
        assert evt.event == "cross_repo_linker_progress"
        params = _payload(evt)["params"]
        assert params["_phase"] == PHASE_CROSS_REPO_LINKER
        assert params["_pair"] == ["wiki-a", "wiki-b"]

    def test_invalid_pair_omitted(self):
        evt = cross_repo_linker_progress(
            "tok-1", 1, 3, "linking", pair=("only-one",)  # type: ignore[arg-type]
        )
        assert "_pair" not in _payload(evt)["params"]


class TestClusteringEvent:
    def test_community_count(self):
        evt = project_clustering_progress(
            "tok-1", 1, 1, "done", project_id="p-1", community_count=12
        )
        assert evt.event == "project_clustering_progress"
        params = _payload(evt)["params"]
        assert params["_phase"] == PHASE_PROJECT_CLUSTERING
        assert params["_communityCount"] == 12

    def test_zero_community_count_kept(self):
        evt = project_clustering_progress(
            "tok-1", 1, 1, "done", community_count=0
        )
        # Zero is meaningful — explicitly test it survives the int() cast.
        assert _payload(evt)["params"]["_communityCount"] == 0
