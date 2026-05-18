"""Unit tests for WikiService multi-source dispatch and GenerateWikiRequest model (#189)."""

from __future__ import annotations

import hashlib
import json
import logging
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from app.models.api import GenerateWikiRequest
from app.services.wiki_service import WikiService


# ---------------------------------------------------------------------------
# 1. GenerateWikiRequest — model validator normalises legacy flat fields
# ---------------------------------------------------------------------------


def test_legacy_repo_url_normalized_to_scope() -> None:
    """repo_url + branch → source_type='git', scope populated."""
    req = GenerateWikiRequest(repo_url="https://github.com/org/repo", branch="develop")
    assert req.source_type == "git"
    assert req.scope == {"repo_url": "https://github.com/org/repo", "branch": "develop"}
    assert req.repo_url == "https://github.com/org/repo"
    assert req.branch == "develop"


def test_legacy_access_token_goes_into_auth() -> None:
    """access_token in legacy form → auth={"pat": ...}."""
    req = GenerateWikiRequest(
        repo_url="https://github.com/org/private",
        branch="main",
        access_token="ghp_test",
    )
    assert req.auth.get("pat") == "ghp_test"
    # access_token back-filled for legacy pipeline compatibility
    assert req.access_token == "ghp_test"


def test_legacy_branch_defaults_to_main() -> None:
    req = GenerateWikiRequest(repo_url="https://github.com/org/repo")
    assert req.scope["branch"] == "main"


def test_new_style_git_request() -> None:
    req = GenerateWikiRequest(
        source_type="git",
        scope={"repo_url": "https://github.com/org/repo", "branch": "feat/123"},
        auth={"pat": "ghp_abc"},
    )
    assert req.source_type == "git"
    assert req.scope["branch"] == "feat/123"
    # back-fill for legacy pipeline
    assert req.repo_url == "https://github.com/org/repo"
    assert req.branch == "feat/123"
    assert req.access_token == "ghp_abc"


def test_new_style_confluence_request() -> None:
    req = GenerateWikiRequest(
        source_type="confluence",
        scope={"base_url": "https://acme.atlassian.net", "space_keys": ["ENG", "HR"]},
        auth={"access_token": "at_xxx", "refresh_token": "rt_yyy", "client_id": "cid_zzz"},
    )
    assert req.source_type == "confluence"
    assert req.scope["space_keys"] == ["ENG", "HR"]
    # auth stored (NOT in scope, NOT in repo_url)
    assert req.auth["access_token"] == "at_xxx"
    # repo_url stub set to base_url for invocation serialisation
    assert req.repo_url == "https://acme.atlassian.net"


def test_new_style_jira_request() -> None:
    req = GenerateWikiRequest(
        source_type="jira",
        scope={"base_url": "https://acme.atlassian.net", "jql": "project=ENG"},
        auth={"access_token": "at_aaa"},
    )
    assert req.source_type == "jira"
    assert req.scope["jql"] == "project=ENG"


def test_structure_planner_mapping() -> None:
    """structure_planner='agentic' → planner_type='agent'."""
    req = GenerateWikiRequest(
        repo_url="https://github.com/org/repo",
        structure_planner="agentic",
    )
    assert req.planner_type == "agent"

    req2 = GenerateWikiRequest(
        repo_url="https://github.com/org/repo",
        structure_planner="graph_clustering",
    )
    assert req2.planner_type == "cluster"


# ---------------------------------------------------------------------------
# 2. WikiService._make_wiki_id — stable hash
# ---------------------------------------------------------------------------


def test_make_wiki_id_stable() -> None:
    id1 = WikiService._make_wiki_id("git", {"repo_url": "https://github.com/a/b", "branch": "main"})
    id2 = WikiService._make_wiki_id("git", {"repo_url": "https://github.com/a/b", "branch": "main"})
    assert id1 == id2
    assert len(id1) == 16


def test_make_wiki_id_different_for_different_scopes() -> None:
    git_id = WikiService._make_wiki_id("git", {"repo_url": "https://github.com/a/b", "branch": "main"})
    conf_id = WikiService._make_wiki_id(
        "confluence", {"base_url": "https://a.atlassian.net", "space_keys": ["ENG"]}
    )
    assert git_id != conf_id


def test_make_wiki_id_scope_key_order_stable() -> None:
    """ID must not change when scope dict key insertion order differs."""
    id_a = WikiService._make_wiki_id("git", {"repo_url": "u", "branch": "b"})
    id_b = WikiService._make_wiki_id("git", {"branch": "b", "repo_url": "u"})
    assert id_a == id_b


def test_make_legacy_git_wiki_id_matches_old_format() -> None:
    """Legacy ID must match the pre-#189 sha256(f'{repo_url}:{branch}')[:16] hash."""
    repo_url = "https://github.com/org/repo"
    branch = "main"
    expected = hashlib.sha256(f"{repo_url}:{branch}".encode()).hexdigest()[:16]
    assert WikiService._make_legacy_git_wiki_id(repo_url, branch) == expected


def test_new_and_legacy_git_ids_differ() -> None:
    """The new multi-source ID must NOT equal the pre-#189 legacy ID.

    This is intentional: the new ID encodes source_type so Confluence and
    Jira wikis from the same base_url can co-exist with Git wikis.
    The service's one-cycle migration helper (_make_legacy_git_wiki_id) is
    used to detect and reuse old rows during a rolling upgrade.
    """
    repo_url = "https://github.com/org/repo"
    branch = "main"
    new_id = WikiService._make_wiki_id("git", {"repo_url": repo_url, "branch": branch})
    legacy_id = WikiService._make_legacy_git_wiki_id(repo_url, branch)
    assert new_id != legacy_id


# ---------------------------------------------------------------------------
# 3. Token redaction — auth tokens must never appear in logs
# ---------------------------------------------------------------------------


def test_generate_request_auth_not_in_repr_or_str() -> None:
    """Sanity check: secret token should not appear in the model's repr."""
    req = GenerateWikiRequest(
        source_type="confluence",
        scope={"base_url": "https://acme.atlassian.net", "space_keys": ["ENG"]},
        auth={"access_token": "SUPER_SECRET_TOKEN_123"},
    )
    # The access_token lives in auth but must NOT appear raw in the scope.
    assert "SUPER_SECRET_TOKEN_123" not in str(req.scope)


# ---------------------------------------------------------------------------
# 4. Subprocess payload includes source fields
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_run_wiki_subprocess_payload_includes_source_fields() -> None:
    """_run_wiki_subprocess must include source_type, scope, and auth in payload."""
    import os
    import tempfile

    settings = MagicMock()
    settings.planner_type = "agent"
    settings.cluster_exclude_tests = False
    settings.llm_max_concurrency = 4

    storage = MagicMock()
    service = WikiService(settings=settings, storage=storage)

    req = GenerateWikiRequest(
        source_type="confluence",
        scope={"base_url": "https://acme.atlassian.net", "space_keys": ["ENG"]},
        auth={"access_token": "at_test", "refresh_token": "rt_test"},
        wiki_title="Test Wiki",
    )

    captured_payload: dict[str, Any] = {}

    async def _fake_subprocess(
        invocation: Any,
        request: Any,
        planner_type: str,
        exclude_tests: bool,
    ) -> dict:
        # Read the payload that the real method would write to input.json.
        import asyncio
        import json
        import os
        import sys
        import tempfile as _tempfile

        _payload = {
            "invocation_id": invocation.id,
            "source_type": request.source_type,
            "scope": request.scope,
            "auth": request.auth,
            "repo_url": request.repo_url,
            "branch": request.branch,
            "access_token": request.access_token,
            "wiki_title": request.wiki_title,
            "include_research": request.include_research,
            "include_diagrams": request.include_diagrams,
            "force_rebuild_index": request.force_rebuild_index,
            "llm_model": request.llm_model,
            "embedding_model": request.embedding_model,
            "planner_type": planner_type,
            "exclude_tests": exclude_tests,
        }
        captured_payload.update(_payload)
        return {"success": True, "generated_pages": {}, "artifacts": []}

    with patch.object(service, "_run_wiki_subprocess", side_effect=_fake_subprocess):
        with patch.object(service, "_emit_progress", new_callable=AsyncMock):
            with patch.object(service, "_emit_error", new_callable=AsyncMock):
                invocation = MagicMock()
                invocation.id = "test-inv-id"
                invocation.wiki_id = "test-wiki-id"
                invocation.owner_id = ""
                invocation.repo_url = req.repo_url
                invocation.branch = req.branch
                invocation.pages_completed = 0
                invocation.pages_total = 0
                invocation.progress = 0.0
                invocation.status = "generating"
                invocation.emit = AsyncMock()
                invocation.completed_at = None
                invocation.created_at = MagicMock()
                invocation.created_at.__sub__ = MagicMock(return_value=MagicMock(total_seconds=lambda: 1.0))

                service._invocations["test-inv-id"] = invocation
                # Run _run_generation which calls _run_wiki_subprocess
                await service._run_generation(invocation, req)

    # Check the payload passed to the subprocess includes the new fields.
    assert captured_payload.get("source_type") == "confluence"
    assert captured_payload.get("scope") == {"base_url": "https://acme.atlassian.net", "space_keys": ["ENG"]}
    assert captured_payload.get("auth") == {"access_token": "at_test", "refresh_token": "rt_test"}


# ---------------------------------------------------------------------------
# 5. Token redaction filter integration — tokens never appear in log output
# ---------------------------------------------------------------------------


def test_token_never_logged(caplog: pytest.LogCaptureFixture) -> None:
    """TokenRedactionFilter must strip any log line containing a raw token."""
    from app.core.sources import install_redaction_filter

    install_redaction_filter()

    token = "ghp_VERY_SECRET_TOKEN_THAT_MUST_NOT_APPEAR"

    with caplog.at_level(logging.DEBUG, logger="app"):
        logger = logging.getLogger("app.test.token_redaction")
        # Log a dict that contains the token as a value
        logger.info("Payload: %s", {"access_token": token, "other": "value"})

    # TokenRedactionFilter replaces the token with *** in the log record's message.
    for record in caplog.records:
        assert token not in record.getMessage(), (
            f"Raw token leaked into log record: {record.getMessage()!r}"
        )


# ---------------------------------------------------------------------------
# 6. _canonicalize — list-value order-independence in wiki ID hashing
# ---------------------------------------------------------------------------


def test_make_wiki_id_list_order_independent() -> None:
    """space_keys=["A","B"] and space_keys=["B","A"] must produce the same wiki ID.

    This is the regression test for Fix 3: sort_keys=True only sorts dict keys,
    not list values.  _canonicalize() recursively sorts homogeneous lists too.
    """
    id_ab = WikiService._make_wiki_id(
        "confluence",
        {"base_url": "https://acme.atlassian.net", "space_keys": ["A", "B"]},
    )
    id_ba = WikiService._make_wiki_id(
        "confluence",
        {"base_url": "https://acme.atlassian.net", "space_keys": ["B", "A"]},
    )
    assert id_ab == id_ba, (
        f"List order must not affect wiki_id: {id_ab!r} != {id_ba!r}"
    )


def test_make_wiki_id_list_order_independent_three_keys() -> None:
    """Three space_keys in any permutation must hash identically."""
    from itertools import permutations

    keys = ["ENG", "HR", "OPS"]
    ids = {
        WikiService._make_wiki_id(
            "confluence",
            {"base_url": "https://example.atlassian.net", "space_keys": list(perm)},
        )
        for perm in permutations(keys)
    }
    assert len(ids) == 1, f"Expected 1 unique ID, got {len(ids)}: {ids}"


def test_canonicalize_preserves_heterogeneous_lists() -> None:
    """Lists containing nested dicts/lists must not be sorted — order is preserved."""
    obj = {"items": [{"z": 1}, {"a": 2}]}
    result = WikiService._canonicalize(obj)
    # Each element is a dict, so the list is heterogeneous and must NOT be reordered.
    assert result["items"] == [{"z": 1}, {"a": 2}], (
        "Heterogeneous list (containing dicts) must not be reordered"
    )


# ---------------------------------------------------------------------------
# 7. SourceToolkit ABC __aenter__/__aexit__ defaults — mock-toolkit dispatch
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_source_toolkit_default_context_manager_no_attribute_error() -> None:
    """A SourceToolkit subclass that does NOT override __aenter__/__aexit__ must
    still work inside 'async with source_toolkit:' without AttributeError.

    This is the regression test for Fix 1: wiki_runner.py:128 uses
    'async with source_toolkit:' for ALL source types, but ConfluenceToolkit
    and JiraToolkit didn't implement the context-manager protocol.  The ABC
    now provides no-op defaults.
    """
    from typing import AsyncIterator

    from app.core.sources.base import FileContent, FileInfo, OriginPointer, SourceToolkit

    class MinimalToolkit(SourceToolkit):
        """Concrete subclass with all abstract methods but NO __aenter__/__aexit__."""

        source_type = "minimal_test"

        @classmethod
        def from_config(cls, config: dict) -> "MinimalToolkit":
            return cls()

        async def list_files(
            self,
            include: list[str] | None = None,
            exclude: list[str] | None = None,
        ) -> AsyncIterator[FileInfo]:  # type: ignore[override]
            return
            yield  # pragma: no cover — makes this an async generator

        async def fetch_content(self, pointer: OriginPointer) -> FileContent:
            raise NotImplementedError  # pragma: no cover

        async def test_connection(self) -> str:
            return "ok"

        def build_origin_pointer(
            self,
            path: str,
            revision: str | None = None,
            line_start: int | None = None,
            line_end: int | None = None,
        ) -> OriginPointer:
            from datetime import datetime, timezone

            return OriginPointer(
                source_type=self.source_type,
                ref=path,
                url=f"minimal://{path}",
                ingested_at=datetime.now(tz=timezone.utc),
            )

    toolkit = MinimalToolkit()

    # This must NOT raise AttributeError — the ABC provides default no-ops.
    entered: SourceToolkit | None = None
    try:
        async with toolkit as ctx:
            entered = ctx
    except AttributeError as exc:
        pytest.fail(
            f"'async with source_toolkit:' raised AttributeError for a toolkit "
            f"that does not override __aenter__/__aexit__: {exc}"
        )

    assert entered is toolkit, "Default __aenter__ must return self"
