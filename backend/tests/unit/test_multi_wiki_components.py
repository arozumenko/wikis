"""Unit tests for build_multi_wiki_components — project description injection.

Covers:
- Project description overrides repo_analysis["description"] when present
- No project description → individual wiki description flows through unaffected
- Exception when fetching project record is swallowed; wiki description intact
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_engine_components(repo_analysis: dict | None = None) -> MagicMock:
    comp = MagicMock()
    comp.retriever_stack = MagicMock()
    comp.graph_manager = MagicMock()
    comp.code_graph = MagicMock()
    comp.repo_analysis = repo_analysis
    comp.llm = MagicMock()
    comp.repo_path = "/tmp/repo"
    return comp


def _make_wiki_record(wiki_id: str) -> MagicMock:
    w = MagicMock()
    w.id = wiki_id
    return w


def _make_project_record(description: str | None) -> MagicMock:
    rec = MagicMock()
    rec.description = description
    return rec


def _make_session_ctx(scalar_return=None, execute_side_effect=None):
    """Build an async context manager that wraps a fake SQLAlchemy session.

    When execute_side_effect is provided the session.execute call raises that
    exception instead of returning a normal result.
    """
    mock_result = MagicMock()
    mock_result.scalar_one_or_none = MagicMock(return_value=scalar_return)

    mock_session = AsyncMock()
    if execute_side_effect is not None:
        mock_session.execute = AsyncMock(side_effect=execute_side_effect)
    else:
        mock_session.execute = AsyncMock(return_value=mock_result)

    ctx = AsyncMock()
    ctx.__aenter__ = AsyncMock(return_value=mock_session)
    ctx.__aexit__ = AsyncMock(return_value=False)
    return ctx


def _build_fake_factory(first_ctx, second_ctx):
    """Return a session factory that yields first_ctx on call 0, second_ctx on call 1+."""
    call_count = [0]

    def factory():
        idx = call_count[0]
        call_count[0] += 1
        return first_ctx if idx == 0 else second_ctx

    return factory


def _common_patches(fake_factory, wiki_comp, proj_record=None, execute_side_effect=None):
    """Assemble all patches needed by every test variant.

    Returns a context-manager stack (via contextlib.ExitStack in the test body).
    """
    import contextlib

    first_ctx = _make_session_ctx()  # used by ProjectService.list_project_wikis
    second_ctx = _make_session_ctx(
        scalar_return=proj_record,
        execute_side_effect=execute_side_effect,
    )
    factory = _build_fake_factory(first_ctx, second_ctx)

    stack = contextlib.ExitStack()

    mock_ps_instance = AsyncMock()
    mock_ps_instance.list_project_wikis = AsyncMock(
        return_value=[_make_wiki_record("wiki-1")]
    )

    stack.enter_context(patch("app.db.get_session_factory", return_value=factory))
    stack.enter_context(
        patch("app.services.project_service.ProjectService", return_value=mock_ps_instance)
    )
    stack.enter_context(
        patch(
            "app.services.multi_wiki_components.build_engine_components",
            new_callable=AsyncMock,
            return_value=wiki_comp,
        )
    )
    stack.enter_context(
        patch("app.core.multi_retriever.MultiWikiRetrieverStack", return_value=MagicMock())
    )
    return stack


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_project_description_overrides_repo_analysis():
    """When the project has a description, merged.repo_analysis["description"] equals it."""
    from app.services.multi_wiki_components import build_multi_wiki_components
    from app.config import Settings
    from pydantic import SecretStr

    wiki_comp = _make_engine_components(repo_analysis={"description": "wiki-level desc"})
    proj_record = _make_project_record(description="Project-level description")

    first_ctx = _make_session_ctx()
    second_ctx = _make_session_ctx(scalar_return=proj_record)
    fake_factory = _build_fake_factory(first_ctx, second_ctx)

    mock_ps_instance = AsyncMock()
    mock_ps_instance.list_project_wikis = AsyncMock(return_value=[_make_wiki_record("wiki-1")])

    settings = Settings(llm_api_key=SecretStr("test"))

    with (
        patch("app.db.get_session_factory", return_value=fake_factory),
        patch("app.services.project_service.ProjectService", return_value=mock_ps_instance),
        patch(
            "app.services.multi_wiki_components.build_engine_components",
            new_callable=AsyncMock,
            return_value=wiki_comp,
        ),
        patch("app.core.multi_retriever.MultiWikiRetrieverStack", return_value=MagicMock()),
    ):
        merged, loaded = await build_multi_wiki_components(
            project_id="proj-1",
            user_id="user-1",
            storage=MagicMock(),
            settings=settings,
        )

    assert merged.repo_analysis["description"] == "Project-level description"


@pytest.mark.asyncio
async def test_no_project_description_leaves_wiki_description_intact():
    """When the project has no description, repo_analysis["description"] keeps the wiki value."""
    from app.services.multi_wiki_components import build_multi_wiki_components
    from app.config import Settings
    from pydantic import SecretStr

    wiki_comp = _make_engine_components(repo_analysis={"description": "wiki-level desc"})
    proj_record = _make_project_record(description=None)  # no project-level desc

    first_ctx = _make_session_ctx()
    second_ctx = _make_session_ctx(scalar_return=proj_record)
    fake_factory = _build_fake_factory(first_ctx, second_ctx)

    mock_ps_instance = AsyncMock()
    mock_ps_instance.list_project_wikis = AsyncMock(return_value=[_make_wiki_record("wiki-1")])

    settings = Settings(llm_api_key=SecretStr("test"))

    with (
        patch("app.db.get_session_factory", return_value=fake_factory),
        patch("app.services.project_service.ProjectService", return_value=mock_ps_instance),
        patch(
            "app.services.multi_wiki_components.build_engine_components",
            new_callable=AsyncMock,
            return_value=wiki_comp,
        ),
        patch("app.core.multi_retriever.MultiWikiRetrieverStack", return_value=MagicMock()),
    ):
        merged, loaded = await build_multi_wiki_components(
            project_id="proj-1",
            user_id="user-1",
            storage=MagicMock(),
            settings=settings,
        )

    # No override — wiki-level description must be preserved
    assert merged.repo_analysis["description"] == "wiki-level desc"


@pytest.mark.asyncio
async def test_exception_fetching_project_is_swallowed():
    """DB error when fetching ProjectRecord is swallowed; no crash, wiki description intact."""
    from app.services.multi_wiki_components import build_multi_wiki_components
    from app.config import Settings
    from pydantic import SecretStr

    wiki_comp = _make_engine_components(repo_analysis={"description": "wiki-level desc"})

    first_ctx = _make_session_ctx()

    error_session = AsyncMock()
    error_session.execute = AsyncMock(side_effect=RuntimeError("DB unavailable"))
    error_ctx = AsyncMock()
    error_ctx.__aenter__ = AsyncMock(return_value=error_session)
    error_ctx.__aexit__ = AsyncMock(return_value=False)

    fake_factory = _build_fake_factory(first_ctx, error_ctx)

    mock_ps_instance = AsyncMock()
    mock_ps_instance.list_project_wikis = AsyncMock(return_value=[_make_wiki_record("wiki-1")])

    settings = Settings(llm_api_key=SecretStr("test"))

    with (
        patch("app.db.get_session_factory", return_value=fake_factory),
        patch("app.services.project_service.ProjectService", return_value=mock_ps_instance),
        patch(
            "app.services.multi_wiki_components.build_engine_components",
            new_callable=AsyncMock,
            return_value=wiki_comp,
        ),
        patch("app.core.multi_retriever.MultiWikiRetrieverStack", return_value=MagicMock()),
    ):
        # Must not raise
        merged, loaded = await build_multi_wiki_components(
            project_id="proj-1",
            user_id="user-1",
            storage=MagicMock(),
            settings=settings,
        )

    # Exception was swallowed — wiki description untouched
    assert merged.repo_analysis["description"] == "wiki-level desc"
