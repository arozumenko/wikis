"""Tests for WikiRecord ORM model and wiki-related API models."""

from __future__ import annotations

from app.models.api import UpdateWikiDescriptionRequest, WikiSummary
from app.models.db_models import WikiRecord


# ---------------------------------------------------------------------------
# WikiRecord ORM model
# ---------------------------------------------------------------------------


def test_wiki_record_description_defaults_to_none():
    """WikiRecord instantiation without description sets it to None."""
    record = WikiRecord(
        id="owner--repo--main",
        owner_id="user-1",
        repo_url="https://github.com/owner/repo",
        branch="main",
        title="Test Wiki",
    )
    assert record.description is None


def test_wiki_record_description_accepts_string():
    """WikiRecord instantiation with description stores the value."""
    record = WikiRecord(
        id="owner--repo--main",
        owner_id="user-1",
        repo_url="https://github.com/owner/repo",
        branch="main",
        title="Test Wiki",
        description="A helpful description of this wiki.",
    )
    assert record.description == "A helpful description of this wiki."


# ---------------------------------------------------------------------------
# WikiSummary Pydantic model
# ---------------------------------------------------------------------------


def test_wiki_summary_description_defaults_to_none():
    """WikiSummary omitting description field returns None."""
    from datetime import datetime

    summary = WikiSummary(
        wiki_id="owner--repo--main",
        repo_url="https://github.com/owner/repo",
        branch="main",
        title="Test Wiki",
        created_at=datetime(2024, 1, 1),
        page_count=5,
    )
    assert summary.description is None


def test_wiki_summary_accepts_description():
    """WikiSummary accepts optional description field."""
    from datetime import datetime

    summary = WikiSummary(
        wiki_id="owner--repo--main",
        repo_url="https://github.com/owner/repo",
        branch="main",
        title="Test Wiki",
        created_at=datetime(2024, 1, 1),
        page_count=5,
        description="Describes the repository architecture.",
    )
    assert summary.description == "Describes the repository architecture."


# ---------------------------------------------------------------------------
# UpdateWikiDescriptionRequest Pydantic model
# ---------------------------------------------------------------------------


def test_update_wiki_description_request_none():
    """UpdateWikiDescriptionRequest accepts description=None."""
    req = UpdateWikiDescriptionRequest(description=None)
    assert req.description is None


def test_update_wiki_description_request_string():
    """UpdateWikiDescriptionRequest accepts a string description."""
    req = UpdateWikiDescriptionRequest(description="New description text.")
    assert req.description == "New description text."


def test_update_wiki_description_request_default_none():
    """UpdateWikiDescriptionRequest defaults description to None when omitted."""
    req = UpdateWikiDescriptionRequest()
    assert req.description is None
