"""Tests for API route error handling — validation errors and 404s."""

from __future__ import annotations

# --------------------------------------------------------------------------- #
# POST /api/v1/generate — validation
# --------------------------------------------------------------------------- #


async def test_generate_missing_repo_url(client):
    """POST /generate with body missing repo_url -> 422."""
    resp = await client.post("/api/v1/generate", json={"branch": "main"})
    assert resp.status_code == 422


async def test_generate_empty_body(client):
    """POST /generate with empty JSON body -> 422."""
    resp = await client.post("/api/v1/generate", json={})
    assert resp.status_code == 422


# --------------------------------------------------------------------------- #
# GET /api/v1/invocations/<nonexistent> — 404
# --------------------------------------------------------------------------- #


async def test_get_invocation_not_found(client):
    """GET /invocations/nonexistent-id -> 404."""
    resp = await client.get("/api/v1/invocations/nonexistent-id")
    assert resp.status_code == 404


# --------------------------------------------------------------------------- #
# DELETE /api/v1/wikis/<nonexistent> — 404
# --------------------------------------------------------------------------- #


async def test_delete_wiki_not_found(client):
    """DELETE /wikis/nonexistent -> 404."""
    resp = await client.delete("/api/v1/wikis/nonexistent")
    assert resp.status_code == 404


# --------------------------------------------------------------------------- #
# POST /api/v1/ask — validation
# --------------------------------------------------------------------------- #


async def test_ask_missing_wiki_id(client):
    """POST /ask with missing wiki_id -> 422."""
    resp = await client.post("/api/v1/ask", json={"question": "What is this?"})
    assert resp.status_code == 422


# --------------------------------------------------------------------------- #
# POST /api/v1/research — validation
# --------------------------------------------------------------------------- #


async def test_research_missing_question(client):
    """POST /research with missing question -> 422."""
    resp = await client.post("/api/v1/research", json={"wiki_id": "some-wiki"})
    assert resp.status_code == 422
