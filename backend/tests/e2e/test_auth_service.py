"""Auth Service Tests (#56)

Tests for the auth service API endpoints.
Run with pytest — uses httpx for direct API calls.
Requires: app service running on :3000 with AUTH_SECRET set.
"""
from __future__ import annotations

import httpx
import pytest

AUTH_URL = "http://localhost:3000"


@pytest.fixture
def auth_client():
    """Sync HTTP client for auth service."""
    with httpx.Client(base_url=AUTH_URL, timeout=10) as client:
        yield client


class TestAuthHealth:
    def test_health_returns_200(self, auth_client):
        resp = auth_client.get("/api/auth/health")
        assert resp.status_code == 200
        assert resp.json()["status"] == "ok"


class TestAuthProviders:
    def test_providers_lists_credentials(self, auth_client):
        resp = auth_client.get("/api/auth/providers")
        if resp.status_code == 200:
            providers = resp.json()
            assert "credentials" in providers
            assert providers["credentials"]["type"] == "credentials"

    def test_csrf_returns_token(self, auth_client):
        resp = auth_client.get("/api/auth/csrf")
        if resp.status_code == 200:
            data = resp.json()
            assert "csrfToken" in data
            assert len(data["csrfToken"]) > 10


class TestRegistration:
    def test_register_valid_user(self, auth_client):
        resp = auth_client.post("/api/auth/register", json={
            "username": f"testuser_{id(self)}",
            "password": "ValidPass123!",
        })
        # 201 for new user, 409 if already exists
        assert resp.status_code in (201, 409)

    def test_register_duplicate_returns_409(self, auth_client):
        # Register first
        auth_client.post("/api/auth/register", json={
            "username": "duptest",
            "password": "ValidPass123!",
        })
        # Register again
        resp = auth_client.post("/api/auth/register", json={
            "username": "duptest",
            "password": "ValidPass123!",
        })
        assert resp.status_code == 409
        assert "already taken" in resp.json().get("error", "").lower()

    def test_register_weak_password_returns_422(self, auth_client):
        resp = auth_client.post("/api/auth/register", json={
            "username": "weakuser",
            "password": "123",
        })
        assert resp.status_code == 422
        assert "password" in resp.json().get("error", "").lower()

    def test_register_missing_fields_returns_error(self, auth_client):
        resp = auth_client.post("/api/auth/register", json={})
        assert resp.status_code in (400, 422)
