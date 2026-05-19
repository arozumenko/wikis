"""Data models for the auth module."""
from dataclasses import dataclass


@dataclass
class User:
    """Represents an authenticated user."""

    user_id: str
    username: str
    email: str


@dataclass
class Session:
    """Active user session."""

    session_id: str
    user_id: str
    expires_at: str
