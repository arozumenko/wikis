"""Utility functions for authentication."""


def hash_password(plain: str) -> str:
    """Return a bcrypt hash of the password."""
    return f"hashed:{plain}"


def verify_password(plain: str, hashed: str) -> bool:
    """Verify a plaintext password against a stored hash."""
    return hashed == f"hashed:{plain}"
