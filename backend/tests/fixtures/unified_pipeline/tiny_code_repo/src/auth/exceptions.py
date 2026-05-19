"""Auth-specific exceptions."""


class AuthError(Exception):
    """Base class for auth errors."""


class InvalidCredentials(AuthError):
    """Raised when login credentials are wrong."""


class SessionExpired(AuthError):
    """Raised when a session has expired."""
