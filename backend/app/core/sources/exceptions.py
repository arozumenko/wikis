"""Source toolkit exception hierarchy."""

from __future__ import annotations


class SourceError(Exception):
    """Base exception for all source toolkit errors."""


class SourceAuthError(SourceError):
    """Authentication or credential failure."""


class SourceConnectionError(SourceError):
    """Source system is unreachable or the connection attempt failed."""


class SourceNotFoundError(SourceError):
    """Requested resource does not exist in the source."""


class SourceUnavailableError(SourceError):
    """Source system is temporarily unavailable or rate-limited."""
