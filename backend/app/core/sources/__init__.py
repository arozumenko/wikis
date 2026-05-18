"""Source toolkit abstraction layer.

Exports the public API for all source connectors. Concrete toolkit classes
(git, confluence, jira, …) are registered in their own modules — not here.
"""

from __future__ import annotations

from app.core.sources._logging import TokenRedactionFilter
from app.core.sources._logging import install as install_redaction_filter
from app.core.sources.base import FileContent, FileInfo, OriginPointer, SourceToolkit
from app.core.sources.exceptions import (
    SourceAuthError,
    SourceConnectionError,
    SourceError,
    SourceNotFoundError,
    SourceUnavailableError,
)
from app.core.sources.registry import ToolkitRegistry, registry

__all__ = [
    "SourceToolkit",
    "FileInfo",
    "FileContent",
    "OriginPointer",
    "SourceError",
    "SourceAuthError",
    "SourceConnectionError",
    "SourceNotFoundError",
    "SourceUnavailableError",
    "ToolkitRegistry",
    "registry",
    "TokenRedactionFilter",
    "install_redaction_filter",
]
