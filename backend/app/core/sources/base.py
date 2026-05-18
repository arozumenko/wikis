"""Abstract base classes and data types for source toolkits."""

from __future__ import annotations

import abc
from dataclasses import dataclass
from datetime import datetime
from typing import AsyncIterator


@dataclass(frozen=True)
class OriginPointer:
    """Lightweight reference back to a file in its origin system.

    Args:
        source_type: Identifies the connector (e.g. "github", "confluence").
        ref: System-specific locator — repo-relative path, page ID, etc.
        url: Direct link for web-based systems; may be empty for local sources.
        ingested_at: When this pointer was created.
        revision: Commit SHA, etag, or version identifier.
        line_start: First line of the referenced section.
        line_end: Last line of the referenced section.
    """

    source_type: str
    ref: str
    url: str
    ingested_at: datetime
    revision: str | None = None
    line_start: int | None = None
    line_end: int | None = None


@dataclass(frozen=True)
class FileInfo:
    """Metadata about a file — no content.

    Args:
        path: Source-relative path.
        language: Detected language identifier, or None if unknown.
        size_bytes: File size in bytes.
        origin: Back-reference to the originating source.
    """

    path: str
    language: str | None
    size_bytes: int
    origin: OriginPointer


@dataclass(frozen=True)
class FileContent:
    """Full content of a single file.

    Args:
        info: Metadata for this file.
        content: Raw text content.
        encoding: Character encoding of *content* (default ``"utf-8"``).
    """

    info: FileInfo
    content: str
    encoding: str = "utf-8"


class SourceToolkit(abc.ABC):
    """Abstract base for all source adapters.

    Subclasses must declare a ``source_type`` class attribute and implement
    all abstract methods.
    """

    source_type: str  # overridden by each concrete subclass

    @abc.abstractmethod
    async def list_files(
        self,
        include: list[str] | None = None,
        exclude: list[str] | None = None,
    ) -> AsyncIterator[FileInfo]:
        """Yield FileInfo for every file reachable in this source.

        Args:
            include: Optional glob patterns; only yield files matching at least one.
            exclude: Optional glob patterns; skip files matching any of these.
        """
        ...

    @abc.abstractmethod
    async def fetch_content(self, pointer: OriginPointer) -> FileContent:
        """Fetch full content for the file identified by *pointer*."""
        ...

    @abc.abstractmethod
    async def test_connection(self) -> str:
        """Verify that the source is reachable with current credentials.

        Returns:
            A human-readable status message.
        """
        ...

    @abc.abstractmethod
    def build_origin_pointer(
        self,
        path: str,
        revision: str | None = None,
        line_start: int | None = None,
        line_end: int | None = None,
    ) -> OriginPointer:
        """Construct an OriginPointer for a file at *path*.

        Args:
            path: Source-relative file path.
            revision: Commit SHA, tag, or version identifier.
            line_start: First line of the referenced section.
            line_end: Last line of the referenced section.
        """
        ...

    @classmethod
    @abc.abstractmethod
    def from_config(cls, config: dict) -> "SourceToolkit":
        """Instantiate the toolkit from a plain credentials dict.

        Args:
            config: Credential and configuration values — no encryption assumed.
        """
        ...

    async def __aenter__(self) -> "SourceToolkit":
        """Async context entry. Default: no-op. Override for resource setup."""
        return self

    async def __aexit__(self, exc_type: object, exc_val: object, exc_tb: object) -> None:
        """Async context exit. Default: no-op. Override for resource cleanup."""
        return None
