"""Abstract artifact storage interface."""

from __future__ import annotations

from abc import ABC, abstractmethod


class ArtifactStorage(ABC):
    """Base class for artifact storage backends."""

    @abstractmethod
    async def upload(self, bucket: str, name: str, data: bytes) -> str:
        """Store artifact, return storage key."""

    @abstractmethod
    async def download(self, bucket: str, name: str) -> bytes:
        """Retrieve artifact by bucket/name."""

    @abstractmethod
    async def delete(self, bucket: str, name: str) -> None:
        """Delete artifact."""

    @abstractmethod
    async def list_artifacts(self, bucket: str, prefix: str = "") -> list[str]:
        """List artifact names in bucket."""
