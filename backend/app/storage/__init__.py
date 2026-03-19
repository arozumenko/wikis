"""Pluggable artifact storage."""

from __future__ import annotations

from app.config import Settings

from .base import ArtifactStorage
from .local import LocalArtifactStorage


def get_storage(settings: Settings) -> ArtifactStorage:
    """Create a storage backend from settings."""
    if settings.storage_backend == "local":
        return LocalArtifactStorage(settings.storage_path)
    raise ValueError(f"Unknown storage backend: {settings.storage_backend}")


__all__ = ["ArtifactStorage", "LocalArtifactStorage", "get_storage"]
