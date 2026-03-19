"""Local filesystem artifact storage."""

from __future__ import annotations

from pathlib import Path

from .base import ArtifactStorage


def _validate_path_component(value: str, label: str) -> None:
    """Reject path traversal attempts."""
    if ".." in value or value.startswith("/") or value.startswith("\\"):
        raise ValueError(f"Invalid {label}: path traversal not allowed: {value!r}")


class LocalArtifactStorage(ArtifactStorage):
    """Store artifacts as files under base_path/<bucket>/<name>."""

    def __init__(self, base_path: str = "./data/artifacts") -> None:
        self.base_path = Path(base_path)

    def _resolve(self, bucket: str, name: str) -> Path:
        _validate_path_component(bucket, "bucket")
        _validate_path_component(name, "name")
        resolved = (self.base_path / bucket / name).resolve()
        # Extra safety: ensure resolved path is under base_path
        if not str(resolved).startswith(str(self.base_path.resolve())):
            raise ValueError("Path traversal detected")
        return resolved

    async def upload(self, bucket: str, name: str, data: bytes) -> str:
        path = self._resolve(bucket, name)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
        return f"{bucket}/{name}"

    async def download(self, bucket: str, name: str) -> bytes:
        path = self._resolve(bucket, name)
        if not path.exists():
            raise FileNotFoundError(f"Artifact not found: {bucket}/{name}")
        return path.read_bytes()

    async def delete(self, bucket: str, name: str) -> None:
        path = self._resolve(bucket, name)
        if path.exists():
            path.unlink()

    async def list_artifacts(self, bucket: str, prefix: str = "") -> list[str]:
        _validate_path_component(bucket, "bucket")
        if prefix:
            _validate_path_component(prefix, "prefix")
        bucket_dir = self.base_path / bucket
        if not bucket_dir.exists():
            return []
        if prefix:
            search_dir = bucket_dir / prefix
            if search_dir.is_dir():
                return sorted(str(p.relative_to(bucket_dir)) for p in search_dir.rglob("*") if p.is_file())
            # Fallback: prefix as glob pattern
            pattern = f"{prefix}*"
            return sorted(p.name for p in bucket_dir.glob(pattern) if p.is_file())
        return sorted(p.name for p in bucket_dir.glob("*") if p.is_file())
