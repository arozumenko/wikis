"""Local filesystem repo support — generate wikis from local directories."""

from __future__ import annotations

import logging
import subprocess
from pathlib import Path

from pydantic import BaseModel

logger = logging.getLogger(__name__)


class LocalRepoInfo(BaseModel):
    """Metadata extracted from a local repository."""

    path: str
    branch: str | None = None
    commit_hash: str | None = None
    remote_url: str | None = None
    is_git: bool = False


def validate_local_path(path: str, allowed_paths: list[str] | None = None) -> Path:
    """Validate a local path for wiki generation.

    Rejects all local paths when ALLOWED_LOCAL_PATHS is unset (opt-in security).
    Rejects paths outside the allowlist and symlinks that resolve outside it.
    """
    # Opt-in: no allowlist configured → reject everything
    if not allowed_paths:
        raise ValueError("Local path access is disabled. Set ALLOWED_LOCAL_PATHS to enable.")

    original = Path(path)
    resolved = original.resolve()

    if not resolved.exists():
        raise ValueError(f"Path does not exist: {path}")
    if not resolved.is_dir():
        raise ValueError(f"Path is not a directory: {path}")

    allowed = [Path(p).resolve() for p in allowed_paths]

    # Check resolved path is within an allowed base
    if not any(resolved == a or str(resolved).startswith(str(a) + "/") for a in allowed):
        raise ValueError(f"Path {path} is outside allowed paths: {', '.join(str(a) for a in allowed)}")

    # Check symlink: if original path (before full resolve) is a symlink, verify its
    # real target is also within an allowed base
    if original.is_symlink():
        real = original.resolve()
        if not any(real == a or str(real).startswith(str(a) + "/") for a in allowed):
            raise ValueError(f"Symlink target {real} is outside allowed paths")

    logger.info("Local path access validated: %s", resolved)
    return resolved


def extract_git_metadata(repo_path: Path) -> LocalRepoInfo:
    """Extract git metadata from a local directory."""
    git_dir = repo_path / ".git"
    if not git_dir.exists():
        return LocalRepoInfo(path=str(repo_path), is_git=False)
    info = LocalRepoInfo(path=str(repo_path), is_git=True)
    for cmd, attr in [
        (["git", "rev-parse", "--abbrev-ref", "HEAD"], "branch"),
        (["git", "rev-parse", "HEAD"], "commit_hash"),
        (["git", "remote", "get-url", "origin"], "remote_url"),
    ]:
        try:
            result = subprocess.run(cmd, cwd=repo_path, capture_output=True, text=True, timeout=5)  # noqa: S603 — git introspection with fixed commands, no user input in cmd
            if result.returncode == 0:
                setattr(info, attr, result.stdout.strip())
        except Exception:  # noqa: S110
            pass
    return info


def is_local_path(repo_url: str) -> bool:
    """Check if a repo_url refers to a local path."""
    return repo_url.startswith("/") or repo_url.startswith("file:///")


def make_local_wiki_id(path: str, branch: str | None = None) -> str:
    """Human-readable wiki ID: local--{dir_name}--{branch_or_none}."""
    dir_name = Path(path).resolve().name
    branch_part = branch or "none"
    return f"local--{dir_name}--{branch_part}"
