"""Repository identifier resolution.

This plugin uses filesystem indexing which creates commit-isolated artifacts.

Problem:
- Runtime callers often refer to a repository as "owner/repo:branch".
- Indexing stores artifacts as "owner/repo:branch:commit8".
- If multiple commits exist, prefix-based selection can pick the wrong one.

Solution:
- Use an explicit pointer in cache_index.json: refs["owner/repo:branch"] = "owner/repo:branch:commit8".
- When refs is missing (legacy), resolve deterministically using either a single cache entry
  or an unambiguous filesystem clone that matches a commit with both FAISS + combined graph.
"""

from __future__ import annotations

import json
import logging
import os
import re
import uuid
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

_RESERVED_TOP_LEVEL_KEYS = {"graphs", "refs", "docs", "bm25"}
_COMMIT_RE = re.compile(r"^[0-9a-f]{8}$")


def split_repo_identifier(repo_identifier: str) -> tuple[str, str | None, str | None]:
    """Split repo identifier into (repo, branch, commit8).

    Accepts:
    - owner/repo
    - owner/repo:branch
    - owner/repo:branch:commit8

    Parsing is from the right to avoid accidental ':' in future formats.
    """
    if not repo_identifier:
        return "", None, None

    if repo_identifier.count(":") >= 2:
        repo, branch, commit = repo_identifier.rsplit(":", 2)
        return repo, branch or None, commit or None

    if repo_identifier.count(":") == 1:
        repo, branch = repo_identifier.rsplit(":", 1)
        return repo, branch or None, None

    return repo_identifier, None, None


def repo_branch_key(repo: str, branch: str) -> str:
    return f"{repo}:{branch}" if branch else repo


def load_cache_index(cache_dir: str | Path) -> dict[str, Any]:
    index_path = Path(cache_dir).expanduser() / "cache_index.json"
    if not index_path.exists():
        return {}

    try:
        with open(index_path, encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, dict):
            return data
    except Exception as e:
        logger.warning(f"Failed to load cache index at {index_path}: {e}")

    return {}


def save_cache_index_atomic(cache_dir: str | Path, index: dict[str, Any]) -> None:
    cache_dir_path = Path(cache_dir).expanduser()
    cache_dir_path.mkdir(parents=True, exist_ok=True)

    index_path = cache_dir_path / "cache_index.json"
    tmp_path = cache_dir_path / f"cache_index.{uuid.uuid4().hex}.tmp"

    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(index, f, indent=2)

    os.replace(tmp_path, index_path)


def ensure_refs(index: dict[str, Any]) -> dict[str, str]:
    refs = index.get("refs")
    if not isinstance(refs, dict):
        refs = {}
        index["refs"] = refs
    return refs  # type: ignore[return-value]


def list_vectorstore_commit_keys(index: dict[str, Any], repo_branch: str) -> list[str]:
    keys: list[str] = []
    for k in index.keys():
        if not isinstance(k, str):
            continue
        if k in _RESERVED_TOP_LEVEL_KEYS:
            continue
        if k.startswith(repo_branch + ":"):
            # expects repo:branch:commit8
            _, _, commit = split_repo_identifier(k)
            if commit and _COMMIT_RE.match(commit):
                keys.append(k)
    return keys


def has_combined_graph(index: dict[str, Any], canonical_repo_id: str) -> bool:
    graphs = index.get("graphs")
    if not isinstance(graphs, dict):
        return False
    return bool(graphs.get(f"{canonical_repo_id}:combined"))


def repository_clone_candidates(repositories_dir: str | Path, repo: str, branch: str) -> list[str]:
    repositories_dir_path = Path(repositories_dir)
    if not repositories_dir_path.exists():
        return []

    safe_repo = repo.replace("/", "_")
    pattern = str(repositories_dir_path / f"{safe_repo}_{branch}_*")
    return [p for p in sorted(glob_glob(pattern)) if os.path.isdir(p)]


def glob_glob(pattern: str) -> list[str]:
    # Local helper to avoid importing glob in every caller.
    import glob

    return glob.glob(pattern)


def commit_from_clone_dir(path: str) -> str | None:
    base = os.path.basename(path.rstrip(os.sep))
    # Commit is always the last '_' segment per LocalRepositoryManager.
    commit = base.split("_")[-1]
    if _COMMIT_RE.match(commit):
        return commit
    return None


def resolve_canonical_repo_identifier(
    *,
    repo_identifier: str,
    cache_dir: str | Path,
    repositories_dir: str | Path | None = None,
) -> str:
    """Resolve repo_identifier to canonical commit-scoped id.

    Raises ValueError when the repo identifier is ambiguous and cannot be
    resolved deterministically.
    """
    repo, branch, commit = split_repo_identifier(repo_identifier)

    # If already commit-scoped, keep as-is.
    if repo and branch and commit:
        return repo_identifier

    if not repo or not branch:
        # We do not support resolving repo-only identifiers here.
        raise ValueError(f"Repository identifier must include branch: '{repo_identifier}'")

    rb = repo_branch_key(repo, branch)
    index = load_cache_index(cache_dir)

    refs = index.get("refs")
    if isinstance(refs, dict) and rb in refs and isinstance(refs.get(rb), str):
        return str(refs[rb])

    # Legacy support: some indexes were stored as repo:branch without commit isolation.
    # If such a key exists, treat it as canonical.
    if isinstance(index.get(rb), str):
        return rb

    # Legacy: use single commit if only one exists.
    commit_keys = list_vectorstore_commit_keys(index, rb)
    if not commit_keys:
        raise ValueError(
            f"No commit-scoped wiki index found for '{rb}'. Reindex (Generate Wiki) to create commit-isolated caches."
        )
    if len(commit_keys) == 1:
        return commit_keys[0]

    # Legacy ambiguity: attempt filesystem-truth.
    if repositories_dir is None:
        repositories_dir = Path(cache_dir) / "repositories"

    candidates = repository_clone_candidates(repositories_dir, repo, branch)
    viable: list[str] = []

    for clone_dir in candidates:
        c = commit_from_clone_dir(clone_dir)
        if not c:
            continue
        canonical = f"{rb}:{c}"
        if canonical in index and has_combined_graph(index, canonical):
            viable.append(canonical)

    if len(viable) == 1:
        return viable[0]

    if commit_keys:
        commits = [split_repo_identifier(k)[2] for k in commit_keys]
    else:
        commits = []

    raise ValueError(
        "Ambiguous wiki indexes for "
        f"'{rb}'. Found commits={sorted([c for c in commits if c])}. "
        "Reindex to create refs mapping or specify commit explicitly."
    )


def resolve_clone_dir_for_canonical_id(
    *,
    canonical_repo_id: str,
    cache_dir: str | Path,
) -> str | None:
    repo, branch, commit = split_repo_identifier(canonical_repo_id)
    if not (repo and branch and commit):
        return None

    repositories_dir = Path(cache_dir) / "repositories"
    safe_repo = repo.replace("/", "_")
    expected = repositories_dir / f"{safe_repo}_{branch}_{commit}"
    if expected.exists() and expected.is_dir():
        return str(expected)

    # Fallback: search for matching commit
    pattern = str(repositories_dir / f"{safe_repo}_{branch}_*")
    for path in sorted(glob_glob(pattern)):
        if os.path.isdir(path) and commit_from_clone_dir(path) == commit:
            return path

    return None
