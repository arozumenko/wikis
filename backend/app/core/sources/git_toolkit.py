"""SourceToolkit implementation for git repositories.

Wraps LocalRepositoryManager + repo_providers/ behind the SourceToolkit interface.
Supports remote URLs (GitHub, GitLab, Bitbucket, Azure DevOps) and local paths.
"""

from __future__ import annotations

import asyncio
import fnmatch
import logging
import shutil
import subprocess
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import AsyncIterator

from app.core.sources.base import FileContent, FileInfo, OriginPointer, SourceToolkit
from app.core.sources.exceptions import (
    SourceAuthError,
    SourceNotFoundError,
    SourceUnavailableError,
)

logger = logging.getLogger(__name__)

# Extension → language identifier (mirrors FilterManager.get_file_language).
# TODO: dedupe with filter_manager.py in #189
_LANG_MAP: dict[str, str] = {
    ".py": "python",
    ".js": "javascript",
    ".ts": "typescript",
    ".jsx": "javascript",
    ".tsx": "typescript",
    ".java": "java",
    ".go": "go",
    ".cs": "c_sharp",
    ".cpp": "cpp",
    ".cxx": "cpp",
    ".cc": "cpp",
    ".c": "c",
    ".h": "c",
    ".hpp": "cpp",
    ".rb": "ruby",
    ".php": "php",
    ".rs": "rust",
    ".kt": "kotlin",
    ".scala": "scala",
    ".swift": "swift",
    ".dart": "dart",
    ".r": "r",
    ".m": "objective_c",
    ".sh": "bash",
    ".bash": "bash",
    ".zsh": "bash",
    ".ps1": "powershell",
    ".html": "html",
    ".htm": "html",
    ".css": "css",
    ".scss": "scss",
    ".sass": "sass",
    ".less": "less",
    ".sql": "sql",
    ".yaml": "yaml",
    ".yml": "yaml",
    ".json": "json",
    ".xml": "xml",
    ".toml": "toml",
    ".dockerfile": "dockerfile",
    ".gradle": "groovy",
    ".kts": "kotlin_script",
    ".wsdl": "xml",
    ".xsd": "xml",
    ".proto": "protobuf",
    ".tf": "terraform",
    ".tfvars": "terraform",
    ".hcl": "hcl",
    ".mod": "gomod",
    ".bat": "batch",
    ".cmd": "batch",
    ".cfg": "config",
    ".conf": "config",
    ".psm1": "powershell",
    ".md": "markdown",
    ".rst": "restructuredtext",
    ".txt": "text",
}


def _detect_language(path: str) -> str | None:
    return _LANG_MAP.get(Path(path).suffix.lower())


def _match_patterns(rel_path: str, patterns: list[str]) -> bool:
    """Return True if *rel_path* matches at least one gitignore-style pattern."""
    for pat in patterns:
        if fnmatch.fnmatch(rel_path, pat):
            return True
        # Match against just the filename component as well (glob convention)
        if fnmatch.fnmatch(Path(rel_path).name, pat):
            return True
        # Allow directory prefix matching: "vendor/**" → "vendor/a/b"
        if pat.endswith("/**") and rel_path.startswith(pat[:-3]):
            return True
    return False


def _strip_url_credentials(url: str) -> str:
    """Return *url* with any embedded userinfo (user:pass) removed."""
    from urllib.parse import urlparse, urlunparse

    parsed = urlparse(url)
    if parsed.username or parsed.password:
        return urlunparse(parsed._replace(netloc=parsed.hostname or ""))
    return url


def _build_blob_url(
    repo_url: str,
    path: str,
    sha: str,
    line_start: int | None,
    line_end: int | None,
) -> str:
    """Produce a web blob URL for the given provider.

    TODO: dedupe with repo_providers/providers.py in #189
    """
    url = repo_url.rstrip("/")
    # Strip .git suffix and embedded credentials for a clean base URL.
    if url.endswith(".git"):
        url = url[:-4]

    # Remove embedded credentials (https://<token>@host/...)
    url = _strip_url_credentials(url)

    from urllib.parse import urlparse

    parsed = urlparse(url)
    host = parsed.hostname or ""

    if "github" in host:
        fragment = f"#L{line_start}-L{line_end}" if line_start and line_end else (f"#L{line_start}" if line_start else "")
        return f"{url}/blob/{sha}/{path}{fragment}"

    if "gitlab" in host:
        if line_start and line_end:
            fragment = f"#L{line_start}-{line_end}"
        elif line_start:
            fragment = f"#L{line_start}"
        else:
            fragment = ""
        return f"{url}/-/blob/{sha}/{path}{fragment}"

    if "bitbucket" in host:
        fragment = f"#lines-{line_start}:{line_end}" if line_start and line_end else (f"#lines-{line_start}" if line_start else "")
        return f"{url}/src/{sha}/{path}{fragment}"

    if "azure" in host or "visualstudio" in host:
        base = f"{url}?path=/{path}&version=GC{sha}"
        if line_start:
            base += f"&line={line_start}"
            if line_end:
                base += f"&lineEnd={line_end}"
        return base

    # Unknown remote — best-effort
    return f"{url}/blob/{sha}/{path}"


class GitToolkit(SourceToolkit):
    """SourceToolkit adapter for git repositories.

    Args:
        repo_url: Remote HTTPS URL or local absolute path / file:// URI.
        branch: Branch to clone (default ``"main"``).
        token: Personal access token for private repositories.
        whitelist: Include-only glob patterns (applied by list_files).
        blacklist: Exclude glob patterns (applied by list_files).
    """

    source_type = "git"

    def __init__(
        self,
        repo_url: str,
        branch: str = "main",
        token: str | None = None,
        whitelist: list[str] | None = None,
        blacklist: list[str] | None = None,
    ) -> None:
        self._repo_url = repo_url
        self._branch = branch
        self._token = token
        self._whitelist = whitelist or []
        self._blacklist = blacklist or []

        # Resolved after first clone/resolve call.
        self._workdir: Path | None = None
        self._commit_sha: str | None = None
        self._is_local: bool = False
        self._tmpdir_owned: str | None = None  # path we must clean up on close

    # ------------------------------------------------------------------
    # Context manager
    # ------------------------------------------------------------------

    async def __aenter__(self) -> "GitToolkit":
        return self

    async def __aexit__(self, *_: object) -> None:
        await self.close()

    async def close(self) -> None:
        """Remove the temporary working tree (if we created one)."""
        if self._tmpdir_owned:
            try:
                shutil.rmtree(self._tmpdir_owned, ignore_errors=True)
            except Exception as exc:  # noqa: BLE001
                logger.warning("Failed to clean up workdir %s: %s", self._tmpdir_owned, exc)
            finally:
                self._tmpdir_owned = None
                self._workdir = None

    # ------------------------------------------------------------------
    # SourceToolkit.from_config
    # ------------------------------------------------------------------

    @classmethod
    def from_config(cls, config: dict) -> "GitToolkit":
        """Instantiate from a plain dict.

        Args:
            config: Keys: ``repo_url`` (required), ``branch``, ``token``,
                ``whitelist`` (str or list), ``blacklist`` (str or list).

        Security precondition:
            The caller is responsible for validating ``repo_url`` against any
            local-path allowlist (``ALLOWED_LOCAL_PATHS``) before constructing
            the toolkit.  ``from_config`` does NOT call ``validate_local_path``.
            When wired into ``WikiService`` in #189, validation must happen at
            the API boundary, not here.
        """
        repo_url: str = config["repo_url"]
        branch: str = config.get("branch", "main")
        token: str | None = config.get("token")

        def _coerce_list(val: object) -> list[str]:
            if val is None:
                return []
            if isinstance(val, str):
                return [val]
            return list(val)

        whitelist = _coerce_list(config.get("whitelist"))
        blacklist = _coerce_list(config.get("blacklist"))

        return cls(
            repo_url=repo_url,
            branch=branch,
            token=token,
            whitelist=whitelist,
            blacklist=blacklist,
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _sanitise(self, text: str) -> str:
        """Replace any occurrence of the raw token with ``***``."""
        if self._token and self._token in text:
            return text.replace(self._token, "***")
        return text

    def _auth_url(self) -> str:
        """Return the clone URL with credentials embedded if a token is set.

        The credential prefix varies by provider:
        - GitHub / Azure DevOps: ``{token}@host``
        - GitLab: ``oauth2:{token}@host``
        - Bitbucket: ``x-token-auth:{token}@host``
        - Default (unknown self-hosted): ``{token}@host``
        """
        if not self._token:
            return self._repo_url

        from urllib.parse import urlparse, urlunparse

        parsed = urlparse(self._repo_url)
        host = parsed.hostname or ""
        port_suffix = f":{parsed.port}" if parsed.port else ""

        if host == "gitlab.com" or host.startswith("gitlab."):
            userinfo = f"oauth2:{self._token}"
        elif host == "bitbucket.org" or host.startswith("bitbucket."):
            userinfo = f"x-token-auth:{self._token}"
        else:
            # GitHub, Azure DevOps (dev.azure.com), and unknown self-hosted
            userinfo = self._token

        authed = parsed._replace(netloc=f"{userinfo}@{host}{port_suffix}")
        return urlunparse(authed)

    def _safe_url(self) -> str:
        """Clone URL without credentials — safe for logging.

        Strips both the instance token (if set) and any credentials that may
        have been embedded directly in ``repo_url`` by the caller.
        """
        return _strip_url_credentials(self._repo_url)

    async def _ensure_workdir(self) -> None:
        """Resolve or clone the repository exactly once.

        For local paths this is a no-op (the path IS the working tree).
        For remote URLs, shallow-clone into a fresh temp directory.

        All ``subprocess.run`` calls are offloaded to a worker thread via
        ``asyncio.to_thread`` (#214) so the FastAPI event loop is never
        blocked by blocking I/O. Combined with the per-call ``timeout=``
        on each subprocess, the worker thread is guaranteed to release.

        Warning:
            This method is NOT safe for concurrent calls on the same instance.
            Two coroutines calling ``_ensure_workdir`` concurrently can both
            pass the ``self._workdir is not None`` guard before either sets
            the attribute, resulting in a double-clone race. Current callers
            (``source_materializer.py``, scan service) are all serial, so
            this is latent rather than active. A proper fix
            (``asyncio.Lock``) is tracked in a follow-up issue.
        """
        if self._workdir is not None:
            return

        from app.core.local_repo_provider import extract_git_metadata, is_local_path

        # TODO(#189): caller (WikiService) must call validate_local_path before reaching this branch
        if is_local_path(self._repo_url):
            self._is_local = True
            raw = self._repo_url.removeprefix("file://")
            resolved = Path(raw).resolve()
            if not resolved.exists():
                raise SourceNotFoundError(f"Local path does not exist: {resolved}")
            self._workdir = resolved

            info = await asyncio.to_thread(extract_git_metadata, resolved)
            if info.is_git:
                self._commit_sha = info.commit_hash
        else:
            self._is_local = False
            tmpdir = tempfile.mkdtemp(prefix="git_toolkit_")
            self._tmpdir_owned = tmpdir
            clone_url = self._auth_url()
            cmd = [
                "git",
                "clone",
                "--branch",
                self._branch,
                "--depth",
                "1",
                "--single-branch",
                clone_url,
                tmpdir,
            ]
            logger.debug("Cloning %s (branch: %s)", self._safe_url(), self._branch)
            try:
                result = await asyncio.to_thread(
                    subprocess.run,  # noqa: S603
                    cmd,
                    capture_output=True,
                    text=True,
                    timeout=75,
                    check=False,
                )
            except subprocess.TimeoutExpired as exc:
                raise SourceUnavailableError(
                    f"git clone timed out for {self._safe_url()}"
                ) from exc
            except Exception as exc:
                raise SourceUnavailableError(self._sanitise(f"git clone failed: {exc}")) from exc

            if result.returncode != 0:
                stderr = self._sanitise(result.stderr)
                if "Authentication failed" in stderr or "Permission denied" in stderr:
                    raise SourceAuthError(f"Authentication failed cloning {self._safe_url()}: {stderr}")
                if "Repository not found" in stderr or "not found" in stderr.lower():
                    raise SourceNotFoundError(f"Repository not found: {self._safe_url()}")
                raise SourceUnavailableError(f"git clone failed ({self._safe_url()}): {stderr}")

            self._workdir = Path(tmpdir)

            # Read the commit SHA from the local clone.
            try:
                rev_result = await asyncio.to_thread(
                    subprocess.run,  # noqa: S603
                    ["git", "rev-parse", "HEAD"],
                    cwd=self._workdir,
                    capture_output=True,
                    text=True,
                    timeout=10,
                    check=False,
                )
            except subprocess.TimeoutExpired:
                logger.warning("git rev-parse HEAD timed out for %s; SHA will be unknown", self._safe_url())
                rev_result = None
            if rev_result is not None and rev_result.returncode == 0:
                self._commit_sha = rev_result.stdout.strip()

    # ------------------------------------------------------------------
    # SourceToolkit abstract method implementations
    # ------------------------------------------------------------------

    async def list_files(
        self,
        include: list[str] | None = None,
        exclude: list[str] | None = None,
    ) -> AsyncIterator[FileInfo]:  # type: ignore[override]
        """Yield FileInfo for every file in the working tree.

        Combines instance-level whitelist/blacklist with the call-level
        include/exclude parameters.

        Args:
            include: Additional include patterns (merged with instance whitelist).
            exclude: Additional exclude patterns (merged with instance blacklist).

        Note:
            ``list_files`` does NOT apply the project-wide
            ``FilesystemRepositoryIndexer`` exclusion defaults (heavy dirs like
            ``node_modules``, extension allowlists, file-size caps).  Callers
            that want indexer-equivalent behavior must pass the same
            ``include``/``exclude`` patterns explicitly.  This keeps the toolkit
            composable — the filter policy lives with the caller.
        """
        await self._ensure_workdir()
        workdir = self._workdir
        assert workdir is not None  # noqa: S101 — guaranteed by _ensure_workdir

        effective_include = self._whitelist + (include or [])
        effective_exclude = self._blacklist + (exclude or [])

        now = datetime.now(tz=timezone.utc)

        for abs_path in workdir.rglob("*"):
            # Skip symlinks before is_file() which follows them.
            if abs_path.is_symlink():
                rel_sym = abs_path.relative_to(workdir)
                logger.debug("Skipping symlink: %s", rel_sym)
                continue
            if not abs_path.is_file():
                continue

            rel = abs_path.relative_to(workdir)
            rel_str = str(rel)

            # Skip .git internals regardless of patterns.
            parts = rel.parts
            if ".git" in parts:
                continue

            # Apply exclusions first.
            if effective_exclude and _match_patterns(rel_str, effective_exclude):
                continue

            # Apply inclusions (if any are specified).
            if effective_include and not _match_patterns(rel_str, effective_include):
                continue

            try:
                size = abs_path.stat().st_size
            except OSError:
                size = 0

            pointer = self.build_origin_pointer(
                path=rel_str,
                revision=self._commit_sha,
            )
            pointer = OriginPointer(
                source_type=pointer.source_type,
                ref=pointer.ref,
                url=pointer.url,
                ingested_at=now,
                revision=pointer.revision,
                line_start=pointer.line_start,
                line_end=pointer.line_end,
            )

            yield FileInfo(
                path=rel_str,
                language=_detect_language(rel_str),
                size_bytes=size,
                origin=pointer,
            )

    async def fetch_content(self, pointer: OriginPointer) -> FileContent:
        """Read the file identified by *pointer*.

        Args:
            pointer: An OriginPointer whose ``ref`` is the repo-relative path.
        """
        await self._ensure_workdir()
        workdir = self._workdir
        assert workdir is not None  # noqa: S101

        resolved = (workdir / pointer.ref).resolve()
        if not resolved.is_relative_to(workdir.resolve()):
            raise SourceNotFoundError(f"Path escapes repository: {pointer.ref!r}")
        if resolved.is_symlink():
            raise SourceNotFoundError(f"Refusing to read symlink: {pointer.ref!r}")

        target = resolved
        if not target.exists():
            raise SourceNotFoundError(f"File not found in working tree: {pointer.ref}")
        try:
            content = target.read_text(encoding="utf-8", errors="replace")
        except OSError as exc:
            raise SourceNotFoundError(f"Cannot read {pointer.ref}: {exc}") from exc

        info = FileInfo(
            path=pointer.ref,
            language=_detect_language(pointer.ref),
            size_bytes=target.stat().st_size,
            origin=pointer,
        )
        return FileContent(info=info, content=content)

    async def test_connection(self) -> str:
        """Verify reachability.

        For remote URLs, runs ``git ls-remote --heads``.
        For local paths, confirms existence.
        """
        from app.core.local_repo_provider import is_local_path

        if is_local_path(self._repo_url):
            from pathlib import Path as _Path

            raw = self._repo_url.removeprefix("file://")
            p = _Path(raw).resolve()
            if not p.exists():
                raise SourceNotFoundError(f"Local path does not exist: {p}")
            git_dir = p / ".git"
            if git_dir.exists():
                return f"Connected to local git repository: {p}"
            return f"Connected to local directory (non-git): {p}"

        auth_url = self._auth_url()
        cmd = ["git", "ls-remote", "--heads", auth_url]
        try:
            result = await asyncio.to_thread(
                subprocess.run, cmd, capture_output=True, text=True, timeout=30, check=False  # noqa: S603
            )
        except subprocess.TimeoutExpired as exc:
            raise SourceUnavailableError(f"Timed out connecting to {self._safe_url()}") from exc
        except Exception as exc:
            raise SourceUnavailableError(self._sanitise(str(exc))) from exc

        if result.returncode != 0:
            stderr = self._sanitise(result.stderr)
            if "Authentication failed" in stderr or "Permission denied" in stderr:
                raise SourceAuthError(f"Auth failure for {self._safe_url()}: {stderr}")
            if "not found" in stderr.lower() or "does not exist" in stderr.lower():
                raise SourceNotFoundError(f"Repository not found: {self._safe_url()}: {stderr}")
            raise SourceUnavailableError(f"ls-remote failed for {self._safe_url()}: {stderr}")

        return f"Connected to {self._safe_url()}"

    def build_origin_pointer(
        self,
        path: str,
        revision: str | None = None,
        line_start: int | None = None,
        line_end: int | None = None,
    ) -> OriginPointer:
        """Construct an OriginPointer for *path*.

        Args:
            path: Repo-relative file path.
            revision: Commit SHA or tag.
            line_start: First referenced line (1-based).
            line_end: Last referenced line (1-based).
        """
        from app.core.local_repo_provider import is_local_path

        sha = revision or self._commit_sha or "HEAD"

        if is_local_path(self._repo_url):
            raw = self._repo_url.removeprefix("file://")
            abs_path = Path(raw).resolve() / path
            url = f"file://{abs_path}"
        else:
            url = _build_blob_url(self._repo_url, path, sha, line_start, line_end)

        return OriginPointer(
            source_type=self.source_type,
            ref=path,
            url=url,
            ingested_at=datetime.now(tz=timezone.utc),
            revision=sha,
            line_start=line_start,
            line_end=line_end,
        )
