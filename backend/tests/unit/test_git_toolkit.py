"""Unit tests for GitToolkit (issue #186)."""

from __future__ import annotations

import asyncio
import socket
import subprocess
import tempfile
import time
from pathlib import Path
from unittest.mock import patch

import pytest

from app.core.sources import GitToolkit, OriginPointer, registry
from app.core.sources.exceptions import SourceAuthError, SourceNotFoundError, SourceUnavailableError


# ---------------------------------------------------------------------------
# Network detection helper
# ---------------------------------------------------------------------------


def _has_network() -> bool:
    try:
        with socket.create_connection(("github.com", 443), timeout=2):
            return True
    except OSError:
        return False


_NETWORK = _has_network()


# ---------------------------------------------------------------------------
# 1. URL builder — parametrized across all 4 providers
# ---------------------------------------------------------------------------

_BLOB_URL_CASES = [
    (
        "github",
        "https://github.com/octocat/Hello-World.git",
        "src/foo.py",
        "abc123",
        10,
        20,
        "https://github.com/octocat/Hello-World/blob/abc123/src/foo.py#L10-L20",
    ),
    (
        "gitlab",
        "https://gitlab.com/group/project.git",
        "src/foo.py",
        "abc123",
        10,
        20,
        "https://gitlab.com/group/project/-/blob/abc123/src/foo.py#L10-20",
    ),
    (
        "bitbucket",
        "https://bitbucket.org/workspace/repo.git",
        "src/foo.py",
        "abc123",
        10,
        20,
        "https://bitbucket.org/workspace/repo/src/abc123/src/foo.py#lines-10:20",
    ),
    (
        "azure_devops",
        "https://dev.azure.com/myorg/myproject/_git/myrepo",
        "src/foo.py",
        "abc123",
        10,
        20,
        "https://dev.azure.com/myorg/myproject/_git/myrepo?path=/src/foo.py&version=GCabc123&line=10&lineEnd=20",
    ),
]


@pytest.mark.parametrize("name,repo_url,path,sha,line_start,line_end,expected", _BLOB_URL_CASES)
def test_build_origin_pointer_blob_url(name, repo_url, path, sha, line_start, line_end, expected):
    toolkit = GitToolkit(repo_url=repo_url)
    ptr = toolkit.build_origin_pointer(path=path, revision=sha, line_start=line_start, line_end=line_end)
    assert ptr.url == expected, f"[{name}] expected {expected!r}, got {ptr.url!r}"


# ---------------------------------------------------------------------------
# 2. URL builder without line anchors
# ---------------------------------------------------------------------------


def test_build_origin_pointer_no_line_anchors():
    toolkit = GitToolkit(repo_url="https://github.com/octocat/Hello-World.git")
    ptr = toolkit.build_origin_pointer("README.md", revision="abc123")
    assert ptr.url == "https://github.com/octocat/Hello-World/blob/abc123/README.md"
    assert "#" not in ptr.url


# ---------------------------------------------------------------------------
# 3. URL builder for local path — produces file:// URL
# ---------------------------------------------------------------------------


def test_build_origin_pointer_local_path(tmp_path):
    toolkit = GitToolkit(repo_url=str(tmp_path))
    ptr = toolkit.build_origin_pointer("src/main.py", revision="deadbeef")
    assert ptr.url.startswith("file://")
    assert "src/main.py" in ptr.url


# ---------------------------------------------------------------------------
# 4. PAT handling — _auth_url embeds token, _sanitise redacts it
# ---------------------------------------------------------------------------


def test_auth_url_embeds_token():
    toolkit = GitToolkit(repo_url="https://github.com/foo/bar.git", token="ghp_secret")
    auth = toolkit._auth_url()
    assert "ghp_secret" in auth


def test_sanitise_replaces_token():
    toolkit = GitToolkit(repo_url="https://github.com/foo/bar.git", token="ghp_secret")
    result = toolkit._sanitise("error: ghp_secret leaked")
    assert "ghp_secret" not in result
    assert "***" in result


def test_sanitise_no_token_passthrough():
    toolkit = GitToolkit(repo_url="https://github.com/foo/bar.git", token=None)
    text = "no token here"
    assert toolkit._sanitise(text) == text


def test_safe_url_removes_token():
    toolkit = GitToolkit(repo_url="https://github.com/foo/bar.git", token="ghp_secret")
    safe = toolkit._safe_url()
    assert "ghp_secret" not in safe


# ---------------------------------------------------------------------------
# 5. from_config round-trip
# ---------------------------------------------------------------------------


def test_from_config_roundtrip():
    cfg = {
        "repo_url": "https://github.com/foo/bar.git",
        "branch": "develop",
        "token": "tok123",
        "whitelist": "*.py",
        "blacklist": ["*.pyc", "node_modules/**"],
    }
    tk = GitToolkit.from_config(cfg)
    assert tk._repo_url == "https://github.com/foo/bar.git"
    assert tk._branch == "develop"
    assert tk._token == "tok123"
    assert tk._whitelist == ["*.py"]
    assert tk._blacklist == ["*.pyc", "node_modules/**"]


def test_from_config_minimal():
    tk = GitToolkit.from_config({"repo_url": "https://github.com/a/b.git"})
    assert tk._repo_url == "https://github.com/a/b.git"
    assert tk._branch == "main"
    assert tk._token is None
    assert tk._whitelist == []
    assert tk._blacklist == []


# ---------------------------------------------------------------------------
# 6. Registry integration
# ---------------------------------------------------------------------------


def test_registry_create_returns_git_toolkit():
    tk = registry.create("git", {"repo_url": "https://github.com/foo/bar.git", "branch": "main"})
    assert isinstance(tk, GitToolkit)


def test_registry_lists_git():
    assert "git" in registry.list_types()


# ---------------------------------------------------------------------------
# 7. OriginPointer fields
# ---------------------------------------------------------------------------


def test_build_origin_pointer_sets_all_fields():
    tk = GitToolkit(repo_url="https://github.com/octocat/Hello-World.git")
    ptr = tk.build_origin_pointer("a/b.py", revision="sha1", line_start=1, line_end=5)
    assert ptr.source_type == "git"
    assert ptr.ref == "a/b.py"
    assert ptr.revision == "sha1"
    assert ptr.line_start == 1
    assert ptr.line_end == 5


# ---------------------------------------------------------------------------
# 8. list_files on a local directory (no network)
# ---------------------------------------------------------------------------


@pytest.fixture()
def small_local_repo(tmp_path: Path) -> Path:
    """Create a tiny fake local repository with a few files."""
    (tmp_path / "README.md").write_text("# hello")
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "main.py").write_text("print('hi')")
    (tmp_path / "src" / "util.js").write_text("module.exports = {}")
    (tmp_path / ".git").mkdir()  # fake git dir so it is skipped
    (tmp_path / ".git" / "config").write_text("[core]")
    (tmp_path / "node_modules").mkdir()
    (tmp_path / "node_modules" / "lodash.js").write_text("// vendored")
    return tmp_path


@pytest.mark.asyncio
async def test_list_files_yields_files(small_local_repo):
    tk = GitToolkit(repo_url=str(small_local_repo))
    files = [fi async for fi in tk.list_files()]
    paths = {fi.path for fi in files}
    assert "README.md" in paths
    assert "src/main.py" in paths
    assert "src/util.js" in paths


@pytest.mark.asyncio
async def test_list_files_skips_git_dir(small_local_repo):
    tk = GitToolkit(repo_url=str(small_local_repo))
    files = [fi async for fi in tk.list_files()]
    paths = {fi.path for fi in files}
    assert not any(".git" in p for p in paths)


@pytest.mark.asyncio
async def test_list_files_exclude_pattern(small_local_repo):
    tk = GitToolkit(repo_url=str(small_local_repo), blacklist=["node_modules/**"])
    files = [fi async for fi in tk.list_files()]
    paths = {fi.path for fi in files}
    assert not any("node_modules" in p for p in paths)


@pytest.mark.asyncio
async def test_list_files_include_pattern(small_local_repo):
    tk = GitToolkit(repo_url=str(small_local_repo), whitelist=["*.py"])
    files = [fi async for fi in tk.list_files()]
    paths = {fi.path for fi in files}
    assert all(p.endswith(".py") for p in paths)
    assert "README.md" not in paths


# ---------------------------------------------------------------------------
# 9. fetch_content on a local file (no network)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_fetch_content_returns_file_contents(small_local_repo):
    tk = GitToolkit(repo_url=str(small_local_repo))
    files = [fi async for fi in tk.list_files()]
    readme = next(fi for fi in files if fi.path == "README.md")
    content = await tk.fetch_content(readme.origin)
    assert "hello" in content.content
    assert content.info.path == "README.md"


@pytest.mark.asyncio
async def test_fetch_content_missing_file_raises(small_local_repo):
    tk = GitToolkit(repo_url=str(small_local_repo))
    ptr = OriginPointer(
        source_type="git",
        ref="does/not/exist.py",
        url="file:///does/not/exist.py",
        ingested_at=__import__("datetime").datetime.now(tz=__import__("datetime").timezone.utc),
    )
    with pytest.raises(SourceNotFoundError):
        await tk.fetch_content(ptr)


# ---------------------------------------------------------------------------
# 10. test_connection on a non-existent local path
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_test_connection_missing_local_path_raises():
    tk = GitToolkit(repo_url="/tmp/__nonexistent_repo_xyz__")
    with pytest.raises(SourceNotFoundError):
        await tk.test_connection()


@pytest.mark.asyncio
async def test_test_connection_local_directory(small_local_repo):
    tk = GitToolkit(repo_url=str(small_local_repo))
    msg = await tk.test_connection()
    assert "Connected" in msg


# ---------------------------------------------------------------------------
# 11. language detection
# ---------------------------------------------------------------------------


def test_language_detection_known():
    from app.core.sources.git_toolkit import _detect_language

    assert _detect_language("app.py") == "python"
    assert _detect_language("index.ts") == "typescript"
    assert _detect_language("main.go") == "go"
    assert _detect_language("Makefile") is None  # no extension


def test_language_detection_unknown_extension():
    from app.core.sources.git_toolkit import _detect_language

    assert _detect_language("file.zzz") is None


# ---------------------------------------------------------------------------
# 12. async context manager cleans up tmpdir
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_context_manager_cleanup(small_local_repo):
    async with GitToolkit(repo_url=str(small_local_repo)) as tk:
        files = [fi async for fi in tk.list_files()]
        assert files  # populated inside context
    # local path — no tmpdir to clean up, but close() must not raise
    assert tk._tmpdir_owned is None


# ---------------------------------------------------------------------------
# 13. Integration test: real network clone
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
@pytest.mark.skipif(not _NETWORK, reason="no network available")
async def test_integration_clone_public_repo():
    """Clone a tiny public repo, list files, fetch content."""
    url = "https://github.com/octocat/Hello-World.git"
    async with GitToolkit(repo_url=url, branch="master") as tk:
        files = [fi async for fi in tk.list_files()]
        assert len(files) >= 1, "Expected at least one file in Hello-World"

        # fetch content for the first file found
        first = files[0]
        fc = await tk.fetch_content(first.origin)
        assert isinstance(fc.content, str)
        assert fc.info.path == first.path


# ---------------------------------------------------------------------------
# 14. Auth URL — provider-specific credential prefix
# ---------------------------------------------------------------------------

_AUTH_URL_CASES = [
    (
        "github",
        "https://github.com/foo/bar.git",
        "ghp_token",
        "https://ghp_token@github.com/foo/bar.git",
    ),
    (
        "gitlab",
        "https://gitlab.com/group/project.git",
        "glpat-token",
        "https://oauth2:glpat-token@gitlab.com/group/project.git",
    ),
    (
        "bitbucket",
        "https://bitbucket.org/workspace/repo.git",
        "bb_token",
        "https://x-token-auth:bb_token@bitbucket.org/workspace/repo.git",
    ),
    (
        "azure_devops",
        "https://dev.azure.com/myorg/myproject/_git/myrepo",
        "pat_token",
        "https://pat_token@dev.azure.com/myorg/myproject/_git/myrepo",
    ),
]


@pytest.mark.parametrize("name,repo_url,token,expected", _AUTH_URL_CASES)
def test_auth_url_provider_prefix(name, repo_url, token, expected):
    """_auth_url uses provider-correct credential prefix."""
    toolkit = GitToolkit(repo_url=repo_url, token=token)
    assert toolkit._auth_url() == expected, f"[{name}] wrong auth URL"


# ---------------------------------------------------------------------------
# 15. list_files — symlinks are skipped
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_list_files_skips_symlinks(tmp_path: Path):
    """Symlinks must not appear in list_files output."""
    real_file = tmp_path / "real.py"
    real_file.write_text("print('real')")
    link_file = tmp_path / "link.py"
    link_file.symlink_to(real_file)

    tk = GitToolkit(repo_url=str(tmp_path))
    files = [fi async for fi in tk.list_files()]
    paths = {fi.path for fi in files}
    assert "real.py" in paths
    assert "link.py" not in paths


# ---------------------------------------------------------------------------
# 16. fetch_content — path-traversal is rejected
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_fetch_content_path_traversal_raises(tmp_path: Path):
    """pointer.ref that escapes workdir must raise SourceNotFoundError."""
    # Create the toolkit pointed at a subdir so ../escape.txt exits it.
    subdir = tmp_path / "repo"
    subdir.mkdir()
    escape_target = tmp_path / "escape.txt"
    escape_target.write_text("secret")

    tk = GitToolkit(repo_url=str(subdir))
    ptr = OriginPointer(
        source_type="git",
        ref="../escape.txt",
        url="file:///escape.txt",
        ingested_at=__import__("datetime").datetime.now(tz=__import__("datetime").timezone.utc),
    )
    with pytest.raises(SourceNotFoundError, match="escapes repository"):
        await tk.fetch_content(ptr)


# ---------------------------------------------------------------------------
# 17. _safe_url — strips credentials even when no token is set on instance
# ---------------------------------------------------------------------------


def test_safe_url_strips_embedded_credentials():
    """_safe_url must remove credentials baked into repo_url."""
    toolkit = GitToolkit(repo_url="https://ghp_xxx@github.com/foo/bar.git", token=None)
    safe = toolkit._safe_url()
    assert "ghp_xxx" not in safe
    assert "github.com" in safe


# ---------------------------------------------------------------------------
# 18. subprocess timeout — git clone raises SourceUnavailableError
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_clone_timeout_raises_source_unavailable():
    """When subprocess.run raises TimeoutExpired, _ensure_workdir must raise
    SourceUnavailableError with the safe (credential-free) URL in the message.
    """
    toolkit = GitToolkit(
        repo_url="https://github.com/example/slow-repo.git",
        branch="main",
    )

    with patch("subprocess.run", side_effect=subprocess.TimeoutExpired(cmd=["git", "clone"], timeout=75)):
        with pytest.raises(SourceUnavailableError, match="git clone timed out for"):
            await toolkit._ensure_workdir()


# ---------------------------------------------------------------------------
# 19. Concurrency proof — _ensure_workdir does not block the event loop
# ---------------------------------------------------------------------------

_SIMULATED_DURATION = 0.15  # seconds each fake clone sleeps
_OVERLAP_THRESHOLD = 0.05   # ≥50 ms overlap proves threads ran concurrently


@pytest.mark.asyncio
async def test_connection_does_not_block_event_loop(tmp_path: Path):
    """Two concurrent asyncio.to_thread calls must overlap by ≥50 ms.

    Proof that _ensure_workdir (and by extension any blocking subprocess call
    wrapped in asyncio.to_thread) does not starve the event loop: both calls
    must run concurrently on the thread pool rather than serialising.
    """
    # Records (start, end) only for the slow git-clone calls.
    call_records: list[tuple[float, float]] = []

    def _fake_subprocess(cmd, *args, **kwargs):
        if "clone" in cmd:
            # Simulate a slow network clone so the two threads must overlap.
            t0 = time.monotonic()
            time.sleep(_SIMULATED_DURATION)
            call_records.append((t0, time.monotonic()))
            return subprocess.CompletedProcess(args=cmd, returncode=0, stdout="", stderr="")
        # git rev-parse HEAD — instant, returns a fake SHA.
        return subprocess.CompletedProcess(args=cmd, returncode=0, stdout="deadbeef\n", stderr="")

    # Two separate toolkit instances so _ensure_workdir runs once each.
    tk1 = GitToolkit(repo_url="https://github.com/example/repo1.git")
    tk2 = GitToolkit(repo_url="https://github.com/example/repo2.git")

    with patch("subprocess.run", side_effect=_fake_subprocess):
        # _ensure_workdir is async (#214) and itself uses asyncio.to_thread
        # for the subprocess calls. Awaiting both directly via gather is
        # what proves the event loop is not blocked.
        await asyncio.gather(
            tk1._ensure_workdir(),
            tk2._ensure_workdir(),
        )

    assert len(call_records) == 2, "Expected exactly 2 git-clone calls (one per toolkit)"

    first_start, first_end = min(call_records, key=lambda r: r[0])
    second_start, _ = max(call_records, key=lambda r: r[0])

    overlap = first_end - second_start
    assert overlap >= _OVERLAP_THRESHOLD, (
        f"Threads did not overlap: first ended at {first_end:.3f}, "
        f"second started at {second_start:.3f}, overlap={overlap:.3f}s"
    )
