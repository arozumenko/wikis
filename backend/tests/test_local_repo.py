"""Tests for local filesystem repo support."""

from __future__ import annotations

import subprocess

import pytest

from app.core.local_repo_provider import (
    extract_git_metadata,
    is_local_path,
    make_local_wiki_id,
    validate_local_path,
)


class TestIsLocalPath:
    def test_absolute_path(self):
        assert is_local_path("/home/user/repo") is True

    def test_file_url(self):
        assert is_local_path("file:///home/user/repo") is True

    def test_https_url(self):
        assert is_local_path("https://github.com/a/b") is False


class TestValidateLocalPath:
    def test_no_allowlist_rejects_all(self, tmp_path):
        """ALLOWED_LOCAL_PATHS unset → all local paths rejected (opt-in)."""
        with pytest.raises(ValueError, match="disabled"):
            validate_local_path(str(tmp_path), allowed_paths=None)

    def test_empty_allowlist_rejects_all(self, tmp_path):
        with pytest.raises(ValueError, match="disabled"):
            validate_local_path(str(tmp_path), allowed_paths=[])

    def test_valid_dir(self, tmp_path):
        assert validate_local_path(str(tmp_path), allowed_paths=[str(tmp_path)]) == tmp_path.resolve()

    def test_nonexistent_raises(self, tmp_path):
        with pytest.raises(ValueError, match="does not exist"):
            validate_local_path(str(tmp_path / "nonexistent"), allowed_paths=[str(tmp_path)])

    def test_file_raises(self, tmp_path):
        f = tmp_path / "f.txt"
        f.write_text("hi")
        with pytest.raises(ValueError, match="not a directory"):
            validate_local_path(str(f), allowed_paths=[str(tmp_path)])

    def test_outside_allowed_raises(self, tmp_path):
        safe = tmp_path / "safe"
        safe.mkdir()
        outside = tmp_path / "other"
        outside.mkdir()
        with pytest.raises(ValueError, match="outside allowed"):
            validate_local_path(str(outside), allowed_paths=[str(safe)])

    def test_path_traversal_rejected(self, tmp_path):
        """/../ tricks are rejected after resolve()."""
        safe = tmp_path / "safe"
        safe.mkdir()
        outside = tmp_path / "other"
        outside.mkdir()
        # Try to sneak out via ../
        with pytest.raises(ValueError, match="outside allowed"):
            validate_local_path(str(safe / ".." / "other"), allowed_paths=[str(safe)])

    def test_inside_allowed(self, tmp_path):
        sub = tmp_path / "repos" / "myrepo"
        sub.mkdir(parents=True)
        assert validate_local_path(str(sub), allowed_paths=[str(tmp_path / "repos")]) == sub.resolve()

    def test_symlink_outside_allowed_rejected(self, tmp_path):
        allowed = tmp_path / "allowed"
        allowed.mkdir()
        secret = tmp_path / "secret"
        secret.mkdir()
        link = allowed / "evil_link"
        link.symlink_to(secret)
        with pytest.raises(ValueError, match="outside allowed"):
            validate_local_path(str(link), allowed_paths=[str(allowed)])

    def test_symlink_inside_allowed_accepted(self, tmp_path):
        allowed = tmp_path / "allowed"
        allowed.mkdir()
        real = allowed / "real_dir"
        real.mkdir()
        link = allowed / "link"
        link.symlink_to(real)
        result = validate_local_path(str(link), allowed_paths=[str(allowed)])
        assert result == real.resolve()


class TestExtractGitMetadata:
    def test_non_git(self, tmp_path):
        info = extract_git_metadata(tmp_path)
        assert info.is_git is False

    def test_git_repo(self, tmp_path):
        subprocess.run(["git", "init"], cwd=tmp_path, capture_output=True)
        subprocess.run(
            ["git", "-c", "user.email=test@test.com", "-c", "user.name=Test",
             "commit", "--allow-empty", "-m", "init"],
            cwd=tmp_path,
            capture_output=True,
        )
        info = extract_git_metadata(tmp_path)
        assert info.is_git is True
        # branch is the current branch name or "HEAD" in detached-HEAD state;
        # either way it must be a non-empty string when a commit exists.
        assert isinstance(info.branch, str) and info.branch != ""


class TestMakeLocalWikiId:
    def test_deterministic(self, tmp_path):
        assert make_local_wiki_id(str(tmp_path), "main") == make_local_wiki_id(str(tmp_path), "main")

    def test_varies_by_branch(self, tmp_path):
        assert make_local_wiki_id(str(tmp_path), "main") != make_local_wiki_id(str(tmp_path), "dev")


class TestGenerateWikiRequestValidator:
    """Tests for the Pydantic model_validator on GenerateWikiRequest."""

    def test_local_path_without_allowlist_raises_422(self, tmp_path, monkeypatch):
        monkeypatch.delenv("ALLOWED_LOCAL_PATHS", raising=False)
        from pydantic import ValidationError

        from app.models.api import GenerateWikiRequest

        with pytest.raises(ValidationError):
            GenerateWikiRequest(repo_url=str(tmp_path))

    def test_local_path_with_allowlist_sets_provider(self, tmp_path, monkeypatch):
        monkeypatch.setenv("ALLOWED_LOCAL_PATHS", str(tmp_path))
        # Need to reload settings — use importlib
        import importlib

        import app.config as cfg_mod
        from app.models.api import GenerateWikiRequest

        importlib.reload(cfg_mod)
        req = GenerateWikiRequest(repo_url=str(tmp_path))
        assert req.provider == "local"

    def test_remote_url_not_affected(self):
        from app.models.api import GenerateWikiRequest

        req = GenerateWikiRequest(repo_url="https://github.com/a/b")
        assert req.provider == "github"

    def test_provider_local_explicit_accepted(self):
        from app.models.api import GenerateWikiRequest

        req = GenerateWikiRequest(repo_url="https://github.com/a/b", provider="local")
        assert req.provider == "local"
