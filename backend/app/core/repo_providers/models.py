"""
Data models for repository providers.

These models define the configuration and output structures for converting
platform toolkit configs to git-clonable URLs.
"""

from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any, Optional
from urllib.parse import urlparse


class ProviderType(StrEnum):
    """Supported repository provider types."""

    GITHUB = "github"
    GITLAB = "gitlab"
    BITBUCKET = "bitbucket"
    AZURE_DEVOPS = "ado"

    @classmethod
    def from_string(cls, value: str) -> "ProviderType":
        """Convert string to ProviderType, with common aliases."""
        aliases = {
            "github": cls.GITHUB,
            "gh": cls.GITHUB,
            "gitlab": cls.GITLAB,
            "gl": cls.GITLAB,
            "bitbucket": cls.BITBUCKET,
            "bb": cls.BITBUCKET,
            "ado": cls.AZURE_DEVOPS,
            "azure": cls.AZURE_DEVOPS,
            "azure_devops": cls.AZURE_DEVOPS,
            "azuredevops": cls.AZURE_DEVOPS,
            "vsts": cls.AZURE_DEVOPS,
        }
        normalized = value.lower().strip()
        if normalized in aliases:
            return aliases[normalized]
        raise ValueError(f"Unknown provider type: {value}")

    @classmethod
    def detect_from_url(cls, url: str) -> Optional["ProviderType"]:
        """Detect provider type from a URL."""
        url_lower = url.lower()

        if "github.com" in url_lower or "github" in url_lower:
            return cls.GITHUB
        elif "gitlab.com" in url_lower or "gitlab" in url_lower:
            return cls.GITLAB
        elif "bitbucket.org" in url_lower or "bitbucket" in url_lower:
            return cls.BITBUCKET
        elif "dev.azure.com" in url_lower or "visualstudio.com" in url_lower:
            return cls.AZURE_DEVOPS

        return None


@dataclass
class GitCloneConfig:
    """
    Configuration for cloning a repository via git.

    This is the output of provider conversion - contains everything needed
    to clone a repository using the git binary.

    Attributes:
        clone_url: The authenticated HTTPS URL for git clone
        repo_identifier: Canonical identifier (e.g., "owner/repo" for GitHub)
        branch: Branch to clone (default: main)
        provider: The provider type
        host: The git host (e.g., github.com, gitlab.example.com)
        api_url: Optional API base URL for the provider
        requires_auth: Whether authentication is required
        auth_method: Authentication method used (token, password, ssh, etc.)
        metadata: Additional provider-specific metadata
    """

    clone_url: str
    repo_identifier: str
    branch: str = "main"
    provider: ProviderType = ProviderType.GITHUB
    host: str = ""
    api_url: str = ""
    requires_auth: bool = False
    auth_method: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        """Validate and normalize fields after initialization."""
        if not self.clone_url:
            raise ValueError("clone_url is required")
        if not self.repo_identifier:
            raise ValueError("repo_identifier is required")

        # Extract host from clone_url if not provided
        if not self.host:
            parsed = urlparse(self.clone_url.replace("git@", "https://").split(":")[0])
            self.host = parsed.netloc or ""

    @property
    def safe_clone_url(self) -> str:
        """Return clone URL with credentials redacted for logging."""
        parsed = urlparse(self.clone_url)
        if parsed.password:
            # Has explicit password
            safe = f"{parsed.scheme}://{parsed.username}:***@{parsed.netloc.split('@')[-1]}{parsed.path}"
            return safe
        elif "@" in self.clone_url and parsed.username:
            # Has token as username
            return f"{parsed.scheme}://***@{parsed.netloc.split('@')[-1]}{parsed.path}"
        return self.clone_url

    @property
    def safe_url(self) -> str:
        """Alias for safe_clone_url for convenience."""
        return self.safe_clone_url

    def sanitize_output(self, text: str) -> str:
        """
        Remove any credentials from output text (e.g., error messages).

        This extracts credentials from the clone_url and replaces any occurrence
        in the provided text with redacted placeholders.
        """
        if not text:
            return text

        result = text
        parsed = urlparse(self.clone_url)

        # Remove password if present
        if parsed.password:
            result = result.replace(parsed.password, "<redacted>")

        # Remove username/token if present and looks like a token
        if parsed.username:
            # Only redact if it looks like a token (not a regular username)
            # Tokens are typically long alphanumeric strings
            if len(parsed.username) > 20 or any(c.isdigit() for c in parsed.username):
                result = result.replace(parsed.username, "<token>")

        return result

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "clone_url": self.safe_clone_url,  # Always use safe URL in dict
            "repo_identifier": self.repo_identifier,
            "branch": self.branch,
            "provider": self.provider.value,
            "host": self.host,
            "api_url": self.api_url,
            "requires_auth": self.requires_auth,
            "auth_method": self.auth_method,
            "metadata": self.metadata,
        }


@dataclass
class LocalPathConfig:
    """Configuration for a local filesystem path (no git clone needed)."""

    local_path: str  # resolved absolute path
    branch: str | None = None  # from git metadata, or None for non-git dirs
    commit_hash: str | None = None  # from git metadata
    remote_url: str | None = None  # from git metadata (for source links)
    provider: str = "local"

    @property
    def repo_identifier(self) -> str:
        from pathlib import Path as _Path

        return _Path(self.local_path).name


@dataclass
class ProviderConfig:
    """
    Normalized provider configuration.

    This is the input format after normalizing toolkit configs from various sources.
    """

    provider: ProviderType
    host: str  # Base host (e.g., github.com, gitlab.example.com)
    api_url: str  # API base URL

    # Repository identification
    repository: str  # Repository path/identifier
    branch: str = "main"

    # Authentication (various methods supported)
    token: str | None = None  # Access token / PAT
    username: str | None = None  # For username/password auth
    password: str | None = None  # Password or app password

    # Provider-specific
    project: str | None = None  # ADO project, Bitbucket workspace
    organization: str | None = None  # ADO organization

    # Flags
    is_cloud: bool = True  # True for SaaS, False for self-hosted

    def has_auth(self) -> bool:
        """Check if any authentication method is configured."""
        return bool(self.token or (self.username and self.password))

    def get_auth_method(self) -> str:
        """Determine which authentication method is configured."""
        if self.token:
            return "token"
        elif self.username and self.password:
            return "password"
        return "anonymous"
