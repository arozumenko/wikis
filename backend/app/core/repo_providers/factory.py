"""
Repository Provider Factory

Converts platform toolkit configurations into GitCloneConfig
objects that can be used by our filesystem-based wiki generator.

Supported toolkit types:
- github: Uses github_configuration with base_url, access_token, username, password, app_id, app_private_key
- gitlab: Uses gitlab_configuration with url, private_token
- bitbucket: Uses bitbucket_configuration with url, username, password
- ado_repos: Uses ado_configuration with organization_url, project, token
"""

import logging
from typing import Any

from .models import GitCloneConfig, LocalPathConfig, ProviderConfig, ProviderType
from .providers import get_provider

logger = logging.getLogger(__name__)


# Map toolkit type names to ProviderType
TOOLKIT_TYPE_MAP = {
    "github": ProviderType.GITHUB,
    "gitlab": ProviderType.GITLAB,
    "bitbucket": ProviderType.BITBUCKET,
    "ado_repos": ProviderType.AZURE_DEVOPS,
    "ado": ProviderType.AZURE_DEVOPS,
}


class RepoProviderFactory:
    """
    Factory for creating GitCloneConfig from various configuration sources.

    This handles the conversion from platform toolkit configurations to
    git-clonable URLs. It normalizes the different configuration formats
    used by each provider (GitHub, GitLab, Bitbucket, Azure DevOps).
    """

    @classmethod
    def from_toolkit_config(
        cls,
        provider_type: str,
        config: dict[str, Any],
        repository: str,
        branch: str = "main",
        project: str | None = None,
    ) -> GitCloneConfig:
        """
        Create GitCloneConfig from a platform toolkit configuration.

        Args:
            provider_type: Provider type string (github, gitlab, bitbucket, ado)
            config: Toolkit configuration dict
            repository: Repository path/URL
            branch: Branch to clone (default: main)
            project: Project/workspace name (required for ADO, optional for others)

        Returns:
            GitCloneConfig ready for git clone

        Example:
            config = RepoProviderFactory.from_toolkit_config(
                provider_type="github",
                config={
                    "base_url": "https://api.github.com",
                    "access_token": "ghp_xxxx"
                },
                repository="owner/repo",
                branch="main"
            )
        """
        # Parse provider type
        ptype = ProviderType.from_string(provider_type)

        # Normalize configuration
        normalized = cls._normalize_config(ptype, config, repository, branch, project)

        # Get provider and build config
        provider = get_provider(ptype)
        return provider.build_clone_config(normalized)

    @classmethod
    def from_url(
        cls,
        url: str,
        token: str | None = None,
        username: str | None = None,
        password: str | None = None,
        branch: str = "main",
    ) -> "GitCloneConfig | LocalPathConfig":
        """
        Create GitCloneConfig (or LocalPathConfig) from a repository URL.

        Auto-detects provider from URL. Returns LocalPathConfig for local paths.
        """
        from pathlib import Path as _Path

        from app.core.local_repo_provider import extract_git_metadata, is_local_path

        if is_local_path(url):
            resolved = str(_Path(url.removeprefix("file://")).resolve())
            info = extract_git_metadata(_Path(resolved))
            return LocalPathConfig(
                local_path=resolved,
                branch=info.branch if info.is_git else None,
                commit_hash=info.commit_hash if info.is_git else None,
                remote_url=info.remote_url if info.is_git else None,
            )

        # Detect provider from URL
        ptype = ProviderType.detect_from_url(url)
        if not ptype:
            raise ValueError(f"Could not detect provider from URL: {url}")

        provider = get_provider(ptype)

        # Parse repository and branch from URL
        repo, url_branch = provider.parse_repository_url(url)
        if url_branch and branch == "main":
            branch = url_branch

        # Build minimal config
        config = ProviderConfig(
            provider=ptype,
            host="",  # Will be extracted from URL
            api_url="",
            repository=repo,
            branch=branch,
            token=token,
            username=username,
            password=password,
        )

        return provider.build_clone_config(config)

    @classmethod
    def _normalize_config(
        cls,
        provider: ProviderType,
        config: dict[str, Any],
        repository: str,
        branch: str,
        project: str | None,
    ) -> ProviderConfig:
        """
        Normalize toolkit configuration to ProviderConfig.

        Maps the various configuration fields to our normalized format.
        """

        # Extract secrets (handle SecretStr objects)
        def get_secret(key: str) -> str | None:
            val = config.get(key)
            if val is None:
                return None
            if hasattr(val, "get_secret_value"):
                return val.get_secret_value()
            return str(val).strip() if val else None

        if provider == ProviderType.GITHUB:
            return ProviderConfig(
                provider=provider,
                host=cls._extract_host(config.get("base_url", "https://api.github.com")),
                api_url=config.get("base_url", "https://api.github.com"),
                repository=repository,
                branch=branch,
                token=get_secret("access_token"),
                username=config.get("username"),
                password=get_secret("password"),
                is_cloud="api.github.com" in config.get("base_url", "https://api.github.com"),
            )

        elif provider == ProviderType.GITLAB:
            url = config.get("url", "https://gitlab.com")
            return ProviderConfig(
                provider=provider,
                host=cls._extract_host(url),
                api_url=url,
                repository=repository,
                branch=branch,
                token=get_secret("private_token"),
                is_cloud="gitlab.com" in url.lower(),
            )

        elif provider == ProviderType.BITBUCKET:
            url = config.get("url", "https://bitbucket.org")
            is_cloud = "bitbucket.org" in url.lower()
            return ProviderConfig(
                provider=provider,
                host=cls._extract_host(url),
                api_url=url,
                repository=repository,
                branch=branch,
                username=config.get("username"),
                password=get_secret("password"),
                project=project or config.get("project"),  # workspace for Cloud, project key for Server
                is_cloud=is_cloud,
            )

        elif provider == ProviderType.AZURE_DEVOPS:
            org_url = config.get("organization_url", "")
            return ProviderConfig(
                provider=provider,
                host=cls._extract_host(org_url),
                api_url=org_url,
                repository=repository,
                branch=branch,
                token=get_secret("token"),
                project=project or config.get("project"),
                organization=cls._extract_ado_org(org_url),
                is_cloud=True,  # ADO is always cloud
            )

        else:
            raise ValueError(f"Unsupported provider: {provider}")

    @staticmethod
    def _extract_host(url: str) -> str:
        """Extract host from URL."""
        if not url:
            return ""
        from urllib.parse import urlparse

        parsed = urlparse(url)
        return parsed.netloc or ""

    @staticmethod
    def _extract_ado_org(url: str) -> str | None:
        """Extract organization name from Azure DevOps URL."""
        import re

        if not url:
            return None

        # dev.azure.com format
        match = re.match(r"^https?://dev\.azure\.com/([^/]+)", url, re.IGNORECASE)
        if match:
            return match.group(1)

        # visualstudio.com format
        match = re.match(r"^https?://([^/.]+)\.visualstudio\.com", url, re.IGNORECASE)
        if match:
            return match.group(1)

        return None


# Convenience function for direct use
def create_clone_config(
    provider_type: str,
    config: dict[str, Any],
    repository: str,
    branch: str = "main",
    project: str | None = None,
) -> GitCloneConfig:
    """
    Convenience function to create a GitCloneConfig.

    See RepoProviderFactory.from_toolkit_config for details.
    """
    return RepoProviderFactory.from_toolkit_config(
        provider_type=provider_type,
        config=config,
        repository=repository,
        branch=branch,
        project=project,
    )


def create_clone_config_from_expanded_toolkit(
    toolkit_type: str,
    toolkit_settings: dict[str, Any],
) -> GitCloneConfig:
    """
    Create GitCloneConfig from an expanded toolkit configuration.

    This is the main entry point for processing toolkits from the platform.
    It extracts the provider-specific configuration and repository info
    from the expanded toolkit settings.

    Args:
        toolkit_type: The toolkit type (github, gitlab, bitbucket, ado_repos)
        toolkit_settings: The expanded 'settings' dict from the toolkit

    Returns:
        GitCloneConfig ready for git clone

    Example (GitHub):
        settings = {
            'github_configuration': {
                'base_url': 'https://api.github.com',
                'access_token': 'ghp_xxx'
            },
            'repository': 'owner/repo',
            'active_branch': 'main',
            'base_branch': 'main'
        }
        config = create_clone_config_from_expanded_toolkit('github', settings)

    Example (GitLab):
        settings = {
            'gitlab_configuration': {
                'url': 'https://gitlab.com',
                'private_token': 'glpat_xxx'
            },
            'repository': 'group/project',
            'branch': 'main'
        }
        config = create_clone_config_from_expanded_toolkit('gitlab', settings)

    Example (Bitbucket):
        settings = {
            'bitbucket_configuration': {
                'url': 'https://bitbucket.org',
                'username': 'user',
                'password': 'app_password'
            },
            'project': 'workspace',
            'repository': 'repo-name',
            'branch': 'main',
            'cloud': True
        }
        config = create_clone_config_from_expanded_toolkit('bitbucket', settings)

    Example (Azure DevOps):
        settings = {
            'ado_configuration': {
                'organization_url': 'https://dev.azure.com/myorg',
                'project': 'MyProject',
                'token': 'xxx'
            },
            'repository_id': 'my-repo',
            'base_branch': 'main'
        }
        config = create_clone_config_from_expanded_toolkit('ado_repos', settings)
    """
    # Resolve provider type
    ptype = TOOLKIT_TYPE_MAP.get(toolkit_type.lower())
    if not ptype:
        # Try to parse using ProviderType
        ptype = ProviderType.from_string(toolkit_type)

    # Extract provider-specific configuration and repository info
    if ptype == ProviderType.GITHUB:
        provider_config = toolkit_settings.get("github_configuration", {})
        repository = toolkit_settings.get("repository", "")
        branch = toolkit_settings.get("active_branch") or toolkit_settings.get("base_branch", "main")
        project = None

    elif ptype == ProviderType.GITLAB:
        provider_config = toolkit_settings.get("gitlab_configuration", {})
        repository = toolkit_settings.get("repository", "")
        branch = toolkit_settings.get("branch", "main")
        project = None

    elif ptype == ProviderType.BITBUCKET:
        provider_config = toolkit_settings.get("bitbucket_configuration", {})
        repository = toolkit_settings.get("repository", "")
        branch = toolkit_settings.get("branch", "main")
        project = toolkit_settings.get("project", "")  # workspace for Cloud, project key for Server
        # Add cloud detection to config if present
        if toolkit_settings.get("cloud") is not None:
            provider_config = {**provider_config, "_cloud": toolkit_settings.get("cloud")}

    elif ptype == ProviderType.AZURE_DEVOPS:
        provider_config = toolkit_settings.get("ado_configuration", {})
        # ADO uses repository_id field
        repository = toolkit_settings.get("repository_id", "")
        branch = toolkit_settings.get("active_branch") or toolkit_settings.get("base_branch", "main")
        # Project comes from ado_configuration
        project = provider_config.get("project", "")

    else:
        raise ValueError(f"Unsupported toolkit type: {toolkit_type}")

    return RepoProviderFactory.from_toolkit_config(
        provider_type=ptype.value,
        config=provider_config,
        repository=repository,
        branch=branch,
        project=project,
    )
