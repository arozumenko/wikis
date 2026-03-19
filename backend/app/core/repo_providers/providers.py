"""
Repository providers - Convert platform toolkit configs to git clone URLs.

Each provider implements the logic for:
1. Parsing repository URLs/identifiers
2. Building authenticated clone URLs
3. Normalizing repository identifiers
"""

import logging
import re
from abc import ABC, abstractmethod
from typing import Any
from urllib.parse import quote, urlparse

from .models import GitCloneConfig, ProviderConfig, ProviderType

logger = logging.getLogger(__name__)


class BaseRepoProvider(ABC):
    """Base class for repository providers."""

    provider_type: ProviderType

    @abstractmethod
    def build_clone_config(self, config: ProviderConfig) -> GitCloneConfig:
        """Build a GitCloneConfig from provider configuration."""
        pass

    @abstractmethod
    def normalize_repository(self, repository: str) -> str:
        """Normalize repository identifier to canonical form."""
        pass

    @abstractmethod
    def parse_repository_url(self, url: str) -> tuple[str, str | None]:
        """
        Parse a repository URL to extract repository identifier and branch.

        Returns:
            Tuple of (repo_identifier, branch or None)
        """
        pass

    def _extract_secret(self, value: Any) -> str | None:
        """Extract secret value from SecretStr or plain string."""
        if value is None:
            return None
        if hasattr(value, "get_secret_value"):
            return value.get_secret_value()
        return str(value).strip() if value else None


class GitHubProvider(BaseRepoProvider):
    """
    GitHub provider - supports github.com and GitHub Enterprise.

    Clone URL formats:
    - Anonymous: https://github.com/owner/repo.git
    - Token: https://{token}@github.com/owner/repo.git
    - User/Pass: https://{user}:{token}@github.com/owner/repo.git
    """

    provider_type = ProviderType.GITHUB
    DEFAULT_HOST = "github.com"
    DEFAULT_API_URL = "https://api.github.com"

    def build_clone_config(self, config: ProviderConfig) -> GitCloneConfig:
        """Build GitHub clone configuration."""
        repo = self.normalize_repository(config.repository)
        host = self._extract_host(config.api_url) or self.DEFAULT_HOST

        # Build clone URL
        if config.token:
            if config.username:
                clone_url = f"https://{config.username}:{config.token}@{host}/{repo}.git"
            else:
                clone_url = f"https://{config.token}@{host}/{repo}.git"
            auth_method = "token"
            requires_auth = True
        elif config.username and config.password:
            clone_url = f"https://{config.username}:{config.password}@{host}/{repo}.git"
            auth_method = "password"
            requires_auth = True
        else:
            clone_url = f"https://{host}/{repo}.git"
            auth_method = "anonymous"
            requires_auth = False

        return GitCloneConfig(
            clone_url=clone_url,
            repo_identifier=repo,
            branch=config.branch,
            provider=self.provider_type,
            host=host,
            api_url=config.api_url or self.DEFAULT_API_URL,
            requires_auth=requires_auth,
            auth_method=auth_method,
        )

    def normalize_repository(self, repository: str) -> str:
        """Normalize GitHub repository to owner/repo format."""
        repo = repository.strip()

        # Handle full URLs
        if repo.startswith(("http://", "https://")):
            parsed = urlparse(repo)
            path = parsed.path.strip("/")
            if path.endswith(".git"):
                path = path[:-4]
            return path

        # Handle SSH format: git@github.com:owner/repo.git
        if repo.startswith("git@"):
            match = re.match(r"^git@[^:]+:([^/]+/[^/]+?)(?:\.git)?$", repo)
            if match:
                return match.group(1)

        # Already in owner/repo format - strip .git suffix if present
        if repo.endswith(".git"):
            repo = repo[:-4]
        return repo.strip("/")

    def parse_repository_url(self, url: str) -> tuple[str, str | None]:
        """Parse GitHub URL to extract repo and branch."""
        repo = self.normalize_repository(url)

        # Check for branch in URL: github.com/owner/repo/tree/branch
        if "/tree/" in url:
            match = re.search(r"/tree/([^/]+)", url)
            if match:
                return repo, match.group(1)

        return repo, None

    def _extract_host(self, api_url: str) -> str:
        """Extract host from API URL."""
        if not api_url:
            return self.DEFAULT_HOST

        # GitHub Enterprise: api_url might be https://github.example.com/api/v3
        parsed = urlparse(api_url)
        if parsed.netloc == "api.github.com":
            return self.DEFAULT_HOST
        return parsed.netloc or self.DEFAULT_HOST


class GitLabProvider(BaseRepoProvider):
    """
    GitLab provider - supports gitlab.com and self-hosted instances.

    Clone URL formats:
    - Anonymous: https://gitlab.com/group/project.git
    - Token: https://oauth2:{token}@gitlab.com/group/project.git
    - User/Pass: https://{user}:{password}@gitlab.com/group/project.git
    """

    provider_type = ProviderType.GITLAB
    DEFAULT_HOST = "gitlab.com"

    def build_clone_config(self, config: ProviderConfig) -> GitCloneConfig:
        """Build GitLab clone configuration."""
        repo = self.normalize_repository(config.repository)
        host = self._extract_host(config.api_url)

        # Build clone URL
        if config.token:
            # GitLab uses oauth2:{token} format for token auth
            clone_url = f"https://oauth2:{config.token}@{host}/{repo}.git"
            auth_method = "token"
            requires_auth = True
        elif config.username and config.password:
            clone_url = f"https://{config.username}:{config.password}@{host}/{repo}.git"
            auth_method = "password"
            requires_auth = True
        else:
            clone_url = f"https://{host}/{repo}.git"
            auth_method = "anonymous"
            requires_auth = False

        return GitCloneConfig(
            clone_url=clone_url,
            repo_identifier=repo,
            branch=config.branch,
            provider=self.provider_type,
            host=host,
            api_url=config.api_url or f"https://{host}",
            requires_auth=requires_auth,
            auth_method=auth_method,
            metadata={"is_cloud": config.is_cloud},
        )

    def normalize_repository(self, repository: str) -> str:
        """Normalize GitLab repository to namespace/project format."""
        repo = repository.strip()

        # Handle full URLs
        if repo.startswith(("http://", "https://")):
            parsed = urlparse(repo)
            path = parsed.path.strip("/")
            if path.endswith(".git"):
                path = path[:-4]
            # Remove /-/... suffixes (merge_requests, issues, etc.)
            if "/-/" in path:
                path = path.split("/-/")[0]
            return path

        # Handle SSH format
        if repo.startswith("git@"):
            match = re.match(r"^git@[^:]+:(.+?)(?:\.git)?$", repo)
            if match:
                return match.group(1)

        return repo.rstrip(".git").strip("/")

    def parse_repository_url(self, url: str) -> tuple[str, str | None]:
        """Parse GitLab URL to extract repo and branch."""
        repo = self.normalize_repository(url)

        # Check for branch in URL: gitlab.com/group/project/-/tree/branch
        if "/-/tree/" in url:
            match = re.search(r"/-/tree/([^/]+)", url)
            if match:
                return repo, match.group(1)

        return repo, None

    def _extract_host(self, api_url: str) -> str:
        """Extract host from GitLab URL."""
        if not api_url:
            return self.DEFAULT_HOST

        parsed = urlparse(api_url.rstrip("/"))
        return parsed.netloc or self.DEFAULT_HOST


class BitbucketProvider(BaseRepoProvider):
    """
    Bitbucket provider - supports Cloud and Server/Data Center.

    Clone URL formats:
    Cloud:
    - Anonymous: https://bitbucket.org/workspace/repo.git
    - Auth: https://{user}:{app_password}@bitbucket.org/workspace/repo.git

    Server:
    - https://{user}:{password}@server/scm/{project}/{repo}.git
    """

    provider_type = ProviderType.BITBUCKET
    DEFAULT_HOST = "bitbucket.org"

    def build_clone_config(self, config: ProviderConfig) -> GitCloneConfig:
        """Build Bitbucket clone configuration."""
        host = self._extract_host(config.api_url)
        is_cloud = "bitbucket.org" in host.lower()

        repo = self.normalize_repository(config.repository)

        # For Server/DC, we need the project key in the path
        if not is_cloud and config.project:
            # Server format: /scm/{project}/{repo}.git
            repo_path = f"scm/{config.project}/{repo}"
        else:
            # Cloud format: /{workspace}/{repo}.git
            # If project is provided, it's the workspace
            if config.project and "/" not in repo:
                repo_path = f"{config.project}/{repo}"
            else:
                repo_path = repo

        # Build clone URL with authentication
        if config.username and config.password:
            clone_url = f"https://{config.username}:{config.password}@{host}/{repo_path}.git"
            auth_method = "password"
            requires_auth = True
        elif config.token:
            # Bitbucket Cloud can use x-token-auth:{token} or just token as password
            clone_url = f"https://x-token-auth:{config.token}@{host}/{repo_path}.git"
            auth_method = "token"
            requires_auth = True
        else:
            clone_url = f"https://{host}/{repo_path}.git"
            auth_method = "anonymous"
            requires_auth = False

        return GitCloneConfig(
            clone_url=clone_url,
            repo_identifier=repo_path.replace("scm/", "") if not is_cloud else repo_path,
            branch=config.branch,
            provider=self.provider_type,
            host=host,
            api_url=config.api_url or f"https://{host}",
            requires_auth=requires_auth,
            auth_method=auth_method,
            metadata={
                "is_cloud": is_cloud,
                "project": config.project,
            },
        )

    def normalize_repository(self, repository: str) -> str:
        """Normalize Bitbucket repository to workspace/repo or project/repo format."""
        repo = repository.strip()

        # Handle full URLs
        if repo.startswith(("http://", "https://")):
            parsed = urlparse(repo)
            path = parsed.path.strip("/")
            if path.endswith(".git"):
                path = path[:-4]
            # Remove /scm/ prefix from Server URLs
            if path.startswith("scm/"):
                path = path[4:]
            return path

        # Handle SSH format
        if repo.startswith("git@"):
            match = re.match(r"^git@[^:]+:(.+?)(?:\.git)?$", repo)
            if match:
                path = match.group(1)
                if path.startswith("scm/"):
                    path = path[4:]
                return path

        return repo.rstrip(".git").strip("/")

    def parse_repository_url(self, url: str) -> tuple[str, str | None]:
        """Parse Bitbucket URL to extract repo and branch."""
        repo = self.normalize_repository(url)

        # Cloud: bitbucket.org/workspace/repo/src/branch
        if "/src/" in url:
            match = re.search(r"/src/([^/]+)", url)
            if match:
                return repo, match.group(1)

        # Server: /browse?at=refs/heads/branch
        if "at=refs/heads/" in url:
            match = re.search(r"at=refs/heads/([^&]+)", url)
            if match:
                return repo, match.group(1)

        return repo, None

    def _extract_host(self, api_url: str) -> str:
        """Extract host from Bitbucket URL."""
        if not api_url:
            return self.DEFAULT_HOST

        parsed = urlparse(api_url.rstrip("/"))
        host = parsed.netloc or self.DEFAULT_HOST

        # Normalize Cloud API URL to domain
        if host == "api.bitbucket.org":
            return self.DEFAULT_HOST

        return host


class AzureDevOpsProvider(BaseRepoProvider):
    """
    Azure DevOps provider - supports dev.azure.com and legacy visualstudio.com.

    Clone URL format:
    - https://{org}@dev.azure.com/{org}/{project}/_git/{repo}
    - https://{pat}@dev.azure.com/{org}/{project}/_git/{repo}

    For legacy visualstudio.com:
    - https://{org}.visualstudio.com/{project}/_git/{repo}
    """

    provider_type = ProviderType.AZURE_DEVOPS

    def build_clone_config(self, config: ProviderConfig) -> GitCloneConfig:
        """Build Azure DevOps clone configuration."""
        org, url_kind = self._parse_organization_url(config.api_url)
        if not org:
            raise ValueError(f"Invalid Azure DevOps organization URL: {config.api_url}")

        project = config.project
        if not project:
            raise ValueError("Azure DevOps requires project name")

        repo = self.normalize_repository(config.repository)

        # URL-encode project and repo for safety
        project_encoded = quote(project, safe="")
        repo_encoded = quote(repo, safe="")

        # Build clone URL based on URL kind
        if url_kind == "dev.azure.com":
            host = "dev.azure.com"
            if config.token:
                # PAT auth: use token in URL
                clone_url = f"https://{config.token}@dev.azure.com/{org}/{project_encoded}/_git/{repo_encoded}"
                auth_method = "token"
                requires_auth = True
            else:
                # Anonymous (only works for public repos)
                clone_url = f"https://dev.azure.com/{org}/{project_encoded}/_git/{repo_encoded}"
                auth_method = "anonymous"
                requires_auth = False
        else:
            # Legacy visualstudio.com format
            host = f"{org}.visualstudio.com"
            if config.token:
                clone_url = f"https://{config.token}@{org}.visualstudio.com/{project_encoded}/_git/{repo_encoded}"
                auth_method = "token"
                requires_auth = True
            else:
                clone_url = f"https://{org}.visualstudio.com/{project_encoded}/_git/{repo_encoded}"
                auth_method = "anonymous"
                requires_auth = False

        return GitCloneConfig(
            clone_url=clone_url,
            repo_identifier=f"{org}/{project}/{repo}",
            branch=config.branch,
            provider=self.provider_type,
            host=host,
            api_url=config.api_url,
            requires_auth=requires_auth,
            auth_method=auth_method,
            metadata={
                "organization": org,
                "project": project,
                "url_kind": url_kind,
            },
        )

    def normalize_repository(self, repository: str) -> str:
        """Normalize ADO repository to repo name only."""
        repo = repository.strip()

        # Handle full URLs
        if "/_git/" in repo:
            # Extract repo name from URL
            match = re.search(r"/_git/([^/?]+)", repo)
            if match:
                return match.group(1)

        # Just the repo name
        return repo.rstrip(".git").strip("/")

    def parse_repository_url(self, url: str) -> tuple[str, str | None]:
        """Parse Azure DevOps URL to extract repo and branch."""
        repo = self.normalize_repository(url)

        # Branch in URL: ?version=GBbranch or ?version=GB{branch}
        if "version=GB" in url:
            match = re.search(r"version=GB([^&]+)", url)
            if match:
                return repo, match.group(1)

        return repo, None

    def _parse_organization_url(self, url: str) -> tuple[str | None, str | None]:
        """
        Parse organization URL to extract org name and URL kind.

        Supported formats:
        - https://dev.azure.com/{org}
        - https://{org}.visualstudio.com

        Returns:
            Tuple of (org_name, url_kind) where url_kind is 'dev.azure.com' or '*.visualstudio.com'
        """
        if not url:
            return None, None

        url = url.rstrip("/")

        # dev.azure.com format
        match = re.match(r"^https?://dev\.azure\.com/([^/]+)/?", url, re.IGNORECASE)
        if match:
            return match.group(1), "dev.azure.com"

        # visualstudio.com format
        match = re.match(r"^https?://([^/.]+)\.visualstudio\.com/?", url, re.IGNORECASE)
        if match:
            return match.group(1), "*.visualstudio.com"

        return None, None


# Provider registry
PROVIDERS = {
    ProviderType.GITHUB: GitHubProvider(),
    ProviderType.GITLAB: GitLabProvider(),
    ProviderType.BITBUCKET: BitbucketProvider(),
    ProviderType.AZURE_DEVOPS: AzureDevOpsProvider(),
}


def get_provider(provider_type: ProviderType) -> BaseRepoProvider:
    """Get provider instance by type."""
    if provider_type not in PROVIDERS:
        raise ValueError(f"Unsupported provider: {provider_type}")
    return PROVIDERS[provider_type]
