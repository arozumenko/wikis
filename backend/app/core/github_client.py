"""
Standalone GitHub Client for Wiki Toolkit

Lightweight GitHub client that extracts only the essential methods needed for repository indexing
and wiki generation, without external SDK dependencies.

This client is designed for use in standalone plugin applications where the full SDK is not available.
"""

import logging
import re
from typing import Any
from urllib.parse import urlparse

from github import Auth, Github
from github.Consts import DEFAULT_BASE_URL

logger = logging.getLogger(__name__)


class StandaloneGitHubClient:
    """
    Standalone GitHub client with only essential methods for wiki generation.

    Simple Python class that provides GitHub API functionality
    without external dependencies. Uses only GitHub personal access tokens for authentication.
    """

    def __init__(
        self,
        repository: str,
        access_token: str | None = None,
        branch: str = "main",
        base_url: str = DEFAULT_BASE_URL,
        rate_limit_delay: float = 0.1,
    ):
        """
        Initialize the standalone GitHub client.

        Args:
            repository: Repository in format 'owner/repo'
            access_token: Optional GitHub personal access token (ghp_* format)
            branch: Default branch to use
            base_url: GitHub API base URL (for GitHub Enterprise)
            rate_limit_delay: Delay between API calls
        """
        # Clean and validate repository name
        self.github_repository = self.clean_repository_name(repository)
        self.github_base_branch = branch
        self.active_branch = branch
        self.github_base_url = base_url
        self.rate_limit_delay = rate_limit_delay

        # Set up authentication - only token-based authentication
        auth = None
        if access_token:
            auth = Auth.Token(access_token)

        # Initialize GitHub client
        if auth is None:
            # Public repositories access (no authentication)
            self.github_api = Github(base_url=base_url)
            logger.info("Initialized GitHub client for public repository access")
        else:
            # Token authentication
            logger.info("Initialized GitHub client with the following base url: %s", base_url)
            self.github_api = Github(auth=auth)
            logger.info("Initialized GitHub client with token authentication")

        # Get repository instance
        try:
            self.github_repo_instance = self.github_api.get_repo(self.github_repository)
            logger.info(f"Successfully connected to repository: {self.github_repository}")
        except Exception as e:
            logger.error(f"Failed to connect to repository {self.github_repository}: {e}")
            raise

    @staticmethod
    def clean_repository_name(repo_link: str) -> str:
        """
        Clean and format a repository name from various input formats.

        Args:
            repo_link: Repository link or name in various formats

        Returns:
            Cleaned repository name in format "owner/repo"

        Raises:
            ValueError: If the repository name is invalid
        """
        # Handle different repository URL formats
        if repo_link.startswith(("http://", "https://")):
            # Parse URL to extract owner/repo
            parsed = urlparse(repo_link)
            path = parsed.path.strip("/")
            if path.endswith(".git"):
                path = path[:-4]
            return path
        elif repo_link.startswith("git@"):
            # SSH format: git@github.com:owner/repo.git
            match = re.match(r"^git@[^:]+:([^/]+/[^/]+?)(?:\.git)?$", repo_link)
            if not match:
                raise ValueError("Invalid SSH repository format")
            return match.group(1)
        else:
            # Assume it's already in owner/repo format
            match = re.match(r"^([^/]+/[^/]+?)(?:\.git)?$", repo_link)
            if not match:
                raise ValueError("Repository should be in 'owner/repo' format")
            return match.group(1)

    @classmethod
    def from_url_and_token(
        cls, repository_url: str, access_token: str | None = None, branch: str = "main"
    ) -> "StandaloneGitHubClient":
        """
        Convenience method to create client from repository URL and optional token.

        Args:
            repository_url: Repository URL or owner/repo string
            access_token: Optional GitHub access token (ghp_* format)
            branch: Branch to use (default: main)

        Returns:
            Configured StandaloneGitHubClient instance
        """
        return cls(repository=repository_url, access_token=access_token, branch=branch)

    def get_repository_info(self) -> dict[str, Any]:
        """
        Get basic repository information.

        Returns:
            Dictionary with repository metadata
        """
        try:
            repo = self.github_repo_instance
            return {
                "name": repo.name,
                "full_name": repo.full_name,
                "description": repo.description,
                "language": repo.language,
                "stargazers_count": repo.stargazers_count,
                "forks_count": repo.forks_count,
                "default_branch": repo.default_branch,
                "created_at": repo.created_at.isoformat() if repo.created_at else None,
                "updated_at": repo.updated_at.isoformat() if repo.updated_at else None,
                "size": repo.size,
                "open_issues_count": repo.open_issues_count,
                "has_wiki": repo.has_wiki,
                "has_pages": repo.has_pages,
                "topics": repo.get_topics(),
            }
        except Exception as e:
            logger.error(f"Failed to get repository info: {e}")
            return {}

    def _get_files(self, directory_path: str, ref: str, repo_name: str | None = None) -> list[str]:
        """
        Get all files in a directory recursively.

        Args:
            directory_path: Path to the directory (empty string for root)
            ref: Branch or commit reference
            repo_name: Optional repository name to override default

        Returns:
            List of file paths
        """
        from github import GithubException

        try:
            repo = self.github_api.get_repo(repo_name) if repo_name else self.github_repo_instance

            # Handle empty directory path (root)
            if not directory_path:
                contents = repo.get_contents("", ref=ref)
            else:
                contents = repo.get_contents(directory_path, ref=ref)

        except GithubException as e:
            logger.error(f"GitHub API error getting contents for {directory_path}: {e}")
            return f"Error: status code {e.status}, {e.message}"

        files = []
        content_queue = list(contents) if isinstance(contents, list) else [contents]

        while content_queue:
            file_content = content_queue.pop(0)

            if file_content.type == "dir":
                try:
                    directory_contents = repo.get_contents(file_content.path, ref=ref)
                    content_queue.extend(
                        directory_contents if isinstance(directory_contents, list) else [directory_contents]
                    )
                except GithubException as e:
                    logger.warning(f"Failed to access directory {file_content.path}: {e}")
                    continue
            else:
                files.append(file_content)

        return [file.path for file in files]

    def _read_file(self, file_path: str, branch: str, repo_name: str | None = None) -> str:
        """
        Read a file from specified branch.

        Args:
            file_path: Path to the file
            branch: Branch to read from
            repo_name: Optional repository name to override default

        Returns:
            File content as string, or error message if failed
        """
        try:
            repo = self.github_api.get_repo(repo_name) if repo_name else self.github_repo_instance
            file = repo.get_contents(file_path, ref=branch)
            return file.decoded_content.decode("utf-8")
        except Exception as e:
            logger.warning(f"Failed to read file {file_path} from branch {branch}: {e}")
            return f"File not found `{file_path}` on branch `{branch}`. Error: {str(e)}"

    def read_file(self, file_path: str, branch: str | None = None, repo_name: str | None = None) -> str:
        """
        Read a file from the active branch.

        Args:
            file_path: Path to the file
            branch: Branch to read from (defaults to active branch)
            repo_name: Optional repository name to override default

        Returns:
            File content as string
        """
        return self._read_file(file_path, branch if branch else self.active_branch, repo_name)

    def list_all_files(self, base_path: str = "") -> list[str]:
        """
        List all files in repository starting from base_path.

        Args:
            base_path: Starting directory (empty for root)

        Returns:
            List of file paths
        """
        return self._get_files(base_path, self.active_branch)

    def get_file_tree(self, max_files: int = 100) -> dict[str, Any]:
        """
        Get a structured representation of the repository file tree.

        Args:
            max_files: Maximum number of files to include

        Returns:
            Dictionary representing the file tree structure
        """
        try:
            all_files = self.list_all_files()

            if isinstance(all_files, str) and all_files.startswith("Error"):
                return {"error": all_files}

            # Limit files
            if len(all_files) > max_files:
                all_files = all_files[:max_files]

            # Build tree structure
            tree = {}
            for file_path in all_files:
                parts = file_path.split("/")
                current = tree

                for part in parts[:-1]:  # Navigate to parent directory
                    if part not in current:
                        current[part] = {}
                    current = current[part]

                # Add file
                current[parts[-1]] = file_path

            return {"tree": tree, "total_files": len(all_files), "truncated": len(self.list_all_files()) > max_files}

        except Exception as e:
            logger.error(f"Failed to build file tree: {e}")
            return {"error": str(e)}

    def check_connection(self) -> bool:
        """
        Check if the connection to GitHub repository is working.

        Returns:
            True if connection is successful, False otherwise
        """
        try:
            # Try to get repository info
            repo_info = self.get_repository_info()
            return bool(repo_info.get("name"))
        except Exception as e:
            logger.error(f"Connection check failed: {e}")
            return False

    def get_branches(self) -> list[str]:
        """
        Get list of available branches.

        Returns:
            List of branch names
        """
        try:
            branches = self.github_repo_instance.get_branches()
            return [branch.name for branch in branches]
        except Exception as e:
            logger.error(f"Failed to get branches: {e}")
            return []

    def set_active_branch(self, branch: str) -> bool:
        """
        Set the active branch for operations.

        Args:
            branch: Branch name to set as active

        Returns:
            True if successful, False otherwise
        """
        try:
            # Verify branch exists
            available_branches = self.get_branches()
            if branch not in available_branches:
                logger.error(f"Branch '{branch}' not found. Available branches: {available_branches}")
                return False

            self.active_branch = branch
            logger.info(f"Active branch set to: {branch}")
            return True

        except Exception as e:
            logger.error(f"Failed to set active branch to {branch}: {e}")
            return False


# Convenience function for quick client creation
def create_github_client(
    repository_url: str, access_token: str | None = None, branch: str = "main"
) -> StandaloneGitHubClient:
    """
    Create a GitHub client from repository URL and optional access token.

    Args:
        repository_url: Repository URL or owner/repo string
        access_token: Optional GitHub access token (ghp_* format) for private repos
        branch: Branch to use (default: main)

    Returns:
        Configured StandaloneGitHubClient instance

    Example:
        ```python
        # Public repository
        client = create_github_client("microsoft/vscode")

        # Private repository with token
        client = create_github_client("owner/private-repo", access_token="ghp_xxx")

        # Different branch
        client = create_github_client("owner/repo", branch="develop")
        ```
    """
    return StandaloneGitHubClient(repository=repository_url, access_token=access_token, branch=branch)
