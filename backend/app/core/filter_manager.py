"""
Filter Manager for GitHub-based indexing

Handles filtering logic based on repo.json configuration and API-level filters.
Supports directory exclusions, file exclusions, extension filtering, and size limits.
"""

import fnmatch
import json
import logging
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


class FilterManager:
    """Manages filtering for repository indexing with GitHub API"""

    def __init__(self, repo_config_path: str | None = None, api_filters: dict[str, Any] | None = None):
        """
        Initialize filter manager

        Args:
            repo_config_path: Path to repo.json configuration file
            api_filters: Additional filters from API level
        """
        self.repo_config = self._load_repo_config(repo_config_path)
        self.api_filters = api_filters or {}

        # Combined filter sets
        self.excluded_dirs = self._build_excluded_dirs()
        self.excluded_files = self._build_excluded_files()
        self.allowed_extensions = self._build_allowed_extensions()
        self.max_file_size_mb = self._get_max_file_size()

        logger.info(
            f"Filter manager initialized with {len(self.excluded_dirs)} excluded dirs, "
            f"{len(self.excluded_files)} excluded files, "
            f"{len(self.allowed_extensions) if self.allowed_extensions else 'all'} allowed extensions"
        )

    def _load_repo_config(self, config_path: str | None) -> dict[str, Any]:
        """Load repository configuration from repo.json"""
        if not config_path:
            # Use default path relative to this file
            default_path = Path(__file__).parent.parent.parent / "api" / "config" / "repo.json"
            config_path = str(default_path)

        try:
            with open(config_path, encoding="utf-8") as f:
                config = json.load(f)
            logger.debug(f"Loaded repo config from {config_path}")
            return config
        except (FileNotFoundError, json.JSONDecodeError) as e:
            logger.warning(f"Could not load repo config from {config_path}: {e}")
            return {}

    def _build_excluded_dirs(self) -> set[str]:
        """Build set of excluded directory patterns"""
        excluded = set()

        # From repo.json
        if "file_filters" in self.repo_config and "excluded_dirs" in self.repo_config["file_filters"]:
            repo_dirs = self.repo_config["file_filters"]["excluded_dirs"]
            for dir_pattern in repo_dirs:
                # Normalize patterns (remove leading ./ and trailing /)
                normalized = dir_pattern.strip("./").rstrip("/")
                excluded.add(normalized)

        # From API filters
        if "excluded_dirs" in self.api_filters:
            api_dirs = self.api_filters["excluded_dirs"]
            for dir_pattern in api_dirs:
                normalized = dir_pattern.strip("./").rstrip("/")
                excluded.add(normalized)

        # Add some common defaults if not already included
        defaults = {
            ".git",
            ".svn",
            ".hg",
            ".bzr",
            ".idea",
            ".vscode",  # IDE folders
            "node_modules",
            "bower_components",
            "jspm_packages",
            "__pycache__",
            ".pytest_cache",
            ".tox",
            "venv",
            ".venv",
            "env",
            "virtualenv",
            "build",
            "dist",
            "target",
            "out",
            "bin",
            ".output",
            "coverage",
            "htmlcov",
            ".nyc_output",
        }
        excluded.update(defaults)

        return excluded

    def _build_excluded_files(self) -> set[str]:
        """Build set of excluded file patterns"""
        excluded = set()

        # From repo.json
        if "file_filters" in self.repo_config and "excluded_files" in self.repo_config["file_filters"]:
            excluded.update(self.repo_config["file_filters"]["excluded_files"])

        # From API filters
        if "excluded_files" in self.api_filters:
            excluded.update(self.api_filters["excluded_files"])

        return excluded

    def _build_allowed_extensions(self) -> set[str] | None:
        """Build set of allowed file extensions"""
        # Check API filters first
        if "allowed_extensions" in self.api_filters:
            extensions = set()
            for ext in self.api_filters["allowed_extensions"]:
                if not ext.startswith("."):
                    ext = "." + ext
                extensions.add(ext.lower())
            return extensions

        # Check repo config
        if "file_filters" in self.repo_config and "allowed_extensions" in self.repo_config["file_filters"]:
            extensions = set()
            for ext in self.repo_config["file_filters"]["allowed_extensions"]:
                if not ext.startswith("."):
                    ext = "." + ext
                extensions.add(ext.lower())
            return extensions

        # If no specific extensions defined, allow common code and doc extensions
        return {
            # Code files
            ".py",
            ".js",
            ".ts",
            ".jsx",
            ".tsx",
            ".java",
            ".go",
            ".cs",
            ".cpp",
            ".cc",
            ".cxx",
            ".c++",
            ".c",
            ".h",
            ".hpp",
            ".hh",
            ".hxx",  # C/C++ complete
            ".rb",
            ".php",
            ".rs",
            ".kt",
            ".scala",
            ".swift",
            ".dart",
            ".r",
            ".m",
            ".sh",
            ".bash",
            ".zsh",
            ".ps1",
            ".bat",
            ".cmd",
            # Config files
            ".json",
            ".yaml",
            ".yml",
            ".toml",
            ".xml",
            ".ini",
            ".cfg",
            ".conf",
            # Build files
            ".gradle",
            ".kts",
            # Schema / IDL files
            ".wsdl",
            ".xsd",
            ".proto",
            # Infrastructure as Code
            ".tf",
            ".tfvars",
            ".hcl",
            # Module definition files
            ".mod",
            # Documentation
            ".md",
            ".rst",
            ".txt",
            ".adoc",
            ".org",
            # Web files
            ".html",
            ".htm",
            ".css",
            ".scss",
            ".sass",
            ".less",
            # Data files
            ".sql",
            ".csv",
            ".tsv",
            # Other
            ".dockerfile",
            ".gitignore",
            ".gitattributes",
        }

    def _get_max_file_size(self) -> float:
        """Get maximum file size in MB"""
        # Check API filters first
        if "max_file_size_mb" in self.api_filters:
            return float(self.api_filters["max_file_size_mb"])

        # Check repo config
        if "repository" in self.repo_config and "max_size_mb" in self.repo_config["repository"]:
            return float(self.repo_config["repository"]["max_size_mb"])

        # Default to 10MB per file
        return 10.0

    def should_process_directory(self, dir_path: str) -> bool:
        """Check if a directory should be processed"""
        if not dir_path:
            return True

        # Normalize path
        normalized_path = dir_path.strip("./").rstrip("/")

        # Check exact matches and patterns
        for excluded_pattern in self.excluded_dirs:
            if (
                normalized_path == excluded_pattern
                or normalized_path.startswith(excluded_pattern + "/")
                or fnmatch.fnmatch(normalized_path, excluded_pattern)
            ):
                logger.debug(f"Directory {dir_path} excluded by pattern {excluded_pattern}")
                return False

        # Check each path component
        path_parts = normalized_path.split("/")
        for part in path_parts:
            if part in self.excluded_dirs:
                logger.debug(f"Directory {dir_path} excluded by component {part}")
                return False

        return True

    def should_process_file(self, file_path: str, file_size_bytes: int | None = None) -> bool:
        """Check if a file should be processed"""
        file_name = Path(file_path).name
        file_ext = Path(file_path).suffix.lower()

        # Check file size limit
        if file_size_bytes is not None:
            file_size_mb = file_size_bytes / (1024 * 1024)
            if file_size_mb > self.max_file_size_mb:
                logger.debug(
                    f"File {file_path} excluded by size limit ({file_size_mb:.2f}MB > {self.max_file_size_mb}MB)"
                )
                return False

        # Check excluded file patterns
        for excluded_pattern in self.excluded_files:
            if (
                file_name == excluded_pattern
                or fnmatch.fnmatch(file_name, excluded_pattern)
                or fnmatch.fnmatch(file_path, excluded_pattern)
            ):
                logger.debug(f"File {file_path} excluded by pattern {excluded_pattern}")
                return False

        # Check allowed extensions
        if self.allowed_extensions is not None:
            if file_ext not in self.allowed_extensions:
                logger.debug(f"File {file_path} excluded by extension filter ({file_ext} not in allowed)")
                return False

        return True

    def get_file_language(self, file_path: str) -> str | None:
        """Determine programming language from file extension"""
        ext = Path(file_path).suffix.lower()

        language_map = {
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
            # Build files
            ".gradle": "groovy",
            ".kts": "kotlin_script",
            # Schema / IDL files
            ".wsdl": "xml",
            ".xsd": "xml",
            ".proto": "protobuf",
            # Infrastructure as Code
            ".tf": "terraform",
            ".tfvars": "terraform",
            ".hcl": "hcl",
            # Module definition files
            ".mod": "gomod",
            # Shell / batch
            ".bat": "batch",
            ".cmd": "batch",
            ".cfg": "config",
            ".conf": "config",
            ".psm1": "powershell",
        }

        return language_map.get(ext)

    def is_code_file(self, file_path: str) -> bool:
        """Check if file is a code file that should be parsed with tree-sitter"""
        return self.get_file_language(file_path) is not None

    def is_text_file(self, file_path: str) -> bool:
        """Check if file is a text/documentation file"""
        ext = Path(file_path).suffix.lower()
        name = Path(file_path).stem.upper()

        text_extensions = {".md", ".txt", ".rst", ".org", ".adoc", ".markdown"}
        text_names = {"README", "CHANGELOG", "LICENSE", "CONTRIBUTING", "AUTHORS"}

        return ext in text_extensions or name in text_names

    def get_filter_summary(self) -> dict[str, Any]:
        """Get summary of active filters"""
        return {
            "excluded_dirs": list(self.excluded_dirs),
            "excluded_files": list(self.excluded_files),
            "allowed_extensions": list(self.allowed_extensions) if self.allowed_extensions else None,
            "max_file_size_mb": self.max_file_size_mb,
            "has_repo_config": bool(self.repo_config),
            "has_api_filters": bool(self.api_filters),
        }
