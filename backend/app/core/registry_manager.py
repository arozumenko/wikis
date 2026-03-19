#!/usr/bin/python3

"""
Wiki Registry Manager

Manages a global registry of all generated wikis stored in a shared bucket.
The registry enables Context7-style wiki discovery and resolution.

Registry location: {bucket}/_registry/wikis.json
"""

import json
import logging
import re
from datetime import datetime

logger = logging.getLogger(__name__)

REGISTRY_PATH = "_registry/wikis.json"
SCHEMA_VERSION = 1


def normalize_wiki_id(repo_identifier: str) -> str:
    """
    Convert repo identifier to wiki folder name.

    Format: {owner}--{repo}--{branch} (all lowercase)

    Examples:
        owner/repo:main:abc123 → owner--repo--main
        Owner/Repo:feature/test:abc123 → owner--repo--feature-test
        owner/repo → owner--repo--main
    """
    # Parse repo_identifier: owner/repo:branch:commit or owner/repo:branch or owner/repo
    parts = repo_identifier.split(":")
    repo_part = parts[0]  # owner/repo
    branch = parts[1] if len(parts) > 1 else "main"

    # Normalize repo: / → --, lowercase
    normalized_repo = repo_part.lower().replace("/", "--")

    # Normalize branch: lowercase, / → -, remove special chars
    normalized_branch = branch.lower()
    normalized_branch = re.sub(r"[^a-z0-9-]", "-", normalized_branch)
    normalized_branch = re.sub(r"-+", "-", normalized_branch)  # collapse multiple -
    normalized_branch = normalized_branch.strip("-")

    return f"{normalized_repo}--{normalized_branch}"


def parse_wiki_id(wiki_id: str) -> dict[str, str]:
    """
    Parse wiki_id back to components.

    Returns:
        {"owner": "...", "repo": "...", "branch": "..."}

    Note: wiki_id format is always owner--repo--branch (3 parts minimum).
    If only 2 parts, branch is unknown (return empty string, caller should handle).
    """
    # Split by -- but handle owner--repo--branch format
    parts = wiki_id.split("--")
    if len(parts) >= 3:
        owner = parts[0]
        repo = parts[1]
        branch = "--".join(parts[2:])  # branch might have -- in it
        return {"owner": owner, "repo": repo, "branch": branch}
    elif len(parts) == 2:
        # No branch in wiki_id - shouldn't happen with normalize_wiki_id
        return {"owner": parts[0], "repo": parts[1], "branch": ""}
    else:
        return {"owner": "", "repo": wiki_id, "branch": ""}


class WikiRegistryManager:
    """
    Manages the global wiki registry in a bucket.

    The registry tracks all generated wikis and their metadata,
    enabling discovery and resolution from natural language queries.
    """

    def __init__(self, artifacts_client, bucket_name: str = "wiki_artifacts"):
        """
        Initialize registry manager.

        Args:
            artifacts_client: Client for bucket operations (create_artifact, download_artifact, etc.)
            bucket_name: Name of the bucket containing wiki artifacts
        """
        self.client = artifacts_client
        self.bucket = bucket_name
        self._cache: dict | None = None
        self._cache_time: datetime | None = None
        self._cache_ttl_seconds = 60  # Cache registry for 60 seconds

    def _is_cache_valid(self) -> bool:
        """Check if cached registry is still valid."""
        if self._cache is None or self._cache_time is None:
            return False
        elapsed = (datetime.utcnow() - self._cache_time).total_seconds()
        return elapsed < self._cache_ttl_seconds

    def load_registry(self, force_refresh: bool = False) -> dict:
        """
        Load registry from bucket, create if missing.

        Args:
            force_refresh: Bypass cache and reload from bucket

        Returns:
            Registry dict with schema_version, wikis list, updated_at
        """
        if not force_refresh and self._is_cache_valid():
            return self._cache

        try:
            data = self.client.download_artifact(self.bucket, REGISTRY_PATH)
            if isinstance(data, bytes):
                data = data.decode("utf-8")
            self._cache = json.loads(data)
            self._cache_time = datetime.utcnow()
            logger.info(f"Loaded wiki registry with {len(self._cache.get('wikis', []))} wikis")
        except Exception as e:
            # Registry doesn't exist yet - create empty one
            logger.info(f"Registry not found, creating new one: {e}")
            self._cache = {
                "schema_version": SCHEMA_VERSION,
                "wikis": [],
                "updated_at": datetime.utcnow().isoformat() + "Z",
            }
            self._cache_time = datetime.utcnow()

        return self._cache

    def save_registry(self, registry: dict) -> None:
        """
        Save registry to bucket.

        Args:
            registry: Registry dict to save
        """
        registry["updated_at"] = datetime.utcnow().isoformat() + "Z"

        try:
            self.client.create_artifact(self.bucket, REGISTRY_PATH, json.dumps(registry, indent=2))
            self._cache = registry
            self._cache_time = datetime.utcnow()
            logger.info(f"Saved wiki registry with {len(registry.get('wikis', []))} wikis")
        except Exception as e:
            logger.error(f"Failed to save registry: {e}")
            raise

    def register_wiki(
        self,
        wiki_id: str,
        repo: str,
        branch: str,
        provider: str = "github",
        host: str = "github.com",
        display_name: str | None = None,
        description: str | None = None,
        topics: list[str] | None = None,
        commit_hash: str | None = None,
        stats: dict[str, int] | None = None,
    ) -> dict:
        """
        Register a wiki in the global registry.

        Args:
            wiki_id: Normalized wiki folder name (e.g., "owner--repo--main")
            repo: Repository in owner/repo format
            branch: Branch name
            provider: Git provider (github, gitlab, bitbucket, ado)
            host: Git host (e.g., github.com)
            display_name: Human-readable name
            description: Wiki/repo description
            topics: List of topics/tags
            commit_hash: Git commit hash
            stats: Stats dict (files_indexed, symbols_in_graph, etc.)

        Returns:
            The created/updated wiki entry
        """
        registry = self.load_registry(force_refresh=True)

        now = datetime.utcnow().isoformat() + "Z"

        # Find existing entry
        existing_idx = next((i for i, w in enumerate(registry["wikis"]) if w["id"] == wiki_id), None)

        wiki_entry = {
            "id": wiki_id,
            "repo": repo,
            "branch": branch,
            "provider": provider,
            "host": host,
            "display_name": display_name or f"{repo} ({branch})",
            "description": description or "",
            "topics": topics or [],
            "folder_path": f"{wiki_id}/",
            "commit_hash": commit_hash,
            "artifact_status": {
                "graph": True,
                "vectorstore": True,
                "bm25": True,
                "wiki_pages": True,
            },
            "stats": stats or {},
            "updated_at": now,
        }

        if existing_idx is not None:
            # Update existing - preserve created_at
            wiki_entry["created_at"] = registry["wikis"][existing_idx].get("created_at", now)
            registry["wikis"][existing_idx] = wiki_entry
            logger.info(f"Updated wiki in registry: {wiki_id}")
        else:
            # New entry
            wiki_entry["created_at"] = now
            registry["wikis"].append(wiki_entry)
            logger.info(f"Added new wiki to registry: {wiki_id}")

        self.save_registry(registry)
        return wiki_entry

    def unregister_wiki(self, wiki_id: str) -> bool:
        """
        Remove a wiki from the registry.

        Args:
            wiki_id: Wiki ID to remove

        Returns:
            True if wiki was found and removed, False otherwise
        """
        registry = self.load_registry(force_refresh=True)

        original_count = len(registry["wikis"])
        registry["wikis"] = [w for w in registry["wikis"] if w["id"] != wiki_id]

        if len(registry["wikis"]) < original_count:
            self.save_registry(registry)
            logger.info(f"Removed wiki from registry: {wiki_id}")
            return True

        return False

    def delete_wiki_with_artifacts(self, wiki_id: str) -> dict:
        """
        Delete a wiki and all its artifacts from the bucket.

        This method:
        1. Lists all artifacts with the wiki_id prefix
        2. Deletes each artifact
        3. Removes the wiki from the registry

        Args:
            wiki_id: Wiki ID to delete

        Returns:
            Dict with deletion results: {deleted_count, errors, registry_removed}
        """
        result = {
            "wiki_id": wiki_id,
            "deleted_count": 0,
            "errors": [],
            "registry_removed": False,
        }

        try:
            # List all artifacts in the bucket
            all_artifacts = self.client.list_artifacts(self.bucket)

            # Filter artifacts that start with wiki_id prefix
            wiki_prefix = f"{wiki_id}/"
            wiki_artifacts = (
                [a for a in (all_artifacts or []) if isinstance(a, dict) and a.get("name", "").startswith(wiki_prefix)]
                if all_artifacts
                else []
            )

            # Also handle direct name comparison if list returns different format
            if not wiki_artifacts and all_artifacts:
                wiki_artifacts = [a for a in all_artifacts if isinstance(a, str) and a.startswith(wiki_prefix)]

            logger.info(f"Found {len(wiki_artifacts)} artifacts to delete for wiki {wiki_id}")

            # Delete each artifact
            for artifact in wiki_artifacts:
                try:
                    artifact_name = artifact.get("name") if isinstance(artifact, dict) else artifact
                    self.client.delete_artifact(self.bucket, artifact_name)
                    result["deleted_count"] += 1
                except Exception as del_err:
                    result["errors"].append(f"Failed to delete {artifact_name}: {str(del_err)}")

            # Remove from registry
            result["registry_removed"] = self.unregister_wiki(wiki_id)

        except Exception as e:
            result["errors"].append(f"Error during deletion: {str(e)}")
            logger.error(f"Failed to delete wiki {wiki_id}: {e}")

        return result

    def get_wiki(self, wiki_id: str) -> dict | None:
        """
        Get a wiki entry by ID.

        Args:
            wiki_id: Wiki ID to find

        Returns:
            Wiki entry dict or None if not found
        """
        registry = self.load_registry()
        return next((w for w in registry["wikis"] if w["id"] == wiki_id), None)

    def list_wikis(self, filter_text: str | None = None) -> list[dict]:
        """
        List all wikis, optionally filtered.

        Args:
            filter_text: Optional text to filter by (matches repo, display_name, topics)

        Returns:
            List of wiki entries
        """
        registry = self.load_registry()
        wikis = registry.get("wikis", [])

        if not filter_text:
            return wikis

        filter_lower = filter_text.lower()

        def matches(wiki: dict) -> bool:
            if filter_lower in wiki.get("repo", "").lower():
                return True
            if filter_lower in wiki.get("display_name", "").lower():
                return True
            if filter_lower in wiki.get("description", "").lower():
                return True
            if any(filter_lower in t.lower() for t in wiki.get("topics", [])):
                return True
            return False

        return [w for w in wikis if matches(w)]

    def find_wiki_by_repo(self, repo: str, branch: str | None = None) -> dict | None:
        """
        Find a wiki by repository name and optional branch.

        Args:
            repo: Repository in owner/repo format
            branch: Optional branch name (if None, returns first match)

        Returns:
            Wiki entry or None
        """
        registry = self.load_registry()

        for wiki in registry.get("wikis", []):
            if wiki.get("repo") == repo:
                if branch is None or wiki.get("branch") == branch:
                    return wiki

        return None

    def get_registry_for_resolution(self) -> str:
        """
        Get a formatted registry list for LLM-based resolution.

        Returns:
            Formatted string with wiki list for prompting
        """
        registry = self.load_registry()
        wikis = registry.get("wikis", [])

        if not wikis:
            return "No wikis available."

        lines = []
        for wiki in wikis:
            topics_str = ", ".join(wiki.get("topics", [])) if wiki.get("topics") else "none"
            lines.append(
                f"- ID: {wiki['id']}\n"
                f"  Repo: {wiki['repo']} (branch: {wiki['branch']})\n"
                f"  Description: {wiki.get('description', 'N/A')}\n"
                f"  Topics: {topics_str}"
            )

        return "\n\n".join(lines)

    def get_wikis_for_resolution(self) -> list[dict]:
        """
        Get wikis as a list of dicts for LLM resolution.

        Returns:
            List of dicts with wiki_id, wiki_title, description, repo, branch
        """
        registry = self.load_registry()
        wikis = registry.get("wikis", [])

        if not wikis:
            return []

        result = []
        for wiki in wikis:
            result.append(
                {
                    "wiki_id": wiki.get("id", ""),
                    "wiki_title": wiki.get("display_name", ""),
                    "description": wiki.get("description", ""),
                    "repo": wiki.get("repo", ""),
                    "branch": wiki.get("branch", "main"),
                }
            )

        return result
