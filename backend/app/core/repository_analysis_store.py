#!/usr/bin/python3

"""
Repository Analysis Store - Persistent storage for full repository analysis.

CRITICAL: Analysis is stored UNTRUNCATED to preserve full repository context.
This is essential for Ask tool to have complete understanding of the codebase,
just like in wiki generation prompts.
"""

import hashlib
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


class RepositoryAnalysisStore:
    """
    Persists FULL repository analysis alongside wiki artifacts.

    Unlike some summarization approaches, we store the COMPLETE analysis
    because truncation loses critical context needed for accurate Q&A.
    """

    def __init__(self, cache_dir: str | None = None):
        """
        Initialize the repository analysis store.

        Args:
            cache_dir: Directory for storing analysis files.
                       Defaults to ./data/cache
        """
        if cache_dir is None:
            cache_dir = "./data/cache"

        self.cache_dir = Path(cache_dir).expanduser()
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"RepositoryAnalysisStore initialized at {self.cache_dir}")

    def _analysis_key_str(
        self,
        repo_identifier: str,
        commit_hash: str | None = None,
        analysis_key_override: str | None = None,
    ) -> str:
        """Compute the stable key string used for cache addressing.

        Backward compatible with the legacy key format:
            f"{repo_identifier}:{commit_hash or 'HEAD'}"

        If analysis_key_override is provided, it is used verbatim.
        """
        if isinstance(analysis_key_override, str) and analysis_key_override.strip():
            return analysis_key_override.strip()

        return f"{repo_identifier}:{commit_hash or 'HEAD'}"

    def _get_cache_key_for_analysis_key(self, analysis_key: str) -> str:
        """Generate cache key (MD5) from an analysis key string."""
        return hashlib.md5(str(analysis_key).encode()).hexdigest()  # noqa: S324 — content fingerprint for cache key, not cryptographic use

    def _get_cache_key(self, repo_identifier: str, commit_hash: str | None = None) -> str:
        """Generate cache key for repository (legacy).

        Prefer using _analysis_key_str + _get_cache_key_for_analysis_key for new code.
        """
        analysis_key = self._analysis_key_str(repo_identifier=repo_identifier, commit_hash=commit_hash)
        return self._get_cache_key_for_analysis_key(analysis_key)

    def _get_analysis_path(self, cache_key: str) -> Path:
        """Get path to analysis JSON file"""
        return self.cache_dir / f"{cache_key}_analysis.json"

    def save_analysis(
        self,
        repo_identifier: str,
        analysis: dict[str, Any],
        commit_hash: str | None = None,
        metadata: dict[str, Any] | None = None,
        analysis_key_override: str | None = None,
    ) -> Path:
        """
        Save FULL repository analysis - NO TRUNCATION.

        The analysis should include:
        - repository_analysis: Full text analysis of the codebase
        - wiki_structure_spec: Complete structure specification
        - key_components: All identified components
        - architecture: Architecture description

        Args:
            repo_identifier: Repository path or URL
            analysis: Complete analysis dictionary (NOT truncated)
            commit_hash: Optional commit hash for versioning
            metadata: Optional additional metadata

        Returns:
            Path to saved analysis file
        """
        analysis_key = self._analysis_key_str(
            repo_identifier=repo_identifier,
            commit_hash=commit_hash,
            analysis_key_override=analysis_key_override,
        )
        cache_key = self._get_cache_key_for_analysis_key(analysis_key)
        path = self._get_analysis_path(cache_key)

        # Store COMPLETE analysis - do not truncate!
        data = {
            "repo_identifier": repo_identifier,
            "commit_hash": commit_hash,
            "analysis_key": analysis_key,
            "analysis": analysis,  # Full analysis, NO truncation
            "stored_at": datetime.now().isoformat(),
            "metadata": metadata or {},
        }

        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

        logger.info(f"Saved repository analysis to {path} (size: {path.stat().st_size} bytes)")
        return path

    def load_analysis(
        self,
        repo_identifier: str,
        commit_hash: str | None = None,
        analysis_key_override: str | None = None,
    ) -> dict[str, Any] | None:
        """
        Load FULL repository analysis if exists.

        Args:
            repo_identifier: Repository path or URL
            commit_hash: Optional commit hash for versioning

        Returns:
            Complete analysis dictionary or None if not found
        """
        analysis_key = self._analysis_key_str(
            repo_identifier=repo_identifier,
            commit_hash=commit_hash,
            analysis_key_override=analysis_key_override,
        )
        cache_key = self._get_cache_key_for_analysis_key(analysis_key)
        path = self._get_analysis_path(cache_key)

        if not path.exists():
            logger.debug(f"No analysis found at {path}")
            return None

        try:
            with open(path, encoding="utf-8") as f:
                data = json.load(f)
                # Return COMPLETE analysis - never truncate on read
                return data.get("analysis")
        except Exception as e:
            logger.warning(f"Failed to load analysis from {path}: {e}")
            return None

    def load_full_record(
        self,
        repo_identifier: str,
        commit_hash: str | None = None,
        analysis_key_override: str | None = None,
    ) -> dict[str, Any] | None:
        """
        Load full analysis record including metadata.

        Args:
            repo_identifier: Repository path or URL
            commit_hash: Optional commit hash for versioning

        Returns:
            Full record with analysis, metadata, timestamps, or None
        """
        analysis_key = self._analysis_key_str(
            repo_identifier=repo_identifier,
            commit_hash=commit_hash,
            analysis_key_override=analysis_key_override,
        )
        cache_key = self._get_cache_key_for_analysis_key(analysis_key)
        path = self._get_analysis_path(cache_key)

        if not path.exists():
            return None

        try:
            with open(path, encoding="utf-8") as f:
                return json.load(f)
        except Exception as e:
            logger.warning(f"Failed to load record from {path}: {e}")
            return None

    def get_analysis_for_prompt(
        self,
        repo_identifier: str,
        commit_hash: str | None = None,
        analysis_key_override: str | None = None,
    ) -> str:
        """
        Get repository analysis for LLM prompts.

        Returns the FULL LLM-generated analysis text.
        This is the comprehensive analysis created from README.md and
        directory structure using a bottom-up exploration approach.

        Supports both exact match and prefix matching for backward compatibility.

        Args:
            repo_identifier: Repository path or URL (e.g., "owner/repo:branch" or "owner/repo:branch:commit")
            commit_hash: Optional commit hash for versioning

        Returns:
            Analysis string (empty if not found)
        """
        # If a specific analysis key is provided, load exactly that record.
        if isinstance(analysis_key_override, str) and analysis_key_override.strip():
            analysis = self.load_analysis(
                repo_identifier=repo_identifier,
                commit_hash=commit_hash,
                analysis_key_override=analysis_key_override,
            )
            if not analysis:
                return ""

            if isinstance(analysis, str):
                return analysis
            if isinstance(analysis, dict):
                return analysis.get("repository_analysis", "")
            return ""

        # First try exact match
        analysis = self.load_analysis(repo_identifier, commit_hash)

        # If caller provided repo:branch (no commit), try to resolve to canonical commit-scoped id
        # via cache_index.json refs mapping.
        canonical_identifier = repo_identifier
        if not analysis and commit_hash is None:
            canonical_identifier = self._resolve_canonical_repo_identifier(repo_identifier)
            if canonical_identifier != repo_identifier:
                analysis = self.load_analysis(canonical_identifier, None)

        # If not found, try to find any analysis file and check repo_identifier prefix
        if not analysis:
            # Prefer deterministic selection: exact commit-scoped identifier if known,
            # otherwise newest matching prefix.
            analysis = self._find_analysis_by_prefix(canonical_identifier)

        if not analysis:
            return ""

        # The analysis is now stored as just the LLM-generated text string
        if isinstance(analysis, str):
            return analysis

        # Legacy support: if it's a dict, try to get repository_analysis field
        if isinstance(analysis, dict):
            return analysis.get("repository_analysis", "")

        return ""

    def _resolve_canonical_repo_identifier(self, repo_identifier: str) -> str:
        """Resolve repo:branch -> repo:branch:commit8 via cache_index.json refs.

        Best-effort only; returns input on any error.
        """
        try:
            from .repo_resolution import resolve_canonical_repo_identifier

            return resolve_canonical_repo_identifier(
                repo_identifier=repo_identifier,
                cache_dir=self.cache_dir,
                repositories_dir=self.cache_dir / "repositories",
            )
        except Exception:
            return repo_identifier

    def _find_analysis_by_prefix(self, repo_identifier: str) -> Any | None:
        """
        Find analysis by repo_identifier prefix match.

        Scans all analysis files to find one where stored repo_identifier
        starts with the given prefix. This handles cases where the query
        is "owner/repo:branch" but the stored key is "owner/repo:branch:commit".
        """
        try:
            prefix = repo_identifier.rstrip(":")
            exact_matches: list[tuple[float, Any]] = []
            prefix_matches: list[tuple[float, Any]] = []

            for analysis_file in self.cache_dir.glob("*_analysis.json"):
                try:
                    with open(analysis_file, encoding="utf-8") as f:
                        data = json.load(f)
                    stored_identifier = data.get("repo_identifier", "")
                    if not stored_identifier:
                        continue

                    mtime = analysis_file.stat().st_mtime

                    if stored_identifier == prefix:
                        exact_matches.append((mtime, data.get("analysis")))
                    elif stored_identifier.startswith(prefix + ":"):
                        prefix_matches.append((mtime, data.get("analysis")))
                except Exception:  # noqa: S112
                    continue

            # Prefer exact match; otherwise newest prefix match.
            if exact_matches:
                exact_matches.sort(key=lambda x: x[0], reverse=True)
                logger.info(f"Found analysis by exact identifier match: {prefix}")
                return exact_matches[0][1]

            if prefix_matches:
                prefix_matches.sort(key=lambda x: x[0], reverse=True)
                logger.info(f"Found analysis by newest prefix match for: {prefix}")
                return prefix_matches[0][1]

        except Exception as e:
            logger.warning(f"Error scanning for analysis files: {e}")

        return None

    def has_analysis(
        self,
        repo_identifier: str,
        commit_hash: str | None = None,
        analysis_key_override: str | None = None,
    ) -> bool:
        """
        Check if analysis exists for repository.

        Args:
            repo_identifier: Repository path or URL
            commit_hash: Optional commit hash for versioning

        Returns:
            True if analysis exists
        """
        analysis_key = self._analysis_key_str(
            repo_identifier=repo_identifier,
            commit_hash=commit_hash,
            analysis_key_override=analysis_key_override,
        )
        cache_key = self._get_cache_key_for_analysis_key(analysis_key)
        path = self._get_analysis_path(cache_key)
        return path.exists()

    def delete_analysis(
        self,
        repo_identifier: str,
        commit_hash: str | None = None,
        analysis_key_override: str | None = None,
    ) -> bool:
        """
        Delete stored analysis.

        Args:
            repo_identifier: Repository path or URL
            commit_hash: Optional commit hash for versioning

        Returns:
            True if deleted, False if not found
        """
        analysis_key = self._analysis_key_str(
            repo_identifier=repo_identifier,
            commit_hash=commit_hash,
            analysis_key_override=analysis_key_override,
        )
        cache_key = self._get_cache_key_for_analysis_key(analysis_key)
        path = self._get_analysis_path(cache_key)

        if path.exists():
            path.unlink()
            logger.info(f"Deleted analysis at {path}")
            return True
        return False


# =============================================================================
# STRUCTURED ANALYSIS UTILITIES
# =============================================================================
# Functions for working with structured JSON analysis output from
# STRUCTURED_REPO_ANALYSIS_PROMPT. These enable:
# - Query-optimized context extraction for Ask tool
# - Focused markdown rendering for LLM prompts
# - Capability matching for retrieval optimization
# =============================================================================


def parse_structured_analysis(analysis_text: str) -> dict[str, Any] | None:
    """
    Parse structured JSON analysis from stored text.

    Handles both new JSON format and legacy markdown format.

    Args:
        analysis_text: Raw analysis string (JSON or markdown)

    Returns:
        Parsed dict if JSON, None if markdown/invalid
    """
    if not analysis_text or not analysis_text.strip():
        return None

    text = analysis_text.strip()

    # Quick check: JSON starts with { or [
    if not (text.startswith("{") or text.startswith("[")):
        return None  # Likely markdown format

    try:
        parsed = json.loads(text)
        # Validate it has expected structure
        if isinstance(parsed, dict) and "capabilities" in parsed:
            return parsed
        return None
    except json.JSONDecodeError:
        # May have extra text before/after JSON, try to extract
        import re

        match = re.search(r"\{[\s\S]*\}", text)
        if match:
            try:
                parsed = json.loads(match.group())
                if isinstance(parsed, dict) and "capabilities" in parsed:
                    return parsed
            except json.JSONDecodeError:
                pass
        return None


def is_structured_analysis(analysis_text: str) -> bool:
    """Check if analysis is structured JSON format."""
    return parse_structured_analysis(analysis_text) is not None


def get_executive_summary(analysis_text: str) -> str:
    """
    Extract executive summary from analysis.

    Works with both structured JSON and legacy markdown.

    Args:
        analysis_text: Raw analysis string

    Returns:
        Executive summary string (~200-500 chars)
    """
    parsed = parse_structured_analysis(analysis_text)

    if parsed:
        # Structured JSON format
        summary = parsed.get("executive_summary", "")
        purpose = parsed.get("core_purpose", "")
        if purpose and purpose not in summary:
            summary = f"{summary} {purpose}".strip()
        return summary

    # Legacy markdown: extract first section
    if "**Executive Summary**" in analysis_text:
        import re

        match = re.search(r"\*\*Executive Summary\*\*\s*\n([\s\S]*?)(?=\n\*\*|\Z)", analysis_text)
        if match:
            return match.group(1).strip()[:1000]

    # Fallback: first 500 chars
    return analysis_text[:500].strip()


def match_capabilities_to_query(analysis_text: str, query: str, max_capabilities: int = 5) -> list[dict[str, Any]]:
    """
    Match user query to relevant capabilities using keyword overlap.

    Simple but effective: tokenize query, match against capability keywords.

    Args:
        analysis_text: Raw analysis string (must be structured JSON)
        query: User's question
        max_capabilities: Maximum capabilities to return

    Returns:
        List of matching capability dicts, sorted by relevance
    """
    parsed = parse_structured_analysis(analysis_text)
    if not parsed:
        return []

    capabilities = parsed.get("capabilities", [])
    if not capabilities:
        return []

    # Tokenize query (simple word-based)
    import re

    query_tokens = set(re.findall(r"\b\w+\b", query.lower()))

    # Score each capability
    scored = []
    for cap in capabilities:
        keywords = set(k.lower() for k in cap.get("keywords", []))
        name_tokens = set(re.findall(r"\b\w+\b", cap.get("name", "").lower()))
        desc_tokens = set(re.findall(r"\b\w+\b", cap.get("description", "").lower()))

        # Weighted scoring
        keyword_matches = len(query_tokens & keywords) * 3  # Keywords most important
        name_matches = len(query_tokens & name_tokens) * 2
        desc_matches = len(query_tokens & desc_tokens) * 1

        score = keyword_matches + name_matches + desc_matches

        if score > 0:
            scored.append((score, cap))

    # Sort by score descending
    scored.sort(key=lambda x: x[0], reverse=True)

    return [cap for _, cap in scored[:max_capabilities]]


def get_query_optimized_context(analysis_text: str, query: str, max_tokens: int = 2000) -> str:
    """
    Extract query-relevant context from structured analysis.

    Returns focused context for Ask tool instead of full 67K analysis.
    Includes executive summary + matching capabilities.

    Args:
        analysis_text: Raw analysis string
        query: User's question
        max_tokens: Approximate token budget (chars * 0.25)

    Returns:
        Focused markdown context for LLM
    """
    parsed = parse_structured_analysis(analysis_text)

    if not parsed:
        # Legacy markdown: just truncate
        char_limit = max_tokens * 4
        return analysis_text[:char_limit]

    char_budget = max_tokens * 4  # ~4 chars per token
    parts = []

    # Always include summary
    summary = parsed.get("executive_summary", "")
    purpose = parsed.get("core_purpose", "")
    tech_stack = parsed.get("tech_stack", [])

    parts.append(f"## Repository Overview\n{summary}")
    if purpose:
        parts.append(f"**Purpose:** {purpose}")
    if tech_stack:
        parts.append(f"**Tech Stack:** {', '.join(tech_stack)}")

    # Add matched capabilities
    matched = match_capabilities_to_query(analysis_text, query, max_capabilities=5)
    if matched:
        parts.append("\n## Relevant Capabilities")
        for cap in matched:
            name = cap.get("name", "Unknown")
            desc = cap.get("description", "")
            files = cap.get("files", [])
            files_str = ", ".join(files[:3]) if files else ""
            if files_str:
                parts.append(f"- **{name}** ({files_str}): {desc}")
            else:
                parts.append(f"- **{name}**: {desc}")

    # Add key patterns if space
    patterns = parsed.get("key_patterns", [])
    if patterns:
        parts.append(f"\n**Key Patterns:** {', '.join(patterns)}")

    # Combine and trim to budget
    result = "\n".join(parts)
    if len(result) > char_budget:
        result = result[:char_budget] + "..."

    return result


def render_structured_analysis_as_markdown(
    analysis_text: str, include_sections: list[str] | None = None, max_chars: int = 12000
) -> str:
    """
    Render structured JSON analysis as readable markdown for LLM.

    Converts JSON structure to prose-like markdown that LLMs understand well.

    Args:
        analysis_text: Raw analysis string (JSON or markdown)
        include_sections: Optional list of sections to include:
            - "summary": Executive summary + purpose
            - "capabilities": All capabilities
            - "workflows": All workflows
            - "patterns": Key patterns and entry points
            - "integrations": External integrations
            - "config": Configuration info
            - "quality": Quality notes
            If None, includes all sections.
        max_chars: Maximum output length

    Returns:
        Markdown string suitable for LLM prompts
    """
    parsed = parse_structured_analysis(analysis_text)

    if not parsed:
        # Already markdown, just return truncated
        return analysis_text[:max_chars]

    if include_sections is None:
        include_sections = ["summary", "capabilities", "workflows", "patterns", "integrations", "config"]

    parts = []

    # Summary section
    if "summary" in include_sections:
        summary = parsed.get("executive_summary", "")
        purpose = parsed.get("core_purpose", "")
        tech = parsed.get("tech_stack", [])

        parts.append("## Executive Summary")
        if summary:
            parts.append(summary)
        if purpose:
            parts.append(f"\n**Core Purpose:** {purpose}")
        if tech:
            parts.append(f"\n**Technology Stack:** {', '.join(tech)}")

    # Capabilities section
    if "capabilities" in include_sections:
        caps = parsed.get("capabilities", [])
        if caps:
            parts.append("\n## Capability Catalog")
            for cap in caps:
                name = cap.get("name", "")
                cat = cap.get("category", "")
                desc = cap.get("description", "")
                files = cap.get("files", [])

                header = f"### {name}"
                if cat:
                    header += f" ({cat})"
                parts.append(header)

                if desc:
                    parts.append(desc)
                if files:
                    parts.append(f"**Files:** {', '.join(files)}")

    # Workflows section
    if "workflows" in include_sections:
        workflows = parsed.get("workflows", [])
        if workflows:
            parts.append("\n## Key Workflows")
            for wf in workflows:
                name = wf.get("name", "")
                wf_type = wf.get("type", "")
                steps = wf.get("steps", [])

                parts.append(f"### {name}" + (f" ({wf_type})" if wf_type else ""))
                if steps:
                    for i, step in enumerate(steps, 1):
                        parts.append(f"{i}. {step}")

    # Patterns section
    if "patterns" in include_sections:
        patterns = parsed.get("key_patterns", [])
        entry_points = parsed.get("entry_points", [])

        if patterns or entry_points:
            parts.append("\n## Architecture Patterns")
            if patterns:
                parts.append(f"**Key Patterns:** {', '.join(patterns)}")
            if entry_points:
                parts.append(f"**Entry Points:** {', '.join(entry_points)}")

    # Integrations section
    if "integrations" in include_sections:
        integrations = parsed.get("external_integrations", [])
        if integrations:
            parts.append("\n## External Integrations")
            for integ in integrations:
                if isinstance(integ, dict):
                    name = integ.get("name", "")
                    int_type = integ.get("type", "")
                    files = integ.get("files", [])
                    parts.append(f"- **{name}** ({int_type}): {', '.join(files)}")
                else:
                    parts.append(f"- {integ}")

    # Config section
    if "config" in include_sections:
        config = parsed.get("configuration", {})
        if config:
            parts.append("\n## Configuration")
            files = config.get("files", [])
            settings = config.get("key_settings", [])
            desc = config.get("description", "")

            if desc:
                parts.append(desc)
            if files:
                parts.append(f"**Config Files:** {', '.join(files)}")
            if settings:
                parts.append(f"**Key Settings:** {', '.join(settings)}")

    # Quality section
    if "quality" in include_sections:
        quality = parsed.get("quality_notes", {})
        if quality:
            strengths = quality.get("strengths", [])
            opportunities = quality.get("opportunities", [])
            complexity = quality.get("complexity", "")

            if strengths or opportunities or complexity:
                parts.append("\n## Quality Notes")
                if complexity:
                    parts.append(f"**Complexity:** {complexity}")
                if strengths:
                    parts.append(f"**Strengths:** {', '.join(strengths)}")
                if opportunities:
                    parts.append(f"**Opportunities:** {', '.join(opportunities)}")

    result = "\n".join(parts)

    # Trim to max_chars
    if len(result) > max_chars:
        result = result[:max_chars] + "\n\n[... truncated ...]"

    return result


def get_capability_keywords(analysis_text: str) -> list[str]:
    """
    Extract all unique keywords from capabilities.

    Useful for building search indexes or query expansion.

    Args:
        analysis_text: Raw analysis string

    Returns:
        List of unique keywords
    """
    parsed = parse_structured_analysis(analysis_text)
    if not parsed:
        return []

    keywords = set()
    for cap in parsed.get("capabilities", []):
        keywords.update(k.lower() for k in cap.get("keywords", []))

    for wf in parsed.get("workflows", []):
        keywords.update(k.lower() for k in wf.get("keywords", []))

    for integ in parsed.get("external_integrations", []):
        if isinstance(integ, dict):
            keywords.update(k.lower() for k in integ.get("keywords", []))

    return sorted(keywords)


def get_files_for_query(analysis_text: str, query: str, max_files: int = 20) -> list[str]:
    """
    Get relevant file paths for a query based on capability matching.

    Can be used to optimize retrieval by filtering to relevant files.

    Args:
        analysis_text: Raw analysis string
        query: User's question
        max_files: Maximum files to return

    Returns:
        List of file paths most relevant to query
    """
    matched = match_capabilities_to_query(analysis_text, query, max_capabilities=10)

    files = []
    seen = set()
    for cap in matched:
        for f in cap.get("files", []):
            if f not in seen:
                files.append(f)
                seen.add(f)
            if len(files) >= max_files:
                return files

    return files
