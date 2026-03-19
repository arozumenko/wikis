"""
Artifact Export Utilities

This module provides utilities for exporting and storing wiki artifacts
via the storage system.

Bucket structure (Context7-style):
    wiki_artifacts/
        ├── _registry/wikis.json           # Global registry of all wikis
        └── {wiki_id}/                     # Folder per wiki
            ├── wiki_manifest_{version}.json   # Versioned manifests
            ├── wiki_pages/                    # Wiki markdown pages
            │   ├── section_name/
            │   │   └── page_name.md
            │   └── README.md
            ├── analysis/                      # Structure and analysis
            │   └── wiki_structure.json
            ├── {hash}.faiss                   # Flat cache artifacts
            ├── {hash}.docstore.bin
            ├── {hash}.doc_index.json
            ├── {hash}.bm25.sqlite
            └── combined.code_graph.gz
"""

import json
import logging
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Any

from .state.wiki_state import WikiPage, WikiStructureSpec

logger = logging.getLogger(__name__)


# Import normalize_wiki_id from registry_manager for consistency
# Note: For standalone use, the function also handles owner/repo/branch parameters
def normalize_wiki_id(
    owner: str | None = None, repo: str | None = None, branch: str | None = None, repository: str | None = None
) -> str:
    """
    Normalize repository identifier to wiki_id format.

    Format: {owner}--{repo}--{branch}
    Example: microsoft--vscode--main

    Args:
        owner: Repository owner (optional if repository is provided)
        repo: Repository name (optional if repository is provided)
        branch: Branch name (defaults to 'main')
        repository: Full repository path like "owner/repo" (alternative to owner+repo)

    Returns:
        Normalized wiki_id string
    """
    # Parse repository if provided
    if repository and "/" in repository:
        parts = repository.split("/")
        if len(parts) >= 2:
            owner = parts[0]
            repo = parts[1]

    if not owner or not repo:
        raise ValueError("Could not determine owner and repo from provided arguments")

    branch = branch or "main"

    # Normalize: lowercase, replace special chars with dashes
    owner = owner.lower().replace("_", "-").replace(".", "-")
    repo = repo.lower().replace("_", "-").replace(".", "-")
    branch = branch.lower().replace("/", "-").replace("_", "-")

    return f"{owner}--{repo}--{branch}"


class ArtifactExporter:
    """Handles export and storage of wiki artifacts with folder structure."""

    def __init__(self, wiki_id: str = None):
        """
        Initialize artifact exporter.

        Args:
            wiki_id: Optional wiki identifier for folder-based storage.
                     If provided, all artifacts will be prefixed with {wiki_id}/
                     Format: {owner}--{repo}--{branch}
        """
        self.export_formats = ["json", "markdown"]
        self.wiki_id = wiki_id

    def _prefix_name(self, subfolder: str, name: str) -> str:
        """
        Add wiki_id folder prefix to artifact name.

        Args:
            subfolder: Subfolder within wiki (e.g., 'wiki_pages', 'analysis')
            name: Base artifact name

        Returns:
            Prefixed name like "{wiki_id}/{subfolder}/{name}" or just "{subfolder}/{name}"
        """
        if self.wiki_id:
            return f"{self.wiki_id}/{subfolder}/{name}"
        return f"{subfolder}/{name}"

    def export_wiki_artifacts(
        self,
        wiki_structure: WikiStructureSpec,
        wiki_pages: list[WikiPage],
        repo_metadata: dict[str, Any],
        export_formats: list[str] | None = None,
    ) -> list:
        """
        Export wiki artifacts in multiple formats.

        Args:
            wiki_structure: The wiki structure
            wiki_pages: List of wiki pages
            repo_metadata: Repository metadata
            export_formats: List of formats to export (default: ["json", "markdown"])

        Returns:
            Dictionary mapping format names to main artifact URLs.
            For markdown format, individual files are also stored as separate artifacts for UI rendering.
        """
        if export_formats is None:
            export_formats = ["json", "markdown"]

        artifacts = []

        try:
            with tempfile.TemporaryDirectory() as temp_dir:
                temp_path = Path(temp_dir)

                # Export in each requested format
                for format_name in export_formats:
                    try:
                        if format_name == "json":
                            jsons_artifacts = self._export_json_format(wiki_structure, wiki_pages)
                            artifacts.extend(jsons_artifacts)
                        elif format_name == "markdown":
                            markdown_artifacts = self._export_markdown_format(
                                wiki_structure, wiki_pages, repo_metadata, temp_path
                            )
                            artifacts.extend(markdown_artifacts)
                    except Exception as e:
                        logger.error(f"Failed to export wiki in {format_name} format: {str(e)}")
                        return []
                return artifacts

        except Exception as e:
            logger.error(f"Error exporting wiki artifacts: {str(e)}")
            return []

    def _export_json_format(self, wiki_structure: WikiStructureSpec, wiki_pages: list[WikiPage]) -> list:
        """Export wiki in simple JSON format optimized for Git repository structure."""
        result_wiki_structure = []
        # Create a mapping of page IDs to pages for easier lookup
        pages_by_id = {page.page_id: page for page in wiki_pages}

        # Use the WikiStructureSpec sections if available, otherwise fallback grouping
        if wiki_structure.sections and len(wiki_structure.sections) > 0:
            # Use the actual sections from wiki_structure
            sections_data = []
            for section_idx, section in enumerate(wiki_structure.sections):
                section_pages = []
                for page_idx, page_spec in enumerate(section.pages):
                    # Create expected page_id based on structure indices
                    expected_page_id = f"{section_idx}#{page_idx}"

                    # Find matching page by the actual page_id format
                    matching_page = pages_by_id.get(expected_page_id)

                    # Fallback: try to find by page_name if direct ID match fails
                    if not matching_page:
                        matching_page = next((p for p in wiki_pages if p.title == page_spec.page_name), None)

                    if matching_page:
                        section_pages.append(
                            {
                                "page_name": f"{self._create_safe_filename(matching_page.title)}.md",
                                "page_content": matching_page.content,
                            }
                        )

                if section_pages:  # Only add sections that have pages
                    sections_data.append({"section_name": section.section_name, "pages": section_pages})
        else:
            # Fallback: Group pages by section index from page_id format (e.g., "0#1" -> section 0)
            pages_by_section = {}
            for page in wiki_pages:
                section_name = "General"

                # Extract section from page_id format like "0#1" -> section 0
                if "#" in page.page_id:
                    try:
                        section_idx = int(page.page_id.split("#")[0])
                        # Try to get section name from wiki_structure if available
                        if wiki_structure.sections and section_idx < len(wiki_structure.sections):
                            section_name = wiki_structure.sections[section_idx].section_name
                        else:
                            section_name = f"Section {section_idx}"
                    except (ValueError, IndexError):
                        section_name = "General"

                if section_name not in pages_by_section:
                    pages_by_section[section_name] = []

                pages_by_section[section_name].append(
                    {"page_name": f"{self._create_safe_filename(page.title)}.md", "page_content": page.content}
                )

            sections_data = [
                {"section_name": section_name, "pages": pages} for section_name, pages in pages_by_section.items()
            ]

        # Create simple, clean JSON structure
        wiki_data = {"wiki_title": wiki_structure.wiki_title, "sections": sections_data}

        # Use folder prefix for Context7-style organization
        base_name = f"wiki_structure_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        artifact_name = self._prefix_name("analysis", base_name)
        result_wiki_structure.append(
            {"name": artifact_name, "type": "application/json", "data": json.dumps(wiki_data, indent=4)}
        )
        return result_wiki_structure

    def _export_markdown_format(
        self,
        wiki_structure: WikiStructureSpec,
        wiki_pages: list[WikiPage],
        repo_metadata: dict[str, Any],
        temp_path: Path,
    ) -> list:
        """Export wiki in markdown format with proper directory structure."""

        markdown_dir = temp_path / "wiki_markdown"
        markdown_dir.mkdir(exist_ok=True)
        markdown_artifacts = []

        # Create directory structure based on sections
        if wiki_structure.sections and len(wiki_structure.sections) > 0:
            # Create a mapping of page IDs to pages for easier lookup
            pages_by_id = {page.page_id: page for page in wiki_pages}

            # Use the actual sections from wiki_structure
            for section_idx, section in enumerate(wiki_structure.sections):
                section_dir = markdown_dir / self._create_safe_filename(section.section_name)
                section_dir.mkdir(exist_ok=True)

                for page_idx, page_spec in enumerate(section.pages):
                    # Create expected page_id based on structure indices
                    expected_page_id = f"{section_idx}#{page_idx}"

                    # Find matching page by the actual page_id format
                    matching_page = pages_by_id.get(expected_page_id)

                    # Fallback: try to find by page_name if direct ID match fails
                    if not matching_page:
                        matching_page = next((p for p in wiki_pages if p.title == page_spec.page_name), None)

                    if matching_page:
                        self._create_markdown_page_in_section(matching_page, section_dir)
        else:
            # Fallback: Group pages by section index from page_id format (e.g., "0#1" -> section 0)
            pages_by_section = {}
            for page in wiki_pages:
                section_name = "general"

                # Extract section from page_id format like "0#1" -> section 0
                if "#" in page.page_id:
                    try:
                        section_idx = int(page.page_id.split("#")[0])
                        # Try to get section name from wiki_structure if available
                        if wiki_structure.sections and section_idx < len(wiki_structure.sections):
                            section_name = wiki_structure.sections[section_idx].section_name
                        else:
                            section_name = f"section_{section_idx}"
                    except (ValueError, IndexError):
                        section_name = "general"

                if section_name not in pages_by_section:
                    pages_by_section[section_name] = []
                pages_by_section[section_name].append(page)

            # Create directories and files
            for section_name, pages in pages_by_section.items():
                section_dir = markdown_dir / self._create_safe_filename(section_name)
                section_dir.mkdir(exist_ok=True)

                for page in pages:
                    self._create_markdown_page_in_section(page, section_dir)

        # Create main README
        self._create_markdown_index(wiki_structure, wiki_pages, markdown_dir)

        # Store individual markdown files as artifacts for UI debugging/rendering
        for file_path in markdown_dir.rglob("*"):
            if file_path.is_file() and file_path.suffix == ".md":
                # Read markdown file
                with open(file_path, "rb") as f:
                    file_data = f.read()

                # Create artifact name preserving directory structure
                # Use folder structure: {wiki_id}/wiki_pages/{section}/{page}.md
                relative_path = file_path.relative_to(markdown_dir)
                # Keep the directory structure for Context7-style organization
                artifact_name = self._prefix_name("wiki_pages", str(relative_path))

                markdown_artifacts.append(
                    {"name": artifact_name, "type": "text/markdown", "data": file_data.decode("utf-8")}
                )

        # # Create archive with all files
        # archive_path = temp_path / "wiki_markdown.zip"
        # with zipfile.ZipFile(archive_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
        #     for file_path in markdown_dir.rglob('*'):
        #         if file_path.is_file():
        #             zipf.write(file_path, file_path.relative_to(markdown_dir))
        #
        # # Read archive data and store as artifact
        # with open(archive_path, 'rb') as f:
        #     artifact_data = f.read()
        #
        # zip_artifact_name = f"wiki_markdown_{datetime.now().strftime('%Y%m%d_%H%M%S')}.zip"
        # self.client.create_artifact(
        #     bucket_name=self.bucket_name,
        #     artifact_name=zip_artifact_name,
        #     artifact_data=artifact_data
        # )
        #
        # logger.info(f"Stored markdown archive: {zip_artifact_name}")
        # logger.info(f"Stored {len(individual_artifacts)} individual markdown files for UI rendering")
        return markdown_artifacts

    def _create_markdown_index(self, wiki_structure: WikiStructureSpec, wiki_pages: list[WikiPage], output_dir: Path):
        """Create markdown index page showing directory structure."""

        content = f"# {wiki_structure.wiki_title}\n\n"
        content += f"{getattr(wiki_structure, 'overview', 'Generated wiki documentation')}\n\n"

        content += "## Wiki Structure\n\n"

        # Show directory structure based on sections
        if wiki_structure.sections and len(wiki_structure.sections) > 0:
            # Create a mapping of page IDs to pages for easier lookup
            pages_by_id = {page.page_id: page for page in wiki_pages}

            for section_idx, section in enumerate(wiki_structure.sections):
                section_name = self._create_safe_filename(section.section_name)
                content += f"### {section.section_name}/\n\n"

                for page_idx, page_spec in enumerate(section.pages):
                    # Create expected page_id based on structure indices
                    expected_page_id = f"{section_idx}#{page_idx}"

                    # Find matching page by the actual page_id format
                    matching_page = pages_by_id.get(expected_page_id)

                    # Fallback: try to find by page_name if direct ID match fails
                    if not matching_page:
                        matching_page = next((p for p in wiki_pages if p.title == page_spec.page_name), None)

                    if matching_page:
                        filename = self._create_safe_filename(matching_page.title)
                        content += f"- [{matching_page.title}]({section_name}/{filename}.md)\n"
                content += "\n"
        else:
            # Fallback: Group pages by section index from page_id format (e.g., "0#1" -> section 0)
            pages_by_section = {}
            for page in wiki_pages:
                section_name = "general"

                # Extract section from page_id format like "0#1" -> section 0
                if "#" in page.page_id:
                    try:
                        section_idx = int(page.page_id.split("#")[0])
                        # Try to get section name from wiki_structure if available
                        if wiki_structure.sections and section_idx < len(wiki_structure.sections):
                            section_name = wiki_structure.sections[section_idx].section_name
                        else:
                            section_name = f"section_{section_idx}"
                    except (ValueError, IndexError):
                        section_name = "general"

                if section_name not in pages_by_section:
                    pages_by_section[section_name] = []
                pages_by_section[section_name].append(page)

            for section_name, pages in pages_by_section.items():
                safe_section = self._create_safe_filename(section_name)
                content += f"### {section_name.title()}/\n\n"
                for page in pages:
                    filename = self._create_safe_filename(page.title)
                    content += f"- [{page.title}]({safe_section}/{filename}.md)\n"
                content += "\n"

        content += "\n---\n\n*Generated by Wikis Wiki Toolkit*\n"

        with open(output_dir / "README.md", "w", encoding="utf-8") as f:
            f.write(content)

    def _create_markdown_page_in_section(self, page: WikiPage, section_dir: Path):
        """Create individual markdown page in a specific section directory."""

        filename = self._create_safe_filename(page.title)
        content = page.content

        with open(section_dir / f"{filename}.md", "w", encoding="utf-8") as f:
            f.write(content)

    def _get_pages_by_category(self, wiki_pages) -> dict[str, list]:
        """Group pages by category."""

        pages_by_category = {}

        for page in wiki_pages:
            category = "General"
            # Tags not supported in current WikiPage structure
            # if page.tags:
            #     category = page.tags[0].title()

            if category not in pages_by_category:
                pages_by_category[category] = []
            pages_by_category[category].append(page)

        return pages_by_category

    def _create_safe_filename(self, title: str) -> str:
        """Create safe filename from title."""
        import re

        safe_name = re.sub(r"[^\w\s-]", "", title)
        safe_name = re.sub(r"[-\s]+", "-", safe_name)
        return safe_name.strip("-").lower()
