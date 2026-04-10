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
import re
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
        """Export wiki in markdown format with Obsidian enrichments."""

        markdown_dir = temp_path / "wiki_markdown"
        markdown_dir.mkdir(exist_ok=True)
        markdown_artifacts = []

        # Build full page→section lookup before iterating so related-page footers
        # and section MOCs have access to all pages.
        page_title_map: dict[str, dict] = {}
        if wiki_structure.sections:
            page_title_map = self._build_page_title_map(wiki_structure, wiki_pages)

        # Aggregate source-file → [referencing page titles] for stub creation
        source_files_map: dict[str, list[str]] = {}

        if wiki_structure.sections and len(wiki_structure.sections) > 0:
            pages_by_id = {page.page_id: page for page in wiki_pages}

            for section_idx, section in enumerate(wiki_structure.sections):
                section_dir = markdown_dir / self._create_safe_filename(section.section_name)
                section_dir.mkdir(exist_ok=True)

                # Section MOC
                self._create_section_moc(
                    section, wiki_structure, page_title_map, section_dir, repo_metadata
                )

                for page_idx, page_spec in enumerate(section.pages):
                    expected_page_id = f"{section_idx}#{page_idx}"
                    matching_page = pages_by_id.get(expected_page_id) or next(
                        (p for p in wiki_pages if p.title == page_spec.page_name), None
                    )
                    if matching_page:
                        refs = self._create_markdown_page_in_section(
                            matching_page,
                            section_dir,
                            page_spec=page_spec,
                            section_spec=section,
                            wiki_structure=wiki_structure,
                            repo_metadata=repo_metadata,
                            page_title_map=page_title_map,
                        )
                        for fpath in refs:
                            source_files_map.setdefault(fpath, [])
                            if matching_page.title not in source_files_map[fpath]:
                                source_files_map[fpath].append(matching_page.title)
        else:
            # Fallback: group by section derived from page_id
            pages_by_section: dict[str, list] = {}
            for page in wiki_pages:
                section_name = "general"
                if "#" in page.page_id:
                    try:
                        section_idx = int(page.page_id.split("#")[0])
                        if wiki_structure.sections and section_idx < len(wiki_structure.sections):
                            section_name = wiki_structure.sections[section_idx].section_name
                        else:
                            section_name = f"section_{section_idx}"
                    except (ValueError, IndexError):
                        pass
                pages_by_section.setdefault(section_name, []).append(page)

            for section_name, pages in pages_by_section.items():
                section_dir = markdown_dir / self._create_safe_filename(section_name)
                section_dir.mkdir(exist_ok=True)
                for page in pages:
                    refs = self._create_markdown_page_in_section(page, section_dir)
                    for fpath in refs:
                        source_files_map.setdefault(fpath, [])
                        if page.title not in source_files_map[fpath]:
                            source_files_map[fpath].append(page.title)

        # Source file stub notes
        if source_files_map:
            source_dir = markdown_dir / "source"
            source_dir.mkdir(exist_ok=True)
            self._create_source_file_stubs(
                source_files_map, source_dir, wiki_structure, repo_metadata
            )

        # Main index
        self._create_markdown_index(wiki_structure, wiki_pages, markdown_dir, repo_metadata)

        # Collect all .md files as artifacts
        for file_path in markdown_dir.rglob("*"):
            if file_path.is_file() and file_path.suffix == ".md":
                with open(file_path, "rb") as f:
                    file_data = f.read()
                relative_path = file_path.relative_to(markdown_dir)
                artifact_name = self._prefix_name("wiki_pages", str(relative_path))
                markdown_artifacts.append(
                    {"name": artifact_name, "type": "text/markdown", "data": file_data.decode("utf-8")}
                )

        return markdown_artifacts

    def _create_markdown_index(
        self,
        wiki_structure: WikiStructureSpec,
        wiki_pages: list[WikiPage],
        output_dir: Path,
        repo_metadata: dict | None = None,
    ) -> None:
        """Create README.md index with YAML frontmatter and [[wikilinks]]."""
        wiki_title = wiki_structure.wiki_title or "Wiki"
        wiki_tag = self._normalize_tag(wiki_title)
        repo_url = (repo_metadata or {}).get("repository_url", "")
        branch = (repo_metadata or {}).get("branch", "main")

        frontmatter = self._generate_frontmatter(
            {
                "title": wiki_title,
                "aliases": [wiki_title, f"{wiki_title} Index"],
                "tags": [f"wiki/{wiki_tag}", "type/wiki-index", "type/moc"],
                "type": "wiki-index",
                "wiki": wiki_title,
                "repository": repo_url,
                "branch": branch,
                "description": getattr(wiki_structure, "overview", ""),
                "created": datetime.now().strftime("%Y-%m-%d"),
            }
        )

        content = frontmatter
        content += f"# {wiki_title}\n\n"
        content += f"{getattr(wiki_structure, 'overview', 'Generated wiki documentation')}\n\n"
        content += "## Wiki Structure\n\n"

        if wiki_structure.sections and len(wiki_structure.sections) > 0:
            pages_by_id = {page.page_id: page for page in wiki_pages}

            for section_idx, section in enumerate(wiki_structure.sections):
                # Link to section MOC
                content += f"### [[{section.section_name}]]\n\n"
                if getattr(section, "description", ""):
                    content += f"{section.description}\n\n"

                for page_idx, page_spec in enumerate(section.pages):
                    expected_page_id = f"{section_idx}#{page_idx}"
                    matching_page = pages_by_id.get(expected_page_id) or next(
                        (p for p in wiki_pages if p.title == page_spec.page_name), None
                    )
                    if matching_page:
                        content += f"- [[{matching_page.title}]]\n"
                content += "\n"
        else:
            # Fallback: group pages by section extracted from page_id
            pages_by_section: dict[str, list] = {}
            for page in wiki_pages:
                section_name = "general"
                if "#" in page.page_id:
                    try:
                        section_idx = int(page.page_id.split("#")[0])
                        if wiki_structure.sections and section_idx < len(wiki_structure.sections):
                            section_name = wiki_structure.sections[section_idx].section_name
                        else:
                            section_name = f"section_{section_idx}"
                    except (ValueError, IndexError):
                        pass
                pages_by_section.setdefault(section_name, []).append(page)

            for section_name, pages in pages_by_section.items():
                content += f"### {section_name.title()}/\n\n"
                for page in pages:
                    content += f"- [[{page.title}]]\n"
                content += "\n"

        content += "\n---\n\n*Generated by Wikis*\n"

        with open(output_dir / "README.md", "w", encoding="utf-8") as f:
            f.write(content)

    def _create_markdown_page_in_section(
        self,
        page: WikiPage,
        section_dir: Path,
        page_spec: Any = None,
        section_spec: Any = None,
        wiki_structure: WikiStructureSpec | None = None,
        repo_metadata: dict | None = None,
        page_title_map: dict | None = None,
    ) -> list[str]:
        """Create individual markdown page with Obsidian frontmatter, wikilinks, and
        a related-pages footer. Returns list of referenced source file paths."""
        filename = self._create_safe_filename(page.title)
        content = page.content

        # Extract source references before any transformation
        source_files = self._extract_code_sources(content)

        # Transform <code_source:> tags → [[source/path|display]] wikilinks in prose
        content = self._transform_code_source_refs(content)

        # Append related pages footer (sibling pages in same section)
        if section_spec and page_title_map:
            content = self._add_related_pages_footer(
                content, section_spec, page_title_map, page.title
            )

        # Prepend YAML frontmatter
        if page_spec and section_spec and wiki_structure:
            frontmatter = self._make_page_frontmatter(
                page, page_spec, section_spec, wiki_structure, repo_metadata or {}, source_files
            )
            content = frontmatter + "\n" + content

        with open(section_dir / f"{filename}.md", "w", encoding="utf-8") as f:
            f.write(content)

        return source_files

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
        safe_name = re.sub(r"[^\w\s-]", "", title)
        safe_name = re.sub(r"[-\s]+", "-", safe_name)
        return safe_name.strip("-").lower()

    # ------------------------------------------------------------------
    # Obsidian enrichment helpers
    # ------------------------------------------------------------------

    def _normalize_tag(self, text: str) -> str:
        """Convert text to Obsidian tag-safe format (lowercase, hyphenated)."""
        tag = re.sub(r"[^\w\s-]", "", text.lower())
        tag = re.sub(r"[\s_]+", "-", tag.strip())
        return tag.strip("-") or "unknown"

    def _generate_frontmatter(self, fields: dict) -> str:
        """Generate a YAML frontmatter block from a dict."""

        def _yaml_scalar(v: Any) -> str:
            if isinstance(v, bool):
                return "true" if v else "false"
            if isinstance(v, (int, float)):
                return str(v)
            s = str(v)
            if any(c in s for c in ':#{}[]|>&*!,\'"'):
                return json.dumps(s)
            return s

        lines = ["---"]
        for key, value in fields.items():
            if value is None or value == "" or value == []:
                continue
            if isinstance(value, list):
                lines.append(f"{key}:")
                for item in value:
                    lines.append(f"  - {_yaml_scalar(item)}")
            else:
                lines.append(f"{key}: {_yaml_scalar(value)}")
        lines.append("---")
        return "\n".join(lines) + "\n"

    def _extract_code_sources(self, content: str) -> list[str]:
        """Return unique file paths from all <code_source: path> tags in content."""
        paths: list[str] = []
        seen: set[str] = set()
        for m in re.finditer(r"<code_source:\s*([^>]+?)\s*>", content):
            raw = m.group(1).strip()
            # Strip optional line reference  :L45-L120
            path = re.sub(r":L\d+(-L?\d+)?$", "", raw).strip()
            if path and path not in seen:
                seen.add(path)
                paths.append(path)
        return paths

    def _transform_code_source_refs(self, content: str) -> str:
        """Replace <code_source: path> with [[source/path|display]] in prose.

        Fenced code blocks are left untouched so comment annotations like
        ``# From: <code_source: path>`` are preserved for context.
        """
        segments = re.split(r"(```[\s\S]*?```)", content)
        result: list[str] = []
        for i, seg in enumerate(segments):
            if i % 2 == 1:  # fenced code block – leave as-is
                result.append(seg)
                continue

            def _replace(m: re.Match) -> str:
                raw = m.group(1).strip()
                line_m = re.match(r"^(.+?)(:[Ll]\d+(?:-[Ll]?\d+)?)$", raw)
                if line_m:
                    path, line_ref = line_m.group(1).strip(), line_m.group(2)
                else:
                    path, line_ref = raw, ""
                safe = path.replace("\\", "/")
                display = f"{path}{line_ref}" if line_ref else path
                return f"[[source/{safe}|{display}]]"

            result.append(re.sub(r"<code_source:\s*([^>]+?)\s*>", _replace, seg))
        return "".join(result)

    def _build_page_title_map(
        self, wiki_structure: WikiStructureSpec, wiki_pages: list[WikiPage]
    ) -> dict[str, dict]:
        """Return mapping: page_id → {title, section_name, section_safe, page_safe,
        page_spec, section_spec}."""
        pages_by_id = {p.page_id: p for p in wiki_pages}
        result: dict[str, dict] = {}
        for section_idx, section in enumerate(wiki_structure.sections or []):
            safe_section = self._create_safe_filename(section.section_name)
            for page_idx, page_spec in enumerate(section.pages):
                expected_id = f"{section_idx}#{page_idx}"
                page = pages_by_id.get(expected_id) or next(
                    (p for p in wiki_pages if p.title == page_spec.page_name), None
                )
                if page:
                    result[page.page_id] = {
                        "title": page.title,
                        "section_name": section.section_name,
                        "section_safe": safe_section,
                        "page_safe": self._create_safe_filename(page.title),
                        "page_spec": page_spec,
                        "section_spec": section,
                    }
        return result

    def _make_page_frontmatter(
        self,
        page: WikiPage,
        page_spec: Any,
        section_spec: Any,
        wiki_structure: WikiStructureSpec,
        repo_metadata: dict,
        source_files: list[str],
    ) -> str:
        wiki_title = wiki_structure.wiki_title or "Wiki"
        wiki_tag = self._normalize_tag(wiki_title)
        section_tag = self._normalize_tag(section_spec.section_name)
        repo_url = (repo_metadata or {}).get("repository_url", "")
        branch = (repo_metadata or {}).get("branch", "main")

        tags = [f"wiki/{wiki_tag}", f"section/{section_tag}", "type/wiki-page"]
        ext_lang = {
            "py": "python", "ts": "typescript", "tsx": "typescript",
            "js": "javascript", "jsx": "javascript", "go": "go",
            "rs": "rust", "java": "java", "cpp": "cpp", "cs": "csharp",
            "rb": "ruby", "kt": "kotlin", "swift": "swift",
        }
        for fpath in source_files:
            ext = Path(fpath).suffix.lstrip(".").lower()
            lang = ext_lang.get(ext)
            if lang and f"lang/{lang}" not in tags:
                tags.append(f"lang/{lang}")

        created = datetime.now().strftime("%Y-%m-%d")
        if hasattr(page, "generated_at") and page.generated_at:
            try:
                created = page.generated_at.strftime("%Y-%m-%d")
            except Exception:
                pass

        fields: dict[str, Any] = {
            "title": page.title,
            "aliases": [page.title],
            "tags": tags,
            "type": "wiki-page",
            "wiki": wiki_title,
            "section": section_spec.section_name,
            "section_order": getattr(section_spec, "section_order", 0),
            "page_order": getattr(page_spec, "page_order", 0),
            "repository": repo_url,
            "branch": branch,
            "description": getattr(page_spec, "description", ""),
            "created": created,
        }
        if source_files:
            fields["source_files"] = [f"source/{f}" for f in source_files]
        target_symbols = getattr(page_spec, "target_symbols", None)
        if target_symbols:
            fields["symbols"] = target_symbols
        key_files = getattr(page_spec, "key_files", None)
        if key_files:
            fields["key_files"] = key_files
        return self._generate_frontmatter(fields)

    def _add_related_pages_footer(
        self,
        content: str,
        section_spec: Any,
        page_title_map: dict[str, dict],
        current_title: str,
    ) -> str:
        """Append a ## Related Pages section linking to sibling pages."""
        siblings = sorted(
            info["title"]
            for info in page_title_map.values()
            if info["section_name"] == section_spec.section_name
            and info["title"] != current_title
        )
        if not siblings:
            return content
        footer = "\n\n---\n\n## Related Pages\n\n"
        for title in siblings:
            footer += f"- [[{title}]]\n"
        return content.rstrip() + footer

    def _create_section_moc(
        self,
        section_spec: Any,
        wiki_structure: WikiStructureSpec,
        page_title_map: dict[str, dict],
        section_dir: Path,
        repo_metadata: dict,
    ) -> None:
        """Write a Map of Content (_index.md) for a section."""
        wiki_title = wiki_structure.wiki_title or "Wiki"
        wiki_tag = self._normalize_tag(wiki_title)
        section_tag = self._normalize_tag(section_spec.section_name)
        repo_url = (repo_metadata or {}).get("repository_url", "")
        branch = (repo_metadata or {}).get("branch", "main")

        section_pages = [
            info
            for info in page_title_map.values()
            if info["section_name"] == section_spec.section_name
        ]

        frontmatter = self._generate_frontmatter(
            {
                "title": section_spec.section_name,
                "aliases": [section_spec.section_name, f"{section_spec.section_name} Overview"],
                "tags": [f"wiki/{wiki_tag}", f"section/{section_tag}", "type/moc", "type/section-index"],
                "type": "section-moc",
                "wiki": wiki_title,
                "section": section_spec.section_name,
                "section_order": getattr(section_spec, "section_order", 0),
                "repository": repo_url,
                "branch": branch,
                "description": getattr(section_spec, "description", ""),
                "created": datetime.now().strftime("%Y-%m-%d"),
            }
        )

        body = frontmatter
        body += f"# {section_spec.section_name}\n\n"
        desc = getattr(section_spec, "description", "")
        if desc:
            body += f"{desc}\n\n"
        body += "## Pages\n\n"
        for info in section_pages:
            page_desc = getattr(info["page_spec"], "description", "") or ""
            if page_desc:
                body += f"- [[{info['title']}]] — {page_desc}\n"
            else:
                body += f"- [[{info['title']}]]\n"
        body += f"\n---\n\n→ [[{wiki_title}|Wiki Index]]\n"

        with open(section_dir / "_index.md", "w", encoding="utf-8") as f:
            f.write(body)

    def _create_source_file_stubs(
        self,
        source_files_map: dict[str, list[str]],
        source_dir: Path,
        wiki_structure: WikiStructureSpec,
        repo_metadata: dict,
    ) -> None:
        """Create stub notes under source/ for every referenced source file."""
        wiki_title = wiki_structure.wiki_title or "Wiki"
        wiki_tag = self._normalize_tag(wiki_title)
        repo_url = (repo_metadata or {}).get("repository_url", "")

        ext_lang = {
            "py": "python", "ts": "typescript", "tsx": "typescript",
            "js": "javascript", "jsx": "javascript", "go": "go",
            "rs": "rust", "java": "java", "cpp": "cpp", "c": "c",
            "cs": "csharp", "rb": "ruby", "php": "php", "kt": "kotlin",
            "swift": "swift", "yaml": "yaml", "yml": "yaml",
            "json": "json", "toml": "toml", "md": "markdown",
        }

        for file_path, referencing_pages in source_files_map.items():
            safe_path = file_path.strip().replace("\\", "/")
            parts = Path(safe_path).parts

            stub_dir = source_dir
            for part in parts[:-1]:
                stub_dir = stub_dir / part
                stub_dir.mkdir(parents=True, exist_ok=True)

            filename = parts[-1] if parts else safe_path
            ext = Path(filename).suffix.lstrip(".").lower()
            lang = ext_lang.get(ext, "")

            tags = [f"wiki/{wiki_tag}", "type/source-file"]
            if lang:
                tags.append(f"lang/{lang}")

            frontmatter = self._generate_frontmatter(
                {
                    "title": filename,
                    "aliases": [filename, safe_path],
                    "tags": tags,
                    "type": "source-file",
                    "file_path": safe_path,
                    "language": lang or None,
                    "repository": repo_url,
                    "referenced_by": referencing_pages,
                    "created": datetime.now().strftime("%Y-%m-%d"),
                }
            )

            ref_lines = "\n".join(f"> - [[{t}]]" for t in referencing_pages)
            body = frontmatter
            body += f"# `{filename}`\n\n"
            body += f"> [!info] Source File\n"
            body += f"> **Path:** `{safe_path}`\n"
            if repo_url:
                body += f"> **Repository:** {repo_url}\n"
            body += ">\n"
            body += "> Referenced by wiki pages:\n"
            body += ref_lines + "\n"

            stub_stem = self._create_safe_filename(Path(filename).stem)
            with open(stub_dir / f"{stub_stem}.md", "w", encoding="utf-8") as f:
                f.write(body)
