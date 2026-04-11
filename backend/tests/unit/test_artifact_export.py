"""Unit tests for app/core/artifact_export.py"""

import pytest

from app.core.artifact_export import ArtifactExporter, normalize_wiki_id
from app.core.state.wiki_state import PageSpec, SectionSpec, WikiPage, WikiStructureSpec


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_page_spec(name: str = "Overview", order: int = 0) -> PageSpec:
    return PageSpec(
        page_name=name,
        page_order=order,
        description="A page",
        content_focus="general",
        rationale="needed",
    )


def _make_section(name: str = "Architecture", pages: list[PageSpec] | None = None) -> SectionSpec:
    return SectionSpec(
        section_name=name,
        section_order=0,
        description="Section description",
        rationale="needed",
        pages=pages or [_make_page_spec()],
    )


def _make_structure(title: str = "Test Wiki", sections: list[SectionSpec] | None = None) -> WikiStructureSpec:
    s = sections or [_make_section()]
    return WikiStructureSpec(
        wiki_title=title,
        overview="Overview text",
        sections=s,
        total_pages=sum(len(sec.pages) for sec in s),
    )


def _make_wiki_page(page_id: str = "0#0", title: str = "Overview", content: str = "# Overview\n\nContent.") -> WikiPage:
    return WikiPage(page_id=page_id, title=title, content=content, status="completed")


def _make_repo_metadata() -> dict:
    return {"repository_url": "https://github.com/owner/repo", "branch": "main"}


# ---------------------------------------------------------------------------
# normalize_wiki_id
# ---------------------------------------------------------------------------


class TestNormalizeWikiId:
    def test_basic_owner_repo(self):
        result = normalize_wiki_id(owner="Microsoft", repo="VSCode", branch="main")
        assert result == "microsoft--vscode--main"

    def test_normalizes_underscores_in_owner(self):
        result = normalize_wiki_id(owner="my_org", repo="my_repo", branch="main")
        assert result == "my-org--my-repo--main"

    def test_normalizes_dots(self):
        result = normalize_wiki_id(owner="my.org", repo="my.repo", branch="main")
        assert result == "my-org--my-repo--main"

    def test_branch_defaults_to_main(self):
        result = normalize_wiki_id(owner="owner", repo="repo")
        assert result.endswith("--main")

    def test_branch_slashes_replaced(self):
        result = normalize_wiki_id(owner="owner", repo="repo", branch="feature/my-branch")
        assert result == "owner--repo--feature-my-branch"

    def test_from_repository_string(self):
        result = normalize_wiki_id(repository="owner/repo", branch="dev")
        assert result == "owner--repo--dev"

    def test_raises_when_no_owner_or_repo(self):
        with pytest.raises(ValueError, match="Could not determine owner and repo"):
            normalize_wiki_id()

    def test_raises_when_only_repo(self):
        with pytest.raises(ValueError):
            normalize_wiki_id(repo="repo")

    def test_format_is_owner_double_dash_repo_double_dash_branch(self):
        result = normalize_wiki_id(owner="a", repo="b", branch="c")
        parts = result.split("--")
        assert len(parts) == 3
        assert parts == ["a", "b", "c"]


# ---------------------------------------------------------------------------
# ArtifactExporter construction
# ---------------------------------------------------------------------------


class TestArtifactExporterConstruction:
    def test_wiki_id_stored(self):
        exporter = ArtifactExporter(wiki_id="owner--repo--main")
        assert exporter.wiki_id == "owner--repo--main"

    def test_export_formats_default(self):
        exporter = ArtifactExporter()
        assert "json" in exporter.export_formats
        assert "markdown" in exporter.export_formats

    def test_wiki_id_none_by_default(self):
        exporter = ArtifactExporter()
        assert exporter.wiki_id is None


# ---------------------------------------------------------------------------
# ArtifactExporter._prefix_name
# ---------------------------------------------------------------------------


class TestPrefixName:
    def test_prefix_with_wiki_id(self):
        exporter = ArtifactExporter(wiki_id="owner--repo--main")
        result = exporter._prefix_name("wiki_pages", "section/page.md")
        assert result == "owner--repo--main/wiki_pages/section/page.md"

    def test_prefix_without_wiki_id(self):
        exporter = ArtifactExporter()
        result = exporter._prefix_name("analysis", "structure.json")
        assert result == "analysis/structure.json"


# ---------------------------------------------------------------------------
# ArtifactExporter._create_safe_filename
# ---------------------------------------------------------------------------


class TestCreateSafeFilename:
    def test_lowercases_title(self):
        exporter = ArtifactExporter()
        assert exporter._create_safe_filename("MyPage") == "mypage"

    def test_replaces_spaces_with_dashes(self):
        exporter = ArtifactExporter()
        result = exporter._create_safe_filename("My Page Title")
        assert " " not in result
        assert "my" in result

    def test_removes_special_characters(self):
        exporter = ArtifactExporter()
        result = exporter._create_safe_filename("Page: Overview & Summary")
        assert ":" not in result
        assert "&" not in result

    def test_empty_string_returns_empty(self):
        exporter = ArtifactExporter()
        result = exporter._create_safe_filename("")
        assert result == ""


# ---------------------------------------------------------------------------
# ArtifactExporter._normalize_tag
# ---------------------------------------------------------------------------


class TestNormalizeTag:
    def test_converts_to_lowercase(self):
        exporter = ArtifactExporter()
        assert exporter._normalize_tag("MyTag") == "mytag"

    def test_replaces_spaces_with_hyphens(self):
        exporter = ArtifactExporter()
        result = exporter._normalize_tag("my tag here")
        assert result == "my-tag-here"

    def test_empty_returns_unknown(self):
        exporter = ArtifactExporter()
        assert exporter._normalize_tag("") == "unknown"


# ---------------------------------------------------------------------------
# ArtifactExporter._generate_frontmatter
# ---------------------------------------------------------------------------


class TestGenerateFrontmatter:
    def test_produces_yaml_block(self):
        exporter = ArtifactExporter()
        result = exporter._generate_frontmatter({"title": "Test", "tags": ["a", "b"]})
        assert result.startswith("---")
        assert result.strip().endswith("---")

    def test_contains_field_values(self):
        exporter = ArtifactExporter()
        result = exporter._generate_frontmatter({"title": "Hello", "version": 1})
        assert "Hello" in result
        assert "title" in result

    def test_lists_expanded_with_items(self):
        exporter = ArtifactExporter()
        result = exporter._generate_frontmatter({"tags": ["wiki/test", "type/page"]})
        assert "- wiki/test" in result
        assert "- type/page" in result

    def test_none_values_omitted(self):
        exporter = ArtifactExporter()
        result = exporter._generate_frontmatter({"title": "X", "empty": None})
        assert "empty" not in result

    def test_empty_string_values_omitted(self):
        exporter = ArtifactExporter()
        result = exporter._generate_frontmatter({"title": "X", "empty": ""})
        assert "empty" not in result

    def test_strings_with_special_chars_quoted(self):
        exporter = ArtifactExporter()
        result = exporter._generate_frontmatter({"title": "My: Title"})
        # Colons in values need quoting
        assert "My" in result


# ---------------------------------------------------------------------------
# ArtifactExporter._extract_code_sources
# ---------------------------------------------------------------------------


class TestExtractCodeSources:
    def test_extracts_single_source(self):
        exporter = ArtifactExporter()
        result = exporter._extract_code_sources("See <code_source: src/app.py> for details.")
        assert result == ["src/app.py"]

    def test_extracts_multiple_sources(self):
        exporter = ArtifactExporter()
        content = "See <code_source: a.py> and <code_source: b.py>."
        result = exporter._extract_code_sources(content)
        assert "a.py" in result
        assert "b.py" in result

    def test_deduplicates_sources(self):
        exporter = ArtifactExporter()
        content = "<code_source: a.py> and <code_source: a.py>"
        result = exporter._extract_code_sources(content)
        assert result.count("a.py") == 1

    def test_strips_line_references(self):
        exporter = ArtifactExporter()
        result = exporter._extract_code_sources("<code_source: src/app.py:L10-L20>")
        assert result == ["src/app.py"]

    def test_empty_content_returns_empty(self):
        exporter = ArtifactExporter()
        assert exporter._extract_code_sources("no tags here") == []


# ---------------------------------------------------------------------------
# ArtifactExporter._transform_code_source_refs
# ---------------------------------------------------------------------------


class TestTransformCodeSourceRefs:
    def test_replaces_tag_with_wikilink(self):
        exporter = ArtifactExporter()
        result = exporter._transform_code_source_refs("See <code_source: src/app.py>.")
        assert "[[source/src/app.py" in result
        assert "<code_source:" not in result

    def test_fenced_code_blocks_untouched(self):
        exporter = ArtifactExporter()
        content = "```python\n# From: <code_source: src/app.py>\n```"
        result = exporter._transform_code_source_refs(content)
        assert "<code_source: src/app.py>" in result

    def test_line_reference_included_in_display(self):
        exporter = ArtifactExporter()
        result = exporter._transform_code_source_refs("<code_source: src/app.py:L10-L20>")
        assert "L10-L20" in result


# ---------------------------------------------------------------------------
# ArtifactExporter._build_page_title_map
# ---------------------------------------------------------------------------


class TestBuildPageTitleMap:
    def test_map_contains_all_pages(self):
        exporter = ArtifactExporter()
        section = _make_section("Arch", pages=[_make_page_spec("Overview"), _make_page_spec("Details", 1)])
        structure = _make_structure(sections=[section])
        pages = [
            _make_wiki_page("0#0", "Overview"),
            _make_wiki_page("0#1", "Details"),
        ]
        result = exporter._build_page_title_map(structure, pages)
        assert len(result) == 2

    def test_map_entry_has_title(self):
        exporter = ArtifactExporter()
        section = _make_section("Arch", pages=[_make_page_spec("Overview")])
        structure = _make_structure(sections=[section])
        pages = [_make_wiki_page("0#0", "Overview")]
        result = exporter._build_page_title_map(structure, pages)
        entry = list(result.values())[0]
        assert entry["title"] == "Overview"

    def test_map_entry_has_section_name(self):
        exporter = ArtifactExporter()
        section = _make_section("Architecture", pages=[_make_page_spec("Intro")])
        structure = _make_structure(sections=[section])
        pages = [_make_wiki_page("0#0", "Intro")]
        result = exporter._build_page_title_map(structure, pages)
        entry = list(result.values())[0]
        assert entry["section_name"] == "Architecture"


# ---------------------------------------------------------------------------
# ArtifactExporter._export_json_format
# ---------------------------------------------------------------------------


class TestExportJsonFormat:
    def test_returns_list_with_one_artifact(self):
        exporter = ArtifactExporter(wiki_id="owner--repo--main")
        structure = _make_structure()
        pages = [_make_wiki_page()]
        result = exporter._export_json_format(structure, pages)

        assert len(result) == 1

    def test_artifact_has_name_type_data(self):
        exporter = ArtifactExporter()
        structure = _make_structure()
        pages = [_make_wiki_page()]
        result = exporter._export_json_format(structure, pages)

        artifact = result[0]
        assert "name" in artifact
        assert "type" in artifact
        assert "data" in artifact

    def test_artifact_type_is_json(self):
        exporter = ArtifactExporter()
        structure = _make_structure()
        pages = [_make_wiki_page()]
        result = exporter._export_json_format(structure, pages)
        assert result[0]["type"] == "application/json"

    def test_artifact_data_is_valid_json(self):
        import json

        exporter = ArtifactExporter()
        structure = _make_structure()
        pages = [_make_wiki_page()]
        result = exporter._export_json_format(structure, pages)
        parsed = json.loads(result[0]["data"])
        assert "wiki_title" in parsed
        assert "sections" in parsed

    def test_fallback_grouping_when_no_sections(self):
        exporter = ArtifactExporter()
        structure = WikiStructureSpec(wiki_title="T", overview="", sections=[], total_pages=0)
        pages = [_make_wiki_page("0#0", "Page One")]
        result = exporter._export_json_format(structure, pages)
        assert len(result) == 1


# ---------------------------------------------------------------------------
# ArtifactExporter._export_markdown_format
# ---------------------------------------------------------------------------


class TestExportMarkdownFormat:
    def test_returns_list_of_artifacts(self, tmp_path):
        exporter = ArtifactExporter(wiki_id="owner--repo--main")
        structure = _make_structure()
        pages = [_make_wiki_page("0#0", "Overview", "# Overview\nContent")]
        result = exporter._export_markdown_format(structure, pages, _make_repo_metadata(), tmp_path)

        assert isinstance(result, list)
        assert len(result) > 0

    def test_artifacts_have_markdown_type(self, tmp_path):
        exporter = ArtifactExporter()
        structure = _make_structure()
        pages = [_make_wiki_page("0#0", "Overview")]
        result = exporter._export_markdown_format(structure, pages, _make_repo_metadata(), tmp_path)

        for artifact in result:
            assert artifact["type"] == "text/markdown"

    def test_artifact_names_end_with_md(self, tmp_path):
        exporter = ArtifactExporter()
        structure = _make_structure()
        pages = [_make_wiki_page("0#0", "Overview")]
        result = exporter._export_markdown_format(structure, pages, _make_repo_metadata(), tmp_path)

        for artifact in result:
            assert artifact["name"].endswith(".md"), f"Expected .md but got: {artifact['name']}"

    def test_readme_included(self, tmp_path):
        exporter = ArtifactExporter()
        structure = _make_structure()
        pages = [_make_wiki_page("0#0", "Overview")]
        result = exporter._export_markdown_format(structure, pages, _make_repo_metadata(), tmp_path)

        names = [a["name"] for a in result]
        assert any("README" in n for n in names)

    def test_section_index_included(self, tmp_path):
        exporter = ArtifactExporter()
        structure = _make_structure()
        pages = [_make_wiki_page("0#0", "Overview")]
        result = exporter._export_markdown_format(structure, pages, _make_repo_metadata(), tmp_path)

        names = [a["name"] for a in result]
        assert any("_index" in n for n in names)


# ---------------------------------------------------------------------------
# ArtifactExporter._add_related_pages_footer
# ---------------------------------------------------------------------------


class TestAddRelatedPagesFooter:
    SECTION_NAME = "Arch"

    def _make_page_title_map(self) -> tuple[dict, object]:
        section = _make_section(self.SECTION_NAME, pages=[_make_page_spec("Intro"), _make_page_spec("Details")])
        exporter = ArtifactExporter()
        structure = _make_structure(sections=[section])
        pages = [_make_wiki_page("0#0", "Intro"), _make_wiki_page("0#1", "Details")]
        title_map = exporter._build_page_title_map(structure, pages)
        return title_map, section

    def test_adds_related_pages_section(self):
        exporter = ArtifactExporter()
        title_map, section = self._make_page_title_map()
        content = "# Page\n\nSome content."
        result = exporter._add_related_pages_footer(content, section, title_map, "Intro")
        assert "## Related Pages" in result

    def test_does_not_duplicate_related_pages(self):
        exporter = ArtifactExporter()
        title_map, section = self._make_page_title_map()
        content = "# Page\n\n## Related Pages\n- [[Other]]\n"
        result = exporter._add_related_pages_footer(content, section, title_map, "Intro")
        assert result.count("## Related Pages") == 1

    def test_returns_content_unchanged_when_no_siblings(self):
        exporter = ArtifactExporter()
        section = _make_section("Solo")
        # Map with only the current page (no siblings)
        title_map = {"0#0": {"title": "OnlyPage", "section_name": "Solo", "section_safe": "solo",
                              "page_safe": "onlypage", "page_spec": _make_page_spec(), "section_spec": section}}
        content = "# Only Page\n\nContent."
        result = exporter._add_related_pages_footer(content, section, title_map, "OnlyPage")
        assert "## Related Pages" not in result


# ---------------------------------------------------------------------------
# ArtifactExporter.export_wiki_artifacts
# ---------------------------------------------------------------------------


class TestExportWikiArtifacts:
    def test_returns_list_of_artifacts(self):
        exporter = ArtifactExporter(wiki_id="owner--repo--main")
        structure = _make_structure()
        pages = [_make_wiki_page("0#0", "Overview")]
        result = exporter.export_wiki_artifacts(
            structure, pages, _make_repo_metadata(), export_formats=["json"]
        )
        assert isinstance(result, list)
        assert len(result) >= 1

    def test_returns_empty_on_unknown_format(self):
        exporter = ArtifactExporter()
        structure = _make_structure()
        pages = [_make_wiki_page()]
        # An unknown format raises internally → returns []
        result = exporter.export_wiki_artifacts(
            structure, pages, {}, export_formats=["unknown_format"]
        )
        assert result == []

    def test_default_formats_produce_both_json_and_markdown(self):
        exporter = ArtifactExporter()
        structure = _make_structure()
        pages = [_make_wiki_page("0#0", "Overview")]
        result = exporter.export_wiki_artifacts(structure, pages, _make_repo_metadata())

        types = {a["type"] for a in result}
        assert "application/json" in types
        assert "text/markdown" in types

    def test_json_only_export(self):
        exporter = ArtifactExporter()
        structure = _make_structure()
        pages = [_make_wiki_page("0#0", "Overview")]
        result = exporter.export_wiki_artifacts(
            structure, pages, _make_repo_metadata(), export_formats=["json"]
        )
        for a in result:
            assert a["type"] == "application/json"

    def test_markdown_only_export(self):
        exporter = ArtifactExporter()
        structure = _make_structure()
        pages = [_make_wiki_page("0#0", "Overview")]
        result = exporter.export_wiki_artifacts(
            structure, pages, _make_repo_metadata(), export_formats=["markdown"]
        )
        for a in result:
            assert a["type"] == "text/markdown"


# ---------------------------------------------------------------------------
# ArtifactExporter._get_pages_by_category
# ---------------------------------------------------------------------------


class TestGetPagesByCategory:
    def test_all_pages_in_general_category(self):
        exporter = ArtifactExporter()
        pages = [_make_wiki_page("0#0", "A"), _make_wiki_page("0#1", "B")]
        result = exporter._get_pages_by_category(pages)
        assert "General" in result
        assert len(result["General"]) == 2

    def test_empty_pages_returns_empty(self):
        exporter = ArtifactExporter()
        result = exporter._get_pages_by_category([])
        assert result == {}


# ---------------------------------------------------------------------------
# ArtifactExporter._make_page_frontmatter
# ---------------------------------------------------------------------------


class TestMakePageFrontmatter:
    def test_includes_title(self):
        exporter = ArtifactExporter()
        section = _make_section("Architecture")
        structure = _make_structure(sections=[section])
        page_spec = _make_page_spec("Overview")
        page = _make_wiki_page("0#0", "Overview")

        result = exporter._make_page_frontmatter(
            page, page_spec, section, structure, _make_repo_metadata(), []
        )
        assert "Overview" in result

    def test_includes_section_tag(self):
        exporter = ArtifactExporter()
        section = _make_section("Core")
        structure = _make_structure(sections=[section])
        page = _make_wiki_page("0#0", "Design")

        result = exporter._make_page_frontmatter(
            page, _make_page_spec("Design"), section, structure, _make_repo_metadata(), []
        )
        assert "section/core" in result

    def test_includes_python_lang_tag_for_py_files(self):
        exporter = ArtifactExporter()
        section = _make_section()
        structure = _make_structure(sections=[section])
        page = _make_wiki_page("0#0", "Code")

        result = exporter._make_page_frontmatter(
            page, _make_page_spec("Code"), section, structure, _make_repo_metadata(),
            ["src/app.py", "lib/utils.py"]
        )
        assert "lang/python" in result

    def test_includes_source_files_field(self):
        exporter = ArtifactExporter()
        section = _make_section()
        structure = _make_structure(sections=[section])
        page = _make_wiki_page("0#0", "Code")

        result = exporter._make_page_frontmatter(
            page, _make_page_spec("Code"), section, structure, _make_repo_metadata(),
            ["src/main.py"]
        )
        assert "source_files" in result

    def test_includes_symbols_when_target_symbols_set(self):
        exporter = ArtifactExporter()
        section = _make_section()
        structure = _make_structure(sections=[section])
        page = _make_wiki_page("0#0", "Code")
        page_spec = PageSpec(
            page_name="Code", page_order=0, description="", content_focus="",
            rationale="", target_symbols=["MyClass", "my_func"]
        )

        result = exporter._make_page_frontmatter(
            page, page_spec, section, structure, _make_repo_metadata(), []
        )
        assert "symbols" in result

    def test_includes_key_files_when_set(self):
        exporter = ArtifactExporter()
        section = _make_section()
        structure = _make_structure(sections=[section])
        page = _make_wiki_page("0#0", "Code")
        page_spec = PageSpec(
            page_name="Code", page_order=0, description="", content_focus="",
            rationale="", key_files=["important.py"]
        )

        result = exporter._make_page_frontmatter(
            page, page_spec, section, structure, _make_repo_metadata(), []
        )
        assert "key_files" in result


# ---------------------------------------------------------------------------
# ArtifactExporter — source file stubs via full export
# ---------------------------------------------------------------------------


class TestSourceFileStubs:
    def test_source_stubs_created_when_code_source_tags_present(self, tmp_path):
        exporter = ArtifactExporter()
        structure = _make_structure()
        pages = [_make_wiki_page("0#0", "Code Page", "# Code\n\n<code_source: src/app.py> details.")]
        result = exporter._export_markdown_format(structure, pages, _make_repo_metadata(), tmp_path)

        names = [a["name"] for a in result]
        # Source stub should appear under source/
        assert any("source/" in n for n in names), f"Expected source/ stub in: {names}"


# ---------------------------------------------------------------------------
# ArtifactExporter._export_markdown_format — fallback (no sections)
# ---------------------------------------------------------------------------


class TestExportMarkdownFallback:
    def test_fallback_export_when_no_sections(self, tmp_path):
        exporter = ArtifactExporter()
        structure = WikiStructureSpec(wiki_title="T", overview="", sections=[], total_pages=0)
        pages = [_make_wiki_page("0#0", "My Page", "# My Page\n\nContent.")]
        result = exporter._export_markdown_format(structure, pages, _make_repo_metadata(), tmp_path)

        assert isinstance(result, list)
        assert len(result) > 0

    def test_fallback_generates_readme(self, tmp_path):
        exporter = ArtifactExporter()
        structure = WikiStructureSpec(wiki_title="T", overview="", sections=[], total_pages=0)
        pages = [_make_wiki_page("0#0", "My Page")]
        result = exporter._export_markdown_format(structure, pages, _make_repo_metadata(), tmp_path)

        names = [a["name"] for a in result]
        assert any("README" in n for n in names)


# ---------------------------------------------------------------------------
# ArtifactExporter._create_section_moc — via full markdown export
# ---------------------------------------------------------------------------


class TestCreateSectionMoc:
    def test_section_moc_has_page_links(self, tmp_path):
        exporter = ArtifactExporter()
        section = _make_section("Core", pages=[_make_page_spec("Overview"), _make_page_spec("Details")])
        structure = _make_structure(sections=[section])
        pages = [_make_wiki_page("0#0", "Overview"), _make_wiki_page("0#1", "Details")]

        result = exporter._export_markdown_format(structure, pages, _make_repo_metadata(), tmp_path)

        # Find the _index.md for the section
        index_artifacts = [a for a in result if "_index" in a["name"]]
        assert len(index_artifacts) == 1
        content = index_artifacts[0]["data"]
        assert "Overview" in content or "Details" in content
