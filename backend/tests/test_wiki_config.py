"""Tests for .wikis/wiki.json config parser."""

from __future__ import annotations

import json

from app.core.wiki_config import load_wiki_config


class TestLoadWikiConfig:
    def test_missing_file_returns_none(self, tmp_path):
        assert load_wiki_config(tmp_path) is None

    def test_valid_config_with_pages(self, tmp_path):
        config_dir = tmp_path / ".wikis"
        config_dir.mkdir()
        (config_dir / "wiki.json").write_text(
            json.dumps(
                {
                    "repo_notes": "This is a FastAPI project.",
                    "pages": [
                        {"title": "Getting Started", "description": "Setup guide"},
                        {"title": "API Reference", "description": "Endpoint docs"},
                    ],
                }
            )
        )
        result = load_wiki_config(tmp_path)
        assert result is not None
        assert result.repo_notes == "This is a FastAPI project."
        assert len(result.pages) == 2
        assert result.pages[0].title == "Getting Started"

    def test_config_with_only_repo_notes(self, tmp_path):
        config_dir = tmp_path / ".wikis"
        config_dir.mkdir()
        (config_dir / "wiki.json").write_text(json.dumps({"repo_notes": "Important context for LLM."}))
        result = load_wiki_config(tmp_path)
        assert result is not None
        assert result.repo_notes == "Important context for LLM."
        assert result.pages == []

    def test_empty_json_object(self, tmp_path):
        config_dir = tmp_path / ".wikis"
        config_dir.mkdir()
        (config_dir / "wiki.json").write_text("{}")
        result = load_wiki_config(tmp_path)
        assert result is not None
        assert result.repo_notes == ""
        assert result.pages == []

    def test_malformed_json_returns_none(self, tmp_path):
        config_dir = tmp_path / ".wikis"
        config_dir.mkdir()
        (config_dir / "wiki.json").write_text("not valid json {{{")
        result = load_wiki_config(tmp_path)
        assert result is None

    def test_invalid_schema_returns_none(self, tmp_path):
        config_dir = tmp_path / ".wikis"
        config_dir.mkdir()
        (config_dir / "wiki.json").write_text(json.dumps({"pages": [{"wrong_field": "no title"}]}))
        result = load_wiki_config(tmp_path)
        assert result is None

    def test_extra_fields_ignored(self, tmp_path):
        config_dir = tmp_path / ".wikis"
        config_dir.mkdir()
        (config_dir / "wiki.json").write_text(
            json.dumps(
                {
                    "repo_notes": "Context",
                    "unknown_field": "ignored",
                }
            )
        )
        result = load_wiki_config(tmp_path)
        assert result is not None
        assert result.repo_notes == "Context"
