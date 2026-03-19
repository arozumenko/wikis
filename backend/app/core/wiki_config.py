"""In-repo wiki configuration parser (.wikis/wiki.json)."""

from __future__ import annotations

import json
import logging
from pathlib import Path

from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

WIKI_CONFIG_PATH = ".wikis/wiki.json"


class PageConfig(BaseModel):
    """A user-defined wiki page specification."""

    title: str
    description: str = ""


class WikiConfig(BaseModel):
    """Parsed .wikis/wiki.json configuration."""

    repo_notes: str = ""
    pages: list[PageConfig] = Field(default_factory=list)


def load_wiki_config(repo_path: str | Path) -> WikiConfig | None:
    """Load .wikis/wiki.json from a cloned repo directory.

    Returns WikiConfig if found and valid, None otherwise.
    Logs warnings on parse errors but never raises.
    """
    config_file = Path(repo_path) / WIKI_CONFIG_PATH
    if not config_file.exists():
        logger.debug(f"No wiki config at {config_file}")
        return None

    try:
        raw = config_file.read_text(encoding="utf-8")
        data = json.loads(raw)
        config = WikiConfig.model_validate(data)
        logger.info(
            f"Loaded wiki config: {len(config.pages)} pages defined, repo_notes={'yes' if config.repo_notes else 'no'}"
        )
        return config
    except json.JSONDecodeError as e:
        logger.warning(f"Malformed wiki config JSON at {config_file}: {e}")
        return None
    except Exception as e:
        logger.warning(f"Failed to parse wiki config at {config_file}: {e}")
        return None
