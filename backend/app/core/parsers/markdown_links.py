"""Markdown link extractor (#228).

Extracts internal, external, wikilink, and attachment references from
source markdown. Used by:

- Doc-graph clustering (#230) to build the markdown link graph
- Wiki-link resolver (#241) to validate ``[[wiki-links]]`` in generated pages

Code fences and inline code are stripped before extraction so embedded link
syntax in code examples is ignored.
"""

from __future__ import annotations

import posixpath
import re
from dataclasses import dataclass
from typing import Literal

LinkKind = Literal["internal", "external", "wikilink", "attachment"]


@dataclass
class Link:
    kind: LinkKind
    target: str
    text: str
    anchor: str | None
    source_path: str | None
    resolved: str | None


_FENCED_BLOCK_RE = re.compile(r"```[\s\S]*?```|~~~[\s\S]*?~~~")
_INLINE_CODE_RE = re.compile(r"`[^`\n]+`")

_ATTACHMENT_PLACEHOLDER_RE = re.compile(r"\[\[attachment:\s*([^\]]+)\]\]")
_IMAGE_RE = re.compile(r"!\[([^\]]*)\]\(([^)\s]+)(?:\s+\"[^\"]*\")?\)")
_LINK_RE = re.compile(r"(?<!!)\[([^\]]+)\]\(([^)\s]+)(?:\s+\"[^\"]*\")?\)")
_WIKILINK_RE = re.compile(r"\[\[([^\]]+)\]\]")

_EXTERNAL_SCHEME_RE = re.compile(r"^[a-z][a-z0-9+.\-]*:", re.IGNORECASE)


def _strip_code(md: str) -> str:
    md = _FENCED_BLOCK_RE.sub("", md)
    md = _INLINE_CODE_RE.sub("", md)
    return md


def _split_anchor(target: str) -> tuple[str, str | None]:
    if "#" in target:
        path, anchor = target.split("#", 1)
        return path, anchor
    return target, None


def _resolve_relative(target: str, source_path: str) -> str:
    if target.startswith("/"):
        return posixpath.normpath(target.lstrip("/"))
    source_dir = posixpath.dirname(source_path)
    if source_dir:
        return posixpath.normpath(posixpath.join(source_dir, target))
    return posixpath.normpath(target)


def _blank_spans(text: str, spans: list[tuple[int, int]]) -> str:
    if not spans:
        return text
    parts = []
    prev = 0
    for start, end in sorted(spans):
        parts.append(text[prev:start])
        parts.append(" " * (end - start))
        prev = end
    parts.append(text[prev:])
    return "".join(parts)


def extract_links(
    md_text: str,
    source_path: str | None = None,
    url_to_file_index: dict[str, str] | None = None,
) -> list[Link]:
    """Extract links from a markdown document.

    Args:
        md_text: Raw markdown content.
        source_path: Path of the source file (used to resolve relative links).
        url_to_file_index: Optional map of original URL → exported file path,
            used to localize Confluence/Jira links back to internal references.

    Returns:
        List of ``Link`` records in document order. ``target`` carries the
        canonical reference for the link's kind (file path for internal,
        full URL for external, page title for wikilink, filename or
        relative path for attachment); the ``#anchor`` portion is split
        out into ``anchor`` for all kinds.
    """
    if not md_text:
        return []

    cleaned = _strip_code(md_text)
    found: list[tuple[int, Link]] = []

    placeholder_spans: list[tuple[int, int]] = []
    for m in _ATTACHMENT_PLACEHOLDER_RE.finditer(cleaned):
        target = m.group(1).strip()
        found.append(
            (
                m.start(),
                Link(
                    kind="attachment",
                    target=target,
                    text=target,
                    anchor=None,
                    source_path=source_path,
                    resolved=None,
                ),
            )
        )
        placeholder_spans.append((m.start(), m.end()))

    masked = _blank_spans(cleaned, placeholder_spans)

    for m in _IMAGE_RE.finditer(masked):
        found.append(
            (
                m.start(),
                Link(
                    kind="attachment",
                    target=m.group(2),
                    text=m.group(1),
                    anchor=None,
                    source_path=source_path,
                    resolved=None,
                ),
            )
        )

    for m in _LINK_RE.finditer(masked):
        text = m.group(1)
        target_raw = m.group(2)
        path, anchor = _split_anchor(target_raw)
        if _EXTERNAL_SCHEME_RE.match(target_raw):
            if url_to_file_index and target_raw in url_to_file_index:
                # Localized: target carries the resolved file path so it has
                # the same meaning as a non-localized internal link.
                resolved_path = url_to_file_index[target_raw]
                found.append(
                    (
                        m.start(),
                        Link(
                            kind="internal",
                            target=resolved_path,
                            text=text,
                            anchor=anchor,
                            source_path=source_path,
                            resolved=resolved_path,
                        ),
                    )
                )
            else:
                found.append(
                    (
                        m.start(),
                        Link(
                            kind="external",
                            target=path,
                            text=text,
                            anchor=anchor,
                            source_path=source_path,
                            resolved=None,
                        ),
                    )
                )
            continue

        resolved = _resolve_relative(path, source_path) if source_path else None
        found.append(
            (
                m.start(),
                Link(
                    kind="internal",
                    target=path,
                    text=text,
                    anchor=anchor,
                    source_path=source_path,
                    resolved=resolved,
                ),
            )
        )

    for m in _WIKILINK_RE.finditer(masked):
        inner = m.group(1).strip()
        if not inner or inner.startswith("source/"):
            continue
        if "|" in inner:
            target, text = inner.split("|", 1)
            target = target.strip()
            text = text.strip()
        else:
            target = inner
            text = inner
        target_clean, anchor = _split_anchor(target)
        found.append(
            (
                m.start(),
                Link(
                    kind="wikilink",
                    target=target_clean.strip(),
                    text=text,
                    anchor=anchor,
                    source_path=source_path,
                    resolved=None,
                ),
            )
        )

    found.sort(key=lambda pair: pair[0])
    return [link for _, link in found]
