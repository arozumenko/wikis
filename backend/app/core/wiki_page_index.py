"""Wiki page index — parses generated wiki pages and builds an in-memory wikilink graph.

This module is pure Python: no LLM, no FAISS, no NetworkX.
"""

from __future__ import annotations

import asyncio
import logging
import re
from collections import OrderedDict, deque
from dataclasses import dataclass

from app.storage.base import ArtifactStorage

logger = logging.getLogger(__name__)

# Matches [[Page Title]] and [[Page Title|Display Text]]
_WIKILINK_RE = re.compile(r"\[\[([^\]]+)\]\]")


@dataclass
class PageMeta:
    """Metadata extracted from a single wiki page.

    Attributes:
        title: Page title from YAML frontmatter.
        description: Page description from YAML frontmatter (empty string if absent).
        file_path: Relative artifact path within the wiki storage bucket.
    """

    title: str
    description: str
    file_path: str


class WikiPageIndex:
    """In-memory wikilink graph built from a wiki's stored markdown pages.

    On construction, reads ALL ``.md`` files for a wiki from artifact storage,
    parses their YAML frontmatter, and builds three data structures:

    * ``pages`` — ``dict[normalized_title, PageMeta]``
    * ``_forward`` — ``dict[title, set[title]]`` — edges *from* each page
    * ``_backward`` — ``dict[title, set[title]]`` — edges *to* each page (backlinks)

    Wikilink parsing rules:

    * ``[[Page Title]]`` → edge to "Page Title"
    * ``[[Page Title|Display Text]]`` → edge to "Page Title" (target only)
    * ``[[source/…]]`` and ``[[source/…|display]]`` → *excluded* (source-code links)

    Title normalization: strip surrounding whitespace, preserve case.

    Args:
        wiki_id: Identifier of the wiki (used to scope artifact listing).
        storage: ArtifactStorage instance to read pages from.
    """

    _MAX_NEIGHBORS = 50

    def __init__(self, wiki_id: str, storage: ArtifactStorage) -> None:
        self._wiki_id = wiki_id
        self._storage = storage
        self.pages: dict[str, PageMeta] = {}
        self._forward: dict[str, set[str]] = {}
        self._backward: dict[str, set[str]] = {}

    # ------------------------------------------------------------------
    # Factory / loader
    # ------------------------------------------------------------------

    @classmethod
    async def build(cls, wiki_id: str, storage: ArtifactStorage) -> WikiPageIndex:
        """Construct and populate a WikiPageIndex from artifact storage.

        Args:
            wiki_id: Wiki identifier used as the storage prefix.
            storage: ArtifactStorage instance.

        Returns:
            A fully populated WikiPageIndex.
        """
        index = cls(wiki_id, storage)
        await index._load()
        return index

    async def _load(self) -> None:
        """Read all .md artifacts and populate the index."""
        artifacts = await self._storage.list_artifacts("wiki_artifacts", prefix=self._wiki_id)
        md_artifacts = [a for a in artifacts if a.endswith(".md") and "wiki_pages/" in a]

        logger.debug("WikiPageIndex: loading %d pages for wiki %s", len(md_artifacts), self._wiki_id)

        # First pass: parse frontmatter and register all pages.
        raw_contents: dict[str, str] = {}
        for artifact_key in md_artifacts:
            try:
                raw_bytes = await self._storage.download("wiki_artifacts", artifact_key)
                content = raw_bytes.decode("utf-8") if isinstance(raw_bytes, bytes) else raw_bytes
            except Exception:
                logger.warning("WikiPageIndex: failed to read %s", artifact_key)
                continue

            title, description = _parse_frontmatter(content)
            if not title:
                # Fall back to filename stem if no title in frontmatter.
                filename = artifact_key.rsplit("/", 1)[-1]
                title = filename[: -len(".md")] if filename.endswith(".md") else filename

            normalized = title.strip()
            if normalized in self.pages:
                existing_path = self.pages[normalized].file_path
                logger.warning(
                    "WikiPageIndex: title collision for '%s' — '%s' overwritten by '%s'",
                    normalized,
                    existing_path,
                    artifact_key,
                )
            self.pages[normalized] = PageMeta(
                title=normalized,
                description=description,
                file_path=artifact_key,
            )
            raw_contents[normalized] = content

        # Second pass: build forward and backward adjacency maps.
        for title, content in raw_contents.items():
            targets = _extract_wikilink_targets(content)
            # Normalize targets in the same way.
            targets = {t.strip() for t in targets}
            self._forward[title] = targets
            for target in targets:
                self._backward.setdefault(target, set()).add(title)

        logger.debug(
            "WikiPageIndex: indexed %d pages, %d forward edges for wiki %s",
            len(self.pages),
            sum(len(v) for v in self._forward.values()),
            self._wiki_id,
        )

    # ------------------------------------------------------------------
    # Public query API
    # ------------------------------------------------------------------

    def neighbors(self, title: str, hop_depth: int = 1) -> list[PageMeta]:
        """Return pages reachable from *title* within *hop_depth* BFS hops.

        Args:
            title: Normalized page title to start from.
            hop_depth: Maximum number of hops. Defaults to 1.

        Returns:
            List of PageMeta for reachable pages, excluding the seed page.
            At most 50 results are returned regardless of graph size.
        """
        if hop_depth < 1:
            return []

        normalized = title.strip()
        if normalized not in self.pages:
            return []

        visited: set[str] = {normalized}
        result: list[PageMeta] = []
        # Queue entries: (page_title, current_depth)
        queue: deque[tuple[str, int]] = deque()

        for neighbor in self._forward.get(normalized, set()):
            if neighbor not in visited:
                queue.append((neighbor, 1))
                visited.add(neighbor)

        while queue and len(result) < self._MAX_NEIGHBORS:
            current, depth = queue.popleft()
            if current in self.pages:
                result.append(self.pages[current])
            if depth < hop_depth and len(result) < self._MAX_NEIGHBORS:
                for neighbor in self._forward.get(current, set()):
                    if neighbor not in visited:
                        visited.add(neighbor)
                        queue.append((neighbor, depth + 1))

        return result

    def backlinks(self, title: str) -> list[str]:
        """Return titles of pages that link to *title*.

        Args:
            title: Normalized page title to look up.

        Returns:
            List of page titles that contain a wikilink pointing to *title*.
        """
        return list(self._backward.get(title.strip(), set()))

    def get_description(self, title: str) -> str:
        """Return the description for *title*, or an empty string if not found.

        Args:
            title: Normalized page title.

        Returns:
            Description string, or ``""`` when the page is absent.
        """
        meta = self.pages.get(title.strip())
        return meta.description if meta is not None else ""


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _parse_frontmatter(content: str) -> tuple[str, str]:
    """Extract *title* and *description* from YAML frontmatter.

    Only the first ``---`` block is considered.  Values are extracted with a
    simple line-by-line scan rather than a full YAML parser — no external dep.

    Args:
        content: Raw markdown content, possibly starting with ``---``.

    Returns:
        ``(title, description)`` — either may be an empty string.
    """
    if not content.startswith("---"):
        return "", ""

    # Use "\n---\n" to avoid matching "---" embedded in body text; also accept
    # "\n---" at end-of-file (no trailing newline) via the regex fallback.
    match = re.search(r"\n---(?:\n|$)", content[3:])
    if not match:
        return "", ""
    end = 3 + match.start()

    frontmatter_block = content[3:end]

    title = ""
    description = ""
    for line in frontmatter_block.splitlines():
        if line.startswith("title:") and not title:
            title = _strip_yaml_value(line[len("title:") :])
        elif line.startswith("description:") and not description:
            description = _strip_yaml_value(line[len("description:") :])

    return title, description


def _strip_yaml_value(raw: str) -> str:
    """Strip surrounding whitespace and optional enclosing quotes from a YAML scalar."""
    value = raw.strip()
    if len(value) >= 2 and value[0] == '"' and value[-1] == '"':
        value = value[1:-1].replace('\\"', '"')
    elif len(value) >= 2 and value[0] == "'" and value[-1] == "'":
        # Note: YAML single-quoted scalars use '' to escape a literal single
        # quote.  That escaping is NOT implemented here — callers should be
        # aware that embedded '' sequences will be preserved as-is.
        value = value[1:-1]
    return value


class WikiPageIndexCache:
    """LRU in-memory cache of WikiPageIndex instances, keyed by wiki_id.

    Indexes are built lazily on first access and evicted from memory when the
    number of loaded wikis exceeds ``max_wikis``.  Evicted indexes are simply
    rebuilt from artifact storage on the next access — there is no disk
    persistence.

    Per-wiki async locks prevent duplicate builds under concurrent requests.

    Args:
        storage: ArtifactStorage instance passed to WikiPageIndex.build().
        max_wikis: Maximum number of indexes to keep in memory (LRU eviction).
    """

    def __init__(self, storage, max_wikis: int = 50) -> None:
        self._storage = storage
        self._max_wikis = max_wikis
        self._indexes: OrderedDict[str, WikiPageIndex] = OrderedDict()
        self._locks: dict[str, asyncio.Lock] = {}

    def _get_lock(self, wiki_id: str) -> asyncio.Lock:
        return self._locks.setdefault(wiki_id, asyncio.Lock())

    async def get(self, wiki_id: str) -> WikiPageIndex:
        """Return the WikiPageIndex for *wiki_id*, building it if necessary.

        On cache hit the entry is moved to the most-recently-used position.
        On cache miss a per-wiki lock is acquired before building to prevent
        duplicate work from concurrent callers; a double-check after acquiring
        the lock handles the case where another coroutine completed the build
        while this one was waiting.

        Args:
            wiki_id: Wiki identifier.

        Returns:
            A fully populated WikiPageIndex.
        """
        # Fast path: already cached — mark as recently used and return.
        if wiki_id in self._indexes:
            self._indexes.move_to_end(wiki_id)
            return self._indexes[wiki_id]

        # Slow path: acquire per-wiki lock, then double-check.
        async with self._get_lock(wiki_id):
            if wiki_id in self._indexes:
                self._indexes.move_to_end(wiki_id)
                return self._indexes[wiki_id]

            logger.debug("WikiPageIndexCache: building index for wiki %s", wiki_id)
            index = await WikiPageIndex.build(wiki_id, self._storage)

            # LRU eviction before inserting.
            while len(self._indexes) >= self._max_wikis:
                evicted_id, _ = self._indexes.popitem(last=False)
                logger.info(
                    "WikiPageIndexCache: evicted wiki %s (LRU, capacity %d)",
                    evicted_id,
                    self._max_wikis,
                )

            self._indexes[wiki_id] = index
            return index


def _extract_wikilink_targets(content: str) -> set[str]:
    """Return the set of link *targets* from all ``[[…]]`` patterns in *content*.

    Rules:
    * ``[[Page Title]]`` → target is "Page Title"
    * ``[[Page Title|Display]]`` → target is "Page Title"
    * ``[[source/…]]`` and ``[[source/…|display]]`` → excluded (source-code links)

    Args:
        content: Markdown page body.

    Returns:
        Set of target title strings (may include unknown pages).
    """
    targets: set[str] = set()
    for match in _WIKILINK_RE.finditer(content):
        inner = match.group(1)
        # Split on pipe to separate target from display text.
        target = inner.split("|", 1)[0]
        # Exclude source-code links.
        if target.startswith("source/"):
            continue
        targets.add(target)
    return targets
