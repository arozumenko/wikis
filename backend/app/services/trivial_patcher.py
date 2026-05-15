"""Trivial-regime citation patcher for #116 PR 3.

When the only changes touching a page are ``MOVED`` (same node_id,
same content_hash, different ``rel_path``), the wiki prose itself stays
correct — only the file paths embedded in ``<code_context path="...">``
blocks need to change. This module does that rewrite without involving
an LLM.

The patcher is **deterministic** and **idempotent**: running it twice
on the same input produces the same output.

Contract:

* Input: page markdown + list of (old_path → new_path) tuples derived
  from :class:`NodeChange` ``MOVED`` entries.
* Output: revised page markdown + a new content_hash (from
  ``compute_content_hash`` on the rewritten body).
* Side effect (handled by the caller): re-upsert the ``wiki_pages``
  row with the new content_hash; ``page_symbols`` rows already point
  at node_ids, not paths, so they need no update.
"""

from __future__ import annotations

import logging
import re
from collections.abc import Iterable
from dataclasses import dataclass

from app.core.storage.incremental import compute_content_hash
from app.services.change_detector import ChangeKind, NodeChange

logger = logging.getLogger(__name__)


# Match <code_context path="..."> and <code_context path='...'>
# Path is captured in group 1 (double-quoted) or group 2 (single-quoted).
# `re.DOTALL` not needed — these attributes don't span newlines.
_CODE_CONTEXT_PATH_RE = re.compile(
    r'<code_context\b([^>]*?)\bpath\s*=\s*(?:"([^"]*)"|\'([^\']*)\')',
)


@dataclass(frozen=True)
class TrivialPatch:
    """Result of patching one page in the trivial regime."""

    page_id: str
    new_content: str
    new_content_hash: str
    paths_rewritten: int


def patch_trivial_page(
    page_id: str,
    content: str,
    changes: Iterable[NodeChange],
) -> TrivialPatch:
    """Rewrite ``<code_context path="old">`` blocks per the MOVED changes.

    Args:
        page_id: the wiki page being patched (for telemetry only).
        content: current page markdown body.
        changes: the :class:`NodeChange` set for this page. Only
            ``MOVED`` kinds are consumed; the patcher ignores other
            kinds even if they leak through (defensive — the classifier
            should already have routed those elsewhere).

    Returns:
        :class:`TrivialPatch` with the rewritten content and its hash.
        ``paths_rewritten`` counts how many ``code_context`` paths
        actually changed. When zero, the content body is byte-identical
        to the input.
    """
    path_remap: dict[str, str] = {}
    for change in changes:
        if change.kind is not ChangeKind.MOVED:
            continue
        if change.old_path and change.new_path and change.old_path != change.new_path:
            path_remap[change.old_path] = change.new_path

    if not path_remap:
        return TrivialPatch(
            page_id=page_id,
            new_content=content,
            new_content_hash=compute_content_hash(content),
            paths_rewritten=0,
        )

    rewritten_count = 0

    def _replace(match: re.Match[str]) -> str:
        nonlocal rewritten_count
        attrs_prefix, dq_path, sq_path = match.group(1), match.group(2), match.group(3)
        old_path = dq_path if dq_path is not None else sq_path
        new_path = path_remap.get(old_path)
        if new_path is None or new_path == old_path:
            return match.group(0)
        rewritten_count += 1
        # Preserve the original quoting style.
        if dq_path is not None:
            return f'<code_context{attrs_prefix}path="{new_path}"'
        return f"<code_context{attrs_prefix}path='{new_path}'"

    new_content = _CODE_CONTEXT_PATH_RE.sub(_replace, content)

    if rewritten_count:
        logger.info(
            "[trivial_patcher] page=%s rewrote %d code_context paths",
            page_id,
            rewritten_count,
        )

    return TrivialPatch(
        page_id=page_id,
        new_content=new_content,
        new_content_hash=compute_content_hash(new_content),
        paths_rewritten=rewritten_count,
    )
