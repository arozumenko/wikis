"""Helpers for #116 incremental wiki regeneration.

Pure functions shared by both storage backends, the parser layer, and the
wiki generation agent. Kept dependency-free so they can be unit-tested
without spinning up a DB or LLM.

Three concerns live here:

* ``compute_content_hash`` — stable SHA-256 of a node's source text, used
  by change detection to decide whether a node has changed since the last
  index.

* ``compute_page_id`` — deterministic 16-hex-char page identifier derived
  from the cluster coordinates and primary symbol. Lets wiki page IDs
  survive across regenerations as long as the underlying cluster + symbol
  are stable.

* ``compute_anchor_slug`` — deterministic URL-safe slug from a page title,
  with collision resolution scoped to a wiki.
"""

from __future__ import annotations

import hashlib
import re
import unicodedata
from collections.abc import Iterable

__all__ = [
    "PAGE_ID_LENGTH",
    "compute_anchor_slug",
    "compute_content_hash",
    "compute_page_id",
    "normalize_for_hash",
]


# 16 hex chars = 64 bits of entropy. Collisions become likely at ~2^32 pages
# per wiki, which is well above any realistic wiki size.
PAGE_ID_LENGTH = 16


def normalize_for_hash(text: str) -> str:
    """Strip trailing whitespace from each line and collapse line endings.

    The goal is to make ``content_hash`` invariant to incidental whitespace
    edits that don't change program meaning (editor newline-style changes,
    trailing-whitespace cleanup, etc.) while still flagging real edits.
    """
    if not text:
        return ""
    # Normalize line endings, strip trailing whitespace per line, drop
    # leading/trailing blank lines.
    lines = text.replace("\r\n", "\n").replace("\r", "\n").split("\n")
    return "\n".join(line.rstrip() for line in lines).strip("\n")


def compute_content_hash(text: str) -> str:
    """Return the SHA-256 hex digest of ``normalize_for_hash(text)``.

    Returns an empty string for empty input so callers can store ``NULL``
    without a special-case branch.
    """
    normalized = normalize_for_hash(text)
    if not normalized:
        return ""
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def compute_page_id(
    wiki_id: str,
    macro_cluster: int | None,
    micro_cluster: int | None,
    primary_symbol_id: str | None,
    title: str,
) -> str:
    """Deterministic page identifier for the 'stable_v1' id_scheme.

    The inputs are the four signals that uniquely position a page in a
    wiki: (cluster_macro, cluster_micro, primary_symbol, title). ``title``
    is normalized so trivial editorial tweaks ("API Overview" →
    "Api overview") don't shift the ID — only an intentional retitle does.

    **Stability caveat (until PR 4 of #116 lands):** ``macro_cluster`` and
    ``micro_cluster`` are raw Leiden partition IDs. Leiden is deterministic
    given a fixed seed *and* an unchanged graph, but cluster IDs themselves
    are assigned in arbitrary partition order. Even small graph changes can
    therefore renumber every cluster and shift every page_id. PR 4 lifts
    this via Jaccard mapping of new clusters onto old IDs. Until then,
    callers relying on cross-regen page_id stability should treat IDs as
    stable *within* a single repo state — not across edits to the repo.
    """
    parts = [
        wiki_id or "",
        str(macro_cluster) if macro_cluster is not None else "",
        str(micro_cluster) if micro_cluster is not None else "",
        primary_symbol_id or "",
        _normalize_title(title),
    ]
    payload = "|".join(parts).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()[:PAGE_ID_LENGTH]


def compute_anchor_slug(title: str, existing_slugs: Iterable[str] = ()) -> str:
    """URL-safe slug for ``title``, with numeric suffix on collision.

    Args:
        title: Human-readable page title. May contain unicode.
        existing_slugs: Slugs already taken in the same wiki. The returned
            slug is guaranteed to be absent from this collection.

    Examples:
        >>> compute_anchor_slug("API Overview")
        'api-overview'
        >>> compute_anchor_slug("API Overview", {"api-overview"})
        'api-overview-2'
    """
    base = _slugify(title) or "page"
    taken = set(existing_slugs)
    if base not in taken:
        return base
    counter = 2
    while f"{base}-{counter}" in taken:
        counter += 1
    return f"{base}-{counter}"


# ---------------------------------------------------------------------------
# internals
# ---------------------------------------------------------------------------

# Matches one-or-more characters that are *not* ASCII alphanumeric.
_NON_SLUG_RE = re.compile(r"[^a-z0-9]+")


def _normalize_title(title: str) -> str:
    """Lowercase + collapse whitespace; used as a stable hash input."""
    return " ".join((title or "").lower().split())


def _slugify(title: str) -> str:
    """Lowercase ASCII slug. Strips diacritics, collapses runs of non-alnum."""
    if not title:
        return ""
    # NFKD splits e.g. "é" into "e" + combining accent, then we drop combiners.
    decomposed = unicodedata.normalize("NFKD", title)
    ascii_only = decomposed.encode("ascii", "ignore").decode("ascii").lower()
    return _NON_SLUG_RE.sub("-", ascii_only).strip("-")
