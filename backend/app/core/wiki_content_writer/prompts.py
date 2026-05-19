"""Citation prompt rules for the wiki content writer (#238).

This module defines the **citation contract** that the writer agent must honour.
Rules are expressed as named string constants so downstream consumers (#243,
which wires them into the LLM system prompt) can compose them without coupling
to the internal text of each rule.

Citation format
---------------
Inline citations appear immediately after the claim they support::

    ...claim text [path:lo-hi].
    ...claim text [path:N].          # single-line, lo == hi == N
    ...claim text [a.py:1-2, b.py:5].  # comma-separated, multiple sources

``path`` is a repo-relative POSIX path.  ``lo`` and ``hi`` are 1-indexed line
numbers.  For a single-line span ``lo == hi == N`` and the short form
``[path:N]`` is preferred.  Citations are emitted as close to the claim as
possible (end of sentence or end of the relevant clause, before the
punctuation or after the final word — either is acceptable).

Prompt rules
------------
``CITATION_RULE_EVERY_CLAIM``       — all substantive claims need a citation.
``CITATION_RULE_NO_README_BLIND``   — README content must be verified by a
                                       ``read_file`` call before citing it.
``CITATION_RULE_NO_HALLUCINATED_SYMBOLS`` — only name identifiers that appear
                                       in the tool trace.

``CITATION_PROMPT_RULES``           — ordered tuple of all three rules.
                                       #243 injects this into the system prompt.

Out of scope in this PR (#238)
------------------------------
- Wiring rules into the LLM call → #243
- Citation verifier (LLM pass) → #239
- Source-kind gating (README/SQL pre-filter) → #240
"""

from __future__ import annotations

# ── Citation format specification ────────────────────────────────────────────

#: Canonical format for a single-line citation token.
#: Substitute ``{path}`` with the repo-relative file path and ``{line}`` with
#: the 1-indexed line number.
CITATION_FORMAT_SINGLE_LINE: str = "[{path}:{line}]"

#: Canonical format for a line-range citation token.
#: Substitute ``{path}``, ``{lo}``, and ``{hi}`` accordingly.
CITATION_FORMAT_RANGE: str = "[{path}:{lo}-{hi}]"

#: Human-readable description of the citation format, suitable for inclusion
#: in a prompt or error message.
CITATION_FORMAT_DESCRIPTION: str = (
    "Inline citation format: [path:N] for a single line or [path:lo-hi] for a "
    "range, where path is a repo-relative POSIX path and N/lo/hi are 1-indexed "
    "line numbers. Multiple citations on the same claim are comma-separated "
    "inside one bracket pair: [a.py:1-2, b.py:5]."
)

# ── Individual prompt rules ───────────────────────────────────────────────────

#: Rule 1 — every substantive claim must be grounded by a citation.
CITATION_RULE_EVERY_CLAIM: str = (
    "Every claim must carry a citation pointing at a span from your tool trace. "
    "Append the citation token immediately after the claim, before the closing "
    "punctuation: '...validates the token [api/auth.py:42-58].' "
    "A paragraph without any citation will be stripped by the verifier."
)

#: Rule 2 — README content must be verified against a ``read_file`` call.
CITATION_RULE_NO_README_BLIND: str = (
    "Do not reproduce README content unless you have called read_file on the "
    "README and the cited line span matches the content you are describing. "
    "Paraphrasing README text without a matching citation is treated as an "
    "uncited claim and will be stripped."
)

#: Rule 3 — identifiers must appear in the tool trace.
CITATION_RULE_NO_HALLUCINATED_SYMBOLS: str = (
    "Do not name a function, class, or environment variable that did not appear "
    "in your tool results (read_file, get_signature, get_callers, get_callees, "
    "grep, list_doc_chunks). If you are uncertain whether a symbol exists, call "
    "get_signature or grep first."
)

# ── Composite constant ────────────────────────────────────────────────────────

#: All citation rules as an ordered tuple.  Inject into the writer system
#: prompt in #243 via ``"\\n".join(CITATION_PROMPT_RULES)``.
CITATION_PROMPT_RULES: tuple[str, ...] = (
    CITATION_RULE_EVERY_CLAIM,
    CITATION_RULE_NO_README_BLIND,
    CITATION_RULE_NO_HALLUCINATED_SYMBOLS,
)
