"""Source-kind gating pre-filter (#240).

Deterministic pre-filter that runs *before* the LLM verifier (#239),
catching obvious hallucination classes without an LLM call.

Three rules
-----------
1. **readme_claim_without_read** — paragraph mentions "README" but no
   ``read_file`` of a README file appears in the agent's tool trace.

2. **sql_claim_without_read** — paragraph contains SQL DDL/DML keywords
   (``CREATE TABLE``, ``SELECT``, ``INSERT``, ``UPDATE``, ``DELETE``,
   ``JOIN``, ``PRIMARY KEY``, ``NOT NULL``, ``REFERENCES``) but no
   ``read_file`` of a ``*.sql`` artifact appears in the trace.

3. **identifier_not_grounded** — paragraph contains CamelCase, snake_case,
   or UPPER_CASE identifier tokens that do not appear in the code graph
   (``known_symbols`` set) *and* are not present in the text of any
   ``read_file`` or ``grep`` result in the tool trace.

Per-rule modes (configurable via the ``mode`` dict)
----------------------------------------------------
``"flag"`` (default) — emit GateFlag, keep the paragraph.
``"strip"``           — emit GateFlag, remove paragraph from output.
``"verifier"``        — emit GateFlag with action="verifier", keep paragraph
                        (signals the LLM verifier to apply extra scrutiny).

Public API
----------
ToolCall       — tool name + args + result_text from the agent trace.
GateFlag       — per-paragraph flag record.
GateReport     — aggregate of all flags + stripped indices.
apply_gate()   — runs all three rules, returns filtered claims + report.

Dependency on #238
------------------
This module imports ``Citation`` and ``CitedClaim`` from
``citation_extractor`` (#238), which is implemented concurrently.
The types are well-defined from the test contract; no local stubs needed.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field

from app.core.wiki_content_writer.citation_extractor import CitedClaim

logger = logging.getLogger(__name__)


# ── ToolCall ─────────────────────────────────────────────────────────────────


@dataclass
class ToolCall:
    """A single tool invocation recorded in the agent's trace.

    Attributes
    ----------
    tool : str
        Tool name, e.g. ``"read_file"``, ``"grep"``, ``"get_signature"``.
    args : dict
        Arguments passed to the tool.
    result_text : str
        Textual output of the tool call.  Used to determine what was "seen"
        by the agent (identifier grounding, README / SQL file detection).
    """

    tool: str
    args: dict
    result_text: str


# ── GateFlag / GateReport ─────────────────────────────────────────────────────


@dataclass
class GateFlag:
    """A single pre-filter finding for one paragraph.

    Attributes
    ----------
    paragraph_index : int
        0-based index of the flagged paragraph.
    paragraph_text : str
        Full text of the flagged paragraph.
    rule : str
        Which rule fired.  One of:
        ``"readme_claim_without_read"``,
        ``"sql_claim_without_read"``,
        ``"identifier_not_grounded"``.
    detail : str
        Human-readable detail — e.g. the matching SQL keyword, the
        identifier name.
    action : str
        What the gate decided.  One of ``"flag"``, ``"strip"``,
        ``"verifier"``.
    """

    paragraph_index: int
    paragraph_text: str
    rule: str
    detail: str
    action: str


@dataclass
class GateReport:
    """Aggregate result of :func:`apply_gate`.

    Attributes
    ----------
    flags : list[GateFlag]
        All flags emitted by the three rules, in paragraph order.
    stripped_paragraphs : list[int]
        Indices of paragraphs that were removed from the returned claims
        because at least one rule fired with ``action="strip"``.
    """

    flags: list[GateFlag] = field(default_factory=list)
    stripped_paragraphs: list[int] = field(default_factory=list)


# ── Regex patterns ────────────────────────────────────────────────────────────

# README detection — word "readme" in any case (also README.md, README.rst etc.)
_README_WORD_RE = re.compile(r"\bREADME\b", re.IGNORECASE)

# README path pattern — matches readme variants in a file path
_README_PATH_RE = re.compile(r"(?i)readme(\.(md|rst|txt))?$")

# SQL keyword patterns — triggers rule 2
_SQL_KEYWORDS: list[tuple[re.Pattern[str], str]] = [
    (re.compile(r"\bCREATE\s+TABLE\b", re.IGNORECASE), "CREATE TABLE"),
    (re.compile(r"\bSELECT\b", re.IGNORECASE), "SELECT"),
    (re.compile(r"\bINSERT\b", re.IGNORECASE), "INSERT"),
    (re.compile(r"\bUPDATE\b", re.IGNORECASE), "UPDATE"),
    (re.compile(r"\bDELETE\b", re.IGNORECASE), "DELETE"),
    (re.compile(r"\bJOIN\b", re.IGNORECASE), "JOIN"),
    (re.compile(r"\bPRIMARY\s+KEY\b", re.IGNORECASE), "PRIMARY KEY"),
    (re.compile(r"\bNOT\s+NULL\b", re.IGNORECASE), "NOT NULL"),
    (re.compile(r"\bREFERENCES\b", re.IGNORECASE), "REFERENCES"),
]

# SQL file extension (case-insensitive)
_SQL_EXT_RE = re.compile(r"\.sql$", re.IGNORECASE)

# Identifier patterns for rule 3.
# Order matters — more specific patterns first.
#
# CamelCase: two or more title-case segments, e.g. WikiContentWriter.
# We require at least TWO uppercase-led segments to avoid flagging lone
# capitalised words (English proper nouns, sentence starts).
_CAMEL_CASE_RE = re.compile(r"\b[A-Z][a-z]+(?:[A-Z][a-z]+)+\b")

# snake_case function/variable: contains at least one underscore and
# consists of lowercase letters, digits, and underscores only.  Each
# underscore-delimited segment must be non-empty (so ``a_`` doesn't match).
_SNAKE_CASE_RE = re.compile(r"\b[a-z][a-z0-9]*(?:_[a-z0-9]+)+\b")

# UPPER_CASE env vars / constants: 3+ chars, contains at least one underscore.
_UPPER_CASE_RE = re.compile(r"\b[A-Z][A-Z0-9]*_[A-Z0-9_]+\b")

# Short tokens to skip — all-caps tokens of 2 chars or less are too noisy
# (HTTP codes, abbreviations).
_SHORT_UPPER_RE = re.compile(r"^[A-Z]{1,2}$")


# ── Internal helpers ──────────────────────────────────────────────────────────


def _trace_has_readme_read(trace: list[ToolCall]) -> bool:
    """Return True if any ``read_file`` call in the trace targets a README."""
    for call in trace:
        if call.tool == "read_file":
            path = call.args.get("path", "")
            if isinstance(path, str) and _README_PATH_RE.search(path):
                return True
    return False


def _trace_has_sql_read(trace: list[ToolCall]) -> bool:
    """Return True if any ``read_file`` call in the trace targets a .sql file."""
    for call in trace:
        if call.tool == "read_file":
            path = call.args.get("path", "")
            if isinstance(path, str) and _SQL_EXT_RE.search(path):
                return True
    return False


def _build_trace_text(trace: list[ToolCall]) -> str:
    """Concatenate the result_text of all read_file and grep calls."""
    parts: list[str] = []
    for call in trace:
        if call.tool in ("read_file", "grep"):
            parts.append(call.result_text)
    return "\n".join(parts)


def _extract_identifiers(text: str) -> list[tuple[str, str]]:
    """Extract (identifier, kind) pairs from *text*.

    Returns a list of ``(token, "camel"|"snake"|"upper")`` tuples.
    Short or noisy tokens are filtered here to reduce false-positive rates.
    """
    tokens: list[tuple[str, str]] = []
    seen: set[str] = set()

    def _add(m: re.Match[str], kind: str) -> None:
        tok = m.group(0)
        if tok not in seen:
            seen.add(tok)
            tokens.append((tok, kind))

    for m in _CAMEL_CASE_RE.finditer(text):
        _add(m, "camel")
    for m in _SNAKE_CASE_RE.finditer(text):
        _add(m, "snake")
    for m in _UPPER_CASE_RE.finditer(text):
        tok = m.group(0)
        # Filter out very short all-caps tokens — too many false positives
        if not _SHORT_UPPER_RE.match(tok) and tok not in seen:
            seen.add(tok)
            tokens.append((tok, "upper"))

    return tokens


# ── apply_gate ────────────────────────────────────────────────────────────────

_DEFAULT_MODE: dict[str, str] = {
    "readme": "flag",
    "sql": "flag",
    "identifier": "flag",
}

_VALID_ACTIONS: frozenset[str] = frozenset({"flag", "strip", "verifier"})
_VALID_RULES: frozenset[str] = frozenset(_DEFAULT_MODE.keys())


def apply_gate(
    cited_claims: list[CitedClaim],
    tool_trace: list[ToolCall],
    *,
    known_symbols: set[str] | None = None,
    mode: dict[str, str] | None = None,
) -> tuple[list[CitedClaim], GateReport]:
    """Run the three source-kind pre-filter rules over *cited_claims*.

    Parameters
    ----------
    cited_claims :
        Paragraphs produced by the writer agent, with their extracted
        citations (output of :func:`extract_citations` from #238).
    tool_trace :
        Ordered record of every tool call made by the writer agent during
        page generation.  Only ``read_file`` and ``grep`` calls contribute
        to identifier grounding; only ``read_file`` paths matter for
        README / SQL checks.
    known_symbols :
        Set of symbol names from the code graph's name index.  Identifiers
        found in this set are considered grounded.  Pass ``None`` to rely
        solely on the tool trace for grounding (identifier rule still runs).
    mode :
        Per-rule action override.  Keys: ``"readme"``, ``"sql"``,
        ``"identifier"``.  Values: ``"flag"`` (default), ``"strip"``,
        ``"verifier"``.  Unspecified rules default to ``"flag"``.

    Returns
    -------
    (kept_claims, report) :
        ``kept_claims`` — paragraphs not stripped by any ``"strip"``-mode rule.
        ``report`` — all flags and the indices of stripped paragraphs.
    """
    if mode:
        for rule_key, action in mode.items():
            if rule_key not in _VALID_RULES:
                raise ValueError(
                    f"Unknown gate rule {rule_key!r}; expected one of {sorted(_VALID_RULES)}"
                )
            if action not in _VALID_ACTIONS:
                raise ValueError(
                    f"Invalid gate action {action!r} for rule {rule_key!r}; "
                    f"expected one of {sorted(_VALID_ACTIONS)}"
                )
    effective_mode = {**_DEFAULT_MODE, **(mode or {})}
    readme_mode = effective_mode["readme"]
    sql_mode = effective_mode["sql"]
    identifier_mode = effective_mode["identifier"]

    # Pre-compute expensive trace queries once
    has_readme_read = _trace_has_readme_read(tool_trace)
    has_sql_read = _trace_has_sql_read(tool_trace)
    trace_text = _build_trace_text(tool_trace)

    report = GateReport()
    kept: list[CitedClaim] = []

    for claim in cited_claims:
        text = claim.paragraph_text
        para_idx = claim.paragraph_index
        stripped_this = False
        flags_this: list[GateFlag] = []

        # ── Rule 1: README claim without README read ──────────────────────
        if _README_WORD_RE.search(text) and not has_readme_read:
            flag = GateFlag(
                paragraph_index=para_idx,
                paragraph_text=text,
                rule="readme_claim_without_read",
                detail="README",
                action=readme_mode,
            )
            flags_this.append(flag)
            if readme_mode == "strip":
                stripped_this = True

        # ── Rule 2: SQL claim without .sql read ───────────────────────────
        if not has_sql_read:
            for sql_re, keyword in _SQL_KEYWORDS:
                if sql_re.search(text):
                    flag = GateFlag(
                        paragraph_index=para_idx,
                        paragraph_text=text,
                        rule="sql_claim_without_read",
                        detail=keyword,
                        action=sql_mode,
                    )
                    flags_this.append(flag)
                    if sql_mode == "strip":
                        stripped_this = True
                    # One flag per paragraph is sufficient to signal the rule
                    break

        # ── Rule 3: Identifier not grounded ───────────────────────────────
        identifiers = _extract_identifiers(text)
        for token, _kind in identifiers:
            # Grounded if in known_symbols set
            if known_symbols is not None and token in known_symbols:
                continue
            # Grounded if token appears verbatim in any read_file/grep output
            if token in trace_text:
                continue
            flag = GateFlag(
                paragraph_index=para_idx,
                paragraph_text=text,
                rule="identifier_not_grounded",
                detail=token,
                action=identifier_mode,
            )
            flags_this.append(flag)
            if identifier_mode == "strip":
                stripped_this = True
                break  # one strip per paragraph is enough

        report.flags.extend(flags_this)

        if stripped_this:
            report.stripped_paragraphs.append(para_idx)
        else:
            kept.append(claim)

    return kept, report
