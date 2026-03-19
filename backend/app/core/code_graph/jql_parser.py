"""
JQL (JSON Query Language) parser for structured code graph queries (SPEC-2).

Parses a mini query language into executable clauses that can be dispatched
to FTS5 index queries, SQL WHERE filters, and graph traversals.

Syntax::

    field:value              — exact/prefix match on field
    field:"multi word"       — quoted value
    field:glob*              — fnmatch glob pattern
    field:>N                 — numeric comparison (>, <, >=, <=)
    clause1 AND clause2      — explicit boolean AND
    clause1 OR clause2       — boolean OR
    clause1 clause2          — implicit AND (space-separated)

Supported Fields::

    type        — symbol_type (class, function, etc.)
    layer       — architectural layer (core_type, public_api, etc.)
    file        — rel_path with glob support
    name        — symbol name (exact or prefix)
    text        — FTS5 concept search
    related     — graph BFS from named symbol
    dir         — direction filter for related (outgoing, incoming, both)
    has_rel     — edge type existence filter
    connections — min/max connection count (>, <, >=, <=)
    limit       — max results

Example queries::

    type:class file:src/auth/*
    related:"BaseHandler" dir:incoming has_rel:inherits
    type:class connections:>10
    layer:core_type text:authentication
    type:class OR type:interface file:src/*

Usage::

    from .jql_parser import parse_jql, JQLQuery

    query = parse_jql('type:class file:src/auth/* text:authentication')
    # query.index_clauses  → clauses handled by FTS5/SQL
    # query.post_filters   → clauses requiring graph traversal
    # query.limit          → result cap
"""

import logging
import re
from dataclasses import dataclass, field
from enum import Enum
from fnmatch import fnmatch

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Clause types
# ---------------------------------------------------------------------------


class ClauseOp(Enum):
    """Comparison operators for JQL clauses."""

    EQ = "eq"  # exact match or glob
    GT = "gt"  # >
    GTE = "gte"  # >=
    LT = "lt"  # <
    LTE = "lte"  # <=
    MATCH = "match"  # FTS5 MATCH (text search)


@dataclass
class JQLClause:
    """A single parsed JQL clause (field:op:value)."""

    field: str
    op: ClauseOp
    value: str
    raw: str = ""  # original text for error messages

    def matches_value(self, candidate: str) -> bool:
        """Check if a candidate string matches this clause.

        For EQ: supports exact match and fnmatch glob patterns.
        For MATCH: simple substring check (FTS5 handles the real matching).
        """
        if self.op == ClauseOp.EQ:
            val = self.value.lower()
            cand = candidate.lower()
            if "*" in val or "?" in val or "[" in val:
                return fnmatch(cand, val)
            return cand == val
        if self.op == ClauseOp.MATCH:
            return self.value.lower() in candidate.lower()
        return False

    def matches_numeric(self, candidate: int) -> bool:
        """Check if a numeric candidate satisfies a comparison clause."""
        try:
            threshold = int(self.value)
        except (ValueError, TypeError):
            return False
        if self.op == ClauseOp.GT:
            return candidate > threshold
        if self.op == ClauseOp.GTE:
            return candidate >= threshold
        if self.op == ClauseOp.LT:
            return candidate < threshold
        if self.op == ClauseOp.LTE:
            return candidate <= threshold
        if self.op == ClauseOp.EQ:
            return candidate == threshold
        return False


# ---------------------------------------------------------------------------
# Parsed query
# ---------------------------------------------------------------------------

# Fields that can be pushed down to FTS5/SQL index queries
INDEX_FIELDS = frozenset({"type", "layer", "file", "name", "text"})

# Fields that require graph-level post-filtering
POST_FILTER_FIELDS = frozenset({"related", "dir", "has_rel", "connections"})


@dataclass
class JQLQuery:
    """Parsed JQL query split into index-friendly and post-filter clauses.

    Attributes:
        index_clauses: Clauses that can be executed via FTS5/SQL
            (type, layer, file, name, text).
        post_filters: Clauses requiring graph traversal
            (related, dir, has_rel, connections).
        limit: Maximum results (from ``limit:N`` clause, default 20).
        is_or: If True, index clauses are combined with OR instead of AND.
        raw_query: Original query string for debugging.
    """

    index_clauses: list[JQLClause] = field(default_factory=list)
    post_filters: list[JQLClause] = field(default_factory=list)
    limit: int = 20
    is_or: bool = False
    raw_query: str = ""

    @property
    def has_text_clause(self) -> bool:
        """Whether the query includes a ``text:`` FTS5 full-text clause."""
        return any(c.field == "text" for c in self.index_clauses)

    @property
    def text_value(self) -> str | None:
        """Return the ``text:`` value, or None."""
        for c in self.index_clauses:
            if c.field == "text":
                return c.value
        return None

    @property
    def type_values(self) -> list[str]:
        """Return all ``type:`` values."""
        return [c.value.lower() for c in self.index_clauses if c.field == "type"]

    @property
    def layer_value(self) -> str | None:
        """Return the ``layer:`` value, or None."""
        for c in self.index_clauses:
            if c.field == "layer":
                return c.value.lower()
        return None

    @property
    def file_value(self) -> str | None:
        """Return the ``file:`` value (path glob), or None."""
        for c in self.index_clauses:
            if c.field == "file":
                return c.value
        return None

    @property
    def name_value(self) -> str | None:
        """Return the ``name:`` value, or None."""
        for c in self.index_clauses:
            if c.field == "name":
                return c.value
        return None

    @property
    def related_value(self) -> str | None:
        """Return the ``related:`` symbol name, or None."""
        for c in self.post_filters:
            if c.field == "related":
                return c.value
        return None

    @property
    def direction_value(self) -> str:
        """Return the ``dir:`` value, defaulting to 'both'."""
        for c in self.post_filters:
            if c.field == "dir":
                return c.value.lower()
        return "both"

    @property
    def has_rel_values(self) -> list[str]:
        """Return all ``has_rel:`` edge type filters."""
        return [c.value.lower() for c in self.post_filters if c.field == "has_rel"]

    @property
    def connections_clause(self) -> JQLClause | None:
        """Return the ``connections:`` comparison clause, or None."""
        for c in self.post_filters:
            if c.field == "connections":
                return c
        return None

    @property
    def is_empty(self) -> bool:
        """Whether the query has no effective clauses."""
        return not self.index_clauses and not self.post_filters


# ---------------------------------------------------------------------------
# Tokenizer
# ---------------------------------------------------------------------------

# Regex for a single clause: field, colon, optional comparison op, value (possibly quoted)
_CLAUSE_RE = re.compile(
    r"""
    (?P<field>[a-z_]+)              # field name
    :                               # separator
    (?:
      "(?P<quoted>[^"]*)"           # "quoted value"
      |
      (?P<comp>[><!]=?)?            # optional comparison operator
      (?P<bare>[^\s"]+)             # bare value (no spaces, no quotes)
    )
    """,
    re.VERBOSE,
)

_KNOWN_FIELDS = INDEX_FIELDS | POST_FILTER_FIELDS | {"limit"}


def _tokenize(expression: str) -> list[tuple[str, str]]:
    """Tokenize a JQL expression into (type, value) tuples.

    Returns:
        List of ('clause', clause_text), ('AND', 'AND'), ('OR', 'OR') tuples.
    """
    tokens: list[tuple[str, str]] = []
    remaining = expression.strip()

    while remaining:
        remaining = remaining.lstrip()

        # Check for boolean operators
        upper = remaining.upper()
        if upper.startswith("AND ") or upper.startswith("AND\t"):
            tokens.append(("AND", "AND"))
            remaining = remaining[3:].lstrip()
            continue
        if upper.startswith("OR ") or upper.startswith("OR\t"):
            tokens.append(("OR", "OR"))
            remaining = remaining[2:].lstrip()
            continue

        # Try to match a clause
        m = _CLAUSE_RE.match(remaining)
        if m:
            tokens.append(("clause", m.group(0)))
            remaining = remaining[m.end() :]
            continue

        # Skip unrecognized tokens
        next_space = remaining.find(" ")
        if next_space == -1:
            # Last token, skip
            break
        remaining = remaining[next_space:].lstrip()

    return tokens


def _parse_clause(text: str) -> JQLClause | None:
    """Parse a single clause string into a JQLClause."""
    m = _CLAUSE_RE.match(text)
    if not m:
        return None

    field_name = m.group("field").lower()
    if field_name not in _KNOWN_FIELDS:
        logger.debug(f"Unknown JQL field: {field_name!r} in {text!r}")
        return None

    quoted = m.group("quoted")
    comp = m.group("comp") or ""
    bare = m.group("bare") or ""

    value = quoted if quoted is not None else bare

    # Determine operator
    if comp == ">":
        op = ClauseOp.GT
    elif comp == ">=":
        op = ClauseOp.GTE
    elif comp == "<":
        op = ClauseOp.LT
    elif comp == "<=":
        op = ClauseOp.LTE
    elif field_name == "text":
        op = ClauseOp.MATCH
    else:
        op = ClauseOp.EQ

    return JQLClause(field=field_name, op=op, value=value, raw=text)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def parse_jql(expression: str) -> JQLQuery:
    """Parse a JQL expression into a structured query.

    Args:
        expression: JQL query string, e.g. ``'type:class file:src/auth/*'``.

    Returns:
        ``JQLQuery`` with index-friendly and post-filter clauses separated.

    Examples::

        >>> q = parse_jql('type:class file:src/auth/*')
        >>> q.type_values
        ['class']
        >>> q.file_value
        'src/auth/*'

        >>> q = parse_jql('related:"BaseHandler" dir:incoming has_rel:inherits')
        >>> q.related_value
        'BaseHandler'
        >>> q.direction_value
        'incoming'

        >>> q = parse_jql('type:class connections:>10 limit:5')
        >>> q.connections_clause.op
        <ClauseOp.GT: 'gt'>
        >>> q.limit
        5
    """
    if not expression or not expression.strip():
        return JQLQuery(raw_query=expression or "")

    tokens = _tokenize(expression)

    query = JQLQuery(raw_query=expression)
    has_or = False

    for token_type, token_value in tokens:
        if token_type == "OR":  # noqa: S105 — token_type is a JQL parser token kind, not a credential
            has_or = True
            continue
        if token_type == "AND":  # noqa: S105 — token_type is a JQL parser token kind, not a credential
            continue
        if token_type != "clause":  # noqa: S105 — token_type is a JQL parser token kind, not a credential
            continue

        clause = _parse_clause(token_value)
        if clause is None:
            continue

        # Handle special fields
        if clause.field == "limit":
            try:
                query.limit = max(1, min(500, int(clause.value)))
            except (ValueError, TypeError):
                pass
            continue

        # Classify clause
        if clause.field in INDEX_FIELDS:
            query.index_clauses.append(clause)
        elif clause.field in POST_FILTER_FIELDS:
            query.post_filters.append(clause)

    query.is_or = has_or
    return query


def is_jql_expression(text: str) -> bool:
    """Quick heuristic to check if text looks like a JQL expression.

    Returns True if the text contains at least one ``field:value`` pattern
    where field is a known JQL field name.

    This is useful for deciding whether to route a query through the JQL
    parser or treat it as a natural-language search.
    """
    if not text:
        return False
    for m in _CLAUSE_RE.finditer(text):
        if m.group("field").lower() in _KNOWN_FIELDS:
            return True
    return False
