"""
Query builder for FTS5 and vector-store search backends.

Constructs optimised queries tailored to different retrieval intents:

* **symbol_resolution** — exact / near-exact symbol name lookup
* **concept_search**    — broad concept search across docstrings & content
* **doc_search**        — documentation-oriented queries
* **for_vector_store**  — cleaned natural-language query for embedding similarity
* **parse_natural_language** — SPEC-6: NL→structured filter extraction

All public methods are ``@staticmethod`` — the class is a pure utility with
no state.  Consumers import individual helpers or the class:

    from .graph_query_builder import GraphQueryBuilder, extract_keywords
    from .graph_query_builder import parse_natural_language

Feature flag
~~~~~~~~~~~~
Always available — query building doesn't depend on ``WIKIS_ENABLE_FTS5``.
Callers decide whether to use FTS5 or brute-force search; this module only
constructs the query strings.
"""

import re
from dataclasses import dataclass, field

# ---------------------------------------------------------------------------
# Stop words — tuned for *code search* (not generic NLP).
# These are words that appear in natural-language questions but carry zero
# information for BM25 matching against symbol names / docstrings / code.
# ---------------------------------------------------------------------------

CODE_STOP_WORDS: frozenset = frozenset(
    {
        # Question / discourse words
        "a",
        "an",
        "the",
        "is",
        "are",
        "was",
        "were",
        "be",
        "been",
        "being",
        "am",
        "do",
        "does",
        "did",
        "has",
        "have",
        "had",
        "having",
        "will",
        "would",
        "shall",
        "should",
        "may",
        "might",
        "must",
        "can",
        "could",
        "how",
        "what",
        "when",
        "where",
        "which",
        "who",
        "whom",
        "why",
        "this",
        "that",
        "these",
        "those",
        "i",
        "me",
        "my",
        "we",
        "us",
        "our",
        "you",
        "your",
        "he",
        "she",
        "it",
        "they",
        "them",
        "its",
        "their",
        # Connectors / prepositions
        "of",
        "in",
        "to",
        "for",
        "with",
        "on",
        "at",
        "by",
        "from",
        "about",
        "as",
        "into",
        "through",
        "during",
        "before",
        "after",
        "above",
        "below",
        "between",
        "under",
        "over",
        # Common verbs meaningless in code search
        "get",
        "set",
        "make",
        "work",
        "works",
        "use",
        "uses",
        "used",
        "find",
        "show",
        "tell",
        "give",
        # Filler
        "and",
        "or",
        "but",
        "not",
        "no",
        "so",
        "if",
        "then",
        "else",
        "all",
        "each",
        "every",
        "any",
        "some",
        "many",
        "much",
        "more",
        "most",
        "other",
        "than",
        "very",
        "just",
        "also",
        "only",
        "here",
        "there",
        "out",
        "up",
        "down",
    }
)

# Words that are stop-words in plain English but carry meaning in code
# (e.g. "class", "function") — do NOT strip these.
CODE_SIGNIFICANT_WORDS: frozenset = frozenset(
    {
        "class",
        "classes",
        "function",
        "functions",
        "method",
        "methods",
        "interface",
        "interfaces",
        "struct",
        "structs",
        "enum",
        "enums",
        "module",
        "modules",
        "package",
        "packages",
        "constant",
        "constants",
        "type",
        "types",
        "variable",
        "variables",
        "parameter",
        "parameters",
        "error",
        "errors",
        "exception",
        "exceptions",
        "handler",
        "handlers",
        "service",
        "services",
        "config",
        "configuration",
        "factory",
        "builder",
        "manager",
        "client",
        "server",
        "request",
        "response",
        "database",
        "cache",
        "queue",
        "stream",
        "test",
        "tests",
        "mock",
        "fixture",
        "auth",
        "authentication",
        "authorization",
        "api",
        "rest",
        "http",
        "grpc",
        "async",
        "await",
        "promise",
        "callback",
        "abstract",
        "virtual",
        "override",
        "static",
        "template",
        "generic",
        "trait",
        "mixin",
        "create",
        "delete",
        "update",
        "read",
        "init",
        "initialize",
        "setup",
        "teardown",
        "parse",
        "parser",
        "serialize",
        "deserialize",
        "validate",
        "validation",
        "schema",
        "import",
        "export",
        "inherit",
        "implement",
        "handle",
        "process",
        "execute",
        "run",
        "connect",
        "disconnect",
        "open",
        "close",
        "send",
        "receive",
        "publish",
        "subscribe",
        "load",
        "save",
        "store",
        "fetch",
        "retrieve",
    }
)

# ---------------------------------------------------------------------------
# CamelCase / snake_case splitting (reuses pattern from graph_text_index)
# ---------------------------------------------------------------------------

_CAMEL_SPLIT_RE = re.compile(r"(?<=[a-z])(?=[A-Z])|(?<=[A-Z])(?=[A-Z][a-z])")


def _split_identifier(name: str) -> list[str]:
    """Split a camelCase / PascalCase / snake_case identifier into parts.

    >>> _split_identifier("AuthServiceManager")
    ['auth', 'service', 'manager']
    >>> _split_identifier("get_user_by_id")
    ['get', 'user', 'by', 'id']
    """
    parts = _CAMEL_SPLIT_RE.sub(" ", name)
    parts = re.sub(r"[^a-zA-Z0-9]", " ", parts)
    return [p for p in parts.lower().split() if p]


# ---------------------------------------------------------------------------
# Keyword extraction
# ---------------------------------------------------------------------------


def extract_keywords(
    natural_query: str,
    *,
    min_length: int = 2,
    keep_identifiers: bool = True,
) -> list[str]:
    """Extract meaningful search keywords from a natural-language query.

    1. Tokenise on non-alphanumeric boundaries.
    2. Drop tokens shorter than *min_length*.
    3. Drop stop words (unless they appear in ``CODE_SIGNIFICANT_WORDS``).
    4. If *keep_identifiers* is True, also add sub-parts of camelCase /
       snake_case tokens (e.g. ``"AuthService"`` → ``["authservice",
       "auth", "service"]``).
    5. Deduplicate while preserving order.

    Examples::

        >>> extract_keywords("How does authentication work?")
        ['authentication']
        >>> extract_keywords("What classes handle payment processing?")
        ['classes', 'handle', 'payment', 'processing']
        >>> extract_keywords("AuthServiceManager login flow")
        ['authservicemanager', 'auth', 'service', 'manager', 'login', 'flow']
    """
    raw_tokens = re.split(r"[^a-zA-Z0-9_]+", natural_query.strip())

    seen: set[str] = set()
    result: list[str] = []

    def _add(tok: str) -> None:
        if tok and tok not in seen and len(tok) >= min_length:
            seen.add(tok)
            result.append(tok)

    for raw in raw_tokens:
        if not raw:
            continue
        lower = raw.lower()

        # Keep code-significant words even if they're also in stop words
        if lower in CODE_SIGNIFICANT_WORDS:
            _add(lower)
            continue

        # Skip stop words
        if lower in CODE_STOP_WORDS:
            continue

        # Add the whole token (lowercased)
        _add(lower)

        # Optionally split compound identifiers
        if keep_identifiers and ("_" in raw or any(c.isupper() for c in raw[1:])):
            for part in _split_identifier(raw):
                if part not in CODE_STOP_WORDS or part in CODE_SIGNIFICANT_WORDS:
                    _add(part)

    return result


# ---------------------------------------------------------------------------
# SPEC-6: NL → Structured Filter Extraction
# ---------------------------------------------------------------------------

# Maps natural-language type words to canonical symbol_type values.
TYPE_KEYWORDS: dict[str, str] = {
    "class": "class",
    "classes": "class",
    "function": "function",
    "functions": "function",
    "func": "function",
    "funcs": "function",
    "method": "method",
    "methods": "method",
    "interface": "interface",
    "interfaces": "interface",
    "struct": "struct",
    "structs": "struct",
    "enum": "enum",
    "enums": "enum",
    "constant": "constant",
    "constants": "constant",
    "trait": "trait",
    "traits": "trait",
    "module": "module_doc",
    "modules": "module_doc",
}

# Words that signal a path/directory reference in the NEXT token.
MODULE_KEYWORDS: frozenset = frozenset(
    {
        "module",
        "modules",
        "package",
        "packages",
        "directory",
        "folder",
        "dir",
        "file",
    }
)

# Maps natural-language relationship words to canonical edge types as stored
# on graph edges (``RelationshipType.value`` strings from ``base_parser.py``).
# The values here MUST match the actual ``relationship_type`` attribute of
# edges returned by the enhanced parsers — e.g. ``"inheritance"`` not
# ``"inherits"``.  Legacy aliases (``"extends"`` etc.) that appear on some
# older edges are normalised by ``EDGE_TYPE_ALIASES`` in the query service.
RELATIONSHIP_KEYWORDS: dict[str, str] = {
    # OOP — inheritance
    "inherits": "inheritance",
    "inherit": "inheritance",
    "extends": "inheritance",
    "extend": "inheritance",
    "extending": "inheritance",
    "subclass": "inheritance",
    "subclasses": "inheritance",
    "inheritance": "inheritance",
    # OOP — implementation (interface)
    "implements": "implementation",
    "implement": "implementation",
    "implementing": "implementation",
    "implementation": "implementation",
    # OOP — composition / aggregation
    "composition": "composition",
    "composes": "composition",
    "composed": "composition",
    "aggregation": "aggregation",
    "aggregates": "aggregation",
    # Calls
    "calls": "calls",
    "call": "calls",
    "calling": "calls",
    "invokes": "calls",
    # Imports / exports
    "imports": "imports",
    "import": "imports",
    "importing": "imports",
    "exports": "exports",
    "export": "exports",
    # Structural
    "defines": "defines",
    "define": "defines",
    "defining": "defines",
    "contains": "contains",
    "contain": "contains",
    "decorates": "decorates",
    "decorator": "decorates",
    "annotates": "annotates",
    # Data flow
    "assigns": "assigns",
    "returns": "returns",
    "parameter": "parameter",
    "reads": "reads",
    "writes": "writes",
    # References / uses
    "references": "references",
    "reference": "references",
    "ref": "references",
    "uses": "uses",
    "use": "uses",
    "using": "uses",
    "uses_type": "uses_type",
    # Object creation
    "creates": "creates",
    "create": "creates",
    "instantiates": "instantiates",
    "constructs": "creates",
    # C++ specific
    "specializes": "specializes",
    "declares": "declares",
    "defines_body": "defines_body",
    "overrides": "overrides",
    "hides": "hides",
    "captures": "captures",
    # Type aliases
    "alias_of": "alias_of",
    "alias": "alias_of",
}

# Maps legacy / informal edge type strings that appear on some graph edges to
# canonical ``RelationshipType.value`` forms.  Used by the query service to
# normalise before comparison so that both ``has_rel:inherits`` and
# ``has_rel:inheritance`` match the same edges.
EDGE_TYPE_ALIASES: dict[str, str] = {
    # NL → canonical
    "inherits": "inheritance",
    "extends": "inheritance",
    "implements": "implementation",
    "invokes": "calls",
    "has_member": "contains",
    "has_method": "contains",
    "composes": "composition",
    "constructs": "creates",
    "accesses": "reads",
    "modifies": "writes",
    "file_imports": "imports",
    "implements_interface": "implementation",
    "field_type": "uses_type",
}

# Preposition words that bridge to a path or related symbol
_BRIDGE_PREPOSITIONS: frozenset = frozenset(
    {
        "in",
        "from",
        "under",
        "inside",
        "within",
        "of",
    }
)

_INTENT_KEYWORDS: dict[str, str] = {
    "architecture": "architecture",
    "architectural": "architecture",
    "structure": "architecture",
    "overview": "architecture",
    "implementation": "implementation",
    "detail": "implementation",
    "security": "security",
    "auth": "security",
    "authentication": "security",
    "integration": "integration",
    "integrates": "integration",
    "debugging": "debugging",
    "debug": "debugging",
    "bug": "debugging",
}


@dataclass
class ParsedNLQuery:
    """Result of natural-language query parsing (SPEC-6).

    Separates structural filters from content keywords so that callers
    can use typed filters (``type``, ``path_hint``) at the SQL/FTS level
    and only pass ``text_keywords`` to BM25 full-text search.

    Attributes:
        text_keywords: Content keywords for FTS5 search (structural words removed).
        type_filter: Canonical symbol type (e.g. 'class', 'function') or None.
        path_hint: Inferred path prefix/pattern or None.
        relationship_hint: Canonical edge type (e.g. 'inherits', 'calls') or None.
        intent: Detected intent ('general', 'symbol', 'concept', 'architecture', etc.)
        related_symbol: Symbol name for relationship queries or None.
    """

    text_keywords: list[str] = field(default_factory=list)
    type_filter: str | None = None
    path_hint: str | None = None
    relationship_hint: str | None = None
    intent: str = "general"
    related_symbol: str | None = None

    def to_jql(self) -> str:
        """Convert to a JQL expression for ``GraphQueryService.query()``.

        Examples::

            >>> p = ParsedNLQuery(type_filter='class', text_keywords=['auth'])
            >>> p.to_jql()
            'type:class text:auth'
        """
        parts: list[str] = []
        if self.type_filter:
            parts.append(f"type:{self.type_filter}")
        if self.path_hint:
            hint = self.path_hint
            if not any(c in hint for c in ("*", "?")):
                hint = hint.rstrip("/") + "/*"
            parts.append(f"file:{hint}")
        if self.related_symbol:
            name = self.related_symbol
            if " " in name:
                name = f'"{name}"'
            parts.append(f"related:{name}")
        if self.relationship_hint:
            parts.append(f"has_rel:{self.relationship_hint}")
        if self.text_keywords:
            # Join keywords as a single text clause
            parts.append(f"text:{' '.join(self.text_keywords)}")
        return " ".join(parts)

    @property
    def has_structural_filters(self) -> bool:
        """True if any structural filter was extracted (type, path, related)."""
        return bool(self.type_filter or self.path_hint or self.relationship_hint or self.related_symbol)


def parse_natural_language(query: str) -> ParsedNLQuery:
    """Parse a natural-language query and extract structured filters (SPEC-6).

    Scans tokens for structural words (type keywords, module references,
    relationship keywords) and extracts them as typed filters, leaving
    only content keywords for FTS5 BM25 search.

    This is a **deterministic, zero-LLM** extraction that works at the
    keyword-dictionary level — no model inference, no network calls.

    Args:
        query: Free-form natural-language question or keyword string.

    Returns:
        ``ParsedNLQuery`` with extracted filters and remaining keywords.

    Examples::

        >>> parse_natural_language("What classes handle authentication?")
        ParsedNLQuery(type_filter='class', text_keywords=['handle', 'authentication'], ...)

        >>> parse_natural_language("functions in the auth module")
        ParsedNLQuery(type_filter='function', path_hint='auth', text_keywords=[], ...)

        >>> parse_natural_language("classes that extend BaseHandler")
        ParsedNLQuery(type_filter='class', relationship_hint='inherits',
                      related_symbol='BaseHandler', text_keywords=[], ...)
    """
    if not query or not query.strip():
        return ParsedNLQuery()

    raw_tokens = re.split(r"[^a-zA-Z0-9_./]+", query.strip())
    tokens = [t for t in raw_tokens if t]
    lower_tokens = [t.lower() for t in tokens]

    result = ParsedNLQuery()
    consumed: set[int] = set()  # indices of tokens consumed as filters

    # Phase 1: Extract type filter
    for i, tok in enumerate(lower_tokens):
        if tok in TYPE_KEYWORDS:
            result.type_filter = TYPE_KEYWORDS[tok]
            consumed.add(i)
            break  # only one type filter

    # Phase 2: Extract path hint from "in/from <path>" or "module <path>" patterns
    for i, tok in enumerate(lower_tokens):
        if i in consumed:
            continue
        # Pattern: "in the auth module" / "from src/auth"
        if tok in _BRIDGE_PREPOSITIONS and i + 1 < len(tokens):
            next_tok = tokens[i + 1]
            next_lower = lower_tokens[i + 1]

            # Skip "the" / "a" between preposition and path
            offset = 1
            if next_lower in ("the", "a", "an") and i + 2 < len(tokens):
                offset = 2
                next_tok = tokens[i + offset]
                next_lower = lower_tokens[i + offset]

            # Check if next token looks like a path or module reference
            if "/" in next_tok or "." in next_tok:
                result.path_hint = next_tok
                consumed.add(i)
                for j in range(1, offset + 1):
                    consumed.add(i + j)
                break
            # "module <name>" after bridge
            if i + offset + 1 < len(tokens) and next_lower in MODULE_KEYWORDS:
                result.path_hint = tokens[i + offset + 1]
                consumed.update(range(i, i + offset + 2))
                break
            # Plain identifier after preposition — treat as path if no type keyword
            if next_lower not in CODE_STOP_WORDS and next_lower not in TYPE_KEYWORDS:
                # Heuristic: only if it's short-ish and not a common content word
                if len(next_tok) <= 30:
                    result.path_hint = next_tok
                    consumed.add(i)
                    for j in range(1, offset + 1):
                        consumed.add(i + j)
                    break

        # Pattern: "module <name>" without preposition
        if tok in MODULE_KEYWORDS and i + 1 < len(tokens):
            next_lower = lower_tokens[i + 1]
            if next_lower not in CODE_STOP_WORDS:
                result.path_hint = tokens[i + 1]
                consumed.add(i)
                consumed.add(i + 1)
                break

    # Phase 3: Extract relationship hint and related symbol
    for i, tok in enumerate(lower_tokens):
        if i in consumed:
            continue
        if tok in RELATIONSHIP_KEYWORDS:
            result.relationship_hint = RELATIONSHIP_KEYWORDS[tok]
            consumed.add(i)
            # The next non-stop word is the related symbol
            for j in range(i + 1, len(tokens)):
                if j in consumed:
                    continue
                candidate = lower_tokens[j]
                if candidate in ("from", "to", "the", "a", "an", "that", "which"):
                    consumed.add(j)
                    continue
                # This looks like a symbol name — use original case
                result.related_symbol = tokens[j]
                consumed.add(j)
                break
            break

    # Phase 4: Detect intent from remaining tokens
    for i, tok in enumerate(lower_tokens):
        if i in consumed:
            continue
        if tok in _INTENT_KEYWORDS:
            result.intent = _INTENT_KEYWORDS[tok]
            break

    # Phase 5: Collect remaining content keywords (non-consumed, non-stop-word)
    for i, tok in enumerate(lower_tokens):
        if i in consumed:
            continue
        if tok in CODE_STOP_WORDS and tok not in CODE_SIGNIFICANT_WORDS:
            continue
        if len(tok) < 2:
            continue
        # Skip type/module keywords already extracted
        if tok in TYPE_KEYWORDS and result.type_filter:
            continue
        if tok in MODULE_KEYWORDS and result.path_hint:
            continue
        result.text_keywords.append(tok)

    return result


# ---------------------------------------------------------------------------
# FTS5 query escaping
# ---------------------------------------------------------------------------


def _escape_fts5_token(token: str) -> str:
    """Escape a token for safe inclusion in FTS5 query strings.

    Doubles any embedded quotes and wraps in quotes.
    """
    return '"' + token.replace('"', '""') + '"'


# ---------------------------------------------------------------------------
# GraphQueryBuilder
# ---------------------------------------------------------------------------


class GraphQueryBuilder:
    """Constructs optimised queries for FTS5 and vector-store backends.

    All methods are ``@staticmethod`` — no state, pure functions.
    """

    # ------------------------------------------------------------------
    # FTS5 queries
    # ------------------------------------------------------------------

    @staticmethod
    def symbol_resolution(name: str) -> str:
        """FTS5 query for exact / near-exact symbol lookup.

        Strategy:
        - Column-scope to ``symbol_name`` (weight 10.0) for exact match.
        - Also search ``name_tokens`` for split variants (e.g. "auth service").
        - Returns empty string if *name* is empty.

        Example::

            >>> GraphQueryBuilder.symbol_resolution("AuthService")
            'symbol_name:"AuthService"* OR name_tokens:"auth" "service"'
        """
        name = name.strip()
        if not name:
            return ""

        parts: list[str] = []

        # 1. Exact / prefix match on symbol_name column
        escaped = _escape_fts5_token(name.lower())
        parts.append(f"symbol_name:{escaped}*")

        # 2. Split-token match on name_tokens column
        sub_tokens = _split_identifier(name)
        if len(sub_tokens) > 1:
            token_query = " ".join(_escape_fts5_token(t) for t in sub_tokens)
            parts.append(f"name_tokens:{token_query}")

        return " OR ".join(parts)

    @staticmethod
    def concept_search(
        keywords: list[str],
        *,
        boost_docstring: bool = True,
    ) -> str:
        """FTS5 query for broad concept search across all columns.

        Strategy:
        - Each keyword becomes a prefix query (``keyword*``).
        - If *boost_docstring* is True, adds an extra column-scoped
          ``docstring:keyword`` clause for each keyword to boost BM25
          when the keyword appears in the docstring.
        - Multiple keywords are joined with OR (any match contributes).

        Example::

            >>> GraphQueryBuilder.concept_search(["authentication", "token"])
            '"authentication"* OR "token"* OR docstring:"authentication" OR docstring:"token"'
        """
        if not keywords:
            return ""

        parts: list[str] = []

        # Broad prefix search across all FTS columns
        for kw in keywords:
            escaped = _escape_fts5_token(kw.lower())
            parts.append(f"{escaped}*")

        # Optional docstring boost
        if boost_docstring:
            for kw in keywords:
                escaped = _escape_fts5_token(kw.lower())
                parts.append(f"docstring:{escaped}")

        return " OR ".join(parts)

    @staticmethod
    def phrase_search(phrase: str, column: str | None = None) -> str:
        """FTS5 phrase query — matches the exact phrase in order.

        Args:
            phrase: Multi-word phrase (e.g. ``"dependency injection"``).
            column: Optional FTS5 column to restrict to (e.g. ``"docstring"``).

        Example::

            >>> GraphQueryBuilder.phrase_search("dependency injection", "docstring")
            'docstring:"dependency injection"'
        """
        phrase = phrase.strip()
        if not phrase:
            return ""
        escaped = _escape_fts5_token(phrase.lower())
        if column:
            return f"{column}:{escaped}"
        return escaped

    @staticmethod
    def proximity_search(
        terms: list[str],
        distance: int = 50,
        column: str | None = None,
    ) -> str:
        """FTS5 NEAR query — matches terms within *distance* tokens of each other.

        Requires at least 2 terms.

        Example::

            >>> GraphQueryBuilder.proximity_search(["UserService", "authenticate"], 50)
            'NEAR("userservice" "authenticate", 50)'
        """
        if len(terms) < 2:
            return ""

        escaped = " ".join(_escape_fts5_token(t.lower()) for t in terms)
        near = f"NEAR({escaped}, {distance})"
        if column:
            return f"{column}:{near}"
        return near

    @staticmethod
    def combined_symbol_and_concept(
        symbol_names: list[str],
        concept_keywords: list[str],
    ) -> str:
        """Build a combined FTS5 query that finds symbols by name AND concept.

        Useful for queries like "find classes related to authentication" where
        we have both structural hints (class names) and concept terms.
        """
        parts: list[str] = []

        for name in symbol_names:
            q = GraphQueryBuilder.symbol_resolution(name)
            if q:
                parts.append(q)

        if concept_keywords:
            q = GraphQueryBuilder.concept_search(concept_keywords, boost_docstring=True)
            if q:
                parts.append(q)

        return " OR ".join(parts) if parts else ""

    # ------------------------------------------------------------------
    # Natural-language → FTS5 (replaces _prepare_fts_query)
    # ------------------------------------------------------------------

    @staticmethod
    def from_natural_language(
        query: str,
        *,
        intent: str = "general",
    ) -> str:
        """Convert a natural-language query into an FTS5 query string.

        This is the main entry point that replaces ``_prepare_fts_query``.

        SPEC-6 enhancement: now pre-parses the query to extract structural
        filters (type keywords, path references) and removes them from the
        FTS5 query.  The **FTS5 string** returned contains only content
        keywords.  Use ``from_natural_language_parsed()`` to also get the
        extracted filter metadata.

        Args:
            query: Free-form natural-language question or keyword string.
            intent: One of ``'general'``, ``'symbol'``, ``'concept'``,
                    ``'doc'``.  Adjusts the query construction strategy.

        Strategy by intent:

        * ``'general'`` — extract keywords, prefix-match across all columns.
        * ``'symbol'``  — treat query as a symbol name, column-scope.
        * ``'concept'`` — extract keywords, boost docstring column.
        * ``'doc'``     — extract keywords, target docstring + content columns.

        Falls back to simple prefix-OR if keyword extraction yields nothing.
        """
        fts_query, _ = GraphQueryBuilder.from_natural_language_parsed(
            query,
            intent=intent,
        )
        return fts_query

    @staticmethod
    def from_natural_language_parsed(
        query: str,
        *,
        intent: str = "general",
    ) -> tuple[str, "ParsedNLQuery"]:
        """Convert NL query into FTS5 string + structured filter metadata.

        SPEC-6: This is the enhanced version that returns both the FTS5
        query AND a ``ParsedNLQuery`` with extracted structural filters
        (type, path, relationship).

        Callers that need the filters (e.g. ``search_smart``) should use
        this method and pass the filters to ``_execute_fts_search()``.

        Returns:
            Tuple of ``(fts5_query_string, parsed_nl_query)``.
        """
        query = query.strip()
        if not query:
            return "", ParsedNLQuery()

        if intent == "symbol":
            return GraphQueryBuilder.symbol_resolution(query), ParsedNLQuery(intent="symbol")

        # SPEC-6: Parse NL to extract structural filters
        parsed = parse_natural_language(query)

        # Use the cleaned text_keywords instead of raw extract_keywords
        keywords = parsed.text_keywords if parsed.text_keywords else extract_keywords(query)

        if not keywords:
            # Fallback: just tokenise and prefix-match everything
            tokens = [t.lower() for t in re.split(r"\W+", query) if len(t) >= 2]
            if not tokens:
                return "", parsed
            return " OR ".join(f'"{t}"*' for t in tokens), parsed

        if intent == "concept":
            return GraphQueryBuilder.concept_search(keywords, boost_docstring=True), parsed

        if intent == "doc":
            parts: list[str] = []
            for kw in keywords:
                escaped = _escape_fts5_token(kw.lower())
                parts.append(f"{escaped}*")
                parts.append(f"docstring:{escaped}")
                parts.append(f"content:{escaped}")
            return " OR ".join(parts), parsed

        # general: simple prefix-match (same as improved _prepare_fts_query)
        return GraphQueryBuilder.concept_search(keywords, boost_docstring=False), parsed

    # ------------------------------------------------------------------
    # Vector-store query preparation
    # ------------------------------------------------------------------

    @staticmethod
    def for_vector_store(
        natural_query: str,
        *,
        context_terms: list[str] | None = None,
    ) -> str:
        """Prepare a query for vector similarity search.

        Unlike FTS5 which needs structured query syntax, vector stores work
        best with clean natural-language text that matches the embedding
        model's training distribution.

        Strategy:
        - Extract keywords from the query.
        - Optionally append *context_terms* (e.g. file names, module names).
        - Join into a clean sentence-like string.

        Example::

            >>> GraphQueryBuilder.for_vector_store("How does authentication work?")
            'authentication'
            >>> GraphQueryBuilder.for_vector_store(
            ...     "payment processing",
            ...     context_terms=["billing", "stripe"]
            ... )
            'payment processing billing stripe'
        """
        keywords = extract_keywords(natural_query)
        parts = keywords if keywords else [natural_query.strip()]

        if context_terms:
            for term in context_terms:
                clean = term.strip().lower()
                if clean and clean not in parts:
                    parts.append(clean)

        return " ".join(parts)

    # ------------------------------------------------------------------
    # Structured helpers
    # ------------------------------------------------------------------

    @staticmethod
    def type_scoped_search(
        keywords: list[str],
        symbol_type: str,
    ) -> tuple[str, frozenset]:
        """Return an FTS5 query string AND a symbol_types filter.

        Useful for queries like "find all classes related to auth":
        produces an FTS5 query for "auth" + a type filter for "class".

        Returns:
            Tuple of (fts_query, symbol_types_filter).
        """
        fts_query = GraphQueryBuilder.concept_search(
            keywords,
            boost_docstring=True,
        )
        type_filter = frozenset({symbol_type.lower()})
        return fts_query, type_filter

    @staticmethod
    def path_scoped_concept(
        keywords: list[str],
        path_prefix: str,
    ) -> tuple[str, str]:
        """Return an FTS5 query AND a path prefix for combined filtering.

        The caller should use the FTS5 query with ``search()`` and
        post-filter or pre-filter by path prefix.

        Returns:
            Tuple of (fts_query, normalised_path_prefix).
        """
        fts_query = GraphQueryBuilder.concept_search(
            keywords,
            boost_docstring=True,
        )
        norm_prefix = path_prefix.strip("/")
        return fts_query, norm_prefix
