"""Credential redaction for source toolkit log output."""

from __future__ import annotations

import logging
import re

# Field names whose values must never appear in logs.
_SENSITIVE_FIELDS = frozenset(
    ["access_token", "refresh_token", "pat", "token", "api_token"]
)

# Match JSON-style ("key": "value") and key=value / key: value forms.
# Group 1 = the matched key name; group 2 = delimiter; group 3 = value.
_JSON_RE = re.compile(
    r'"(' + "|".join(_SENSITIVE_FIELDS) + r')"(\s*:\s*)"[^"]*"',
    re.IGNORECASE,
)
_KV_RE = re.compile(
    r"\b(" + "|".join(_SENSITIVE_FIELDS) + r")(=|:\s*)(\S+)",
    re.IGNORECASE,
)


def _redact_string(text: str) -> str:
    """Return *text* with sensitive field values replaced by ``***``."""
    text = _JSON_RE.sub(r'"\1"\2"***"', text)
    text = _KV_RE.sub(r"\1\2***", text)
    return text


def _redact_value(value: object) -> object:
    """Recursively redact sensitive values in dicts/lists."""
    if isinstance(value, dict):
        return {k: ("***" if k in _SENSITIVE_FIELDS else _redact_value(v)) for k, v in value.items()}
    if isinstance(value, list):
        return [_redact_value(item) for item in value]
    if isinstance(value, str):
        return _redact_string(value)
    return value


class TokenRedactionFilter(logging.Filter):
    """Logging filter that redacts credential field values from log records.

    Handles:
    - ``record.msg`` as a plain string (JSON and key=value patterns).
    - ``record.args`` when it is a single dict (``%(field)s``-style or
      ``logger.info("…", {"key": "val"})``-style).
    """

    def filter(self, record: logging.LogRecord) -> bool:  # noqa: A003
        if isinstance(record.msg, str):
            record.msg = _redact_string(record.msg)

        if isinstance(record.args, dict):
            record.args = _redact_value(record.args)  # type: ignore[assignment]
        # Tuple positional args are left untouched — we can't know which
        # slot maps to which field name without parsing the format string.

        return True


def install() -> None:
    """Add TokenRedactionFilter to the root logger and uvicorn loggers.

    Safe to call multiple times — duplicate filters are a no-op in practice
    because Python checks filter identity, not equality.
    """
    fltr = TokenRedactionFilter()
    for name in ("", "uvicorn.access", "uvicorn.error"):
        log = logging.getLogger(name)
        # Avoid adding the same filter class twice.
        if not any(isinstance(f, TokenRedactionFilter) for f in log.filters):
            log.addFilter(fltr)
