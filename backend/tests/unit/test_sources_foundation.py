"""Unit tests for backend/app/core/sources/ scaffolding (issue #185)."""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import AsyncIterator

import pytest

from app.core.sources import (
    FileContent,
    FileInfo,
    OriginPointer,
    SourceAuthError,
    SourceConnectionError,
    SourceError,
    SourceNotFoundError,
    SourceToolkit,
    SourceUnavailableError,
    TokenRedactionFilter,
    ToolkitRegistry,
    install_redaction_filter,
    registry,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _now() -> datetime:
    return datetime.now(tz=timezone.utc)


class _ConcreteToolkit(SourceToolkit):
    """Minimal concrete subclass used across registry tests."""

    source_type = "fake"

    def __init__(self, cfg: dict) -> None:
        self.cfg = cfg

    @classmethod
    def from_config(cls, config: dict) -> "_ConcreteToolkit":
        return cls(config)

    async def list_files(
        self,
        include: list[str] | None = None,
        exclude: list[str] | None = None,
    ) -> AsyncIterator[FileInfo]:
        return
        yield  # make it an async generator

    async def fetch_content(self, pointer: OriginPointer) -> FileContent:
        raise NotImplementedError

    async def test_connection(self) -> str:
        return "ok"

    def build_origin_pointer(
        self,
        path: str,
        revision: str | None = None,
        line_start: int | None = None,
        line_end: int | None = None,
    ) -> OriginPointer:
        return OriginPointer(
            source_type=self.source_type,
            ref=path,
            url=f"fake://{path}",
            ingested_at=_now(),
            revision=revision,
            line_start=line_start,
            line_end=line_end,
        )


# ---------------------------------------------------------------------------
# 1. ABC enforcement
# ---------------------------------------------------------------------------


def test_abc_enforcement_raises_type_error():
    """Instantiating a subclass without overriding abstract methods raises TypeError."""

    class EmptySubclass(SourceToolkit):
        source_type = "empty"

    with pytest.raises(TypeError):
        EmptySubclass()  # type: ignore[abstract]


# ---------------------------------------------------------------------------
# 2. Dataclass shape — OriginPointer
# ---------------------------------------------------------------------------


def test_origin_pointer_has_no_source_id():
    assert not hasattr(OriginPointer, "source_id"), "source_id must not exist"


def test_origin_pointer_has_no_content_hash():
    assert not hasattr(OriginPointer, "content_hash"), "content_hash must not exist"


def test_origin_pointer_is_frozen():
    ptr = OriginPointer(
        source_type="git",
        ref="README.md",
        url="https://github.com/acme/repo/blob/main/README.md",
        ingested_at=_now(),
    )
    with pytest.raises((AttributeError, TypeError)):
        ptr.ref = "other"  # type: ignore[misc]


def test_origin_pointer_optional_fields_default_none():
    ptr = OriginPointer(
        source_type="git",
        ref="src/main.py",
        url="https://example.com",
        ingested_at=_now(),
    )
    assert ptr.revision is None
    assert ptr.line_start is None
    assert ptr.line_end is None


# ---------------------------------------------------------------------------
# 3. Registry — register, get, list_types, create
# ---------------------------------------------------------------------------


@pytest.fixture()
def fresh_registry() -> ToolkitRegistry:
    return ToolkitRegistry()


def test_registry_register_and_get(fresh_registry: ToolkitRegistry):
    fresh_registry.register(_ConcreteToolkit)
    assert fresh_registry.get("fake") is _ConcreteToolkit


def test_registry_register_as_decorator(fresh_registry: ToolkitRegistry):
    @fresh_registry.register
    class _DecoratedToolkit(_ConcreteToolkit):
        source_type = "decorated"

    assert fresh_registry.get("decorated") is _DecoratedToolkit


def test_registry_list_types_includes_registered(fresh_registry: ToolkitRegistry):
    fresh_registry.register(_ConcreteToolkit)
    assert "fake" in fresh_registry.list_types()


def test_registry_list_types_is_sorted(fresh_registry: ToolkitRegistry):
    class _B(_ConcreteToolkit):
        source_type = "bbb"

    class _A(_ConcreteToolkit):
        source_type = "aaa"

    fresh_registry.register(_B)
    fresh_registry.register(_A)
    types = fresh_registry.list_types()
    assert types == sorted(types)


def test_registry_create_calls_from_config(fresh_registry: ToolkitRegistry):
    fresh_registry.register(_ConcreteToolkit)
    cfg = {"host": "localhost", "token": "t"}
    toolkit = fresh_registry.create("fake", cfg)
    assert isinstance(toolkit, _ConcreteToolkit)
    assert toolkit.cfg == cfg


# ---------------------------------------------------------------------------
# 4. Registry missing key
# ---------------------------------------------------------------------------


def test_registry_get_missing_key_raises_key_error_with_available(fresh_registry: ToolkitRegistry):
    fresh_registry.register(_ConcreteToolkit)
    with pytest.raises(KeyError) as exc_info:
        fresh_registry.get("nonexistent")
    msg = str(exc_info.value)
    assert "fake" in msg, f"Expected available types in message, got: {msg}"


def test_registry_get_missing_key_empty_registry(fresh_registry: ToolkitRegistry):
    with pytest.raises(KeyError) as exc_info:
        fresh_registry.get("anything")
    assert "anything" in str(exc_info.value)


# ---------------------------------------------------------------------------
# 5. Exception hierarchy
# ---------------------------------------------------------------------------


def test_source_auth_error_is_source_error():
    assert issubclass(SourceAuthError, SourceError)


def test_source_connection_error_is_source_error():
    assert issubclass(SourceConnectionError, SourceError)


def test_source_not_found_error_is_source_error():
    assert issubclass(SourceNotFoundError, SourceError)


def test_source_unavailable_error_is_source_error():
    assert issubclass(SourceUnavailableError, SourceError)


def test_source_error_is_exception():
    assert issubclass(SourceError, Exception)


# ---------------------------------------------------------------------------
# Redaction filter helpers
# ---------------------------------------------------------------------------


@pytest.fixture()
def redaction_logger(caplog):
    """Return a logger with the TokenRedactionFilter installed, plus caplog."""
    log = logging.getLogger("test_redaction")
    log.setLevel(logging.DEBUG)
    fltr = TokenRedactionFilter()
    # Add filter to the caplog handler so records passing through are redacted
    for handler in caplog.handler.__class__.__mro__:
        break
    caplog.handler.addFilter(fltr)
    # Also add to logger itself so propagation picks it up
    log.addFilter(fltr)
    return log


# ---------------------------------------------------------------------------
# 6. Redaction — dict arg
# ---------------------------------------------------------------------------


def test_redaction_dict_arg_token_replaced(redaction_logger, caplog):
    with caplog.at_level(logging.DEBUG, logger="test_redaction"):
        redaction_logger.info("body=%s", {"access_token": "secret123", "other": "ok"})
    combined = "\n".join(r.getMessage() for r in caplog.records)
    assert "secret123" not in combined
    assert "***" in combined
    assert "ok" in combined


# ---------------------------------------------------------------------------
# 7. Redaction — JSON string in msg
# ---------------------------------------------------------------------------


def test_redaction_json_string_in_msg(redaction_logger, caplog):
    with caplog.at_level(logging.DEBUG, logger="test_redaction"):
        redaction_logger.info('body={"access_token": "secret123"}')
    combined = "\n".join(r.getMessage() for r in caplog.records)
    assert "secret123" not in combined
    assert "***" in combined


# ---------------------------------------------------------------------------
# 8. Redaction — key=value form
# ---------------------------------------------------------------------------


def test_redaction_key_value_form(redaction_logger, caplog):
    with caplog.at_level(logging.DEBUG, logger="test_redaction"):
        redaction_logger.info("got token=secret123 and pat=foo")
    combined = "\n".join(r.getMessage() for r in caplog.records)
    assert "secret123" not in combined
    assert "foo" not in combined
    assert "***" in combined


# ---------------------------------------------------------------------------
# 9. Redaction — nested dict
# ---------------------------------------------------------------------------


def test_redaction_nested_dict(redaction_logger, caplog):
    with caplog.at_level(logging.DEBUG, logger="test_redaction"):
        redaction_logger.info("body=%s", {"auth": {"refresh_token": "xyz"}})
    combined = "\n".join(r.getMessage() for r in caplog.records)
    assert "xyz" not in combined
    assert "***" in combined


# ---------------------------------------------------------------------------
# 10. Redaction — unrelated fields untouched
# ---------------------------------------------------------------------------


def test_redaction_unrelated_fields_untouched(redaction_logger, caplog):
    with caplog.at_level(logging.DEBUG, logger="test_redaction"):
        redaction_logger.info('hello world {"name": "alice"}')
    combined = "\n".join(r.getMessage() for r in caplog.records)
    assert "alice" in combined


# ---------------------------------------------------------------------------
# install_redaction_filter smoke test
# ---------------------------------------------------------------------------


def test_install_redaction_filter_does_not_raise():
    """install() must not raise even when called multiple times."""
    install_redaction_filter()
    install_redaction_filter()
