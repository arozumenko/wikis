"""Registry mapping source_type strings to SourceToolkit classes."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from app.core.sources.base import SourceToolkit

logger = logging.getLogger(__name__)


class ToolkitRegistry:
    """Central registry for SourceToolkit subclasses.

    Typical usage::

        @registry.register
        class GitToolkit(SourceToolkit):
            source_type = "github"
            ...
    """

    def __init__(self) -> None:
        self._registry: dict[str, type[SourceToolkit]] = {}

    def register(self, toolkit_cls: type[SourceToolkit]) -> type[SourceToolkit]:
        """Register a toolkit class by its ``source_type``. Also usable as a decorator."""
        self._registry[toolkit_cls.source_type] = toolkit_cls
        logger.debug("Registered source toolkit: %s", toolkit_cls.source_type)
        return toolkit_cls

    def get(self, source_type: str) -> type[SourceToolkit]:
        """Return the toolkit class for *source_type*.

        Args:
            source_type: The string key the class was registered under.

        Raises:
            KeyError: If *source_type* is not registered.
        """
        try:
            return self._registry[source_type]
        except KeyError:
            available = ", ".join(sorted(self._registry)) or "(none)"
            raise KeyError(
                f"No toolkit registered for source_type={source_type!r}. "
                f"Available: {available}"
            ) from None

    def list_types(self) -> list[str]:
        """Return a sorted list of registered source type strings."""
        return sorted(self._registry)

    def create(self, source_type: str, config: dict) -> SourceToolkit:
        """Look up *source_type* and instantiate it via ``from_config``.

        Args:
            source_type: Registered source type string.
            config: Credential/configuration dict passed to ``from_config``.

        Returns:
            A ready SourceToolkit instance.

        Raises:
            KeyError: If *source_type* is not registered.

        Errors raised by ``from_config`` propagate unchanged — callers (e.g. the wiki
        ingestion service) are responsible for catching and wrapping credential or
        configuration errors as appropriate (e.g. ``SourceAuthError``).
        """
        cls = self.get(source_type)
        return cls.from_config(config)


# Module-level singleton — concrete toolkits register themselves here.
registry = ToolkitRegistry()
