"""Image extractor — LLM vision describes the image as documentation text (#118).

For any image format the wiki ingests, build a multimodal ``HumanMessage``
(image bytes as base64 + a "describe this for a documentation index"
prompt) and call the project-configured LLM. The returned natural-language
description goes into FTS5 + embeddings exactly like prose markdown would —
so a user searching "the deploy architecture diagram" can find an image
that has no surrounding text.

Cost handling: every call logs ``[extractor.image] file=… input_tokens=…
output_tokens=…`` at INFO. Per the project's call, there is no gate or
cap — operators monitor the log stream and intervene if a repo turns out
to have hundreds of images. The extractor never retries on LLM errors; a
failed extraction returns ``None`` and the file is skipped (no node, no
silent retry, no token waste).

Vision support is provider-dependent. Claude 3+, GPT-4o, Gemini 1.5+ all
work without configuration. Older models (GPT-3.5, text-only Ollama
configs) will fail the LLM call; the extractor catches the exception and
returns ``None`` so the index pass continues.
"""

from __future__ import annotations

import base64
import logging
from pathlib import Path
from typing import TYPE_CHECKING

from app.core.extractors.protocol import ExtractedDocument

if TYPE_CHECKING:
    from langchain_core.language_models import BaseChatModel

logger = logging.getLogger(__name__)


_DESCRIBE_PROMPT = (
    "You are indexing this image for a technical documentation wiki. "
    "Describe what the image contains in detail — visible text, diagrams, "
    "charts, code snippets, UI elements, architecture, data flow. "
    "Include any text you can read verbatim. Aim for a complete textual "
    "representation that a search engine can match against natural-"
    "language queries. Do not add commentary about the image's quality "
    "or style; describe content only. If the image is blank or contains "
    "no useful content, respond with exactly: EMPTY_IMAGE."
)


_MIME_BY_SUFFIX = {
    ".png": "image/png",
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".gif": "image/gif",
    ".webp": "image/webp",
}


# Anthropic caps image-input at 5 MiB base64-encoded (~3.75 MiB raw);
# OpenAI/Gemini are higher but we use the strictest cap as a
# provider-agnostic safety net. base64-encoded size is ~4/3 the raw
# size, so a raw cap of 3 MiB stays well under 5 MiB base64 and still
# admits typical screenshots / scanned pages.
#
# Files above the cap are skipped with a WARNING (rather than letting
# the provider return a generic 400) so operators see the actual cause.
# Future: per-provider caps could be threaded through if we add provider
# detection to the LLM factory.
_MAX_RAW_BYTES = 3 * 1024 * 1024


class ImageExtractor:
    """Describe images via LLM vision so they're retrievable as text."""

    def __init__(self, *, llm: "BaseChatModel") -> None:
        self._llm = llm

    @property
    def supported_extensions(self) -> tuple[str, ...]:
        return tuple(_MIME_BY_SUFFIX.keys())

    def extract(self, file_path: Path) -> ExtractedDocument | None:
        from langchain_core.messages import HumanMessage

        suffix = file_path.suffix.lower()
        mime = _MIME_BY_SUFFIX.get(suffix)
        if mime is None:
            return None

        try:
            raw = file_path.read_bytes()
        except OSError as exc:
            logger.warning(
                "[extractors.image] failed to read %s: %s", file_path, exc,
            )
            return None

        if not raw:
            return None

        if len(raw) > _MAX_RAW_BYTES:
            # Pre-empt the provider's 400 with an explicit reason so the
            # operator log doesn't say "rate limited" or "invalid_request"
            # for what's actually a too-large image.
            logger.warning(
                "[extractors.image] %s skipped: %.1f MiB exceeds the "
                "%.1f MiB raw-byte cap (Anthropic's strictest "
                "vision-input limit). Resize the image or run a "
                "preprocessor to ingest it.",
                file_path,
                len(raw) / (1024 * 1024),
                _MAX_RAW_BYTES / (1024 * 1024),
            )
            return None

        b64 = base64.b64encode(raw).decode("ascii")
        message = HumanMessage(content=[
            {
                "type": "image_url",
                "image_url": {"url": f"data:{mime};base64,{b64}"},
            },
            {"type": "text", "text": _DESCRIBE_PROMPT},
        ])

        try:
            response = self._llm.invoke([message])
        except Exception as exc:  # noqa: BLE001
            # Provider-side failures (rate limit, vision not supported by
            # this model, bad credentials) all funnel here. Log once at
            # WARNING — re-running the index later will retry.
            logger.warning(
                "[extractors.image] LLM call failed for %s: %s",
                file_path, exc,
            )
            return None

        text = _extract_text(response).strip()
        if not text or text == "EMPTY_IMAGE":
            return None

        in_tokens, out_tokens = _token_usage(response)
        logger.info(
            "[extractors.image] file=%s input_tokens=%d output_tokens=%d",
            file_path.name, in_tokens, out_tokens,
        )

        return ExtractedDocument(
            text=text,
            page_count=1,
            extraction_method="llm-vision",
            input_tokens=in_tokens,
            output_tokens=out_tokens,
        )


def _extract_text(response) -> str:
    """Pull plain text out of a LangChain LLM response.

    Different providers wrap content differently: a single string, a list
    of content blocks, or AIMessage-style ``.content``. Handle the
    common shapes; unknown shapes fall back to ``str(response)``.
    """
    content = getattr(response, "content", response)
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for block in content:
            if isinstance(block, str):
                parts.append(block)
            elif isinstance(block, dict) and block.get("type") == "text":
                parts.append(block.get("text", ""))
        return "".join(parts)
    return str(content)


def _token_usage(response) -> tuple[int, int]:
    """Best-effort token-usage extraction across LangChain providers.

    ``usage_metadata`` is the LangChain-canonical key (set by all major
    providers in recent versions). Falls back to ``response_metadata``
    shapes for older clients. Returns ``(0, 0)`` if neither exists —
    cost logging degrades gracefully rather than failing.
    """
    usage = getattr(response, "usage_metadata", None)
    if isinstance(usage, dict):
        return (
            int(usage.get("input_tokens", 0)),
            int(usage.get("output_tokens", 0)),
        )

    metadata = getattr(response, "response_metadata", None)
    if isinstance(metadata, dict):
        token_usage = metadata.get("token_usage") or metadata.get("usage", {})
        if isinstance(token_usage, dict):
            return (
                int(token_usage.get("prompt_tokens", 0)
                    or token_usage.get("input_tokens", 0)),
                int(token_usage.get("completion_tokens", 0)
                    or token_usage.get("output_tokens", 0)),
            )

    return (0, 0)
