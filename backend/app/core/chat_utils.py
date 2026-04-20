"""Shared utilities for conversation history handling across Q&A engines."""

from __future__ import annotations


def format_chat_history(
    messages: list[dict[str, str]],
    max_messages: int = 4,
    max_chars_per_message: int = 300,
) -> str:
    """Format a list of chat messages into a readable history string.

    Takes the most recent ``max_messages`` turns and truncates each message
    body to ``max_chars_per_message`` characters. Used by both AskEngine and
    DeepResearchEngine to inject prior conversation context into prompts.

    Args:
        messages: List of ``{"role": "user"|"assistant", "content": "..."}`` dicts.
        max_messages: Maximum number of messages to include (from the end).
        max_chars_per_message: Maximum characters per message body before truncation.

    Returns:
        A formatted multi-line string, or an empty string if messages is empty.
    """
    if not messages:
        return ""

    recent = messages[-max_messages:]
    lines = []
    for m in recent:
        role = "User" if m.get("role") == "user" else "Assistant"
        content = m.get("content", "")
        if len(content) > max_chars_per_message:
            content = content[:max_chars_per_message] + "…"
        lines.append(f"{role}: {content}")

    return "\n".join(lines)
