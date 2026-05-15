"""LLM prompts for the surgical-edit regime (#116 PR 3).

The surgical patcher is invoked when a page's only changes are
``MODIFIED`` (or ``MODIFIED + MOVED``) — i.e. the page structure is
intact, only specific symbols' source text shifted. The prompt's job is
to revise prose discussing those symbols without rewriting unrelated
sections, retitling the page, or altering citation paths that weren't
moved.

The quality gate in :class:`PagePatcher` compares the LLM's output to
the original via a token-set similarity score. If too much changed, the
patcher falls back to structural regen.
"""

from __future__ import annotations


SURGICAL_EDIT_SYSTEM = """You are a documentation editor with one job:
update an existing wiki page so its prose accurately describes the
*newly modified versions* of specific code symbols, while leaving every
other part of the page byte-identical.

Hard rules:

1. **Do not change the page title or its section headings.** They feed
   stable URL anchors that other pages link to.
2. **Do not rewrite sections that aren't about the changed symbols.**
   If a paragraph mentions an unchanged symbol, leave it alone.
3. **Preserve every ``<code_context path="...">`` block exactly,** unless
   the path is in the ``moved_paths`` map below (in which case rewrite
   the path attribute only; the body is unchanged).
4. **Match the original tone, voice, and approximate length.** A two-
   paragraph description should remain a two-paragraph description.
5. **Do not invent new code examples or API references.** If you can't
   describe the new symbol behavior from the diff alone, say so in a
   single sentence rather than fabricating details.

Output the full revised page markdown. No commentary, no surrounding
fences, no leading title."""


SURGICAL_EDIT_USER_TEMPLATE = """## Page being edited

Title: {page_title}
Primary symbol: {primary_symbol_id}

## Symbols whose source changed in this regen

{symbol_diffs}

## File paths that moved (rewrite <code_context path=...> attributes only)

{moved_paths}

## Current page markdown

The full current body follows between the ``<original>`` markers.
Replace ONLY the prose about the changed symbols above. Everything else
must be preserved byte-for-byte.

<original>
{current_content}
</original>

Now return the revised page markdown. Start with the first heading and
end with the last line of body; no fences, no explanation."""


def format_symbol_diff(
    symbol_name: str,
    old_source: str,
    new_source: str,
    signature_change: str | None = None,
) -> str:
    """Render one symbol's before/after for inclusion in the user prompt.

    Kept as a standalone function so the prompt assembly is testable
    without instantiating an LLM client. Output is one block per
    symbol, separated by a blank line.
    """
    parts = [f"### {symbol_name}"]
    if signature_change:
        parts.append(f"Signature change: {signature_change}")
    parts.append("```before")
    parts.append(old_source.rstrip())
    parts.append("```")
    parts.append("```after")
    parts.append(new_source.rstrip())
    parts.append("```")
    return "\n".join(parts)
