"""Mermaid diagram extraction and sanitization utilities.

Phase 1 implementation (pure Python heuristics, no external mermaid CLI dependency):
- Extract ```mermaid fenced blocks preserving order & indexes
- Normalize whitespace & line endings
- Heuristic repairs for common breakage patterns (missing header, missing direction, unquoted labels, arrow label quoting, sequence participants)
- Idempotent: valid diagrams remain unchanged
- Structured result objects for future enrichment (hashing, metrics, optional CLI validation)

Future (Phase 2+):
- Optional Node-based mermaid.parse() authoritative validation
- LLM fallback repair for stubborn failures
- Caching (hash -> status)
- More diagram-type specific semantic checks (gantt, classDiagram, stateDiagram)
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, field

# Match a mermaid fence whose closing ``` is on its own line (multiline aware)
MERMAID_FENCE_RE = re.compile(r"^```mermaid[^\n]*\n(.*?)^\s*```[ \t]*\n?", re.DOTALL | re.IGNORECASE | re.MULTILINE)
HEADER_TYPES = {
    "graph",
    "flowchart",
    "sequenceDiagram",
    "classDiagram",
    "stateDiagram",
    "erDiagram",
    "gantt",
    "journey",
    "mindmap",
    "pie",
}
FLOWCHART_LIKE = {"graph", "flowchart"}
SEQUENCE = "sequenceDiagram"
DEFAULT_GRAPH_DIR = "TD"

LABEL_NEEDS_QUOTES_RE = re.compile(r"^(?P<id>[A-Za-z0-9_\-]+)\[(?P<label>[^\[]+?)\]", re.MULTILINE)
# Arrow label simple matcher capturing optional surrounding spaces: A --> |  Label text  | B
RAW_ARROW_LABEL_RE = re.compile(r"-->\s*\|(?P<label>[^\|]+)\|")
PIPE_LABEL_SPACE_RE = re.compile(r"\|\s+([^|]+?)\s+\|")
_PID = r"[A-Za-z0-9_](?:[A-Za-z0-9_\-]*[A-Za-z0-9_])?"  # no trailing hyphen
PARTICIPANT_USE_RE = re.compile(rf"^\s*(?P<lhs>{_PID})\s*-{{1,2}}>>\s*(?P<rhs>{_PID})", re.MULTILINE)
PARTICIPANT_DEF_RE = re.compile(r"^\s*participant\s+(?P<id>[A-Za-z0-9_\-]+)\b", re.MULTILINE)
SMART_QUOTES = {"“": '"', "”": '"', "‘": "'", "’": "'"}
UNICODE_ARROWS = {"→": "->", "⇒": "->", "⟶": "->"}

# Sequence keywords (reserved) intentionally NOT auto-renamed yet per user request.


@dataclass
class DiagramRecord:
    index: int
    original: str
    sanitized: str = ""
    header: str = ""
    was_modified: bool = False
    errors: list[str] = field(default_factory=list)
    fixes: list[str] = field(default_factory=list)
    status: str = "unknown"  # valid | fixed | failed
    hash: str = ""


@dataclass
class SanitizationSummary:
    total: int
    valid: int
    fixed: int
    failed: int
    records: list[DiagramRecord]


@dataclass
class SanitizerConfig:
    default_direction: str = DEFAULT_GRAPH_DIR
    enforce_single_flowchart_arrow: bool = True  # only -->
    auto_add_sequence_participants: bool = True
    max_diagram_chars: int = 8000
    enable_repairs: bool = True


def _hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8", "replace")).hexdigest()[:12]


def extract_mermaid_diagrams(content: str) -> list[DiagramRecord]:
    records: list[DiagramRecord] = []
    for i, m in enumerate(MERMAID_FENCE_RE.finditer(content)):
        body = m.group(1)
        records.append(DiagramRecord(index=i, original=body))
    return records


def _normalize(text: str) -> str:
    # Normalize newlines & strip leading/trailing blank lines
    t = text.replace("\r\n", "\n").replace("\r", "\n")
    lines = t.split("\n")
    # Trim blank edges
    while lines and lines[0].strip() == "":
        lines.pop(0)
    while lines and lines[-1].strip() == "":
        lines.pop()
    # Detab
    lines = [ln.replace("\t", "  ") for ln in lines]
    return "\n".join(lines) + "\n"


def _replace_smart_chars(text: str) -> tuple[str, list[str]]:
    fixes = []
    for bad, good in SMART_QUOTES.items():
        if bad in text:
            text = text.replace(bad, good)
            fixes.append("smart_quotes")
    for bad, good in UNICODE_ARROWS.items():
        if bad in text:
            text = text.replace(bad, good)
            fixes.append("unicode_arrows")
    return text, fixes


def _detect_header(lines: list[str]) -> tuple[str, int]:
    if not lines:
        return "", -1
    first = lines[0].strip()
    # Direct match
    base = first.split()[0] if first else ""
    if base in HEADER_TYPES:
        return base, 0
    return "", -1


def _repair_missing_header(lines: list[str], cfg: SanitizerConfig, fixes: list[str]) -> list[str]:
    # Heuristic: if first lines contain node-like patterns A[Label] or participant usage
    joined = "\n".join(lines[:5])
    if re.search(r"^[A-Za-z0-9_\-]+\[.+\]", joined, re.MULTILINE):
        lines.insert(0, f"flowchart {cfg.default_direction}")
        fixes.append("add_header_flowchart")
    elif re.search(r"^[A-Za-z0-9_\-]+-{1,2}>>", joined, re.MULTILINE):
        lines.insert(0, SEQUENCE)
        fixes.append("add_header_sequence")
    return lines


def _ensure_flowchart_direction(line: str, cfg: SanitizerConfig, fixes: list[str]) -> str:
    parts = line.split()
    if len(parts) == 1:  # just 'flowchart' or 'graph'
        fixes.append("add_direction")
        return f"{parts[0]} {cfg.default_direction}"
    if parts[0] in ("flowchart", "graph") and parts[1] not in {"TD", "LR", "RL", "BT", "TB"}:
        fixes.append("replace_direction")
        return f"{parts[0]} {cfg.default_direction}" + (" " + " ".join(parts[1:]) if len(parts) > 1 else "")
    return line


def _quote_labels(flow_text: str, fixes: list[str]) -> str:
    def repl(m: re.Match) -> str:
        node_id = m.group("id")
        label = m.group("label").rstrip()
        if label.startswith('"') and label.endswith('"'):
            return m.group(0)  # already quoted fully
        # If label contains quotes already, leave inner quotes but wrap entire
        if any(ch in label for ch in (" ", "(", ")", ":", ".", "/", "'")) and not (
            label.startswith('"') and label.endswith('"')
        ):
            fixes.append("quote_labels")
            # Escape existing double quotes by converting to single if needed
            label_clean = label
            return f'{node_id}["{label_clean}"]'
        return m.group(0)

    return LABEL_NEEDS_QUOTES_RE.sub(repl, flow_text)


def _quote_arrow_labels(flow_text: str, fixes: list[str]) -> str:
    def raw_repl(m: re.Match) -> str:
        raw = m.group("label")
        trimmed = raw.strip()
        if raw != trimmed:
            fixes.append("arrow_label_trim")
        if not (trimmed.startswith('"') and trimmed.endswith('"')):
            fixes.append("arrow_label_quotes")
            return f'--> |"{trimmed}"|'
        return f"--> |{trimmed}|"

    return RAW_ARROW_LABEL_RE.sub(raw_repl, flow_text)


def _add_sequence_participants(text: str, fixes: list[str], cfg: SanitizerConfig) -> str:
    if not cfg.auto_add_sequence_participants:
        return text
    lines = text.split("\n")
    if not lines:
        return text
    if lines[0].strip() != SEQUENCE:
        return text
    # Gather declared participants
    declared = {m.group("id") for m in PARTICIPANT_DEF_RE.finditer(text)}
    # Find used participants in message roles
    used_pairs = [(m.group("lhs"), m.group("rhs")) for m in PARTICIPANT_USE_RE.finditer(text)]
    used = set()
    for a, b in used_pairs:
        used.add(a)
        used.add(b)
    # First, correct near-miss usages based on declared participants (e.g., PAR -> PAR_)
    near_miss_map: dict[str, str] = {}
    for u in sorted(used):
        if u in declared:
            continue
        candidates = set()
        # If missing trailing underscore variant exists
        if u + "_" in declared:
            candidates.add(u + "_")
        # If extra trailing underscore present and base exists
        if u.endswith("_") and u[:-1] in declared:
            candidates.add(u[:-1])
        # Apply correction only if exactly one unambiguous candidate
        if len(candidates) == 1:
            near_miss_map[u] = next(iter(candidates))
    if near_miss_map:
        # Replace sender and receiver positions only, per-line, token-aware
        msg_pat = re.compile(
            rf"^(?P<indent>\s*)(?P<src>{_PID})\s*(?P<arrow>-{{1,2}}>>|-->>|->>)\s*(?P<dst>{_PID})(?P<rest>\s*:.*)$"
        )
        new_lines = []
        changed_any = False
        for ln in text.split("\n"):
            m = msg_pat.match(ln)
            if not m:
                new_lines.append(ln)
                continue
            src = m.group("src")
            dst = m.group("dst")
            new_src = near_miss_map.get(src, src)
            new_dst = near_miss_map.get(dst, dst)
            if new_src != src or new_dst != dst:
                changed_any = True
            new_lines.append(f"{m.group('indent')}{new_src}{m.group('arrow')}{new_dst}{m.group('rest')}")
        if changed_any:
            text = "\n".join(new_lines)
        if "sequence_correct_near_miss_participant" not in fixes:
            fixes.append("sequence_correct_near_miss_participant")
        # Recompute used after corrections
        used_pairs = [(m.group("lhs"), m.group("rhs")) for m in PARTICIPANT_USE_RE.finditer(text)]
        used = set()
        for a, b in used_pairs:
            used.add(a)
            used.add(b)
        # Update lines for potential participant insertion
        lines = text.split("\n")
    # Add participants for any remaining missing ones
    missing = [u for u in sorted(used) if u not in declared]
    if missing:
        insertion_index = 1
        for p in reversed(missing):
            lines.insert(insertion_index, f"participant {p} as {p}")
        fixes.append("add_participants")
    return "\n".join(lines)


DUPE_NODE_LABEL_RE = re.compile(r'\[""([^"\]]+?)""\]')  # [""Label""] -> ["Label"]
DUPE_ARROW_LABEL_RE = re.compile(r'\|""([^"|]+?)""\|')  # |""Label""| -> |"Label"|
ARROW_MISBALANCE_LEADING_RE = re.compile(r'\|""([^"|]+?)"\|')  # |""Label"| -> |"Label"|
ARROW_MISBALANCE_TRAILING_RE = re.compile(r'\|"([^"|]+?)""\|')  # |"Label""| -> |"Label"|
NODE_LABEL_NEWLINE_RE = re.compile(r'(\[["])([^\]]*?)(\])')  # crude match to then replace \n inside quoted labels
RESERVED_SEQ_KEYWORDS = {  # subset of mermaid sequence keywords & control words
    "link",
    "click",
    "note",
    "over",
    "alt",
    "loop",
    "par",
    "and",
    "opt",
    "actor",
    "participant",
    "end",
}

# Reserved / problematic identifiers in flowcharts that can collide with control semantics
RESERVED_FLOW_KEYWORDS = {"end"}  # can expand later


def _post_normalize(text: str, fixes: list[str]) -> str:

    # 1. Flowchart reserved node id rename (e.g., end["Done"]) -> end_ / end__
    def _rename_flow_reserved_ids(t: str) -> str:
        if not RESERVED_FLOW_KEYWORDS:
            return t
        pattern = re.compile(
            r"(^|\n)(?P<indent>\s*)(?P<id>" + "|".join(map(re.escape, RESERVED_FLOW_KEYWORDS)) + r')\["'
        )
        changed = False
        used_ids = {m.group(2) for m in re.finditer(r"(^|\n)\s*([A-Za-z0-9_\-]+)\[", t)}

        def repl(m: re.Match) -> str:
            nonlocal changed
            orig = m.group("id")
            new_id = orig + "_"
            while new_id in used_ids or new_id in RESERVED_FLOW_KEYWORDS:
                new_id += "_"
            used_ids.add(new_id)
            changed = True
            return f'{m.group(1)}{m.group("indent")}{new_id}["'

        new_t = pattern.sub(repl, t)
        if changed:
            fixes.append("flow_reserved_node_rename")
        return new_t

    text = _rename_flow_reserved_ids(text)
    # Collapse duplicated quotes in node labels
    new_text = DUPE_NODE_LABEL_RE.sub(r'["\1"]', text)
    if new_text != text:
        fixes.append("collapse_double_quotes")
        text = new_text
    # Collapse duplicated quotes in arrow labels
    new_text = DUPE_ARROW_LABEL_RE.sub(r'|"\1"|', text)
    if new_text != text:
        fixes.append("collapse_double_quotes")
        text = new_text
    # Fix misbalanced leading/trailing duplication cases
    new_text = ARROW_MISBALANCE_LEADING_RE.sub(r'|"\1"|', text)
    if new_text != text:
        fixes.append("collapse_double_quotes")
        text = new_text
    new_text = ARROW_MISBALANCE_TRAILING_RE.sub(r'|"\1"|', text)
    if new_text != text:
        fixes.append("collapse_double_quotes")
        text = new_text

    # Replace literal \n inside quoted node labels with <br/>
    def _newline_repl(m: re.Match) -> str:
        inner = m.group(2)
        if "\\n" in inner:
            replaced = inner.replace("\\n", "<br/>")
            if replaced != inner:
                return f"{m.group(1)}{replaced}{m.group(3)}"
        return m.group(0)

    new_text = NODE_LABEL_NEWLINE_RE.sub(_newline_repl, text)
    if new_text != text:
        fixes.append("label_newline_br")
        text = new_text

    # Arrow label inner quote cleanup: |""some text"?"| -> |"some text?"|
    def _arrow_inner_cleanup(m: re.Match) -> str:
        full = m.group(0)
        inner = m.group(1)
        cleaned = inner
        # remove leading/trailing stray quotes
        cleaned = re.sub(r'^"+', "", cleaned)
        cleaned = re.sub(r'"+$', "", cleaned)
        # collapse quote before punctuation
        cleaned = re.sub(r'"([?!.,;])', r"\1", cleaned)
        if cleaned != inner:
            fixes.append("arrow_label_inner_quote_cleanup")
            return f'|"{cleaned}"|'
        return full

    # Allow internal quotes (stray) inside arrow label so we can clean them; non-greedy match
    text_after = re.sub(r'\|"(.*?)"\|', _arrow_inner_cleanup, text)
    if text_after != text:
        text = text_after

    # Node label inner quote cleanup: [""some text"?" ] -> ["some text?" ]
    def _node_label_inner_cleanup(m: re.Match) -> str:
        full = m.group(0)
        inner = m.group(1)
        # Special early fix: malformed empty list pattern with duplicated opening quotes and escaped bracket
        special = re.sub(r'^"+Return\s+\[\"\]"?$', "Return []", inner)
        if special != inner:
            if "collapse_extra_label_quotes" not in fixes:
                fixes.append("collapse_extra_label_quotes")
            fixes.append("node_label_inner_quote_cleanup")
            return f'["{special}"]'
        cleaned = inner
        # collapse duplicated leading/trailing quotes inside the main quotes
        cleaned = re.sub(r'^"+', "", cleaned)
        cleaned = re.sub(r'"+$', "", cleaned)
        # remove stray quote immediately before punctuation
        cleaned = re.sub(r'"([?!.,;])', r"\1", cleaned)
        # normalize malformed Return empty array variants e.g. Return ["] or Return ["\"] -> Return []
        cleaned2 = re.sub(r'Return\s+\[(?:"\"|"|\"|)\]', "Return []", cleaned)
        if cleaned2 != cleaned:
            cleaned = cleaned2
            if "collapse_extra_label_quotes" not in fixes:
                fixes.append("collapse_extra_label_quotes")
        if cleaned != inner:
            fixes.append("node_label_inner_quote_cleanup")
            if "collapse_extra_label_quotes" not in fixes:
                fixes.append("collapse_extra_label_quotes")
            return f'["{cleaned}"]'
        return full

    new_text2 = re.sub(r'\["((?:[^"]|"(?!\]))*)"\s*\]', _node_label_inner_cleanup, text)
    if new_text2 != text:
        text = new_text2

    # Second-stage fix: if any label still contains Return ["] replace inside quoted label
    def _return_array_stage2(m: re.Match) -> str:
        inner = m.group(1)
        new_inner = re.sub(r'Return\s+\["\]', "Return []", inner)
        if new_inner != inner:
            if "collapse_extra_label_quotes" not in fixes:
                fixes.append("collapse_extra_label_quotes")
            return f'["{new_inner}"]'
        return m.group(0)

    stage2 = re.sub(r'\["((?:[^"]|"(?!\]))*)"\]', _return_array_stage2, text)
    if stage2 != text:
        text = stage2
    # Sequence-specific semicolon -> <br/> inside message bodies (not inside quotes)
    if "sequenceDiagram" in text:
        # 2a. Split multi-target deactivate lines: deactivate A, B -> separate lines
        def _split_multi_deactivate(seq_t: str) -> str:
            lines = seq_t.split("\n")
            out = []
            changed = False
            for ln in lines:
                m = re.match(
                    r"^\s*deactivate\s+([A-Za-z0-9_\-]+\s*,\s*[A-Za-z0-9_\-]+(?:\s*,\s*[A-Za-z0-9_\-]+)*)\s*$", ln
                )
                if m:
                    parts = [p.strip() for p in m.group(1).split(",") if p.strip()]
                    for p in parts:
                        out.append(f"deactivate {p}")
                    changed = True
                else:
                    out.append(ln)
            if changed:
                fixes.append("sequence_split_multi_deactivate")
            return "\n".join(out)

        text = _split_multi_deactivate(text)

        # 2b. Remove unmatched deactivate lines (no prior activate)
        def _remove_unmatched_deactivates(seq_t: str) -> str:
            lines = seq_t.split("\n")
            active = set()
            result = []
            changed = False
            act_pat = re.compile(r"^\s*activate\s+([A-Za-z0-9_\-]+)\s*$")
            de_pat = re.compile(r"^\s*deactivate\s+([A-Za-z0-9_\-]+)\s*$")
            for ln in lines:
                am = act_pat.match(ln)
                if am:
                    active.add(am.group(1))
                    result.append(ln)
                    continue
                dm = de_pat.match(ln)
                if dm and dm.group(1) not in active:
                    changed = True
                    continue  # skip unmatched deactivate
                result.append(ln)
            if changed:
                fixes.append("sequence_remove_unmatched_deactivate")
            return "\n".join(result)

        text = _remove_unmatched_deactivates(text)
        # 2c. Remove standalone 'return' lines
        new_text_ret = re.sub(r"(?m)^\s*return\s*$", "", text)
        if new_text_ret != text:
            fixes.append("sequence_remove_standalone_return")
            text = "\n".join([ln for ln in new_text_ret.split("\n") if ln.strip() != ""])
        # 2d. Remove standalone 'break' lines (not a valid single-line statement in mermaid sequence)
        new_text_break = re.sub(r"(?m)^\s*break\s*$", "", text)
        if new_text_break != text:
            fixes.append("sequence_remove_standalone_break")
            text = "\n".join([ln for ln in new_text_break.split("\n") if ln.strip() != ""])

        def _seq_semicolon(line: str) -> str:
            # match message line with colon
            if ":" not in line or "->" not in line and "-->>" not in line and "->>" not in line:
                return line
            parts = line.split(":", 1)
            prefix, msg = parts[0], parts[1]
            if ";" not in msg:
                return line
            # simple split on semicolons not inside quotes (approx by temporary removal)
            segments = []
            current = ""
            in_quotes = False
            for ch in msg:
                if ch == '"':
                    in_quotes = not in_quotes
                if ch == ";" and not in_quotes:
                    if current.strip():
                        segments.append(current.strip())
                    current = ""
                else:
                    current += ch
            if current.strip():
                segments.append(current.strip())
            if len(segments) > 1:
                replaced = "<br/>".join(segments)
                if replaced != msg.strip():
                    fixes.append("sequence_semicolon_linebreak")
                    return f"{prefix}: {replaced}"
            return line

        new_lines = []
        for ln in text.split("\n"):
            new_lines.append(_seq_semicolon(ln))
        new_text2 = "\n".join(new_lines)
        if new_text2 != text:
            text = new_text2
    # Collapse duplicate deactivate lines for same participant (keep last)
    if "deactivate " in text and "alt " in text:
        lines = text.split("\n")
        last_deactivate_idx: dict[str, int] = {}
        pattern = re.compile(r"^\s*deactivate\s+([A-Za-z0-9_\-]+)\s*$")
        for i, l in enumerate(lines):
            m = pattern.match(l)
            if m:
                last_deactivate_idx[m.group(1)] = i
        if last_deactivate_idx:
            changed_any = False
            for i, l in enumerate(lines):
                m = pattern.match(l)
                if m and last_deactivate_idx.get(m.group(1)) != i:
                    lines[i] = ""
                    changed_any = True
            if changed_any:
                fixes.append("collapse_duplicate_deactivate")
                text = "\n".join([l for l in lines if l != ""])
    return text

    # (unreachable - kept structure clarity)


def sanitize_mermaid_diagram(diagram: str, cfg: SanitizerConfig | None = None) -> tuple[str, list[str], list[str]]:
    """Return (sanitized_text, fixes, errors)."""
    cfg = cfg or SanitizerConfig()
    fixes: list[str] = []
    errors: list[str] = []
    if len(diagram) > cfg.max_diagram_chars:
        errors.append("too_large")
        return diagram, fixes, errors
    text = _normalize(diagram)
    text, smart_fixes = _replace_smart_chars(text)
    fixes.extend(smart_fixes)
    lines = text.split("\n")
    if lines and lines[-1] == "":
        lines.pop()
    header, idx = _detect_header(lines)
    if not header and cfg.enable_repairs:
        before_len = len(lines)
        lines = _repair_missing_header(lines, cfg, fixes)
        if len(lines) != before_len:
            header, idx = _detect_header(lines)
    if header in FLOWCHART_LIKE:
        lines[0] = _ensure_flowchart_direction(lines[0], cfg, fixes)
    # Rejoin for further flowchart-only repairs
    text2 = "\n".join(lines) + "\n"
    if header in FLOWCHART_LIKE or (header == "graph"):
        # Early rename of reserved raw node ids (e.g., end["...") before other transforms
        if re.search(r'(?<![A-Za-z0-9_])end\["', text2):
            text2 = re.sub(r'(?<![A-Za-z0-9_])end\["', 'end_["', text2)
            fixes.append("flow_reserved_node_rename")
        if cfg.enforce_single_flowchart_arrow:
            # Phase 1: Collapse any repeated dashes before > (legacy behavior)
            preliminary = re.sub(r"-+>+", "-->", text2)
            if preliminary != text2:
                fixes.append("normalize_arrows")
                text2 = preliminary

            # Phase 2: Strict normalization – convert dotted / open / cross / mixed arrows to standard -->
            # Mermaid allows variants like -.->, -.-, --o-->, --x-->, but user requirement: unify to '-->'
            def _strict_arrow_norm(match: re.Match) -> str:
                return "-->"  # canonical form

            # Pattern matches: a dash then any run of ., -, o, x (case-insensitive) ending with one or more >
            strict_pattern = re.compile(r"-[.oxX-]*>+")
            strict_applied = strict_pattern.sub(_strict_arrow_norm, text2)
            if strict_applied != text2:
                fixes.append("strict_flow_arrow")
                text2 = strict_applied
            # Phase 3: Clean up any accidental corruption artifacts (#) inside arrow tokens
            corruption_pattern = re.compile(r"-+#>+")
            cleaned = corruption_pattern.sub("-->", text2)
            if cleaned != text2:
                if "strict_flow_arrow" not in fixes:
                    fixes.append("strict_flow_arrow")
                text2 = cleaned

        # Pass: Auto-create node for orphan label following an arrow label (pattern: A --> |"Yes"|"Some Node")
        # Safety: collapse ["'x'"] -> ['x'] globally before other label passes to avoid residual quotes
        pre_fix = re.sub(r"\[\s*\"\s*\'([^\']+)\'\s*\"\s*\]", r"['\1']", text2)
        if pre_fix != text2:
            text2 = pre_fix
            if "label_index_quote_single" not in fixes:
                fixes.append("label_index_quote_single")

        def _auto_nodes_for_orphan_labels(flow: str) -> str:
            lines = flow.split("\n")
            generated_ids = set()
            id_pattern = re.compile(r"^[A-Za-z0-9_]+$")
            # Pre-collect existing ids to avoid collisions
            existing_ids = set(re.findall(r'(^|\n)\s*([A-Za-z0-9_]+)\["', flow))
            existing_ids = {m[1] for m in existing_ids}
            orphan_re = re.compile(
                r'^(?P<indent>\s*)(?P<src>[A-Za-z0-9_]+)\s*-->\s*\|"(?P<edge>[^"]+)"\|\s*"(?P<label>[^"\n]+)"\s*$'
            )
            changed = False

            def _slug(label: str) -> str:
                import unicodedata

                base = unicodedata.normalize("NFKD", label)
                base = re.sub(r"[^A-Za-z0-9]+", "_", base).strip("_").lower() or "auto"
                base = base[:20]
                cand = base
                i = 1
                while cand in existing_ids or cand in generated_ids or not id_pattern.match(cand):
                    cand = f"{base}_{i}"
                    i += 1
                generated_ids.add(cand)
                return cand

            for i, line in enumerate(lines):
                m = orphan_re.match(line)
                if not m:
                    continue
                node_label = m.group("label").strip()
                new_id = _slug(node_label.replace("()", ""))
                lines[i] = f'{m.group("indent")}{m.group("src")} --> |"{m.group("edge")}"| {new_id}["{node_label}"]'
                fixes.append("auto_node_from_orphan_label")
                changed = True
            return "\n".join(lines) if changed else flow

        text2 = _auto_nodes_for_orphan_labels(text2)
        # Quote labels & arrow labels
        text2_before = text2
        text2 = _quote_labels(text2, fixes)
        text2 = _quote_arrow_labels(text2, fixes)
        if text2 != text2_before:
            pass

        # Flow reserved node id rename pass (after quoting to ensure pattern includes [" )
        def _rename_flow_reserved_ids_fc(t: str) -> str:
            if not RESERVED_FLOW_KEYWORDS:
                return t
            pattern = re.compile(
                r"(^|\n)(?P<indent>\s*)(?P<id>" + "|".join(map(re.escape, RESERVED_FLOW_KEYWORDS)) + r')\["'
            )
            used_ids = {m.group(2) for m in re.finditer(r"(^|\n)\s*([A-Za-z0-9_\-]+)\[", t)}
            changed = False

            def repl(m: re.Match) -> str:
                nonlocal changed
                orig = m.group("id")
                new_id = orig + "_"
                while new_id in used_ids or new_id in RESERVED_FLOW_KEYWORDS:
                    new_id += "_"
                used_ids.add(new_id)
                changed = True
                return f'{m.group(1)}{m.group("indent")}{new_id}["'

            new_t = pattern.sub(repl, t)
            if changed:
                fixes.append("flow_reserved_node_rename")
            return new_t

        text2 = _rename_flow_reserved_ids_fc(text2)
        # Fallback simple substitution if still raw reserved id present
        if re.search(r'(^|\n)\s*end\["', text2):
            # Ensure not a subgraph terminator; pattern only matches node definition
            text2 = re.sub(r'(^|\n)(\s*)end\["', lambda m: f'{m.group(1)}{m.group(2)}end_["', text2)
            if "flow_reserved_node_rename" not in fixes:
                fixes.append("flow_reserved_node_rename")

        # Unwrap quoted labels inside decision nodes (diamond): id{"Label"} -> id{Label}
        def _unwrap_decision_quotes(t: str) -> str:
            # Match anywhere in the line: id{"Label"} -> id{Label}
            pattern = re.compile(r'(?P<id>\b[A-Za-z0-9_\-]+)\{\s*"(?P<label>[^"\n]+)"\s*\}')
            changed = False

            def repl(m: re.Match) -> str:
                nonlocal changed
                changed = True
                return f"{m.group('id')}{{{m.group('label')}}}"

            new_t = pattern.sub(repl, t)
            if changed:
                fixes.append("flow_decision_unwrap_quotes")
            return new_t

        text2 = _unwrap_decision_quotes(text2)
    if header == SEQUENCE:
        text2 = _add_sequence_participants(text2, fixes, cfg)

        # Avoid reserved keyword aliases (post participant insertion)
        def _avoid_reserved_aliases(seq_text: str) -> str:
            lines = seq_text.split("\n")
            alias_map: dict[str, str] = {}
            # First pass: adjust participant definitions
            for i, l in enumerate(lines):
                ls = l.strip()
                if not ls.lower().startswith("participant "):
                    continue
                rest = ls[len("participant ") :]
                if not rest:
                    continue
                first = rest.split()[0]
                # Already mapped or safe
                if first.lower() not in RESERVED_SEQ_KEYWORDS:
                    continue
                # Generate new alias
                base = first + "_"
                while base.lower() in RESERVED_SEQ_KEYWORDS or base in alias_map.values():
                    base += "_"
                alias_map[first] = base
                # Replace only the first token after 'participant '
                lines[i] = l.replace("participant " + first, "participant " + base, 1)
            if not alias_map:
                return seq_text
            # Second pass: replace occurrences in messages (sender & receiver positions)
            new_text = "\n".join(lines)
            for old, new in alias_map.items():
                # sender position (start or after newline/indent) before arrow
                # Use doubled braces to represent literal { and } inside f-string
                sender_pattern = re.compile(rf"(?m)(^|\n)([^\n]*?)\b{old}\b(?=-{{1,2}}>>)", re.IGNORECASE)

                def sender_repl(m: re.Match, _old=old, _new=new) -> str:
                    segment = m.group(0)
                    # case sensitive replace last occurrence of old before arrow
                    return re.sub(rf"\b{_old}\b(?=-{{1,2}}>>)", _new, segment)

                new_text = sender_pattern.sub(sender_repl, new_text)
                # receiver position after arrow before colon
                receiver_pattern = re.compile(rf"(-{{1,2}}>>\s*){old}(?=\s*:)", re.IGNORECASE)
                new_text = receiver_pattern.sub(rf"\1{new}", new_text)
            fixes.append("reserved_participant_alias")
            return new_text

        text2 = _avoid_reserved_aliases(text2)

        # Sequence Note semicolon splitting (multi-clause note -> <br/>)
        def _split_note_semicolons(seq_t: str) -> str:
            note_re = re.compile(r"^(?P<prefix>\s*Note\s+(?:right|left|over)\s+[^:]+:\s*)(?P<body>.+)$", re.IGNORECASE)
            out_lines = []
            changed = False
            for ln in seq_t.split("\n"):
                m = note_re.match(ln)
                if not m:
                    out_lines.append(ln)
                    continue
                body = m.group("body")
                segments = []
                cur = ""
                in_q = False
                for ch in body:
                    if ch == '"':
                        in_q = not in_q
                    if ch == ";" and not in_q:
                        if cur.strip():
                            segments.append(cur.strip())
                        cur = ""
                    else:
                        cur += ch
                if cur.strip():
                    segments.append(cur.strip())
                if len(segments) > 1:
                    new_body = "<br/>".join(segments)
                    out_lines.append(f"{m.group('prefix')}{new_body}")
                    fixes.append("sequence_note_semicolon_linebreak")
                    changed = True
                else:
                    out_lines.append(ln)
            return "\n".join(out_lines) if changed else seq_t

        text2 = _split_note_semicolons(text2)

        # Final: ensure usages align with declared participant ids differing only by trailing underscores
        def _align_near_miss_aliases(seq_t: str) -> str:
            declared_ids = [m.group("id") for m in PARTICIPANT_DEF_RE.finditer(seq_t)]
            # Map base id -> underscored id if unique
            base_to_declared: dict[str, str] = {}
            for did in declared_ids:
                if did.endswith("_"):
                    base = did.rstrip("_")
                    # only if base is not also declared explicitly
                    if base not in declared_ids and base not in base_to_declared:
                        base_to_declared[base] = did
            if not base_to_declared:
                return seq_t
            changed = False

            def fix_line(ln: str) -> str:
                nonlocal changed
                m = re.match(
                    r"^(?P<indent>\s*)(?P<src>[A-Za-z0-9_\-]+)\s*(?P<arrow>-{1,2}>>|-->>|->>)\s*(?P<dst>[A-Za-z0-9_\-]+)(?P<rest>\s*:.*)$",
                    ln,
                )
                if not m:
                    return ln
                src = m.group("src")
                dst = m.group("dst")
                src2 = base_to_declared.get(src, src)
                dst2 = base_to_declared.get(dst, dst)
                if src2 != src or dst2 != dst:
                    changed = True
                return f"{m.group('indent')}{src2}{m.group('arrow')}{dst2}{m.group('rest')}"

            new_lines = [fix_line(ln) for ln in seq_t.split("\n")]
            if changed and "sequence_correct_near_miss_participant" not in fixes:
                fixes.append("sequence_correct_near_miss_participant")
            return "\n".join(new_lines)

        text2 = _align_near_miss_aliases(text2)
    # Post normalization (quote collapse, newline/angle handling)
    text2 = _post_normalize(text2, fixes)

    # Label inner-text normalization (dictionary/index quotes, escaped quotes to single, bracket-adjacent quotes)
    def _label_inner_normalize(t: str) -> str:
        # Scan for node labels of the form ["..."], honoring escapes, and fix inner [\"key\"] -> ['key'].
        def fix_inner(inner: str) -> str:
            changed_local = False
            # 0) Fix malformed mixed-quote indexer: ["'x'] or [\"'x'] -> ['x']
            new_inner = re.sub(r"\[\s*\"\s*\'([^\']+)\'\s*\]", r"['\1']", inner)
            if new_inner != inner:
                changed_local = True
            inner = new_inner
            # 1) Convert double-quoted indexers inside label text to single-quoted
            #    Unescaped form: ["key"] -> ['key']
            new_inner = re.sub(r"\[\s*\"([^\"\]\n]+)\"\s*\]", r"['\1']", inner)
            if new_inner != inner:
                changed_local = True
            inner = new_inner
            #    Escaped form (if present in source): [\"key\"] -> ['key']
            new_inner = re.sub(r'\[\s*\\"([^"\\\]]+)\\"\s*\]', r"['\1']", inner)
            if new_inner != inner:
                changed_local = True
            inner = new_inner
            # 2) Trim bracket-adjacent quotes if any artifacts remain
            new_inner = re.sub(r'\[([^\]"\n]+)"\]', r"[\1]", inner)
            if new_inner != inner:
                changed_local = True
            inner = new_inner
            new_inner = re.sub(r'\["([^\]"\n]+)\]', r"[\1]", inner)
            if new_inner != inner:
                changed_local = True
            inner = new_inner
            # 3) Convert escaped quotes around words to single quotes inside the label text: \"text\" -> 'text'
            new_inner = re.sub(r'\\"([^"\\]+)\\"', r"'\1'", inner)
            if new_inner != inner:
                changed_local = True
            inner = new_inner
            # 4) Remove stray double quote immediately before a bracketed single-quoted indexer
            new_inner = re.sub(r'"\s*(\[\s*\'[^\]]+\'\s*\])', r"\1", inner)
            if new_inner != inner:
                changed_local = True
            inner = new_inner

            # 5) Normalize generic-type brackets robustly (scan): BaseStore["str", Document] -> BaseStore[str, Document]
            def _normalize_generics(s: str) -> tuple[str, bool]:
                i = 0
                out: list[str] = []
                changed = False
                n = len(s)
                while i < n:
                    m = re.match(r"([A-Za-z0-9_]+)\[", s[i:])
                    if not m:
                        out.append(s[i])
                        i += 1
                        continue
                    type_name = m.group(1)
                    start = i + m.end()  # position after '['
                    # find matching closing ']' accounting for nested [] and quotes
                    j = start
                    depth = 1
                    in_q = False
                    qch = ""
                    while j < n:
                        ch = s[j]
                        if in_q:
                            if ch == qch:
                                in_q = False
                            elif ch == "\\" and j + 1 < n:
                                j += 1  # skip escaped char
                            j += 1
                            continue
                        if ch in ('"', "'"):
                            in_q = True
                            qch = ch
                            j += 1
                            continue
                        if ch == "[":
                            depth += 1
                            j += 1
                            continue
                        if ch == "]":
                            depth -= 1
                            j += 1
                            if depth == 0:
                                break
                            continue
                        j += 1
                    if depth != 0:
                        # Unmatched, emit char and continue
                        out.append(s[i])
                        i += 1
                        continue
                    content = s[start : j - 1] if j - 1 >= start else ""  # exclude the ']'
                    # Split content by commas at top-level (ignore brackets/quotes)
                    tokens: list[str] = []
                    buf = ""
                    d = 0
                    inq = False
                    q = ""
                    for ch in content:
                        if inq:
                            if ch == q:
                                inq = False
                            elif ch == "\\" and q == '"':
                                # keep escape but move on
                                pass
                            buf += ch
                            continue
                        if ch in ('"', "'"):
                            inq = True
                            q = ch
                            buf += ch
                            continue
                        if ch == "[":
                            d += 1
                            buf += ch
                            continue
                        if ch == "]":
                            if d > 0:
                                d -= 1
                            buf += ch
                            continue
                        if ch == "," and not inq and d == 0:
                            tokens.append(buf.strip())
                            buf = ""
                            continue
                        buf += ch
                    if buf.strip():
                        tokens.append(buf.strip())
                    if len(tokens) >= 2:

                        def strip_outer_quotes(tok: str) -> str:
                            return re.sub(r'^["\']?(.*?)["\']?$', r"\1", tok.strip())

                        cleaned = [strip_outer_quotes(tk) for tk in tokens]
                        out.append(f"{type_name}[{', '.join(cleaned)}]")
                        i = j
                        changed = True
                    else:
                        # Not a generic (no top-level comma); emit one char and continue
                        out.append(s[i])
                        i += 1
                return "".join(out), changed

            new_inner, gen_changed = _normalize_generics(inner)
            if gen_changed:
                inner = new_inner
                changed_local = True
                if "label_generic_bracket_tokens_unquote" not in fixes:
                    fixes.append("label_generic_bracket_tokens_unquote")
            # 5b) Fallback: remove double-quotes around simple tokens inside [ ... ] when followed by comma or closing bracket
            # This helps when the generic scanner fails due to unexpected constructs
            new_inner = re.sub(r'(\[|,\s*)"([^"\[\],]+)"(?=\s*(?:,|\]))', r"\1\2", inner)
            if new_inner != inner:
                inner = new_inner
                changed_local = True
                if "label_generic_bracket_tokens_unquote" not in fixes:
                    fixes.append("label_generic_bracket_tokens_unquote")
            if changed_local and "label_index_quote_single" not in fixes:
                fixes.append("label_index_quote_single")
            return inner

        res: list[str] = []
        i = 0
        n = len(t)
        while i < n:
            if t[i] == "[" and i + 1 < n and t[i + 1] == '"':
                j = i + 2
                found = False
                depth = 0
                while j < n:
                    ch = t[j]
                    prev = t[j - 1] if j > 0 else ""
                    if ch == "[" and prev != "\\":
                        depth += 1
                        j += 1
                        continue
                    if ch == "]" and prev != "\\" and depth > 0:
                        depth -= 1
                        j += 1
                        continue
                    if ch == '"' and prev != "\\" and depth == 0:
                        # Allow optional whitespace before closing bracket
                        k = j + 1
                        while k < n and t[k] in (" ", "\t"):  # preserve newline as non-terminating
                            k += 1
                        if k < n and t[k] == "]":
                            inner = t[i + 2 : j]
                            fixed = fix_inner(inner)
                            res.append('["' + fixed + '"]')
                            i = k + 1
                            found = True
                            break
                    j += 1
                if not found:
                    # Not a well-formed label, emit current char and continue
                    res.append(t[i])
                    i += 1
            else:
                res.append(t[i])
                i += 1
        t2 = "".join(res)

        def arrow_repl(m: re.Match) -> str:
            inner = m.group(1)
            changed = False
            # d) Escaped double quotes -> single quotes inside arrow labels
            inner2 = re.sub(r'\\"([^\\"]+?)\\"', r"'\\1'", inner)
            if inner2 != inner:
                changed = True
            inner = inner2
            # a) Indexer double-quotes to single within arrow labels (no extra quotes around brackets)
            inner2 = re.sub(r'\[(\s)*"([^"\]]+?)"(\s*)\]', r"[\1'\2'\3]", inner)
            if inner2 != inner:
                changed = True
            inner = inner2
            # a2) Remove stray double quotes around bracketed single-quoted indexers
            inner2 = re.sub(r"\[\s*\"\s*\'([^\']+)\'\s*\"\s*\]", r"['\1']", inner)
            if inner2 != inner:
                changed = True
            inner = inner2
            # c) Trim bracket-adjacent quotes
            inner2 = re.sub(r"\[([^\"\n]+)\"\]", r"[\1]", inner)
            if inner2 != inner:
                changed = True
            inner = inner2
            inner2 = re.sub(r"\[\"([^\"\n]+)\]", r"[\1]", inner)
            if inner2 != inner:
                changed = True
            inner = inner2
            if changed:
                if "label_escaped_dquote_to_single" not in fixes:
                    fixes.append("label_escaped_dquote_to_single")
                return f'|"{inner}"|'
            return m.group(0)

        t3 = re.sub(r"\|\"(.+?)\"\|", arrow_repl, t2)
        # Final arrow pass: collapse [\"'x'\"] -> ['x'] if still present
        t3_fix = re.sub(r"\[\s*\"\s*\'([^\']+)\'\s*\"\s*\]", r"['\1']", t3)
        if t3_fix != t3:
            t3 = t3_fix
        return t3

    text2 = _label_inner_normalize(text2)

    # Normalize generic-type brackets inside node labels globally: BaseStore["str", Document] -> BaseStore[str, Document]
    def _normalize_label_generics(t: str) -> str:
        def node_repl(m: re.Match) -> str:
            inner = m.group("inner")
            # Unescape generic inner quotes for processing: \"str\" -> "str"
            inner = inner.replace('\\"', '"')
            # Fix stray double-quote immediately before a generic bracket after an identifier: Foo"[T] -> Foo[T]
            inner = re.sub(r'([A-Za-z0-9_])"\[', r"\1[", inner)

            def gen_repl(gm: re.Match) -> str:
                type_name = gm.group(1)
                content = gm.group(2)
                content = content.replace('\\"', '"')
                parts = [p.strip() for p in content.split(",")]
                cleaned = []
                for p in parts:
                    p2 = re.sub(r'^["\']', "", p)
                    p2 = re.sub(r'["\']$', "", p2)
                    cleaned.append(p2)
                return f"{type_name}[{', '.join(cleaned)}]"

            new_inner = re.sub(r"([A-Za-z0-9_]+)\[\s*([^\[\]]*?,[^\[\]]*?)\s*\]", gen_repl, inner)
            if new_inner != inner and "label_generic_bracket_tokens_unquote" not in fixes:
                fixes.append("label_generic_bracket_tokens_unquote")
            return f'{m.group("id")}["{new_inner}"]'

        return re.sub(r'(?P<id>[A-Za-z0-9_\-]+)\["(?P<inner>(?:[^"\\]|\\.(?!\]))*)"\]', node_repl, t)

    text2 = _normalize_label_generics(text2)
    # Global escape-to-single inside diagrams as safety net
    text2_new = re.sub(r'\\"([^\\"]+?)\\"', r"'\1'", text2)
    # Also convert literal double-quoted arguments inside function-like calls in labels: ("...") -> ('...')
    text2_new2 = re.sub(r'\((\s*)"([^"\n]+)"(\s*)\)', r"(\1'\2'\3)", text2_new)
    if text2_new2 != text2_new:
        text2_new = text2_new2
    if text2_new != text2:
        text2 = text2_new
        if "label_escaped_dquote_to_single" not in fixes:
            fixes.append("label_escaped_dquote_to_single")
    # Global collapse of ["'x'"] -> ['x'] in any context (safety net for arrow labels)
    text2_fix2 = re.sub(r"\[\s*\"\s*\'([^\']+)\'\s*\"\s*\]", r"['\1']", text2)
    if text2_fix2 != text2:
        text2 = text2_fix2
        if "label_index_quote_single" not in fixes:
            fixes.append("label_index_quote_single")
    # Remove stray quote before a bracketed single-quoted index globally: "['x'] -> ['x']
    text2_fix = re.sub(r'"\s*(\[\s*\'[^\]]+\'\s*\])', r"\1", text2)
    if text2_fix != text2:
        text2 = text2_fix
        if "label_index_quote_single" not in fixes:
            fixes.append("label_index_quote_single")

    # 3. Late pass: quote any remaining unquoted labels containing space/punct
    def _late_label_quote(t: str) -> str:
        # Quote id[label] patterns that are outside of any double-quoted segments in the line
        pat = re.compile(r'(?P<id>[A-Za-z0-9_\-]+)\[(?P<label>(?!")[^"\]\n]+)\]')
        triggers_base = " ()/:.<>{}!?"
        triggers = set(triggers_base) | {'"'}
        changed = False
        out_lines: list[str] = []
        for line in t.split("\n"):
            i = 0
            n = len(line)
            in_q = False
            res: list[str] = []
            while i < n:
                ch = line[i]
                if ch == '"':
                    in_q = not in_q
                    res.append(ch)
                    i += 1
                    continue
                if not in_q:
                    m = pat.match(line, i)
                    if m:
                        lab = m.group("label")
                        if any(c in lab for c in triggers):
                            res.append(f'{m.group("id")}["{lab}"]')
                            i = m.end()
                            changed = True
                            continue
                res.append(ch)
                i += 1
            out_lines.append("".join(res))
        if changed:
            fixes.append("late_label_quote")
        return "\n".join(out_lines)

    text2 = _late_label_quote(text2)

    # 4. Merge fragmented multiline labels "part1"<br/>"part2" (nodes & arrows)
    def _merge_fragments(t: str) -> str:
        changed = False

        def node_repl(m: re.Match) -> str:
            nonlocal changed
            changed = True
            return f'["{m.group(1)}<br/>{m.group(2)}"]'

        def arrow_repl(m: re.Match) -> str:
            nonlocal changed
            changed = True
            return f'|"{m.group(1)}<br/>{m.group(2)}"|'

        new_t = re.sub(r'\["([^"\]]+)"<br/>"([^"\]]+)"\]', node_repl, t)
        new_t = re.sub(r'\|"([^"|]+)"<br/>"([^"|]+)"\|', arrow_repl, new_t)
        if changed:
            fixes.append("merge_fragmented_multiline_label")
        return new_t

    text2 = _merge_fragments(text2)

    # 5. Normalize inner escaped double quotes to single quotes inside labels
    def _normalize_inner_quotes(t: str) -> str:
        changed = False

        def norm_label(match: re.Match) -> str:
            nonlocal changed
            inner = match.group(1)
            new_inner = re.sub(r'\\"([^\\"]+?)\\"', r"'\\1'", inner)
            # Also convert occurrences of {'key': "value"} inside larger labels to single-quoted values
            # First handle escaped pattern {'key': \"value\"}
            converted = re.sub(r"(\{'[^']+'\s*:\s*)\\\"([^\\\"]+)\\\"", r"\1'\2'", new_inner)
            # Then handle unescaped pattern {'key': "value"}
            converted2 = re.sub(r"(\{'[^']+'\s*:\s*)\"([^\"\n]+)\"", r"\1'\2'", converted)
            if converted2 != new_inner:
                new_inner = converted2
            if converted != new_inner:
                new_inner = converted

                # Normalize generic-type brackets inside labels: BaseStore["str", Document] -> BaseStore[str, Document]
                def _generic_repl(m: re.Match) -> str:
                    type_name = m.group(1)
                    content = m.group(2)
                    parts = [p.strip() for p in content.split(",")]
                    cleaned: list[str] = []
                    for p in parts:
                        # Remove a single leading/trailing quote if present
                        p2 = re.sub(r'^["\']', "", p)
                        p2 = re.sub(r'["\']$', "", p2)
                        cleaned.append(p2)
                    return f"{type_name}[{', '.join(cleaned)}]"

                pattern_generics = re.compile(r"([A-Za-z0-9_]+)\[\s*([^\[\]]*?,[^\[\]]*?)\s*\]")
                new_inner2 = pattern_generics.sub(_generic_repl, new_inner)
                if new_inner2 != new_inner:
                    new_inner = new_inner2
                    changed = True
                    if "label_generic_bracket_tokens_unquote" not in fixes:
                        fixes.append("label_generic_bracket_tokens_unquote")
            if new_inner != inner:
                changed = True
                return f'["{new_inner}"]'
            return match.group(0)

        t2 = re.sub(r'\["((?:[^\"]|\"(?!\]))*)"\]', norm_label, t)

        # Arrow labels
        def norm_arrow(ma: re.Match) -> str:
            nonlocal changed
            inner = ma.group(1)
            new_inner = re.sub(r'\\"([^\\"]+?)\\"', r"'\\1'", inner)
            if new_inner != inner:
                changed = True
                return f'|"{new_inner}"|'
            return ma.group(0)

        t3 = re.sub(r'\|"([^"|]+)"\|', norm_arrow, t2)

        # Fallback: within node/arrow labels already processed, catch any remaining {'key': "value"}
        def fallback(m: re.Match) -> str:
            inner = m.group(1)
            converted = re.sub(r"(\{'[^']+'\s*:\s*)\"([^\"\n]+)\"", r"\1'\2'", inner)
            return f'"{converted}"'

        t4 = re.sub(r'"([^"\n]*\{[^}]*\}[^"\n]*?)"', fallback, t3)
        if t4 != t3 and "normalize_inner_quotes" not in fixes:
            fixes.append("normalize_inner_quotes")
            return t4
        if changed and "normalize_inner_quotes" not in fixes:
            fixes.append("normalize_inner_quotes")
        return t3

    text2 = _normalize_inner_quotes(text2)

    # Subgraph rewrite: replace edges using subgraph id with representative first node id
    def _subgraph_rewrite(flow: str) -> str:
        lines = flow.split("\n")
        reps: dict[str, str] = {}
        inside = None
        for _i, ln in enumerate(lines):
            m = re.match(r'^\s*subgraph\s+([A-Za-z0-9_\-]+)(?:\["[^\]]+"\])?', ln)
            if m:
                inside = m.group(1)
                reps.setdefault(inside, None)
                continue
            if inside and re.match(r"^\s*end\s*$", ln):
                inside = None
                continue
            if inside and reps.get(inside) is None:
                n = re.match(r'^\s*([A-Za-z0-9_\-]+)\["', ln)
                if n:
                    reps[inside] = n.group(1)
        # Autocreate representative if missing
        for sg, rep in list(reps.items()):
            if not rep:
                reps[sg] = f"{sg}_entry"
        # Rewrite edges referencing subgraph id to representative
        changed = False
        out = []
        for ln in lines:
            for sg, rep in reps.items():
                # src replacement
                ln2 = re.sub(rf"(^|\s)({re.escape(sg)})(\s*-->)", rf"\1{rep}\3", ln)
                # dst replacement
                ln2 = re.sub(rf'(-->|\|"[^"|]+"\|\s*)({re.escape(sg)})(\s*\[)', rf"\1{rep}\3", ln2)
                if ln2 != ln:
                    changed = True
                ln = ln2
            out.append(ln)
        if changed:
            fixes.append("subgraph_id_edge_rewrite")
        return "\n".join(out)

    if header in FLOWCHART_LIKE:
        text2 = _subgraph_rewrite(text2)
    # Sequence: opt followed by else => alt
    if header == SEQUENCE:

        def _opt_to_alt_if_else(t: str) -> str:
            lines = t.split("\n")
            out = []
            i = 0
            changed = False
            while i < len(lines):
                ln = lines[i]
                m = re.match(r"^\s*opt\b(.*)$", ln)
                if m:
                    # scan ahead until end
                    j = i + 1
                    saw_else = False
                    while j < len(lines) and not re.match(r"^\s*end\s*$", lines[j]):
                        if re.match(r"^\s*else\b", lines[j]):
                            saw_else = True
                            break
                        j += 1
                    if saw_else:
                        out.append("alt" + m.group(1))
                        changed = True
                    else:
                        out.append(ln)
                else:
                    out.append(ln)
                i += 1
            if changed:
                fixes.append("sequence_opt_to_alt_for_else")
            return "\n".join(out)

        text2 = _opt_to_alt_if_else(text2)

        # Collapse backslash-newline continuations in message labels
        def _collapse_line_continuations(t: str) -> str:
            changed = False

            def msg_repl(m: re.Match) -> str:
                nonlocal changed
                msg = m.group("msg")
                joined = re.sub(r"\\\s*\n\s*", " ", msg)
                # Normalize commas: no spaces before ',', exactly one space after
                joined = re.sub(r"\s+,", ",", joined)
                joined = re.sub(r",\s*", ", ", joined)
                # Trim spaces right after '(' and before ')'
                joined = re.sub(r"\(\s+", "(", joined)
                joined = re.sub(r"\s+\)", ")", joined)
                if joined != msg:
                    changed = True
                return f"{m.group('lhs')}{m.group('arrow')}{m.group('rhs')}: {joined}"

            new_t = re.sub(
                r"^(?P<lhs>[A-Za-z0-9_\-]+)\s*(?P<arrow>-{1,2}>>|-->>|->>)\s*(?P<rhs>[A-Za-z0-9_\-]+):\s*(?P<msg>.+)$",
                msg_repl,
                t,
                flags=re.MULTILINE,
            )
            if changed:
                fixes.append("sequence_label_line_continuation_collapse")
            return new_t

        text2 = _collapse_line_continuations(text2)
        # Global fallback: collapse any backslash-newline continuations across lines (inside sequence only)
        seq_collapse = re.sub(r"\\\s*\n\s*", " ", text2)
        if seq_collapse != text2:
            text2 = seq_collapse
            if "sequence_label_line_continuation_collapse" not in fixes:
                fixes.append("sequence_label_line_continuation_collapse")

        # Normalize spacing inside message parts after colon on sequence lines
        def _norm_msg_spaces(seq_t: str) -> str:
            def repl(m: re.Match) -> str:
                prefix = m.group(1)
                msg = m.group(2)
                msg2 = re.sub(r"\(\s+", "(", msg)
                msg2 = re.sub(r"\s+\)", ")", msg2)
                msg2 = re.sub(r"\s+,", ",", msg2)
                msg2 = re.sub(r",\s*", ", ", msg2)
                return f"{prefix}{msg2}"

            return re.sub(r"^(.*?->>.*?:\s*)(.+)$", repl, text2, flags=re.MULTILINE)

        text2 = _norm_msg_spaces(text2)

        # Fallback: collapse multiline parentheses in messages even without backslashes
        def _join_multiline_paren(seq_t: str) -> str:
            lines = seq_t.split("\n")
            out_lines: list[str] = []
            i = 0
            changed = False
            msg_re = re.compile(r"^(?P<head>\s*[A-Za-z0-9_\-]+\s*-{1,2}>>\s*[A-Za-z0-9_\-]+:\s*)(?P<msg>.+)$")
            while i < len(lines):
                ln = lines[i]
                m = msg_re.match(ln)
                if not m:
                    out_lines.append(ln)
                    i += 1
                    continue
                head = m.group("head")
                msg = m.group("msg")
                # If message contains '(' and unbalanced at end of line, accumulate
                if "(" in msg and msg.count("(") > msg.count(")"):
                    buf = [msg]
                    bal = msg.count("(") - msg.count(")")
                    j = i + 1
                    while j < len(lines) and bal > 0:
                        seg = lines[j].strip()
                        buf.append(seg)
                        bal += seg.count("(") - seg.count(")")
                        j += 1
                    joined = " ".join(buf)
                    # spacing normalization
                    joined = re.sub(r"\s+,", ",", joined)
                    joined = re.sub(r",\s*", ", ", joined)
                    joined = re.sub(r"\(\s+", "(", joined)
                    joined = re.sub(r"\s+\)", ")", joined)
                    out_lines.append(head + joined)
                    i = j
                    changed = True
                else:
                    out_lines.append(ln)
                    i += 1
            return "\n".join(out_lines) if changed else seq_t

        text2 = _join_multiline_paren(text2)

    # Final flow-specific cleanup for extra leading/trailing quotes, empty array artifacts inside labels
    def _final_flow_label_cleanup(t: str) -> str:
        before = t
        # 1. Collapse duplicated leading quotes only if they start a quoted label token
        t = re.sub(r'(\[[ \t]*)"{2,}', r'\1"', t)
        # 2. Collapse duplicated trailing quotes directly before ] but retain one quote
        t = re.sub(r'"{2,}(\])', r'"\1', t)
        # 3. Normalize specific malformed Return [] pattern
        t2 = re.sub(r'\["Return\s+\["\]"\]', '["Return []"]', t)
        if t2 != t:
            t = t2

        # 3b. Inside any node label content, replace Return ["] -> Return []
        def return_arr_fix(m: re.Match) -> str:
            inner = m.group("inner")
            new_inner = re.sub(r'Return\s+\["\]', "Return []", inner)
            if new_inner != inner:
                return f'{m.group("id")}["{new_inner}"]'
            return m.group(0)

        t3 = re.sub(r'(?P<id>[A-Za-z0-9_\-]+)\["(?P<inner>[^"\]\n]+)"\]', return_arr_fix, t)
        if t3 != t:
            t = t3
            if "collapse_extra_label_quotes" not in fixes:
                fixes.append("collapse_extra_label_quotes")
        # 4. Restore any labels missing a closing quote (pattern: id["label])
        restore_pat = re.compile(r'(?P<id>[A-Za-z0-9_\-]+)\["(?P<label>[^"\]\n]+)\]')

        def restore(m: re.Match) -> str:
            return f'{m.group("id")}["{m.group("label")}"]'

        t3 = restore_pat.sub(restore, t)
        if t3 != t:
            t = t3
            if "restore_label_trailing_quote" not in fixes:
                fixes.append("restore_label_trailing_quote")
        # 5. Tag if anything changed overall
        if t != before and "collapse_extra_label_quotes" not in fixes:
            fixes.append("collapse_extra_label_quotes")
        return t

    text2 = _final_flow_label_cleanup(text2)
    # Global bracket-adjacent stray quote fix: [name"] -> [name]
    text2_g = re.sub(r'\[([A-Za-z0-9_\.]+)"\]', r"[\1]", text2)
    if text2_g != text2:
        text2 = text2_g
        if "label_bracket_adjacent_quote_trim" not in fixes:
            fixes.append("label_bracket_adjacent_quote_trim")
    # Trim accidental "]"] at node ends inside labels
    text2_trim = text2.replace('"]"]', "]]")
    if text2_trim != text2:
        text2 = text2_trim
        if "label_array_close_quote_trim" not in fixes:
            fixes.append("label_array_close_quote_trim")

    # Final pass: robust normalization of any lingering Return [<only quotes or escapes>] -> Return []
    def _return_array_final(m: re.Match) -> str:
        inner = m.group(1)
        new_inner = re.sub(r'Return\s+\[(?:\\?"\s*)+\]', "Return []", inner)
        if new_inner == inner and re.search(r"Return\s+\[$", inner):
            new_inner = re.sub(r"Return\s+\[$", "Return []", inner)
        if new_inner != inner:
            if "collapse_extra_label_quotes" not in fixes:
                fixes.append("collapse_extra_label_quotes")
            return f'["{new_inner}"]'
        return m.group(0)

    text2_final = re.sub(r'\["((?:[^"]|"(?!\]))*)"\]', _return_array_final, text2)
    if text2_final != text2:
        text2 = text2_final
    # Last-chance cleanup: collapse ["'x'"] -> ['x'] anywhere in the final text
    text2_final2 = re.sub(r"\[\s*\"\s*\'([^\']+)\'\s*\"\s*\]", r"['\1']", text2)
    if text2_final2 != text2:
        text2 = text2_final2
        if "label_index_quote_single" not in fixes:
            fixes.append("label_index_quote_single")
    # Last-chance cleanup 2: handle leading-only double quote inside bracket ["'x'] -> ['x']
    text2_final3 = re.sub(r"\[\s*\"\s*\'([^\']+)\'\s*\]", r"['\1']", text2)
    if text2_final3 != text2:
        text2 = text2_final3
        if "label_index_quote_single" not in fixes:
            fixes.append("label_index_quote_single")

    # Late pass: inside node labels, remove stray double quote immediately before a single-quoted indexer
    def _strip_quote_before_single_indexer(m: re.Match) -> str:
        inner = m.group("inner")
        fixed = re.sub(r'"\s*(\[\s*\'[^\]]+\'\s*\])', r"\1", inner)
        if fixed != inner and "label_index_quote_single" not in fixes:
            fixes.append("label_index_quote_single")
        return f'{m.group("id")}["{fixed}"]'

    text2_late = re.sub(
        r'(?P<id>[A-Za-z0-9_\-]+)\["(?P<inner>(?:[^"\\]|\\.(?!\]))*)"\]', _strip_quote_before_single_indexer, text2
    )
    if text2_late != text2:
        text2 = text2_late

    # Final generics cleanup inside node labels: remove stray quote before '[' after identifiers and unquote simple tokens inside []
    def _final_generics_fix(m: re.Match) -> str:
        inner = m.group("inner")
        # Remove stray quote before bracket following an identifier: Foo"[T] -> Foo[T]
        inner2 = re.sub(r'([A-Za-z0-9_])"\[', r"\1[", inner)

        # Inside each bracket group, remove double-quotes around simple tokens at top level
        def fix_group(mm: re.Match) -> str:
            content = mm.group(1)
            # Unescape any escaped quotes first
            content = content.replace('\\"', '"')
            # Unquote simple tokens separated by commas at top-level
            parts = []
            buf = ""
            depth = 0
            inq = False
            q = ""
            for ch in content:
                if inq:
                    if ch == q:
                        inq = False
                    elif ch == "\\" and q == '"':
                        pass
                    buf += ch
                    continue
                if ch in ('"', "'"):
                    inq = True
                    q = ch
                    buf += ch
                    continue
                if ch == "[":
                    depth += 1
                    buf += ch
                    continue
                if ch == "]":
                    if depth > 0:
                        depth -= 1
                    buf += ch
                    continue
                if ch == "," and not inq and depth == 0:
                    parts.append(buf.strip())
                    buf = ""
                    continue
                buf += ch
            if buf.strip():
                parts.append(buf.strip())
            if len(parts) >= 1:

                def stripq(tok: str) -> str:
                    return re.sub(r'^["\']?(.*?)["\']?$', r"\1", tok)

                parts2 = [stripq(p) for p in parts]
                return "[" + ", ".join(parts2) + "]"
            return "[" + content + "]"

        inner3 = re.sub(r"\[([^\[\]]*?)\]", fix_group, inner2)
        return f'{m.group("id")}["{inner3}"]'

    text2_final_gen = re.sub(
        r'(?P<id>[A-Za-z0-9_\-]+)\["(?P<inner>(?:[^"\\]|\\.(?!\]))*)"\]', _final_generics_fix, text2
    )
    if text2_final_gen != text2:
        text2 = text2_final_gen
    return text2, list(dict.fromkeys(fixes)), errors


def sanitize_content(content: str, cfg: SanitizerConfig | None = None) -> tuple[str, SanitizationSummary]:
    cfg = cfg or SanitizerConfig()

    # Log original content for debugging fence issues
    # logger = logging.getLogger(__name__)
    # logger.info(f"SANITIZER_INPUT: {repr(content)}")

    # Use finditer to get the exact spans for replacement
    output_parts: list[str] = []
    last_index = 0
    records: list[DiagramRecord] = []

    # Pre-pass: fence normalization (mermaid-only): split inline closers within mermaid blocks, autoclose unmatched
    def _normalize_fences(md: str) -> tuple[str, list[str]]:
        fx: list[str] = []
        out: list[str] = []
        inside_mermaid = False

        def _find_ticks_outside_quotes(s: str) -> int:
            in_quotes = False
            i = 0
            while i <= len(s) - 3:
                ch = s[i]
                if ch == '"':
                    # toggle quotes if not escaped
                    if i == 0 or s[i - 1] != "\\":
                        in_quotes = not in_quotes
                    i += 1
                    continue
                if not in_quotes and s[i : i + 3] == "```":
                    return i
                i += 1
            return -1

        for ln in md.split("\n"):
            stripped = ln.strip()
            if not inside_mermaid:
                if re.match(r"^```mermaid\b", stripped, re.IGNORECASE):
                    inside_mermaid = True
                    out.append(ln)
                else:
                    out.append(ln)
                continue
            # inside mermaid
            # 1) Standalone closer
            if stripped == "```":
                inside_mermaid = False
                out.append(ln)
                continue
            # 2) Inline closer anywhere in the line: split into pre, ``` on its own line, and post outside mermaid
            # But don't split if this line starts with ```mermaid (opening fence)
            tick_pos = _find_ticks_outside_quotes(ln)
            if tick_pos != -1 and not re.match(r"^\s*```mermaid\b", ln, re.IGNORECASE):
                pre = ln[:tick_pos].rstrip()
                post = ln[tick_pos + 3 :]
                if pre:
                    out.append(pre)
                out.append("```")
                inside_mermaid = False
                if post:
                    # Leave trailing content unchanged; only move the closing fence to its own line
                    out.append(post)
                fx.append("fence_split_inline_closer")
                continue
            out.append(ln)
        if inside_mermaid:
            out.append("```")
            fx.append("fence_autoclose_unmatched")
        return ("\n".join(out), fx)

    content, fence_fixes = _normalize_fences(content)

    for i, m in enumerate(MERMAID_FENCE_RE.finditer(content)):
        start, end = m.span()
        body = m.group(1)
        # Append text before diagram (from last cursor to this fence start)
        output_parts.append(content[last_index:start])
        record = DiagramRecord(index=i, original=body)
        sanitized, fixes, errors = sanitize_mermaid_diagram(body, cfg)
        record.sanitized = sanitized
        record.was_modified = sanitized != body
        # Consider trailing-newline-only differences as not modified (idempotency for valid diagrams)
        if record.was_modified and sanitized.rstrip("\n") == body.rstrip("\n"):
            record.was_modified = False
        record.fixes = fixes
        record.errors = errors
        record.status = "failed" if errors else ("fixed" if record.was_modified else "valid")
        record.hash = _hash(record.sanitized if not errors else record.original)
        # Reconstruct fenced block safely:
        # 1) Ensure the diagram body ends with a newline so the closing fence is on its own line
        #    (prevents inline closers like '... deactivate R```')
        body = record.sanitized
        if not body.endswith("\n"):
            body = body + "\n"
        # 2) Preserve a trailing newline after the closing fence only if one existed in the original content
        needs_newline = not (end < len(content) and content[end : end + 1] == "\n")
        trailing = "\n" if needs_newline else ""
        output_parts.append(f"```mermaid\n{body}```{trailing}")
        records.append(record)
        # Advance cursor to the end of this fence
        last_index = end
    # Append remaining tail after the last fence
    output_parts.append(content[last_index:])
    new_content = "".join(output_parts)

    # Attach fence fixes to each record for visibility (non-destructive)
    if fence_fixes and records:
        # store on first record as a container-wide note
        records[0].fixes = list(dict.fromkeys(records[0].fixes + fence_fixes))

    # Log sanitized content for debugging fence issues
    # logger.info(f"SANITIZER_OUTPUT: {repr(new_content)}")

    summary = SanitizationSummary(
        total=len(records),
        valid=sum(1 for r in records if r.status == "valid"),
        fixed=sum(1 for r in records if r.status == "fixed"),
        failed=sum(1 for r in records if r.status == "failed"),
        records=records,
    )
    return new_content, summary


__all__ = [
    "SanitizerConfig",
    "DiagramRecord",
    "SanitizationSummary",
    "extract_mermaid_diagrams",
    "sanitize_mermaid_diagram",
    "sanitize_content",
]
