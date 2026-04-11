"""Deep research service — multi-step research against generated wikis."""

from __future__ import annotations

import json
import logging
import os
from collections import defaultdict
from collections.abc import AsyncGenerator

from app.config import Settings
from app.core.code_graph.graph_query_service import GraphQueryService
from app.core.deep_research.research_engine import DeepResearchEngine, ResearchConfig
from app.models.api import (
    CallStack,
    CallStackStep,
    CodeMapData,
    CodeMapSection,
    CodeMapSymbol,
    ResearchRequest,
    ResearchResponse,
    SourceReference,
)
from app.services.toolkit_bridge import ComponentCache, EngineComponents, build_engine_components
from app.storage.base import ArtifactStorage

logger = logging.getLogger(__name__)

# Relationship types worth showing in the tree (skip structural noise)
_INTERESTING_REL_TYPES = frozenset({
    "calls", "imports", "inherits", "implements", "uses", "instantiates",
    "references", "composes", "extends", "overrides",
})

# Symbol types to exclude (file-level containers add no tree value)
_EXCLUDE_SYMBOL_TYPES = frozenset({"file", "module", "package", "directory"})

_MAX_NEIGHBORS_PER_SEED = 15
_MAX_SECTIONS = 8
_MAX_SYMBOLS_PER_SECTION = 8

# Relationship types for call-tree expansion (tight: only actual call flow)
_CALL_FLOW_REL_TYPES = frozenset({
    "calls", "imports", "inherits", "implements", "extends", "overrides",
})

# Generic symbol names to skip during graph expansion (too common, add noise)
_GENERIC_EXPANSION_NAMES = frozenset({
    "get", "set", "put", "post", "delete", "head", "patch", "run", "call",
    "init", "new", "main", "test", "data", "self", "this", "args", "kwargs",
    "name", "type", "value", "key", "item", "list", "map", "id", "url",
    "start", "stop", "open", "close", "read", "write", "send", "split",
    "true", "false", "none", "null", "error", "result", "response", "request",
})


# ---------------------------------------------------------------------------
# Step 2: Build call tree from ask sources (replaces FTS5-only seeding)
# ---------------------------------------------------------------------------


def _build_call_tree_from_sources(
    sources: list[SourceReference],
    gqs: GraphQueryService,
) -> CodeMapData | None:
    """Build a call tree seeded from ask-engine source references.

    Unlike the old FTS5-only approach, seeds come from the LLM-driven ask
    pipeline which finds semantically relevant symbols — not just keyword
    matches.  Each seed is resolved to a graph node, then expanded via
    relationship traversal to build a complete call-flow picture.
    """
    if not sources:
        return None

    # file_path → {node_id → {name, type, line_start, relationships[]}}
    file_symbols: dict[str, dict[str, dict]] = defaultdict(dict)

    def _rel_path(data: dict, fallback: str = "") -> str:
        return data.get("rel_path", "") or os.path.basename(
            data.get("file_path", fallback)
        )

    def _ensure_sym(
        node_id: str, name: str, sym_type: str, fp: str, line: int | None = None,
    ) -> None:
        if not name or not fp:
            return
        if node_id not in file_symbols[fp]:
            file_symbols[fp][node_id] = {
                "name": name,
                "type": sym_type,
                "line_start": line,
                "relationships": [],
            }

    # --- Resolve ask sources to graph seeds ---
    seed_node_ids: list[str] = []
    for src in sources:
        node_id: str | None = None
        # Try resolving by symbol name first (most precise)
        if src.symbol:
            node_id = gqs.resolve_symbol(src.symbol, file_path=src.file_path or "")
        # Fall back to searching by file path for any symbol in that file
        if not node_id and src.file_path:
            file_hits = gqs.search(
                os.path.basename(src.file_path),
                k=5,
                exclude_types=_EXCLUDE_SYMBOL_TYPES,
            )
            for hit in file_hits:
                hit_fp = hit.rel_path or hit.file_path
                if hit_fp and src.file_path in hit_fp:
                    node_id = hit.node_id
                    break
        if node_id and node_id not in seed_node_ids:
            seed_node_ids.append(node_id)

    if not seed_node_ids:
        return None

    # --- Expand each seed via graph traversal (1-hop, call-flow edges only) ---
    for node_id in seed_node_ids:
        data = gqs.graph.nodes.get(node_id, {})
        name = data.get("symbol_name", "") or data.get("name", "")
        sym_type = (data.get("symbol_type") or "").lower()
        if sym_type in _EXCLUDE_SYMBOL_TYPES:
            continue
        fp = _rel_path(data)
        line = data.get("line_start")
        _ensure_sym(node_id, name, sym_type, fp, line)

        rels = gqs.get_relationships(
            node_id, direction="both", max_depth=1,
            max_results=_MAX_NEIGHBORS_PER_SEED,
        )
        for rel in rels:
            rel_label = rel.relationship_type.lower()
            # Only expand via call-flow edges (not references/uses which are noisy)
            if rel_label not in _CALL_FLOW_REL_TYPES:
                continue

            src_raw = gqs.resolve_symbol(rel.source_name)
            tgt_raw = gqs.resolve_symbol(rel.target_name)
            if not src_raw or not tgt_raw:
                continue

            # Skip file/module/package containers and generic symbols
            src_data = gqs.graph.nodes.get(src_raw, {})
            tgt_data = gqs.graph.nodes.get(tgt_raw, {})
            src_type = (src_data.get("symbol_type") or "").lower()
            tgt_type = (tgt_data.get("symbol_type") or "").lower()
            if src_type in _EXCLUDE_SYMBOL_TYPES or tgt_type in _EXCLUDE_SYMBOL_TYPES:
                continue
            # Don't expand into generic/short-named neighbors (they add noise)
            tgt_name_simple = rel.target_name.rsplit(".", 1)[-1].rsplit("::", 1)[-1].lower()
            if len(tgt_name_simple) < 4 or tgt_name_simple in _GENERIC_EXPANSION_NAMES:
                # Still record the relationship label, just don't add the target as a node
                if src_raw in file_symbols.get(_rel_path(src_data), {}):
                    rel_str = f"{rel_label}: {rel.target_name}"
                    rels_list = file_symbols[_rel_path(src_data)][src_raw]["relationships"]
                    if rel_str not in rels_list:
                        rels_list.append(rel_str)
                continue

            src_fp = _rel_path(src_data)
            tgt_fp = _rel_path(tgt_data)

            if src_fp:
                _ensure_sym(
                    src_raw, rel.source_name, rel.source_type,
                    src_fp, src_data.get("line_start"),
                )
            if tgt_fp:
                _ensure_sym(
                    tgt_raw, rel.target_name, rel.target_type,
                    tgt_fp, tgt_data.get("line_start"),
                )

            # Record directional relationship on the source symbol
            if src_raw in file_symbols.get(src_fp, {}):
                rel_str = f"{rel_label}: {rel.target_name}"
                rels_list = file_symbols[src_fp][src_raw]["relationships"]
                if rel_str not in rels_list:
                    rels_list.append(rel_str)

    if not file_symbols:
        return None

    # --- Order files by graph connectivity (files sharing edges cluster together) ---
    # Build a file adjacency graph: files that have symbols calling each other stay close
    file_edges: dict[str, set[str]] = defaultdict(set)
    all_node_to_file: dict[str, str] = {}
    for fp, syms in file_symbols.items():
        for nid in syms:
            all_node_to_file[nid] = fp

    for fp, syms in file_symbols.items():
        for nid, info in syms.items():
            for rel_str in info["relationships"]:
                # rel_str is like "calls: target_name"
                parts = rel_str.split(": ", 1)
                if len(parts) == 2:
                    tgt_name = parts[1]
                    tgt_id = gqs.resolve_symbol(tgt_name) if gqs else None
                    if tgt_id and tgt_id in all_node_to_file:
                        other_fp = all_node_to_file[tgt_id]
                        if other_fp != fp:
                            file_edges[fp].add(other_fp)
                            file_edges[other_fp].add(fp)

    # BFS to group connected files into clusters
    remaining = set(file_symbols.keys())
    clusters: list[list[str]] = []
    while remaining:
        start = next(iter(remaining))
        cluster: list[str] = []
        queue = [start]
        while queue:
            f = queue.pop(0)
            if f not in remaining:
                continue
            remaining.discard(f)
            cluster.append(f)
            for neighbor in file_edges.get(f, []):
                if neighbor in remaining:
                    queue.append(neighbor)
        clusters.append(cluster)

    # Flatten clusters, largest first; within each cluster sort by symbol count
    ordered_files: list[str] = []
    for cluster in sorted(clusters, key=len, reverse=True):
        cluster.sort(key=lambda fp: len(file_symbols[fp]), reverse=True)
        ordered_files.extend(cluster)

    sections: list[CodeMapSection] = []
    for idx, fp in enumerate(ordered_files[:_MAX_SECTIONS]):
        syms = file_symbols[fp]
        section_symbols: list[CodeMapSymbol] = []
        for sym_idx, (node_id, info) in enumerate(
            sorted(syms.items(), key=lambda kv: (kv[1].get("line_start") or 9999))
        ):
            if sym_idx >= _MAX_SYMBOLS_PER_SECTION:
                break
            section_symbols.append(CodeMapSymbol(
                id=f"s{idx}_sym{sym_idx}",
                name=info["name"],
                symbol_type=info["type"] or "symbol",
                file_path=fp,
                line_start=info.get("line_start"),
                relationships=info["relationships"][:5],
            ))
        if section_symbols:
            sections.append(CodeMapSection(
                id=f"section_{idx}",
                title=os.path.basename(fp),
                file_path=fp,
                symbols=section_symbols,
            ))

    return CodeMapData(sections=sections) if sections else None


# ---------------------------------------------------------------------------
# Step 3: Explain every node in the tree with a single batch LLM call
# ---------------------------------------------------------------------------


async def _explain_tree_nodes(
    code_map: CodeMapData,
    question: str,
    llm: object,
) -> CodeMapData:
    """Annotate each symbol with a 1-sentence description using a low-tier LLM.

    Sends a single batch prompt listing all symbols; parses the JSON response
    back onto each ``CodeMapSymbol.description``.  Also sets
    ``CodeMapData.summary`` to a short call-flow overview.
    """
    from langchain_core.messages import HumanMessage

    # Collect all symbols for the prompt
    symbol_entries: list[dict[str, str]] = []
    for sec in code_map.sections:
        for sym in sec.symbols:
            rels_str = ", ".join(sym.relationships[:3]) if sym.relationships else "none"
            symbol_entries.append({
                "id": sym.id,
                "name": sym.name,
                "type": sym.symbol_type,
                "file": sym.file_path,
                "relationships": rels_str,
            })

    if not symbol_entries:
        return code_map

    symbols_text = "\n".join(
        f"- {e['id']}: {e['name']} ({e['type']}) in {e['file']}  rels: {e['relationships']}"
        for e in symbol_entries
    )

    # Build id→{name,file} map for post-processing
    id_to_info: dict[str, dict[str, str]] = {}
    for e in symbol_entries:
        id_to_info[e["id"]] = {"name": e["name"], "file": e["file"]}

    prompt = (
        "You are a code analysis assistant.  Given the user's question and a list of "
        "code symbols with their relationships, produce a JSON object with:\n\n"
        "1. \"descriptions\": mapping each symbol id to a single sentence (max 15 words) "
        "explaining that symbol's role.\n\n"
        "2. \"summary\": 1-2 sentence overview.\n\n"
        "3. \"call_stacks\": an array of independent call-flow chains. Each chain is:\n"
        "   {\"title\": \"short flow name\", \"steps\": [{\"symbol\": \"<actual function/class name>\", "
        "\"file_path\": \"<file path>\", \"description\": \"<what happens at this step>\"}]}\n"
        "   IMPORTANT: Use the actual symbol NAME (e.g. \"useAuth\", \"MCPAuthMiddleware\"), "
        "NOT the id (e.g. NOT \"s0_sym0\"). "
        "List steps in execution order (entry point first). "
        "Create separate chains for independent flows.\n\n"
        "Return ONLY valid JSON, no markdown fences.\n\n"
        f"Question: {question}\n\n"
        f"Symbols:\n{symbols_text}\n\n"
        "JSON:"
    )

    try:
        response = await llm.ainvoke([HumanMessage(content=prompt)])
        raw = response.content if hasattr(response, "content") else str(response)
        # Strip markdown fences if the model wraps them anyway
        raw = raw.strip()
        if raw.startswith("```"):
            raw = raw.split("\n", 1)[1] if "\n" in raw else raw[3:]
            if raw.endswith("```"):
                raw = raw[:-3]
            raw = raw.strip()

        data = json.loads(raw)
        descriptions: dict[str, str] = data.get("descriptions", {})
        summary: str = data.get("summary", "")

        # Apply descriptions back to the tree
        for sec in code_map.sections:
            for sym in sec.symbols:
                if sym.id in descriptions:
                    sym.description = descriptions[sym.id]

        if summary:
            code_map.summary = summary

        # Build call stacks — fix any IDs the LLM used instead of names
        raw_stacks = data.get("call_stacks", [])
        for stack_data in raw_stacks:
            if not isinstance(stack_data, dict):
                continue
            title = stack_data.get("title", "")
            raw_steps = stack_data.get("steps", [])
            steps: list[CallStackStep] = []
            for s in raw_steps:
                if not isinstance(s, dict) or not s.get("symbol"):
                    continue
                sym = s["symbol"]
                fp = s.get("file_path", "")
                # If LLM used the id (e.g. "s0_sym1"), replace with actual name
                if sym in id_to_info:
                    info = id_to_info[sym]
                    sym = info["name"]
                    fp = fp or info["file"]
                steps.append(CallStackStep(
                    symbol=sym,
                    file_path=fp,
                    description=s.get("description", ""),
                ))
            if title and steps:
                code_map.call_stacks.append(CallStack(title=title, steps=steps))

    except Exception:
        logger.exception("Failed to generate code-map node explanations")

    return code_map


class ResearchService:
    """Handles deep research queries using multi-step agents."""

    def __init__(self, settings: Settings, storage: ArtifactStorage) -> None:
        self.settings = settings
        self.storage = storage
        self._cache = ComponentCache(
            max_size=settings.ask_cache_max_wikis,
            ttl_seconds=settings.ask_cache_ttl_seconds,
        )

    def evict_cache(self, wiki_id: str) -> bool:
        """Remove cached engine components for a wiki."""
        return self._cache.evict(wiki_id)

    async def _get_components(self, wiki_id: str) -> EngineComponents:
        """Load or return cached engine components for a wiki."""
        return await self._cache.get_or_build(
            wiki_id,
            factory=lambda: build_engine_components(
                wiki_id,
                self.storage,
                self.settings,
                tier="high",
            ),
        )

    async def _get_multi_wiki_components(
        self,
        project_id: str,
        user_id: str | None,
    ) -> EngineComponents:
        """Load and compose EngineComponents for all wikis in a project."""
        from app.db import get_session_factory
        from app.services.multi_wiki_components import build_multi_wiki_components

        session_factory = get_session_factory()
        return await build_multi_wiki_components(
            project_id=project_id,
            user_id=user_id,
            get_components_fn=self._get_components,
            session_factory=session_factory,
        )

    async def research_stream(
        self,
        request: ResearchRequest,
        user_id: str | None = None,
    ) -> AsyncGenerator[dict, None]:
        """Stream research events (research_start, thinking_step, research_complete)."""
        if request.project_id:
            components = await self._get_multi_wiki_components(request.project_id, user_id)
        else:
            components = await self._get_components(request.wiki_id)
        engine = DeepResearchEngine(
            retriever_stack=components.retriever_stack,
            graph_manager=components.graph_manager,
            code_graph=components.code_graph,
            repo_analysis=components.repo_analysis,
            llm_client=components.llm,
            config=ResearchConfig(
                research_type=request.research_type,
                similarity_threshold=self.settings.research_similarity_threshold,
            ),
            repo_path=components.repo_path,
        )

        chat_history = [{"role": m.role, "content": m.content} for m in request.chat_history] or None
        async for event in engine.research(question=request.question, chat_history=chat_history):
            yield event

    async def research_sync(
        self,
        request: ResearchRequest,
        user_id: str | None = None,
    ) -> ResearchResponse:
        """Non-streaming: collect final answer from event stream."""
        final_answer = ""
        sources: list[SourceReference] = []
        steps: list[str] = []
        tool_events: list[dict] = []

        async for event in self.research_stream(request, user_id=user_id):
            event_type = event.get("event_type", "")
            if event_type == "thinking_step":
                step_data = event.get("data", {})
                step_desc = step_data.get("tool", step_data.get("description", ""))
                if step_desc:
                    steps.append(step_desc)
                # Collect raw tool event data (kept for compat; no longer used for code_map)
                tool_events.append(step_data)
            # Support both legacy (research_complete/research_error) and MCP (task_complete/task_failed) events
            elif event_type in ("research_complete", "task_complete"):
                data = event.get("data", {})
                final_answer = data.get("report", "") or data.get("answer", "")
                raw_sources = data.get("sources", [])
                sources = [
                    SourceReference(
                        file_path=s.get("file_path") or s.get("source", ""),
                        line_start=s.get("line_start"),
                        line_end=s.get("line_end"),
                        snippet=s.get("snippet"),
                        symbol=s.get("symbol"),
                        symbol_type=s.get("symbol_type") or s.get("type"),
                        relevance_score=s.get("relevance_score"),
                    )
                    for s in raw_sources
                    if isinstance(s, dict)
                ]
            elif event_type in ("research_error", "task_failed"):
                raise RuntimeError(event.get("data", {}).get("error", "Research failed"))

        return ResearchResponse(
            answer=final_answer,
            sources=sources,
            research_steps=steps,
            code_map=None,
        )

    async def codemap_stream(
        self,
        request: ResearchRequest,
        user_id: str | None = None,
    ) -> AsyncGenerator[dict, None]:
        """Stream code-map pipeline events (tool calls from ask, then final result).

        Pipeline:
        1. Run the ask engine — forward its thinking_step events so the UI shows
           tool calls (search_codebase, etc.) in real time.
        2. Build a call tree from the ask sources (graph traversal, no LLM).
        3. Explain every node with a single batch LLM call.
        4. Re-answer the question grounded in the enriched tree.
        5. Emit research_complete with answer + code_map.
        """
        from langchain_core.messages import HumanMessage

        from app.core.ask_engine import AskConfig, AskEngine
        from app.services.llm_factory import create_llm

        if request.project_id:
            components = await self._get_multi_wiki_components(request.project_id, user_id)
        else:
            components = await self._get_components(request.wiki_id)

        # --- Step 1: Run ask engine, forwarding tool-call events to the UI ---
        ask_answer = ""
        ask_sources: list[SourceReference] = []
        import re
        _SYM_LINE_RE = re.compile(
            r"`(\w+)`\s+\((\w+)\)\s+(?:in|—)\s+([\w/.]+)"
        )

        # Collect ALL symbols from tool results; we'll filter after we have the answer
        all_tool_symbols: list[SourceReference] = []

        try:
            ask_llm = create_llm(self.settings, tier="low", skip_retry=True)
            engine = AskEngine(
                retriever_stack=components.retriever_stack,
                graph_manager=components.graph_manager,
                code_graph=components.code_graph,
                repo_analysis=components.repo_analysis,
                llm_client=ask_llm,
                config=AskConfig(
                    similarity_threshold=self.settings.ask_similarity_threshold,
                ),
            )
            chat_history = (
                [{"role": m.role, "content": m.content} for m in request.chat_history]
                or None
            )
            seen_symbols: set[str] = set()
            async for event in engine.ask(
                question=request.question, chat_history=chat_history,
            ):
                event_type = event.get("event_type", "")

                # Forward tool-call / tool-result events so the right panel shows activity
                if event_type == "thinking_step":
                    yield event
                    # Collect symbols from tool_result events
                    step_data = event.get("data", {})
                    step_type = step_data.get("stepType") or step_data.get("step_type", "")
                    if step_type == "tool_result":
                        output = step_data.get("output", "")
                        tool = step_data.get("tool", "")
                        if tool in ("search_symbols", "get_code", "query_graph", "search_docs"):
                            for m in _SYM_LINE_RE.finditer(output):
                                sym_name, sym_type, file_path = m.groups()
                                key = f"{sym_name}:{file_path}"
                                if key not in seen_symbols:
                                    seen_symbols.add(key)
                                    all_tool_symbols.append(SourceReference(
                                        file_path=file_path,
                                        symbol=sym_name,
                                        symbol_type=sym_type,
                                    ))

                if event_type in ("ask_complete", "task_complete"):
                    data = event.get("data", {})
                    ask_answer = data.get("answer", "")
                elif event_type in ("ask_error", "task_failed"):
                    logger.warning("Ask engine failed in codemap: %s", event)
        except Exception:
            logger.exception("Ask engine failed in codemap pipeline")

        # Filter: only keep symbols whose name or file appears in the answer.
        # Skip short/generic names that match too broadly (e.g. "get", "post", "set").
        _GENERIC_NAMES = frozenset({
            "get", "set", "put", "post", "delete", "head", "patch", "run", "call",
            "init", "new", "main", "test", "data", "self", "this", "args", "kwargs",
            "name", "type", "value", "key", "item", "list", "map", "id", "url",
            "start", "stop", "open", "close", "read", "write", "send", "split",
        })
        answer_lower = ask_answer.lower()
        for src in all_tool_symbols:
            sym_name = src.symbol or ""
            file_name = os.path.basename(src.file_path) if src.file_path else ""
            sym_lower = sym_name.lower()
            # Skip generic/short names — they match everywhere
            if sym_lower in _GENERIC_NAMES or len(sym_name) < 4:
                continue
            # Keep if the symbol name or filename is referenced in the answer
            file_stem = os.path.splitext(file_name)[0].lower() if file_name else ""
            if (sym_name and sym_lower in answer_lower) or \
               (file_stem and len(file_stem) > 3 and file_stem in answer_lower):
                ask_sources.append(src)

        logger.info(
            "Codemap: %d/%d symbols referenced in answer",
            len(ask_sources), len(all_tool_symbols),
        )

        from datetime import datetime, timezone

        def _ts() -> str:
            return datetime.now(timezone.utc).isoformat()

        # --- Step 2: Build call tree from ask sources ---
        t0 = _ts()
        yield {"event_type": "thinking_step", "data": {
            "stepType": "tool_call", "tool": "build_call_tree",
            "input": f"{len(ask_sources)} source references",
            "timestamp": t0,
        }}

        code_map: CodeMapData | None = None
        fts_index = components.graph_manager.fts_index if components.graph_manager else None
        gqs = GraphQueryService(components.code_graph, fts_index)

        try:
            code_map = _build_call_tree_from_sources(ask_sources, gqs)
        except Exception:
            logger.exception("Failed to build call tree from ask sources")

        node_count = sum(len(s.symbols) for s in code_map.sections) if code_map else 0
        yield {"event_type": "thinking_step", "data": {
            "stepType": "tool_result", "tool": "build_call_tree",
            "output": f"{node_count} nodes across {len(code_map.sections) if code_map else 0} files",
            "timestamp": _ts(),
        }}

        # --- Step 3: Explain every node with a batch LLM call ---
        if code_map and code_map.sections:
            t0 = _ts()
            yield {"event_type": "thinking_step", "data": {
                "stepType": "tool_call", "tool": "explain_nodes",
                "input": f"Explaining {node_count} symbols",
                "timestamp": t0,
            }}
            try:
                explain_llm = create_llm(self.settings, tier="low", skip_retry=True)
                code_map = await _explain_tree_nodes(
                    code_map, request.question, explain_llm,
                )
            except Exception:
                logger.exception("Node explanation step failed")
            yield {"event_type": "thinking_step", "data": {
                "stepType": "tool_result", "tool": "explain_nodes",
                "output": code_map.summary[:200] if code_map and code_map.summary else "done",
                "timestamp": _ts(),
            }}

        # --- Emit code map early so the UI renders call stacks immediately ---
        if code_map:
            yield {
                "event_type": "code_map_ready",
                "data": {"code_map": code_map.model_dump()},
            }

        # --- Step 4: Re-answer grounded in the enriched tree ---
        answer = ask_answer
        sources = ask_sources
        if code_map and code_map.sections:
            t0 = _ts()
            yield {"event_type": "thinking_step", "data": {
                "stepType": "tool_call", "tool": "refine_answer",
                "input": "Re-answering with enriched call-tree context",
                "timestamp": t0,
            }}
            try:
                context_parts: list[str] = []
                for sec in code_map.sections:
                    sym_lines = []
                    for sym in sec.symbols:
                        line_ref = f":{sym.line_start}" if sym.line_start else ""
                        desc = f" — {sym.description}" if sym.description else ""
                        rels = ", ".join(sym.relationships[:3]) if sym.relationships else ""
                        sym_lines.append(
                            f"  - {sym.name} ({sym.symbol_type}){line_ref}{desc}"
                            + (f"  [{rels}]" if rels else "")
                        )
                    block = f"File: {sec.file_path}\n" + "\n".join(sym_lines)
                    context_parts.append(block)

                tree_context = "\n\n---\n\n".join(context_parts)
                tree_summary = code_map.summary or ""

                prompt = (
                    "You are a code analysis assistant. You have the initial answer from "
                    "a code search agent and a detailed call-tree with per-node explanations.\n\n"
                    "Re-write the answer to accurately explain the call flow, referencing "
                    "specific files, functions, and their relationships. Be precise about "
                    "which function calls which, and how data flows through the system.\n\n"
                    "IMPORTANT: Group the explanation by independent call flows if there are "
                    "multiple distinct auth/call paths. For example, frontend auth, backend API "
                    "auth, and MCP server auth are separate flows — present them as separate sections "
                    "ordered from user-facing (frontend) to internal (backend services).\n\n"
                    f"Question: {request.question}\n\n"
                    f"Initial answer:\n{ask_answer}\n\n"
                    + (f"Call-flow summary:\n{tree_summary}\n\n" if tree_summary else "")
                    + f"Detailed call tree:\n{tree_context}\n\n"
                    "Refined answer:"
                )

                response = await components.llm.ainvoke([HumanMessage(content=prompt)])
                answer = response.content if hasattr(response, "content") else str(response)

                # Enrich sources from the tree
                tree_files = {sec.file_path for sec in code_map.sections}
                existing_files = {s.file_path for s in sources}
                for fp in tree_files - existing_files:
                    sources.append(SourceReference(file_path=fp))

            except Exception:
                logger.exception("Refined answer generation failed — using ask answer")

            yield {"event_type": "thinking_step", "data": {
                "stepType": "tool_result", "tool": "refine_answer",
                "output": "done",
                "timestamp": _ts(),
            }}

        # --- Final: emit research_complete with code_map ---
        yield {
            "event_type": "research_complete",
            "data": {
                "report": answer,
                "sources": [s.model_dump() for s in sources],
                "code_map": code_map.model_dump() if code_map else None,
            },
        }

    async def codemap_sync(
        self,
        request: ResearchRequest,
        user_id: str | None = None,
    ) -> ResearchResponse:
        """Non-streaming wrapper for codemap_stream."""
        final_answer = ""
        sources: list[SourceReference] = []
        steps: list[str] = []
        code_map: CodeMapData | None = None

        async for event in self.codemap_stream(request, user_id=user_id):
            event_type = event.get("event_type", "")
            if event_type == "thinking_step":
                tool = event.get("data", {}).get("tool", "")
                if tool:
                    steps.append(tool)
            elif event_type == "research_complete":
                data = event.get("data", {})
                final_answer = data.get("report", "")
                raw_sources = data.get("sources", [])
                sources = [
                    SourceReference(**s) for s in raw_sources if isinstance(s, dict)
                ]
                raw_map = data.get("code_map")
                if raw_map:
                    code_map = CodeMapData(**raw_map)

        return ResearchResponse(
            answer=final_answer,
            sources=sources,
            research_steps=steps,
            code_map=code_map,
        )
