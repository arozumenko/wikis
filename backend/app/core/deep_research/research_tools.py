"""
Research Tools - Custom tools for Deep Research and Ask agents

These tools wrap our existing infrastructure with bounded, high-quality results,
following a progressive disclosure pattern: cheap discovery first, then targeted
detail fetches.

Tools included:
- search_symbols: Discover symbols by name/keywords (compact summaries, no source)
- search_docs: Search documentation chunks in the vector store
- get_code: Fetch full source for a specific symbol
- query_graph: Query symbols by path prefix and optional text filter
- get_relationships_tool: Inspect code graph relationships of a symbol
- think: Strategic reflection and planning
- read_source_file: Read raw source files from the cloned repository
- list_repo_files: Browse the repository directory structure
"""

import logging
import os
from collections.abc import Callable
from datetime import datetime
from typing import Any

from langchain_core.tools import tool

from ..code_graph.graph_query_service import GraphQueryService
from ..code_graph.storage_query_service import StorageQueryService
from ..constants import ARCHITECTURAL_SYMBOLS, DOC_SYMBOL_TYPES

logger = logging.getLogger(__name__)

# Container types that can own child methods/functions via 'defines' edges.
# Used by orphan FTS fallback to discover string-based references.
CONTAINER_SYMBOL_TYPES = frozenset({"class", "interface", "struct", "enum", "trait"})

# Try to import EmbeddingsFilter for semantic reranking
try:
    from langchain_classic.retrievers.document_compressors.embeddings_filter import EmbeddingsFilter

    EMBEDDINGS_FILTER_AVAILABLE = True
except ImportError:
    EmbeddingsFilter = None
    EMBEDDINGS_FILTER_AVAILABLE = False
    logger.warning("langchain_classic EmbeddingsFilter not available - search results won't be reranked")



def create_codebase_tools(
    retriever_stack: Any,  # UnifiedRetriever
    graph_manager: Any,  # GraphManager
    code_graph: Any,  # NetworkX graph (may be None when `storage` is provided)
    repo_analysis: dict | None = None,
    event_callback: Callable | None = None,
    similarity_threshold: float = 0.75,
    graph_text_index: Any = None,  # GraphTextIndex / StorageTextIndex (FTS5)
    unified_db_path: str | None = None,  # Path to .wiki.db
    repo_path: str | None = None,  # Path to cloned repo for direct file access
    storage: Any = None,  # WikiStorageProtocol (UnifiedWikiDB / PostgresWikiStorage)
) -> list:
    """
    Create custom tools for deep research and Ask agents.

    Returns the progressive disclosure tool set: cheap discovery
    (search_symbols, search_docs, query_graph, get_relationships_tool)
    plus the expensive detail fetch (get_code).

    When ``storage`` is provided (and ``code_graph`` is either ``None``
    or an empty placeholder), tools bind to ``StorageQueryService`` and
    read node metadata directly from the unified DB.  Otherwise they
    use ``GraphQueryService`` over the in-memory NX graph.

    Args:
        retriever_stack: UnifiedRetriever instance for codebase search
        graph_manager: GraphManager instance for graph operations
        code_graph: The loaded code graph (NetworkX) — may be None when
            ``storage`` is supplied.
        repo_analysis: Pre-loaded repository analysis (optional)
        event_callback: Callback for emitting thinking events
        similarity_threshold: Minimum similarity for EmbeddingsFilter (0.0-1.0)
        unified_db_path: Path to unified .wiki.db (legacy; prefer ``storage``)
        repo_path: Path to cloned repo for direct file access
        storage: ``WikiStorageProtocol`` for storage-native symbol/edge lookups

    Returns:
        List of LangChain tools for DeepAgents
    """
    emit = event_callback or (lambda x: None)

    # ------------------------------------------------------------------
    # Decide which query service to use.
    # Priority: explicit ``storage`` wins when the graph is unusable
    # (None or empty), even if a retriever-attached DB is present.
    # ------------------------------------------------------------------
    def _graph_has_nodes(g: Any) -> bool:
        if g is None:
            return False
        try:
            return g.number_of_nodes() > 0
        except Exception:
            return False

    use_storage = storage is not None and not _graph_has_nodes(code_graph)

    if use_storage:
        # ``graph_text_index`` should already be a StorageTextIndex backed
        # by ``storage``.  Fall back to building one if the caller didn't
        # provide it.
        if graph_text_index is None:
            try:
                from ..storage.text_index import StorageTextIndex

                graph_text_index = StorageTextIndex(storage)
            except Exception as exc:  # pragma: no cover - defensive
                logger.debug("Failed to build StorageTextIndex: %s", exc)
        query_service: Any = StorageQueryService(
            storage=storage, text_index=graph_text_index
        )
        logger.info("[RESEARCH_TOOLS] Using StorageQueryService (storage-native)")
    else:
        query_service = (
            GraphQueryService(code_graph, fts_index=graph_text_index)
            if code_graph is not None
            else None
        )

    def _node_data(nid: str) -> dict[str, Any]:
        """Unified node metadata lookup — graph or storage."""
        if not nid:
            return {}
        if use_storage:
            return storage.get_node(nid) or {}
        if code_graph is not None and nid in code_graph:
            return dict(code_graph.nodes.get(nid, {}) or {})
        return {}

    def _defines_children(nid: str) -> list[dict[str, Any]]:
        """Return child nodes (methods/functions) defined by ``nid``."""
        if not nid:
            return []
        if use_storage:
            try:
                return storage.find_related_nodes(
                    nid,
                    rel_types=["defines"],
                    direction="out",
                    target_symbol_types=["method", "function"],
                    limit=64,
                )
            except Exception as exc:
                logger.debug("_defines_children storage lookup failed: %s", exc)
                return []
        # NX fallback preserved for wiki-gen callers.
        if code_graph is None or nid not in code_graph:
            return []
        out: list[dict[str, Any]] = []
        for succ in code_graph.successors(nid):
            edge_data = code_graph.get_edge_data(nid, succ)
            if edge_data is None:
                continue
            edges = (
                edge_data.values()
                if isinstance(edge_data, dict)
                and all(isinstance(v, dict) for v in edge_data.values())
                else [edge_data]
            )
            if not any(
                (e.get("relationship_type") or "").lower() == "defines"
                for e in edges
            ):
                continue
            data = code_graph.nodes.get(succ, {}) or {}
            if (data.get("symbol_type") or "").lower() not in ("method", "function"):
                continue
            out.append({"node_id": succ, **data})
        return out

    # Resolve unified_db_path: auto-detect when path not provided
    _udb_path = unified_db_path
    if _udb_path is None and not use_storage and code_graph is None:
        # Legacy: try to get path from retriever_stack (UnifiedRetriever stores it)
        _udb_path = getattr(getattr(retriever_stack, "db", None), "db_path", None)
        if _udb_path:
            _udb_path = str(_udb_path)
    _use_db_fallback = bool(_udb_path and not use_storage and code_graph is None)
    if _use_db_fallback:
        logger.info("[RESEARCH_TOOLS] Unified DB fallback enabled: %s", _udb_path)

    @tool(parse_docstring=True)
    def think(reflection: str) -> str:
        """Strategic reflection tool for analyzing findings and planning next steps.

        Use this to:
        - Analyze what you've learned so far
        - Identify gaps in your understanding
        - Plan your next research steps
        - Synthesize information from multiple sources
        - Decide if you have enough information to answer the question

        This makes your reasoning visible and helps organize research.

        Args:
            reflection: Your full analysis, observations, or reasoning
        """
        emit({"type": "thinking", "content": reflection, "timestamp": datetime.now().isoformat()})

        # Return full confirmation (the middleware handles this)
        return f"✓ Thinking recorded.\n\nReflection:\n{reflection}"

    # ================================================================
    # Progressive Disclosure Tools
    # ================================================================
    # Cheap discovery tools (search_symbols, get_relationships_tool, search_docs)
    # + expensive detail tool (get_code).

    @tool(parse_docstring=True)
    def search_symbols(query: str, k: int = 20, symbol_type: str = "", file_prefix: str = "") -> str:
        """Search for code symbols by name or keywords. Returns compact summaries (no source code).

        This is the primary discovery tool. Use it to find relevant classes, functions,
        and constants, then use get_code to read the ones you need.

        Args:
            query: Search query (symbol name, concept, or keywords)
            k: Maximum number of results (default 20)
            symbol_type: Optional filter by type: class, function, constant, interface, etc.
            file_prefix: Optional filter by file path prefix (e.g. 'src/auth')
        """
        emit({"type": "tool_start", "tool": "search_symbols", "input": query, "timestamp": datetime.now().isoformat()})

        try:
            results = []
            sym_types = frozenset({symbol_type.lower()}) if symbol_type else None
            path_prefix = file_prefix if file_prefix else None

            if query_service:
                svc_results = query_service.search(
                    query,
                    k=k,
                    symbol_types=sym_types,
                    exclude_types=DOC_SYMBOL_TYPES,
                    path_prefix=path_prefix,
                )
                for r in svc_results:
                    # Extract one-line doc (first sentence of docstring)
                    one_line = ""
                    if r.node_id:
                        doc = (_node_data(r.node_id).get("docstring") or "")
                        if doc:
                            first_line = doc.strip().split("\n")[0].strip()
                            if len(first_line) > 80:
                                first_line = first_line[:77] + "..."
                            one_line = first_line

                    results.append(
                        {
                            "name": r.symbol_name,
                            "kind": r.symbol_type,
                            "layer": r.layer,
                            "file": r.rel_path or r.file_path,
                            "refs": r.connections,
                            "doc": one_line,
                        }
                    )
            elif graph_text_index and graph_text_index.is_open:
                docs = graph_text_index.search_smart(query, k=k, intent="symbol", exclude_types=DOC_SYMBOL_TYPES)
                for doc in docs:
                    m = doc.metadata
                    results.append(
                        {
                            "name": m.get("symbol_name", ""),
                            "kind": m.get("symbol_type", ""),
                            "layer": m.get("layer", ""),
                            "file": m.get("rel_path", m.get("source", "")),
                            "refs": 0,
                            "doc": "",
                        }
                    )
            else:
                return "No search index available."

            emit(
                {
                    "type": "tool_end",
                    "tool": "search_symbols",
                    "result_count": len(results),
                    "timestamp": datetime.now().isoformat(),
                }
            )

            if not results:
                return f"No symbols found for: {query}"

            lines = [f"Found {len(results)} symbols for: {query}\n"]
            for i, r in enumerate(results):
                doc_str = f" — {r['doc']}" if r["doc"] else ""
                layer_str = f" [{r['layer']}]" if r["layer"] else ""
                lines.append(
                    f"{i + 1}. `{r['name']}` ({r['kind']}{layer_str}) in {r['file']} [{r['refs']} refs]{doc_str}"
                )
            return "\n".join(lines)

        except Exception as e:
            logger.error(f"search_symbols failed: {e}", exc_info=True)
            return f"Symbol search failed: {str(e)}"

    # ----------------------------------------------------------------
    # Orphan FTS fallback — shared by get_relationships_tool
    # ----------------------------------------------------------------
    def _orphan_fts_fallback(
        node_id: str,
        fts_index: Any,
        emit_fn: Callable,
    ) -> str | None:
        """Discover implicit string-based references for orphan container symbols.

        When a class/struct/trait/enum/interface has zero graph-edge
        relationships its method names may still be referenced as string
        literals in framework or RPC code (e.g. ``Event.on_startup``
        dispatched via ``emit("on_startup")``).

        Extracts child method/function names via ``defines`` edges and
        runs FTS5 text search for each name.  Returns a formatted result
        string if any hits are found, otherwise ``None``.
        """
        if fts_index is None:
            return None
        if not (hasattr(fts_index, "search") and hasattr(fts_index, "is_open")):
            return None
        if not fts_index.is_open:
            return None

        node_data = _node_data(node_id)
        if not node_data:
            return None
        sym_type = (node_data.get("symbol_type") or "").lower()
        if sym_type not in CONTAINER_SYMBOL_TYPES:
            return None

        sym_name = node_data.get("symbol_name", "") or node_data.get("name", "")

        # Collect child method/function names via 'defines' edges
        method_names: list[str] = []
        for child in _defines_children(node_id):
            name = child.get("symbol_name", "") or child.get("name", "")
            if name and not name.startswith("__"):
                method_names.append(name)

        if not method_names:
            return None

        logger.debug(
            f"[ORPHAN_FTS] {sym_type} '{sym_name}' has no graph rels, "
            f"searching FTS for child methods: {method_names[:8]}"
        )

        emit_fn(
            {
                "type": "tool_start",
                "tool": "orphan_fts_fallback",
                "input": f"{sym_name} methods: {method_names[:8]}",
                "timestamp": datetime.now().isoformat(),
            }
        )

        # FTS5 search for each method name — look for string-based references
        fts_filter_types = ARCHITECTURAL_SYMBOLS - frozenset({"macro"})
        hits: list[dict] = []
        seen_nodes: set = set()
        for method_name in method_names[:8]:  # Cap to avoid too many FTS queries
            try:
                docs = fts_index.search(
                    method_name,
                    k=3,
                    symbol_types=fts_filter_types,
                )
            except Exception:  # noqa: S112
                continue
            for doc in docs:
                nid = doc.metadata.get("node_id", "")
                if not nid or nid == node_id or nid in seen_nodes:
                    continue
                seen_nodes.add(nid)
                ref_name = doc.metadata.get("symbol_name", "") or doc.metadata.get("name", "")
                ref_type = doc.metadata.get("symbol_type", "")
                ref_file = doc.metadata.get("rel_path", doc.metadata.get("source", ""))
                hits.append(
                    {
                        "method": method_name,
                        "ref_name": ref_name,
                        "ref_type": ref_type,
                        "ref_file": ref_file,
                    }
                )

        emit_fn(
            {
                "type": "tool_end",
                "tool": "orphan_fts_fallback",
                "result_count": len(hits),
                "timestamp": datetime.now().isoformat(),
            }
        )

        if not hits:
            return None

        lines = [
            f"`{sym_name}` ({sym_type}) has no direct graph relationships.",
            f"However, its methods are referenced as strings in {len(hits)} symbols:\n",
        ]
        for h in hits:
            lines.append(
                f"  \u26a1 `{h['method']}` referenced in `{h['ref_name']}` ({h['ref_type']}) \u2014 {h['ref_file']}"
            )
        lines.append("\nTip: use get_code on these referencing symbols to see the full context.")
        return "\n".join(lines)

    @tool(parse_docstring=True)
    def get_relationships_tool(symbol_name: str, direction: str = "both", max_depth: int = 2) -> str:
        """Get relationships for a code symbol. Returns compact edge list (no source code).

        Use this after search_symbols to understand how a symbol connects to the codebase.

        Args:
            symbol_name: Name of the symbol to analyze (e.g. "AuthService", "login")
            direction: 'both', 'outgoing', or 'incoming'
            max_depth: How many hops to traverse (default 2)
        """
        emit(
            {
                "type": "tool_start",
                "tool": "get_relationships",
                "input": symbol_name,
                "timestamp": datetime.now().isoformat(),
            }
        )

        if not query_service:
            return "Code graph not available for relationship analysis"

        try:
            node_id, rels = query_service.resolve_and_traverse(
                symbol_name,
                direction=direction,
                max_depth=max_depth,
                max_results=50,
            )

            if not node_id:
                return f"Symbol '{symbol_name}' not found. Try search_symbols first."

            emit(
                {
                    "type": "tool_end",
                    "tool": "get_relationships",
                    "result_count": len(rels),
                    "timestamp": datetime.now().isoformat(),
                }
            )

            if not rels:
                # === Orphan FTS fallback ===
                # Container types (class, struct, trait, enum, interface) may
                # have no graph edges but their method names can appear as
                # string literals in framework/RPC code. Extract child method
                # names and search FTS5 for string-based references.
                orphan_hits = _orphan_fts_fallback(
                    node_id,
                    graph_text_index,
                    emit,
                )
                if orphan_hits:
                    return orphan_hits
                return f"`{symbol_name}` has no relationships within {max_depth} hops."

            lines = [f"Relationships for `{symbol_name}` (node: `{node_id}`):\n"]
            for r in rels:
                # Determine arrow direction relative to the root symbol:
                # If node_id is the source (or source_name matches), it's outgoing
                src_low = (r.source_name or "").lower()
                is_outgoing = (
                    r.source_name == node_id or src_low == symbol_name.lower() or symbol_name.lower() in src_low
                )
                if is_outgoing:
                    arrow, other, other_type = "→", r.target_name, r.target_type
                else:
                    arrow, other, other_type = "←", r.source_name, r.source_type
                hop_str = f" (hop {r.hop_distance})" if r.hop_distance > 1 else ""
                type_str = f" ({other_type})" if other_type else ""
                lines.append(f"  {arrow} `{other}`{type_str} [{r.relationship_type}]{hop_str}")

            # If ALL rels are structural 'defines' edges, the container
            # only has child-containment links — also run orphan FTS to
            # discover implicit string-based references.
            if all(r.relationship_type.lower() == "defines" for r in rels):
                orphan_hits = _orphan_fts_fallback(
                    node_id,
                    graph_text_index,
                    emit,
                )
                if orphan_hits:
                    lines.append("")
                    lines.append(orphan_hits)

            return "\n".join(lines)

        except Exception as e:
            logger.error(f"get_relationships failed: {e}", exc_info=True)
            return f"Relationship analysis failed: {str(e)}"

    @tool(parse_docstring=True)
    def get_code(symbol_name: str, max_lines: int = 200) -> str:
        """Retrieve full source code for a specific symbol. Use sparingly — read only what you need.

        Call search_symbols first to discover symbols, then get_code for the 3-5 most
        relevant ones.

        Args:
            symbol_name: Exact symbol name from search_symbols results
            max_lines: Maximum lines of source to return (default 200)
        """
        emit({"type": "tool_start", "tool": "get_code", "input": symbol_name, "timestamp": datetime.now().isoformat()})

        if query_service is None:
            return "Code graph not available"

        try:
            node_id = query_service.resolve_symbol(symbol_name)
            if not node_id:
                return f"Symbol '{symbol_name}' not found. Check the exact name from search_symbols."

            data = _node_data(node_id)
            if not data:
                return f"No data for node '{node_id}'"

            # Try multiple content sources (match FTS index build logic)
            symbol_obj = data.get("symbol")
            if symbol_obj and hasattr(symbol_obj, "source_text") and symbol_obj.source_text:
                content = symbol_obj.source_text
            else:
                content = data.get("content") or data.get("source_text") or data.get("docstring") or ""

            sym_name = data.get("symbol_name", "") or data.get("name", symbol_name)
            sym_type = (data.get("symbol_type") or "").lower()
            rel_path = data.get("rel_path", data.get("file_path", ""))
            start_line = data.get("start_line", "")
            end_line = data.get("end_line", "")

            # Fallback: try FTS5 index for stored content when graph node is empty
            if not content and graph_text_index is not None:
                try:
                    if hasattr(graph_text_index, "get_by_node_id"):
                        fts_row = graph_text_index.get_by_node_id(node_id)
                        if fts_row:
                            content = fts_row.get("content") or fts_row.get("docstring") or ""
                except Exception:  # noqa: S110
                    pass  # Best-effort fallback

            # Truncate to max_lines
            lines = content.split("\n")
            truncated = False
            if len(lines) > max_lines:
                lines = lines[:max_lines]
                truncated = True

            emit({"type": "tool_end", "tool": "get_code", "result_count": 1, "timestamp": datetime.now().isoformat()})

            if not content:
                return (
                    f"### `{sym_name}` ({sym_type}) — {rel_path}\n\n"
                    f"Source code not available for this symbol "
                    f"(graph was built without code bodies).\n"
                    f"Tip: use get_relationships to explore connections, "
                    f"or search_docs for documentation."
                )

            header = f"### `{sym_name}` ({sym_type}) — {rel_path}:{start_line}-{end_line}\n"
            code_block = "\n".join(lines)
            suffix = f"\n... (truncated at {max_lines} lines)" if truncated else ""

            return f"{header}\n```\n{code_block}\n```{suffix}"

        except Exception as e:
            logger.error(f"get_code failed: {e}", exc_info=True)
            return f"Code retrieval failed: {str(e)}"

    @tool(parse_docstring=True)
    def search_docs(query: str, k: int = 5) -> str:
        """Search documentation (README, guides, comments) — NOT code symbols.

        Use this when the question is about project documentation, configuration,
        or high-level architecture described in markdown/text files.

        Args:
            query: Search query for documentation content
            k: Maximum number of document chunks to return (default 5)
        """
        emit({"type": "tool_start", "tool": "search_docs", "input": query, "timestamp": datetime.now().isoformat()})

        try:
            docs = []
            if retriever_stack:
                try:
                    # Unified DB hybrid search already ranks results well
                    vs_docs = retriever_stack.search_repository(query=query, k=k, apply_expansion=False)
                    docs.extend(vs_docs)
                except Exception as e:
                    logger.warning(f"search_docs VS failed: {e}")

            emit(
                {
                    "type": "tool_end",
                    "tool": "search_docs",
                    "result_count": len(docs),
                    "timestamp": datetime.now().isoformat(),
                }
            )

            if not docs:
                return f"No documentation found for: {query}"

            sections = []
            for i, doc in enumerate(docs):
                source = doc.metadata.get("source", doc.metadata.get("rel_path", "unknown"))
                sections.append(f"### [{i + 1}] {source}\n\n{doc.page_content}")

            return f"## Documentation: {query}\n\n" + "\n\n---\n\n".join(sections)

        except Exception as e:
            logger.error(f"search_docs failed: {e}", exc_info=True)
            return f"Documentation search failed: {str(e)}"

    # ================================================================
    # SPEC-2: JQL Structured Query Tool
    # ================================================================

    @tool(parse_docstring=True)
    def query_graph(expression: str) -> str:
        """Execute a structured JQL query against the code graph.

        Use this for precise, multi-filter searches when you know what kind of
        symbol you're looking for. Combines type, layer, path, name, relationship,
        and connection count filters in a single expression.

        Syntax: field:value clauses separated by spaces (implicit AND).
        - type:class          — filter by symbol type (class, function, method, interface, etc.)
        - layer:core_type     — filter by architectural layer (core_type, public_api, entry_point, internal, infrastructure, constant)
        - file:src/auth/*     — filter by file path (glob patterns supported)
        - name:AuthService    — filter by symbol name (glob patterns supported)
        - text:authentication — full-text search across names, docstrings, code
        - related:"BaseClass" — filter to symbols connected to a specific symbol
        - dir:outgoing        — direction for related filter (outgoing, incoming, both)
        - has_rel:inheritance — filter to symbols with a specific relationship type.
          Common types: calls, inheritance, implementation, composition, aggregation,
          imports, exports, defines, contains, references, creates, instantiates,
          decorates, annotates, overrides, uses, uses_type, reads, writes,
          defines_body, specializes, captures, alias_of, assigns, returns, parameter.
          Aliases accepted: inherits→inheritance, extends→inheritance,
          implements→implementation, invokes→calls, composes→composition
        - connections:>10     — filter by connection count (>, <, >=, <=)
        - limit:30            — maximum number of results (default 20)

        Examples:
            type:class file:src/auth/*
            related:"AuthService" dir:outgoing has_rel:calls
            type:function connections:>5 limit:10
            layer:core_type text:handler

        Args:
            expression: JQL query expression with field:value clauses
        """
        emit(
            {"type": "tool_start", "tool": "query_graph", "input": expression, "timestamp": datetime.now().isoformat()}
        )

        try:
            if not query_service:
                return "Graph query service not available."

            results = query_service.query(expression)

            emit(
                {
                    "type": "tool_end",
                    "tool": "query_graph",
                    "result_count": len(results),
                    "timestamp": datetime.now().isoformat(),
                }
            )

            if not results:
                return f"No symbols matched query: {expression}"

            lines = [f"## Query Results: `{expression}` ({len(results)} matches)\n"]
            for r in results:
                location = r.rel_path or r.file_path or "?"
                layer_tag = f" [{r.layer}]" if r.layer else ""
                conn_tag = f" ({r.connections} connections)" if r.connections else ""
                lines.append(f"- **{r.symbol_name}** ({r.symbol_type}){layer_tag} — `{location}`{conn_tag}")

            return "\n".join(lines)

        except Exception as e:
            logger.error(f"query_graph failed: {e}", exc_info=True)
            return f"Query failed: {str(e)}"

    # ================================================================
    # read_source_file — direct file access to the cloned repository
    # ================================================================

    _repo_available = repo_path is not None and os.path.isdir(repo_path)

    @tool(parse_docstring=True)
    def read_source_file(file_path: str, offset: int = 0, limit: int = 200) -> str:
        """Read a source file directly from the repository.

        Use this to read files that weren't returned by search_symbols / search_docs,
        such as config files, Dockerfiles, Makefiles, scripts, or to see
        full file context around a search result snippet.

        Args:
            file_path: Relative path from repo root (e.g. 'src/auth/manager.py')
            offset: Starting line number, 0-based (default 0)
            limit: Maximum number of lines to return (default 200)
        """
        if not _repo_available:
            return "Error: Repository clone is not available on disk. Use search_symbols / get_code instead."

        # Resolve and validate the path stays within the repo
        try:
            full = os.path.realpath(os.path.join(repo_path, file_path))
            repo_real = os.path.realpath(repo_path)
            if not full.startswith(repo_real + os.sep) and full != repo_real:
                return f"Error: Path '{file_path}' is outside the repository."
        except (ValueError, OSError):
            return f"Error: Invalid path '{file_path}'."

        if not os.path.isfile(full):
            return f"Error: File not found: {file_path}"

        try:
            with open(full, errors="replace") as f:
                lines = f.readlines()
            total = len(lines)
            selected = lines[offset : offset + limit]
            numbered = [f"{offset + i + 1:>5} | {line.rstrip()}" for i, line in enumerate(selected)]
            header = f"# {file_path} ({total} lines total, showing {offset + 1}-{offset + len(selected)})"
            return header + "\n" + "\n".join(numbered)
        except Exception as e:
            return f"Error reading {file_path}: {e}"

    @tool(parse_docstring=True)
    def list_repo_files(directory: str = ".", pattern: str = "") -> str:
        """List files and directories in the repository.

        Use this to explore the repository structure before reading specific files.
        Returns up to 200 entries.

        Args:
            directory: Relative path from repo root (default: root directory)
            pattern: Optional glob pattern to filter results (e.g. '*.py', '*.yml')
        """
        if not _repo_available:
            return "Error: Repository clone is not available on disk. Use search_symbols / get_code instead."

        try:
            full = os.path.realpath(os.path.join(repo_path, directory))
            repo_real = os.path.realpath(repo_path)
            if not full.startswith(repo_real + os.sep) and full != repo_real:
                return f"Error: Path '{directory}' is outside the repository."
        except (ValueError, OSError):
            return f"Error: Invalid path '{directory}'."

        if not os.path.isdir(full):
            return f"Error: Directory not found: {directory}"

        try:
            import fnmatch

            entries = sorted(os.listdir(full))
            if pattern:
                entries = [e for e in entries if fnmatch.fnmatch(e, pattern)]

            lines = []
            for entry in entries[:200]:
                entry_path = os.path.join(full, entry)
                suffix = "/" if os.path.isdir(entry_path) else ""
                lines.append(f"  {entry}{suffix}")

            header = f"# {directory}/ ({len(entries)} entries"
            if len(entries) > 200:
                header += ", showing first 200"
            header += ")"
            return header + "\n" + "\n".join(lines)
        except Exception as e:
            return f"Error listing {directory}: {e}"

    # ================================================================
    # Tool Selection
    # ================================================================
    # File access tools — included only when the cloned repo is on disk
    _file_tools = [read_source_file, list_repo_files] if _repo_available else []
    if _repo_available:
        logger.info("[RESEARCH_TOOLS] read_source_file + list_repo_files enabled (repo_path=%s)", repo_path)

    return [
        search_symbols,
        get_relationships_tool,
        get_code,
        search_docs,
        query_graph,
        think,
    ] + _file_tools
