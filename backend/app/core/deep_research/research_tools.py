"""
Research Tools - Custom tools for DeepAgents

These tools wrap our existing infrastructure with bounded, high-quality results.
EmbeddingsFilter provides semantic reranking and threshold filtering to ensure
only relevant results are returned.

Tools included:
- search_codebase: Search repository using vector store + code graph hybrid
- get_symbol_relationships: Analyze code graph relationships
- think: Strategic reflection and planning
- read_source_file: Read raw source files from the cloned repository
- list_repo_files: Browse the repository directory structure
"""

import logging
import os
import re
from collections.abc import Callable
from datetime import datetime
from typing import Any

from langchain_core.documents import Document
from langchain_core.tools import tool

from ..code_graph.graph_query_service import GraphQueryService
from ..constants import ARCHITECTURAL_SYMBOLS, DOC_SYMBOL_TYPES
from .hybrid_fusion import (
    HYBRID_FUSION_ENABLED,
    fuse_search_results,
)

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


def _search_graph_by_text(code_graph: Any, query: str, k: int = 10) -> list[Document]:
    """Full-text search over code graph node names, content, and docstrings.

    This is the graph-as-first-class-citizen search: queries go directly to the
    graph instead of (or in addition to) the vector store.  Works even when
    SEPARATE_DOC_INDEX is enabled and the vector store only contains docs.

    Matching strategy (scored):
      1. Exact symbol name match (highest priority)
      2. Substring match in symbol name
      3. Keyword match in content/docstring
    """
    if code_graph is None or code_graph.number_of_nodes() == 0:
        return []

    # Tokenize query into searchable keywords
    query_lower = query.lower()
    keywords = [w for w in re.split(r"\W+", query_lower) if len(w) >= 2]
    if not keywords:
        return []

    scored: list[tuple] = []  # (score, node_id, node_data)

    for node_id, node_data in code_graph.nodes(data=True):
        symbol_name = (node_data.get("symbol_name", "") or node_data.get("name", "")).lower()
        symbol_type = (node_data.get("symbol_type") or "").lower()
        content = (node_data.get("content", "") or "").lower()
        docstring = (node_data.get("docstring", "") or "").lower()
        if not docstring:
            _sym_obj = node_data.get("symbol")
            if _sym_obj and hasattr(_sym_obj, "docstring") and _sym_obj.docstring:
                docstring = _sym_obj.docstring.lower()

        # Skip doc nodes — those come from the vector store
        if symbol_type in DOC_SYMBOL_TYPES:
            continue

        score = 0

        # 1. Exact symbol name match (case-insensitive)
        if query_lower == symbol_name:
            score += 100
        # 2. Query appears as substring in symbol name
        elif query_lower in symbol_name:
            score += 50

        # 3. Individual keyword matches in name
        name_kw_hits = sum(1 for kw in keywords if kw in symbol_name)
        score += name_kw_hits * 10

        # 4. Keyword matches in docstring (medium value)
        if docstring:
            doc_kw_hits = sum(1 for kw in keywords if kw in docstring)
            score += doc_kw_hits * 3

        # 5. Keyword matches in content (lower value — content is large)
        if content and score < 20:  # Only check content if no strong name match
            content_kw_hits = sum(1 for kw in keywords if kw in content[:2000])
            score += content_kw_hits * 1

        if score > 0:
            scored.append((score, node_id, node_data))

    # Sort by score descending, take top k
    scored.sort(key=lambda x: -x[0])

    documents = []
    for score, _node_id, node_data in scored[:k]:
        content = node_data.get("content", "") or node_data.get("docstring", "") or ""
        if not content:
            # Comprehensive parser nodes store source/docstring inside Symbol object
            _sym_obj = node_data.get("symbol")
            if _sym_obj:
                content = getattr(_sym_obj, "source_text", "") or getattr(_sym_obj, "docstring", "") or ""
        if not content.strip():
            continue

        doc = Document(
            page_content=content,
            metadata={
                "source": node_data.get("rel_path", node_data.get("file_path", "unknown")),
                "rel_path": node_data.get("rel_path", ""),
                "symbol_name": node_data.get("symbol_name", "") or node_data.get("name", ""),
                "symbol_type": node_data.get("symbol_type", "unknown"),
                "start_line": node_data.get("start_line", ""),
                "end_line": node_data.get("end_line", ""),
                "search_score": score,
                "search_source": "graph",
            },
        )
        documents.append(doc)

    return documents




def _find_graph_node(code_graph: Any, symbol_name: str, rel_path: str = "") -> str | None:
    """Find the best-matching node ID in the graph for a given symbol.

    Tries exact node_id first (often ``rel_path::symbol_name``), then falls
    back to substring matching on symbol_name.
    """
    if code_graph is None or not symbol_name:
        return None

    # Strategy 1: exact composite key (common format: "path/file.py::ClassName")
    if rel_path:
        candidate = f"{rel_path}::{symbol_name}"
        if candidate in code_graph:
            return candidate

    # Strategy 2: scan for best substring match
    candidates = []
    name_lower = symbol_name.lower()
    for node_id in code_graph.nodes():
        if name_lower in node_id.lower():
            candidates.append(node_id)

    if not candidates:
        return None

    # Prefer shortest (most specific) match
    return min(candidates, key=len)


def _format_neighbors(code_graph: Any, node_id: str, max_per_direction: int = 8) -> list[str]:
    """Format 1-hop neighbors of *node_id* as readable lines.

    Returns a list like:
        - → `ChildClass` [inherits]
        - ← `CallerFunction` [calls]
    """
    lines: list[str] = []

    # Outgoing edges
    out_count = 0
    for successor in code_graph.successors(node_id):
        if out_count >= max_per_direction:
            remaining = sum(1 for _ in code_graph.successors(node_id)) - max_per_direction
            if remaining > 0:
                lines.append(f"  - → ... and {remaining} more")
            break
        edge_data = code_graph.get_edge_data(node_id, successor)
        rel_type = _extract_rel_type(edge_data)
        lines.append(f"  - → `{successor}` [{rel_type}]")
        out_count += 1

    # Incoming edges
    in_count = 0
    for predecessor in code_graph.predecessors(node_id):
        if in_count >= max_per_direction:
            remaining = sum(1 for _ in code_graph.predecessors(node_id)) - max_per_direction
            if remaining > 0:
                lines.append(f"  - ← ... and {remaining} more")
            break
        edge_data = code_graph.get_edge_data(predecessor, node_id)
        rel_type = _extract_rel_type(edge_data)
        lines.append(f"  - ← `{predecessor}` [{rel_type}]")
        in_count += 1

    return lines


def _extract_rel_type(edge_data: Any) -> str:
    """Extract relationship type from edge data (handles MultiDiGraph format)."""
    if not edge_data:
        return "related"
    # MultiDiGraph: edge_data is {key: {attr_dict}}
    if isinstance(edge_data, dict):
        for _key, data in edge_data.items():
            if isinstance(data, dict):
                return data.get("relationship_type", "related")
    return "related"


def create_codebase_tools(
    retriever_stack: Any,  # WikiRetrieverStack
    graph_manager: Any,  # GraphManager
    code_graph: Any,  # NetworkX graph
    repo_analysis: dict | None = None,
    event_callback: Callable | None = None,
    similarity_threshold: float = 0.75,
    graph_text_index: Any = None,  # GraphTextIndex (FTS5)
    repo_path: str | None = None,  # Path to cloned repo for direct file access
    query_service: Any = None,  # Pre-built GraphQueryService or MultiGraphQueryService (projects)
) -> list:
    """
    Create custom tools for deep research.

    search_codebase now uses EmbeddingsFilter for:
    - Semantic reranking of results
    - Bounded output (configurable k)
    - Threshold filtering (default 0.75)

    Args:
        retriever_stack: WikiRetrieverStack instance for codebase search
        graph_manager: GraphManager instance for graph operations
        code_graph: The loaded code graph (NetworkX)
        repo_analysis: Pre-loaded repository analysis (optional)
        event_callback: Callback for emitting thinking events
        similarity_threshold: Minimum similarity for EmbeddingsFilter (0.0-1.0)

    Returns:
        List of LangChain tools for DeepAgents
    """
    emit = event_callback or (lambda x: None)

    # Build unified query service (SPEC-1: replaces scattered O(N) scans)
    # Use pre-built query_service when provided (e.g. MultiGraphQueryService for projects)
    if query_service is None:
        query_service = GraphQueryService(code_graph, fts_index=graph_text_index) if code_graph else None

    # Get embeddings from retriever_stack for reranking
    embeddings = None
    if retriever_stack and hasattr(retriever_stack, "vectorstore_manager"):
        embeddings = getattr(retriever_stack.vectorstore_manager, "embeddings", None)

    # Doc search cap: the vector store mostly holds documentation chunks,
    # so we cap its contribution to avoid drowning out code results from FTS5.
    _MAX_DOC_RESULTS = int(os.environ.get("WIKIS_MAX_DOC_RESULTS", "3"))

    @tool(parse_docstring=True)
    def search_codebase(query: str, k: int = 10) -> str:
        """Search the repository codebase for relevant code, classes, functions, and documentation.

        Uses hybrid search: vector store (semantic, capped at 3 doc results)
        + code graph FTS5 (full-text, up to k results).
        Returns actual code symbols even when docs are separated from code.

        Args:
            query: Search query describing what you're looking for
            k: Maximum number of results after reranking (default 10)
        """
        emit({"type": "tool_start", "tool": "search_codebase", "input": query, "timestamp": datetime.now().isoformat()})

        try:
            all_docs = []

            # === Branch 1: Vector store search (semantic — doc-heavy, capped) ===
            # The vector store primarily contains documentation chunks.
            # We cap results to _MAX_DOC_RESULTS (default 3) so code results
            # from FTS5 aren't drowned out.
            k_doc = min(k, _MAX_DOC_RESULTS)
            if retriever_stack:
                try:
                    # Skip EmbeddingsFilter reranking — it can reject valid results
                    # when the filter uses a different embedding model than the FAISS
                    # index (e.g. text-embedding-3-large vs all-MiniLM-L6-v2).
                    # FAISS similarity is already a good relevance signal.
                    vs_docs = retriever_stack.search_repository(query=query, k=k_doc, apply_expansion=False)

                    for doc in vs_docs:
                        doc.metadata["search_source"] = "vectorstore"
                    all_docs.extend(vs_docs)
                    logger.info(f"[SEARCH_CODEBASE] VS returned {len(vs_docs)} docs (cap={k_doc}, query={query!r:.60})")
                except Exception as e:
                    logger.warning(f"[SEARCH_CODEBASE] Vector store search failed: {e}")

            # === Branch 2: Graph full-text search (code symbols, at least k) ===
            # Always run graph search to ensure code symbols are found even when
            # SEPARATE_DOC_INDEX routes docs away from the graph.
            k_code = max(k, 10)  # At least 10 code results from FTS5
            graph_docs = []
            try:
                if graph_text_index is not None and graph_text_index.is_open:
                    # FTS5-backed search: use search_smart for better query
                    # construction (keyword extraction + intent-aware FTS5).
                    graph_docs = graph_text_index.search_smart(
                        query,
                        k=k_code,
                        intent="general",
                        exclude_types=DOC_SYMBOL_TYPES,
                    )
                    if graph_docs:
                        top_names = [d.metadata.get("symbol_name", "?") for d in graph_docs[:5]]
                        logger.info(
                            f"[SEARCH_CODEBASE][FTS5] {len(graph_docs)} results "
                            f"(k={k_code}, query={query!r:.60}, "
                            f"top={top_names})"
                        )
                    else:
                        logger.info(f"[SEARCH_CODEBASE][FTS5] 0 results (query={query!r:.60})")
                elif code_graph is not None:
                    # Fallback: brute-force keyword scan
                    graph_docs = _search_graph_by_text(code_graph, query, k=k_code)
                    logger.info(f"[SEARCH_CODEBASE][BRUTE] {len(graph_docs)} results (query={query!r:.60})")
            except Exception as e:
                logger.warning(f"[SEARCH_CODEBASE] Graph search failed: {e}")

            # Tag graph results with source (preserve 'graph_fts' if set by FTS5)
            for doc in graph_docs:
                doc.metadata.setdefault("search_source", "graph")

            # === Fusion: combine VS + graph results ===
            if HYBRID_FUSION_ENABLED and (all_docs or graph_docs):
                # Reciprocal Rank Fusion: rank-based merging of both lists.
                # Produces a single ranked list where documents appearing in
                # BOTH sources get boosted.
                ranked_lists = {}
                if all_docs:
                    ranked_lists["vectorstore"] = all_docs
                if graph_docs:
                    ranked_lists["graph"] = graph_docs
                all_docs = fuse_search_results(
                    ranked_lists,
                    cap=k * 2,
                    dedup_key="symbol_name",
                )
                logger.info(f"[SEARCH_CODEBASE][RRF] Fused {len(all_docs)} results from {list(ranked_lists.keys())}")
            else:
                # Legacy path: simple concatenation (VS first, graph deduped)
                if graph_docs:
                    vs_symbols = {
                        doc.metadata.get("symbol_name", "").lower()
                        for doc in all_docs
                        if doc.metadata.get("symbol_name")
                    }
                    for doc in graph_docs:
                        sym = doc.metadata.get("symbol_name", "").lower()
                        if sym and sym not in vs_symbols:
                            all_docs.append(doc)
                            vs_symbols.add(sym)

                # Sort: prefer vector store results (semantic), then graph
                all_docs.sort(key=lambda d: 0 if d.metadata.get("search_source") == "vectorstore" else 1)

                # Cap total results
                all_docs = all_docs[: k * 2]

            results = []
            for i, doc in enumerate(all_docs):
                source = doc.metadata.get("source", doc.metadata.get("rel_path", "unknown"))
                symbol = doc.metadata.get("symbol_name", "")
                symbol_type = doc.metadata.get("symbol_type", "")
                start_line = doc.metadata.get("start_line", "")
                end_line = doc.metadata.get("end_line", "")
                search_src = doc.metadata.get("search_source", "?")

                result_block = f"""
### [{i + 1}] {source} ({search_src})
**Symbol:** `{symbol}` ({symbol_type})
**Lines:** {start_line}-{end_line}

```
{doc.page_content}
```
"""
                results.append(result_block)

            emit(
                {
                    "type": "tool_end",
                    "tool": "search_codebase",
                    "result_count": len(all_docs),
                    "timestamp": datetime.now().isoformat(),
                }
            )

            if not results:
                return f"No results found for query: {query}"

            return f"## Search Results for: {query}\n\nFound {len(all_docs)} relevant items:\n" + "\n---\n".join(
                results
            )

        except Exception as e:
            logger.error(f"Search failed: {e}", exc_info=True)
            emit(
                {
                    "type": "tool_error",
                    "tool": "search_codebase",
                    "error": str(e),
                    "timestamp": datetime.now().isoformat(),
                }
            )
            return f"Search failed: {str(e)}"

    @tool(parse_docstring=True)
    def get_symbol_relationships(symbol_name: str, max_depth: int = 2) -> str:
        """Get relationships for a code symbol (class, function, module).

        Use this to understand how a symbol connects to other parts of the codebase:
        - What classes it inherits from or implements
        - What functions it calls or is called by
        - What modules depend on it

        Args:
            symbol_name: Name of the symbol to analyze (e.g., "AuthService", "login")
            max_depth: How many relationship levels to traverse (default 2)
        """
        emit(
            {
                "type": "tool_start",
                "tool": "get_symbol_relationships",
                "input": symbol_name,
                "timestamp": datetime.now().isoformat(),
            }
        )

        if code_graph is None:
            return "Code graph not available for relationship analysis"

        try:
            # Use GraphQueryService for O(1) resolution (SPEC-1)
            target_node = None
            matching_nodes = []
            if query_service:
                target_node = query_service.resolve_symbol(symbol_name)
                if target_node:
                    matching_nodes = [target_node]

            # Fallback: brute-force scan if service didn't find it
            if not target_node:
                for node_id in code_graph.nodes():
                    if symbol_name.lower() in node_id.lower():
                        matching_nodes.append(node_id)
                if matching_nodes:
                    target_node = min(matching_nodes, key=len)

            if not target_node:
                return f"Symbol '{symbol_name}' not found in code graph. Try using search_codebase first to find exact symbol names."

            # Use service for bounded traversal (SPEC-1)
            if query_service:
                rels = query_service.get_relationships(
                    target_node,
                    direction="both",
                    max_depth=max_depth,
                    max_results=50,
                )

                lines = [f"# Relationships for `{symbol_name}`\n"]
                lines.append(f"**Matched Node:** `{target_node}`")
                lines.append(f"**Total Related Nodes:** {len(rels)}\n")

                if len(matching_nodes) > 1:
                    lines.append(
                        f"**Other Matching Nodes:** {', '.join(f'`{n}`' for n in matching_nodes[:10] if n != target_node)}\n"
                    )

                # Group by depth and direction
                [r for r in rels if r.source_name != symbol_name or r.hop_distance > 0]
                incoming = [
                    r
                    for r in rels
                    if r.target_name == target_node
                    or (r.source_name != target_node and any(r.target_name == target_node for _ in [1]))
                ]

                by_depth = {}
                for r in rels:
                    depth = r.hop_distance
                    if depth not in by_depth:
                        by_depth[depth] = []
                    by_depth[depth].append(r)

                for depth in sorted(by_depth.keys()):
                    depth_rels = by_depth[depth]
                    lines.append(f"\n## Depth {depth} Relationships")
                    for rel in depth_rels:
                        lines.append(f"- `{rel.source_name}` → `{rel.target_name}` [{rel.relationship_type}]")

                emit(
                    {
                        "type": "tool_end",
                        "tool": "get_symbol_relationships",
                        "result_count": len(rels),
                        "timestamp": datetime.now().isoformat(),
                    }
                )

                return "\n".join(lines)

            # Legacy fallback: manual graph traversal

            # Get neighbors at different depths
            neighbors_by_depth = {}
            visited = {target_node}
            current_level = {target_node}

            for depth in range(1, max_depth + 1):
                next_level = set()
                for node in current_level:
                    for neighbor in code_graph.neighbors(node):
                        if neighbor not in visited:
                            if depth not in neighbors_by_depth:
                                neighbors_by_depth[depth] = []
                            neighbors_by_depth[depth].append(
                                {"node": neighbor, "from": node, "edge_data": code_graph.get_edge_data(node, neighbor)}
                            )
                            next_level.add(neighbor)
                            visited.add(neighbor)
                current_level = next_level

            # Also check incoming edges
            incoming = []
            for predecessor in code_graph.predecessors(target_node):
                edge_data = code_graph.get_edge_data(predecessor, target_node)
                incoming.append({"node": predecessor, "edge_data": edge_data})

            # Format comprehensive output (NO truncation)
            lines = [f"# Relationships for `{symbol_name}`\n"]
            lines.append(f"**Matched Node:** `{target_node}`")
            lines.append(f"**Total Related Nodes:** {len(visited) - 1}\n")

            if matching_nodes and len(matching_nodes) > 1:
                lines.append(
                    f"**Other Matching Nodes:** {', '.join(f'`{n}`' for n in matching_nodes[:10] if n != target_node)}\n"
                )

            # Outgoing relationships by depth
            for depth, neighbors in neighbors_by_depth.items():
                lines.append(f"\n## Depth {depth} Relationships (Outgoing)")
                for rel in neighbors:
                    edge_info = ""
                    if rel["edge_data"]:
                        for _key, data in rel["edge_data"].items():
                            if isinstance(data, dict):
                                rel_type = data.get("relationship_type", "related")
                                edge_info = f" [{rel_type}]"
                    lines.append(f"- `{rel['from']}` → `{rel['node']}`{edge_info}")

            # Incoming relationships
            if incoming:
                lines.append("\n## Incoming Relationships (called by / used by)")
                for rel in incoming:
                    edge_info = ""
                    if rel["edge_data"]:
                        for _key, data in rel["edge_data"].items():
                            if isinstance(data, dict):
                                rel_type = data.get("relationship_type", "related")
                                edge_info = f" [{rel_type}]"
                    lines.append(f"- `{rel['node']}` → `{target_node}`{edge_info}")

            emit(
                {
                    "type": "tool_end",
                    "tool": "get_symbol_relationships",
                    "result_count": len(visited) - 1,
                    "timestamp": datetime.now().isoformat(),
                }
            )

            return "\n".join(lines)

        except Exception as e:
            logger.error(f"Graph analysis failed: {e}")
            return f"Relationship analysis failed: {str(e)}"

    @tool(parse_docstring=True)
    def search_graph(query: str, k: int = 5, include_neighbors: bool = True) -> str:
        """Search for code symbols and return them with their graph relationships.

        Combines FTS5 full-text search with graph traversal to return rich context:
        each matched symbol comes with its immediate neighbors (callers, callees,
        parent classes, implementations, etc.).

        Use this when you need to understand HOW a symbol fits into the codebase,
        not just WHAT it contains.  For pure content search, use search_codebase.

        Args:
            query: Search query (symbol name, concept, or keywords)
            k: Maximum number of primary symbols to return (default 5)
            include_neighbors: Whether to include 1-hop graph neighbors (default True)
        """
        emit({"type": "tool_start", "tool": "search_graph", "input": query, "timestamp": datetime.now().isoformat()})

        if code_graph is None:
            return "Code graph not available for graph search."

        try:
            # Step 1: Find matching symbols via FTS5 or brute-force
            matched_docs: list[Document] = []
            if graph_text_index is not None and graph_text_index.is_open:
                matched_docs = graph_text_index.search_smart(
                    query,
                    k=k,
                    intent="symbol",
                    exclude_types=DOC_SYMBOL_TYPES,
                )
            if not matched_docs and code_graph is not None:
                matched_docs = _search_graph_by_text(code_graph, query, k=k)

            if not matched_docs:
                return f"No symbols found matching: {query}"

            # Step 2: For each match, gather graph neighbors
            sections: list[str] = []
            for i, doc in enumerate(matched_docs[:k]):
                sym_name = doc.metadata.get("symbol_name", "") or doc.metadata.get("name", "")
                sym_type = doc.metadata.get("symbol_type", "")
                rel_path = doc.metadata.get("rel_path", doc.metadata.get("source", ""))
                start_line = doc.metadata.get("start_line", "")
                end_line = doc.metadata.get("end_line", "")

                section_lines = [
                    f"### [{i + 1}] `{sym_name}` ({sym_type})",
                    f"**File:** {rel_path}  **Lines:** {start_line}-{end_line}",
                ]

                # Truncate content for summary (full content in search_codebase)
                content_preview = (doc.page_content or "")[:500]
                if len(doc.page_content or "") > 500:
                    content_preview += "\n... (truncated, use search_codebase for full)"
                section_lines.append(f"\n```\n{content_preview}\n```")

                if include_neighbors:
                    # Use GraphQueryService for O(1) resolution (SPEC-1)
                    node_id = None
                    if query_service:
                        node_id = query_service.resolve_symbol(sym_name, file_path=rel_path)
                    if not node_id:
                        node_id = _find_graph_node(code_graph, sym_name, rel_path)
                    if node_id:
                        neighbor_lines = _format_neighbors(code_graph, node_id)
                        if neighbor_lines:
                            section_lines.append("\n**Relationships:**")
                            section_lines.extend(neighbor_lines)

                sections.append("\n".join(section_lines))

            emit(
                {
                    "type": "tool_end",
                    "tool": "search_graph",
                    "result_count": len(sections),
                    "timestamp": datetime.now().isoformat(),
                }
            )

            header = f"## Graph Search: {query}\n\nFound {len(sections)} symbols with relationships:\n"
            return header + "\n\n---\n\n".join(sections)

        except Exception as e:
            logger.error(f"search_graph failed: {e}", exc_info=True)
            return f"Graph search failed: {str(e)}"

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
    # SPEC-5: Progressive Disclosure Tools
    # ================================================================
    # Cheap discovery tools (search_symbols, get_relationships_tool, search_docs)
    # + expensive detail tool (get_code).
    # Gated by WIKIS_PROGRESSIVE_TOOLS=1 (off by default).

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
                    if r.node_id and r.node_id in code_graph:
                        doc = (code_graph.nodes.get(r.node_id, {}) or {}).get("docstring", "") or ""
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
                return "No search index available. Use search_codebase instead."

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
        graph: Any,
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
        if graph is None or fts_index is None:
            return None
        if not (hasattr(fts_index, "search") and hasattr(fts_index, "is_open")):
            return None
        if not fts_index.is_open:
            return None

        node_data = graph.nodes.get(node_id, {})
        sym_type = (node_data.get("symbol_type") or "").lower()
        if sym_type not in CONTAINER_SYMBOL_TYPES:
            return None

        sym_name = node_data.get("symbol_name", "") or node_data.get("name", "")

        # Collect child method/function names via 'defines' edges
        method_names: list[str] = []
        for succ in graph.successors(node_id):
            # Check all edges between node_id → succ for a 'defines' relationship
            edge_data = graph.get_edge_data(node_id, succ)
            if edge_data is None:
                continue
            # MultiDiGraph returns {0: {...}, 1: {...}}, DiGraph returns {...}
            edges = (
                edge_data.values()
                if isinstance(edge_data, dict) and all(isinstance(v, dict) for v in edge_data.values())
                else [edge_data]
            )
            is_defines = any(e.get("relationship_type", "").lower() == "defines" for e in edges)
            if not is_defines:
                continue

            succ_data = graph.nodes.get(succ, {})
            succ_type = (succ_data.get("symbol_type") or "").lower()
            if succ_type not in ("method", "function"):
                continue
            name = succ_data.get("symbol_name", "") or succ_data.get("name", "")
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
                    code_graph,
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
                    code_graph,
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

        if code_graph is None:
            return "Code graph not available"

        try:
            node_id = None
            if query_service:
                node_id = query_service.resolve_symbol(symbol_name)
            if not node_id:
                node_id = _find_graph_node(code_graph, symbol_name)
            if not node_id:
                return f"Symbol '{symbol_name}' not found. Check the exact name from search_symbols."

            data = code_graph.nodes.get(node_id, {})
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
                    # Doc search uses FAISS similarity directly — skip EmbeddingsFilter
                    # reranking because the filter may use a different embedding model
                    # than the index, causing valid results to be rejected.
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

        Use this to read files that weren't returned by search_codebase,
        such as config files, Dockerfiles, Makefiles, scripts, or to see
        full file context around a search result snippet.

        Args:
            file_path: Relative path from repo root (e.g. 'src/auth/manager.py')
            offset: Starting line number, 0-based (default 0)
            limit: Maximum number of lines to return (default 200)
        """
        if not _repo_available:
            return "Error: Repository clone is not available on disk. Use search_codebase instead."

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
            return "Error: Repository clone is not available on disk. Use search_codebase instead."

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
    # Tool Selection: feature-flag gated
    # ================================================================
    _progressive = os.environ.get("WIKIS_PROGRESSIVE_TOOLS", "").strip()
    _use_progressive = _progressive in ("1", "true", "yes")

    # File access tools — included only when the cloned repo is on disk
    _file_tools = [read_source_file, list_repo_files] if _repo_available else []
    if _repo_available:
        logger.info("[RESEARCH_TOOLS] read_source_file + list_repo_files enabled (repo_path=%s)", repo_path)

    if _use_progressive:
        logger.info("[RESEARCH_TOOLS] Using progressive disclosure tools (SPEC-5)")
        return [
            search_symbols,
            get_relationships_tool,
            get_code,
            search_docs,
            query_graph,
            think,
        ] + _file_tools
    else:
        return [
            search_codebase,
            get_symbol_relationships,
            search_graph,
            think,
        ] + _file_tools
