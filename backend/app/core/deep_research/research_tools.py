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
from ..constants import ARCHITECTURAL_SYMBOLS, CODE_SYMBOL_TYPES, DOC_SYMBOL_TYPES
from .hybrid_fusion import (
    HYBRID_FUSION_ENABLED,
    fuse_search_results,
)

logger = logging.getLogger(__name__)


# Container types that can own child methods/functions via 'defines' edges.
# Used by orphan FTS fallback to discover string-based references.
CONTAINER_SYMBOL_TYPES = frozenset({"class", "interface", "struct", "enum", "trait"})
PROGRESSIVE_SYMBOL_TYPES = frozenset(CODE_SYMBOL_TYPES | {"method"})

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
    repo_path: str | dict[str, str] | None = None,  # Path(s) to cloned repo(s) for direct file access
    query_service: Any = None,  # Pre-built GraphQueryService or MultiGraphQueryService (projects)
    min_confidence: str | None = None,  # #120/#157: edge-confidence floor
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

    def _node_content(data: dict[str, Any]) -> str:
        symbol_obj = data.get("symbol") if data else None
        if symbol_obj is not None and getattr(symbol_obj, "source_text", None):
            return symbol_obj.source_text
        return (
            data.get("content")
            or data.get("source_text")
            or data.get("docstring")
            or ""
        )

    def _doc_from_symbol_result(result: Any, search_source: str) -> Document | None:
        """Materialize a query-service symbol hit into a tool result document."""
        data: dict[str, Any] = {}
        if query_service is not None and hasattr(query_service, "get_node"):
            try:
                data = query_service.get_node(result.node_id) or {}
            except Exception:
                data = {}
        if not data and code_graph is not None:
            try:
                data = code_graph.nodes.get(result.node_id, {}) or {}
            except Exception:
                data = {}

        content = _node_content(data)
        if not content.strip():
            content = result.docstring or ""
        if not content.strip():
            return None

        wiki_id = (
            getattr(result, "wiki_id", "")
            or getattr(result, "source_wiki_id", "")
            or data.get("wiki_id", "")
        )
        rel_path = result.rel_path or data.get("rel_path", "") or result.file_path or data.get("file_path", "")
        return Document(
            page_content=content,
            metadata={
                "source": rel_path or "unknown",
                "rel_path": rel_path,
                "file_path": result.file_path or data.get("file_path", ""),
                "symbol_name": result.symbol_name or data.get("symbol_name", "") or data.get("name", ""),
                "symbol_type": result.symbol_type or data.get("symbol_type", "unknown"),
                "start_line": data.get("start_line", "") or data.get("line_start", ""),
                "end_line": data.get("end_line", "") or data.get("line_end", ""),
                "node_id": result.node_id,
                "raw_node_id": getattr(result, "raw_node_id", "") or data.get("raw_node_id", ""),
                "source_wiki_id": wiki_id,
                "wiki_id": wiki_id,
                "search_score": result.score,
                "score": result.score,
                "search_source": search_source,
                "project_dedup_key": f"{wiki_id}::{result.node_id or rel_path}",
            },
        )

    def _format_doc_source(doc: Document) -> str:
        source = doc.metadata.get("source", doc.metadata.get("rel_path", "unknown"))
        wiki_id = doc.metadata.get("source_wiki_id") or doc.metadata.get("wiki_id")
        return f"{wiki_id}:{source}" if wiki_id else source

    def _is_document_hit(doc: Document) -> bool:
        metadata = doc.metadata or {}
        symbol_type = (metadata.get("symbol_type") or metadata.get("type") or "").lower()
        return bool(
            metadata.get("is_documentation")
            or metadata.get("semantic_retrieved")
            or metadata.get("is_doc")
            or symbol_type in DOC_SYMBOL_TYPES
            or symbol_type.endswith("_document")
        )

    def _diversify_project_results(results: list[Any], limit: int) -> list[Any]:
        """Keep project discovery from being dominated by one wiki."""
        if limit <= 0 or len(results) <= limit:
            return results[:limit]
        buckets: dict[str, list[Any]] = {}
        for result in results:
            wiki_id = getattr(result, "wiki_id", "") or getattr(result, "source_wiki_id", "") or ""
            buckets.setdefault(wiki_id, []).append(result)
        if len(buckets) <= 1:
            return results[:limit]

        selected: list[Any] = []
        seen_ids: set[str] = set()
        active_keys = [key for key, bucket in buckets.items() if bucket]
        while active_keys and len(selected) < limit:
            next_active: list[str] = []
            for key in active_keys:
                bucket = buckets[key]
                if not bucket:
                    continue
                item = bucket.pop(0)
                item_key = getattr(item, "node_id", "") or f"{key}:{getattr(item, 'symbol_name', '')}"
                if item_key not in seen_ids:
                    selected.append(item)
                    seen_ids.add(item_key)
                    if len(selected) >= limit:
                        break
                if bucket:
                    next_active.append(key)
            active_keys = next_active

        if len(selected) < limit:
            for item in results:
                item_key = getattr(item, "node_id", "") or getattr(item, "symbol_name", "")
                if item_key and item_key in seen_ids:
                    continue
                selected.append(item)
                if item_key:
                    seen_ids.add(item_key)
                if len(selected) >= limit:
                    break
        return selected[:limit]

    def _call_cross_repo_edges(node_id: str, direction: str = "both") -> list[dict[str, Any]]:
        if query_service is None or not hasattr(query_service, "cross_repo_edges"):
            return []
        try:
            return query_service.cross_repo_edges(node_id, direction=direction) or []
        except TypeError:
            try:
                return query_service.cross_repo_edges(node_id) or []
            except Exception:
                return []
        except Exception:
            return []

    def _relationship_extra(r: Any, other_node_id: str = "") -> str:
        extras: list[str] = []
        other_wiki_id = ""
        if other_node_id and "::" in other_node_id:
            other_wiki_id = other_node_id.split("::", 1)[0]
        other_wiki_id = other_wiki_id or getattr(r, "target_wiki_id", "") or getattr(r, "source_wiki_id", "")
        if other_wiki_id:
            extras.append(f"wiki: {other_wiki_id}")
        if other_node_id:
            extras.append(f"id: `{other_node_id}`")
        provenance = getattr(r, "provenance", None) or {}
        surface = provenance.get("surface") if isinstance(provenance, dict) else ""
        matcher = provenance.get("matcher") if isinstance(provenance, dict) else ""
        weight = getattr(r, "weight", None)
        if surface:
            extras.append(f"surface: `{surface}`")
        if matcher:
            extras.append(f"matcher: {matcher}")
        if isinstance(weight, (int, float)) and weight != 1.0:
            extras.append(f"weight: {weight:.3f}")
        return " " + " ".join(f"[{extra}]" for extra in extras) if extras else ""

    def _format_cross_repo_edge(row: dict[str, Any]) -> str:
        source_node_id = row.get("source_node_id", "") or ""
        target_node_id = row.get("target_node_id", "") or ""
        provenance = dict(row.get("provenance") or {})
        source_data = query_service.get_node(source_node_id) if hasattr(query_service, "get_node") else {}
        target_data = query_service.get_node(target_node_id) if hasattr(query_service, "get_node") else {}
        source_data = source_data or {}
        target_data = target_data or {}
        source_name = source_data.get("symbol_name") or source_data.get("name") or source_node_id
        target_name = target_data.get("symbol_name") or target_data.get("name") or target_node_id
        source_type = source_data.get("symbol_type", "") or "symbol"
        target_type = target_data.get("symbol_type", "") or "symbol"
        source_file = source_data.get("rel_path") or source_data.get("file_path") or "unknown"
        target_file = target_data.get("rel_path") or target_data.get("file_path") or "unknown"
        rel_type = provenance.get("source_relationship_type") or row.get("edge_class") or "cross_repo"
        source_wiki = provenance.get("source_wiki_id") or (source_node_id.split("::", 1)[0] if "::" in source_node_id else "")
        target_wiki = provenance.get("target_wiki_id") or (target_node_id.split("::", 1)[0] if "::" in target_node_id else "")
        surface = provenance.get("surface", "")
        matcher = provenance.get("matcher", "")
        detail = [f"{rel_type}"]
        if surface:
            detail.append(f"surface `{surface}`")
        if matcher:
            detail.append(f"matcher {matcher}")
        try:
            detail.append(f"weight {float(row.get('weight', 1.0) or 1.0):.3f}")
        except (TypeError, ValueError):
            pass
        return (
            f"`{source_name}` ({source_type}) [wiki: {source_wiki}] [id: `{source_node_id}`] in {source_file} "
            f"→ `{target_name}` ({target_type}) [wiki: {target_wiki}] [id: `{target_node_id}`] in {target_file} "
            f"[{'; '.join(detail)}]"
        )

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
                        wiki_id = doc.metadata.get("source_wiki_id") or doc.metadata.get("wiki_id") or ""
                        node_id = doc.metadata.get("node_id") or doc.metadata.get("source") or doc.metadata.get("rel_path") or ""
                        symbol_name = doc.metadata.get("symbol_name") or ""
                        doc.metadata.setdefault(
                            "project_dedup_key",
                            f"{wiki_id}::{node_id or symbol_name}",
                        )
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
                if query_service is not None:
                    svc_results = query_service.search(
                        query,
                        k=k_code,
                        exclude_types=DOC_SYMBOL_TYPES,
                    )
                    for result in svc_results:
                        doc = _doc_from_symbol_result(result, "graph_query")
                        if doc is not None:
                            graph_docs.append(doc)
                    if graph_docs:
                        top_names = [d.metadata.get("symbol_name", "?") for d in graph_docs[:5]]
                        logger.info(
                            f"[SEARCH_CODEBASE][QUERY_SERVICE] {len(graph_docs)} results "
                            f"(k={k_code}, query={query!r:.60}, top={top_names})"
                        )
                    else:
                        logger.info(f"[SEARCH_CODEBASE][QUERY_SERVICE] 0 results (query={query!r:.60})")
                elif graph_text_index is not None and graph_text_index.is_open:
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
                    dedup_key="project_dedup_key",
                )
                logger.info(f"[SEARCH_CODEBASE][RRF] Fused {len(all_docs)} results from {list(ranked_lists.keys())}")
            else:
                # Legacy path: simple concatenation (VS first, graph deduped)
                if graph_docs:
                    vs_symbols = {
                        (
                            doc.metadata.get("source_wiki_id") or doc.metadata.get("wiki_id") or "",
                            doc.metadata.get("symbol_name", "").lower(),
                        )
                        for doc in all_docs
                        if doc.metadata.get("symbol_name")
                    }
                    for doc in graph_docs:
                        sym = doc.metadata.get("symbol_name", "").lower()
                        wiki_id = doc.metadata.get("source_wiki_id") or doc.metadata.get("wiki_id") or ""
                        key = (wiki_id, sym)
                        if sym and key not in vs_symbols:
                            all_docs.append(doc)
                            vs_symbols.add(key)

                # Sort: prefer vector store results (semantic), then graph
                all_docs.sort(key=lambda d: 0 if d.metadata.get("search_source") == "vectorstore" else 1)

                # Cap total results
                all_docs = all_docs[: k * 2]

            results = []
            for i, doc in enumerate(all_docs):
                source = _format_doc_source(doc)
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

        if query_service is None and code_graph is None:
            return "Code graph not available for relationship analysis"

        try:
            # Use GraphQueryService for O(1) resolution (SPEC-1)
            target_node = None
            matching_nodes = []
            if query_service:
                if hasattr(query_service, "get_node"):
                    try:
                        if query_service.get_node(symbol_name) is not None:
                            target_node = symbol_name
                    except Exception:
                        target_node = None
                target_node = target_node or query_service.resolve_symbol(symbol_name)
                if target_node:
                    matching_nodes = [target_node]

            # Fallback: brute-force scan if service didn't find it
            if not target_node and code_graph is not None:
                for node_id in code_graph.nodes():
                    if symbol_name.lower() in node_id.lower():
                        matching_nodes.append(node_id)
                if matching_nodes:
                    target_node = min(matching_nodes, key=len)

            if not target_node:
                return f"Symbol '{symbol_name}' not found in code graph. Try using search_codebase first to find exact symbol names."

            # Use service for bounded traversal (SPEC-1).
            # #120/#157: closure-captured ``min_confidence`` flows
            # through to the live agent path here.
            if query_service:
                rels = query_service.get_relationships(
                    target_node,
                    direction="both",
                    max_depth=max_depth,
                    max_results=50,
                    min_confidence=min_confidence,
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

        if query_service is None and code_graph is None:
            return "Code graph not available for graph search."

        try:
            # Step 1: Find matching symbols via FTS5 or brute-force
            matched_docs: list[Document] = []
            if query_service is not None:
                svc_results = query_service.search(
                    query,
                    k=k,
                    exclude_types=DOC_SYMBOL_TYPES,
                )
                matched_docs = [
                    doc
                    for result in svc_results
                    if (doc := _doc_from_symbol_result(result, "graph_query")) is not None
                ]
            elif graph_text_index is not None and graph_text_index.is_open:
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
                    node_id = doc.metadata.get("node_id") or None
                    if query_service:
                        node_id = node_id or query_service.resolve_symbol(sym_name, file_path=rel_path)
                    if not node_id and code_graph is not None:
                        node_id = _find_graph_node(code_graph, sym_name, rel_path)
                    if node_id:
                        neighbor_lines = []
                        if query_service is not None:
                            rels = query_service.get_relationships(
                                node_id,
                                direction="both",
                                max_depth=1,
                                max_results=8,
                            )
                            for rel in rels:
                                neighbor_lines.append(
                                    f"  - `{rel.source_name}` → `{rel.target_name}` [{rel.relationship_type}]"
                                )
                        elif code_graph is not None:
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
            if symbol_type:
                requested_types = frozenset({symbol_type.lower()})
                sym_types = requested_types & PROGRESSIVE_SYMBOL_TYPES
                if not sym_types:
                    return f"No architectural symbols found for unsupported symbol_type: {symbol_type}"
            else:
                sym_types = PROGRESSIVE_SYMBOL_TYPES
            path_prefix = file_prefix if file_prefix else None

            if query_service:
                wiki_count = 1
                if hasattr(query_service, "wiki_ids"):
                    try:
                        wiki_count = max(len(query_service.wiki_ids()), 1)
                    except Exception:
                        wiki_count = 1
                svc_results = query_service.search(
                    query,
                    k=max(k * wiki_count, k),
                    symbol_types=sym_types,
                    exclude_types=DOC_SYMBOL_TYPES,
                    path_prefix=path_prefix,
                )
                svc_results = _diversify_project_results(svc_results, k)
                for r in svc_results:
                    # Extract one-line doc (first sentence of docstring).
                    # Use ``query_service.get_node`` so the lookup works for
                    # nodes from any wiki in a project, not just the primary
                    # wiki's in-memory NX graph.
                    one_line = ""
                    node_data: dict = {}
                    if r.node_id and hasattr(query_service, "get_node"):
                        node_data = query_service.get_node(r.node_id) or {}
                    if not node_data and r.node_id and code_graph is not None and r.node_id in code_graph:
                        node_data = code_graph.nodes.get(r.node_id, {}) or {}
                    doc = node_data.get("docstring", "") or ""
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
                            "node_id": r.node_id,
                            "wiki_id": getattr(r, "wiki_id", "") or getattr(r, "source_wiki_id", ""),
                        }
                    )
            elif graph_text_index and graph_text_index.is_open:
                docs = graph_text_index.search_smart(
                    query,
                    k=k,
                    intent="symbol",
                    symbol_types=sym_types,
                    exclude_types=DOC_SYMBOL_TYPES,
                )
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
                            "node_id": m.get("node_id", ""),
                            "wiki_id": m.get("wiki_id", "") or m.get("source_wiki_id", ""),
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
                wiki_str = f" [wiki: {r['wiki_id']}]" if r.get("wiki_id") else ""
                id_str = f" [id: `{r['node_id']}`]" if r.get("node_id") else ""
                lines.append(
                    f"{i + 1}. `{r['name']}` ({r['kind']}{layer_str}) in {r['file']} "
                    f"[{r['refs']} refs]{wiki_str}{id_str}{doc_str}"
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
        qs: Any = None,
    ) -> str | None:
        """Discover implicit string-based references for orphan container symbols.

        When a class/struct/trait/enum/interface has zero graph-edge
        relationships its method names may still be referenced as string
        literals in framework or RPC code (e.g. ``Event.on_startup``
        dispatched via ``emit("on_startup")``).

        Resolves child method names via the abstract query service
        (cross-wiki + storage-backend safe), and runs FTS5 text search
        for each name.  Falls back to direct ``code_graph`` traversal
        only when ``qs`` is unavailable (legacy single-wiki path).
        Returns a formatted result string if any hits are found,
        otherwise ``None``.
        """
        if fts_index is None:
            return None
        if not (hasattr(fts_index, "search") and hasattr(fts_index, "is_open")):
            return None
        if not fts_index.is_open:
            return None

        # --- Resolve container node attributes ---
        # Prefer query_service.get_node so we transparently support
        # storage-backed wikis and cross-wiki node ownership.
        node_data: dict = {}
        if qs is not None and hasattr(qs, "get_node"):
            node_data = qs.get_node(node_id) or {}
        if not node_data and graph is not None:
            try:
                node_data = graph.nodes.get(node_id, {}) or {}
            except Exception:
                node_data = {}
        if not node_data:
            return None

        sym_type = (node_data.get("symbol_type") or "").lower()
        if sym_type not in CONTAINER_SYMBOL_TYPES:
            return None

        sym_name = node_data.get("symbol_name", "") or node_data.get("name", "")

        # --- Collect child method/function names via 'defines' edges ---
        method_names: list[str] = []
        if qs is not None and hasattr(qs, "get_relationships"):
            try:
                rels = qs.get_relationships(
                    node_id,
                    direction="outgoing",
                    max_depth=1,
                    max_results=200,
                )
            except Exception:
                rels = []
            for r in rels or []:
                if (getattr(r, "relationship_type", "") or "").lower() != "defines":
                    continue
                tgt_type = (getattr(r, "target_type", "") or "").lower()
                if tgt_type not in ("method", "function"):
                    continue
                name = getattr(r, "target_name", "") or ""
                if name and not name.startswith("__") and "::" not in name:
                    method_names.append(name)
                elif name:
                    # target_name may be a fully-qualified node_id; take the
                    # last segment as the bare method name for FTS.
                    last = name.rsplit("::", 1)[-1]
                    if last and not last.startswith("__"):
                        method_names.append(last)

        if not method_names and graph is not None:
            # Legacy single-wiki path: walk the in-memory NX graph directly.
            try:
                successors_iter = list(graph.successors(node_id))
            except Exception:
                successors_iter = []
            for succ in successors_iter:
                edge_data = graph.get_edge_data(node_id, succ)
                if edge_data is None:
                    continue
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

    @tool("get_relationships", parse_docstring=True)
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
            # Prefer exact node IDs when project-mode tools pass a namespaced
            # project node ID; otherwise resolve by symbol name. Main's
            # confidence floor is forwarded when the backing query service
            # supports it.
            node_id = None
            if hasattr(query_service, "get_node"):
                try:
                    if query_service.get_node(symbol_name) is not None:
                        node_id = symbol_name
                except Exception:
                    node_id = None
            if node_id:
                try:
                    rels = query_service.get_relationships(
                        node_id,
                        direction=direction,
                        max_depth=max_depth,
                        max_results=50,
                        min_confidence=min_confidence,
                    )
                except TypeError:
                    rels = query_service.get_relationships(
                        node_id,
                        direction=direction,
                        max_depth=max_depth,
                        max_results=50,
                    )
            else:
                try:
                    node_id, rels = query_service.resolve_and_traverse(
                        symbol_name,
                        direction=direction,
                        max_depth=max_depth,
                        max_results=50,
                        min_confidence=min_confidence,
                    )
                except TypeError:
                    node_id, rels = query_service.resolve_and_traverse(
                        symbol_name,
                        direction=direction,
                        max_depth=max_depth,
                        max_results=50,
                    )

            if not node_id:
                return f"Symbol '{symbol_name}' not found. Try search_symbols first."

            root_display_name = symbol_name
            root_names = {symbol_name.lower(), node_id.lower()}
            if hasattr(query_service, "get_node"):
                try:
                    root_data = query_service.get_node(node_id) or {}
                except Exception:
                    root_data = {}
                root_symbol_name = root_data.get("symbol_name", "") or root_data.get("name", "")
                if root_symbol_name:
                    root_display_name = root_symbol_name
                    root_names.add(root_symbol_name.lower())

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
                    qs=query_service,
                )
                if orphan_hits:
                    return orphan_hits
                return f"`{symbol_name}` has no relationships within {max_depth} hops."

            lines = [f"Relationships for `{root_display_name}` (node: `{node_id}`):\n"]
            for r in rels:
                # Determine arrow direction relative to the root symbol:
                # If node_id is the source (or source_name matches), it's outgoing
                source_node_id = getattr(r, "source_node_id", "") or ""
                target_node_id = getattr(r, "target_node_id", "") or ""
                src_low = (r.source_name or "").lower()
                is_outgoing = source_node_id == node_id or r.source_name == node_id or src_low in root_names
                if is_outgoing:
                    arrow, other, other_type, other_node_id = "→", r.target_name, r.target_type, target_node_id
                else:
                    arrow, other, other_type, other_node_id = "←", r.source_name, r.source_type, source_node_id
                hop_str = f" (hop {r.hop_distance})" if r.hop_distance > 1 else ""
                type_str = f" ({other_type})" if other_type else ""
                extra_str = _relationship_extra(r, other_node_id)
                lines.append(f"  {arrow} `{other}`{type_str} [{r.relationship_type}]{hop_str}{extra_str}")

            # If ALL rels are structural 'defines' edges, the container
            # only has child-containment links — also run orphan FTS to
            # discover implicit string-based references.
            if all(r.relationship_type.lower() == "defines" for r in rels):
                orphan_hits = _orphan_fts_fallback(
                    node_id,
                    code_graph,
                    graph_text_index,
                    emit,
                    qs=query_service,
                )
                if orphan_hits:
                    lines.append("")
                    lines.append(orphan_hits)

            return "\n".join(lines)

        except Exception as e:
            logger.error(f"get_relationships failed: {e}", exc_info=True)
            return f"Relationship analysis failed: {str(e)}"

    @tool(parse_docstring=True)
    def find_cross_repo_links(query: str = "", symbol_name: str = "", k: int = 10) -> str:
        """Find direct cross-repo evidence links in project mode.

        Use this for project-level questions about how repositories integrate.
        These links come from explicit API-surface matches such as REST endpoints,
        DTO/object shapes, FFI/ABI calls, protobuf/gRPC services, GraphQL fields,
        BDD steps, CLI commands, and related direct contracts.

        Args:
            query: Search text used to find starting symbols when symbol_name is not provided
            symbol_name: Exact symbol or project node id from search_symbols results
            k: Maximum number of direct links to return (default 10)
        """
        emit({"type": "tool_start", "tool": "find_cross_repo_links", "input": symbol_name or query, "timestamp": datetime.now().isoformat()})

        if query_service is None or not hasattr(query_service, "cross_repo_edges"):
            return "Direct cross-repo links are available only in project mode."

        try:
            limit = max(1, min(int(k or 10), 25))
        except (TypeError, ValueError):
            limit = 10

        try:
            candidate_ids: list[str] = []
            if symbol_name:
                if hasattr(query_service, "get_node"):
                    try:
                        if query_service.get_node(symbol_name) is not None:
                            candidate_ids.append(symbol_name)
                    except Exception:
                        pass
                if not candidate_ids and hasattr(query_service, "resolve_symbol"):
                    resolved = query_service.resolve_symbol(symbol_name)
                    if resolved:
                        candidate_ids.append(resolved)
            elif query:
                results = query_service.search(
                    query,
                    k=max(limit, 10),
                    symbol_types=PROGRESSIVE_SYMBOL_TYPES,
                    exclude_types=DOC_SYMBOL_TYPES,
                )
                for result in _diversify_project_results(results, max(limit, 10)):
                    node_id = getattr(result, "node_id", "")
                    if node_id:
                        candidate_ids.append(node_id)
            else:
                return "Provide either query or symbol_name to find direct cross-repo links."

            seen_candidates: set[str] = set()
            candidate_ids = [node_id for node_id in candidate_ids if not (node_id in seen_candidates or seen_candidates.add(node_id))]
            if not candidate_ids:
                return f"No project symbols found for: {symbol_name or query}"

            rows: list[dict[str, Any]] = []
            seen_edges: set[tuple[str, str, str]] = set()
            for node_id in candidate_ids:
                for row in _call_cross_repo_edges(node_id, direction="both"):
                    key = (
                        row.get("source_node_id", ""),
                        row.get("target_node_id", ""),
                        row.get("edge_class", ""),
                    )
                    if key in seen_edges:
                        continue
                    seen_edges.add(key)
                    rows.append(row)
                    if len(rows) >= limit:
                        break
                if len(rows) >= limit:
                    break

            emit(
                {
                    "type": "tool_end",
                    "tool": "find_cross_repo_links",
                    "result_count": len(rows),
                    "timestamp": datetime.now().isoformat(),
                }
            )

            if not rows:
                return f"No direct cross-repo links found for: {symbol_name or query}"

            lines = [f"Found {len(rows)} direct cross-repo links for: {symbol_name or query}\n"]
            for index, row in enumerate(rows, 1):
                lines.append(f"{index}. {_format_cross_repo_edge(row)}")
            return "\n".join(lines)
        except Exception as e:
            logger.error(f"find_cross_repo_links failed: {e}", exc_info=True)
            return f"Cross-repo link search failed: {str(e)}"

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

        # Multi-wiki and storage-backed wikis route everything through
        # ``query_service`` — it is the only abstraction that knows how to
        # resolve a node and read its attributes across every wiki in the
        # project.  ``code_graph`` is only ever the *primary* wiki's NX
        # graph (or None for storage-backed primaries), so reading node
        # attributes from it directly silently drops cross-wiki nodes.
        if query_service is None and code_graph is None:
            return "Code graph not available"

        try:
            node_id = None
            if query_service:
                if hasattr(query_service, "get_node"):
                    try:
                        if query_service.get_node(symbol_name) is not None:
                            node_id = symbol_name
                    except Exception:
                        node_id = None
                node_id = node_id or query_service.resolve_symbol(symbol_name)
            if not node_id and code_graph is not None:
                node_id = _find_graph_node(code_graph, symbol_name)
            if not node_id:
                return f"Symbol '{symbol_name}' not found. Check the exact name from search_symbols."

            # Prefer ``query_service.get_node`` (cross-wiki / storage-aware);
            # only fall back to the in-memory ``code_graph`` when no service
            # is wired (legacy single-wiki NX-only path).
            data: dict = {}
            if query_service is not None and hasattr(query_service, "get_node"):
                data = query_service.get_node(node_id) or {}
            if not data and code_graph is not None:
                data = code_graph.nodes.get(node_id, {}) or {}
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
                    doc_search = getattr(retriever_stack, "search_docs_semantic", None)
                    if doc_search is not None:
                        vs_docs = doc_search(
                            query=query,
                            k=k,
                            similarity_threshold=0.0,
                        )
                    else:
                        # Fallback for older retrievers: use repository search,
                        # but keep only documentation nodes/chunks.
                        vs_docs = retriever_stack.search_repository(
                            query=query,
                            k=max(k * 3, 10),
                            apply_expansion=False,
                        )
                        vs_docs = [doc for doc in vs_docs if _is_document_hit(doc)]
                    docs.extend(vs_docs[:k])
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
                source = _format_doc_source(doc)
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
                wiki_id = getattr(r, "wiki_id", "") or getattr(r, "source_wiki_id", "")
                wiki_tag = f" [wiki: {wiki_id}]" if wiki_id else ""
                id_tag = f" [id: `{r.node_id}`]" if r.node_id else ""
                lines.append(
                    f"- **{r.symbol_name}** ({r.symbol_type}){layer_tag} — "
                    f"`{location}`{conn_tag}{wiki_tag}{id_tag}"
                )

            return "\n".join(lines)

        except Exception as e:
            logger.error(f"query_graph failed: {e}", exc_info=True)
            return f"Query failed: {str(e)}"

    # ================================================================
    # read_source_file — direct file access to the cloned repository
    # ================================================================

    def _build_repo_roots(value: str | dict[str, str] | None) -> dict[str, str]:
        if isinstance(value, dict):
            roots = {}
            for wiki_id, path in value.items():
                if path and os.path.isdir(path):
                    roots[str(wiki_id)] = os.path.realpath(path)
            return roots
        if isinstance(value, str) and os.path.isdir(value):
            return {"": os.path.realpath(value)}
        return {}

    _repo_roots = _build_repo_roots(repo_path)
    _multi_repo_files = len(_repo_roots) > 1
    _repo_available = bool(_repo_roots)

    def _repo_aliases() -> dict[str, str]:
        aliases: dict[str, str] = {}
        for wiki_id, root in _repo_roots.items():
            raw_aliases = {wiki_id, os.path.basename(root)}
            basename = os.path.basename(root)
            parts = basename.split("_")
            if len(parts) >= 2:
                raw_aliases.add(parts[-1])
                raw_aliases.add("_".join(parts[:2]))
            for alias in raw_aliases:
                alias = (alias or "").strip().strip("/")
                if alias:
                    aliases.setdefault(alias, wiki_id)
                    aliases.setdefault(alias.replace("-", "_"), wiki_id)
                    aliases.setdefault(alias.lower(), wiki_id)
        return aliases

    _repo_alias_map = _repo_aliases()

    def _split_repo_qualified_path(path_value: str) -> tuple[str | None, str]:
        clean = (path_value or ".").strip().lstrip("/")
        if not clean or clean == ".":
            return None, "."
        first, sep, rest = clean.partition("/")
        wiki_id = _repo_alias_map.get(first) or _repo_alias_map.get(first.lower())
        if wiki_id is not None:
            return wiki_id, rest if sep else "."
        return None, clean

    def _safe_join(root: str, rel_path: str) -> str | None:
        try:
            full = os.path.realpath(os.path.join(root, rel_path))
            root_real = os.path.realpath(root)
            if not full.startswith(root_real + os.sep) and full != root_real:
                return None
            return full
        except (ValueError, OSError):
            return None

    def _matching_repo_paths(path_value: str, *, expect_dir: bool = False, expect_file: bool = False) -> list[tuple[str, str, str]]:
        explicit_wiki_id, rel_path = _split_repo_qualified_path(path_value)
        candidates = (
            [(explicit_wiki_id, _repo_roots[explicit_wiki_id])]
            if explicit_wiki_id in _repo_roots
            else list(_repo_roots.items())
        )
        matches = []
        for wiki_id, root in candidates:
            full = _safe_join(root, rel_path)
            if not full:
                continue
            if expect_dir and not os.path.isdir(full):
                continue
            if expect_file and not os.path.isfile(full):
                continue
            matches.append((wiki_id, rel_path, full))
        return matches

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

        matches = _matching_repo_paths(file_path, expect_file=True)
        if not matches:
            searched = ", ".join(_repo_roots.keys()) if _multi_repo_files else "repository"
            return f"Error: File not found: {file_path} (searched: {searched})"
        if len(matches) > 1:
            options = [f"{wiki_id}/{rel_path}" for wiki_id, rel_path, _full in matches]
            return "Error: File path is ambiguous across repositories. Use one of: " + ", ".join(options)

        wiki_id, rel_path, full = matches[0]

        try:
            with open(full, errors="replace") as f:
                lines = f.readlines()
            total = len(lines)
            selected = lines[offset : offset + limit]
            numbered = [f"{offset + i + 1:>5} | {line.rstrip()}" for i, line in enumerate(selected)]
            display_path = f"{wiki_id}:{rel_path}" if _multi_repo_files else rel_path
            header = f"# {display_path} ({total} lines total, showing {offset + 1}-{offset + len(selected)})"
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

        matches = _matching_repo_paths(directory, expect_dir=True)
        if not matches:
            searched = ", ".join(_repo_roots.keys()) if _multi_repo_files else "repository"
            return f"Error: Directory not found: {directory} (searched: {searched})"

        try:
            import fnmatch

            sections = []
            for wiki_id, rel_path, full in matches:
                entries = sorted(os.listdir(full))
                if pattern:
                    entries = [e for e in entries if fnmatch.fnmatch(e, pattern)]

                lines = []
                for entry in entries[:200]:
                    entry_path = os.path.join(full, entry)
                    suffix = "/" if os.path.isdir(entry_path) else ""
                    lines.append(f"  {entry}{suffix}")

                display_path = f"{wiki_id}:{rel_path}" if _multi_repo_files else rel_path
                header = f"# {display_path}/ ({len(entries)} entries"
                if len(entries) > 200:
                    header += ", showing first 200"
                header += ")"
                sections.append(header + "\n" + "\n".join(lines))
            return "\n\n".join(sections)
        except Exception as e:
            return f"Error listing {directory}: {e}"

    # ================================================================
    # Tool Selection: feature-flag gated
    # ================================================================
    _progressive = os.environ.get("WIKIS_PROGRESSIVE_TOOLS", "").strip()
    _use_progressive = _progressive in ("1", "true", "yes")

    # File access tools — included only when the cloned repo is on disk
    _file_tools = [read_source_file, list_repo_files] if _repo_available else []
    _project_link_tools = [find_cross_repo_links] if query_service is not None and hasattr(query_service, "cross_repo_edges") else []
    if _repo_available:
        logger.info("[RESEARCH_TOOLS] read_source_file + list_repo_files enabled (repo_roots=%s)", list(_repo_roots.keys()))

    if _use_progressive:
        logger.info("[RESEARCH_TOOLS] Using progressive disclosure tools (SPEC-5)")
        return [
            search_symbols,
            *_project_link_tools,
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
